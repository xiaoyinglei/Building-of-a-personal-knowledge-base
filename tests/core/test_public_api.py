from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from rag import RAGRuntime, StorageComponentConfig, StorageConfig
from rag.assembly import (
    AssemblyConfig,
    AssemblyOverrides,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityRequirements,
    ProviderConfig,
)
from rag.ingest.pipeline import DirectContentItem
from rag.retrieval.models import QueryOptions
from rag.schema.core import GraphEdge, GraphNode
from tests.support import make_runtime


def test_rag_exposes_insert_query_delete_rebuild() -> None:
    core = make_runtime()

    assert hasattr(core, "insert")
    assert hasattr(core, "query")
    assert hasattr(core, "delete")
    assert hasattr(core, "rebuild")


def test_rag_exposes_batch_ingest_and_custom_kg_operations() -> None:
    core = make_runtime()

    assert hasattr(core, "insert_many")
    assert hasattr(core, "insert_content_list")
    assert hasattr(core, "insert_custom_kg")
    assert hasattr(core, "upsert_node")
    assert hasattr(core, "upsert_edge")
    assert hasattr(core, "delete_node")
    assert hasattr(core, "delete_edge")


def test_query_options_defaults_to_mix_mode() -> None:
    options = QueryOptions()

    assert options.mode == "mix"


def test_query_options_accepts_bypass_mode() -> None:
    options = QueryOptions(mode="bypass")

    assert options.mode == "bypass"


def test_rag_bypass_mode_uses_direct_llm_path_without_retrieval() -> None:
    core = make_runtime()
    try:
        result = core.query(
            "直接回答：Alpha Engine 是什么？",
            options=QueryOptions(
                mode="bypass",
                response_type="Single Paragraph",
                user_prompt="简洁回答，不要引用。",
            ),
        )

        assert result.mode == "bypass"
        assert result.retrieval.evidence.internal == []
        assert result.retrieval.reranked_chunk_ids == []
        assert "Alpha Engine" in result.context.prompt
        assert result.context.evidence == []
    finally:
        core.stores.close()


def test_storage_config_accepts_string_root(tmp_path: Path) -> None:
    root = tmp_path / ".rag"

    core = make_runtime(storage=StorageConfig(root=str(root)))

    assert core.stores.root == root


def test_storage_config_accepts_component_backends(tmp_path: Path) -> None:
    root = tmp_path / ".rag-components"
    core = make_runtime(
        storage=StorageConfig(
            root=root,
            metadata=StorageComponentConfig(backend="sqlite"),
            vectors=StorageComponentConfig(backend="sqlite"),
            graph=StorageComponentConfig(backend="sqlite"),
            cache=StorageComponentConfig(backend="metadata"),
            object_store=StorageComponentConfig(backend="local"),
            fts=StorageComponentConfig(backend="sqlite"),
        )
    )

    assert core.stores.root == root
    assert type(cast(object, core.stores.cache_repo)) is type(cast(object, core.stores.metadata_repo))


def test_storage_config_requires_dsn_for_remote_component_backend(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires a DSN/URI"):
        StorageConfig(
            root=tmp_path / ".rag-milvus",
            vectors=StorageComponentConfig(backend="milvus"),
        ).build()


class _FakeEmbedOnlyProvider:
    provider_name = "fake-embed"
    embedding_model_name = "fake-embed-model"
    is_embed_configured = True
    is_chat_configured = False
    is_rerank_configured = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        raise AssertionError("engine should not wire fallback rerank providers when is_rerank_configured is false")


def test_rag_custom_kg_operations_round_trip() -> None:
    core = make_runtime()
    try:
        node = GraphNode(node_id="entity-alpha", node_type="entity", label="Alpha Engine")
        edge = GraphEdge(
            edge_id="edge-supports",
            from_node_id="entity-alpha",
            to_node_id="entity-beta",
            relation_type="supports",
            confidence=0.9,
            evidence_chunk_ids=["chunk-1"],
        )

        core.upsert_node(node, evidence_chunk_ids=["chunk-1"])
        core.upsert_edge(edge)

        assert core.list_nodes(node_type="entity") == [node]
        assert core.get_node(node.node_id) == node
        assert core.get_edge(edge.edge_id) == edge

        result = core.insert_custom_kg(
            nodes=[GraphNode(node_id="entity-beta", node_type="entity", label="Beta Service")],
            edges=[edge],
        )

        assert result["node_count"] == 1
        assert result["edge_count"] == 1

        core.delete_edge(edge.edge_id)
        core.delete_node(node.node_id)

        assert core.get_edge(edge.edge_id) is None
        assert core.get_node(node.node_id) is None
    finally:
        core.stores.close()


def test_rag_insert_content_list_normalizes_mixed_content_inputs(tmp_path: Path) -> None:
    core = make_runtime()
    try:
        markdown_path = tmp_path / "note.md"
        markdown_path.write_text("# Alpha Note\n\nAlpha Engine supports Beta Service.\n", encoding="utf-8")

        result = core.insert_content_list(
            [
                DirectContentItem(
                    location="memory://alpha.txt",
                    source_type="plain_text",
                    content="Alpha Engine supports Beta Service.",
                ),
                DirectContentItem(
                    location=str(markdown_path),
                    source_type="markdown",
                    content=markdown_path,
                ),
                DirectContentItem(
                    location="https://example.com/article",
                    source_type="web",
                    content=(
                        "<html><body><article><h1>Alpha Web</h1>"
                        "<p>Alpha Engine overview.</p></article></body></html>"
                    ),
                ),
            ]
        )

        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.results) == 3
        assert {item.source.source_type.value for item in result.results} == {"plain_text", "markdown", "web"}
    finally:
        core.stores.close()


def test_rag_query_skips_unconfigured_rerank_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _FakeEmbedOnlyProvider()
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(service, "_compatibility_config_from_environment", lambda: (AssemblyConfig(), {}))
    monkeypatch.setattr(service, "_build_provider", lambda config: provider)
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            requirements=CapabilityRequirements(),
            overrides=AssemblyOverrides(
                embedding=ProviderConfig(
                    provider_kind="fake-embed",
                    embedding_model="fake-embed-model",
                )
            ),
        ),
        assembly_service=service,
    )
    try:
        runtime.insert(
            source_type="plain_text",
            location="memory://alpha",
            owner="user",
            content_text="Alpha Engine supports Beta Service through graph-aware retrieval.",
        )
        result = runtime.query("What does Alpha Engine support?")

        assert result.retrieval.diagnostics.rerank_provider is None
        assert result.answer.answer_text
    finally:
        runtime.close()
