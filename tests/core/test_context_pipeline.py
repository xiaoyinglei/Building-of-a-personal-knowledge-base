from __future__ import annotations

import json

from pytest import MonkeyPatch

from rag import RAGRuntime
from rag.llm._providers.fallback_embedding_repo import FallbackEmbeddingRepo
from rag.llm.assembly import (
    AssemblyConfig,
    AssemblyOverrides,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityRequirements,
    ProviderConfig,
)
from rag.query.query import QueryOptions
from rag.schema._types.access import ExecutionLocationPreference
from rag.storage import StorageConfig
from tests.support import make_runtime


class FakeGenerationProvider:
    provider_name = "fake-core"
    embedding_model_name = "fake-embed"
    chat_model_name = "fake-chat"
    is_embed_configured = True
    is_chat_configured = True

    def __init__(self) -> None:
        self._fallback = FallbackEmbeddingRepo()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._fallback.embed(texts)

    def chat(self, prompt: str) -> str:
        del prompt
        return json.dumps(
            {
                "answer_text": "Beta Service depends on Alpha Engine for upstream context.",
                "answer_sections": [
                    {
                        "title": "Dependency",
                        "text": "Beta Service depends on Alpha Engine for upstream context.",
                        "evidence_ids": ["E1"],
                    }
                ],
                "insufficient_evidence_flag": False,
            }
        )


def test_ragcore_query_returns_grounded_answer_retrieval_and_context() -> None:
    core = make_runtime()
    core.insert(
        source_type="plain_text",
        location="memory://alpha-engine",
        owner="user",
        content_text=(
            "Alpha Engine processes ingestion requests. "
            "Beta Service depends on Alpha Engine for upstream context. "
            "Gamma Index stores chunk vectors for retrieval."
        ),
    )

    result = core.query("What does Alpha Engine do?")

    assert result.answer.answer_text
    assert result.retrieval.evidence.internal
    assert result.context.evidence
    assert "E1" in result.context.prompt
    assert "kind=internal" in result.context.prompt
    assert result.context.token_count <= result.context.token_budget
    assert result.context.grounded_candidate


def test_ragcore_query_uses_generation_provider_when_available(monkeypatch: MonkeyPatch) -> None:
    provider = FakeGenerationProvider()
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(service, "_compatibility_config_from_environment", lambda: (AssemblyConfig(), {}))
    monkeypatch.setattr(service, "_build_provider", lambda config: provider)
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            requirements=CapabilityRequirements(require_chat=True),
            overrides=AssemblyOverrides(
                embedding=ProviderConfig(
                    provider_kind="fake-core",
                    embedding_model="fake-embed",
                ),
                chat=ProviderConfig(
                    provider_kind="fake-core",
                    chat_model="fake-chat",
                ),
            ),
        ),
        assembly_service=service,
    )
    runtime.insert(
        source_type="plain_text",
        location="memory://alpha-beta",
        owner="user",
        content_text=(
            "Alpha Engine processes ingestion requests. Beta Service depends on Alpha Engine for upstream context."
        ),
    )

    result = runtime.query(
        "How is Beta Service related to Alpha Engine?",
        options=QueryOptions(
            mode="hybrid",
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
        ),
    )

    assert result.generation_provider == "fake-core"
    assert result.answer.answer_text == "Beta Service depends on Alpha Engine for upstream context."
    assert result.answer.answer_sections
    runtime.close()


def test_ragcore_query_truncates_context_to_budget() -> None:
    core = make_runtime()
    core.insert(
        source_type="plain_text",
        location="memory://long-context",
        owner="user",
        content_text=" ".join(["Alpha Engine processes ingestion requests and validates chunks." for _ in range(120)]),
    )

    result = core.query(
        "What does Alpha Engine do?",
        options=QueryOptions(max_context_tokens=40, max_evidence_chunks=1),
    )

    assert len(result.context.evidence) == 1
    assert result.context.token_count <= 40
    assert result.context.truncated_count >= 0
    assert result.context.evidence[0].selected_token_count <= 40
