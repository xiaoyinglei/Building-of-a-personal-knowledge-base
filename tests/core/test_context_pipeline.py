from __future__ import annotations

import json
from types import SimpleNamespace

from pytest import MonkeyPatch

from rag import RAGRuntime
from rag.assembly import (
    AssemblyConfig,
    AssemblyOverrides,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityRequirements,
    ProviderConfig,
)
from rag.providers.generation import AnswerGenerationResult
from rag.providers.adapters import FallbackEmbeddingRepo
from rag.retrieval.context import EvidenceTruncator
from rag.retrieval.analysis import RoutingDecision
from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult
from rag.runtime import _QueryPipeline
from rag.retrieval.models import PublicQueryResult, QueryOptions, RetrievalResult
from rag.schema.query import EvidenceItem, GroundedAnswer, TaskType, ComplexityLevel
from rag.schema.runtime import ExecutionLocationPreference, RetrievalDiagnostics, RuntimeMode
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
        self.prompts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._fallback.embed(texts)

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
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
            mode="mix",
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
        ),
    )

    assert result.generation_provider == "fake-core"
    assert "Beta Service depends on Alpha Engine for upstream context." in result.answer.answer_text
    assert "[" in result.answer.answer_text and "]" in result.answer.answer_text
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


def test_ragcore_query_limits_answer_context_independently_from_retrieval() -> None:
    provider = FakeGenerationProvider()
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch = MonkeyPatch()
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
        location="memory://many-evidence",
        owner="user",
        content_text=(
            "Alpha Engine validates chunks. "
            "Beta Service depends on Alpha Engine. "
            "Gamma Index stores vectors. "
            "Delta Router chooses query branches."
        ),
    )

    result = runtime.query(
        "How does Beta Service relate to Alpha Engine?",
        options=QueryOptions(
            mode="mix",
            max_evidence_chunks=4,
            answer_context_top_k=1,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
        ),
    )

    assert len(result.context.evidence) == 1
    assert "E1" in result.context.prompt
    assert "E2 | kind=internal" not in result.context.prompt
    assert provider.prompts
    runtime.close()
    monkeypatch.undo()


def test_ragcore_query_applies_grounding_service_before_prompt_build() -> None:
    base_evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            doc_id="doc-1",
            citation_anchor="Alpha / Engine",
            text="Alpha Engine processes ingestion requests.",
            score=0.9,
            retrieval_channels=["vector"],
        )
    ]

    class _RetrievalService:
        def retrieve(self, *_args, **_kwargs):
            return RetrievalResult(
                decision=RoutingDecision(
                    task_type=TaskType.LOOKUP,
                    complexity_level=ComplexityLevel.L1_DIRECT,
                    runtime_mode=RuntimeMode.FAST,
                ),
                diagnostics=RetrievalDiagnostics(),
                evidence=EvidenceBundle(internal=list(base_evidence)),
                self_check=SelfCheckResult(
                    retrieve_more=False,
                    evidence_sufficient=True,
                    claim_supported=True,
                ),
            )

    class _ContextMerger:
        def merge(self, retrieval):
            return list(retrieval.evidence.internal)

    class _GroundingService:
        def ground(self, *, query: str, evidence):
            del query
            return [
                evidence[0].model_copy(
                    update={
                        "text": "GROUNDING-OVERRIDE",
                        "citation_anchor": "Grounded / Override",
                    }
                )
            ]

    class _PromptBuilder:
        token_accounting = SimpleNamespace(
            count=lambda prompt: len(prompt.split()),
            clip=lambda prompt, _budget: prompt,
        )

        def build(
            self,
            *,
            query: str,
            grounded_candidate: str,
            evidence,
            runtime_mode,
            response_type: str,
            user_prompt,
            conversation_history,
            prompt_style: str = "full",
        ):
            del grounded_candidate, runtime_mode, response_type, user_prompt, conversation_history, prompt_style
            prompt = f"{query}\n{evidence[0].text}"
            return SimpleNamespace(
                grounded_candidate=evidence[0].text,
                prompt=prompt,
                token_count=len(prompt.split()),
            )

    class _AnswerGenerator:
        def grounded_candidate(self, query, evidence_pack, *, query_understanding=None):
            del query, query_understanding
            return evidence_pack[0].text

        def generate(
            self,
            *,
            query: str,
            prompt: str,
            evidence_pack,
            grounded_candidate: str,
            runtime_mode,
            access_policy,
            execution_location_preference,
        ):
            del query, evidence_pack, grounded_candidate, runtime_mode, access_policy, execution_location_preference
            return AnswerGenerationResult(
                answer=GroundedAnswer(
                    answer_text=prompt,
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                provider="fake",
                model="fake-model",
                attempts=[],
            )

    pipeline = _QueryPipeline(
        retrieval=_RetrievalService(),
        context_merger=_ContextMerger(),
        grounding_service=_GroundingService(),
        truncator=EvidenceTruncator(),
        prompt_builder=_PromptBuilder(),
        answer_generator=_AnswerGenerator(),
    )

    result = pipeline.run("What does Alpha Engine do?", options=QueryOptions())

    assert result.context.evidence
    assert result.context.evidence[0].text == "GROUNDING-OVERRIDE"
    assert "GROUNDING-OVERRIDE" in result.context.prompt


def test_query_pipeline_resolves_user_authorization_before_retrieval() -> None:
    captured: dict[str, object] = {}
    base_evidence = [
        EvidenceItem(
            chunk_id="chunk-1",
            doc_id="doc-allowed",
            citation_anchor="Alpha / Engine",
            text="Alpha Engine processes ingestion requests.",
            score=0.9,
            retrieval_channels=["vector"],
        )
    ]

    class _AuthorizationService:
        def resolve_query(self, *, user_id: str | None, access_policy, source_scope):
            captured["user_id"] = user_id
            captured["incoming_scope"] = source_scope
            return SimpleNamespace(
                user_id=user_id,
                access_policy=access_policy,
                source_scope=("doc-allowed",),
                allowed_doc_ids=("doc-allowed",),
            )

    class _RetrievalService:
        def retrieve(self, *_args, **kwargs):
            captured["retrieval_scope"] = kwargs["source_scope"]
            return RetrievalResult(
                decision=RoutingDecision(
                    task_type=TaskType.LOOKUP,
                    complexity_level=ComplexityLevel.L1_DIRECT,
                    runtime_mode=RuntimeMode.FAST,
                ),
                diagnostics=RetrievalDiagnostics(),
                evidence=EvidenceBundle(internal=list(base_evidence)),
                self_check=SelfCheckResult(
                    retrieve_more=False,
                    evidence_sufficient=True,
                    claim_supported=True,
                ),
            )

    class _ContextMerger:
        def merge(self, retrieval):
            return list(retrieval.evidence.internal)

    class _PromptBuilder:
        token_accounting = SimpleNamespace(
            count=lambda prompt: len(prompt.split()),
            clip=lambda prompt, _budget: prompt,
        )

        def build(
            self,
            *,
            query: str,
            grounded_candidate: str,
            evidence,
            runtime_mode,
            response_type: str,
            user_prompt,
            conversation_history,
            prompt_style: str = "full",
        ):
            del grounded_candidate, runtime_mode, response_type, user_prompt, conversation_history, prompt_style
            prompt = f"{query}\n{evidence[0].text}"
            return SimpleNamespace(
                grounded_candidate=evidence[0].text,
                prompt=prompt,
                token_count=len(prompt.split()),
            )

    class _AnswerGenerator:
        def grounded_candidate(self, query, evidence_pack, *, query_understanding=None):
            del query, query_understanding
            return evidence_pack[0].text

        def generate(
            self,
            *,
            query: str,
            prompt: str,
            evidence_pack,
            grounded_candidate: str,
            runtime_mode,
            access_policy,
            execution_location_preference,
        ):
            del query, evidence_pack, grounded_candidate, runtime_mode, access_policy, execution_location_preference
            return AnswerGenerationResult(
                answer=GroundedAnswer(
                    answer_text=prompt,
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                provider="fake",
                model="fake-model",
                attempts=[],
            )

    pipeline = _QueryPipeline(
        retrieval=_RetrievalService(),
        context_merger=_ContextMerger(),
        grounding_service=None,
        truncator=EvidenceTruncator(),
        prompt_builder=_PromptBuilder(),
        answer_generator=_AnswerGenerator(),
        synthesis_service=None,
        authorization_service=_AuthorizationService(),
    )

    result = pipeline.run("What does Alpha Engine do?", options=QueryOptions(user_id="alice"))

    assert result.answer.answer_text
    assert captured["user_id"] == "alice"
    assert captured["retrieval_scope"] == ("doc-allowed",)


def test_ragcore_query_public_returns_lean_contract_without_retrieval_result() -> None:
    core = make_runtime()
    core.insert(
        source_type="plain_text",
        location="memory://alpha-public",
        owner="user",
        content_text="Alpha Engine handles ingestion and retrieval orchestration.",
    )

    result = core.query_public("What does Alpha Engine handle?")

    assert isinstance(result, PublicQueryResult)
    assert result.answer.answer_text
    assert result.context.evidence
    assert result.retrieval_diagnostics.mode_executor is not None
    assert not hasattr(result, "retrieval")
    core.close()
