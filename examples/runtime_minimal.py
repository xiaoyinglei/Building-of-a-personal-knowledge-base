from __future__ import annotations

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageConfig
from rag.query import QueryOptions


def main() -> None:
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            profile_id="test_minimal",
            requirements=CapabilityRequirements(
                require_chat=False,
                default_context_tokens=1024,
            ),
        ),
    )
    try:
        runtime.insert(
            source_type="plain_text",
            location="memory://runtime-demo",
            owner="demo",
            content_text="Assembly drives model selection. The runtime entrypoint owns construction.",
        )
        result = runtime.query(
            "What owns construction?",
            options=QueryOptions(mode="mix"),
        )
        print("diagnostics:", runtime.diagnostics_payload())
        print("answer:", result.answer.answer_text)
        print("evidence_count:", len(result.context.evidence))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
