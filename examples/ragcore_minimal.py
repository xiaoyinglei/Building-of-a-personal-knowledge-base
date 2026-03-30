from pathlib import Path

from pkp import RAGCore, StorageConfig
from pkp.query import QueryOptions


def main() -> None:
    core = RAGCore(storage=StorageConfig(root=Path(".ragcore-demo")))

    ingest_result = core.insert(
        source_type="plain_text",
        location="memory://note-1",
        owner="user",
        content_text="Alpha Engine processes ingestion requests.",
    )

    query_result = core.query(
        "Alpha Engine 是做什么的？",
        options=QueryOptions(mode="mix"),
    )

    print("document_id:", ingest_result.document_id)
    print("chunk_count:", ingest_result.chunk_count)
    print("answer:", query_result.answer.answer_text)
    print("evidence:")
    for item in query_result.context.evidence[:3]:
        print(f"  - {item.citation_anchor} | score={item.score:.3f}")

    delete_result = core.delete(location="memory://note-1")
    print("deleted_doc_ids:", delete_result.deleted_doc_ids)


if __name__ == "__main__":
    main()
