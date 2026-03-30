from pkp import RAGCore, StorageConfig
from pkp.query import QueryOptions

core = RAGCore(storage=StorageConfig(root=".ragcore"))

core.insert(
    source_type="plain_text",
    location="memory://note-1",
    owner="user",
    content_text="Alpha Engine processes ingestion requests.",
)

result = core.query(
    "Alpha Engine 是做什么的？",
    options=QueryOptions(mode="mix"),
)

print(result.answer.answer_text)

core.delete(location="memory://note-1")
core.rebuild(location="memory://note-1")