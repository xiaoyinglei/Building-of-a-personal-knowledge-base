# RAG

一个以 RAG 本体为中心的仓库。当前目标很直接：对齐 LightRAG 和 RAG-Anything 的核心能力，把 ingest、index、retrieve、graph、answer 这条主链做硬。

现在只保留两层公开面：

- `rag/`：核心库
- `rag/cli.py`：最小 CLI

旧的 `pkp` 包名、`interfaces/runtime/api/eval/workbench` 包装层不再保留。

## 核心能力

- 文档接入：`plain_text / markdown / pdf / docx / image / web`
- 存储形态：`KV + Vector + Graph + FTS`
- 检索模式：`naive / local / global / hybrid / mix`
- 图增强：实体、关系、证据 chunk 回指、跨文档聚合
- 多模态补充：`special chunk` 会进入图和向量索引
- 生命周期：`insert / query / delete / rebuild`

主流程：

`ingest -> parse -> chunk -> graph/vector index -> retrieve -> context build -> grounded answer`

## 目录

```text
rag/
├── __init__.py
├── cli.py
├── engine.py
├── document/
├── ingest/
├── llm/
├── query/
├── schema/
├── storage/
└── utils/
```

其中：

- `engine.py` 提供唯一主入口 `RAG`
- `ingest/` 负责解析、切块、索引、重建
- `query/` 负责检索、图扩展、上下文拼装
- `storage/` 负责 SQLite、FTS 和对象存储封装

## 快速开始

```bash
uv sync
```

默认索引目录是当前工作目录下的 `.rag/`。

### 库用法

```python
from rag import RAG, StorageConfig
from rag.query import QueryOptions

core = RAG(storage=StorageConfig(root=".rag"))

core.insert(
    source_type="plain_text",
    location="memory://note-1",
    owner="user",
    content_text="Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
)

result = core.query(
    "What does Alpha Engine handle?",
    options=QueryOptions(mode="mix"),
)

print(result.answer.answer_text)
print(result.context.evidence[0].text)

core.delete(location="memory://note-1")
core.rebuild(location="memory://note-1")
```

完整可运行示例见 [examples/rag_minimal.py](examples/rag_minimal.py)。

### CLI 用法

插入：

```bash
uv run rag ingest \
  --storage-root .rag \
  --source-type plain_text \
  --location memory://note-1 \
  --content "Alpha Engine handles ingestion. Beta Service depends on Alpha Engine."
```

查询：

```bash
uv run rag query \
  --storage-root .rag \
  --mode mix \
  --query "What does Alpha Engine handle?" \
  --json
```

删除和重建：

```bash
uv run rag delete --storage-root .rag --location memory://note-1
uv run rag rebuild --storage-root .rag --location memory://note-1
```

## 当前边界

- 这是核心 RAG 仓库，不再以内置 API、Deep Research runtime、评测入口为主
- 公开入口只有 `RAG` 和最小 CLI
- 保留下来的内部代码，只要还能直接增强核心 RAG，就继续留在现有目录里，而不是再长一层新框架
