# RAG

一个以核心 RAG 引擎为中心的仓库。当前公开面只有两层：

- `rag/`：核心库
- `rag` CLI：最小命令行入口

这个项目现在的目标不是做 API 包装层，不是做 demo workbench，而是把 `ingest -> parse -> chunk -> index -> retrieve -> answer` 这条链路做成可直接使用的 RAG 后端。

## 这份 README 解决什么问题

如果你现在想真正把项目跑起来，最重要的是先分清两种用法：

- 最小本地用法：直接用 CLI 或默认 `StorageConfig(root=".rag")`，走本地 SQLite 路径，适合快速上手
- 正式接入用法：用 Python API 显式绑定模型和存储后端，适合接 OpenAI / Ollama / Milvus / Neo4j / Redis / Postgres

当前 CLI 是极简入口，只覆盖本地 SQLite 工作流。多后端装配、模型绑定、生产配置，应该走 Python API。

## 你能用它做什么

- 接入文档：`plain_text`、`markdown`、`pdf`、`docx`、`pptx`、`xlsx`、`image`、`web`
- 建索引：chunk、vector、graph、FTS、对象存储
- 查询模式：`bypass`、`naive`、`local`、`global`、`hybrid`、`mix`
- 批量导入：`insert_many()`、`insert_content_list()`
- 自定义图谱：`insert_custom_kg()`、`upsert_node()`、`upsert_edge()`
- 生命周期操作：`insert`、`query`、`delete`、`rebuild`

## 安装

要求：

- Python `>=3.12,<3.14`
- 推荐使用 [`uv`](https://docs.astral.sh/uv/)

安装依赖：

```bash
uv sync
```

安装后可以确认 CLI 已经可用：

```bash
uv run rag --help
```

## 3 分钟跑通

### 1. 用 CLI 插入一段文本

```bash
uv run rag ingest \
  --storage-root .rag \
  --source-type plain_text \
  --location memory://note-1 \
  --content "Alpha Engine handles ingestion. Beta Service depends on Alpha Engine."
```

### 2. 查询

```bash
uv run rag query \
  --storage-root .rag \
  --mode mix \
  --query "What does Alpha Engine handle?" \
  --json
```

### 3. 删除或重建

```bash
uv run rag delete --storage-root .rag --location memory://note-1
uv run rag rebuild --storage-root .rag --location memory://note-1
```

这条路径默认使用本地 `.rag/` 目录，底层是本地 SQLite 系列存储，适合最小可跑。

## 完整 CLI 流程

如果你就是想像现在这样：

1. 导入一个文档
2. 立刻开始提问
3. 知道答案靠不靠谱

那就按这条完整流程走，不要跳步骤。

### 第 1 步：给这次实验单独建一个索引目录

不要一上来就把所有文档都塞进同一个 `.rag`，否则你很快就不知道答案是从哪份文档来的。

```bash
export RAG_DEMO_ROOT=.rag-demo
rm -rf "$RAG_DEMO_ROOT"
```

### 第 2 步：导入一份文档

以你刚才那份 markdown 为例：

```bash
uv run rag ingest \
  --storage-root "$RAG_DEMO_ROOT" \
  --source-type markdown \
  --location /Users/leixiaoying/LLM/RAG学习/data/test_corpus/tech_docs/hf_transformers_document_question_answering.md
```

导入成功后，你会看到一大段 JSON。现在你只需要先看 4 个字段：

- `document.title`
- `document.doc_id`
- `chunk_count`
- `status`

如果 `status` 不是 `ready`，先不要继续查。

注意：

- 这段 JSON 是 CLI 直接打印到终端的结果，不会自动写成文件
- 真正持久化落盘的是索引目录里的 `metadata.sqlite3`、`vectors.sqlite3`、`graph.sqlite3`、`fts.sqlite3` 和 `objects/`

如果你想把 ingest 结果保存成文件，请显式重定向：

```bash
uv run rag ingest ... > ingest-result.json
```

如果你想一边看终端输出、一边保存文件：

```bash
uv run rag ingest ... | tee ingest-result.json
```

### 第 3 步：先问最容易命中的问题

刚导入完一份英文技术文档时，先问贴近原文表述的问题，不要一上来就问太抽象的总结题。

对这份文档，先试这些：

```bash
uv run rag query \
  --storage-root "$RAG_DEMO_ROOT" \
  --mode mix \
  --query "What is Document Question Answering?"
```

```bash
uv run rag query \
  --storage-root "$RAG_DEMO_ROOT" \
  --mode mix \
  --query "What dataset is used in this guide?"
```

```bash
uv run rag query \
  --storage-root "$RAG_DEMO_ROOT" \
  --mode mix \
  --query "What is the model checkpoint?"
```

```bash
uv run rag query \
  --storage-root "$RAG_DEMO_ROOT" \
  --mode mix \
  --query "How does the guide say to run inference after fine-tuning?"
```

### 第 4 步：想看证据时，加 `--json`

```bash
uv run rag query \
  --storage-root "$RAG_DEMO_ROOT" \
  --mode mix \
  --query "What dataset is used in this guide?" \
  --json
```

这时不要盯着整份 JSON 看。先只看这几块：

- `answer.answer_text`
- `answer.insufficient_evidence_flag`
- `context.evidence`
- `retrieval.diagnostics.branch_hits`

判断方法：

- `answer.insufficient_evidence_flag = false` 且 `context.evidence` 不为空：说明至少取回了证据
- `context.evidence[0].section_path`：告诉你答案主要来自文档哪个章节
- `context.evidence[0].citation_anchor`：告诉你命中的具体锚点
- `retrieval.diagnostics.branch_hits`：告诉你这次到底命中了哪些检索分支

### 第 5 步：如果回答不对，先改问法

这是当前项目里非常实际的一点。

对于英文文档：

- 优先用英文问
- 优先沿着文档原句问
- 优先问单一事实
- 优先带上章节语义

比方说，不要一上来问：

```text
What are the key ideas of this file?
```

先改成：

```text
What dependencies does LayoutLMv2 require according to the guide?
```

或者：

```text
In the "Load the data" section, what dataset is used?
```

或者：

```text
What does the guide say about inference?
```

对当前系统来说，这几类问法更稳：

- 定义题：`What is X?`
- 依赖题：`What does X depend on?`
- 参数题：`What is the model checkpoint?`
- 章节题：`In the "Load the data" section, what dataset is used?`
- 流程题：`How does the guide say to run inference?`

## 我刚 ingest 完一个文件，下一步怎么问

如果你已经执行过：

```bash
uv run rag ingest \
  --storage-root .rag \
  --source-type markdown \
  --location /Users/leixiaoying/LLM/RAG学习/data/test_corpus/tech_docs/hf_transformers_document_question_answering.md
```

那你下一步直接执行：

```bash
uv run rag query \
  --storage-root .rag \
  --mode mix \
  --query "What is Document Question Answering?"
```

如果想看证据：

```bash
uv run rag query \
  --storage-root .rag \
  --mode mix \
  --query "What is Document Question Answering?" \
  --json
```

然后继续问这些：

- `What dataset is used in this guide?`
- `What is the model checkpoint?`
- `What dependencies does LayoutLMv2 require?`
- `How does the guide say to run inference after fine-tuning?`

如果这是中文文档，就把问题换成中文；如果这是英文文档，建议先用英文问。

## 最小 Python 用法

这是最容易理解当前项目的入口。

```python
from rag import RAG, StorageConfig
from rag.query import QueryOptions

core = RAG(storage=StorageConfig(root=".rag"))

try:
    ingest_result = core.insert(
        source_type="plain_text",
        location="memory://note-1",
        owner="user",
        content_text="Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
    )

    result = core.query(
        "What does Alpha Engine handle?",
        options=QueryOptions(mode="mix"),
    )

    print("document_id:", ingest_result.document_id)
    print("chunk_count:", ingest_result.chunk_count)
    print("answer:", result.answer.answer_text)
    print("evidence_count:", len(result.context.evidence))
finally:
    core.stores.close()
```

说明：

- `StorageConfig(root=".rag")` 是持久化本地索引
- `StorageConfig.in_memory()` 是临时目录，适合测试或脚本内一次性运行
- `core.stores.close()` 建议在脚本结束时调用，尤其是远程后端

完整示例见 `examples/rag_minimal.py`。

## 文件和网页怎么导入

### 本地文件

CLI 下，文件类型通常只要把 `location` 指向本地文件路径即可，RAG 会自己读文件。

```bash
uv run rag ingest \
  --storage-root .rag \
  --source-type markdown \
  --location ./docs/intro.md
```

```bash
uv run rag ingest \
  --storage-root .rag \
  --source-type pdf \
  --location ./samples/report.pdf
```

支持的文件源类型：

- `markdown`
- `pdf`
- `docx`
- `pptx`
- `xlsx`
- `image`

### 网页

如果 `source_type=web` 且没有显式传 `content`，系统会按 `location` 抓取网页内容。

```bash
uv run rag ingest \
  --storage-root .rag \
  --source-type web \
  --location https://example.com/article
```

### 直接注入内容列表

如果你已经在应用层拿到了文本、HTML、文件路径，可以用 `insert_content_list()`。

```python
from pathlib import Path

from rag import RAG, StorageConfig
from rag.ingest.ingest import DirectContentItem

core = RAG(storage=StorageConfig.in_memory())

try:
    result = core.insert_content_list(
        [
            DirectContentItem(
                location="memory://alpha.txt",
                source_type="plain_text",
                content="Alpha Engine supports Beta Service.",
            ),
            DirectContentItem(
                location="https://example.com/article",
                source_type="web",
                content="<html><body><h1>Alpha Web</h1><p>Alpha Engine overview.</p></body></html>",
            ),
            DirectContentItem(
                location="./docs/alpha.md",
                source_type="markdown",
                content=Path("./docs/alpha.md"),
            ),
        ]
    )
    print(result.success_count)
finally:
    core.stores.close()
```

## 查询模式怎么选

当前支持 6 种模式：

| 模式 | 适用场景 |
| --- | --- |
| `bypass` | 直接走聊天模型，不做检索。适合纯生成，不适合知识库问答 |
| `naive` | 基础向量检索 |
| `local` | 更偏局部实体 / 局部上下文检索 |
| `global` | 更偏关系 / 跨段聚合 |
| `hybrid` | 结合局部和全局图检索 |
| `mix` | 当前默认正式模式，综合 vector / sparse / graph / structure / special 信号 |

最常用写法：

```python
from rag.query import QueryOptions

options = QueryOptions(
    mode="mix",
    top_k=8,
    max_context_tokens=1200,
    response_type="Multiple Paragraphs",
)
```

注意：

- `bypass` 要求有可用聊天模型；如果没有聊天模型，只会返回“没有可用 chat provider”
- 其余模式即使没有聊天模型，也还能基于证据走 grounded fallback answer

## 如果不配置模型，会发生什么

项目可以“跑起来”，但你要知道质量边界。

默认情况下：

- embedding 会退回到内置 fallback embedding
- 回答生成如果没有聊天模型，会退回到基于证据的 grounded fallback
- `bypass` 没有聊天模型时不可用

所以：

- 想验证链路是否通：可以先不接真实模型
- 想要真正可用的检索和回答质量：必须显式配置 embedding + chat，最好再加 reranker

## 正式接入模型

### 方案 1：OpenAI

```python
import os

from rag import RAG, StorageConfig
from rag.llm.embedding import EmbeddingProviderBinding, OpenAIProviderRepo

provider = OpenAIProviderRepo(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
    embedding_model="text-embedding-3-small",
)

core = RAG(
    storage=StorageConfig(root=".rag"),
    embedding_bindings=(
        EmbeddingProviderBinding(
            provider=provider,
            space="default",
            location="cloud",
        ),
    ),
)
```

### 方案 2：Ollama

```python
from rag import RAG, StorageConfig
from rag.llm.embedding import EmbeddingProviderBinding, OllamaProviderRepo

provider = OllamaProviderRepo(
    base_url="http://localhost:11434",
    chat_model="qwen3.5:9b",
    embedding_model="qwen3-embedding:8b",
)

core = RAG(
    storage=StorageConfig(root=".rag"),
    embedding_bindings=(
        EmbeddingProviderBinding(
            provider=provider,
            space="default",
            location="local",
        ),
    ),
)
```

### 方案 3：本地 BGE embedding + rerank

```python
from rag import RAG, StorageConfig
from rag.llm.embedding import EmbeddingProviderBinding, LocalBgeProviderRepo

provider = LocalBgeProviderRepo(
    embedding_model="BAAI/bge-m3",
    rerank_model="BAAI/bge-reranker-v2-m3",
)

core = RAG(
    storage=StorageConfig(root=".rag"),
    embedding_bindings=(
        EmbeddingProviderBinding(
            provider=provider,
            space="default",
            location="local",
        ),
    ),
)
```

### rerank 是怎么启用的

当前项目不再内置 heuristic rerank。

正式行为只有两种：

- 你提供的 provider 本身支持 `rerank()`
- 或者设置环境变量，让引擎加载本地 BGE reranker

例如：

```bash
export RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

也可以指定本地模型路径：

```bash
export RAG_RERANK_MODEL_PATH=/path/to/local/reranker
```

## embedding / tokenizer / chunking 合同

这个项目现在已经把 embedding、chunking、token 预算绑到了统一 contract 上。

你最需要记住两条：

1. 索引阶段和查询阶段必须使用同一个 embedding model
2. tokenizer、chunking、prompt budget 由 `RAG_*` 环境变量统一约束

常用环境变量：

```bash
export RAG_EMBEDDING_MODEL=text-embedding-3-small
export RAG_TOKENIZER_MODEL=text-embedding-3-small
export RAG_CHUNKING_TOKENIZER_MODEL=text-embedding-3-small
export RAG_CHUNK_TOKEN_SIZE=480
export RAG_CHUNK_OVERLAP_TOKENS=64
export RAG_MAX_CONTEXT_TOKENS=1200
export RAG_PROMPT_RESERVED_TOKENS=256
```

如果你设置了 `RAG_EMBEDDING_MODEL`，但实际 provider 的 `embedding_model_name` 不一致，引擎会直接报错，而不是悄悄混用索引。

## 存储怎么选

### 1. 最小本地模式

这是 CLI 默认模式，也是最容易开始的模式：

- metadata：SQLite
- vectors：SQLite
- graph：SQLite
- cache：metadata
- object store：本地目录
- FTS：SQLite

对应代码：

```python
from rag import RAG, StorageConfig

core = RAG(storage=StorageConfig(root=".rag"))
```

### 2. 正式版多后端模式

如果你要把它作为后台服务能力，建议走组件式装配。

```python
from rag import RAG, StorageComponentConfig, StorageConfig

storage = StorageConfig(
    root=".rag-prod",
    metadata=StorageComponentConfig(
        backend="postgres",
        namespace="rag",
    ),
    vectors=StorageComponentConfig(
        backend="milvus",
        namespace="rag",
        collection="rag_vectors",
    ),
    graph=StorageComponentConfig(
        backend="neo4j",
        namespace="neo4j",
    ),
    cache=StorageComponentConfig(
        backend="redis",
        namespace="rag-cache",
    ),
    object_store=StorageComponentConfig(
        backend="minio",
        bucket="rag-objects",
        collection="rag",
    ),
    fts=StorageComponentConfig(
        backend="postgres",
        namespace="rag",
    ),
)

core = RAG(storage=storage)
```

当前支持的后端：

| 组件 | 后端 |
| --- | --- |
| metadata / documents / chunks | `sqlite`、`postgres` |
| vectors | `sqlite`、`pgvector`、`milvus` |
| graph | `sqlite`、`neo4j` |
| cache | `metadata`、`redis` |
| object store | `local`、`s3`、`minio` |
| FTS | `sqlite`、`postgres` |

### 常见环境变量

你可以把 DSN/URI 写在 `StorageComponentConfig` 里，也可以放进环境变量。

Postgres / pgvector：

```bash
export RAG_POSTGRES_DSN=postgresql://user:password@localhost:5432/rag
export RAG_METADATA_DSN=postgresql://user:password@localhost:5432/rag
export RAG_VECTOR_DSN=postgresql://user:password@localhost:5432/rag
export RAG_PGVECTOR_DSN=postgresql://user:password@localhost:5432/rag
export RAG_FTS_DSN=postgresql://user:password@localhost:5432/rag
```

Milvus：

```bash
export RAG_MILVUS_URI=http://localhost:19530
export RAG_MILVUS_TOKEN=username:password
export RAG_MILVUS_DB=default
```

Neo4j：

```bash
export RAG_NEO4J_URI=bolt://localhost:7687
export RAG_NEO4J_USERNAME=neo4j
export RAG_NEO4J_PASSWORD=your-password
export RAG_NEO4J_DATABASE=neo4j
```

Redis：

```bash
export RAG_REDIS_URL=redis://localhost:6379/0
```

S3 / MinIO：

```bash
export RAG_OBJECT_BUCKET=rag-objects
export RAG_OBJECT_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
```

说明：

- `backend="minio"` 时，通常要配 `RAG_OBJECT_ENDPOINT`
- `backend="s3"` 时，通常直接用 AWS 标准环境变量

## 批量导入和自定义图谱

### 批量导入

```python
from rag import RAG, StorageConfig
from rag.ingest.ingest import IngestRequest

core = RAG(storage=StorageConfig.in_memory())

try:
    result = core.insert_many(
        [
            IngestRequest(
                location="memory://1",
                source_type="plain_text",
                owner="user",
                content_text="Alpha Engine supports Beta Service.",
            ),
            IngestRequest(
                location="memory://2",
                source_type="plain_text",
                owner="user",
                content_text="Gamma Layer depends on Alpha Engine.",
            ),
        ]
    )
    print(result.success_count)
finally:
    core.stores.close()
```

### 自定义 KG

```python
from rag import RAG, StorageConfig
from rag.schema.graph import GraphEdge, GraphNode

core = RAG(storage=StorageConfig.in_memory())

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
finally:
    core.stores.close()
```

## 当前 CLI 边界

这点很重要。

当前 `rag` CLI 只做最小入口：

- `rag ingest`
- `rag query`
- `rag delete`
- `rag rebuild`

而且 CLI 内部是固定：

```python
RAG(storage=StorageConfig(root=storage_root))
```

这意味着：

- CLI 默认是本地 SQLite 路径
- CLI 现在不能直接从参数里装配 `Postgres + Milvus + Neo4j + Redis + MinIO`
- 如果你要用多后端、真实模型绑定、正式 rerank，应该走 Python API

## 常见问题

### 1. 为什么 `bypass` 不工作

因为 `bypass` 是直接走聊天模型的模式。如果没有可用 chat provider，它不会回退成普通检索问答。

### 2. 为什么项目能回答，但回答不像真正 LLM

因为没有配置聊天模型时，系统会退回 grounded fallback answer。它是“基于证据给出答案”，不是“真正的 LLM 生成”。

### 3. 为什么换了 embedding model 之后索引报错

这是故意的。当前 runtime contract 会校验：

- embedding model
- tokenizer model
- chunking tokenizer model
- tokenizer backend
- chunk size
- overlap

只要这些关键项不一致，就会要求你重建索引。

### 4. 文档更新后怎么重建

最简单的方式：

```bash
uv run rag rebuild --storage-root .rag --location /path/to/file.pdf
```

或者在 Python API 里调用：

```python
core.rebuild(location="/path/to/file.pdf")
```

### 5. 为什么导入能成功，但查询时报 runtime contract mismatch

这是当前项目的硬约束，不是偶发问题。

索引会记录一份 runtime contract，包括：

- embedding model
- tokenizer model
- chunking tokenizer model
- tokenizer backend
- chunk size
- overlap

如果你导入时和查询时这些关键项不一致，系统会直接拒绝查询。

最简单的处理方式有两个：

1. 导入和查询都在同一个 shell 环境下执行，不要中途改 `RAG_*` 环境变量
2. 如果你改过 tokenizer / embedding / chunking 配置，就删掉旧索引后重新导入

例如：

```bash
rm -rf .rag
```

然后重新执行：

```bash
uv run rag ingest ...
uv run rag query ...
```

### 7. 为什么我只看到 sqlite3 文件，没有看到 graph 的 json 文件

这是正常现象。

默认本地模式下，系统不会把图谱单独写成 json 文件，而是直接落在：

- `graph.sqlite3`

也就是说：

- `metadata.sqlite3`：文档、segment、chunk、状态、cache
- `vectors.sqlite3`：向量索引
- `graph.sqlite3`：节点、边、证据关联
- `fts.sqlite3`：全文检索

所以“没有 graph json 文件”并不等于“没有走图构造”。当前默认实现本来就是把图存进 SQLite 图仓库。

### 6. 为什么 answer 不对，但 evidence 看起来像对的

这通常不是“完全没检索到”，而是以下几种情况：

- 你没接真实 chat model，只是在走 grounded fallback
- 你问得太抽象，超出了当前 evidence aggregation 的稳态范围
- 你问法和文档原文差得太远

排查顺序建议固定成这样：

1. 先看 `context.evidence` 有没有命中正确章节
2. 再看 `answer.insufficient_evidence_flag` 是否为 `true`
3. 如果 evidence 对、answer 不好，先把问题改写得更贴近原文
4. 如果你要更稳定的生成质量，接真实 chat provider 和 reranker

## 代码入口导览

如果你要继续读源码，优先看这些文件：

- `rag/engine.py`：主入口 `RAG`
- `rag/cli.py`：最小 CLI
- `rag/ingest/ingest.py`：导入、解析、建索引
- `rag/ingest/chunk.py`：文档切分
- `rag/query/retrieve.py`：查询主链
- `rag/query/understanding.py`：query understanding
- `rag/query/routing.py`：路由
- `rag/storage/__init__.py`：多后端存储装配

## 当前最推荐的使用方式

如果你是第一次用这个项目：

1. 先用 CLI 或 `StorageConfig(root=".rag")` 跑通最小链路
2. 再用 Python API 接入真实 embedding + chat 模型
3. 最后再切到组件化后端存储

如果你要把它当成正式知识库后台：

1. 用 Python API，不要只靠 CLI
2. 固定 embedding model，并用 `RAG_*` 环境变量锁定 tokenizer / chunking contract
3. 接上真实 reranker
4. 按组件拆分存储后端，而不是继续用单机 SQLite
