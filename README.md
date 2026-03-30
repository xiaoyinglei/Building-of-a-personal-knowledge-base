# 个人知识库 RAG Core

一个以 `RAGCore` 为中心的中文 RAG 项目。

当前实现已经完成一次骨架收口：项目不再以 FastAPI 或 runtime 作为主骨架，而是以纯库形态的核心链路为中心；接口层只作为薄包装保留。整体设计吸收了 LightRAG 的主流程、四类存储和查询模式，也吸收了 RAG-Anything 的多模态 special chunk 思路。

## 当前定位

- 核心形态：纯库优先，`RAGCore` 是唯一核心入口
- 包装层：CLI、FastAPI、runtime、eval 仍可用，但全部下沉到 `interfaces`
- 核心目标：把文档接入、索引、检索、上下文构建、答案生成做成长期可维护的 RAG Core，而不是先做 Web 框架

## 核心链路

当前主链路是：

`文档输入 -> 解析 -> 切块 -> 实体/关系抽取 -> KV/Vector/Graph/DocStatus -> 多模式检索 -> Context Build -> LLM/grounded 生成`

具体能力包括：

- 文档接入：`markdown / pdf / docx / image / web / plain_text / pasted_text / browser_clip`
- 切块策略：结构切块 + token 切块 + `special` 多模态切块
- 图索引：实体、关系、证据 chunk 回指、跨文档聚合
- 查询模式：`naive / local / global / hybrid / mix`
- 上下文流水线：`search -> truncation -> merge/fusion -> prompt build -> generation`
- 文档生命周期：`insert / query / delete / rebuild`

## 目录结构

现在 `src/pkp` 顶层只保留下面这些目录和文件：

```text
src/pkp/
├── engine.py
├── ingest/
├── query/
├── llm/
├── storage/
├── document/
├── schema/
├── interfaces/
└── utils/
```

各目录职责：

- `engine.py`
  `RAGCore` 总入口，统一调度 `insert / query / delete / rebuild`
- `ingest/`
  入库主流程、切块、实体关系抽取
- `query/`
  查询主流程、检索、图扩展、上下文构建
- `llm/`
  embedding、rerank、生成，以及模型 provider 适配
- `storage/`
  存储逻辑层和底层 SQLite/FTS/对象存储实现
- `document/`
  文档加载、解析、OCR、多模态解析工具
- `schema/`
  领域模型、查询结构、图结构、存储状态定义
- `interfaces/`
  CLI、FastAPI、runtime、配置装配、评测等薄包装
- `utils/`
  公共文本工具、契约、telemetry 工具

说明：

- 目录里带 `_` 前缀的子包，表示内部实现细节，不作为主骨架对外暴露。
- 顶层已经移除了旧的 `algorithms / core / repo / service / runtime / ui / types / stores` 平行骨架。

## 存储设计

当前存储设计已经对齐 LightRAG 的四类核心存储，只是做了更细的逻辑拆分。

逻辑层：

- `KV`
  文档、segment、chunk、缓存
- `Vector`
  `chunk / entity / relation / multimodal` 向量
- `Graph`
  实体关系图、证据回指、alias
- `Doc Status`
  文档处理状态和阶段

默认物理落盘：

```text
.ragcore/
├── metadata.sqlite3
├── vectors.sqlite3
├── graph.sqlite3
├── fts.sqlite3
└── objects/
```

直接对应实现：

- `storage/kv_store.py`
- `storage/vector_store.py`
- `storage/graph_store.py`
- `storage/doc_status.py`

## 检索模式

当前公开查询模式在 `pkp.query.QueryMode` 中定义：

- `naive`
  只走普通 chunk 检索
- `local`
  偏实体、偏局部的图检索
- `global`
  偏关系、偏全局的图检索
- `hybrid`
  `local + global`
- `mix`
  图检索和普通 chunk 检索一起使用，是默认模式

此外，系统内部还会把表格、图片、caption、OCR 区域、公式等 special chunk 作为 companion evidence 带入检索和 context build。

## 快速开始

安装依赖：

```bash
uv sync --all-extras
cp .env.example .env
```

### 最小库调用

最小公开 API：

```python
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
```

如果你只想跑临时内存实例：

```python
from pkp import RAGCore, StorageConfig

core = RAGCore(storage=StorageConfig.in_memory())
```

### 按 `.env` 自动装配 provider

如果你希望根据 `.env` 自动构建本地/云端 provider，可以使用工程装配入口：

```python
from pkp.interfaces._bootstrap import build_rag_core, load_settings

core = build_rag_core(load_settings())
```

说明：

- `pkp` 顶层公开的是纯库 API
- `interfaces._bootstrap` 是工程装配辅助入口，适合项目内使用

### 场景 1：本地 Ollama / BGE 全链路接入

这是最贴近“本地个人知识库”的用法：本地聊天模型负责生成，本地 BGE 负责 embedding 和 rerank。

先准备 `.env`：

```bash
PKP_OLLAMA__BASE_URL=http://localhost:11434
PKP_OLLAMA__CHAT_MODEL=qwen3.5:9b
PKP_OLLAMA__EMBEDDING_MODEL=qwen3-embedding:8b

PKP_LOCAL_BGE__ENABLED=true
PKP_LOCAL_BGE__EMBEDDING_MODEL=BAAI/bge-m3
PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH=~/.cache/huggingface/hub/models--BAAI--bge-m3
PKP_LOCAL_BGE__RERANK_MODEL=BAAI/bge-reranker-v2-m3
PKP_LOCAL_BGE__RERANK_MODEL_PATH=~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3

PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE=local_only
```

然后用库入口跑一条完整链路：

```python
from pkp.interfaces._bootstrap import build_rag_core, load_settings
from pkp.query import QueryOptions

core = build_rag_core(load_settings())

core.insert(
    source_type="markdown",
    location="README.md",
    owner="user",
)

result = core.query(
    "这个项目的核心架构是什么？",
    options=QueryOptions(mode="mix"),
)

print(result.answer.answer_text)
print(result.generation_provider)
print(result.generation_model)
print([item.citation_anchor for item in result.context.evidence[:3]])
```

这个场景下通常是：

- 本地 BGE 负责 chunk / entity / relation 向量和 rerank
- Ollama 聊天模型负责 grounded answer 生成
- `mix` 模式会同时使用图检索和普通 chunk 检索

### 场景 2：用 PDF 和网页做一次完整 ingest + query

下面这个示例更贴近日常知识整理：一份本地 PDF，加一篇网页文章，然后做综合查询。

```python
from pkp.interfaces._bootstrap import build_rag_core, load_settings
from pkp.query import QueryOptions

core = build_rag_core(load_settings())

pdf_result = core.insert(
    source_type="pdf",
    location="/absolute/path/annual-report.pdf",
    owner="user",
)

web_result = core.insert(
    source_type="web",
    location="https://example.com/industry-analysis",
    owner="user",
)

answer = core.query(
    "结合 PDF 报告和网页文章，总结 Alpha Engine 的业务重点和风险点。",
    options=QueryOptions(
        mode="hybrid",
        max_context_tokens=1800,
        max_evidence_chunks=10,
    ),
)

print(answer.answer.answer_text)
for item in answer.context.evidence[:5]:
    print(item.doc_id, item.citation_anchor, item.score)
```

如果你只想用 CLI 跑同样的流程：

```bash
uv run python -m pkp.interfaces.cli ingest --source-type pdf --location /absolute/path/annual-report.pdf
uv run python -m pkp.interfaces.cli ingest --source-type web --location https://example.com/industry-analysis
uv run python -m pkp.interfaces.cli query --mode fast --query "总结 Alpha Engine 的业务重点和风险点"
```

### 场景 3：用 delete / rebuild 管理知识库生命周期

当原始文件更新、索引异常、或者你想强制重建图和向量时，可以直接用 `location` 做生命周期管理。

```python
from pkp.interfaces._bootstrap import build_rag_core, load_settings

core = build_rag_core(load_settings())

core.insert(
    source_type="pdf",
    location="/absolute/path/design-doc.pdf",
    owner="user",
)

delete_result = core.delete(location="/absolute/path/design-doc.pdf")
print(delete_result.deleted_doc_ids)
print(delete_result.deleted_chunk_ids[:5])

rebuild_result = core.rebuild(location="/absolute/path/design-doc.pdf")
print(rebuild_result.rebuilt_doc_ids)
```

说明：

- `delete` 会清理该文档对应的 FTS、vector、graph 索引，并把文档状态落成 `DELETED`
- `rebuild` 会重新解析原始文档并重建 chunk、向量和图索引
- `delete / rebuild` 都支持三种选择器，但一次只能传一种：
  `doc_id`、`source_id`、`location`

如果你更习惯先看入库结果再做生命周期管理，可以这样写：

```python
result = core.insert(
    source_type="plain_text",
    location="memory://note-ops",
    owner="user",
    content_text="Alpha Engine v2 replaces the v1 ingest scheduler.",
)

core.delete(doc_id=result.document_id)
core.rebuild(location="memory://note-ops")
```

## 模型配置

### 本地 Ollama + BGE

```bash
PKP_OLLAMA__BASE_URL=http://localhost:11434
PKP_OLLAMA__CHAT_MODEL=qwen3.5:9b
PKP_OLLAMA__EMBEDDING_MODEL=qwen3-embedding:8b

PKP_LOCAL_BGE__ENABLED=true
PKP_LOCAL_BGE__EMBEDDING_MODEL=BAAI/bge-m3
PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH=~/.cache/huggingface/hub/models--BAAI--bge-m3
PKP_LOCAL_BGE__RERANK_MODEL=BAAI/bge-reranker-v2-m3
PKP_LOCAL_BGE__RERANK_MODEL_PATH=~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3

PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE=local_only
```

常见准备命令：

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:8b
```

### OpenAI 或兼容 OpenAI API

```bash
PKP_OPENAI__API_KEY=your-key
PKP_OPENAI__BASE_URL=https://api.openai.com/v1
PKP_OPENAI__MODEL=gpt-4.1-mini
PKP_OPENAI__EMBEDDING_MODEL=text-embedding-3-small
PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE=cloud_first
```

说明：

- 没有 chat provider 时，系统仍然会返回基于证据的 grounded answer，不会直接中断主流程
- embedding、rerank、generation 都支持本地优先或云端优先的装配方式

## CLI

CLI 现在是包装层，入口在 `pkp.interfaces.cli`。

健康检查：

```bash
uv run python -m pkp.interfaces.cli health
```

文档接入：

```bash
uv run python -m pkp.interfaces.cli ingest --source-type markdown --location README.md
uv run python -m pkp.interfaces.cli ingest --source-type pdf --location /absolute/path/demo.pdf
uv run python -m pkp.interfaces.cli ingest --source-type image --location /absolute/path/demo.png
uv run python -m pkp.interfaces.cli ingest --source-type web --location https://example.com/article
uv run python -m pkp.interfaces.cli ingest --source-type pasted_text --content "这里是一段笔记内容" --title "临时笔记"
```

统一文件入口：

```bash
uv run python -m pkp.interfaces.cli process-file --location /absolute/path/demo.pdf
```

查询：

```bash
uv run python -m pkp.interfaces.cli query --mode fast --query "这个项目做什么？"
uv run python -m pkp.interfaces.cli query --mode deep --query "比较 local 和 global 检索有什么差异？"
```

评测：

```bash
uv run python -m pkp.interfaces.cli evaluate-retrieval
uv run python -m pkp.interfaces.cli evaluate-file --location README.md --questions-file data/eval/questions.json
```

## FastAPI

FastAPI 仍然可用，但它只是包装层，入口在 `pkp.interfaces.api`。

启动方式：

```bash
uv run uvicorn pkp.interfaces.api:create_app --factory --reload
```

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"markdown","location":"README.md"}'

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"这个项目做什么？","mode":"fast"}'
```

## 实现说明

当前版本已经落地的关键点：

- `RAGCore` 已打通 `insert / query / delete / rebuild`
- 目录骨架已经按主流程收口，不再以 FastAPI、runtime 或 service 分层为中心
- `query` 侧已经有 `naive / local / global / hybrid / mix`
- `storage` 侧已经按 `KV / Vector / Graph / DocStatus` 组织
- `document` 和 `ingest` 侧保留了结构切块、token 切块、多模态 special chunk

当前仍然属于后续可继续加强的方向：

- 更强的跨文档实体消歧和规范化
- 更深的全局子图检索与图摘要
- 更完整的多模态图节点和跨模态关系建模

## 开发验证

常用验证命令：

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```
