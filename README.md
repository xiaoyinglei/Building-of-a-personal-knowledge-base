# RAG

一个以核心 RAG 后端为中心的仓库。现在的推荐用法已经统一到 `assembly -> runtime -> ingest/query/workbench` 这条链路上。

当前公开面分成 3 层：

- `rag/`：核心库
- `RAGRuntime`：唯一推荐的新主入口
- `rag` CLI / `rag workbench`：基于 `RAGRuntime` 的本地入口

## 先看结论

如果你只是想把项目跑起来并开始测试，按这个顺序：

1. `uv sync`
2. `uv run rag profiles`
3. 选一个 profile
4. `uv run rag ingest ...`
5. `uv run rag query ... --json`
6. 或者直接 `uv run rag workbench ...` 在浏览器里测试

推荐入口是：

- `RAGRuntime.from_profile(...)`
- `RAGRuntime.from_request(...)`

`legacy_embedding_bindings` 这条兼容链已经移除。`EmbeddingProviderBinding` 不再是使用方式，`RAG` 也不再是公开入口。

## 安装

要求：

- Python `>=3.12,<3.14`
- 推荐使用 [`uv`](https://docs.astral.sh/uv/)

安装：

```bash
uv sync
```

确认 CLI 可用：

```bash
uv run rag --help
uv run rag profiles
```

## 现在支持什么

- 文档源：`plain_text`、`markdown`、`pdf`、`docx`、`pptx`、`xlsx`、`image`、`web`
- 查询模式：`bypass`、`naive`、`local`、`global`、`hybrid`、`mix`
- 图谱操作：`insert_custom_kg()`、`upsert_node()`、`upsert_edge()`
- 多存储后端：`sqlite`、`postgres`、`pgvector`、`milvus`、`neo4j`、`redis`、`s3/minio/local object store`
- 浏览器工作台：真实读写本地目录，左侧文件树、中间证据/路由、右侧问答

## 推荐入口

### CLI

适合快速跑通：

- `uv run rag profiles`
- `uv run rag ingest`
- `uv run rag query`
- `uv run rag delete`
- `uv run rag rebuild`
- `uv run rag workbench`

### Python

适合正式接入：

- `RAGRuntime.from_profile(...)`
- `RAGRuntime.from_request(...)`

## 推荐 profile

先用这个命令看当前环境下有哪些可用 profile：

```bash
uv run rag profiles
```

当前内置的推荐 profile：

| profile | 用途 |
| --- | --- |
| `local_full` | 本地 chat + embedding + rerank，适合纯本地 |
| `local_retrieval_cloud_chat` | 本地 embedding/rerank + 云端 chat，适合你现在这种本地检索、Gemini/OpenAI 生成 |
| `cloud_full` | 云端 chat + embedding，适合全云 |
| `test_minimal` | 最小测试档，允许降级，适合不接真实模型先跑链路 |

## 3 分钟跑通 CLI

### 1. 建一个独立索引目录

```bash
export RAG_DEMO_ROOT=.rag-demo
rm -rf "$RAG_DEMO_ROOT"
```

### 2. 导入一段文本

```bash
uv run rag ingest \
  --storage-root "$RAG_DEMO_ROOT" \
  --profile test_minimal \
  --source-type plain_text \
  --location memory://note-1 \
  --content "Alpha Engine handles ingestion. Beta Service depends on Alpha Engine."
```

### 3. 查询

```bash
uv run rag query \
  --storage-root "$RAG_DEMO_ROOT" \
  --profile test_minimal \
  --mode mix \
  --query "What does Alpha Engine handle?" \
  --json
```

### 4. 删除或重建

```bash
uv run rag delete --storage-root "$RAG_DEMO_ROOT" --location memory://note-1
uv run rag rebuild --storage-root "$RAG_DEMO_ROOT" --location memory://note-1
```

## 一个完整的文档问答流程

下面这套流程是最适合你现在直接照抄的。

### 第 1 步：先看 profile

```bash
uv run rag profiles
```

如果你已经配了 Gemini/OpenAI 之类的云端聊天模型，同时希望本地 embedding/rerank 优先，用：

```bash
--profile local_retrieval_cloud_chat
```

如果只是先看链路能不能跑通，用：

```bash
--profile test_minimal
```

### 第 2 步：导入文档

英文文档：

```bash
uv run rag ingest \
  --storage-root .rag \
  --profile test_minimal \
  --source-type markdown \
  --location /Users/leixiaoying/LLM/RAG学习/data/test_corpus/tech_docs/hf_transformers_document_question_answering.md
```

中文文档：

```bash
uv run rag ingest \
  --storage-root .rag \
  --profile test_minimal \
  --source-type markdown \
  --location /Users/leixiaoying/LLM/RAG学习/data/test_corpus/tech_docs/chinese_enterprise_rag_practice_guide.md
```

注意：

- ingest 的 JSON 直接打印到终端，不会自动写文件
- 真正落盘的是 `.rag/` 下的 `metadata.sqlite3`、`vectors.sqlite3`、`graph.sqlite3`、`fts.sqlite3`、`objects/`
- 如果你想保存 ingest 结果，要显式重定向：

```bash
uv run rag ingest ... | tee ingest-result.json
```

### 第 3 步：开始问

英文文档先用英文问，贴近原文：

```bash
uv run rag query \
  --storage-root .rag \
  --profile test_minimal \
  --mode mix \
  --query "What is Document Question Answering?"
```

```bash
uv run rag query \
  --storage-root .rag \
  --profile test_minimal \
  --mode mix \
  --query "What dataset is used in this guide?"
```

中文文档可以先问这些：

```bash
uv run rag query \
  --storage-root .rag \
  --profile test_minimal \
  --mode mix \
  --query "第一批核心资料包括哪些内容？"
```

```bash
uv run rag query \
  --storage-root .rag \
  --profile test_minimal \
  --mode mix \
  --query "图谱中至少维护哪些节点类型？"
```

```bash
uv run rag query \
  --storage-root .rag \
  --profile test_minimal \
  --mode mix \
  --query "试点阶段的默认 chunk 参数是什么？"
```

### 第 4 步：看证据，不只看 answer

```bash
uv run rag query \
  --storage-root .rag \
  --profile test_minimal \
  --mode mix \
  --query "试点阶段的默认 chunk 参数是什么？" \
  --json > query.json
```

你最该先看的字段：

- `answer.answer_text`
- `answer.insufficient_evidence_flag`
- `context.evidence`
- `context.evidence[0].citation_anchor`
- `context.evidence[0].section_path`
- `retrieval.diagnostics.branch_hits`

快速打印证据：

```bash
uv run python - <<'PY'
import json
data = json.load(open("query.json", "r", encoding="utf-8"))
for i, item in enumerate(data["context"]["evidence"][:5], 1):
    print(f"[{i}] score={item['score']:.3f}")
    print("citation_anchor =", item.get("citation_anchor"))
    print("section_path =", " > ".join(item.get("section_path", [])))
    print("text =", item.get("text", "")[:220])
    print()
PY
```

### 第 5 步：如果回答不稳，先改问法

当前系统最稳的问法：

- 定义题：`What is X?`
- 参数题：`What is the model checkpoint?`
- 流程题：`How does the guide say to run inference?`
- 章节题：`In the "Load the data" section, what dataset is used?`
- 中文事实题：`第一批核心资料包括哪些内容？`
- 中文结构题：`在“图谱建模原则”部分提到了哪些节点类型？`

## 浏览器工作台完整流程

这是最适合你现在在网页端测试的入口。

### 1. 启动 workbench

```bash
uv run rag workbench \
  --storage-root .rag \
  --workspace-root /Users/leixiaoying/LLM/RAG学习/data/test_corpus/tech_docs
```

启动后会打印一个本地地址，例如：

```text
http://127.0.0.1:8765
```

浏览器打开它。

### 2. 页面怎么用

左侧：

- 真实本地文档目录
- 新建 Markdown、上传文件、重建、删除
- 页面改文件会直接同步到本地磁盘
- 本地外部增删改文件，页面会自动同步

中间：

- 命中的证据卡
- 每条证据的分数
- `citation_anchor`
- `section_path`
- `retrieval_family`
- `routing / budgets / diagnostics`

右侧：

- 用左右箭头切换 profile
- 选择查询模式
- 输入问题
- 只显示用户提问和 LLM 回复

### 3. 推荐的网页测试顺序

1. 先切到一个 profile  
   没配真实模型时先用 `test_minimal`；配了 Gemini 之后优先试 `local_retrieval_cloud_chat`

2. 在左侧选中文档  
   比如 `chinese_enterprise_rag_practice_guide.md`

3. 在右侧提问

```text
第一批核心资料包括哪些内容？
```

```text
图谱中至少维护哪些节点类型？
```

```text
试点阶段的默认 chunk 参数是什么？
```

4. 到中间看证据卡  
   不要只看右侧答案，要确认：
   - 分数是不是合理
   - `citation_anchor` 是否落在对应章节
   - `section_path` 是否匹配问题语义
   - `diagnostics.branch_hits` 有没有命中你预期的分支

### 4. 网页端和本地文件是否同步

是同步的，而且是双向的：

- 网页里新建、上传、删除文件，会直接改本地文件系统
- 本地外部修改工作目录，页面会自动同步
- workbench 会同步更新索引：新增会 ingest，改动会 rebuild，删除会删索引

## Python API

## 最小推荐写法：按 profile 创建 runtime

```python
from rag import CapabilityRequirements, RAGRuntime, StorageConfig

runtime = RAGRuntime.from_profile(
    storage=StorageConfig(root=".rag"),
    profile_id="test_minimal",
    requirements=CapabilityRequirements(
        require_chat=False,
        default_context_tokens=1200,
    ),
)

try:
    runtime.insert(
        source_type="plain_text",
        location="memory://note-1",
        owner="demo",
        content_text="Alpha Engine handles ingestion.",
    )
    result = runtime.query("What does Alpha Engine handle?")
    print(result.answer.answer_text)
finally:
    runtime.close()
```

## 显式装配：按 request 创建 runtime

```python
from rag import RAGRuntime, StorageConfig
from rag.llm import (
    AssemblyConfig,
    AssemblyOverrides,
    AssemblyRequest,
    CapabilityRequirements,
    ProviderConfig,
    TokenizerConfig,
)

runtime = RAGRuntime.from_request(
    storage=StorageConfig(root=".rag"),
    request=AssemblyRequest(
        profile_id="cloud_full",
        requirements=CapabilityRequirements(
            require_chat=True,
            require_embedding=True,
            default_context_tokens=1200,
        ),
        config=AssemblyConfig(
            tokenizer=TokenizerConfig(
                embedding_model_name="text-embedding-3-small",
                tokenizer_model_name="text-embedding-3-small",
                chunking_tokenizer_model_name="text-embedding-3-small",
                chunk_token_size=480,
                chunk_overlap_tokens=64,
                max_context_tokens=1200,
                prompt_reserved_tokens=256,
            ),
        ),
        overrides=AssemblyOverrides(
            chat=ProviderConfig(
                provider_kind="openai-compatible",
                location="cloud",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                api_key="YOUR_API_KEY",
                chat_model="gemini-2.5-pro",
            ),
            embedding=ProviderConfig(
                provider_kind="openai-compatible",
                location="cloud",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                api_key="YOUR_API_KEY",
                embedding_model="text-embedding-3-small",
            ),
        ),
    ),
)

try:
    print(runtime.diagnostics_payload())
finally:
    runtime.close()
```

## 内容列表注入

```python
from pathlib import Path

from rag import CapabilityRequirements, RAGRuntime, StorageConfig
from rag.ingest.ingest import DirectContentItem

runtime = RAGRuntime.from_profile(
    storage=StorageConfig.in_memory(),
    profile_id="test_minimal",
    requirements=CapabilityRequirements(require_chat=False),
)

try:
    result = runtime.insert_content_list(
        [
            DirectContentItem(
                location="memory://alpha.txt",
                source_type="plain_text",
                content="Alpha Engine supports Beta Service.",
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
    runtime.close()
```

## 模型与环境变量

assembly 现在是唯一的模型决策入口。模型选择优先级固定为：

1. 显式参数
2. profile
3. 新版结构化配置
4. 兼容旧 env
5. 默认值

### 推荐环境变量

OpenAI-compatible / Gemini：

```bash
export OPENAI_API_KEY=...
export GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
export GEMINI_CHAT_MODEL=gemini-2.5-pro
```

Ollama：

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_CHAT_MODEL=qwen3.5:14b
export OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b
```

本地 BGE：

```bash
export RAG_LOCAL_BGE_ENABLED=1
export RAG_LOCAL_BGE_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

统一 token / chunking 合同：

```bash
export RAG_EMBEDDING_MODEL=text-embedding-3-small
export RAG_TOKENIZER_MODEL=text-embedding-3-small
export RAG_CHUNKING_TOKENIZER_MODEL=text-embedding-3-small
export RAG_CHUNK_TOKEN_SIZE=480
export RAG_CHUNK_OVERLAP_TOKENS=64
export RAG_MAX_CONTEXT_TOKENS=1200
export RAG_PROMPT_RESERVED_TOKENS=256
```

### 兼容旧 env

旧键名还会被 assembly 翻译成 compatibility input，例如：

- `PKP_OPENAI__API_KEY`
- `PKP_OPENAI__BASE_URL`
- `PKP_OPENAI__MODEL`
- `PKP_OPENAI__EMBEDDING_MODEL`
- `PKP_OLLAMA__BASE_URL`
- `PKP_OLLAMA__CHAT_MODEL`
- `PKP_OLLAMA__EMBEDDING_MODEL`
- `PKP_LOCAL_BGE__ENABLED`

但这些已经只是兼容输入，不再是推荐配置方式。

## tokenizer / embedding / index contract

这个项目现在强约束下面这件事：

- 索引阶段和查询阶段必须使用同一个 embedding model

同时下面这些合同也由同一条 assembly 链统一治理：

- embedding model
- tokenizer model
- chunking tokenizer model
- max context tokens
- chunk token size
- overlap
- prompt reserved tokens

如果你换了 embedding 或 tokenizer 合同，而索引里还是旧合同，运行时会报 contract mismatch，而不是偷偷混用。

## 存储后端

最小本地模式：

- metadata：SQLite
- vectors：SQLite
- graph：SQLite
- cache：metadata
- object store：本地目录
- FTS：SQLite

正式组件化模式支持：

| 组件 | 后端 |
| --- | --- |
| metadata / documents / chunks | `sqlite`、`postgres` |
| vectors | `sqlite`、`pgvector`、`milvus` |
| graph | `sqlite`、`neo4j` |
| cache | `memory`、`redis` |
| object store | `local`、`s3`、`minio` |
| fts | `sqlite`、`postgres` |

## 常见问题

### 1. 为什么 ingest 后没看到 JSON 文件

因为 CLI 只把 JSON 打到终端，不会自动落盘。  
要保存请用：

```bash
uv run rag ingest ... | tee ingest-result.json
```

### 2. 为什么目录里只有 sqlite3 文件

这说明索引已经落盘了，不说明“没建图”。  
本地默认模式本来就把图存到 `graph.sqlite3`。

### 3. 我怎么知道是否真的接上了聊天模型

CLI：

```bash
uv run rag query ... --json > query.json
```

然后看：

- `generation_provider`
- `generation_model`
- `retrieval.diagnostics.rerank_provider`

### 4. 我怎么知道证据是不是命中了

看：

- `context.evidence`
- `citation_anchor`
- `section_path`
- `retrieval.diagnostics.branch_hits`

workbench 里这些信息在中间栏直接可见。

## 入口治理状态

长期保留：

- `RAGRuntime.from_profile(...)`
- `RAGRuntime.from_request(...)`
- `AssemblyRequest / CapabilityRequirements / AssemblyConfig / AssemblyOverrides`

过渡兼容：

- 旧 env 键名

已经移除：

- `legacy_embedding_bindings`
- `EmbeddingProviderBinding` 作为运行时主入口
- `RAG(...)` 作为公开入口
