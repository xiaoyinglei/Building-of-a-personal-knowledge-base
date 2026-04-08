# RAG

一个正式版的 RAG / GraphRAG 后端骨架，主线已经收敛到两条：

- 查询主线：`查询理解 -> 路由/检索 -> 融合/重排 -> 证据处理 -> 上下文构建 -> 生成`
- 入库主线：`文档解析 -> 切分 -> 抽取 -> 入库 -> 索引/图谱构建`

当前推荐入口只有三个：

- `rag/runtime.py`：唯一系统入口
- `uv run rag ...`：CLI 入口
- `uv run rag workbench ...`：浏览器工作台入口

## 目录结构

说明：`assembly` 现在保留为一个小目录，而不是单文件 `assembly.py`，原因很简单，它已经形成稳定子系统，强行压成一个文件只会变成新的神文件。

```text
rag/
├── runtime.py                     # 系统入口：统一组装 runtime、ingest、query、delete、rebuild
├── cli.py                         # 命令行入口：profiles / ingest / query / delete / rebuild / workbench
├── assembly/                      # 装配中心：profile、provider 选择、tokenizer/runtime contract
│   ├── __init__.py                # 对外导出 Assembly API
│   ├── models.py                  # AssemblyRequest / CapabilityRequirements / Diagnostics 等装配模型
│   ├── service.py                 # CapabilityAssemblyService：解析 profile、选择 provider、生成 capability bundle
│   ├── support.py                 # 内置 profile、环境变量兼容、默认装配策略
│   ├── bindings.py                # chat / embedding / rerank 的 capability binding
│   └── tokenizer.py               # tokenizer contract 与 token accounting
├── schema/                        # 数据契约：只放跨层共享的模型，不放业务逻辑
│   ├── core.py                    # Document / Chunk / Segment / Source / GraphNode 等核心实体
│   ├── query.py                   # QueryUnderstanding / EvidenceItem / GroundedAnswer / QueryRequest 等查询契约
│   └── runtime.py                 # AccessPolicy / Diagnostics / RuntimeMode / Status / Telemetry 等运行契约
├── ingest/                        # 文档入库层：解析、切分、抽取、写入索引
│   ├── pipeline.py                # 入库总流程：insert / insert_many / rebuild / delete 的主执行链
│   ├── parser.py                  # 解析调度：按 source_type 选择 parser
│   ├── chunking.py                # 切分调度：结构化切分、多模态切分、token 切分的统一入口
│   ├── extract.py                 # 实体/关系抽取与图谱写入前处理
│   ├── policy.py                  # 入库策略：重复内容、内容哈希、处理策略等
│   ├── parsers/                   # 具体解析实现
│   │   ├── docling_parser_repo.py # Docling 文档解析
│   │   ├── pdf_parser_repo.py     # PDF 解析
│   │   ├── markdown_parser_repo.py# Markdown 解析
│   │   ├── plain_text_parser_repo.py # 纯文本解析
│   │   ├── image_parser_repo.py   # 图片/OCR 解析
│   │   ├── web_fetch_repo.py      # 网页抓取
│   │   ├── web_parser_repo.py     # 网页正文解析
│   │   └── util.py                # 解析相关文本清洗与通用辅助
│   └── chunkers/                  # 具体切分实现
│       ├── structured_chunker.py  # 结构感知切分
│       ├── multimodal_chunk_router.py # 多模态 chunk 路由
│       └── token_chunker.py       # token 级切分
├── retrieval/                     # 检索编排层：查询理解、检索调度、证据处理、上下文构建
│   ├── analysis.py                # Query Understanding、policy narrowing、RoutingDecision
│   ├── orchestrator.py            # 多路检索编排：branch collect、fusion、rerank、graph/web 补检索
│   ├── evidence.py                # 证据过滤、bundle、自检、artifact suggestion
│   ├── context.py                 # 上下文构建：budget control、truncation、prompt build
│   ├── graph.py                   # query-time graph retrieval / graph expansion
│   └── models.py                  # QueryMode / QueryOptions / RetrievalResult / RAGQueryResult
├── providers/                     # 基础设施适配：模型能力本身，不负责 runtime 装配
│   ├── generation.py              # LLM 生成、grounded answer、回答结构化输出
│   ├── embedding.py               # embedding provider 适配
│   ├── rerank.py                  # reranker 适配与排序流水线
│   └── adapters.py                # OpenAI / Gemini / Ollama / Local BGE 等供应商适配
├── storage/                       # 持久化与搜索
│   ├── vector.py                  # 向量存储抽象与聚合入口
│   ├── graph.py                   # 图存储抽象与聚合入口
│   ├── metadata.py                # 文档、chunk、状态等元数据存储聚合
│   ├── search.py                  # FTS / web search 聚合入口
│   ├── object_store.py            # 文件对象存储聚合入口
│   ├── cache.py                   # cache 抽象与聚合入口
│   ├── search_backends/           # 具体搜索/向量实现
│   │   ├── sqlite_vector_repo.py  # SQLite 向量索引
│   │   ├── sqlite_fts_repo.py     # SQLite 全文检索
│   │   ├── postgres_fts_repo.py   # Postgres FTS
│   │   ├── pgvector_vector_repo.py# PGVector
│   │   ├── milvus_vector_repo.py  # Milvus
│   │   ├── in_memory_vector_repo.py # 内存向量实现
│   │   └── web_search_repo.py     # 外部搜索实现
│   ├── graph_backends/            # 具体图存储实现
│   │   ├── sqlite_graph_repo.py   # SQLite graph backend
│   │   └── neo4j_graph_repo.py    # Neo4j backend
│   └── repositories/              # 具体元数据 / cache / object store 实现
│       ├── sqlite_metadata_repo.py# SQLite metadata repo
│       ├── postgres_metadata_repo.py # Postgres metadata repo
│       ├── redis_cache_repo.py    # Redis cache repo
│       ├── file_object_store.py   # 本地文件对象存储
│       └── s3_object_store.py     # S3 / MinIO 对象存储
├── utils/                         # 底层工具：不放主业务，只放基础设施
│   ├── text.py                    # 文本清洗、切句、FTS query、token 辅助
│   └── telemetry.py               # 遥测事件与记录辅助
└── workbench/                     # 浏览器工作台：文件树、检索证据、问答调试
    ├── server.py                  # HTTP 服务入口
    ├── service.py                 # workbench 后端逻辑
    ├── models.py                  # workbench 数据结构
    └── static/                    # 前端静态资源
```

## 这个项目现在能做什么

- 文档导入：`plain_text`、`markdown`、`pdf`、`docx`、`pptx`、`xlsx`、`image`、`web`
- 查询模式：`bypass`、`naive`、`local`、`global`、`hybrid`、`mix`
- Query Understanding：显式硬约束抽取 + LLM structured output
- Retrieval：多 branch 检索、RRF / fusion、rerank、graph expansion、web 补检索
- Evidence：过滤、自检、上下文预算控制、citation-grounded answer
- Storage：`sqlite`、`postgres`、`pgvector`、`milvus`、`neo4j`、`redis`、`s3/minio/local object store`
- 调试：CLI JSON 输出 + 浏览器 workbench + retrieval diagnostics

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

## 推荐 profile

先查看当前环境下有哪些 profile：

```bash
uv run rag profiles
```

当前默认会根据环境变量和本地 provider 情况生成推荐 profile，常见的是：

| profile | 用途 |
| --- | --- |
| `local_full` | 本地 chat + embedding + rerank |
| `local_retrieval_cloud_chat` | 本地检索，云端回答 |
| `cloud_full` | 云端 chat + embedding |
| `test_minimal` | 最小测试链路，适合不接真实模型先跑通 |

## 快速开始

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

### 3. 发起查询

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

## 怎么用：只看检索效果

如果你现在的目标不是回答写得漂不漂亮，而是看“检索有没有把正确证据找回来”，做法很简单：

1. 用你准备上线的检索 profile，不要用纯测试 profile 误判效果
2. 查询时加 `--json`
3. 重点看 `retrieval` 和 `context`，先不要盯 `answer.answer_text`

示例：

```bash
uv run rag query \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --mode mix \
  --query "第一批核心资料包括哪些内容？" \
  --json > retrieval.json
```

你应该优先检查这些字段：

- `retrieval.diagnostics.branch_hits`：各 branch 命中了多少候选
- `retrieval.diagnostics.reranked_chunk_ids`：rerank 后保留下来的 chunk 顺序
- `retrieval.decision`：本次查询的路由/执行决策
- `retrieval.self_check`：证据是否足够、是否建议继续检索
- `retrieval.diagnostics.query_understanding`：Query Understanding 最终结果
- `context.evidence`：真正进入生成上下文的证据列表

可以直接这样看：

```bash
jq '.retrieval.diagnostics.branch_hits' retrieval.json
jq '.retrieval.self_check' retrieval.json
jq '.retrieval.decision' retrieval.json
jq '.context.evidence[] | {chunk_id, score, file_name, section_path, retrieval_channels}' retrieval.json
```

判断检索效果时，最值得看的不是一句 answer，而是：

- 正确文档有没有被召回
- 正确 chunk 有没有排到前面
- `context.evidence` 里有没有混进明显无关证据
- `self_check.evidence_sufficient` 是否稳定
- `branch_hits` 和 `reranked_chunk_ids` 是否符合你的预期

如果你要做更严肃的 retrieval eval，建议固定一批 query，然后比较：

- `mode=local`
- `mode=global`
- `mode=hybrid`
- `mode=mix`

看哪种模式下 `context.evidence` 最稳定地包含 gold chunk。

## 怎么用：端到端 RAG 检验

如果你要看的是完整 RAG 效果，而不是单纯检索，检查顺序应该是：

1. `answer.answer_text`：回答本身是否正确
2. `answer.citations`：引用是否真的落在对的 chunk 上
3. `answer.groundedness_flag`：回答是否被证据支撑
4. `answer.insufficient_evidence_flag`：证据不足时是否明确拒答/降级
5. `context.token_count / token_budget`：上下文是否超预算或被截断得过狠
6. `generation_provider / generation_model`：本次到底用了哪个生成模型

示例：

```bash
uv run rag query \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --mode mix \
  --query "这个方案为什么强调证据优先，而不是先做复杂路由？" \
  --json > rag-result.json
```

建议重点看：

```bash
jq '.answer.answer_text' rag-result.json
jq '.answer.citations' rag-result.json
jq '.answer.groundedness_flag, .answer.insufficient_evidence_flag' rag-result.json
jq '.context.token_budget, .context.token_count, .context.truncated_count' rag-result.json
jq '.generation_provider, .generation_model' rag-result.json
```

如果你要做端到端对比，建议固定：

- 同一批 query
- 同一批文档
- 同一 profile
- 分别比较 `mix / local / hybrid`

然后记录：

- answer 是否正确
- citation 是否合理
- groundedness 是否为 `true`
- insufficient evidence 时是否乱答

## 从头到尾的完整流程

### 方案 A：CLI 跑完整链路

#### 第 1 步：看 profile

```bash
uv run rag profiles
```

#### 第 2 步：导入文档

```bash
uv run rag ingest \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --source-type markdown \
  --location data/test_corpus/tech_docs/chinese_enterprise_rag_practice_guide.md
```

或者：

```bash
uv run rag ingest \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --source-type pdf \
  --location /absolute/path/to/your.pdf
```

#### 第 3 步：发起查询

```bash
uv run rag query \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --mode mix \
  --query "第一批核心资料包括哪些内容？" \
  --json > query-result.json
```

#### 第 4 步：看结果，不要只看一段 answer

先看回答：

```bash
jq '.answer.answer_text' query-result.json
```

再看证据：

```bash
jq '.context.evidence[] | {chunk_id, file_name, section_path, score}' query-result.json
```

再看诊断：

```bash
jq '.retrieval.diagnostics' query-result.json
```

#### 第 5 步：如果文档更新，做 rebuild

```bash
uv run rag rebuild \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --location data/test_corpus/tech_docs/chinese_enterprise_rag_practice_guide.md
```

#### 第 6 步：如果不再需要，做 delete

```bash
uv run rag delete \
  --storage-root .rag \
  --profile local_retrieval_cloud_chat \
  --location data/test_corpus/tech_docs/chinese_enterprise_rag_practice_guide.md
```

### 方案 B：浏览器工作台

启动：

```bash
uv run rag workbench \
  --storage-root .rag \
  --workspace-root data/test_corpus/tech_docs
```

工作台适合做三件事：

- 左侧看文件树和索引状态
- 中间看检索证据、路由信息、context budget
- 右侧直接问答，快速观察 answer / evidence / diagnostics 是否一致

如果你在调 retrieval，而不是调回答文案，workbench 的中间栏比终端更高效。

## Python API

### 推荐写法：按 profile 创建 runtime

```python
from rag import CapabilityRequirements, RAGRuntime, StorageConfig
from rag.retrieval import QueryOptions

runtime = RAGRuntime.from_profile(
    storage=StorageConfig(root=".rag"),
    profile_id="local_retrieval_cloud_chat",
    requirements=CapabilityRequirements(
        require_embedding=True,
        require_chat=False,
        require_rerank=True,
    ),
)

try:
    runtime.insert(
        source_type="markdown",
        location="data/test_corpus/tech_docs/chinese_enterprise_rag_practice_guide.md",
        owner="user",
    )

    result = runtime.query(
        "第一批核心资料包括哪些内容？",
        options=QueryOptions(mode="mix", max_context_tokens=1200),
    )

    print(result.answer.answer_text)
    print(result.retrieval.diagnostics.branch_hits)
    print([item.chunk_id for item in result.context.evidence])
finally:
    runtime.close()
```

### 批量注入内容列表

```python
from pathlib import Path

from rag import CapabilityRequirements, RAGRuntime, StorageConfig
from rag.ingest.pipeline import DirectContentItem

runtime = RAGRuntime.from_profile(
    storage=StorageConfig(root=".rag"),
    profile_id="test_minimal",
    requirements=CapabilityRequirements(require_chat=False),
)

try:
    result = runtime.insert_content_list(
        [
            DirectContentItem(
                location="memory://note-1",
                source_type="plain_text",
                content="Alpha Engine handles ingestion.",
            ),
            DirectContentItem(
                location="memory://note-2",
                source_type="markdown",
                content=Path("data/test_corpus/tech_docs/chinese_enterprise_rag_practice_guide.md"),
            ),
        ]
    )
    print(result.success_count, result.failure_count)
finally:
    runtime.close()
```

### 显式装配：按 request 创建 runtime

```python
from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageConfig

runtime = RAGRuntime.from_request(
    storage=StorageConfig(root=".rag"),
    request=AssemblyRequest(
        requirements=CapabilityRequirements(
            require_embedding=True,
            require_chat=False,
            require_rerank=True,
        )
    ),
)

try:
    result = runtime.query("What does Alpha Engine handle?")
    print(result.retrieval.diagnostics.branch_hits)
finally:
    runtime.close()
```

## 常用环境变量

云端 chat / OpenAI-compatible：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4.1-mini
```

Gemini 兼容写法：

```bash
export GEMINI_API_KEY=...
export GEMINI_CHAT_MODEL=gemini-2.5-pro
```

Ollama：

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_CHAT_MODEL=qwen3:14b
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

本地 BGE 检索：

```bash
export RAG_LOCAL_BGE_ENABLED=1
export RAG_LOCAL_BGE_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

tokenizer / chunk / context：

```bash
export RAG_TOKENIZER_MODEL=text-embedding-3-small
export RAG_CHUNK_TOKEN_SIZE=480
export RAG_CHUNK_OVERLAP_TOKENS=64
export RAG_MAX_CONTEXT_TOKENS=1200
export RAG_PROMPT_RESERVED_TOKENS=256
```

## 怎么判断自己现在到底在测什么

如果你现在在做的是：

- 检索评估：重点看 `retrieval.*` 和 `context.evidence`
- RAG 评估：同时看 `answer.*`、`context.*`、`retrieval.*`
- 解析/切分评估：重点看 ingest 结果、chunk 数量、graph 抽取、rebuild 前后差异

最容易犯的错是：

- 用 `answer.answer_text` 代替 retrieval 评估
- 用 `test_minimal` 的结果判断真实上线效果
- 不看 `citations / groundedness / insufficient_evidence_flag`
- 不保留 `--json` 结果，导致后面没法回溯 branch hits 和证据链

## 一句话建议

- 想先跑通：用 `test_minimal`
- 想看检索：看 `retrieval` 和 `context.evidence`
- 想看端到端效果：看 `answer + citations + groundedness + diagnostics`
- 想最高效调试：开 `workbench`
