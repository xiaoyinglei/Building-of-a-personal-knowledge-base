# RAG

一个以 **正式版 RAG / GraphRAG 后端** 为目标的项目骨架。这个仓库不把重点放在 demo 页面，而是把下面几条链路做完整并可评测：

- 查询主线：`查询理解 -> 检索编排 -> 融合/重排 -> 证据处理 -> 上下文构建 -> 生成`
- 入库主线：`文档解析 -> 切分 -> 抽取 -> 入库 -> 索引/图谱构建`
- Benchmark 主线：`公开数据下载 -> 统一格式转换 -> 正式 ingest -> retrieval baseline -> 离线诊断 -> answer eval`

当前最稳定的系统入口只有三个：

- [rag/runtime.py](/Users/leixiaoying/LLM/RAG学习/rag/runtime.py)：统一 Python 运行时入口
- [rag/cli.py](/Users/leixiaoying/LLM/RAG学习/rag/cli.py)：CLI 入口
- [rag/workbench/server.py](/Users/leixiaoying/LLM/RAG学习/rag/workbench/server.py)：浏览器工作台入口

## 项目根目录

```text
RAG学习/
├── README.md                                  # 项目说明、目录地图、命令、评测说明
├── pyproject.toml                             # 项目依赖与工具配置
├── uv.lock                                    # uv 锁文件
├── rag/                                       # 正式代码主目录
├── scripts/                                   # benchmark / profiling / diagnose 入口脚本
├── tests/                                     # 回归测试
├── data/                                      # 统一数据根目录：benchmark 数据、索引、评测结果
└── .venv/                                     # 本地虚拟环境（通常不提交）
```

## `rag/` 目录结构

```text
rag/
├── runtime.py                                 # 系统唯一运行时入口：组装 storage / ingest / retrieval / generation
├── cli.py                                     # CLI 命令入口：ingest/query/rebuild/benchmark/workbench
├── benchmarks.py                              # 公开 benchmark 主逻辑：下载、prepare、ingest、retrieval eval、profiling
├── benchmark_diagnostics.py                   # retrieval 后置诊断：failure analysis、branch diagnostics、recommendations
├── answer_benchmarks.py                       # answer-level eval：证据一致性 + judge 子集正确性评测
│
├── assembly/                                  # 能力装配中心
│   ├── __init__.py                            # 对外导出 assembly 能力
│   ├── models.py                              # AssemblyRequest / ProviderConfig / CapabilityBundle 等模型
│   ├── bindings.py                            # embedding/chat/rerank 的 capability binding
│   ├── service.py                             # CapabilityAssemblyService：按 profile/requirements 装配 provider
│   ├── support.py                             # profile、环境变量兼容、provider 默认策略
│   └── tokenizer.py                           # tokenizer contract、token accounting
│
├── ingest/                                    # 入库主线
│   ├── pipeline.py                            # IngestPipeline：source/document/chunk/vector/fts/graph 串联总控
│   ├── parser.py                              # source_type -> parser 调度入口
│   ├── chunking.py                            # 正式切分入口：chunk route、多模态/结构化/token 切分
│   ├── extract.py                             # 实体/关系抽取与图谱前处理
│   ├── policy.py                              # ingest 策略解析
│   ├── parsers/                               # 各类具体解析器实现
│   └── chunkers/                              # 具体 chunking 实现细节
│
├── retrieval/                                 # 查询时主线
│   ├── analysis.py                            # QueryUnderstanding + RoutingDecision + access policy narrowing
│   ├── orchestrator.py                        # RetrievalService：branch 检索、fusion、rerank、graph/web 补检索
│   ├── evidence.py                            # evidence filtering、bundle、自检、artifact suggestion
│   ├── context.py                             # prompt build、context budget、truncation
│   ├── graph.py                               # graph retrieval / graph expansion
│   └── models.py                              # QueryOptions、RetrievalResult、BuiltContext 等查询侧模型
│
├── providers/                                 # 模型/推理适配层
│   ├── adapters.py                            # OpenAI/Ollama/Local-BGE 等 provider repo 实现
│   ├── embedding.py                           # embedding 适配导出入口
│   ├── rerank.py                              # rerank 适配导出入口
│   └── generation.py                          # Grounded answer 生成、citation 构造、fallback、judge prompt
│
├── schema/                                    # 跨层共享契约，只放公共模型
│   ├── core.py                                # Document / Chunk / Source / Content 等核心实体
│   ├── query.py                               # QueryUnderstanding / GroundedAnswer / EvidenceItem / Citation 等
│   └── runtime.py                             # AccessPolicy / RetrievalDiagnostics / ProviderAttempt / telemetry 相关模型
│
├── storage/                                   # 存储抽象与后端实现
│   ├── __init__.py                            # StorageBundle 组装根
│   ├── vector.py                              # 向量存储抽象入口
│   ├── graph.py                               # 图谱存储抽象入口
│   ├── metadata.py                            # 元数据存储抽象入口
│   ├── search.py                              # FTS / web search 抽象入口
│   ├── object_store.py                        # 对象存储抽象入口
│   ├── cache.py                               # 缓存能力入口
│   ├── search_backends/                       # sqlite / pgvector / milvus / web search 等具体实现
│   ├── repositories/                          # sqlite/postgres/object store 等 repository 实现
│   └── graph_backends/                        # sqlite / neo4j 图谱后端实现
│
├── utils/                                     # 通用工具
│   ├── text.py                                # 文本处理、token/span/fts 查询辅助
│   └── telemetry.py                           # 运行指标与评测聚合辅助
│
└── workbench/                                 # 浏览器工作台
    ├── models.py                              # workbench 请求/响应模型
    ├── service.py                             # workbench 服务层
    ├── server.py                              # HTTP 服务入口
    └── static/                                # 前端静态资源
```

## `data/` 目录结构

所有 benchmark 数据、索引和评测输出统一放在 `data/` 下。

```text
data/
├── benchmarks/
│   ├── fiqa/
│   │   ├── raw/                               # 原始公开数据：corpus / queries / qrels
│   │   ├── prepared/
│   │   │   ├── full/                          # 统一格式 JSONL（完整集）
│   │   │   └── mini/                          # 统一格式 JSONL（小子集）
│   │   ├── index/                             # 预留检索库目录（当前未保留正式索引快照）
│   │   └── eval/                              # 预留评测目录（当前未保留正式结果快照）
│   │
│   └── medical_retrieval/
│       ├── raw/                               # C-MTEB MedicalRetrieval 原始数据
│       ├── prepared/
│       │   ├── full/                          # full 文档/问题/qrels JSONL
│       │   └── mini/                          # mini 文档/问题/qrels JSONL
│       ├── index/
│       │   ├── mini/                          # 历史参考线索引：BAAI/bge-m3 + sqlite
│       │   ├── mini-milvus-bge-v2/            # 低时延线索引壳：BAAI/bge-m3 + Milvus
│       │   └── mini-milvus-qwen8b-v1/         # 质量优先线索引壳：qwen3-embedding:8b + Milvus
│       ├── eval/
│       │   ├── retrieval/
│       │   │   └── mini/                      # 仅保留 3 条 retrieval 基线的历史与逐 query 输出
│       │   └── ingest/                        # ingest profiling 输出（按需生成，不长期保留）
│       └── subsets/                           # 预留实验子集目录（当前已清空）
│
└── eval/
    └── baselines/
        └── medical_retrieval.json             # 单数据集总表：当前保留的 retrieval 基线、主线标记、核心对比
```

## 主线路：代码如何流转

### 1. 运行时装配主线

入口是 [rag/runtime.py](/Users/leixiaoying/LLM/RAG学习/rag/runtime.py)。

流转顺序：

1. `RAGRuntime.from_request()` / `RAGRuntime.from_profile()`
2. 调 [rag/assembly/service.py](/Users/leixiaoying/LLM/RAG学习/rag/assembly/service.py) 的 `CapabilityAssemblyService`
3. 根据 [rag/assembly/support.py](/Users/leixiaoying/LLM/RAG学习/rag/assembly/support.py) 的 profile 和环境变量，装配：
   - embedding binding
   - chat binding
   - rerank binding
   - tokenizer contract
4. `storage.build()` 构建 storage bundle
5. `runtime._build_pipelines()` 组装：
   - ingest pipeline
   - delete / rebuild pipeline
   - retrieval service
   - query pipeline

你可以把 `runtime.py` 理解成：
**全系统唯一的组合根**。

---

### 2. 入库主线

入口通常来自：
- CLI：`uv run rag ingest ...`
- benchmark ingest：`scripts/ingest_public_benchmark.py`
- Python：`runtime.insert(...)` / `runtime.insert_many(...)`

代码流转：

1. [rag/cli.py](/Users/leixiaoying/LLM/RAG学习/rag/cli.py) / [scripts/ingest_public_benchmark.py](/Users/leixiaoying/LLM/RAG学习/scripts/ingest_public_benchmark.py)
2. [rag/runtime.py](/Users/leixiaoying/LLM/RAG学习/rag/runtime.py)
   - `runtime.insert()` 或 `runtime.insert_many()`
3. [rag/ingest/pipeline.py](/Users/leixiaoying/LLM/RAG学习/rag/ingest/pipeline.py)
   - `IngestPipeline`
4. [rag/ingest/parser.py](/Users/leixiaoying/LLM/RAG学习/rag/ingest/parser.py)
   - 根据 `source_type` 分派到 `ingest/parsers/*`
5. [rag/ingest/chunking.py](/Users/leixiaoying/LLM/RAG学习/rag/ingest/chunking.py)
   - 文档结构分析
   - segment -> chunk
   - benchmark metadata 透传（如 `benchmark_doc_id`）
6. [rag/ingest/extract.py](/Users/leixiaoying/LLM/RAG学习/rag/ingest/extract.py)
   - 实体/关系抽取
7. [rag/storage/*](/Users/leixiaoying/LLM/RAG学习/rag/storage)
   - metadata repo
   - vector repo
   - FTS repo
   - graph repo
   - object store

入库主线最终产物：
- `document`
- `chunk`
- `vector`
- `fts`
- `graph`
- `status`

---

### 3. 查询主线

入口通常来自：
- CLI：`uv run rag query ...`
- workbench
- benchmark retrieval eval
- benchmark answer eval

代码流转：

1. [rag/runtime.py](/Users/leixiaoying/LLM/RAG学习/rag/runtime.py)
   - `runtime.query(query, options=QueryOptions(...))`
2. `_QueryPipeline.run()`
3. [rag/retrieval/orchestrator.py](/Users/leixiaoying/LLM/RAG学习/rag/retrieval/orchestrator.py)
   - `RetrievalService.retrieve(...)`
4. [rag/retrieval/analysis.py](/Users/leixiaoying/LLM/RAG学习/rag/retrieval/analysis.py)
   - Query Understanding
   - RoutingDecision
   - access policy 收缩
5. [rag/retrieval/orchestrator.py](/Users/leixiaoying/LLM/RAG学习/rag/retrieval/orchestrator.py)
   - branch 检索
   - RRF / fusion
   - rerank
   - graph / web 等补召回
6. [rag/retrieval/evidence.py](/Users/leixiaoying/LLM/RAG学习/rag/retrieval/evidence.py)
   - evidence filtering
   - bundle
   - self-check
7. [rag/runtime.py](/Users/leixiaoying/LLM/RAG学习/rag/runtime.py)
   - `_build_bounded_context()`
8. [rag/retrieval/context.py](/Users/leixiaoying/LLM/RAG学习/rag/retrieval/context.py)
   - context truncation
   - prompt build
9. [rag/providers/generation.py](/Users/leixiaoying/LLM/RAG学习/rag/providers/generation.py)
   - grounded candidate
   - answer generation
   - citation / evidence link 组装
10. 返回 [RAGQueryResult](/Users/leixiaoying/LLM/RAG学习/rag/retrieval/models.py)
    - `answer`
    - `retrieval`
    - `context`
    - `generation_provider`
    - `generation_model`

---

### 4. Retrieval Benchmark 主线

入口：
- [scripts/download_public_benchmark.py](/Users/leixiaoying/LLM/RAG学习/scripts/download_public_benchmark.py)
- [scripts/prepare_public_benchmark.py](/Users/leixiaoying/LLM/RAG学习/scripts/prepare_public_benchmark.py)
- [scripts/ingest_public_benchmark.py](/Users/leixiaoying/LLM/RAG学习/scripts/ingest_public_benchmark.py)
- [scripts/evaluate_retrieval_benchmark.py](/Users/leixiaoying/LLM/RAG学习/scripts/evaluate_retrieval_benchmark.py)

代码流转：

1. `scripts/*`
2. [rag/benchmarks.py](/Users/leixiaoying/LLM/RAG学习/rag/benchmarks.py)
3. `build_runtime_for_benchmark(...)`
4. 复用正式：
   - ingest pipeline
   - retrieval service
5. 落盘：
   - `baseline.csv`
   - `per_query.jsonl`
   - `run_summary.json`
   - `run_history.jsonl`
   - `runs/<run_id>/...`

这条链主要回答：
- 检索能不能命中 gold
- 排名好不好
- 模式、embedding、rerank 对 retrieval 指标的影响

---

### 5. Retrieval 离线诊断主线

入口：
- [scripts/diagnose_benchmark_run.py](/Users/leixiaoying/LLM/RAG学习/scripts/diagnose_benchmark_run.py)

代码流转：

1. 脚本读取已有 retrieval run
2. [rag/benchmark_diagnostics.py](/Users/leixiaoying/LLM/RAG学习/rag/benchmark_diagnostics.py)
3. `BenchmarkDiagnosticsPostProcessor`
4. 复跑 retrieval 主链并补 branch 信息
5. 落盘：
   - `failure_analysis.jsonl`
   - `failure_summary.json`
   - `branch_diagnostics.jsonl`
   - `branch_summary.json`
   - `recall_failure_profile.json`
   - `rerank_profile.json`
   - `full_text_profile.json`
   - `recommendations.json`

这条链主要回答：
- 当前失败主要是召回失败还是排序失败
- full-text 是否真有独立价值
- rerank 在帮哪些 query、伤哪些 query
- 下一步更应该投哪一层

---

### 6. Answer Benchmark 主线

入口：
- [scripts/evaluate_answer_benchmark.py](/Users/leixiaoying/LLM/RAG学习/scripts/evaluate_answer_benchmark.py)

代码流转：

1. 脚本读取 benchmark 的 `queries/qrels/documents`
2. [rag/answer_benchmarks.py](/Users/leixiaoying/LLM/RAG学习/rag/answer_benchmarks.py)
3. 对每个 query 调 `runtime.query(...)`
4. 基于 `RAGQueryResult` 生成：
   - `AnswerPerQueryRecord`
   - 证据一致性统计
5. 对固定子集运行 judge：
   - 本地 judge 初筛
   - 强模型复核（可选）
6. 落盘：
   - `per_query_answers.jsonl`
   - `evidence_consistency_summary.json`
   - `judge_subset.jsonl`
   - `judge_summary.json`
   - `answer_recommendations.json`
   - `run_summary.json`
   - `baseline.csv`

这条链主要回答：
- 生成有没有基于证据
- 检索命中了但答案为什么还没答出来
- 哪些 query 更适合改 prompt / answer guard

默认索引选择：
- 对 `medical_retrieval mini`
  - 不传 `--storage-root` 且不传 embedding override 时，脚本默认走低时延线：
    - `data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2`
    - `vector_backend=milvus`
    - `vector_collection_prefix=medical_retrieval_mini_bge_v2`
  - 如果显式传：
    - `--embedding-provider ollama`
    - `--embedding-model qwen3-embedding:8b`
    脚本默认切到质量优先线：
    - `data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1`
    - `vector_backend=milvus`
    - `vector_collection_prefix=medical_retrieval_mini_qwen8b_v1`

注意：
- answer benchmark 会复用**现有索引**
- 但不会复用旧的 retrieval run 结果
- 它会对每个 query 重新跑一次完整 `runtime.query(...)`
  - retrieval
  - rerank
  - context build
  - grounded answer generation
- 所以 answer benchmark 的时延通常明显高于 retrieval benchmark

## 当前保留的 retrieval 基线

先看这个文件：

- [medical_retrieval.json](/Users/leixiaoying/LLM/RAG学习/data/eval/baselines/medical_retrieval.json)

它是当前 `medical_retrieval` 数据集唯一的收口总表。以后看结果，先看它，不用先翻：
- `baseline.csv`
- `run_history.jsonl`
- `per_query.jsonl`

### 质量优先线

- embedding：`qwen3-embedding:8b`
- mode：`naive`
- chunk：`480/64`
- chunk_top_k：`20`
- rerank：`on`
- rerank_pool_k：`10`
- 向量后端：`Milvus`
- 索引目录：`data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1`

### 低时延线

- embedding：`BAAI/bge-m3`
- mode：`naive`
- chunk：`480/64`
- chunk_top_k：`20`
- rerank：`on`
- rerank_pool_k：`10`
- 向量后端：`Milvus`
- 索引目录：`data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2`

### 历史参考线

- embedding：`BAAI/bge-m3`
- 向量后端：`sqlite`
- 索引目录：`data/benchmarks/medical_retrieval/index/mini`
- 用途：保留旧主线对照，不再作为当前低时延主线

## 指标说明

### Retrieval 指标

- `Recall@10`
  - 前 10 个结果里有没有找回 gold 文档
- `MRR@10`
  - 第一个正确文档排得有多靠前
- `NDCG@10`
  - top10 的整体排序质量
- `avg_latency_ms`
  - 平均单 query 时延
- `p95_latency_ms`
  - 95 分位时延，反映尾部慢查询
- `queries_per_second`
  - 检索吞吐

### Ingest Profiling 指标

- `docs_per_second`
  - 每秒入库多少文档
- `chunks_per_second`
  - 每秒处理多少 chunk
- `embedding_elapsed_ms`
  - embedding 编码阶段耗时

### Diagnostics 指标

- `top1_hit_ratio`
  - 第 1 名就是正确文档的比例
- `top10_hit_but_low_rank_ratio`
  - 前 10 命中，但排位不够靠前的比例
- `top10_miss_ratio`
  - 前 10 完全 miss 的比例
- `recall_failure_ratio`
  - 召回层根本没找回 gold 的比例
- `fusion_loss_ratio`
  - 某分支召回了 gold，但融合后丢掉的比例
- `mapping_failure_ratio`
  - chunk -> benchmark_doc_id 映射异常的比例
- `independent_hit_ratio`
  - 某分支独立命中 gold 的比例
- `rerank_helped_query_count`
  - rerank 帮到的 query 数
- `rerank_hurt_query_count`
  - rerank 伤到的 query 数

### Answer Benchmark 指标

- `evidence_consistent_rate`
  - 回答是否总体与给定证据一致
- `grounded_answer_rate`
  - 回答是否以 grounded answer 结构产出
- `citation_presence_rate`
  - 是否带 citation
- `citation_gold_hit_rate`
  - citation 映射到 benchmark doc 后，是否覆盖 gold 文档
- `answer_correct_rate`
  - judge 子集上的答案正确率
- `avg_generation_latency_ms`
  - 生成阶段平均耗时，只统计 chat / answer generation
- `p95_generation_latency_ms`
  - 生成阶段 95 分位耗时
- `avg_non_generation_latency_ms`
  - 非生成阶段平均耗时，近似等于 retrieval + rerank + context build
- `p95_non_generation_latency_ms`
  - 非生成阶段 95 分位耗时
- `avg_context_evidence_count`
  - 生成时实际送入 prompt 的 evidence 条数
- `avg_context_token_count`
  - 生成时实际送入 prompt 的上下文 token 数

## 常用命令

### 1. 下载与准备 benchmark

FiQA：

```bash
uv run python scripts/download_public_benchmark.py --dataset fiqa
uv run python scripts/prepare_public_benchmark.py --dataset fiqa
```

MedicalRetrieval：

```bash
uv run python scripts/download_public_benchmark.py --dataset medical_retrieval
uv run python scripts/prepare_public_benchmark.py --dataset medical_retrieval
```

### 2. 正式 ingest

如果用 Milvus，先设置：

```bash
export RAG_MILVUS_URI=http://127.0.0.1:19530
```

低时延线：

```bash
uv run python scripts/ingest_public_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_bge_v2 \
  --batch-size 32 \
  --embedding-batch-size 8 \
  --embedding-device mps \
  --chunk-token-size 480 \
  --chunk-overlap-tokens 64 \
  --skip-graph-extraction
```

质量优先线：

```bash
uv run python scripts/ingest_public_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --batch-size 32 \
  --embedding-batch-size 8 \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b \
  --chunk-token-size 480 \
  --chunk-overlap-tokens 64 \
  --skip-graph-extraction
```

### 3. Retrieval baseline

低时延线：

```bash
uv run python scripts/evaluate_retrieval_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_bge_v2 \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10
```

质量优先线：

```bash
uv run python scripts/evaluate_retrieval_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b
```

### 4. Retrieval 离线诊断

低时延线：

```bash
uv run python scripts/diagnose_benchmark_run.py \
  --dataset medical_retrieval \
  --variant mini \
  --run-id medical_retrieval-20260412T155018448051Z \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_bge_v2 \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --rerank \
  --rerank-pool-k 10
```

质量优先线：

```bash
uv run python scripts/diagnose_benchmark_run.py \
  --dataset medical_retrieval \
  --variant mini \
  --run-id medical_retrieval-20260413T022639961261Z \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --rerank \
  --rerank-pool-k 10 \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b
```

### 5. Answer benchmark

低时延线：

```bash
uv run python scripts/evaluate_answer_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-bge-v2 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_bge_v2 \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --judge-subset-size 250
```

质量优先线：

```bash
uv run python scripts/evaluate_answer_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --storage-root data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1 \
  --vector-backend milvus \
  --vector-collection-prefix medical_retrieval_mini_qwen8b_v1 \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b \
  --judge-subset-size 250
```

如果你只是想直接走当前默认 Milvus 线，也可以省略 `--storage-root / --vector-backend / --vector-collection-prefix`：

- 默认低时延线：
```bash
uv run python scripts/evaluate_answer_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --judge-subset-size 250
```

- 默认质量优先线：
```bash
uv run python scripts/evaluate_answer_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --embedding-provider ollama \
  --embedding-model qwen3-embedding:8b \
  --judge-subset-size 250
```

只做证据一致性，不跑 judge：

```bash
uv run python scripts/evaluate_answer_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --judge-subset-size 0 \
  --disable-review
```

做 `answer_context_top_k` A/B：

```bash
uv run python scripts/evaluate_answer_benchmark.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --mode naive \
  --top-k 10 \
  --chunk-top-k 20 \
  --retrieval-pool-k 20 \
  --rerank-pool-k 10 \
  --answer-context-top-k 4 \
  --judge-subset-size 0
```

把 `4` 改成 `6 / 8` 即可做生成侧 A/B。这个参数只影响生成用 evidence，不影响 retrieval `chunk_top_k=20`。

### 6. ingest profiling

```bash
uv run python scripts/profile_benchmark_ingest.py \
  --dataset medical_retrieval \
  --variant mini \
  --profile local_full \
  --doc-counts 100 \
  --ingest-batch-sizes 32 \
  --encode-batch-sizes 8,16,32,64 \
  --embedding-device mps \
  --skip-graph-extraction
```

## 当前实验结论（基于已跑结果）

### Retrieval

已收敛：
- `mode = naive`
- `chunk = 480/64`
- `chunk_top_k = 20`
- `rerank = on`
- `rerank_pool_k = 10`

已证伪或已收敛：
- `stream` 入库不优于 `preload`
- `local / global / hybrid` 没独立价值
- `mix` 比 `naive` 更差
- `chunk_top_k=20/30/40` 分数相同
- `rerank_pool_k=20/40` 不涨分，只拉高时延
- `parent_backfill` 无收益
- `256` 小 chunk 系列没有优于 `480/64`

当前保留的 3 条 retrieval 基线：
- `BAAI/bge-m3 + sqlite`
  - `Recall@10 = 0.776667`
  - `MRR@10 = 0.690972`
  - `NDCG@10 = 0.712199`
  - `avg_latency_ms = 2472.225`
- `BAAI/bge-m3 + milvus`
  - `Recall@10 = 0.67`
  - `MRR@10 = 0.588259`
  - `NDCG@10 = 0.608173`
  - `avg_latency_ms = 563.793`
- `qwen3-embedding:8b + milvus`
  - `Recall@10 = 0.82`
  - `MRR@10 = 0.705854`
  - `NDCG@10 = 0.733644`
  - `avg_latency_ms = 695.559`

当前主线选择：
- 新质量优先线：`qwen3-embedding:8b + milvus`
- 新低时延线：`BAAI/bge-m3 + milvus`
- `BAAI/bge-m3 + sqlite` 只作为历史参考

### Diagnostics

已知事实：
- 当前 retrieval 主问题是 `recall_failure`
- `fusion_loss = 0`
- `mapping_failure = 0`
- `vector` 是唯一主力分支
- `full_text` 有极窄场景价值
- `local/global` 可以继续降级处理
- `rerank` 有净收益，但副作用不小，后续更适合做 guard 实验

### Answer Eval

目前已经确认：
- 生成层真正的问题不是“纯乱编”，而是：
  - 检索命中了但仍然模板化拒答
  - citation benchmark 映射之前被内部 `document-*` 污染，现在已修正为优先用 `benchmark_doc_id`
  - 本地长上下文生成非常慢，不适合直接当客户在线链路
- 新增的 `answer_context_top_k` 用来只压 generation context，不影响 retrieval 结果，便于做 `4 / 6 / 8` A/B
- answer benchmark 现在会额外记录：
  - `avg_generation_latency_ms`
  - `avg_non_generation_latency_ms`
  - `avg_context_evidence_count`
  - `avg_context_token_count`
  这样可以直接判断慢点主要在生成，还是在 retrieval/context build

## 下一步更值得投的方向

按当前诊断和实验结果，优先级建议：

1. `query normalization / alias expansion`
   - 先解决 recall failure 的表达问题
2. `conditional full-text`
   - 不恢复常驻 hybrid，只做条件触发式稀疏检索
3. `rerank guard`
   - 不默认所有 query 都强 rerank
4. `answer_context_top_k` A/B
   - 降生成时延，减少“检索命中但生成拒答”
5. `在线链路和评测链路分线`
   - benchmark 用高质量参数
   - 在线用更小上下文、更轻 chat 模型或云端 chat

不建议优先再投：
- `hybrid / mix` 重写
- `local / global` 调参
- `fusion` 重写
- 再开新的 embedding 大范围扫参

## Workbench

启动：

```bash
uv run rag workbench --storage-root .rag
```

它适合做：
- 文档导入
- 单 query 调试
- 查看 retrieval diagnostics
- 查看 context evidence
- 快速验证 answer/citation 输出

## 测试

常用：

```bash
uv run pytest -q
uv run ruff check rag scripts tests
python3 -m py_compile $(find rag scripts tests -name '*.py' -not -path '*/__pycache__/*')
```

如果只验证 benchmark 相关：

```bash
uv run pytest \
  tests/core/test_public_benchmark.py \
  tests/core/test_benchmark_diagnostics.py \
  tests/core/test_benchmark_diagnostics_context.py \
  tests/core/test_answer_benchmarks.py -q
```
