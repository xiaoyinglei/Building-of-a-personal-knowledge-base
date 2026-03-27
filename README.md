# 个人知识平台

一个以可靠性为优先的个人知识平台，用来完成资料接入、索引构建、检索问答、深度研究和知识沉淀。

项目采用严格分层架构：

`Types -> Config -> Repo -> Service -> Runtime -> UI`

## 架构

- `Types`：领域模型、枚举、请求响应契约
- `Config`：配置加载、默认策略、运行参数
- `Repo`：解析、存储、检索、图谱、模型适配
- `Service`：ingest、chunking、retrieval、evidence、artifact 等领域逻辑
- `Runtime`：Fast Path、Deep Path、artifact promotion、session orchestration
- `UI`：FastAPI 和 CLI 对外入口

## 核心模块

- `Ingest Pipeline`：接入 PDF、Markdown、纯文本、图片、网页和内联内容
- `Index Layer`：构建 metadata、FTS、vector、graph 等索引
- `Model Gateway`：统一接入 OpenAI、Ollama，并支持 fallback
- `Retrieval Orchestrator`：融合全文检索、向量检索、章节检索、图扩展
- `Research Agent`：支持 Fast Path 和 Deep Path，两种研究路径
- `Knowledge Layer`：支持 artifact 生成、审批、重索引和复用

## 如何使用

安装依赖：

```bash
uv sync --all-extras
cp .env.example .env
```

项目会自动读取根目录 `.env`。

## Ollama 接入

1. 确保本地 Ollama 服务可用
2. 准备一个聊天模型，例如 `qwen3.5:9b`
3. 在 `.env` 中配置：

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

如果你本地已经装好了 Ollama，常见准备命令是：

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:8b
```

说明：

- 配好后，Deep Path 和需要 synthesis 的回答会优先走本地 `Qwen`
- embedding 和 rerank 会优先走本地 HuggingFace/FlagEmbedding 的 `bge-m3` 与 `bge-reranker-v2-m3`
- `PKP_OLLAMA__EMBEDDING_MODEL` 会保留为可选回退，不再主导向量索引
- 如果你只想本地运行，建议把 `PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE` 设成 `local_only`

## 云端大模型接入

如果你要接 OpenAI 或兼容 OpenAI API 的云端网关，在 `.env` 中配置：

```bash
PKP_OPENAI__API_KEY=your-key
PKP_OPENAI__BASE_URL=https://api.openai.com/v1
PKP_OPENAI__MODEL=gpt-4.1-mini
PKP_OPENAI__EMBEDDING_MODEL=text-embedding-3-small
PKP_RUNTIME__EXECUTION_LOCATION_PREFERENCE=cloud_first
```

说明：

- 官方 OpenAI 直接用默认 `BASE_URL` 即可
- 如果你用兼容 OpenAI 的第三方网关，只需要把 `PKP_OPENAI__BASE_URL` 改成对应地址
- 没有云端 key 也没有本地模型时，项目仍可运行 ingest、索引、检索和 artifact 主流程，问答会退化为 `retrieval-only`

## 如何上传文档

支持这些 `source_type`：

- `markdown`
- `pdf`
- `docx`
- `plain_text`
- `pasted_text`
- `image`
- `web`
- `browser_clip`

CLI 示例：

```bash
uv run python -m pkp.ui.cli health

# 本地 Markdown / 文本 / PDF / 图片
uv run python -m pkp.ui.cli ingest --source-type markdown --location README.md
uv run python -m pkp.ui.cli ingest --source-type plain_text --location data/samples/plain-notes.txt
uv run python -m pkp.ui.cli ingest --source-type pdf --location /absolute/path/demo.pdf
uv run python -m pkp.ui.cli ingest --source-type docx --location /absolute/path/demo.docx
uv run python -m pkp.ui.cli ingest --source-type image --location /absolute/path/screenshot.png

# 统一文档解析与切分入口
uv run python -m pkp.ui.cli process-file --location /absolute/path/demo.pdf
uv run python -m pkp.ui.cli process-file --location /absolute/path/demo.md
uv run python -m pkp.ui.cli process-file --location /absolute/path/demo.docx
uv run python -m pkp.ui.cli process-file --location /absolute/path/demo.png

# 远程网页
uv run python -m pkp.ui.cli ingest --source-type web --location https://example.com/article

# 直接粘贴内容
uv run python -m pkp.ui.cli ingest --source-type pasted_text --content "这里是一段笔记内容" --title "临时笔记"

# 查询
uv run python -m pkp.ui.cli query --mode fast --query "这个项目做什么？"
uv run python -m pkp.ui.cli query --mode deep --query "比较 Fast Path 和 Deep Path"
uv run python -m pkp.ui.cli list-artifacts
```

说明：

- `--location` 可以是相对路径、绝对路径，或网页 URL
- 不传 `--location` 时，可以直接用 `--content` 走内联 ingest
- ingest 后文档会进入本地索引和对象存储，默认目录是 `data/runtime`

启动 API：

```bash
uv run uvicorn pkp.ui.api.app:create_app --factory --reload
```

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"markdown","location":"README.md"}'

curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"web","location":"https://example.com/article"}'

curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type":"pasted_text","content":"这里是一段笔记内容","title":"临时笔记"}'

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"这个项目做什么？","mode":"fast"}'
```

## 开发验证

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run python -m scripts.check_repo_only_imports
```
