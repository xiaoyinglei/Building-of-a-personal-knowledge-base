from __future__ import annotations

from html import escape

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from pkp.interfaces._ui.dependencies import get_request_container

router = APIRouter()

_NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("query", "Query / 提问", "/workbench/query"),
    ("ingest", "Ingest / 导入", "/workbench/ingest"),
    ("artifacts", "Artifacts / 产物", "/workbench/artifacts"),
    ("system", "System / 系统", "/workbench/system"),
)


def _render_layout(
    *,
    active: str,
    eyebrow: str,
    title: str,
    description: str,
    body: str,
) -> HTMLResponse:
    nav = "".join(
        (
            f'<a class="wb-nav-link{" is-active" if key == active else ""}" href="{href}">{label}</a>'
            for key, label, href in _NAV_ITEMS
        )
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PKP Workbench · {escape(title)}</title>
    <link rel="stylesheet" href="/workbench/assets/workbench.css" />
  </head>
  <body data-page="{escape(active)}">
    <div class="wb-shell">
      <aside class="wb-sidebar">
        <a class="wb-brand" href="/workbench/query">
          <span class="wb-brand-mark">PKP</span>
          <span class="wb-brand-copy">
            <strong>PKP Workbench</strong>
            <span>Knowledge Base Test Console / 知识库测试台</span>
          </span>
        </a>
        <nav class="wb-nav">{nav}</nav>
        <section class="wb-sidebar-card">
          <h2>用途 / Use It For</h2>
          <p>上传资料、检查检索命中、追踪 provider fallback，并快速判断知识库质量是否真的在提升。</p>
        </section>
      </aside>
      <main class="wb-main">
        <header class="wb-page-head">
          <p class="wb-eyebrow">{escape(eyebrow)}</p>
          <h1>{escape(title)}</h1>
          <p class="wb-description">{escape(description)}</p>
        </header>
        {body}
      </main>
    </div>
    <script src="/workbench/assets/workbench.js" defer></script>
  </body>
</html>
"""
    return HTMLResponse(html)


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/workbench/query", status_code=307)


@router.get("/workbench", include_in_schema=False)
def workbench_root() -> RedirectResponse:
    return RedirectResponse(url="/workbench/query", status_code=307)


@router.get("/workbench/query", response_class=HTMLResponse, include_in_schema=False)
def query_page() -> HTMLResponse:
    return _render_layout(
        active="query",
        eyebrow="Workspace",
        title="Query Workspace",
        description=(
            "Ask the knowledge base a question, compare fast and deep paths, "
            "inspect grounded answers plus citations, and keep a lightweight evaluation sheet on the page."
        ),
        body="""
        <section class="wb-grid wb-grid-query">
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Ask / 提问</h2>
              <button class="wb-button wb-button-secondary" id="query-reset" type="button">Reset</button>
            </div>
            <form id="query-form" class="wb-form">
              <label>
                Question / 问题
                <textarea name="query" rows="6" placeholder="比如：这个项目现在的检索链路有哪些明显短板？"></textarea>
              </label>
              <div class="wb-form-row">
                <label>
                  Mode / 模式
                  <select name="mode">
                    <option value="deep" selected>Deep</option>
                    <option value="fast">Fast</option>
                  </select>
                </label>
                <label>
                  Session ID / 会话 ID
                  <input name="session_id" placeholder="default" />
                </label>
              </div>
              <label>
                Source Scope / 限定来源
                <input name="source_scope" placeholder="doc-a,doc-b" />
              </label>
              <div class="wb-form-row">
                <label>
                  Latency Budget / 延迟预算
                  <input name="latency_budget" type="number" min="1" placeholder="optional" />
                </label>
                <label>
                  Token Budget / Token 预算
                  <input name="token_budget" type="number" min="1" placeholder="optional" />
                </label>
              </div>
              <div class="wb-form-row">
                <label>
                  Location Preference / 执行位置偏好
                  <select name="execution_location_preference">
                    <option value="">Default</option>
                    <option value="local_first">Local First</option>
                    <option value="cloud_first">Cloud First</option>
                    <option value="local_only">Local Only</option>
                  </select>
                </label>
                <label class="wb-checkbox">
                  <input name="fallback_allowed" type="checkbox" checked />
                  Allow Fallback / 允许回退
                </label>
              </div>
              <label class="wb-checkbox">
                <input name="compare_modes" type="checkbox" checked />
                Compare Fast / Deep
              </label>
              <button class="wb-button" type="submit">Run Query / 开始查询</button>
            </form>
            <section class="wb-subpanel">
              <h3>参数说明</h3>
              <div class="wb-list">
                <article class="wb-list-item">
                  <h4>Fast / Deep</h4>
                  <p>
                    Fast：优先快速检索与抽取式回答，必要时才升级到 Deep。
                    Deep：会根据证据做综合回答，更适合评估知识库质量。
                  </p>
                </article>
                <article class="wb-list-item">
                  <h4>Cloud First / Local First / Local Only</h4>
                  <p>
                    Cloud First：优先云端执行检索和综合回答；若对应向量空间缺失或 provider 不可用，
                    会回退到本地。Local First：优先本地。Local Only：只允许本地。
                  </p>
                </article>
                <article class="wb-list-item">
                  <h4>Source Scope / Session ID</h4>
                  <p>
                    Source Scope 用逗号分隔 `doc_id` 或 `source_id`，用于缩小检索范围。
                    Session ID 用于串联 Deep 模式的子问题、证据矩阵和记忆线索。
                  </p>
                </article>
              </div>
            </section>
            <section class="wb-subpanel">
              <h3>Evaluation Pad / 测评辅助</h3>
              <form id="query-eval-form" class="wb-form">
                <label>
                  Expected Terms / 期望关键词
                  <input name="expected_terms" placeholder="比如：月返抽查, 整改跟进, 异常复核" />
                </label>
                <label>
                  Expected Chunk IDs / 期望命中 chunk
                  <input name="expected_chunk_ids" placeholder="比如：chunk-a, table-1" />
                </label>
                <label>
                  Expected Special Types / 特殊块类型
                  <input name="expected_special_types" placeholder="比如：table, image_summary" />
                </label>
              </form>
              <div id="query-eval" class="wb-list wb-empty">运行查询后，这里会生成页面端测评提示。</div>
            </section>
          </section>
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Answer Comparison / 回答对比</h2>
              <span class="wb-status" id="query-status">空闲</span>
            </div>
            <div id="query-run-summary" class="wb-metric-grid wb-empty">运行后，这里会显示本次测评摘要。</div>
            <div id="query-results" class="wb-result-grid wb-empty">运行一次查询后，这里会显示结果卡片。</div>
            <section class="wb-subpanel">
              <h3>Evidence / 证据</h3>
              <div id="query-evidence" class="wb-list wb-empty">还没有证据。</div>
            </section>
          </section>
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Diagnostics / 诊断</h2>
              <button class="wb-button wb-button-secondary" id="query-copy-json" type="button">
                Copy JSON / 复制 JSON
              </button>
            </div>
            <div id="query-diagnostics-cards" class="wb-metric-grid wb-empty">还没有诊断信息。</div>
            <section class="wb-subpanel">
              <h3>Query Understanding / 查询理解</h3>
              <div id="query-understanding" class="wb-list wb-empty">还没有查询理解结果。</div>
            </section>
            <section class="wb-subpanel">
              <h3>Citations / 引用</h3>
              <div id="query-citations" class="wb-list wb-empty">还没有引用。</div>
            </section>
            <section class="wb-subpanel">
              <h3>Attempts / 调用记录</h3>
              <div id="query-attempts" class="wb-list wb-empty">还没有 provider 调用记录。</div>
            </section>
            <section class="wb-subpanel">
              <h3>Deep Session Trace / Deep 会话轨迹</h3>
              <div id="query-session" class="wb-list wb-empty">使用 Deep 模式后，这里会显示子问题和证据矩阵。</div>
            </section>
            <section class="wb-subpanel">
              <h3>Raw Response / 原始响应</h3>
              <pre id="query-raw" class="wb-code wb-empty">还没有响应。</pre>
            </section>
          </section>
        </section>
        """,
    )


@router.get("/workbench/ingest", response_class=HTMLResponse, include_in_schema=False)
def ingest_page() -> HTMLResponse:
    return _render_layout(
        active="ingest",
        eyebrow="Operations",
        title="Ingest Console",
        description=(
            "Upload local files or paste source text directly. "
            "Watch new documents appear without touching the CLI."
        ),
        body="""
        <section class="wb-grid wb-grid-ingest">
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Upload File / 上传文件</h2>
              <span class="wb-status" id="upload-status">空闲</span>
            </div>
            <form id="upload-form" class="wb-form">
              <label>
                File / 文件
                <input name="file" type="file" required />
              </label>
              <label>
                Title / 标题
                <input name="title" placeholder="Optional display title" />
              </label>
              <button class="wb-button" type="submit">Upload And Ingest / 上传并导入</button>
            </form>
          </section>
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Paste Content / 粘贴内容</h2>
              <span class="wb-status" id="inline-status">空闲</span>
            </div>
            <form id="inline-ingest-form" class="wb-form">
              <div class="wb-form-row">
                <label>
                  Source Type / 来源类型
                  <select name="source_type">
                    <option value="markdown">Markdown</option>
                    <option value="plain_text">Plain Text</option>
                    <option value="pasted_text">Pasted Text</option>
                  </select>
                </label>
                <label>
                  Title / 标题
                  <input name="title" placeholder="Optional title" />
                </label>
              </div>
              <label>
                Content / 内容
                <textarea name="content" rows="12" placeholder="直接粘贴一段文档内容。"></textarea>
              </label>
              <button class="wb-button" type="submit">Ingest Text / 导入文本</button>
            </form>
          </section>
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Recent Sources / 最近资料</h2>
              <button class="wb-button wb-button-secondary" id="sources-refresh" type="button">Refresh / 刷新</button>
            </div>
            <div id="sources-list" class="wb-list wb-empty">还没有资料。</div>
          </section>
        </section>
        """,
    )


@router.get("/workbench/artifacts", response_class=HTMLResponse, include_in_schema=False)
def artifacts_page() -> HTMLResponse:
    return _render_layout(
        active="artifacts",
        eyebrow="Review",
        title="Artifacts",
        description=(
            "Inspect promoted knowledge artifacts, approve candidates, "
            "and open the raw payload when you need to audit quality."
        ),
        body="""
        <section class="wb-grid wb-grid-artifacts">
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Artifact Queue / 产物队列</h2>
              <button class="wb-button wb-button-secondary" id="artifacts-refresh" type="button">Refresh / 刷新</button>
            </div>
            <div id="artifacts-list" class="wb-list wb-empty">还没有 artifact。</div>
          </section>
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Artifact Detail / 产物详情</h2>
              <span class="wb-status" id="artifact-status">空闲</span>
            </div>
            <pre id="artifact-detail" class="wb-code wb-empty">选择一个 artifact 后，这里会显示详情。</pre>
          </section>
        </section>
        """,
    )


@router.get("/workbench/system", response_class=HTMLResponse, include_in_schema=False)
def system_page() -> HTMLResponse:
    return _render_layout(
        active="system",
        eyebrow="Diagnostics",
        title="System Overview",
        description=(
            "Check provider health, vector/index counts, "
            "and recent ingested sources before trusting the answers."
        ),
        body="""
        <section class="wb-grid wb-grid-system">
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Health / 健康检查</h2>
              <button class="wb-button wb-button-secondary" id="health-refresh" type="button">Refresh / 刷新</button>
            </div>
            <div id="health-summary" class="wb-metric-grid wb-empty">还没有健康数据。</div>
            <div id="health-providers" class="wb-list wb-empty">还没有 provider 健康信息。</div>
          </section>
          <section class="wb-panel">
            <div class="wb-panel-head">
              <h2>Recent Sources / 最近资料</h2>
              <button class="wb-button wb-button-secondary" id="system-sources-refresh" type="button">
                Refresh / 刷新
              </button>
            </div>
            <div id="system-sources-list" class="wb-list wb-empty">还没有资料。</div>
          </section>
        </section>
        """,
    )


@router.get("/sources")
def list_sources(request: Request) -> list[dict[str, object]]:
    container = get_request_container(request)
    metadata_repo = getattr(container, "metadata_repo", None)
    if metadata_repo is None:
        return []

    sources = {source.source_id: source for source in metadata_repo.list_sources()}
    payload: list[dict[str, object]] = []
    for document in metadata_repo.list_documents(active_only=True):
        source = sources.get(document.source_id)
        location = document.metadata.get("location") if document.metadata else None
        if location is None and source is not None:
            location = source.location
        payload.append(
            {
                "doc_id": document.doc_id,
                "source_id": document.source_id,
                "title": document.title,
                "doc_type": document.doc_type.value,
                "language": document.language,
                "created_at": document.created_at.isoformat(),
                "location": location,
                "source_type": source.source_type.value if source is not None else None,
                "ingest_version": source.ingest_version if source is not None else None,
            }
        )
    payload.sort(key=lambda item: str(item["created_at"]), reverse=True)
    return payload
