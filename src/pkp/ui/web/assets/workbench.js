const $ = (selector) => document.querySelector(selector);

function setText(selector, value) {
  const node = $(selector);
  if (node) {
    node.textContent = value;
    node.classList.toggle("wb-empty", !value || value.startsWith("No ") || value === "Idle");
  }
}

function setStatus(selector, value, tone = "") {
  const node = $(selector);
  if (!node) return;
  node.textContent = value;
  node.classList.remove("wb-good", "wb-warn", "wb-bad");
  if (tone) node.classList.add(tone);
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.message || response.statusText;
    throw new Error(String(detail));
  }
  return payload;
}

function renderCards(targetSelector, cards) {
  const target = $(targetSelector);
  if (!target) return;
  if (!cards.length) {
    target.innerHTML = "No data.";
    target.classList.add("wb-empty");
    return;
  }
  target.classList.remove("wb-empty");
  target.innerHTML = cards
    .map(
      (card) => `
      <article class="wb-metric">
        <h4>${escapeHtml(card.label)}</h4>
        <strong>${escapeHtml(card.value)}</strong>
      </article>
    `
    )
    .join("");
}

function renderList(targetSelector, items, emptyText = "No data.") {
  const target = $(targetSelector);
  if (!target) return;
  if (!items.length) {
    target.innerHTML = emptyText;
    target.classList.add("wb-empty");
    return;
  }
  target.classList.remove("wb-empty");
  target.innerHTML = items.join("");
}

function buildAttemptMarkup(attempt) {
  const error = attempt.error ? `<p>${escapeHtml(attempt.error)}</p>` : "";
  return `
    <article class="wb-attempt">
      <strong>${escapeHtml(`${attempt.stage} · ${attempt.provider}`)}</strong>
      <p>${escapeHtml(`capability=${attempt.capability} · location=${attempt.location} · status=${attempt.status}${attempt.model ? ` · model=${attempt.model}` : ""}`)}</p>
      ${error}
    </article>
  `;
}

function buildEvidenceMarkup(item) {
  return `
    <article class="wb-list-item">
      <h4>${escapeHtml(item.citation_anchor || item.chunk_id)}</h4>
      <p>${escapeHtml(item.text || "")}</p>
      <div class="wb-pill-row">
        <span class="wb-pill">${escapeHtml(`doc=${item.doc_id}`)}</span>
        <span class="wb-pill">${escapeHtml(`chunk=${item.chunk_id}`)}</span>
        <span class="wb-pill">${escapeHtml(`score=${Number(item.score || 0).toFixed(3)}`)}</span>
      </div>
    </article>
  `;
}

async function loadSources(targetSelector) {
  const payload = await fetchJson("/sources");
  renderList(
    targetSelector,
    payload.map(
      (item) => `
      <article class="wb-source-card">
        <h4>${escapeHtml(item.title || item.doc_id)}</h4>
        <p>${escapeHtml(`${item.source_type || "unknown"} · ${item.location || "no location"}`)}</p>
        <div class="wb-pill-row">
          <span class="wb-pill">${escapeHtml(item.doc_id)}</span>
          <span class="wb-pill">${escapeHtml(item.source_id)}</span>
          <span class="wb-pill">${escapeHtml(item.language || "unknown")}</span>
        </div>
      </article>
    `
    ),
    "No sources yet."
  );
}

function initQueryPage() {
  const form = $("#query-form");
  if (!form) return;

  $("#query-reset")?.addEventListener("click", () => {
    form.reset();
    setStatus("#query-status", "空闲");
    setText("#query-answer", "运行一次查询后，这里会显示最终回答。");
    setText("#query-evidence", "还没有证据。");
    setText("#query-attempts", "还没有 provider 调用记录。");
    setText("#query-session", "使用 Deep 模式后，这里会显示子问题和证据矩阵。");
    setText("#query-raw", "还没有响应。");
    setText("#query-diagnostics-cards", "还没有诊断信息。");
  });

  $("#query-copy-json")?.addEventListener("click", async () => {
    const raw = $("#query-raw")?.textContent || "";
    if (!raw || raw === "还没有响应。") return;
    await navigator.clipboard.writeText(raw);
    setStatus("#query-status", "JSON 已复制", "wb-good");
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const payload = {
      query: data.get("query"),
      mode: data.get("mode"),
      session_id: data.get("session_id") || undefined,
      source_scope: String(data.get("source_scope") || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
      latency_budget: data.get("latency_budget") ? Number(data.get("latency_budget")) : undefined,
      token_budget: data.get("token_budget") ? Number(data.get("token_budget")) : undefined,
      execution_location_preference: data.get("execution_location_preference") || undefined,
      fallback_allowed: data.get("fallback_allowed") === "on",
    };

    try {
      setStatus("#query-status", "查询中");
      const response = await fetchJson("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setStatus("#query-status", `完成 · ${response.runtime_mode}`, "wb-good");
      setText("#query-answer", response.conclusion || "没有生成回答。");
      renderList(
        "#query-evidence",
        (response.evidence || []).map(buildEvidenceMarkup),
        "没有返回证据。"
      );

      const retrieval = response.diagnostics?.retrieval || {};
      const model = response.diagnostics?.model || {};
      renderCards("#query-diagnostics-cards", [
        { label: "向量嵌入", value: retrieval.embedding_provider || "未使用" },
        { label: "重排", value: retrieval.rerank_provider || "未使用" },
        { label: "综合回答", value: model.synthesis_provider || "未使用" },
        { label: "回退原因", value: model.fallback_reason || "无" },
        { label: "是否降级", value: Boolean(model.degraded_to_retrieval_only) ? "是" : "否" },
        { label: "不确定性", value: response.uncertainty || "unknown" },
      ]);
      renderList(
        "#query-attempts",
        [...(retrieval.attempts || []), ...(model.attempts || [])].map(buildAttemptMarkup),
        "还没有 provider 调用记录。"
      );
      setText("#query-raw", prettyJson(response));

      if (payload.mode === "deep") {
        const sessionId = payload.session_id || "default";
        const session = await fetchJson(`/sessions/${encodeURIComponent(sessionId)}`);
        renderList(
          "#query-session",
          [
            session.sub_questions?.length
              ? `<article class="wb-list-item"><h4>Sub Questions</h4><p>${escapeHtml(session.sub_questions.join("\n"))}</p></article>`
              : "",
            session.memory_hints?.length
              ? `<article class="wb-list-item"><h4>Memory Hints</h4><p>${escapeHtml(session.memory_hints.join("\n"))}</p></article>`
              : "",
            session.evidence_matrix?.length
              ? `<article class="wb-list-item"><h4>Evidence Matrix</h4><pre class="wb-code">${escapeHtml(prettyJson(session.evidence_matrix))}</pre></article>`
              : "",
          ].filter(Boolean),
          "当前会话没有保存 Deep 轨迹。"
        );
      } else {
        setText("#query-session", "使用 Deep 模式后，这里会显示子问题和证据矩阵。");
      }
    } catch (error) {
      setStatus("#query-status", `失败 · ${error.message}`, "wb-bad");
      setText("#query-raw", error.stack || String(error));
    }
  });
}

function initIngestPage() {
  const uploadForm = $("#upload-form");
  const inlineForm = $("#inline-ingest-form");
  if (!uploadForm || !inlineForm) return;

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = new FormData(uploadForm);
    try {
      setStatus("#upload-status", "上传中");
      const response = await fetchJson("/ingest/upload", {
        method: "POST",
        body: payload,
      });
      setStatus("#upload-status", `完成 · ${response.chunk_count} 个 chunk`, "wb-good");
      await loadSources("#sources-list");
    } catch (error) {
      setStatus("#upload-status", `失败 · ${error.message}`, "wb-bad");
    }
  });

  inlineForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(inlineForm);
    try {
      setStatus("#inline-status", "写入中");
      const response = await fetchJson("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_type: data.get("source_type"),
          title: data.get("title") || undefined,
          content: data.get("content"),
        }),
      });
      setStatus("#inline-status", `完成 · ${response.chunk_count} 个 chunk`, "wb-good");
      await loadSources("#sources-list");
    } catch (error) {
      setStatus("#inline-status", `失败 · ${error.message}`, "wb-bad");
    }
  });

  $("#sources-refresh")?.addEventListener("click", () => void loadSources("#sources-list"));
  void loadSources("#sources-list");
}

async function loadArtifacts() {
  const payload = await fetchJson("/artifacts");
  renderList(
    "#artifacts-list",
    payload.map(
      (item) => `
        <article class="wb-list-item">
          <h4>${escapeHtml(item.artifact_id)}</h4>
          <p>${escapeHtml(`状态=${item.status || "unknown"}`)}</p>
          <div class="wb-list-item-actions">
          <button class="wb-button wb-button-secondary" data-artifact-open="${escapeHtml(item.artifact_id)}" type="button">查看</button>
          <button class="wb-button" data-artifact-approve="${escapeHtml(item.artifact_id)}" type="button">批准</button>
        </div>
      </article>
    `
    ),
    "还没有 artifact。"
  );
}

function initArtifactsPage() {
  if (!$("#artifacts-list")) return;

  $("#artifacts-refresh")?.addEventListener("click", () => void loadArtifacts());
  $("#artifacts-list")?.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const openId = target.dataset.artifactOpen;
    const approveId = target.dataset.artifactApprove;
    try {
      if (openId) {
        setStatus("#artifact-status", `加载中 · ${openId}`);
        const payload = await fetchJson(`/artifacts/${encodeURIComponent(openId)}`);
        setText("#artifact-detail", prettyJson(payload));
        setStatus("#artifact-status", `已打开 · ${openId}`, "wb-good");
      }
      if (approveId) {
        setStatus("#artifact-status", `批准中 · ${approveId}`);
        const payload = await fetchJson("/artifacts/approve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ artifact_id: approveId }),
        });
        setText("#artifact-detail", prettyJson(payload));
        setStatus("#artifact-status", `已批准 · ${approveId}`, "wb-good");
        await loadArtifacts();
      }
    } catch (error) {
      setStatus("#artifact-status", `失败 · ${error.message}`, "wb-bad");
    }
  });
  void loadArtifacts();
}

async function loadHealth() {
  const payload = await fetchJson("/health");
  renderCards("#health-summary", [
    { label: "系统状态", value: payload.status || "unknown" },
    { label: "文档数", value: String(payload.indices?.documents ?? 0) },
    { label: "Chunk 数", value: String(payload.indices?.chunks ?? 0) },
    { label: "已建向量文档", value: String(payload.indices?.vectors ?? 0) },
    { label: "缺失向量文档", value: String(payload.indices?.missing_vectors ?? 0) },
  ]);
  renderList(
    "#health-providers",
    (payload.providers || []).map((provider) => {
      const capabilities = Object.entries(provider.capabilities || {})
        .map(([name, capability]) => `${name}: ${capability.available ? "up" : "down"}${capability.model ? ` (${capability.model})` : ""}`)
        .join(" · ");
      return `
        <article class="wb-list-item">
          <h4>${escapeHtml(`${provider.provider} · ${provider.location}`)}</h4>
          <p>${escapeHtml(capabilities || "No capability details")}</p>
        </article>
      `;
    }),
    "还没有 provider 健康信息。"
  );
}

function initSystemPage() {
  if (!$("#health-summary")) return;
  $("#health-refresh")?.addEventListener("click", () => void loadHealth());
  $("#system-sources-refresh")?.addEventListener("click", () => void loadSources("#system-sources-list"));
  void loadHealth();
  void loadSources("#system-sources-list");
}

document.addEventListener("DOMContentLoaded", () => {
  initQueryPage();
  initIngestPage();
  initArtifactsPage();
  initSystemPage();
});
