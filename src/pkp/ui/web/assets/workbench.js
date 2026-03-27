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
  const sectionPath = Array.isArray(item.section_path) && item.section_path.length
    ? item.section_path.join(" > ")
    : "";
  const chunkType = item.chunk_type || item.special_chunk_type || item.chunk_role || "child";
  return `
    <article class="wb-list-item">
      <h4>${escapeHtml(item.citation_anchor || item.chunk_id)}</h4>
      <p>${escapeHtml(item.text || "")}</p>
      <div class="wb-pill-row">
        <span class="wb-pill">${escapeHtml(`doc=${item.doc_id}`)}</span>
        <span class="wb-pill">${escapeHtml(`chunk=${item.chunk_id}`)}</span>
        <span class="wb-pill">${escapeHtml(`type=${chunkType}`)}</span>
        <span class="wb-pill">${escapeHtml(`score=${Number(item.score || 0).toFixed(3)}`)}</span>
        ${sectionPath ? `<span class="wb-pill">${escapeHtml(sectionPath)}</span>` : ""}
      </div>
    </article>
  `;
}

function parseCsvInput(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function modeLabel(mode) {
  return String(mode || "").toLowerCase() === "fast" ? "Fast" : "Deep";
}

function buildPill(value) {
  return `<span class="wb-pill">${escapeHtml(value)}</span>`;
}

function buildCitationMarkup(citation) {
  const sectionPath = Array.isArray(citation.section_path) && citation.section_path.length
    ? citation.section_path.join(" > ")
    : "未提供章节";
  const pageLabel = citation.page_start == null
    ? "page=?"
    : citation.page_end && citation.page_end !== citation.page_start
      ? `pages=${citation.page_start}-${citation.page_end}`
      : `page=${citation.page_start}`;
  return `
    <article class="wb-list-item">
      <h4>${escapeHtml(citation.file_name || citation.chunk_id)}</h4>
      <p>${escapeHtml(sectionPath)}</p>
      <div class="wb-pill-row">
        ${buildPill(`citation=${citation.citation_id}`)}
        ${buildPill(`chunk=${citation.chunk_id}`)}
        ${buildPill(`type=${citation.chunk_type || "unknown"}`)}
        ${buildPill(pageLabel)}
      </div>
    </article>
  `;
}

function buildAnswerSectionMarkup(section, response) {
  const linkedCitations = (response.citations || []).filter((citation) =>
    (section.citation_ids || []).includes(citation.citation_id)
  );
  return `
    <article class="wb-list-item">
      <h4>${escapeHtml(section.title || section.section_id || "回答片段")}</h4>
      <p>${escapeHtml(section.text || "")}</p>
      <div class="wb-pill-row">
        ${buildPill(`evidence=${(section.evidence_chunk_ids || []).join(", ") || "none"}`)}
        ${buildPill(`citations=${(section.citation_ids || []).join(", ") || "none"}`)}
      </div>
      ${linkedCitations.length
        ? `<div class="wb-inline-notes">${linkedCitations
            .map((citation) => `<span>${escapeHtml(`${citation.file_name || citation.chunk_id} · ${citation.chunk_type}`)}</span>`)
            .join("")}</div>`
        : ""}
    </article>
  `;
}

function buildResponseCard(response) {
  const answerText = response.answer_text || response.conclusion || "没有生成回答。";
  const sections = response.answer_sections || [];
  const citations = response.citations || [];
  const evidenceLinks = response.evidence_links || [];
  const evidenceItems = response.evidence || [];
  return `
    <article class="wb-result-card">
      <div class="wb-result-card-head">
        <div>
          <p class="wb-result-kicker">${escapeHtml(modeLabel(response.runtime_mode))}</p>
          <h3>${escapeHtml(`${modeLabel(response.runtime_mode)} Result`)}</h3>
        </div>
        <span class="wb-status ${response.insufficient_evidence_flag ? "wb-warn" : response.groundedness_flag ? "wb-good" : "wb-bad"}">
          ${escapeHtml(
            response.insufficient_evidence_flag
              ? "证据不足"
              : response.groundedness_flag
                ? "Grounded"
                : "Needs Review"
          )}
        </span>
      </div>
      <article class="wb-answer">${escapeHtml(answerText)}</article>
      <div class="wb-pill-row">
        ${buildPill(`uncertainty=${response.uncertainty || "unknown"}`)}
        ${buildPill(`citations=${citations.length}`)}
        ${buildPill(`sections=${sections.length}`)}
        ${buildPill(`links=${evidenceLinks.length}`)}
      </div>
      <section class="wb-subpanel">
        <h3>Answer Sections / 回答片段</h3>
        <div class="wb-list ${sections.length ? "" : "wb-empty"}">
          ${sections.length
            ? sections.map((section) => buildAnswerSectionMarkup(section, response)).join("")
            : "没有结构化回答片段。"}
        </div>
      </section>
      <section class="wb-subpanel">
        <h3>Inline Citations / 内联引用</h3>
        <div class="wb-list ${citations.length ? "" : "wb-empty"}">
          ${citations.length
            ? citations.map(buildCitationMarkup).join("")
            : "没有引用。"}
        </div>
      </section>
      <section class="wb-subpanel">
        <h3>Linked Evidence / 关联证据</h3>
        <div class="wb-list ${evidenceItems.length ? "" : "wb-empty"}">
          ${evidenceItems.length
            ? evidenceItems.slice(0, 5).map(buildEvidenceMarkup).join("")
            : "没有证据。"}
        </div>
      </section>
    </article>
  `;
}

function collectCombinedEvidence(results) {
  const seen = new Set();
  const merged = [];
  for (const result of results) {
    for (const item of result.response.evidence || []) {
      if (seen.has(item.chunk_id)) continue;
      seen.add(item.chunk_id);
      merged.push(item);
    }
  }
  return merged;
}

function collectCombinedCitations(results) {
  const seen = new Set();
  const merged = [];
  for (const result of results) {
    for (const item of result.response.citations || []) {
      if (seen.has(item.citation_id)) continue;
      seen.add(item.citation_id);
      merged.push(item);
    }
  }
  return merged;
}

function renderRunSummary(results) {
  const cards = results.map(({ response }) => {
    const retrieval = response.diagnostics?.retrieval || {};
    return {
      label: modeLabel(response.runtime_mode),
      value: `${response.groundedness_flag ? "grounded" : "review"} · parent+${retrieval.parent_backfilled_count || 0}`,
    };
  });
  renderCards("#query-run-summary", cards);
}

function renderEvaluationPad(results, evalInputs) {
  const expectedTerms = evalInputs.expectedTerms;
  const expectedChunkIds = evalInputs.expectedChunkIds;
  const expectedSpecialTypes = evalInputs.expectedSpecialTypes;
  const cards = results.map(({ response }) => {
    const answerText = response.answer_text || response.conclusion || "";
    const evidenceText = (response.evidence || []).map((item) => item.text || "").join(" ");
    const citationTypes = new Set((response.citations || []).map((item) => item.chunk_type));
    const hitChunkIds = new Set([
      ...(response.evidence || []).map((item) => item.chunk_id),
      ...(response.evidence_links || []).map((item) => item.evidence_chunk_id),
    ]);
    const matchedTerms = expectedTerms.filter((term) => `${answerText} ${evidenceText}`.includes(term));
    const matchedChunkIds = expectedChunkIds.filter((item) => hitChunkIds.has(item));
    const matchedSpecialTypes = expectedSpecialTypes.filter((item) => citationTypes.has(item));
    return `
      <article class="wb-list-item">
        <h4>${escapeHtml(`${modeLabel(response.runtime_mode)} Judge`)}</h4>
        <div class="wb-pill-row">
          ${buildPill(`grounded=${response.groundedness_flag ? "yes" : "no"}`)}
          ${buildPill(`insufficient=${response.insufficient_evidence_flag ? "yes" : "no"}`)}
          ${buildPill(`terms=${matchedTerms.length}/${expectedTerms.length}`)}
          ${buildPill(`chunks=${matchedChunkIds.length}/${expectedChunkIds.length}`)}
          ${buildPill(`special=${matchedSpecialTypes.length}/${expectedSpecialTypes.length}`)}
        </div>
        <div class="wb-inline-notes">
          <span>${escapeHtml(`命中关键词：${matchedTerms.join("、") || "无"}`)}</span>
          <span>${escapeHtml(`命中 chunk：${matchedChunkIds.join(", ") || "无"}`)}</span>
          <span>${escapeHtml(`命中特殊块：${matchedSpecialTypes.join(", ") || "无"}`)}</span>
        </div>
      </article>
    `;
  });
  renderList(
    "#query-eval",
    cards,
    "运行查询后，这里会生成页面端测评提示。"
  );
}

function renderQueryUnderstanding(results) {
  const items = results.map(({ response }) => {
    const understanding = response.diagnostics?.retrieval?.query_understanding;
    if (!understanding) {
      return `
        <article class="wb-list-item">
          <h4>${escapeHtml(`${modeLabel(response.runtime_mode)} · Query Understanding`)}</h4>
          <p>当前结果没有输出 query understanding。</p>
        </article>
      `;
    }
    const flags = [
      `dense=${understanding.needs_dense ? "on" : "off"}`,
      `sparse=${understanding.needs_sparse ? "on" : "off"}`,
      `special=${understanding.needs_special ? "on" : "off"}`,
      `structure=${understanding.needs_structure ? "on" : "off"}`,
      `metadata=${understanding.needs_metadata ? "on" : "off"}`,
    ];
    return `
      <article class="wb-list-item">
        <h4>${escapeHtml(`${modeLabel(response.runtime_mode)} · ${understanding.intent || "unknown"}`)}</h4>
        <p>${escapeHtml(`query_type=${understanding.query_type || "unknown"} · confidence=${Number(understanding.confidence || 0).toFixed(2)}`)}</p>
        <div class="wb-pill-row">
          ${flags.map(buildPill).join("")}
          ${(understanding.special_targets || []).map((item) => buildPill(`target=${item}`)).join("")}
        </div>
      </article>
    `;
  });
  renderList("#query-understanding", items, "还没有查询理解结果。");
}

function renderQueryDiagnostics(results) {
  const primary = results.find(({ response }) => response.runtime_mode === "deep") || results[0];
  const retrieval = primary?.response.diagnostics?.retrieval || {};
  const model = primary?.response.diagnostics?.model || {};
  renderCards("#query-diagnostics-cards", [
    { label: "Primary Result", value: primary ? modeLabel(primary.response.runtime_mode) : "无" },
    { label: "向量嵌入", value: retrieval.embedding_provider || "未使用" },
    { label: "重排", value: retrieval.rerank_provider || "未使用" },
    { label: "综合回答", value: model.synthesis_provider || "未使用" },
    { label: "Parent 回填", value: String(retrieval.parent_backfilled_count || 0) },
    { label: "去噪折叠", value: String(retrieval.collapsed_candidate_count || 0) },
    { label: "Fusion 输入", value: String(retrieval.fusion_input_count || 0) },
    { label: "Fusion 输出", value: String(retrieval.fused_count || 0) },
  ]);
  renderList(
    "#query-attempts",
    results.flatMap(({ response }) => {
      const retrievalAttempts = (response.diagnostics?.retrieval?.attempts || []).map(buildAttemptMarkup);
      const modelAttempts = (response.diagnostics?.model?.attempts || []).map(buildAttemptMarkup);
      return retrievalAttempts.concat(modelAttempts);
    }),
    "还没有 provider 调用记录。"
  );
}

function buildRawPayload(results) {
  if (results.length === 1) return results[0].response;
  return {
    compare: true,
    responses: results.map((item) => item.response),
    sessions: Object.fromEntries(
      results
        .filter((item) => item.session)
        .map((item) => [String(item.response.runtime_mode), item.session])
    ),
  };
}

async function runQueryRequest(payload, sessionId) {
  const response = await fetchJson("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      session_id: sessionId,
    }),
  });
  let session = null;
  if (response.runtime_mode === "deep") {
    session = await fetchJson(`/sessions/${encodeURIComponent(sessionId || "default")}`);
  }
  return { response, session };
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
  const evalForm = $("#query-eval-form");
  if (!form) return;

  $("#query-reset")?.addEventListener("click", () => {
    form.reset();
    setStatus("#query-status", "空闲");
    evalForm?.reset();
    setText("#query-run-summary", "运行后，这里会显示本次测评摘要。");
    setText("#query-results", "运行一次查询后，这里会显示结果卡片。");
    setText("#query-evidence", "还没有证据。");
    setText("#query-citations", "还没有引用。");
    setText("#query-understanding", "还没有查询理解结果。");
    setText("#query-eval", "运行查询后，这里会生成页面端测评提示。");
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
      source_scope: String(data.get("source_scope") || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
      latency_budget: data.get("latency_budget") ? Number(data.get("latency_budget")) : undefined,
      token_budget: data.get("token_budget") ? Number(data.get("token_budget")) : undefined,
      execution_location_preference: data.get("execution_location_preference") || undefined,
      fallback_allowed: data.get("fallback_allowed") === "on",
    };
    const compareModes = data.get("compare_modes") === "on";
    const baseSessionId = String(data.get("session_id") || "default");
    const evalInputs = {
      expectedTerms: parseCsvInput(new FormData(evalForm || form).get("expected_terms")),
      expectedChunkIds: parseCsvInput(new FormData(evalForm || form).get("expected_chunk_ids")),
      expectedSpecialTypes: parseCsvInput(new FormData(evalForm || form).get("expected_special_types")),
    };

    try {
      setStatus("#query-status", "查询中");
      const runs = compareModes
        ? await Promise.all([
            runQueryRequest({ ...payload, mode: "fast" }, `${baseSessionId}-fast`),
            runQueryRequest({ ...payload, mode: "deep" }, `${baseSessionId}-deep`),
          ])
        : [await runQueryRequest({ ...payload, mode: payload.mode }, payload.mode === "deep" ? baseSessionId : `${baseSessionId}-fast`)];
      setStatus(
        "#query-status",
        compareModes
          ? "完成 · Fast + Deep"
          : `完成 · ${modeLabel(runs[0].response.runtime_mode)}`,
        "wb-good"
      );
      renderRunSummary(runs);
      renderList(
        "#query-results",
        runs.map(({ response }) => buildResponseCard(response)),
        "没有返回结果。"
      );
      const mergedEvidence = collectCombinedEvidence(runs);
      renderList(
        "#query-evidence",
        mergedEvidence.map(buildEvidenceMarkup),
        "没有返回证据。"
      );
      renderList(
        "#query-citations",
        collectCombinedCitations(runs).map(buildCitationMarkup),
        "还没有引用。"
      );
      renderQueryUnderstanding(runs);
      renderQueryDiagnostics(runs);
      renderEvaluationPad(runs, evalInputs);
      setText("#query-raw", prettyJson(buildRawPayload(runs)));

      const deepRun = runs.find(({ response }) => response.runtime_mode === "deep");
      if (deepRun?.session) {
        renderList(
          "#query-session",
          [
            deepRun.session.sub_questions?.length
              ? `<article class="wb-list-item"><h4>Sub Questions</h4><p>${escapeHtml(deepRun.session.sub_questions.join("\n"))}</p></article>`
              : "",
            deepRun.session.memory_hints?.length
              ? `<article class="wb-list-item"><h4>Memory Hints</h4><p>${escapeHtml(deepRun.session.memory_hints.join("\n"))}</p></article>`
              : "",
            deepRun.session.evidence_matrix?.length
              ? `<article class="wb-list-item"><h4>Evidence Matrix</h4><pre class="wb-code">${escapeHtml(prettyJson(deepRun.session.evidence_matrix))}</pre></article>`
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
