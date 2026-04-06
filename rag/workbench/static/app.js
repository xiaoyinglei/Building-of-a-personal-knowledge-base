const appState = {
  activeProfileId: null,
  selectedPath: null,
  selectedFile: null,
  filesVersion: null,
  latestQuery: null,
};

const dom = {
  roots: document.getElementById("roots"),
  indexSummary: document.getElementById("index-summary"),
  syncMessages: document.getElementById("sync-messages"),
  fileTree: document.getElementById("file-tree"),
  fileFilter: document.getElementById("file-filter"),
  fileDetailPanel: document.getElementById("file-detail-panel"),
  queryBadges: document.getElementById("query-badges"),
  evidenceList: document.getElementById("evidence-list"),
  routingPanel: document.getElementById("routing-panel"),
  budgetPanel: document.getElementById("budget-panel"),
  diagnosticsPanel: document.getElementById("diagnostics-panel"),
  activeModel: document.getElementById("active-model"),
  activeProfileMeta: document.getElementById("active-profile-meta"),
  activeProfileStatus: document.getElementById("active-profile-status"),
  modeSelect: document.getElementById("mode-select"),
  scopeSelected: document.getElementById("scope-selected"),
  chatLog: document.getElementById("chat-log"),
  queryForm: document.getElementById("query-form"),
  queryInput: document.getElementById("query-input"),
  prevModel: document.getElementById("prev-model"),
  nextModel: document.getElementById("next-model"),
  syncButton: document.getElementById("sync-button"),
  newMarkdownButton: document.getElementById("new-markdown-button"),
  uploadButton: document.getElementById("upload-button"),
  uploadInput: document.getElementById("upload-input"),
  rebuildButton: document.getElementById("rebuild-button"),
  deleteButton: document.getElementById("delete-button"),
};

function setText(node, value) {
  if (node) {
    node.textContent = value;
  }
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || payload.message || "Request failed");
  }
  return payload;
}

function profileIndex(profiles, profileId) {
  return profiles.findIndex((profile) => profile.profile_id === profileId);
}

function renderState(state, { keepSelection = true } = {}) {
  window.__lastState = state;
  appState.activeProfileId = state.active_profile_id;
  dom.roots.textContent = `storage_root: ${state.storage_root}  |  workspace_root: ${state.workspace_root}`;
  dom.indexSummary.textContent = [
    `${state.index_summary.active_documents} docs`,
    `${state.index_summary.chunks} chunks`,
    `${state.index_summary.vectors} vectors`,
    `${state.index_summary.graph_nodes} nodes`,
    `${state.index_summary.graph_edges} edges`,
  ].join(" · ");
  dom.syncMessages.textContent = state.sync_messages.join(" | ");
  const currentSelection = keepSelection ? appState.selectedPath : null;
  appState.filesVersion = state.files_version;
  renderProfiles(state.model_profiles);
  renderFiles(state.files, currentSelection, dom.fileFilter.value.trim().toLowerCase());
  if (currentSelection) {
    const selected = findFileByPath(state.files, currentSelection);
    if (selected) {
      appState.selectedPath = selected.abs_path;
      appState.selectedFile = selected;
      renderSelectedFile(selected);
    } else {
      appState.selectedPath = null;
      appState.selectedFile = null;
      renderSelectedFile(null);
    }
  } else if (appState.selectedFile) {
    renderSelectedFile(appState.selectedFile);
  }
}

function renderProfiles(profiles) {
  if (!profiles.length) {
    setText(dom.activeModel, "No configured profile");
    setText(dom.activeProfileMeta, "没有可用的 assembly profile。");
    setText(dom.activeProfileStatus, "请先配置环境变量，或使用 test_minimal。");
    return;
  }
  const active = profiles.find((profile) => profile.profile_id === appState.activeProfileId) || profiles[0];
  appState.activeProfileId = active.profile_id;
  const suffix = active.compatible_with_index ? "" : " (index mismatch)";
  setText(dom.activeModel, `${active.label}${suffix}`);
  const capabilities = [
    active.chat_model ? `chat=${active.chat_model}` : "chat=none",
    active.embedding_model ? `embedding=${active.embedding_model}` : "embedding=none",
    active.rerank_model ? `rerank=${active.rerank_model}` : "rerank=none",
  ];
  setText(dom.activeProfileMeta, [
    `profile_id: ${active.profile_id}`,
    active.description || "No description.",
    capabilities.join(" · "),
  ].join("\n"));
  setText(
    dom.activeProfileStatus,
    active.compatible_with_index
      ? "当前 profile 与索引 contract 兼容。"
      : `当前 profile 与索引不兼容: ${active.compatibility_error || "请重建索引或切换 profile。"}`,
  );
}

function renderFiles(files, selectedPath, filterText) {
  dom.fileTree.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const entry of files) {
    const node = renderFileNode(entry, selectedPath, filterText);
    if (node) fragment.appendChild(node);
  }
  if (!fragment.childNodes.length) {
    const empty = document.createElement("div");
    empty.className = "placeholder-card";
    empty.textContent = "当前目录没有匹配的文件。";
    fragment.appendChild(empty);
  }
  dom.fileTree.appendChild(fragment);
}

function renderFileNode(entry, selectedPath, filterText) {
  if (!matchesFilter(entry, filterText)) {
    return null;
  }
  const wrapper = document.createElement("div");
  wrapper.className = "tree-node";
  const label = document.createElement("div");
  label.className = `tree-label${selectedPath === entry.abs_path ? " selected" : ""}`;
  label.addEventListener("click", () => selectEntry(entry));

  const main = document.createElement("div");
  main.className = "tree-main";
  const title = document.createElement("div");
  title.className = "tree-title";
  title.textContent = entry.node_type === "directory" ? `📁 ${entry.name}` : entry.name;
  const meta = document.createElement("div");
  meta.className = "tree-meta";
  meta.textContent = buildFileMeta(entry);
  main.appendChild(title);
  main.appendChild(meta);

  const badge = document.createElement("div");
  badge.className = "file-badge";
  badge.textContent = entry.sync_state;

  label.appendChild(main);
  label.appendChild(badge);
  wrapper.appendChild(label);

  if (entry.children && entry.children.length) {
    const children = document.createElement("div");
    children.className = "tree-children";
    for (const child of entry.children) {
      const childNode = renderFileNode(child, selectedPath, filterText);
      if (childNode) children.appendChild(childNode);
    }
    if (children.childNodes.length) {
      wrapper.appendChild(children);
    }
  }
  return wrapper;
}

function buildFileMeta(entry) {
  if (entry.node_type === "directory") {
    return entry.rel_path || ".";
  }
  const parts = [entry.rel_path];
  if (entry.source_type) parts.push(entry.source_type);
  if (entry.chunk_count) parts.push(`${entry.chunk_count} chunks`);
  if (entry.status) parts.push(`status=${entry.status}`);
  return parts.join(" · ");
}

function matchesFilter(entry, filterText) {
  if (!filterText) return true;
  const haystack = `${entry.name} ${entry.rel_path}`.toLowerCase();
  if (haystack.includes(filterText)) return true;
  return Boolean(entry.children && entry.children.some((child) => matchesFilter(child, filterText)));
}

function findFileByPath(entries, absPath) {
  for (const entry of entries) {
    if (entry.abs_path === absPath) return entry;
    if (entry.children) {
      const match = findFileByPath(entry.children, absPath);
      if (match) return match;
    }
  }
  return null;
}

function selectEntry(entry) {
  appState.selectedPath = entry.abs_path;
  appState.selectedFile = entry;
  renderSelectedFile(entry);
  refreshState({ keepSelection: true, sync: false }).catch(showError);
}

function renderSelectedFile(entry) {
  if (!entry) {
    dom.fileDetailPanel.textContent = "选择左侧文件后，这里显示索引状态和 chunk 数。";
    return;
  }
  const lines = [
    `path: ${entry.rel_path}`,
    `sync_state: ${entry.sync_state}`,
    `source_type: ${entry.source_type || "unsupported"}`,
    `status: ${entry.status || "-"}`,
    `stage: ${entry.stage || "-"}`,
    `doc_id: ${entry.doc_id || "-"}`,
    `source_id: ${entry.source_id || "-"}`,
    `chunks: ${entry.chunk_count || 0}`,
  ];
  if (entry.error_message) {
    lines.push(`error: ${entry.error_message}`);
  }
  dom.fileDetailPanel.textContent = lines.join("\n");
}

function renderQuery(result) {
  appState.latestQuery = result;
  const badges = [
    `mode: ${result.mode}`,
    `executor: ${result.mode_executor || "-"}`,
    `generation: ${result.generation_provider || "fallback"}`,
    `rerank: ${result.rerank_provider || "none"}`,
  ];
  dom.queryBadges.innerHTML = "";
  for (const text of badges) {
    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = text;
    dom.queryBadges.appendChild(badge);
  }

  renderEvidence(result.evidence || []);
  renderRouting(result);
  renderBudget(result);
  renderDiagnostics(result);
  appendChat("user", result.query);
  appendChat("assistant", result.answer_text || "模型没有返回内容。");
}

function renderEvidence(items) {
  dom.evidenceList.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "placeholder-card";
    empty.textContent = "当前查询没有命中证据。";
    dom.evidenceList.appendChild(empty);
    return;
  }
  items.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = `evidence-card${index === 0 ? " primary" : ""}`;
    card.innerHTML = `
      <div class="evidence-header">
        <strong>证据 #${index + 1}</strong>
        <span class="evidence-score">${item.score.toFixed(3)}</span>
      </div>
      <div class="meta-text">
        family: ${item.retrieval_family || "-"} · channels: ${(item.retrieval_channels || []).join(", ") || "-"}<br>
        citation: ${item.citation_anchor || "-"}<br>
        section: ${(item.section_path || []).join(" > ") || "-"}<br>
        tokens: ${item.selected_token_count}/${item.token_count}
      </div>
      <div class="excerpt">${item.text}</div>
    `;
    dom.evidenceList.appendChild(card);
  });
}

function renderRouting(result) {
  const understanding = result.query_understanding || {};
  const metadata = understanding.metadata_filters || {};
  const structure = understanding.structure_constraints || {};
  const lines = [
    `query_type: ${understanding.query_type || "-"}`,
    `needs_structure: ${Boolean(understanding.needs_structure)}`,
    `needs_metadata: ${Boolean(understanding.needs_metadata)}`,
    `needs_special: ${Boolean(understanding.needs_special)}`,
    `needs_graph_expansion: ${Boolean(understanding.needs_graph_expansion)}`,
    `section_terms: ${(understanding.preferred_section_terms || []).join(", ") || "-"}`,
    `special_targets: ${(understanding.special_targets || []).join(", ") || "-"}`,
    `page_numbers: ${(metadata.page_numbers || []).join(", ") || "-"}`,
    `source_types: ${(metadata.source_types || []).join(", ") || "-"}`,
    `section_families: ${(structure.semantic_section_families || []).join(", ") || "-"}`,
  ];
  dom.routingPanel.textContent = lines.join("\n");
}

function renderBudget(result) {
  const lines = [
    `token_budget: ${result.token_budget}`,
    `token_count: ${result.token_count}`,
    `truncated_count: ${result.truncated_count}`,
    `insufficient_evidence: ${Boolean(result.insufficient_evidence)}`,
    `generation_model: ${result.generation_model || "-"}`,
  ];
  dom.budgetPanel.textContent = lines.join("\n");
}

function renderDiagnostics(result) {
  const diagnostics = result.diagnostics || {};
  const branchHits = diagnostics.branch_hits || {};
  const branchLimits = diagnostics.branch_limits || {};
  const lines = [
    `embedding_provider: ${diagnostics.embedding_provider || "-"}`,
    `rerank_provider: ${diagnostics.rerank_provider || "-"}`,
    `fusion_input_count: ${diagnostics.fusion_input_count ?? "-"}`,
    `fused_count: ${diagnostics.fused_count ?? "-"}`,
    `graph_expanded: ${Boolean(diagnostics.graph_expanded)}`,
    `branch_hits: ${JSON.stringify(branchHits)}`,
    `branch_limits: ${JSON.stringify(branchLimits)}`,
  ];
  dom.diagnosticsPanel.textContent = lines.join("\n");
}

function appendChat(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  dom.chatLog.appendChild(bubble);
  dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
}

function cycleProfile(direction) {
  const profiles = window.__lastProfiles || [];
  if (!profiles.length) return;
  const current = profileIndex(profiles, appState.activeProfileId);
  const next = current < 0 ? 0 : (current + direction + profiles.length) % profiles.length;
  appState.activeProfileId = profiles[next].profile_id;
  renderProfiles(profiles);
  refreshState({ keepSelection: true, sync: false }).catch(showError);
}

async function refreshState({ keepSelection = true, sync = true } = {}) {
  const profileParam = appState.activeProfileId ? `&profile_id=${encodeURIComponent(appState.activeProfileId)}` : "";
  const state = await request(`/api/state?sync=${sync ? "1" : "0"}${profileParam}`);
  window.__lastProfiles = state.model_profiles || [];
  renderState(state, { keepSelection });
  return state;
}

async function submitQuery(event) {
  event.preventDefault();
  const query = dom.queryInput.value.trim();
  if (!query) return;
  const sourceScope = [];
  if (
    dom.scopeSelected.checked &&
    appState.selectedFile &&
    appState.selectedFile.doc_id
  ) {
    sourceScope.push(appState.selectedFile.doc_id);
  }
  const result = await request("/api/query", {
    method: "POST",
    body: JSON.stringify({
      query,
      mode: dom.modeSelect.value,
      profile_id: appState.activeProfileId,
      source_scope: sourceScope,
    }),
  });
  renderQuery(result);
  dom.queryInput.value = "";
}

async function createMarkdown() {
  const relativePath = window.prompt("输入相对路径，例如 notes/new-note.md", "notes/new-note.md");
  if (!relativePath) return;
  const content = window.prompt("输入文档内容", "# 新文档\n\n这里写内容。");
  if (content === null) return;
  const result = await request("/api/files/save", {
    method: "POST",
    body: JSON.stringify({
      relative_path: relativePath,
      content_text: content,
      profile_id: appState.activeProfileId,
      auto_ingest: true,
    }),
  });
  showMessage(result.message);
  window.__lastProfiles = result.state.model_profiles || window.__lastProfiles;
  renderState(result.state, { keepSelection: false });
}

async function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const relativePath = window.prompt("保存到相对路径", file.name);
  if (!relativePath) return;
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  bytes.forEach((value) => { binary += String.fromCharCode(value); });
  const contentBase64 = btoa(binary);
  const result = await request("/api/files/save", {
    method: "POST",
    body: JSON.stringify({
      relative_path: relativePath,
      content_base64: contentBase64,
      profile_id: appState.activeProfileId,
      auto_ingest: true,
    }),
  });
  showMessage(result.message);
  renderState(result.state, { keepSelection: false });
  event.target.value = "";
}

async function rebuildSelected() {
  if (!appState.selectedFile || appState.selectedFile.node_type !== "file") {
    showMessage("先选择一个文件");
    return;
  }
  const result = await request("/api/files/rebuild", {
    method: "POST",
    body: JSON.stringify({
      relative_path: appState.selectedFile.rel_path,
      profile_id: appState.activeProfileId,
    }),
  });
  showMessage(result.message);
  renderState(result.state, { keepSelection: true });
}

async function deleteSelected() {
  if (!appState.selectedFile || appState.selectedFile.node_type !== "file") {
    showMessage("先选择一个文件");
    return;
  }
  const confirmed = window.confirm(`确认删除 ${appState.selectedFile.rel_path} ? 这会同时删除本地文件和索引记录。`);
  if (!confirmed) return;
  const result = await request("/api/files/delete", {
    method: "POST",
    body: JSON.stringify({
      relative_path: appState.selectedFile.rel_path,
      profile_id: appState.activeProfileId,
    }),
  });
  showMessage(result.message);
  appState.selectedFile = null;
  appState.selectedPath = null;
  renderState(result.state, { keepSelection: false });
}

async function syncNow() {
  const state = await request("/api/sync", {
    method: "POST",
    body: JSON.stringify({ profile_id: appState.activeProfileId }),
  });
  window.__lastProfiles = state.model_profiles || [];
  renderState(state, { keepSelection: true });
}

function showMessage(message) {
  dom.syncMessages.textContent = message;
}

function showError(error) {
  dom.syncMessages.textContent = error.message || String(error);
}

dom.queryForm.addEventListener("submit", (event) => {
  submitQuery(event).catch(showError);
});
dom.prevModel.addEventListener("click", () => cycleProfile(-1));
dom.nextModel.addEventListener("click", () => cycleProfile(1));
dom.syncButton.addEventListener("click", () => syncNow().catch(showError));
dom.newMarkdownButton.addEventListener("click", () => createMarkdown().catch(showError));
dom.uploadButton.addEventListener("click", () => dom.uploadInput.click());
dom.uploadInput.addEventListener("change", (event) => handleUpload(event).catch(showError));
dom.rebuildButton.addEventListener("click", () => rebuildSelected().catch(showError));
dom.deleteButton.addEventListener("click", () => deleteSelected().catch(showError));
dom.fileFilter.addEventListener("input", () => {
  if (window.__lastState) {
    renderFiles(window.__lastState.files || [], appState.selectedPath, dom.fileFilter.value.trim().toLowerCase());
  }
});

async function boot() {
  const state = await refreshState({ sync: false });
  window.__lastState = state;
  syncNow().catch(showError);
  setInterval(async () => {
    try {
      const nextState = await refreshState({ keepSelection: true, sync: true });
      window.__lastState = nextState;
    } catch (error) {
      showError(error);
    }
  }, 5000);
}

boot().catch(showError);
