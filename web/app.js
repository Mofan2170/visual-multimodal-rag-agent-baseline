const state = {
  imageFile: null,
  imageId: null,
  previewUrl: null,
  detections: [],
};

const $ = (selector) => document.querySelector(selector);

const runtimeForm = $("#runtime-form");
const modelPathForm = $("#model-path-form");
const modelUploadForm = $("#model-upload-form");
const documentForm = $("#document-form");
const imageForm = $("#image-form");
const askForm = $("#ask-form");

const apiKey = $("#api-key");
const clearApiKey = $("#clear-api-key");
const baseUrl = $("#base-url");
const chatModel = $("#chat-model");
const embeddingModel = $("#embedding-model");
const modelPath = $("#model-path");
const modelFile = $("#model-file");
const documentFile = $("#document-file");
const imageFile = $("#image-file");
const question = $("#question");

const runtimeSummary = $("#runtime-summary");
const runtimeStatus = $("#runtime-status");
const modelStatus = $("#model-status");
const classList = $("#class-list");
const documentStatus = $("#document-status");
const imageStatus = $("#image-status");
const storeMode = $("#store-mode");
const detectionCount = $("#detection-count");
const countPills = $("#count-pills");

const imagePreview = $("#image-preview");
const imagePreviewWrap = $("#image-preview-wrap");
const bboxLayer = $("#bbox-layer");
const detectionsEl = $("#detections");
const answerEl = $("#answer");
const citationsEl = $("#citations");
const warningsEl = $("#warnings");

window.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  await refreshStatus();
});

$("#refresh-status").addEventListener("click", refreshStatus);

runtimeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(runtimeForm, true);
  runtimeStatus.textContent = "保存中...";
  try {
    const apiKeyValue = clearApiKey.checked ? "" : (apiKey.value.trim() || null);
    const data = await postJson("/api/runtime/config", {
      api_key: apiKeyValue,
      base_url: baseUrl.value,
      chat_model: chatModel.value,
      embedding_model: embeddingModel.value || "local",
    });
    renderRuntimeConfig(data);
    clearApiKey.checked = false;
    renderWarnings(data.warnings || []);
    await refreshStatus();
  } catch (error) {
    runtimeStatus.textContent = error.message;
  } finally {
    setBusy(runtimeForm, false);
  }
});

modelPathForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = modelPath.value.trim();
  if (!value) {
    modelStatus.textContent = "请输入模型路径";
    return;
  }
  setBusy(modelPathForm, true);
  modelStatus.textContent = "加载中...";
  try {
    const data = await postJson("/api/models/select", { model_path: value });
    resetImageAnalysis();
    renderModelInfo(data.yolo_model);
    renderWarnings(data.warnings || []);
    await refreshStatus();
  } catch (error) {
    modelStatus.textContent = error.message;
  } finally {
    setBusy(modelPathForm, false);
  }
});

modelUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = modelFile.files[0];
  if (!file) {
    modelStatus.textContent = "请选择 .pt 模型";
    return;
  }
  setBusy(modelUploadForm, true);
  modelStatus.textContent = "上传中...";
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/models/upload", form);
    resetImageAnalysis();
    modelPath.value = data.model_path;
    renderModelInfo(data.yolo_model);
    renderWarnings(data.warnings || []);
    await refreshStatus();
  } catch (error) {
    modelStatus.textContent = error.message;
  } finally {
    setBusy(modelUploadForm, false);
  }
});

documentFile.addEventListener("change", () => {
  documentStatus.textContent = documentFile.files[0]?.name || "未入库";
});

imageFile.addEventListener("change", () => {
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }
  state.imageFile = imageFile.files[0] || null;
  state.imageId = null;
  state.previewUrl = null;
  state.detections = [];
  renderDetections([], {});
  renderWarnings([]);
  if (!state.imageFile) {
    imagePreview.removeAttribute("src");
    imagePreviewWrap.classList.remove("has-image");
    imageStatus.textContent = "未检测";
    return;
  }
  state.previewUrl = URL.createObjectURL(state.imageFile);
  imagePreview.src = state.previewUrl;
  imagePreviewWrap.classList.add("has-image");
  imageStatus.textContent = state.imageFile.name;
});

imagePreview.addEventListener("load", () => {
  renderBoxes(state.detections);
});

documentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = documentFile.files[0];
  if (!file) {
    documentStatus.textContent = "请选择规则文档";
    return;
  }

  setBusy(documentForm, true);
  documentStatus.textContent = "入库中...";
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/documents/upload", form);
    documentStatus.textContent = data.deduplicated
      ? `${data.chunks} 个 chunk 已存在，已跳过重复入库`
      : `${data.chunks} 个 chunk 已入库`;
    storeMode.textContent = data.store_mode;
    renderWarnings(data.warnings || []);
  } catch (error) {
    documentStatus.textContent = error.message;
  } finally {
    setBusy(documentForm, false);
  }
});

imageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.imageFile) {
    imageStatus.textContent = "请选择图片";
    return;
  }

  setBusy(imageForm, true);
  imageStatus.textContent = "检测中...";
  try {
    const form = new FormData();
    form.append("image", state.imageFile);
    const data = await postForm("/api/images/analyze", form);
    state.imageId = data.image_id;
    state.detections = data.detections || [];
    imageStatus.textContent = data.summary;
    renderDetections(state.detections, data.detection_counts || {});
    renderModelInfo({
      model_name: data.yolo_model_name,
      classes: data.yolo_classes || [],
      loaded: true,
    });
    renderWarnings(data.warnings || []);
  } catch (error) {
    imageStatus.textContent = error.message;
  } finally {
    setBusy(imageForm, false);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = question.value.trim();
  if (!value) {
    answerEl.textContent = "请输入问题";
    return;
  }

  setBusy(askForm, true);
  answerEl.textContent = "运行中...";
  citationsEl.innerHTML = "";
  renderWarnings([]);

  try {
    const form = new FormData();
    form.append("question", value);
    if (state.imageId) {
      form.append("image_id", state.imageId);
    } else if (state.imageFile) {
      form.append("image", state.imageFile);
    }
    const data = await postForm("/api/ask", form);
    state.imageId = data.image_id || state.imageId;
    state.detections = data.detections || state.detections;
    renderMarkdownAnswer(data.answer || "无回答");
    storeMode.textContent = data.store_mode || "local-json";
    if (data.visual_summary) {
      imageStatus.textContent = data.visual_summary;
    }
    renderDetections(state.detections, data.detection_counts || {});
    renderModelInfo({
      model_name: data.yolo_model_name,
      classes: data.yolo_classes || [],
      loaded: Boolean(data.yolo_model_name),
    });
    renderCitations(data.citations || []);
    renderWarnings(data.warnings || []);
  } catch (error) {
    answerEl.textContent = error.message;
  } finally {
    setBusy(askForm, false);
  }
});

async function refreshStatus() {
  try {
    const response = await fetch("/api/runtime/status");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `状态请求失败：${response.status}`);
    }
    renderRuntimeStatus(data);
    renderModelInfo(data.yolo_model);
    renderWarnings(data.warnings || []);
  } catch (error) {
    runtimeSummary.textContent = "状态不可用";
    runtimeStatus.textContent = error.message;
  }
}

function renderRuntimeStatus(data) {
  runtimeSummary.textContent = data.model_configured ? "API 已填写" : "本地 fallback";
  runtimeStatus.textContent = data.model_configured ? "API 已填写，运行时验证" : "未配置 API";
  baseUrl.value = data.base_url || baseUrl.value;
  chatModel.value = data.chat_model || chatModel.value;
  embeddingModel.value = data.embedding_model || "local";
  storeMode.textContent = data.store_mode || "local-json";
}

function renderRuntimeConfig(data) {
  runtimeStatus.textContent = data.model_configured ? "API 已填写，运行时验证" : "未配置 API";
  runtimeSummary.textContent = data.model_configured ? "API 已填写" : "本地 fallback";
  apiKey.value = "";
}

function renderModelInfo(info = {}) {
  const name = info.model_name || "未选择";
  const loaded = info.loaded ? "已加载" : "未加载";
  modelStatus.textContent = `${name} · ${loaded}`;
  classList.innerHTML = "";
  for (const cls of info.classes || []) {
    const pill = document.createElement("span");
    pill.className = "class-pill";
    pill.textContent = cls;
    classList.appendChild(pill);
  }
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `请求失败：${response.status}`);
  }
  return data;
}

async function postForm(url, form) {
  const response = await fetch(url, {
    method: "POST",
    body: form,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `请求失败：${response.status}`);
  }
  return data;
}

function renderDetections(detections, counts) {
  detectionCount.textContent = String(detections.length);
  renderCountPills(counts);
  detectionsEl.innerHTML = "";
  for (const detection of detections) {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `<strong>${escapeHtml(detection.label)}</strong> · ${(detection.confidence * 100).toFixed(1)}%<br />bbox: ${detection.bbox.map((v) => v.toFixed(1)).join(", ")}`;
    detectionsEl.appendChild(item);
  }
  renderBoxes(detections);
}

function renderCountPills(counts) {
  countPills.innerHTML = "";
  for (const [label, count] of Object.entries(counts || {})) {
    const pill = document.createElement("span");
    pill.className = "count-pill";
    pill.textContent = `${label} ${count}`;
    countPills.appendChild(pill);
  }
}

function renderBoxes(detections) {
  bboxLayer.innerHTML = "";
  if (!imagePreview.naturalWidth || !imagePreview.naturalHeight) {
    return;
  }

  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.bbox;
    const box = document.createElement("div");
    box.className = "box";
    box.style.left = `${(x1 / imagePreview.naturalWidth) * 100}%`;
    box.style.top = `${(y1 / imagePreview.naturalHeight) * 100}%`;
    box.style.width = `${((x2 - x1) / imagePreview.naturalWidth) * 100}%`;
    box.style.height = `${((y2 - y1) / imagePreview.naturalHeight) * 100}%`;

    const label = document.createElement("span");
    label.className = "box-label";
    label.textContent = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
    box.appendChild(label);
    bboxLayer.appendChild(box);
  }
}

function renderCitations(citations) {
  citationsEl.innerHTML = "";
  for (const citation of citations) {
    const item = document.createElement("div");
    item.className = "list-item";
    const text = citation.text.length > 260 ? `${citation.text.slice(0, 260)}...` : citation.text;
    item.innerHTML = `<strong>${escapeHtml(citation.source)}</strong> · score ${citation.score.toFixed(3)}<br />${escapeHtml(text)}`;
    citationsEl.appendChild(item);
  }
}

function renderWarnings(warnings) {
  warningsEl.innerHTML = "";
  for (const warning of warnings) {
    const item = document.createElement("div");
    item.className = "warning-item";
    item.textContent = warning;
    warningsEl.appendChild(item);
  }
}

function renderMarkdownAnswer(markdown) {
  if (window.marked?.parse && window.DOMPurify?.sanitize) {
    const html = window.marked.parse(markdown, {
      breaks: true,
      gfm: true,
    });
    answerEl.innerHTML = window.DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true },
    });
    return;
  }
  answerEl.textContent = markdown;
}

function setBusy(element, busy) {
  element.classList.toggle("is-busy", busy);
}

function resetImageAnalysis() {
  state.imageId = null;
  state.detections = [];
  renderDetections([], {});
  if (state.imageFile) {
    imageStatus.textContent = `${state.imageFile.name} · 模型已切换，请重新检测`;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
