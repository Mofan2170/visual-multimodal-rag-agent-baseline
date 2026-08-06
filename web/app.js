const state = {
  imageFile: null,
  detections: [],
};

const documentForm = document.querySelector("#document-form");
const imageForm = document.querySelector("#image-form");
const askForm = document.querySelector("#ask-form");
const documentFile = document.querySelector("#document-file");
const imageFile = document.querySelector("#image-file");
const question = document.querySelector("#question");
const documentStatus = document.querySelector("#document-status");
const imageStatus = document.querySelector("#image-status");
const imagePreview = document.querySelector("#image-preview");
const imagePreviewWrap = document.querySelector("#image-preview-wrap");
const bboxLayer = document.querySelector("#bbox-layer");
const detectionsEl = document.querySelector("#detections");
const detectionCount = document.querySelector("#detection-count");
const answerEl = document.querySelector("#answer");
const citationsEl = document.querySelector("#citations");
const warningsEl = document.querySelector("#warnings");
const storeMode = document.querySelector("#store-mode");

window.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
});

documentFile.addEventListener("change", () => {
  documentStatus.textContent = documentFile.files[0]?.name || "未入库";
});

imageFile.addEventListener("change", () => {
  state.imageFile = imageFile.files[0] || null;
  state.detections = [];
  renderDetections([]);
  renderWarnings([]);
  if (!state.imageFile) {
    imagePreview.removeAttribute("src");
    imagePreviewWrap.classList.remove("has-image");
    imageStatus.textContent = "未检测";
    return;
  }
  imagePreview.src = URL.createObjectURL(state.imageFile);
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
    documentStatus.textContent = "请选择文档";
    return;
  }

  setBusy(documentForm, true);
  documentStatus.textContent = "入库中...";
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/documents/upload", form);
    documentStatus.textContent = `${data.chunks} 个 chunk 已入库`;
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
    state.detections = data.detections || [];
    imageStatus.textContent = data.summary;
    renderDetections(state.detections);
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
  answerEl.textContent = "生成中...";
  citationsEl.innerHTML = "";
  renderWarnings([]);

  try {
    const form = new FormData();
    form.append("question", value);
    form.append("top_k", "5");
    if (state.imageFile) {
      form.append("image", state.imageFile);
    }
    const data = await postForm("/api/ask", form);
    state.detections = data.detections || state.detections;
    answerEl.textContent = data.answer || "无回答";
    storeMode.textContent = data.store_mode || "local-json";
    if (data.visual_summary) {
      imageStatus.textContent = data.visual_summary;
    }
    renderDetections(state.detections);
    renderCitations(data.citations || []);
    renderWarnings(data.warnings || []);
  } catch (error) {
    answerEl.textContent = error.message;
  } finally {
    setBusy(askForm, false);
  }
});

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

function renderDetections(detections) {
  detectionCount.textContent = `${detections.length} 个目标`;
  detectionsEl.innerHTML = "";
  for (const detection of detections) {
    const item = document.createElement("div");
    item.className = "detection-item";
    item.innerHTML = `<strong>${escapeHtml(detection.label)}</strong> · ${(detection.confidence * 100).toFixed(1)}%<br />bbox: ${detection.bbox.map((v) => v.toFixed(1)).join(", ")}`;
    detectionsEl.appendChild(item);
  }
  renderBoxes(detections);
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
    item.className = "citation-item";
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

function setBusy(element, busy) {
  element.classList.toggle("is-busy", busy);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
