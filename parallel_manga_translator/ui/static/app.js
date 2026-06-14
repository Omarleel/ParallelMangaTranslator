const state = {
  job: null,
  pageIndex: 0,
  selectedRegion: null,
  variant: 'current',
  tool: 'select',
  regions: [],
  brushStrokes: [],
  naturalWidth: 1,
  naturalHeight: 1,
  polling: null,
  dragging: null,
  drawingRegion: null,
  drawingStroke: null,
  dirty: false,
  pageStamp: '',
  clipboardRegion: null,
  deletedStack: [],
  lastJob: null,
  zoom: 1,
  fitZoom: 1,
  autosaveTimer: null,
  autosaveInFlight: false,
  autosaveError: '',
  brushCursor: { x: 0, y: 0, visible: false },
  overlayRaf: null,
  deferredOverlayRender: false,
  inlineEditingIndex: null,
};

const $ = (id) => document.getElementById(id);
const setupView = $('setupView');
const workView = $('workView');
const uploadForm = $('uploadForm');
const folderInput = $('folderInput');
const zipInput = $('zipInput');
const folderLabel = $('folderLabel');
const zipLabel = $('zipLabel');
const titleInput = $('titleInput');
const sourceLanguage = $('sourceLanguage');
const targetLanguage = $('targetLanguage');
const translatorSelect = $('translatorSelect');
const detectionEngine = $('detectionEngine');
const transcriptionEngine = $('transcriptionEngine');
const advancedToggle = $('advancedToggle');
const advancedOptions = $('advancedOptions');
const continueLastBtn = $('continueLastBtn');
const newJobBtn = $('newJobBtn');
const progressPanel = $('progressPanel');
const pagesPanel = $('pagesPanel');
const pageList = $('pageList');
const jobTitle = $('jobTitle');
const jobBadge = $('jobBadge');
const jobMessage = $('jobMessage');
const jobOptionsSummary = $('jobOptionsSummary');
const progressBar = $('progressBar');
const progressText = $('progressText');
const exportBtn = $('exportBtn');
const reviewLayout = $('reviewLayout');
const waitState = $('waitState');
const imageStage = $('imageStage');
const canvasCard = document.querySelector('.canvas-card');
const pageImage = $('pageImage');
const overlayLayer = $('overlayLayer');
const pageHeading = $('pageHeading');
const pageStatus = $('pageStatus');
const regionCount = $('regionCount');
const noRegion = $('noRegion');
const regionEditor = $('regionEditor');
const regionTypeBadge = $('regionTypeBadge');
const originalText = $('originalText');
const translatedText = $('translatedText');
const boxX = $('boxX');
const boxY = $('boxY');
const boxW = $('boxW');
const boxH = $('boxH');
const restoreOriginal = $('restoreOriginal');
const visibleText = $('visibleText');
const autoFontSize = $('autoFontSize');
const fontSize = $('fontSize');
const fontSizeNumber = $('fontSizeNumber');
const fontSizeValue = $('fontSizeValue');
const saveBtn = $('saveBtn');
const resetBtn = $('resetBtn');
const prevBtn = $('prevBtn');
const nextBtn = $('nextBtn');
const toast = $('toast');
const selectTool = $('selectTool');
const newRegionTool = $('newRegionTool');
const brushTool = $('brushTool');
const brushMode = $('brushMode');
const brushSize = $('brushSize');
const brushSizeValue = $('brushSizeValue');
const undoBrushBtn = $('undoBrushBtn');
const undoDeleteBtn = $('undoDeleteBtn');
const applyInpaintBtn = $('applyInpaintBtn');
const zoomOutBtn = $('zoomOutBtn');
const zoomInBtn = $('zoomInBtn');
const zoomFitBtn = $('zoomFitBtn');
const focusBtn = $('focusBtn');
const zoomSlider = $('zoomSlider');
const zoomValue = $('zoomValue');
const autosaveDot = $('autosaveDot');
const autosaveStatus = $('autosaveStatus');
const toolHelpTitle = $('toolHelpTitle');
const toolHelpText = $('toolHelpText');
const ocrRegionBtn = $('ocrRegionBtn');
const copyRegionBtn = $('copyRegionBtn');
const deleteRegionBtn = $('deleteRegionBtn');

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove('hidden');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add('hidden'), 3800);
}

function setAutosaveStatus(kind, message) {
  if (!autosaveDot || !autosaveStatus) return;
  autosaveDot.className = `autosave-dot ${kind || 'saved'}`;
  autosaveStatus.textContent = message || 'Cambios guardados automáticamente.';
}

function hasPendingInpaintStroke() {
  return state.brushStrokes.some((stroke) => stroke.mode === 'inpaint' && !stroke.applied);
}

function hasMaskEraserStroke() {
  return state.brushStrokes.some((stroke) => ['mask_eraser', 'erase_mask', 'eraser'].includes(stroke.mode));
}

function hasAnyInpaintStroke() {
  return state.brushStrokes.some((stroke) => stroke.mode === 'inpaint');
}

function isMaskEraserMode(mode) {
  return ['mask_eraser', 'erase_mask', 'eraser'].includes(String(mode || '').toLowerCase());
}

function clampFontSize(value, fallback = null) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.max(6, Math.min(160, Math.round(parsed)));
}

const MANGA_FONT_FAMILY = '"New Wild Words", "Comic Sans MS", "Trebuchet MS", Arial, sans-serif';
const TEXT_RENDERER_MAX_FONT_SIZE = 160;
const TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE = 7;
const TEXT_RENDERER_INNER_MARGIN_RATIO = 0.09;
const TEXT_RENDERER_LINE_SPACING = 0.22;
let textMeasureCanvas = null;

function regionStyle(region) {
  return String(region?.style || 'dialogo').toLowerCase();
}

function innerMarginRatioForRegion(region) {
  const style = regionStyle(region);
  if (style.startsWith('onomatopeya')) {
    return Math.max(0.035, TEXT_RENDERER_INNER_MARGIN_RATIO * 0.45);
  }
  return TEXT_RENDERER_INNER_MARGIN_RATIO;
}

function textRenderArea(region) {
  const [, , rawW, rawH] = region?.bbox || [0, 0, 80, 40];
  const w = Math.max(1, Number(rawW) || 1);
  const h = Math.max(1, Number(rawH) || 1);
  const marginRatio = innerMarginRatioForRegion(region);
  const marginX = Math.max(2, Math.floor(w * marginRatio));
  const marginY = Math.max(2, Math.floor(h * marginRatio));
  return {
    x: marginX,
    y: marginY,
    width: Math.max(1, w - 2 * marginX),
    height: Math.max(1, h - 2 * marginY),
  };
}

function normalizeRendererText(text) {
  return String(text || ' ')
    .replace(/\r/g, '\n')
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim() || ' ';
}

function preparedRendererText(region, text) {
  const style = regionStyle(region);
  let value = normalizeRendererText(text);
  if (!style.startsWith('onomatopeya')) return value;
  if (style === 'onomatopeya_subtitle' && value.includes('\n')) {
    const [first, ...rest] = value.split('\n');
    let subtitle = rest.join('\n').trim();
    if (/[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]/.test(subtitle)) subtitle = subtitle.toUpperCase();
    return `${first.trim()}\n${subtitle}`;
  }
  return /[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]/.test(value) ? value.toUpperCase() : value;
}

function textMeasureContext(fontSize) {
  if (!textMeasureCanvas) textMeasureCanvas = document.createElement('canvas');
  const ctx = textMeasureCanvas.getContext('2d');
  ctx.font = `900 ${Math.max(1, Math.round(fontSize))}px ${MANGA_FONT_FAMILY}`;
  return ctx;
}

function measureMangaText(text, fontSize) {
  return textMeasureContext(fontSize).measureText(text || ' ').width;
}

function splitPreviewLines(text, fontSize, maxWidth) {
  const lines = [];
  for (const paragraph of normalizeRendererText(text).split('\n')) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    if (!words.length) {
      if (lines.length) lines.push('');
      continue;
    }
    let current = '';
    for (const word of words) {
      const candidate = current ? `${current} ${word}` : word;
      if (current && measureMangaText(candidate, fontSize) > maxWidth) {
        lines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    if (current) lines.push(current);
  }
  return lines.length ? lines : [' '];
}

function fitsPreviewText(text, fontSize, width, height) {
  const lines = splitPreviewLines(text, fontSize, width);
  const maxLineWidth = Math.max(...lines.map((line) => measureMangaText(line, fontSize)), 0);
  const paragraphHeight = lines.length * fontSize + Math.max(0, lines.length - 1) * fontSize * TEXT_RENDERER_LINE_SPACING;
  return maxLineWidth <= width && paragraphHeight <= height;
}

function rendererLikeAutoFontSize(region) {
  const text = preparedRendererText(region, region?.translated_text || region?.original_text || 'Texto');
  const area = textRenderArea(region);
  const style = regionStyle(region);
  let startFactor = 0.70;
  if (style.startsWith('onomatopeya')) startFactor = style === 'onomatopeya_subtitle' ? 0.82 : 0.92;
  else if (style === 'narracion') startFactor = 0.58;
  const start = Math.min(TEXT_RENDERER_MAX_FONT_SIZE, Math.max(TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE, Math.floor(area.height * startFactor)));
  for (let size = start; size >= TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE; size -= 1) {
    if (fitsPreviewText(text, size, area.width, area.height)) return size;
  }
  return TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE;
}

function applyRendererTextLayout(element, region) {
  if (!element || !region) return;
  const { scaleX, scaleY } = getScale();
  const area = textRenderArea(region);
  element.style.left = `${area.x * scaleX}px`;
  element.style.top = `${area.y * scaleY}px`;
  element.style.width = `${area.width * scaleX}px`;
  element.style.height = `${area.height * scaleY}px`;
  element.style.fontSize = `${previewFontSize(region)}px`;
  element.style.lineHeight = regionStyle(region).startsWith('onomatopeya') ? '1' : '1.05';
}

function usesAutoFontSize(region) {
  return !region || region.auto_font_size !== false;
}

function estimatedAutoFontSize(region) {
  return rendererLikeAutoFontSize(region);
}

function effectiveManualFontSize(region) {
  return clampFontSize(region?.font_size, estimatedAutoFontSize(region) || 24);
}

function syncFontControlsFromValue(value) {
  const size = clampFontSize(value, 24);
  if (fontSize) fontSize.value = String(Math.max(Number(fontSize.min || 6), Math.min(Number(fontSize.max || 120), size)));
  if (fontSizeNumber) fontSizeNumber.value = String(size);
  if (fontSizeValue) fontSizeValue.textContent = `${size} px`;
  return size;
}

function updateFontControlsDisabled() {
  const disabled = Boolean(autoFontSize?.checked);
  if (fontSize) fontSize.disabled = disabled;
  if (fontSizeNumber) fontSizeNumber.disabled = disabled;
}

function shouldDisplayStroke(stroke, isDrawing = false) {
  if (!stroke || !stroke.points?.length) return false;
  const mode = String(stroke.mode || 'restore_clean').toLowerCase();
  // Los trazos de limpieza/restauración/borrador se hornean en la imagen al guardar;
  // no deben quedarse como manchas encima de la página. Solo se muestra el trazo
  // mientras se dibuja y las máscaras de inpaint pendientes de aplicar.
  if (isDrawing) return true;
  return mode === 'inpaint' && !stroke.applied;
}

function updateHeavyActions() {
  if (applyInpaintBtn) applyInpaintBtn.disabled = !hasPendingInpaintStroke();
}

function scheduleAutoSave(reason = 'auto', delay = 1200) {
  const page = currentPage();
  if (!page || page.status !== 'ready') return;
  if (hasPendingInpaintStroke()) {
    setAutosaveStatus('pending', 'Hay una máscara de inpaint pendiente. Pulsa “Aplicar inpaint” para procesarla.');
    updateHeavyActions();
    return;
  }
  clearTimeout(state.autosaveTimer);
  setAutosaveStatus('pending', 'Cambios pendientes…');
  state.autosaveTimer = setTimeout(() => saveCurrentPage({ silent: true, reason }), delay);
}

function markDirty(options = {}) {
  state.dirty = true;
  updateHeavyActions();
  if (options.autosave) scheduleAutoSave(options.reason || 'auto');
}

function showSetupView() {
  setupView.classList.remove('hidden');
  workView.classList.add('hidden');
}

function showWorkView() {
  setupView.classList.add('hidden');
  workView.classList.remove('hidden');
  requestAnimationFrame(() => renderOverlay());
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;
  try { payload = await response.json(); } catch (_) {}
  if (!response.ok) {
    const detail = payload?.detail || response.statusText || 'Error desconocido';
    throw new Error(detail);
  }
  return payload;
}

advancedToggle.addEventListener('click', () => {
  const willOpen = advancedOptions.classList.contains('hidden');
  advancedOptions.classList.toggle('hidden');
  advancedToggle.textContent = willOpen ? 'Ocultar opciones avanzadas' : 'Mostrar opciones avanzadas';
});

folderInput.addEventListener('change', () => {
  const files = Array.from(folderInput.files || []).filter(isImageFile);
  folderLabel.textContent = files.length ? `${files.length} imagen(es) seleccionadas` : 'Selecciona una carpeta con .jpg, .png, .webp o .bmp.';
});

zipInput.addEventListener('change', () => {
  const file = zipInput.files?.[0];
  zipLabel.textContent = file ? file.name : 'O carga un zip con las páginas.';
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const folderFiles = Array.from(folderInput.files || []).filter(isImageFile);
  const zipFile = zipInput.files?.[0] || null;
  if (!folderFiles.length && !zipFile) {
    showToast('Selecciona una carpeta o un archivo zip.');
    return;
  }

  const data = new FormData();
  data.append('title', titleInput.value || '');
  data.append('source_language', sourceLanguage.value || 'Japonés');
  data.append('target_language', targetLanguage.value || 'Español');
  data.append('translator', translatorSelect.value || 'llm');
  data.append('detection_engine', detectionEngine.value || 'auto');
  data.append('transcription_engine', transcriptionEngine.value || 'auto');
  if (zipFile) data.append('zip_file', zipFile, zipFile.name);
  for (const file of folderFiles) {
    data.append('images', file, file.webkitRelativePath || file.name);
  }

  try {
    uploadForm.querySelector('button[type="submit"]').disabled = true;
    showToast('Proyecto cargado. Iniciando procesamiento…');
    const job = await requestJson('/api/jobs', { method: 'POST', body: data });
    resetEditorState();
    state.job = job;
    state.pageIndex = 0;
    setTool('select');
    showWorkView();
    renderJob(job);
    startPolling(job.job_id);
  } catch (error) {
    showToast(error.message);
  } finally {
    uploadForm.querySelector('button[type="submit"]').disabled = false;
  }
});

continueLastBtn.addEventListener('click', () => {
  if (!state.lastJob) return;
  resetEditorState();
  state.job = state.lastJob;
  state.pageIndex = Math.min(state.lastJob.active_page || 0, Math.max(0, (state.lastJob.pages || []).length - 1));
  setTool('select');
  showWorkView();
  renderJob(state.lastJob);
  if (!['ready', 'failed'].includes(state.lastJob.status)) startPolling(state.lastJob.job_id);
});

newJobBtn.addEventListener('click', () => {
  showSetupView();
  showToast('Configura un nuevo trabajo. El procesamiento anterior sigue guardado.');
});

function isImageFile(file) {
  return Boolean(file && (file.type?.startsWith('image/') || /\.(jpe?g|png|bmp|webp)$/i.test(file.name)));
}

function resetEditorState() {
  state.selectedRegion = null;
  state.variant = 'current';
  state.regions = [];
  state.brushStrokes = [];
  state.dragging = null;
  state.drawingRegion = null;
  state.drawingStroke = null;
  state.dirty = false;
  state.pageStamp = '';
  state.deletedStack = [];
  state.zoom = 1;
  state.fitZoom = 1;
  clearTimeout(state.autosaveTimer);
  state.deferredOverlayRender = false;
  state.inlineEditingIndex = null;
  setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
  updateHeavyActions();
  markActiveVariant();
}

function startPolling(jobId) {
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(async () => {
    try {
      const job = await requestJson(`/api/jobs/${jobId}`);
      const currentPageBefore = state.job?.pages?.[state.pageIndex]?.updated_at;
      state.job = job;
      renderJob(job);
      const currentPageAfter = job.pages?.[state.pageIndex]?.updated_at;
      if (currentPageBefore !== currentPageAfter) renderCurrentPage();
      if (['ready', 'failed'].includes(job.status)) clearInterval(state.polling);
    } catch (error) {
      console.warn(error);
    }
  }, 1800);
}

function renderJob(job) {
  if (!job) return;
  jobTitle.textContent = job.title || 'Proyecto';
  jobMessage.textContent = job.message || '';
  const opts = job.options || {};
  jobOptionsSummary.textContent = `${opts.source_language || 'Entrada'} → ${opts.target_language || 'Salida'} · OCR det: ${opts.detection_engine || 'auto'} · OCR trans: ${opts.transcription_engine || 'auto'} · ${opts.translator === 'google' ? 'Google' : 'LLM'}`;
  jobBadge.textContent = readableStatus(job.status);
  jobBadge.className = `badge ${job.status === 'ready' ? 'ready' : job.status === 'failed' ? 'failed' : ''}`;
  progressBar.style.width = `${job.progress || 0}%`;
  progressText.textContent = `${job.processed_count || 0} listas · ${job.failed_count || 0} fallidas · ${job.total_count || 0} total`;
  const exportablePages = (job.pages || []).filter((page) => page.status === 'ready' || page.has_corrected).length;
  exportBtn.disabled = exportablePages === 0;
  exportBtn.textContent = exportablePages > 0 ? `Exportar ZIP (${exportablePages})` : 'Exportar ZIP';
  renderPageList(job.pages || []);
  renderCurrentPage();
}

function readableStatus(status) {
  return {
    queued: 'En cola',
    pending: 'Pendiente',
    processing: 'Procesando',
    ready: 'Listo',
    failed: 'Con errores',
    corrected: 'Corregido',
  }[status] || status;
}

function renderPageList(pages) {
  pageList.innerHTML = '';
  for (const page of pages) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `page-item ${page.index === state.pageIndex ? 'active' : ''}`;
    button.innerHTML = `
      <span class="page-number">${String(page.index + 1).padStart(2, '0')}</span>
      <span class="page-name" title="${escapeHtml(page.source_filename)}">${escapeHtml(page.source_filename)}</span>
      <span class="status-dot ${page.display_status || page.status}"></span>
    `;
    button.addEventListener('click', () => {
      const inlineEditor = activeInlineTextEditor();
      if (inlineEditor) syncInlineTextToRegion(inlineEditor, { autosave: false });
      if (state.dirty) saveCurrentPage({ silent: true, force: true, reason: 'cambio de página' }).catch(console.warn);
      state.pageIndex = page.index;
      state.selectedRegion = null;
      state.dirty = false;
      state.drawingRegion = null;
      state.drawingStroke = null;
      state.deletedStack = [];
      clearTimeout(state.autosaveTimer);
      setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
      renderJob(state.job);
    });
    pageList.appendChild(button);
  }
}

function currentPage() {
  return state.job?.pages?.[state.pageIndex] || null;
}

function renderCurrentPage() {
  const page = currentPage();
  if (!page) return;
  const activeCount = countActiveRegions(page.regions || []);
  pageHeading.textContent = `Página ${page.index + 1}`;
  pageStatus.textContent = page.message || readableStatus(page.status);
  regionCount.textContent = `${activeCount} regiones`;
  prevBtn.disabled = state.pageIndex <= 0;
  nextBtn.disabled = state.pageIndex >= (state.job?.pages?.length || 1) - 1;

  if (page.status !== 'ready') {
    imageStage.classList.add('hidden');
    waitState.classList.remove('hidden');
    overlayLayer.innerHTML = '';
    state.regions = [];
    state.brushStrokes = [];
    showNoRegion();
    return;
  }

  waitState.classList.add('hidden');
  imageStage.classList.remove('hidden');
  const pageStamp = `${state.job.job_id}:${page.index}:${page.updated_at || ''}`;
  if (!state.dirty && state.pageStamp !== pageStamp) {
    state.regions = cloneRegions(page.regions || []);
    state.brushStrokes = cloneBrushStrokes(page.brush_strokes || []);
    state.pageStamp = pageStamp;
    if (state.selectedRegion != null && !isSelectableRegion(state.regions[state.selectedRegion])) state.selectedRegion = null;
  }
  const url = page.images[state.variant] || page.images.current;
  const bustedUrl = `${url}?t=${encodeURIComponent(page.updated_at || Date.now())}`;
  if (pageImage.src !== new URL(bustedUrl, location.href).href) pageImage.src = bustedUrl;
  pageImage.onload = () => {
    state.naturalWidth = pageImage.naturalWidth || 1;
    state.naturalHeight = pageImage.naturalHeight || 1;
    setFitZoomIfNeeded();
    applyZoom();
    renderOverlay();
  };
  if (pageImage.complete) {
    state.naturalWidth = pageImage.naturalWidth || 1;
    state.naturalHeight = pageImage.naturalHeight || 1;
    setFitZoomIfNeeded();
    applyZoom();
    renderOverlay();
  }
  if (state.selectedRegion == null && state.tool === 'select') {
    const first = state.regions.findIndex(isSelectableRegion);
    if (first >= 0) selectRegion(first, false);
  }
  if (!state.regions.some(isSelectableRegion)) showNoRegion();
  regionCount.textContent = `${state.regions.filter(isSelectableRegion).length} regiones`;
  updateHeavyActions();
}


function countActiveRegions(regions) {
  return regions.filter((region) => !region.deleted).length;
}

function isSelectableRegion(region) {
  return Boolean(region && !region.deleted);
}

function cloneRegions(regions) {
  return regions.map((region, idx) => {
    const bbox = normalizeBox(region.bbox || [0, 0, 1, 1]);
    const sourceBox = normalizeBox(region.source_bbox || region.original_bbox || bbox);
    return {
      index: Number(region.index ?? idx),
      bbox,
      source_bbox: sourceBox,
      original_text: region.original_text || '',
      translated_text: region.translated_text ?? region.text ?? '',
      style: region.style || 'dialogo',
      type: region.type || '',
      confidence: region.confidence || 0,
      restore_original: Boolean(region.restore_original),
      visible: region.visible !== false,
      modified: Boolean(region.modified || region.corrected),
      manual: Boolean(region.manual || region.created_manually),
      deleted: Boolean(region.deleted),
      auto_font_size: region.auto_font_size !== false,
      font_size: clampFontSize(region.font_size, null),
    };
  });
}

function normalizeBox(raw) {
  const values = (raw || [0, 0, 1, 1]).slice(0, 4).map((value) => Number(value));
  while (values.length < 4) values.push(1);
  return values.map((value, index) => Math.round(Number.isFinite(value) ? value : index < 2 ? 0 : 1));
}

function cloneBrushStrokes(strokes) {
  return (strokes || []).map((stroke) => ({
    points: (stroke.points || []).map((p) => [Number(p[0]), Number(p[1])]).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])),
    radius: Number(stroke.radius || 18),
    mode: stroke.mode || 'restore_clean',
    applied: Boolean(stroke.applied),
  })).filter((stroke) => stroke.points.length);
}

function markRegionModified(region) {
  if (!region) return;
  region.modified = true;
  markDirty();
}


function imagePointFromEvent(event) {
  const rect = pageImage.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * state.naturalWidth;
  const y = ((event.clientY - rect.top) / rect.height) * state.naturalHeight;
  return [
    Math.max(0, Math.min(state.naturalWidth, Math.round(x))),
    Math.max(0, Math.min(state.naturalHeight, Math.round(y))),
  ];
}

function updateBrushCursorFromEvent(event, immediate = false) {
  if (state.tool !== 'brush' || !pageImage || imageStage.classList.contains('hidden')) {
    if (state.brushCursor.visible) {
      state.brushCursor.visible = false;
      requestOverlayRender();
    }
    return;
  }
  const rect = pageImage.getBoundingClientRect();
  const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
  if (!inside) {
    if (state.brushCursor.visible) {
      state.brushCursor.visible = false;
      requestOverlayRender();
    }
    return;
  }
  const [x, y] = imagePointFromEvent(event);
  state.brushCursor = { x, y, visible: true };
  if (immediate) renderOverlay();
  else requestOverlayRender();
}

function hideBrushCursor() {
  if (!state.brushCursor.visible) return;
  state.brushCursor.visible = false;
  requestOverlayRender();
}

function clampZoom(value) {
  return Math.max(0.25, Math.min(3, Number(value) || 1));
}

function setFitZoomIfNeeded(force = false) {
  if (!canvasCard || !state.naturalWidth) return;
  const available = Math.max(260, canvasCard.clientWidth - 18);
  const fit = clampZoom(available / state.naturalWidth);
  state.fitZoom = fit;
  if (force || !state.zoom || Math.abs(state.zoom - 1) < 0.001) {
    state.zoom = fit;
  }
}

function applyZoom() {
  if (!pageImage || !imageStage || !state.naturalWidth) return;
  state.zoom = clampZoom(state.zoom);
  const width = Math.max(1, Math.round(state.naturalWidth * state.zoom));
  pageImage.style.width = `${width}px`;
  pageImage.style.height = 'auto';
  imageStage.style.width = `${width}px`;
  imageStage.style.height = `${Math.max(1, Math.round(state.naturalHeight * state.zoom))}px`;
  if (zoomSlider) zoomSlider.value = String(Math.round(state.zoom * 100));
  if (zoomValue) zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
  requestAnimationFrame(renderOverlay);
}

function zoomTo(nextZoom, anchorEvent = null) {
  if (!canvasCard || !pageImage) {
    state.zoom = clampZoom(nextZoom);
    applyZoom();
    return;
  }
  const oldRect = pageImage.getBoundingClientRect();
  const oldZoom = state.zoom;
  const localX = anchorEvent ? anchorEvent.clientX - oldRect.left : oldRect.width / 2;
  const localY = anchorEvent ? anchorEvent.clientY - oldRect.top : oldRect.height / 2;
  const ratioX = oldRect.width ? localX / oldRect.width : 0.5;
  const ratioY = oldRect.height ? localY / oldRect.height : 0.5;
  state.zoom = clampZoom(nextZoom);
  applyZoom();
  const newWidth = state.naturalWidth * state.zoom;
  const newHeight = state.naturalHeight * state.zoom;
  const viewportX = anchorEvent ? anchorEvent.clientX - canvasCard.getBoundingClientRect().left : canvasCard.clientWidth / 2;
  const viewportY = anchorEvent ? anchorEvent.clientY - canvasCard.getBoundingClientRect().top : canvasCard.clientHeight / 2;
  canvasCard.scrollLeft = Math.max(0, ratioX * newWidth - viewportX);
  canvasCard.scrollTop = Math.max(0, ratioY * newHeight - viewportY);
  if (Math.abs(oldZoom - state.zoom) > 0.001) renderOverlay();
}

function getScale() {
  const rect = pageImage.getBoundingClientRect();
  return { scaleX: rect.width / state.naturalWidth, scaleY: rect.height / state.naturalHeight, width: rect.width, height: rect.height };
}

function requestOverlayRender() {
  if (shouldDeferOverlayRender()) {
    state.deferredOverlayRender = true;
    return;
  }
  if (state.overlayRaf) return;
  state.overlayRaf = requestAnimationFrame(() => {
    state.overlayRaf = null;
    renderOverlay();
  });
}

function updateBrushSizeLabel() {
  if (brushSizeValue && brushSize) brushSizeValue.textContent = `${Number(brushSize.value || 22)} px`;
}

function brushModeLabel(mode) {
  const normalized = String(mode || 'restore_clean').toLowerCase();
  if (normalized === 'inpaint') return 'inpaint';
  if (isMaskEraserMode(normalized)) return 'restaurar original';
  return 'limpieza';
}

function shouldShowLiveText(region, idx) {
  if (!isSelectableRegion(region) || region.visible === false) return false;
  return idx === state.selectedRegion && (state.dirty || state.dragging || state.tool === 'select');
}

function previewFontSize(region) {
  const { scaleX, scaleY } = getScale();
  const scale = (scaleX + scaleY) / 2;
  const baseSize = usesAutoFontSize(region) ? estimatedAutoFontSize(region) : effectiveManualFontSize(region);
  return Math.max(4, Math.min(240, Math.round(baseSize * scale)));
}

function activeInlineTextEditor() {
  const active = document.activeElement;
  return active?.classList?.contains('region-text-editor') ? active : null;
}

function shouldDeferOverlayRender() {
  return Boolean(activeInlineTextEditor() && !state.dragging && !state.drawingRegion && !state.drawingStroke);
}

function flushDeferredOverlayRender() {
  if (!state.deferredOverlayRender) return;
  state.deferredOverlayRender = false;
  requestAnimationFrame(() => renderOverlay({ force: true }));
}

function renderOverlay(options = {}) {
  if (!options.force && shouldDeferOverlayRender()) {
    state.deferredOverlayRender = true;
    return;
  }
  state.deferredOverlayRender = false;
  if (!pageImage || !overlayLayer || workView.classList.contains('hidden')) return;
  overlayLayer.innerHTML = '';
  const { scaleX, scaleY, width, height } = getScale();
  if (!Number.isFinite(width) || width <= 0) return;
  overlayLayer.style.width = `${width}px`;
  overlayLayer.style.height = `${height}px`;
  overlayLayer.className = `overlay-layer tool-${state.tool}`;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'stroke-layer');
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const allStrokes = state.drawingStroke
    ? [...state.brushStrokes.map((stroke) => ({ stroke, drawing: false })), { stroke: state.drawingStroke, drawing: true }]
    : state.brushStrokes.map((stroke) => ({ stroke, drawing: false }));
  for (const item of allStrokes) {
    const stroke = item.stroke;
    if (!shouldDisplayStroke(stroke, item.drawing)) continue;
    const mode = String(stroke.mode || 'restore_clean').toLowerCase();
    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', stroke.points.map(([x, y]) => `${x * scaleX},${y * scaleY}`).join(' '));
    polyline.setAttribute('class', `mask-stroke ${mode === 'restore_original' ? 'original-stroke' : mode === 'inpaint' ? 'inpaint-stroke' : isMaskEraserMode(mode) ? 'eraser-stroke' : 'clean-stroke'}`);
    polyline.setAttribute('stroke-width', String(Math.max(2, stroke.radius * 2 * ((scaleX + scaleY) / 2))));
    svg.appendChild(polyline);
  }
  overlayLayer.appendChild(svg);

  state.regions.forEach((region, idx) => {
    if (!isSelectableRegion(region)) return;
    const [x, y, w, h] = region.bbox;
    const box = document.createElement('div');
    const isSelected = idx === state.selectedRegion;
    box.className = `region-box ${isSelected ? 'selected' : ''} ${region.restore_original ? 'restore' : ''} ${region.manual ? 'manual' : ''} ${region.visible === false ? 'hidden-region' : ''}`;
    box.style.left = `${x * scaleX}px`;
    box.style.top = `${y * scaleY}px`;
    box.style.width = `${w * scaleX}px`;
    box.style.height = `${h * scaleY}px`;
    box.dataset.index = idx;
    box.tabIndex = 0;

    const label = document.createElement('span');
    label.className = 'region-label';
    label.textContent = `#${idx + 1}${region.manual ? ' · manual' : ''}`;
    box.appendChild(label);

    const previewText = region.translated_text || region.original_text || '';
    const showInlineEditor = isSelected && state.tool === 'select' && region.visible !== false;
    const showPreview = !showInlineEditor && shouldShowLiveText(region, idx);
    if (showInlineEditor) {
      box.classList.add('live-editing');
      const editor = document.createElement('div');
      editor.className = `region-text-editor ${previewText.trim() ? '' : 'empty'}`;
      editor.contentEditable = 'plaintext-only';
      editor.spellcheck = false;
      editor.dataset.index = String(idx);
      applyRendererTextLayout(editor, region);
      editor.textContent = previewText;
      editor.title = 'Escribe la traducción directamente aquí';
      editor.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
      });
      editor.addEventListener('click', (event) => {
        event.stopPropagation();
        selectRegion(idx, false);
      });
      editor.addEventListener('focus', onInlineTextFocus);
      editor.addEventListener('blur', onInlineTextBlur);
      editor.addEventListener('paste', onInlineTextPaste);
      editor.addEventListener('input', onInlineTextInput);
      editor.addEventListener('keydown', onInlineTextKeyDown);
      box.appendChild(editor);
    } else if (showPreview) {
      box.classList.add('live-preview');
      const preview = document.createElement('span');
      preview.className = `region-text-preview ${previewText.trim() ? '' : 'empty'}`;
      applyRendererTextLayout(preview, region);
      preview.textContent = previewText || 'Texto';
      box.appendChild(preview);
    }

    const handle = document.createElement('span');
    handle.className = 'resize-handle';
    handle.title = 'Redimensionar';
    box.appendChild(handle);

    box.addEventListener('pointerdown', onBoxPointerDown);
    box.addEventListener('click', (event) => {
      event.stopPropagation();
      if (state.tool === 'select') selectRegion(idx);
    });
    overlayLayer.appendChild(box);
  });

  if (state.drawingRegion) {
    const [x, y, w, h] = state.drawingRegion.bbox;
    const temp = document.createElement('div');
    temp.className = 'region-box drawing';
    temp.style.left = `${x * scaleX}px`;
    temp.style.top = `${y * scaleY}px`;
    temp.style.width = `${w * scaleX}px`;
    temp.style.height = `${h * scaleY}px`;
    temp.innerHTML = '<span class="region-label">nueva</span>';
    overlayLayer.appendChild(temp);
  }

  if (state.tool === 'brush' && state.brushCursor.visible) {
    const radius = Number(brushSize?.value || 22);
    const scale = (scaleX + scaleY) / 2;
    const visualRadius = Math.max(2, radius * scale);
    const cursor = document.createElement('div');
    const mode = String(brushMode?.value || 'restore_clean').toLowerCase();
    cursor.className = `brush-cursor mode-${isMaskEraserMode(mode) ? 'mask_eraser' : mode}`;
    cursor.style.left = `${state.brushCursor.x * scaleX - visualRadius}px`;
    cursor.style.top = `${state.brushCursor.y * scaleY - visualRadius}px`;
    cursor.style.width = `${visualRadius * 2}px`;
    cursor.style.height = `${visualRadius * 2}px`;
    cursor.title = `Pincel de ${brushModeLabel(mode)} · ${radius}px`;
    overlayLayer.appendChild(cursor);
  }
}

function textFromEditable(element) {
  return String(element?.innerText || '').replace(/\r/g, '').replace(/\u00a0/g, ' ');
}

function syncInlineTextToRegion(element, options = {}) {
  const idx = Number(element?.dataset?.index);
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return;
  const nextText = textFromEditable(element);
  element.classList.toggle('empty', !nextText.trim());
  if (region.translated_text === nextText) return;
  region.translated_text = nextText;
  region.modified = true;
  if (idx === state.selectedRegion && translatedText) translatedText.value = nextText;
  markDirty();
  if (options.autosave) scheduleAutoSave(options.reason || 'edición de texto directa');
}

function onInlineTextFocus(event) {
  state.inlineEditingIndex = Number(event.currentTarget?.dataset?.index);
  clearTimeout(state.autosaveTimer);
  setAutosaveStatus('pending', 'Editando texto… se guardará al salir del cuadro.');
}

function onInlineTextBlur(event) {
  syncInlineTextToRegion(event.currentTarget, { autosave: false });
  state.inlineEditingIndex = null;
  if (state.dirty) scheduleAutoSave('salir de edición directa', 180);
  flushDeferredOverlayRender();
}

function onInlineTextInput(event) {
  syncInlineTextToRegion(event.currentTarget, { autosave: false });
  clearTimeout(state.autosaveTimer);
  setAutosaveStatus('pending', 'Editando texto… se guardará al salir del cuadro.');
}

function onInlineTextPaste(event) {
  event.preventDefault();
  const text = event.clipboardData?.getData('text/plain') || '';
  document.execCommand('insertText', false, text);
}

function onInlineTextKeyDown(event) {
  // Dentro de la caja, Enter permite saltos de línea; Escape devuelve el foco al lienzo.
  if (event.key === 'Escape') {
    event.preventDefault();
    event.currentTarget.blur();
    overlayLayer?.focus?.();
  }
}

function onBoxPointerDown(event) {
  if (state.tool !== 'select') return;
  event.preventDefault();
  event.stopPropagation();
  const box = event.currentTarget;
  const idx = Number(box.dataset.index);
  if (!isSelectableRegion(state.regions[idx])) return;
  selectRegion(idx);
  const isResize = event.target.classList.contains('resize-handle');
  const region = state.regions[idx];
  state.dragging = {
    idx,
    isResize,
    startX: event.clientX,
    startY: event.clientY,
    startBox: [...region.bbox],
  };
  box.setPointerCapture(event.pointerId);
}

overlayLayer.addEventListener('pointerdown', (event) => {
  const page = currentPage();
  if (!page || page.status !== 'ready') return;
  if (event.target.closest?.('.region-box') && state.tool === 'select') return;
  event.preventDefault();
  const [x, y] = imagePointFromEvent(event);

  if (state.tool === 'region') {
    state.selectedRegion = null;
    state.drawingRegion = { startX: x, startY: y, bbox: [x, y, 1, 1] };
    overlayLayer.setPointerCapture(event.pointerId);
    showNoRegion();
    renderOverlay();
    return;
  }

  if (state.tool === 'brush') {
    state.brushCursor = { x, y, visible: true };
    state.drawingStroke = { points: [[x, y]], radius: Number(brushSize.value || 22), mode: brushMode.value || 'restore_clean', applied: false };
    overlayLayer.setPointerCapture(event.pointerId);
    markDirty();
    renderOverlay();
    return;
  }

  state.selectedRegion = null;
  showNoRegion();
  renderOverlay();
});

overlayLayer.addEventListener('pointermove', (event) => updateBrushCursorFromEvent(event));
overlayLayer.addEventListener('pointerenter', (event) => updateBrushCursorFromEvent(event));
overlayLayer.addEventListener('pointerleave', hideBrushCursor);

document.addEventListener('pointermove', (event) => {
  if (state.dragging) {
    const drag = state.dragging;
    const region = state.regions[drag.idx];
    if (!isSelectableRegion(region)) return;
    const rect = pageImage.getBoundingClientRect();
    const dx = (event.clientX - drag.startX) / (rect.width / state.naturalWidth);
    const dy = (event.clientY - drag.startY) / (rect.height / state.naturalHeight);
    let [x, y, w, h] = drag.startBox;
    if (drag.isResize) {
      w = Math.max(12, Math.min(state.naturalWidth - x, w + dx));
      h = Math.max(12, Math.min(state.naturalHeight - y, h + dy));
    } else {
      x = Math.max(0, Math.min(state.naturalWidth - w, x + dx));
      y = Math.max(0, Math.min(state.naturalHeight - h, y + dy));
    }
    region.bbox = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
    markRegionModified(region);
    updateEditorFromRegion();
    renderOverlay();
    return;
  }

  if (state.drawingRegion) {
    const [x2, y2] = imagePointFromEvent(event);
    const x1 = state.drawingRegion.startX;
    const y1 = state.drawingRegion.startY;
    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const w = Math.max(1, Math.abs(x2 - x1));
    const h = Math.max(1, Math.abs(y2 - y1));
    state.drawingRegion.bbox = [x, y, w, h].map(Math.round);
    renderOverlay();
    return;
  }

  if (state.drawingStroke) {
    const [x, y] = imagePointFromEvent(event);
    state.brushCursor = { x, y, visible: true };
    const last = state.drawingStroke.points[state.drawingStroke.points.length - 1];
    if (!last || Math.hypot(last[0] - x, last[1] - y) >= 2) {
      state.drawingStroke.points.push([x, y]);
      renderOverlay();
    } else {
      requestOverlayRender();
    }
  }
});

document.addEventListener('pointerup', () => {
  if (state.dragging) scheduleAutoSave(state.dragging.isResize ? 'redimensionar región' : 'mover región');
  state.dragging = null;

  if (state.drawingRegion) {
    const bbox = clampBox(state.drawingRegion.bbox);
    state.drawingRegion = null;
    if (bbox[2] >= 12 && bbox[3] >= 12) {
      const nextIndex = nextRegionIndex();
      const region = {
        index: nextIndex,
        bbox,
        source_bbox: [...bbox],
        original_text: '',
        translated_text: '',
        style: 'dialogo',
        type: 'manual',
        confidence: 0,
        restore_original: false,
        visible: true,
        modified: true,
        manual: true,
        deleted: false,
        auto_font_size: true,
        font_size: 24,
      };
      state.regions.push(region);
      state.selectedRegion = state.regions.length - 1;
      markDirty({ autosave: true, reason: 'nueva región' });
      setTool('select');
      selectRegion(state.selectedRegion, false);
      showToast('Región manual creada. Puedes ejecutar OCR sobre ella.');
    }
    renderOverlay();
  }

  if (state.drawingStroke) {
    if (state.drawingStroke.points.length > 0) {
      state.brushStrokes.push(state.drawingStroke);
      if (state.drawingStroke.mode === 'inpaint') {
        markDirty();
        setAutosaveStatus('pending', 'Máscara de inpaint lista. Pulsa “Aplicar inpaint”.');
        showToast('Máscara de inpaint agregada. Pulsa Aplicar inpaint para procesarla.');
      } else if (isMaskEraserMode(state.drawingStroke.mode)) {
        markDirty();
        if (hasPendingInpaintStroke()) {
          setAutosaveStatus('pending', 'Zona quitada de la máscara pendiente. Pulsa “Aplicar inpaint”.');
          showToast('Zona quitada de la máscara pendiente.');
        } else {
          setAutosaveStatus('saving', 'Restaurando manga original en la zona pintada…');
          showToast(hasAnyInpaintStroke() ? 'Restaurando manga original sobre el inpaint…' : 'Restaurando manga original.');
          saveCurrentPage({ silent: true, force: true, reason: 'borrador de máscara', operation: 'mask_eraser' }).catch(() => {});
        }
      } else {
        markDirty({ autosave: true, reason: 'pincel' });
        showToast(state.drawingStroke.mode === 'restore_original' ? 'Restauración agregada. Se guardará sola.' : 'Limpieza agregada. Se guardará sola.');
      }
    }
    state.drawingStroke = null;
    updateHeavyActions();
    renderOverlay();
  }
});
window.addEventListener('resize', () => { setFitZoomIfNeeded(false); applyZoom(); renderOverlay(); });
window.addEventListener('beforeunload', (event) => {
  if (!state.dirty) return;
  event.preventDefault();
  event.returnValue = '';
});

zoomSlider?.addEventListener('input', () => zoomTo(Number(zoomSlider.value || 100) / 100));
zoomOutBtn?.addEventListener('click', () => zoomTo(state.zoom / 1.15));
zoomInBtn?.addEventListener('click', () => zoomTo(state.zoom * 1.15));
zoomFitBtn?.addEventListener('click', () => { setFitZoomIfNeeded(true); applyZoom(); });
focusBtn?.addEventListener('click', () => {
  if (!workView) return;
  workView.classList.toggle('focus-canvas');
  const active = workView.classList.contains('focus-canvas');
  focusBtn.textContent = active ? 'Salir enfoque' : 'Enfoque';
  focusBtn.setAttribute('aria-pressed', active ? 'true' : 'false');
  requestAnimationFrame(() => { setFitZoomIfNeeded(true); applyZoom(); renderOverlay(); });
});
canvasCard?.addEventListener('wheel', (event) => {
  if (workView.classList.contains('hidden') || imageStage.classList.contains('hidden')) return;
  if (!event.ctrlKey) return;
  event.preventDefault();
  const factor = event.deltaY > 0 ? 1 / 1.12 : 1.12;
  zoomTo(state.zoom * factor, event);
}, { passive: false });

function setTool(tool) {
  state.tool = tool;
  [selectTool, newRegionTool, brushTool].forEach((button) => button.classList.toggle('active', button.dataset.tool === tool));
  const help = {
    select: ['Herramienta activa: seleccionar', 'Haz clic en una región para editarla, muévela o redimensiónala. Atajos: Supr, Ctrl+C, Ctrl+X y Ctrl+V.'],
    region: ['Herramienta activa: nueva región', 'Arrastra sobre la página para crear una caja. Luego usa OCR + traducir región o escribe el texto manualmente.'],
    brush: ['Herramienta activa: pincel', 'Usa Limpiar texto para borrado rápido, Inpaint para reconstrucciones pesadas y Restaurar original/borrar inpaint para devolver la zona pintada al manga original. El círculo del cursor muestra el tamaño real.'],
  }[tool];
  toolHelpTitle.textContent = help[0];
  toolHelpText.textContent = help[1];
  renderOverlay();
}

[selectTool, newRegionTool, brushTool].forEach((button) => {
  button.addEventListener('click', () => setTool(button.dataset.tool));
});
brushSize.addEventListener('input', () => {
  updateBrushSizeLabel();
  setTool('brush');
  requestOverlayRender();
});
brushMode.addEventListener('change', () => {
  setTool('brush');
  requestOverlayRender();
});
updateBrushSizeLabel();
undoBrushBtn.addEventListener('click', () => {
  if (!state.brushStrokes.length) return showToast('No hay pinceladas para deshacer.');
  state.brushStrokes.pop();
  markDirty({ autosave: true, reason: 'deshacer pincel' });
  renderOverlay();
  showToast('Última pincelada eliminada.');
});
undoDeleteBtn.addEventListener('click', undoLastDelete);

function selectRegion(idx, rerender = true) {
  if (!isSelectableRegion(state.regions[idx])) return showNoRegion();
  state.selectedRegion = idx;
  noRegion.classList.add('hidden');
  regionEditor.classList.remove('hidden');
  updateEditorFromRegion();
  if (rerender) renderOverlay();
}

function showNoRegion() {
  state.selectedRegion = null;
  noRegion.classList.remove('hidden');
  regionEditor.classList.add('hidden');
}

function updateEditorFromRegion() {
  const region = state.regions[state.selectedRegion];
  if (!isSelectableRegion(region)) return;
  originalText.value = region.original_text || '';
  translatedText.value = region.translated_text || '';
  const [x, y, w, h] = region.bbox;
  boxX.value = x; boxY.value = y; boxW.value = w; boxH.value = h;
  restoreOriginal.checked = Boolean(region.restore_original);
  visibleText.checked = region.visible !== false;
  if (autoFontSize) autoFontSize.checked = usesAutoFontSize(region);
  syncFontControlsFromValue(effectiveManualFontSize(region));
  updateFontControlsDisabled();
  regionTypeBadge.textContent = region.manual ? 'Región manual' : 'Región detectada';
}

function updateRegionFromEditor(options = {}) {
  const region = state.regions[state.selectedRegion];
  if (!isSelectableRegion(region)) return;
  const nextOriginalText = originalText.value;
  const nextText = translatedText.value;
  const nextBox = clampBox([Number(boxX.value || 0), Number(boxY.value || 0), Number(boxW.value || 12), Number(boxH.value || 12)]);
  const nextRestore = restoreOriginal.checked;
  const nextVisible = visibleText.checked;
  const nextAutoFont = autoFontSize ? autoFontSize.checked : true;
  const nextFontSize = clampFontSize(fontSizeNumber?.value || fontSize?.value, effectiveManualFontSize(region));
  const currentFontSize = clampFontSize(region.font_size, null);
  const changed =
    region.original_text !== nextOriginalText ||
    region.translated_text !== nextText ||
    region.restore_original !== nextRestore ||
    region.visible !== nextVisible ||
    usesAutoFontSize(region) !== nextAutoFont ||
    currentFontSize !== nextFontSize ||
    region.bbox.some((value, index) => value !== nextBox[index]);

  region.original_text = nextOriginalText;
  region.translated_text = nextText;
  region.bbox = nextBox;
  region.restore_original = nextRestore;
  region.visible = nextVisible;
  region.auto_font_size = nextAutoFont;
  region.font_size = nextFontSize;
  updateFontControlsDisabled();
  if (changed) {
    markRegionModified(region);
    if (options.autosave !== false) scheduleAutoSave('edición de región');
  }
  if (options.render !== false) renderOverlay();
}

function clampBox(rawBox) {
  let [x, y, w, h] = normalizeBox(rawBox);
  w = Math.max(12, Math.min(w, state.naturalWidth));
  h = Math.max(12, Math.min(h, state.naturalHeight));
  x = Math.max(0, Math.min(state.naturalWidth - w, x));
  y = Math.max(0, Math.min(state.naturalHeight - h, y));
  return [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
}

function nextRegionIndex() {
  return state.regions.reduce((max, region) => Math.max(max, Number(region.index || 0)), -1) + 1;
}

[originalText, translatedText, boxX, boxY, boxW, boxH, restoreOriginal, visibleText, autoFontSize].forEach((element) => {
  if (!element) return;
  element.addEventListener('input', updateRegionFromEditor);
  element.addEventListener('change', updateRegionFromEditor);
});
fontSize?.addEventListener('input', () => {
  syncFontControlsFromValue(fontSize.value);
  updateRegionFromEditor();
});
fontSizeNumber?.addEventListener('input', () => {
  syncFontControlsFromValue(fontSizeNumber.value);
  updateRegionFromEditor();
});
fontSizeNumber?.addEventListener('change', () => {
  syncFontControlsFromValue(fontSizeNumber.value);
  updateRegionFromEditor();
});
autoFontSize?.addEventListener('change', () => {
  updateFontControlsDisabled();
  updateRegionFromEditor();
});

ocrRegionBtn.addEventListener('click', async () => {
  const page = currentPage();
  const region = state.regions[state.selectedRegion];
  if (!page || page.status !== 'ready' || !isSelectableRegion(region)) return showToast('Selecciona una región lista.');
  updateRegionFromEditor();
  ocrRegionBtn.disabled = true;
  try {
    showToast('Ejecutando OCR sobre la región…');
    const result = await requestJson(`/api/jobs/${state.job.job_id}/pages/${page.index}/ocr-region`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox: region.bbox, translate: true }),
    });
    region.original_text = result.original_text || '';
    if (result.translated_text) region.translated_text = result.translated_text;
    region.bbox = result.bbox || region.bbox;
    region.source_bbox = region.source_bbox || [...region.bbox];
    markRegionModified(region);
    updateEditorFromRegion();
    renderOverlay();
    showToast(result.translated_text ? 'OCR y traducción aplicados.' : 'OCR aplicado. No hubo texto para traducir.');
  } catch (error) {
    showToast(error.message);
  } finally {
    ocrRegionBtn.disabled = false;
  }
});

copyRegionBtn.addEventListener('click', () => copySelectedRegion(false));
deleteRegionBtn.addEventListener('click', () => deleteSelectedRegion());

function copySelectedRegion(cut = false) {
  const region = state.regions[state.selectedRegion];
  if (!isSelectableRegion(region)) return showToast('Selecciona una región para copiar.');
  state.clipboardRegion = cloneRegionForClipboard(region);
  if (cut) {
    deleteSelectedRegion('Región cortada. Usa Ctrl+V para pegarla.');
  } else {
    showToast('Región copiada. Usa Ctrl+V para pegarla.');
  }
}

function cloneRegionForClipboard(region) {
  return JSON.parse(JSON.stringify({
    ...region,
    deleted: false,
    visible: region.visible !== false,
  }));
}

function pasteRegion() {
  if (!state.clipboardRegion) return showToast('No hay una región copiada.');
  const page = currentPage();
  if (!page || page.status !== 'ready') return;
  const base = cloneRegionForClipboard(state.clipboardRegion);
  const [x, y, w, h] = base.bbox;
  const offset = 18;
  const bbox = clampBox([x + offset, y + offset, w, h]);
  const pasted = {
    ...base,
    index: nextRegionIndex(),
    bbox,
    source_bbox: [...bbox],
    manual: true,
    modified: true,
    deleted: false,
  };
  state.regions.push(pasted);
  state.selectedRegion = state.regions.length - 1;
  markDirty({ autosave: true, reason: 'pegar región' });
  setTool('select');
  selectRegion(state.selectedRegion);
  showToast('Región pegada como nueva caja manual.');
}

function deleteSelectedRegion(message = 'Región eliminada. Guarda para aplicar el borrado limpio.') {
  const idx = state.selectedRegion;
  if (idx == null || !isSelectableRegion(state.regions[idx])) return showToast('Selecciona una región para eliminar.');
  deleteRegionAt(idx, message);
}

function deleteRegionAt(idx, message) {
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return;
  state.deletedStack.push({ idx, snapshot: cloneRegionForClipboard(region) });
  region.deleted = true;
  region.visible = false;
  region.modified = true;
  region.restore_original = false;
  region.source_bbox = region.source_bbox || [...region.bbox];
  markDirty({ autosave: true, reason: 'borrar región' });
  showNoRegion();
  renderOverlay();
  regionCount.textContent = `${state.regions.filter(isSelectableRegion).length} regiones`;
  showToast(message);
}

function undoLastDelete() {
  const item = state.deletedStack.pop();
  if (!item) return showToast('No hay borrados para deshacer.');
  const current = state.regions[item.idx];
  if (current) {
    state.regions[item.idx] = { ...item.snapshot, deleted: false, visible: item.snapshot.visible !== false, modified: true };
    state.selectedRegion = item.idx;
  } else {
    state.regions.push({ ...item.snapshot, deleted: false, visible: item.snapshot.visible !== false, modified: true });
    state.selectedRegion = state.regions.length - 1;
  }
  markDirty({ autosave: true, reason: 'deshacer borrado' });
  selectRegion(state.selectedRegion);
  regionCount.textContent = `${state.regions.filter(isSelectableRegion).length} regiones`;
  showToast('Borrado deshecho.');
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = target.tagName?.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable;
}

document.addEventListener('keydown', (event) => {
  if (isTypingTarget(event.target)) return;
  const page = currentPage();
  if (!page || page.status !== 'ready' || workView.classList.contains('hidden')) return;

  const key = event.key.toLowerCase();
  const ctrl = event.ctrlKey || event.metaKey;
  if (key === 'delete' || key === 'backspace') {
    event.preventDefault();
    deleteSelectedRegion();
    return;
  }
  if (!ctrl) return;
  if (key === 'c') {
    event.preventDefault();
    copySelectedRegion(false);
  } else if (key === 'x') {
    event.preventDefault();
    copySelectedRegion(true);
  } else if (key === 'v') {
    event.preventDefault();
    pasteRegion();
  }
});

exportBtn.addEventListener('click', async () => {
  if (!state.job) return;
  exportBtn.disabled = true;
  try {
    showToast('Preparando ZIP de resultados…');
    const response = await fetch(`/api/jobs/${state.job.job_id}/export`);
    if (!response.ok) {
      let payload = null;
      try { payload = await response.json(); } catch (_) {}
      throw new Error(payload?.detail || response.statusText || 'No se pudo exportar.');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const disposition = response.headers.get('content-disposition') || '';
    const match = disposition.match(/filename\*?=(?:UTF-8''|\")?([^\";]+)/i);
    link.href = url;
    link.download = decodeURIComponent(match?.[1] || `${state.job.title || 'manga'}_resultado.zip`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast('ZIP exportado.');
  } catch (error) {
    showToast(error.message);
  } finally {
    renderJob(state.job);
  }
});

function buildRenderPayload(operation = 'render') {
  return {
    operation,
    regions: state.regions.map((region, idx) => ({
      index: region.index ?? idx,
      bbox: region.bbox,
      text: region.translated_text || '',
      original_text: region.original_text || '',
      style: region.style || 'dialogo',
      restore_original: Boolean(region.restore_original),
      visible: region.visible !== false,
      modified: Boolean(region.modified || region.manual || region.deleted),
      manual: Boolean(region.manual),
      deleted: Boolean(region.deleted),
      source_bbox: region.source_bbox || region.bbox,
      auto_font_size: region.auto_font_size !== false,
      font_size: clampFontSize(region.font_size, null),
    })),
    brush_strokes: state.brushStrokes.map((stroke) => ({
      points: stroke.points,
      radius: Number(stroke.radius || 18),
      mode: stroke.mode || 'restore_clean',
      applied: Boolean(stroke.applied),
    })),
  };
}

async function saveCurrentPage({ silent = false, force = false, reason = 'manual', markInpaintApplied = false, operation = null } = {}) {
  const page = currentPage();
  if (!page || page.status !== 'ready') {
    if (!silent) showToast('La página aún no está lista.');
    return null;
  }
  const inlineEditor = activeInlineTextEditor();
  if (inlineEditor) syncInlineTextToRegion(inlineEditor, { autosave: false });
  else if (state.selectedRegion != null) updateRegionFromEditor({ render: false, autosave: false });
  if (!markInpaintApplied && operation !== 'mask_eraser' && hasPendingInpaintStroke()) {
    setAutosaveStatus('pending', 'Hay una máscara de inpaint pendiente. Pulsa “Aplicar inpaint”.');
    updateHeavyActions();
    return null;
  }
  clearTimeout(state.autosaveTimer);
  saveBtn.disabled = true;
  state.autosaveInFlight = true;
  setAutosaveStatus(markInpaintApplied ? 'saving' : 'saving', markInpaintApplied ? 'Aplicando inpaint…' : 'Guardando cambios…');
  try {
    const updatedPage = await requestJson(`/api/jobs/${state.job.job_id}/pages/${page.index}/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildRenderPayload(operation || (markInpaintApplied ? 'inpaint' : 'render'))),
    });
    state.job.pages[page.index] = updatedPage;
    state.dirty = false;
    state.pageStamp = `${state.job.job_id}:${page.index}:${updatedPage.updated_at || ''}`;
    state.variant = 'current';
    state.deletedStack = [];
    state.brushStrokes = cloneBrushStrokes(updatedPage.brush_strokes || []);
    markActiveVariant();
    setAutosaveStatus('saved', markInpaintApplied ? 'Inpaint aplicado y guardado.' : 'Cambios guardados automáticamente.');
    if (!silent || markInpaintApplied) showToast(markInpaintApplied ? 'Inpaint aplicado.' : 'Corrección guardada.');
    if (activeInlineTextEditor()) {
      state.deferredOverlayRender = true;
      updateHeavyActions();
    } else {
      renderCurrentPage();
      updateHeavyActions();
    }
    return updatedPage;
  } catch (error) {
    setAutosaveStatus('error', error.message || 'No se pudo guardar.');
    if (!silent) showToast(error.message);
    throw error;
  } finally {
    state.autosaveInFlight = false;
    saveBtn.disabled = false;
  }
}

saveBtn.addEventListener('click', () => {
  saveCurrentPage({ silent: false, force: true, reason: 'guardar ahora' }).catch(() => {});
});

applyInpaintBtn?.addEventListener('click', () => {
  if (!hasPendingInpaintStroke()) return showToast('Dibuja primero una máscara con el pincel en modo inpaint.');
  saveCurrentPage({ silent: false, force: true, reason: 'inpaint', markInpaintApplied: true }).catch(() => {});
});

resetBtn.addEventListener('click', async () => {
  const page = currentPage();
  if (!page || page.status !== 'ready') return;
  resetBtn.disabled = true;
  try {
    const updatedPage = await requestJson(`/api/jobs/${state.job.job_id}/pages/${page.index}/reset`, { method: 'POST' });
    state.job.pages[page.index] = updatedPage;
    state.dirty = false;
    state.pageStamp = `${state.job.job_id}:${page.index}:${updatedPage.updated_at || ''}`;
    state.variant = 'translated';
    state.selectedRegion = null;
    state.brushStrokes = [];
    state.deletedStack = [];
    markActiveVariant();
    setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
    updateHeavyActions();
    showToast('Se restauró la salida automática.');
    renderCurrentPage();
  } catch (error) {
    showToast(error.message);
  } finally {
    resetBtn.disabled = false;
  }
});

prevBtn.addEventListener('click', () => {
  if (!state.job || state.pageIndex <= 0) return;
  state.pageIndex -= 1;
  state.selectedRegion = null;
  state.dirty = false;
  state.deletedStack = [];
  clearTimeout(state.autosaveTimer);
  state.deferredOverlayRender = false;
  state.inlineEditingIndex = null;
  setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
  renderJob(state.job);
});
nextBtn.addEventListener('click', () => {
  if (!state.job || state.pageIndex >= state.job.pages.length - 1) return;
  state.pageIndex += 1;
  state.selectedRegion = null;
  state.dirty = false;
  state.deletedStack = [];
  clearTimeout(state.autosaveTimer);
  state.deferredOverlayRender = false;
  state.inlineEditingIndex = null;
  setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
  renderJob(state.job);
});

document.querySelectorAll('.view-switcher button').forEach((button) => {
  button.addEventListener('click', () => {
    state.variant = button.dataset.variant;
    markActiveVariant();
    renderCurrentPage();
  });
});

function markActiveVariant() {
  document.querySelectorAll('.view-switcher button').forEach((button) => {
    button.classList.toggle('active', button.dataset.variant === state.variant);
  });
}

function escapeHtml(value) {
  return String(value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

(async function prepareInitialView() {
  showSetupView();
  try {
    const result = await requestJson('/api/jobs');
    const lastJob = result.jobs?.[0];
    if (lastJob) {
      state.lastJob = lastJob;
      continueLastBtn.classList.remove('hidden');
      continueLastBtn.textContent = `Continuar último trabajo: ${lastJob.title || 'Proyecto'}`;
    }
  } catch (_) {}
})();
