const state = {
  job: null,
  pageIndex: 0,
  selectedRegion: null,
  variant: 'current',
  tool: 'select',
  regions: [],
  brushStrokes: [],
  backgroundRevision: 'base',
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
  spacePan: false,
  panning: null,
  previewCache: new Map(),
  previewInFlightKeys: new Set(),
  lastPointerImagePoint: null,
  undoStack: [],
  redoStack: [],
  historyLimit: 80,
  lastHistoryPush: null,
  applyingHistory: false,
  editRevision: 0,
  pendingSaveOptions: null,
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
const inpaintModel = $('inpaintModel');
const detectionEngine = $('detectionEngine');
const transcriptionEngine = $('transcriptionEngine');
const pageMaxRetries = $('pageMaxRetries');
const retryBackoffSeconds = $('retryBackoffSeconds');
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
const pauseJobBtn = $('pauseJobBtn');
const resumeJobBtn = $('resumeJobBtn');
const cancelJobBtn = $('cancelJobBtn');
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
const mobilePagesBtn = $('mobilePagesBtn');
const mobileEditorBtn = $('mobileEditorBtn');
const closePagesBtn = $('closePagesBtn');
const closeEditorBtn = $('closeEditorBtn');
const panelBackdrop = $('panelBackdrop');
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
const rotationAngle = $('rotationAngle');
const rotationAngleNumber = $('rotationAngleNumber');
const rotationAngleValue = $('rotationAngleValue');
const resetRotationBtn = $('resetRotationBtn');
const regionStyleSelect = $('regionStyleSelect');
const lineSpacing = $('lineSpacing');
const lineSpacingNumber = $('lineSpacingNumber');
const lineSpacingValue = $('lineSpacingValue');
const textOffsetX = $('textOffsetX');
const textOffsetY = $('textOffsetY');
const resetTextOffsetBtn = $('resetTextOffsetBtn');
const resetTypographyBtn = $('resetTypographyBtn');
const centerRegionHorizontalBtn = $('centerRegionHorizontalBtn');
const centerRegionVerticalBtn = $('centerRegionVerticalBtn');
const fitRegionToTextBtn = $('fitRegionToTextBtn');
const alignLeftBtn = $('alignLeftBtn');
const alignCenterBtn = $('alignCenterBtn');
const alignRightBtn = $('alignRightBtn');
const alignTopBtn = $('alignTopBtn');
const alignMiddleBtn = $('alignMiddleBtn');
const alignBottomBtn = $('alignBottomBtn');
const saveBtn = $('saveBtn');
const resetBtn = $('resetBtn');
const prevBtn = $('prevBtn');
const nextBtn = $('nextBtn');
const toast = $('toast');
const selectTool = $('selectTool');
const newRegionTool = $('newRegionTool');
const brushTool = $('brushTool');
const panTool = $('panTool');
const dockSelectTool = $('dockSelectTool');
const dockRegionTool = $('dockRegionTool');
const dockBrushTool = $('dockBrushTool');
const dockPanTool = $('dockPanTool');
const dockFitBtn = $('dockFitBtn');
const brushMode = $('brushMode');
const brushSize = $('brushSize');
const brushInpaintModel = $('brushInpaintModel');
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
const translateOriginalBtn = $('translateOriginalBtn');
const copyRegionBtn = $('copyRegionBtn');
const deleteRegionBtn = $('deleteRegionBtn');
const regionList = $('regionList');
const canvasHint = $('canvasHint');
const canvasReadout = $('canvasReadout');
const appJobTitle = $('appJobTitle');
const appPageName = $('appPageName');
const documentDirtyIndicator = $('documentDirtyIndicator');
const undoHistoryBtn = $('undoHistoryBtn');
const redoHistoryBtn = $('redoHistoryBtn');
const quickSaveBtn = $('quickSaveBtn');
const contextToolIcon = $('contextToolIcon');
const contextToolName = $('contextToolName');
const contextToolSummary = $('contextToolSummary');
const statusTool = $('statusTool');
const statusDocumentSize = $('statusDocumentSize');
const statusSelection = $('statusSelection');
const statusZoom = $('statusZoom');
const shortcutHelpBtn = $('shortcutHelpBtn');
const shortcutModal = $('shortcutModal');
const shortcutModalBackdrop = $('shortcutModalBackdrop');
const closeShortcutModalBtn = $('closeShortcutModalBtn');

const TOOL_PRESENTATION = {
  select: {
    icon: '↖',
    name: 'Seleccionar y transformar',
    shortcut: 'V',
    summary: 'Selecciona una región para moverla, redimensionarla o editar su texto.',
  },
  region: {
    icon: '▭',
    name: 'Crear región de texto',
    shortcut: 'R',
    summary: 'Arrastra sobre la página para crear una caja de texto editable.',
  },
  brush: {
    icon: '●',
    name: 'Pincel de corrección',
    shortcut: 'B',
    summary: 'Pinta una máscara para limpiar texto, reconstruir fondo o restaurar el original.',
  },
  pan: {
    icon: '✋',
    name: 'Desplazar lienzo',
    shortcut: 'H',
    summary: 'Arrastra la página. Mantén Espacio para usar esta herramienta temporalmente.',
  },
};

function updateWorkspaceChrome() {
  const page = currentPage();
  const tool = TOOL_PRESENTATION[activeTool()] || TOOL_PRESENTATION.select;
  const selected = selectedRegion();
  const pageTotal = state.job?.pages?.length || 0;
  const ready = page?.status === 'ready';

  if (appJobTitle) appJobTitle.textContent = state.job?.title || 'Proyecto';
  if (appPageName) {
    appPageName.textContent = page
      ? `Página ${page.index + 1}${pageTotal ? ` de ${pageTotal}` : ''} · ${page.source_filename || 'sin nombre'}`
      : 'Sin página seleccionada';
  }
  if (documentDirtyIndicator) {
    const kind = state.autosaveError ? 'error' : state.autosaveInFlight ? 'saving pending' : state.dirty ? 'pending' : '';
    documentDirtyIndicator.className = `document-dirty ${kind}`.trim();
    const label = state.autosaveError ? 'Error al guardar' : state.autosaveInFlight ? 'Guardando documento' : state.dirty ? 'Cambios pendientes' : 'Documento guardado';
    documentDirtyIndicator.title = label;
    documentDirtyIndicator.setAttribute('aria-label', label);
  }
  if (undoHistoryBtn) undoHistoryBtn.disabled = state.undoStack.length === 0;
  if (redoHistoryBtn) redoHistoryBtn.disabled = state.redoStack.length === 0;
  if (quickSaveBtn) quickSaveBtn.disabled = !ready || state.autosaveInFlight;

  if (contextToolIcon) contextToolIcon.textContent = tool.icon;
  if (contextToolName) contextToolName.textContent = tool.name;
  if (contextToolSummary) contextToolSummary.textContent = tool.summary;
  if (statusTool) statusTool.textContent = `${tool.name} (${tool.shortcut})`;
  if (statusDocumentSize) statusDocumentSize.textContent = ready ? `${Math.round(state.naturalWidth)} × ${Math.round(state.naturalHeight)} px` : '— × — px';
  if (statusSelection) {
    if (!ready) statusSelection.textContent = 'Sin documento';
    else if (!selected) statusSelection.textContent = 'Sin selección';
    else {
      const [x, y, w, h] = selected.bbox || [0, 0, 0, 0];
      statusSelection.textContent = `Región ${state.selectedRegion + 1} · ${Math.round(w)} × ${Math.round(h)} px · X ${Math.round(x)} Y ${Math.round(y)}`;
    }
  }
  if (statusZoom) statusZoom.textContent = `${Math.round(state.zoom * 100)}%`;
}

function openShortcutModal() {
  if (!shortcutModal) return;
  shortcutModal.classList.remove('hidden');
  closeShortcutModalBtn?.focus();
}

function closeShortcutModal() {
  if (!shortcutModal) return;
  shortcutModal.classList.add('hidden');
  shortcutHelpBtn?.focus();
}


function showToast(message) {
  toast.textContent = message;
  toast.classList.remove('hidden');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add('hidden'), 3800);
}

function setAutosaveStatus(kind, message) {
  if (autosaveDot && autosaveStatus) {
    autosaveDot.className = `autosave-dot ${kind || 'saved'}`;
    autosaveStatus.textContent = message || 'Cambios guardados automáticamente.';
  }
  state.autosaveError = kind === 'error' ? String(message || 'Error al guardar') : '';
  updateWorkspaceChrome();
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

function activeTool() {
  return state.spacePan ? 'pan' : state.tool;
}

function isPanMode() {
  return activeTool() === 'pan';
}

function setCanvasPanning(active) {
  canvasCard?.classList.toggle('panning', Boolean(active));
  workView?.classList.toggle('space-pan', Boolean(state.spacePan));
}

function nudgeBrushSize(delta) {
  if (!brushSize) return;
  const next = Math.max(Number(brushSize.min || 4), Math.min(Number(brushSize.max || 90), Number(brushSize.value || 22) + delta));
  brushSize.value = String(next);
  updateBrushSizeLabel();
  requestOverlayRender();
}

function clampFontSize(value, fallback = null) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.max(6, Math.min(160, Math.round(parsed)));
}

function clampRotationAngle(value, fallback = 0) {
  let parsed = Number(value);
  if (!Number.isFinite(parsed)) parsed = Number(fallback) || 0;
  while (parsed <= -90) parsed += 180;
  while (parsed > 90) parsed -= 180;
  parsed = Math.max(-89, Math.min(89, parsed));
  return Math.abs(parsed) < 0.65 ? 0 : Math.round(parsed * 10) / 10;
}

function normalizeTextAlign(value) {
  const normalized = String(value || 'center').toLowerCase();
  return ['left', 'center', 'right'].includes(normalized) ? normalized : 'center';
}

function normalizeVerticalAlign(value) {
  const normalized = String(value || 'middle').toLowerCase();
  return ['top', 'middle', 'bottom'].includes(normalized) ? normalized : 'middle';
}

function clampLineSpacing(value, fallback = 1) {
  const parsed = Number(value);
  const safe = Number.isFinite(parsed) ? parsed : Number(fallback) || 1;
  return Math.round(Math.max(0.55, Math.min(2, safe)) * 100) / 100;
}

function clampTextOffset(value, fallback = 0) {
  const parsed = Number(value);
  const safe = Number.isFinite(parsed) ? parsed : Number(fallback) || 0;
  return Math.round(Math.max(-1000, Math.min(1000, safe)) * 10) / 10;
}

function syncLineSpacingControls(value) {
  const factor = clampLineSpacing(value, 1);
  if (lineSpacing) lineSpacing.value = String(factor);
  if (lineSpacingNumber) lineSpacingNumber.value = String(factor);
  if (lineSpacingValue) lineSpacingValue.textContent = `${Math.round(factor * 100)}%`;
  return factor;
}

function syncAlignmentControls(region) {
  const horizontal = normalizeTextAlign(region?.text_align);
  const vertical = normalizeVerticalAlign(region?.vertical_align);
  document.querySelectorAll('[data-text-align]').forEach((button) => button.classList.toggle('active', button.dataset.textAlign === horizontal));
  document.querySelectorAll('[data-vertical-align]').forEach((button) => button.classList.toggle('active', button.dataset.verticalAlign === vertical));
}

function syncRotationControls(value) {
  const angle = clampRotationAngle(value, 0);
  if (rotationAngle) rotationAngle.value = String(Math.max(Number(rotationAngle.min || -45), Math.min(Number(rotationAngle.max || 45), angle)));
  if (rotationAngleNumber) rotationAngleNumber.value = String(angle);
  if (rotationAngleValue) rotationAngleValue.textContent = `${angle > 0 ? '+' : ''}${angle}°`;
  return angle;
}

const MANGA_FONT_FAMILY = '"New Wild Words", "Comic Sans MS", "Trebuchet MS", Arial, sans-serif';
const TEXT_RENDERER_MAX_FONT_SIZE = 96;
const TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE = 7;
const TEXT_RENDERER_INNER_MARGIN_RATIO = 0.09;
const TEXT_RENDERER_LINE_SPACING = 0.22;
let textMeasureCanvas = null;
const inlinePreviewTimers = new Map();
const inlinePreviewControllers = new Map();
let inlinePreviewSerial = 0;
const RASTER_PREVIEW_CACHE_LIMIT = 72;

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
  const strokePad = Math.max(2, Math.ceil(fontSize * 0.14));
  const safeWidth = Math.max(1, width - strokePad * 2);
  const safeHeight = Math.max(1, height - strokePad * 2);
  const lines = splitPreviewLines(text, fontSize, safeWidth);
  const maxLineWidth = Math.max(...lines.map((line) => measureMangaText(line, fontSize)), 0);
  const paragraphHeight = lines.length * fontSize + Math.max(0, lines.length - 1) * fontSize * TEXT_RENDERER_LINE_SPACING;
  return maxLineWidth <= safeWidth && paragraphHeight <= safeHeight;
}

function cloneUiLayout(layout) {
  if (!layout || typeof layout !== 'object') return null;
  try { return JSON.parse(JSON.stringify(layout)); } catch (_) { return null; }
}

function boxValues(raw, fallback = [0, 0, 1, 1]) {
  const values = Array.isArray(raw) ? raw.slice(0, 4).map((value) => Number(value)) : [];
  while (values.length < 4) values.push(fallback[values.length] ?? (values.length < 2 ? 0 : 1));
  return values.map((value, index) => Math.round(Number.isFinite(value) ? value : fallback[index] ?? (index < 2 ? 0 : 1)));
}

function layoutBlocksForRegion(region) {
  const layout = region?.ui_layout;
  const blocks = Array.isArray(layout?.blocks) ? layout.blocks : [];
  if (!blocks.length) {
    const area = textRenderArea(region);
    return [{
      x: area.x,
      y: area.y,
      width: area.width,
      height: area.height,
      text: region?.translated_text || region?.original_text || '',
      fontSize: null,
      lines: null,
      writingMode: region?.ui_layout?.writing_mode || 'horizontal',
      source: 'fallback',
    }];
  }
  const [, , regionW, regionH] = boxValues(region?.bbox, [0, 0, 1, 1]);
  const [, , layoutW, layoutH] = boxValues(layout.bbox, [0, 0, regionW || 1, regionH || 1]);
  const scaleX = Math.max(1, regionW) / Math.max(1, layoutW);
  const scaleY = Math.max(1, regionH) / Math.max(1, layoutH);
  return blocks.map((block) => {
    const [x, y, w, h] = boxValues(block.area || block.slot, [0, 0, regionW || 1, regionH || 1]);
    return {
      x: Math.round(x * scaleX),
      y: Math.round(y * scaleY),
      width: Math.max(1, Math.round(w * scaleX)),
      height: Math.max(1, Math.round(h * scaleY)),
      text: String(block.text || ''),
      fontSize: clampFontSize(block.font_size, null),
      lines: Array.isArray(block.lines) ? block.lines.map((line) => String(line)) : null,
      writingMode: layout?.writing_mode || 'horizontal',
      source: 'ui_layout',
    };
  });
}

function sentenceUnits(text) {
  const normalized = String(text || ' ').replace(/\s+/g, ' ').trim() || ' ';
  const matches = normalized.match(/.+?(?:[.!?。！？…]+|$)(?:\s+|$)/gs) || [normalized];
  return matches.map((unit) => unit.replace(/\s+/g, ' ').trim()).filter(Boolean);
}

function splitTextForLayout(text, slotCount) {
  const count = Math.max(1, Number(slotCount) || 1);
  const normalized = normalizeRendererText(text);
  if (count === 1) return [normalized];
  const units = sentenceUnits(normalized);
  if (units.length >= count) {
    const chunks = [];
    const remaining = [...units];
    for (let slotIdx = 0; slotIdx < count; slotIdx += 1) {
      const remainingSlots = count - slotIdx;
      if (remainingSlots === 1) {
        chunks.push(remaining.join(' ').trim() || ' ');
        break;
      }
      const remainingChars = remaining.reduce((sum, unit) => sum + unit.length, 0);
      const target = Math.max(1, remainingChars / remainingSlots);
      const current = [];
      let currentLen = 0;
      while (remaining.length && remaining.length > remainingSlots - 1) {
        const next = remaining[0];
        if (current.length && currentLen + next.length > target * 1.18) break;
        current.push(remaining.shift());
        currentLen += next.length;
        if (currentLen >= target * 0.82) break;
      }
      chunks.push(current.join(' ').trim() || ' ');
    }
    return chunks.slice(0, count);
  }
  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length < count * 2) {
    const chunks = [];
    let start = 0;
    for (let slotIdx = 0; slotIdx < count; slotIdx += 1) {
      const end = slotIdx === count - 1 ? normalized.length : Math.round(normalized.length * (slotIdx + 1) / count);
      chunks.push(normalized.slice(start, end).trim() || ' ');
      start = end;
    }
    return chunks;
  }
  const chunks = [];
  let start = 0;
  for (let slotIdx = 0; slotIdx < count; slotIdx += 1) {
    const remainingSlots = count - slotIdx;
    if (remainingSlots === 1) {
      chunks.push(words.slice(start).join(' ').trim() || ' ');
      break;
    }
    const remainingWords = words.length - start;
    const take = Math.max(1, Math.round(remainingWords / remainingSlots));
    chunks.push(words.slice(start, start + take).join(' ').trim() || ' ');
    start += take;
  }
  return chunks;
}

function rendererLikeAutoFontSizeForArea(region, text, area) {
  const prepared = preparedRendererText(region, text);
  const style = regionStyle(region);
  let startFactor = 0.70;
  if (style.startsWith('onomatopeya')) startFactor = style === 'onomatopeya_subtitle' ? 0.82 : 0.92;
  else if (style === 'narracion') startFactor = 0.58;
  const start = Math.min(TEXT_RENDERER_MAX_FONT_SIZE, Math.max(TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE, Math.floor(area.height * startFactor)));
  for (let size = start; size >= TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE; size -= 1) {
    if (fitsPreviewText(prepared, size, area.width, area.height)) return size;
  }
  return TEXT_RENDERER_ABSOLUTE_MIN_FONT_SIZE;
}

function sameRendererText(a, b) {
  return normalizeRendererText(a) === normalizeRendererText(b);
}

function rendererLikeAutoFontSize(region) {
  const text = preparedRendererText(region, region?.translated_text || region?.original_text || 'Texto');
  const [firstBlock] = layoutBlocksForRegion(region);
  if (firstBlock?.fontSize && sameRendererText(text, firstBlock.text)) return firstBlock.fontSize;
  return rendererLikeAutoFontSizeForArea(region, text, firstBlock || textRenderArea(region));
}

function previewFontSizeForBlock(region, block, text) {
  if (!usesAutoFontSize(region)) return effectiveManualFontSize(region);
  const prepared = preparedRendererText(region, text || 'Texto');
  if (block?.fontSize && sameRendererText(prepared, block.text)) return block.fontSize;
  return rendererLikeAutoFontSizeForArea(region, prepared, block || textRenderArea(region));
}

function verticalCharacterPreviewText(text) {
  const normalized = normalizeRendererText(text).replace(/\s+/g, ' ').trim() || ' ';
  return Array.from(normalized).join('\n');
}

function usesVerticalCharacterPreview(region, block, text, fontSize) {
  if (Math.abs(clampRotationAngle(region?.rotation_angle || 0)) < 0.65) return false;
  const prepared = preparedRendererText(region, text || 'Texto');
  if (block?.writingMode === 'vertical_chars' && sameRendererText(prepared, block.text)) return true;
  return splitPreviewLines(prepared, fontSize, Math.max(1, block?.width || 1)).length === 1;
}

function previewLinesForBlock(region, block, text, fontSize) {
  const prepared = preparedRendererText(region, text || 'Texto');
  if (usesVerticalCharacterPreview(region, block, prepared, fontSize)) return verticalCharacterPreviewText(prepared);
  if (block?.lines?.length && sameRendererText(prepared, block.text)) return block.lines.join('\n');
  return splitPreviewLines(prepared, fontSize, Math.max(1, block?.width || 1)).join('\n');
}

function applyRendererTextLayout(element, region, block = null, blockText = null) {
  if (!element || !region) return;
  const { scaleX, scaleY } = getScale();
  const area = block || textRenderArea(region);
  const text = blockText ?? region?.translated_text ?? region?.original_text ?? 'Texto';
  const size = previewFontSizeForBlock(region, area, text);
  const verticalCharacters = usesVerticalCharacterPreview(region, area, text, size);
  element.style.left = `${area.x * scaleX}px`;
  element.style.top = `${area.y * scaleY}px`;
  element.style.width = `${area.width * scaleX}px`;
  element.style.height = `${area.height * scaleY}px`;
  element.style.transformOrigin = `${(region.bbox[2] / 2 - area.x) * scaleX}px ${(region.bbox[3] / 2 - area.y) * scaleY}px`;
  element.style.transform = verticalCharacters ? 'rotate(0deg)' : '';
  element.style.fontSize = `${Math.max(4, Math.min(240, Math.round(size * ((scaleX + scaleY) / 2))))}px`;
  element.style.lineHeight = regionStyle(region).startsWith('onomatopeya') ? '1' : '1.05';
}


function applyInlineEditorLayout(element, region, block = null, blockText = null) {
  if (!element || !region) return;
  const { scaleX, scaleY } = getScale();
  const area = block || textRenderArea(region);
  const text = blockText ?? region?.translated_text ?? region?.original_text ?? 'Texto';
  const baseSize = previewFontSizeForBlock(region, area, text);
  const fontPx = Math.max(4, Math.min(240, Math.round(baseSize * ((scaleX + scaleY) / 2))));
  const lineFactor = regionStyle(region).startsWith('onomatopeya') ? 1.0 : 1.05;
  const lineHeightPx = Math.max(5, Math.round(fontPx * lineFactor));
  const widthPx = Math.max(1, area.width * scaleX);
  const heightPx = Math.max(1, area.height * scaleY);
  const prepared = preparedRendererText(region, text || 'Texto');
  const lineCount = Math.max(1, splitPreviewLines(prepared, Math.max(1, baseSize), Math.max(1, area.width)).length);
  const verticalCharacters = Math.abs(clampRotationAngle(region?.rotation_angle || 0)) >= 0.65 && lineCount === 1;
  const padY = Math.max(0, Math.floor((heightPx - lineCount * lineHeightPx) / 2));
  const padX = Math.max(1, Math.round(Math.max(scaleX, scaleY) * 1.5));

  element.style.left = `${area.x * scaleX}px`;
  element.style.top = `${area.y * scaleY}px`;
  element.style.width = `${widthPx}px`;
  element.style.height = `${heightPx}px`;
  element.style.transformOrigin = `${(region.bbox[2] / 2 - area.x) * scaleX}px ${(region.bbox[3] / 2 - area.y) * scaleY}px`;
  element.style.transform = verticalCharacters ? 'rotate(0deg)' : '';
  element.style.writingMode = verticalCharacters ? 'vertical-rl' : '';
  element.style.textOrientation = verticalCharacters ? 'upright' : '';
  element.style.fontSize = `${fontPx}px`;
  element.style.lineHeight = `${lineHeightPx}px`;
  element.style.padding = `${padY}px ${padX}px`;
}

function refreshInlineEditorLayout(element) {
  const idx = Number(element?.dataset?.index);
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return;
  const block = layoutBlocksForRegion(region)[0] || textRenderArea(region);
  applyInlineEditorLayout(element, region, block, textFromEditable(element) || region.translated_text || region.original_text || 'Texto');
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
  state.editRevision += 1;
  state.dirty = true;
  updateHeavyActions();
  updateWorkspaceChrome();
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
  data.append('inpaint_model', inpaintModel.value || 'auto');
  data.append('detection_engine', detectionEngine.value || 'auto');
  data.append('transcription_engine', transcriptionEngine.value || 'auto');
  data.append('page_max_retries', pageMaxRetries.value || '0');
  data.append('retry_backoff_seconds', retryBackoffSeconds.value || '0');
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
  if (!['ready', 'failed', 'cancelled'].includes(state.lastJob.status)) startPolling(state.lastJob.job_id);
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
  state.backgroundRevision = 'base';
  state.dragging = null;
  state.drawingRegion = null;
  state.drawingStroke = null;
  state.dirty = false;
  state.editRevision = 0;
  state.pendingSaveOptions = null;
  state.pageStamp = '';
  state.deletedStack = [];
  clearHistory();
  state.zoom = 1;
  state.fitZoom = 1;
  clearTimeout(state.autosaveTimer);
  cleanupInlinePreviewResources();
  clearRasterPreviewCache();
  state.deferredOverlayRender = false;
  state.inlineEditingIndex = null;
  setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
  updateHeavyActions();
  markActiveVariant();
}

function pageUpdatedAtValue(page) {
  const value = Number(page?.updated_at || 0);
  return Number.isFinite(value) ? value : 0;
}

function mergePolledJob(currentJob, incomingJob) {
  if (!incomingJob) return currentJob;
  if (!currentJob || currentJob.job_id !== incomingJob.job_id) return incomingJob;

  const currentByIndex = new Map(
    (currentJob.pages || []).map((page) => [Number(page?.index), page]),
  );
  const seen = new Set();
  const mergedPages = (incomingJob.pages || []).map((incomingPage) => {
    const index = Number(incomingPage?.index);
    seen.add(index);
    const currentPage = currentByIndex.get(index);
    if (!currentPage) return incomingPage;

    // Una respuesta de polling puede haber sido generada antes de que terminara un
    // guardado manual y llegar después. Nunca permitimos que esa copia atrasada haga
    // retroceder background_revision, regiones o la imagen corregida de la página.
    if (pageUpdatedAtValue(currentPage) > pageUpdatedAtValue(incomingPage)) {
      return currentPage;
    }
    return incomingPage;
  });

  // Defensa para respuestas parciales/transitorias: no perder páginas que el cliente
  // ya conoce si una instantánea de polling todavía no las incluyó.
  for (const currentPage of currentJob.pages || []) {
    const index = Number(currentPage?.index);
    if (!seen.has(index)) mergedPages.push(currentPage);
  }

  return {
    ...currentJob,
    ...incomingJob,
    pages: mergedPages,
  };
}

function startPolling(jobId) {
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(async () => {
    try {
      const incomingJob = await requestJson(`/api/jobs/${jobId}`);
      if (state.job?.job_id && state.job.job_id !== jobId) return;
      const job = mergePolledJob(state.job, incomingJob);
      state.job = job;
      renderJob(job);
      if (['ready', 'failed', 'cancelled'].includes(job.status)) clearInterval(state.polling);
    } catch (error) {
      console.warn(error);
    }
  }, 1800);
}

function inpaintModelLabel(value) {
  return {
    auto: 'Automático',
    'opencv-tela': 'OpenCV Telea',
    lama_mpe: 'LaMa MPE',
    lama_large_512px: 'LaMa Large 512',
    aot: 'AOT',
    'B/N': 'B/N',
  }[value] || value || 'Automático';
}

function renderJob(job) {
  if (!job) return;
  jobTitle.textContent = job.title || 'Proyecto';
  jobMessage.textContent = job.message || '';
  const opts = job.options || {};
  jobOptionsSummary.textContent = `${opts.source_language || 'Entrada'} → ${opts.target_language || 'Salida'} · ${opts.translator === 'google' ? 'Google' : 'LLM'} · Inpainting: ${inpaintModelLabel(opts.inpaint_model)}`;
  jobBadge.textContent = readableStatus(job.status);
  jobBadge.className = `badge ${job.status === 'ready' ? 'ready' : job.status === 'failed' ? 'failed' : ''}`;
  progressBar.style.width = `${job.progress || 0}%`;
  progressText.textContent = `${job.processed_count || 0} listas · ${job.failed_count || 0} fallidas · ${job.total_count || 0} total`;
  const terminal = ['ready', 'failed', 'cancelled'].includes(job.status);
  pauseJobBtn.disabled = terminal || ['paused', 'pausing', 'resuming', 'cancelling'].includes(job.status);
  resumeJobBtn.disabled = !['paused', 'pausing', 'resuming'].includes(job.status);
  cancelJobBtn.disabled = terminal || job.status === 'cancelling';
  const exportablePages = (job.pages || []).filter((page) => page.status === 'ready' || page.has_corrected).length;
  exportBtn.disabled = exportablePages === 0;
  exportBtn.textContent = exportablePages > 0 ? `Exportar ZIP (${exportablePages})` : 'Exportar ZIP';
  renderPageList(job.pages || []);
  renderCurrentPage();
  updateWorkspaceChrome();
}

function readableStatus(status) {
  return {
    queued: 'En cola',
    pending: 'Pendiente',
    processing: 'Procesando',
    pausing: 'Pausando',
    paused: 'Pausado',
    resuming: 'Reanudando',
    cancelling: 'Cancelando',
    cancelled: 'Cancelado',
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
      if (inlineEditor) {
        syncInlineTextToRegion(inlineEditor, { autosave: false, preview: false });
        inlineEditor.blur();
      }
      if (state.dirty) saveCurrentPage({ silent: true, force: true, reason: 'cambio de página' }).catch(console.warn);
      state.pageIndex = page.index;
      state.selectedRegion = null;
      state.dirty = false;
      state.drawingRegion = null;
      state.drawingStroke = null;
      state.deletedStack = [];
      clearTimeout(state.autosaveTimer);
      cleanupInlinePreviewResources();
      clearRasterPreviewCache();
      state.deferredOverlayRender = false;
      state.inlineEditingIndex = null;
      setAutosaveStatus('saved', 'Cambios guardados automáticamente.');
      renderJob(state.job);
    });
    pageList.appendChild(button);
  }
}

function currentPage() {
  return state.job?.pages?.[state.pageIndex] || null;
}

function selectedRegion() {
  return state.selectedRegion == null ? null : state.regions[state.selectedRegion];
}

function useCleanBaseForUiText() {
  // Round 7: no cambiamos toda la hoja a la imagen limpia al seleccionar una
  // región. Mantener la página actual como base evita que las demás regiones se
  // re-rastericen con previews temporales y elimina el parpadeo por clics.
  return false;
}

function activePageImageVariant() {
  return useCleanBaseForUiText() ? 'clean' : state.variant;
}

function isTextInteractiveTarget(target) {
  return Boolean(target?.closest?.('.region-text-editor, .region-text-preview, .region-text-hit, .region-label, .resize-handle'));
}

function isNearBoxEdge(event, box, tolerance = 9) {
  const rect = box.getBoundingClientRect();
  const left = event.clientX - rect.left;
  const top = event.clientY - rect.top;
  const right = rect.width - left;
  const bottom = rect.height - top;
  return Math.min(left, top, right, bottom) <= tolerance;
}

function setPageImageSource(page = currentPage()) {
  if (!page || page.status !== 'ready' || !pageImage) return;
  const imageVariant = activePageImageVariant();
  const url = page.images[imageVariant] || page.images[state.variant] || page.images.current;
  const bustedUrl = `${url}?t=${encodeURIComponent(`${page.updated_at || Date.now()}:${imageVariant}`)}`;
  const absoluteUrl = new URL(bustedUrl, location.href).href;
  pageImage.onload = () => {
    state.naturalWidth = pageImage.naturalWidth || 1;
    state.naturalHeight = pageImage.naturalHeight || 1;
    setFitZoomIfNeeded();
    applyZoom();
    renderOverlay();
  };
  if (pageImage.src !== absoluteUrl) {
    pageImage.src = bustedUrl;
    return;
  }
  if (pageImage.complete) {
    state.naturalWidth = pageImage.naturalWidth || 1;
    state.naturalHeight = pageImage.naturalHeight || 1;
    setFitZoomIfNeeded();
    applyZoom();
    renderOverlay();
  }
}

function deselectRegion(options = {}) {
  if (activeInlineTextEditor()) activeInlineTextEditor().blur();
  state.selectedRegion = null;
  showNoRegion();
  if (options.render !== false) setPageImageSource();
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
    state.backgroundRevision = 'base';
    showNoRegion();
    renderRegionList();
    return;
  }

  waitState.classList.add('hidden');
  imageStage.classList.remove('hidden');
  const pageStamp = `${state.job.job_id}:${page.index}:${page.updated_at || ''}`;
  if (!state.dirty && state.pageStamp !== pageStamp) {
    clearRasterPreviewCache();
    state.regions = cloneRegions(page.regions || []);
    state.brushStrokes = cloneBrushStrokes(page.brush_strokes || []);
    state.backgroundRevision = page.background_revision || 'base';
    state.pageStamp = pageStamp;
    state.editRevision = 0;
    state.pendingSaveOptions = null;
    clearHistory();
    if (state.selectedRegion != null && !isSelectableRegion(state.regions[state.selectedRegion])) state.selectedRegion = null;
  }
  setPageImageSource(page);
  if (state.selectedRegion == null && state.tool === 'select') {
    const first = state.regions.findIndex(isSelectableRegion);
    if (first >= 0) selectRegion(first, false);
  }
  if (!state.regions.some(isSelectableRegion)) showNoRegion();
  regionCount.textContent = `${state.regions.filter(isSelectableRegion).length} regiones`;
  renderRegionList();
  updateHeavyActions();
}


function countActiveRegions(regions) {
  return regions.filter((region) => !region.deleted).length;
}

function shortRegionText(region) {
  const text = String(region?.translated_text || region?.original_text || '').replace(/\s+/g, ' ').trim();
  return text || (region?.manual ? 'Región manual sin texto' : 'Sin texto');
}

function renderRegionList() {
  if (!regionList) return;
  const selectable = state.regions
    .map((region, idx) => ({ region, idx }))
    .filter(({ region }) => isSelectableRegion(region));
  regionList.innerHTML = '';
  regionList.classList.toggle('empty', !selectable.length);
  if (!selectable.length) {
    regionList.textContent = 'No hay regiones en esta página.';
    return;
  }
  for (const { region, idx } of selectable) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `layer-item ${idx === state.selectedRegion ? 'active' : ''} ${region.visible === false ? 'hidden-layer' : ''}`;
    const [x, y, w, h] = region.bbox || [0, 0, 0, 0];
    item.innerHTML = `
      <span class="layer-thumb">${idx + 1}</span>
      <span class="layer-main">
        <span class="layer-title">${escapeHtml(shortRegionText(region))}</span>
        <span class="layer-sub">${escapeHtml(region.manual ? 'Manual' : 'Detectada')} · ${Math.round(w)}×${Math.round(h)} · ${Math.round(x)},${Math.round(y)}</span>
      </span>
      <span class="layer-eye" title="Mostrar/ocultar texto">${region.visible === false ? '○' : '●'}</span>
    `;
    item.addEventListener('click', (event) => {
      const inlineEditor = activeInlineTextEditor();
      if (inlineEditor) {
        syncInlineTextToRegion(inlineEditor, { autosave: false, preview: false });
        inlineEditor.blur();
      }
      const eye = event.target.closest?.('.layer-eye');
      if (eye) {
        pushUndoSnapshot('visibilidad de región');
        region.visible = region.visible === false;
        region.modified = true;
        markDirty({ autosave: true, reason: 'visibilidad de región' });
        renderRegionList();
        renderOverlay({ force: true });
        return;
      }
      setTool('select');
      selectRegion(idx);
      scrollSelectedRegionIntoView(region);
    });
    regionList.appendChild(item);
  }
}

function scrollSelectedRegionIntoView(region = selectedRegion()) {
  if (!region || !canvasCard || !pageImage) return;
  const [x, y, w, h] = region.bbox;
  const targetX = (x + w / 2) * state.zoom - canvasCard.clientWidth / 2;
  const targetY = (y + h / 2) * state.zoom - canvasCard.clientHeight / 2;
  canvasCard.scrollTo({ left: Math.max(0, targetX), top: Math.max(0, targetY), behavior: 'smooth' });
}

function isSelectableRegion(region) {
  return Boolean(region && !region.deleted);
}

function unionBoxes(boxes) {
  const valid = (boxes || []).filter((box) => Array.isArray(box) && box.length >= 4 && Number(box[2]) > 0 && Number(box[3]) > 0);
  if (!valid.length) return null;
  const x1 = Math.min(...valid.map((box) => Number(box[0]) || 0));
  const y1 = Math.min(...valid.map((box) => Number(box[1]) || 0));
  const x2 = Math.max(...valid.map((box) => (Number(box[0]) || 0) + Math.max(1, Number(box[2]) || 1)));
  const y2 = Math.max(...valid.map((box) => (Number(box[1]) || 0) + Math.max(1, Number(box[3]) || 1)));
  return [Math.round(x1), Math.round(y1), Math.max(1, Math.round(x2 - x1)), Math.max(1, Math.round(y2 - y1))];
}

function clampLocalBox(rawBox, maxWidth, maxHeight) {
  let [x, y, w, h] = boxValues(rawBox, [0, 0, Math.max(1, maxWidth || 1), Math.max(1, maxHeight || 1)]);
  const width = Math.max(1, Math.round(maxWidth || 1));
  const height = Math.max(1, Math.round(maxHeight || 1));
  x = Math.max(0, Math.min(width - 1, x));
  y = Math.max(0, Math.min(height - 1, y));
  w = Math.max(1, Math.min(w, width - x));
  h = Math.max(1, Math.min(h, height - y));
  return [x, y, w, h];
}
function createFullBoxUiLayout(bbox, style = 'dialogo') {
  const [, , w, h] = normalizeBox(bbox || [0, 0, 1, 1]);
  return {
    version: 2,
    bbox: [0, 0, Math.max(1, w), Math.max(1, h)],
    style: style || 'dialogo',
    rotation_angle: 0,
    text_align: 'center',
    vertical_align: 'middle',
    line_spacing_factor: 1,
    text_offset_x: 0,
    text_offset_y: 0,
    block_count: 1,
    blocks: [{
      slot: [0, 0, Math.max(1, w), Math.max(1, h)],
      area: [0, 0, Math.max(1, w), Math.max(1, h)],
      text: '',
      lines: null,
      font_size: null,
      line_spacing: null,
      stroke_width: null,
    }],
    uses_clip_mask: false,
    ui_text_region: true,
  };
}


function adaptLayoutToTextRegion(layout, bbox) {
  const cloned = cloneUiLayout(layout);
  const blocks = Array.isArray(cloned?.blocks) ? cloned.blocks : [];
  if (!cloned || !blocks.length || cloned.ui_text_region === true) return { bbox, layout: cloned, source_bbox: bbox, adapted: false };

  const [, , boxW, boxH] = bbox;
  const [, , layoutW, layoutH] = boxValues(cloned.bbox, [0, 0, Math.max(1, boxW), Math.max(1, boxH)]);
  const scaleX = Math.max(1, boxW) / Math.max(1, layoutW);
  const scaleY = Math.max(1, boxH) / Math.max(1, layoutH);
  const scaledBlocks = blocks.map((block) => {
    const area = boxValues(block.area || block.slot, [0, 0, layoutW, layoutH]);
    const slot = boxValues(block.slot || block.area, area);
    const scaleBox = (raw) => [
      Math.round(raw[0] * scaleX),
      Math.round(raw[1] * scaleY),
      Math.max(1, Math.round(raw[2] * scaleX)),
      Math.max(1, Math.round(raw[3] * scaleY)),
    ];
    return { block, area: scaleBox(area), slot: scaleBox(slot) };
  });
  const textUnion = unionBoxes(scaledBlocks.map((item) => item.area));
  if (!textUnion) return { bbox, layout: cloned, source_bbox: bbox, adapted: false };

  const [unionX, unionY, unionW, unionH] = clampLocalBox(textUnion, boxW, boxH);
  const adaptedBlocks = scaledBlocks.map(({ block, area, slot }) => {
    const shiftedArea = clampLocalBox([area[0] - unionX, area[1] - unionY, area[2], area[3]], unionW, unionH);
    const shiftedSlot = clampLocalBox([slot[0] - unionX, slot[1] - unionY, slot[2], slot[3]], unionW, unionH);
    return {
      ...block,
      area: shiftedArea,
      slot: shiftedSlot,
    };
  });

  return {
    bbox: [bbox[0] + unionX, bbox[1] + unionY, unionW, unionH],
    source_bbox: [...bbox],
    layout: {
      ...cloned,
      bbox: [0, 0, unionW, unionH],
      blocks: adaptedBlocks,
      block_count: adaptedBlocks.length,
      ui_text_region: true,
      original_region_bbox: [...bbox],
    },
    adapted: true,
  };
}

function cloneRegions(regions) {
  return regions.map((region, idx) => {
    const rawBBox = normalizeBox(region.bbox || [0, 0, 1, 1]);
    const rawLayout = cloneUiLayout(region.ui_layout || region.layout_ui || region['Layout UI'] || null);
    const manual = Boolean(region.manual || region.created_manually);
    const style = region.style || 'dialogo';
    const layoutResult = manual
      ? { bbox: rawBBox, layout: rawLayout || createFullBoxUiLayout(rawBBox, style), source_bbox: rawBBox, adapted: false }
      : adaptLayoutToTextRegion(rawLayout, rawBBox);
    if (!layoutResult.layout && manual) layoutResult.layout = createFullBoxUiLayout(rawBBox, style);
    const bbox = normalizeBox(layoutResult.bbox || rawBBox);
    const sourceBox = normalizeBox(region.source_bbox || region.original_bbox || layoutResult.source_bbox || rawBBox);
    return {
      index: Number(region.index ?? idx),
      bbox,
      source_bbox: sourceBox,
      original_text: region.original_text || '',
      translated_text: region.translated_text ?? region.text ?? '',
      style,
      type: region.type || '',
      confidence: region.confidence || 0,
      restore_original: Boolean(region.restore_original),
      visible: region.visible !== false,
      modified: Boolean(region.modified || region.corrected),
      manual,
      deleted: Boolean(region.deleted),
      auto_font_size: region.auto_font_size !== false,
      font_size: region.auto_font_size === false ? clampFontSize(region.font_size, null) : null,
      rotation_angle: clampRotationAngle(region.rotation_angle ?? region.text_rotation_angle ?? layoutResult.layout?.requested_rotation_angle ?? layoutResult.layout?.rotation_angle ?? 0),
      rotation_confidence: Number(region.rotation_confidence || 0),
      text_align: normalizeTextAlign(region.text_align ?? layoutResult.layout?.text_align ?? 'center'),
      vertical_align: normalizeVerticalAlign(region.vertical_align ?? layoutResult.layout?.vertical_align ?? 'middle'),
      line_spacing_factor: clampLineSpacing(region.line_spacing_factor ?? region.line_spacing ?? layoutResult.layout?.line_spacing_factor ?? 1),
      text_offset_x: clampTextOffset(region.text_offset_x ?? layoutResult.layout?.text_offset_x ?? 0),
      text_offset_y: clampTextOffset(region.text_offset_y ?? layoutResult.layout?.text_offset_y ?? 0),
      ui_layout: layoutResult.layout,
      ui_text_region: Boolean(layoutResult.layout?.ui_text_region),
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

function brushStrokeFingerprint(stroke) {
  const normalized = cloneBrushStrokes([stroke])[0];
  return normalized ? JSON.stringify(normalized) : '';
}

function stripCommittedBrushPrefix(currentStrokes, committedStrokes) {
  const current = cloneBrushStrokes(currentStrokes || []);
  const committed = cloneBrushStrokes(committedStrokes || []);
  if (!committed.length || current.length < committed.length) {
    return { matched: false, remaining: current };
  }
  for (let idx = 0; idx < committed.length; idx += 1) {
    if (brushStrokeFingerprint(current[idx]) !== brushStrokeFingerprint(committed[idx])) {
      return { matched: false, remaining: current };
    }
  }
  return { matched: true, remaining: current.slice(committed.length) };
}

function rebaseHistoryAfterBackgroundCommit({ pageKey, parentRevision, nextRevision, committedStrokes }) {
  if (!pageKey || !nextRevision || nextRevision === parentRevision) return;
  const rebaseSnapshot = (snapshot) => {
    if (!snapshot || snapshot.pageKey !== pageKey) return;
    if ((snapshot.backgroundRevision || 'base') !== parentRevision) return;
    const stripped = stripCommittedBrushPrefix(snapshot.brushStrokes || [], committedStrokes || []);
    // Una instantánea anterior a la pincelada no contiene esos trazos y debe seguir
    // apuntando al padre: esa es precisamente la entrada que permite deshacer el inpaint.
    if (!stripped.matched) return;
    snapshot.backgroundRevision = nextRevision;
    snapshot.brushStrokes = stripped.remaining;
  };
  state.undoStack.forEach(rebaseSnapshot);
  state.redoStack.forEach(rebaseSnapshot);
}

function reconcileCommittedBackgroundRevision(payload, updatedPage, pageKey) {
  const parentRevision = payload?.background_revision || 'base';
  const nextRevision = updatedPage?.background_revision || parentRevision;
  if (nextRevision === parentRevision) return false;

  // Si el usuario siguió pintando mientras el inpaint estaba calculándose, la respuesta
  // representa un prefijo de los trazos locales. Hay que avanzar la revisión de fondo a
  // la recién creada y quitar SOLO ese prefijo ya horneado, conservando los trazos nuevos.
  // Si el prefijo ya no existe (por ejemplo, el usuario pulsó Deshacer durante el cálculo),
  // no rebasamos el estado local: el historial elegido por el usuario tiene prioridad.
  if ((state.backgroundRevision || 'base') !== parentRevision) return false;
  const stripped = stripCommittedBrushPrefix(state.brushStrokes || [], payload?.brush_strokes || []);
  if (!stripped.matched) return false;

  state.backgroundRevision = nextRevision;
  state.brushStrokes = stripped.remaining;
  rebaseHistoryAfterBackgroundCommit({
    pageKey,
    parentRevision,
    nextRevision,
    committedStrokes: payload?.brush_strokes || [],
  });
  return true;
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
  if (!pageImage || imageStage.classList.contains('hidden')) {
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
  updateCanvasReadout([x, y]);
  if (activeTool() !== 'brush') {
    if (state.brushCursor.visible) {
      state.brushCursor.visible = false;
      requestOverlayRender();
    }
    return;
  }
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
  updateCanvasReadout();
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

function toolLabel(tool = activeTool()) {
  return {
    select: 'Seleccionar',
    region: 'Nueva región',
    brush: `Pincel ${brushSize?.value || 22}px`,
    pan: 'Mano',
  }[tool] || tool;
}

function updateCanvasReadout(point = state.lastPointerImagePoint) {
  const page = currentPage();
  if (!page || page.status !== 'ready') {
    if (canvasReadout) canvasReadout.textContent = 'Sin página';
    updateWorkspaceChrome();
    return;
  }
  if (Array.isArray(point)) state.lastPointerImagePoint = point;
  const selected = state.selectedRegion == null ? 'sin región' : `R${state.selectedRegion + 1}`;
  const dirty = state.dirty ? ' · sin guardar' : '';
  const coords = Array.isArray(state.lastPointerImagePoint) ? ` · ${state.lastPointerImagePoint[0]}, ${state.lastPointerImagePoint[1]}` : '';
  if (canvasReadout) canvasReadout.textContent = `P${page.index + 1}/${state.job?.pages?.length || 1} · ${Math.round(state.zoom * 100)}% · ${toolLabel()} · ${selected}${coords}${dirty}`;
  updateWorkspaceChrome();
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
function focusInlineEditorForRegion(idx, options = {}) {
  let attempts = 0;
  const tryFocus = () => {
    const editor = overlayLayer?.querySelector?.(`.region-text-editor[data-index="${idx}"]`);
    if (!editor) {
      if (attempts < 8) {
        attempts += 1;
        setTimeout(tryFocus, 55);
      }
      return;
    }
    editor.focus({ preventScroll: true });
    if (options.selectAll) {
      if (typeof editor.select === 'function') {
        editor.select();
      } else {
        const range = document.createRange();
        range.selectNodeContents(editor);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
      }
    }
  };
  requestAnimationFrame(tryFocus);
}


function rememberRasterPreview(key, url) {
  const previous = state.previewCache.get(key);
  if (previous?.url === url) return;
  if (previous?.url) URL.revokeObjectURL(previous.url);
  state.previewCache.set(key, { url, touched: Date.now() });
  while (state.previewCache.size > RASTER_PREVIEW_CACHE_LIMIT) {
    let oldestKey = null;
    let oldestTime = Infinity;
    state.previewCache.forEach((entry, entryKey) => {
      if (entry.touched < oldestTime) {
        oldestTime = entry.touched;
        oldestKey = entryKey;
      }
    });
    if (!oldestKey) break;
    const evicted = state.previewCache.get(oldestKey);
    if (evicted?.url) URL.revokeObjectURL(evicted.url);
    state.previewCache.delete(oldestKey);
  }
}

function cachedRasterPreview(key) {
  const entry = state.previewCache.get(key);
  if (!entry) return null;
  entry.touched = Date.now();
  return entry.url;
}

async function fetchRegionMetrics(idx) {
  const page = currentPage();
  const region = state.regions[idx];
  if (!page || page.status !== 'ready' || !state.job?.job_id || !isSelectableRegion(region)) return null;
  const response = await fetch(`/api/jobs/${state.job.job_id}/pages/${page.index}/region-metrics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ region: regionToRenderPatch(region, idx) }),
  });
  if (!response.ok) {
    let payload = null;
    try { payload = await response.json(); } catch (_) {}
    throw new Error(payload?.detail || response.statusText || 'No se pudieron medir las métricas.');
  }
  return response.json();
}

function clearRasterPreviewCache() {
  // Mantener este reseteo tolerante a estados parciales evita que una recarga
  // durante la inicialización bloquee la apertura del editor.
  state.previewCache ??= new Map();
  state.previewInFlightKeys ??= new Set();
  state.previewCache.forEach((entry) => {
    if (entry?.url) URL.revokeObjectURL(entry.url);
  });
  state.previewCache.clear();
  state.previewInFlightKeys.clear();
}

function currentHistoryPageKey() {
  const page = currentPage();
  if (!state.job || !page) return '';
  return `${state.job.job_id}:${page.index}`;
}

function clearHistory() {
  state.undoStack = [];
  state.redoStack = [];
  state.lastHistoryPush = null;
  updateWorkspaceChrome();
}

function createHistorySnapshot(label = 'cambio') {
  return {
    label,
    pageKey: currentHistoryPageKey(),
    regions: cloneRegions(state.regions || []),
    brushStrokes: cloneBrushStrokes(state.brushStrokes || []),
    backgroundRevision: state.backgroundRevision || 'base',
    selectedRegion: state.selectedRegion,
    dirty: Boolean(state.dirty),
    timestamp: Date.now(),
  };
}

function snapshotsAreEquivalent(a, b) {
  if (!a || !b) return false;
  try {
    return JSON.stringify({ regions: a.regions, brushStrokes: a.brushStrokes, backgroundRevision: a.backgroundRevision, selectedRegion: a.selectedRegion }) ===
      JSON.stringify({ regions: b.regions, brushStrokes: b.brushStrokes, backgroundRevision: b.backgroundRevision, selectedRegion: b.selectedRegion });
  } catch (_) {
    return false;
  }
}

function pushUndoSnapshot(label = 'cambio', options = {}) {
  if (state.applyingHistory) return;
  const pageKey = currentHistoryPageKey();
  if (!pageKey) return;
  const now = Date.now();
  const coalesceKey = options.coalesceKey || null;
  const coalesceMs = Number(options.coalesceMs ?? 850);
  if (coalesceKey && state.lastHistoryPush?.key === coalesceKey && state.lastHistoryPush?.pageKey === pageKey && now - state.lastHistoryPush.time < coalesceMs) {
    return;
  }
  const snapshot = createHistorySnapshot(label);
  const previous = state.undoStack[state.undoStack.length - 1];
  if (snapshotsAreEquivalent(previous, snapshot)) return;
  state.undoStack.push(snapshot);
  while (state.undoStack.length > state.historyLimit) state.undoStack.shift();
  state.redoStack = [];
  state.lastHistoryPush = { key: coalesceKey || `single:${now}`, pageKey, time: now };
  updateCanvasReadout();
  updateWorkspaceChrome();
}

function restoreHistorySnapshot(snapshot, reason = 'historial') {
  if (!snapshot || snapshot.pageKey !== currentHistoryPageKey()) {
    showToast('El historial de esta página ya no está disponible.');
    return false;
  }
  state.applyingHistory = true;
  try {
    cleanupInlinePreviewResources();
    clearRasterPreviewCache();
    state.regions = cloneRegions(snapshot.regions || []);
    state.brushStrokes = cloneBrushStrokes(snapshot.brushStrokes || []);
    state.backgroundRevision = snapshot.backgroundRevision || 'base';
    state.selectedRegion = Number.isInteger(snapshot.selectedRegion) && isSelectableRegion(state.regions[snapshot.selectedRegion]) ? snapshot.selectedRegion : null;
    state.deletedStack = [];
    state.drawingRegion = null;
    state.drawingStroke = null;
    state.dragging = null;
    state.dirty = true;
    if (state.selectedRegion == null) showNoRegion();
    else {
      noRegion.classList.add('hidden');
      regionEditor.classList.remove('hidden');
      updateEditorFromRegion();
    }
    renderRegionList();
    regionCount.textContent = `${state.regions.filter(isSelectableRegion).length} regiones`;
    setPageImageSource();
    renderOverlay({ force: true });
    markDirty({ autosave: true, reason });
    return true;
  } finally {
    state.applyingHistory = false;
    state.lastHistoryPush = null;
    updateCanvasReadout();
  }
}

function undoChange() {
  if (!state.undoStack.length) {
    showToast('No hay cambios para deshacer.');
    return;
  }
  const snapshot = state.undoStack.pop();
  if (snapshot.pageKey !== currentHistoryPageKey()) {
    clearHistory();
    showToast('El historial pertenece a otra página.');
    return;
  }
  state.redoStack.push(createHistorySnapshot('rehacer'));
  restoreHistorySnapshot(snapshot, 'deshacer');
  updateWorkspaceChrome();
  showToast(`Deshecho: ${snapshot.label || 'cambio'}.`);
}

function redoChange() {
  if (!state.redoStack.length) {
    showToast('No hay cambios para rehacer.');
    return;
  }
  const snapshot = state.redoStack.pop();
  if (snapshot.pageKey !== currentHistoryPageKey()) {
    clearHistory();
    showToast('El historial pertenece a otra página.');
    return;
  }
  state.undoStack.push(createHistorySnapshot('deshacer rehacer'));
  restoreHistorySnapshot(snapshot, 'rehacer');
  updateWorkspaceChrome();
  showToast('Cambio rehecho.');
}

function cleanupInlinePreviewResources() {
  inlinePreviewTimers.forEach((timer) => clearTimeout(timer));
  inlinePreviewTimers.clear();
  inlinePreviewControllers.forEach((controller) => controller.abort());
  inlinePreviewControllers.clear();
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
  cleanupInlinePreviewResources();
  overlayLayer.innerHTML = '';
  const { scaleX, scaleY, width, height } = getScale();
  if (!Number.isFinite(width) || width <= 0) return;
  overlayLayer.style.width = `${width}px`;
  overlayLayer.style.height = `${height}px`;
  overlayLayer.className = `overlay-layer tool-${activeTool()}`;

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
    box.className = `region-box ${region.ui_text_region ? 'text-region' : ''} ${isSelected ? 'selected' : ''} ${region.restore_original ? 'restore' : ''} ${region.manual ? 'manual' : ''} ${region.visible === false ? 'hidden-region' : ''}`;
    box.style.left = `${x * scaleX}px`;
    box.style.top = `${y * scaleY}px`;
    box.style.width = `${w * scaleX}px`;
    box.style.height = `${h * scaleY}px`;
    box.style.setProperty('--text-rotation', `${clampRotationAngle(region.rotation_angle || 0)}deg`);
    box.dataset.index = idx;
    box.tabIndex = 0;

    const label = document.createElement('span');
    label.className = 'region-label';
    label.textContent = `#${idx + 1}${region.manual ? ' · manual' : ''}${Math.abs(region.rotation_angle || 0) >= 0.65 ? ` · ${region.rotation_angle > 0 ? '+' : ''}${region.rotation_angle}°` : ''}`;
    box.appendChild(label);

    const previewText = region.translated_text || region.original_text || '';
    const layoutBlocks = layoutBlocksForRegion(region);
    const blockTexts = splitTextForLayout(previewText || 'Texto', layoutBlocks.length);
    const hasMultipleSlots = layoutBlocks.length > 1;
    const cleanUiBase = useCleanBaseForUiText();
    const showInlineEditor = isSelected && state.tool === 'select' && region.visible !== false && !hasMultipleSlots;
    const showPreview = (!showInlineEditor && shouldShowLiveText(region, idx))
      || (isSelected && hasMultipleSlots && region.visible !== false)
      || (cleanUiBase && region.visible !== false);
    if (showInlineEditor) {
      box.classList.add('live-editing');
      const editor = document.createElement('textarea');
      const block = layoutBlocks[0] || textRenderArea(region);
      editor.className = `region-text-editor ui-live-input ${previewText.trim() ? '' : 'empty'}`;
      editor.spellcheck = false;
      editor.dataset.index = String(idx);
      editor.value = previewText;
      editor.placeholder = 'Escribe aquí';
      editor.title = 'Edita directamente el texto visible. La previsualización final se actualiza al salir o guardar.';
      applyInlineEditorLayout(editor, region, block, previewText);
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
      const useExactRasterPreview = region.visible !== false && (cleanUiBase || isSelected);
      if (useExactRasterPreview) {
        box.classList.add('live-raster-preview');
        const raster = document.createElement('img');
        raster.className = 'region-raster-preview';
        raster.dataset.index = String(idx);
        raster.alt = '';
        raster.decoding = 'async';
        raster.draggable = false;
        box.appendChild(raster);
        queueInlineRasterPreview(idx, isSelected ? 0 : 35);
        if (isSelected && hasMultipleSlots) {
          const slotHint = document.createElement('span');
          slotHint.className = 'region-slot-hint';
          slotHint.textContent = 'Edita este diálogo en el panel derecho.';
          box.appendChild(slotHint);
        }
      } else {
        layoutBlocks.forEach((block, blockIndex) => {
          const blockText = blockTexts[blockIndex] || 'Texto';
          const font = previewFontSizeForBlock(region, block, blockText);
          const preview = document.createElement('span');
          preview.className = `region-text-preview ${previewText.trim() ? '' : 'empty'} ${hasMultipleSlots ? 'split-slot' : ''}`;
          applyRendererTextLayout(preview, region, block, blockText);
          preview.textContent = previewText.trim() ? previewLinesForBlock(region, block, blockText, font) : 'Texto';
          box.appendChild(preview);
        });
      }
    }

    if (!showInlineEditor && !showPreview && region.visible !== false) {
      layoutBlocks.forEach((block) => {
        const hit = document.createElement('span');
        hit.className = 'region-text-hit';
        applyRendererTextLayout(hit, region, block, previewText || 'Texto');
        hit.title = 'Seleccionar texto';
        box.appendChild(hit);
      });
    }

    if (isSelected && state.tool === 'select') {
      for (const direction of ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']) {
        const handle = document.createElement('span');
        handle.className = `resize-handle handle-${direction}`;
        handle.dataset.handle = direction;
        handle.title = `Redimensionar ${direction.toUpperCase()}`;
        box.appendChild(handle);
      }
      const rotateHandle = document.createElement('span');
      rotateHandle.className = 'rotation-handle';
      rotateHandle.title = 'Arrastra para girar el texto';
      box.appendChild(rotateHandle);
    }

    box.addEventListener('pointerdown', onBoxPointerDown);
    box.addEventListener('click', (event) => {
      event.stopPropagation();
      if (activeTool() !== 'select') return;
      if (event.target.closest?.('.region-text-editor')) return;
      selectRegion(idx);
    });
    box.addEventListener('dblclick', (event) => {
      event.stopPropagation();
      if (activeTool() !== 'select') return;
      selectRegion(idx);
      focusInlineEditorForRegion(idx, { selectAll: true });
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

  if (activeTool() === 'brush' && state.brushCursor.visible) {
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
  updateCanvasReadout();
}

function textFromEditable(element) {
  const raw = element && 'value' in element ? element.value : element?.innerText;
  return String(raw || '').replace(/\r/g, '').replace(/\u00a0/g, ' ');
}

function syncInlineTextToRegion(element, options = {}) {
  const idx = Number(element?.dataset?.index);
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return;
  const nextText = textFromEditable(element);
  element.classList.toggle('empty', !nextText.trim());
  if (region.translated_text === nextText) {
    if (element.classList.contains('ui-live-input')) refreshInlineEditorLayout(element);
    if (options.preview !== false && overlayLayer?.querySelector?.(`.region-raster-preview[data-index="${idx}"]`)) queueInlineRasterPreview(idx, 220);
    return;
  }
  if (options.history !== false) pushUndoSnapshot('edición de texto directa', { coalesceKey: `inline:${idx}`, coalesceMs: 1200 });
  region.translated_text = nextText;
  region.modified = true;
  if (idx === state.selectedRegion && translatedText) translatedText.value = nextText;
  renderRegionList();
  markDirty();
  if (element.classList.contains('ui-live-input')) refreshInlineEditorLayout(element);
  if (options.preview !== false && overlayLayer?.querySelector?.(`.region-raster-preview[data-index="${idx}"]`)) queueInlineRasterPreview(idx, 220);
  if (options.autosave) scheduleAutoSave(options.reason || 'edición de texto directa');
}

function onInlineTextFocus(event) {
  state.inlineEditingIndex = Number(event.currentTarget?.dataset?.index);
  if (Number.isInteger(state.inlineEditingIndex)) {
    pushUndoSnapshot('edición de texto directa', { coalesceKey: `inline:${state.inlineEditingIndex}`, coalesceMs: 1200 });
  }
  clearTimeout(state.autosaveTimer);
  setAutosaveStatus('pending', 'Editando texto… se guardará al salir del cuadro.');
}

function onInlineTextBlur(event) {
  syncInlineTextToRegion(event.currentTarget, { autosave: false, preview: false });
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
  const target = event.currentTarget;
  if (target && 'setRangeText' in target) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? target.value.length;
    target.setRangeText(text, start, end, 'end');
    target.dispatchEvent(new Event('input', { bubbles: true }));
    return;
  }
  document.execCommand('insertText', false, text);
}

function onInlineTextKeyDown(event) {
  // Dentro de la caja, Enter permite saltos de línea; Ctrl+Enter confirma y Escape devuelve el foco al lienzo.
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    syncInlineTextToRegion(event.currentTarget, { autosave: false, preview: false });
    event.currentTarget.blur();
    saveCurrentPage({ silent: true, force: true, reason: 'confirmar edición directa' }).catch(() => {});
    return;
  }
  if (event.key === 'Escape') {
    event.preventDefault();
    event.currentTarget.blur();
    overlayLayer?.focus?.();
  }
}

function safeSetPointerCapture(element, pointerId) {
  if (!element?.isConnected || typeof element.setPointerCapture !== 'function') return false;
  try {
    element.setPointerCapture(pointerId);
    return true;
  } catch (error) {
    // Chromium can throw InvalidStateError when a render replaces the pointer target.
    // Dragging still works because movement and release are tracked on document.
    if (!['InvalidStateError', 'NotFoundError'].includes(error?.name)) {
      console.warn('No se pudo capturar el puntero.', error);
    }
    return false;
  }
}

function safeReleasePointerCapture(element, pointerId) {
  if (!element?.isConnected || typeof element.releasePointerCapture !== 'function') return;
  try {
    if (typeof element.hasPointerCapture !== 'function' || element.hasPointerCapture(pointerId)) {
      element.releasePointerCapture(pointerId);
    }
  } catch (error) {
    if (!['InvalidStateError', 'NotFoundError'].includes(error?.name)) {
      console.warn('No se pudo liberar el puntero.', error);
    }
  }
}

function onBoxPointerDown(event) {
  if (activeTool() !== 'select') return;
  event.stopPropagation();
  const box = event.currentTarget;
  const idx = Number(box.dataset.index);
  if (!isSelectableRegion(state.regions[idx])) return;

  const targetIsResize = event.target.classList.contains('resize-handle');
  const targetIsRotate = event.target.classList.contains('rotation-handle');
  const targetIsEditor = Boolean(event.target.closest?.('.region-text-editor'));
  const edgeDrag = isNearBoxEdge(event, box);

  if (targetIsEditor) return;

  event.preventDefault();
  // Capture on the stable overlay before selection can rerender and detach `box`.
  safeSetPointerCapture(overlayLayer, event.pointerId);
  selectRegion(idx, false);

  const canMove = edgeDrag || event.target === box || Boolean(event.target.closest?.('.region-label'));
  if (!targetIsResize && !targetIsRotate && !canMove) {
    safeReleasePointerCapture(overlayLayer, event.pointerId);
    return;
  }

  const region = state.regions[idx];
  const mode = targetIsRotate ? 'rotate' : targetIsResize ? 'resize' : 'move';
  pushUndoSnapshot(mode === 'rotate' ? 'girar texto' : mode === 'resize' ? 'redimensionar región' : 'mover región', { coalesceKey: `drag:${idx}:${mode}`, coalesceMs: 500 });
  const imageRect = pageImage.getBoundingClientRect();
  const [rx, ry, rw, rh] = region.bbox;
  const centerX = imageRect.left + (rx + rw / 2) * (imageRect.width / state.naturalWidth);
  const centerY = imageRect.top + (ry + rh / 2) * (imageRect.height / state.naturalHeight);
  state.dragging = {
    idx,
    mode,
    pointerId: event.pointerId,
    isResize: mode === 'resize',
    isRotate: mode === 'rotate',
    handle: event.target.dataset.handle || 'se',
    startX: event.clientX,
    startY: event.clientY,
    startBox: [...region.bbox],
    startRotation: clampRotationAngle(region.rotation_angle || 0),
    centerX,
    centerY,
    startPointerAngle: Math.atan2(event.clientY - centerY, event.clientX - centerX) * 180 / Math.PI,
  };
}


document.addEventListener('pointerdown', (event) => {
  const editor = activeInlineTextEditor();
  if (!editor) return;
  if (event.target.closest?.('.region-text-editor')) return;
  syncInlineTextToRegion(editor, { autosave: false, preview: false });
  editor.blur();
}, true);

overlayLayer.addEventListener('pointerdown', (event) => {
  const page = currentPage();
  if (!page || page.status !== 'ready') return;
  if (event.target.closest?.('.region-box') && activeTool() === 'select') return;
  event.preventDefault();
  const [x, y] = imagePointFromEvent(event);

  if (isPanMode()) {
    state.panning = {
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: canvasCard?.scrollLeft || 0,
      scrollTop: canvasCard?.scrollTop || 0,
    };
    setCanvasPanning(true);
    safeSetPointerCapture(overlayLayer, event.pointerId);
    updateCanvasReadout([x, y]);
    return;
  }

  if (state.tool === 'region') {
    state.selectedRegion = null;
    state.drawingRegion = { startX: x, startY: y, bbox: [x, y, 1, 1] };
    safeSetPointerCapture(overlayLayer, event.pointerId);
    showNoRegion();
    renderOverlay();
    return;
  }

  if (activeTool() === 'brush') {
    state.brushCursor = { x, y, visible: true };
    pushUndoSnapshot('pincel', { coalesceKey: 'brush-stroke', coalesceMs: 250 });
    state.drawingStroke = { points: [[x, y]], radius: Number(brushSize.value || 22), mode: brushMode.value || 'restore_clean', applied: false };
    safeSetPointerCapture(overlayLayer, event.pointerId);
    markDirty();
    renderOverlay();
    return;
  }

  deselectRegion();
});

overlayLayer.addEventListener('pointermove', (event) => updateBrushCursorFromEvent(event));
overlayLayer.addEventListener('pointerenter', (event) => updateBrushCursorFromEvent(event));
overlayLayer.addEventListener('pointerleave', hideBrushCursor);

document.addEventListener('pointermove', (event) => {
  if (state.panning) {
    const pan = state.panning;
    if (canvasCard) {
      canvasCard.scrollLeft = pan.scrollLeft - (event.clientX - pan.startX);
      canvasCard.scrollTop = pan.scrollTop - (event.clientY - pan.startY);
    }
    const rect = pageImage?.getBoundingClientRect?.();
    if (rect && event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom) {
      updateCanvasReadout(imagePointFromEvent(event));
    }
    return;
  }

  if (state.dragging) {
    const drag = state.dragging;
    const region = state.regions[drag.idx];
    if (!isSelectableRegion(region)) return;
    const rect = pageImage.getBoundingClientRect();
    const dx = (event.clientX - drag.startX) / (rect.width / state.naturalWidth);
    const dy = (event.clientY - drag.startY) / (rect.height / state.naturalHeight);
    let [x, y, w, h] = drag.startBox;
    if (drag.mode === 'rotate') {
      const pointerAngle = Math.atan2(event.clientY - drag.centerY, event.clientX - drag.centerX) * 180 / Math.PI;
      const nextAngle = clampRotationAngle(drag.startRotation + pointerAngle - drag.startPointerAngle);
      region.rotation_angle = nextAngle;
      if (region.ui_layout) region.ui_layout.rotation_angle = nextAngle;
    } else if (drag.mode === 'resize') {
      let left = x;
      let top = y;
      let right = x + w;
      let bottom = y + h;
      const handle = drag.handle || 'se';
      if (handle.includes('w')) left = Math.max(0, Math.min(right - 12, left + dx));
      if (handle.includes('e')) right = Math.min(state.naturalWidth, Math.max(left + 12, right + dx));
      if (handle.includes('n')) top = Math.max(0, Math.min(bottom - 12, top + dy));
      if (handle.includes('s')) bottom = Math.min(state.naturalHeight, Math.max(top + 12, bottom + dy));
      x = left;
      y = top;
      w = right - left;
      h = bottom - top;
      region.bbox = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
    } else {
      x = Math.max(0, Math.min(state.naturalWidth - w, x + dx));
      y = Math.max(0, Math.min(state.naturalHeight - h, y + dy));
      region.bbox = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
    }
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

document.addEventListener('pointerup', (event) => {
  safeReleasePointerCapture(overlayLayer, event.pointerId);
  if (state.panning) {
    state.panning = null;
    setCanvasPanning(false);
  }

  if (state.dragging) {
    const reason = state.dragging.mode === 'rotate' ? 'girar texto' : state.dragging.mode === 'resize' ? 'redimensionar región' : 'mover región';
    scheduleAutoSave(reason);
    queueInlineRasterPreview(state.dragging.idx, 0);
  }
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
        font_size: null,
        rotation_angle: 0,
        rotation_confidence: 0,
        text_align: 'center',
        vertical_align: 'middle',
        line_spacing_factor: 1,
        text_offset_x: 0,
        text_offset_y: 0,
        ui_layout: createFullBoxUiLayout(bbox, 'dialogo'),
        ui_text_region: true,
      };
      pushUndoSnapshot('nueva región');
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

document.addEventListener('pointercancel', (event) => {
  safeReleasePointerCapture(overlayLayer, event.pointerId);
  state.panning = null;
  state.dragging = null;
  state.drawingRegion = null;
  state.drawingStroke = null;
  setCanvasPanning(false);
  renderOverlay();
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

function activateEditorTab(name) {
  const target = ['text', 'layers', 'brush'].includes(name) ? name : 'text';
  document.querySelectorAll('[data-editor-tab]').forEach((button) => {
    const active = button.dataset.editorTab === target;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  document.querySelectorAll('[data-editor-pane]').forEach((pane) => {
    pane.classList.toggle('active', pane.dataset.editorPane === target);
  });
}

document.querySelectorAll('[data-editor-tab]').forEach((button) => {
  button.addEventListener('click', () => activateEditorTab(button.dataset.editorTab));
});

function closeMobilePanels() {
  workView?.classList.remove('pages-open', 'editor-open');
  mobilePagesBtn?.setAttribute('aria-expanded', 'false');
  mobileEditorBtn?.setAttribute('aria-expanded', 'false');
  panelBackdrop?.classList.add('hidden');
}

function toggleMobilePanel(panel) {
  const className = panel === 'pages' ? 'pages-open' : 'editor-open';
  const willOpen = !workView?.classList.contains(className);
  closeMobilePanels();
  if (!willOpen || !workView) return;
  workView.classList.add(className);
  (panel === 'pages' ? mobilePagesBtn : mobileEditorBtn)?.setAttribute('aria-expanded', 'true');
  panelBackdrop?.classList.remove('hidden');
}

mobilePagesBtn?.addEventListener('click', () => toggleMobilePanel('pages'));
mobileEditorBtn?.addEventListener('click', () => toggleMobilePanel('editor'));
closePagesBtn?.addEventListener('click', closeMobilePanels);
closeEditorBtn?.addEventListener('click', closeMobilePanels);
panelBackdrop?.addEventListener('click', closeMobilePanels);
shortcutHelpBtn?.addEventListener('click', openShortcutModal);
shortcutModalBackdrop?.addEventListener('click', closeShortcutModal);
closeShortcutModalBtn?.addEventListener('click', closeShortcutModal);
undoHistoryBtn?.addEventListener('click', undoChange);
redoHistoryBtn?.addEventListener('click', redoChange);
quickSaveBtn?.addEventListener('click', () => {
  saveCurrentPage({ silent: false, force: true, reason: 'guardar desde barra superior' }).catch(() => {});
});
window.addEventListener('resize', () => {
  if (window.innerWidth > 1050) closeMobilePanels();
});

function setTool(tool) {
  state.tool = ['select', 'region', 'brush', 'pan'].includes(tool) ? tool : 'select';
  const allToolButtons = [selectTool, newRegionTool, brushTool, panTool, dockSelectTool, dockRegionTool, dockBrushTool, dockPanTool].filter(Boolean);
  allToolButtons.forEach((button) => {
    const active = button.dataset.tool === state.tool;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  const help = {
    select: ['Herramienta activa: seleccionar', 'Haz clic en una región para editarla, muévela o redimensiónala. Atajos: V, Supr, Ctrl+C, Ctrl+X y Ctrl+V.'],
    region: ['Herramienta activa: nueva región', 'Arrastra sobre la página para crear una caja. Luego usa OCR + traducir región o escribe el texto manualmente. Atajo: R.'],
    brush: ['Herramienta activa: pincel', 'Usa Limpiar texto, Inpaint o Restaurar original. Ajusta tamaño con [ y ]. Atajo: B.'],
    pan: ['Herramienta activa: mano', 'Arrastra el lienzo para desplazarte como en un editor de imágenes. También puedes mantener Espacio con cualquier herramienta. Atajo: H.'],
  }[state.tool];
  toolHelpTitle.textContent = help[0];
  toolHelpText.textContent = help[1];
  hideBrushCursor();
  if (state.tool === 'brush') activateEditorTab('brush');
  else if (state.tool === 'region') activateEditorTab('text');
  setPageImageSource();
  updateCanvasReadout();
  updateWorkspaceChrome();
}

[selectTool, newRegionTool, brushTool, panTool, dockSelectTool, dockRegionTool, dockBrushTool, dockPanTool]
  .filter(Boolean)
  .forEach((button) => {
    button.addEventListener('click', () => setTool(button.dataset.tool));
  });
dockFitBtn?.addEventListener('click', () => { setFitZoomIfNeeded(true); applyZoom(); });
brushSize?.addEventListener('input', () => {
  updateBrushSizeLabel();
  setTool('brush');
  requestOverlayRender();
});
brushMode?.addEventListener('change', () => {
  setTool('brush');
  requestOverlayRender();
});
updateBrushSizeLabel();
undoBrushBtn?.addEventListener('click', () => {
  if (!state.brushStrokes.length) return showToast('No hay pinceladas para deshacer.');
  pushUndoSnapshot('deshacer pincel');
  state.brushStrokes.pop();
  markDirty({ autosave: true, reason: 'deshacer pincel' });
  renderOverlay();
  showToast('Última pincelada eliminada.');
});
undoDeleteBtn?.addEventListener('click', undoLastDelete);

function selectRegion(idx, rerender = true) {
  if (!isSelectableRegion(state.regions[idx])) return showNoRegion();
  state.selectedRegion = idx;
  activateEditorTab('text');
  noRegion.classList.add('hidden');
  regionEditor.classList.remove('hidden');
  updateEditorFromRegion();
  renderRegionList();
  updateCanvasReadout();
  if (rerender) setPageImageSource();
}

function showNoRegion() {
  state.selectedRegion = null;
  noRegion.classList.remove('hidden');
  regionEditor.classList.add('hidden');
  renderRegionList();
  updateCanvasReadout();
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
  syncRotationControls(region.rotation_angle || 0);
  if (regionStyleSelect) regionStyleSelect.value = region.style || 'dialogo';
  syncAlignmentControls(region);
  syncLineSpacingControls(region.line_spacing_factor || 1);
  if (textOffsetX) textOffsetX.value = String(clampTextOffset(region.text_offset_x || 0));
  if (textOffsetY) textOffsetY.value = String(clampTextOffset(region.text_offset_y || 0));
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
  const manualFontSize = clampFontSize(fontSizeNumber?.value || fontSize?.value, effectiveManualFontSize(region));
  const nextFontSize = nextAutoFont ? null : manualFontSize;
  const nextRotationAngle = clampRotationAngle(rotationAngleNumber?.value ?? rotationAngle?.value ?? region.rotation_angle ?? 0);
  const nextStyle = String(regionStyleSelect?.value || region.style || 'dialogo');
  const nextTextAlign = normalizeTextAlign(document.querySelector('[data-text-align].active')?.dataset?.textAlign || region.text_align);
  const nextVerticalAlign = normalizeVerticalAlign(document.querySelector('[data-vertical-align].active')?.dataset?.verticalAlign || region.vertical_align);
  const nextLineSpacing = clampLineSpacing(lineSpacingNumber?.value ?? lineSpacing?.value ?? region.line_spacing_factor ?? 1);
  const nextTextOffsetX = clampTextOffset(textOffsetX?.value ?? region.text_offset_x ?? 0);
  const nextTextOffsetY = clampTextOffset(textOffsetY?.value ?? region.text_offset_y ?? 0);
  const currentFontSize = usesAutoFontSize(region) ? null : clampFontSize(region.font_size, null);
  const changed =
    region.original_text !== nextOriginalText ||
    region.translated_text !== nextText ||
    region.restore_original !== nextRestore ||
    region.visible !== nextVisible ||
    usesAutoFontSize(region) !== nextAutoFont ||
    currentFontSize !== nextFontSize ||
    clampRotationAngle(region.rotation_angle || 0) !== nextRotationAngle ||
    String(region.style || 'dialogo') !== nextStyle ||
    normalizeTextAlign(region.text_align) !== nextTextAlign ||
    normalizeVerticalAlign(region.vertical_align) !== nextVerticalAlign ||
    clampLineSpacing(region.line_spacing_factor || 1) !== nextLineSpacing ||
    clampTextOffset(region.text_offset_x || 0) !== nextTextOffsetX ||
    clampTextOffset(region.text_offset_y || 0) !== nextTextOffsetY ||
    region.bbox.some((value, index) => value !== nextBox[index]);

  if (changed && options.history !== false) {
    pushUndoSnapshot('edición de región', { coalesceKey: `panel:${state.selectedRegion}`, coalesceMs: 900 });
  }

  region.original_text = nextOriginalText;
  region.translated_text = nextText;
  region.bbox = nextBox;
  region.restore_original = nextRestore;
  region.visible = nextVisible;
  region.auto_font_size = nextAutoFont;
  region.font_size = nextFontSize;
  region.rotation_angle = nextRotationAngle;
  region.style = nextStyle;
  region.text_align = nextTextAlign;
  region.vertical_align = nextVerticalAlign;
  region.line_spacing_factor = nextLineSpacing;
  region.text_offset_x = nextTextOffsetX;
  region.text_offset_y = nextTextOffsetY;
  if (region.ui_layout) {
    region.ui_layout.rotation_angle = nextRotationAngle;
    region.ui_layout.style = nextStyle;
    region.ui_layout.text_align = nextTextAlign;
    region.ui_layout.vertical_align = nextVerticalAlign;
    region.ui_layout.line_spacing_factor = nextLineSpacing;
    region.ui_layout.text_offset_x = nextTextOffsetX;
    region.ui_layout.text_offset_y = nextTextOffsetY;
  }
  syncRotationControls(nextRotationAngle);
  syncLineSpacingControls(nextLineSpacing);
  syncAlignmentControls(region);
  updateFontControlsDisabled();
  if (changed) {
    markRegionModified(region);
    renderRegionList();
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
function nudgeSelectedRegion(dx, dy, options = {}) {
  const idx = state.selectedRegion;
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return false;
  const [x, y, w, h] = region.bbox;
  const resize = Boolean(options.resize);
  const nextBox = resize
    ? clampBox([x, y, w + dx, h + dy])
    : clampBox([x + dx, y + dy, w, h]);
  if (region.bbox.every((value, index) => value === nextBox[index])) return true;
  pushUndoSnapshot(resize ? 'redimensionar con teclado' : 'mover con teclado', { coalesceKey: `nudge:${idx}:${resize ? 'resize' : 'move'}`, coalesceMs: 700 });
  region.bbox = nextBox;
  markRegionModified(region);
  updateEditorFromRegion();
  renderOverlay({ force: true });
  scheduleAutoSave(resize ? 'redimensionar con teclado' : 'mover con teclado', 650);
  return true;
}


function nudgeSelectedText(dx, dy) {
  const idx = state.selectedRegion;
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return false;
  pushUndoSnapshot('mover texto dentro de la región', { coalesceKey: `text-offset:${idx}`, coalesceMs: 700 });
  region.text_offset_x = clampTextOffset((region.text_offset_x || 0) + dx);
  region.text_offset_y = clampTextOffset((region.text_offset_y || 0) + dy);
  if (region.ui_layout) {
    region.ui_layout.text_offset_x = region.text_offset_x;
    region.ui_layout.text_offset_y = region.text_offset_y;
  }
  markRegionModified(region);
  updateEditorFromRegion();
  renderOverlay({ force: true });
  scheduleAutoSave('mover texto', 650);
  return true;
}

function centerSelectedRegion(axis) {
  const idx = state.selectedRegion;
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return;
  const [x, y, w, h] = region.bbox;
  const next = axis === 'horizontal'
    ? clampBox([(state.naturalWidth - w) / 2, y, w, h])
    : clampBox([x, (state.naturalHeight - h) / 2, w, h]);
  if (region.bbox.every((value, index) => value === next[index])) return;
  pushUndoSnapshot(`centrar región ${axis}`);
  region.bbox = next;
  markRegionModified(region);
  updateEditorFromRegion();
  renderOverlay({ force: true });
  scheduleAutoSave(`centrar región ${axis}`, 500);
}

async function fitSelectedRegionToText() {
  const idx = state.selectedRegion;
  const region = state.regions[idx];
  if (!isSelectableRegion(region)) return showToast('Selecciona una región.');
  fitRegionToTextBtn.disabled = true;
  try {
    const metrics = await fetchRegionMetrics(idx);
    const blocks = Array.isArray(metrics?.blocks) ? metrics.blocks : [];
    const lines = blocks.flatMap((block) => (block.lines || []).map((line) => ({ ...line, stroke: Number(block.stroke_width || 0) })));
    if (!lines.length) return showToast('Escribe texto antes de ajustar la caja.');
    const minX = Math.min(...lines.map((line) => Number(line.visual_x || 0) - line.stroke));
    const minY = Math.min(...lines.map((line) => Number(line.visual_y || 0) - line.stroke));
    const maxX = Math.max(...lines.map((line) => Number(line.visual_x || 0) + Number(line.width || 0) + line.stroke));
    const maxY = Math.max(...lines.map((line) => Number(line.visual_y || 0) + Number(line.height || 0) + line.stroke));
    const padding = Math.max(8, Math.round(Math.max(...blocks.map((block) => Number(block.font_size || 12))) * 0.35));
    const desiredW = Math.max(24, Math.ceil(maxX - minX + padding * 2));
    const desiredH = Math.max(24, Math.ceil(maxY - minY + padding * 2));
    const [x, y, w, h] = region.bbox;
    const centerX = x + w / 2;
    const centerY = y + h / 2;
    pushUndoSnapshot('ajustar región al texto');
    region.bbox = clampBox([centerX - desiredW / 2, centerY - desiredH / 2, desiredW, desiredH]);
    const exactSize = Number(blocks[0]?.font_size);
    if (Number.isFinite(exactSize)) {
      region.auto_font_size = false;
      region.font_size = clampFontSize(exactSize, region.font_size || 24);
    }
    region.text_offset_x = 0;
    region.text_offset_y = 0;
    region.ui_layout = createFullBoxUiLayout(region.bbox, region.style || 'dialogo');
    region.ui_layout.rotation_angle = region.rotation_angle || 0;
    region.ui_layout.text_align = region.text_align || 'center';
    region.ui_layout.vertical_align = region.vertical_align || 'middle';
    region.ui_layout.line_spacing_factor = region.line_spacing_factor || 1;
    markRegionModified(region);
    updateEditorFromRegion();
    renderOverlay({ force: true });
    scheduleAutoSave('ajustar región al texto', 500);
    showToast('Caja ajustada al texto.');
  } catch (error) {
    showToast(error.message || 'No se pudo ajustar la caja.');
  } finally {
    fitRegionToTextBtn.disabled = false;
  }
}

function resetSelectedTypography() {
  const region = selectedRegion();
  if (!region) return;
  pushUndoSnapshot('restablecer tipografía');
  region.text_align = 'center';
  region.vertical_align = 'middle';
  region.line_spacing_factor = 1;
  region.text_offset_x = 0;
  region.text_offset_y = 0;
  region.rotation_angle = 0;
  if (region.ui_layout) {
    region.ui_layout.text_align = 'center';
    region.ui_layout.vertical_align = 'middle';
    region.ui_layout.line_spacing_factor = 1;
    region.ui_layout.text_offset_x = 0;
    region.ui_layout.text_offset_y = 0;
    region.ui_layout.rotation_angle = 0;
  }
  markRegionModified(region);
  updateEditorFromRegion();
  renderOverlay({ force: true });
  scheduleAutoSave('restablecer tipografía', 500);
}


function nextRegionIndex() {
  return state.regions.reduce((max, region) => Math.max(max, Number(region.index || 0)), -1) + 1;
}

[originalText, translatedText, boxX, boxY, boxW, boxH, restoreOriginal, visibleText, autoFontSize].forEach((element) => {
  if (!element) return;
  element.addEventListener('input', updateRegionFromEditor);
  element.addEventListener('change', updateRegionFromEditor);
});
rotationAngle?.addEventListener('input', () => {
  const value = clampRotationAngle(rotationAngle.value);
  if (rotationAngleNumber) rotationAngleNumber.value = String(value);
  updateRegionFromEditor();
});
rotationAngleNumber?.addEventListener('input', () => {
  const value = clampRotationAngle(rotationAngleNumber.value);
  if (rotationAngle) rotationAngle.value = String(Math.max(-45, Math.min(45, value)));
  updateRegionFromEditor();
});
rotationAngleNumber?.addEventListener('change', () => {
  syncRotationControls(rotationAngleNumber.value);
  updateRegionFromEditor();
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
regionStyleSelect?.addEventListener('change', updateRegionFromEditor);
[lineSpacing, lineSpacingNumber].filter(Boolean).forEach((control) => {
  control.addEventListener('input', () => {
    syncLineSpacingControls(control.value);
    updateRegionFromEditor();
  });
  control.addEventListener('change', () => {
    syncLineSpacingControls(control.value);
    updateRegionFromEditor();
  });
});
[textOffsetX, textOffsetY].filter(Boolean).forEach((control) => {
  control.addEventListener('input', updateRegionFromEditor);
  control.addEventListener('change', updateRegionFromEditor);
});
document.querySelectorAll('[data-text-align]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-text-align]').forEach((item) => item.classList.toggle('active', item === button));
    updateRegionFromEditor();
  });
});
document.querySelectorAll('[data-vertical-align]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-vertical-align]').forEach((item) => item.classList.toggle('active', item === button));
    updateRegionFromEditor();
  });
});
resetTextOffsetBtn?.addEventListener('click', () => {
  if (textOffsetX) textOffsetX.value = '0';
  if (textOffsetY) textOffsetY.value = '0';
  updateRegionFromEditor();
});
resetTypographyBtn?.addEventListener('click', resetSelectedTypography);
centerRegionHorizontalBtn?.addEventListener('click', () => centerSelectedRegion('horizontal'));
centerRegionVerticalBtn?.addEventListener('click', () => centerSelectedRegion('vertical'));
fitRegionToTextBtn?.addEventListener('click', fitSelectedRegionToText);

translateOriginalBtn?.addEventListener('click', async () => {
  const page = currentPage();
  const region = state.regions[state.selectedRegion];
  if (!page || page.status !== 'ready' || !isSelectableRegion(region)) return showToast('Selecciona una región lista.');

  updateRegionFromEditor({ autosave: false });
  const sourceText = originalText.value;
  if (!sourceText.trim()) return showToast('La transcripción está vacía.');

  translateOriginalBtn.disabled = true;
  try {
    showToast('Traduciendo la transcripción corregida…');
    const result = await requestJson(`/api/jobs/${state.job.job_id}/pages/${page.index}/translate-region`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ original_text: sourceText }),
    });
    pushUndoSnapshot('retraducir transcripción');
    region.original_text = sourceText;
    region.translated_text = result.translated_text || '';
    markRegionModified(region);
    updateEditorFromRegion();
    renderOverlay({ force: true });
    scheduleAutoSave('retraducir transcripción', 180);
    showToast(region.translated_text ? 'Traducción actualizada desde la transcripción corregida.' : 'El traductor no devolvió texto.');
  } catch (error) {
    showToast(error.message);
  } finally {
    translateOriginalBtn.disabled = false;
  }
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
    pushUndoSnapshot('OCR de región');
    region.original_text = result.original_text || '';
    if (result.translated_text) region.translated_text = result.translated_text;
    if (Number.isFinite(Number(result.rotation_angle))) {
      region.rotation_angle = clampRotationAngle(result.rotation_angle, region.rotation_angle || 0);
      region.rotation_confidence = Number(result.rotation_confidence || 0);
      if (region.ui_layout) region.ui_layout.rotation_angle = region.rotation_angle;
    }
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
  pushUndoSnapshot('pegar región');
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
  pushUndoSnapshot('eliminar región');
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
  pushUndoSnapshot('deshacer borrado');
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
  if (workView.classList.contains('hidden')) return;

  const key = event.key.toLowerCase();
  if (shortcutModal && !shortcutModal.classList.contains('hidden')) {
    if (key === 'escape') {
      event.preventDefault();
      closeShortcutModal();
    }
    return;
  }
  const ctrl = event.ctrlKey || event.metaKey;

  if (ctrl && key === 'z' && !isTypingTarget(event.target)) {
    event.preventDefault();
    if (event.shiftKey) redoChange();
    else undoChange();
    return;
  }
  if (ctrl && key === 'y' && !isTypingTarget(event.target)) {
    event.preventDefault();
    redoChange();
    return;
  }

  if (isTypingTarget(event.target)) return;

  if (event.code === 'Space') {
    event.preventDefault();
    if (!state.spacePan) {
      state.spacePan = true;
      setCanvasPanning(false);
      requestOverlayRender();
      updateCanvasReadout();
    }
    return;
  }

  const page = currentPage();

  if (ctrl && key === 'j' && state.selectedRegion != null && page?.status === 'ready') {
    event.preventDefault();
    copySelectedRegion(false);
    pasteRegion();
    return;
  }
  if (ctrl && key === 's') {
    event.preventDefault();
    saveCurrentPage({ silent: false, force: true, reason: 'atajo guardar' }).catch(() => {});
    return;
  }
  if (ctrl && (key === '0' || event.code === 'Digit0')) {
    event.preventDefault();
    setFitZoomIfNeeded(true);
    applyZoom();
    return;
  }
  if (ctrl && ['+', '='].includes(key)) {
    event.preventDefault();
    zoomTo(state.zoom * 1.15);
    return;
  }
  if (ctrl && key === '-') {
    event.preventDefault();
    zoomTo(state.zoom / 1.15);
    return;
  }

  if (event.altKey && key.startsWith('arrow') && state.selectedRegion != null && page?.status === 'ready') {
    const step = event.shiftKey ? 10 : 1;
    const vector = {
      arrowleft: [-step, 0],
      arrowright: [step, 0],
      arrowup: [0, -step],
      arrowdown: [0, step],
    }[key];
    if (vector && nudgeSelectedText(vector[0], vector[1])) {
      event.preventDefault();
      return;
    }
  }

  if (key.startsWith('arrow') && state.selectedRegion != null && page?.status === 'ready') {
    const step = event.shiftKey ? 10 : 1;
    const vector = {
      arrowleft: [-step, 0],
      arrowright: [step, 0],
      arrowup: [0, -step],
      arrowdown: [0, step],
    }[key];
    if (vector && nudgeSelectedRegion(vector[0], vector[1], { resize: ctrl })) {
      event.preventDefault();
      return;
    }
  }

  if (!ctrl) {
    if (key === '?' || (key === '/' && event.shiftKey)) { event.preventDefault(); openShortcutModal(); return; }
    if (key === 'v') { event.preventDefault(); setTool('select'); return; }
    if (key === 'b') { event.preventDefault(); setTool('brush'); return; }
    if (key === 'r') { event.preventDefault(); setTool('region'); return; }
    if (key === 'h') { event.preventDefault(); setTool('pan'); return; }
    if (key === '[') { event.preventDefault(); nudgeBrushSize(-2); return; }
    if (key === ']') { event.preventDefault(); nudgeBrushSize(2); return; }
    if (key === 'arrowleft' && state.job && state.pageIndex > 0) { event.preventDefault(); prevBtn.click(); return; }
    if (key === 'arrowright' && state.job && state.pageIndex < (state.job.pages?.length || 1) - 1) { event.preventDefault(); nextBtn.click(); return; }
  }

  if (!page || page.status !== 'ready') return;

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

document.addEventListener('keyup', (event) => {
  if (event.code !== 'Space' || !state.spacePan) return;
  state.spacePan = false;
  if (state.panning) state.panning = null;
  setCanvasPanning(false);
  requestOverlayRender();
  updateCanvasReadout();
});

async function controlCurrentJob(action) {
  if (!state.job) return;
  const labels = { pause: 'Pausando…', resume: 'Reanudando…', cancel: 'Cancelando…' };
  showToast(labels[action] || 'Actualizando trabajo…');
  try {
    const job = await requestJson(`/api/jobs/${state.job.job_id}/${action}`, { method: 'POST' });
    state.job = job;
    renderJob(job);
    if (!['ready', 'failed', 'cancelled'].includes(job.status)) startPolling(job.job_id);
  } catch (error) {
    showToast(error.message);
  }
}

pauseJobBtn.addEventListener('click', () => controlCurrentJob('pause'));
resetRotationBtn?.addEventListener('click', () => {
  syncRotationControls(0);
  updateRegionFromEditor();
});

resumeJobBtn.addEventListener('click', () => controlCurrentJob('resume'));
cancelJobBtn.addEventListener('click', () => controlCurrentJob('cancel'));

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

function regionToRenderPatch(region, idx) {
  return {
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
    font_size: region.auto_font_size === false ? clampFontSize(region.font_size, null) : null,
    rotation_angle: clampRotationAngle(region.rotation_angle || 0),
    text_align: normalizeTextAlign(region.text_align),
    vertical_align: normalizeVerticalAlign(region.vertical_align),
    line_spacing_factor: clampLineSpacing(region.line_spacing_factor || 1),
    text_offset_x: clampTextOffset(region.text_offset_x || 0),
    text_offset_y: clampTextOffset(region.text_offset_y || 0),
    ui_layout: region.ui_layout || null,
    ui_text_region: Boolean(region.ui_text_region || region.ui_layout?.ui_text_region),
  };
}

function inlinePreviewCacheKey(region, idx) {
  return JSON.stringify(regionToRenderPatch(region, idx));
}

function queueInlineRasterPreview(idxOrElement, delay = 120) {
  const idx = typeof idxOrElement === 'number' ? idxOrElement : Number(idxOrElement?.dataset?.index);
  if (!Number.isInteger(idx) || !isSelectableRegion(state.regions[idx])) return;
  // Durante la edición directa mantenemos visible el textarea HTML para que
  // el cursor, la selección y la escritura sean estables y predecibles.
  if (state.inlineEditingIndex === idx) return;
  const page = currentPage();
  const img = overlayLayer?.querySelector?.(`.region-raster-preview[data-index="${idx}"]`);
  if (!page || !img) return;
  const key = `${state.job?.job_id || ''}:${page.index}:${inlinePreviewCacheKey(state.regions[idx], idx)}`;
  if (img.dataset.previewKey === key && img.src) return;
  const cachedUrl = cachedRasterPreview(key);
  if (cachedUrl) {
    img.dataset.previewKey = key;
    img.classList.remove('loading');
    img.src = cachedUrl;
    return;
  }
  if (state.previewInFlightKeys.has(key)) return;
  clearTimeout(inlinePreviewTimers.get(idx));
  inlinePreviewTimers.set(idx, setTimeout(() => updateInlineRasterPreview(idx).catch((error) => {
    if (error?.name !== 'AbortError') console.warn('No se pudo generar la previsualización rasterizada:', error);
  }), Math.max(0, delay)));
}

async function updateInlineRasterPreview(idx) {
  const page = currentPage();
  const region = state.regions[idx];
  if (!page || page.status !== 'ready' || !state.job?.job_id || !isSelectableRegion(region)) return;
  const img = overlayLayer?.querySelector?.(`.region-raster-preview[data-index="${idx}"]`);
  if (!img) return;

  const key = `${state.job.job_id}:${page.index}:${inlinePreviewCacheKey(region, idx)}`;
  if (img.dataset.previewKey === key && img.src) return;
  const cachedUrl = cachedRasterPreview(key);
  if (cachedUrl) {
    img.dataset.previewKey = key;
    img.classList.remove('loading');
    img.src = cachedUrl;
    return;
  }
  if (state.previewInFlightKeys.has(key)) return;
  state.previewInFlightKeys.add(key);

  inlinePreviewControllers.get(idx)?.abort();
  const controller = new AbortController();
  inlinePreviewControllers.set(idx, controller);
  const serial = String(++inlinePreviewSerial);
  img.dataset.serial = serial;
  img.classList.add('loading');

  try {
    const response = await fetch(`/api/jobs/${state.job.job_id}/pages/${page.index}/region-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ region: regionToRenderPatch(region, idx) }),
      signal: controller.signal,
    });
    if (!response.ok) {
      let payload = null;
      try { payload = await response.json(); } catch (_) {}
      throw new Error(payload?.detail || response.statusText || 'No se pudo previsualizar la región.');
    }
    const blob = await response.blob();
    if (controller.signal.aborted) return;
    const currentImg = overlayLayer?.querySelector?.(`.region-raster-preview[data-index="${idx}"]`);
    if (!currentImg || currentImg.dataset.serial !== serial) return;
    const url = URL.createObjectURL(blob);
    rememberRasterPreview(key, url);
    currentImg.onload = () => {
      currentImg.classList.remove('loading');
    };
    currentImg.onerror = () => {
      currentImg.classList.remove('loading');
    };
    currentImg.dataset.previewKey = key;
    currentImg.src = url;
  } finally {
    state.previewInFlightKeys.delete(key);
  }
}

function buildRenderPayload(operation = 'render') {
  return {
    operation,
    inpaint_model: brushInpaintModel?.value || 'job',
    background_revision: state.backgroundRevision || 'base',
    regions: state.regions.map((region, idx) => regionToRenderPatch(region, idx)),
    brush_strokes: state.brushStrokes.map((stroke) => ({
      points: stroke.points,
      radius: Number(stroke.radius || 18),
      mode: stroke.mode || 'restore_clean',
      applied: Boolean(stroke.applied),
    })),
  };
}

function syncCommittedRegionSources(updatedRegions = []) {
  const committedByIndex = new Map(
    (updatedRegions || []).map((region, idx) => [Number(region?.index ?? idx), region]),
  );
  state.regions.forEach((region, idx) => {
    const committed = committedByIndex.get(Number(region?.index ?? idx));
    const sourceBox = committed?.source_bbox || committed?.bbox;
    if (!region || !Array.isArray(sourceBox) || sourceBox.length < 4) return;
    // Aunque haya cambios más nuevos en bbox/texto, la posición que acaba de quedar
    // rasterizada en el servidor sí debe convertirse en el nuevo origen a limpiar.
    region.source_bbox = normalizeBox(sourceBox);
  });
}

function queuePendingSave(options = {}) {
  const pageKey = currentHistoryPageKey();
  if (!pageKey) return;
  const next = {
    pageKey,
    silent: options.silent !== false,
    force: true,
    reason: options.reason || 'cambios durante guardado',
    markInpaintApplied: Boolean(options.markInpaintApplied),
    operation: options.operation || null,
  };
  const previous = state.pendingSaveOptions;
  if (!previous || previous.pageKey !== pageKey) {
    state.pendingSaveOptions = next;
    return;
  }
  state.pendingSaveOptions = {
    ...previous,
    silent: previous.silent && next.silent,
    force: true,
    reason: next.reason || previous.reason,
    markInpaintApplied: previous.markInpaintApplied || next.markInpaintApplied,
    // Una operación explícita (restaurar original/inpaint) no debe perderse si
    // llega otro autosave genérico mientras el guardado actual sigue en curso.
    operation: next.operation || previous.operation || null,
  };
}

async function saveCurrentPage({ silent = false, force = false, reason = 'manual', markInpaintApplied = false, operation = null } = {}) {
  const page = currentPage();
  if (!page || page.status !== 'ready') {
    if (!silent) showToast('La página aún no está lista.');
    return null;
  }
  const inlineEditor = activeInlineTextEditor();
  if (inlineEditor) syncInlineTextToRegion(inlineEditor, { autosave: false, preview: false });
  else if (state.selectedRegion != null) updateRegionFromEditor({ render: false, autosave: false });
  if (!markInpaintApplied && operation !== 'mask_eraser' && hasPendingInpaintStroke()) {
    setAutosaveStatus('pending', 'Hay una máscara de inpaint pendiente. Pulsa “Aplicar inpaint”.');
    updateHeavyActions();
    return null;
  }

  if (state.autosaveInFlight) {
    queuePendingSave({ silent, force, reason, markInpaintApplied, operation });
    setAutosaveStatus('pending', 'Hay cambios nuevos pendientes de guardar…');
    return null;
  }

  clearTimeout(state.autosaveTimer);
  const pageKey = currentHistoryPageKey();
  const saveRevision = state.editRevision;
  const resolvedOperation = operation || (markInpaintApplied ? 'inpaint' : 'render');
  const payload = buildRenderPayload(resolvedOperation);
  saveBtn.disabled = true;
  state.autosaveInFlight = true;
  updateWorkspaceChrome();
  setAutosaveStatus(markInpaintApplied ? 'saving' : 'saving', markInpaintApplied ? 'Aplicando inpaint…' : 'Guardando cambios…');
  try {
    const updatedPage = await requestJson(`/api/jobs/${state.job.job_id}/pages/${page.index}/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.job.pages[page.index] = updatedPage;
    const stillOnSavedPage = currentHistoryPageKey() === pageKey;
    if (stillOnSavedPage) {
      syncCommittedRegionSources(updatedPage.regions || []);
      const hasNewerChanges = state.editRevision !== saveRevision;
      const rebasedBackground = hasNewerChanges
        ? reconcileCommittedBackgroundRevision(payload, updatedPage, pageKey)
        : false;
      state.pageStamp = `${state.job.job_id}:${page.index}:${updatedPage.updated_at || ''}`;
      state.variant = 'current';
      markActiveVariant();

      if (hasNewerChanges) {
        // No reemplazar el estado local con una respuesta que representa una versión
        // anterior. Ese comportamiento era la causa intermitente de pinceladas que
        // desaparecían cuando el usuario seguía editando durante un autosave.
        state.dirty = true;
        queuePendingSave({ silent: true, force: true, reason: rebasedBackground ? 'continuar sobre fondo actualizado' : 'cambios durante guardado' });
        setAutosaveStatus(
          'pending',
          rebasedBackground
            ? 'Inpaint confirmado; guardando las pinceladas posteriores sobre ese resultado…'
            : 'Cambios nuevos pendientes… se guardarán a continuación.',
        );
        setPageImageSource(updatedPage);
        renderOverlay({ force: true });
      } else {
        state.backgroundRevision = updatedPage.background_revision || state.backgroundRevision || 'base';
        state.dirty = false;
        state.deletedStack = [];
        state.brushStrokes = cloneBrushStrokes(updatedPage.brush_strokes || []);
        setAutosaveStatus('saved', markInpaintApplied ? 'Inpaint aplicado y guardado.' : 'Cambios guardados automáticamente.');
        if (!silent || markInpaintApplied) showToast(markInpaintApplied ? 'Inpaint aplicado.' : 'Corrección guardada.');
        if (activeInlineTextEditor()) {
          state.deferredOverlayRender = true;
          updateHeavyActions();
        } else {
          renderCurrentPage();
          updateHeavyActions();
        }
      }
    }
    updateWorkspaceChrome();
    return updatedPage;
  } catch (error) {
    setAutosaveStatus('error', error.message || 'No se pudo guardar.');
    if (!silent) showToast(error.message);
    throw error;
  } finally {
    state.autosaveInFlight = false;
    saveBtn.disabled = false;
    updateWorkspaceChrome();
    const pending = state.pendingSaveOptions;
    if (pending && pending.pageKey === currentHistoryPageKey()) {
      state.pendingSaveOptions = null;
      setTimeout(() => {
        saveCurrentPage(pending).catch(() => {});
      }, 0);
    }
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
    state.editRevision = 0;
    state.pendingSaveOptions = null;
    state.pageStamp = `${state.job.job_id}:${page.index}:${updatedPage.updated_at || ''}`;
    state.variant = 'translated';
    state.selectedRegion = null;
    state.regions = cloneRegions(updatedPage.regions || []);
    state.brushStrokes = [];
    state.backgroundRevision = updatedPage.background_revision || 'base';
    state.deletedStack = [];
    clearHistory();
    cleanupInlinePreviewResources();
    clearRasterPreviewCache();
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
  cleanupInlinePreviewResources();
  clearRasterPreviewCache();
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
  cleanupInlinePreviewResources();
  clearRasterPreviewCache();
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

updateWorkspaceChrome();
