from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "parallel_manga_translator" / "ui" / "static"


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)


def test_editor_html_has_unique_ids_and_workspace_controls() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    parser = _IdCollector()
    parser.feed(html)

    duplicates = {value for value in parser.ids if parser.ids.count(value) > 1}
    assert not duplicates
    for element_id in (
        "pauseJobBtn",
        "resumeJobBtn",
        "cancelJobBtn",
        "undoHistoryBtn",
        "redoHistoryBtn",
        "quickSaveBtn",
        "shortcutModal",
        "statusSelection",
        "fitRegionToTextBtn",
        "alignLeftBtn",
        "alignMiddleBtn",
        "lineSpacing",
        "textOffsetX",
        "resetTypographyBtn",
        "translateOriginalBtn",
        "retranslateBtn",
        "retranslateModal",
        "retranslateTranslator",
        "retranslateTargetLanguage",
        "retranslateOverwrite",
        "confirmRetranslateBtn",
    ):
        assert element_id in parser.ids


def test_job_controls_and_mobile_workspace_cannot_collapse() -> None:
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert ".job-control-row #cancelJobBtn { grid-column: 1 / -1; }" in css
    assert "min-height: 40px;" in css
    assert "white-space: nowrap;" in css
    assert "grid-template-columns: minmax(0, 1fr);" in css
    assert ".tool-dock {\n    width: 100%;" in css


def test_workspace_javascript_connects_history_save_and_shortcut_help() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "undoHistoryBtn?.addEventListener('click', undoChange);" in javascript
    assert "redoHistoryBtn?.addEventListener('click', redoChange);" in javascript
    assert "quickSaveBtn?.addEventListener('click'" in javascript
    assert "shortcutHelpBtn?.addEventListener('click', openShortcutModal);" in javascript
    assert "function updateWorkspaceChrome()" in javascript
    assert "backgroundRevision: state.backgroundRevision || 'base'" in javascript
    assert "background_revision: state.backgroundRevision || 'base'" in javascript
    assert "state.backgroundRevision = snapshot.backgroundRevision || 'base';" in javascript


def test_job_retranslation_dialog_is_wired_to_the_api() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert "retranslateBtn?.addEventListener('click', openRetranslateModal);" in javascript
    assert "confirmRetranslateBtn?.addEventListener('click', submitRetranslation);" in javascript
    assert "/retranslate`" in javascript
    assert "overwrite_manual: !!retranslateOverwrite?.checked," in javascript
    # Solo tiene sentido sobre un trabajo terminado con páginas listas.
    assert "retranslateBtn.disabled = !terminal || readyPages === 0;" in javascript
    assert ".retranslate-dialog" in css


def test_editable_preview_and_photoshop_transform_tools_are_wired() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert "/region-preview" in javascript
    assert "/region-metrics" in javascript
    assert "if (state.inlineEditingIndex === idx) return;" in javascript
    assert "classList.add('live-editing');" in javascript
    assert "classList.add('raster-ready')" not in javascript
    assert "for (const direction of ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'])" in javascript
    assert "rotation-handle" in javascript
    assert "function fitSelectedRegionToText()" in javascript
    assert "function nudgeSelectedText(dx, dy)" in javascript
    assert '@font-face' not in css
    assert '/api/editor/font' not in css
    assert '.region-box.live-editing.raster-ready .region-text-editor.ui-live-input' not in css
    assert '.handle-nw' in css and '.handle-se' in css and '.rotation-handle' in css
    assert "Alt + flechas" in html
    assert "Ctrl J" in html


def test_raster_preview_cache_is_initialized_before_editor_reset() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    state_block = javascript.split("\n};", 1)[0]
    assert "previewCache: new Map()," in state_block
    assert "previewInFlightKeys: new Set()," in state_block
    assert "metricsCache" not in state_block
    assert "metricsInFlightKeys" not in state_block
    assert "state.previewCache ??= new Map();" in javascript
    assert "state.previewInFlightKeys ??= new Set();" in javascript
    assert javascript.index("previewCache: new Map(),") < javascript.index("function resetEditorState()")



def test_pointer_capture_survives_overlay_rerenders() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function safeSetPointerCapture(element, pointerId)" in javascript
    assert "function safeReleasePointerCapture(element, pointerId)" in javascript
    assert "safeSetPointerCapture(overlayLayer, event.pointerId);" in javascript
    assert "selectRegion(idx, false);" in javascript
    assert "box.setPointerCapture(event.pointerId);" not in javascript
    assert "document.addEventListener('pointercancel'" in javascript
    assert javascript.count(".setPointerCapture(") == 1


def test_autosave_serializes_requests_and_keeps_committed_region_origin() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function queuePendingSave(options = {})" in javascript
    assert "if (state.autosaveInFlight) {" in javascript
    assert "const saveRevision = state.editRevision;" in javascript
    assert "const hasNewerChanges = state.editRevision !== saveRevision;" in javascript
    assert "function syncCommittedRegionSources(updatedRegions = [])" in javascript
    assert "syncCommittedRegionSources(updatedPage.regions || []);" in javascript
    assert javascript.index("syncCommittedRegionSources(updatedPage.regions || []);") < javascript.index(
        "const hasNewerChanges = state.editRevision !== saveRevision;"
    )


def test_transcription_retranslate_button_is_wired_to_corrected_source_text() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'id="translateOriginalBtn"' in html
    assert "const translateOriginalBtn = $('translateOriginalBtn');" in javascript
    assert "/translate-region" in javascript
    assert "body: JSON.stringify({ original_text: sourceText })" in javascript
    assert "region.translated_text = result.translated_text || '';" in javascript


def test_renderer_preview_does_not_collapse_repeated_line_breaks() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert ".replace(/\n{3,}/g, '\n\n')" not in javascript


def test_background_commit_is_rebased_when_user_paints_during_slow_inpaint() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function stripCommittedBrushPrefix(currentStrokes, committedStrokes)" in javascript
    assert "function reconcileCommittedBackgroundRevision(payload, updatedPage, pageKey)" in javascript
    assert "state.backgroundRevision = nextRevision;" in javascript
    assert "state.brushStrokes = stripped.remaining;" in javascript
    assert "rebaseHistoryAfterBackgroundCommit" in javascript
    assert "const rebasedBackground = hasNewerChanges" in javascript
    assert "continuar sobre fondo actualizado" in javascript

def test_polling_cannot_roll_back_a_newer_manual_background_revision() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function pageUpdatedAtValue(page)" in javascript
    assert "function mergePolledJob(currentJob, incomingJob)" in javascript
    assert "if (pageUpdatedAtValue(currentPage) > pageUpdatedAtValue(incomingPage))" in javascript
    assert "return currentPage;" in javascript
    assert "const job = mergePolledJob(state.job, incomingJob);" in javascript
    assert "state.job = job;" in javascript


def test_new_manual_region_focuses_inline_text_editor_immediately() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    creation_start = javascript.index("pushUndoSnapshot('nueva región');")
    creation_end = javascript.index("if (state.drawingStroke)", creation_start)
    creation_block = javascript[creation_start:creation_end]

    assert "setTool('select');" in creation_block
    assert "selectRegion(state.selectedRegion, false);" in creation_block
    assert "renderOverlay();" in creation_block
    assert "focusInlineEditorForRegion(state.selectedRegion);" in creation_block
    assert creation_block.index("renderOverlay();") < creation_block.index("focusInlineEditorForRegion(state.selectedRegion);")
    assert '<script src="/static/app.js?v=6"></script>' in html
