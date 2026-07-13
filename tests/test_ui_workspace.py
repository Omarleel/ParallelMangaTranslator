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
