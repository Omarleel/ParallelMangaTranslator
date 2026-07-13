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
