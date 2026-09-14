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
        "retranslateModel",
        "confirmRetranslateBtn",
        "translationEventsBtn",
        "translationEventsModal",
        "translationRunList",
        "translationEventList",
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
    assert "llm_model:" in javascript
    assert "/translation-runs`" in javascript
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
    assert '<script src="/static/app.js?v=16"></script>' in html


def test_setup_actions_keep_the_primary_button_usable_with_a_long_project_title() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="continueLastTitle"' in html
    assert '<span class="continue-last-label">Continuar último trabajo</span>' in html
    assert ".continue-last-title {" in css
    assert "text-overflow: ellipsis;" in css
    # El título largo ya no viaja en el textContent del botón, que era lo que estrechaba
    # al botón principal de la fila.
    assert "continueLastBtn.textContent = `Continuar último trabajo" not in javascript
    assert "if (continueLastTitle) continueLastTitle.textContent = lastJob.title || 'Proyecto';" in javascript


def test_translation_history_menu_is_wired_in_both_views() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'id="historyModal"' in html
    assert 'id="openHistoryBtn"' in html and 'id="historyBtn"' in html
    assert 'id="historyList"' in html and 'id="historySearch"' in html
    assert "openHistoryBtn?.addEventListener('click', () => openHistoryModal());" in javascript
    assert "historyBtn?.addEventListener('click', () => openHistoryModal());" in javascript
    assert "async function loadJobHistory()" in javascript
    assert "async function openJobById(jobId, { pageIndex = null, fromRoute = false } = {})" in javascript
    # El listado no puede arrastrar las páginas de cada trabajo: hay proyectos de cientos.
    assert "'/api/jobs?include_pages=false'" in javascript
    assert ".history-list {" in css


def test_moving_a_region_hides_the_text_already_rasterized_in_the_page() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert "function regionGhostBox(region)" in javascript
    assert "function appendGhostPatches(scaleX, scaleY, width, height)" in javascript
    assert "appendGhostPatches(scaleX, scaleY, width, height);" in javascript
    assert "pageLayerUrl(page, 'background')" in javascript
    assert ".region-ghost-patch {" in css
    # Una máscara de inpaint pendiente ya no puede bloquear el guardado: ese bloqueo
    # dejaba el texto rasterizado en la posición anterior de la región.
    assert "Hay una máscara de inpaint pendiente. Pulsa “Aplicar inpaint”." not in javascript


def test_closing_a_job_returns_to_the_setup_view() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'id="closeJobBtn"' in html
    assert "async function closeCurrentJob(" in javascript
    assert "closeJobBtn?.addEventListener('click'" in javascript
    # El botón de cerrar el cajón de páginas es solo de móvil: en escritorio salía
    # igualmente porque `.icon-button` redefinía el display después de `.mobile-only`.
    assert ".mobile-only { display: none; }" in css
    assert css.index(".icon-button {") < css.index(".mobile-only { display: none; }")


def test_hash_routes_survive_a_reload() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function buildRouteHash()" in javascript
    assert "function parseRouteHash(rawHash)" in javascript
    assert "function syncRoute(" in javascript
    assert "async function applyRoute(route)" in javascript
    assert "window.addEventListener('hashchange'" in javascript
    assert "`#/trabajo/${encodeURIComponent(state.job.job_id)}/pagina/${Number(state.pageIndex || 0) + 1}`" in javascript
    # El arranque ya no muestra siempre el asistente: aplica la ruta de la URL.
    assert "const route = parseRouteHash(location.hash);" in javascript
    assert "await applyRoute(route);" in javascript
    # Un cambio de página es una ruta más, y pasa por un único camino.
    assert "function goToPage(index, { save = true } = {})" in javascript
    assert "prevBtn.addEventListener('click', () => goToPage(state.pageIndex - 1));" in javascript
    assert "button.addEventListener('click', () => goToPage(page.index));" in javascript


def test_deleting_a_job_from_the_history_asks_for_confirmation_first() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert "async function deleteJobById(jobId)" in javascript
    assert "method: 'DELETE'" in javascript
    # Primer clic: solo marca la tarjeta. El borrado real vive tras el segundo botón.
    assert "state.historyPendingDelete = button.dataset.historyDelete;" in javascript
    assert "data-history-confirm-delete=" in javascript
    assert "data-history-cancel-delete=" in javascript
    assert "Eliminar definitivamente" in javascript
    assert ".history-confirm {" in css
    delete_call = javascript.index("async function deleteJobById(jobId)")
    assert javascript.index("state.historyPendingDelete = button.dataset.historyDelete;") < delete_call


def test_delete_endpoint_is_exposed_and_allowed_by_cors() -> None:
    app_source = (
        Path(__file__).resolve().parents[1]
        / "parallel_manga_translator" / "ui" / "app.py"
    ).read_text(encoding="utf-8")

    assert '@app.delete("/api/jobs/{job_id}")' in app_source
    assert 'allow_methods=["GET", "POST", "DELETE"]' in app_source
    assert "status_code=409" in app_source


def test_promoting_a_corrected_job_to_dataset_eval_is_wired() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'id="datasetBtn"' in html and 'id="datasetModal"' in html
    assert 'id="datasetName"' in html and 'id="datasetOverwrite"' in html
    assert "datasetBtn?.addEventListener('click', openDatasetModal);" in javascript
    assert "async function openDatasetModal()" in javascript
    assert "async function submitDatasetCase()" in javascript
    assert "/dataset-case`" in javascript
    # El botón no se ofrece si nadie corrigió nada: el caso no valdría como referencia.
    assert "function correctedPageCount(job)" in javascript
    assert "datasetBtn.disabled = !terminal || corrected === 0;" in javascript
    assert ".dataset-summary {" in css


def test_dataset_case_endpoints_exist_and_reuse_the_cli_builder() -> None:
    ui_dir = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui"
    app_source = (ui_dir / "app.py").read_text(encoding="utf-8")
    manager_source = (ui_dir / "job_manager.py").read_text(encoding="utf-8")

    assert '@app.get("/api/jobs/{job_id}/dataset-case")' in app_source
    assert '@app.post("/api/jobs/{job_id}/dataset-case")' in app_source
    # El mismo constructor que `python -m ...eval_dataset build`: un solo formato de caso.
    assert "build_case_from_ui_job" in manager_source
    assert "resolve_dataset_dir" in manager_source
    assert "build_ground_truth_page" in manager_source


def test_job_actions_live_in_a_modal_so_the_page_list_gets_the_room() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    sidebar = html[html.index('id="progressPanel"'):html.index('id="pagesPanel"')]
    # En la barra lateral solo quedan progreso, transporte y el disparador del modal.
    assert 'id="jobActionsBtn"' in sidebar
    for moved in ("retranslateBtn", "translationEventsBtn", "exportBtn", "datasetBtn", "historyBtn", "newJobBtn"):
        assert f'id="{moved}"' not in sidebar, f"{moved} debería vivir en el modal de acciones"
        assert f'id="{moved}"' in html

    modal = html[html.index('id="jobActionsModal"'):html.index('id="datasetModal"')]
    # Retraducir, eventos, exportar ZIP, exportar textos, importar textos, dataset_eval,
    # historial y nuevo trabajo.
    assert modal.count("data-job-action") == 8

    assert "function openJobActionsModal()" in javascript
    assert "jobActionsBtn?.addEventListener('click', openJobActionsModal);" in javascript
    # Una acción abre su propia ventana: el menú no puede quedarse debajo.
    assert "if (event.target.closest?.('button[data-job-action]')) closeJobActionsModal();" in javascript
    # El contador del ZIP ya no puede escribirse como textContent del botón.
    assert "exportBtn.textContent =" not in javascript
    assert "if (exportBtnLabel) exportBtnLabel.textContent" in javascript
    assert ".job-action {" in css


def test_finished_jobs_do_not_keep_three_dead_transport_buttons() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "document.querySelector('.job-control-row')?.classList.toggle('hidden', terminal);" in javascript
    # Y "Nuevo trabajo" cierra el trabajo, para que la ruta no siga apuntando a él.
    assert "closeCurrentJob({ silent: true }).catch((error) => console.warn(error));" in javascript


def test_a_freshly_processed_page_is_never_covered_by_ghost_patches() -> None:
    """Regresión real: los globos aparecían vacíos en la vista «Actual».

    El cliente encoge `bbox` al área de texto para editar, mientras `source_bbox` sigue
    siendo la caja del globo. Comparar una con otra daba "movida" en TODAS las regiones de
    un trabajo recién procesado, y el parche tapaba cada globo con el fondo limpio.
    El sello `rasterized_bbox` guarda la caja tal como el servidor la dibujó.
    """
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "rasterized_bbox: [...bbox]," in javascript
    assert "const source = normalizeBox(region.rasterized_bbox || region.bbox);" in javascript
    # Comparar contra source_bbox es justamente lo que causaba el fallo.
    assert "normalizeBox(region.source_bbox || region.bbox)" not in javascript
    # Y el sello se renueva con lo que el servidor acaba de rasterizar.
    assert "region.rasterized_bbox = normalizeBox(enviada);" in javascript
