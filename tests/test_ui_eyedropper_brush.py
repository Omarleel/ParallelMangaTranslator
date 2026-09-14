"""Cuentagotas y pincel de pintar, para tapar restos de tinta a mano.

El inpaint reconstruye el fondo, que es lo correcto sobre tramas, pero dentro de un globo
plano a veces lo que hace falta es taparlo con el color del propio globo. Eso es este par:
el cuentagotas toma el color de la página y el pincel lo pinta.

El color viaja **con el trazo**: cambiar el selector después no puede repintar lo ya hecho.
Y se guarda en RGB por todo el recorrido —es lo que manda el navegador—; solo se voltea a
BGR al escribir píxeles, en `_apply_brush_strokes`, que es el único sitio que habla OpenCV.
"""

from __future__ import annotations

from pathlib import Path

from parallel_manga_translator.ui.manual_renderer import PAINT_MODES, BrushStroke, brush_stroke_to_dict, parse_brush_strokes

STATIC = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "static"


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_la_ui_ofrece_la_herramienta_y_el_color() -> None:
    html = _html()

    assert 'data-tool="eyedropper"' in html, "falta el cuentagotas en el dock"
    assert '<option value="paint">' in html, "falta el modo de pincel que pinta"
    assert 'id="brushColor"' in html, "hace falta poder elegir el color a mano también"


def test_el_boton_del_dock_esta_enganchado() -> None:
    """Un botón en el dock sin su listener se ve perfecto y no hace nada.

    Las herramientas no se enganchan por `querySelectorAll`, sino por una lista escrita a
    mano, así que añadir el botón al HTML no basta: hay que estar en esa lista.
    """
    javascript = _js()
    listas = [linea for linea in javascript.splitlines() if "dockBrushTool" in linea and "dockPanTool" in linea]

    assert listas, "no encuentro las listas de botones de herramienta"
    for linea in listas:
        assert "dockEyedropperTool" in linea, f"el cuentagotas falta en: {linea.strip()[:80]}"


def test_el_cuentagotas_toma_el_color_y_pasa_a_pintar() -> None:
    javascript = _js()

    assert "function sampleImageColor(" in javascript
    assert "state.tool = ['select', 'region', 'brush', 'eyedropper', 'pan']" in javascript
    # Tomar un color es el paso previo a pintar, nunca el objetivo.
    assert "brushMode.value = 'paint'" in javascript
    assert "setTool('brush')" in javascript
    # Alt+clic con el pincel hace lo mismo, que es como funciona en cualquier editor.
    assert "event.altKey" in javascript


def test_el_trazo_se_lleva_su_color() -> None:
    javascript = _js()

    assert "if (isPaintMode(modoPincel)) state.drawingStroke.color = currentPaintColor();" in javascript
    # Y sobrevive a deshacer/rehacer.
    assert "Array.isArray(stroke.color) ? { color: stroke.color.map(Number) } : {}" in javascript


def test_el_color_viaja_en_la_peticion_al_servidor() -> None:
    """El fallo real del primer intento, y por qué el pincel "no pintaba nada".

    El payload construía cada trazo con una lista blanca de cuatro campos y tiraba el
    color. El cuentagotas lo tomaba bien, el trazo lo llevaba en el cliente y sobrevivía a
    deshacer; el POST lo perdía. El servidor recibía `paint` sin color y, por su regla de
    «sin color no se pinta», no hacía nada: parecía roto sin estarlo.

    Que `cloneBrushStrokes` lo conserve NO basta: son dos serializadores distintos y el
    que decide si algo llega al disco es este.
    """
    javascript = _js()
    inicio = javascript.index("brush_strokes: state.brushStrokes.map(")
    bloque = javascript[inicio:inicio + 800]

    assert "stroke.color" in bloque, "el payload que se envía al servidor no manda el color"


def test_el_servidor_entiende_el_color_y_lo_acota() -> None:
    trazos = parse_brush_strokes(
        [
            {"points": [[5, 5]], "radius": 4, "mode": "paint", "color": [300, -20, 128.6]},
            {"points": [[6, 6]], "radius": 4, "mode": "paint", "color": "rojo"},
            {"points": [[7, 7]], "radius": 4, "mode": "restore_clean"},
        ],
        50,
        50,
    )

    assert trazos[0].color == (255, 0, 129), "el color se acota a 0-255, no se rechaza el trazo"
    assert trazos[1].color is None, "un color ilegible no puede convertirse en uno cualquiera"
    assert trazos[2].color is None


def test_el_color_solo_se_guarda_cuando_lo_hay() -> None:
    """Emitirlo como `null` en todos los trazos cambiaría la forma de lo ya guardado."""
    pintado = brush_stroke_to_dict(BrushStroke(points=[(1, 1)], mode="paint", color=(10, 20, 30)))
    normal = brush_stroke_to_dict(BrushStroke(points=[(1, 1)], mode="restore_clean"))

    assert pintado["color"] == [10, 20, 30]
    assert "color" not in normal
    assert set(normal) == {"points", "radius", "mode", "applied"}


def test_los_modos_que_pintan_estan_declarados_en_un_solo_sitio() -> None:
    assert "paint" in PAINT_MODES
