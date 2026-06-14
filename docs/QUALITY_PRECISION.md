# Precisión y calidad visual

PMT mantiene la arquitectura `YOLO globo -> OCR guard -> clean_mask -> inpaint -> render`.

## Detección fina de texto

El OCR de localización ya no se usa solo como `bbox` rectangular. Cuando el motor OCR devuelve polígonos, PMT construye una `text_mask` fina por región:

- `region.mask`: zona segura del globo o región.
- `region.text_mask`: semilla fina de texto basada en polígonos OCR.
- `region.clean_mask`: tinta real a borrar.

Configuración:

```yaml
quality:
  fine_text_detection: true
  fine_text_mask_dilate: 2
```

## Máscara de tinta refinada

La máscara de limpieza se refina con componentes conectados anclados a `text_mask`/`text_bbox`. Esto reduce dos errores comunes:

- limpiar todo el globo cuando solo se debe borrar tinta;
- borrar tramas o ruido lejos del texto.

Configuración:

```yaml
quality:
  ink_mask_refinement: true
  ink_mask_min_component_area: 3
  ink_mask_component_anchor_overlap: 0.03
  ink_mask_component_anchor_max_gap_ratio: 0.45
```

Los valores recomendados son conservadores. Sube `ink_mask_min_component_area` si borra puntos de trama; baja `fine_text_mask_dilate` si come demasiado fondo alrededor de letras.

## Orden por panel

Antes de ordenar globos/textos, PMT puede detectar paneles con OpenCV y asignar cada región a un panel. Si la detección falla, vuelve automáticamente al orden base de PMT.

```yaml
quality:
  panel_aware_reading_order: true
  panel_detection_min_area_ratio: 0.015
  panel_detection_max_area_ratio: 0.96
  panel_detection_gutter_px: 10
```

En japonés, los paneles de una misma fila se leen de derecha a izquierda; dentro de cada panel se mantiene el `ReadingOrderResolver` existente.

## Render tipográfico

El renderer mantiene el recorte por `clip_mask`, pero añade mejoras de wrapping:

- cortes suaves para palabras latinas largas;
- balance de líneas para evitar líneas huérfanas muy cortas;
- factor de espaciado configurable.

```yaml
quality:
  typography_smart_wrap: true
  typography_hyphenation: true
  typography_balance_lines: true
  typography_line_spacing_factor: 1.0
```

Para debug visual, desactiva `typography_balance_lines` si quieres ver el wrap greedily original.
