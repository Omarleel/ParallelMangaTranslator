# Ground truth para evaluación

Crea un JSON por página con cajas y textos revisados manualmente. El evaluador compara estas anotaciones contra `Limpieza/Transcripción.json` y `Traduccion/Traducción.json`.

Ejemplo:

```json
{
  "page": "0001.png",
  "regions": [
    {
      "bbox": [10, 20, 180, 90],
      "type": "dialogue",
      "text_ja": "行くぞ",
      "translation_es": "¡Vamos!"
    }
  ]
}
```

`bbox` usa formato `[x, y, w, h]`. También se aceptan coordenadas `[[x1, y1], [x2, y2]]`.
