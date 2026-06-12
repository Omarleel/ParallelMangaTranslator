# Diccionarios de onomatopeyas

Los diccionarios están separados por idioma para que se puedan ampliar sin tocar el código.
Cada carpeta usa el código corto del idioma y contiene un archivo `onomatopoeias.yaml`.

Estructura esperada:

```yaml
language: Japonés
aliases:
  - ja
  - Japanese
entries:
  - key: impact
    target: ドン!
    sources:
      - ドン
      - ドーン
      - ズドン
```

Campos:

- `language`: nombre canónico que usa la aplicación, por ejemplo `Japonés` o `Español`.
- `aliases`: nombres o códigos alternativos aceptados por el loader.
- `entries`: lista de onomatopeyas.
- `key`: clave semántica compartida entre idiomas, como `impact`, `hit`, `slash` o `heartbeat`.
- `target`: forma recomendada al traducir/renderizar hacia ese idioma.
- `sources`: variantes que pueden aparecer en OCR o texto de entrada.

Para agregar un idioma nuevo, crea una carpeta nueva, por ejemplo `de/onomatopoeias.yaml`, y usa las mismas claves semánticas.
