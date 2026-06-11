# ParallelMangaTranslator

ParallelMangaTranslator es una herramienta diseñada para simplificar y optimizar el proceso de limpieza y traducción de mangas, especialmente para los aficionados del manga y los grupos de scanlation. Utilizando tecnología OCR (Optical Character Recognition), esta herramienta detecta automáticamente el texto dentro de los mangas escaneados y proporciona funcionalidades para traducir y limpiar las páginas usando GPU CUDA de forma paralela, en ese sentido, ParallelMangaTranslator es una mejora del programa que desarrollé con anterioridad: [MangaTranslate](https://github.com/Omarleel/MangaTranslate).

## Características

- **Procesamiento por Carpetas:** Permite procesar múltiples imágenes al seleccionar o especificar manualmente la ruta de la carpeta que contiene las imágenes. Admite formatos como .jpg, .png, .jpeg, .bmp y .webp. Además, puede descargar y descomprimir automáticamente archivos .zip desde Google Drive. 
- **Detección Automática de Texto:** Utiliza algoritmos OCR avanzados para identificar texto en las páginas de manga, independientemente del estilo de dibujo o letra.
- **Limpieza de Páginas:** Ofrece una herramienta para eliminar los textos de las páginas, facilitando la lectura y la posterior traducción.
- **Camuflaje Avanzado:** Permite camuflar los textos en fondos de páginas a color, adaptándose incluso a fondos irregulares.
- **Traducción Precisa con Google:** Utiliza GoogleTranslator de deep_translator para traducir el manga a varios idiomas, incluyendo japonés, inglés, español, coreano y chino, manteniendo la fidelidad del texto original.
- **Almacenamiento y Organización de Imágenes Procesadas:** Todas las imágenes procesadas se almacenan en la carpeta "Outputs" y se organizan en subcarpetas según la acción realizada sobre ellas, ya sea "Limpieza" o "Traducción".
- **Exportación de Datos en Formato JSON:** Genera archivos .json estructurados que contienen las transcripciones y traducciones de las páginas procesadas, facilitando su análisis o integración con otras aplicaciones.
- **Traducción Contextual con LLM (Nuevo):** Además de la traducción tradicional (Google Translator), ahora integra **Modelos de Lenguaje (LLMs)** a través de Groq. Esto permite traducciones coherentes que entienden el contexto, el *lore* del manga, el tono de los personajes y la jerga, superando las limitaciones de la traducción literal.

##  Requerimientos

Antes de utilizar ParallelMangaTranslator, asegúrate de tener instalados los siguientes requisitos o instalarlos desde requirements.txt:
- Python 3.10.10
- **CUDA**: Esencial para el funcionamiento de algunas bibliotecas OCR que requieren procesamiento en GPU.
- OpenCV: Una biblioteca de procesamiento de imágenes y visión por computadora.
- EasyOCR: Una biblioteca para el reconocimiento óptico de caracteres (OCR) fácil de usar.
- Manga-OCR: Una biblioteca para el reconocimiento óptico de caracteres (OCR) especializada en mangas.
- PaddleOCR: OCR opcional. Si se usa junto a YOLO/PyTorch GPU, se ejecuta en un subproceso aislado para evitar conflictos CUDA.
- Deep_Translator: Una biblioteca flexible, gratuita e ilimitada para traducir entre diferentes idiomas de forma sencilla utilizando varios traductores.
- Pillow: Una biblioteca para manipulación de imágenes en Python.
- pydrive2: Una biblioteca de Python que envuelve la API de Google Drive, facilitando las operaciones de carga y descarga de archivos.

## Instalación de CUDA compatible con Torch
Ejecuta los siguientes comandos:
```bash
# Desinstala cualquier versión de Torch que tengas
pip uninstall torch torchvision torchaudio
# Limpia la caché de pip para evitar conflictos
pip cache purge
# Instala la versión específica de Torch compatible con CUDA 12.4:
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

## Contribuciones

Si deseas contribuir al desarrollo de ParallelMangaTranslator, ¡no dudes en hacerlo! Puedes enviar pull requests o reportar problemas en el repositorio del proyecto.

## Pruebas
Para correr el programa, puedes ejecutar el siguiente conjunto de comandos:
```bash
# Clonar el repositorio y acceder a la carpeta del programa
git clone https://github.com/Omarleel/ParallelMangaTranslator
# Accede al proyecto
cd ParallelMangaTranslator
# Instala los requerimientos
pip install -r requirements.txt
# Ejecutar el script
py ParallelMangaTranslator.py
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omarleel/ParallelMangaTranslator/blob/main/ParallelMangaTranslator.ipynb)
## Mejoras profesionales incluidas

Esta versión usa **solo modelos preentrenados** para detectar globos de texto mediante YOLO/Ultralytics. El objetivo principal es limpiar y renderizar sobre la **máscara del globo**, no solo sobre la caja OCR de las letras. La detección heurística de globos fue eliminada: si el modelo profesional no está instalado, no se puede descargar o no existe la ruta indicada, el programa lanza error en vez de inventar globos por reglas OpenCV.

### Uso recomendado

```bash
python ParallelMangaTranslator.py
```

Para máxima velocidad durante pruebas:

```bash
PMT_INPAINT_MODE=bubble_only PMT_SKIP_PDF=1 PMT_CACHE=1 python ParallelMangaTranslator.py
```

Para calidad equilibrada:

```bash
PMT_BUBBLE_DETECTOR=professional PMT_INPAINT_MODE=auto PMT_CACHE=1 python ParallelMangaTranslator.py
```

El modelo profesional es obligatorio por defecto. Para comprobarlo explícitamente:

```bash
PMT_REQUIRE_PROFESSIONAL_BUBBLE=1 python ParallelMangaTranslator.py
```

Para usar un modelo local de segmentación de globos:

```bash
PMT_BUBBLE_MODEL_PATH=/ruta/al/modelo/best.pt python ParallelMangaTranslator.py
```

Para conservar onomatopeyas originales:

```bash
PMT_ONOMATOPOEIA_MODE=keep python ParallelMangaTranslator.py
```

Para dejar onomatopeya original más traducción pequeña:

```bash
PMT_ONOMATOPOEIA_MODE=subtitle python ParallelMangaTranslator.py
```


### OCR y PaddleOCR aislado

Por defecto, el modo `auto` usa MangaOCR/EasyOCR sin cargar PaddleOCR en el proceso principal:

```bash
PMT_OCR_ENGINE=auto python ParallelMangaTranslator.py
```

Para usar PaddleOCR con GPU sin chocar con YOLO/PyTorch, usa el worker aislado:

```bash
PMT_OCR_ENGINE=paddle PMT_OCR_GPU=1 PMT_PADDLE_SUBPROCESS=1 python ParallelMangaTranslator.py
```

También puedes pedirlo directamente:

```bash
PMT_OCR_ENGINE=paddle_subprocess PMT_OCR_GPU=1 python ParallelMangaTranslator.py
```

Instala PaddleOCR solo si lo vas a usar:

```bash
pip install -r requirements-paddle-optional.txt
```

### Configuración YAML

Puedes copiar el ejemplo:

```bash
cp config.example.yaml config.yaml
```

Y modificar idiomas, modo de inpainting, exportación, caché y onomatopeyas desde ese archivo. Las variables `PMT_*` tienen prioridad sobre el YAML.

### Métricas

Después de procesar se genera:

```text
Dataset/Outputs/Metricas/reporte.json
```

Ahí puedes revisar duración, globos detectados, onomatopeyas, OCR vacío, traducciones vacías y páginas fallidas.

## Organización interna

El código mantiene las firmas públicas del pipeline, pero las responsabilidades transversales se separaron para que sea más fácil mantenerlo:

- `Applications/Environment.py`: lectura segura de variables de entorno y publicación de valores por defecto.
- `Applications/Geometry.py`: operaciones comunes sobre cajas `x, y, w, h`.
- `Applications/BubbleDetectorConfig.py`: parámetros del detector profesional y filtros de texto libre.
- `Applications/OnomatopoeiaRepository.py`: carga de diccionarios YAML de onomatopeyas; la detección/traducción sigue en `OnomatopoeiaManager`.
- `Applications/TextNormalization.py`: normalización de texto OCR/traducido antes de traducir o renderizar.

`config.yaml` es la fuente recomendada para la configuración no privada. `.env` debe reservarse para secretos como claves API y no debería versionarse.

### Evaluación real de precisión

Se añadió un evaluador para comparar el pipeline contra anotaciones manuales. Esto permite medir detección de regiones, OCR y traducción en vez de depender solo de métricas internas.

Estructura recomendada:

```text
DatasetEval/ground_truth/
  0001.json
  0002.json
```

Formato mínimo de cada página:

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

Ejecuta la evaluación después de procesar el manga:

```bash
python evaluate_manga.py \
  --ground-truth DatasetEval/ground_truth \
  --transcription-json Dataset/Outputs/Limpieza/Transcripción.json \
  --translation-json Dataset/Outputs/Traduccion/Traducción.json \
  --output Dataset/Outputs/Metricas/reporte_precision.json
```

El reporte incluye `detection_precision`, `detection_recall`, `detection_f1`, `mean_iou`, `mean_ocr_cer` y `mean_translation_cer`.

### JSON estricto para traducción LLM

La traducción LLM ahora valida localmente la respuesta antes de usarla. Cuando el proveedor lo permite, intenta `json_schema` estricto; si el proveedor no lo soporta, cae a `json_object` y mantiene validación local estricta.

Salida aceptada:

```json
{
  "traducciones": [
    {"id": 0, "traduccion": "texto traducido"}
  ]
}
```

No se aceptan campos extra, ids duplicados, ids faltantes ni tipos incorrectos. Puedes controlar el intento de schema estricto con:

```yaml
llm:
  strict_json_schema: true
```

### Memoria automática de personajes y hablantes

En modo `LLM`, el sistema construye automáticamente una memoria persistente con IA antes de traducir cada página. Usa el OCR, tipo de región, orden de lectura, contexto previo y memoria acumulada para inferir hablantes, estilos de habla y posibles aliases sin pedirlo manualmente.

Por defecto se guarda en:

```text
Dataset/character_memory.json
```

Configuración:

```yaml
character_memory:
  enabled: true
  path: ""
  max_context_pages: 8
```

La memoria se añade al prompt de traducción y también se exportan campos como `Hablante` y `Confianza hablante` en los JSON de transcripción/traducción cuando están disponibles. Si no hay cliente LLM configurado, el sistema no inventa personajes: usa asignaciones conservadoras como `unknown`, `narrator` o `sfx`.
