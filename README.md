# ParallelMangaTranslator

ParallelMangaTranslator es una herramienta diseñada para simplificar y optimizar el proceso de limpieza y traducción de mangas, especialmente para los aficionados del manga y los grupos de scanlation. Utilizando tecnología OCR (Optical Character Recognition), esta herramienta detecta automáticamente el texto dentro de los mangas escaneados y proporciona funcionalidades para traducir y limpiar las páginas usando GPU CUDA de forma paralela, en ese sentido, ParallelMangaTranslator es una mejora del programa que desarrollé con anterioridad: [MangaTranslate](https://github.com/Omarleel/MangaTranslate).

## Características

- **Procesamiento por Carpetas:** Permite procesar múltiples imágenes al seleccionar o especificar manualmente la ruta de la carpeta que contiene las imágenes. Admite formatos como .jpg, .png, .jpeg, .bmp y .webp. Además, puede descargar y descomprimir automáticamente archivos .zip desde Google Drive. 
- **Detección Automática de Texto:** Utiliza algoritmos OCR avanzados para identificar texto en las páginas de manga, independientemente del estilo de dibujo o letra.
- **Limpieza de Páginas:** Ofrece una herramienta para eliminar los textos de las páginas, facilitando la lectura y la posterior traducción.
- **Camuflaje Avanzado:** Permite camuflar los textos en fondos de páginas a color, adaptándose incluso a fondos irregulares.
- **Traducción Precisa con Google:** Utiliza GoogleTranslator de deep_translator para traducir el manga a varios idiomas, incluyendo japonés, inglés, español, coreano y chino, manteniendo la fidelidad del texto original.
- **Almacenamiento y Organización de Imágenes Procesadas:** Todas las imágenes procesadas se almacenan en la carpeta "outputs" y se organizan en subcarpetas según la acción realizada sobre ellas, ya sea "limpieza" o "traduccion".
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



## UI local de revisión humana

Además de la CLI, el proyecto incluye una interfaz web local para revisar páginas a medida que se procesan. Permite cargar una carpeta desde el navegador o un archivo `.zip`, ver la salida automática de traducción y limpieza, alternar entre original/limpieza/traducción/corregida, editar traducciones manualmente, mover o redimensionar regiones de texto y revertir la limpieza por región antes de guardar una nueva imagen corregida.

Instala las dependencias y lanza la UI:

```bash
pip install -r requirements.txt
python ParallelMangaTranslatorUI.py
# o, si instalaste el paquete en modo editable:
pmt-ui
```

Abre `http://127.0.0.1:7860`. Los trabajos se guardan por defecto en `.pmt_ui_jobs/<job_id>/outputs/` con estas carpetas:

```text
limpieza/       salida limpia automática
traduccion/     salida traducida automática
corregida/      imágenes guardadas desde el asistente corrector
correcciones/   JSON con textos, cajas y flags manuales
```

La UI procesa página por página en segundo plano para mejorar la experiencia: una página pendiente muestra un mensaje de espera, pero las páginas ya listas se pueden revisar y corregir inmediatamente. El botón **Exportar ZIP** descarga un paquete con `imagenes_finales/`, usando la versión corregida si existe y, si no, la traducción automática lista; también incluye `correcciones/` y `manifest_export.json` cuando corresponda. Usa la misma configuración funcional de `config.yaml`; puedes apuntar a otro archivo con `PMT_CONFIG=/ruta/config.yaml python ParallelMangaTranslatorUI.py`.

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
## Mejoras YOLOes incluidas

Esta versión usa **YOLO11-seg fine-tuned como detector principal** para globos/texto de manga mediante Ultralytics. La máscara del globo se conserva como **zona segura para OCR/renderizado** y se separa de `clean_mask`, que representa la **tinta/texto original a borrar**. Así una detección de globo no limpia todo el interior por accidente. La detección heurística de globos fue eliminada: si el modelo YOLO no está instalado, no se puede descargar o no existe la ruta indicada, el programa lanza error en vez de inventar globos por reglas OpenCV.

### Uso recomendado

```bash
python ParallelMangaTranslator.py
```

Para máxima velocidad durante pruebas, cambia estos valores en `config.yaml`:

```yaml
quality:
  inpaint_mode: bubble_only

export:
  skip_pdf: true

processing:
  cache: true
```

Para calidad equilibrada:

```yaml
quality:
  bubble_detector: yolo11-seg
  inpaint_mode: auto
  require_yolo: true
  bubble_retina_masks: true
```

La máscara del globo y la máscara de tinta están separadas: `region.mask` es la zona segura y `region.clean_mask` es lo que se borra. Por defecto, el borrado de tinta dentro de globos usa el modelo configurado en `translation.inpaint_model`, así que si eliges `lama_mpe` no se rellena con blanco/sólido salvo fallback por error o modelo sin API de máscara.

```yaml
translation:
  inpaint_model: lama_mpe  # auto | opencv-tela | lama_mpe | lama_large_512px | aot | B/N

quality:
  # inpaint = siempre usa translation.inpaint_model sobre clean_mask
  # auto = sólido en fondos uniformes, inpaint en fondos complejos
  # solid = siempre color sólido local
  bubble_fill_strategy: inpaint
  bubble_fill_background_std_threshold: 18.0  # solo aplica con auto
  bubble_fill_inpaint_padding: 18
```


Para usar tus pesos locales YOLO11-seg fine-tuned:

```yaml
quality:
  bubble_detector: yolo11-seg
  bubble_model_path: /ruta/al/modelo/yolo11s-manga-seg.pt
  # Opcional: limita por IDs de clase si tu modelo detecta más cosas.
  bubble_model_classes: "0,1,2"
  # Opcional: excluye clases que nunca deben limpiarse.
  bubble_exclude_labels: "ignore_art,panel,page,background"
```

Para usar pesos desde Hugging Face, conserva `bubble_model_repo` y `bubble_model_file`.

Para conservar onomatopeyas originales:

```yaml
onomatopoeia:
  mode: keep
```

Para dejar onomatopeya original más traducción pequeña:

```yaml
onomatopoeia:
  mode: subtitle
```

### OCR y PaddleOCR aislado

Por defecto, el modo `auto` usa MangaOCR/EasyOCR sin cargar PaddleOCR en el proceso principal:

```yaml
ocr:
  engine: auto
```

Para usar PaddleOCR con GPU sin chocar con YOLO/PyTorch, usa el worker aislado:

```yaml
ocr:
  engine: paddle
  gpu: true
  paddle_subprocess: true
```

También puedes pedirlo directamente:

```yaml
ocr:
  engine: paddle_subprocess
  gpu: true
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

Modifica idiomas, OCR, traducción, inpainting, exportación, caché y onomatopeyas desde `config.yaml`. El archivo `.env` queda reservado para secretos como `GROQ_API_KEY` y `DEEPL_API_KEY`.

### Métricas

Después de procesar se genera:

```text
dataset/outputs/metricas/reporte.json
```

Ahí puedes revisar duración, globos detectados, onomatopeyas, OCR vacío, traducciones vacías y páginas fallidas.


## Arquitectura abierta a cambio

OCR y traducción usan factories/registries:

```text
parallel_manga_translator/ocr/engines/factory.py
parallel_manga_translator/translation/providers/factory.py
```

Para agregar un motor nuevo, implementa el contrato correspondiente y regístralo en la factory. El pipeline principal no necesita cambios. Consulta `PLUGIN_ARCHITECTURE.md`.

## Organización interna

El código fue reorganizado como paquete Python con nombres `snake_case` y subpaquetes por responsabilidad. El script raíz `ParallelMangaTranslator.py` queda como wrapper mínimo y la lógica de arranque vive en `parallel_manga_translator/cli.py`.

```text
parallel_manga_translator/
  config/          configuración YAML, constantes y carga de secretos
  detection/       detección de globos y regiones YOLOes
  geometry/        utilidades geométricas para cajas y máscaras
  infrastructure/  logging, caché y manejo estructurado de errores
  inpainting/      adaptadores de modelos de limpieza/inpainting
  io/              archivos, descargas, exportación y utilidades
  language/        onomatopeyas y filtros de idioma fuente
  layout/          orden de lectura
  models/          modelos de datos compartidos
  ocr/             OCR y workers aislados de PaddleOCR
  processing/      orquestación de limpieza, traducción y procesamiento paralelo
  quality/         métricas y evaluación contra ground truth
  rendering/       renderizado de texto sobre páginas limpias
  translation/     traducción, glosario, memoria de personajes y normalización
```

Puedes seguir ejecutando el wrapper clásico:

```bash
python ParallelMangaTranslator.py
```

O instalar el proyecto en modo editable y usar la CLI:

```bash
pip install -e .
pmt
pmt-evaluate --help
```

`config.yaml` es la única fuente de configuración funcional. `.env` debe reservarse solo para secretos como claves API y no debería versionarse.

### Evaluación real de precisión

Se añadió un evaluador para comparar el pipeline contra anotaciones manuales. Esto permite medir detección de regiones, OCR y traducción en vez de depender solo de métricas internas.

Estructura recomendada:

```text
dataset_eval/ground_truth/
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
  --ground-truth dataset_eval/ground_truth \
  --transcription-json dataset/outputs/limpieza/Transcripción.json \
  --translation-json dataset/outputs/traduccion/Traducción.json \
  --output dataset/outputs/metricas/reporte_precision.json
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
dataset/character_memory.json
```

Configuración:

```yaml
character_memory:
  enabled: true
  path: ""
  max_context_pages: 8
```

La memoria se añade al prompt de traducción y también se exportan campos como `Hablante` y `Confianza hablante` en los JSON de transcripción/traducción cuando están disponibles. Si no hay cliente LLM configurado, el sistema no inventa personajes: usa asignaciones conservadoras como `unknown`, `narrator` o `sfx`.


## Refactor SOLID / Clean Code

La arquitectura fue dividida en fachadas pequeñas y módulos por responsabilidad. Ver `SOLID_REFACTOR.md` para el mapa completo de responsabilidades y los contratos en `parallel_manga_translator/architecture/ports.py`.

### Precisión visual avanzada

Esta versión separa explícitamente `region.mask` (zona segura del globo) de `region.clean_mask` (tinta real a borrar):

- detección fina de texto con polígonos OCR (`text_mask`);
- refinamiento de tinta por componentes conectados;
- orden de lectura sensible a paneles;
- render tipográfico con cortes suaves y balance de líneas.

Ver `docs/QUALITY_PRECISION.md` para ajustar `fine_text_detection`, `ink_mask_refinement`, `panel_aware_reading_order` y opciones tipográficas.

## UI de corrección humana

La interfaz local se ejecuta con:

```bash
python ParallelMangaTranslatorUI.py
```

Abre `http://127.0.0.1:7860` y configura el trabajo antes de procesar:

- carpeta de imágenes o archivo ZIP;
- idioma de entrada y salida;
- traductor: Google/tradicional o LLM;
- opciones avanzadas de OCR para detección/localización y transcripción.

La UI tiene dos vistas separadas: primero **Configuración** y después **Trabajo/Revisión**. Durante la revisión puedes avanzar por las páginas ya listas mientras el resto se procesa en segundo plano. En cada página hay herramientas para:

- editar el texto traducido **directamente dentro de la región**, en tiempo real y con la tipografía/tamaño de vista previa;
- elegir por región entre **tamaño de fuente automático** o **tamaño manual fijo** con slider/número;
- mover o redimensionar regiones detectadas o manuales con previsualización del texto en tiempo real;
- eliminar regiones detectadas o manuales con borrado limpio;
- usar atajos sobre regiones: `Supr`, `Ctrl+C`, `Ctrl+X` y `Ctrl+V`;
- hacer zoom con el control manual, los botones `+`/`-`, **Ajustar** o **Ctrl + rueda del ratón** sobre la página;
- revertir limpieza por región;
- crear nuevas regiones manualmente;
- ejecutar OCR + traducción sobre una región creada o seleccionada;
- usar un pincel simplificado con tres acciones: **Limpiar texto**, **Inpaint** y **Restaurar / borrar máscara**; el cursor circular muestra en tiempo real el tamaño exacto del pincel antes de pintar;
- activar **Enfoque** para ocultar paneles y ampliar la página del manga;
- exportar un ZIP final.

Los cambios ligeros se guardan solos en `outputs/corregida/` y `outputs/correcciones/`. La escritura dentro del cuadro es instantánea y solo dispara autoguardado después de una pausa, sin esperar al backend para cada tecla. **Aplicar inpaint** actúa sobre la imagen actual sin volver a redibujar regiones, para no recalcular tamaños de fuente. Las regiones con tamaño manual mantienen el valor fijado aunque cambie el área de la caja. Antes de aplicar inpaint se guarda una copia interna de la página actual, de modo que **Restaurar / borrar máscara** pueda recuperar zonas inpainted si te pasas con la máscara. Los botones explícitos quedan reservados para tareas pesadas o destructivas: **OCR + traducir región**, **Aplicar inpaint**, **Restaurar automático** y **Exportar ZIP**. El ZIP final usa la versión corregida cuando existe, o la traducción automática si la página no fue editada.
