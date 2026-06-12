# Motores OCR: localización vs transcripción

El proyecto usa dos responsabilidades OCR distintas. Están separadas a propósito para que puedas combinar motores distintos sin tocar el pipeline principal.

1. **OCR de localización (`ocr.detection_engine`)**
   - Detecta bounding boxes de texto en la página completa.
   - Se usa para limpieza, máscaras, pistas de texto, texto libre/SFX y división de globos cercanos.
   - No es la lectura final.

2. **OCR de transcripción (`ocr.transcription_engine`)**
   - Lee el texto dentro de cada región ya detectada/recortada.
   - Se usa para producir el texto que se traduce.

## Configuración

```yaml
ocr:
  detection_engine: easyocr        # auto | easyocr | paddle | paddle_subprocess
  transcription_engine: mangaocr   # auto | easyocr | mangaocr | paddle | paddle_subprocess
  gpu: false
  paddle_subprocess: auto
  fast_mode: false
```

## Ejemplos de uso

### EasyOCR para cajas + MangaOCR para lectura

```yaml
ocr:
  detection_engine: easyocr
  transcription_engine: mangaocr
  gpu: false
```

### PaddleOCR para cajas + MangaOCR para lectura

```yaml
ocr:
  detection_engine: paddle
  transcription_engine: mangaocr
  gpu: false
```

### PaddleOCR aislado para cajas y lectura

```yaml
ocr:
  detection_engine: paddle_subprocess
  transcription_engine: paddle_subprocess
  gpu: false
  paddle_subprocess: true
```

## Motores incluidos

### EasyOCR

- Sirve para **detección** y **transcripción**.
- Detector: `ocr/text_detection/easyocr_detector.py`
- Transcriptor: `ocr/engines/easyocr_engine.py`
- Cliente compartido: `ocr/easyocr_adapter.py`

EasyOCR devuelve cajas y texto. Por eso puede participar en las dos fases.

### MangaOCR

- Sirve solo para **transcripción**.
- Implementación: `ocr/engines/manga_ocr_engine.py`

`mangaocr` no devuelve bounding boxes. Por eso, si se coloca como `detection_engine`, el factory lo redirige a `easyocr` para mantener un localizador válido.

### PaddleOCR

- Sirve para **detección** y **transcripción**.
- Detector directo: `ocr/text_detection/paddle_detector.py`
- Detector subproceso: `ocr/text_detection/paddle_subprocess_detector.py`
- Transcriptor directo: `ocr/engines/paddle_ocr_engine.py`
- Transcriptor subproceso: `ocr/engines/paddle_subprocess_engine.py`
- Cliente compartido: `ocr/paddle_adapter.py`
- Normalización compartida: `ocr/paddle_result.py`

La opción `paddle_subprocess` permite aislar PaddleOCR en un worker separado. Esto evita conflictos de dependencias o problemas de GPU dentro del proceso principal.

## Estructura interna refactorizada

La implementación de OCR evita duplicación entre localización y transcripción:

- `ocr/settings.py`: settings, idiomas de EasyOCR/PaddleOCR y regla única para `paddle_subprocess`.
- `ocr/engine_registry.py`: registry compartido para alias, motores soportados y fallback a worker de PaddleOCR.
- `ocr/easyocr_adapter.py`: inicialización perezosa compartida de EasyOCR.
- `ocr/paddle_adapter.py`: cliente único para PaddleOCR directo o en subproceso.
- `ocr/paddle_result.py`: normalización común de salidas PaddleOCR 2.x/3.x.

Para agregar un motor nuevo, registra un adaptador en el factory correspondiente. Solo necesitas tocar `ocr/engines/factory.py` si es un motor de transcripción, o `ocr/text_detection/factory.py` si es un motor de localización.

---

# Cómo implementar un OCR nuevo

Hay tres casos:

1. Un OCR que solo **transcribe** texto.
2. Un OCR que solo **detecta cajas** de texto.
3. Un OCR que hace ambas cosas.

La regla general es no modificar el pipeline principal. Se añade una clase nueva, se registra en su factory y se activa desde `config.yaml`.

## 1. Implementar un OCR de transcripción

Usa este camino cuando la librería recibe un recorte de imagen y devuelve texto.

Crea un archivo nuevo en:

```text
parallel_manga_translator/ocr/engines/my_ocr_engine.py
```

Plantilla mínima:

```python
from __future__ import annotations

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.engines.base import OcrEngineBase, OcrEngineSettings

logger = get_logger(__name__)


class MyOcrEngine(OcrEngineBase):
    def __init__(self, settings: OcrEngineSettings) -> None:
        super().__init__(settings)
        self._client = None

    @property
    def engine_id(self) -> str:
        return "myocr"

    def _client_instance(self):
        if self._client is None:
            # Import lazy: evita romper el proyecto si la dependencia opcional
            # no está instalada hasta que el usuario active este motor.
            from my_ocr_library import MyOcr  # type: ignore

            self._client = MyOcr(
                language=self.settings.language,
                gpu=self.settings.gpu,
            )
        return self._client

    def extract_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""

        image = self.upscale_if_needed(image)

        try:
            result = self._client_instance().read(image)
        except Exception as exc:
            logger.warning("MyOCR falló transcribiendo texto: %s", exc)
            return ""

        # Caso 1: la librería devuelve directamente un string.
        if isinstance(result, str):
            return self.normalize_text(result)

        # Caso 2: la librería devuelve líneas con texto, confianza y caja.
        lines = []
        for item in result or []:
            text = item.get("text", "")
            confidence = float(item.get("confidence", 1.0))
            box = item.get("box", [])
            if text:
                lines.append({"text": text, "confidence": confidence, "box": box})

        return self.join_ocr_lines(lines)
```

Después registra el motor en:

```text
parallel_manga_translator/ocr/engines/factory.py
```

Añade el import:

```python
from parallel_manga_translator.ocr.engines.my_ocr_engine import MyOcrEngine
```

Y registra el nombre:

```python
OcrFactory.register("myocr", MyOcrEngine)
```

Configúralo así:

```yaml
ocr:
  transcription_engine: myocr
```

### Contrato que debe cumplir

```python
extract_text(image: np.ndarray) -> str
```

- Entrada: imagen `numpy.ndarray`, normalmente en formato BGR porque viene de OpenCV.
- Salida: texto normalizado como `str`.
- Si falla o no encuentra texto, devuelve `""`.
- No debe lanzar errores hacia el pipeline salvo que sea un fallo de programación.

## 2. Implementar un OCR de detección de cajas

Usa este camino cuando la librería recibe una página completa y devuelve coordenadas de texto.

Crea un archivo nuevo en:

```text
parallel_manga_translator/ocr/text_detection/my_ocr_detector.py
```

Plantilla mínima:

```python
from __future__ import annotations

from typing import List

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.text_detection.base import (
    TextDetection,
    TextDetectionEngineBase,
    TextDetectionSettings,
)

logger = get_logger(__name__)


class MyOcrTextDetector(TextDetectionEngineBase):
    def __init__(self, settings: TextDetectionSettings) -> None:
        super().__init__(settings)
        self._client = None

    @property
    def engine_id(self) -> str:
        return "myocr"

    def _client_instance(self):
        if self._client is None:
            from my_ocr_library import MyOcrDetector  # type: ignore

            self._client = MyOcrDetector(
                language=self.settings.language,
                gpu=self.settings.gpu,
            )
        return self._client

    def detect_text_boxes(self, image: np.ndarray) -> List[TextDetection]:
        if image is None or image.size == 0:
            return []

        try:
            results = self._client_instance().detect(image)
        except Exception as exc:
            logger.warning("MyOCR falló detectando cajas de texto: %s", exc)
            return []

        detections: List[TextDetection] = []

        for item in results or []:
            # Ajusta estas claves al formato real de tu librería.
            box = item.get("box")
            text = item.get("text", "")
            confidence = item.get("confidence", 0.0)

            detection = self.normalize_detection(box, text, confidence)
            if detection is not None:
                detections.append(detection)

        return detections
```

Después registra el detector en:

```text
parallel_manga_translator/ocr/text_detection/factory.py
```

Añade el import:

```python
from parallel_manga_translator.ocr.text_detection.my_ocr_detector import MyOcrTextDetector
```

Y registra el nombre:

```python
TextDetectionFactory.register("myocr", MyOcrTextDetector)
```

Configúralo así:

```yaml
ocr:
  detection_engine: myocr
```

### Contrato que debe cumplir

```python
detect_text_boxes(image: np.ndarray) -> List[TextDetection]
```

`TextDetection` tiene este formato:

```python
(
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "texto opcional",
    confianza_float,
)
```

- Las coordenadas deben estar en píxeles de la imagen original que recibe el detector.
- Si haces resize interno, escala las cajas de vuelta antes de devolverlas.
- Si la librería solo devuelve rectángulos `(x, y, w, h)`, conviértelos a cuatro puntos.
- Si no hay texto, devuelve `[]`.

Ejemplo para convertir `(x, y, w, h)` a cuatro puntos:

```python
box = [
    [x, y],
    [x + w, y],
    [x + w, y + h],
    [x, y + h],
]
```

## 3. Implementar un OCR que detecta y transcribe

Si la librería sirve para ambas fases, evita duplicar inicialización. Crea un adaptador compartido:

```text
parallel_manga_translator/ocr/myocr_adapter.py
```

Ejemplo:

```python
from __future__ import annotations

import numpy as np

from parallel_manga_translator.ocr.settings import OcrSettings


class MyOcrAdapter:
    def __init__(self, settings: OcrSettings) -> None:
        self.settings = settings
        self._client = None

    def client(self):
        if self._client is None:
            from my_ocr_library import MyOcr  # type: ignore

            self._client = MyOcr(
                language=self.settings.language,
                gpu=self.settings.gpu,
            )
        return self._client

    def read(self, image: np.ndarray):
        return self.client().read(image)

    def detect(self, image: np.ndarray):
        return self.client().detect(image)
```

Luego crea dos wrappers pequeños:

```text
parallel_manga_translator/ocr/engines/my_ocr_engine.py
parallel_manga_translator/ocr/text_detection/my_ocr_detector.py
```

El transcriptor usa `adapter.read(image)` y el detector usa `adapter.detect(image)`. Así se comparte:

- import lazy;
- inicialización del cliente;
- settings;
- manejo de dependencias opcionales;
- cualquier conversión común de formato.

Configúralo en ambas fases:

```yaml
ocr:
  detection_engine: myocr
  transcription_engine: myocr
```

## 4. Agregar alias opcionales

Si quieres permitir nombres alternativos, edita:

```text
parallel_manga_translator/ocr/engine_registry.py
```

Ejemplo:

```python
ENGINE_ALIASES = {
    ...
    "my_ocr": "myocr",
    "my-ocr": "myocr",
}
```

Con eso estas configuraciones apuntan al mismo motor:

```yaml
ocr:
  transcription_engine: myocr
```

```yaml
ocr:
  transcription_engine: my_ocr
```

```yaml
ocr:
  transcription_engine: my-ocr
```

## 5. Agregar dependencias opcionales

Si el OCR nuevo requiere una librería extra, agrégala como dependencia opcional cuando sea posible. Evita imports globales en los módulos del motor.

Preferido:

```python
def _client_instance(self):
    if self._client is None:
        from my_ocr_library import MyOcr  # type: ignore
        self._client = MyOcr()
    return self._client
```

Evita:

```python
from my_ocr_library import MyOcr
```

La razón es que un usuario que no use ese motor no debería necesitar instalar esa dependencia para arrancar el proyecto.

## 6. Probar un OCR nuevo

Agrega pruebas pequeñas que no descarguen modelos. Lo ideal es mockear la librería externa.

Ejemplo de prueba para el registry/factory:

```python
from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.ocr.engines.factory import OcrFactory


def test_myocr_is_registered():
    config = OcrConfig(transcription_engine="myocr")
    engine = OcrFactory.create("Japonés", config)
    assert engine.engine_id == "myocr"
```

Ejemplo de prueba para normalización de detecciones:

```python
from parallel_manga_translator.ocr.text_detection.base import TextDetectionEngineBase


def test_normalize_detection_accepts_four_points():
    detection = TextDetectionEngineBase.normalize_detection(
        [[1, 2], [3, 2], [3, 4], [1, 4]],
        "text",
        0.9,
    )
    assert detection == ([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]], "text", 0.9)
```

## 7. Checklist para un motor nuevo

### Para transcripción

- [ ] Crear `ocr/engines/my_ocr_engine.py`.
- [ ] Heredar de `OcrEngineBase`.
- [ ] Implementar `engine_id`.
- [ ] Implementar `extract_text(image) -> str`.
- [ ] Usar `upscale_if_needed(image)` si trabaja sobre recortes pequeños.
- [ ] Normalizar salida con `normalize_text(...)` o `join_ocr_lines(...)`.
- [ ] Registrar en `ocr/engines/factory.py`.
- [ ] Activar con `ocr.transcription_engine: myocr`.

### Para detección

- [ ] Crear `ocr/text_detection/my_ocr_detector.py`.
- [ ] Heredar de `TextDetectionEngineBase`.
- [ ] Implementar `engine_id`.
- [ ] Implementar `detect_text_boxes(image) -> List[TextDetection]`.
- [ ] Devolver cajas como cuatro puntos.
- [ ] Mantener coordenadas en escala original.
- [ ] Registrar en `ocr/text_detection/factory.py`.
- [ ] Activar con `ocr.detection_engine: myocr`.

### Para un motor dual

- [ ] Crear `ocr/myocr_adapter.py`.
- [ ] Poner ahí la inicialización pesada.
- [ ] Crear un wrapper de transcripción.
- [ ] Crear un wrapper de detección.
- [ ] Registrar ambos wrappers.
- [ ] Usar el mismo nombre si quieres configurar ambas fases con `myocr`.

## 8. Qué no hacer

- No agregues `if engine == "myocr"` dentro del pipeline principal.
- No dupliques settings si ya existen en `OcrSettings`.
- No hagas imports pesados arriba del archivo si el motor es opcional.
- No devuelvas cajas en una escala distinta a la imagen original.
- No uses MangaOCR como detector: no entrega bounding boxes.
