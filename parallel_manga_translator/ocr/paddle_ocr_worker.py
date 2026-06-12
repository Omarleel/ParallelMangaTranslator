from __future__ import annotations

"""Worker aislado para PaddleOCR.

Este módulo se ejecuta con `python -m parallel_manga_translator.ocr.paddle_ocr_worker` desde el proceso
principal. La razón es práctica: en Windows/Colab, PaddlePaddle GPU y PyTorch/YOLO
pueden chocar si ambos se importan dentro del mismo intérprete. Al aislar PaddleOCR
en este worker, el proceso principal puede usar YOLO/Ultralytics con GPU y este
worker puede usar PaddleOCR CPU/GPU sin registrar CUDA en el mismo proceso.
"""

import argparse
import base64
import json
import os
import sys
import traceback

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

from parallel_manga_translator.ocr.paddle_result import normalize_paddle_result


def _decode_image(payload: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(payload.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen enviada al worker PaddleOCR")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    use_gpu = str(args.gpu).strip().lower() in {"1", "true", "yes", "on"}

    try:
        from paddleocr import PaddleOCR  # type: ignore

        try:
            ocr = PaddleOCR(use_angle_cls=True, lang=args.lang, use_gpu=use_gpu, show_log=False)
        except TypeError:
            # Compatibilidad básica con versiones nuevas que cambien argumentos.
            ocr = PaddleOCR(lang=args.lang)
    except Exception as exc:
        print(json.dumps({"type": "fatal", "error": f"No se pudo iniciar PaddleOCR: {exc}", "traceback": traceback.format_exc()}, ensure_ascii=False), flush=True)
        return 2

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            if request.get("cmd") == "shutdown":
                print(json.dumps({"id": request_id, "ok": True, "shutdown": True}, ensure_ascii=False), flush=True)
                return 0
            image = _decode_image(request["image"])
            try:
                result = ocr.ocr(img=image, cls=True)
            except TypeError:
                result = ocr.ocr(image)
            lines = normalize_paddle_result(result)
            print(json.dumps({"id": request_id, "ok": True, "lines": lines}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"id": None, "ok": False, "error": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
