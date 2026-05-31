from __future__ import annotations

"""Worker aislado para PaddleOCR.

Este módulo se ejecuta con `python -m Applications.PaddleOcrWorker` desde el proceso
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
from typing import Any, Dict, List

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")


def _decode_image(payload: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(payload.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen enviada al worker PaddleOCR")
    return image


def _line_to_json(line) -> Dict[str, Any]:
    try:
        box = line[0]
        text, conf = line[-1]
        return {
            "box": [[float(p[0]), float(p[1])] for p in box],
            "text": str(text),
            "confidence": float(conf),
        }
    except Exception:
        return {"box": [], "text": "", "confidence": 0.0}


def _normalize_paddle_result(result) -> List[Dict[str, Any]]:
    if not result:
        return []

    # PaddleOCR 2.x suele devolver: [ [ [box, (text, conf)], ... ] ]
    if isinstance(result, list):
        lines = result[0] if result and isinstance(result[0], list) else result
        return [_line_to_json(line) for line in lines if line]

    # PaddleOCR 3.x puede devolver estructuras diferentes. Intentamos extraer campos comunes.
    if isinstance(result, dict):
        texts = result.get("rec_texts") or result.get("texts") or []
        scores = result.get("rec_scores") or result.get("scores") or []
        boxes = result.get("dt_polys") or result.get("boxes") or []
        output = []
        for idx, text in enumerate(texts):
            box = boxes[idx] if idx < len(boxes) else []
            conf = scores[idx] if idx < len(scores) else 0.0
            output.append({"box": box, "text": str(text), "confidence": float(conf)})
        return output

    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=os.getenv("PMT_PADDLE_LANG", "en"))
    parser.add_argument("--gpu", default=os.getenv("PMT_OCR_GPU", "0"))
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
            lines = _normalize_paddle_result(result)
            print(json.dumps({"id": request_id, "ok": True, "lines": lines}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"id": None, "ok": False, "error": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
