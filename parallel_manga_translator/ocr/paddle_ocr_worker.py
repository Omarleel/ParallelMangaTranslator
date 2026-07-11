from __future__ import annotations

"""Worker aislado y persistente para PaddleOCR 2.x/3.x."""

import argparse
import base64
import json
import os
import re
import sys
import traceback
from typing import Any

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.45")

from parallel_manga_translator.ocr.paddle_result import normalize_paddle_result


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def _decode_image(payload: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(payload.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen enviada al worker PaddleOCR")
    return image


def _major_version(version: str) -> int:
    match = re.match(r"\s*(\d+)", str(version or ""))
    return int(match.group(1)) if match else 0


_WINDOWS_DLL_HANDLES: list[Any] = []


def _preload_torch_runtime() -> dict[str, Any]:
    """No carga PyTorch dentro del worker PaddleOCR.

    En Windows, las ruedas GPU de PyTorch y Paddle pueden incluir versiones
    distintas de cuDNN/cuBLAS aun cuando ambas indiquen CUDA 12.9. El worker
    Paddle debe permanecer libre de imports de Torch para evitar colisiones
    de DLL como ``cudnn_engines_precompiled64_9.dll``.
    """
    return {
        "torch_preloaded": False,
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_lib": None,
    }


def _build_ocr(lang: str, use_gpu: bool):
    torch_runtime = _preload_torch_runtime()

    import paddle  # type: ignore
    import paddleocr  # type: ignore
    from paddleocr import PaddleOCR  # type: ignore

    paddle_version = str(getattr(paddle, "__version__", "desconocida"))
    paddleocr_version = str(getattr(paddleocr, "__version__", "desconocida"))
    cuda_compiled = bool(paddle.is_compiled_with_cuda())
    try:
        gpu_count = int(paddle.device.cuda.device_count()) if cuda_compiled else 0
    except Exception:
        gpu_count = 0

    if use_gpu and (not cuda_compiled or gpu_count < 1):
        raise RuntimeError(
            "Se solicitó PaddleOCR GPU, pero el paquete PaddlePaddle instalado no tiene CUDA disponible. "
            f"paddle={paddle_version}, paddleocr={paddleocr_version}, "
            f"is_compiled_with_cuda={cuda_compiled}, gpu_count={gpu_count}. "
            "Instala paddlepaddle-gpu compatible o configura ocr.gpu=false."
        )

    major = _major_version(paddleocr_version)
    device = "gpu:0" if use_gpu else "cpu"

    if major >= 3:
        # API oficial de PaddleOCR 3.x. Desactivamos módulos de documento que no
        # aportan valor en recortes de manga y consumen memoria/tiempo adicional.
        ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        api_mode = "3.x"
    else:
        # Compatibilidad con PaddleOCR 2.x.
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        api_mode = "2.x"

    return ocr, {
        **torch_runtime,
        "paddle_version": paddle_version,
        "paddleocr_version": paddleocr_version,
        "api_mode": api_mode,
        "device": device,
        "cuda_compiled": cuda_compiled,
        "gpu_count": gpu_count,
    }


def _run_ocr(ocr, image, api_mode: str):
    if api_mode == "3.x":
        # ``ocr`` permanece como alias obsoleto en 3.x; ``predict`` es la API
        # oficial y devuelve objetos Result de PaddleX.
        return ocr.predict(image)

    try:
        return ocr.ocr(img=image, cls=True)
    except TypeError:
        return ocr.ocr(image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    use_gpu = str(args.gpu).strip().lower() in {"1", "true", "yes", "on"}

    try:
        ocr, runtime = _build_ocr(args.lang, use_gpu)
    except Exception as exc:
        _emit(
            {
                "type": "fatal",
                "error": f"No se pudo iniciar PaddleOCR: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 2

    _emit({"type": "ready", "ok": True, "lang": args.lang, "gpu_requested": use_gpu, **runtime})

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request_id = None
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            if request.get("cmd") == "shutdown":
                _emit({"id": request_id, "ok": True, "shutdown": True})
                return 0
            image = _decode_image(request["image"])
            result = _run_ocr(ocr, image, runtime["api_mode"])
            lines = normalize_paddle_result(result)
            _emit({"id": request_id, "ok": True, "lines": lines})
        except Exception as exc:
            _emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
