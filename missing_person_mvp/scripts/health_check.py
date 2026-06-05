from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from missing_person_mvp import config


OPTIONAL_PACKAGES = {
    "cv2": "OpenCV video IO",
    "gradio": "local UI",
    "ultralytics": "YOLO person tracking",
    "faiss": "vector index",
    "reportlab": "PDF export",
    "google.generativeai": "Gemini image generation",
    "torchreid": "OSNet Re-ID",
    "diffusers": "Flux local image generation",
    "transformers": "Flux text encoder runtime",
    "accelerate": "Flux device/offload helpers",
}


def _status(module_name: str) -> str:
    return "ok" if importlib.util.find_spec(module_name) else "missing"


def main() -> int:
    print(f"base_dir={config.BASE_DIR}")
    print(f"data_dir={config.DATA_DIR}")
    print(f"results_dir={config.RESULTS_DIR}")
    print(f"reid_threshold={config.REID_THRESHOLD}")
    print(f"sample_every={config.SAMPLE_EVERY}")
    print(f"max_workers={config.MAX_WORKERS}")
    print(f"camera_rtsp_url={config.CAMERA_RTSP_URL}")
    print(f"image_provider_order={','.join(config.IMAGE_PROVIDER_ORDER)}")
    print(f"flux_enabled={config.FLUX_ENABLED}")
    print(f"flux_model_id={config.FLUX_MODEL_ID}")
    print(f"flux_mode={config.FLUX_MODE}")
    print(f"flux_cache_dir={config.FLUX_CACHE_DIR}")
    for module_name, label in OPTIONAL_PACKAGES.items():
        print(f"{module_name}: {_status(module_name)} ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
