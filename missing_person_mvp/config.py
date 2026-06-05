from __future__ import annotations

import os
from pathlib import Path

# OpenMP 런타임 중복(faiss + torch) → Error #15 방지. 모든 진입점이 config를
# 일찍 import하므로 여기서도 설정 (torch/faiss 로드 전).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def _load_dotenv(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_DIR = Path(os.getenv("DB_DIR", DATA_DIR / "db"))
FOOTAGE_DIR = Path(os.getenv("FOOTAGE_DIR", DATA_DIR / "footage"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", DATA_DIR / "results"))

PERSONS_JSON = DB_DIR / "persons.json"
EMBEDDINGS_NPZ = DB_DIR / "embeddings.npz"
FAISS_INDEX = DB_DIR / "faiss.index"

# 웹 데모용 번들 테스트 영상 (업로드 없이 바로 선택 가능)
DEMO_VIDEO_DIR = DATA_DIR / "video"
DEMO_VIDEO_PATH = DEMO_VIDEO_DIR / "VIGI C330 1.20_C8F3_20260601225048_615_label.mp4"
DEMO_VIDEO_XML = DEMO_VIDEO_DIR / "cvat_xml" / "annotations.xml"
DEMO_VIDEO_LABEL = "노란색 상의, 검은색바지 인물 정답 영상"
DEMO_VIDEO_XML_LABEL = "target"  # CVAT 정답 라벨명
DEMO_VIDEO_OUTFIT = "노란색 상의에 검은색 바지"  # 정답 인물 복장 (색상 보정용)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FASHN_API_KEY  = os.getenv("FASHN_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")
CAMERA_RTSP_URL = os.getenv("CAMERA_RTSP_URL", "")
OSNET_MODEL_PATH = os.getenv("OSNET_MODEL_PATH", "./weights/osnet_x1_0_market.pth")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n-pose.pt")
YOLO_FALLBACK_MODEL = os.getenv("YOLO_FALLBACK_MODEL", "yolo11n.pt")
YOLO_TRACKER = os.getenv("YOLO_TRACKER", "bytetrack.yaml")

IMAGE_PROVIDER_ORDER = [
    item.strip().lower()
    for item in os.getenv("IMAGE_PROVIDER_ORDER", "gemini,flux,fashn,placeholder").split(",")
    if item.strip()
]

FLUX_ENABLED = os.getenv("FLUX_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
FLUX_MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
FLUX_DEVICE = os.getenv("FLUX_DEVICE", "auto")
FLUX_DTYPE = os.getenv("FLUX_DTYPE", "bfloat16")
FLUX_NUM_INFERENCE_STEPS = int(os.getenv("FLUX_NUM_INFERENCE_STEPS", "4"))
FLUX_GUIDANCE_SCALE = float(os.getenv("FLUX_GUIDANCE_SCALE", "0.0"))
FLUX_SEED = int(os.getenv("FLUX_SEED", "42"))
FLUX_CACHE_DIR = Path(os.getenv("FLUX_CACHE_DIR", BASE_DIR / "weights" / "flux"))
FLUX_MODE = os.getenv("FLUX_MODE", "inpaint").lower()
FLUX_INPAINT_STRENGTH = float(os.getenv("FLUX_INPAINT_STRENGTH", "0.65"))
FLUX_INPAINT_HEIGHT = int(os.getenv("FLUX_INPAINT_HEIGHT", "1024"))
FLUX_INPAINT_WIDTH = int(os.getenv("FLUX_INPAINT_WIDTH", "768"))
FLUX_REQUIRE_THREE_VIEWS = os.getenv("FLUX_REQUIRE_THREE_VIEWS", "true").lower() in {"1", "true", "yes", "on"}
FLUX_MASK_DILATE = int(os.getenv("FLUX_MASK_DILATE", "12"))
FLUX_CPU_OFFLOAD = os.getenv("FLUX_CPU_OFFLOAD", "true").lower() in {"1", "true", "yes", "on"}

REID_THRESHOLD = float(os.getenv("REID_THRESHOLD", "0.72"))
SAMPLE_EVERY = int(os.getenv("SAMPLE_EVERY", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
EMBEDDING_DIM = 512


def ensure_dirs() -> None:
    for path in (DATA_DIR, DB_DIR, FOOTAGE_DIR, RESULTS_DIR, FLUX_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()
