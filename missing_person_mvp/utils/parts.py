"""Part-based body region utilities for Re-ID.

등록 시: full image bbox → 고정 비율로 part crop → 각 part OSNet 임베딩 평균
분석 시: YOLO-Pose keypoints → visible parts 결정 → 보이는 부위만 비교
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PARTS = ["full", "upper", "torso", "lower", "legs"]

# COCO 17-keypoint 인덱스
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6
_LEFT_HIP, _RIGHT_HIP = 11, 12
_LEFT_KNEE, _RIGHT_KNEE = 13, 14
_LEFT_ANKLE, _RIGHT_ANKLE = 15, 16

# bbox 높이 기준 각 part의 (top_ratio, bottom_ratio)
_PART_RATIOS: dict[str, tuple[float, float]] = {
    "full":  (0.00, 1.00),
    "upper": (0.00, 0.50),  # 어깨~허리
    "torso": (0.00, 0.65),  # 어깨~골반
    "lower": (0.35, 1.00),  # 골반~발
    "legs":  (0.50, 1.00),  # 허벅지~발
}

_MIN_CROP_PX = 16  # 너무 작은 crop 무시


def crop_part(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    part: str,
    margin: float = 0.04,
) -> np.ndarray:
    """bbox와 고정 비율로 body part crop을 반환. 실패 시 빈 배열."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    if h <= 0 or x2 <= x1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    t_ratio, b_ratio = _PART_RATIOS.get(part, (0.0, 1.0))
    top = y1 + int(h * t_ratio)
    bottom = y1 + int(h * b_ratio)

    H, W = image.shape[:2]
    pad_x = int((x2 - x1) * margin)
    pad_y = int((bottom - top) * margin)
    left  = max(0, x1 - pad_x)
    right = min(W, x2 + pad_x)
    top   = max(0, top - pad_y)
    bottom = min(H, bottom + pad_y)

    if right - left < _MIN_CROP_PX or bottom - top < _MIN_CROP_PX:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[top:bottom, left:right].copy()


def crop_all_parts(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float = 0.04,
) -> dict[str, np.ndarray]:
    """PARTS 전체에 대해 crop dict 반환. 빈 crop은 제외."""
    result = {}
    for part in PARTS:
        crop = crop_part(image, bbox, part, margin)
        if crop.size > 0:
            result[part] = crop
    return result


def visible_parts_from_keypoints(
    kp_conf: np.ndarray,
    threshold: float = 0.4,
) -> list[str]:
    """YOLO-Pose keypoint confidence → 비교 가능한 part 목록 반환.

    keypoints가 없거나 신뢰도가 낮으면 ["full"] 폴백.
    """
    if kp_conf is None or len(kp_conf) < 17:
        return ["full"]

    def vis(*indices: int) -> bool:
        return any(float(kp_conf[i]) > threshold for i in indices)

    shoulder_vis = vis(_LEFT_SHOULDER, _RIGHT_SHOULDER)
    hip_vis      = vis(_LEFT_HIP, _RIGHT_HIP)
    knee_vis     = vis(_LEFT_KNEE, _RIGHT_KNEE)
    ankle_vis    = vis(_LEFT_ANKLE, _RIGHT_ANKLE)

    parts: list[str] = []
    if shoulder_vis and ankle_vis:
        parts.append("full")
    if shoulder_vis and hip_vis:
        parts.extend(["upper", "torso"])
    if hip_vis and knee_vis:
        parts.append("lower")
    if knee_vis and ankle_vis:
        parts.append("legs")

    # 중복 제거 + 순서 유지
    seen: set[str] = set()
    unique = [p for p in parts if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]
    return unique if unique else ["full"]


def part_weighted_score(
    candidate: dict[str, np.ndarray],
    bank: dict[str, np.ndarray],
    visible: list[str],
) -> float:
    """visible parts 기준 코사인 유사도 가중 평균.

    보이지 않는 부위는 계산에서 완전히 제외 (0점 처리 안 함).
    """
    scores: list[float] = []
    for part in visible:
        c_vec = candidate.get(part)
        b_vec = bank.get(part)
        if c_vec is not None and b_vec is not None:
            sim = float(np.dot(c_vec, b_vec))
            scores.append(sim)
    return float(np.mean(scores)) if scores else 0.0


def _detect_person_bbox(img_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """YOLO-Pose로 가장 큰 사람 bbox 탐지. 실패 시 None.

    생성 이미지·등록 사진은 배경·워터마크를 포함하므로, 사람만 타이트하게
    크롭해야 영상 속 YOLO 크롭과 임베딩 분포가 일치한다.
    """
    try:
        import cv2
        from missing_person_mvp.utils.track import _load_yolo

        model = _load_yolo()
        if model is None:
            return None
        arr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = model.predict(arr, classes=[0], verbose=False)
        best: tuple[int, tuple[int, int, int, int]] | None = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue
            for row in boxes.xyxy:
                x1, y1, x2, y2 = (int(v) for v in row.tolist())
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if best is None or area > best[0]:
                    best = (area, (x1, y1, x2, y2))
        return best[1] if best else None
    except Exception:
        return None


def build_part_bank(
    images: list[Image.Image],
    extract_fn,
    representative_fn,
) -> dict[str, np.ndarray]:
    """등록 사진 목록으로 part embedding bank 생성.

    각 사진에서 YOLO로 사람 bbox를 잡아(실패 시 전체 이미지) full/upper/torso/
    lower/legs crop → OSNet 임베딩 → part별 평균 대표 벡터 반환.
    """
    accum: dict[str, list[np.ndarray]] = {p: [] for p in PARTS}

    for pil_img in images:
        img_rgb = np.array(pil_img.convert("RGB"))
        h, w = img_rgb.shape[:2]
        # 사람 영역만 타이트 크롭 (배경·워터마크 배제) — 실패 시 전체 이미지
        bbox = _detect_person_bbox(img_rgb) or (0, 0, w, h)

        for part, crop in crop_all_parts(img_rgb, bbox).items():
            crop_rgb = crop  # 이미 RGB
            feat = extract_fn([crop_rgb])
            if feat.shape[0] > 0:
                accum[part].append(feat[0])

    bank: dict[str, np.ndarray] = {}
    for part, feats in accum.items():
        if feats:
            bank[part] = representative_fn(np.stack(feats))
    return bank
