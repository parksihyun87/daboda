"""Part crop 색상 vs 착장 설명 색상 비교 → Re-ID score boost.

흐름:
1. crop BGR → K-means(k=3) → 지배색 BGR
2. 기대 색상(RGB, outfit_parser.color_rgb()) → HSV 변환
3. HSV 거리 → match score [0,1]
4. 가시 파트 평균 → color_boost [0,1]
5. final_score = reid_score × (1 + COLOR_ALPHA × color_boost)
"""
from __future__ import annotations

import cv2
import numpy as np

from missing_person_mvp.vton.outfit_parser import OutfitSpec, color_rgb

COLOR_ALPHA: float = 0.30  # 최대 30% 부스트

# 파트 → OutfitSpec 필드 매핑
_PART_COLOR_FIELD: dict[str, str | None] = {
    "upper": "upper_color",
    "torso": "upper_color",
    "lower": "lower_color",
    "legs":  "lower_color",
    "full":  None,  # primary_color() 사용
}


def dominant_color_bgr(crop_bgr: np.ndarray, k: int = 3) -> tuple[int, int, int]:
    """K-means로 crop에서 가장 많이 나타나는 색상 반환 (BGR)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return (128, 128, 128)
    small = cv2.resize(crop_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        counts = np.bincount(labels.flatten())
        dominant = centers[int(np.argmax(counts))]
        return (int(dominant[0]), int(dominant[1]), int(dominant[2]))
    except Exception:
        mean = pixels.mean(axis=0)
        return (int(mean[0]), int(mean[1]), int(mean[2]))


def color_match(
    actual_bgr: tuple[int, int, int],
    expected_rgb: tuple[int, int, int],
) -> float:
    """HSV 공간에서 두 색상의 유사도 [0,1].

    무채색(expected S<40): V 거리 중심 비교.
    유채색: H(원형거리) 70% + S 30% 가중합.
    """
    exp_bgr = (expected_rgb[2], expected_rgb[1], expected_rgb[0])

    actual_hsv = cv2.cvtColor(
        np.array([[actual_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV
    )[0, 0].astype(int)
    exp_hsv = cv2.cvtColor(
        np.array([[exp_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV
    )[0, 0].astype(int)

    # 무채색 기대(검정·흰색·회색): V 차이 기준
    if exp_hsv[1] < 40:
        v_diff = abs(actual_hsv[2] - exp_hsv[2]) / 255.0
        # 채도 높은 실제 색상은 감점
        s_penalty = actual_hsv[1] / 255.0 * 0.25
        return float(max(0.0, 1.0 - v_diff - s_penalty))

    # 유채색: Hue 원형거리 (OpenCV H: 0-180)
    h_diff = abs(actual_hsv[0] - exp_hsv[0])
    h_diff = min(h_diff, 180 - h_diff)
    h_score = float(max(0.0, 1.0 - h_diff / 60.0))  # 60° 이상이면 0점

    s_score = 1.0 - abs(actual_hsv[1] - exp_hsv[1]) / 255.0
    return h_score * 0.7 + s_score * 0.3


def part_color_boost(
    part_crops: dict[str, np.ndarray],
    outfit_spec: OutfitSpec,
) -> float:
    """가시 파트별 색상 매치 평균 [0,1].

    매핑되는 기대 색상이 없는 파트는 계산에서 제외.
    """
    scores: list[float] = []
    for part, crop in part_crops.items():
        field = _PART_COLOR_FIELD.get(part)
        if field is not None:
            expected_name: str | None = getattr(outfit_spec, field, None)
        else:
            expected_name = outfit_spec.primary_color()

        if not expected_name:
            continue

        expected_rgb = color_rgb(expected_name)
        actual_bgr = dominant_color_bgr(crop)
        scores.append(color_match(actual_bgr, expected_rgb))

    return float(np.mean(scores)) if scores else 0.0
