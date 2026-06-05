"""Skeleton-based gait feature extraction — simplified SkeletonGait++.

Registration (1회):
    보행 동영상 2개 이상 → YOLO-Pose → 정규화 골격 시퀀스 → 32-dim gait embedding

CCTV 매칭 (분석 중):
    TrackInfo.keypoints_xy 누적 → 동일 방법으로 임베딩 → 코사인 유사도 비교
"""
from __future__ import annotations

import cv2
import numpy as np

# COCO 17 keypoints 중 보행에 유효한 8개: 양쪽 어깨(5,6), 골반(11,12), 무릎(13,14), 발목(15,16)
_GAIT_KP_IDX = [5, 6, 11, 12, 13, 14, 15, 16]
_HIP_L, _HIP_R = 11, 12
_SHOULDER_L, _SHOULDER_R = 5, 6

GAIT_DIM = 32        # 8 joints × (mean_x, mean_y, std_x, std_y)
_MIN_FRAMES = 15     # 유효한 임베딩 생성에 필요한 최소 프레임


def normalize_skeleton(kp_xy: np.ndarray) -> np.ndarray | None:
    """(17, 2) pixel coords → pelvis 중심·torso 스케일 정규화된 (8, 2).

    골반 중점을 원점, 어깨-골반 거리를 단위 길이로 정규화.
    핵심 관절이 유효하지 않으면 None 반환.
    """
    if kp_xy is None or kp_xy.shape != (17, 2):
        return None
    pelvis = (kp_xy[_HIP_L] + kp_xy[_HIP_R]) / 2.0
    shoulder_mid = (kp_xy[_SHOULDER_L] + kp_xy[_SHOULDER_R]) / 2.0
    scale = float(np.linalg.norm(shoulder_mid - pelvis))
    if scale < 1e-6:
        return None
    normed = (kp_xy[_GAIT_KP_IDX] - pelvis) / scale  # (8, 2)
    return normed.astype("float32")


def build_gait_embedding(frames: list[np.ndarray]) -> np.ndarray | None:
    """정규화 골격 프레임 목록 → L2-정규화된 32-dim gait 임베딩.

    특징: 각 관절의 [평균_x, 평균_y, 표준편차_x, 표준편차_y] 연결.
    충분한 프레임 없으면 None 반환.
    """
    valid = [f for f in frames if f is not None and f.shape == (8, 2)]
    if len(valid) < _MIN_FRAMES:
        return None
    stack = np.stack(valid, axis=0)        # (N, 8, 2)
    mean = stack.mean(axis=0).flatten()    # (16,)
    std  = stack.std(axis=0).flatten()     # (16,)
    feat = np.concatenate([mean, std]).astype("float32")  # (32,)
    norm = float(np.linalg.norm(feat))
    return feat / norm if norm > 1e-8 else None


def extract_gait_frames_from_video(
    video_path: str,
    conf_threshold: float = 0.35,
) -> list[np.ndarray]:
    """보행 등록용 — 동영상에서 YOLO-Pose 실행하여 정규화 골격 시퀀스 추출.

    (CCTV 분석 중에는 TrackInfo.keypoints_xy를 직접 사용하므로 이 함수 불필요)
    """
    from missing_person_mvp.utils.track import _load_yolo

    model = _load_yolo()
    if model is None:
        return []

    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, classes=[0], verbose=False)
            for result in results:
                kps = getattr(result, "keypoints", None)
                if kps is None or len(kps.data) == 0:
                    continue
                boxes = getattr(result, "boxes", None)
                best_idx = int(boxes.conf.argmax()) if (boxes is not None and boxes.conf is not None) else 0
                kp_data = kps.data[best_idx]           # (17, 3): xy + conf
                kp_conf = kp_data[:, 2].cpu().numpy()
                if kp_conf[[5, 6, 11, 12, 13, 14]].mean() < conf_threshold:
                    continue
                xy = kp_data[:, :2].cpu().numpy()
                normed = normalize_skeleton(xy)
                if normed is not None:
                    frames.append(normed)
    finally:
        cap.release()
    return frames


def gait_similarity(
    registered_embedding: np.ndarray,
    track_frames: list[np.ndarray],
) -> float:
    """등록된 gait 임베딩과 CCTV 트랙 골격 시퀀스의 코사인 유사도 [0, 1].

    트랙 프레임이 부족하면 0.0 반환 (Re-ID 결과에 영향 없음).
    """
    if registered_embedding is None or not track_frames:
        return 0.0
    cand = build_gait_embedding(track_frames)
    if cand is None:
        return 0.0
    return float(np.clip(np.dot(registered_embedding, cand), 0.0, 1.0))
