from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from missing_person_mvp import config


@dataclass
class TrackInfo:
    track_id: int
    bbox: tuple[int, int, int, int]
    conf: float
    keypoints_conf: Optional[np.ndarray] = field(default=None, repr=False)
    keypoints_xy:   Optional[np.ndarray] = field(default=None, repr=False)  # (17, 2) pixel coords


_YOLO_MODEL = None
_YOLO_LOAD_ATTEMPTED = False
_IS_POSE_MODEL = False


def _load_yolo():
    global _YOLO_MODEL, _YOLO_LOAD_ATTEMPTED, _IS_POSE_MODEL
    if _YOLO_LOAD_ATTEMPTED:
        return _YOLO_MODEL
    _YOLO_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO  # type: ignore

        _YOLO_MODEL = YOLO(config.YOLO_MODEL)
        _IS_POSE_MODEL = True
    except Exception as exc:
        warnings.warn(
            f"Failed to load YOLO_MODEL={config.YOLO_MODEL}: {exc}. "
            f"Trying YOLO_FALLBACK_MODEL={config.YOLO_FALLBACK_MODEL}.",
            stacklevel=2,
        )
        try:
            from ultralytics import YOLO  # type: ignore

            _YOLO_MODEL = YOLO(config.YOLO_FALLBACK_MODEL)
            _IS_POSE_MODEL = False
        except Exception as fallback_exc:
            warnings.warn(
                f"Failed to load YOLO fallback model={config.YOLO_FALLBACK_MODEL}: {fallback_exc}.",
                stacklevel=2,
            )
            _YOLO_MODEL = None
    return _YOLO_MODEL


def detect_persons(frame: np.ndarray, tracker: str = "") -> list[TrackInfo]:
    """YOLO-Pose로 프레임에서 인물 감지·추적.

    Args:
        tracker: "bytetrack.yaml" | "botsort.yaml" | "" → config.YOLO_TRACKER 기본값
    """
    model = _load_yolo()
    if model is None:
        return []

    effective_tracker = tracker or config.YOLO_TRACKER
    try:
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker=effective_tracker,
            verbose=False,
        )
    except Exception as exc:
        warnings.warn(
            f"YOLO tracking failed with tracker={effective_tracker}: {exc}",
            stacklevel=2,
        )
        return []

    tracks: list[TrackInfo] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue

        ids = boxes.id
        confs = boxes.conf
        keypoints = getattr(result, "keypoints", None)

        for idx, xyxy in enumerate(boxes.xyxy):
            track_id = int(ids[idx].item()) if ids is not None else idx
            conf = float(confs[idx].item()) if confs is not None else 0.0
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]

            kp_conf = None
            kp_xy   = None
            if keypoints is not None:
                try:
                    kp_conf = keypoints.conf[idx].cpu().numpy().astype("float32")
                    kp_xy   = keypoints.xy[idx].cpu().numpy().astype("float32")  # (17, 2)
                except Exception:
                    pass

            tracks.append(
                TrackInfo(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    keypoints_conf=kp_conf,
                    keypoints_xy=kp_xy,
                )
            )
    return tracks
