from __future__ import annotations

import numpy as np


VIEW_FRONT = "front"
VIEW_SIDE = "side"
VIEW_BACK = "back"
VIEW_UNKNOWN = "unknown"
VIEWS = [VIEW_FRONT, VIEW_SIDE, VIEW_BACK]


def classify_viewpoint(
    keypoints_conf: np.ndarray | None,
    keypoints_xy: np.ndarray | None = None,
    threshold: float = 0.35,
) -> str:
    """Estimate front/side/back from COCO keypoint confidence.

    YOLO-Pose does not emit a viewpoint class. This is a routing heuristic:
    visible face + balanced body suggests front; one-sided body visibility
    suggests side; visible body with weak face suggests back.
    """
    if keypoints_conf is None or len(keypoints_conf) < 17:
        return VIEW_UNKNOWN
    conf = np.asarray(keypoints_conf, dtype="float32")
    face_score = float(np.mean(conf[[0, 1, 2, 3, 4]]))
    left_score = float(np.mean(conf[[5, 7, 9, 11, 13, 15]]))
    right_score = float(np.mean(conf[[6, 8, 10, 12, 14, 16]]))
    body_score = max(left_score, right_score)
    side_delta = abs(left_score - right_score)
    if face_score >= threshold and side_delta < 0.25:
        return VIEW_FRONT
    if body_score >= threshold and side_delta >= 0.25:
        return VIEW_SIDE
    if body_score >= threshold and face_score < threshold * 0.55:
        return VIEW_BACK
    return VIEW_UNKNOWN


def viewpoint_score(view: str, keypoints_conf: np.ndarray | None) -> float:
    if keypoints_conf is None or len(keypoints_conf) < 17:
        return 0.0
    conf = np.asarray(keypoints_conf, dtype="float32")
    if view == VIEW_FRONT:
        return float(np.mean(conf[[0, 1, 2, 3, 4, 5, 6]]))
    if view == VIEW_SIDE:
        left = float(np.mean(conf[[5, 7, 9, 11, 13, 15]]))
        right = float(np.mean(conf[[6, 8, 10, 12, 14, 16]]))
        return abs(left - right) + max(left, right) * 0.25
    if view == VIEW_BACK:
        body = float(np.mean(conf[[5, 6, 11, 12, 13, 14, 15, 16]]))
        face = float(np.mean(conf[[0, 1, 2, 3, 4]]))
        return max(0.0, body - face)
    return 0.0

