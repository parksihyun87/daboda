from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from missing_person_mvp.utils.video import hms_to_seconds


@dataclass(slots=True)
class GTBox:
    video_id: str
    gt_track_id: str
    frame: int
    timestamp_sec: float
    bbox: tuple[float, float, float, float]
    outside: bool = False
    occluded: bool = False


@dataclass(slots=True)
class Hit:
    timestamp_sec: float
    track_id: str
    camera_id: str
    bbox: tuple[float, float, float, float] | None
    similarity: float = 0.0
    crop_image_path: str = ""


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes"}


def _parse_bbox(value) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(float(part) for part in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    cleaned = value.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [part.strip() for part in cleaned.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _parse_time(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    value = str(value).strip()
    if ":" in value:
        return hms_to_seconds(value)
    return float(value)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _timestamp_from_frame(frame: int, fps: float) -> float:
    return frame / fps if fps > 0 else float(frame)


def load_cvat_xml(path: str | Path, fps: float = 30.0, label: str = "person_of_interest") -> list[GTBox]:
    root = ET.parse(path).getroot()
    video_id = Path(path).stem
    boxes: list[GTBox] = []

    # CVAT video tracks: <track id="..." label="..."><box frame="..." xtl=... /></track>
    for track in root.findall(".//track"):
        track_label = track.attrib.get("label", "")
        if label and track_label != label:
            continue
        track_id = track.attrib.get("id", "0")
        for box in track.findall("box"):
            frame = int(float(box.attrib.get("frame", "0")))
            bbox = (
                float(box.attrib.get("xtl", "0")),
                float(box.attrib.get("ytl", "0")),
                float(box.attrib.get("xbr", "0")),
                float(box.attrib.get("ybr", "0")),
            )
            boxes.append(
                GTBox(
                    video_id=video_id,
                    gt_track_id=track_id,
                    frame=frame,
                    timestamp_sec=_timestamp_from_frame(frame, fps),
                    bbox=bbox,
                    outside=_truthy(box.attrib.get("outside")),
                    occluded=_truthy(box.attrib.get("occluded")),
                )
            )

    # CVAT image export can store boxes with track_id attributes.
    for image in root.findall(".//image"):
        frame = int(float(image.attrib.get("id", image.attrib.get("frame", "0"))))
        for box in image.findall("box"):
            if label and box.attrib.get("label", "") != label:
                continue
            track_id = box.attrib.get("track_id", "shape")
            bbox = (
                float(box.attrib.get("xtl", "0")),
                float(box.attrib.get("ytl", "0")),
                float(box.attrib.get("xbr", "0")),
                float(box.attrib.get("ybr", "0")),
            )
            boxes.append(
                GTBox(
                    video_id=video_id,
                    gt_track_id=track_id,
                    frame=frame,
                    timestamp_sec=_timestamp_from_frame(frame, fps),
                    bbox=bbox,
                    outside=False,
                    occluded=_truthy(box.attrib.get("occluded")),
                )
            )
    return sorted(boxes, key=lambda box: (box.gt_track_id, box.frame))


def load_mot(path: str | Path, fps: float = 30.0, video_id: str | None = None) -> list[GTBox]:
    boxes: list[GTBox] = []
    vid = video_id or Path(path).stem
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = parts[1]
            x, y, w, h = [float(value) for value in parts[2:6]]
            visible = float(parts[8]) if len(parts) > 8 and parts[8] else 1.0
            boxes.append(
                GTBox(
                    video_id=vid,
                    gt_track_id=track_id,
                    frame=frame,
                    timestamp_sec=_timestamp_from_frame(frame, fps),
                    bbox=(x, y, x + w, y + h),
                    outside=False,
                    occluded=visible < 0.5,
                )
            )
    return sorted(boxes, key=lambda box: (box.gt_track_id, box.frame))


def load_hits(path: str | Path) -> list[Hit]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("hits", data if isinstance(data, list) else [])
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    hits: list[Hit] = []
    for row in rows:
        bbox = row.get("bounding_box") or row.get("bbox")
        parsed_bbox = _parse_bbox(bbox) if isinstance(row, dict) else None
        timestamp = (
            row.get("timestamp_sec")
            or row.get("time_sec")
            or row.get("timestamp")
            or row.get("time")
            or "0"
        )
        hits.append(
            Hit(
                timestamp_sec=_parse_time(str(timestamp)),
                track_id=str(row.get("track_id", "")),
                camera_id=str(row.get("camera_id", row.get("video_id", ""))),
                bbox=parsed_bbox,
                similarity=float(row.get("similarity", row.get("score", 0.0)) or 0.0),
                crop_image_path=str(row.get("crop_image_path", "")),
            )
        )
    return sorted(hits, key=lambda hit: hit.timestamp_sec)


def _group_gt(boxes: Iterable[GTBox]) -> dict[str, list[GTBox]]:
    grouped: dict[str, list[GTBox]] = {}
    for box in boxes:
        if box.outside:
            continue
        grouped.setdefault(box.gt_track_id, []).append(box)
    return {key: sorted(value, key=lambda box: box.timestamp_sec) for key, value in grouped.items()}


def _gt_interval(boxes: list[GTBox]) -> tuple[float, float]:
    return boxes[0].timestamp_sec, boxes[-1].timestamp_sec


def _temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0.0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _nearest_gt_box(gt_boxes: list[GTBox], timestamp_sec: float) -> GTBox:
    return min(gt_boxes, key=lambda box: abs(box.timestamp_sec - timestamp_sec))


def evaluate_detection(
    gt_boxes: list[GTBox],
    hits: list[Hit],
    time_tolerance_sec: float = 2.0,
    iou_threshold: float = 0.3,
    video_duration_sec: float | None = None,
    runtime_sec: float | None = None,
) -> dict:
    grouped = _group_gt(gt_boxes)
    matched_rows: list[dict] = []
    false_rows: list[dict] = []
    missed_rows: list[dict] = []
    used_hit_indices: set[int] = set()
    gt_hit_map: dict[str, list[tuple[int, Hit, float]]] = {track_id: [] for track_id in grouped}

    for hit_idx, hit in enumerate(hits):
        best: tuple[str, GTBox, float, float] | None = None
        for track_id, boxes in grouped.items():
            gt = _nearest_gt_box(boxes, hit.timestamp_sec)
            dt = abs(gt.timestamp_sec - hit.timestamp_sec)
            if dt > time_tolerance_sec:
                continue
            iou = _iou(gt.bbox, hit.bbox) if hit.bbox is not None else math.nan
            if hit.bbox is not None and iou < iou_threshold:
                continue
            rank = iou if not math.isnan(iou) else 1.0 / (1.0 + dt)
            if best is None or rank > best[3]:
                best = (track_id, gt, iou, rank)
        if best is None:
            false_rows.append({**asdict(hit), "reason": "no_matching_gt"})
            continue
        track_id, gt, iou, _rank = best
        used_hit_indices.add(hit_idx)
        gt_hit_map[track_id].append((hit_idx, hit, iou))
        matched_rows.append(
            {
                "gt_track_id": track_id,
                "gt_timestamp_sec": gt.timestamp_sec,
                "hit_timestamp_sec": hit.timestamp_sec,
                "delay_sec": hit.timestamp_sec - grouped[track_id][0].timestamp_sec,
                "pred_track_id": hit.track_id,
                "iou": "" if math.isnan(iou) else iou,
                "similarity": hit.similarity,
                "crop_image_path": hit.crop_image_path,
            }
        )

    matched_tracks = {track_id for track_id, matches in gt_hit_map.items() if matches}
    for track_id, boxes in grouped.items():
        if track_id not in matched_tracks:
            start, end = _gt_interval(boxes)
            missed_rows.append({"gt_track_id": track_id, "start_sec": start, "end_sec": end})

    gt_count = len(grouped)
    track_recall = len(matched_tracks) / gt_count if gt_count else 0.0

    first_delays = []
    temporal_ious = []
    bbox_ious = []
    id_switch_count = 0
    fragmentation_count = 0
    for track_id, boxes in grouped.items():
        matches = sorted(gt_hit_map[track_id], key=lambda item: item[1].timestamp_sec)
        if not matches:
            continue
        first_delays.append(matches[0][1].timestamp_sec - boxes[0].timestamp_sec)
        gt_interval = _gt_interval(boxes)
        hit_interval = (matches[0][1].timestamp_sec, matches[-1][1].timestamp_sec)
        temporal_ious.append(_temporal_iou(gt_interval, hit_interval))
        pred_ids = [item[1].track_id for item in matches]
        unique_pred_ids = [value for idx, value in enumerate(pred_ids) if value and value not in pred_ids[:idx]]
        id_switch_count += max(0, len(unique_pred_ids) - 1)
        gaps = [
            matches[idx + 1][1].timestamp_sec - matches[idx][1].timestamp_sec
            for idx in range(len(matches) - 1)
        ]
        fragmentation_count += sum(1 for gap in gaps if gap > time_tolerance_sec * 2)
        bbox_ious.extend(item[2] for item in matches if not math.isnan(item[2]))

    duration = video_duration_sec
    if duration is None:
        gt_end = max((box.timestamp_sec for box in gt_boxes), default=0.0)
        hit_end = max((hit.timestamp_sec for hit in hits), default=0.0)
        duration = max(gt_end, hit_end, 1.0)

    false_hits_per_10min = len(false_rows) / max(duration / 600.0, 1e-9)
    mean_first_hit_delay_sec = sum(first_delays) / len(first_delays) if first_delays else None
    mean_temporal_iou = sum(temporal_ious) / len(temporal_ious) if temporal_ious else 0.0
    mean_bbox_iou = sum(bbox_ious) / len(bbox_ious) if bbox_ious else None
    runtime_ratio = runtime_sec / duration if runtime_sec is not None and duration > 0 else None

    false_hit_score = max(0.0, 1.0 - min(false_hits_per_10min, 20.0) / 20.0)
    tracking_stability = max(0.0, 1.0 - min(id_switch_count + fragmentation_count, max(gt_count, 1) * 3) / (max(gt_count, 1) * 3))
    speed_score = 1.0 if runtime_ratio is None else max(0.0, 1.0 - min(runtime_ratio, 2.0) / 2.0)
    total_score = (
        track_recall * 0.35
        + mean_temporal_iou * 0.15
        + false_hit_score * 0.20
        + tracking_stability * 0.15
        + speed_score * 0.15
    )

    return {
        "metrics": {
            "gt_track_count": gt_count,
            "hit_count": len(hits),
            "matched_track_count": len(matched_tracks),
            "track_recall": track_recall,
            "mean_first_hit_delay_sec": mean_first_hit_delay_sec,
            "false_hit_count": len(false_rows),
            "false_hits_per_10min": false_hits_per_10min,
            "temporal_iou": mean_temporal_iou,
            "mean_bbox_iou": mean_bbox_iou,
            "id_switch_count": id_switch_count,
            "fragmentation_count": fragmentation_count,
            "runtime_ratio": runtime_ratio,
            "total_score": total_score,
            "recall_gate_pass": track_recall >= 0.8,
        },
        "matched_hits": matched_rows,
        "false_hits": false_rows,
        "missed_tracks": missed_rows,
    }


def write_outputs(result: dict, output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "matched_hits.csv", result["matched_hits"])
    _write_csv(out_dir / "false_hits.csv", result["false_hits"])
    _write_csv(out_dir / "missed_tracks.csv", result["missed_tracks"])
    metrics = result["metrics"]
    lines = [
        "# Detection Evaluation Summary",
        "",
        f"- total_score: {metrics['total_score']:.4f}",
        f"- track_recall: {metrics['track_recall']:.4f}",
        f"- false_hits_per_10min: {metrics['false_hits_per_10min']:.4f}",
        f"- temporal_iou: {metrics['temporal_iou']:.4f}",
        f"- mean_bbox_iou: {metrics['mean_bbox_iou']}",
        f"- id_switch_count: {metrics['id_switch_count']}",
        f"- fragmentation_count: {metrics['fragmentation_count']}",
        f"- runtime_ratio: {metrics['runtime_ratio']}",
        f"- recall_gate_pass: {metrics['recall_gate_pass']}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CCTV detection hits against CVAT/MOT ground truth.")
    parser.add_argument("--ground-truth", required=True, help="CVAT XML or MOT text file")
    parser.add_argument("--hits", required=True, help="hits.csv or hits.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=["auto", "cvat", "mot"], default="auto")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--label", default="person_of_interest")
    parser.add_argument("--time-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--video-duration-sec", type=float, default=None)
    parser.add_argument("--runtime-sec", type=float, default=None)
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    fmt = args.format
    if fmt == "auto":
        fmt = "cvat" if gt_path.suffix.lower() == ".xml" else "mot"
    gt_boxes = (
        load_cvat_xml(gt_path, fps=args.fps, label=args.label)
        if fmt == "cvat"
        else load_mot(gt_path, fps=args.fps)
    )
    hits = load_hits(args.hits)
    result = evaluate_detection(
        gt_boxes,
        hits,
        time_tolerance_sec=args.time_tolerance_sec,
        iou_threshold=args.iou_threshold,
        video_duration_sec=args.video_duration_sec,
        runtime_sec=args.runtime_sec,
    )
    write_outputs(result, args.output_dir)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


def hits_from_report(report) -> list[Hit]:
    """AnalysisReport.hits → list[Hit] — evaluate_detection()에 직접 전달 가능."""
    result: list[Hit] = []
    for h in report.hits:
        result.append(Hit(
            timestamp_sec=hms_to_seconds(h.timestamp),
            track_id=str(h.track_id),
            camera_id=h.camera_id,
            bbox=tuple(float(v) for v in h.bounding_box),  # type: ignore[arg-type]
            similarity=h.similarity,
            crop_image_path=h.crop_image_path,
        ))
    return sorted(result, key=lambda hit: hit.timestamp_sec)


if __name__ == "__main__":
    raise SystemExit(main())
