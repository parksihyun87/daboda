from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def parse_recording_start(video_path: str) -> datetime | None:
    """파일명에서 CCTV 녹화 시작 시각 추출 → datetime.

    VIGI, Hikvision, Dahua 등 주요 CCTV 파일명 패턴 지원:
      - 20260601225048   (YYYYMMDDHHmmss 14자리 연속)
      - 20260601_225048  (YYYYMMDD_HHmmss)

    실패하면 None 반환 (graceful).
    """
    stem = Path(video_path).stem
    # 패턴 1: 14자리 연속 (예: 20260601225048)
    m = re.search(r'(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([01]\d|2[0-3])([0-5]\d)([0-5]\d)(?!\d)', stem)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)), int(m.group(6)))
        except ValueError:
            pass
    # 패턴 2: YYYYMMDD_HHmmss (언더스코어 구분)
    m2 = re.search(r'(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])_([01]\d|2[0-3])([0-5]\d)([0-5]\d)', stem)
    if m2:
        try:
            return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)),
                            int(m2.group(4)), int(m2.group(5)), int(m2.group(6)))
        except ValueError:
            pass
    return None


def recording_datetime_str(recording_start: datetime | None, offset_sec: float) -> str | None:
    """녹화 시작 datetime + 영상 내 경과 초 → 실제 시각 문자열 'YYYY-MM-DD HH:MM:SS'.

    recording_start가 None이면 None 반환.
    """
    if recording_start is None:
        return None
    return (recording_start + timedelta(seconds=offset_sec)).strftime("%Y-%m-%d %H:%M:%S")


def hms_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("시간은 HH:MM:SS 형식이어야 합니다.")
    try:
        hours, minutes, seconds = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("시간은 HH:MM:SS 형식이어야 합니다.") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("시간 값이 올바르지 않습니다.")
    return float(hours * 3600 + minutes * 60 + seconds)


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def iter_frames(
    video_path: str,
    start_time: str,
    end_time: str,
    sample_every: int = 3,
) -> Generator[tuple[np.ndarray, float], None, None]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    if sample_every < 1:
        raise ValueError("sample_every는 1 이상이어야 합니다.")

    start_sec = hms_to_seconds(start_time)
    end_sec = hms_to_seconds(end_time)
    if end_sec < start_sec:
        raise ValueError("종료 시각은 시작 시각보다 빠를 수 없습니다.")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_sec = start_sec + frame_idx / fps
            if current_sec > end_sec:
                break
            if frame_idx % sample_every == 0:
                yield frame, current_sec
            frame_idx += 1
    finally:
        cap.release()


def extract_frame_with_bbox(
    video_path: str,
    timestamp_hms: str,
    bbox: tuple[int, int, int, int],
    similarity: float = 0.0,
) -> np.ndarray:
    """타임스탬프 위치 프레임을 추출하고 bbox를 빨간 사각형으로 표시. RGB ndarray 반환."""
    sec = hms_to_seconds(timestamp_hms)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        return blank
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 220), 3)
    if similarity > 0:
        label = f"{similarity * 100:.1f}%"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 220), 2,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_video_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    padding_sec: float = 1.5,
) -> str:
    """트래킹 구간(start_sec~end_sec)을 H.264 MP4 클립으로 추출 → output_path 반환.

    브라우저 재생을 위해 H.264(libx264)로 인코딩.
    av 라이브러리 사용 → ffmpeg 바이너리 불필요.
    padding_sec: 시작 전·종료 후 여유 구간(맥락 파악용).
    """
    import av as _av  # PyAV — H.264 인코딩 (브라우저 재생 가능)

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일 없음: {video_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _av.open(str(path)) as inp:
        v_stream = inp.streams.video[0]
        fps_frac = v_stream.average_rate
        fps = float(fps_frac) if fps_frac else 25.0
        total_sec = float(v_stream.duration * v_stream.time_base) if v_stream.duration else 0.0

        clip_start = max(0.0, start_sec - padding_sec)
        clip_end = min(total_sec, end_sec + padding_sec) if total_sec > 0 else end_sec + padding_sec

        # keyframe 직전으로 seek
        inp.seek(int(clip_start * _av.time_base ** -1) if hasattr(_av, 'time_base') else
                 int(clip_start / float(v_stream.time_base)),
                 stream=v_stream)

        with _av.open(str(out_path), "w", format="mp4") as out:
            out_stream = out.add_stream("libx264", rate=fps_frac or 25)
            out_stream.width = v_stream.width
            out_stream.height = v_stream.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "23", "preset": "fast", "movflags": "+faststart"}

            for packet in inp.demux(v_stream):
                if packet.dts is None:
                    continue
                pkt_sec = float(packet.pts * v_stream.time_base) if packet.pts is not None else 0.0
                if pkt_sec < clip_start - 0.5:
                    continue
                if pkt_sec > clip_end + 0.1:
                    break
                for frame in packet.decode():
                    f_sec = float(frame.pts * v_stream.time_base) if frame.pts is not None else 0.0
                    if f_sec < clip_start:
                        continue
                    if f_sec > clip_end:
                        break
                    for enc_pkt in out_stream.encode(frame):
                        out.mux(enc_pkt)

            # flush
            for enc_pkt in out_stream.encode():
                out.mux(enc_pkt)

    return str(out_path)


def crop_person(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float = 0.08,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right <= left or bottom <= top:
        return np.empty((0, 0, frame.shape[2]), dtype=frame.dtype)
    return frame[top:bottom, left:right].copy()

