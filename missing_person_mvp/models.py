from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import config


@dataclass(slots=True)
class PersonProfile:
    person_id: str
    name: str
    contact: str
    image_paths: list[str]
    gait_video_path: Optional[str] = None
    view_image_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PersonEmbedding:
    person_id: str
    vector: list[float]
    image_count: int


@dataclass(slots=True)
class SearchRequest:
    person_id: str
    video_paths: list[str]
    start_time: str
    end_time: str
    threshold: float = config.REID_THRESHOLD
    sample_every: int = config.SAMPLE_EVERY
    outfit_desc: str = ""    # 착장 설명 — 색상 보정에 사용
    tracker: str = ""        # "" → config.YOLO_TRACKER 기본값 사용
    use_gait: bool = False   # 보행 특징 보조 활성화
    gait_weight: float = 0.3 # Gait 기여 비율 (0 = Re-ID만, 1 = Gait만)


@dataclass(slots=True)
class DetectionHit:
    timestamp: str                          # 영상 내 경과 시각 HH:MM:SS
    similarity: float
    track_id: int
    camera_id: str
    crop_image_path: str
    bounding_box: tuple[int, int, int, int]
    recording_datetime: Optional[str] = None  # 실제 녹화 시각 "YYYY-MM-DD HH:MM:SS"
    tracking_end: Optional[str] = None        # 트래킹 마지막 등장 HH:MM:SS (클립 종료 기준)


@dataclass(slots=True)
class TrackFeature:
    """2차 검색(가림 대응) 재점수용 per-track 중간 특징.

    1차 분석 때 계산된 값을 보관 → 영상 재처리 없이 즉시 재점수.
    """
    track_id: int
    camera_id: str
    timestamp: str
    crop_image_path: str
    bounding_box: tuple[int, int, int, int]
    primary_score: float                       # 1차 최종 점수
    part_bank: dict                            # {part: np.ndarray(512)} 후보 대표벡터
    gait_frames: list                          # [(8,2) ...] 누적 정규화 골격
    visible_parts: list[str]                   # 종합 가시 부위
    occlusion_ratio: float                     # 0(완전보임)~1(많이가림)
    tracking_end: Optional[str] = None
    recording_datetime: Optional[str] = None


@dataclass(slots=True)
class AnalysisReport:
    request: SearchRequest
    hits: list[DetectionHit] = field(default_factory=list)
    summary: str = ""
    diagnostics: list[str] = field(default_factory=list)  # 단계별 진단 로그 (UI 표시용)
    candidates: list[DetectionHit] = field(default_factory=list)  # 임계값 무관 유사도 상위 후보
    track_features: list = field(default_factory=list)  # 2차 검색 재점수용 (list[TrackFeature])

    def sorted_hits(self) -> list[DetectionHit]:
        return sorted(self.hits, key=lambda hit: (hit.timestamp, hit.camera_id, hit.track_id))
