"""CCTV 저장 영상 구간 분석 — Part-based Re-ID + Batch Embedding.

흐름:
1. iter_frames → YOLO-Pose → TrackInfo(bbox, keypoints_conf)
2. 후보마다: visible_parts_from_keypoints → part crop
3. track 단위로 part crop 누적 (EMBED_EVERY 프레임마다 batch 임베딩)
4. track 종료 후: part별 대표 벡터 → 등록자 bank와 비교
5. part_weighted_score ≥ threshold → DetectionHit 생성
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from missing_person_mvp import config
from missing_person_mvp.models import AnalysisReport, DetectionHit, SearchRequest, TrackFeature
from missing_person_mvp.pipeline.register import get_embedding, get_part_bank
from missing_person_mvp.utils import embedding as emb_util
from missing_person_mvp.utils import track as track_util
from missing_person_mvp.utils import color_score as color_util
from missing_person_mvp.utils.gait import gait_similarity, normalize_skeleton
from missing_person_mvp.utils.parts import (
    PARTS,
    crop_part,
    part_weighted_score,
    visible_parts_from_keypoints,
)
from missing_person_mvp.utils.video import (
    crop_person, hms_to_seconds, iter_frames,
    parse_recording_start, recording_datetime_str, seconds_to_hms,
)
from missing_person_mvp.vton.outfit_parser import OutfitSpec, parse_outfit

# 진행 콜백: (메시지, 진척률 0~1). 진척률 모르면 -1.
ProgressCallback = Callable[[str, float], None]

# track당 N 프레임마다 1회 batch 임베딩 (매 프레임 하지 않음)
_EMBED_EVERY = 8
# YOLO bbox 최소 면적 (너무 작은 후보 제외)
_MIN_BBOX_AREA = 32 * 32
# 후보로 인정할 최소 트랙 지속(샘플 프레임 등장 횟수) — 순간 오탐(전등 등) 제외
_MIN_TRACK_DETS = 5
# 세그먼트당 보관할 상위 후보 수 (임계값 무관)
_CAND_TOPN = 10
# UI에 보여줄 글로벌 상위 후보 수 (정답1+비정답3=4명, 재추적 ID 분리 감안해 여유 있게)
CANDIDATE_TOPN = 8


def _camera_id(video_path: str) -> str:
    return Path(video_path).stem


def _save_crop(crop: np.ndarray, camera_id: str, track_id: int, timestamp: str) -> str:
    config.ensure_dirs()
    safe_ts = timestamp.replace(":", "")
    path = config.RESULTS_DIR / f"{camera_id}_{track_id}_{safe_ts}.jpg"
    cv2.imwrite(str(path), crop)
    return str(path)


def _bbox_area(bbox: tuple) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def analyze_segment(
    video_path: str,
    start_time: str,
    end_time: str,
    query_embedding: np.ndarray,       # fallback: full-body 단일 벡터
    part_bank: dict[str, np.ndarray],  # {part: (512,)} 등록자 bank
    threshold: float = config.REID_THRESHOLD,
    sample_every: int = config.SAMPLE_EVERY,
    progress_cb: ProgressCallback | None = None,
    outfit_spec: OutfitSpec | None = None,   # 색상 보정용 착장 스펙
    tracker: str = "",                       # "" → config.YOLO_TRACKER
    gait_embedding: np.ndarray | None = None,# 등록된 gait 임베딩
    gait_weight: float = 0.3,                # gait 기여 비율
    part_bank_multi_view: dict[str, dict[str, np.ndarray]] | None = None,  # {view: {part: embedding}}
    diag: list[str] | None = None,           # 단계별 진단 로그 수집(UI 표시용)
    candidates_out: list[DetectionHit] | None = None,  # 임계값 무관 상위 후보 수집
    track_features_out: list | None = None,  # 2차 검색용 TrackFeature 수집
) -> list[DetectionHit]:
    camera_id = _camera_id(video_path)

    def _log(msg: str) -> None:
        if diag is not None:
            diag.append(f"[{camera_id}] {msg}")

    # 단계 0: 영상 열림 확인 + 메타데이터
    cap0 = cv2.VideoCapture(video_path)
    opened = cap0.isOpened()
    v_fps = cap0.get(cv2.CAP_PROP_FPS)
    v_frames = cap0.get(cv2.CAP_PROP_FRAME_COUNT)
    cap0.release()
    if not opened:
        _log(f"❌ 영상 열기 실패: {video_path}")
        return []
    # 파일명에서 CCTV 녹화 시작 시각 파싱 (실패하면 None → 실제 시각 미표시)
    _recording_start = parse_recording_start(video_path)
    _rec_str = _recording_start.strftime("%Y-%m-%d %H:%M:%S") if _recording_start else "불명"
    _log(f"① 영상 OK: fps={v_fps:.1f} frames={int(v_frames)} 경로={Path(video_path).name} 녹화시작={_rec_str}")
    _log(f"   bank parts={list(part_bank.keys()) or '없음'} | 3면={'있음' if part_bank_multi_view else '없음'} "
         f"| 색상보정={'ON' if outfit_spec else 'OFF'} | 임계값={threshold} | tracker={tracker or config.YOLO_TRACKER}")

    # 진척률 계산용 구간(초). 종료시각이 영상 길이를 넘으면 영상 끝으로 보정.
    _start_sec = hms_to_seconds(start_time)
    _video_dur = (v_frames / v_fps) if v_fps > 0 else 0.0
    _end_sec = hms_to_seconds(end_time)
    if _video_dur > 0:
        _end_sec = min(_end_sec, _video_dur)
    _span = max(1.0, _end_sec - _start_sec)

    # 진단 카운터
    n_frames = 0
    n_dets_raw = 0
    n_dets_kept = 0

    # track_id → {part: [feat_vectors]}
    track_part_feats: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    # track_id → best crop 메타
    track_meta: dict[int, dict] = {}
    # track_id → 프레임 카운터 (batch 간격 조절용)
    track_frame_cnt: dict[int, int] = defaultdict(int)
    # 배치 대기 큐: [(track_id, part, crop_bgr)]
    pending_batch: list[tuple[int, str, np.ndarray]] = []
    # 색상 보정용: track_id → {part: 대표 crop BGR} (최초 1장만 저장)
    track_part_crop: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    # gait 보조: track_id → 정규화된 골격 프레임 목록
    track_gait_frames: dict[int, list[np.ndarray]] = defaultdict(list)
    # 2차 검색용: track_id → 프레임별 가시 부위 수 (가림 비율 계산)
    track_vis_counts: dict[int, list[int]] = defaultdict(list)

    def _flush_batch():
        if not pending_batch:
            return
        crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, _, c in pending_batch]
        feats = emb_util.extract(crops_rgb)  # (N, 512) batch
        for (tid, part, _), feat in zip(pending_batch, feats):
            track_part_feats[tid][part].append(feat)
        pending_batch.clear()

    for frame, current_sec in iter_frames(video_path, start_time, end_time, sample_every):
        n_frames += 1
        if progress_cb:
            frac = min(1.0, max(0.0, (current_sec - _start_sec) / _span))
            progress_cb(f"{camera_id} {seconds_to_hms(current_sec)} 분석 중", frac)

        persons = track_util.detect_persons(frame, tracker)
        n_dets_raw += len(persons)
        # 1차 필터: 너무 작은 bbox 제외
        persons = [p for p in persons if _bbox_area(p.bbox) >= _MIN_BBOX_AREA]
        n_dets_kept += len(persons)

        for info in persons:
            tid = info.track_id
            track_frame_cnt[tid] += 1

            # visible parts 결정
            vis_parts = visible_parts_from_keypoints(info.keypoints_conf)

            # ── 매 샘플 프레임 누적 (cheap, gait·2차 검색용) ──
            # EMBED_EVERY로 솎으면 골격이 너무 드물어져 gait 최소 프레임(15)을 못 채움.
            # 가시부위 수·정규화 골격은 연산이 가벼우므로 매 샘플 프레임 누적.
            track_vis_counts[tid].append(len(vis_parts))
            if info.keypoints_xy is not None:
                normed = normalize_skeleton(info.keypoints_xy)
                if normed is not None:
                    track_gait_frames[tid].append(normed)

            # EMBED_EVERY 간격 — part crop OSNet 임베딩(비쌈)만 솎음
            if track_frame_cnt[tid] % _EMBED_EVERY != 0:
                continue

            # best crop 메타 갱신 (full crop 기준)
            full_crop = crop_person(frame, info.bbox)
            if full_crop.size == 0:
                continue

            # part crop 큐에 추가 + 색상 보정용 첫 crop 저장
            for part in vis_parts:
                part_crop = crop_part(frame, info.bbox, part)
                if part_crop.size > 0:
                    pending_batch.append((tid, part, part_crop))
                    if part not in track_part_crop[tid]:
                        track_part_crop[tid][part] = part_crop.copy()

            # 메타: 첫 등장 시각 + best crop 저장
            if tid not in track_meta:
                track_meta[tid] = {
                    "timestamp": seconds_to_hms(current_sec),
                    "offset_sec": current_sec,          # 실제 시각 계산용
                    "crop": full_crop.copy(),
                    "bbox": info.bbox,
                }
            # 마지막 등장 시각 갱신 (클립 추출 구간 계산용)
            track_meta[tid]["last_seen_sec"] = current_sec

        # 배치가 일정 크기 이상이면 즉시 flush
        if len(pending_batch) >= 32:
            _flush_batch()

    _flush_batch()

    # 단계 ②③: 프레임/탐지 요약
    _log(f"② 프레임 {n_frames}개 처리 · YOLO 사람탐지 {n_dets_raw}개(필터후 {n_dets_kept}개) "
         f"· 추적된 track {len(track_part_feats)}개")
    if n_frames == 0:
        _log("❌ 처리된 프레임이 0 — 구간(시작/종료 시각)이 영상 범위 밖이거나 영상이 안 읽힘")
    elif n_dets_kept == 0:
        _log("❌ 사람이 한 명도 탐지되지 않음 — YOLO 미작동 또는 bbox가 너무 작음")

    # track별 최종 판정
    hits: list[DetectionHit] = []
    all_scores: list[tuple[float, int]] = []  # (score, track_id) — 임계값 미만 포함 진단용
    track_rescore: dict[int, dict] = {}       # 2차 검색용: tid → {bank, visible, occlusion}
    for tid, part_feats in track_part_feats.items():
        # part별 대표 벡터 계산
        candidate_bank: dict[str, np.ndarray] = {}
        for part, feats in part_feats.items():
            if feats:
                candidate_bank[part] = emb_util.representative(np.stack(feats))

        if not candidate_bank:
            continue

        visible = list(candidate_bank.keys())

        # 2차 검색용: 가림 비율 (가시 부위 평균 / 전체 5부위)
        _vc = track_vis_counts.get(tid, [])
        _occ = 1.0 - (float(np.mean(_vc)) / len(PARTS)) if _vc else 0.0
        track_rescore[tid] = {
            "candidate_bank": candidate_bank,
            "visible": visible,
            "occlusion_ratio": max(0.0, min(1.0, _occ)),
        }

        # part bank가 있으면 part-weighted score, 없으면 full 단일 비교
        if part_bank:
            # 다중 view bank가 있으면 max similarity 사용
            if part_bank_multi_view:
                scores_by_view = []
                for view, view_bank in part_bank_multi_view.items():
                    if view_bank:  # empty dict가 아니면
                        view_score = part_weighted_score(candidate_bank, view_bank, visible)
                        scores_by_view.append(view_score)
                if scores_by_view:
                    score = max(scores_by_view)
                else:
                    # fallback: 기존 bank 사용
                    score = part_weighted_score(candidate_bank, part_bank, visible)
            else:
                score = part_weighted_score(candidate_bank, part_bank, visible)
        else:
            # fallback: full embedding 단일 비교
            full_feat = candidate_bank.get("full")
            if full_feat is None:
                full_feat = emb_util.representative(
                    np.stack([v for vecs in part_feats.values() for v in vecs])
                )
            score = float(np.dot(query_embedding, full_feat))

        # 색상 보정 boost (outfit_spec이 있을 때만)
        if outfit_spec is not None and tid in track_part_crop:
            boost = color_util.part_color_boost(track_part_crop[tid], outfit_spec)
            score = score * (1.0 + color_util.COLOR_ALPHA * boost)

        # Gait 보조 (등록된 gait 임베딩이 있을 때 가중 결합)
        if gait_embedding is not None and track_gait_frames.get(tid):
            g_score = gait_similarity(gait_embedding, track_gait_frames[tid])
            score = (1.0 - gait_weight) * score + gait_weight * g_score

        all_scores.append((float(score), tid))

    # 크롭은 track당 1회만 저장 (hit·후보 공용 캐시)
    _crop_cache: dict[int, str] = {}

    def _mk_hit(score: float, tid: int) -> DetectionHit:
        meta = track_meta[tid]
        if tid not in _crop_cache:
            _crop_cache[tid] = _save_crop(meta["crop"], camera_id, tid, meta["timestamp"])
        return DetectionHit(
            timestamp=meta["timestamp"],
            similarity=round(score, 4),
            track_id=tid,
            camera_id=camera_id,
            crop_image_path=_crop_cache[tid],
            bounding_box=tuple(meta["bbox"]),
            recording_datetime=recording_datetime_str(_recording_start, meta["offset_sec"]),
            tracking_end=seconds_to_hms(meta["last_seen_sec"]) if "last_seen_sec" in meta else None,
        )

    # 임계값 통과 = 탐지(hits)
    hits = [_mk_hit(s, t) for s, t in all_scores if s >= threshold and t in track_meta]

    # 임계값 무관 상위 후보 (세그먼트당 최대 N개) → 호출측에서 글로벌 상위 N 선별.
    # 몇 프레임만 스친 순간 오탐(전등 등)은 _MIN_TRACK_DETS 미만이라 후보에서 제외.
    if candidates_out is not None:
        _persistent = [
            (s, t) for s, t in all_scores
            if t in track_meta and track_frame_cnt[t] >= _MIN_TRACK_DETS
        ]
        n_dropped = len([1 for s, t in all_scores if t in track_meta]) - len(_persistent)
        for s, t in sorted(_persistent, reverse=True)[:_CAND_TOPN]:
            candidates_out.append(_mk_hit(s, t))
        if n_dropped:
            _log(f"   (후보에서 순간 오탐 {n_dropped}개 제외: {_MIN_TRACK_DETS}프레임 미만 트랙)")

    # 2차 검색용 TrackFeature 수집 (persistent 트랙 전체 — 가림으로 밀린 후보 구출용)
    if track_features_out is not None:
        for s, t in all_scores:
            if t not in track_meta or track_frame_cnt[t] < _MIN_TRACK_DETS:
                continue
            rs = track_rescore.get(t)
            if rs is None:
                continue
            meta = track_meta[t]
            if t not in _crop_cache:
                _crop_cache[t] = _save_crop(meta["crop"], camera_id, t, meta["timestamp"])
            track_features_out.append(TrackFeature(
                track_id=t,
                camera_id=camera_id,
                timestamp=meta["timestamp"],
                crop_image_path=_crop_cache[t],
                bounding_box=tuple(meta["bbox"]),
                primary_score=round(float(s), 4),
                part_bank=rs["candidate_bank"],
                gait_frames=list(track_gait_frames.get(t, [])),
                visible_parts=rs["visible"],
                occlusion_ratio=round(rs["occlusion_ratio"], 3),
                recording_datetime=recording_datetime_str(_recording_start, meta["offset_sec"]),
                tracking_end=seconds_to_hms(meta["last_seen_sec"]) if "last_seen_sec" in meta else None,
            ))

    # 단계 ④: 점수 분포 (임계값 미만 포함) — 왜 0건인지 진단
    if all_scores:
        top = sorted(all_scores, reverse=True)[:5]
        _log("④ track 점수 상위(임계값 미만 포함): "
             + ", ".join(f"track{t}={s:.3f}" for s, t in top))
        best = top[0][0]
        if not hits:
            _log(f"❌ 임계값 {threshold} 통과 0건. 최고점 {best:.3f}. "
                 + ("색상 보정을 켜거나 " if outfit_spec is None else "")
                 + f"임계값을 {max(0.1, best - 0.03):.2f} 이하로 낮춰보세요. (아래 상위 후보 참고)")
    else:
        _log("④ 점수 계산된 track 0개 (임베딩 추출 실패 가능)")
    _log(f"⑤ 최종 탐지 {len(hits)}건 (임계값 {threshold} 통과)")

    return sorted(hits, key=lambda h: (h.timestamp, h.track_id))


def secondary_search_rescore(
    track_features: list,
    part_bank: dict[str, np.ndarray],
    part_bank_multi_view: dict[str, dict[str, np.ndarray]] | None,
    gait_emb: np.ndarray,
) -> list[dict]:
    """2차 검색: 외형 독립 재검증 (걸음걸이 + 체형). 영상 재처리 없음.

    색상·옷과 무관한 gait + 가시부위 Re-ID로 1차 결과를 교차검증한다.
    gait 가중치는 *깨끗한 걸음 데이터가 얼마나 잡혔나*로 결정 (가림 기반 아님).

    각 트랙:
      part_score = 보이는 부위만 코사인 (3면이면 view별 max)
      gait_score = 등록 gait vs 트랙 골격
        n_gait < 15 → w_gait=0 (부위 점수만), 아니면 0.5*min(1, n_gait/40)
      secondary = (1 - w_gait)*part + w_gait*gait

    반환: [{feature, secondary_score, part_score, gait_score,
            w_gait, gait_frames_n}] — 점수 내림차순.
    """
    from missing_person_mvp.utils.gait import _MIN_FRAMES

    results: list[dict] = []
    for tf in track_features:
        bank = tf.part_bank
        visible = tf.visible_parts or list(bank.keys())

        # 부위 점수 (3면 있으면 view별 max)
        if part_bank_multi_view:
            view_scores = [
                part_weighted_score(bank, vb, visible)
                for vb in part_bank_multi_view.values() if vb
            ]
            part_score = max(view_scores) if view_scores else part_weighted_score(bank, part_bank, visible)
        else:
            part_score = part_weighted_score(bank, part_bank, visible)

        # gait 점수 + 데이터량 기반 신뢰도
        n_gait = len(tf.gait_frames)
        gait_score = gait_similarity(gait_emb, tf.gait_frames) if gait_emb is not None else 0.0
        if n_gait < _MIN_FRAMES or gait_emb is None:
            w_gait = 0.0          # 깨끗한 걸음 데이터 부족 → gait 무시, 부위 점수만
        else:
            w_gait = 0.5 * min(1.0, n_gait / 40.0)

        secondary = (1.0 - w_gait) * part_score + w_gait * gait_score

        results.append({
            "feature": tf,
            "secondary_score": round(float(secondary), 4),
            "part_score": round(float(part_score), 4),
            "gait_score": round(float(gait_score), 4),
            "w_gait": round(float(w_gait), 3),
            "gait_frames_n": int(n_gait),
        })

    results.sort(key=lambda r: r["secondary_score"], reverse=True)
    return results


def _build_summary(request: SearchRequest, hits: list[DetectionHit]) -> str:
    if not hits:
        return f"{request.start_time}~{request.end_time} 구간에서 유사 인물이 탐지되지 않았습니다."
    ordered = sorted(hits, key=lambda h: h.timestamp)
    best = max(ordered, key=lambda h: h.similarity)
    cameras = " -> ".join(dict.fromkeys(h.camera_id for h in ordered))
    return (
        f"총 {len(hits)}건 탐지. "
        f"최초 {ordered[0].timestamp}({ordered[0].camera_id}), "
        f"마지막 {ordered[-1].timestamp}({ordered[-1].camera_id}), "
        f"최고 유사도 {best.similarity:.3f}({best.camera_id}). "
        f"이동 추정: {cameras}"
    )


def analyze_multiple_cameras(
    request: SearchRequest,
    progress_cb: ProgressCallback | None = None,
) -> AnalysisReport:
    query_embedding = get_embedding(request.person_id)
    part_bank = get_part_bank(request.person_id)
    # 3면 뷰별 임베딩이 있으면 우선 사용 (max similarity 매칭용)
    from missing_person_mvp.pipeline.register import get_part_bank_multi_view
    part_bank_multi_view = get_part_bank_multi_view(request.person_id)
    outfit_spec: OutfitSpec | None = parse_outfit(request.outfit_desc) if request.outfit_desc.strip() else None
    tracker = request.tracker or config.YOLO_TRACKER

    gait_emb: np.ndarray | None = None
    if request.use_gait:
        from missing_person_mvp.pipeline.register import get_gait_embedding
        gait_emb = get_gait_embedding(request.person_id)

    all_hits: list[DetectionHit] = []
    all_candidates: list[DetectionHit] = []
    all_track_features: list = []
    diagnostics: list[str] = []
    # 준비 단계 진단: 임베딩/뱅크 상태
    diagnostics.append(
        f"⓪ 준비: person={request.person_id} | query_emb={'OK' if query_embedding is not None else '없음'} "
        f"| part_bank={list(part_bank.keys()) or '없음'} | 3면뱅크={'있음' if part_bank_multi_view else '없음'} "
        f"| 색상보정={'ON('+request.outfit_desc+')' if outfit_spec else 'OFF'} | gait={'ON' if gait_emb is not None else 'OFF'}"
    )
    if outfit_spec is not None:
        diagnostics.append(
            f"   색상파싱: upper={outfit_spec.upper_color} lower={outfit_spec.lower_color} "
            f"shoe={outfit_spec.shoe_color}"
        )
    max_workers = min(config.MAX_WORKERS, max(1, len(request.video_paths)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                analyze_segment,
                path,
                request.start_time,
                request.end_time,
                query_embedding,
                part_bank,
                request.threshold,
                request.sample_every,
                progress_cb,
                outfit_spec,
                tracker,
                gait_emb,
                request.gait_weight,
                part_bank_multi_view,
                diagnostics,         # diag — 스레드에서 직접 append
                all_candidates,      # candidates_out — 스레드에서 직접 append
                all_track_features,  # track_features_out — 2차 검색용
            ): path
            for path in request.video_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                all_hits.extend(future.result())
            except Exception as exc:
                import traceback
                diagnostics.append(
                    f"❌ [{Path(path).name}] 분석 중 예외 발생: {exc}\n{traceback.format_exc()}"
                )

    all_hits.sort(key=lambda h: (h.timestamp, h.camera_id, h.track_id))
    # 글로벌 유사도 상위 후보 N개 (임계값 무관, 중복 track 제거)
    seen_keys: set = set()
    top_candidates: list[DetectionHit] = []
    for c in sorted(all_candidates, key=lambda h: h.similarity, reverse=True):
        key = (c.camera_id, c.track_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        top_candidates.append(c)
        if len(top_candidates) >= CANDIDATE_TOPN:
            break
    if top_candidates:
        diagnostics.append(
            "⑥ 유사도 상위 후보(임계값 무관): "
            + ", ".join(f"{c.similarity*100:.1f}%@{c.timestamp}" for c in top_candidates)
        )

    return AnalysisReport(
        request=request,
        hits=all_hits,
        summary=_build_summary(request, all_hits),
        diagnostics=diagnostics,
        candidates=top_candidates,
        track_features=all_track_features,
    )
