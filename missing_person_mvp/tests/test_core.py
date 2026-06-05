from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from missing_person_mvp import config
from missing_person_mvp.models import AnalysisReport, DetectionHit, PersonProfile, SearchRequest
from missing_person_mvp.pipeline import analyze_vod, generate_image, register, report as report_mod
from missing_person_mvp.utils import embedding
from missing_person_mvp.utils.track import TrackInfo
from missing_person_mvp.utils.video import crop_person, hms_to_seconds, iter_frames


@pytest.fixture()
def isolated_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    db_dir = data_dir / "db"
    results_dir = data_dir / "results"
    footage_dir = data_dir / "footage"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DB_DIR", db_dir)
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "FOOTAGE_DIR", footage_dir)
    monkeypatch.setattr(config, "PERSONS_JSON", db_dir / "persons.json")
    monkeypatch.setattr(config, "EMBEDDINGS_NPZ", db_dir / "embeddings.npz")
    monkeypatch.setattr(config, "FAISS_INDEX", db_dir / "faiss.index")
    config.ensure_dirs()
    return data_dir


def _make_image(path: Path, color=(120, 90, 60)) -> Path:
    Image.new("RGB", (64, 128), color).save(path)
    return path


def _make_video(path: Path, frame_count: int = 6) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        3.0,
        (64, 64),
    )
    for idx in range(frame_count):
        frame = np.full((64, 64, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_hms_to_seconds_validation():
    assert hms_to_seconds("01:02:03") == 3723
    with pytest.raises(ValueError):
        hms_to_seconds("01:99:00")


def test_crop_person_clips_to_frame():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_person(frame, (-10, -10, 15, 15), margin=0)
    assert crop.shape == (15, 15, 3)


def test_embedding_representative_is_l2_normalized():
    vectors = np.ones((2, config.EMBEDDING_DIM), dtype="float32")
    rep = embedding.representative(vectors)
    assert rep.shape == (config.EMBEDDING_DIM,)
    assert np.linalg.norm(rep) == pytest.approx(1.0)


def test_register_list_get_delete_flow(isolated_data, tmp_path):
    image_path = _make_image(tmp_path / "person.jpg")
    profile = PersonProfile(
        person_id="p001",
        name="홍길동",
        contact="010-0000-0000",
        image_paths=[str(image_path)],
    )
    result = register.register_person(profile)
    assert result.person_id == "p001"
    assert result.image_count == 1
    assert register.list_persons()[0]["name"] == "홍길동"
    assert register.get_embedding("p001").shape == (config.EMBEDDING_DIM,)
    assert register.delete_person("p001") is True
    assert register.list_persons() == []


def test_generate_image_falls_back_from_gemini_to_fashn(isolated_data, tmp_path, monkeypatch):
    image_path = _make_image(tmp_path / "base.jpg")
    register.register_person(
        PersonProfile("p002", "김테스트", "010", [str(image_path)])
    )

    def fail_gemini(base_path, outfit_desc):
        raise generate_image.ImageGenerationError("gemini failed")

    def ok_fashn(base_path, outfit_desc):
        return Image.new("RGB", (128, 256), (10, 20, 30))

    monkeypatch.setattr(generate_image, "_gemini_generate", fail_gemini)
    monkeypatch.setattr(generate_image, "_fashn_generate", ok_fashn)
    image, path, errors = generate_image.generate_outfit_image("p002", "파란 점퍼")
    assert image.size == (128, 256)
    assert Path(path).exists()
    assert "gemini" in " ".join(errors).lower()  # gemini 실패 에러가 errors에 기록됨


def test_generate_image_flux_uses_three_view_paths(isolated_data, tmp_path, monkeypatch):
    view_paths = {
        "front": str(_make_image(tmp_path / "front.jpg", (120, 90, 60))),
        "side": str(_make_image(tmp_path / "side.jpg", (110, 90, 60))),
        "back": str(_make_image(tmp_path / "back.jpg", (100, 90, 60))),
    }
    register.register_person(
        PersonProfile(
            person_id="p_flux",
            name="Flux",
            contact="010",
            image_paths=[],
            view_image_paths=view_paths,
        )
    )
    seen = {}

    def fake_flux(base_path, outfit_desc):
        seen["base_path"] = base_path
        seen["outfit_desc"] = outfit_desc
        return Image.new("RGB", (128, 256), (10, 40, 90))

    monkeypatch.setattr(generate_image, "_flux_generate", fake_flux)
    image, path, errors = generate_image.generate_outfit_image("p_flux", "blue jacket", ["flux"])
    assert image.size == (128, 256)
    assert seen["base_path"] == view_paths
    assert errors == []
    assert Path(path).exists()


def test_generate_image_flux_single_provider_reports_missing_views(isolated_data, tmp_path):
    image_path = _make_image(tmp_path / "base.jpg")
    register.register_person(PersonProfile("p_no_views", "No Views", "010", [str(image_path)]))
    with pytest.raises(generate_image.ImageGenerationError, match="front/side/back"):
        generate_image.generate_outfit_image("p_no_views", "blue jacket", ["flux"])


def test_iter_frames_reads_requested_segment(tmp_path):
    video = _make_video(tmp_path / "sample.mp4")
    frames = list(iter_frames(str(video), "00:00:00", "00:00:01", sample_every=2))
    assert len(frames) >= 1
    assert frames[0][0].shape[:2] == (64, 64)


def test_analyze_multiple_cameras_sorts_hits_and_saves_crop(isolated_data, tmp_path, monkeypatch):
    video_a = _make_video(tmp_path / "cam_b.mp4")
    video_b = _make_video(tmp_path / "cam_a.mp4")
    query = np.zeros(config.EMBEDDING_DIM, dtype="float32")
    query[0] = 1.0
    part_bank = {part: query.copy() for part in ["full", "upper", "torso", "lower", "legs"]}

    monkeypatch.setattr(analyze_vod, "get_embedding", lambda person_id: query)
    monkeypatch.setattr(analyze_vod, "get_part_bank", lambda person_id: part_bank)
    monkeypatch.setattr(
        analyze_vod.track_util,
        "detect_persons",
        lambda frame, tracker="": [TrackInfo(track_id=7, bbox=(5, 5, 40, 55), conf=0.9)],
    )
    monkeypatch.setattr(
        analyze_vod.emb_util,
        "extract",
        lambda images: np.asarray([query for _ in images], dtype="float32"),
    )
    # sample_every=1, _EMBED_EVERY를 1로 패치해 작은 동영상에서도 임베딩이 실행되게
    monkeypatch.setattr(analyze_vod, "_EMBED_EVERY", 1)
    request = SearchRequest(
        person_id="p003",
        video_paths=[str(video_a), str(video_b)],
        start_time="00:00:00",
        end_time="00:00:02",
        threshold=0.9,
        sample_every=1,
    )
    result = analyze_vod.analyze_multiple_cameras(request)
    assert isinstance(result, AnalysisReport)
    assert [hit.camera_id for hit in result.hits] == sorted(hit.camera_id for hit in result.hits)


def test_report_handles_empty_hits(isolated_data):
    request = SearchRequest("p004", [], "00:00:00", "00:01:00")
    empty = AnalysisReport(request=request, hits=[], summary="없음")
    assert report_mod.build_timeline(empty) == []
    assert "추정할 수 없습니다" in report_mod.infer_movement([])
    output = report_mod.export_pdf(empty, str(config.RESULTS_DIR / "empty.pdf"))
    assert Path(output).exists()


def test_app_imports():
    import missing_person_mvp.app as app

    assert callable(app.build_app)


def test_streamlit_app_imports():
    import missing_person_mvp.streamlit_app  # noqa: F401 — import 오류만 검사


def test_visible_parts_from_keypoints():
    from missing_person_mvp.utils.parts import visible_parts_from_keypoints

    # 전신 keypoints (어깨+골반+무릎+발목 모두 visible)
    kp_conf = np.zeros(17, dtype="float32")
    for idx in [5, 6, 11, 12, 13, 14, 15, 16]:
        kp_conf[idx] = 0.9
    parts = visible_parts_from_keypoints(kp_conf)
    assert "full" in parts
    assert "upper" in parts

    # 상반신만 (어깨+골반만)
    kp_conf2 = np.zeros(17, dtype="float32")
    for idx in [5, 6, 11, 12]:
        kp_conf2[idx] = 0.9
    parts2 = visible_parts_from_keypoints(kp_conf2)
    assert "upper" in parts2
    assert "legs" not in parts2

    # 빈 keypoints → fallback
    assert visible_parts_from_keypoints(None) == ["full"]
    assert visible_parts_from_keypoints(np.zeros(17)) == ["full"]


def test_part_weighted_score():
    from missing_person_mvp.utils.parts import part_weighted_score

    vec = np.array([1.0, 0.0, 0.0], dtype="float32")
    candidate = {"upper": vec, "torso": vec}
    bank      = {"upper": vec, "torso": vec, "full": vec}
    score = part_weighted_score(candidate, bank, ["upper", "torso"])
    assert abs(score - 1.0) < 1e-5

    # 보이지 않는 part는 제외 — 0점 처리 안 함
    zero_vec = np.array([0.0, 1.0, 0.0], dtype="float32")
    candidate2 = {"upper": vec}
    bank2      = {"upper": vec, "lower": zero_vec}
    score2 = part_weighted_score(candidate2, bank2, ["upper"])
    assert abs(score2 - 1.0) < 1e-5


def test_secondary_search_rescore_weights_by_gait_data():
    """2차 검색: gait 데이터량(유효 골격 프레임 수)에 따라 w_gait이 정해지는지 검증.

    가림이 아니라 걸음 데이터량 기반 — 프레임 부족이면 gait 무시(부위 점수만).
    """
    from missing_person_mvp.models import TrackFeature
    from missing_person_mvp.pipeline.analyze_vod import secondary_search_rescore

    dim = config.EMBEDDING_DIM
    part_vec = np.zeros(dim, dtype="float32"); part_vec[0] = 1.0
    bank = {"full": part_vec, "upper": part_vec, "torso": part_vec,
            "lower": part_vec, "legs": part_vec}

    gait_emb = np.zeros(32, dtype="float32"); gait_emb[0] = 1.0

    def _tf(tid, n_frames):
        return TrackFeature(
            track_id=tid, camera_id="cam", timestamp="00:00:01",
            crop_image_path="", bounding_box=(0, 0, 10, 10),
            primary_score=0.5,
            part_bank={"full": part_vec},
            gait_frames=[np.ones((8, 2), dtype="float32") for _ in range(n_frames)],
            visible_parts=["full"], occlusion_ratio=0.5,  # 가림은 동일 → 영향 없어야
        )

    # 프레임 10장(<15) → w_gait=0 / 40장 → gait_conf=1.0 → w_gait=0.5
    res = secondary_search_rescore([_tf(1, 10), _tf(2, 40)], bank, None, gait_emb)
    by_tid = {r["feature"].track_id: r for r in res}
    assert by_tid[1]["w_gait"] == 0.0           # 데이터 부족 → gait 무시
    assert abs(by_tid[2]["w_gait"] - 0.5) < 1e-6  # 충분 → 최대 가중
    assert by_tid[1]["gait_frames_n"] == 10
    assert by_tid[2]["gait_frames_n"] == 40
    # 점수 내림차순 정렬
    scores = [r["secondary_score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    # gait_emb=None이면 w_gait=0 (부위 점수만)
    res_none = secondary_search_rescore([_tf(3, 40)], bank, None, None)
    assert res_none[0]["w_gait"] == 0.0


