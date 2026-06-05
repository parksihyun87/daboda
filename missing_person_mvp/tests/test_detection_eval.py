from __future__ import annotations

import textwrap

import pytest

from missing_person_mvp.scripts.evaluate_detection import (
    GTBox,
    Hit,
    _iou,
    evaluate_detection,
    hits_from_report,
    load_cvat_xml,
    load_mot,
)


_CVAT_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <track id="0" label="person_of_interest">
        <box frame="30" xtl="100" ytl="200" xbr="200" ybr="400" outside="0" occluded="0"/>
        <box frame="90" xtl="110" ytl="205" xbr="210" ybr="405" outside="0" occluded="0"/>
      </track>
    </annotations>
""")

_MOT_LINES = textwrap.dedent("""\
    30,1,100,200,100,200,1,-1,-1,-1
    90,1,110,205,100,200,1,-1,-1,-1
""")


def test_iou_exact_overlap():
    bbox = (0.0, 0.0, 100.0, 100.0)
    assert _iou(bbox, bbox) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert _iou(a, b) == pytest.approx(0.0)


def test_iou_partial():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection=5*10=50, union=100+100-50=150
    assert _iou(a, b) == pytest.approx(50.0 / 150.0)


def test_load_cvat_xml(tmp_path):
    xml_file = tmp_path / "test.xml"
    xml_file.write_text(_CVAT_XML, encoding="utf-8")
    boxes = load_cvat_xml(xml_file, fps=30.0, label="person_of_interest")
    assert len(boxes) == 2
    b0 = boxes[0]
    assert b0.gt_track_id == "0"
    assert b0.frame == 30
    assert b0.timestamp_sec == pytest.approx(1.0)   # 30/30 fps
    assert b0.bbox == (100.0, 200.0, 200.0, 400.0)
    assert b0.outside is False


def test_load_cvat_xml_wrong_label(tmp_path):
    xml_file = tmp_path / "test.xml"
    xml_file.write_text(_CVAT_XML, encoding="utf-8")
    boxes = load_cvat_xml(xml_file, fps=30.0, label="other_label")
    assert boxes == []


def test_load_mot(tmp_path):
    mot_file = tmp_path / "gt.txt"
    mot_file.write_text(_MOT_LINES, encoding="utf-8")
    boxes = load_mot(mot_file, fps=30.0)
    assert len(boxes) == 2
    b = boxes[0]
    assert b.gt_track_id == "1"
    assert b.frame == 30
    assert b.timestamp_sec == pytest.approx(1.0)
    # MOT: x,y,w,h=100,200,100,200 → x2=200, y2=400
    assert b.bbox == (100.0, 200.0, 200.0, 400.0)


def test_evaluate_perfect_match():
    gt = [GTBox(video_id="v", gt_track_id="0", frame=30,
                timestamp_sec=1.0, bbox=(100, 200, 200, 400))]
    hits = [Hit(timestamp_sec=1.0, track_id="7", camera_id="v",
                bbox=(100, 200, 200, 400), similarity=0.95)]
    result = evaluate_detection(gt, hits, time_tolerance_sec=2.0, iou_threshold=0.3)
    m = result["metrics"]
    assert m["track_recall"] == pytest.approx(1.0)
    assert m["false_hit_count"] == 0
    assert m["matched_track_count"] == 1


def test_evaluate_no_match():
    gt = [GTBox(video_id="v", gt_track_id="0", frame=30,
                timestamp_sec=1.0, bbox=(100, 200, 200, 400))]
    # hit at different time → false positive + missed track
    hits = [Hit(timestamp_sec=60.0, track_id="9", camera_id="v",
                bbox=(100, 200, 200, 400), similarity=0.8)]
    result = evaluate_detection(gt, hits, time_tolerance_sec=2.0, iou_threshold=0.3)
    m = result["metrics"]
    assert m["track_recall"] == pytest.approx(0.0)
    assert m["false_hit_count"] == 1
    assert len(result["missed_tracks"]) == 1


def test_hits_from_report():
    from missing_person_mvp.models import AnalysisReport, DetectionHit, SearchRequest

    request = SearchRequest(
        person_id="p001", video_paths=[], start_time="00:00:00", end_time="00:01:00"
    )
    det_hit = DetectionHit(
        timestamp="00:01:05",
        similarity=0.88,
        track_id=3,
        camera_id="cam1",
        crop_image_path="",
        bounding_box=(10, 20, 110, 220),
    )
    report = AnalysisReport(request=request, hits=[det_hit])
    hits = hits_from_report(report)
    assert len(hits) == 1
    h = hits[0]
    assert h.timestamp_sec == pytest.approx(65.0)   # 1*60 + 5
    assert h.similarity == pytest.approx(0.88)
    assert h.bbox == (10.0, 20.0, 110.0, 220.0)
    assert h.track_id == "3"
    assert h.camera_id == "cam1"
