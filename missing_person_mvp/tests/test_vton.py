from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from missing_person_mvp.scripts.evaluate_generation import evaluate
from missing_person_mvp.utils.viewpoint import VIEW_BACK, VIEW_FRONT, VIEW_SIDE, classify_viewpoint
from missing_person_mvp.vton import flux
from missing_person_mvp.vton.flux import (
    FluxGenerationError,
    create_clothing_mask,
    create_contact_sheet,
    generate_flux_views,
    validate_three_view_paths,
)
from missing_person_mvp.vton.garment_reference import create_garment_reference
from missing_person_mvp.vton.outfit_parser import parse_outfit


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (320, 480), color).save(path)
    return path


def test_parse_korean_outfit_text():
    spec = parse_outfit("파란 점퍼, 검은 바지, 흰 운동화")
    assert "blue" in spec.colors
    assert "black" in spec.colors
    assert "jacket" in spec.items
    assert "pants" in spec.items
    assert spec.category == "full_body"


def test_create_garment_reference_from_text(tmp_path):
    image, path, spec = create_garment_reference("파란 점퍼 검은 바지", tmp_path)
    assert image.size == (768, 1024)
    assert Path(path).exists()
    assert spec.primary_color() == "blue"


def test_viewpoint_heuristic():
    front = np.zeros(17, dtype="float32")
    front[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = 0.9
    assert classify_viewpoint(front) == VIEW_FRONT

    side = np.zeros(17, dtype="float32")
    side[[5, 7, 9, 11, 13, 15]] = 0.9
    side[[6, 8, 10, 12, 14, 16]] = 0.1
    assert classify_viewpoint(side) == VIEW_SIDE

    back = np.zeros(17, dtype="float32")
    back[[5, 6, 11, 12, 13, 14, 15, 16]] = 0.8
    assert classify_viewpoint(back) == VIEW_BACK


def test_flux_placeholder_three_view_generation(tmp_path):
    person = _image(tmp_path / "person.jpg", (120, 100, 90))
    images, sheet_path = generate_flux_views(str(person), "blue jacket", tmp_path, allow_placeholder=True)
    assert sorted(images) == ["back", "front", "side"]
    assert Path(sheet_path).exists()
    sheet = create_contact_sheet(images)
    assert sheet.size[0] > sheet.size[1]


def test_flux_requires_complete_three_view_paths(tmp_path):
    front = _image(tmp_path / "front.jpg", (120, 100, 90))
    with np.testing.assert_raises(FluxGenerationError):
        validate_three_view_paths({"front": str(front)})


def test_flux_three_view_mapping_placeholder(tmp_path):
    paths = {
        "front": str(_image(tmp_path / "front.jpg", (120, 100, 90))),
        "side": str(_image(tmp_path / "side.jpg", (110, 100, 90))),
        "back": str(_image(tmp_path / "back.jpg", (100, 100, 90))),
    }
    images, sheet_path = generate_flux_views(paths, "blue jacket", tmp_path, allow_placeholder=True)
    assert sorted(images) == ["back", "front", "side"]
    assert Path(sheet_path).exists()


def test_clothing_mask_protects_face_region(monkeypatch):
    image = Image.new("RGB", (768, 1024), (150, 130, 110))
    monkeypatch.setattr(flux, "_detect_person_bbox_and_keypoints", lambda img: ((180, 80, 590, 980), None))
    monkeypatch.setattr(flux, "_detect_face_boxes", lambda img: [(320, 110, 450, 250)])
    mask = create_clothing_mask(image)
    arr = np.array(mask)
    assert arr[170, 385] < 32
    assert arr[560, 385] > 180


def test_clothing_mask_falls_back_without_detectors(monkeypatch):
    image = Image.new("RGB", (768, 1024), (150, 130, 110))
    monkeypatch.setattr(flux, "_detect_person_bbox_and_keypoints", lambda img: (None, None))
    monkeypatch.setattr(flux, "_detect_face_boxes", lambda img: [])
    mask = create_clothing_mask(image)
    arr = np.array(mask)
    assert arr.max() > 180
    assert arr[40, 384] < 32


def test_evaluate_generation_outputs_files(tmp_path):
    person = _image(tmp_path / "person.jpg", (160, 130, 110))
    gemini = _image(tmp_path / "gemini.jpg", (40, 90, 190))
    flux = _image(tmp_path / "flux.jpg", (30, 80, 180))
    out_dir = tmp_path / "eval"
    result = evaluate(
        person_image=str(person),
        outfit_desc="파란 점퍼",
        gemini_image=str(gemini),
        flux_image=str(flux),
        person_id="p001",
        output_dir=out_dir,
    )
    assert result["winner"] in {"gemini", "flux"}
    assert (out_dir / "scores.json").exists()
    assert (out_dir / "scores.csv").exists()
    assert (out_dir / "comparison.jpg").exists()
