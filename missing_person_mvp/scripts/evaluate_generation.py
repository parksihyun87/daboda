from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from missing_person_mvp import config
from missing_person_mvp.utils import embedding as emb_util
from missing_person_mvp.utils.parts import build_part_bank
from missing_person_mvp.vton.outfit_parser import color_rgb, parse_outfit


WEIGHTS = {
    "prompt_color_score": 0.25,
    "identity_preservation_score": 0.20,
    "body_integrity_score": 0.20,
    "artifact_score": 0.15,
    "reid_utility_score": 0.20,
}


def _load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _arr(image: Image.Image, size: tuple[int, int] = (384, 512)) -> np.ndarray:
    return np.asarray(image.resize(size), dtype="float32")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _upper_body(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((0, 0, w, int(h * 0.55)))


def _central_body(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((int(w * 0.18), int(h * 0.20), int(w * 0.82), int(h * 0.88)))


def prompt_color_score(image: Image.Image, outfit_desc: str) -> float:
    spec = parse_outfit(outfit_desc)
    if not spec.colors:
        return 0.5
    target = np.asarray(color_rgb(spec.primary_color()), dtype="float32")
    body = _arr(_central_body(image))
    distances = np.linalg.norm(body.reshape(-1, 3) - target[None, :], axis=1)
    close_ratio = float(np.mean(distances < 90.0))
    mean_distance = float(np.linalg.norm(body.reshape(-1, 3).mean(axis=0) - target))
    return _clip01(close_ratio * 0.75 + (1.0 - mean_distance / 255.0) * 0.25)


def identity_preservation_score(person: Image.Image, generated: Image.Image) -> float:
    base = _arr(_upper_body(person), (256, 192))
    gen = _arr(_upper_body(generated), (256, 192))
    diff = float(np.mean(np.abs(base - gen)))
    return _clip01(1.0 - diff / 140.0)


def body_integrity_score(image: Image.Image) -> float:
    w, h = image.size
    if w < 256 or h < 256:
        return 0.1
    aspect = h / max(w, 1)
    aspect_score = _clip01(1.0 - abs(aspect - 1.33) / 1.33)
    body = _arr(_central_body(image))
    non_blank = float(np.std(body) > 12.0)
    return _clip01(aspect_score * 0.65 + non_blank * 0.35)


def artifact_score(image: Image.Image) -> float:
    stat = ImageStat.Stat(image)
    channel_std = float(np.mean(stat.stddev))
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_std = float(np.asarray(edges, dtype="float32").std())
    return _clip01(_clip01(channel_std / 70.0) * 0.55 + _clip01(edge_std / 45.0) * 0.45)


def reid_utility_score(image: Image.Image) -> float:
    try:
        bank = build_part_bank([image], emb_util.extract, emb_util.representative)
        if "full" not in bank:
            return 0.0
        norm = float(np.linalg.norm(bank["full"]))
        return _clip01((1.0 - abs(norm - 1.0)) * 0.5 + min(1.0, len(bank) / 5.0) * 0.5)
    except Exception:
        return 0.0


def score_image(provider: str, person: Image.Image, generated: Image.Image, outfit_desc: str, latency_sec: float = 0.0) -> dict:
    scores = {
        "provider": provider,
        "prompt_color_score": prompt_color_score(generated, outfit_desc),
        "identity_preservation_score": identity_preservation_score(person, generated),
        "body_integrity_score": body_integrity_score(generated),
        "artifact_score": artifact_score(generated),
        "reid_utility_score": reid_utility_score(generated),
        "latency_sec": float(latency_sec),
    }
    scores["total"] = sum(scores[name] * weight for name, weight in WEIGHTS.items())
    return scores


def _comparison(person: Image.Image, gemini: Image.Image, flux: Image.Image, output_path: Path) -> None:
    cell_w, cell_h = 360, 520
    canvas = Image.new("RGB", (cell_w * 3, cell_h + 54), (245, 245, 242))
    labels = [("person", person), ("gemini", gemini), ("flux", flux)]
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for idx, (label, image) in enumerate(labels):
        thumb = image.copy()
        thumb.thumbnail((cell_w - 24, cell_h - 70))
        x = idx * cell_w + (cell_w - thumb.width) // 2
        y = 48 + (cell_h - 70 - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text((idx * cell_w + 18, 14), label, fill=(20, 20, 20), font=font)
    canvas.save(output_path)


def evaluate(
    person_image: str,
    outfit_desc: str,
    gemini_image: str,
    flux_image: str,
    person_id: str = "unknown",
    output_dir: str | Path | None = None,
) -> dict:
    started = time.perf_counter()
    out_dir = Path(output_dir) if output_dir else config.RESULTS_DIR / "vton_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    person = _load(person_image)
    gemini = _load(gemini_image)
    flux = _load(flux_image)
    rows = [
        score_image("gemini", person, gemini, outfit_desc),
        score_image("flux", person, flux, outfit_desc),
    ]
    result = {
        "person_id": person_id,
        "outfit_desc": outfit_desc,
        "elapsed_sec": time.perf_counter() - started,
        "scores": rows,
        "winner": max(rows, key=lambda row: row["total"])["provider"],
    }

    (out_dir / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _comparison(person, gemini, flux, out_dir / "comparison.jpg")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Gemini and Flux outfit-generation outputs.")
    parser.add_argument("--person-image", required=True)
    parser.add_argument("--outfit-desc", required=True)
    parser.add_argument("--gemini-image", required=True)
    parser.add_argument("--flux-image", required=True)
    parser.add_argument("--person-id", default="unknown")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = evaluate(
        person_image=args.person_image,
        outfit_desc=args.outfit_desc,
        gemini_image=args.gemini_image,
        flux_image=args.flux_image,
        person_id=args.person_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
