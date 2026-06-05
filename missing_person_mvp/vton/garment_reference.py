from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from missing_person_mvp import config
from missing_person_mvp.vton.outfit_parser import OutfitSpec, color_rgb, parse_outfit


def _font(size: int):
    try:
        return ImageFont.truetype("malgun.ttf", size)
    except Exception:
        return ImageFont.load_default()


def create_garment_reference(
    outfit_desc: str,
    output_dir: str | Path | None = None,
    size: tuple[int, int] = (768, 1024),
) -> tuple[Image.Image, str, OutfitSpec]:
    spec = parse_outfit(outfit_desc)
    width, height = size
    image = Image.new("RGB", size, (236, 236, 232))
    draw = ImageDraw.Draw(image)

    upper = color_rgb(spec.upper_color or spec.primary_color())
    lower = color_rgb(spec.lower_color or ("black" if spec.upper_color else spec.primary_color()))
    shoes = color_rgb(spec.shoe_color or "white")

    cx = width // 2
    # A simple full-body garment reference for local generation/evaluation.
    draw.rounded_rectangle((cx - 155, 120, cx + 155, 470), radius=38, fill=upper, outline=(40, 40, 40), width=4)
    draw.rectangle((cx - 128, 455, cx - 18, 810), fill=lower, outline=(40, 40, 40), width=4)
    draw.rectangle((cx + 18, 455, cx + 128, 810), fill=lower, outline=(40, 40, 40), width=4)
    draw.rounded_rectangle((cx - 150, 812, cx - 10, 872), radius=20, fill=shoes, outline=(40, 40, 40), width=3)
    draw.rounded_rectangle((cx + 10, 812, cx + 150, 872), radius=20, fill=shoes, outline=(40, 40, 40), width=3)

    label = outfit_desc.strip() or "neutral outfit"
    draw.text((36, 36), "Garment reference", fill=(35, 35, 35), font=_font(30))
    y = 905
    for line in [label[i : i + 34] for i in range(0, len(label), 34)][:2]:
        draw.text((36, y), line, fill=(35, 35, 35), font=_font(24))
        y += 34

    out_dir = Path(output_dir) if output_dir else config.RESULTS_DIR / "garments"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1((outfit_desc or "neutral").encode("utf-8")).hexdigest()[:10]
    path = out_dir / f"garment_{digest}.png"
    image.save(path)
    return image, str(path), spec
