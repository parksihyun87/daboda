from __future__ import annotations

import time
import gc
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from missing_person_mvp import config
from missing_person_mvp.utils.viewpoint import VIEWS

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class FluxGenerationError(RuntimeError):
    pass


_PIPE = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR = ""


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _load_pipeline():
    global _PIPE, _LOAD_ATTEMPTED, _LOAD_ERROR
    if _LOAD_ATTEMPTED:
        if _PIPE is None:
            detail = f" Previous error: {_LOAD_ERROR}" if _LOAD_ERROR else ""
            raise FluxGenerationError(f"Flux inpaint pipeline is not loaded.{detail}")
        return _PIPE
    _LOAD_ATTEMPTED = True
    if not config.FLUX_ENABLED:
        raise FluxGenerationError("FLUX_ENABLED=false.")
    if config.FLUX_MODE != "inpaint":
        raise FluxGenerationError(f"Unsupported FLUX_MODE={config.FLUX_MODE}; expected inpaint.")
    try:
        import torch
        from diffusers import FluxInpaintPipeline  # type: ignore
    except Exception as exc:
        raise FluxGenerationError(
            "Flux inpaint dependencies are missing. Install diffusers, transformers, accelerate, and safetensors."
        ) from exc

    device = "cuda" if config.FLUX_DEVICE == "auto" and torch.cuda.is_available() else config.FLUX_DEVICE
    if device == "auto":
        device = "cpu"
    dtype = torch.float32
    if device == "cuda" and config.FLUX_DTYPE == "bfloat16":
        dtype = torch.bfloat16
    elif device == "cuda" and config.FLUX_DTYPE == "float16":
        dtype = torch.float16

    try:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        load_kwargs: dict = {
            "torch_dtype": dtype,
            "cache_dir": str(config.FLUX_CACHE_DIR),
        }
        if config.HF_TOKEN:
            load_kwargs["token"] = config.HF_TOKEN
        _PIPE = FluxInpaintPipeline.from_pretrained(config.FLUX_MODEL_ID, **load_kwargs)
        if hasattr(_PIPE, "enable_vae_tiling"):
            _PIPE.enable_vae_tiling()
        if hasattr(_PIPE, "enable_vae_slicing"):
            _PIPE.enable_vae_slicing()
        if device == "cuda" and not config.FLUX_CPU_OFFLOAD:
            _PIPE.to("cuda")
        elif device == "cuda":
            _PIPE.enable_model_cpu_offload()
        else:
            _PIPE.enable_model_cpu_offload()
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        raise FluxGenerationError(f"Flux inpaint pipeline load failed: {exc}") from exc
    _LOAD_ERROR = ""
    return _PIPE


def _prompt(outfit_desc: str, view: str) -> str:
    view_text = {
        "front": "front view, same person, facing the camera",
        "side": "side profile view, same person, standing sideways",
        "back": "back view, same person, facing away from the camera",
    }[view]
    return (
        f"Edit only the clothing region. Keep the same person, face, head, hair, body shape, "
        f"pose, hands, shoes if not specified, and background. {view_text}. "
        f"Change the outfit to: {outfit_desc}. Realistic public safety reference photo, "
        "clear clothing colors, no text, no watermark."
    )


def _placeholder_view(outfit_desc: str, view: str, size: tuple[int, int] = (768, 1024)) -> Image.Image:
    image = Image.new("RGB", size, (240, 240, 236))
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), f"Flux inpaint unavailable: {view}", fill=(30, 30, 30), font=_font(32))
    draw.text((40, 92), outfit_desc[:60], fill=(60, 60, 60), font=_font(24))
    return image


def _resize_for_flux(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(
        (config.FLUX_INPAINT_WIDTH, config.FLUX_INPAINT_HEIGHT),
        Image.Resampling.LANCZOS,
    )


def _detect_face_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    try:
        import cv2

        arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        boxes = cascade.detectMultiScale(arr, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
        return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in boxes]
    except Exception:
        return []


def _detect_person_bbox_and_keypoints(image: Image.Image):
    try:
        import cv2

        from missing_person_mvp.utils.track import _load_yolo

        model = _load_yolo()
        if model is None:
            return None, None
        arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        results = model.predict(arr, classes=[0], verbose=False)
        best = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue
            confs = getattr(boxes, "conf", None)
            idx = int(confs.argmax()) if confs is not None else 0
            xyxy = tuple(int(v) for v in boxes.xyxy[idx].tolist())
            area = max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])
            keypoints = getattr(result, "keypoints", None)
            kp_xy = None
            kp_conf = None
            if keypoints is not None:
                try:
                    kp_xy = keypoints.xy[idx].cpu().numpy().astype("float32")
                    kp_conf = keypoints.conf[idx].cpu().numpy().astype("float32")
                except Exception:
                    pass
            if best is None or area > best[0]:
                best = (area, xyxy, kp_xy, kp_conf)
        if best is None:
            return None, None
        return best[1], (best[2], best[3])
    except Exception:
        return None, None


def _clothing_vertical_bounds(
    bbox: tuple[int, int, int, int],
    keypoints: tuple[np.ndarray | None, np.ndarray | None] | None,
) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    height = max(1, y2 - y1)
    top = y1 + int(height * 0.18)
    bottom = y2
    if keypoints is not None:
        kp_xy, kp_conf = keypoints
        if kp_xy is not None and kp_conf is not None and len(kp_conf) >= 17:
            shoulders = [idx for idx in (5, 6) if float(kp_conf[idx]) > 0.35]
            ankles = [idx for idx in (15, 16) if float(kp_conf[idx]) > 0.35]
            if shoulders:
                top = max(y1, int(min(float(kp_xy[idx][1]) for idx in shoulders)) - int(height * 0.04))
            if ankles:
                bottom = min(y2, int(max(float(kp_xy[idx][1]) for idx in ankles)) + int(height * 0.03))
    return top, bottom


def create_clothing_mask(image: Image.Image) -> Image.Image:
    """White pixels are edited by inpainting; black pixels are preserved."""
    src = _resize_for_flux(image)
    width, height = src.size
    bbox, keypoints = _detect_person_bbox_and_keypoints(src)
    if bbox is None:
        margin_x = int(width * 0.18)
        bbox = (margin_x, int(height * 0.12), width - margin_x, int(height * 0.98))
        keypoints = None

    x1, y1, x2, y2 = bbox
    top, bottom = _clothing_vertical_bounds(bbox, keypoints)
    pad_x = int((x2 - x1) * 0.08)
    mask = Image.new("L", src.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (max(0, x1 - pad_x), max(0, top), min(width, x2 + pad_x), min(height, bottom)),
        radius=max(12, int((x2 - x1) * 0.08)),
        fill=255,
    )

    protect = Image.new("L", src.size, 0)
    pdraw = ImageDraw.Draw(protect)
    for fx1, fy1, fx2, fy2 in _detect_face_boxes(src):
        fw, fh = fx2 - fx1, fy2 - fy1
        pdraw.ellipse(
            (
                max(0, fx1 - int(fw * 0.55)),
                max(0, fy1 - int(fh * 0.95)),
                min(width, fx2 + int(fw * 0.55)),
                min(height, fy2 + int(fh * 0.75)),
            ),
            fill=255,
        )

    # Always protect the likely head band inside the person box, including side/back photos.
    head_bottom = max(y1 + int((y2 - y1) * 0.22), top)
    pdraw.rectangle((max(0, x1 - pad_x), max(0, y1), min(width, x2 + pad_x), min(height, head_bottom)), fill=255)

    if config.FLUX_MASK_DILATE > 0:
        mask = mask.filter(ImageFilter.MaxFilter(config.FLUX_MASK_DILATE | 1))
    protect = protect.filter(ImageFilter.GaussianBlur(radius=4))
    mask_arr = np.array(mask, dtype=np.int16)
    protect_arr = np.array(protect, dtype=np.int16)
    final = np.clip(mask_arr - protect_arr, 0, 255).astype("uint8")
    return Image.fromarray(final, mode="L").filter(ImageFilter.GaussianBlur(radius=3))


def validate_three_view_paths(view_image_paths: dict[str, str]) -> dict[str, str]:
    missing = [view for view in VIEWS if not view_image_paths.get(view)]
    if missing:
        raise FluxGenerationError(f"Flux inpainting requires front/side/back images. Missing: {', '.join(missing)}")
    resolved: dict[str, str] = {}
    for view in VIEWS:
        path = Path(view_image_paths[view])
        if not path.exists():
            raise FluxGenerationError(f"Flux {view} image does not exist: {path}")
        resolved[view] = str(path)
    return resolved


def create_contact_sheet(images: dict[str, Image.Image], output_path: str | Path | None = None) -> Image.Image:
    cell_w, cell_h = 360, 520
    sheet = Image.new("RGB", (cell_w * 3, cell_h + 48), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, view in enumerate(VIEWS):
        img = images[view].copy()
        img.thumbnail((cell_w - 24, cell_h - 70))
        x = idx * cell_w + (cell_w - img.width) // 2
        y = 48 + (cell_h - 70 - img.height) // 2
        sheet.paste(img, (x, y))
        draw.text((idx * cell_w + 16, 12), view, fill=(20, 20, 20), font=_font(24))
    if output_path:
        sheet.save(output_path)
    return sheet


def generate_flux_views(
    person_image_path: str | dict[str, str],
    outfit_desc: str,
    output_dir: str | Path | None = None,
    allow_placeholder: bool = False,
) -> tuple[dict[str, Image.Image], str]:
    out_dir = Path(output_dir) if output_dir else config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(person_image_path, dict):
        view_paths = validate_three_view_paths(person_image_path)
    elif allow_placeholder and not config.FLUX_REQUIRE_THREE_VIEWS:
        view_paths = {view: person_image_path for view in VIEWS}
    elif allow_placeholder:
        view_paths = {view: person_image_path for view in VIEWS}
    else:
        raise FluxGenerationError("Flux inpainting requires view_image_paths with front/side/back images.")

    if allow_placeholder:
        stem = f"flux_inpaint_3view_{int(time.time())}"
        images = {view: _placeholder_view(outfit_desc, view) for view in VIEWS}
        for view, image in images.items():
            image.save(out_dir / f"{stem}_{view}.png")
        sheet_path = out_dir / f"{stem}_sheet.jpg"
        create_contact_sheet(images, sheet_path)
        return images, str(sheet_path)

    images: dict[str, Image.Image] = {}
    try:
        pipe = _load_pipeline()
        import torch

        device = "cuda" if torch.cuda.is_available() and config.FLUX_DEVICE != "cpu" else "cpu"
        generator = torch.Generator(device=device).manual_seed(config.FLUX_SEED)
        for view in VIEWS:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            base = _resize_for_flux(Image.open(view_paths[view]).convert("RGB"))
            mask = create_clothing_mask(base)
            result = pipe(
                prompt=_prompt(outfit_desc, view),
                image=base,
                mask_image=mask,
                height=config.FLUX_INPAINT_HEIGHT,
                width=config.FLUX_INPAINT_WIDTH,
                strength=config.FLUX_INPAINT_STRENGTH,
                num_inference_steps=config.FLUX_NUM_INFERENCE_STEPS,
                guidance_scale=config.FLUX_GUIDANCE_SCALE,
                generator=generator,
            )
            images[view] = result.images[0].convert("RGB")
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
    except Exception as exc:
        if not allow_placeholder:
            raise FluxGenerationError(str(exc)) from exc
        images = {view: _placeholder_view(outfit_desc, view) for view in VIEWS}

    stem = f"flux_inpaint_3view_{int(time.time())}"
    for view, image in images.items():
        image.save(out_dir / f"{stem}_{view}.png")
    sheet_path = out_dir / f"{stem}_sheet.jpg"
    create_contact_sheet(images, sheet_path)
    return images, str(sheet_path)


def generate_flux_contact_sheet(
    person_image_path: str | dict[str, str],
    outfit_desc: str,
    output_dir: str | Path | None = None,
) -> Image.Image:
    _images, sheet_path = generate_flux_views(person_image_path, outfit_desc, output_dir)
    return Image.open(sheet_path).convert("RGB")
