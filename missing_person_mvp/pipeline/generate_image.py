from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from missing_person_mvp import config
from missing_person_mvp.pipeline.register import get_profile, get_view_image_paths, update_outfit_embedding, update_outfit_embedding_three_views
from missing_person_mvp.vton.flux import FluxGenerationError, generate_flux_contact_sheet


class ImageGenerationError(RuntimeError):
    pass


def _font(size: int):
    try:
        return ImageFont.truetype("malgun.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _add_watermark(image: Image.Image, text: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    banner_height = max(52, height // 10)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, height - banner_height, width, height), fill=(190, 0, 0))
    message = f"Missing person: {text}"
    font = _font(max(18, banner_height // 3))
    bbox = draw.textbbox((0, 0), message, font=font)
    x = max(10, (width - (bbox[2] - bbox[0])) // 2)
    y = height - banner_height + max(4, (banner_height - (bbox[3] - bbox[1])) // 2)
    draw.text((x, y), message, fill="white", font=font)
    return canvas


def _wrap_text(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    return [text[idx : idx + width] for idx in range(0, len(text), width)]


def _placeholder_image(name: str, outfit_desc: str) -> Image.Image:
    image = Image.new("RGB", (768, 1024), (242, 242, 238))
    draw = ImageDraw.Draw(image)
    draw.text((56, 80), name, fill=(20, 20, 20), font=_font(42))
    draw.text((56, 150), "Temporary outfit image", fill=(70, 70, 70), font=_font(28))
    y = 230
    for line in _wrap_text(outfit_desc, 28):
        draw.text((56, y), line, fill=(30, 30, 30), font=_font(26))
        y += 40
    return image


def _gemini_generate(base_path: str, outfit_desc: str) -> Image.Image:
    if not config.GEMINI_API_KEY:
        raise ImageGenerationError("GEMINI_API_KEY is not configured.")
    try:
        import io
        import google.generativeai as genai           # type: ignore

        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-image",
            generation_config={"response_modalities": ["IMAGE"]},
        )

        pil_img = Image.open(base_path).convert("RGB")
        prompt = (
            "Generate a full-body photo of this person wearing: "
            f"{outfit_desc}. Preserve identity, body shape, pose, and a neutral background. "
            "Output an image only."
        )

        for attempt in range(3):
            response = model.generate_content([prompt, pil_img])
            candidates = getattr(response, "candidates", None)
            if candidates:
                cand = candidates[0]
                parts = getattr(getattr(cand, "content", None), "parts", None) or []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        raw = inline.data
                        if not isinstance(raw, (bytes, bytearray)):
                            raw = bytes(raw)
                        return Image.open(io.BytesIO(raw)).convert("RGB")
                if attempt < 2:
                    time.sleep(2)
                    continue
                fr = getattr(cand, "finish_reason", "UNKNOWN")
                raise ImageGenerationError(f"Gemini 이미지 없음 (finish_reason={fr})")
            fb = getattr(response, "prompt_feedback", None)
            raise ImageGenerationError(f"Gemini 응답에 candidates 없음 (prompt_feedback={fb})")

    except ImageGenerationError:
        raise
    except Exception as exc:
        raise ImageGenerationError(f"Gemini generation failed: {exc}") from exc


def _gemini_generate_three_views(base_path: str, outfit_desc: str) -> dict[str, Image.Image]:
    """Gemini로 정면/측면/후면 3개 이미지 생성 → {view: PIL.Image} 반환."""
    if not config.GEMINI_API_KEY:
        raise ImageGenerationError("GEMINI_API_KEY is not configured.")
    try:
        import io
        import google.generativeai as genai           # type: ignore

        genai.configure(api_key=config.GEMINI_API_KEY)
        # response_modalities=IMAGE 명시 → 텍스트 응답 방지
        model = genai.GenerativeModel(
            "gemini-2.5-flash-image",
            generation_config={"response_modalities": ["IMAGE"]},
        )

        pil_img = Image.open(base_path).convert("RGB")

        prompts = {
            "front": (
                f"Edit the outfit of the person in this photo so they wear: {outfit_desc}. "
                "Keep the EXACT same camera angle and pose (front-facing). "
                "Only change the clothing, preserve everything else. Output an image only."
            ),
            "side": (
                f"Using this person as reference, generate a NEW full-body photo of the SAME person "
                f"photographed from a 90-degree side angle (lateral profile). "
                f"The person is turned sideways so the camera sees only their left side profile — "
                f"face pointing left, body perpendicular to the camera. "
                f"They are wearing: {outfit_desc}. Neutral indoor background. Output an image only."
            ),
            "back": (
                f"Using this person as reference, generate a NEW full-body photo of the SAME person "
                f"photographed from directly behind (180-degree back view). "
                f"The person's back faces the camera completely — no face visible, "
                f"only the rear of their head, back, and the back of their legs. "
                f"They are wearing: {outfit_desc}. Neutral indoor background. Output an image only."
            ),
        }
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _call_one(view: str, prompt: str) -> tuple[str, Image.Image]:
            for attempt in range(3):
                resp = model.generate_content([prompt, pil_img])
                candidates = getattr(resp, "candidates", None)
                if candidates:
                    cand = candidates[0]
                    parts = getattr(getattr(cand, "content", None), "parts", None) or []
                    for part in parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            raw = inline.data
                            if not isinstance(raw, (bytes, bytearray)):
                                raw = bytes(raw)
                            return view, Image.open(io.BytesIO(raw)).convert("RGB")
                    # 이미지 없이 텍스트만 반환 — 재시도
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    fr = getattr(cand, "finish_reason", "UNKNOWN")
                    raise ImageGenerationError(
                        f"Gemini {view} 이미지 없음 after {attempt+1} attempts (finish_reason={fr})"
                    )
                fb = getattr(resp, "prompt_feedback", None)
                raise ImageGenerationError(
                    f"Gemini {view} 응답에 candidates 없음 (prompt_feedback={fb})"
                )
            raise ImageGenerationError(f"Gemini {view} 이미지 없음 (최대 재시도 초과)")

        # 정면/측면/후면 3개 동시 호출 — 순차 대비 ~3배 빠름
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_call_one, v, p): v for v, p in prompts.items()}
            for fut in as_completed(futures):
                view, img = fut.result()
                results[view] = img

        return results
    except ImageGenerationError:
        raise
    except Exception as exc:
        raise ImageGenerationError(f"Gemini 3-view generation failed: {exc}") from exc


def _flux_generate(base_path: str | dict[str, str], outfit_desc: str) -> Image.Image:
    try:
        return generate_flux_contact_sheet(base_path, outfit_desc, config.RESULTS_DIR)
    except FluxGenerationError as exc:
        raise ImageGenerationError(str(exc)) from exc
    except Exception as exc:
        raise ImageGenerationError(f"Flux generation failed: {exc}") from exc


def _fashn_generate(base_path: str, outfit_desc: str) -> Image.Image:
    if not config.FASHN_API_KEY:
        raise ImageGenerationError("FASHN_API_KEY is not configured.")
    headers = {"Authorization": f"Bearer {config.FASHN_API_KEY}"}
    try:
        with open(base_path, "rb") as f:
            response = requests.post(
                "https://api.fashn.ai/v1/run",
                headers=headers,
                files={"model_image": (Path(base_path).name, f, "image/jpeg")},
                data={"garment_description": outfit_desc, "category": "full_body"},
                timeout=30,
            )
        response.raise_for_status()
        run_id = response.json().get("id")
        if not run_id:
            raise ImageGenerationError("FASHN.ai response did not include run id.")
        for _ in range(30):
            poll = requests.get(f"https://api.fashn.ai/v1/status/{run_id}", headers=headers, timeout=15)
            poll.raise_for_status()
            data = poll.json()
            if data.get("status") == "completed" and data.get("output"):
                image_response = requests.get(data["output"][0], timeout=30)
                image_response.raise_for_status()
                from io import BytesIO

                return Image.open(BytesIO(image_response.content)).convert("RGB")
            if data.get("status") == "failed":
                raise ImageGenerationError("FASHN.ai generation failed.")
            time.sleep(2)
        raise ImageGenerationError("FASHN.ai generation timed out.")
    except ImageGenerationError:
        raise
    except Exception as exc:
        raise ImageGenerationError(f"FASHN.ai generation failed: {exc}") from exc


def _provider_map():
    return {
        "gemini": _gemini_generate,
        "flux": _flux_generate,
        "fashn": _fashn_generate,
        "placeholder": None,
    }


def _run_provider(provider_name: str, base_path: str, view_paths: dict[str, str], outfit_desc: str) -> Image.Image:
    provider_fns = _provider_map()
    generator = provider_fns.get(provider_name)
    if generator is None:
        raise ImageGenerationError(f"Unknown image provider: {provider_name}")
    if provider_name == "flux":
        return generator(view_paths, outfit_desc)
    return generator(base_path, outfit_desc)


def _select_front_image(image_paths: list[str]) -> str:
    """등록 사진 중 얼굴 키포인트가 가장 잘 보이는 정면 사진 자동 선택.

    YOLO-Pose의 face keypoints(0=코, 1-2=눈, 3-4=귀) 평균 confidence 기준.
    탐지 실패 시 첫 번째 사진 반환.
    """
    if len(image_paths) <= 1:
        return image_paths[0]
    try:
        import cv2 as _cv2
        from missing_person_mvp.utils.track import _load_yolo
        model = _load_yolo()
        if model is None:
            return image_paths[0]
        best_path, best_score = image_paths[0], -1.0
        for path in image_paths:
            img = _cv2.imread(path)
            if img is None:
                continue
            results = model.predict(img, classes=[0], verbose=False)
            for result in results:
                kps = getattr(result, "keypoints", None)
                if kps is None or len(kps.data) == 0:
                    continue
                boxes = getattr(result, "boxes", None)
                best_idx = int(boxes.conf.argmax()) if (boxes is not None and boxes.conf is not None) else 0
                kp_conf = kps.data[best_idx][:, 2].cpu().numpy()
                # 코(0) + 눈(1,2) + 귀(3,4) 평균 — 높을수록 정면
                face_score = float(kp_conf[[0, 1, 2, 3, 4]].mean())
                if face_score > best_score:
                    best_score, best_path = face_score, path
        return best_path
    except Exception:
        return image_paths[0]


def generate_outfit_image(
    person_id: str,
    outfit_desc: str,
    providers: list[str] | None = None,
) -> tuple[Image.Image, str, list[str]]:
    profile = get_profile(person_id)
    if not profile.image_paths:
        raise ImageGenerationError("No registered base image exists.")
    base_path = _select_front_image(profile.image_paths)
    view_paths = get_view_image_paths(person_id)

    errors: list[str] = []
    image: Image.Image | None = None
    is_placeholder = False
    provider_order = providers if providers is not None else config.IMAGE_PROVIDER_ORDER
    provider_fns = _provider_map()
    for provider_name in provider_order:
        if provider_name == "placeholder":
            image = _placeholder_image(profile.name, outfit_desc)
            is_placeholder = True
            break
        try:
            image = _run_provider(provider_name, base_path, view_paths, outfit_desc)
            break
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")

    if image is None:
        if providers is not None and len(providers) == 1 and providers[0] != "placeholder":
            raise ImageGenerationError(errors[-1] if errors else "Image generation failed.")
        image = _placeholder_image(profile.name, outfit_desc)
        is_placeholder = True
        errors.append("All configured providers failed; placeholder was used.")

    watermarked = _add_watermark(image, profile.name)
    config.ensure_dirs()
    output_path = config.RESULTS_DIR / f"{person_id}_outfit_{int(time.time())}.jpg"
    watermarked.save(output_path, quality=92)
    # 플레이스홀더(회색 박스)는 검색 벡터를 오염시키므로 임베딩 저장 안 함.
    # 워터마크 없는 원본(image)에서 사람 크롭 후 임베딩 추출.
    if not is_placeholder:
        update_outfit_embedding(person_id, image=image)
        # 날짜·프롬프트 붙은 피쳐로 자동 저장 → 검색 시 재사용(무료)
        try:
            from missing_person_mvp.pipeline.register import save_outfit_preset
            save_outfit_preset(person_id, prompt=outfit_desc)
        except Exception:
            pass
    return watermarked, str(output_path), errors


def compare_generate_outfit_image(
    person_id: str,
    outfit_desc: str,
    compare_providers: list[str] | None = None,
) -> dict[str, tuple[Image.Image, str] | str]:
    """두 제공자를 병렬 실행. 임베딩 저장 안 함 — 사용자가 직접 선택 후 저장.

    Returns: {provider: (PIL.Image, path)} or {provider: error_message}
    """
    profile = get_profile(person_id)
    if not profile.image_paths:
        raise ImageGenerationError("No registered base image exists.")
    base_path = _select_front_image(profile.image_paths)
    view_paths = get_view_image_paths(person_id)

    providers_to_run = compare_providers or ["gemini", "flux"]
    provider_fns = _provider_map()

    def _run_one(pname: str) -> tuple[str, tuple[Image.Image, str] | str]:
        if pname == "placeholder":
            img = _placeholder_image(profile.name, outfit_desc)
            config.ensure_dirs()
            out = config.RESULTS_DIR / f"{person_id}_placeholder_{int(time.time())}.jpg"
            img.save(out, quality=92)
            return pname, (img, str(out))
        try:
            img = _run_provider(pname, base_path, view_paths, outfit_desc)
            config.ensure_dirs()
            out = config.RESULTS_DIR / f"{person_id}_{pname}_{int(time.time())}.jpg"
            _add_watermark(img, profile.name).save(out, quality=92)
            return pname, (img, str(out))
        except Exception as exc:
            return pname, str(exc)

    results: dict[str, tuple[Image.Image, str] | str] = {}
    with ThreadPoolExecutor(max_workers=len(providers_to_run)) as executor:
        futs = {executor.submit(_run_one, p): p for p in providers_to_run}
        for fut in _as_completed(futs):
            pname, res = fut.result()
            results[pname] = res

    return results


def generate_outfit_image_three_views(
    person_id: str,
    outfit_desc: str,
    providers: list[str] | None = None,
) -> tuple[dict[str, Image.Image], dict[str, str], list[str]]:
    """3면 이미지 생성 (Gemini/Flux 전용). {view: image}, {view: path}, errors 반환."""
    profile = get_profile(person_id)
    if not profile.image_paths:
        raise ImageGenerationError("No registered base image exists.")
    base_path = _select_front_image(profile.image_paths)

    errors: list[str] = []
    images: dict[str, Image.Image] | None = None
    is_placeholder = False
    provider_order = providers if providers is not None else config.IMAGE_PROVIDER_ORDER

    for provider_name in provider_order:
        if provider_name == "placeholder":
            images = {v: _placeholder_image(profile.name, outfit_desc) for v in ["front", "side", "back"]}
            is_placeholder = True
            break
        try:
            if provider_name == "gemini":
                images = _gemini_generate_three_views(base_path, outfit_desc)
                break
            elif provider_name == "flux":
                view_paths = get_view_image_paths(person_id)
                if not all(v in view_paths for v in ["front", "side", "back"]):
                    errors.append("flux: Requires front/side/back view paths")
                    continue
                from missing_person_mvp.vton.flux import generate_flux_views
                _imgs, _ = generate_flux_views(view_paths, outfit_desc, config.RESULTS_DIR)
                images = _imgs
                break
            else:
                errors.append(f"{provider_name}: Not available for 3-view generation")
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")

    if images is None:
        images = {v: _placeholder_image(profile.name, outfit_desc) for v in ["front", "side", "back"]}
        is_placeholder = True
        errors.append("All providers failed; placeholders used.")

    # 저장 + 뷰별 경로
    config.ensure_dirs()
    paths: dict[str, str] = {}
    ts = int(time.time())
    for view, img in images.items():
        watermarked = _add_watermark(img, profile.name)
        out_path = config.RESULTS_DIR / f"{person_id}_outfit_{view}_{ts}.jpg"
        watermarked.save(out_path, quality=92)
        paths[view] = str(out_path)

    # 플레이스홀더는 검색 벡터를 오염시키므로 저장 안 함.
    # 워터마크 없는 원본(images)에서 사람 크롭 후 뷰별 임베딩 추출.
    if not is_placeholder:
        update_outfit_embedding_three_views(person_id, view_images=images)
        # 날짜·프롬프트 붙은 피쳐로 자동 저장 → 검색 시 재사용(무료)
        try:
            from missing_person_mvp.pipeline.register import save_outfit_preset
            save_outfit_preset(person_id, prompt=outfit_desc)
        except Exception:
            pass

    return images, paths, errors
