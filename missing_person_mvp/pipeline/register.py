from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from missing_person_mvp import config
from missing_person_mvp.models import PersonEmbedding, PersonProfile
from missing_person_mvp.utils import embedding as emb_util


VIEW_KEYS = ("front", "side", "back")


def _load_db() -> dict:
    config.ensure_dirs()
    if not config.PERSONS_JSON.exists():
        return {"persons": {}, "faiss_to_id": []}
    return json.loads(config.PERSONS_JSON.read_text(encoding="utf-8"))


def _save_db(db: dict) -> None:
    config.ensure_dirs()
    config.PERSONS_JSON.write_text(
        json.dumps(db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_embeddings() -> dict[str, np.ndarray]:
    if not config.EMBEDDINGS_NPZ.exists():
        return {}
    loaded = np.load(config.EMBEDDINGS_NPZ, allow_pickle=False)
    return {key: loaded[key].astype("float32") for key in loaded.files}


def _save_embeddings(vectors: dict[str, np.ndarray]) -> None:
    config.ensure_dirs()
    if vectors:
        np.savez(config.EMBEDDINGS_NPZ, **vectors)
    elif config.EMBEDDINGS_NPZ.exists():
        config.EMBEDDINGS_NPZ.unlink()
    _rebuild_faiss(vectors)


def _rebuild_faiss(vectors: dict[str, np.ndarray]) -> None:
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        ids = sorted(vectors)
        if ids:
            matrix = np.stack([vectors[person_id] for person_id in ids]).astype("float32")
            index.add(matrix)
        faiss.write_index(index, str(config.FAISS_INDEX))
        db = _load_db()
        db["faiss_to_id"] = ids
        _save_db(db)
    except Exception:
        if config.FAISS_INDEX.exists():
            config.FAISS_INDEX.unlink()


def _open_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"등록 이미지 파일을 찾을 수 없습니다: {path}")
        images.append(Image.open(image_path).convert("RGB"))
    if not images:
        raise ValueError("등록에는 최소 1장의 이미지가 필요합니다.")
    return images


def _normalized_profile(profile: PersonProfile) -> PersonProfile:
    view_paths = {
        view: str(path)
        for view, path in (profile.view_image_paths or {}).items()
        if view in VIEW_KEYS and str(path).strip()
    }
    image_paths = list(profile.image_paths or [])
    if view_paths:
        for view in VIEW_KEYS:
            path = view_paths.get(view)
            if path and path not in image_paths:
                image_paths.append(path)
    return PersonProfile(
        person_id=profile.person_id,
        name=profile.name,
        contact=profile.contact,
        image_paths=image_paths,
        gait_video_path=profile.gait_video_path,
        view_image_paths=view_paths,
    )


def register_person(profile: PersonProfile) -> PersonEmbedding:
    profile = _normalized_profile(profile)
    if not profile.person_id.strip():
        raise ValueError("person_id는 비어 있을 수 없습니다.")
    images = _open_images(profile.image_paths)

    from missing_person_mvp.utils.parts import PARTS, build_part_bank

    # Part embedding bank 생성 (등록 시 1회만 계산)
    bank = build_part_bank(images, emb_util.extract, emb_util.representative)

    db = _load_db()
    db["persons"][profile.person_id] = asdict(profile)
    _save_db(db)

    vectors = _load_embeddings()
    for part, vec in bank.items():
        vectors[f"{profile.person_id}_{part}"] = vec.astype("float32")
    # full vector를 기본 키로도 저장 (하위 호환 + FAISS)
    if "full" in bank:
        vectors[profile.person_id] = bank["full"].astype("float32")
    _save_embeddings(vectors)

    return PersonEmbedding(
        person_id=profile.person_id,
        vector=bank.get("full", np.zeros(config.EMBEDDING_DIM, dtype="float32")).tolist(),
        image_count=len(images),
    )


def get_part_bank(person_id: str) -> dict[str, np.ndarray]:
    """등록된 인물의 part embedding bank 반환. 없는 part는 제외."""
    from missing_person_mvp.utils.parts import PARTS

    vectors = _load_embeddings()
    outfit_prefix = f"{person_id}_outfit_"
    base_prefix   = f"{person_id}_"
    bank: dict[str, np.ndarray] = {}
    for part in PARTS:
        outfit_key = f"{person_id}_outfit_{part}"
        base_key   = f"{person_id}_{part}"
        if outfit_key in vectors:
            bank[part] = vectors[outfit_key]
        elif base_key in vectors:
            bank[part] = vectors[base_key]
    return bank


def get_part_bank_multi_view(person_id: str) -> dict[str, dict[str, np.ndarray]]:
    """3면 뷰별 part bank 반환. {view: {part: embedding}}.

    없으면 None 반환. CCTV 매칭 시 max similarity 사용.
    """
    from missing_person_mvp.utils.parts import PARTS

    vectors = _load_embeddings()
    views = {v: {} for v in ["front", "side", "back"]}
    found_any = False

    for view in ["front", "side", "back"]:
        for part in PARTS:
            key = f"{person_id}_outfit_{view}_{part}"
            if key in vectors:
                views[view][part] = vectors[key]
                found_any = True

    return views if found_any else {}


def update_outfit_embedding(
    person_id: str,
    image_path: str | None = None,
    image: Image.Image | None = None,
) -> None:
    """VTON 생성 이미지에서 part별 임베딩 추출 → {person_id}_outfit_{part} 키로 저장.

    image(워터마크 없는 PIL) 또는 image_path 중 하나를 받는다.
    build_part_bank가 내부적으로 YOLO로 사람 영역을 타이트 크롭한다.
    """
    from missing_person_mvp.utils.parts import build_part_bank

    if image is None:
        if image_path is None:
            raise ValueError("update_outfit_embedding: image 또는 image_path 필요")
        image = Image.open(image_path)
    img = image.convert("RGB")
    bank = build_part_bank([img], emb_util.extract, emb_util.representative)
    vectors = _load_embeddings()
    for part, vec in bank.items():
        vectors[f"{person_id}_outfit_{part}"] = vec.astype("float32")
    # 하위 호환: single outfit key도 유지
    if "full" in bank:
        vectors[f"{person_id}_outfit"] = bank["full"].astype("float32")
    _save_embeddings(vectors)


def update_outfit_embedding_three_views(
    person_id: str,
    view_paths: dict[str, str] | None = None,
    view_images: dict[str, Image.Image] | None = None,
) -> None:
    """3면 VTON 이미지에서 뷰별·part별 임베딩 추출 → {person_id}_outfit_{view}_{part} 저장.

    view_images(워터마크 없는 PIL dict) 또는 view_paths 중 하나를 받는다.
    """
    from missing_person_mvp.utils.parts import build_part_bank

    if view_images is None:
        if view_paths is None:
            raise ValueError("update_outfit_embedding_three_views: view_images 또는 view_paths 필요")
        view_images = {v: Image.open(p) for v, p in view_paths.items()}

    vectors = _load_embeddings()
    for view, img in view_images.items():
        bank = build_part_bank([img.convert("RGB")], emb_util.extract, emb_util.representative)
        for part, vec in bank.items():
            vectors[f"{person_id}_outfit_{view}_{part}"] = vec.astype("float32")
    _save_embeddings(vectors)


def get_embedding(person_id: str) -> np.ndarray:
    """착장 임베딩(VTON 이미지 기반) 우선 반환. 없으면 등록 사진 기반 임베딩 반환."""
    vectors = _load_embeddings()
    outfit_key = f"{person_id}_outfit"
    if outfit_key in vectors:
        return vectors[outfit_key]
    if person_id not in vectors:
        raise KeyError(f"등록된 인물을 찾을 수 없습니다: {person_id}")
    return vectors[person_id]


def get_embedding_source(person_id: str) -> str:
    """현재 분석에 사용될 임베딩 출처를 반환. UI 표시용."""
    vectors = _load_embeddings()
    if f"{person_id}_outfit" in vectors:
        return "착장 이미지 기반 (VTON)"
    if person_id in vectors:
        return "등록 사진 기반 (평시)"
    return "임베딩 없음"


def embedding_states(person_id: str) -> dict[str, bool]:
    """대상자의 임베딩 보유 상태를 구분해서 반환.

    - base   : 평소(등록 사진) 임베딩 존재
    - outfit : 생성된 착장(VTON) 임베딩 존재 — 검색 시 이게 우선 사용됨
    - outfit_3view : 3면(정면·측면·후면) 착장 임베딩 존재
    """
    vectors = _load_embeddings()
    return {
        "base": (person_id in vectors) or (f"{person_id}_full" in vectors),
        "outfit": f"{person_id}_outfit" in vectors,
        "outfit_3view": any(
            f"{person_id}_outfit_{v}_full" in vectors for v in ("front", "side", "back")
        ),
    }


# ── 착장 피쳐(벡터) 저장/불러오기 — 날짜·프롬프트 메타데이터 포함 ──────────
# 생성 시 자동 저장되고, 검색 시 "언제·어떤 프롬프트로 만든 피쳐"를 골라 쓴다.
# Gemini 재호출 없이 순수 벡터 복사라 즉시·무료.

def _outfit_keys(person_id: str, vectors: dict[str, np.ndarray]) -> list[str]:
    """대상자의 생성 착장(VTON) 임베딩 키 전체 (단면 + 3면)."""
    return [
        k for k in vectors
        if k == f"{person_id}_outfit" or k.startswith(f"{person_id}_outfit_")
    ]


def _preset_dir(person_id: str) -> Path:
    d = config.DB_DIR / "presets" / person_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _preset_index_path(person_id: str) -> Path:
    return _preset_dir(person_id) / "index.json"


def _load_preset_index(person_id: str) -> dict:
    p = _preset_index_path(person_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_preset_index(person_id: str, index: dict) -> None:
    _preset_index_path(person_id).write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_outfit_preset(person_id: str, prompt: str = "") -> str:
    """현재 생성 착장 임베딩(단면+3면)을 날짜·프롬프트와 함께 피쳐로 저장.

    순수 벡터 복사 → Gemini 호출 없음. 반환: 피쳐 id (timestamp).
    """
    from datetime import datetime

    vectors = _load_embeddings()
    keys = _outfit_keys(person_id, vectors)
    if not keys:
        raise ValueError("저장할 생성 착장 임베딩이 없습니다. 먼저 착장을 생성하세요.")

    now = datetime.now()
    preset_id = now.strftime("%Y%m%d_%H%M%S")
    sub = {k: vectors[k].astype("float32") for k in keys}
    np.savez(_preset_dir(person_id) / f"{preset_id}.npz", **sub)

    has_3view = any("_outfit_front_" in k or "_outfit_side_" in k or "_outfit_back_" in k for k in keys)
    index = _load_preset_index(person_id)
    index[preset_id] = {
        "id": preset_id,
        "created_at": now.isoformat(timespec="seconds"),
        "prompt": (prompt or "").strip(),
        "has_3view": has_3view,
    }
    _save_preset_index(person_id, index)
    return preset_id


def list_outfit_presets(person_id: str) -> list[dict]:
    """저장된 착장 피쳐 목록 (최신순). 각 항목: id, created_at, prompt, has_3view."""
    index = _load_preset_index(person_id)
    d = config.DB_DIR / "presets" / person_id
    items = []
    for p in (d.glob("*.npz") if d.exists() else []):
        meta = index.get(p.stem, {"id": p.stem, "created_at": p.stem, "prompt": "", "has_3view": False})
        items.append(meta)
    return sorted(items, key=lambda m: m.get("created_at", ""), reverse=True)


def preset_label(meta: dict) -> str:
    """UI 표시용 라벨: '06-02 14:30 · 노란색 상의에 검은색 바지 (3면)'."""
    ca = meta.get("created_at", meta.get("id", ""))
    when = ca[5:16].replace("T", " ") if len(ca) >= 16 else ca
    prompt = meta.get("prompt") or "(설명 없음)"
    tag = " (3면)" if meta.get("has_3view") else " (단면)"
    return f"{when} · {prompt}{tag}"


def load_outfit_preset(person_id: str, preset_id: str) -> list[str]:
    """저장된 피쳐를 활성 착장 임베딩으로 복원. Gemini 호출 없음. 반환: 복원된 키 목록."""
    path = config.DB_DIR / "presets" / person_id / f"{preset_id}.npz"
    if not path.exists():
        raise KeyError(f"피쳐를 찾을 수 없습니다: {preset_id}")
    saved = np.load(path, allow_pickle=False)
    vectors = _load_embeddings()
    for k in _outfit_keys(person_id, vectors):
        vectors.pop(k, None)
    for k in saved.files:
        vectors[k] = saved[k].astype("float32")
    _save_embeddings(vectors)
    return list(saved.files)


def delete_outfit_preset(person_id: str, preset_id: str) -> bool:
    path = config.DB_DIR / "presets" / person_id / f"{preset_id}.npz"
    existed = path.exists()
    if existed:
        path.unlink()
    index = _load_preset_index(person_id)
    if preset_id in index:
        index.pop(preset_id)
        _save_preset_index(person_id, index)
    return existed


def get_profile(person_id: str) -> PersonProfile:
    db = _load_db()
    data = db.get("persons", {}).get(person_id)
    if data is None:
        raise KeyError(f"등록된 인물을 찾을 수 없습니다: {person_id}")
    return PersonProfile(**data)


def get_view_image_paths(person_id: str) -> dict[str, str]:
    profile = get_profile(person_id)
    return {
        view: str(path)
        for view, path in (profile.view_image_paths or {}).items()
        if view in VIEW_KEYS and str(path).strip()
    }


def list_persons() -> list[dict]:
    db = _load_db()
    return [
        {"person_id": person_id, **data}
        for person_id, data in sorted(db.get("persons", {}).items())
    ]


def delete_person(person_id: str) -> bool:
    db = _load_db()
    existed = person_id in db.get("persons", {})
    if not existed:
        return False
    db["persons"].pop(person_id, None)
    _save_db(db)
    vectors = _load_embeddings()
    vectors.pop(person_id, None)
    vectors.pop(f"{person_id}_outfit", None)
    vectors.pop(f"{person_id}_gait", None)
    _save_embeddings(vectors)
    return True


# ── Gait 등록 ───────────────────────────────────────────────────────

def register_gait_videos(person_id: str, video_paths: list[str]) -> bool:
    """보행 동영상들 → YOLO-Pose → gait embedding → {person_id}_gait 키로 저장.

    Returns True if successful, False if not enough valid frames.
    """
    from missing_person_mvp.utils.gait import (
        build_gait_embedding,
        extract_gait_frames_from_video,
    )

    all_frames: list = []
    for vpath in video_paths:
        all_frames.extend(extract_gait_frames_from_video(vpath))

    emb = build_gait_embedding(all_frames)
    if emb is None:
        return False

    vectors = _load_embeddings()
    vectors[f"{person_id}_gait"] = emb.astype("float32")
    _save_embeddings(vectors)
    return True


def get_gait_embedding(person_id: str) -> np.ndarray | None:
    """등록된 gait 임베딩 반환. 없으면 None."""
    vectors = _load_embeddings()
    return vectors.get(f"{person_id}_gait")


def has_gait_embedding(person_id: str) -> bool:
    vectors = _load_embeddings()
    return f"{person_id}_gait" in vectors
