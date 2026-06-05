from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from PIL import Image

from missing_person_mvp import config


_MODEL = None
_MODEL_LOAD_ATTEMPTED = False


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype("float32")
    return (vector / norm).astype("float32")


def _load_osnet():
    global _MODEL, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL
    _MODEL_LOAD_ATTEMPTED = True
    try:
        import torchreid  # type: ignore

        _MODEL = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True,
        )
        _MODEL.eval()
    except Exception:
        _MODEL = None
    return _MODEL


def _to_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr.astype("uint8")


def _dummy_embedding(image) -> np.ndarray:
    arr = _to_rgb_array(image)
    digest = hashlib.sha256(arr.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    color = arr.reshape(-1, 3).mean(axis=0) / 255.0
    vec = rng.normal(size=config.EMBEDDING_DIM).astype("float32")
    vec[:3] += color.astype("float32") * 4.0
    return _l2_normalize(vec)


def _osnet_embed(model, images: list) -> np.ndarray:
    import torch
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = [transform(Image.fromarray(_to_rgb_array(img))) for img in images]
    batch = torch.stack(tensors)
    with torch.no_grad():
        feats = model(batch)
    feats_np = feats.cpu().numpy().astype("float32")
    norms = np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-12
    return feats_np / norms


def extract(images: Iterable[Image.Image | np.ndarray]) -> np.ndarray:
    image_list = list(images)
    if not image_list:
        return np.empty((0, config.EMBEDDING_DIM), dtype="float32")

    model = _load_osnet()
    if model is None:
        return np.stack([_dummy_embedding(img) for img in image_list]).astype("float32")

    try:
        return _osnet_embed(model, image_list)
    except Exception:
        return np.stack([_dummy_embedding(img) for img in image_list]).astype("float32")


def representative(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype="float32")
    if arr.ndim == 1:
        return _l2_normalize(arr)
    if arr.size == 0:
        return np.zeros(config.EMBEDDING_DIM, dtype="float32")
    return _l2_normalize(arr.mean(axis=0))

