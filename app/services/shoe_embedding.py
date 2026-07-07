from __future__ import annotations

import threading

import numpy as np
import torch

from app.core.config import settings

_model = None
_model_lock = threading.Lock()

_cache: dict[str, np.ndarray] = {}
_cache_loaded = False
_cache_lock = threading.Lock()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported embedding device: {device_arg}")
    return device_arg


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = resolve_device(settings.shoe_embedding_device)
                _model = SentenceTransformer(settings.shoe_embedding_model_name, device=device)
    return _model


def _load_cache() -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    with _cache_lock:
        if _cache_loaded:
            return
        cache_path = settings.shoe_review_embedding_cache_path
        if cache_path.exists():
            data = np.load(cache_path)
            keys = data["keys"]
            vectors = data["vectors"]
            _cache.update({str(keys[i]): vectors[i] for i in range(len(keys))})
        _cache_loaded = True


def _save_cache() -> None:
    cache_path = settings.shoe_review_embedding_cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not _cache:
        return
    keys = np.array(list(_cache.keys()))
    vectors = np.stack(list(_cache.values())).astype(np.float32)
    np.savez(cache_path, keys=keys, vectors=vectors)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def get_or_embed_texts(key_to_text: dict[str, str]) -> dict[str, np.ndarray]:
    """Embed sentences, reusing a disk cache keyed by an arbitrary string id
    (e.g. "{review_id}:{sentence_index}") since embeddings only depend on
    the sentence content and are reused across every user's request."""
    _load_cache()

    missing_keys = [key for key in key_to_text if key not in _cache]
    if missing_keys:
        missing_texts = [key_to_text[key] for key in missing_keys]
        vectors = embed_texts(missing_texts)
        with _cache_lock:
            for key, vector in zip(missing_keys, vectors):
                _cache[key] = vector
            _save_cache()

    return {key: _cache[key] for key in key_to_text}


def rank_by_similarity(query_vector: np.ndarray, candidates: dict[str, np.ndarray], top_k: int) -> list[tuple[str, float]]:
    if not candidates:
        return []
    keys = list(candidates.keys())
    matrix = np.stack([candidates[key] for key in keys])
    scores = matrix @ query_vector
    ranked_indices = np.argsort(-scores)[:top_k]
    return [(keys[i], float(scores[i])) for i in ranked_indices]
