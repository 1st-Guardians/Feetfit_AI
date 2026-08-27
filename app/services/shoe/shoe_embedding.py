from __future__ import annotations

import hashlib
import gc
import logging
import threading

import numpy as np
import torch

from app.core.config import settings

_model = None
_model_device: str | None = None
_model_lock = threading.Lock()

_cache: dict[str, np.ndarray] = {}
_cache_loaded = False
_cache_dirty = False
_cache_lock = threading.Lock()

logger = logging.getLogger(__name__)

_CACHE_FORMAT = "feetfit-shoe-review-embedding-v2"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported embedding device: {device_arg}")
    if device_arg == "cuda" and not torch.cuda.is_available():
        logger.warning("BGE-M3 CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device_arg


def embedding_runtime_status() -> dict[str, str | bool | None]:
    cuda_available = torch.cuda.is_available()
    return {
        "cudaAvailable": cuda_available,
        "gpuName": torch.cuda.get_device_name(0) if cuda_available else None,
        "pytorchCudaVersion": torch.version.cuda,
        "bgeM3ConfiguredDevice": settings.shoe_embedding_device,
        "bgeM3ResolvedDevice": _model_device or resolve_device(settings.shoe_embedding_device),
    }


def get_model():
    global _model, _model_device
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = resolve_device(settings.shoe_embedding_device)
                try:
                    _model = SentenceTransformer(
                        settings.shoe_embedding_model_name, device=device
                    )
                    _model_device = device
                except Exception:
                    if device != "cuda":
                        raise
                    logger.warning(
                        "BGE-M3 CUDA initialization failed; retrying once on CPU.",
                        exc_info=True,
                    )
                    torch.cuda.empty_cache()
                    _model = SentenceTransformer(
                        settings.shoe_embedding_model_name, device="cpu"
                    )
                    _model_device = "cpu"
                logger.info(
                    "BGE-M3 initialized: device=%s model=%s",
                    _model_device,
                    settings.shoe_embedding_model_name,
                )
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
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    cache_format = str(data["cacheFormat"].item())
                    model_name = str(data["modelName"].item())
                    vector_dim = int(data["vectorDim"].item())
                    keys = data["keys"]
                    vectors = data["vectors"]
                    valid = (
                        cache_format == _CACHE_FORMAT
                        and model_name == settings.shoe_embedding_model_name
                        and vectors.ndim == 2
                        and vectors.shape[0] == len(keys)
                        and vectors.shape[1] == vector_dim
                    )
                    if valid:
                        _cache.update(
                            {str(keys[i]): vectors[i] for i in range(len(keys))}
                        )
                    else:
                        logger.info(
                            "Ignoring incompatible BGE-M3 cache; it will be atomically rebuilt."
                        )
            except (KeyError, OSError, ValueError, TypeError):
                logger.warning(
                    "Ignoring corrupt/legacy BGE-M3 cache; it will be atomically rebuilt.",
                    exc_info=True,
                )
        _cache_loaded = True


def _text_cache_key(text: str) -> str:
    material = (
        f"shoe-review-embedding-v2\0{settings.shoe_embedding_model_name}\0{text}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def flush_embedding_cache() -> None:
    """Atomically persist once after a recommendation run, not once per shoe."""
    global _cache_dirty
    _load_cache()
    cache_path = settings.shoe_review_embedding_cache_path
    with _cache_lock:
        if not _cache or not _cache_dirty:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f"{cache_path.name}.tmp.npz")
        keys = np.array(list(_cache.keys()))
        vectors = np.stack(list(_cache.values())).astype(np.float32)
        np.savez(
            temporary_path,
            cacheFormat=np.array(_CACHE_FORMAT),
            modelName=np.array(settings.shoe_embedding_model_name),
            vectorDim=np.array(vectors.shape[1], dtype=np.int64),
            keys=keys,
            vectors=vectors,
        )
        temporary_path.replace(cache_path)
        _cache_dirty = False


def embed_texts(texts: list[str]) -> np.ndarray:
    global _model, _model_device
    model = get_model()
    try:
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
    except Exception:
        if _model_device != "cuda":
            raise
        logger.warning(
            "BGE-M3 CUDA encode failed; retrying the batch once on CPU.",
            exc_info=True,
        )
        with _model_lock:
            from sentence_transformers import SentenceTransformer

            _model = None
            torch.cuda.empty_cache()
            _model = SentenceTransformer(
                settings.shoe_embedding_model_name, device="cpu"
            )
            _model_device = "cpu"
            model = _model
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
    return vectors.astype(np.float32)


def get_or_embed_texts(key_to_text: dict[str, str]) -> dict[str, np.ndarray]:
    """Embed sentences, reusing a disk cache keyed by an arbitrary string id
    (e.g. "{review_id}:{sentence_index}") since embeddings only depend on
    the sentence content and are reused across every user's request."""
    _load_cache()

    global _cache_dirty
    requested_cache_keys = {
        requested_key: _text_cache_key(text)
        for requested_key, text in key_to_text.items()
    }
    missing_cache_keys = list(
        dict.fromkeys(
            cache_key
            for cache_key in requested_cache_keys.values()
            if cache_key not in _cache
        )
    )
    if missing_cache_keys:
        text_by_cache_key = {
            _text_cache_key(text): text for text in key_to_text.values()
        }
        missing_texts = [text_by_cache_key[key] for key in missing_cache_keys]
        vectors = embed_texts(missing_texts)
        if vectors.ndim != 2 or vectors.shape[0] != len(missing_cache_keys):
            raise ValueError("BGE-M3 returned an invalid embedding matrix shape")
        rebuild_all = (
            bool(_cache)
            and len(next(iter(_cache.values()))) != vectors.shape[1]
        )
        if rebuild_all:
            logger.warning(
                "BGE-M3 vector dimension changed; rebuilding the cache atomically."
            )
            missing_cache_keys = list(dict.fromkeys(requested_cache_keys.values()))
            missing_texts = [text_by_cache_key[key] for key in missing_cache_keys]
            vectors = embed_texts(missing_texts)
            if vectors.ndim != 2 or vectors.shape[0] != len(missing_cache_keys):
                raise ValueError("BGE-M3 returned an invalid embedding matrix shape")
        with _cache_lock:
            if rebuild_all:
                _cache.clear()
            for key, vector in zip(missing_cache_keys, vectors, strict=True):
                _cache[key] = vector
            _cache_dirty = True

    return {
        requested_key: _cache[cache_key]
        for requested_key, cache_key in requested_cache_keys.items()
    }


def release_embedding_model() -> None:
    """Release BGE-M3 VRAM after the automatic batch before Ollama is used."""
    global _model, _model_device
    if not settings.shoe_release_embedding_model_after_batch:
        return
    with _model_lock:
        released_device = _model_device
        _model = None
        _model_device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("BGE-M3 model released after batch (previous_device=%s).", released_device)


def rank_by_similarity(query_vector: np.ndarray, candidates: dict[str, np.ndarray], top_k: int) -> list[tuple[str, float]]:
    if not candidates:
        return []
    keys = list(candidates.keys())
    matrix = np.stack([candidates[key] for key in keys])
    scores = matrix @ query_vector
    ranked_indices = np.argsort(-scores)[:top_k]
    return [(keys[i], float(scores[i])) for i in ranked_indices]
