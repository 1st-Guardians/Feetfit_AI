from __future__ import annotations

from dataclasses import dataclass
import threading

from app.core.config import settings
from app.schemas.shoe_server import ServerRecommendationContext, ServerShoe
from app.services.shoe.shoe_embedding import (
    embed_texts,
    flush_embedding_cache,
    get_or_embed_texts,
    rank_by_similarity,
    release_embedding_model,
)
from app.services.shoe.shoe_feature_rules import (
    REASON_TYPES,
    build_need_query,
    derive_foot_need,
    extract_reason_evidence,
    normalize_review_text,
    split_sentences,
)
from app.services.shoe.shoe_fit_policy import (
    AreaScore,
    ShoeFitPolicyError,
    overall_fit_score,
    risk_level_from_score,
    score_shoe_fit,
)


class ShoeRecommendationError(RuntimeError):
    pass


class ShoeRecommendationBusyError(ShoeRecommendationError):
    pass


_batch_lock = threading.Lock()


@dataclass(frozen=True)
class ReviewCandidate:
    review_id: int
    sentence: str
    similarity: float


@dataclass(frozen=True)
class ReasonFacts:
    """A quantitative area result plus shoe-local MUSINSA evidence.

    ``score`` and ``risk_level`` come only from measurementSession + RunRepeat
    measurements. Review text never changes either value.
    """

    reason_type: str
    score: float
    title: str
    risk_level: str
    review_ids: list[int]
    review_candidates: tuple[ReviewCandidate, ...] = ()


@dataclass(frozen=True)
class ShoeFacts:
    shoe_id: int
    fit_score: float
    reasons: list[ReasonFacts]
    point_summary: str | None = None


@dataclass(frozen=True)
class ShoeRecommendationBatch:
    user_id: int
    measurement_session_id: int
    items: list[ShoeFacts]


def _sentence_candidates(shoe: ServerShoe) -> dict[str, tuple[int, str]]:
    candidates: dict[str, tuple[int, str]] = {}
    seen_review_bodies: set[str] = set()
    seen_sentences: set[str] = set()
    # The smallest reviewId is the deterministic canonical identity when
    # MUSINSA contains separate review records with the same body.
    for review in sorted(shoe.reviews, key=lambda item: item.id):
        body_fingerprint = normalize_review_text(review.review_text)
        if not body_fingerprint or body_fingerprint in seen_review_bodies:
            continue
        seen_review_bodies.add(body_fingerprint)
        for index, sentence in enumerate(split_sentences(review.review_text)):
            sentence_fingerprint = normalize_review_text(sentence)
            if not sentence_fingerprint or sentence_fingerprint in seen_sentences:
                continue
            seen_sentences.add(sentence_fingerprint)
            candidates[f"{shoe.id}:{review.id}:{index}"] = (review.id, sentence)
    return candidates


def _semantic_shortlists(
    shoe: ServerShoe, query_vectors: dict[str, object]
) -> dict[str, tuple[ReviewCandidate, ...]]:
    sentences = _sentence_candidates(shoe)
    if not sentences:
        return {reason_type: () for reason_type in REASON_TYPES}

    embeddings = get_or_embed_texts(
        {key: sentence for key, (_review_id, sentence) in sentences.items()}
    )
    results: dict[str, tuple[ReviewCandidate, ...]] = {}
    for reason_type in REASON_TYPES:
        ranked = rank_by_similarity(
            query_vectors[reason_type],
            embeddings,
            len(embeddings),
        )
        selected: list[ReviewCandidate] = []
        seen_review_ids: set[int] = set()
        for key, similarity in ranked:
            review_id, sentence = sentences[key]
            evidence = extract_reason_evidence(reason_type, sentence)
            if evidence is None:
                continue
            if review_id in seen_review_ids:
                continue
            seen_review_ids.add(review_id)
            # Apply the candidate cap after collapsing multiple sentences from
            # one review. Otherwise a long/duplicated body can crowd unique
            # reviews out of the ranking window.
            if len(seen_review_ids) > settings.shoe_max_candidate_reviews_per_reason:
                break
            if similarity < settings.shoe_review_semantic_min_score:
                continue
            selected.append(
                ReviewCandidate(
                    review_id=review_id,
                    sentence=evidence,
                    similarity=round(float(similarity), 6),
                )
            )
            if len(selected) >= settings.shoe_reviews_per_reason:
                break
        results[reason_type] = tuple(selected)
    return results


def _to_reason_facts(
    area: AreaScore, candidates: tuple[ReviewCandidate, ...]
) -> ReasonFacts:
    return ReasonFacts(
        reason_type=area.reason_type,
        score=area.score,
        title=area.title,
        risk_level=area.risk_level,
        review_ids=[candidate.review_id for candidate in candidates],
        review_candidates=candidates,
    )


def _score_shoe_facts(
    shoe: ServerShoe,
    context: ServerRecommendationContext,
    query_vectors: dict[str, object] | None,
) -> ShoeFacts:
    try:
        areas = score_shoe_fit(shoe, context.foot_state)
    except ShoeFitPolicyError as exc:
        raise ShoeRecommendationError(str(exc)) from exc
    shortlists = (
        _semantic_shortlists(shoe, query_vectors)
        if query_vectors is not None and shoe.reviews
        else {reason_type: () for reason_type in REASON_TYPES}
    )
    return ShoeFacts(
        shoe_id=shoe.id,
        fit_score=overall_fit_score(areas),
        reasons=[
            _to_reason_facts(area, shortlists[area.reason_type]) for area in areas
        ],
    )


def _compute_shoe_recommendations(
    context: ServerRecommendationContext,
) -> ShoeRecommendationBatch:
    """Compute an idempotent, session-scoped recommendation for every shoe.

    The function uses only the supplied completed-session context. Reviews are
    optional evidence and never gate or affect quantitative scoring. Missing
    5/7 characteristics are not synthesized; each area reweights only its
    actually present RunRepeat components.
    """
    foot_state = context.foot_state
    if not any(
        [
            foot_state.daily_foot_analysis,
            foot_state.tina_pedis_analysis,
            foot_state.hallux_valgus_analysis,
            foot_state.static_pressure_analyses,
        ]
    ):
        raise ShoeRecommendationError(
            f"No measurement data found for measurement session {context.measurement_session_id}"
        )
    if not context.shoes:
        raise ShoeRecommendationError("Feetfit_Server returned an empty shoe catalog.")
    shoe_ids = [shoe.id for shoe in context.shoes]
    if len(shoe_ids) != len(set(shoe_ids)):
        raise ShoeRecommendationError("Recommendation context contains duplicate shoeId values.")

    need = derive_foot_need(foot_state)
    query_vectors: dict[str, object] | None = None
    if any(shoe.reviews for shoe in context.shoes):
        queries = [build_need_query(reason_type, need) for reason_type in REASON_TYPES]
        query_vectors = dict(zip(REASON_TYPES, embed_texts(queries), strict=True))

    try:
        items = [
            _score_shoe_facts(shoe, context, query_vectors) for shoe in context.shoes
        ]
    finally:
        try:
            flush_embedding_cache()
        finally:
            release_embedding_model()
    if [item.shoe_id for item in items] != shoe_ids:
        raise ShoeRecommendationError("Recommendation output does not preserve full shoe coverage.")
    return ShoeRecommendationBatch(
        user_id=context.user_id,
        measurement_session_id=context.measurement_session_id,
        items=items,
    )


def compute_shoe_recommendations(
    context: ServerRecommendationContext,
) -> ShoeRecommendationBatch:
    acquired = _batch_lock.acquire(
        timeout=settings.shoe_recommendation_batch_lock_timeout_seconds
    )
    if not acquired:
        raise ShoeRecommendationBusyError(
            "Another Phase D recommendation batch is using the shared BGE-M3 runtime."
        )
    try:
        return _compute_shoe_recommendations(context)
    finally:
        _batch_lock.release()


__all__ = [
    "ReasonFacts",
    "ReviewCandidate",
    "ShoeFacts",
    "ShoeRecommendationBatch",
    "ShoeRecommendationBusyError",
    "ShoeRecommendationError",
    "compute_shoe_recommendations",
    "risk_level_from_score",
]
