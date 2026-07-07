from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.services.shoe_db import (
    ShoeRow,
    connect_shoe_db,
    fetch_shoes_with_reviews,
    fetch_user_foot_state,
    resolve_user_id,
)
from app.services.shoe_embedding import embed_texts, get_or_embed_texts, rank_by_similarity
from app.services.shoe_feature_rules import (
    REASON_TYPES,
    bucket_review_by_reason,
    build_need_query,
    default_title,
    derive_foot_need,
    polarity_score,
    split_sentences,
)

_REASON_RISK_SCORE = {"LOW": 90.0, "MEDIUM": 60.0, "HIGH": 30.0}


class ShoeRecommendationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReasonFacts:
    """부위 하나에 대한 판단 결과 (fitScore/riskLevel/근거 리뷰). 문장(pointSummary/reviewSummary)은
    여기서 만들지 않는다 — 신발 상세 조회 시 필요할 때만 Ollama로 따로 생성한다."""

    reason_type: str
    title: str
    risk_level: str
    review_ids: list[int]


@dataclass(frozen=True)
class ShoeFacts:
    shoe_id: int
    fit_score: float
    reasons: list[ReasonFacts]


@dataclass(frozen=True)
class ShoeRecommendationBatch:
    user_id: int
    measurement_session_id: int
    items: list[ShoeFacts]


def risk_level_from_score(score: float) -> str:
    if score >= settings.shoe_risk_low_min_score:
        return "LOW"
    if score >= settings.shoe_risk_medium_min_score:
        return "MEDIUM"
    return "HIGH"


def _dedup_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _collect_candidate_sentences(shoe: ShoeRow, reason_type: str) -> dict[str, tuple[int, str]]:
    """cache_key -> (review_id, sentence_text) for sentences that mention this reason_type."""
    candidates: dict[str, tuple[int, str]] = {}
    for review in shoe.reviews:
        for index, sentence in enumerate(split_sentences(review.review_text)):
            if reason_type in bucket_review_by_reason(sentence):
                candidates[f"{review.id}:{index}"] = (review.id, sentence)
                if len(candidates) >= settings.shoe_max_candidate_reviews_per_reason:
                    return candidates
    return candidates


def _collect_fallback_sentences(
    shoe: ShoeRow, exclude_keys: set[str], limit: int
) -> dict[str, tuple[int, str]]:
    """이 항목을 직접 언급하는 리뷰 문장이 부족할 때, 그 신발의 다른 리뷰 문장으로라도 채워서
    근거 리뷰가 비어 보이지 않게 한다."""
    fallback: dict[str, tuple[int, str]] = {}
    for review in shoe.reviews:
        for index, sentence in enumerate(split_sentences(review.review_text)):
            key = f"{review.id}:{index}"
            if key in exclude_keys:
                continue
            fallback[key] = (review.id, sentence)
            if len(fallback) >= limit:
                return fallback
    return fallback


def _score_reason_facts(shoe: ShoeRow, reason_type: str, query_vector) -> ReasonFacts:
    candidates = _collect_candidate_sentences(shoe, reason_type)
    if len(candidates) < settings.shoe_reviews_per_reason:
        candidates = {
            **candidates,
            **_collect_fallback_sentences(shoe, set(candidates), settings.shoe_max_candidate_reviews_per_reason),
        }

    selected_sentences: list[str] = []
    selected_review_ids: list[int] = []
    if candidates:
        key_to_text = {key: sentence for key, (_, sentence) in candidates.items()}
        embeddings = get_or_embed_texts(key_to_text)
        # 후보 전체를 유사도순으로 받아온 뒤, 리뷰당 하나의 문장만 채택한다.
        # 그래야 서로 다른 review_id 최대 3개가 뽑혀 실제 화면에도 리뷰 3개가 노출된다.
        ranked = rank_by_similarity(query_vector, embeddings, len(embeddings))
        seen_review_ids: set[int] = set()
        for key, _score in ranked:
            review_id, sentence = candidates[key]
            if review_id in seen_review_ids:
                continue
            selected_sentences.append(sentence)
            selected_review_ids.append(review_id)
            seen_review_ids.add(review_id)
            if len(selected_review_ids) >= settings.shoe_reviews_per_reason:
                break

    if selected_sentences:
        polarities = [polarity_score(sentence) for sentence in selected_sentences]
        polarity_avg = sum(polarities) / len(polarities)
        risk_level = "LOW" if polarity_avg > 0 else ("HIGH" if polarity_avg < 0 else "MEDIUM")
    else:
        risk_level = "MEDIUM"

    return ReasonFacts(
        reason_type=reason_type,
        title=default_title(reason_type, risk_level),
        risk_level=risk_level,
        review_ids=_dedup_preserve_order(selected_review_ids),
    )


def _score_shoe_facts(shoe: ShoeRow, query_vectors: dict[str, object]) -> ShoeFacts:
    reasons = [_score_reason_facts(shoe, reason_type, query_vectors[reason_type]) for reason_type in REASON_TYPES]

    reason_scores = {reason.reason_type: _REASON_RISK_SCORE[reason.risk_level] for reason in reasons}
    fit_score = sum(reason_scores.values()) / len(reason_scores)

    return ShoeFacts(shoe_id=shoe.id, fit_score=fit_score, reasons=reasons)


def compute_shoe_recommendations(measurement_session_id: int) -> ShoeRecommendationBatch:
    """measurement_session_id로 유저를 특정한 뒤, 그 유저의 최신 발 상태를 기준으로
    DB에 있는 전체 신발의 fitScore/riskLevel/근거 리뷰를 매 요청마다 새로 계산한다
    (저장된 값을 불러오지 않음). pointSummary/reviewSummary 문장은 여기서 만들지 않는다 —
    신발 상세 조회 시 필요한 신발 하나에 대해서만 별도로 생성한다 (app/api/routes/shoes.py)."""
    connection = connect_shoe_db()
    try:
        user_id = resolve_user_id(connection, measurement_session_id)
        foot_state = fetch_user_foot_state(connection, user_id)
        shoes = fetch_shoes_with_reviews(connection)
    finally:
        connection.close()

    if not any([foot_state.daily_foot_analysis, foot_state.tina_pedis_analysis, foot_state.hallux_valgus_analysis]):
        raise ShoeRecommendationError(f"No measurement data found for user {user_id}")

    need = derive_foot_need(foot_state)
    need_queries = {reason_type: build_need_query(reason_type, need) for reason_type in REASON_TYPES}
    query_vectors = dict(zip(REASON_TYPES, embed_texts([need_queries[rt] for rt in REASON_TYPES])))

    items = [_score_shoe_facts(shoe, query_vectors) for shoe in shoes if shoe.reviews]

    return ShoeRecommendationBatch(user_id=user_id, measurement_session_id=measurement_session_id, items=items)
