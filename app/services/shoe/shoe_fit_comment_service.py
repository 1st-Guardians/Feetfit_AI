from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from dataclasses import dataclass

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError

from app.core.config import settings
from app.prompts.shoe_fit_comment_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.schemas.shoe_fit_comment import ShoeFitSummary, ShoePointSummaryClaim
from app.schemas.shoe_server import ServerShoeCharacteristics
from app.services.shoe.shoe_feature_rules import (
    REASON_TYPES,
    extract_reason_evidence,
    normalize_review_text,
)
from app.services.shoe.shoe_point_summary import (
    PointEvidenceFact,
    format_point_evidence_facts,
    join_point_summary_claims,
    prepare_point_evidence_facts,
    render_product_point_summary,
)

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_DISPLAY_MODEL_CODE_SUFFIX_PATTERN = re.compile(
    r"\s*/\s*[A-Z0-9][A-Z0-9._-]{4,}$", re.IGNORECASE
)
_NO_REVIEW_NOTE = "없음 (판단만 근거로 사용하고, 구체적인 리뷰 내용을 지어내지 말 것)"

_REASON_LABEL = {"FOREFOOT": "발볼/앞코", "HEEL": "뒤꿈치", "INSOLE": "깔창"}
_SUMMARY_FIELD_BY_REASON = {
    "FOREFOOT": "forefoot_summary",
    "HEEL": "heel_summary",
    "INSOLE": "insole_summary",
}
_REVIEW_IDS_FIELD_BY_REASON = {
    "FOREFOOT": "forefoot_review_ids",
    "HEEL": "heel_review_ids",
    "INSOLE": "insole_review_ids",
}
_RISK_LABEL = {"LOW": "좋음", "MEDIUM": "보통", "HIGH": "주의"}

# These topics cannot be valid evidence for FOREFOOT/HEEL/INSOLE fit. They are
# rejected even when they appear in an untrusted review body; this closes the
# exact failure mode where fashion text or an emotional body metaphor was
# rewritten as a shoe-fit claim.
_REASON_OUT_OF_SCOPE_SUMMARY_TERMS = (
    "바지",
    "밑단",
    "코디",
    "패션",
    "옷",
    "예뻐",
    "예쁘",
    "이뻐",
    "디자인",
    "색상",
    "가슴",
    "허리",
    "어깨",
    "치료",
    "진단",
    "질환",
    "병원",
)
_POINT_OUT_OF_SCOPE_SUMMARY_TERMS = (
    "가슴",
    "허리",
    "어깨",
    "치료",
    "진단",
    "질환",
    "병원",
)
_REVIEW_GROUNDED_CONCEPTS = (
    ("통기", "통풍", "습도", "땀", "열감", "답답"),
    ("쿠션", "푹신", "딱딱", "충격"),
    ("발목",),
    ("사이즈", "크기", "사이즈 업", "사이즈업", "사이즈 다운", "사이즈다운", "1업", "반업"),
    ("무게", "무겁"),
    ("가볍",),
    ("마찰", "쓸림", "까짐", "까졌"),
    ("압박", "눌림", "좁"),
    ("넓", "여유", "넉넉"),
    ("적응",),
    ("해결",),
    ("개선",),
    ("완화",),
)
_POINT_GROUNDED_CONCEPTS = (
    ("색감", "색상"),
    ("디자인", "실루엣"),
    ("예쁘", "예뻐", "이뻐"),
    ("청바지", "데님"),
    ("슬랙스",),
    ("스커트", "치마"),
    ("통큰 바지", "와이드 팬츠"),
    ("바지",),
    ("데일리", "일상용"),
    ("코디",),
    ("날렵",),
    ("로우", "낮은 굽", "굽은 낮"),
    ("정사이즈",),
    ("딱 맞", "정사이즈"),
    ("반 사이즈", "반업"),
    ("사이즈 업", "사이즈업", "1업"),
)
_AREA_CONCEPTS = {
    "FOREFOOT": ("발볼", "앞볼", "앞코", "앞꿈치", "발가락", "엄지"),
    "HEEL": ("뒤꿈치", "뒤축", "뒷굽", "발목"),
    "INSOLE": ("깔창", "발바닥"),
}
_CAUSAL_CLAIM_SUBJECTS = (
    "발볼",
    "앞코",
    "뒤꿈치",
    "발목",
    "깔창",
    "쿠션",
    "통기성",
    "통풍",
    "바닥",
    "무게",
)

_llm: ChatOllama | None = None
_cpu_llm: ChatOllama | None = None
_llm_lock = threading.Lock()
_ollama_semaphore = asyncio.Semaphore(settings.ollama_max_concurrency)

logger = logging.getLogger(__name__)


class ShoeFitCommentError(RuntimeError):
    pass


def _display_shoe_name(shoe_name: str) -> str:
    return _DISPLAY_MODEL_CODE_SUFFIX_PATTERN.sub("", shoe_name).strip() or shoe_name


@dataclass(frozen=True)
class ReasonFactsForPrompt:
    reason_type: str  # "FOREFOOT" | "HEEL" | "INSOLE"
    title: str
    risk_level: str
    review_ids: list[int]
    review_texts: list[str]

    def __post_init__(self) -> None:
        if self.reason_type not in REASON_TYPES:
            raise ValueError(f"Unknown reason_type: {self.reason_type}")
        if self.risk_level not in _RISK_LABEL:
            raise ValueError(f"Unknown risk_level: {self.risk_level}")
        if len(self.review_ids) != len(self.review_texts):
            raise ValueError("review_ids and review_texts must have the same length")
        if len(self.review_ids) > 3 or len(self.review_ids) != len(set(self.review_ids)):
            raise ValueError("review_ids must be unique and contain at most three ids")
        if any(review_id < 1 for review_id in self.review_ids):
            raise ValueError("review_ids must contain positive ids")


def prepare_grounded_reasons(
    reasons: list[ReasonFactsForPrompt],
) -> list[ReasonFactsForPrompt]:
    """Canonicalize legacy evidence and expose only reason-specific clauses.

    The first recommendation stage persists review IDs, not BGE's selected
    sentence. A later Server response therefore contains each full review body.
    This boundary reconstructs a conservative excerpt and removes duplicate
    bodies before any untrusted text reaches Ollama.
    """

    reason_types = [reason.reason_type for reason in reasons]
    if len(reason_types) != len(set(reason_types)) or set(reason_types) != set(REASON_TYPES):
        raise ShoeFitCommentError(
            "Summary facts must contain FOREFOOT, HEEL, and INSOLE exactly once."
        )

    prepared: list[ReasonFactsForPrompt] = []
    for reason_type in REASON_TYPES:
        reason = next(item for item in reasons if item.reason_type == reason_type)

        # Preserve the first relevance-ranked position, while using the
        # smallest reviewId as the stable canonical ID for a duplicate body.
        body_groups: dict[str, tuple[int, int, str]] = {}
        for index, (review_id, review_text) in enumerate(
            zip(reason.review_ids, reason.review_texts, strict=True)
        ):
            body_fingerprint = normalize_review_text(review_text)
            if not body_fingerprint:
                continue
            current = body_groups.get(body_fingerprint)
            if current is None:
                body_groups[body_fingerprint] = (index, review_id, review_text)
            elif review_id < current[1]:
                body_groups[body_fingerprint] = (current[0], review_id, review_text)

        canonical_pairs: list[tuple[int, str]] = []
        seen_evidence: set[str] = set()
        for _index, review_id, review_text in sorted(
            body_groups.values(), key=lambda item: item[0]
        ):
            evidence = extract_reason_evidence(reason_type, review_text)
            if evidence is None:
                continue
            evidence_fingerprint = normalize_review_text(evidence)
            if evidence_fingerprint in seen_evidence:
                continue
            seen_evidence.add(evidence_fingerprint)
            canonical_pairs.append((review_id, evidence))
            if len(canonical_pairs) >= 3:
                break

        prepared.append(
            ReasonFactsForPrompt(
                reason_type=reason.reason_type,
                title=reason.title,
                risk_level=reason.risk_level,
                review_ids=[review_id for review_id, _evidence in canonical_pairs],
                review_texts=[evidence for _review_id, evidence in canonical_pairs],
            )
        )
    return prepared


def prepare_point_summary_evidence(
    reasons: list[ReasonFactsForPrompt],
    characteristics: ServerShoeCharacteristics | None = None,
) -> list[PointEvidenceFact]:
    """Aggregate typed facts without losing reviewId provenance or consensus."""

    review_pairs = [
        (review_id, review_text)
        for reason_type in REASON_TYPES
        for reason in reasons
        if reason.reason_type == reason_type
        for review_id, review_text in zip(
            reason.review_ids, reason.review_texts, strict=True
        )
    ]
    try:
        return prepare_point_evidence_facts(review_pairs, characteristics)
    except ValueError as exc:
        raise ShoeFitCommentError(str(exc)) from exc


def _new_llm(num_gpu: int) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        format="json",
        num_gpu=num_gpu,
        client_kwargs={"timeout": settings.ollama_request_timeout_seconds},
    )


def _get_llm(*, force_cpu: bool = False) -> ChatOllama:
    global _llm, _cpu_llm
    if force_cpu:
        if _cpu_llm is None:
            with _llm_lock:
                if _cpu_llm is None:
                    _cpu_llm = _new_llm(0)
        return _cpu_llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = _new_llm(settings.ollama_num_gpu)
    return _llm


async def ollama_runtime_status() -> dict[str, object]:
    """Read-only Ollama GPU preflight. It never loads a model or triggers inference."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/ps"
    try:
        async with httpx.AsyncClient(
            timeout=min(settings.ollama_request_timeout_seconds, 10.0)
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "reachable": False,
            "model": settings.ollama_model,
            "gpuInUse": None,
            "detail": str(exc),
        }
    models = payload.get("models", []) if isinstance(payload, dict) else []
    loaded = next(
        (
            model
            for model in models
            if isinstance(model, dict)
            and str(model.get("name", "")).split(":")[0]
            == settings.ollama_model.split(":")[0]
        ),
        None,
    )
    size_vram = loaded.get("size_vram") if loaded else None
    return {
        "reachable": True,
        "model": settings.ollama_model,
        "gpuInUse": bool(size_vram) if size_vram is not None else None,
        "sizeVram": size_vram,
    }


def _format_reason_block(reason: ReasonFactsForPrompt) -> str:
    if reason.review_texts:
        review_lines = "\n".join(
            f'  {i + 1}. reviewId={review_id}: "{text}"'
            for i, (review_id, text) in enumerate(
                zip(reason.review_ids, reason.review_texts, strict=True)
            )
        )
    else:
        review_lines = f"  {_NO_REVIEW_NOTE}"

    label = _REASON_LABEL[reason.reason_type]
    return (
        f"- 항목: {label} ({reason.reason_type})\n"
        f"- 제목: {reason.title}\n"
        f"- 위험도: {reason.risk_level}\n"
        f"- 참고 리뷰 문장:\n{review_lines}"
    )


def _format_point_evidence(point_evidence: list[PointEvidenceFact]) -> str:
    return format_point_evidence_facts(point_evidence)


def _build_user_prompt(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> str:
    reason_by_type = {reason.reason_type: reason for reason in reasons}
    return USER_PROMPT_TEMPLATE.format(
        shoe_name=shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        forefoot_block=_format_reason_block(reason_by_type["FOREFOOT"]),
        heel_block=_format_reason_block(reason_by_type["HEEL"]),
        insole_block=_format_reason_block(reason_by_type["INSOLE"]),
        point_evidence_block=_format_point_evidence(point_evidence),
    )


def _parse_llm_json(raw_text: str) -> dict:
    """정상적인 JSON이면 그대로 파싱하고, 실패하면 텍스트 안에서 JSON 블록만 추출해 재파싱한다."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_PATTERN.search(text)
    if not match:
        raise ShoeFitCommentError(f"Ollama 응답에서 JSON을 찾을 수 없습니다: {text[:200]}")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ShoeFitCommentError(f"Ollama 응답 JSON 파싱에 실패했습니다: {text[:200]}") from exc


def _validate_review_selection(
    summary: ShoeFitSummary, reasons: list[ReasonFactsForPrompt]
) -> None:
    allowed = {reason.reason_type: set(reason.review_ids) for reason in reasons}
    selected = {
        "FOREFOOT": summary.forefoot_review_ids,
        "HEEL": summary.heel_review_ids,
        "INSOLE": summary.insole_review_ids,
    }
    for reason_type, review_ids in selected.items():
        if not set(review_ids).issubset(allowed[reason_type]):
            raise ShoeFitCommentError(
                "Ollama selected a reviewId outside the exact shoe/reason candidate set "
                f"(reasonType={reason_type})."
            )


def _ensure_review_selection_minimum(
    summary: ShoeFitSummary, reasons: list[ReasonFactsForPrompt]
) -> ShoeFitSummary:
    """Keep 1-3 selected reviews whenever grounded candidates exist.

    An empty list remains valid only when the semantic + lexical grounding
    stages found no related evidence. We never fill it with an unrelated review.
    """

    updates: dict[str, list[int]] = {}
    for reason in reasons:
        field_name = _REVIEW_IDS_FIELD_BY_REASON[reason.reason_type]
        selected_ids = list(getattr(summary, field_name))
        if reason.review_ids and not selected_ids:
            selected_ids = [reason.review_ids[0]]
        updates[field_name] = selected_ids
    return summary.model_copy(update=updates)


def _ensure_point_evidence_reviews_selected(
    summary: ShoeFitSummary,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> ShoeFitSummary:
    """Keep reviews cited by pointSummary claims in the persisted 1-3 review set."""

    facts_by_id = {fact.evidence_id: fact for fact in point_evidence}
    cited_review_ids: set[int] = set()
    for claim in summary.point_summary_claims:
        for evidence_id in claim.evidence_ids:
            fact = facts_by_id.get(evidence_id)
            if fact is not None:
                cited_review_ids.update(fact.review_ids)

    updates: dict[str, list[int]] = {}
    for reason in reasons:
        field_name = _REVIEW_IDS_FIELD_BY_REASON[reason.reason_type]
        existing = list(getattr(summary, field_name))
        cited_candidates = [
            review_id
            for review_id in reason.review_ids
            if review_id in cited_review_ids
        ]
        selected = list(dict.fromkeys([*cited_candidates, *existing]))[:3]
        if reason.review_ids and not selected:
            selected = [reason.review_ids[0]]
        updates[field_name] = selected
    return summary.model_copy(update=updates)


def _contains_out_of_scope_content(text: str, blocked_terms: tuple[str, ...]) -> bool:
    normalized = normalize_review_text(text)
    return any(term in normalized for term in blocked_terms)


def _uses_unsupported_concept(
    text: str,
    source_text: str,
    *,
    additional_concepts: tuple[tuple[str, ...], ...] = (),
) -> bool:
    normalized_text = normalize_review_text(text)
    normalized_source = normalize_review_text(source_text)

    for terms in (*_REVIEW_GROUNDED_CONCEPTS, *additional_concepts):
        if any(term in normalized_text for term in terms) and not any(
            term in normalized_source for term in terms
        ):
            return True
    for terms in _AREA_CONCEPTS.values():
        for term in terms:
            if term in normalized_text and term not in normalized_source:
                return True
    for subject in _CAUSAL_CLAIM_SUBJECTS:
        output_causal_forms = (
            f"{subject} 때문",
            f"{subject}때문",
            f"{subject}으로 인해",
            f"{subject}로 인해",
        )
        if any(form in normalized_text for form in output_causal_forms) and not any(
            form in normalized_source for form in output_causal_forms
        ):
            return True
    return False


_POINT_REPORT_STYLE_PATTERNS = (
    "후기를 참고",
    "후기에 따르면",
    "후기상",
    "리뷰에 따르면",
    "리뷰상",
    "구매평",
    "착화평",
    "평가에 따르면",
    "확인하는 것이 좋",
    "체크하는 것이 좋",
    "체크해 보",
    "참고할 만",
    "라는 의견이 있고",
    "라는 후기가 있고",
    "한편",
    "발볼 리뷰에서는",
    "뒤꿈치 리뷰에서는",
    "깔창 리뷰에서는",
)
_POINT_HEDGE_MARKERS = (
    "일부 착화",
    "한 착화",
    "한 사례",
    "개인차",
    "착화에 따라",
    "느껴질 수",
    "생길 수",
)
_POINT_MIXED_SCOPE_MARKERS = (
    "개인차",
    "엇갈",
    "서로 다르",
    "착화에 따라",
    "착화 경험에 따라",
)
_POINT_UNIVERSAL_CLAIM_TERMS = (
    "누구나",
    "대부분",
    "항상",
    "모든 사람",
    "모든 사용자",
    "무조건",
)
_POINT_UNTRACKED_ATTRIBUTE_TERMS = (
    "내구성",
    "내구력이",
    "방수",
    "방풍",
    "접지력",
    "미끄럼 방지",
    "보온성",
    "가격 대비",
    "가성비",
)
_POINT_EXAGGERATION_TERMS = (
    "최고",
    "최상",
    "압도적",
    "완벽",
    "극강",
    "구름 위",
    "탁월",
)

_SIZE_THREE_DIGIT_PATTERN = re.compile(r"(?<!\d)(?:2\d{2}|3\d{2})(?!\d)")
_SIZE_EXACT_STEP_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*업")


def _claim_uses_uncited_signal(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    normalized = normalize_review_text(text)

    def cited(subject: str, *values: str) -> bool:
        return any(
            fact.signal.subject == subject
            and (not values or fact.signal.value in values)
            and fact.mode != "CONFLICT"
            for fact in cited_facts
        )

    checks: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
        (("한 사이즈 업", "1업"), "SIZE_OPTION", ("FULL_UP",)),
        (("반 사이즈 업", "반업"), "SIZE_OPTION", ("HALF_UP",)),
        (("정사이즈",), "SIZE_OPTION", ("TRUE_SIZE",)),
        (("슬림", "좁은 편", "밀착되는 핏"), "WIDTH_SPACE", ("TIGHT",)),
        (("여유로운 편", "넉넉한 편"), "WIDTH_SPACE", ("ROOMY",)),
        (("안정적으로 잡", "발목을 잡"), "HEEL_HOLD", ("STABLE",)),
        (("헐렁", "들뜨"), "HEEL_HOLD", ("LOOSE",)),
        (("바닥이 얇", "바닥은 얇"), "SOLE_THICKNESS", ("THIN",)),
        (("바닥이 두껍", "바닥은 두껍"), "SOLE_THICKNESS", ("THICK",)),
        (
            ("쿠션감은 높은", "쿠션감이 높은", "쿠션은 부드러운", "부드럽게 느껴"),
            "CUSHION_SOFTNESS",
            ("SOFT",),
        ),
        (("쿠션감은 낮", "쿠션감이 낮", "푹신한 타입보다"), "CUSHION_SOFTNESS", ("FIRM",)),
        (("오래 서", "장시간", "종일"), "LONG_WEAR_COMFORT", ("DISCOMFORT", "COMFORTABLE")),
        (("무게감", "무겁"), "WEIGHT", ("HEAVY",)),
        (("가볍",), "WEIGHT", ("LIGHT",)),
        (("색감", "디자인"), "APPEARANCE", ("POSITIVE", "MENTIONED")),
        (("청바지", "데님"), "OUTFIT", ("JEANS",)),
        (("슬랙스",), "OUTFIT", ("SLACKS",)),
        (("스커트", "치마"), "OUTFIT", ("SKIRT",)),
        (("통큰 바지", "와이드 팬츠"), "OUTFIT", ("WIDE_PANTS",)),
        (("데일리", "일상용"), "OUTFIT", ("DAILY",)),
        (("날렵",), "SILHOUETTE", ("SLIM",)),
        (("로우 프로파일", "낮은 실루엣"), "PROFILE", ("LOW",)),
    )
    for terms, subject, values in checks:
        if any(term in normalized for term in terms) and not cited(subject, *values):
            return True

    if (
        ("사이즈 업" in normalized or "사이즈업" in normalized)
        and "반 사이즈 업" not in normalized
        and "한 사이즈 업" not in normalized
        and not cited("SIZE_OPTION", "GENERIC_UP")
    ):
        return True
    if (
        any(term in normalized for term in ("사이즈 다운", "사이즈다운", "사이즈를 줄", "작게 신"))
        and not cited("SIZE_OPTION", "DOWN")
    ):
        return True
    if "runrepeat" in normalized and not any(
        fact.source == "RUNREPEAT" for fact in cited_facts
    ):
        return True
    return False


def _single_facts_are_safely_scoped(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    single_facts = [
        fact
        for fact in cited_facts
        if fact.source == "MUSINSA" and fact.mode == "SINGLE"
    ]
    if not single_facts:
        return True
    normalized = normalize_review_text(text)
    if any(term in normalized for term in _POINT_UNIVERSAL_CLAIM_TERMS):
        return False
    prefix_markers = ("일부 착화", "한 착화", "한 사례", "착화에 따라")
    prefix = normalized[:40]
    return any(marker in prefix for marker in prefix_markers) or any(
        marker in normalized for marker in ("느껴질 수", "생길 수", "맞을 수")
    )


def _mixed_facts_are_safely_scoped(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    if not any(
        fact.source == "MUSINSA" and fact.mode == "MIXED"
        for fact in cited_facts
    ):
        return True
    normalized = normalize_review_text(text)
    return any(marker in normalized for marker in _POINT_MIXED_SCOPE_MARKERS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _lab_fact_is_accurately_expressed(
    text: str, fact: PointEvidenceFact
) -> bool:
    """Allow a LAB citation only for its exact characteristic and level direction."""

    normalized = normalize_review_text(text)
    subject = fact.signal.subject
    value = fact.signal.value
    specifications: dict[
        str,
        tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]],
    ] = {
        "CUSHION_SOFTNESS": (
            ("쿠션",),
            {
                "SOFT": ("높은 편", "높게", "부드럽", "부드러", "푹신"),
                "BALANCED": ("보통", "중간", "균형"),
                "FIRM": ("낮은 편", "낮게", "단단", "딱딱", "바닥감"),
            },
            ("높은 편", "높게", "부드럽", "부드러", "푹신", "보통", "중간", "균형", "낮은 편", "낮게", "단단", "딱딱", "바닥감"),
        ),
        "LAB_WIDTH_SPACE": (
            ("발볼 공간", "발볼 너비"),
            {
                "TIGHT": ("낮은 편", "좁", "타이트", "밀착"),
                "BALANCED": ("보통", "중간", "균형"),
                "ROOMY": ("높은 편", "넓", "여유", "넉넉"),
            },
            ("낮은 편", "좁", "타이트", "밀착", "보통", "중간", "균형", "높은 편", "넓", "여유", "넉넉"),
        ),
        "LAB_TOEBOX_SPACE": (
            ("앞코 공간", "토박스 공간"),
            {
                "TIGHT": ("낮은 편", "좁", "타이트", "밀착"),
                "BALANCED": ("보통", "중간", "균형"),
                "ROOMY": ("높은 편", "넓", "여유", "넉넉"),
            },
            ("낮은 편", "좁", "타이트", "밀착", "보통", "중간", "균형", "높은 편", "넓", "여유", "넉넉"),
        ),
        "HEEL_STRUCTURE": (
            ("뒤꿈치 구조", "힐 카운터", "구조 강성"),
            {
                "FLEXIBLE": ("낮은 편", "유연"),
                "BALANCED": ("보통", "중간", "균형"),
                "FIRM": ("높은 편", "단단", "강한 편"),
            },
            ("낮은 편", "유연", "보통", "중간", "균형", "높은 편", "단단", "강한 편"),
        ),
        "BREATHABILITY": (
            ("통기성", "통풍"),
            {
                "LOW": ("낮은 편", "낮게"),
                "MEDIUM": ("보통", "중간"),
                "HIGH": ("높은 편", "높게"),
            },
            ("낮은 편", "낮게", "보통", "중간", "높은 편", "높게"),
        ),
        "SHOCK_ABSORPTION": (
            ("충격 완화", "충격 흡수"),
            {
                "LOW": ("낮은 편", "낮게"),
                "MEDIUM": ("보통", "중간"),
                "HIGH": ("높은 편", "높게"),
            },
            ("낮은 편", "낮게", "보통", "중간", "높은 편", "높게"),
        ),
        "ENERGY_RETURN": (
            ("반발력", "에너지 리턴"),
            {
                "LOW": ("낮은 편", "낮게"),
                "MEDIUM": ("보통", "중간"),
                "HIGH": ("높은 편", "높게"),
            },
            ("낮은 편", "낮게", "보통", "중간", "높은 편", "높게"),
        ),
    }
    specification = specifications.get(subject)
    if specification is None:
        return False
    subject_terms, value_terms, all_direction_terms = specification
    allowed_terms = value_terms.get(value)
    if allowed_terms is None or not _contains_any(normalized, subject_terms):
        return False
    if not _contains_any(normalized, allowed_terms):
        return False
    opposite_terms = tuple(
        term for term in all_direction_terms if term not in allowed_terms
    )
    return not _contains_any(normalized, opposite_terms)


def _review_fact_is_expressed(text: str, fact: PointEvidenceFact) -> bool:
    normalized = normalize_review_text(text)
    expressions: dict[tuple[str, str], tuple[str, ...]] = {
        ("SIZE_OPTION", "FULL_UP"): ("한 사이즈 업", "1업"),
        ("SIZE_OPTION", "HALF_UP"): ("반 사이즈 업", "반업", "0.5업"),
        ("SIZE_OPTION", "TRUE_SIZE"): ("정사이즈",),
        ("SIZE_OPTION", "GENERIC_UP"): ("사이즈 업", "사이즈업", "사이즈를 올"),
        ("SIZE_OPTION", "DOWN"): ("사이즈 다운", "사이즈다운", "사이즈를 줄", "작게 신"),
        ("FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP"): ("발볼", "발등"),
        ("WIDTH_SPACE", "TIGHT"): ("좁", "타이트", "밀착", "압박"),
        ("WIDTH_SPACE", "ROOMY"): ("넓", "여유", "넉넉"),
        ("HEEL_HOLD", "STABLE"): ("안정적으로 잡", "발목을 잡", "받쳐", "지지"),
        ("HEEL_HOLD", "LOOSE"): ("헐렁", "들뜨", "벗겨", "미끄러"),
        ("HEEL_COMFORT", "RUBBING"): ("쓸", "까", "마찰", "불편"),
        ("HEEL_COMFORT", "COMFORTABLE"): ("편하", "폭신", "부드럽"),
        ("CUSHION_SOFTNESS", "SOFT"): ("푹신", "폭신", "부드럽"),
        ("CUSHION_SOFTNESS", "FIRM"): ("딱딱", "단단", "쿠션감이 낮", "바닥감"),
        ("SOLE_THICKNESS", "THIN"): ("바닥이 얇", "바닥은 얇"),
        ("SOLE_THICKNESS", "THICK"): ("바닥이 두껍", "바닥은 두껍"),
        ("LONG_WEAR_COMFORT", "DISCOMFORT"): ("오래 서", "장시간", "종일", "발바닥 부담"),
        ("LONG_WEAR_COMFORT", "COMFORTABLE"): ("오래 신어도 편", "장시간 편", "종일 편"),
        ("WEIGHT", "HEAVY"): ("무게감", "무겁"),
        ("WEIGHT", "LIGHT"): ("가볍",),
        ("APPEARANCE", "POSITIVE"): ("색감", "디자인", "예쁘", "만족"),
        ("APPEARANCE", "MENTIONED"): ("색감", "디자인", "색상"),
        ("SILHOUETTE", "SLIM"): ("날렵", "슬림"),
        ("SILHOUETTE", "ROUNDED"): ("둥글",),
        ("PROFILE", "LOW"): ("로우 프로파일", "낮은 실루엣", "굽이 낮"),
        ("OUTFIT", "JEANS"): ("청바지", "데님"),
        ("OUTFIT", "SLACKS"): ("슬랙스",),
        ("OUTFIT", "SKIRT"): ("스커트", "치마"),
        ("OUTFIT", "WIDE_PANTS"): ("통큰 바지", "와이드 팬츠"),
        ("OUTFIT", "DAILY"): ("데일리", "일상용"),
        ("OUTFIT", "GENERAL"): ("코디", "잘 어울"),
        ("PRODUCT_DETAIL", "MENTIONED"): ("소재", "굽", "착화감"),
    }
    terms = expressions.get((fact.signal.subject, fact.signal.value))
    return bool(terms and _contains_any(normalized, terms))


def _fact_is_expressed(text: str, fact: PointEvidenceFact) -> bool:
    if fact.source == "RUNREPEAT" or fact.mode == "LAB":
        return _lab_fact_is_accurately_expressed(text, fact)
    return _review_fact_is_expressed(text, fact)


def _size_claim_is_consistent(
    text: str, cited_facts: list[PointEvidenceFact]
) -> bool:
    normalized = normalize_review_text(text)

    def has_size_value(*values: str) -> bool:
        return any(
            fact.signal.subject == "SIZE_OPTION"
            and fact.signal.value in values
            and fact.mode != "CONFLICT"
            for fact in cited_facts
        )

    for raw_step in _SIZE_EXACT_STEP_PATTERN.findall(normalized):
        step = float(raw_step)
        if step == 1.0:
            if not has_size_value("FULL_UP"):
                return False
        elif step == 0.5:
            if not has_size_value("HALF_UP"):
                return False
        else:
            return False

    without_exact_units = re.sub(
        r"한\s*사이즈\s*업|반\s*사이즈\s*업|반업|사이즈업",
        "",
        normalized,
    )
    if "사이즈 업" in without_exact_units and not has_size_value("GENERIC_UP"):
        return False
    if any(
        term in normalized
        for term in ("사이즈 다운", "사이즈다운", "사이즈를 줄", "작게 신")
    ) and not has_size_value("DOWN"):
        return False

    claimed_sizes = _SIZE_THREE_DIGIT_PATTERN.findall(normalized)
    if len(claimed_sizes) >= 2:
        supported_order = False
        for fact in cited_facts:
            for evidence_text in fact.evidence_texts:
                source_sizes = _SIZE_THREE_DIGIT_PATTERN.findall(
                    normalize_review_text(evidence_text)
                )
                cursor = 0
                for source_size in source_sizes:
                    if cursor < len(claimed_sizes) and source_size == claimed_sizes[cursor]:
                        cursor += 1
                if cursor == len(claimed_sizes):
                    supported_order = True
                    break
            if supported_order:
                break
        if not supported_order:
            return False
    return True


def _point_claims_are_traceable(
    summary: ShoeFitSummary,
    *,
    shoe_name: str,
    facts: list[PointEvidenceFact],
) -> bool:
    if not facts:
        # Preserve only the legacy non-assertive smoke-test placeholder. Any
        # product claim without typed evidence must be replaced by the trusted
        # deterministic fallback.
        neutral = normalize_review_text(summary.point_summary)
        return not summary.point_summary_claims and neutral in {
            "종합",
            "종합 결과입니다",
            "종합 적합도 결과입니다",
        }
    if not summary.point_summary_claims:
        return False
    expected_summary = " ".join(
        f"{claim.text.rstrip('.')}." for claim in summary.point_summary_claims
    )
    if normalize_review_text(expected_summary) != normalize_review_text(
        summary.point_summary
    ):
        return False

    facts_by_id = {fact.evidence_id: fact for fact in facts}
    for claim in summary.point_summary_claims:
        if not claim.evidence_ids or not set(claim.evidence_ids).issubset(facts_by_id):
            return False
        normalized_claim = normalize_review_text(claim.text)
        if any(
            term in normalized_claim
            for term in (
                *_POINT_UNIVERSAL_CLAIM_TERMS,
                *_POINT_UNTRACKED_ATTRIBUTE_TERMS,
                *_POINT_EXAGGERATION_TERMS,
            )
        ):
            return False
        cited_facts = [facts_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        if any(fact.mode == "CONFLICT" for fact in cited_facts):
            return False
        cited_text = " ".join(
            [shoe_name, *(text for fact in cited_facts for text in fact.evidence_texts)]
        )
        if _uses_unsupported_concept(
            claim.text,
            cited_text,
            additional_concepts=_POINT_GROUNDED_CONCEPTS,
        ):
            return False
        if _claim_uses_uncited_signal(claim.text, cited_facts):
            return False
        if not all(_fact_is_expressed(claim.text, fact) for fact in cited_facts):
            return False
        if not _single_facts_are_safely_scoped(claim.text, cited_facts):
            return False
        if not _mixed_facts_are_safely_scoped(claim.text, cited_facts):
            return False
        if not _size_claim_is_consistent(claim.text, cited_facts):
            return False
        numeric_claims = set(re.findall(r"\d+(?:\.\d+)?", claim.text))
        grounded_numbers = set(re.findall(r"\d+(?:\.\d+)?", cited_text))
        if numeric_claims - grounded_numbers:
            return False
    return True


def _contains_mechanical_area_result_summary(text: str) -> bool:
    """Reject a pointSummary that merely repeats the three fit-result labels.

    The individual reason cards already expose those results.  The product-level
    summary should explain concrete fit/product evidence instead of opening with
    a compressed ``FOREFOOT / HEEL / INSOLE`` status list.
    """

    normalized = normalize_review_text(text)
    if any(enum_name.lower() in normalized for enum_name in REASON_TYPES):
        return True

    result_terms = ("적합도", "주의", "좋음", "좋습니다", "보통", "위험도")
    area_result_sentence_count = 0
    area_subjects: set[str] = set()
    for sentence in re.split(r"[.!?]+", normalized):
        mentioned_areas = sum(
            any(term in sentence for term in terms)
            for terms in _AREA_CONCEPTS.values()
        )
        if mentioned_areas >= 2 and any(term in sentence for term in result_terms):
            return True
        if mentioned_areas and any(term in sentence for term in result_terms):
            area_result_sentence_count += 1
        for reason_type, terms in _AREA_CONCEPTS.items():
            if any(
                re.search(rf"{re.escape(term)}\s*(?:은|는|:)", sentence)
                for term in terms
            ):
                area_subjects.add(reason_type)
    if area_result_sentence_count >= 2 or len(area_subjects) >= 3:
        return True
    return any(pattern in normalized for pattern in _POINT_REPORT_STYLE_PATTERNS)


def _generated_reason_summaries_are_grounded(
    summary: ShoeFitSummary,
    *,
    reasons: list[ReasonFactsForPrompt],
) -> bool:
    for reason in reasons:
        generated = getattr(summary, _SUMMARY_FIELD_BY_REASON[reason.reason_type])
        source_text = " ".join([reason.title, *reason.review_texts])
        if _contains_out_of_scope_content(
            generated, _REASON_OUT_OF_SCOPE_SUMMARY_TERMS
        ):
            return False
        if not normalize_review_text(generated).startswith(
            normalize_review_text(reason.title)
        ):
            return False
        if _uses_unsupported_concept(generated, source_text):
            return False
    return True


def _generated_point_summary_is_grounded(
    summary: ShoeFitSummary,
    *,
    fit_score: float,
    shoe_name: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> bool:
    if _contains_mechanical_area_result_summary(summary.point_summary):
        return False
    if _contains_out_of_scope_content(
        summary.point_summary, _POINT_OUT_OF_SCOPE_SUMMARY_TERMS
    ):
        return False
    normalized_summary = normalize_review_text(summary.point_summary)
    if sum(
        normalized_summary.count(term) for term in ("리뷰", "후기", "의견")
    ) > 1:
        return False
    if not _point_claims_are_traceable(
        summary, shoe_name=shoe_name, facts=point_evidence
    ):
        return False
    point_source = " ".join(
        [
            shoe_name,
            str(fit_score),
            *(reason.title for reason in reasons),
            *(
                text
                for fact in point_evidence
                for text in fact.evidence_texts
            ),
        ]
    )
    if _uses_unsupported_concept(
        summary.point_summary,
        point_source,
        additional_concepts=_POINT_GROUNDED_CONCEPTS,
    ):
        return False
    numeric_claims = set(re.findall(r"\d+(?:\.\d+)?", summary.point_summary))
    grounded_numbers = set(re.findall(r"\d+(?:\.\d+)?", point_source))
    grounded_numbers.update(
        {
            str(fit_score),
            f"{fit_score:g}",
            f"{fit_score:.1f}",
            f"{fit_score:.2f}",
        }
    )
    if numeric_claims - grounded_numbers:
        return False
    return True


def _fallback_point_summary(
    *,
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
) -> tuple[str, list[ShoePointSummaryClaim]]:
    del fit_score, overall_risk_level, reasons
    claims = render_product_point_summary(shoe_name, point_evidence)
    return (
        join_point_summary_claims(claims),
        [
            ShoePointSummaryClaim(
                text=claim.text,
                evidence_ids=list(claim.evidence_ids),
            )
            for claim in claims
        ],
    )


def _deterministic_grounded_summary(
    *,
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    point_evidence: list[PointEvidenceFact],
    selected_summary: ShoeFitSummary,
) -> ShoeFitSummary:
    reason_summaries: dict[str, str] = {}
    for reason in reasons:
        review_count = len(
            getattr(selected_summary, _REVIEW_IDS_FIELD_BY_REASON[reason.reason_type])
        )
        evidence_note = (
            f"관련 착화 리뷰 {review_count}개를 함께 확인해 주세요."
            if review_count
            else "관련성이 확인된 리뷰가 없어 정량 분석 결과만 안내합니다."
        )
        reason_summaries[reason.reason_type] = (
            f"{reason.title}. 정량 분석 기준 {_REASON_LABEL[reason.reason_type]} "
            f"위험도는 {_RISK_LABEL[reason.risk_level]}입니다. {evidence_note}"
        )

    point_summary, point_summary_claims = _fallback_point_summary(
        shoe_name=shoe_name,
        fit_score=fit_score,
        overall_risk_level=overall_risk_level,
        reasons=reasons,
        point_evidence=point_evidence,
    )
    return ShoeFitSummary(
        point_summary=point_summary,
        point_summary_claims=point_summary_claims,
        forefoot_summary=reason_summaries["FOREFOOT"],
        heel_summary=reason_summaries["HEEL"],
        insole_summary=reason_summaries["INSOLE"],
        forefoot_review_ids=selected_summary.forefoot_review_ids,
        heel_review_ids=selected_summary.heel_review_ids,
        insole_review_ids=selected_summary.insole_review_ids,
    )


async def generate_shoe_summaries(
    shoe_name: str,
    fit_score: float,
    overall_risk_level: str,
    reasons: list[ReasonFactsForPrompt],
    characteristics: ServerShoeCharacteristics | None = None,
) -> ShoeFitSummary:
    """신발 하나의 fitScore/riskLevel/부위별 판단 결과를 바탕으로, Ollama로
    pointSummary + 부위별 reviewSummary(발볼/뒤꿈치/깔창)를 한 번의 호출로 생성한다."""
    if overall_risk_level not in _RISK_LABEL:
        raise ShoeFitCommentError(f"Unknown overall risk level: {overall_risk_level}")
    if not 0 <= fit_score <= 100:
        raise ShoeFitCommentError("fit_score must be between 0 and 100")
    display_shoe_name = _display_shoe_name(shoe_name)
    point_evidence = prepare_point_summary_evidence(reasons, characteristics)
    grounded_reasons = prepare_grounded_reasons(reasons)
    user_prompt = _build_user_prompt(
        display_shoe_name,
        fit_score,
        overall_risk_level,
        grounded_reasons,
        point_evidence,
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    try:
        await asyncio.wait_for(
            _ollama_semaphore.acquire(),
            timeout=settings.ollama_queue_timeout_seconds,
        )
    except TimeoutError as exc:
        raise ShoeFitCommentError(
            "Ollama summary runtime is busy; the exact-session request can be retried."
        ) from exc
    try:
        try:
            response = await _get_llm().ainvoke(messages)
        except (httpx.HTTPError, ResponseError, RuntimeError) as exc:
            if not settings.ollama_cpu_fallback_enabled or settings.ollama_num_gpu == 0:
                raise ShoeFitCommentError(f"Ollama 요청에 실패했습니다: {exc}") from exc
            logger.warning(
                "Ollama GPU 우선 요청 실패; 동일 prompt를 CPU로 한 번 fallback합니다.",
                exc_info=True,
            )
            try:
                response = await _get_llm(force_cpu=True).ainvoke(messages)
            except (httpx.HTTPError, ResponseError, RuntimeError) as cpu_exc:
                raise ShoeFitCommentError(
                    f"Ollama GPU/CPU 요청이 모두 실패했습니다: {cpu_exc}"
                ) from cpu_exc
    finally:
        _ollama_semaphore.release()

    parsed = _parse_llm_json(response.content)
    try:
        summary = ShoeFitSummary.model_validate(parsed)
    except Exception as exc:
        logger.warning(
            "Ollama response did not match the shoe-summary schema; using the "
            "typed-evidence fallback.",
            exc_info=True,
        )
        seed = ShoeFitSummary(
            point_summary="typed evidence fallback",
            forefoot_summary=next(
                reason.title for reason in grounded_reasons if reason.reason_type == "FOREFOOT"
            ),
            heel_summary=next(
                reason.title for reason in grounded_reasons if reason.reason_type == "HEEL"
            ),
            insole_summary=next(
                reason.title for reason in grounded_reasons if reason.reason_type == "INSOLE"
            ),
            forefoot_review_ids=next(
                reason.review_ids for reason in grounded_reasons if reason.reason_type == "FOREFOOT"
            ),
            heel_review_ids=next(
                reason.review_ids for reason in grounded_reasons if reason.reason_type == "HEEL"
            ),
            insole_review_ids=next(
                reason.review_ids for reason in grounded_reasons if reason.reason_type == "INSOLE"
            ),
        )
        fallback = _deterministic_grounded_summary(
            shoe_name=display_shoe_name,
            fit_score=fit_score,
            overall_risk_level=overall_risk_level,
            reasons=grounded_reasons,
            point_evidence=point_evidence,
            selected_summary=seed,
        )
        return _ensure_point_evidence_reviews_selected(
            fallback, grounded_reasons, point_evidence
        )
    _validate_review_selection(summary, grounded_reasons)
    summary = _ensure_review_selection_minimum(summary, grounded_reasons)
    reason_summaries_grounded = _generated_reason_summaries_are_grounded(
        summary, reasons=grounded_reasons
    )
    point_summary_grounded = _generated_point_summary_is_grounded(
        summary,
        shoe_name=display_shoe_name,
        fit_score=fit_score,
        reasons=grounded_reasons,
        point_evidence=point_evidence,
    )
    if not reason_summaries_grounded or not point_summary_grounded:
        logger.warning(
            "Ollama shoe summary failed deterministic grounding checks "
            "(reasonSummaries=%s, pointSummary=%s); replacing only unsafe fields.",
            reason_summaries_grounded,
            point_summary_grounded,
        )
        fallback = _deterministic_grounded_summary(
            shoe_name=display_shoe_name,
            fit_score=fit_score,
            overall_risk_level=overall_risk_level,
            reasons=grounded_reasons,
            point_evidence=point_evidence,
            selected_summary=summary,
        )
        updates: dict[str, object] = {}
        if not point_summary_grounded:
            updates["point_summary"] = fallback.point_summary
            updates["point_summary_claims"] = fallback.point_summary_claims
        if not reason_summaries_grounded:
            updates.update(
                {
                    "forefoot_summary": fallback.forefoot_summary,
                    "heel_summary": fallback.heel_summary,
                    "insole_summary": fallback.insole_summary,
                }
            )
        result = summary.model_copy(update=updates)
        return _ensure_point_evidence_reviews_selected(
            result, grounded_reasons, point_evidence
        )
    return _ensure_point_evidence_reviews_selected(
        summary, grounded_reasons, point_evidence
    )
