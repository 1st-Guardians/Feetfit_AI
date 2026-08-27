from __future__ import annotations

import math
from collections import Counter

from app.schemas.shoe_server import ServerRecommendationContext
from app.services.shoe.shoe_fit_policy import (
    CLINICAL_VALIDATION_STATUS,
    POLICY_CLASSIFICATION,
    POLICY_VERSION,
)
from app.services.shoe.shoe_recommendation import ShoeRecommendationBatch


def build_dry_run_report(
    context: ServerRecommendationContext,
    batch: ShoeRecommendationBatch,
) -> dict[str, object]:
    issues: list[str] = []
    input_ids = [shoe.id for shoe in context.shoes]
    output_ids = [item.shoe_id for item in batch.items]
    input_counts = Counter(input_ids)
    output_counts = Counter(output_ids)
    duplicate_input = sorted(key for key, count in input_counts.items() if count > 1)
    duplicate_output = sorted(key for key, count in output_counts.items() if count > 1)
    if duplicate_input:
        issues.append("DUPLICATE_INPUT_SHOE_ID")
    if duplicate_output:
        issues.append("DUPLICATE_OUTPUT_SHOE_ID")
    if input_ids != output_ids:
        issues.append("SHOE_COVERAGE_OR_ORDER_MISMATCH")
    if batch.measurement_session_id != context.measurement_session_id:
        issues.append("MEASUREMENT_SESSION_MISMATCH")
    if batch.user_id != context.user_id:
        issues.append("USER_SCOPE_MISMATCH")

    shoes_by_id = {shoe.id: shoe for shoe in context.shoes}
    score_distribution: Counter[str] = Counter()
    evidence_distribution: Counter[str] = Counter()
    for item in batch.items:
        if not math.isfinite(item.fit_score) or not 0 <= item.fit_score <= 100:
            issues.append(f"INVALID_FIT_SCORE:{item.shoe_id}")
        score_distribution[str(int(item.fit_score // 10) * 10)] += 1
        reason_types = [reason.reason_type for reason in item.reasons]
        if reason_types != ["FOREFOOT", "HEEL", "INSOLE"]:
            issues.append(f"INVALID_REASON_TYPES:{item.shoe_id}")
        shoe = shoes_by_id.get(item.shoe_id)
        owned_review_ids = {review.id for review in shoe.reviews} if shoe else set()
        for reason in item.reasons:
            review_ids = reason.review_ids
            evidence_distribution[str(len(review_ids))] += 1
            if len(review_ids) > 3 or len(review_ids) != len(set(review_ids)):
                issues.append(
                    f"INVALID_REVIEW_ID_CARDINALITY:{item.shoe_id}:{reason.reason_type}"
                )
            if not set(review_ids).issubset(owned_review_ids):
                issues.append(
                    f"REVIEW_OWNERSHIP_CONFLICT:{item.shoe_id}:{reason.reason_type}"
                )

    return {
        "format": "feetfit-phase-d-read-only-dry-run",
        "status": "PASS" if not issues else "FAIL",
        "measurementSessionId": context.measurement_session_id,
        "userId": context.user_id,
        "policy": {
            "version": POLICY_VERSION,
            "classification": POLICY_CLASSIFICATION,
            "clinicalValidation": CLINICAL_VALIDATION_STATUS,
        },
        "coverage": {
            "inputShoeCount": len(input_ids),
            "outputShoeCount": len(output_ids),
            "missingShoeIds": sorted(set(input_ids) - set(output_ids)),
            "unexpectedShoeIds": sorted(set(output_ids) - set(input_ids)),
            "duplicateInputShoeIds": duplicate_input,
            "duplicateOutputShoeIds": duplicate_output,
        },
        "fitScoreDecileDistribution": dict(sorted(score_distribution.items())),
        "reviewIdsPerReasonDistribution": dict(sorted(evidence_distribution.items())),
        "issues": issues,
        "safety": {
            "dryRun": True,
            "serverHttpMethods": ["GET"],
            "serverMutationRequested": False,
            "databaseAccess": False,
            "ollamaCalled": False,
            "shoeComparisonFeatureIncluded": False,
            "reviewTextIncludedInReport": False,
        },
    }
