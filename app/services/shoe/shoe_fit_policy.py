from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Iterable

from app.core.config import settings
from app.schemas.shoe_server import (
    ServerShoe,
    ServerShoeLabMeasurement,
    ServerShoeRawMetric,
    ServerUserFootState,
)
from app.services.shoe.shoe_feature_rules import default_title


POLICY_CLASSIFICATION = "TEMPORARY_HEURISTIC"
CLINICAL_VALIDATION_STATUS = "NOT_CLINICALLY_VALIDATED"
POLICY_VERSION = "phase-d-v1"
REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")


class ShoeFitPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentScore:
    name: str
    score: float
    weight: float


@dataclass(frozen=True)
class AreaScore:
    reason_type: str
    score: float
    risk_level: str
    title: str
    components: tuple[ComponentScore, ...]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _to_float(value) -> float | None:
    return None if value is None else float(value)


def risk_level_from_score(score: float) -> str:
    if score >= settings.shoe_risk_low_min_score:
        return "LOW"
    if score >= settings.shoe_risk_medium_min_score:
        return "MEDIUM"
    return "HIGH"


def _weighted_score(components: Iterable[ComponentScore]) -> float:
    present = tuple(components)
    total_weight = sum(component.weight for component in present)
    if total_weight <= 0:
        raise ShoeFitPolicyError(
            "No usable RunRepeat component exists for a required fit area; refusing to synthesize a score."
        )
    return sum(component.score * component.weight for component in present) / total_weight


def _area(reason_type: str, components: Iterable[ComponentScore]) -> AreaScore:
    present = tuple(components)
    score = round(_weighted_score(present), 2)
    risk = risk_level_from_score(score)
    return AreaScore(
        reason_type=reason_type,
        score=score,
        risk_level=risk,
        title=default_title(reason_type, risk),
        components=present,
    )


def _latest_runrepeat_measurement(shoe: ServerShoe) -> ServerShoeLabMeasurement:
    measurements = [
        measurement
        for measurement in shoe.lab_measurements
        if measurement.source.strip().upper() == "RUNREPEAT"
    ]
    if not measurements:
        raise ShoeFitPolicyError(
            f"shoeId={shoe.id} has no RunRepeat lab measurement; quantitative scoring is impossible."
        )
    return max(
        measurements,
        key=lambda measurement: (
            measurement.captured_at is not None,
            measurement.captured_at,
            measurement.measurement_id,
        ),
    )


def _metric_priority(metric: ServerShoeRawMetric) -> tuple[int, int, int]:
    version = (metric.method_version or "").strip().lower()
    variant = (metric.variant or "").strip().lower()
    return (
        0 if version == "old method" else (2 if version == "new method" else 1),
        1 if variant in {"primary", "width"} else 0,
        metric.metric_id,
    )


def _metric(
    measurement: ServerShoeLabMeasurement,
    characteristic: str,
    *,
    location: str | None = None,
    variant: str | None = None,
) -> ServerShoeRawMetric | None:
    candidates = [
        metric
        for metric in measurement.raw_metrics
        if metric.canonical_characteristic == characteristic and metric.value is not None
    ]
    if location is not None:
        normalized = location.casefold()
        candidates = [
            metric for metric in candidates if (metric.location or "").casefold() == normalized
        ]
    if variant is not None:
        normalized = variant.casefold()
        candidates = [
            metric for metric in candidates if (metric.variant or "").casefold() == normalized
        ]
    return max(candidates, key=_metric_priority) if candidates else None


def _percentile(metric: ServerShoeRawMetric, *, higher_is_better: bool = True) -> float:
    value = float(metric.value)
    minimum = _to_float(metric.source_min_value)
    maximum = _to_float(metric.source_max_value)
    average = _to_float(metric.average_value)
    if minimum is not None and maximum is not None and maximum > minimum:
        result = _clamp((value - minimum) / (maximum - minimum))
    elif average is not None:
        scale = max(abs(average) * 0.35, 1.0)
        result = _clamp(0.5 + 0.5 * tanh((value - average) / scale))
    else:
        result = 0.5
    return result if higher_is_better else 1.0 - result


def _target_score(percentile: float, target: float, tolerance: float | None = None) -> float:
    if tolerance is None:
        tolerance = settings.shoe_metric_target_tolerance
    return 100.0 * _clamp(1.0 - abs(percentile - target) / tolerance)


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _ratio_severity(value: float | None, low: float, high: float) -> float:
    if value is None or high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _deficit_severity(value: float | None, safe: float, severe: float) -> float:
    if value is None or safe <= severe:
        return 0.0
    return _clamp((safe - value) / (safe - severe))


def _user_factors(state: ServerUserFootState) -> dict[str, float | None]:
    daily = state.daily_foot_analysis
    tinea = state.tina_pedis_analysis
    hallux = state.hallux_valgus_analysis

    foot_length = _average(
        (
            daily.measured_left_foot_size_mm if daily else None,
            daily.measured_right_foot_size_mm if daily else None,
        )
    )
    foot_width = _average(
        (
            daily.left_foot_width_mm if daily else None,
            daily.right_foot_width_mm if daily else None,
        )
    )
    width_ratio = (
        foot_width / foot_length
        if foot_width is not None and foot_length not in {None, 0.0}
        else None
    )
    max_hallux = max(
        (
            value
            for value in (
                hallux.left_toe_angle_degree if hallux else None,
                hallux.right_toe_angle_degree if hallux else None,
            )
            if value is not None
        ),
        default=None,
    )
    width_need = max(
        _ratio_severity(
            width_ratio,
            settings.shoe_forefoot_width_ratio_neutral,
            settings.shoe_forefoot_width_ratio_high,
        ),
        _ratio_severity(
            max_hallux,
            settings.shoe_hallux_angle_neutral_degree,
            settings.shoe_hallux_angle_high_degree,
        ),
    )

    pressure_imbalance = None
    if daily and daily.left_pressure_percent is not None and daily.right_pressure_percent is not None:
        pressure_imbalance = abs(daily.left_pressure_percent - daily.right_pressure_percent)
    daily_balance = daily.balance_score if daily else None
    static_balance = _average(row.balance_score for row in state.static_pressure_analyses)
    balance = min(
        (value for value in (daily_balance, static_balance) if value is not None),
        default=None,
    )
    rearfoot_pressure = max(
        (
            row.rearfoot_pressure_ratio
            for row in state.static_pressure_analyses
            if row.rearfoot_pressure_ratio is not None
        ),
        default=None,
    )
    skin_sensitivity = _deficit_severity(
        tinea.skin_reaction_safety_score if tinea else None,
        settings.shoe_skin_safety_neutral_score,
        settings.shoe_skin_safety_high_risk_score,
    )
    heel_need = max(
        _ratio_severity(
            pressure_imbalance,
            settings.shoe_pressure_imbalance_neutral_percent,
            settings.shoe_pressure_imbalance_high_percent,
        ),
        _deficit_severity(
            balance,
            settings.shoe_balance_neutral_score,
            settings.shoe_balance_high_risk_score,
        ),
        _ratio_severity(
            rearfoot_pressure,
            settings.shoe_rearfoot_pressure_neutral_percent,
            settings.shoe_rearfoot_pressure_high_percent,
        ),
    )
    humidity_need = _ratio_severity(
        daily.avg_humidity_percent if daily else None,
        settings.shoe_humidity_neutral_percent,
        settings.shoe_humidity_high_percent,
    )
    fungal_need = _deficit_severity(
        tinea.fungal_suspicion_safety_score if tinea else None,
        settings.shoe_fungal_safety_neutral_score,
        settings.shoe_fungal_safety_high_risk_score,
    )
    return {
        "foot_length": foot_length,
        "foot_width": foot_width,
        "width_need": width_need,
        "heel_need": heel_need,
        "skin_sensitivity": skin_sensitivity,
        "breathability_need": max(humidity_need, fungal_need),
        # Pressure/balance belongs to HEEL. INSOLE personalization is driven
        # by humidity/fungal evidence; keeping these independent prevents one
        # measurement signal from silently changing multiple UI areas.
        "cushion_need": heel_need,
    }


def _forefoot_score(
    measurement: ServerShoeLabMeasurement, factors: dict[str, float | None]
) -> AreaScore:
    components: list[ComponentScore] = []
    width_need = float(factors["width_need"] or 0.0)
    foot_length = factors["foot_length"]
    foot_width = factors["foot_width"]
    if (
        measurement.width_mm is not None
        and measurement.internal_length_mm not in {None, 0.0}
        and foot_length is not None
        and foot_width is not None
    ):
        scaled_internal_width = measurement.width_mm * foot_length / measurement.internal_length_mm
        actual_allowance = scaled_internal_width - foot_width
        desired = (
            settings.shoe_forefoot_width_allowance_mm
            + width_need * settings.shoe_forefoot_width_extra_allowance_mm
        )
        shortfall = max(0.0, desired - actual_allowance)
        excess = max(0.0, actual_allowance - desired - settings.shoe_forefoot_width_excess_free_mm)
        score = 100.0 - (
            shortfall * settings.shoe_forefoot_width_shortfall_penalty_per_mm
            + excess * settings.shoe_forefoot_width_excess_penalty_per_mm
        )
        components.append(
            ComponentScore(
                "WIDTH_ALLOWANCE",
                _clamp(score, 0, 100),
                settings.shoe_forefoot_width_component_weight,
            )
        )
    else:
        width = _metric(measurement, "WIDTH_SPACE")
        if width is not None:
            target = 0.5 + 0.45 * width_need
            components.append(
                ComponentScore(
                    "WIDTH_SPACE",
                    _target_score(_percentile(width), target),
                    settings.shoe_forefoot_width_component_weight,
                )
            )

    toebox = _metric(measurement, "TOEBOX_SPACE", variant="width")
    if toebox is None:
        toebox = _metric(measurement, "TOEBOX_SPACE")
    if toebox is not None:
        target = 0.5 + 0.45 * width_need
        components.append(
            ComponentScore(
                "TOEBOX_SPACE",
                _target_score(_percentile(toebox), target),
                settings.shoe_forefoot_toebox_component_weight,
            )
        )
    return _area("FOREFOOT", components)


def _heel_score(
    measurement: ServerShoeLabMeasurement, factors: dict[str, float | None]
) -> AreaScore:
    components: list[ComponentScore] = []
    heel_need = float(factors["heel_need"] or 0.0)
    skin_sensitivity = float(factors["skin_sensitivity"] or 0.0)
    cushion_need = float(factors["cushion_need"] or 0.0)

    heel_hold = _metric(measurement, "HEEL_HOLD", location="HEEL") or _metric(
        measurement, "HEEL_HOLD"
    )
    if heel_hold is not None:
        target = _clamp(0.5 + 0.4 * heel_need - 0.2 * skin_sensitivity)
        components.append(
            ComponentScore(
                "HEEL_HOLD",
                _target_score(_percentile(heel_hold), target),
                settings.shoe_heel_hold_component_weight,
            )
        )
    shock = _metric(measurement, "SHOCK_ABSORPTION", location="HEEL") or _metric(
        measurement, "SHOCK_ABSORPTION"
    )
    if shock is not None:
        components.append(
            ComponentScore(
                "SHOCK_ABSORPTION",
                _target_score(_percentile(shock), 0.5 + 0.45 * cushion_need),
                settings.shoe_heel_shock_component_weight,
            )
        )
    energy = _metric(measurement, "ENERGY_RETURN", location="HEEL") or _metric(
        measurement, "ENERGY_RETURN"
    )
    if energy is not None:
        components.append(
            ComponentScore(
                "ENERGY_RETURN",
                _target_score(_percentile(energy), 0.5 + 0.3 * heel_need),
                settings.shoe_heel_energy_component_weight,
            )
        )
    cushion = _metric(measurement, "CUSHION", variant="primary") or _metric(
        measurement, "CUSHION"
    )
    if cushion is not None:
        components.append(
            ComponentScore(
                "CUSHION",
                _target_score(
                    _percentile(cushion, higher_is_better=False),
                    0.5 + 0.35 * cushion_need,
                ),
                settings.shoe_heel_cushion_component_weight,
            )
        )
    return _area("HEEL", components)


def _insole_score(
    measurement: ServerShoeLabMeasurement, factors: dict[str, float | None]
) -> AreaScore:
    components: list[ComponentScore] = []
    breathability_need = float(factors["breathability_need"] or 0.0)
    # Cushion and shock remain quantitative shoe characteristics, but there is
    # currently no validated session measurement that personalizes their
    # target. Keep the neutral target until such a signal is explicitly added.
    cushion_need = 0.0

    breathability = _metric(measurement, "BREATHABILITY")
    if breathability is not None:
        components.append(
            ComponentScore(
                "BREATHABILITY",
                _target_score(
                    _percentile(breathability), 0.5 + 0.45 * breathability_need
                ),
                settings.shoe_insole_breathability_component_weight,
            )
        )
    cushion = _metric(measurement, "CUSHION", variant="primary") or _metric(
        measurement, "CUSHION"
    )
    if cushion is not None:
        components.append(
            ComponentScore(
                "CUSHION",
                _target_score(
                    _percentile(cushion, higher_is_better=False),
                    0.5 + 0.4 * cushion_need,
                ),
                settings.shoe_insole_cushion_component_weight,
            )
        )
    shock = _metric(measurement, "SHOCK_ABSORPTION", location="FOREFOOT") or _metric(
        measurement, "SHOCK_ABSORPTION", location="HEEL"
    ) or _metric(measurement, "SHOCK_ABSORPTION")
    if shock is not None:
        components.append(
            ComponentScore(
                "SHOCK_ABSORPTION",
                _target_score(_percentile(shock), 0.5 + 0.45 * cushion_need),
                settings.shoe_insole_shock_component_weight,
            )
        )
    return _area("INSOLE", components)


def score_shoe_fit(shoe: ServerShoe, state: ServerUserFootState) -> tuple[AreaScore, ...]:
    measurement = _latest_runrepeat_measurement(shoe)
    factors = _user_factors(state)
    return (
        _forefoot_score(measurement, factors),
        _heel_score(measurement, factors),
        _insole_score(measurement, factors),
    )


def overall_fit_score(areas: Iterable[AreaScore]) -> float:
    weights = {
        "FOREFOOT": settings.shoe_forefoot_area_weight,
        "HEEL": settings.shoe_heel_area_weight,
        "INSOLE": settings.shoe_insole_area_weight,
    }
    present = tuple(areas)
    total = sum(weights[area.reason_type] for area in present)
    if total <= 0:
        raise ShoeFitPolicyError("Fit area weights must sum to a positive value.")
    return round(
        sum(area.score * weights[area.reason_type] for area in present) / total,
        2,
    )
