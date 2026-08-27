"""Read-only DB and API audit for the fixed 338-shoe ingestion cohort.

The audit has three fail-closed evidence boundaries:

* the crawler's existing authoritative loader is replayed in its own virtual
  environment;
* every dry-run member hash and the completed full execution state are checked
  against that replay;
* observations come from an injectable SELECT-only DB reader and GET-only API
  reader.

Reports contain no credentials, tokens, response bodies, review text, or raw
payloads.  The one allowed write is an atomic local report with a SHA-256
integrity signature.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

import httpx
import pymysql

from app.core.config import PROJECT_ROOT, settings


FORMAT = "feetfit-read-only-full-ingestion-audit"
VERSION = 1
EXPECTED_SHOE_COUNT = 338
EXPECTED_REVIEW_COUNT = 8517
EXPECTED_MEASUREMENT_COUNT = 338
EXPECTED_METRIC_COUNT = 3029
EXPECTED_USABLE_DISTRIBUTION = MappingProxyType({5: 27, 6: 25, 7: 286})
PINNED_DRY_RUN_MANIFEST_SHA256 = (
    "afd64dbea59828968e62c0a8e840994794f02e805a8d208f9e28439de14dc46e"
)
EXPECTED_BATCH_SIZE = 25

CANONICAL_CHARACTERISTICS = frozenset(
    {
        "CUSHION",
        "SHOCK_ABSORPTION",
        "ENERGY_RETURN",
        "WIDTH_SPACE",
        "TOEBOX_SPACE",
        "HEEL_HOLD",
        "BREATHABILITY",
    }
)

CRAWLER_ROOT = PROJECT_ROOT.parent / "shoe_crawler"
DEFAULT_DRY_RUN = CRAWLER_ROOT / "selection" / "20260826-03-ingestion-dry-run"
DEFAULT_EXECUTION_STATE = (
    CRAWLER_ROOT
    / "selection"
    / "20260826-03-full-ingestion-live"
    / "execution-state.json"
)
DEFAULT_LOADER_PYTHON = CRAWLER_ROOT / "venv" / "Scripts" / "python.exe"
LOADER_BRIDGE = PROJECT_ROOT / "scripts" / "replay_verified_ingestion_loader.py"

_MANIFEST_FILE = "manifest.json"
_DRY_RUN_FILES = frozenset(
    {
        "dry-run-report.json",
        "dry-run-decisions.json",
        "combined-items.json",
        "musinsa-server-payload.json",
        "runrepeat-targeted-server-payload.json",
    }
)
_TABLES = ("shoe", "shoe_review", "shoe_lab_measurement", "shoe_lab_metric")
_SUCCESS_OPERATIONS = frozenset({"CREATED", "UPDATED"})
_CHAR_REQUIRED_FIELDS = frozenset(
    {
        "type",
        "level",
        "value",
        "averageValue",
        "minValue",
        "maxValue",
        "unit",
        "testedSize",
    }
)


class FullAuditError(RuntimeError):
    """Authoritative evidence or a read-only observation is unsafe/invalid."""


class VerifiedLoader(Protocol):
    def __call__(self, dry_run_dir: Path) -> Mapping[str, Any]: ...


class DbAuditReader(Protocol):
    def read(self, expectation: "FullExpectation") -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class HttpGetObservation:
    status_code: int
    document: Mapping[str, Any] | None


class ApiAuditReader(Protocol):
    def detail(self, shoe_id: int) -> HttpGetObservation | Mapping[str, Any]: ...

    def characteristics(
        self, shoe_id: int
    ) -> HttpGetObservation | Mapping[str, Any]: ...


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _goods_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _chunks(values: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [
        tuple(values[index : index + size])
        for index in range(0, len(values), size)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FullAuditError("authoritative JSON file is missing or unsafe")
    try:
        value = json.loads(resolved.read_text("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullAuditError("authoritative JSON file is invalid") from exc
    if not isinstance(value, dict):
        raise FullAuditError("authoritative JSON root must be an object")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FullAuditError(f"{label} must be non-blank")
    return value.strip()


def _require_optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FullAuditError(f"{label} must be text or null")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FullAuditError(f"{label} must be an integer >= {minimum}")
    return value


def _as_decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except Exception:
        return None
    return result if result.is_finite() else None


def _require_decimal(value: object, label: str, *, nullable: bool = False) -> Decimal | None:
    result = _as_decimal(value)
    if result is None and not (nullable and value is None):
        raise FullAuditError(f"{label} must be a finite decimal")
    return result


def _time_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        result = value.isoformat()
    elif isinstance(value, str):
        result = value.strip().replace(" ", "T")
    else:
        return None
    if "." in result:
        head, tail = result.split(".", 1)
        suffix = ""
        for marker in ("+", "-"):
            if marker in tail:
                fraction, zone = tail.split(marker, 1)
                suffix = marker + zone
                tail = fraction
                break
        result = head + ("." + tail.rstrip("0") if tail.rstrip("0") else "") + suffix
    return result


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def replay_verified_loader(
    dry_run_dir: Path,
    *,
    crawler_root: Path = CRAWLER_ROOT,
    python_executable: Path = DEFAULT_LOADER_PYTHON,
) -> Mapping[str, Any]:
    """Run ``load_prepared_ingestion`` without importing crawler deps here."""

    root = crawler_root.expanduser().resolve()
    interpreter = python_executable.expanduser().resolve()
    bridge = LOADER_BRIDGE.resolve()
    if (
        root.is_symlink()
        or not root.is_dir()
        or interpreter.is_symlink()
        or not interpreter.is_file()
        or bridge.is_symlink()
        or not bridge.is_file()
    ):
        raise FullAuditError("authoritative loader runtime is missing or unsafe")
    command = [
        str(interpreter),
        str(bridge),
        "--crawler-root",
        str(root),
        "--dry-run-dir",
        str(dry_run_dir.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
    except Exception as exc:
        raise FullAuditError(
            f"authoritative loader replay failed ({type(exc).__name__})"
        ) from None
    if completed.returncode != 0:
        raise FullAuditError("authoritative loader replay was rejected; output suppressed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FullAuditError("authoritative loader replay output is invalid") from exc
    if not isinstance(value, dict):
        raise FullAuditError("authoritative loader replay root is invalid")
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class ExpectedReview:
    source_review_id: str
    rating: Decimal
    review_text: str
    collected_at: str

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.source_review_id,
            self.rating,
            self.review_text,
            self.collected_at,
        )


@dataclass(frozen=True, slots=True)
class ExpectedMetric:
    canonical_characteristic: str
    source_metric_name: str
    value: Decimal | None
    average_value: Decimal | None
    source_min_value: Decimal | None
    source_max_value: Decimal | None
    unit: str | None
    tested_size: str | None
    method_name: str | None
    method_version: str | None
    location: str | None
    variant: str | None
    comparison_sample_count: int | None
    comparison_cohort: str | None
    raw_value_text: str | None

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.canonical_characteristic,
            self.source_metric_name,
            self.value,
            self.average_value,
            self.source_min_value,
            self.source_max_value,
            self.unit,
            self.tested_size,
            self.method_name,
            self.method_version,
            self.location,
            self.variant,
            self.comparison_sample_count,
            self.comparison_cohort,
            self.raw_value_text,
        )


@dataclass(frozen=True, slots=True)
class ExpectedMeasurement:
    source_url: str
    source_brand_name: str
    source_shoe_name: str
    source_model_code: str | None
    captured_at: str
    parser_version: str
    tested_size: str | None
    internal_length_mm: Decimal | None
    width_mm: Decimal | None
    toebox_width_mm: Decimal | None
    toebox_height_mm: Decimal | None
    insole_thickness_mm: Decimal | None
    heel_stack_mm: Decimal | None
    forefoot_stack_mm: Decimal | None
    metrics: tuple[ExpectedMetric, ...]
    usable_characteristics: frozenset[str]
    missing_characteristics: frozenset[str]
    display_metrics: Mapping[str, ExpectedMetric]


@dataclass(frozen=True, slots=True)
class ExpectedShoe:
    goods_no: str
    shoe_id: int
    brand_name: str
    shoe_name: str
    model_code: str
    musinsa_url: str
    price: int
    image_url: str
    overall_rating: Decimal
    review_count: int
    reviews: tuple[ExpectedReview, ...]
    measurement: ExpectedMeasurement

    @property
    def public_fields(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "id": self.shoe_id,
                "brandName": self.brand_name,
                "shoeName": self.shoe_name,
                "modelCode": self.model_code,
                "musinsaUrl": self.musinsa_url,
                "price": self.price,
                "imageUrl": self.image_url,
                "overallRating": self.overall_rating,
                "reviewCount": self.review_count,
            }
        )


@dataclass(frozen=True, slots=True)
class FullExpectation:
    goods_nos: tuple[str, ...]
    canary_goods_nos: frozenset[str]
    shoes: Mapping[str, ExpectedShoe]
    dry_run_manifest_sha256: str
    dry_run_file_sha256: Mapping[str, str]
    execution_state_sha256: str
    selection_manifest_sha256: str
    final_audit_manifest_sha256: str
    runrepeat_audit_ids_by_goods_no: Mapping[str, int]
    execution_summary: Mapping[str, Any]

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "shoe": len(self.shoes),
                "shoe_review": sum(len(shoe.reviews) for shoe in self.shoes.values()),
                "shoe_lab_measurement": len(self.shoes),
                "shoe_lab_metric": sum(
                    len(shoe.measurement.metrics) for shoe in self.shoes.values()
                ),
            }
        )

    @property
    def shoe_ids(self) -> Mapping[str, int]:
        return MappingProxyType(
            {goods_no: self.shoes[goods_no].shoe_id for goods_no in self.goods_nos}
        )


def _metric_from_payload(row: Mapping[str, Any]) -> ExpectedMetric:
    characteristic = _require_text(
        row.get("canonicalCharacteristic"), "canonicalCharacteristic"
    )
    if characteristic not in CANONICAL_CHARACTERISTICS:
        raise FullAuditError("raw metric characteristic is not canonical")
    sample_count = row.get("comparisonSampleCount")
    if sample_count is not None:
        sample_count = _require_int(sample_count, "comparisonSampleCount")
    return ExpectedMetric(
        canonical_characteristic=characteristic,
        source_metric_name=_require_text(row.get("sourceMetricName"), "sourceMetricName"),
        value=_require_decimal(row.get("value"), "metric value", nullable=True),
        average_value=_require_decimal(
            row.get("averageValue"), "averageValue", nullable=True
        ),
        source_min_value=_require_decimal(
            row.get("sourceMinValue"), "sourceMinValue", nullable=True
        ),
        source_max_value=_require_decimal(
            row.get("sourceMaxValue"), "sourceMaxValue", nullable=True
        ),
        unit=_require_optional_text(row.get("unit"), "unit"),
        tested_size=_require_optional_text(row.get("testedSize"), "testedSize"),
        method_name=_require_optional_text(row.get("methodName"), "methodName"),
        method_version=_require_optional_text(
            row.get("methodVersion"), "methodVersion"
        ),
        location=_require_optional_text(row.get("location"), "location"),
        variant=_require_optional_text(row.get("variant"), "variant"),
        comparison_sample_count=sample_count,
        comparison_cohort=_require_optional_text(
            row.get("comparisonCohort"), "comparisonCohort"
        ),
        raw_value_text=_require_optional_text(row.get("rawValueText"), "rawValueText"),
    )


def _source_base(value: str) -> str:
    return re.sub(
        r"\s*\((?:new|old) method\)\s*$", "", " ".join(value.split()).casefold()
    )


def _same_normalized(left: str | None, right: str) -> bool:
    return left is not None and " ".join(left.split()).casefold() == right.casefold()


def _display_metrics(metrics: Sequence[ExpectedMetric]) -> Mapping[str, ExpectedMetric]:
    selected: dict[str, ExpectedMetric] = {}
    for characteristic in CANONICAL_CHARACTERISTICS:
        eligible: list[tuple[int, ExpectedMetric]] = []
        for metric in metrics:
            if metric.canonical_characteristic != characteristic or metric.value is None:
                continue
            base = _source_base(metric.source_metric_name)
            ok = False
            priority = 0
            if characteristic == "CUSHION":
                ok = (
                    base == "midsole softness"
                    and _same_normalized(metric.variant, "primary")
                    and _normalize_text(metric.location) is None
                    and metric.unit is not None
                    and metric.unit.casefold() in {"ac", "ha"}
                )
                priority = 0 if metric.unit and metric.unit.casefold() == "ac" else 1
            elif characteristic == "SHOCK_ABSORPTION":
                ok = (
                    base == "shock absorption heel"
                    and _same_normalized(metric.location, "HEEL")
                    and _same_normalized(metric.unit, "SA")
                )
            elif characteristic == "ENERGY_RETURN":
                ok = (
                    base == "energy return heel"
                    and _same_normalized(metric.location, "HEEL")
                    and _same_normalized(metric.unit, "%")
                )
            elif characteristic == "WIDTH_SPACE":
                ok = base == "width / fit" and _same_normalized(metric.unit, "mm")
            elif characteristic == "TOEBOX_SPACE":
                ok = (
                    base == "toebox width"
                    and _same_normalized(metric.variant, "width")
                    and _same_normalized(metric.unit, "mm")
                )
            elif characteristic == "HEEL_HOLD":
                ok = base == "heel counter stiffness" and _same_normalized(
                    metric.location, "HEEL"
                )
            elif characteristic == "BREATHABILITY":
                ok = base == "breathability" and metric.unit is not None and (
                    metric.unit.casefold() in {"br", "score"}
                )
                priority = 0 if metric.unit and metric.unit.casefold() == "br" else 1
            if ok:
                eligible.append((priority, metric))
        if not eligible:
            continue
        best = min(priority for priority, _ in eligible)
        winners = [metric for priority, metric in eligible if priority == best]
        if len(winners) == 1:
            selected[characteristic] = winners[0]
    return MappingProxyType(selected)


def _mapping_rows(document: Mapping[str, Any], key: str, label: str) -> list[Mapping[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise FullAuditError(f"{label} rows are invalid")
    return value


def _validate_execution_state(
    state: Mapping[str, Any],
    *,
    goods_nos: tuple[str, ...],
    canary_goods_nos: frozenset[str],
    loader_evidence: Mapping[str, Any],
) -> tuple[Mapping[str, int], Mapping[str, int], Mapping[str, Any]]:
    expected_provenance = {
        "selectionManifestSha256": loader_evidence["selectionManifestSha256"],
        "finalAuditManifestSha256": loader_evidence["finalAuditManifestSha256"],
        "dryRunManifestSha256": loader_evidence["dryRunManifestSha256"],
        "canaryExecutionStateSha256": loader_evidence[
            "canaryExecutionStateSha256"
        ],
    }
    if (
        state.get("format") != "feetfit-verified-ingestion-execution"
        or state.get("version") != 1
        or state.get("mode") != "EXECUTE"
        or state.get("scope") != "FULL"
        or state.get("status") != "COMPLETED"
        or state.get("provenance") != expected_provenance
        or state.get("selectedGoodsNos") != list(goods_nos)
        or state.get("batchSize") != EXPECTED_BATCH_SIZE
        or state.get("fullOperationPolicy")
        != loader_evidence.get("fullOperationPolicy")
        or state.get("inFlight") is not None
    ):
        raise FullAuditError("completed full execution state contract is invalid")

    goods_set = set(goods_nos)
    for key in ("musinsaSuccessfulGoodsNos", "runRepeatSuccessfulGoodsNos"):
        values = state.get(key)
        if (
            not isinstance(values, list)
            or len(values) != EXPECTED_SHOE_COUNT
            or len(set(map(str, values))) != EXPECTED_SHOE_COUNT
            or set(map(str, values)) != goods_set
        ):
            raise FullAuditError(f"execution state {key} is incomplete")
    shoe_ids_raw = state.get("shoeIdsByGoodsNo")
    if not isinstance(shoe_ids_raw, Mapping):
        raise FullAuditError("execution state shoeId mapping is missing")
    shoe_ids = {str(key): value for key, value in shoe_ids_raw.items()}
    if (
        set(shoe_ids) != goods_set
        or any(type(value) is not int or value <= 0 for value in shoe_ids.values())
        or len(set(shoe_ids.values())) != EXPECTED_SHOE_COUNT
    ):
        raise FullAuditError("execution state shoeId mapping is invalid")
    canary_ids_raw = loader_evidence.get("canaryShoeIdsByGoodsNo")
    if (
        not isinstance(canary_ids_raw, Mapping)
        or {str(key): value for key, value in canary_ids_raw.items()}
        != {goods_no: shoe_ids[goods_no] for goods_no in canary_goods_nos}
    ):
        raise FullAuditError("full execution changed a verified canary shoeId")

    safety = state.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("executeFlag") is not True
        or safety.get("serverCalled") is not True
        or safety.get("shoeComparisonImplemented") is not False
    ):
        raise FullAuditError("execution state safety evidence is invalid")
    batches = state.get("batches")
    if not isinstance(batches, list) or not all(isinstance(row, Mapping) for row in batches):
        raise FullAuditError("execution batch evidence is invalid")
    allowances_raw = state.get("uncertainRetryAllowances")
    if (
        not isinstance(allowances_raw, Mapping)
        or set(allowances_raw) != {"MUSINSA", "RUNREPEAT"}
        or any(
            not isinstance(allowances_raw[phase], list)
            or len(allowances_raw[phase]) != len(set(allowances_raw[phase]))
            or not set(map(str, allowances_raw[phase])) <= goods_set
            for phase in ("MUSINSA", "RUNREPEAT")
        )
    ):
        raise FullAuditError("execution uncertain-retry allowances are invalid")
    uncertain_allowances = {
        phase: set(map(str, allowances_raw[phase]))
        for phase in ("MUSINSA", "RUNREPEAT")
    }

    successful_events: list[Mapping[str, Any]] = []
    failed_events: list[Mapping[str, Any]] = []
    timeout_event_count = 0
    for event in batches:
        if event.get("idempotencyReplay") is not False:
            raise FullAuditError("full execution unexpectedly contains replay batches")
        errors = event.get("errors")
        failed_goods = event.get("failedGoodsNos")
        if not isinstance(errors, list) or not isinstance(failed_goods, list):
            raise FullAuditError("execution event failure evidence is invalid")
        successful = (
            event.get("serverResponse") is not None
            and event.get("failedCount") == 0
            and event.get("successCount") == event.get("requestedCount")
            and not errors
            and not failed_goods
        )
        if successful:
            successful_events.append(event)
        else:
            failed_events.append(event)
            if any("timeout" in str(error).casefold() for error in errors):
                timeout_event_count += 1

    expected_chunks = _chunks(goods_nos, EXPECTED_BATCH_SIZE)
    if len(expected_chunks) != 14 or len(successful_events) != len(expected_chunks) * 2:
        raise FullAuditError("successful execution batch coverage is incomplete")
    expected_sequence = [
        (phase, number, chunk)
        for number, chunk in enumerate(expected_chunks, start=1)
        for phase in ("MUSINSA", "RUNREPEAT")
    ]
    rr_audit_ids: dict[str, int] = {}
    operation_counts: dict[str, Counter[str]] = {
        "MUSINSA": Counter(),
        "RUNREPEAT": Counter(),
    }
    uncertain_updates: dict[str, set[str]] = {
        "MUSINSA": set(),
        "RUNREPEAT": set(),
    }
    batch_summaries: list[dict[str, Any]] = []
    for event, (phase, batch_number, chunk) in zip(
        successful_events, expected_sequence, strict=True
    ):
        event_goods_raw = event.get("goodsNos")
        if (
            event.get("phase") != phase
            or event.get("batchNumber") != batch_number
            or event_goods_raw != list(chunk)
            or event.get("requestedCount") != len(chunk)
            or event.get("successCount") != len(chunk)
            or event.get("failedCount") != 0
            or event.get("httpStatus") != 200
        ):
            raise FullAuditError(
                "full execution is not ordered MUSINSA-then-RunRepeat per batch"
            )
        response = event.get("serverResponse")
        if not isinstance(response, Mapping):
            raise FullAuditError("successful batch response is invalid")
        response_rows = response.get("items")
        if (
            response.get("requestedCount") != len(chunk)
            or response.get("processedCount") != len(chunk)
            or not isinstance(response_rows, list)
            or len(response_rows) != len(chunk)
            or not all(isinstance(row, Mapping) for row in response_rows)
        ):
            raise FullAuditError("successful Server response counts are invalid")
        by_goods: dict[str, Mapping[str, Any]] = {}
        batch_operations: Counter[str] = Counter()
        for row in response_rows:
            goods_no = str(row.get("externalKey") or "")
            if goods_no not in chunk or goods_no in by_goods:
                raise FullAuditError("successful Server response identity is invalid")
            operation = str(row.get("operation") or "")
            expected_operation = (
                "UPDATED" if goods_no in canary_goods_nos else "CREATED"
            )
            operation_allowed = operation == expected_operation or (
                expected_operation == "CREATED"
                and operation == "UPDATED"
                and goods_no in uncertain_allowances[phase]
            )
            if (
                row.get("matchStatus") != "MATCHED"
                or row.get("shoeId") != shoe_ids[goods_no]
                or operation not in _SUCCESS_OPERATIONS
                or not operation_allowed
            ):
                raise FullAuditError("successful Server response linkage/operation is invalid")
            candidate_ids = row.get("candidateShoeIds")
            if candidate_ids is not None and candidate_ids != [shoe_ids[goods_no]]:
                raise FullAuditError("successful response candidate linkage is invalid")
            audit_id = row.get("auditId")
            if type(audit_id) is not int or audit_id <= 0:
                raise FullAuditError("successful response auditId is invalid")
            if phase == "RUNREPEAT":
                if audit_id in rr_audit_ids.values():
                    raise FullAuditError("RunRepeat auditId is duplicated")
                rr_audit_ids[goods_no] = audit_id
            if (
                goods_no not in canary_goods_nos
                and operation == "UPDATED"
                and goods_no in uncertain_allowances[phase]
            ):
                uncertain_updates[phase].add(goods_no)
            operation_counts[phase][operation] += 1
            batch_operations[operation] += 1
            by_goods[goods_no] = row
        if set(by_goods) != set(chunk):
            raise FullAuditError("successful response omitted a batch goodsNo")
        batch_summaries.append(
            {
                "batchNumber": batch_number,
                "phase": phase,
                "requestedCount": len(chunk),
                "successCount": len(chunk),
                "failedCount": 0,
                "httpStatus": 200,
                "operations": dict(sorted(batch_operations.items())),
                "executionTimeSeconds": event.get("executionTimeSeconds"),
            }
        )

    if set(rr_audit_ids) != goods_set:
        raise FullAuditError("full execution created/updated operation totals are invalid")
    for phase in ("MUSINSA", "RUNREPEAT"):
        expected_updated = canary_goods_nos | uncertain_updates[phase]
        if operation_counts[phase]["UPDATED"] != len(expected_updated):
            raise FullAuditError("full execution update allowances are inconsistent")
        if operation_counts[phase]["CREATED"] != EXPECTED_SHOE_COUNT - len(
            expected_updated
        ):
            raise FullAuditError("full execution create count is inconsistent")
    summary = state.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("requestedCount") != EXPECTED_SHOE_COUNT
        or summary.get("musinsaSuccessCount") != EXPECTED_SHOE_COUNT
        or summary.get("runRepeatSuccessCount") != EXPECTED_SHOE_COUNT
        or summary.get("failedBatchCount") != len(failed_events)
    ):
        raise FullAuditError("execution summary is incomplete")
    resume_evidence = state.get("resumeWarnings")
    resume_event_count = len(resume_evidence) if isinstance(resume_evidence, list) else 0
    acknowledged: dict[str, set[str]] = {"MUSINSA": set(), "RUNREPEAT": set()}
    if resume_evidence is not None and not isinstance(resume_evidence, list):
        raise FullAuditError("execution resume evidence is invalid")
    for warning in resume_evidence or []:
        batch = warning.get("batch") if isinstance(warning, Mapping) else None
        if (
            not isinstance(warning, Mapping)
            or warning.get("warning")
            != "UNCERTAIN_IN_FLIGHT_BATCH_RETRY_ACKNOWLEDGED"
            or warning.get("databaseCommitOutcomeChecked") is not True
            or not isinstance(batch, Mapping)
            or batch.get("phase") not in acknowledged
            or not isinstance(batch.get("goodsNos"), list)
            or not set(map(str, batch["goodsNos"])) <= goods_set
        ):
            raise FullAuditError("timeout resume lacks DB commit verification evidence")
        acknowledged[str(batch["phase"])].update(map(str, batch["goodsNos"]))
    if acknowledged != uncertain_allowances:
        raise FullAuditError("timeout resume allowances disagree with acknowledgements")
    uncertain_failed = [
        event for event in failed_events if event.get("requestOutcomeUncertain") is True
    ]
    for event in uncertain_failed:
        phase = str(event.get("phase") or "")
        event_goods = event.get("goodsNos")
        if (
            phase not in acknowledged
            or not isinstance(event_goods, list)
            or not set(map(str, event_goods)) <= acknowledged[phase]
        ):
            raise FullAuditError("uncertain timeout was resumed without acknowledgement")
    execution_summary = MappingProxyType(
        {
            "status": "COMPLETED",
            "logicalBatchCount": len(expected_chunks),
            "successfulRequestCount": len(successful_events),
            "historicalFailedRequestCount": len(failed_events),
            "timeoutEventCount": timeout_event_count,
            "resumeEvidenceCount": resume_event_count,
            "resumeOccurred": bool(failed_events or resume_event_count),
            "uncertainRetryAllowances": {
                phase: sorted(values, key=_goods_key)
                for phase, values in uncertain_allowances.items()
            },
            "operations": {
                phase: dict(sorted(counts.items()))
                for phase, counts in operation_counts.items()
            },
            "batches": batch_summaries,
        }
    )
    return (
        MappingProxyType({key: int(value) for key, value in shoe_ids.items()}),
        MappingProxyType(rr_audit_ids),
        execution_summary,
    )


def load_full_expectation(
    *,
    dry_run_dir: str | Path = DEFAULT_DRY_RUN,
    execution_state_path: str | Path = DEFAULT_EXECUTION_STATE,
    verified_loader: VerifiedLoader | None = None,
) -> FullExpectation:
    """Build the exact 338-shoe expected state after replaying all provenance."""

    dry_run = Path(dry_run_dir).expanduser().resolve()
    state_path = Path(execution_state_path).expanduser().resolve()
    if dry_run.is_symlink() or not dry_run.is_dir():
        raise FullAuditError("authoritative dry-run directory is missing or unsafe")
    loader = verified_loader or replay_verified_loader
    loader_evidence = loader(dry_run)
    if (
        not isinstance(loader_evidence, Mapping)
        or loader_evidence.get("format")
        != "feetfit-authoritative-ingestion-loader-replay"
        or loader_evidence.get("version") != 1
        or Path(str(loader_evidence.get("dryRunRoot") or "")).resolve() != dry_run
        or not isinstance(loader_evidence.get("canaryExecutionStateSha256"), str)
        or not isinstance(loader_evidence.get("canaryShoeIdsByGoodsNo"), Mapping)
        or not isinstance(loader_evidence.get("fullOperationPolicy"), Mapping)
    ):
        raise FullAuditError("authoritative loader evidence is invalid")
    ready_raw = loader_evidence.get("readyGoodsNos")
    canary_raw = loader_evidence.get("canaryGoodsNos")
    if (
        not isinstance(ready_raw, list)
        or not isinstance(canary_raw, list)
        or len(ready_raw) != EXPECTED_SHOE_COUNT
        or len(set(map(str, ready_raw))) != EXPECTED_SHOE_COUNT
        or len(canary_raw) != 8
        or len(set(map(str, canary_raw))) != 8
    ):
        raise FullAuditError("authoritative loader cohort count/identity is invalid")
    goods_nos = tuple(str(value) for value in ready_raw)
    canary_goods_nos = frozenset(str(value) for value in canary_raw)
    if not canary_goods_nos <= set(goods_nos):
        raise FullAuditError("authoritative canary escaped the full cohort")

    manifest_path = dry_run / _MANIFEST_FILE
    manifest = _load_object(manifest_path)
    manifest_sha = _sha256(manifest_path)
    if (
        manifest_sha != PINNED_DRY_RUN_MANIFEST_SHA256
        or loader_evidence.get("dryRunManifestSha256") != manifest_sha
        or manifest.get("format") != "feetfit-server-ingestion-dry-run-bundle"
        or manifest.get("version") != 1
        or set(path.name for path in dry_run.iterdir())
        != {_MANIFEST_FILE, *_DRY_RUN_FILES}
    ):
        raise FullAuditError("pinned dry-run manifest/filesystem identity changed")
    summary = manifest.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("serverReadyCount") != EXPECTED_SHOE_COUNT
        or summary.get("musinsaPayloadShoeCount") != EXPECTED_SHOE_COUNT
        or summary.get("runRepeatTargetedPayloadItemCount") != EXPECTED_SHOE_COUNT
        or summary.get("combinedItemCount") != EXPECTED_SHOE_COUNT
    ):
        raise FullAuditError("dry-run manifest ready counts changed")
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("sourcePreviewManifestSha256")
        != loader_evidence.get("selectionManifestSha256")
        or provenance.get("finalAuditManifestSha256")
        != loader_evidence.get("finalAuditManifestSha256")
        or provenance.get("requiredAllowlistLoader")
        != "load_verified_ingestion_eligibility"
    ):
        raise FullAuditError("dry-run/loader provenance changed")

    records = manifest.get("files")
    if not isinstance(records, Mapping) or set(records) != _DRY_RUN_FILES:
        raise FullAuditError("dry-run manifest file records are incomplete")
    documents: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    for filename in sorted(_DRY_RUN_FILES):
        record = records.get(filename)
        path = dry_run / filename
        if (
            not isinstance(record, Mapping)
            or path.is_symlink()
            or not path.is_file()
            or record.get("path") != filename
            or record.get("sha256") != _sha256(path)
            or record.get("byteSize") != path.stat().st_size
        ):
            raise FullAuditError("dry-run member hash/size/path is invalid")
        file_hashes[filename] = str(record["sha256"])
        documents[filename] = _load_object(path)

    combined_rows = _mapping_rows(
        documents["combined-items.json"], "items", "combined dry-run"
    )
    musinsa_rows = _mapping_rows(
        documents["musinsa-server-payload.json"], "shoes", "MUSINSA payload"
    )
    rr_rows = _mapping_rows(
        documents["runrepeat-targeted-server-payload.json"],
        "items",
        "RunRepeat payload",
    )
    if any(len(rows) != EXPECTED_SHOE_COUNT for rows in (combined_rows, musinsa_rows, rr_rows)):
        raise FullAuditError("dry-run payload cohort count changed")
    combined_goods = [str(row.get("goodsNo") or "") for row in combined_rows]
    musinsa_goods = [str(row.get("goodsNo") or "") for row in musinsa_rows]
    rr_goods = [str(row.get("targetGoodsNo") or "") for row in rr_rows]
    if not combined_goods == musinsa_goods == rr_goods == list(goods_nos):
        raise FullAuditError("authoritative payload cohort/order mismatch")

    state = _load_object(state_path)
    shoe_ids, rr_audit_ids, execution_summary = _validate_execution_state(
        state,
        goods_nos=goods_nos,
        canary_goods_nos=canary_goods_nos,
        loader_evidence=loader_evidence,
    )

    global_review_ids: set[str] = set()
    shoes: dict[str, ExpectedShoe] = {}
    usable_distribution: Counter[int] = Counter()
    for combined, musinsa, rr in zip(combined_rows, musinsa_rows, rr_rows, strict=True):
        goods_no = str(combined["goodsNo"])
        if (
            str(rr.get("externalKey") or "") != goods_no
            or str(rr.get("targetGoodsNo") or "") != goods_no
        ):
            raise FullAuditError("RunRepeat externalKey/targetGoodsNo mismatch")
        product_fields = (
            "brandName",
            "shoeName",
            "modelCode",
            "musinsaUrl",
            "price",
            "imageUrl",
            "overallRating",
            "reviewCount",
        )
        if any(combined.get(key) != musinsa.get(key) for key in product_fields):
            raise FullAuditError("combined/MUSINSA product payload mismatch")
        combined_reviews = _mapping_rows(combined, "reviews", "combined reviews")
        musinsa_reviews = _mapping_rows(musinsa, "reviews", "MUSINSA reviews")
        if len(combined_reviews) != len(musinsa_reviews):
            raise FullAuditError("combined/MUSINSA review count mismatch")
        reviews: list[ExpectedReview] = []
        for combined_review, musinsa_review in zip(
            combined_reviews, musinsa_reviews, strict=True
        ):
            review_id = _require_text(combined_review.get("reviewId"), "reviewId")
            if (
                review_id != str(musinsa_review.get("sourceReviewId") or "")
                or combined_review.get("rating") != musinsa_review.get("rating")
                or combined_review.get("reviewText") != musinsa_review.get("reviewText")
                or combined_review.get("collectedAt") != musinsa_review.get("collectedAt")
                or combined_review.get("source") != "MUSINSA"
            ):
                raise FullAuditError("combined/MUSINSA review payload mismatch")
            if review_id in global_review_ids:
                raise FullAuditError("authoritative dry-run has duplicate review identity")
            global_review_ids.add(review_id)
            reviews.append(
                ExpectedReview(
                    source_review_id=review_id,
                    rating=_require_decimal(combined_review.get("rating"), "review rating"),
                    review_text=_require_text(
                        combined_review.get("reviewText"), "reviewText"
                    ),
                    collected_at=_require_text(
                        combined_review.get("collectedAt"), "review collectedAt"
                    ),
                )
            )

        combined_rr = combined.get("runRepeat")
        if not isinstance(combined_rr, Mapping):
            raise FullAuditError("combined RunRepeat evidence is invalid")
        combined_metrics_raw = _mapping_rows(
            combined_rr, "rawMetrics", "combined raw metrics"
        )
        rr_metrics_raw = _mapping_rows(rr, "rawMetrics", "RunRepeat raw metrics")
        if combined_metrics_raw != rr_metrics_raw:
            raise FullAuditError("combined/RunRepeat raw metrics mismatch")
        if (
            combined_rr.get("sourceUrl") != rr.get("sourceUrl")
            or combined_rr.get("testedSize") != rr.get("testedSize")
        ):
            raise FullAuditError("combined/RunRepeat source evidence mismatch")
        metrics = tuple(_metric_from_payload(metric) for metric in rr_metrics_raw)
        usable_raw = combined_rr.get("usableCharacteristics")
        missing_raw = combined_rr.get("missingCharacteristics")
        if not isinstance(usable_raw, list) or not isinstance(missing_raw, list):
            raise FullAuditError("characteristic partition is missing")
        usable = frozenset(str(value) for value in usable_raw)
        missing = frozenset(str(value) for value in missing_raw)
        raw_types = frozenset(metric.canonical_characteristic for metric in metrics)
        display = _display_metrics(metrics)
        if (
            usable | missing != CANONICAL_CHARACTERISTICS
            or usable & missing
            or usable != raw_types
            or usable != frozenset(display)
            or len(usable) not in {5, 6, 7}
            or combined_rr.get("usableCharacteristicCount") != len(usable)
        ):
            raise FullAuditError("authoritative characteristic partition is inconsistent")
        usable_distribution[len(usable)] += 1

        measurement = ExpectedMeasurement(
            source_url=_require_text(rr.get("sourceUrl"), "RunRepeat sourceUrl"),
            source_brand_name=_require_text(rr.get("brandName"), "RunRepeat brandName"),
            source_shoe_name=_require_text(rr.get("shoeName"), "RunRepeat shoeName"),
            source_model_code=_require_optional_text(
                rr.get("modelCode"), "RunRepeat modelCode"
            ),
            captured_at=_require_text(rr.get("capturedAt"), "RunRepeat capturedAt"),
            parser_version=_require_text(rr.get("parserVersion"), "parserVersion"),
            tested_size=_require_optional_text(rr.get("testedSize"), "testedSize"),
            internal_length_mm=_require_decimal(
                rr.get("internalLengthMm"), "internalLengthMm", nullable=True
            ),
            width_mm=_require_decimal(rr.get("widthMm"), "widthMm", nullable=True),
            toebox_width_mm=_require_decimal(
                rr.get("toeboxWidthMm"), "toeboxWidthMm", nullable=True
            ),
            toebox_height_mm=_require_decimal(
                rr.get("toeboxHeightMm"), "toeboxHeightMm", nullable=True
            ),
            insole_thickness_mm=_require_decimal(
                rr.get("insoleThicknessMm"), "insoleThicknessMm", nullable=True
            ),
            heel_stack_mm=_require_decimal(
                rr.get("heelStackMm"), "heelStackMm", nullable=True
            ),
            forefoot_stack_mm=_require_decimal(
                rr.get("forefootStackMm"), "forefootStackMm", nullable=True
            ),
            metrics=metrics,
            usable_characteristics=usable,
            missing_characteristics=missing,
            display_metrics=display,
        )
        price = _require_int(musinsa.get("price"), "price")
        overall_rating = _require_decimal(
            musinsa.get("overallRating"), "overallRating"
        )
        assert overall_rating is not None
        shoes[goods_no] = ExpectedShoe(
            goods_no=goods_no,
            shoe_id=shoe_ids[goods_no],
            brand_name=_require_text(musinsa.get("brandName"), "brandName"),
            shoe_name=_require_text(musinsa.get("shoeName"), "shoeName"),
            model_code=_require_text(musinsa.get("modelCode"), "modelCode"),
            musinsa_url=_require_text(musinsa.get("musinsaUrl"), "musinsaUrl"),
            price=price,
            image_url=_require_text(musinsa.get("imageUrl"), "imageUrl"),
            overall_rating=overall_rating,
            review_count=_require_int(musinsa.get("reviewCount"), "reviewCount"),
            reviews=tuple(reviews),
            measurement=measurement,
        )

    counts = {
        "shoe": len(shoes),
        "shoe_review": sum(len(shoe.reviews) for shoe in shoes.values()),
        "shoe_lab_measurement": len(shoes),
        "shoe_lab_metric": sum(
            len(shoe.measurement.metrics) for shoe in shoes.values()
        ),
    }
    if counts != {
        "shoe": EXPECTED_SHOE_COUNT,
        "shoe_review": EXPECTED_REVIEW_COUNT,
        "shoe_lab_measurement": EXPECTED_MEASUREMENT_COUNT,
        "shoe_lab_metric": EXPECTED_METRIC_COUNT,
    } or dict(usable_distribution) != dict(EXPECTED_USABLE_DISTRIBUTION):
        raise FullAuditError("pinned authoritative dry-run totals changed")

    return FullExpectation(
        goods_nos=goods_nos,
        canary_goods_nos=canary_goods_nos,
        shoes=MappingProxyType(shoes),
        dry_run_manifest_sha256=manifest_sha,
        dry_run_file_sha256=MappingProxyType(file_hashes),
        execution_state_sha256=_sha256(state_path),
        selection_manifest_sha256=str(loader_evidence["selectionManifestSha256"]),
        final_audit_manifest_sha256=str(loader_evidence["finalAuditManifestSha256"]),
        runrepeat_audit_ids_by_goods_no=rr_audit_ids,
        execution_summary=execution_summary,
    )


@dataclass(frozen=True, slots=True)
class _MysqlConfig:
    host: str
    port: int
    database: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    charset: str = "utf8mb4"


def _mysql_config_from_settings() -> _MysqlConfig:
    jdbc_url = settings.shoe_db_url
    username = settings.shoe_db_username
    password = settings.shoe_db_password
    if (
        not jdbc_url
        or not username
        or not password
        or not jdbc_url.startswith("jdbc:mysql://")
    ):
        raise FullAuditError("SHOE_DB_* configuration is incomplete")
    parsed = urlparse(jdbc_url.replace("jdbc:mysql://", "mysql://", 1))
    if (
        not parsed.hostname
        or not parsed.path
        or parsed.path == "/"
        or parsed.username
        or parsed.password
    ):
        raise FullAuditError("SHOE_DB_URL host/database structure is invalid")
    query = parse_qs(parsed.query)
    source_charset = query.get("characterEncoding", ["UTF-8"])[0]
    charset = (
        "utf8mb4"
        if source_charset.upper().replace("-", "") == "UTF8"
        else source_charset
    )
    return _MysqlConfig(
        host=parsed.hostname,
        port=parsed.port or 3306,
        database=parsed.path.lstrip("/"),
        username=username,
        password=password,
        charset=charset,
    )


class MySqlReadOnlyAuditReader:
    """One consistent MySQL snapshot, using only READ ONLY and SELECT."""

    def __init__(self, connection_factory: Any | None = None) -> None:
        self._connection_factory = connection_factory

    def _connect(self):
        config = _mysql_config_from_settings()
        factory = self._connection_factory or pymysql.connect
        try:
            return factory(
                host=config.host,
                port=config.port,
                user=config.username,
                password=config.password,
                database=config.database,
                charset=config.charset,
                autocommit=False,
                connect_timeout=15,
                read_timeout=120,
                write_timeout=30,
                cursorclass=pymysql.cursors.DictCursor,
            )
        except Exception as exc:
            raise FullAuditError(
                f"DB connection failed ({type(exc).__name__}); credentials suppressed"
            ) from None

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("START TRANSACTION READ ONLY")
                yield cursor
        finally:
            connection.rollback()
            connection.close()

    @staticmethod
    def _fetch(cursor: Any, sql: str, params: object = None) -> list[dict[str, Any]]:
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def read(self, expectation: FullExpectation) -> Mapping[str, Any]:
        goods_nos = expectation.goods_nos
        if len(goods_nos) != EXPECTED_SHOE_COUNT or len(set(goods_nos)) != len(
            goods_nos
        ):
            raise FullAuditError("DB reader requires exactly 338 unique goodsNo")
        placeholders = ", ".join(["%s"] * len(goods_nos))
        params = tuple(goods_nos)
        with self._cursor() as cursor:
            counts: dict[str, int] = {}
            for table in _TABLES:
                rows = self._fetch(
                    cursor,
                    f"/* full-audit:count:{table} */ SELECT COUNT(*) AS record_count FROM {table}",
                )
                counts[table] = int(rows[0]["record_count"])
            shoes = self._fetch(
                cursor,
                f"""/* full-audit:shoes */
                SELECT id AS shoe_id, musinsa_goods_no AS goods_no, brand_name,
                       shoe_name, model_code, musinsa_url, price, image_url,
                       overall_rating, review_count, click_count
                FROM shoe WHERE musinsa_goods_no IN ({placeholders})
                ORDER BY musinsa_goods_no, id""",
                params,
            )
            reviews = self._fetch(
                cursor,
                f"""/* full-audit:reviews */
                SELECT r.id AS review_id, r.shoe_id, r.rating, r.review_text,
                       r.source, r.source_review_id, r.content_hash, r.collected_at,
                       s.musinsa_goods_no AS goods_no
                FROM shoe_review r JOIN shoe s ON s.id = r.shoe_id
                WHERE s.musinsa_goods_no IN ({placeholders})
                ORDER BY s.musinsa_goods_no, r.id""",
                params,
            )
            measurements = self._fetch(
                cursor,
                f"""/* full-audit:measurements */
                SELECT m.shoe_lab_measurement_id AS measurement_id, m.shoe_id,
                       m.source, m.source_url, m.snapshot_key, m.tested_size,
                       m.source_brand_name, m.source_shoe_name, m.source_model_code,
                       m.captured_at, m.parser_version, m.internal_length_mm,
                       m.width_mm, m.toebox_width_mm, m.toebox_height_mm,
                       m.insole_thickness_mm, m.heel_stack_mm, m.forefoot_stack_mm,
                       s.musinsa_goods_no AS goods_no
                FROM shoe_lab_measurement m JOIN shoe s ON s.id = m.shoe_id
                WHERE s.musinsa_goods_no IN ({placeholders})
                ORDER BY s.musinsa_goods_no, m.shoe_lab_measurement_id""",
                params,
            )
            metrics = self._fetch(
                cursor,
                f"""/* full-audit:metrics */
                SELECT lm.shoe_lab_metric_id AS metric_id,
                       lm.shoe_lab_measurement_id AS measurement_id,
                       m.shoe_id, s.musinsa_goods_no AS goods_no,
                       lm.canonical_characteristic, lm.source_metric_name,
                       lm.metric_value, lm.average_value, lm.source_min_value,
                       lm.source_max_value, lm.unit, lm.tested_size,
                       lm.method_name, lm.method_version, lm.location, lm.variant,
                       lm.comparison_sample_count, lm.comparison_cohort,
                       lm.raw_value_text
                FROM shoe_lab_metric lm
                JOIN shoe_lab_measurement m
                  ON m.shoe_lab_measurement_id = lm.shoe_lab_measurement_id
                JOIN shoe s ON s.id = m.shoe_id
                WHERE s.musinsa_goods_no IN ({placeholders})
                ORDER BY s.musinsa_goods_no, lm.shoe_lab_metric_id""",
                params,
            )
            import_audits = self._fetch(
                cursor,
                f"""/* full-audit:runrepeat-import-audits */
                SELECT shoe_import_audit_id AS audit_id, source, external_key,
                       source_url, match_status, matched_shoe_id, raw_payload
                FROM shoe_import_audit
                WHERE source = %s AND match_status = %s
                  AND external_key IN ({placeholders})
                ORDER BY shoe_import_audit_id""",
                ("RUNREPEAT", "MATCHED", *params),
            )
            outside_cohort = self._fetch(
                cursor,
                f"""/* full-audit:outside-cohort */
                SELECT id AS shoe_id, musinsa_goods_no AS goods_no
                FROM shoe
                WHERE musinsa_goods_no IS NULL
                   OR musinsa_goods_no NOT IN ({placeholders})
                ORDER BY id""",
                params,
            )
            duplicate_goods = self._fetch(
                cursor,
                """/* full-audit:duplicate-goods */
                SELECT musinsa_goods_no AS goods_no, COUNT(*) AS record_count
                FROM shoe WHERE musinsa_goods_no IS NOT NULL
                GROUP BY musinsa_goods_no HAVING COUNT(*) > 1""",
            )
            duplicate_reviews = self._fetch(
                cursor,
                """/* full-audit:duplicate-review-identity */
                SELECT shoe_id, source, source_review_id, COUNT(*) AS record_count
                FROM shoe_review WHERE source_review_id IS NOT NULL
                GROUP BY shoe_id, source, source_review_id HAVING COUNT(*) > 1""",
            )
            cross_shoe_review_ids = self._fetch(
                cursor,
                """/* full-audit:cross-shoe-review-identity */
                SELECT source, source_review_id,
                       COUNT(DISTINCT shoe_id) AS record_count
                FROM shoe_review WHERE source_review_id IS NOT NULL
                GROUP BY source, source_review_id
                HAVING COUNT(DISTINCT shoe_id) > 1""",
            )
            duplicate_snapshots = self._fetch(
                cursor,
                """/* full-audit:duplicate-snapshot */
                SELECT snapshot_key, COUNT(*) AS record_count
                FROM shoe_lab_measurement WHERE snapshot_key IS NOT NULL
                GROUP BY snapshot_key HAVING COUNT(*) > 1""",
            )
            orphan_counts: dict[str, int] = {}
            orphan_sql = {
                "shoeReviewWithoutShoe": """SELECT COUNT(*) AS record_count FROM shoe_review r LEFT JOIN shoe s ON s.id=r.shoe_id WHERE s.id IS NULL""",
                "labMeasurementWithoutShoe": """SELECT COUNT(*) AS record_count FROM shoe_lab_measurement m LEFT JOIN shoe s ON s.id=m.shoe_id WHERE s.id IS NULL""",
                "labMetricWithoutMeasurement": """SELECT COUNT(*) AS record_count FROM shoe_lab_metric lm LEFT JOIN shoe_lab_measurement m ON m.shoe_lab_measurement_id=lm.shoe_lab_measurement_id WHERE m.shoe_lab_measurement_id IS NULL""",
            }
            for name, sql in orphan_sql.items():
                rows = self._fetch(cursor, f"/* full-audit:orphan:{name} */ {sql}")
                orphan_counts[name] = int(rows[0]["record_count"])
        return MappingProxyType(
            {
                "counts": counts,
                "shoes": shoes,
                "reviews": reviews,
                "measurements": measurements,
                "metrics": metrics,
                "importAudits": import_audits,
                "outsideCohort": outside_cohort,
                "duplicates": {
                    "goodsNo": duplicate_goods,
                    "reviewIdentity": duplicate_reviews,
                    "reviewIdAcrossShoes": cross_shoe_review_ids,
                    "snapshotKey": duplicate_snapshots,
                },
                "orphans": orphan_counts,
            }
        )


def _row_goods(row: Mapping[str, Any]) -> str:
    return str(row.get("goods_no") or "")


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(expected, Decimal):
        return _as_decimal(actual) == expected
    return actual == expected


def _same_float(actual: object, expected: Decimal | None) -> bool:
    if expected is None:
        return actual is None
    actual_decimal = _as_decimal(actual)
    if actual_decimal is None:
        return False
    return math.isclose(
        float(actual_decimal), float(expected), rel_tol=1e-6, abs_tol=1e-5
    )


def _actual_metric_signature(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        str(row.get("canonical_characteristic") or ""),
        str(row.get("source_metric_name") or ""),
        _as_decimal(row.get("metric_value")),
        _as_decimal(row.get("average_value")),
        _as_decimal(row.get("source_min_value")),
        _as_decimal(row.get("source_max_value")),
        row.get("unit"),
        row.get("tested_size"),
        row.get("method_name"),
        row.get("method_version"),
        row.get("location"),
        row.get("variant"),
        row.get("comparison_sample_count"),
        row.get("comparison_cohort"),
        row.get("raw_value_text"),
    )


def _decode_raw_payload(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value, parse_float=Decimal)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def audit_database(
    observation: Mapping[str, Any], expectation: FullExpectation
) -> Mapping[str, Any]:
    """Compare one consistent DB snapshot with every persisted dry-run row."""

    issues: list[str] = []
    counts_raw = observation.get("counts")
    if not isinstance(counts_raw, Mapping):
        raise FullAuditError("DB observation counts are missing")
    counts: dict[str, int] = {}
    for table in _TABLES:
        value = counts_raw.get(table)
        if type(value) is not int or value < 0:
            raise FullAuditError(f"DB count is invalid for {table}")
        counts[table] = value
        if value != expectation.counts[table]:
            issues.append(f"FINAL_COUNT_MISMATCH:{table}")

    collections: dict[str, list[Mapping[str, Any]]] = {}
    for key in ("shoes", "reviews", "measurements", "metrics"):
        value = observation.get(key)
        if not isinstance(value, list) or not all(
            isinstance(row, Mapping) for row in value
        ):
            raise FullAuditError(f"DB observation {key} is invalid")
        collections[key] = value
    import_audits = observation.get("importAudits")
    outside_cohort = observation.get("outsideCohort")
    if (
        not isinstance(import_audits, list)
        or not all(isinstance(row, Mapping) for row in import_audits)
        or not isinstance(outside_cohort, list)
        or not all(isinstance(row, Mapping) for row in outside_cohort)
    ):
        raise FullAuditError("DB audit/cohort evidence is invalid")
    if outside_cohort:
        issues.append("OUTSIDE_AUTHORITATIVE_COHORT")

    duplicates = observation.get("duplicates")
    orphans = observation.get("orphans")
    if not isinstance(duplicates, Mapping) or not isinstance(orphans, Mapping):
        raise FullAuditError("DB duplicate/orphan evidence is invalid")
    duplicate_counts: dict[str, int] = {}
    for key in ("goodsNo", "reviewIdentity", "reviewIdAcrossShoes", "snapshotKey"):
        rows = duplicates.get(key)
        if not isinstance(rows, list):
            raise FullAuditError(f"DB duplicate evidence is invalid for {key}")
        duplicate_counts[key] = len(rows)
        if rows:
            issues.append(f"DUPLICATE_FOUND:{key}")
    orphan_counts: dict[str, int] = {}
    for key in (
        "shoeReviewWithoutShoe",
        "labMeasurementWithoutShoe",
        "labMetricWithoutMeasurement",
    ):
        value = orphans.get(key)
        if type(value) is not int or value < 0:
            raise FullAuditError(f"DB orphan count is invalid for {key}")
        orphan_counts[key] = value
        if value:
            issues.append(f"ORPHAN_FOUND:{key}")

    goods_set = set(expectation.goods_nos)
    for key, rows in collections.items():
        if any(_row_goods(row) not in goods_set for row in rows):
            issues.append(f"COHORT_ESCAPE:{key}")
    by_key = {
        key: {
            goods_no: [row for row in rows if _row_goods(row) == goods_no]
            for goods_no in expectation.goods_nos
        }
        for key, rows in collections.items()
    }
    audit_rows_by_goods: dict[str, list[Mapping[str, Any]]] = {
        goods_no: [] for goods_no in expectation.goods_nos
    }
    for row in import_audits:
        goods_no = str(row.get("external_key") or "")
        if goods_no in audit_rows_by_goods:
            audit_rows_by_goods[goods_no].append(row)
        else:
            issues.append("COHORT_ESCAPE:importAudits")

    per_goods: dict[str, Any] = {}
    identity_material: list[str] = []
    product_difference_count = 0
    review_difference_count = 0
    measurement_difference_count = 0
    metric_difference_count = 0
    below_five_count = 0
    wrong_target_count = 0
    observed_state_audit_ids: set[int] = set()
    for goods_no in expectation.goods_nos:
        expected = expectation.shoes[goods_no]
        shoe_rows = by_key["shoes"][goods_no]
        review_rows = by_key["reviews"][goods_no]
        measurement_rows = by_key["measurements"][goods_no]
        metric_rows = by_key["metrics"][goods_no]
        row_issues: list[str] = []
        expected_counts = {
            "shoe": 1,
            "shoe_review": len(expected.reviews),
            "shoe_lab_measurement": 1,
            "shoe_lab_metric": len(expected.measurement.metrics),
        }
        actual_counts = {
            "shoe": len(shoe_rows),
            "shoe_review": len(review_rows),
            "shoe_lab_measurement": len(measurement_rows),
            "shoe_lab_metric": len(metric_rows),
        }
        if actual_counts != expected_counts:
            row_issues.append("ROW_COUNT_MISMATCH")

        if len(shoe_rows) == 1:
            shoe = shoe_rows[0]
            if shoe.get("shoe_id") != expected.shoe_id:
                row_issues.append("WRONG_SHOE_ID_TARGET")
                wrong_target_count += 1
            db_fields = {
                "brand_name": expected.brand_name,
                "shoe_name": expected.shoe_name,
                "model_code": expected.model_code,
                "musinsa_url": expected.musinsa_url,
                "price": expected.price,
                "image_url": expected.image_url,
                "overall_rating": expected.overall_rating,
                "review_count": expected.review_count,
            }
            mismatches = [
                field_name
                for field_name, expected_value in db_fields.items()
                if not _same_value(shoe.get(field_name), expected_value)
            ]
            if mismatches:
                product_difference_count += 1
                row_issues.extend(f"PRODUCT_FIELD_MISMATCH:{name}" for name in mismatches)
            identity_material.append(f"shoe:{goods_no}:{shoe.get('shoe_id')}")

        expected_review_signatures = Counter(
            review.signature for review in expected.reviews
        )
        actual_review_signatures: Counter[tuple[object, ...]] = Counter()
        actual_review_ids: set[str] = set()
        for review in review_rows:
            if review.get("shoe_id") != expected.shoe_id:
                row_issues.append("REVIEW_WRONG_SHOE_ID")
                wrong_target_count += 1
            source_id = str(review.get("source_review_id") or "")
            if review.get("source") != "MUSINSA" or not source_id:
                row_issues.append("REVIEW_IDENTITY_INVALID")
            if review.get("content_hash") is not None:
                row_issues.append("REVIEW_CONTENT_HASH_UNEXPECTED")
            actual_review_ids.add(source_id)
            actual_review_signatures[
                (
                    source_id,
                    _as_decimal(review.get("rating")),
                    review.get("review_text"),
                    _time_text(review.get("collected_at")),
                )
            ] += 1
            identity_material.append(
                f"review:{goods_no}:{review.get('review_id')}:{source_id}"
            )
        expected_review_ids = {
            review.source_review_id for review in expected.reviews
        }
        if actual_review_ids != expected_review_ids:
            row_issues.append("REVIEW_ID_SET_MISMATCH")
        if actual_review_signatures != expected_review_signatures:
            review_difference_count += 1
            row_issues.append("REVIEW_PAYLOAD_MISMATCH")

        measurement_id: int | None = None
        if len(measurement_rows) == 1:
            measurement = measurement_rows[0]
            measurement_id_raw = measurement.get("measurement_id")
            if type(measurement_id_raw) is int and measurement_id_raw > 0:
                measurement_id = measurement_id_raw
            else:
                row_issues.append("MEASUREMENT_ID_INVALID")
            if measurement.get("shoe_id") != expected.shoe_id:
                row_issues.append("MEASUREMENT_WRONG_SHOE_ID")
                wrong_target_count += 1
            expected_snapshot_key = hashlib.sha256(
                (
                    f"{expected.shoe_id}|RUNREPEAT|"
                    f"{expected.measurement.source_url}|{expected.measurement.captured_at}"
                ).encode("utf-8")
            ).hexdigest()
            measurement_fields = {
                "source": "RUNREPEAT",
                "source_url": expected.measurement.source_url,
                "snapshot_key": expected_snapshot_key,
                "tested_size": expected.measurement.tested_size,
                "source_brand_name": expected.measurement.source_brand_name,
                "source_shoe_name": expected.measurement.source_shoe_name,
                "source_model_code": expected.measurement.source_model_code,
                "parser_version": expected.measurement.parser_version,
            }
            mismatched = [
                name
                for name, expected_value in measurement_fields.items()
                if measurement.get(name) != expected_value
            ]
            if _time_text(measurement.get("captured_at")) != _time_text(
                expected.measurement.captured_at
            ):
                mismatched.append("captured_at")
            float_fields = {
                "internal_length_mm": expected.measurement.internal_length_mm,
                "width_mm": expected.measurement.width_mm,
                "toebox_width_mm": expected.measurement.toebox_width_mm,
                "toebox_height_mm": expected.measurement.toebox_height_mm,
                "insole_thickness_mm": expected.measurement.insole_thickness_mm,
                "heel_stack_mm": expected.measurement.heel_stack_mm,
                "forefoot_stack_mm": expected.measurement.forefoot_stack_mm,
            }
            mismatched.extend(
                name
                for name, expected_value in float_fields.items()
                if not _same_float(measurement.get(name), expected_value)
            )
            if mismatched:
                measurement_difference_count += 1
                row_issues.extend(
                    f"MEASUREMENT_FIELD_MISMATCH:{name}" for name in mismatched
                )
            identity_material.append(
                f"measurement:{goods_no}:{measurement.get('measurement_id')}"
            )

        actual_metric_signatures: Counter[tuple[object, ...]] = Counter()
        metric_types: set[str] = set()
        for metric in metric_rows:
            if metric.get("shoe_id") != expected.shoe_id:
                row_issues.append("METRIC_WRONG_SHOE_ID")
                wrong_target_count += 1
            if measurement_id is None or metric.get("measurement_id") != measurement_id:
                row_issues.append("METRIC_WRONG_MEASUREMENT_ID")
                wrong_target_count += 1
            characteristic = str(metric.get("canonical_characteristic") or "")
            if characteristic in CANONICAL_CHARACTERISTICS:
                metric_types.add(characteristic)
            else:
                row_issues.append("METRIC_CHARACTERISTIC_INVALID")
            actual_metric_signatures[_actual_metric_signature(metric)] += 1
            identity_material.append(
                f"metric:{goods_no}:{metric.get('metric_id')}:{metric.get('measurement_id')}"
            )
        expected_metric_signatures = Counter(
            metric.signature for metric in expected.measurement.metrics
        )
        if actual_metric_signatures != expected_metric_signatures:
            metric_difference_count += 1
            row_issues.append("RAW_METRIC_PAYLOAD_MISMATCH")
        if frozenset(metric_types) != expected.measurement.usable_characteristics:
            row_issues.append("USABLE_CHARACTERISTICS_MISMATCH")
        missing_types = CANONICAL_CHARACTERISTICS - metric_types
        if frozenset(missing_types) != expected.measurement.missing_characteristics:
            row_issues.append("MISSING_CHARACTERISTICS_MISMATCH")
        if len(metric_types) < 5:
            below_five_count += 1
            row_issues.append("BELOW_FIVE_CHARACTERISTICS")

        expected_state_audit_id = expectation.runrepeat_audit_ids_by_goods_no[goods_no]
        matching_state_audit = False
        for audit_row in audit_rows_by_goods[goods_no]:
            audit_id = audit_row.get("audit_id")
            raw_payload = _decode_raw_payload(audit_row.get("raw_payload"))
            audit_wrong = (
                audit_row.get("source") != "RUNREPEAT"
                or audit_row.get("match_status") != "MATCHED"
                or audit_row.get("matched_shoe_id") != expected.shoe_id
                or audit_row.get("source_url") != expected.measurement.source_url
                or raw_payload is None
                or str(raw_payload.get("targetGoodsNo") or "") != goods_no
                or str(raw_payload.get("externalKey") or "") != goods_no
                or raw_payload.get("sourceUrl") != expected.measurement.source_url
            )
            if audit_wrong:
                wrong_target_count += 1
                row_issues.append("RUNREPEAT_AUDIT_TARGET_MISMATCH")
            if audit_id == expected_state_audit_id:
                matching_state_audit = not audit_wrong
                if type(audit_id) is int:
                    observed_state_audit_ids.add(audit_id)
        if not matching_state_audit:
            wrong_target_count += 1
            row_issues.append("EXECUTION_RUNREPEAT_AUDIT_MISSING_OR_WRONG")

        unique_row_issues = sorted(set(row_issues))
        if unique_row_issues:
            issues.extend(f"{issue}:{goods_no}" for issue in unique_row_issues)
        per_goods[goods_no] = {
            "shoeId": expected.shoe_id,
            "expectedCounts": expected_counts,
            "actualCounts": actual_counts,
            "reviewIdentityCount": len(actual_review_ids),
            "usableCharacteristics": sorted(metric_types),
            "missingCharacteristics": sorted(missing_types),
            "usableCharacteristicCount": len(metric_types),
            "issues": unique_row_issues,
            "status": "PASS" if not unique_row_issues else "FAIL",
        }

    if observed_state_audit_ids != set(
        expectation.runrepeat_audit_ids_by_goods_no.values()
    ):
        issues.append("EXECUTION_RUNREPEAT_AUDIT_COVERAGE_MISMATCH")
    fingerprint = hashlib.sha256(
        "\n".join(sorted(identity_material)).encode("utf-8")
    ).hexdigest()
    return MappingProxyType(
        {
            "status": "PASS" if not issues else "FAIL",
            "counts": counts,
            "expectedCounts": dict(expectation.counts),
            "outsideCohortCount": len(outside_cohort),
            "duplicateCounts": duplicate_counts,
            "orphanCounts": orphan_counts,
            "wrongRunRepeatTargetCount": wrong_target_count,
            "characteristicBelowFiveCount": below_five_count,
            "dryRunDifferenceCounts": {
                "product": product_difference_count,
                "reviewPayload": review_difference_count,
                "labMeasurement": measurement_difference_count,
                "rawMetric": metric_difference_count,
            },
            "perGoods": per_goods,
            "identityFingerprint": fingerprint,
            "issues": sorted(set(issues)),
        }
    )


class HttpGetAuditReader:
    """Loopback-only adapter exposing exactly the two permitted GET routes."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_connections: int = 8,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise FullAuditError("live API audit is restricted to loopback HTTP")
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or type(max_connections) is not int
            or not 1 <= max_connections <= 32
        ):
            raise FullAuditError("API timeout/concurrency configuration is invalid")
        self._base_url = base_url.rstrip("/")
        self._owned = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            headers={"Accept": "application/json", "User-Agent": "FeetFit-Full-Audit/1"},
        )

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def __enter__(self) -> "HttpGetAuditReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str) -> HttpGetObservation:
        if not re.fullmatch(r"/api/shoes/[1-9][0-9]*(?:/characteristics)?", path):
            raise FullAuditError("API audit path escaped the two approved GET routes")
        try:
            response = self._client.get(f"{self._base_url}{path}")
        except httpx.HTTPError as exc:
            raise FullAuditError(
                f"GET audit transport failed ({type(exc).__name__}); details suppressed"
            ) from None
        document: Mapping[str, Any] | None = None
        try:
            decoded = response.json(parse_float=Decimal)
            if isinstance(decoded, Mapping):
                document = decoded
        except ValueError:
            pass
        return HttpGetObservation(response.status_code, document)

    def detail(self, shoe_id: int) -> HttpGetObservation:
        if type(shoe_id) is not int or shoe_id <= 0:
            raise FullAuditError("shoeId must be a positive integer")
        return self._get(f"/api/shoes/{shoe_id}")

    def characteristics(self, shoe_id: int) -> HttpGetObservation:
        if type(shoe_id) is not int or shoe_id <= 0:
            raise FullAuditError("shoeId must be a positive integer")
        return self._get(f"/api/shoes/{shoe_id}/characteristics")


def _observation(value: HttpGetObservation | Mapping[str, Any]) -> HttpGetObservation:
    if isinstance(value, HttpGetObservation):
        return value
    if isinstance(value, Mapping):
        return HttpGetObservation(200, value)
    raise FullAuditError("API reader returned an invalid observation")


def _api_result(document: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if (
        not isinstance(document, Mapping)
        or document.get("isSuccess") is not True
        or not isinstance(document.get("code"), str)
        or not isinstance(document.get("message"), str)
        or not isinstance(document.get("result"), Mapping)
    ):
        return None
    return document["result"]


def _audit_one_api_shoe(
    reader: ApiAuditReader, expected: ExpectedShoe
) -> Mapping[str, Any]:
    issues: list[str] = []
    detail_status: int | None = None
    characteristic_status: int | None = None
    detail_result: Mapping[str, Any] | None = None
    characteristic_result: Mapping[str, Any] | None = None
    try:
        detail_observation = _observation(reader.detail(expected.shoe_id))
        detail_status = detail_observation.status_code
        if detail_status != 200:
            issues.append("DETAIL_HTTP_NOT_200")
        else:
            detail_result = _api_result(detail_observation.document)
            if detail_result is None:
                issues.append("DETAIL_API_RESPONSE_INVALID")
    except Exception as exc:
        issues.append(f"DETAIL_GET_FAILED:{type(exc).__name__}")
    try:
        characteristic_observation = _observation(
            reader.characteristics(expected.shoe_id)
        )
        characteristic_status = characteristic_observation.status_code
        if characteristic_status != 200:
            issues.append("CHARACTERISTICS_HTTP_NOT_200")
        else:
            characteristic_result = _api_result(characteristic_observation.document)
            if characteristic_result is None:
                issues.append("CHARACTERISTICS_API_RESPONSE_INVALID")
    except Exception as exc:
        issues.append(f"CHARACTERISTICS_GET_FAILED:{type(exc).__name__}")

    required_detail_fields = {
        "id",
        "brandName",
        "shoeName",
        "modelCode",
        "musinsaUrl",
        "price",
        "imageUrl",
        "overallRating",
        "clickCount",
        "reviewCount",
        "fitScore",
        "pointSummary",
        "reasons",
    }
    if detail_result is not None:
        if not required_detail_fields <= set(detail_result):
            issues.append("DETAIL_FIELDS_MISSING")
        for name, expected_value in expected.public_fields.items():
            if not _same_value(detail_result.get(name), expected_value):
                issues.append(f"DETAIL_FIELD_MISMATCH:{name}")
        if (
            not isinstance(detail_result.get("brandName"), str)
            or not detail_result["brandName"].strip()
            or not isinstance(detail_result.get("shoeName"), str)
            or not detail_result["shoeName"].strip()
        ):
            issues.append("DETAIL_LOCALIZED_NAME_INVALID")
        if (
            type(detail_result.get("price")) is not int
            or detail_result["price"] < 0
            or not isinstance(detail_result.get("imageUrl"), str)
            or not detail_result["imageUrl"].startswith(("http://", "https://"))
            or _as_decimal(detail_result.get("overallRating")) is None
        ):
            issues.append("DETAIL_PRODUCT_VALUE_INVALID")

    returned_types: list[str] = []
    summary_nonblank = False
    if characteristic_result is not None:
        if characteristic_result.get("shoeId") != expected.shoe_id:
            issues.append("CHARACTERISTIC_SHOE_ID_MISMATCH")
        rows = characteristic_result.get("characteristics")
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            issues.append("CHARACTERISTIC_ROWS_INVALID")
            rows = []
        for row in rows:
            if not _CHAR_REQUIRED_FIELDS <= set(row):
                issues.append("CHARACTERISTIC_FIELDS_MISSING")
            characteristic = str(row.get("type") or "")
            returned_types.append(characteristic)
            expected_metric = expected.measurement.display_metrics.get(characteristic)
            if expected_metric is None:
                issues.append("SYNTHESIZED_OR_UNKNOWN_CHARACTERISTIC")
                continue
            if row.get("level") not in {None, "LOW", "MEDIUM", "HIGH"}:
                issues.append("CHARACTERISTIC_LEVEL_INVALID")
            expected_values = {
                "value": expected_metric.value,
                "averageValue": expected_metric.average_value,
                "unit": expected_metric.unit,
                "testedSize": expected_metric.tested_size
                or expected.measurement.tested_size,
            }
            for name, expected_value in expected_values.items():
                if isinstance(expected_value, Decimal):
                    equal = _as_decimal(row.get(name)) == expected_value
                else:
                    equal = row.get(name) == expected_value
                if not equal:
                    issues.append(f"CHARACTERISTIC_FIELD_MISMATCH:{characteristic}:{name}")
            if expected_metric.source_min_value is not None and (
                _as_decimal(row.get("minValue")) != expected_metric.source_min_value
            ):
                issues.append(
                    f"CHARACTERISTIC_FIELD_MISMATCH:{characteristic}:minValue"
                )
            if expected_metric.source_max_value is not None and (
                _as_decimal(row.get("maxValue")) != expected_metric.source_max_value
            ):
                issues.append(
                    f"CHARACTERISTIC_FIELD_MISMATCH:{characteristic}:maxValue"
                )
        if len(returned_types) != len(set(returned_types)):
            issues.append("CHARACTERISTIC_TYPE_DUPLICATE")
        returned_set = frozenset(returned_types)
        if returned_set != expected.measurement.usable_characteristics:
            issues.append("CHARACTERISTIC_EXACT_SET_MISMATCH")
        if not 5 <= len(rows) <= 7:
            issues.append("CHARACTERISTIC_COUNT_OUT_OF_RANGE")
        summary = characteristic_result.get("summary")
        summary_nonblank = isinstance(summary, str) and bool(summary.strip())
        if not summary_nonblank:
            issues.append("CHARACTERISTIC_SUMMARY_MISSING")

    unique_issues = sorted(set(issues))
    return MappingProxyType(
        {
            "shoeId": expected.shoe_id,
            "detailHttpStatus": detail_status,
            "characteristicsHttpStatus": characteristic_status,
            "usableCharacteristics": sorted(set(returned_types)),
            "usableCharacteristicCount": len(set(returned_types)),
            "summaryNonblank": summary_nonblank,
            "issues": unique_issues,
            "status": "PASS" if not unique_issues else "FAIL",
        }
    )


def audit_api(
    reader: ApiAuditReader,
    expectation: FullExpectation,
    *,
    max_concurrency: int = 8,
) -> Mapping[str, Any]:
    """Run exactly 338 detail + 338 characteristics GETs with bounded workers."""

    if (
        type(max_concurrency) is not int
        or not 1 <= max_concurrency <= 32
        or len(expectation.goods_nos) != EXPECTED_SHOE_COUNT
    ):
        raise FullAuditError("API audit requires 338 shoes and concurrency 1..32")
    completed: dict[str, Mapping[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=max_concurrency, thread_name_prefix="shoe-full-audit"
    ) as executor:
        futures = {
            executor.submit(
                _audit_one_api_shoe, reader, expectation.shoes[goods_no]
            ): goods_no
            for goods_no in expectation.goods_nos
        }
        for future in as_completed(futures):
            goods_no = futures[future]
            try:
                completed[goods_no] = future.result()
            except Exception as exc:
                completed[goods_no] = {
                    "shoeId": expectation.shoes[goods_no].shoe_id,
                    "detailHttpStatus": None,
                    "characteristicsHttpStatus": None,
                    "usableCharacteristics": [],
                    "usableCharacteristicCount": 0,
                    "summaryNonblank": False,
                    "issues": [f"UNEXPECTED_AUDIT_FAILURE:{type(exc).__name__}"],
                    "status": "FAIL",
                }
    per_goods = {
        goods_no: dict(completed[goods_no]) for goods_no in expectation.goods_nos
    }
    issues = [
        f"{issue}:{goods_no}"
        for goods_no in expectation.goods_nos
        for issue in per_goods[goods_no]["issues"]
    ]
    detail_200 = sum(
        row["detailHttpStatus"] == 200 for row in per_goods.values()
    )
    characteristics_200 = sum(
        row["characteristicsHttpStatus"] == 200 for row in per_goods.values()
    )
    return MappingProxyType(
        {
            "status": "PASS" if not issues else "FAIL",
            "auditedShoeCount": len(per_goods),
            "requestCounts": {
                "detail": EXPECTED_SHOE_COUNT,
                "characteristics": EXPECTED_SHOE_COUNT,
                "total": EXPECTED_SHOE_COUNT * 2,
            },
            "http200Counts": {
                "detail": detail_200,
                "characteristics": characteristics_200,
                "total": detail_200 + characteristics_200,
            },
            "summaryNonblankCount": sum(
                row["summaryNonblank"] for row in per_goods.values()
            ),
            "characteristicCountDistribution": dict(
                sorted(
                    Counter(
                        row["usableCharacteristicCount"]
                        for row in per_goods.values()
                    ).items()
                )
            ),
            "synthesizedOrUnknownCharacteristicCount": sum(
                any(
                    issue == "SYNTHESIZED_OR_UNKNOWN_CHARACTERISTIC"
                    for issue in row["issues"]
                )
                for row in per_goods.values()
            ),
            "perGoods": per_goods,
            "issues": sorted(set(issues)),
            "maxConcurrency": max_concurrency,
        }
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_hash(document: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    return hashlib.sha256(_json_bytes(unsigned)).hexdigest()


def _sign(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result["integrity"] = {
        "algorithm": "SHA-256",
        "payloadSha256": _payload_hash(result),
    }
    return result


def create_audit(
    *,
    expectation: FullExpectation,
    db_reader: DbAuditReader,
    api_reader: ApiAuditReader,
    api_max_concurrency: int = 8,
) -> Mapping[str, Any]:
    db = audit_database(db_reader.read(expectation), expectation)
    api = audit_api(
        api_reader, expectation, max_concurrency=api_max_concurrency
    )
    issues = [
        *(f"DB:{issue}" for issue in db["issues"]),
        *(f"API:{issue}" for issue in api["issues"]),
    ]
    document = {
        "format": FORMAT,
        "version": VERSION,
        "createdAt": _now(),
        "status": "PASS" if not issues else "FAIL",
        "provenance": {
            "dryRunManifestSha256": expectation.dry_run_manifest_sha256,
            "dryRunFileSha256": dict(expectation.dry_run_file_sha256),
            "executionStateSha256": expectation.execution_state_sha256,
            "selectionManifestSha256": expectation.selection_manifest_sha256,
            "finalAuditManifestSha256": expectation.final_audit_manifest_sha256,
            "authoritativeLoaderReplayed": True,
            "authoritativeCohortCount": EXPECTED_SHOE_COUNT,
            "cohortSha256": hashlib.sha256(
                "\n".join(expectation.goods_nos).encode("utf-8")
            ).hexdigest(),
        },
        "execution": dict(expectation.execution_summary),
        "expected": {
            "counts": dict(expectation.counts),
            "usableCharacteristicCountDistribution": dict(
                EXPECTED_USABLE_DISTRIBUTION
            ),
            "canaryUpdateCount": len(expectation.canary_goods_nos),
            "newCreateCount": len(expectation.goods_nos)
            - len(expectation.canary_goods_nos),
        },
        "database": dict(db),
        "api": dict(api),
        "issues": issues,
        "safety": {
            "databaseTransactionReadOnly": True,
            "databaseStatements": ["START TRANSACTION READ ONLY", "SELECT"],
            "httpMethods": ["GET"],
            "httpRoutes": [
                "/api/shoes/{shoeId}",
                "/api/shoes/{shoeId}/characteristics",
            ],
            "serverMutationRequested": False,
            "databaseMutationRequested": False,
            "secretValuesIncluded": False,
            "responseBodiesIncluded": False,
            "reviewTextIncluded": False,
            "shoeComparisonFeatureIncluded": False,
            "comparison": False,
        },
    }
    return MappingProxyType(_sign(document))


def load_verified_audit(path: str | Path) -> Mapping[str, Any]:
    document = _load_object(Path(path))
    integrity = document.get("integrity")
    if (
        document.get("format") != FORMAT
        or document.get("version") != VERSION
        or not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "SHA-256"
        or integrity.get("payloadSha256") != _payload_hash(document)
    ):
        raise FullAuditError("audit report format/integrity is invalid")
    return MappingProxyType(document)


def write_atomic_audit(path: str | Path, audit: Mapping[str, Any]) -> Path:
    if (
        audit.get("format") != FORMAT
        or not isinstance(audit.get("integrity"), Mapping)
        or audit["integrity"].get("payloadSha256") != _payload_hash(audit)
    ):
        raise FullAuditError("refusing to write an invalid audit document")
    target = Path(path).expanduser().resolve()
    if target.exists() and target.is_symlink():
        raise FullAuditError("refusing to replace a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target.name}-",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(_json_bytes(audit))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "ApiAuditReader",
    "CANONICAL_CHARACTERISTICS",
    "DEFAULT_DRY_RUN",
    "DEFAULT_EXECUTION_STATE",
    "DbAuditReader",
    "EXPECTED_BATCH_SIZE",
    "EXPECTED_MEASUREMENT_COUNT",
    "EXPECTED_METRIC_COUNT",
    "EXPECTED_REVIEW_COUNT",
    "EXPECTED_SHOE_COUNT",
    "FullAuditError",
    "FullExpectation",
    "HttpGetAuditReader",
    "HttpGetObservation",
    "MySqlReadOnlyAuditReader",
    "PINNED_DRY_RUN_MANIFEST_SHA256",
    "audit_api",
    "audit_database",
    "create_audit",
    "load_full_expectation",
    "load_verified_audit",
    "replay_verified_loader",
    "write_atomic_audit",
]
