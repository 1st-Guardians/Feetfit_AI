"""Read-only audit for the verified eight-shoe ingestion canary.

The module has two deliberately separate boundaries:

* evidence loading rebuilds expectations from the crawler dry-run and the
  ingestion runner's recorded response state;
* injectable DB/API readers supply observations.  The production readers use
  only ``START TRANSACTION READ ONLY``/``SELECT`` and HTTP ``GET``.

No secret, JWT, database URL, username, or password is copied into reports.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

import httpx
import pymysql

from app.core.config import PROJECT_ROOT, settings


FORMAT = "feetfit-read-only-canary-ingestion-audit"
VERSION = 1
EXPECTED_CANARY_COUNT = 8
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
    / "20260826-03-canary-ingestion-live"
    / "execution-state.json"
)

_MANIFEST_FILE = "manifest.json"
_COMBINED_FILE = "combined-items.json"
_SUCCESS_OPERATIONS = frozenset({"CREATED", "UPDATED"})
_TABLES = ("shoe", "shoe_review", "shoe_lab_measurement", "shoe_lab_metric")


class CanaryAuditError(RuntimeError):
    """Raised when authoritative evidence or an observation is unsafe."""


class DbAuditReader(Protocol):
    def read(self, goods_nos: Sequence[str]) -> Mapping[str, Any]: ...


class ApiAuditReader(Protocol):
    def list_by_rating(self) -> Mapping[str, Any]: ...

    def detail(self, shoe_id: int) -> Mapping[str, Any]: ...

    def characteristics(self, shoe_id: int) -> Mapping[str, Any]: ...


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
    source_review_count: int
    imported_review_count: int
    raw_metric_count: int
    runrepeat_source_url: str
    usable_characteristics: frozenset[str]
    missing_characteristics: frozenset[str]

    @property
    def public_fields(self) -> Mapping[str, Any]:
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
                "reviewCount": self.source_review_count,
            }
        )


@dataclass(frozen=True, slots=True)
class CanaryExpectation:
    goods_nos: tuple[str, ...]
    shoes: Mapping[str, ExpectedShoe]
    dry_run_manifest_sha256: str
    execution_state_sha256: str
    replay_recorded: bool

    @property
    def shoe_ids(self) -> Mapping[str, int]:
        return MappingProxyType(
            {goods_no: self.shoes[goods_no].shoe_id for goods_no in self.goods_nos}
        )

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "shoe": len(self.shoes),
                "shoe_review": sum(
                    shoe.imported_review_count for shoe in self.shoes.values()
                ),
                "shoe_lab_measurement": len(self.shoes),
                "shoe_lab_metric": sum(
                    shoe.raw_metric_count for shoe in self.shoes.values()
                ),
            }
        )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _goods_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise CanaryAuditError("authoritative JSON file is missing or unsafe")
    try:
        value = json.loads(resolved.read_text("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryAuditError("authoritative JSON file is invalid") from exc
    if not isinstance(value, dict):
        raise CanaryAuditError("authoritative JSON root must be an object")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanaryAuditError(f"{label} must be non-blank")
    return value.strip()


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CanaryAuditError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CanaryAuditError(f"{label} must be a non-negative integer")
    return value


def _phase_successes(
    state: Mapping[str, Any], phase: str, goods_nos: Sequence[str], *, replay: bool
) -> dict[str, int]:
    expected = set(goods_nos)
    container: object
    if replay:
        idempotency = state.get("idempotency")
        container = idempotency.get("batches") if isinstance(idempotency, Mapping) else None
    else:
        container = state.get("batches")
    if not isinstance(container, list):
        raise CanaryAuditError("execution batch evidence is invalid")
    successes: dict[str, int] = {}
    for event in container:
        if not isinstance(event, Mapping) or event.get("phase") != phase:
            continue
        if bool(event.get("idempotencyReplay")) is not replay:
            continue
        response = event.get("serverResponse")
        if response is None:  # a recorded fail-stop attempt is audit evidence, not success
            continue
        if not isinstance(response, Mapping) or not isinstance(response.get("items"), list):
            raise CanaryAuditError("recorded Server response is invalid")
        rows = response["items"]
        if (
            type(response.get("requestedCount")) is not int
            or response["requestedCount"] != len(rows)
            or type(response.get("processedCount")) is not int
            or response["processedCount"] != len(rows)
        ):
            raise CanaryAuditError("recorded Server response counts are invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise CanaryAuditError("recorded Server response item is invalid")
            goods_no = str(row.get("externalKey") or "")
            shoe_id = row.get("shoeId")
            operation = str(row.get("operation") or "")
            if goods_no not in expected:
                raise CanaryAuditError("recorded response escaped canary cohort")
            if (
                row.get("matchStatus") != "MATCHED"
                or type(shoe_id) is not int
                or shoe_id <= 0
                or operation not in _SUCCESS_OPERATIONS
                or (replay and operation == "CREATED")
            ):
                raise CanaryAuditError("recorded response is not a successful exact match")
            previous = successes.setdefault(goods_no, shoe_id)
            if previous != shoe_id:
                raise CanaryAuditError("recorded response changed shoeId for one goodsNo")
    if set(successes) != expected:
        raise CanaryAuditError(f"{phase} response evidence is incomplete")
    if len(set(successes.values())) != len(successes):
        raise CanaryAuditError("multiple goodsNo resolved to the same shoeId")
    return successes


def load_canary_expectation(
    *,
    dry_run_dir: str | Path = DEFAULT_DRY_RUN,
    execution_state_path: str | Path = DEFAULT_EXECUTION_STATE,
) -> CanaryExpectation:
    """Replay artifact hashes and build the exact eight-shoe expected state."""

    dry_run = Path(dry_run_dir).expanduser().resolve()
    manifest_path = dry_run / _MANIFEST_FILE
    combined_path = dry_run / _COMBINED_FILE
    state_path = Path(execution_state_path).expanduser().resolve()
    manifest = _load_object(manifest_path)
    state = _load_object(state_path)
    manifest_sha = _sha256(manifest_path)
    state_sha = _sha256(state_path)
    if (
        manifest.get("format") != "feetfit-server-ingestion-dry-run-bundle"
        or manifest.get("version") != 1
        or not isinstance(manifest.get("files"), Mapping)
    ):
        raise CanaryAuditError("dry-run manifest format/version is invalid")
    combined_entry = manifest["files"].get(_COMBINED_FILE)
    if (
        not isinstance(combined_entry, Mapping)
        or combined_entry.get("sha256") != _sha256(combined_path)
    ):
        raise CanaryAuditError("combined dry-run hash does not match manifest")
    if (
        state.get("format") != "feetfit-verified-ingestion-execution"
        or state.get("version") != 1
        or state.get("scope") != "CANARY"
        or state.get("status") not in {"COMPLETED", "IDEMPOTENCY_FAILED"}
        or not isinstance(state.get("provenance"), Mapping)
        or state["provenance"].get("dryRunManifestSha256") != manifest_sha
    ):
        raise CanaryAuditError("execution state provenance/status is invalid")
    selected_raw = state.get("selectedGoodsNos")
    if not isinstance(selected_raw, list):
        raise CanaryAuditError("execution state selectedGoodsNos is invalid")
    goods_nos = tuple(str(value) for value in selected_raw)
    if len(goods_nos) != EXPECTED_CANARY_COUNT or len(set(goods_nos)) != len(goods_nos):
        raise CanaryAuditError("canary must contain exactly eight unique goodsNo")
    expected_set = set(goods_nos)
    for key in ("musinsaSuccessfulGoodsNos", "runRepeatSuccessfulGoodsNos"):
        values = state.get(key)
        if not isinstance(values, list) or {str(value) for value in values} != expected_set:
            raise CanaryAuditError(f"execution state {key} is incomplete")
    musinsa_ids = _phase_successes(state, "MUSINSA", goods_nos, replay=False)
    runrepeat_ids = _phase_successes(state, "RUNREPEAT", goods_nos, replay=False)
    if musinsa_ids != runrepeat_ids:
        raise CanaryAuditError("RunRepeat response shoeId/target linkage changed")
    state_ids = state.get("shoeIdsByGoodsNo")
    if (
        not isinstance(state_ids, Mapping)
        or {str(key): value for key, value in state_ids.items()} != musinsa_ids
    ):
        raise CanaryAuditError("execution state shoeId mapping disagrees with responses")
    replay_recorded = False
    idempotency = state.get("idempotency")
    if isinstance(idempotency, Mapping) and idempotency.get("requested") is True:
        replay_recorded = True
        if idempotency.get("status") not in {
            "PASSED",
            "API_RESPONSE_PASSED_DB_AUDIT_REQUIRED",
        }:
            raise CanaryAuditError("runner idempotency replay did not pass")
        if _phase_successes(state, "MUSINSA", goods_nos, replay=True) != musinsa_ids:
            raise CanaryAuditError("MUSINSA replay changed shoeId")
        if _phase_successes(state, "RUNREPEAT", goods_nos, replay=True) != musinsa_ids:
            raise CanaryAuditError("RunRepeat replay changed shoeId")

    combined = _load_object(combined_path)
    rows = combined.get("items")
    if not isinstance(rows, list):
        raise CanaryAuditError("combined dry-run items are invalid")
    by_goods: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CanaryAuditError("combined dry-run item is invalid")
        goods_no = str(row.get("goodsNo") or "")
        if goods_no in expected_set:
            if goods_no in by_goods:
                raise CanaryAuditError("combined dry-run has duplicate goodsNo")
            by_goods[goods_no] = row
    if set(by_goods) != expected_set:
        raise CanaryAuditError("canary escaped the verified combined payload")

    review_ids: set[str] = set()
    shoes: dict[str, ExpectedShoe] = {}
    for goods_no in goods_nos:
        row = by_goods[goods_no]
        reviews = row.get("reviews")
        rr = row.get("runRepeat")
        if not isinstance(reviews, list) or not isinstance(rr, Mapping):
            raise CanaryAuditError("combined review/RunRepeat evidence is invalid")
        for review in reviews:
            if not isinstance(review, Mapping):
                raise CanaryAuditError("combined review is invalid")
            review_id = _require_text(review.get("reviewId"), "reviewId")
            if review_id in review_ids:
                raise CanaryAuditError("combined canary has duplicate reviewId")
            review_ids.add(review_id)
        raw_metrics = rr.get("rawMetrics")
        usable_raw = rr.get("usableCharacteristics")
        missing_raw = rr.get("missingCharacteristics")
        if not all(isinstance(value, list) for value in (raw_metrics, usable_raw, missing_raw)):
            raise CanaryAuditError("combined RunRepeat metric evidence is invalid")
        usable = frozenset(str(value) for value in usable_raw)
        missing = frozenset(str(value) for value in missing_raw)
        raw_characteristics = frozenset(
            str(metric.get("canonicalCharacteristic"))
            for metric in raw_metrics
            if isinstance(metric, Mapping)
        )
        if (
            len(usable) < 5
            or usable | missing != CANONICAL_CHARACTERISTICS
            or usable & missing
            or raw_characteristics != usable
            or rr.get("usableCharacteristicCount") != len(usable)
        ):
            raise CanaryAuditError("combined characteristic completeness is inconsistent")
        price = row.get("price")
        rating = row.get("overallRating")
        if type(price) is not int or price < 0 or isinstance(rating, bool):
            raise CanaryAuditError("combined price/rating is invalid")
        try:
            rating_decimal = Decimal(str(rating))
        except Exception as exc:
            raise CanaryAuditError("combined overallRating is invalid") from exc
        image_url = _require_text(row.get("imageUrl"), "imageUrl")
        shoes[goods_no] = ExpectedShoe(
            goods_no=goods_no,
            shoe_id=musinsa_ids[goods_no],
            brand_name=_require_text(row.get("brandName"), "brandName"),
            shoe_name=_require_text(row.get("shoeName"), "shoeName"),
            model_code=_require_text(row.get("modelCode"), "modelCode"),
            musinsa_url=_require_text(row.get("musinsaUrl"), "musinsaUrl"),
            price=price,
            image_url=image_url,
            overall_rating=rating_decimal,
            source_review_count=_require_nonnegative_int(
                row.get("reviewCount"), "reviewCount"
            ),
            imported_review_count=len(reviews),
            raw_metric_count=len(raw_metrics),
            runrepeat_source_url=_require_text(rr.get("sourceUrl"), "RunRepeat sourceUrl"),
            usable_characteristics=usable,
            missing_characteristics=missing,
        )
    return CanaryExpectation(
        goods_nos=goods_nos,
        shoes=MappingProxyType(shoes),
        dry_run_manifest_sha256=manifest_sha,
        execution_state_sha256=state_sha,
        replay_recorded=replay_recorded,
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
    if not jdbc_url or not username or not password or not jdbc_url.startswith("jdbc:mysql://"):
        raise CanaryAuditError("SHOE_DB_* configuration is incomplete")
    parsed = urlparse(jdbc_url.replace("jdbc:mysql://", "mysql://", 1))
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise CanaryAuditError("SHOE_DB_URL host/database structure is invalid")
    query = parse_qs(parsed.query)
    source_charset = query.get("characterEncoding", ["UTF-8"])[0]
    charset = "utf8mb4" if source_charset.upper().replace("-", "") == "UTF8" else source_charset
    return _MysqlConfig(
        host=parsed.hostname,
        port=parsed.port or 3306,
        database=parsed.path.lstrip("/"),
        username=username,
        password=password,
        charset=charset,
    )


class MySqlReadOnlyAuditReader:
    """MySQL reader whose transaction and statement set are both read-only."""

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
                connect_timeout=10,
                read_timeout=30,
                write_timeout=30,
                cursorclass=pymysql.cursors.DictCursor,
            )
        except Exception as exc:
            raise CanaryAuditError(
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

    def read(self, goods_nos: Sequence[str]) -> Mapping[str, Any]:
        if len(goods_nos) != EXPECTED_CANARY_COUNT or len(set(goods_nos)) != len(goods_nos):
            raise CanaryAuditError("DB reader requires exactly eight unique goodsNo")
        placeholders = ", ".join(["%s"] * len(goods_nos))
        params = tuple(goods_nos)
        with self._cursor() as cursor:
            shoes = self._fetch(
                cursor,
                f"""/* canary:shoes */
                SELECT id AS shoe_id, musinsa_goods_no AS goods_no, brand_name,
                       shoe_name, model_code, musinsa_url, price, image_url,
                       overall_rating, review_count
                FROM shoe WHERE musinsa_goods_no IN ({placeholders})
                ORDER BY musinsa_goods_no, id""",
                params,
            )
            reviews = self._fetch(
                cursor,
                f"""/* canary:reviews */
                SELECT r.id AS review_id, r.shoe_id, r.source, r.source_review_id,
                       s.musinsa_goods_no AS goods_no
                FROM shoe_review r JOIN shoe s ON s.id = r.shoe_id
                WHERE s.musinsa_goods_no IN ({placeholders})
                ORDER BY s.musinsa_goods_no, r.id""",
                params,
            )
            measurements = self._fetch(
                cursor,
                f"""/* canary:measurements */
                SELECT m.shoe_lab_measurement_id AS measurement_id, m.shoe_id,
                       m.source, m.source_url, m.snapshot_key,
                       s.musinsa_goods_no AS goods_no
                FROM shoe_lab_measurement m JOIN shoe s ON s.id = m.shoe_id
                WHERE s.musinsa_goods_no IN ({placeholders})
                ORDER BY s.musinsa_goods_no, m.shoe_lab_measurement_id""",
                params,
            )
            metrics = self._fetch(
                cursor,
                f"""/* canary:metrics */
                SELECT lm.shoe_lab_metric_id AS metric_id,
                       lm.shoe_lab_measurement_id AS measurement_id,
                       m.shoe_id, s.musinsa_goods_no AS goods_no,
                       lm.canonical_characteristic
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
                f"""/* canary:runrepeat_import_audits */
                SELECT shoe_import_audit_id AS audit_id, source, external_key,
                       match_status, matched_shoe_id, raw_payload
                FROM shoe_import_audit
                WHERE source = %s AND match_status = %s
                  AND external_key IN ({placeholders})
                ORDER BY shoe_import_audit_id""",
                ("RUNREPEAT", "MATCHED", *params),
            )
            duplicate_goods = self._fetch(
                cursor,
                """/* canary:duplicate_goods */
                SELECT musinsa_goods_no AS goods_no, COUNT(*) AS record_count
                FROM shoe WHERE musinsa_goods_no IS NOT NULL
                GROUP BY musinsa_goods_no HAVING COUNT(*) > 1""",
            )
            duplicate_reviews = self._fetch(
                cursor,
                """/* canary:duplicate_reviews */
                SELECT shoe_id, source, source_review_id, COUNT(*) AS record_count
                FROM shoe_review WHERE source_review_id IS NOT NULL
                GROUP BY shoe_id, source, source_review_id HAVING COUNT(*) > 1""",
            )
            cross_shoe_review_ids = self._fetch(
                cursor,
                """/* canary:cross_shoe_review_ids */
                SELECT source, source_review_id,
                       COUNT(DISTINCT shoe_id) AS record_count
                FROM shoe_review WHERE source_review_id IS NOT NULL
                GROUP BY source, source_review_id
                HAVING COUNT(DISTINCT shoe_id) > 1""",
            )
            duplicate_snapshots = self._fetch(
                cursor,
                """/* canary:duplicate_snapshots */
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
                rows = self._fetch(cursor, f"/* canary:orphan:{name} */ {sql}")
                orphan_counts[name] = int(rows[0]["record_count"])
        return MappingProxyType(
            {
                # Counts are deliberately scoped to the exact canary cohort.
                "counts": {
                    "shoe": len(shoes),
                    "shoe_review": len(reviews),
                    "shoe_lab_measurement": len(measurements),
                    "shoe_lab_metric": len(metrics),
                },
                "shoes": shoes,
                "reviews": reviews,
                "measurements": measurements,
                "metrics": metrics,
                "importAudits": import_audits,
                "duplicates": {
                    "goodsNo": duplicate_goods,
                    "reviewIdentity": duplicate_reviews,
                    "reviewIdAcrossShoes": cross_shoe_review_ids,
                    "snapshotKey": duplicate_snapshots,
                },
                "orphans": orphan_counts,
            }
        )


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_local_audit_jwt(secret_base64: str, user_id: int, *, now: int | None = None) -> str:
    """Create the same HS256 subject token as Feetfit_Server's TokenProvider."""

    if not secret_base64 or type(user_id) is not int or user_id <= 0:
        raise CanaryAuditError("JWT_SECRET and a positive audit userId are required")
    try:
        padded = secret_base64 + "=" * (-len(secret_base64) % 4)
        key = base64.b64decode(padded, validate=True)
    except Exception as exc:
        raise CanaryAuditError("JWT_SECRET must be valid base64") from exc
    if len(key) < 32:
        raise CanaryAuditError("JWT_SECRET is too short for HS256")
    issued_at = int(time.time()) if now is None else now
    header = _b64url(json.dumps({"alg": "HS256"}, separators=(",", ":")).encode())
    payload = _b64url(
        json.dumps(
            {"sub": str(user_id), "iat": issued_at, "exp": issued_at + 900},
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = _b64url(hmac.new(key, signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


class HttpGetAuditReader:
    """Loopback-only GET adapter.  It has no mutation methods."""

    def __init__(
        self,
        *,
        base_url: str,
        jwt_secret: str,
        user_id: int,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise CanaryAuditError("live API audit is restricted to a loopback HTTP server")
        self._base_url = base_url.rstrip("/")
        self._token = create_local_audit_jwt(jwt_secret, user_id)
        self._client = client

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if not path.startswith("/api/shoes"):
            raise CanaryAuditError("API audit path escaped /api/shoes")
        owned = self._client is None
        client = self._client or httpx.Client(timeout=15.0)
        try:
            response = client.get(
                f"{self._base_url}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
            )
            response.raise_for_status()
            value = response.json(parse_float=Decimal)
        except (httpx.HTTPError, ValueError) as exc:
            raise CanaryAuditError(f"GET audit failed ({type(exc).__name__}); secrets suppressed") from None
        finally:
            if owned:
                client.close()
        if not isinstance(value, Mapping):
            raise CanaryAuditError("GET response root is invalid")
        return value

    def list_by_rating(self) -> Mapping[str, Any]:
        return self._get("/api/shoes", {"sort": "RATING", "page": 0, "size": 100})

    def detail(self, shoe_id: int) -> Mapping[str, Any]:
        return self._get(f"/api/shoes/{shoe_id}")

    def characteristics(self, shoe_id: int) -> Mapping[str, Any]:
        return self._get(f"/api/shoes/{shoe_id}/characteristics")


def http_reader_from_environment(*, base_url: str, user_id: int) -> HttpGetAuditReader:
    secret = os.environ.get("JWT_SECRET", "")
    return HttpGetAuditReader(base_url=base_url, jwt_secret=secret, user_id=user_id)


def _as_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except Exception:
        return None
    return result if result.is_finite() else None


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(expected, Decimal):
        return _as_decimal(actual) == expected
    return actual == expected


def _api_result(document: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if (
        document.get("isSuccess") is not True
        or not isinstance(document.get("code"), str)
        or not isinstance(document.get("message"), str)
        or not isinstance(document.get("result"), Mapping)
    ):
        raise CanaryAuditError(f"{label} is not a successful ApiResponse")
    return document["result"]


def _row_goods(row: Mapping[str, Any]) -> str:
    return str(row.get("goods_no") or "")


def audit_database(
    observation: Mapping[str, Any], expectation: CanaryExpectation
) -> Mapping[str, Any]:
    """Validate exact table/cohort counts and all DB identity relationships."""

    issues: list[str] = []
    counts = observation.get("counts")
    if not isinstance(counts, Mapping):
        raise CanaryAuditError("DB observation counts are missing")
    normalized_counts: dict[str, int] = {}
    for table in _TABLES:
        value = counts.get(table)
        if type(value) is not int or value < 0:
            raise CanaryAuditError(f"DB count is invalid for {table}")
        normalized_counts[table] = value
        if value != expectation.counts[table]:
            issues.append(f"GLOBAL_COUNT_MISMATCH:{table}")
    collections: dict[str, list[Mapping[str, Any]]] = {}
    for key in ("shoes", "reviews", "measurements", "metrics"):
        value = observation.get(key)
        if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
            raise CanaryAuditError(f"DB observation {key} is invalid")
        collections[key] = value
    import_audits = observation.get("importAudits")
    if not isinstance(import_audits, list) or not all(
        isinstance(row, Mapping) for row in import_audits
    ):
        raise CanaryAuditError("DB RunRepeat import audit evidence is invalid")
    duplicates = observation.get("duplicates")
    orphans = observation.get("orphans")
    if not isinstance(duplicates, Mapping) or not isinstance(orphans, Mapping):
        raise CanaryAuditError("DB duplicate/orphan evidence is invalid")
    duplicate_counts: dict[str, int] = {}
    for key in ("goodsNo", "reviewIdentity", "reviewIdAcrossShoes", "snapshotKey"):
        rows = duplicates.get(key)
        if not isinstance(rows, list):
            raise CanaryAuditError(f"DB duplicate evidence is invalid for {key}")
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
            raise CanaryAuditError(f"DB orphan count is invalid for {key}")
        orphan_counts[key] = value
        if value:
            issues.append(f"ORPHAN_FOUND:{key}")

    goods_set = set(expectation.goods_nos)
    for key, rows in collections.items():
        escaped = sorted(
            {_row_goods(row) for row in rows if _row_goods(row) not in goods_set},
            key=_goods_key,
        )
        if escaped:
            issues.append(f"COHORT_ESCAPE:{key}")
    escaped_audits = sorted(
        {
            str(row.get("external_key") or "")
            for row in import_audits
            if str(row.get("external_key") or "") not in goods_set
        },
        key=_goods_key,
    )
    if escaped_audits:
        issues.append("COHORT_ESCAPE:importAudits")
    per_goods: dict[str, Any] = {}
    global_review_ids: set[str] = set()
    identity_material: list[str] = []
    wrong_target_count = len(escaped_audits)
    for goods_no in expectation.goods_nos:
        expected = expectation.shoes[goods_no]
        shoe_rows = [row for row in collections["shoes"] if _row_goods(row) == goods_no]
        review_rows = [row for row in collections["reviews"] if _row_goods(row) == goods_no]
        measurement_rows = [
            row for row in collections["measurements"] if _row_goods(row) == goods_no
        ]
        metric_rows = [row for row in collections["metrics"] if _row_goods(row) == goods_no]
        audit_rows = [
            row
            for row in import_audits
            if str(row.get("external_key") or "") == goods_no
        ]
        row_issues: list[str] = []
        expected_counts = {
            "shoe": 1,
            "shoe_review": expected.imported_review_count,
            "shoe_lab_measurement": 1,
            "shoe_lab_metric": expected.raw_metric_count,
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
            db_expected = {
                "brand_name": expected.brand_name,
                "shoe_name": expected.shoe_name,
                "model_code": expected.model_code,
                "musinsa_url": expected.musinsa_url,
                "price": expected.price,
                "image_url": expected.image_url,
                "overall_rating": expected.overall_rating,
                "review_count": expected.source_review_count,
            }
            for field_name, expected_value in db_expected.items():
                if not _same_value(shoe.get(field_name), expected_value):
                    row_issues.append(f"SHOE_FIELD_MISMATCH:{field_name}")
            identity_material.append(f"shoe:{goods_no}:{shoe.get('shoe_id')}")
        review_ids: list[str] = []
        for review in review_rows:
            if review.get("shoe_id") != expected.shoe_id:
                row_issues.append("REVIEW_WRONG_SHOE_ID")
                wrong_target_count += 1
            if review.get("source") != "MUSINSA":
                row_issues.append("REVIEW_SOURCE_MISMATCH")
            source_id = review.get("source_review_id")
            if not isinstance(source_id, str) or not source_id.strip():
                row_issues.append("REVIEW_ID_MISSING")
                continue
            if source_id in global_review_ids:
                row_issues.append("DUPLICATE_REVIEW_ID")
            global_review_ids.add(source_id)
            review_ids.append(source_id)
            identity_material.append(f"review:{goods_no}:{review.get('review_id')}:{source_id}")
        measurement_ids: set[int] = set()
        for measurement in measurement_rows:
            if measurement.get("shoe_id") != expected.shoe_id:
                row_issues.append("MEASUREMENT_WRONG_SHOE_ID")
                wrong_target_count += 1
            if measurement.get("source") != "RUNREPEAT":
                row_issues.append("MEASUREMENT_SOURCE_MISMATCH")
            if measurement.get("source_url") != expected.runrepeat_source_url:
                row_issues.append("RUNREPEAT_TARGET_URL_MISMATCH")
                wrong_target_count += 1
            measurement_id = measurement.get("measurement_id")
            if type(measurement_id) is not int or measurement_id <= 0:
                row_issues.append("MEASUREMENT_ID_INVALID")
            else:
                measurement_ids.add(measurement_id)
                identity_material.append(f"measurement:{goods_no}:{measurement_id}")
        metric_types: list[str] = []
        for metric in metric_rows:
            if metric.get("shoe_id") != expected.shoe_id:
                row_issues.append("METRIC_WRONG_SHOE_ID")
                wrong_target_count += 1
            if metric.get("measurement_id") not in measurement_ids:
                row_issues.append("METRIC_WRONG_MEASUREMENT_ID")
                wrong_target_count += 1
            characteristic = str(metric.get("canonical_characteristic") or "")
            if characteristic not in CANONICAL_CHARACTERISTICS:
                row_issues.append("METRIC_CHARACTERISTIC_INVALID")
            else:
                metric_types.append(characteristic)
            identity_material.append(
                f"metric:{goods_no}:{metric.get('metric_id')}:{metric.get('measurement_id')}"
            )
        usable_in_db = frozenset(metric_types)
        missing_in_db = CANONICAL_CHARACTERISTICS - usable_in_db
        if usable_in_db != expected.usable_characteristics:
            row_issues.append("USABLE_CHARACTERISTICS_MISMATCH")
        if missing_in_db != expected.missing_characteristics:
            row_issues.append("MISSING_CHARACTERISTICS_MISMATCH")
        if len(usable_in_db) < 5:
            row_issues.append("BELOW_FIVE_CHARACTERISTICS")
        expected_audit_count = 2 if expectation.replay_recorded else 1
        if len(audit_rows) != expected_audit_count:
            row_issues.append("RUNREPEAT_MATCHED_AUDIT_COUNT_MISMATCH")
        for import_audit in audit_rows:
            if import_audit.get("source") != "RUNREPEAT":
                row_issues.append("RUNREPEAT_AUDIT_SOURCE_MISMATCH")
            if import_audit.get("match_status") != "MATCHED":
                row_issues.append("RUNREPEAT_AUDIT_STATUS_MISMATCH")
            if import_audit.get("matched_shoe_id") != expected.shoe_id:
                row_issues.append("RUNREPEAT_AUDIT_SHOE_ID_MISMATCH")
                wrong_target_count += 1
            raw_payload = import_audit.get("raw_payload")
            if isinstance(raw_payload, Mapping):
                raw_object = raw_payload
            elif isinstance(raw_payload, str):
                try:
                    decoded = json.loads(raw_payload, parse_float=Decimal)
                except json.JSONDecodeError:
                    decoded = None
                raw_object = decoded if isinstance(decoded, Mapping) else None
            else:
                raw_object = None
            if raw_object is None:
                row_issues.append("RUNREPEAT_AUDIT_RAW_PAYLOAD_INVALID")
                wrong_target_count += 1
            elif raw_object.get("targetGoodsNo") != goods_no:
                # Exact Python comparison intentionally avoids DB collation rules.
                row_issues.append("RUNREPEAT_AUDIT_TARGET_GOODS_NO_MISMATCH")
                wrong_target_count += 1
        if row_issues:
            issues.extend(f"{issue}:{goods_no}" for issue in row_issues)
        per_goods[goods_no] = {
            "shoeId": expected.shoe_id,
            "expectedCounts": expected_counts,
            "actualCounts": actual_counts,
            "usableCharacteristics": sorted(usable_in_db),
            "missingCharacteristics": sorted(missing_in_db),
            "usableCharacteristicCount": len(usable_in_db),
            "reviewIdCount": len(review_ids),
            "runRepeatMatchedAuditCount": len(audit_rows),
            "issues": sorted(set(row_issues)),
            "status": "PASS" if not row_issues else "FAIL",
        }
    fingerprint = hashlib.sha256(
        "\n".join(sorted(identity_material)).encode("utf-8")
    ).hexdigest()
    return MappingProxyType(
        {
            "status": "PASS" if not issues else "FAIL",
            "counts": normalized_counts,
            "expectedCounts": dict(expectation.counts),
            "duplicateCounts": duplicate_counts,
            "orphanCounts": orphan_counts,
            "runRepeatMatchedAuditCount": len(import_audits),
            "wrongTargetCount": wrong_target_count,
            "characteristicBelowFiveCount": sum(
                "BELOW_FIVE_CHARACTERISTICS" in issue for issue in issues
            ),
            "perGoods": per_goods,
            "identityFingerprint": fingerprint,
            "issues": sorted(set(issues)),
        }
    )


def _validate_public_fields(
    row: Mapping[str, Any], expected: ExpectedShoe
) -> list[str]:
    errors: list[str] = []
    for name, expected_value in expected.public_fields.items():
        if not _same_value(row.get(name), expected_value):
            errors.append(f"FIELD_MISMATCH:{name}")
    return errors


def audit_api(reader: ApiAuditReader, expectation: CanaryExpectation) -> Mapping[str, Any]:
    """Issue only GETs through the adapter and validate public contracts."""

    issues: list[str] = []
    list_result = _api_result(reader.list_by_rating(), "shoe list")
    list_rows = list_result.get("shoes")
    if not isinstance(list_rows, list) or not all(isinstance(row, Mapping) for row in list_rows):
        raise CanaryAuditError("shoe list rows are invalid")
    expected_ids = set(expectation.shoe_ids.values())
    actual_ids = [row.get("id") for row in list_rows]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        issues.append("LIST_SHOE_ID_COHORT_MISMATCH")
    if list_result.get("totalElements") != EXPECTED_CANARY_COUNT:
        issues.append("LIST_TOTAL_ELEMENTS_MISMATCH")
    if list_result.get("currentPage") != 0 or list_result.get("hasNext") is not False:
        issues.append("LIST_PAGINATION_MISMATCH")
    ratings = [_as_decimal(row.get("overallRating")) for row in list_rows]
    if any(value is None for value in ratings) or any(
        ratings[index] < ratings[index + 1] for index in range(len(ratings) - 1)
    ):
        issues.append("LIST_RATING_SORT_MISMATCH")
    expected_by_id = {shoe.shoe_id: shoe for shoe in expectation.shoes.values()}
    for row in list_rows:
        shoe = expected_by_id.get(row.get("id"))
        if shoe is None:
            continue
        issues.extend(
            f"LIST_{error}:{shoe.goods_no}" for error in _validate_public_fields(row, shoe)
        )

    per_goods: dict[str, Any] = {}
    for goods_no in expectation.goods_nos:
        expected = expectation.shoes[goods_no]
        row_issues: list[str] = []
        detail = _api_result(reader.detail(expected.shoe_id), "shoe detail")
        row_issues.extend(
            f"DETAIL_{error}" for error in _validate_public_fields(detail, expected)
        )
        characteristic_result = _api_result(
            reader.characteristics(expected.shoe_id), "shoe characteristics"
        )
        if characteristic_result.get("shoeId") != expected.shoe_id:
            row_issues.append("CHARACTERISTIC_SHOE_ID_MISMATCH")
        characteristic_rows = characteristic_result.get("characteristics")
        if not isinstance(characteristic_rows, list) or not all(
            isinstance(row, Mapping) for row in characteristic_rows
        ):
            raise CanaryAuditError("shoe characteristic rows are invalid")
        types: list[str] = []
        required_fields = {
            "type",
            "level",
            "value",
            "averageValue",
            "minValue",
            "maxValue",
            "unit",
            "testedSize",
        }
        for row in characteristic_rows:
            if not required_fields <= set(row):
                row_issues.append("CHARACTERISTIC_FIELDS_MISSING")
            characteristic = str(row.get("type") or "")
            if characteristic not in CANONICAL_CHARACTERISTICS:
                row_issues.append("CHARACTERISTIC_TYPE_INVALID")
            else:
                types.append(characteristic)
            # A level can be unavailable when the metric's compatible cohort is
            # too small.  Preserve that source truth as null; never invent it.
            if row.get("level") not in {None, "LOW", "MEDIUM", "HIGH"}:
                row_issues.append("CHARACTERISTIC_LEVEL_INVALID")
            if _as_decimal(row.get("value")) is None:
                row_issues.append("CHARACTERISTIC_VALUE_MISSING")
        if len(types) != len(set(types)):
            row_issues.append("CHARACTERISTIC_TYPE_DUPLICATE")
        returned_usable = frozenset(types)
        returned_missing = CANONICAL_CHARACTERISTICS - returned_usable
        if returned_usable != expected.usable_characteristics:
            row_issues.append("USABLE_CHARACTERISTICS_MISMATCH")
        if returned_missing != expected.missing_characteristics:
            row_issues.append("MISSING_CHARACTERISTICS_MISMATCH")
        if len(returned_usable) < 5:
            row_issues.append("BELOW_FIVE_CHARACTERISTICS")
        if not isinstance(characteristic_result.get("summary"), str) or not characteristic_result["summary"].strip():
            row_issues.append("CHARACTERISTIC_SUMMARY_MISSING")
        if row_issues:
            issues.extend(f"{issue}:{goods_no}" for issue in row_issues)
        per_goods[goods_no] = {
            "shoeId": expected.shoe_id,
            "usableCharacteristics": sorted(returned_usable),
            "missingCharacteristics": sorted(returned_missing),
            "usableCharacteristicCount": len(returned_usable),
            "issues": sorted(set(row_issues)),
            "status": "PASS" if not row_issues else "FAIL",
        }
    return MappingProxyType(
        {
            "status": "PASS" if not issues else "FAIL",
            "list": {
                "requestedSort": "RATING",
                "returnedCount": len(list_rows),
                "totalElements": list_result.get("totalElements"),
                "shoeIds": actual_ids,
            },
            "perGoods": per_goods,
            "issues": sorted(set(issues)),
            "requestCounts": {"list": 1, "detail": 8, "characteristics": 8},
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
        raise CanaryAuditError("prior audit format or integrity is invalid")
    return MappingProxyType(document)


def create_audit(
    *,
    expectation: CanaryExpectation,
    db_reader: DbAuditReader,
    api_reader: ApiAuditReader,
    phase: str,
    prior_audit: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run one read-only audit and optionally prove retry idempotency."""

    phase_name = _require_text(phase, "phase")
    db = audit_database(db_reader.read(expectation.goods_nos), expectation)
    api = audit_api(api_reader, expectation)
    idempotency: dict[str, Any] = {
        "status": "NOT_CHECKED",
        "evidenceMode": "NO_RECORDED_REPLAY",
        "runnerReplayRecorded": expectation.replay_recorded,
        "priorComparisonStatus": "NOT_PROVIDED",
        "priorAuditPayloadSha256": None,
        "issues": [],
    }
    if expectation.replay_recorded:
        replay_issues = [] if db["status"] == "PASS" else ["EXACT_DB_STATE_FAILED"]
        idempotency = {
            "status": "PASS" if not replay_issues else "FAIL",
            "evidenceMode": "RECORDED_UPDATED_REPLAY_AND_EXACT_DB_STATE",
            "runnerReplayRecorded": True,
            "priorComparisonStatus": "NOT_PROVIDED",
            "priorAuditPayloadSha256": None,
            "issues": replay_issues,
        }
    if prior_audit is not None:
        prior_provenance = prior_audit.get("provenance")
        prior_db = prior_audit.get("database")
        idem_issues: list[str] = []
        if not isinstance(prior_provenance, Mapping) or (
            prior_provenance.get("dryRunManifestSha256")
            != expectation.dry_run_manifest_sha256
        ):
            idem_issues.append("PRIOR_PROVENANCE_MISMATCH")
        if not isinstance(prior_db, Mapping):
            idem_issues.append("PRIOR_DB_EVIDENCE_MISSING")
        else:
            if prior_db.get("counts") != db["counts"]:
                idem_issues.append("IDEMPOTENCY_COUNT_CHANGED")
            if prior_db.get("identityFingerprint") != db["identityFingerprint"]:
                idem_issues.append("IDEMPOTENCY_ROW_IDENTITY_CHANGED")
        if not expectation.replay_recorded:
            idem_issues.append("RUNNER_REPLAY_NOT_RECORDED")
        if db["status"] != "PASS":
            idem_issues.append("EXACT_DB_STATE_FAILED")
        idempotency = {
            "status": "PASS" if not idem_issues else "FAIL",
            "evidenceMode": (
                "RECORDED_UPDATED_REPLAY_AND_EXACT_DB_STATE"
                if expectation.replay_recorded
                else "PRIOR_ARTIFACT_COMPARISON_WITHOUT_RECORDED_REPLAY"
            ),
            "runnerReplayRecorded": expectation.replay_recorded,
            "priorComparisonStatus": "PASS" if not idem_issues else "FAIL",
            "priorAuditPayloadSha256": prior_audit.get("integrity", {}).get(
                "payloadSha256"
            )
            if isinstance(prior_audit.get("integrity"), Mapping)
            else None,
            "issues": idem_issues,
        }
    issues = [
        *(f"DB:{item}" for item in db["issues"]),
        *(f"API:{item}" for item in api["issues"]),
        *(f"IDEMPOTENCY:{item}" for item in idempotency["issues"]),
    ]
    document = {
        "format": FORMAT,
        "version": VERSION,
        "createdAt": _now(),
        "phase": phase_name,
        "status": "PASS" if not issues else "FAIL",
        "provenance": {
            "dryRunManifestSha256": expectation.dry_run_manifest_sha256,
            "executionStateSha256": expectation.execution_state_sha256,
            "authoritativeCanaryCount": EXPECTED_CANARY_COUNT,
        },
        "expected": {
            "goodsNos": list(expectation.goods_nos),
            "shoeIdsByGoodsNo": dict(expectation.shoe_ids),
            "counts": dict(expectation.counts),
            "perGoods": {
                goods_no: {
                    "shoeId": shoe.shoe_id,
                    "importedReviewCount": shoe.imported_review_count,
                    "rawMetricCount": shoe.raw_metric_count,
                    "usableCharacteristics": sorted(shoe.usable_characteristics),
                    "missingCharacteristics": sorted(shoe.missing_characteristics),
                    "usableCharacteristicCount": len(shoe.usable_characteristics),
                }
                for goods_no, shoe in expectation.shoes.items()
            },
        },
        "database": dict(db),
        "api": dict(api),
        "idempotency": idempotency,
        "issues": issues,
        "safety": {
            "databaseTransactionReadOnly": True,
            "httpMethods": ["GET"],
            "serverMutationRequested": False,
            "databaseMutationRequested": False,
            "secretValuesIncluded": False,
            "shoeComparisonFeatureIncluded": False,
        },
    }
    return MappingProxyType(_sign(document))


def write_atomic_audit(path: str | Path, audit: Mapping[str, Any]) -> Path:
    if audit.get("format") != FORMAT or audit.get("integrity", {}).get(
        "payloadSha256"
    ) != _payload_hash(audit):
        raise CanaryAuditError("refusing to write an invalid audit document")
    target = Path(path).expanduser().resolve()
    if target.exists() and target.is_symlink():
        raise CanaryAuditError("refusing to replace a symlink")
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
    "CanaryAuditError",
    "CanaryExpectation",
    "DbAuditReader",
    "ExpectedShoe",
    "HttpGetAuditReader",
    "MySqlReadOnlyAuditReader",
    "audit_api",
    "audit_database",
    "create_audit",
    "create_local_audit_jwt",
    "http_reader_from_environment",
    "load_canary_expectation",
    "load_verified_audit",
    "write_atomic_audit",
]
