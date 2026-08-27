"""Fail-closed authority gate for the one reviewed FeetFit production migration.

This module is intentionally narrower than the clone rehearsal tool.  It can
only migrate the exact rehearsed ``feetfit`` schema, from the exact V1/V2
success + V3 failure state, with the exact Server boot jar and a fresh external
assertion that every Server/ingestion writer is stopped.

There is no cleanup, rollback, retry, resume, HTTP, or Server-API code here.
After Flyway starts, every failure is an indeterminate partial migration that
requires manual inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode, urlunparse

import pymysql

from app.core.config import PROJECT_ROOT
from app.services.shoe.flyway_clone_rehearsal import (
    DEFAULT_SERVER_BOOT_JAR,
    DEFAULT_SERVER_ROOT,
    EXPECTED_MIGRATION_VERSIONS,
    PROTECTED_TABLES,
    FlywayRehearsalError,
    MigrationManifest,
    MysqlConfig,
    TableDigest,
    _HISTORY_COLUMNS,
    _atomic_json,
    _connect,
    _digests_public,
    _ordered_select_sql,
    _quote_identifier,
    _read_source_shapes,
    _redact_process_output,
    _safe_extract_boot_runtime,
    _schema_digest,
    load_migration_manifest,
    read_production_reconciliation_invariants,
    resolve_java_home,
    sha256_file,
    update_row_digest,
    validate_post_flyway_history,
)
from app.services.shoe.writer_absence_evidence import (
    POWERSHELL_PROBE_SHA256,
    PRODUCER as WRITER_EVIDENCE_PRODUCER,
    collect_writer_absence_evidence,
)


FORMAT = "feetfit-flyway-production-authority"
VERSION = 1
WRITER_EVIDENCE_FORMAT = "feetfit-server-writer-absence-evidence"
WRITER_EVIDENCE_VERSION = 1
CONFIRMATION_DOMAIN = "FEETFIT_PRODUCTION_FLYWAY_V1_EXACT_REHEARSAL_AUTHORITY"
EXPECTED_PRODUCTION_DATABASE = "feetfit"
EXPECTED_JDBC_QUERY_ITEMS = (
    ("serverTimezone", "Asia/Seoul"),
    ("characterEncoding", "UTF-8"),
    ("useSSL", "false"),
    ("allowPublicKeyRetrieval", "true"),
)
TARGET_VERSION = "8"
REMOTE_WRITER_FENCE_AUTHORIZED = False

PINNED_MIGRATION_MANIFEST_SHA256 = (
    "ea252a5e6074e69f402e0bedf2b5a12462aa93aac3ff58d01f7567782bbf2251"
)
PINNED_BOOT_JAR_SHA256 = (
    "a80e7f3439afe058465343a6bf9f5a37e8dc9fad11a9a8116a06886194086a0e"
)
PINNED_CLONE_REPORT_SHA256 = (
    "bd9ebd43ce1e2c99c4523c8df12a50b5d06642e24864cc4c5193a51bd0f0f864"
)
PINNED_JAVA_HELPER_SHA256 = (
    "c4367f76d4cb29359b73d6f18db0c2d059e27b7baf25a91de91c04cac15ad47c"
)
PINNED_WRITER_EVIDENCE_PRODUCER_SHA256 = (
    "0430d6784f1fb950b6344ef0714781b15788d8b22e5e400149cce9af626d7853"
)
PINNED_PRE_MIGRATION_SCHEMA_SHA256 = (
    "3431cb81835f8bc7bdd44a021fdd125a7700f8e366fbf7d47a07ad069f0cf1e3"
)
PINNED_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "1",
        "V1__baseline_shoe_tables.sql",
        "877090e1984ebc8faa585f271549e2908f102906e6fe835ecefcc9b6f0f7703b",
    ),
    (
        "2",
        "V2__shoe_ingestion_and_analysis_contract.sql",
        "f00adfbe38ecee67655c02dd9e41c54b1b3551969db1d9aaed57c7cb768422ff",
    ),
    (
        "3",
        "V3__add_tina_pedis_sole_analysis_images.sql",
        "0e2f8700b7081744733cc0ed66a5120be5bfaab06f7ee90ee237794a44d7a688",
    ),
    (
        "4",
        "V4__add_daily_foot_analysis_plantar_footprint_fields.sql",
        "720dc8b21caf6f2179795c895045d5be747418310457e2113621fdf965452612",
    ),
    (
        "5",
        "V5__create_measurement_analysis_status.sql",
        "38a3f49848f4b69d7315d8800308e00ca9e55de1674537026174a91bc5cbc572",
    ),
    (
        "6",
        "V6__replace_processing_measurement_status.sql",
        "cbf9afc44072c1ebed8a34871b8f1c54a2678546dda6dded72df3b71ab777982",
    ),
    (
        "7",
        "V7__schema_authority_reconciliation.sql",
        "41ebf5277246dc84194cc4de3c0ae29b0db7243b51bd3af30a07e6ddd4c6f12e",
    ),
    (
        "8",
        "V8__recommendation_session_scope.sql",
        "499ebad1c3535462e8f7740045c6aba35de6401a57c59b2ba8e6c2fede24a074",
    ),
)

DEFAULT_SUCCESSFUL_CLONE_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "flyway-rehearsal"
    / "feetfit_flyway_rehearsal_20260826T095848978641Z_61199ed8"
    / "rehearsal-report.json"
)
DEFAULT_JAVA_SOURCE = (
    PROJECT_ROOT / "scripts" / "java" / "FeetfitFlywayProductionApply.java"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "flyway-production-apply"
MIGRATION_LOCK_NAME = "feetfit_flyway_production_apply_v1"
MAINTENANCE_WINDOW_AUTHORITY = (
    "FEETFIT_SOLE_WRITER_HOST_AND_RESTART_INHIBITED_V1"
)
MAX_WRITER_EVIDENCE_AGE = timedelta(minutes=5)
MAX_WRITER_EVIDENCE_FUTURE_SKEW = timedelta(seconds=30)

RECOMMENDATION_TABLES_BEFORE = (
    "shoe_recommendation",
    "shoe_recommendation_reason",
    "shoe_recommendation_reason_review",
)
RECOMMENDATION_RUN_TABLE = "shoe_recommendation_run"
RECOMMENDATION_TABLES_AFTER = RECOMMENDATION_TABLES_BEFORE + (
    RECOMMENDATION_RUN_TABLE,
)

PINNED_PROTECTED_DIGESTS: Mapping[str, TableDigest] = {
    "shoe": TableDigest(
        338, "a7d69648a2b774033ba5a140c99d8310faad9453234a05d1fd708086c681aafe"
    ),
    "shoe_review": TableDigest(
        8517, "1661ec538caba255b26705dc47a3bafbe703e90c2a8b7413844841013b0d80cb"
    ),
    "shoe_lab_measurement": TableDigest(
        338, "7ef8d1df908394ceeacbbfd542d4786f4e7a8b9e72120c1f6ebb7cbdec90c7e6"
    ),
    "shoe_lab_metric": TableDigest(
        3029, "c29e98c3fd91c82636c6de0fd6a17b7cc2b232903405d28edba4ccfc7ab441b9"
    ),
}

EXPECTED_INITIAL_HISTORY: tuple[Mapping[str, Any], ...] = (
    {
        "installed_rank": 1,
        "version": "1",
        "description": "baseline shoe tables",
        "type": "SQL",
        "script": "V1__baseline_shoe_tables.sql",
        "checksum": 402784791,
        "installed_on": "2026-08-24T06:50:39.000000",
        "execution_time": 2930,
        "success": True,
    },
    {
        "installed_rank": 2,
        "version": "2",
        "description": "shoe ingestion and analysis contract",
        "type": "SQL",
        "script": "V2__shoe_ingestion_and_analysis_contract.sql",
        "checksum": 4853176,
        "installed_on": "2026-08-24T06:50:45.000000",
        "execution_time": 3117,
        "success": True,
    },
    {
        "installed_rank": 3,
        "version": "3",
        "description": "add tina pedis sole analysis images",
        "type": "SQL",
        "script": "V3__add_tina_pedis_sole_analysis_images.sql",
        "checksum": 1248033283,
        "installed_on": "2026-08-24T06:50:49.000000",
        "execution_time": 1282,
        "success": False,
    },
)

EXPECTED_FINAL_HISTORY_CORE: tuple[Mapping[str, Any], ...] = (
    {
        "installed_rank": 1,
        "version": "1",
        "description": "baseline shoe tables",
        "type": "SQL",
        "script": "V1__baseline_shoe_tables.sql",
        "checksum": 402784791,
        "success": True,
    },
    {
        "installed_rank": 2,
        "version": "2",
        "description": "shoe ingestion and analysis contract",
        "type": "SQL",
        "script": "V2__shoe_ingestion_and_analysis_contract.sql",
        "checksum": 4853176,
        "success": True,
    },
    {
        "installed_rank": 3,
        "version": "3",
        "description": "add tina pedis sole analysis images",
        "type": "SQL",
        "script": "V3__add_tina_pedis_sole_analysis_images.sql",
        "checksum": -771828580,
        "success": True,
    },
    {
        "installed_rank": 4,
        "version": "4",
        "description": "add daily foot analysis plantar footprint fields",
        "type": "SQL",
        "script": "V4__add_daily_foot_analysis_plantar_footprint_fields.sql",
        "checksum": 1807917992,
        "success": True,
    },
    {
        "installed_rank": 5,
        "version": "5",
        "description": "create measurement analysis status",
        "type": "SQL",
        "script": "V5__create_measurement_analysis_status.sql",
        "checksum": -865623862,
        "success": True,
    },
    {
        "installed_rank": 6,
        "version": "6",
        "description": "replace processing measurement status",
        "type": "SQL",
        "script": "V6__replace_processing_measurement_status.sql",
        "checksum": 1692137629,
        "success": True,
    },
    {
        "installed_rank": 7,
        "version": "7",
        "description": "schema authority reconciliation",
        "type": "SQL",
        "script": "V7__schema_authority_reconciliation.sql",
        "checksum": -987218186,
        "success": True,
    },
    {
        "installed_rank": 8,
        "version": "8",
        "description": "recommendation session scope",
        "type": "SQL",
        "script": "V8__recommendation_session_scope.sql",
        "checksum": -1853854896,
        "success": True,
    },
)

_HISTORY_CORE_FIELDS = (
    "installed_rank",
    "version",
    "description",
    "type",
    "script",
    "checksum",
    "success",
)
_WRITER_EVIDENCE_KEYS = {
    "format",
    "version",
    "status",
    "checkedAt",
    "sourceDatabase",
    "databaseFingerprint",
    "serverUuid",
    "serverWriterAbsent",
    "activeServerWriterProcessIds",
    "activeDatabaseWriterSessionIds",
    "javaServerProcessIds",
    "uninspectableJavaProcessIds",
    "pythonWriterProcessIds",
    "uninspectablePythonProcessIds",
    "listener8080ProcessIds",
    "listener8081ProcessIds",
    "mysqlProcessPrivilegeVerified",
    "producer",
    "producerSha256",
    "powershellProbeSha256",
}


class ProductionAuthorityError(FlywayRehearsalError):
    """A production authority precondition or postcondition failed."""


class ProductionRuntimeError(ProductionAuthorityError):
    """A Flyway subprocess failed after the durable ARMED report was written."""

    def __init__(
        self,
        message: str,
        *,
        runtime_result: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.runtime_result = dict(runtime_result) if runtime_result else None


@dataclass(frozen=True, slots=True)
class WriterAbsenceEvidence:
    sha256: str
    checked_at: datetime
    server_uuid: str
    producer: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "checkedAt": self.checked_at.isoformat(),
            "serverUuidFingerprint": hashlib.sha256(
                self.server_uuid.encode("utf-8")
            ).hexdigest()[:16],
            "producer": self.producer,
            "serverWriterAbsent": True,
            "activeServerWriterProcessCount": 0,
            "activeDatabaseWriterSessionCount": 0,
        }


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    server_uuid: str
    target_fingerprint: str
    schema_sha256: str
    history: tuple[Mapping[str, Any], ...]
    protected: Mapping[str, TableDigest]
    recommendation_counts: Mapping[str, int]
    run_table_present: bool
    captured_at: str
    sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "targetFingerprint": self.target_fingerprint,
            "schemaSha256": self.schema_sha256,
            "history": list(self.history),
            "protectedTables": _digests_public(self.protected),
            "recommendationTableCounts": dict(self.recommendation_counts),
            "recommendationRunTablePresent": self.run_table_present,
            "capturedAt": self.captured_at,
            "snapshotSha256": self.sha256,
        }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def database_name_fingerprint(database: str) -> str:
    return hashlib.sha256(database.casefold().encode("utf-8")).hexdigest()[:16]


def _assert_exact_database(config: MysqlConfig) -> None:
    if config.database != EXPECTED_PRODUCTION_DATABASE:
        raise ProductionAuthorityError(
            "configured database is not the exact authorized production database name"
        )
    if config.query_items != EXPECTED_JDBC_QUERY_ITEMS:
        raise ProductionAuthorityError(
            "production JDBC query parameters are not the exact canonical allowlist"
        )


def _production_jdbc_url(config: MysqlConfig) -> str:
    _assert_exact_database(config)
    query = urlencode(config.query_items)
    result = urlunparse(
        ("mysql", f"{config.host}:{config.port}", f"/{config.database}", "", query, "")
    ).replace("mysql://", "jdbc:mysql://", 1)
    return result


def _regular_file_bytes(path: Path, *, label: str, maximum: int | None = None) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or path.expanduser().is_symlink():
        raise ProductionAuthorityError(f"{label} must be a regular non-symlink file")
    size = resolved.stat().st_size
    if maximum is not None and size > maximum:
        raise ProductionAuthorityError(f"{label} exceeds the maximum audited size")
    return resolved.read_bytes()


def load_verified_clone_report(
    path: Path = DEFAULT_SUCCESSFUL_CLONE_REPORT,
) -> tuple[Mapping[str, Any], str]:
    raw = _regular_file_bytes(path, label="successful clone report", maximum=1024 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_CLONE_REPORT_SHA256:
        raise ProductionAuthorityError("successful clone report hash is not pinned value")
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionAuthorityError("successful clone report is invalid JSON") from exc
    if not isinstance(report, dict):
        raise ProductionAuthorityError("successful clone report root is not an object")

    expected_fingerprint = database_name_fingerprint(EXPECTED_PRODUCTION_DATABASE)
    safety = report.get("safety")
    flyway = report.get("flyway")
    runtime = flyway.get("runtimeResult") if isinstance(flyway, dict) else None
    migration = report.get("migrationManifest")
    protected = report.get("protectedTables")
    counts = report.get("allTableCountsBeforeFlyway")
    required_invariants = {
        "recommendationMeasurementSessionNotNull": True,
        "requiredUniqueIndexes": [
            "uq_reason_review",
            "uq_shoe_recommendation_reason_type",
            "uq_shoe_recommendation_session_shoe",
        ],
        "duplicateEquivalentForeignKeyCount": 0,
        "duplicateEquivalentIndexCount": 0,
        "labMetricForeignKey": {
            "name": "fk_shoe_lab_metric_measurement",
            "deleteRule": "CASCADE",
        },
    }
    if (
        report.get("format") != "feetfit-flyway-clone-rehearsal"
        or report.get("version") != 1
        or report.get("status") != "PASS"
        or report.get("mode") != "EXECUTE_CLONE_ONLY"
        or report.get("sourceDatabaseFingerprint") != expected_fingerprint
        or report.get("sourceSchemaSha256") != PINNED_PRE_MIGRATION_SCHEMA_SHA256
        or not isinstance(migration, dict)
        or migration.get("sha256") != PINNED_MIGRATION_MANIFEST_SHA256
        or migration.get("bootJarSha256") != PINNED_BOOT_JAR_SHA256
        or tuple(
            (item.get("version"), item.get("name"), item.get("sha256"))
            for item in (migration.get("migrations") or ())
            if isinstance(item, dict)
        )
        != PINNED_MIGRATIONS
        or not isinstance(runtime, dict)
        or runtime.get("status") != "PASS"
        or runtime.get("stage") != "complete"
        or runtime.get("migrationsExecuted") != 6
        or runtime.get("pendingBefore") != 6
        or runtime.get("pendingAfter") != 0
        or runtime.get("currentVersion") != TARGET_VERSION
        or flyway.get("operationOrder") != ["repair", "validate", "migrate", "validate"]
        or flyway.get("target") != TARGET_VERSION
        or flyway.get("reconciliationInvariants") != required_invariants
        or not isinstance(safety, dict)
        or safety.get("productionDatabaseWrites") is not False
        or safety.get("generatedCloneOnly") is not True
        or safety.get("credentialsIncludedInReport") is not False
        or safety.get("cleanupImplemented") is not False
        or safety.get("cloneRetained") is not True
    ):
        raise ProductionAuthorityError("successful clone report contract mismatch")

    if tuple(flyway.get("historyBefore") or ()) != EXPECTED_INITIAL_HISTORY:
        raise ProductionAuthorityError("clone report initial history is not authoritative")
    _validate_exact_final_history(tuple(flyway.get("historyAfter") or ()))
    expected_public = _digests_public(PINNED_PROTECTED_DIGESTS)
    if (
        not isinstance(protected, dict)
        or protected.get("allCountsAndShaPreserved") is not True
        or any(
            protected.get(stage) != expected_public
            for stage in (
                "sourceSnapshot",
                "sourceAfterFlyway",
                "cloneBeforeFlyway",
                "cloneAfterFlyway",
            )
        )
    ):
        raise ProductionAuthorityError("clone report protected table evidence mismatch")
    if not isinstance(counts, dict) or any(
        counts.get(table_name) != 0 for table_name in RECOMMENDATION_TABLES_BEFORE
    ):
        raise ProductionAuthorityError("clone report recommendation tables were not empty")
    return report, digest


def _parse_utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProductionAuthorityError("writer evidence checkedAt must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProductionAuthorityError("writer evidence checkedAt is invalid") from exc
    if parsed.tzinfo is None:
        raise ProductionAuthorityError("writer evidence checkedAt must include timezone")
    return parsed.astimezone(timezone.utc)


def load_writer_absence_evidence(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_server_uuid: str | None = None,
    now: datetime | None = None,
) -> WriterAbsenceEvidence:
    raw = _regular_file_bytes(path, label="writer absence evidence", maximum=64 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and not secrets.compare_digest(
        digest, expected_sha256
    ):
        raise ProductionAuthorityError("writer absence evidence hash pin mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionAuthorityError("writer absence evidence is invalid JSON") from exc
    if not isinstance(document, dict) or set(document) != _WRITER_EVIDENCE_KEYS:
        raise ProductionAuthorityError("writer absence evidence contract mismatch")
    checked_at = _parse_utc_timestamp(document.get("checkedAt"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if checked_at - current > MAX_WRITER_EVIDENCE_FUTURE_SKEW:
        raise ProductionAuthorityError("writer absence evidence is future-dated")
    if current - checked_at > MAX_WRITER_EVIDENCE_AGE:
        raise ProductionAuthorityError("writer absence evidence is stale")
    server_uuid = document.get("serverUuid")
    producer = document.get("producer")
    if (
        document.get("format") != WRITER_EVIDENCE_FORMAT
        or document.get("version") != WRITER_EVIDENCE_VERSION
        or document.get("status") != "PASS"
        or document.get("sourceDatabase") != EXPECTED_PRODUCTION_DATABASE
        or document.get("databaseFingerprint")
        != database_name_fingerprint(EXPECTED_PRODUCTION_DATABASE)
        or document.get("serverWriterAbsent") is not True
        or document.get("activeServerWriterProcessIds") != []
        or document.get("activeDatabaseWriterSessionIds") != []
        or document.get("javaServerProcessIds") != []
        or document.get("uninspectableJavaProcessIds") != []
        or document.get("pythonWriterProcessIds") != []
        or document.get("uninspectablePythonProcessIds") != []
        or document.get("listener8080ProcessIds") != []
        or document.get("listener8081ProcessIds") != []
        or document.get("mysqlProcessPrivilegeVerified") is not True
        or document.get("producer") != WRITER_EVIDENCE_PRODUCER
        or document.get("producerSha256")
        != PINNED_WRITER_EVIDENCE_PRODUCER_SHA256
        or document.get("powershellProbeSha256") != POWERSHELL_PROBE_SHA256
        or not isinstance(server_uuid, str)
        or not re.fullmatch(r"[0-9a-fA-F-]{16,64}", server_uuid)
        or not isinstance(producer, str)
        or not producer.strip()
    ):
        raise ProductionAuthorityError("writer absence evidence did not prove quiescence")
    if expected_server_uuid is not None and server_uuid.casefold() != expected_server_uuid.casefold():
        raise ProductionAuthorityError("writer evidence targets a different MySQL server")
    return WriterAbsenceEvidence(digest, checked_at, server_uuid, producer.strip())


def _history_rows_from_cursor(cursor: Any, database: str) -> tuple[Mapping[str, Any], ...]:
    cursor.execute(
        f"SELECT {','.join(_quote_identifier(value) for value in _HISTORY_COLUMNS)} "
        f"FROM {_quote_identifier(database)}.`flyway_schema_history` "
        "ORDER BY installed_rank"
    )
    rows = cursor.fetchall()
    return tuple(
        {
            key: (
                value.isoformat(timespec="microseconds")
                if isinstance(value, datetime)
                else bool(value)
                if key == "success"
                else value
            )
            for key, value in zip(_HISTORY_COLUMNS, row, strict=True)
        }
        for row in rows
    )


def _target_fingerprint(config: MysqlConfig, server_uuid: str) -> str:
    material = (
        f"{config.host.casefold()}\0{config.port}\0{config.database}\0"
        f"{server_uuid.casefold()}\0"
        + "&".join(f"{key}={value}" for key, value in config.query_items)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def read_authority_snapshot(config: MysqlConfig) -> AuthoritySnapshot:
    """Read all authority inputs from one consistent, read-only transaction."""

    _assert_exact_database(config)
    connection = _connect(config, database=config.database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
            cursor.execute("SELECT DATABASE(), @@server_uuid")
            identity = cursor.fetchone()
            if identity is None or identity[0] != EXPECTED_PRODUCTION_DATABASE:
                raise ProductionAuthorityError("MySQL connection database identity mismatch")
            server_uuid = str(identity[1])

        shapes, _, _ = _read_source_shapes(connection, config.database)
        schema_sha256 = _schema_digest(shapes)
        protected: dict[str, TableDigest] = {}
        for table_name in PROTECTED_TABLES:
            shape = shapes[table_name]
            stream = connection.cursor(pymysql.cursors.SSCursor)
            digest = hashlib.sha256()
            count = 0
            try:
                stream.execute(_ordered_select_sql(config.database, shape))
                while True:
                    rows = stream.fetchmany(500)
                    if not rows:
                        break
                    for row in rows:
                        update_row_digest(digest, row)
                    count += len(rows)
            finally:
                stream.close()
            protected[table_name] = TableDigest(count, digest.hexdigest())

        with connection.cursor() as cursor:
            history = _history_rows_from_cursor(cursor, config.database)
            recommendation_counts: dict[str, int] = {}
            for table_name in RECOMMENDATION_TABLES_BEFORE:
                if table_name not in shapes:
                    raise ProductionAuthorityError(
                        f"required recommendation table is missing: {table_name}"
                    )
                cursor.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(config.database)}."
                    f"{_quote_identifier(table_name)}"
                )
                recommendation_counts[table_name] = int(cursor.fetchone()[0])
            run_table_present = RECOMMENDATION_RUN_TABLE in shapes
            if run_table_present:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(config.database)}."
                    f"{_quote_identifier(RECOMMENDATION_RUN_TABLE)}"
                )
                recommendation_counts[RECOMMENDATION_RUN_TABLE] = int(
                    cursor.fetchone()[0]
                )
        connection.rollback()
    finally:
        connection.close()

    target_fingerprint = _target_fingerprint(config, server_uuid)
    snapshot_payload = {
        "targetFingerprint": target_fingerprint,
        "schemaSha256": schema_sha256,
        "history": list(history),
        "protectedTables": _digests_public(protected),
        "recommendationTableCounts": recommendation_counts,
        "recommendationRunTablePresent": run_table_present,
    }
    captured_at = datetime.now(timezone.utc).isoformat()
    return AuthoritySnapshot(
        server_uuid=server_uuid,
        target_fingerprint=target_fingerprint,
        schema_sha256=schema_sha256,
        history=history,
        protected=protected,
        recommendation_counts=recommendation_counts,
        run_table_present=run_table_present,
        captured_at=captured_at,
        sha256=_canonical_sha256(snapshot_payload),
    )


def validate_pinned_manifest(manifest: MigrationManifest) -> None:
    if (
        manifest.sha256 != PINNED_MIGRATION_MANIFEST_SHA256
        or manifest.boot_jar_sha256 != PINNED_BOOT_JAR_SHA256
        or tuple(item.version for item in manifest.migrations)
        != EXPECTED_MIGRATION_VERSIONS
        or tuple(
            (item.version, item.name, item.sha256) for item in manifest.migrations
        )
        != PINNED_MIGRATIONS
    ):
        raise ProductionAuthorityError("Server migration manifest/boot jar is not pinned")


def validate_initial_snapshot(snapshot: AuthoritySnapshot) -> None:
    if snapshot.schema_sha256 != PINNED_PRE_MIGRATION_SCHEMA_SHA256:
        raise ProductionAuthorityError("production schema differs from rehearsed source")
    if snapshot.history != EXPECTED_INITIAL_HISTORY:
        raise ProductionAuthorityError(
            "initial Flyway history must be exactly V1/V2 success plus V3 failed"
        )
    if dict(snapshot.protected) != dict(PINNED_PROTECTED_DIGESTS):
        raise ProductionAuthorityError("protected production data differs from pinned state")
    if snapshot.run_table_present:
        raise ProductionAuthorityError("V8 recommendation run table already exists")
    if snapshot.recommendation_counts != {
        table_name: 0 for table_name in RECOMMENDATION_TABLES_BEFORE
    }:
        raise ProductionAuthorityError("recommendation tables must all be empty")


def _history_core(history: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {field: row.get(field) for field in _HISTORY_CORE_FIELDS} for row in history
    )


def _validate_exact_final_history(history: Sequence[Mapping[str, Any]]) -> None:
    if len(history) != 8 or _history_core(history) != EXPECTED_FINAL_HISTORY_CORE:
        raise ProductionAuthorityError("post-Flyway history is not exactly V1 through V8")
    if any(row.get("version") is None for row in history):
        raise ProductionAuthorityError("repeatable/versionless Flyway history is forbidden")


def production_confirmation_token(
    *,
    snapshot: AuthoritySnapshot,
    writer_evidence: WriterAbsenceEvidence,
    clone_report_sha256: str,
    manifest: MigrationManifest,
) -> str:
    payload = {
        "domain": CONFIRMATION_DOMAIN,
        "database": EXPECTED_PRODUCTION_DATABASE,
        "targetFingerprint": snapshot.target_fingerprint,
        "snapshotSha256": snapshot.sha256,
        "manifestSha256": manifest.sha256,
        "bootJarSha256": manifest.boot_jar_sha256,
        "cloneReportSha256": clone_report_sha256,
        "writerEvidenceSha256": writer_evidence.sha256,
        "writerEvidenceCheckedAt": writer_evidence.checked_at.isoformat(),
        "targetVersion": TARGET_VERSION,
        "operationOrder": ["repair", "validate", "migrate", "validate"],
        "maintenanceWindowAuthority": MAINTENANCE_WINDOW_AUTHORITY,
    }
    return _canonical_sha256(payload)


def build_production_plan(
    *,
    config: MysqlConfig,
    manifest: MigrationManifest,
    clone_report_path: Path = DEFAULT_SUCCESSFUL_CLONE_REPORT,
    writer_evidence_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    _assert_exact_database(config)
    validate_pinned_manifest(manifest)
    _, clone_report_sha256 = load_verified_clone_report(clone_report_path)
    snapshot = read_authority_snapshot(config)
    validate_initial_snapshot(snapshot)
    evidence: WriterAbsenceEvidence | None = None
    token: str | None = None
    if writer_evidence_path is not None:
        evidence = load_writer_absence_evidence(
            writer_evidence_path,
            expected_server_uuid=snapshot.server_uuid,
            now=now,
        )
        if REMOTE_WRITER_FENCE_AUTHORIZED:
            token = production_confirmation_token(
                snapshot=snapshot,
                writer_evidence=evidence,
                clone_report_sha256=clone_report_sha256,
                manifest=manifest,
            )
    return {
        "format": FORMAT,
        "version": VERSION,
        "mode": "PLAN_ONLY",
        "status": "PASS",
        "targetDatabase": EXPECTED_PRODUCTION_DATABASE,
        "targetFingerprint": snapshot.target_fingerprint,
        "migrationManifestSha256": manifest.sha256,
        "bootJarSha256": manifest.boot_jar_sha256,
        "successfulCloneReportSha256": clone_report_sha256,
        "initialAuthoritySnapshot": snapshot.public_dict(),
        "writerAbsenceEvidence": evidence.public_dict() if evidence else None,
        "executeReady": evidence is not None and REMOTE_WRITER_FENCE_AUTHORIZED,
        "confirmationToken": token,
        "requiredExecutePins": {
            "migrationManifestSha256": PINNED_MIGRATION_MANIFEST_SHA256,
            "bootJarSha256": PINNED_BOOT_JAR_SHA256,
            "successfulCloneReportSha256": PINNED_CLONE_REPORT_SHA256,
            "writerEvidenceSha256": evidence.sha256 if evidence else None,
        },
        "operationOrder": ["repair", "validate", "migrate", "validate"],
        "targetVersion": TARGET_VERSION,
        "requiredOperationalAuthority": {
            "value": MAINTENANCE_WINDOW_AUTHORITY,
            "meaning": (
                "this host is the sole Server/ingestion writer host and writer "
                "restart is inhibited for the full runtime"
            ),
            "technicallyCompleteFence": False,
            "externalNewConnectionRaceFullyPrevented": False,
            "failurePolicy": "INDETERMINATE_MANUAL_INSPECTION_NO_RETRY_OR_ROLLBACK",
            "currentAuthorityStatus": (
                "AUTHORIZED"
                if REMOTE_WRITER_FENCE_AUTHORIZED
                else "NO_GO_REMOTE_RDS_WRITER_RESTART_FENCE_ABSENT"
            ),
        },
        "safety": {
            "productionWritesPermitted": False,
            "executeRequiresExplicitFlag": True,
            "automaticRetry": False,
            "automaticRollback": False,
            "httpOrServerApiCalls": False,
        },
    }


class ProductionMigrationLock:
    """One executor lock; it does not replace the external writer-stop proof."""

    def __init__(self, config: MysqlConfig) -> None:
        self._config = config
        self._connection: Any | None = None
        self._server_uuid: str | None = None
        self._connection_id: int | None = None

    @property
    def server_uuid(self) -> str:
        if self._server_uuid is None:
            raise ProductionAuthorityError("production migration lock is not held")
        return self._server_uuid

    @property
    def connection_id(self) -> int:
        if self._connection_id is None:
            raise ProductionAuthorityError("production migration lock is not held")
        return self._connection_id

    def __enter__(self) -> "ProductionMigrationLock":
        _assert_exact_database(self._config)
        connection = _connect(
            self._config, database=self._config.database, autocommit=True
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT GET_LOCK(%s, 0)", (MIGRATION_LOCK_NAME,))
                row = cursor.fetchone()
            if row is None or int(row[0]) != 1:
                raise ProductionAuthorityError("another production migration holds the lock")
            with connection.cursor() as cursor:
                cursor.execute("SELECT DATABASE(), @@server_uuid, CONNECTION_ID()")
                identity = cursor.fetchone()
            if identity is None or identity[0] != EXPECTED_PRODUCTION_DATABASE:
                raise ProductionAuthorityError(
                    "production migration lock connection identity mismatch"
                )
        except Exception:
            connection.close()
            raise
        self._connection = connection
        self._server_uuid = str(identity[1])
        self._connection_id = int(identity[2])
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        connection = self._connection
        self._connection = None
        self._server_uuid = None
        self._connection_id = None
        if connection is None:
            return
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT RELEASE_LOCK(%s)", (MIGRATION_LOCK_NAME,))
        finally:
            connection.close()


def _copy_and_pin_boot_jar(source: Path, destination: Path) -> Path:
    resolved = source.expanduser().resolve()
    if not resolved.is_file() or source.expanduser().is_symlink():
        raise ProductionAuthorityError("Server boot jar must be a regular non-symlink file")
    if sha256_file(resolved) != PINNED_BOOT_JAR_SHA256:
        raise ProductionAuthorityError("Server boot jar changed after manifest validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, destination)
    if sha256_file(destination) != PINNED_BOOT_JAR_SHA256:
        raise ProductionAuthorityError("staged Server boot jar hash mismatch")
    return destination


def _redact_production_output(value: str, config: MysqlConfig) -> str:
    redacted = value
    for secret in (config.password, config.username, _production_jdbc_url(config)):
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"jdbc:mysql://\S+", "[JDBC_REDACTED]", redacted)
    redacted = "".join(
        character
        for character in redacted
        if character in "\n\r\t" or ord(character) >= 32
    )
    return redacted[-4000:].strip()


def run_server_runtime_production_flyway(
    *,
    config: MysqlConfig,
    server_uuid: str,
    staged_boot_jar: Path,
    java_home: Path | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    _assert_exact_database(config)
    source_bytes = _regular_file_bytes(
        DEFAULT_JAVA_SOURCE, label="production Flyway Java helper", maximum=128 * 1024
    )
    if hashlib.sha256(source_bytes).hexdigest() != PINNED_JAVA_HELPER_SHA256:
        raise ProductionAuthorityError("production Flyway Java helper hash mismatch")
    jdk = resolve_java_home(java_home)
    javac = jdk / "bin" / ("javac.exe" if os.name == "nt" else "javac")
    java = jdk / "bin" / ("java.exe" if os.name == "nt" else "java")

    with tempfile.TemporaryDirectory(prefix="feetfit-production-flyway-runtime-") as name:
        temp = Path(name)
        source = temp / "FeetfitFlywayProductionApply.java"
        source.write_bytes(source_bytes)
        classes, libraries = _safe_extract_boot_runtime(
            staged_boot_jar, temp / "runtime"
        )
        helper_classes = temp / "helper-classes"
        helper_classes.mkdir()
        runtime_classpath = os.pathsep.join((str(classes), str(libraries / "*")))
        compile_result = subprocess.run(
            [
                str(javac),
                "-encoding",
                "UTF-8",
                "-classpath",
                runtime_classpath,
                "-d",
                str(helper_classes),
                str(source),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if compile_result.returncode != 0:
            diagnostic = _redact_production_output(
                compile_result.stdout + "\n" + compile_result.stderr, config
            )
            raise ProductionRuntimeError(
                "production Flyway helper compilation failed; "
                f"diagnostic={diagnostic}",
                runtime_result={
                    "status": "FAIL",
                    "stage": "helper_compilation",
                    "errorClass": "JavaCompilationFailure",
                },
            )

        result_path = temp / "flyway-result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "FEETFIT_PRODUCTION_JDBC_URL": _production_jdbc_url(config),
                "FEETFIT_PRODUCTION_DB_USERNAME": config.username,
                "FEETFIT_PRODUCTION_DB_PASSWORD": config.password,
                "FEETFIT_PRODUCTION_EXPECTED_DATABASE": EXPECTED_PRODUCTION_DATABASE,
                "FEETFIT_PRODUCTION_EXPECTED_SERVER_UUID": server_uuid,
                "FEETFIT_PRODUCTION_RESULT_PATH": str(result_path),
            }
        )
        classpath = os.pathsep.join((str(helper_classes), runtime_classpath))
        try:
            run_result = subprocess.run(
                [str(java), "-classpath", classpath, "FeetfitFlywayProductionApply"],
                cwd=str(PROJECT_ROOT),
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProductionRuntimeError(
                "production Flyway timed out; migration state is indeterminate and "
                "requires manual inspection",
                runtime_result={
                    "status": "INDETERMINATE",
                    "stage": "subprocess_timeout",
                    "errorClass": "TimeoutExpired",
                },
            ) from exc
        if not result_path.is_file():
            diagnostic = _redact_production_output(
                run_result.stdout + "\n" + run_result.stderr, config
            )
            raise ProductionRuntimeError(
                "production Flyway returned no result; state is indeterminate; "
                f"diagnostic={diagnostic}",
                runtime_result={
                    "status": "INDETERMINATE",
                    "stage": "result_missing",
                    "errorClass": "MissingResult",
                },
            )
        try:
            document = json.loads(result_path.read_text("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductionRuntimeError(
                "production Flyway result is invalid; state is indeterminate",
                runtime_result={
                    "status": "INDETERMINATE",
                    "stage": "result_decode",
                    "errorClass": type(exc).__name__,
                },
            ) from exc
        expected_keys = {
            "status",
            "stage",
            "migrationsExecuted",
            "pendingBefore",
            "pendingAfter",
            "currentVersion",
            "errorClass",
        }
        if (
            run_result.returncode != 0
            or set(document) != expected_keys
            or document.get("status") != "PASS"
            or document.get("stage") != "complete"
            or document.get("migrationsExecuted") != 6
            or document.get("pendingBefore") != 6
            or document.get("pendingAfter") != 0
            or document.get("currentVersion") != TARGET_VERSION
            or document.get("errorClass") != ""
        ):
            diagnostic = _redact_production_output(
                run_result.stdout + "\n" + run_result.stderr, config
            )
            safe_runtime_result = {
                key: document.get(key)
                for key in expected_keys
                if key in document
            }
            raise ProductionRuntimeError(
                "production Flyway runtime contract failed; state is indeterminate; "
                f"diagnostic={diagnostic}",
                runtime_result=safe_runtime_result,
            )
        return document


def validate_post_snapshot(
    before: AuthoritySnapshot,
    after: AuthoritySnapshot,
) -> None:
    validate_post_flyway_history(before.history, after.history)
    _validate_exact_final_history(after.history)
    if dict(after.protected) != dict(PINNED_PROTECTED_DIGESTS):
        raise ProductionAuthorityError("protected production data changed during Flyway")
    if not after.run_table_present or after.recommendation_counts != {
        table_name: 0 for table_name in RECOMMENDATION_TABLES_AFTER
    }:
        raise ProductionAuthorityError("post-Flyway recommendation tables are not empty")
    if before.target_fingerprint != after.target_fingerprint:
        raise ProductionAuthorityError("production target identity changed during Flyway")


def read_identity_checked_history(
    config: MysqlConfig, *, expected_server_uuid: str
) -> tuple[Mapping[str, Any], ...]:
    """Best-effort history reader that cannot silently cross DB/server identity."""

    connection = _connect(config, database=config.database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute("SELECT DATABASE(), @@server_uuid")
            identity = cursor.fetchone()
            if (
                identity is None
                or identity[0] != EXPECTED_PRODUCTION_DATABASE
                or str(identity[1]).casefold() != expected_server_uuid.casefold()
            ):
                raise ProductionAuthorityError(
                    "failure audit history connection identity mismatch"
                )
            history = _history_rows_from_cursor(cursor, config.database)
            connection.rollback()
            return history
    finally:
        connection.close()


def best_effort_failure_audit(
    config: MysqlConfig, *, expected_server_uuid: str
) -> dict[str, Any]:
    """Perform each post-failure read once; never retry or mutate/compensate."""

    diagnostics: dict[str, Any] = {"attempted": True, "automaticRetry": False}
    try:
        snapshot = read_authority_snapshot(config)
        if snapshot.server_uuid.casefold() != expected_server_uuid.casefold():
            raise ProductionAuthorityError("failure snapshot server identity mismatch")
        diagnostics["authoritySnapshot"] = {
            "status": "READ_SUCCESS",
            "value": snapshot.public_dict(),
        }
    except BaseException as exc:
        diagnostics["authoritySnapshot"] = {
            "status": "READ_FAILED",
            "failureClass": type(exc).__name__,
        }
    try:
        history = read_identity_checked_history(
            config, expected_server_uuid=expected_server_uuid
        )
        diagnostics["flywayHistory"] = {
            "status": "READ_SUCCESS",
            "value": list(history),
        }
    except BaseException as exc:
        diagnostics["flywayHistory"] = {
            "status": "READ_FAILED",
            "failureClass": type(exc).__name__,
        }
    try:
        invariants = read_production_reconciliation_invariants(
            config,
            expected_server_uuid=expected_server_uuid,
        )
        diagnostics["reconciliationInvariants"] = {
            "status": "READ_SUCCESS",
            "value": invariants,
        }
    except BaseException as exc:
        diagnostics["reconciliationInvariants"] = {
            "status": "READ_FAILED",
            "failureClass": type(exc).__name__,
        }
    return diagnostics


def _attempt_name(now: datetime | None = None) -> str:
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return "production-apply-" + instant.strftime("%Y%m%dT%H%M%S%fZ")


def assert_no_unresolved_or_replayed_attempt(
    artifact_root: Path, *, confirmation: str
) -> None:
    """Refuse unresolved attempts and reuse of an already recorded authority token."""

    resolved = artifact_root.expanduser().resolve()
    if not resolved.exists():
        return
    if not resolved.is_dir() or artifact_root.expanduser().is_symlink():
        raise ProductionAuthorityError("production artifact root is not a safe directory")
    token_sha256 = hashlib.sha256(confirmation.encode("ascii")).hexdigest()
    for report_path in sorted(resolved.glob("production-apply-*/production-apply-report.json")):
        raw = _regular_file_bytes(
            report_path, label="prior production attempt report", maximum=1024 * 1024
        )
        try:
            report = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductionAuthorityError(
                "prior production attempt report is unreadable; audited clearance required"
            ) from exc
        if not isinstance(report, dict) or report.get("format") != FORMAT:
            raise ProductionAuthorityError(
                "prior production attempt report contract mismatch; audited clearance required"
            )
        status = report.get("status")
        if status in {"ARMED", "INDETERMINATE"}:
            raise ProductionAuthorityError(
                "unresolved production attempt exists; explicit audited clearance required"
            )
        if status != "PASS":
            raise ProductionAuthorityError(
                "unknown prior production attempt state; audited clearance required"
            )
        if report.get("authorityConfirmationSha256") == token_sha256:
            raise ProductionAuthorityError("production authority token replay is forbidden")


def execute_production_migration(
    *,
    config: MysqlConfig,
    manifest: MigrationManifest,
    expected_manifest_sha256: str,
    expected_boot_jar_sha256: str,
    expected_clone_report_sha256: str,
    writer_evidence_path: Path,
    expected_writer_evidence_sha256: str,
    confirmation: str,
    maintenance_window_authority: str,
    boot_jar: Path = DEFAULT_SERVER_BOOT_JAR,
    clone_report_path: Path = DEFAULT_SUCCESSFUL_CLONE_REPORT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    java_home: Path | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], Path]:
    """Perform the one authorized migration after every exact pin is proven."""

    _assert_exact_database(config)
    validate_pinned_manifest(manifest)
    if not REMOTE_WRITER_FENCE_AUTHORIZED:
        raise ProductionAuthorityError(
            "NO-GO: remote RDS writer restart fence evidence is absent; production "
            "execution is disabled"
        )
    if (
        expected_manifest_sha256 != PINNED_MIGRATION_MANIFEST_SHA256
        or expected_boot_jar_sha256 != PINNED_BOOT_JAR_SHA256
        or expected_clone_report_sha256 != PINNED_CLONE_REPORT_SHA256
        or not re.fullmatch(r"[0-9a-f]{64}", expected_writer_evidence_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", confirmation)
        or maintenance_window_authority != MAINTENANCE_WINDOW_AUTHORITY
    ):
        raise ProductionAuthorityError("execute-time authority pins are incomplete or wrong")

    _, clone_report_sha256 = load_verified_clone_report(clone_report_path)
    if clone_report_sha256 != expected_clone_report_sha256:
        raise ProductionAuthorityError("execute clone report pin mismatch")

    assert_no_unresolved_or_replayed_attempt(
        artifact_root,
        confirmation=confirmation,
    )
    output_dir = artifact_root.expanduser().resolve() / _attempt_name(now)
    if output_dir.exists():
        raise ProductionAuthorityError("production migration artifact already exists")
    output_dir.mkdir(parents=True)
    report_path = output_dir / "production-apply-report.json"

    with tempfile.TemporaryDirectory(prefix="feetfit-production-boot-") as temp_name:
        staged_boot = _copy_and_pin_boot_jar(
            boot_jar, Path(temp_name) / "server-runtime-pinned.jar"
        )
        with ProductionMigrationLock(config) as migration_lock:
            snapshot = read_authority_snapshot(config)
            validate_initial_snapshot(snapshot)
            if snapshot.server_uuid.casefold() != migration_lock.server_uuid.casefold():
                raise ProductionAuthorityError(
                    "migration lock and authority snapshot target different servers"
                )
            evidence = load_writer_absence_evidence(
                writer_evidence_path,
                expected_sha256=expected_writer_evidence_sha256,
                expected_server_uuid=snapshot.server_uuid,
            )
            expected_confirmation = production_confirmation_token(
                snapshot=snapshot,
                writer_evidence=evidence,
                clone_report_sha256=clone_report_sha256,
                manifest=manifest,
            )
            if not secrets.compare_digest(confirmation, expected_confirmation):
                raise ProductionAuthorityError(
                    "explicit production confirmation does not match current authority state"
                )

            # Re-read state and re-check evidence immediately before crossing
            # the only mutation boundary.  Any drift invalidates the token.
            immediate = read_authority_snapshot(config)
            validate_initial_snapshot(immediate)
            if immediate.server_uuid.casefold() != migration_lock.server_uuid.casefold():
                raise ProductionAuthorityError(
                    "migration lock and immediate snapshot target different servers"
                )
            if immediate.sha256 != snapshot.sha256:
                raise ProductionAuthorityError("production state changed during preflight")
            evidence = load_writer_absence_evidence(
                writer_evidence_path,
                expected_sha256=expected_writer_evidence_sha256,
                expected_server_uuid=immediate.server_uuid,
            )
            if not secrets.compare_digest(
                confirmation,
                production_confirmation_token(
                    snapshot=immediate,
                    writer_evidence=evidence,
                    clone_report_sha256=clone_report_sha256,
                    manifest=manifest,
                ),
            ):
                raise ProductionAuthorityError("production confirmation expired or drifted")

            live_writer_state = collect_writer_absence_evidence(
                config,
                excluded_mysql_session_ids=(migration_lock.connection_id,),
            )
            if (
                live_writer_state.get("status") != "PASS"
                or live_writer_state.get("serverWriterAbsent") is not True
                or str(live_writer_state.get("serverUuid", "")).casefold()
                != immediate.server_uuid.casefold()
                or live_writer_state.get("producerSha256")
                != PINNED_WRITER_EVIDENCE_PRODUCER_SHA256
                or live_writer_state.get("powershellProbeSha256")
                != POWERSHELL_PROBE_SHA256
            ):
                raise ProductionAuthorityError(
                    "immediate live writer probe did not prove a quiescent target"
                )

            armed_report = {
                "format": FORMAT,
                "version": VERSION,
                "status": "ARMED",
                "mode": "EXECUTE_PRODUCTION",
                "targetDatabase": EXPECTED_PRODUCTION_DATABASE,
                "targetFingerprint": immediate.target_fingerprint,
                "migrationManifestSha256": manifest.sha256,
                "bootJarSha256": manifest.boot_jar_sha256,
                "successfulCloneReportSha256": clone_report_sha256,
                "authorityConfirmationSha256": hashlib.sha256(
                    confirmation.encode("ascii")
                ).hexdigest(),
                "writerAbsenceEvidence": evidence.public_dict(),
                "immediateLiveWriterProbe": {
                    "status": live_writer_state["status"],
                    "checkedAt": live_writer_state["checkedAt"],
                    "serverWriterAbsent": live_writer_state["serverWriterAbsent"],
                    "activeServerWriterProcessCount": len(
                        live_writer_state["activeServerWriterProcessIds"]
                    ),
                    "activeDatabaseWriterSessionCount": len(
                        live_writer_state["activeDatabaseWriterSessionIds"]
                    ),
                    "mysqlProcessPrivilegeVerified": live_writer_state[
                        "mysqlProcessPrivilegeVerified"
                    ],
                },
                "operationalMaintenanceWindowAuthority": {
                    "value": maintenance_window_authority,
                    "soleWriterHostConfirmedByOperator": True,
                    "writerRestartInhibitedByOperator": True,
                    "technicallyCompleteFence": False,
                    "externalNewConnectionRaceFullyPrevented": False,
                    "requiredThroughoutRuntime": True,
                    "failurePolicy": (
                        "INDETERMINATE_MANUAL_INSPECTION_NO_RETRY_OR_ROLLBACK"
                    ),
                },
                "authoritySnapshotBefore": immediate.public_dict(),
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "safety": {
                    "automaticRetry": False,
                    "automaticRollback": False,
                    "automaticResume": False,
                    "httpOrServerApiCalls": False,
                    "failureRequiresManualInspection": True,
                },
            }
            _atomic_json(report_path, armed_report)

            try:
                runtime = run_server_runtime_production_flyway(
                    config=config,
                    server_uuid=immediate.server_uuid,
                    staged_boot_jar=staged_boot,
                    java_home=java_home,
                )
                post_runtime_writer_state = collect_writer_absence_evidence(
                    config,
                    excluded_mysql_session_ids=(migration_lock.connection_id,),
                )
                if (
                    post_runtime_writer_state.get("status") != "PASS"
                    or str(post_runtime_writer_state.get("serverUuid", "")).casefold()
                    != immediate.server_uuid.casefold()
                    or post_runtime_writer_state.get("producerSha256")
                    != PINNED_WRITER_EVIDENCE_PRODUCER_SHA256
                    or post_runtime_writer_state.get("powershellProbeSha256")
                    != POWERSHELL_PROBE_SHA256
                ):
                    raise ProductionAuthorityError(
                        "writer appeared during/after runtime; state is indeterminate"
                    )
                after = read_authority_snapshot(config)
                validate_post_snapshot(immediate, after)
                invariants = read_production_reconciliation_invariants(
                    config,
                    expected_server_uuid=immediate.server_uuid,
                )
            except BaseException as exc:
                failure_report = dict(armed_report)
                failure_report.update(
                    {
                        "status": "INDETERMINATE",
                        "failedAt": datetime.now(timezone.utc).isoformat(),
                        "failureClass": type(exc).__name__,
                        "flywayFailureRuntimeResult": (
                            exc.runtime_result
                            if isinstance(exc, ProductionRuntimeError)
                            else None
                        ),
                        "bestEffortPostFailureAudit": best_effort_failure_audit(
                            config,
                            expected_server_uuid=immediate.server_uuid,
                        ),
                        "mutationBoundaryMayHaveBeenCrossed": True,
                        "manualInspectionRequired": True,
                    }
                )
                try:
                    _atomic_json(report_path, failure_report)
                except BaseException:
                    # The already-fsynced ARMED report remains the durable
                    # conservative state if even the failure update cannot be
                    # persisted.  Never mask the migration failure or retry.
                    pass
                raise

            completed = dict(armed_report)
            completed.update(
                {
                    "status": "PASS",
                    "completedAt": datetime.now(timezone.utc).isoformat(),
                    "flywayRuntimeResult": runtime,
                    "authoritySnapshotAfter": after.public_dict(),
                    "reconciliationInvariants": invariants,
                    "postRuntimeWriterProbe": {
                        "status": post_runtime_writer_state["status"],
                        "checkedAt": post_runtime_writer_state["checkedAt"],
                        "serverWriterAbsent": post_runtime_writer_state[
                            "serverWriterAbsent"
                        ],
                        "activeServerWriterProcessCount": len(
                            post_runtime_writer_state[
                                "activeServerWriterProcessIds"
                            ]
                        ),
                        "activeDatabaseWriterSessionCount": len(
                            post_runtime_writer_state[
                                "activeDatabaseWriterSessionIds"
                            ]
                        ),
                    },
                    "manualInspectionRequired": False,
                }
            )
            try:
                _atomic_json(report_path, completed)
            except BaseException as exc:
                indeterminate = dict(armed_report)
                indeterminate.update(
                    {
                        "status": "INDETERMINATE",
                        "failedAt": datetime.now(timezone.utc).isoformat(),
                        "failureClass": type(exc).__name__,
                        "completedPostAuditButPassReportWriteFailed": True,
                        "bestEffortPostFailureAudit": best_effort_failure_audit(
                            config,
                            expected_server_uuid=immediate.server_uuid,
                        ),
                        "mutationBoundaryMayHaveBeenCrossed": True,
                        "manualInspectionRequired": True,
                    }
                )
                try:
                    _atomic_json(report_path, indeterminate)
                except BaseException:
                    # ARMED is already durable and is intentionally never
                    # replaced with a false PASS on persistence uncertainty.
                    pass
                raise ProductionAuthorityError(
                    "post-audit PASS report could not be persisted; durable state "
                    "is ARMED/INDETERMINATE and manual inspection is required"
                ) from exc
            return completed, report_path


def load_current_pinned_manifest() -> MigrationManifest:
    return load_migration_manifest(
        server_root=DEFAULT_SERVER_ROOT,
        boot_jar=DEFAULT_SERVER_BOOT_JAR,
    )
