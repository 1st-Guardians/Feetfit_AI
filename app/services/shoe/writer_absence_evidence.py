"""Audited, read-only producer for production writer-absence evidence.

The Windows probe checks FeetFit Java processes and listeners on ports 8080
and 8081.  The MySQL probe verifies exact database/server identity, requires
global PROCESS visibility, and rejects every non-current active session for the
FeetFit schema as well as every non-current open InnoDB transaction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.core.config import PROJECT_ROOT
from app.services.shoe.flyway_clone_rehearsal import (
    FlywayRehearsalError,
    MysqlConfig,
    _atomic_json,
    _connect,
    sha256_file,
)


FORMAT = "feetfit-server-writer-absence-evidence"
VERSION = 1
PRODUCER = "feetfit-audited-writer-absence-producer-v1"
EXPECTED_DATABASE = "feetfit"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "writer-absence-evidence"

POWERSHELL_PROBE = r"""
$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$javaProcesses = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @('java.exe', 'javaw.exe') }
)
$pythonProcesses = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @('python.exe', 'pythonw.exe') }
)
$serverJava = @(
    $javaProcesses |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
            $_.CommandLine -match '(?i)(FeetFit_Server|com[.]feetfit[.]server|FeetFitServerApplication)'
        } |
        ForEach-Object { [int64]$_.ProcessId } |
        Sort-Object -Unique
)
$uninspectableJava = @(
    $javaProcesses |
        Where-Object { [string]::IsNullOrWhiteSpace($_.CommandLine) } |
        ForEach-Object { [int64]$_.ProcessId } |
        Sort-Object -Unique
)
$writerPython = @(
    $pythonProcesses |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
            $_.CommandLine -match '(?i)(shoe_crawler|bidirectional_selection_pipeline|verified_ingestion|ingestion[_-]?runner)'
        } |
        ForEach-Object { [int64]$_.ProcessId } |
        Sort-Object -Unique
)
$uninspectablePython = @(
    $pythonProcesses |
        Where-Object { [string]::IsNullOrWhiteSpace($_.CommandLine) } |
        ForEach-Object { [int64]$_.ProcessId } |
        Sort-Object -Unique
)
$allListeners = @(
    Get-NetTCPConnection -State Listen -ErrorAction Stop
)
$listener8080 = @(
    $allListeners |
        Where-Object { $_.LocalPort -eq 8080 } |
        ForEach-Object { [int64]$_.OwningProcess } |
        Sort-Object -Unique
)
$listener8081 = @(
    $allListeners |
        Where-Object { $_.LocalPort -eq 8081 } |
        ForEach-Object { [int64]$_.OwningProcess } |
        Sort-Object -Unique
)
$active = @(
    @($serverJava) + @($uninspectableJava) + @($writerPython) +
        @($uninspectablePython) + @($listener8080) + @($listener8081) |
        Sort-Object -Unique
)
[ordered]@{
    javaServerProcessIds = @($serverJava)
    uninspectableJavaProcessIds = @($uninspectableJava)
    pythonWriterProcessIds = @($writerPython)
    uninspectablePythonProcessIds = @($uninspectablePython)
    listener8080ProcessIds = @($listener8080)
    listener8081ProcessIds = @($listener8081)
    activeServerWriterProcessIds = @($active)
} | ConvertTo-Json -Compress
""".strip()

POWERSHELL_PROBE_SHA256 = hashlib.sha256(
    POWERSHELL_PROBE.encode("utf-8")
).hexdigest()


class WriterEvidenceProducerError(FlywayRehearsalError):
    """A writer probe could not produce authoritative evidence."""


@dataclass(frozen=True, slots=True)
class WindowsWriterState:
    java_server_process_ids: tuple[int, ...]
    uninspectable_java_process_ids: tuple[int, ...]
    python_writer_process_ids: tuple[int, ...]
    uninspectable_python_process_ids: tuple[int, ...]
    listener_8080_process_ids: tuple[int, ...]
    listener_8081_process_ids: tuple[int, ...]
    active_server_writer_process_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MysqlWriterState:
    server_uuid: str
    active_database_writer_session_ids: tuple[int, ...]
    process_privilege_verified: bool


def _strict_integer_ids(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise WriterEvidenceProducerError(f"PowerShell {label} contract mismatch")
    result = tuple(sorted(set(value)))
    if len(result) != len(value) or list(result) != value:
        raise WriterEvidenceProducerError(f"PowerShell {label} is not unique/sorted")
    return result


def _powershell_executable() -> Path:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    candidates = [
        system_root
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe",
    ]
    discovered = shutil.which("powershell.exe") or shutil.which("powershell")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise WriterEvidenceProducerError("Windows PowerShell is required for writer probe")


def read_windows_writer_state(timeout_seconds: int = 30) -> WindowsWriterState:
    try:
        result = subprocess.run(
            [
                str(_powershell_executable()),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                POWERSHELL_PROBE,
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise WriterEvidenceProducerError("Windows writer probe timed out") from exc
    if result.returncode != 0:
        raise WriterEvidenceProducerError("Windows writer probe failed")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WriterEvidenceProducerError(
            "Windows writer probe returned invalid JSON"
        ) from exc
    expected_keys = {
        "javaServerProcessIds",
        "uninspectableJavaProcessIds",
        "pythonWriterProcessIds",
        "uninspectablePythonProcessIds",
        "listener8080ProcessIds",
        "listener8081ProcessIds",
        "activeServerWriterProcessIds",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise WriterEvidenceProducerError("Windows writer probe contract mismatch")
    java_server = _strict_integer_ids(
        document["javaServerProcessIds"], "javaServerProcessIds"
    )
    uninspectable = _strict_integer_ids(
        document["uninspectableJavaProcessIds"], "uninspectableJavaProcessIds"
    )
    writer_python = _strict_integer_ids(
        document["pythonWriterProcessIds"], "pythonWriterProcessIds"
    )
    uninspectable_python = _strict_integer_ids(
        document["uninspectablePythonProcessIds"],
        "uninspectablePythonProcessIds",
    )
    listener_8080 = _strict_integer_ids(
        document["listener8080ProcessIds"], "listener8080ProcessIds"
    )
    listener_8081 = _strict_integer_ids(
        document["listener8081ProcessIds"], "listener8081ProcessIds"
    )
    active = _strict_integer_ids(
        document["activeServerWriterProcessIds"], "activeServerWriterProcessIds"
    )
    expected_active = tuple(
        sorted(
            set(
                java_server
                + uninspectable
                + writer_python
                + uninspectable_python
                + listener_8080
                + listener_8081
            )
        )
    )
    if active != expected_active:
        raise WriterEvidenceProducerError("Windows active writer union mismatch")
    return WindowsWriterState(
        java_server,
        uninspectable,
        writer_python,
        uninspectable_python,
        listener_8080,
        listener_8081,
        active,
    )


def _has_global_process_privilege(grants: tuple[str, ...]) -> bool:
    return any(
        re.search(r"GRANT\s+.*\bPROCESS\b.*\s+ON\s+[`*]+[.][`*]+", grant, re.I)
        or re.search(
            r"GRANT\s+ALL\s+PRIVILEGES\s+ON\s+[`*]+[.][`*]+", grant, re.I
        )
        for grant in grants
    )


def read_mysql_writer_state(
    config: MysqlConfig,
    *,
    excluded_session_ids: tuple[int, ...] = (),
) -> MysqlWriterState:
    if config.database != EXPECTED_DATABASE:
        raise WriterEvidenceProducerError("writer probe database name is not authorized")
    connection = _connect(config, database=config.database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute("SELECT DATABASE(), @@server_uuid, CONNECTION_ID()")
            identity = cursor.fetchone()
            if identity is None or identity[0] != EXPECTED_DATABASE:
                raise WriterEvidenceProducerError("writer probe MySQL identity mismatch")
            server_uuid = str(identity[1])
            connection_id = int(identity[2])
            cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
            grants = tuple(str(row[0]) for row in cursor.fetchall())
            if not _has_global_process_privilege(grants):
                raise WriterEvidenceProducerError(
                    "writer probe requires global PROCESS visibility"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in excluded_session_ids
            ):
                raise WriterEvidenceProducerError("invalid excluded MySQL session id")
            exclusion_sql = ""
            parameters: list[Any] = [connection_id, EXPECTED_DATABASE]
            if excluded_session_ids:
                placeholders = ",".join("%s" for _ in excluded_session_ids)
                exclusion_sql = f" AND p.ID NOT IN ({placeholders})"
                parameters.extend(excluded_session_ids)
            cursor.execute(
                """
                SELECT DISTINCT p.ID
                FROM information_schema.PROCESSLIST p
                LEFT JOIN information_schema.INNODB_TRX t
                  ON t.trx_mysql_thread_id = p.ID
                WHERE p.ID <> %s
                  AND (
                      p.DB = %s
                      OR t.trx_mysql_thread_id IS NOT NULL
                  )
                """
                + exclusion_sql
                + """
                ORDER BY p.ID
                """,
                tuple(parameters),
            )
            active_sessions = tuple(int(row[0]) for row in cursor.fetchall())
            connection.rollback()
    finally:
        connection.close()
    return MysqlWriterState(server_uuid, active_sessions, True)


def _database_fingerprint(database: str) -> str:
    return hashlib.sha256(database.casefold().encode("utf-8")).hexdigest()[:16]


def producer_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def collect_writer_absence_evidence(
    config: MysqlConfig,
    *,
    now: datetime | None = None,
    excluded_mysql_session_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    windows = read_windows_writer_state()
    mysql = read_mysql_writer_state(
        config, excluded_session_ids=excluded_mysql_session_ids
    )
    writer_absent = (
        not windows.active_server_writer_process_ids
        and not windows.uninspectable_java_process_ids
        and not windows.python_writer_process_ids
        and not windows.uninspectable_python_process_ids
        and not mysql.active_database_writer_session_ids
    )
    return {
        "format": FORMAT,
        "version": VERSION,
        "status": "PASS" if writer_absent else "FAIL",
        "checkedAt": (now or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        ).isoformat(),
        "sourceDatabase": EXPECTED_DATABASE,
        "databaseFingerprint": _database_fingerprint(EXPECTED_DATABASE),
        "serverUuid": mysql.server_uuid,
        "serverWriterAbsent": writer_absent,
        "activeServerWriterProcessIds": list(
            windows.active_server_writer_process_ids
        ),
        "activeDatabaseWriterSessionIds": list(
            mysql.active_database_writer_session_ids
        ),
        "javaServerProcessIds": list(windows.java_server_process_ids),
        "uninspectableJavaProcessIds": list(
            windows.uninspectable_java_process_ids
        ),
        "pythonWriterProcessIds": list(windows.python_writer_process_ids),
        "uninspectablePythonProcessIds": list(
            windows.uninspectable_python_process_ids
        ),
        "listener8080ProcessIds": list(windows.listener_8080_process_ids),
        "listener8081ProcessIds": list(windows.listener_8081_process_ids),
        "mysqlProcessPrivilegeVerified": mysql.process_privilege_verified,
        "producer": PRODUCER,
        "producerSha256": producer_sha256(),
        "powershellProbeSha256": POWERSHELL_PROBE_SHA256,
    }


def write_writer_absence_evidence(
    config: MysqlConfig,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    now: datetime | None = None,
) -> tuple[Mapping[str, Any], Path]:
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    document = collect_writer_absence_evidence(config, now=instant)
    output = (
        artifact_root.expanduser().resolve()
        / (
            "writer-absence-"
            + instant.strftime("%Y%m%dT%H%M%S%fZ")
            + ".json"
        )
    )
    if output.exists():
        raise WriterEvidenceProducerError("writer evidence artifact already exists")
    _atomic_json(output, document)
    return document, output
