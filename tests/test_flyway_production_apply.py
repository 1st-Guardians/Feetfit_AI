from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.shoe.flyway_clone_rehearsal import (
    MigrationFile,
    MigrationManifest,
    MysqlConfig,
    TableDigest,
    _atomic_json as real_atomic_json,
)
from app.services.shoe.flyway_production_apply import (
    DEFAULT_JAVA_SOURCE,
    EXPECTED_FINAL_HISTORY_CORE,
    EXPECTED_INITIAL_HISTORY,
    EXPECTED_PRODUCTION_DATABASE,
    MAINTENANCE_WINDOW_AUTHORITY,
    PINNED_BOOT_JAR_SHA256,
    PINNED_CLONE_REPORT_SHA256,
    PINNED_MIGRATION_MANIFEST_SHA256,
    PINNED_MIGRATIONS,
    PINNED_PRE_MIGRATION_SCHEMA_SHA256,
    PINNED_PROTECTED_DIGESTS,
    PINNED_WRITER_EVIDENCE_PRODUCER_SHA256,
    POWERSHELL_PROBE_SHA256,
    RECOMMENDATION_TABLES_AFTER,
    RECOMMENDATION_TABLES_BEFORE,
    AuthoritySnapshot,
    ProductionAuthorityError,
    ProductionMigrationLock,
    ProductionRuntimeError,
    WriterAbsenceEvidence,
    build_production_plan,
    database_name_fingerprint,
    execute_production_migration,
    load_verified_clone_report,
    load_writer_absence_evidence,
    production_confirmation_token,
    validate_initial_snapshot,
    validate_post_snapshot,
)


SERVER_UUID = "12345678-1234-1234-1234-123456789abc"


class FlywayProductionAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 26, 11, 0, tzinfo=timezone.utc)
        self.config = MysqlConfig(
            host="db.internal",
            port=3306,
            database=EXPECTED_PRODUCTION_DATABASE,
            username="secret-user",
            password="secret-password",
            query_items=(
                ("serverTimezone", "Asia/Seoul"),
                ("characterEncoding", "UTF-8"),
                ("useSSL", "false"),
                ("allowPublicKeyRetrieval", "true"),
            ),
        )
        self.manifest = MigrationManifest(
            sha256=PINNED_MIGRATION_MANIFEST_SHA256,
            migrations=tuple(
                MigrationFile(version, name, digest)
                for version, name, digest in PINNED_MIGRATIONS
            ),
            boot_jar_sha256=PINNED_BOOT_JAR_SHA256,
        )
        self.initial = self._snapshot(initial=True)
        self.after = self._snapshot(initial=False)
        self.evidence = WriterAbsenceEvidence(
            sha256="e" * 64,
            checked_at=self.now,
            server_uuid=SERVER_UUID,
            producer="feetfit-audited-writer-absence-producer-v1",
        )

    def _snapshot(self, *, initial: bool) -> AuthoritySnapshot:
        history = (
            tuple(dict(row) for row in EXPECTED_INITIAL_HISTORY)
            if initial
            else tuple(
                {
                    **dict(row),
                    "installed_on": f"2026-08-26T11:00:0{index}.000000",
                    "execution_time": index,
                }
                for index, row in enumerate(EXPECTED_FINAL_HISTORY_CORE, start=1)
            )
        )
        counts = {
            table_name: 0
            for table_name in (
                RECOMMENDATION_TABLES_BEFORE
                if initial
                else RECOMMENDATION_TABLES_AFTER
            )
        }
        snapshot_hash = "1" * 64 if initial else "2" * 64
        return AuthoritySnapshot(
            server_uuid=SERVER_UUID,
            target_fingerprint="f" * 64,
            schema_sha256=(
                PINNED_PRE_MIGRATION_SCHEMA_SHA256 if initial else "9" * 64
            ),
            history=history,
            protected=dict(PINNED_PROTECTED_DIGESTS),
            recommendation_counts=counts,
            run_table_present=not initial,
            captured_at=self.now.isoformat(),
            sha256=snapshot_hash,
        )

    def _write_evidence(
        self,
        directory: Path,
        *,
        checked_at: datetime | None = None,
        process_ids: list[int] | None = None,
        session_ids: list[int] | None = None,
        server_uuid: str = SERVER_UUID,
    ) -> tuple[Path, str]:
        document = {
            "format": "feetfit-server-writer-absence-evidence",
            "version": 1,
            "status": "PASS",
            "checkedAt": (checked_at or self.now).isoformat(),
            "sourceDatabase": EXPECTED_PRODUCTION_DATABASE,
            "databaseFingerprint": database_name_fingerprint(
                EXPECTED_PRODUCTION_DATABASE
            ),
            "serverUuid": server_uuid,
            "serverWriterAbsent": True,
            "activeServerWriterProcessIds": process_ids or [],
            "activeDatabaseWriterSessionIds": session_ids or [],
            "javaServerProcessIds": process_ids or [],
            "uninspectableJavaProcessIds": [],
            "pythonWriterProcessIds": [],
            "uninspectablePythonProcessIds": [],
            "listener8080ProcessIds": [],
            "listener8081ProcessIds": [],
            "mysqlProcessPrivilegeVerified": True,
            "producer": "feetfit-audited-writer-absence-producer-v1",
            "producerSha256": PINNED_WRITER_EVIDENCE_PRODUCER_SHA256,
            "powershellProbeSha256": POWERSHELL_PROBE_SHA256,
        }
        path = directory / "writer-evidence.json"
        payload = json.dumps(document, sort_keys=True).encode("utf-8")
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def test_writer_evidence_is_fresh_exact_and_server_bound(self):
        with tempfile.TemporaryDirectory() as name:
            path, digest = self._write_evidence(Path(name))
            loaded = load_writer_absence_evidence(
                path,
                expected_sha256=digest,
                expected_server_uuid=SERVER_UUID,
                now=self.now,
            )
            self.assertEqual(loaded.sha256, digest)
            self.assertEqual(loaded.server_uuid, SERVER_UUID)

    def test_writer_evidence_rejects_stale_or_any_writer(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stale, _ = self._write_evidence(
                root, checked_at=self.now - timedelta(minutes=6)
            )
            with self.assertRaises(ProductionAuthorityError):
                load_writer_absence_evidence(stale, now=self.now)

            writer, _ = self._write_evidence(root, process_ids=[42])
            with self.assertRaises(ProductionAuthorityError):
                load_writer_absence_evidence(writer, now=self.now)

            session, _ = self._write_evidence(root, session_ids=[99])
            with self.assertRaises(ProductionAuthorityError):
                load_writer_absence_evidence(session, now=self.now)

    def test_writer_evidence_rejects_future_wrong_hash_and_wrong_server(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            future, digest = self._write_evidence(
                root, checked_at=self.now + timedelta(minutes=1)
            )
            with self.assertRaises(ProductionAuthorityError):
                load_writer_absence_evidence(future, now=self.now)

            path, digest = self._write_evidence(root)
            with self.assertRaises(ProductionAuthorityError):
                load_writer_absence_evidence(path, expected_sha256="0" * 64)
            with self.assertRaises(ProductionAuthorityError):
                load_writer_absence_evidence(
                    path,
                    expected_sha256=digest,
                    expected_server_uuid="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    now=self.now,
                )

    def test_initial_snapshot_is_exactly_pinned(self):
        validate_initial_snapshot(self.initial)
        with self.assertRaises(ProductionAuthorityError):
            validate_initial_snapshot(
                replace(self.initial, schema_sha256="0" * 64)
            )
        with self.assertRaises(ProductionAuthorityError):
            validate_initial_snapshot(
                replace(
                    self.initial,
                    history=self.initial.history
                    + ({"version": None, "success": True},),
                )
            )
        changed = dict(self.initial.protected)
        changed["shoe"] = TableDigest(339, changed["shoe"].sha256)
        with self.assertRaises(ProductionAuthorityError):
            validate_initial_snapshot(replace(self.initial, protected=changed))
        counts = dict(self.initial.recommendation_counts)
        counts["shoe_recommendation"] = 1
        with self.assertRaises(ProductionAuthorityError):
            validate_initial_snapshot(
                replace(self.initial, recommendation_counts=counts)
            )
        with self.assertRaises(ProductionAuthorityError):
            validate_initial_snapshot(replace(self.initial, run_table_present=True))

    def test_confirmation_binds_every_authority_artifact(self):
        token = production_confirmation_token(
            snapshot=self.initial,
            writer_evidence=self.evidence,
            clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
            manifest=self.manifest,
        )
        self.assertEqual(len(token), 64)
        changed_snapshot = replace(self.initial, sha256="0" * 64)
        self.assertNotEqual(
            token,
            production_confirmation_token(
                snapshot=changed_snapshot,
                writer_evidence=self.evidence,
                clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                manifest=self.manifest,
            ),
        )
        changed_evidence = replace(self.evidence, sha256="0" * 64)
        self.assertNotEqual(
            token,
            production_confirmation_token(
                snapshot=self.initial,
                writer_evidence=changed_evidence,
                clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                manifest=self.manifest,
            ),
        )

    def test_post_snapshot_requires_exact_history_zero_recommendations_and_hashes(self):
        validate_post_snapshot(self.initial, self.after)
        with self.assertRaises(ProductionAuthorityError):
            validate_post_snapshot(
                self.initial,
                replace(
                    self.after,
                    history=self.after.history + ({"version": None, "success": True},),
                ),
            )
        counts = dict(self.after.recommendation_counts)
        counts["shoe_recommendation_run"] = 1
        with self.assertRaises(ProductionAuthorityError):
            validate_post_snapshot(
                self.initial, replace(self.after, recommendation_counts=counts)
            )

    def test_plan_without_external_evidence_never_emits_confirmation(self):
        with patch(
            "app.services.shoe.flyway_production_apply.load_verified_clone_report",
            return_value=({}, PINNED_CLONE_REPORT_SHA256),
        ), patch(
            "app.services.shoe.flyway_production_apply.read_authority_snapshot",
            return_value=self.initial,
        ):
            plan = build_production_plan(
                config=self.config,
                manifest=self.manifest,
            )
        self.assertFalse(plan["executeReady"])
        self.assertIsNone(plan["confirmationToken"])
        self.assertFalse(plan["safety"]["productionWritesPermitted"])

    def test_wrong_database_or_execute_pin_fails_before_runtime(self):
        wrong = replace(self.config, database="feetfit_dev")
        with self.assertRaises(ProductionAuthorityError):
            build_production_plan(config=wrong, manifest=self.manifest)

        with patch(
            "app.services.shoe.flyway_production_apply.run_server_runtime_production_flyway"
        ) as runtime, patch(
            "app.services.shoe.flyway_production_apply.load_verified_clone_report"
        ) as report, patch(
            "app.services.shoe.flyway_production_apply.REMOTE_WRITER_FENCE_AUTHORIZED",
            True,
        ):
            with self.assertRaises(ProductionAuthorityError):
                execute_production_migration(
                    config=self.config,
                    manifest=self.manifest,
                    expected_manifest_sha256="bad",
                    expected_boot_jar_sha256=PINNED_BOOT_JAR_SHA256,
                    expected_clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                    writer_evidence_path=Path("missing"),
                    expected_writer_evidence_sha256="e" * 64,
                    confirmation="c" * 64,
                    maintenance_window_authority=MAINTENANCE_WINDOW_AUTHORITY,
                )
            runtime.assert_not_called()
            report.assert_not_called()

    def test_migration_lock_reads_and_exposes_exact_connection_identity(self):
        class Cursor:
            def __init__(self):
                self.query = ""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def execute(self, query, parameters=None):
                self.query = str(query)

            def fetchone(self):
                if "GET_LOCK" in self.query:
                    return (1,)
                if "@@server_uuid" in self.query:
                    return (EXPECTED_PRODUCTION_DATABASE, SERVER_UUID, 777)
                raise AssertionError(self.query)

        class Connection:
            def __init__(self):
                self.closed = False

            def cursor(self):
                return Cursor()

            def close(self):
                self.closed = True

        connection = Connection()
        with patch(
            "app.services.shoe.flyway_production_apply._connect",
            return_value=connection,
        ):
            with ProductionMigrationLock(self.config) as held:
                self.assertEqual(held.server_uuid, SERVER_UUID)
                self.assertEqual(held.connection_id, 777)
        self.assertTrue(connection.closed)

    def test_execute_crosses_runtime_once_and_writes_pass_report(self):
        confirmation = production_confirmation_token(
            snapshot=self.initial,
            writer_evidence=self.evidence,
            clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
            manifest=self.manifest,
        )
        runtime_result = {
            "status": "PASS",
            "stage": "complete",
            "migrationsExecuted": 6,
            "pendingBefore": 6,
            "pendingAfter": 0,
            "currentVersion": "8",
            "errorClass": "",
        }
        lock = MagicMock()
        lock.__enter__.return_value = lock
        lock.server_uuid = SERVER_UUID
        lock.connection_id = 777
        with tempfile.TemporaryDirectory() as name, patch(
            "app.services.shoe.flyway_production_apply.REMOTE_WRITER_FENCE_AUTHORIZED",
            True,
        ), patch(
            "app.services.shoe.flyway_production_apply.load_verified_clone_report",
            return_value=({}, PINNED_CLONE_REPORT_SHA256),
        ), patch(
            "app.services.shoe.flyway_production_apply._copy_and_pin_boot_jar",
            return_value=Path(name) / "staged.jar",
        ), patch(
            "app.services.shoe.flyway_production_apply.ProductionMigrationLock",
            return_value=lock,
        ), patch(
            "app.services.shoe.flyway_production_apply.read_authority_snapshot",
            side_effect=[self.initial, self.initial, self.after],
        ), patch(
            "app.services.shoe.flyway_production_apply.load_writer_absence_evidence",
            return_value=self.evidence,
        ), patch(
            "app.services.shoe.flyway_production_apply.collect_writer_absence_evidence",
            return_value={
                "status": "PASS",
                "checkedAt": self.now.isoformat(),
                "serverWriterAbsent": True,
                "serverUuid": SERVER_UUID,
                "activeServerWriterProcessIds": [],
                "activeDatabaseWriterSessionIds": [],
                "mysqlProcessPrivilegeVerified": True,
                "producerSha256": PINNED_WRITER_EVIDENCE_PRODUCER_SHA256,
                "powershellProbeSha256": POWERSHELL_PROBE_SHA256,
            },
        ) as writer_probe, patch(
            "app.services.shoe.flyway_production_apply.run_server_runtime_production_flyway",
            return_value=runtime_result,
        ) as runtime, patch(
            "app.services.shoe.flyway_production_apply.read_production_reconciliation_invariants",
            return_value={"status": "PASS"},
        ):
            report, path = execute_production_migration(
                config=self.config,
                manifest=self.manifest,
                expected_manifest_sha256=PINNED_MIGRATION_MANIFEST_SHA256,
                expected_boot_jar_sha256=PINNED_BOOT_JAR_SHA256,
                expected_clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                writer_evidence_path=Path(name) / "evidence.json",
                expected_writer_evidence_sha256=self.evidence.sha256,
                confirmation=confirmation,
                maintenance_window_authority=MAINTENANCE_WINDOW_AUTHORITY,
                artifact_root=Path(name) / "artifacts",
                now=self.now,
            )
            runtime.assert_called_once()
            self.assertEqual(writer_probe.call_count, 2)
            self.assertEqual(
                writer_probe.call_args_list[0].kwargs["excluded_mysql_session_ids"],
                (777,),
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(json.loads(path.read_text("utf-8"))["status"], "PASS")

    def test_lock_server_identity_mismatch_blocks_before_runtime(self):
        lock = MagicMock()
        lock.__enter__.return_value = lock
        lock.server_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        lock.connection_id = 777
        with tempfile.TemporaryDirectory() as name, patch(
            "app.services.shoe.flyway_production_apply.REMOTE_WRITER_FENCE_AUTHORIZED",
            True,
        ), patch(
            "app.services.shoe.flyway_production_apply.load_verified_clone_report",
            return_value=({}, PINNED_CLONE_REPORT_SHA256),
        ), patch(
            "app.services.shoe.flyway_production_apply._copy_and_pin_boot_jar",
            return_value=Path(name) / "staged.jar",
        ), patch(
            "app.services.shoe.flyway_production_apply.ProductionMigrationLock",
            return_value=lock,
        ), patch(
            "app.services.shoe.flyway_production_apply.read_authority_snapshot",
            return_value=self.initial,
        ), patch(
            "app.services.shoe.flyway_production_apply.run_server_runtime_production_flyway"
        ) as runtime:
            with self.assertRaises(ProductionAuthorityError):
                execute_production_migration(
                    config=self.config,
                    manifest=self.manifest,
                    expected_manifest_sha256=PINNED_MIGRATION_MANIFEST_SHA256,
                    expected_boot_jar_sha256=PINNED_BOOT_JAR_SHA256,
                    expected_clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                    writer_evidence_path=Path(name) / "unused.json",
                    expected_writer_evidence_sha256="e" * 64,
                    confirmation="c" * 64,
                    maintenance_window_authority=MAINTENANCE_WINDOW_AUTHORITY,
                    artifact_root=Path(name) / "artifacts",
                    now=self.now,
                )
            runtime.assert_not_called()

    def test_runtime_failure_is_indeterminate_and_never_retried(self):
        confirmation = production_confirmation_token(
            snapshot=self.initial,
            writer_evidence=self.evidence,
            clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
            manifest=self.manifest,
        )
        lock = MagicMock()
        lock.__enter__.return_value = lock
        lock.server_uuid = SERVER_UUID
        lock.connection_id = 777
        with tempfile.TemporaryDirectory() as name, patch(
            "app.services.shoe.flyway_production_apply.REMOTE_WRITER_FENCE_AUTHORIZED",
            True,
        ), patch(
            "app.services.shoe.flyway_production_apply.load_verified_clone_report",
            return_value=({}, PINNED_CLONE_REPORT_SHA256),
        ), patch(
            "app.services.shoe.flyway_production_apply._copy_and_pin_boot_jar",
            return_value=Path(name) / "staged.jar",
        ), patch(
            "app.services.shoe.flyway_production_apply.ProductionMigrationLock",
            return_value=lock,
        ), patch(
            "app.services.shoe.flyway_production_apply.read_authority_snapshot",
            side_effect=[self.initial, self.initial],
        ), patch(
            "app.services.shoe.flyway_production_apply.load_writer_absence_evidence",
            return_value=self.evidence,
        ), patch(
            "app.services.shoe.flyway_production_apply.collect_writer_absence_evidence",
            return_value={
                "status": "PASS",
                "checkedAt": self.now.isoformat(),
                "serverWriterAbsent": True,
                "serverUuid": SERVER_UUID,
                "activeServerWriterProcessIds": [],
                "activeDatabaseWriterSessionIds": [],
                "mysqlProcessPrivilegeVerified": True,
                "producerSha256": PINNED_WRITER_EVIDENCE_PRODUCER_SHA256,
                "powershellProbeSha256": POWERSHELL_PROBE_SHA256,
            },
        ), patch(
            "app.services.shoe.flyway_production_apply.run_server_runtime_production_flyway",
            side_effect=ProductionRuntimeError(
                "indeterminate",
                runtime_result={
                    "status": "FAIL",
                    "stage": "migrate",
                    "errorClass": "FlywayMigrateException",
                },
            ),
        ) as runtime, patch(
            "app.services.shoe.flyway_production_apply.best_effort_failure_audit",
            return_value={"attempted": True, "automaticRetry": False},
        ):
            with self.assertRaises(ProductionAuthorityError):
                execute_production_migration(
                    config=self.config,
                    manifest=self.manifest,
                    expected_manifest_sha256=PINNED_MIGRATION_MANIFEST_SHA256,
                    expected_boot_jar_sha256=PINNED_BOOT_JAR_SHA256,
                    expected_clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                    writer_evidence_path=Path(name) / "evidence.json",
                    expected_writer_evidence_sha256=self.evidence.sha256,
                    confirmation=confirmation,
                    maintenance_window_authority=MAINTENANCE_WINDOW_AUTHORITY,
                    artifact_root=Path(name) / "artifacts",
                    now=self.now,
                )
            runtime.assert_called_once()
            report_paths = list(Path(name).glob("artifacts/*/production-apply-report.json"))
            self.assertEqual(len(report_paths), 1)
            failure = json.loads(report_paths[0].read_text("utf-8"))
            self.assertEqual(failure["status"], "INDETERMINATE")
            self.assertTrue(failure["manualInspectionRequired"])
            self.assertEqual(
                failure["flywayFailureRuntimeResult"]["stage"], "migrate"
            )
            self.assertIn("bestEffortPostFailureAudit", failure)
            self.assertFalse(failure["safety"]["automaticRetry"])
            self.assertFalse(failure["safety"]["automaticRollback"])

    def test_final_pass_write_failure_never_erases_durable_conservative_state(self):
        confirmation = production_confirmation_token(
            snapshot=self.initial,
            writer_evidence=self.evidence,
            clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
            manifest=self.manifest,
        )
        lock = MagicMock()
        lock.__enter__.return_value = lock
        lock.server_uuid = SERVER_UUID
        lock.connection_id = 777
        live = {
            "status": "PASS",
            "checkedAt": self.now.isoformat(),
            "serverWriterAbsent": True,
            "serverUuid": SERVER_UUID,
            "activeServerWriterProcessIds": [],
            "activeDatabaseWriterSessionIds": [],
            "mysqlProcessPrivilegeVerified": True,
            "producerSha256": PINNED_WRITER_EVIDENCE_PRODUCER_SHA256,
            "powershellProbeSha256": POWERSHELL_PROBE_SHA256,
        }
        runtime_result = {
            "status": "PASS",
            "stage": "complete",
            "migrationsExecuted": 6,
            "pendingBefore": 6,
            "pendingAfter": 0,
            "currentVersion": "8",
            "errorClass": "",
        }
        writes = 0

        def fail_only_completed(path, document):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("simulated final fsync failure")
            return real_atomic_json(path, document)

        with tempfile.TemporaryDirectory() as name, patch(
            "app.services.shoe.flyway_production_apply.REMOTE_WRITER_FENCE_AUTHORIZED",
            True,
        ), patch(
            "app.services.shoe.flyway_production_apply.load_verified_clone_report",
            return_value=({}, PINNED_CLONE_REPORT_SHA256),
        ), patch(
            "app.services.shoe.flyway_production_apply._copy_and_pin_boot_jar",
            return_value=Path(name) / "staged.jar",
        ), patch(
            "app.services.shoe.flyway_production_apply.ProductionMigrationLock",
            return_value=lock,
        ), patch(
            "app.services.shoe.flyway_production_apply.read_authority_snapshot",
            side_effect=[self.initial, self.initial, self.after],
        ), patch(
            "app.services.shoe.flyway_production_apply.load_writer_absence_evidence",
            return_value=self.evidence,
        ), patch(
            "app.services.shoe.flyway_production_apply.collect_writer_absence_evidence",
            return_value=live,
        ), patch(
            "app.services.shoe.flyway_production_apply.run_server_runtime_production_flyway",
            return_value=runtime_result,
        ), patch(
            "app.services.shoe.flyway_production_apply.read_production_reconciliation_invariants",
            return_value={"status": "PASS"},
        ), patch(
            "app.services.shoe.flyway_production_apply.best_effort_failure_audit",
            return_value={"attempted": True, "automaticRetry": False},
        ), patch(
            "app.services.shoe.flyway_production_apply._atomic_json",
            side_effect=fail_only_completed,
        ):
            with self.assertRaises(ProductionAuthorityError):
                execute_production_migration(
                    config=self.config,
                    manifest=self.manifest,
                    expected_manifest_sha256=PINNED_MIGRATION_MANIFEST_SHA256,
                    expected_boot_jar_sha256=PINNED_BOOT_JAR_SHA256,
                    expected_clone_report_sha256=PINNED_CLONE_REPORT_SHA256,
                    writer_evidence_path=Path(name) / "evidence.json",
                    expected_writer_evidence_sha256=self.evidence.sha256,
                    confirmation=confirmation,
                    maintenance_window_authority=MAINTENANCE_WINDOW_AUTHORITY,
                    artifact_root=Path(name) / "artifacts",
                    now=self.now,
                )
            reports = list(Path(name).glob("artifacts/*/production-apply-report.json"))
            self.assertEqual(len(reports), 1)
            persisted = json.loads(reports[0].read_text("utf-8"))
            self.assertEqual(persisted["status"], "INDETERMINATE")
            self.assertTrue(persisted["completedPostAuditButPassReportWriteFailed"])

    def test_java_helper_has_fixed_safe_operation_order(self):
        source = DEFAULT_JAVA_SOURCE.read_text("utf-8")
        self.assertIn("private static final String AUTHORIZED_DATABASE = \"feetfit\"", source)
        self.assertIn(".cleanDisabled(true)", source)
        self.assertIn(".baselineOnMigrate(false)", source)
        self.assertIn(".connectRetries(0)", source)
        self.assertIn("class GuardedDataSource implements DataSource", source)
        self.assertIn(".dataSource(guardedDataSource)", source)
        self.assertIn("assertExactConnection(connection", source)
        self.assertNotIn(".clean(", source)
        repair = source.index("flyway.repair()")
        first_validate = source.index("flyway.validateWithResult()", repair)
        migrate = source.index("flyway.migrate()", first_validate)
        second_validate = source.index("flyway.validateWithResult()", migrate)
        self.assertLess(repair, first_validate)
        self.assertLess(first_validate, migrate)
        self.assertLess(migrate, second_validate)


if __name__ == "__main__":
    unittest.main()
