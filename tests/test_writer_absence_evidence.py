from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.shoe.flyway_clone_rehearsal import MysqlConfig, sha256_file
from app.services.shoe.writer_absence_evidence import (
    POWERSHELL_PROBE,
    MysqlWriterState,
    WindowsWriterState,
    WriterEvidenceProducerError,
    _has_global_process_privilege,
    collect_writer_absence_evidence,
    read_mysql_writer_state,
    read_windows_writer_state,
    write_writer_absence_evidence,
)


SERVER_UUID = "12345678-1234-1234-1234-123456789abc"


class _FakeCursor:
    def __init__(self, *, process_grant: bool = True, sessions: tuple[int, ...] = ()):
        self.process_grant = process_grant
        self.sessions = sessions
        self.query = ""
        self.statements: list[str] = []
        self.parameters: list[object] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, query, parameters=None):
        self.query = " ".join(str(query).split())
        self.statements.append(self.query)
        self.parameters.append(parameters)

    def fetchone(self):
        if self.query.startswith("SELECT DATABASE()"):
            return ("feetfit", SERVER_UUID, 501)
        raise AssertionError(f"unexpected fetchone query: {self.query}")

    def fetchall(self):
        if self.query.startswith("SHOW GRANTS"):
            grant = (
                "GRANT PROCESS ON *.* TO `feetfit`@`%`"
                if self.process_grant
                else "GRANT SELECT ON `feetfit`.* TO `feetfit`@`%`"
            )
            return ((grant,),)
        if self.query.startswith("SELECT DISTINCT p.ID"):
            return tuple((value,) for value in self.sessions)
        raise AssertionError(f"unexpected fetchall query: {self.query}")


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor):
        self.fake_cursor = cursor
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.fake_cursor

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class WriterAbsenceEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MysqlConfig(
            host="db.internal",
            port=3306,
            database="feetfit",
            username="secret-user",
            password="secret-password",
        )
        self.now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)

    def test_powershell_probe_is_read_only_and_covers_all_writer_classes(self):
        self.assertIn("Get-CimInstance Win32_Process", POWERSHELL_PROBE)
        self.assertIn("Get-NetTCPConnection", POWERSHELL_PROBE)
        self.assertIn("FeetFit_Server", POWERSHELL_PROBE)
        self.assertIn("shoe_crawler", POWERSHELL_PROBE)
        self.assertIn("bidirectional_selection_pipeline", POWERSHELL_PROBE)
        self.assertIn("verified_ingestion", POWERSHELL_PROBE)
        self.assertNotIn("Stop-Process", POWERSHELL_PROBE)
        self.assertNotIn("Start-Process", POWERSHELL_PROBE)
        self.assertNotIn("Remove-", POWERSHELL_PROBE)
        self.assertNotIn("SilentlyContinue", POWERSHELL_PROBE)
        self.assertIn("Get-NetTCPConnection -State Listen -ErrorAction Stop", POWERSHELL_PROBE)

    def test_windows_probe_strictly_unions_java_python_and_listener_ids(self):
        document = {
            "javaServerProcessIds": [10],
            "uninspectableJavaProcessIds": [11],
            "pythonWriterProcessIds": [20],
            "uninspectablePythonProcessIds": [21],
            "listener8080ProcessIds": [30],
            "listener8081ProcessIds": [31],
            "activeServerWriterProcessIds": [10, 11, 20, 21, 30, 31],
        }
        completed = MagicMock(
            returncode=0, stdout=json.dumps(document), stderr=""
        )
        with patch(
            "app.services.shoe.writer_absence_evidence._powershell_executable",
            return_value=Path("powershell.exe"),
        ), patch(
            "app.services.shoe.writer_absence_evidence.subprocess.run",
            return_value=completed,
        ):
            state = read_windows_writer_state()
        self.assertEqual(state.python_writer_process_ids, (20,))
        self.assertEqual(
            state.active_server_writer_process_ids, (10, 11, 20, 21, 30, 31)
        )

    def test_windows_probe_rejects_incomplete_union(self):
        document = {
            "javaServerProcessIds": [10],
            "uninspectableJavaProcessIds": [],
            "pythonWriterProcessIds": [],
            "uninspectablePythonProcessIds": [],
            "listener8080ProcessIds": [],
            "listener8081ProcessIds": [],
            "activeServerWriterProcessIds": [],
        }
        completed = MagicMock(returncode=0, stdout=json.dumps(document), stderr="")
        with patch(
            "app.services.shoe.writer_absence_evidence._powershell_executable",
            return_value=Path("powershell.exe"),
        ), patch(
            "app.services.shoe.writer_absence_evidence.subprocess.run",
            return_value=completed,
        ), self.assertRaises(WriterEvidenceProducerError):
            read_windows_writer_state()

    def test_mysql_probe_requires_process_visibility_and_is_read_only(self):
        cursor = _FakeCursor(sessions=(700, 701))
        connection = _FakeConnection(cursor)
        with patch(
            "app.services.shoe.writer_absence_evidence._connect",
            return_value=connection,
        ):
            state = read_mysql_writer_state(
                self.config, excluded_session_ids=(777,)
            )
        self.assertEqual(state.active_database_writer_session_ids, (700, 701))
        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)
        rendered = " ".join(cursor.statements).upper()
        self.assertIn("START TRANSACTION READ ONLY", rendered)
        self.assertIn("P.DB = %S", rendered)
        self.assertIn("INNODB_TRX", rendered)
        self.assertIn("P.ID NOT IN", rendered)
        forbidden = (" INSERT ", " UPDATE ", " DELETE ", " ALTER ", " DROP ", " CREATE ")
        padded = " " + rendered + " "
        self.assertFalse(any(token in padded for token in forbidden))

    def test_mysql_probe_fails_without_global_process_visibility(self):
        cursor = _FakeCursor(process_grant=False)
        connection = _FakeConnection(cursor)
        with patch(
            "app.services.shoe.writer_absence_evidence._connect",
            return_value=connection,
        ), self.assertRaises(WriterEvidenceProducerError):
            read_mysql_writer_state(self.config)

    def test_collect_and_atomic_write_pass_only_when_every_probe_is_empty(self):
        empty_windows = WindowsWriterState((), (), (), (), (), (), ())
        empty_mysql = MysqlWriterState(SERVER_UUID, (), True)
        with tempfile.TemporaryDirectory() as name, patch(
            "app.services.shoe.writer_absence_evidence.read_windows_writer_state",
            return_value=empty_windows,
        ), patch(
            "app.services.shoe.writer_absence_evidence.read_mysql_writer_state",
            return_value=empty_mysql,
        ):
            document, path = write_writer_absence_evidence(
                self.config, artifact_root=Path(name), now=self.now
            )
            self.assertEqual(document["status"], "PASS")
            self.assertTrue(document["serverWriterAbsent"])
            self.assertEqual(json.loads(path.read_text("utf-8"))["status"], "PASS")
            self.assertEqual(len(sha256_file(path)), 64)

    def test_any_python_writer_makes_evidence_fail(self):
        windows = WindowsWriterState((), (), (44,), (), (), (), (44,))
        mysql = MysqlWriterState(SERVER_UUID, (), True)
        with patch(
            "app.services.shoe.writer_absence_evidence.read_windows_writer_state",
            return_value=windows,
        ), patch(
            "app.services.shoe.writer_absence_evidence.read_mysql_writer_state",
            return_value=mysql,
        ):
            document = collect_writer_absence_evidence(self.config, now=self.now)
        self.assertEqual(document["status"], "FAIL")
        self.assertFalse(document["serverWriterAbsent"])

    def test_global_process_grant_parser_is_fail_closed(self):
        self.assertTrue(_has_global_process_privilege(("GRANT PROCESS ON *.* TO x",)))
        self.assertTrue(
            _has_global_process_privilege(("GRANT ALL PRIVILEGES ON *.* TO x",))
        )
        self.assertFalse(
            _has_global_process_privilege(("GRANT SELECT ON feetfit.* TO x",))
        )


if __name__ == "__main__":
    unittest.main()
