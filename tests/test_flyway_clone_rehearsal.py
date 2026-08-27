from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import zipfile

from app.services.shoe.flyway_clone_rehearsal import (
    CONFIRMATION_DOMAIN,
    EXPECTED_MIGRATION_VERSIONS,
    FlywayRehearsalError,
    MigrationFile,
    MigrationManifest,
    MysqlConfig,
    TableShape,
    _is_generated_column,
    _schema_digest,
    assert_safe_clone_name,
    build_plan,
    confirmation_token,
    execute_clone_rehearsal,
    generated_clone_name,
    load_migration_manifest,
    parse_jdbc_mysql_url,
    read_production_reconciliation_invariants,
    update_row_digest,
    validate_post_flyway_history,
)


class FlywayCloneRehearsalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MysqlConfig(
            host="db.internal",
            port=3306,
            database="feetfit",
            username="secret-user",
            password="secret-password",
            query_items=(("serverTimezone", "Asia/Seoul"),),
        )
        self.manifest = MigrationManifest(
            sha256="a" * 64,
            migrations=tuple(
                MigrationFile(str(index), f"V{index}__x.sql", "b" * 64)
                for index in range(1, 9)
            ),
            boot_jar_sha256="c" * 64,
        )

    def test_credentials_are_redacted_from_config_repr_and_plan(self):
        parsed = parse_jdbc_mysql_url(
            "jdbc:mysql://db.internal:3307/feetfit?serverTimezone=Asia%2FSeoul",
            "private-user",
            "private-password",
        )
        self.assertNotIn("private-user", repr(parsed))
        self.assertNotIn("private-password", repr(parsed))
        rendered = json.dumps(build_plan(parsed, self.manifest))
        self.assertNotIn("private-user", rendered)
        self.assertNotIn("private-password", rendered)
        self.assertNotIn("jdbc:mysql", rendered)

    def test_clone_url_changes_only_database_and_keeps_parameters(self):
        clone = generated_clone_name(
            datetime(2026, 8, 26, 1, 2, 3, 456789, tzinfo=timezone.utc),
            "deadbeef",
        )
        url = self.config.clone_jdbc_url(clone)
        self.assertEqual(
            url,
            f"jdbc:mysql://db.internal:3306/{clone}?serverTimezone=Asia%2FSeoul",
        )
        self.assertNotIn("secret", url)

    def test_clone_name_is_generated_and_production_name_is_rejected(self):
        clone = generated_clone_name(
            datetime(2026, 8, 26, tzinfo=timezone.utc), "0123abcd"
        )
        assert_safe_clone_name(clone, "feetfit")
        with self.assertRaises(FlywayRehearsalError):
            assert_safe_clone_name("feetfit", "feetfit")
        with self.assertRaises(FlywayRehearsalError):
            assert_safe_clone_name("attacker_selected_clone", "feetfit")

    def test_rehearsal_clone_cannot_be_used_as_source(self):
        source = generated_clone_name(
            datetime(2026, 8, 26, tzinfo=timezone.utc), "0123abcd"
        )
        target = generated_clone_name(
            datetime(2026, 8, 27, tzinfo=timezone.utc), "0123abce"
        )
        with self.assertRaises(FlywayRehearsalError):
            assert_safe_clone_name(target, source)

    def test_confirmation_is_manifest_and_database_bound(self):
        token = confirmation_token("feetfit", "1" * 64)
        self.assertEqual(len(token), 64)
        self.assertNotEqual(token, confirmation_token("other", "1" * 64))
        self.assertNotEqual(token, confirmation_token("feetfit", "2" * 64))
        self.assertTrue(CONFIRMATION_DOMAIN)

    def test_row_digest_is_unambiguous_and_deterministic(self):
        first = hashlib.sha256()
        second = hashlib.sha256()
        update_row_digest(first, (1, "23", None, b"x"))
        update_row_digest(second, (1, "23", None, b"x"))
        self.assertEqual(first.hexdigest(), second.hexdigest())
        different = hashlib.sha256()
        update_row_digest(different, (12, "3", None, b"x"))
        self.assertNotEqual(first.hexdigest(), different.hexdigest())

    def test_schema_digest_ignores_mysql_redundant_character_set_rendering(self):
        source = TableShape(
            name="example",
            create_sql=(
                "CREATE TABLE `example` (\n"
                "  `value` text COLLATE utf8mb4_unicode_ci\n"
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
            ),
            columns=(),
            primary_key=(),
        )
        clone = TableShape(
            name="example",
            create_sql=(
                "CREATE TABLE `example` (\n"
                "  `value` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci\n"
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
            ),
            columns=(),
            primary_key=(),
        )
        self.assertEqual(
            _schema_digest({"example": source}),
            _schema_digest({"example": clone}),
        )

    def test_default_generated_timestamp_remains_insertable(self):
        self.assertFalse(_is_generated_column("DEFAULT_GENERATED", ""))
        self.assertFalse(_is_generated_column("auto_increment", ""))
        self.assertTrue(_is_generated_column("VIRTUAL GENERATED", "x + 1"))
        self.assertTrue(_is_generated_column("STORED GENERATED", "x + 1"))

    def test_execute_rejects_bad_pin_before_any_database_access(self):
        with patch(
            "app.services.shoe.flyway_clone_rehearsal._source_schema_metadata"
        ) as source:
            with self.assertRaises(FlywayRehearsalError):
                execute_clone_rehearsal(
                    config=self.config,
                    manifest=self.manifest,
                    expected_manifest_sha256="bad",
                    confirmation="bad",
                )
            source.assert_not_called()

    def test_execute_rejects_bad_confirmation_before_any_database_access(self):
        with patch(
            "app.services.shoe.flyway_clone_rehearsal._source_schema_metadata"
        ) as source:
            with self.assertRaises(FlywayRehearsalError):
                execute_clone_rehearsal(
                    config=self.config,
                    manifest=self.manifest,
                    expected_manifest_sha256=self.manifest.sha256,
                    confirmation="bad",
                )
            source.assert_not_called()

    def test_history_validation_preserves_successful_metadata(self):
        before = [
            {
                "version": "1",
                "description": "one",
                "type": "SQL",
                "script": "V1__one.sql",
                "checksum": 10,
                "success": True,
            },
            {
                "version": "3",
                "description": "failed",
                "type": "SQL",
                "script": "V3__failed.sql",
                "checksum": 30,
                "success": False,
            },
        ]
        after = []
        for version in EXPECTED_MIGRATION_VERSIONS:
            row = {
                "version": version,
                "description": "one" if version == "1" else f"v{version}",
                "type": "SQL",
                "script": "V1__one.sql" if version == "1" else f"V{version}__x.sql",
                "checksum": 10 if version == "1" else int(version) * 10,
                "success": True,
            }
            after.append(row)
        validate_post_flyway_history(before, after)

    def test_history_validation_rejects_repair_checksum_rewrite(self):
        before = [
            {
                "version": "1",
                "description": "one",
                "type": "SQL",
                "script": "V1__one.sql",
                "checksum": 10,
                "success": True,
            }
        ]
        after = [
            {
                "version": version,
                "description": "one" if version == "1" else f"v{version}",
                "type": "SQL",
                "script": "V1__one.sql" if version == "1" else f"V{version}__x.sql",
                "checksum": 999 if version == "1" else int(version),
                "success": True,
            }
            for version in EXPECTED_MIGRATION_VERSIONS
        ]
        with self.assertRaises(FlywayRehearsalError):
            validate_post_flyway_history(before, after)

    def test_manifest_requires_exact_source_and_embedded_v1_to_v8(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "server"
            migration_dir = root / "src" / "main" / "resources" / "db" / "migration"
            migration_dir.mkdir(parents=True)
            boot_jar = root / "server.jar"
            embedded: dict[str, bytes] = {}
            for version in EXPECTED_MIGRATION_VERSIONS:
                name = f"V{version}__migration_{version}.sql"
                payload = f"SELECT {version};\n".encode()
                (migration_dir / name).write_bytes(payload)
                embedded[f"BOOT-INF/classes/db/migration/{name}"] = payload
            with zipfile.ZipFile(boot_jar, "w") as archive:
                for name, payload in embedded.items():
                    archive.writestr(name, payload)

            manifest = load_migration_manifest(server_root=root, boot_jar=boot_jar)
            self.assertEqual(
                tuple(item.version for item in manifest.migrations),
                EXPECTED_MIGRATION_VERSIONS,
            )
            self.assertEqual(len(manifest.sha256), 64)

            (migration_dir / "V8__migration_8.sql").write_text("SELECT 99;\n")
            with self.assertRaises(FlywayRehearsalError):
                load_migration_manifest(server_root=root, boot_jar=boot_jar)

    def test_production_invariant_audit_rejects_connection_identity_drift(self):
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
                if "@@server_uuid" in self.query:
                    return ("feetfit", "wrong-server-uuid")
                raise AssertionError(self.query)

        class Connection:
            def cursor(self):
                return Cursor()

            def close(self):
                return None

        with patch(
            "app.services.shoe.flyway_clone_rehearsal._connect",
            return_value=Connection(),
        ), self.assertRaises(FlywayRehearsalError):
            read_production_reconciliation_invariants(
                self.config,
                expected_server_uuid="expected-server-uuid",
            )


if __name__ == "__main__":
    unittest.main()
