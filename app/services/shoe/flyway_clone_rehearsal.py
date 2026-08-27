"""Fail-closed rehearsal of FeetFit Server Flyway migrations on a DB clone.

The production/source schema is opened with a consistent read-only transaction.
Every SQL mutation is directed at a generated, timestamped rehearsal database.
Flyway itself runs from the Server boot jar's runtime classpath so the rehearsal
uses the exact driver, Flyway implementation, and embedded migrations that the
Server would use at runtime.

This module deliberately does not offer cleanup/drop functionality.  A failed
rehearsal clone is retained as evidence and can only be removed through a
separately reviewed database-administration operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pymysql

from app.core.config import PROJECT_ROOT, settings


FORMAT = "feetfit-flyway-clone-rehearsal"
VERSION = 1
EXPECTED_MIGRATION_VERSIONS = tuple(str(value) for value in range(1, 9))
PROTECTED_TABLES = (
    "shoe",
    "shoe_review",
    "shoe_lab_measurement",
    "shoe_lab_metric",
)
CLONE_PREFIX = "feetfit_flyway_rehearsal_"
CONFIRMATION_DOMAIN = "FEETFIT_CLONE_ONLY_FLYWAY_REHEARSAL_V1"
DEFAULT_SERVER_ROOT = PROJECT_ROOT.parent / "FeetFit_Server"
DEFAULT_SERVER_BOOT_JAR = (
    DEFAULT_SERVER_ROOT / "build" / "libs" / "FeetFit_Server-1.0.0.jar"
)
DEFAULT_JAVA_SOURCE = (
    PROJECT_ROOT / "scripts" / "java" / "FeetfitFlywayCloneRehearsal.java"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "flyway-rehearsal"

_MIGRATION_RE = re.compile(r"^V(?P<version>[1-9][0-9]*)__.+\.sql$")
_SAFE_CLONE_RE = re.compile(
    rf"^{re.escape(CLONE_PREFIX)}[0-9]{{8}}T[0-9]{{12}}Z_[0-9a-f]{{8}}$"
)


class FlywayRehearsalError(RuntimeError):
    """A rehearsal precondition or invariant failed."""


@dataclass(frozen=True, slots=True)
class MysqlConfig:
    host: str
    port: int
    database: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    query_items: tuple[tuple[str, str], ...] = ()

    def clone_jdbc_url(self, clone_database: str) -> str:
        assert_safe_clone_name(clone_database, self.database)
        query = urlencode(self.query_items)
        netloc = f"{self.host}:{self.port}"
        return urlunparse(
            ("mysql", netloc, f"/{clone_database}", "", query, "")
        ).replace("mysql://", "jdbc:mysql://", 1)


@dataclass(frozen=True, slots=True)
class MigrationFile:
    version: str
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MigrationManifest:
    sha256: str
    migrations: tuple[MigrationFile, ...]
    boot_jar_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "bootJarSha256": self.boot_jar_sha256,
            "migrations": [
                {
                    "version": item.version,
                    "name": item.name,
                    "sha256": item.sha256,
                }
                for item in self.migrations
            ],
        }


@dataclass(frozen=True, slots=True)
class TableDigest:
    row_count: int
    sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {"rowCount": self.row_count, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ColumnShape:
    name: str
    generated: bool


@dataclass(frozen=True, slots=True)
class TableShape:
    name: str
    create_sql: str
    columns: tuple[ColumnShape, ...]
    primary_key: tuple[str, ...]

    @property
    def insert_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if not column.generated)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_jdbc_mysql_url(jdbc_url: str, username: str, password: str) -> MysqlConfig:
    if not jdbc_url.startswith("jdbc:mysql://"):
        raise FlywayRehearsalError("SHOE_DB_URL must be a jdbc:mysql URL")
    parsed = urlparse(jdbc_url.replace("jdbc:mysql://", "mysql://", 1))
    database = parsed.path.lstrip("/")
    if (
        not parsed.hostname
        or not database
        or "/" in database
        or not username
        or not password
    ):
        raise FlywayRehearsalError("SHOE_DB_* configuration is incomplete")
    return MysqlConfig(
        host=parsed.hostname,
        port=parsed.port or 3306,
        database=database,
        username=username,
        password=password,
        query_items=tuple(parse_qsl(parsed.query, keep_blank_values=True)),
    )


def mysql_config_from_settings() -> MysqlConfig:
    return parse_jdbc_mysql_url(
        settings.shoe_db_url,
        settings.shoe_db_username,
        settings.shoe_db_password,
    )


def assert_safe_clone_name(clone_database: str, source_database: str) -> None:
    if not _SAFE_CLONE_RE.fullmatch(clone_database):
        raise FlywayRehearsalError("clone database name is not generated by this tool")
    if clone_database.casefold() == source_database.casefold():
        raise FlywayRehearsalError("clone database must differ from source database")
    if source_database.casefold().startswith(CLONE_PREFIX.casefold()):
        raise FlywayRehearsalError("a rehearsal clone cannot be used as the source")


def generated_clone_name(now: datetime | None = None, nonce: str | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    stamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    resolved_nonce = nonce or secrets.token_hex(4)
    if not re.fullmatch(r"[0-9a-f]{8}", resolved_nonce):
        raise FlywayRehearsalError("clone nonce must be eight lowercase hex digits")
    name = f"{CLONE_PREFIX}{stamp}_{resolved_nonce}"
    if len(name) > 64:
        raise FlywayRehearsalError("generated clone database name exceeds MySQL limit")
    return name


def _migration_map_from_directory(directory: Path) -> dict[str, bytes]:
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise FlywayRehearsalError("Server migration directory is missing")
    result: dict[str, bytes] = {}
    for path in sorted(resolved.glob("*.sql"), key=lambda item: item.name):
        match = _MIGRATION_RE.fullmatch(path.name)
        if match is None:
            raise FlywayRehearsalError(
                f"unsupported SQL migration filename: {path.name}"
            )
        version = match.group("version")
        if version in result:
            raise FlywayRehearsalError(f"duplicate migration version: V{version}")
        result[version] = path.read_bytes()
    return result


def _migration_map_from_boot_jar(boot_jar: Path) -> dict[str, tuple[str, bytes]]:
    resolved = boot_jar.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FlywayRehearsalError("Server boot jar is missing or unsafe")
    prefix = "BOOT-INF/classes/db/migration/"
    result: dict[str, tuple[str, bytes]] = {}
    try:
        with zipfile.ZipFile(resolved) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                name = member.filename[len(prefix) :]
                if "/" in name or not name.endswith(".sql"):
                    continue
                match = _MIGRATION_RE.fullmatch(name)
                if match is None:
                    raise FlywayRehearsalError(
                        f"unsupported embedded SQL migration filename: {name}"
                    )
                version = match.group("version")
                if version in result:
                    raise FlywayRehearsalError(
                        f"duplicate embedded migration version: V{version}"
                    )
                result[version] = (name, archive.read(member))
    except zipfile.BadZipFile as exc:
        raise FlywayRehearsalError("Server boot jar is invalid") from exc
    return result


def load_migration_manifest(
    *, server_root: Path = DEFAULT_SERVER_ROOT, boot_jar: Path = DEFAULT_SERVER_BOOT_JAR
) -> MigrationManifest:
    source_dir = server_root.resolve() / "src" / "main" / "resources" / "db" / "migration"
    source = _migration_map_from_directory(source_dir)
    embedded = _migration_map_from_boot_jar(boot_jar)
    expected = set(EXPECTED_MIGRATION_VERSIONS)
    if set(source) != expected or set(embedded) != expected:
        raise FlywayRehearsalError(
            "Server source and boot jar must each contain exactly V1 through V8"
        )

    files: list[MigrationFile] = []
    for version in EXPECTED_MIGRATION_VERSIONS:
        embedded_name, embedded_bytes = embedded[version]
        source_candidates = [
            path
            for path in source_dir.glob(f"V{version}__*.sql")
            if path.is_file()
        ]
        if len(source_candidates) != 1:
            raise FlywayRehearsalError(f"V{version} source migration is ambiguous")
        source_name = source_candidates[0].name
        if source_name != embedded_name or source[version] != embedded_bytes:
            raise FlywayRehearsalError(
                f"Server boot jar migration V{version} differs from source"
            )
        files.append(
            MigrationFile(
                version=version,
                name=source_name,
                sha256=_sha256_bytes(embedded_bytes),
            )
        )

    manifest_payload = json.dumps(
        [
            {"version": item.version, "name": item.name, "sha256": item.sha256}
            for item in files
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return MigrationManifest(
        sha256=_sha256_bytes(manifest_payload),
        migrations=tuple(files),
        boot_jar_sha256=sha256_file(boot_jar.resolve()),
    )


def confirmation_token(source_database: str, manifest_sha256: str) -> str:
    payload = (
        f"{CONFIRMATION_DOMAIN}\0{source_database.casefold()}\0{manifest_sha256}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_plan(config: MysqlConfig, manifest: MigrationManifest) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "version": VERSION,
        "mode": "PLAN_ONLY",
        "sourceDatabaseFingerprint": hashlib.sha256(
            config.database.casefold().encode("utf-8")
        ).hexdigest()[:16],
        "targetPolicy": "GENERATED_TIMESTAMPED_CLONE_ONLY",
        "targetMigration": "8",
        "protectedTables": list(PROTECTED_TABLES),
        "migrationManifest": manifest.public_dict(),
        "confirmationToken": confirmation_token(config.database, manifest.sha256),
        "productionWritesPermitted": False,
    }


def _quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise FlywayRehearsalError("unsafe MySQL identifier")
    return "`" + value.replace("`", "``") + "`"


def _is_generated_column(extra: object, generation_expression: object) -> bool:
    normalized = str(extra).upper()
    return bool(generation_expression) or "VIRTUAL GENERATED" in normalized or "STORED GENERATED" in normalized


def _canonical_value(value: Any) -> bytes:
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"B1" if value else b"B0"
    if isinstance(value, int):
        return b"I" + str(value).encode("ascii")
    if isinstance(value, Decimal):
        return b"D" + format(value, "f").encode("ascii")
    if isinstance(value, float):
        if math.isnan(value):
            return b"Fnan"
        return b"F" + struct.pack("!d", value)
    if isinstance(value, bytes):
        return b"Y" + value
    if isinstance(value, str):
        return b"S" + value.encode("utf-8")
    if isinstance(value, datetime):
        return b"T" + value.isoformat(timespec="microseconds").encode("ascii")
    if isinstance(value, date):
        return b"A" + value.isoformat().encode("ascii")
    if isinstance(value, time):
        return b"C" + value.isoformat(timespec="microseconds").encode("ascii")
    if isinstance(value, timedelta):
        return b"L" + str(value.total_seconds()).encode("ascii")
    raise FlywayRehearsalError(
        f"unsupported MySQL value type in digest: {type(value).__name__}"
    )


def update_row_digest(digest: "hashlib._Hash", row: Sequence[Any]) -> None:
    digest.update(struct.pack("!I", len(row)))
    for value in row:
        encoded = _canonical_value(value)
        digest.update(struct.pack("!Q", len(encoded)))
        digest.update(encoded)


def _connect(
    config: MysqlConfig,
    *,
    database: str | None,
    autocommit: bool,
    cursorclass: type | None = None,
) -> pymysql.connections.Connection:
    kwargs: dict[str, Any] = {
        "host": config.host,
        "port": config.port,
        "user": config.username,
        "password": config.password,
        "database": database,
        "autocommit": autocommit,
        "charset": "utf8mb4",
        "connect_timeout": 15,
        "read_timeout": 300,
        "write_timeout": 300,
    }
    if cursorclass is not None:
        kwargs["cursorclass"] = cursorclass
    return pymysql.connect(**kwargs)


def _assert_current_database(
    connection: pymysql.connections.Connection, expected: str
) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE()")
        row = cursor.fetchone()
    current = row[0] if row else None
    if current != expected:
        raise FlywayRehearsalError("write connection is not scoped to rehearsal clone")


def _read_source_shapes(
    connection: pymysql.connections.Connection, source_database: str
) -> tuple[dict[str, TableShape], str, str]:
    """Read base-table DDL and fail if unsupported DB objects would be omitted."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT default_character_set_name, default_collation_name
            FROM information_schema.schemata
            WHERE schema_name = %s
            """,
            (source_database,),
        )
        schema_row = cursor.fetchone()
        if schema_row is None:
            raise FlywayRehearsalError("source database does not exist")
        charset, collation = str(schema_row[0]), str(schema_row[1])

        cursor.execute(
            """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = %s
            ORDER BY table_name
            """,
            (source_database,),
        )
        table_rows = cursor.fetchall()
        unsupported = [str(row[0]) for row in table_rows if row[1] != "BASE TABLE"]
        if unsupported:
            raise FlywayRehearsalError(
                "source contains views; exact clone requires an audited view strategy"
            )

        object_queries = {
            "triggers": (
                "SELECT COUNT(*) FROM information_schema.triggers "
                "WHERE trigger_schema = %s"
            ),
            "routines": (
                "SELECT COUNT(*) FROM information_schema.routines "
                "WHERE routine_schema = %s"
            ),
            "events": (
                "SELECT COUNT(*) FROM information_schema.events "
                "WHERE event_schema = %s"
            ),
        }
        for label, query in object_queries.items():
            cursor.execute(query, (source_database,))
            if int(cursor.fetchone()[0]) != 0:
                raise FlywayRehearsalError(
                    f"source contains {label}; exact clone strategy is intentionally fail-closed"
                )

        shapes: dict[str, TableShape] = {}
        source_marker = _quote_identifier(source_database) + "."
        for table_name_value, _ in table_rows:
            table_name = str(table_name_value)
            cursor.execute(
                f"SHOW CREATE TABLE {_quote_identifier(source_database)}."
                f"{_quote_identifier(table_name)}"
            )
            create_row = cursor.fetchone()
            if create_row is None:
                raise FlywayRehearsalError("SHOW CREATE TABLE returned no row")
            create_sql = str(create_row[1])
            if source_marker.casefold() in create_sql.casefold():
                raise FlywayRehearsalError(
                    "source-qualified reference detected in table DDL"
                )

            cursor.execute(
                """
                SELECT column_name, extra, generation_expression
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (source_database, table_name),
            )
            columns = tuple(
                ColumnShape(
                    name=str(column_name),
                    # DEFAULT_GENERATED means a normal writable column whose
                    # default expression was generated by MySQL.  Omitting it
                    # would silently replace exact timestamp values during a
                    # clone.  Only true virtual/stored generated columns are
                    # non-insertable.
                    generated=_is_generated_column(extra, generation_expression),
                )
                for column_name, extra, generation_expression in cursor.fetchall()
            )
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.key_column_usage
                WHERE table_schema = %s AND table_name = %s
                  AND constraint_name = 'PRIMARY'
                ORDER BY ordinal_position
                """,
                (source_database, table_name),
            )
            primary_key = tuple(str(row[0]) for row in cursor.fetchall())
            if not columns:
                raise FlywayRehearsalError(f"source table has no columns: {table_name}")
            shapes[table_name] = TableShape(
                name=table_name,
                create_sql=create_sql,
                columns=columns,
                primary_key=primary_key,
            )

    missing_protected = set(PROTECTED_TABLES) - set(shapes)
    if missing_protected:
        raise FlywayRehearsalError("one or more protected tables are missing")
    return shapes, charset, collation


def _schema_digest(shapes: Mapping[str, TableShape]) -> str:
    # MySQL SHOW CREATE may expand ``COLLATE utf8mb4_*`` to the semantically
    # identical ``CHARACTER SET utf8mb4 COLLATE utf8mb4_*`` when the same DDL is
    # replayed in another schema.  Collation already fixes the character set,
    # so discard that presentation-only expansion while preserving every
    # column type, collation, key, constraint, and table option.
    def normalized(create_sql: str) -> str:
        return re.sub(
            r" CHARACTER SET [A-Za-z0-9_]+(?= COLLATE [A-Za-z0-9_]+)",
            "",
            create_sql,
        )

    digest = hashlib.sha256()
    for table_name in sorted(shapes):
        shape = shapes[table_name]
        for value in (table_name, normalized(shape.create_sql)):
            encoded = value.encode("utf-8")
            digest.update(struct.pack("!Q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _create_clone_schema(
    config: MysqlConfig,
    clone_database: str,
    *,
    charset: str,
    collation: str,
    shapes: Mapping[str, TableShape],
) -> None:
    assert_safe_clone_name(clone_database, config.database)
    if not re.fullmatch(r"[A-Za-z0-9_]+", charset) or not re.fullmatch(
        r"[A-Za-z0-9_]+", collation
    ):
        raise FlywayRehearsalError("unsafe source charset or collation")

    admin = _connect(config, database=None, autocommit=True)
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = %s",
                (clone_database,),
            )
            if int(cursor.fetchone()[0]) != 0:
                raise FlywayRehearsalError("generated clone database already exists")
            cursor.execute(
                f"CREATE DATABASE {_quote_identifier(clone_database)} "
                f"CHARACTER SET {charset} COLLATE {collation}"
            )
    finally:
        admin.close()

    clone = _connect(config, database=clone_database, autocommit=True)
    try:
        _assert_current_database(clone, clone_database)
        with clone.cursor() as cursor:
            cursor.execute("SET SESSION FOREIGN_KEY_CHECKS = 0")
            try:
                for table_name in sorted(shapes):
                    cursor.execute(shapes[table_name].create_sql)
            finally:
                cursor.execute("SET SESSION FOREIGN_KEY_CHECKS = 1")
    finally:
        clone.close()


def _ordered_select_sql(database: str, shape: TableShape) -> str:
    columns = ",".join(_quote_identifier(column.name) for column in shape.columns)
    sql = (
        f"SELECT {columns} FROM {_quote_identifier(database)}."
        f"{_quote_identifier(shape.name)}"
    )
    if shape.primary_key:
        sql += " ORDER BY " + ",".join(
            _quote_identifier(column) for column in shape.primary_key
        )
    else:
        # The digest remains deterministic for current FeetFit tables, all of
        # which have primary keys.  Refuse a non-empty keyless table rather than
        # claim a stable SHA over unspecified row order.
        sql += " LIMIT 0"
    return sql


def _copy_snapshot_data(
    config: MysqlConfig,
    clone_database: str,
    shapes: Mapping[str, TableShape],
    *,
    batch_size: int = 500,
) -> tuple[dict[str, int], dict[str, TableDigest], str]:
    """Copy one consistent production snapshot into the clone.

    Source rows are read through a non-locking, read-only consistent snapshot
    and inserted over a separate connection whose current schema is the clone.
    """

    assert_safe_clone_name(clone_database, config.database)
    source = _connect(config, database=config.database, autocommit=False)
    clone = _connect(config, database=clone_database, autocommit=False)
    table_counts: dict[str, int] = {}
    protected: dict[str, TableDigest] = {}
    try:
        with source.cursor() as cursor:
            cursor.execute(
                "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
            cursor.execute("SELECT DATE_FORMAT(UTC_TIMESTAMP(6), '%Y-%m-%dT%H:%i:%s.%fZ')")
            snapshot_at = str(cursor.fetchone()[0])

        _assert_current_database(clone, clone_database)
        with clone.cursor() as cursor:
            cursor.execute("SET SESSION FOREIGN_KEY_CHECKS = 0")
            cursor.execute("SET SESSION UNIQUE_CHECKS = 0")

        for table_name in sorted(shapes):
            shape = shapes[table_name]
            if not shape.primary_key:
                with source.cursor() as count_cursor:
                    count_cursor.execute(
                        f"SELECT COUNT(*) FROM {_quote_identifier(config.database)}."
                        f"{_quote_identifier(table_name)}"
                    )
                    if int(count_cursor.fetchone()[0]) != 0:
                        raise FlywayRehearsalError(
                            f"non-empty keyless table cannot be hashed safely: {table_name}"
                        )

            digest = hashlib.sha256()
            row_count = 0
            all_column_names = tuple(column.name for column in shape.columns)
            insert_indices = tuple(
                index
                for index, column in enumerate(shape.columns)
                if not column.generated
            )
            insert_columns = shape.insert_columns
            placeholders = ",".join("%s" for _ in insert_columns)
            insert_sql = (
                f"INSERT INTO {_quote_identifier(clone_database)}."
                f"{_quote_identifier(table_name)} "
                f"({','.join(_quote_identifier(value) for value in insert_columns)}) "
                f"VALUES ({placeholders})"
            )

            stream = source.cursor(pymysql.cursors.SSCursor)
            try:
                stream.execute(_ordered_select_sql(config.database, shape))
                while True:
                    rows = stream.fetchmany(batch_size)
                    if not rows:
                        break
                    for row in rows:
                        if len(row) != len(all_column_names):
                            raise FlywayRehearsalError("source row column count changed")
                        update_row_digest(digest, row)
                    insert_rows = [
                        tuple(row[index] for index in insert_indices) for row in rows
                    ]
                    if insert_rows:
                        with clone.cursor() as insert_cursor:
                            insert_cursor.executemany(insert_sql, insert_rows)
                    row_count += len(rows)
            finally:
                stream.close()

            table_counts[table_name] = row_count
            if table_name in PROTECTED_TABLES:
                protected[table_name] = TableDigest(row_count, digest.hexdigest())

        clone.commit()
        with clone.cursor() as cursor:
            cursor.execute("SET SESSION UNIQUE_CHECKS = 1")
            cursor.execute("SET SESSION FOREIGN_KEY_CHECKS = 1")
        source.rollback()
        return table_counts, protected, snapshot_at
    except Exception:
        clone.rollback()
        source.rollback()
        raise
    finally:
        clone.close()
        source.close()


def read_table_digest(
    config: MysqlConfig, database: str, shape: TableShape
) -> TableDigest:
    connection = _connect(config, database=database, autocommit=False)
    digest = hashlib.sha256()
    count = 0
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
        stream = connection.cursor(pymysql.cursors.SSCursor)
        try:
            stream.execute(_ordered_select_sql(database, shape))
            while True:
                rows = stream.fetchmany(500)
                if not rows:
                    break
                for row in rows:
                    update_row_digest(digest, row)
                count += len(rows)
        finally:
            stream.close()
        connection.rollback()
        return TableDigest(count, digest.hexdigest())
    finally:
        connection.close()


def read_protected_digests(
    config: MysqlConfig, database: str, shapes: Mapping[str, TableShape]
) -> dict[str, TableDigest]:
    return {
        table_name: read_table_digest(config, database, shapes[table_name])
        for table_name in PROTECTED_TABLES
    }


def _read_clone_shapes(
    config: MysqlConfig, clone_database: str
) -> dict[str, TableShape]:
    connection = _connect(config, database=clone_database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
        shapes, _, _ = _read_source_shapes(connection, clone_database)
        connection.rollback()
        return shapes
    finally:
        connection.close()


def _table_counts(
    config: MysqlConfig, database: str, table_names: Iterable[str]
) -> dict[str, int]:
    connection = _connect(config, database=database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            result: dict[str, int] = {}
            for table_name in sorted(table_names):
                cursor.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(database)}."
                    f"{_quote_identifier(table_name)}"
                )
                result[table_name] = int(cursor.fetchone()[0])
            connection.rollback()
            return result
    finally:
        connection.close()


def _source_shapes(config: MysqlConfig) -> dict[str, TableShape]:
    connection = _connect(config, database=config.database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
        shapes, _, _ = _read_source_shapes(connection, config.database)
        connection.rollback()
        return shapes
    finally:
        connection.close()


def _source_schema_metadata(
    config: MysqlConfig,
) -> tuple[dict[str, TableShape], str, str]:
    connection = _connect(config, database=config.database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
        result = _read_source_shapes(connection, config.database)
        connection.rollback()
        return result
    finally:
        connection.close()


_HISTORY_COLUMNS = (
    "installed_rank",
    "version",
    "description",
    "type",
    "script",
    "checksum",
    "installed_on",
    "execution_time",
    "success",
)


def read_flyway_history(
    config: MysqlConfig, database: str
) -> list[dict[str, Any]]:
    connection = _connect(config, database=database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'flyway_schema_history'
                """,
                (database,),
            )
            if int(cursor.fetchone()[0]) == 0:
                connection.rollback()
                return []
            cursor.execute(
                f"SELECT {','.join(_quote_identifier(value) for value in _HISTORY_COLUMNS)} "
                f"FROM {_quote_identifier(database)}.`flyway_schema_history` "
                "ORDER BY installed_rank"
            )
            rows = cursor.fetchall()
            connection.rollback()
    finally:
        connection.close()

    history: list[dict[str, Any]] = []
    for row in rows:
        history.append(
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
        )
    return history


def _read_reconciliation_invariants(
    config: MysqlConfig,
    database: str,
    *,
    expected_server_uuid: str | None = None,
) -> dict[str, Any]:
    """Assert the material V7/V8 schema outcomes on an explicitly scoped DB."""

    connection = _connect(config, database=database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute("SELECT DATABASE(), @@server_uuid")
            identity = cursor.fetchone()
            if (
                identity is None
                or identity[0] != database
                or (
                    expected_server_uuid is not None
                    and str(identity[1]).casefold()
                    != expected_server_uuid.casefold()
                )
            ):
                raise FlywayRehearsalError(
                    "reconciliation audit connection identity mismatch"
                )
            cursor.execute(
                """
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'shoe_recommendation'
                  AND column_name = 'measurement_session_id'
                """,
                (database,),
            )
            nullable_row = cursor.fetchone()

            expected_indexes = {
                "uq_shoe_recommendation_session_shoe": (
                    "shoe_recommendation",
                    "measurement_session_id,shoe_id",
                ),
                "uq_shoe_recommendation_reason_type": (
                    "shoe_recommendation_reason",
                    "shoe_recommendation_id,reason_type",
                ),
                "uq_reason_review": (
                    "shoe_recommendation_reason_review",
                    "reason_id,review_id",
                ),
            }
            observed_indexes: dict[str, tuple[str, str, int]] = {}
            for index_name, (table_name, _) in expected_indexes.items():
                cursor.execute(
                    """
                    SELECT table_name, non_unique,
                           GROUP_CONCAT(column_name ORDER BY seq_in_index)
                    FROM information_schema.statistics
                    WHERE table_schema = %s AND table_name = %s AND index_name = %s
                    GROUP BY table_name, non_unique
                    """,
                    (database, table_name, index_name),
                )
                row = cursor.fetchone()
                if row is not None:
                    observed_indexes[index_name] = (
                        str(row[0]),
                        str(row[2]),
                        int(row[1]),
                    )

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT k.table_name, k.column_name,
                           k.referenced_table_name, k.referenced_column_name,
                           r.delete_rule
                    FROM information_schema.key_column_usage k
                    JOIN information_schema.referential_constraints r
                      ON r.constraint_schema = k.constraint_schema
                     AND r.table_name = k.table_name
                     AND r.constraint_name = k.constraint_name
                    WHERE k.table_schema = %s
                      AND k.referenced_table_name IS NOT NULL
                    GROUP BY k.table_name, k.column_name,
                             k.referenced_table_name, k.referenced_column_name,
                             r.delete_rule
                    HAVING COUNT(*) > 1
                ) duplicate_fks
                """,
                (database,),
            )
            duplicate_fk_count = int(cursor.fetchone()[0])

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT table_name, non_unique, columns_key
                    FROM (
                        SELECT table_name, index_name, non_unique,
                               GROUP_CONCAT(column_name ORDER BY seq_in_index) columns_key
                        FROM information_schema.statistics
                        WHERE table_schema = %s AND index_name <> 'PRIMARY'
                        GROUP BY table_name, index_name, non_unique
                    ) indexes_by_name
                    GROUP BY table_name, non_unique, columns_key
                    HAVING COUNT(*) > 1
                ) duplicate_indexes
                """,
                (database,),
            )
            duplicate_index_count = int(cursor.fetchone()[0])

            cursor.execute(
                """
                SELECT k.constraint_name, r.delete_rule
                FROM information_schema.key_column_usage k
                JOIN information_schema.referential_constraints r
                  ON r.constraint_schema = k.constraint_schema
                 AND r.table_name = k.table_name
                 AND r.constraint_name = k.constraint_name
                WHERE k.table_schema = %s
                  AND k.table_name = 'shoe_lab_metric'
                  AND k.column_name = 'shoe_lab_measurement_id'
                  AND k.referenced_table_name = 'shoe_lab_measurement'
                """,
                (database,),
            )
            lab_fks = tuple((str(row[0]), str(row[1])) for row in cursor.fetchall())
            connection.rollback()
    finally:
        connection.close()

    if nullable_row is None or nullable_row[0] != "NO":
        raise FlywayRehearsalError(
            "V8 did not make recommendation measurement_session_id NOT NULL"
        )
    for index_name, (table_name, columns) in expected_indexes.items():
        if observed_indexes.get(index_name) != (table_name, columns, 0):
            raise FlywayRehearsalError(
                f"required V7/V8 unique index is missing: {index_name}"
            )
    if duplicate_fk_count or duplicate_index_count:
        raise FlywayRehearsalError("V7 left duplicate equivalent FK/index objects")
    if lab_fks != (("fk_shoe_lab_metric_measurement", "CASCADE"),):
        raise FlywayRehearsalError("V7 lab metric delete rule is not canonical")

    return {
        "recommendationMeasurementSessionNotNull": True,
        "requiredUniqueIndexes": sorted(expected_indexes),
        "duplicateEquivalentForeignKeyCount": duplicate_fk_count,
        "duplicateEquivalentIndexCount": duplicate_index_count,
        "labMetricForeignKey": {
            "name": lab_fks[0][0],
            "deleteRule": lab_fks[0][1],
        },
    }


def read_clone_reconciliation_invariants(
    config: MysqlConfig, clone_database: str
) -> dict[str, Any]:
    """Assert the material V7/V8 schema outcomes on a generated clone."""

    assert_safe_clone_name(clone_database, config.database)
    return _read_reconciliation_invariants(config, clone_database)


def read_production_reconciliation_invariants(
    config: MysqlConfig, *, expected_server_uuid: str
) -> dict[str, Any]:
    """Read-only post-migration assertions for the configured production DB.

    The separate production authority executor is responsible for proving the
    exact database identity before this deliberately narrow helper is called.
    """

    result = _read_reconciliation_invariants(
        config,
        config.database,
        expected_server_uuid=expected_server_uuid,
    )
    connection = _connect(config, database=config.database, autocommit=False)
    try:
        with connection.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute("SELECT DATABASE(), @@server_uuid")
            identity = cursor.fetchone()
            if (
                identity is None
                or identity[0] != config.database
                or str(identity[1]).casefold() != expected_server_uuid.casefold()
            ):
                raise FlywayRehearsalError(
                    "recommendation run audit connection identity mismatch"
                )
            cursor.execute(
                """
                SELECT column_name, column_type, is_nullable, column_default, extra
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'shoe_recommendation_run'
                ORDER BY ordinal_position
                """,
                (config.database,),
            )
            columns = tuple(
                (
                    str(row[0]),
                    str(row[1]).casefold(),
                    str(row[2]),
                    None if row[3] is None else str(row[3]),
                    str(row[4]).casefold(),
                )
                for row in cursor.fetchall()
            )
            cursor.execute(
                """
                SELECT index_name, non_unique,
                       GROUP_CONCAT(column_name ORDER BY seq_in_index)
                FROM information_schema.statistics
                WHERE table_schema = %s
                  AND table_name = 'shoe_recommendation_run'
                  AND index_name IN (
                      'uq_shoe_recommendation_run_session',
                      'idx_shoe_recommendation_run_current'
                  )
                GROUP BY index_name, non_unique
                ORDER BY index_name
                """,
                (config.database,),
            )
            indexes = {
                str(row[0]): (int(row[1]), str(row[2])) for row in cursor.fetchall()
            }
            cursor.execute(
                """
                SELECT k.constraint_name, k.referenced_table_name,
                       k.referenced_column_name, r.delete_rule
                FROM information_schema.key_column_usage k
                JOIN information_schema.referential_constraints r
                  ON r.constraint_schema = k.constraint_schema
                 AND r.table_name = k.table_name
                 AND r.constraint_name = k.constraint_name
                WHERE k.table_schema = %s
                  AND k.table_name = 'shoe_recommendation_run'
                  AND k.column_name = 'measurement_session_id'
                """,
                (config.database,),
            )
            run_fks = tuple(
                (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
                for row in cursor.fetchall()
            )
            connection.rollback()
    finally:
        connection.close()

    expected_columns = (
        ("shoe_recommendation_run_id", "bigint", "NO", None, "auto_increment"),
        ("measurement_session_id", "bigint", "NO", None, ""),
        ("status", "varchar(16)", "NO", None, ""),
        ("expected_count", "int", "NO", None, ""),
        ("processed_count", "int", "NO", "0", ""),
        ("started_at", "datetime(6)", "YES", None, ""),
        ("completed_at", "datetime(6)", "YES", None, ""),
        ("failure_detail", "text", "YES", None, ""),
        ("created_at", "datetime(6)", "YES", None, ""),
        ("updated_at", "datetime(6)", "YES", None, ""),
    )
    if columns != expected_columns:
        raise FlywayRehearsalError("V8 recommendation run table shape is not canonical")
    expected_indexes = {
        "uq_shoe_recommendation_run_session": (0, "measurement_session_id"),
        "idx_shoe_recommendation_run_current": (
            1,
            "status,completed_at,measurement_session_id",
        ),
    }
    if indexes != expected_indexes:
        raise FlywayRehearsalError("V8 recommendation run indexes are not canonical")
    if run_fks != (
        (
            "fk_shoe_recommendation_run_session",
            "measurement_session",
            "id",
            "RESTRICT",
        ),
    ) and run_fks != (
        (
            "fk_shoe_recommendation_run_session",
            "measurement_session",
            "id",
            "NO ACTION",
        ),
    ):
        raise FlywayRehearsalError("V8 recommendation run FK is not canonical")
    result["recommendationRun"] = {
        "columnCount": len(columns),
        "uniqueSessionIndex": "uq_shoe_recommendation_run_session",
        "currentLookupIndex": "idx_shoe_recommendation_run_current",
        "measurementSessionForeignKey": run_fks[0][0],
        "deleteRule": run_fks[0][3],
    }
    return result


def validate_post_flyway_history(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> None:
    successful_before = {
        str(row["version"]): row
        for row in before
        if row.get("version") is not None and row.get("success") is True
    }
    successful_after = [
        row
        for row in after
        if row.get("version") is not None and row.get("success") is True
    ]
    versions = [str(row["version"]) for row in successful_after]
    if sorted(versions, key=int) != list(EXPECTED_MIGRATION_VERSIONS):
        raise FlywayRehearsalError(
            "clone Flyway history is not exactly successful V1 through V8"
        )
    if len(set(versions)) != len(versions):
        raise FlywayRehearsalError("clone Flyway history contains duplicate versions")
    if any(row.get("success") is not True for row in after):
        raise FlywayRehearsalError("clone Flyway history still contains a failed row")

    after_by_version = {str(row["version"]): row for row in successful_after}
    stable_fields = ("version", "description", "type", "script", "checksum")
    for version, old in successful_before.items():
        if version not in after_by_version:
            raise FlywayRehearsalError("repair removed a previously successful migration")
        new = after_by_version[version]
        if any(old.get(field) != new.get(field) for field in stable_fields):
            raise FlywayRehearsalError(
                "repair altered metadata for a previously successful migration"
            )


def _safe_extract_boot_runtime(boot_jar: Path, destination: Path) -> tuple[Path, Path]:
    classes = destination / "classes"
    libraries = destination / "lib"
    classes.mkdir(parents=True, exist_ok=False)
    libraries.mkdir(parents=True, exist_ok=False)
    class_prefix = "BOOT-INF/classes/"
    library_prefix = "BOOT-INF/lib/"
    with zipfile.ZipFile(boot_jar.resolve()) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target: Path | None = None
            if member.filename.startswith(class_prefix):
                relative = member.filename[len(class_prefix) :]
                if relative:
                    target = classes / Path(relative)
            elif member.filename.startswith(library_prefix):
                relative = member.filename[len(library_prefix) :]
                if relative and "/" not in relative and relative.endswith(".jar"):
                    target = libraries / relative
            if target is None:
                continue
            resolved_target = target.resolve()
            if destination.resolve() not in resolved_target.parents:
                raise FlywayRehearsalError("unsafe entry in Server boot jar")
            resolved_target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, resolved_target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if not any(libraries.glob("flyway-core-*.jar")):
        raise FlywayRehearsalError("Server runtime does not contain Flyway core")
    if not any(libraries.glob("mysql-connector-j-*.jar")):
        raise FlywayRehearsalError("Server runtime does not contain MySQL driver")
    return classes, libraries


def resolve_java_home(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home))
    javac_on_path = shutil.which("javac")
    if javac_on_path:
        candidates.append(Path(javac_on_path).resolve().parent.parent)
    candidates.extend(
        [
            Path(r"C:\Program Files\JetBrains\IntelliJ IDEA 2026.1\jbr"),
            Path(r"C:\Program Files\Java\jdk-17"),
        ]
    )
    executable = "javac.exe" if os.name == "nt" else "javac"
    java_executable = "java.exe" if os.name == "nt" else "java"
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "bin" / executable).is_file() and (
            resolved / "bin" / java_executable
        ).is_file():
            return resolved
    raise FlywayRehearsalError("Java 17+ JDK with java and javac is required")


def _redact_process_output(
    value: str, config: MysqlConfig, clone_database: str
) -> str:
    redacted = value
    secrets_to_remove = (
        config.password,
        config.username,
        config.clone_jdbc_url(clone_database),
    )
    for secret in secrets_to_remove:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"jdbc:mysql://\S+", "[JDBC_REDACTED]", redacted)
    # Limit diagnostics to a compact tail and remove control characters other
    # than normal line breaks/tabs before it can reach a report or terminal.
    redacted = "".join(
        character
        for character in redacted
        if character in "\n\r\t" or ord(character) >= 32
    )
    return redacted[-4000:].strip()


def _sanitized_process_failure(
    stage: str,
    return_code: int,
    *,
    diagnostic: str = "",
) -> FlywayRehearsalError:
    suffix = f"; diagnostic={diagnostic}" if diagnostic else ""
    return FlywayRehearsalError(
        f"Server-runtime Flyway {stage} failed with exit code {return_code}{suffix}"
    )


def run_server_runtime_flyway(
    *,
    config: MysqlConfig,
    clone_database: str,
    boot_jar: Path,
    java_source: Path = DEFAULT_JAVA_SOURCE,
    java_home: Path | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    assert_safe_clone_name(clone_database, config.database)
    source = java_source.resolve()
    if not source.is_file() or source.is_symlink():
        raise FlywayRehearsalError("Flyway Java helper is missing or unsafe")
    jdk = resolve_java_home(java_home)
    javac = jdk / "bin" / ("javac.exe" if os.name == "nt" else "javac")
    java = jdk / "bin" / ("java.exe" if os.name == "nt" else "java")

    with tempfile.TemporaryDirectory(prefix="feetfit-flyway-runtime-") as temp_name:
        temp = Path(temp_name)
        classes, libraries = _safe_extract_boot_runtime(boot_jar, temp / "runtime")
        helper_classes = temp / "helper-classes"
        helper_classes.mkdir()
        wildcard = str(libraries / "*")
        runtime_classpath = os.pathsep.join((str(classes), wildcard))
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
            diagnostic = _redact_process_output(
                compile_result.stdout + "\n" + compile_result.stderr,
                config,
                clone_database,
            )
            raise _sanitized_process_failure(
                "helper compilation", compile_result.returncode, diagnostic=diagnostic
            )

        result_path = temp / "flyway-result.json"
        run_environment = os.environ.copy()
        run_environment.update(
            {
                "FEETFIT_REHEARSAL_JDBC_URL": config.clone_jdbc_url(clone_database),
                "FEETFIT_REHEARSAL_DB_USERNAME": config.username,
                "FEETFIT_REHEARSAL_DB_PASSWORD": config.password,
                "FEETFIT_REHEARSAL_CLONE_DATABASE": clone_database,
                "FEETFIT_REHEARSAL_PROTECTED_DATABASE": config.database,
                "FEETFIT_REHEARSAL_RESULT_PATH": str(result_path),
            }
        )
        classpath = os.pathsep.join((str(helper_classes), runtime_classpath))
        run_result = subprocess.run(
            [str(java), "-classpath", classpath, "FeetfitFlywayCloneRehearsal"],
            cwd=str(PROJECT_ROOT),
            env=run_environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if not result_path.is_file():
            diagnostic = _redact_process_output(
                run_result.stdout + "\n" + run_result.stderr,
                config,
                clone_database,
            )
            raise _sanitized_process_failure(
                "execution", run_result.returncode, diagnostic=diagnostic
            )
        try:
            document = json.loads(result_path.read_text("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlywayRehearsalError("Flyway helper produced an invalid result") from exc
        if run_result.returncode != 0 or document.get("status") != "PASS":
            stage = str(document.get("stage") or "execution")
            diagnostic = _redact_process_output(
                run_result.stdout + "\n" + run_result.stderr,
                config,
                clone_database,
            )
            raise _sanitized_process_failure(
                stage, run_result.returncode, diagnostic=diagnostic
            )
        expected_keys = {
            "status",
            "stage",
            "migrationsExecuted",
            "pendingBefore",
            "pendingAfter",
            "currentVersion",
            "errorClass",
        }
        if set(document) != expected_keys or document.get("currentVersion") != "8":
            raise FlywayRehearsalError("Flyway helper result contract mismatch")
        return document


def _atomic_json(path: Path, document: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=resolved.parent, prefix=resolved.name + ".", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(resolved)
    return resolved


def _digests_public(values: Mapping[str, TableDigest]) -> dict[str, Any]:
    return {
        table_name: values[table_name].public_dict()
        for table_name in PROTECTED_TABLES
    }


def execute_clone_rehearsal(
    *,
    config: MysqlConfig,
    manifest: MigrationManifest,
    expected_manifest_sha256: str,
    confirmation: str,
    boot_jar: Path = DEFAULT_SERVER_BOOT_JAR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    java_source: Path = DEFAULT_JAVA_SOURCE,
    java_home: Path | None = None,
    now: datetime | None = None,
    nonce: str | None = None,
) -> tuple[dict[str, Any], Path]:
    if expected_manifest_sha256 != manifest.sha256:
        raise FlywayRehearsalError("pinned migration manifest does not match runtime")
    expected_confirmation = confirmation_token(config.database, manifest.sha256)
    if not secrets.compare_digest(confirmation, expected_confirmation):
        raise FlywayRehearsalError("explicit clone-only confirmation is invalid")

    clone_database = generated_clone_name(now=now, nonce=nonce)
    assert_safe_clone_name(clone_database, config.database)
    started_at = datetime.now(timezone.utc)
    output_dir = artifact_root.expanduser().resolve() / clone_database
    if output_dir.exists():
        raise FlywayRehearsalError("rehearsal artifact directory already exists")
    output_dir.mkdir(parents=True)

    source_shapes, charset, collation = _source_schema_metadata(config)
    source_schema_sha256 = _schema_digest(source_shapes)
    source_history = read_flyway_history(config, config.database)

    _create_clone_schema(
        config,
        clone_database,
        charset=charset,
        collation=collation,
        shapes=source_shapes,
    )
    copied_counts, source_digests, snapshot_at = _copy_snapshot_data(
        config, clone_database, source_shapes
    )

    clone_shapes_before = _read_clone_shapes(config, clone_database)
    if _schema_digest(clone_shapes_before) != source_schema_sha256:
        raise FlywayRehearsalError("clone schema differs before Flyway execution")
    clone_counts_before = _table_counts(
        config, clone_database, clone_shapes_before.keys()
    )
    if clone_counts_before != copied_counts:
        raise FlywayRehearsalError("clone row counts differ immediately after copy")
    clone_digests_before = read_protected_digests(
        config, clone_database, source_shapes
    )
    if clone_digests_before != source_digests:
        raise FlywayRehearsalError("protected table digest differs before Flyway")
    clone_history_before = read_flyway_history(config, clone_database)
    if clone_history_before != source_history:
        raise FlywayRehearsalError("Flyway history was not cloned exactly")

    flyway_result = run_server_runtime_flyway(
        config=config,
        clone_database=clone_database,
        boot_jar=boot_jar,
        java_source=java_source,
        java_home=java_home,
    )

    clone_history_after = read_flyway_history(config, clone_database)
    validate_post_flyway_history(clone_history_before, clone_history_after)
    reconciliation_invariants = read_clone_reconciliation_invariants(
        config, clone_database
    )
    clone_digests_after = read_protected_digests(
        config, clone_database, source_shapes
    )
    if clone_digests_after != source_digests:
        raise FlywayRehearsalError("protected table digest changed during Flyway")

    # Re-read the protected source after all clone writes and Flyway work.  This
    # is intentionally separate from the consistent baseline snapshot: it is
    # direct evidence that the source schema/data/history remained untouched.
    source_shapes_after = _source_shapes(config)
    if _schema_digest(source_shapes_after) != source_schema_sha256:
        raise FlywayRehearsalError("source schema changed during clone rehearsal")
    source_digests_after = read_protected_digests(
        config, config.database, source_shapes_after
    )
    if source_digests_after != source_digests:
        raise FlywayRehearsalError("source protected data changed during clone rehearsal")
    source_history_after = read_flyway_history(config, config.database)
    if source_history_after != source_history:
        raise FlywayRehearsalError("source Flyway history changed during clone rehearsal")

    completed_at = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "status": "PASS",
        "mode": "EXECUTE_CLONE_ONLY",
        "cloneDatabase": clone_database,
        "sourceDatabaseFingerprint": hashlib.sha256(
            config.database.casefold().encode("utf-8")
        ).hexdigest()[:16],
        "sourceSnapshotAt": snapshot_at,
        "startedAt": started_at.isoformat(),
        "completedAt": completed_at.isoformat(),
        "executionTimeSeconds": round(
            (completed_at - started_at).total_seconds(), 3
        ),
        "migrationManifest": manifest.public_dict(),
        "sourceSchemaSha256": source_schema_sha256,
        "tableCount": len(source_shapes),
        "allTableCountsBeforeFlyway": clone_counts_before,
        "protectedTables": {
            "sourceSnapshot": _digests_public(source_digests),
            "sourceAfterFlyway": _digests_public(source_digests_after),
            "cloneBeforeFlyway": _digests_public(clone_digests_before),
            "cloneAfterFlyway": _digests_public(clone_digests_after),
            "allCountsAndShaPreserved": True,
        },
        "flyway": {
            "operationOrder": ["repair", "validate", "migrate", "validate"],
            "target": "8",
            "runtimeResult": flyway_result,
            "historyBefore": clone_history_before,
            "historyAfter": clone_history_after,
            "reconciliationInvariants": reconciliation_invariants,
        },
        "safety": {
            "sourceTransaction": "CONSISTENT_SNAPSHOT_READ_ONLY",
            "generatedCloneOnly": True,
            "productionDatabaseWrites": False,
            "cloneRetained": True,
            "credentialsIncludedInReport": False,
            "cleanupImplemented": False,
        },
    }
    report_path = _atomic_json(output_dir / "rehearsal-report.json", report)
    return report, report_path
