"""Plan or execute Server Flyway V1..V8 against a timestamped MySQL clone only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.shoe.flyway_clone_rehearsal import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_JAVA_SOURCE,
    DEFAULT_SERVER_BOOT_JAR,
    DEFAULT_SERVER_ROOT,
    FlywayRehearsalError,
    build_plan,
    execute_clone_rehearsal,
    load_migration_manifest,
    mysql_config_from_settings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="create a generated clone and run Flyway; absent means plan-only",
    )
    parser.add_argument(
        "--expected-migration-manifest-sha256",
        help="required execute-time pin from the plan output",
    )
    parser.add_argument(
        "--confirmation",
        help="required execute-time clone-only token from the plan output",
    )
    parser.add_argument("--server-root", default=str(DEFAULT_SERVER_ROOT))
    parser.add_argument("--server-boot-jar", default=str(DEFAULT_SERVER_BOOT_JAR))
    parser.add_argument("--java-source", default=str(DEFAULT_JAVA_SOURCE))
    parser.add_argument("--java-home")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = mysql_config_from_settings()
    manifest = load_migration_manifest(
        server_root=Path(args.server_root), boot_jar=Path(args.server_boot_jar)
    )
    plan = build_plan(config, manifest)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if not args.expected_migration_manifest_sha256 or not args.confirmation:
        raise FlywayRehearsalError(
            "--execute requires both manifest pin and explicit confirmation from plan"
        )
    report, report_path = execute_clone_rehearsal(
        config=config,
        manifest=manifest,
        expected_manifest_sha256=args.expected_migration_manifest_sha256,
        confirmation=args.confirmation,
        boot_jar=Path(args.server_boot_jar),
        artifact_root=Path(args.artifact_root),
        java_source=Path(args.java_source),
        java_home=Path(args.java_home) if args.java_home else None,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "cloneDatabase": report["cloneDatabase"],
                "migrationManifestSha256": report["migrationManifest"]["sha256"],
                "protectedTables": report["protectedTables"],
                "flyway": report["flyway"]["runtimeResult"],
                "productionDatabaseWrites": report["safety"][
                    "productionDatabaseWrites"
                ],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FlywayRehearsalError as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": str(error),
                    "credentialsIncluded": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
