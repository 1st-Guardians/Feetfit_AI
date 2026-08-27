"""Plan or explicitly execute the pinned FeetFit Server V1..V8 production migration.

Default mode is a read-only authority audit.  Execution is impossible without
all hashes and the derived confirmation emitted by a fresh, writer-free plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.shoe.flyway_clone_rehearsal import (
    FlywayRehearsalError,
    mysql_config_from_settings,
)
from app.services.shoe.flyway_production_apply import (
    DEFAULT_ARTIFACT_ROOT,
    MAINTENANCE_WINDOW_AUTHORITY,
    ProductionAuthorityError,
    build_production_plan,
    execute_production_migration,
    load_current_pinned_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="cross the production mutation boundary; absent means read-only plan",
    )
    parser.add_argument(
        "--writer-absence-evidence",
        help="fresh externally produced JSON proving Server/DB writers are absent",
    )
    parser.add_argument("--expected-writer-evidence-sha256")
    parser.add_argument("--expected-migration-manifest-sha256")
    parser.add_argument("--expected-boot-jar-sha256")
    parser.add_argument("--expected-clone-report-sha256")
    parser.add_argument("--confirmation")
    parser.add_argument(
        "--maintenance-window-authority",
        help=(
            "exact operational acknowledgment from plan: sole writer host and "
            "writer restart inhibited for the full migration"
        ),
    )
    parser.add_argument("--java-home")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser


def _required_execute_value(value: str | None, option: str) -> str:
    if not value:
        raise ProductionAuthorityError(f"--execute requires {option}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = mysql_config_from_settings()
    manifest = load_current_pinned_manifest()
    evidence_path = (
        Path(args.writer_absence_evidence)
        if args.writer_absence_evidence
        else None
    )
    if not args.execute:
        plan = build_production_plan(
            config=config,
            manifest=manifest,
            writer_evidence_path=evidence_path,
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if evidence_path is None:
        raise ProductionAuthorityError(
            "--execute requires --writer-absence-evidence from an external check"
        )
    report, report_path = execute_production_migration(
        config=config,
        manifest=manifest,
        expected_manifest_sha256=_required_execute_value(
            args.expected_migration_manifest_sha256,
            "--expected-migration-manifest-sha256",
        ),
        expected_boot_jar_sha256=_required_execute_value(
            args.expected_boot_jar_sha256, "--expected-boot-jar-sha256"
        ),
        expected_clone_report_sha256=_required_execute_value(
            args.expected_clone_report_sha256, "--expected-clone-report-sha256"
        ),
        writer_evidence_path=evidence_path,
        expected_writer_evidence_sha256=_required_execute_value(
            args.expected_writer_evidence_sha256,
            "--expected-writer-evidence-sha256",
        ),
        confirmation=_required_execute_value(args.confirmation, "--confirmation"),
        maintenance_window_authority=_required_execute_value(
            args.maintenance_window_authority,
            "--maintenance-window-authority " + MAINTENANCE_WINDOW_AUTHORITY,
        ),
        artifact_root=Path(args.artifact_root),
        java_home=Path(args.java_home) if args.java_home else None,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "targetDatabase": report["targetDatabase"],
                "targetFingerprint": report["targetFingerprint"],
                "migrationManifestSha256": report["migrationManifestSha256"],
                "successfulCloneReportSha256": report[
                    "successfulCloneReportSha256"
                ],
                "flywayRuntimeResult": report["flywayRuntimeResult"],
                "reconciliationInvariants": report["reconciliationInvariants"],
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
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "INDETERMINATE",
                    "error": "execution interrupted; inspect durable ARMED report",
                    "manualInspectionRequired": True,
                    "automaticRetry": False,
                    "automaticRollback": False,
                    "credentialsIncluded": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(130)
    except FlywayRehearsalError as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": str(error),
                    "credentialsIncluded": False,
                    "automaticRetry": False,
                    "automaticRollback": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
