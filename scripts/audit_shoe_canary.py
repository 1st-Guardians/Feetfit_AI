"""Run the verified canary DB/API audit using read-only adapters only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.shoe.canary_ingestion_audit import (
    DEFAULT_DRY_RUN,
    DEFAULT_EXECUTION_STATE,
    CanaryAuditError,
    MySqlReadOnlyAuditReader,
    create_audit,
    http_reader_from_environment,
    load_canary_expectation,
    load_verified_audit,
    write_atomic_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-read-only",
        action="store_true",
        help="required safety latch; enables DB SELECTs and loopback HTTP GETs only",
    )
    parser.add_argument("--dry-run-dir", default=str(DEFAULT_DRY_RUN))
    parser.add_argument("--execution-state", default=str(DEFAULT_EXECUTION_STATE))
    parser.add_argument("--server-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--prior-audit")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.live_read_only:
        raise CanaryAuditError(
            "--live-read-only is required; this command still performs no writes"
        )
    expectation = load_canary_expectation(
        dry_run_dir=args.dry_run_dir,
        execution_state_path=args.execution_state,
    )
    prior = None if args.prior_audit is None else load_verified_audit(args.prior_audit)
    audit = create_audit(
        expectation=expectation,
        db_reader=MySqlReadOnlyAuditReader(),
        api_reader=http_reader_from_environment(
            base_url=args.server_base_url, user_id=args.user_id
        ),
        phase=args.phase,
        prior_audit=prior,
    )
    output = write_atomic_audit(Path(args.output), audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "phase": audit["phase"],
                "output": str(output),
                "counts": audit["database"]["counts"],
                "duplicateCounts": audit["database"]["duplicateCounts"],
                "orphanCounts": audit["database"]["orphanCounts"],
                "idempotencyStatus": audit["idempotency"]["status"],
                "issueCount": len(audit["issues"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
