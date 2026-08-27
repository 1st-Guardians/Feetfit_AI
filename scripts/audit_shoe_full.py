"""Run the fixed 338-shoe DB/API audit with SELECT and GET only."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_default(value: object) -> str:
    """Preserve exact decimal audit values in the compact CLI summary."""
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

from app.services.shoe.full_ingestion_audit import (
    DEFAULT_DRY_RUN,
    DEFAULT_EXECUTION_STATE,
    FullAuditError,
    HttpGetAuditReader,
    MySqlReadOnlyAuditReader,
    create_audit,
    load_full_expectation,
    write_atomic_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-read-only",
        action="store_true",
        help="required safety latch; permits DB SELECTs and loopback HTTP GETs only",
    )
    parser.add_argument("--dry-run-dir", default=str(DEFAULT_DRY_RUN))
    parser.add_argument("--execution-state", default=str(DEFAULT_EXECUTION_STATE))
    parser.add_argument("--server-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--api-concurrency", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.live_read_only:
        raise FullAuditError(
            "--live-read-only is required; this command still performs no DB/API writes"
        )
    expectation = load_full_expectation(
        dry_run_dir=args.dry_run_dir,
        execution_state_path=args.execution_state,
    )
    with HttpGetAuditReader(
        base_url=args.server_base_url,
        timeout_seconds=args.timeout_seconds,
        max_connections=args.api_concurrency,
    ) as api_reader:
        audit = create_audit(
            expectation=expectation,
            db_reader=MySqlReadOnlyAuditReader(),
            api_reader=api_reader,
            api_max_concurrency=args.api_concurrency,
        )
    output = write_atomic_audit(Path(args.output), audit)
    database = audit["database"]
    api = audit["api"]
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(output),
                "execution": audit["execution"],
                "counts": database["counts"],
                "duplicateCounts": database["duplicateCounts"],
                "orphanCounts": database["orphanCounts"],
                "wrongRunRepeatTargetCount": database[
                    "wrongRunRepeatTargetCount"
                ],
                "dryRunDifferenceCounts": database["dryRunDifferenceCounts"],
                "apiAuditedShoeCount": api["auditedShoeCount"],
                "apiHttp200Counts": api["http200Counts"],
                "apiSummaryNonblankCount": api["summaryNonblankCount"],
                "issueCount": len(audit["issues"]),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
