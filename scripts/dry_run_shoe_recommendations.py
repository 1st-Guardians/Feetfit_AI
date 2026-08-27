from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.shoe.shoe_recommendation import compute_shoe_recommendations
from app.services.shoe.shoe_recommendation_dry_run import build_dry_run_report
from app.services.shoe.shoe_server_client import ShoeServerClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a completed Server session and validate Phase D without writes."
    )
    parser.add_argument("--dry-run", action="store_true", help="required read-only safety latch")
    parser.add_argument("--measurement-session-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def execute(measurement_session_id: int, authorization_header: str) -> dict[str, object]:
    client = ShoeServerClient(authorization_header)
    context = await client.fetch_recommendation_context(measurement_session_id)
    batch = await asyncio.to_thread(compute_shoe_recommendations, context)
    return build_dry_run_report(context, batch)


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        print("Refusing to run without explicit --dry-run.", file=sys.stderr)
        return 2
    if args.measurement_session_id < 1:
        print("measurement-session-id must be positive.", file=sys.stderr)
        return 2
    authorization_header = os.environ.get("FEETFIT_DRY_RUN_AUTHORIZATION", "").strip()
    if not authorization_header.startswith("Bearer "):
        print(
            "FEETFIT_DRY_RUN_AUTHORIZATION must contain a Bearer header; it is never printed.",
            file=sys.stderr,
        )
        return 2
    try:
        report = asyncio.run(execute(args.measurement_session_id, authorization_header))
    except Exception as exc:
        print(f"Phase D dry-run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
