"""Replay the crawler's authoritative ingestion loader and emit safe evidence.

This bridge intentionally uses only the Python standard library after importing
the crawler.  It is launched with the crawler virtual environment because the
Feetfit_AI environment does not carry the crawler's parsing dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FORMAT = "feetfit-authoritative-ingestion-loader-replay"
VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawler-root", required=True)
    parser.add_argument("--dry-run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    crawler_root = Path(args.crawler_root).expanduser().resolve()
    dry_run_dir = Path(args.dry_run_dir).expanduser().resolve()
    if (
        crawler_root.is_symlink()
        or not crawler_root.is_dir()
        or dry_run_dir.is_symlink()
        or not dry_run_dir.is_dir()
    ):
        raise RuntimeError("authoritative crawler/dry-run path is missing or unsafe")

    sys.path.insert(0, str(crawler_root))
    from verified_ingestion_runner import (
        _full_operation_policy_document,
        _load_full_operation_policy,
        load_prepared_ingestion,
    )

    prepared = load_prepared_ingestion(dry_run_bundle=dry_run_dir)
    full_policy = _load_full_operation_policy(prepared)
    document = {
        "format": FORMAT,
        "version": VERSION,
        "dryRunRoot": str(prepared.dry_run.root),
        "dryRunManifestSha256": prepared.dry_run.manifest_sha256,
        "selectionManifestSha256": prepared.eligibility.source_manifest_sha256,
        "finalAuditManifestSha256": prepared.eligibility.audit_manifest_sha256,
        "readyGoodsNos": list(prepared.dry_run.ready_goods_nos),
        "canaryGoodsNos": list(prepared.canary.goods_nos),
        "canaryExecutionStateSha256": full_policy.canary_state_sha256,
        "canaryShoeIdsByGoodsNo": dict(full_policy.canary_shoe_ids),
        "fullOperationPolicy": _full_operation_policy_document(full_policy),
    }
    print(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
