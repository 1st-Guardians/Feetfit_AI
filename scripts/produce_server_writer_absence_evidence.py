"""Produce audited read-only Server/MySQL writer-absence evidence.

This script never starts/stops processes, changes listeners, or writes to the
database.  It writes only a timestamped local audit JSON.  A FAIL document is
retained for diagnosis but cannot authorize the production executor.
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
    sha256_file,
)
from app.services.shoe.writer_absence_evidence import (
    DEFAULT_ARTIFACT_ROOT,
    write_writer_absence_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document, output = write_writer_absence_evidence(
        mysql_config_from_settings(), artifact_root=Path(args.artifact_root)
    )
    summary = {
        "status": document["status"],
        "checkedAt": document["checkedAt"],
        "sourceDatabase": document["sourceDatabase"],
        "serverWriterAbsent": document["serverWriterAbsent"],
        "activeServerWriterProcessCount": len(
            document["activeServerWriterProcessIds"]
        ),
        "activeDatabaseWriterSessionCount": len(
            document["activeDatabaseWriterSessionIds"]
        ),
        "mysqlProcessPrivilegeVerified": document[
            "mysqlProcessPrivilegeVerified"
        ],
        "evidenceSha256": sha256_file(output),
        "evidence": str(output),
        "databaseCredentialsIncluded": False,
        "databaseWrites": False,
        "processMutations": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if document["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FlywayRehearsalError as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": str(error),
                    "databaseCredentialsIncluded": False,
                    "databaseWrites": False,
                    "processMutations": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
