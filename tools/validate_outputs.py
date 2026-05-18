#!/usr/bin/env python3
"""Validate generated ASTEROID-NEO report and table artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asteroid_neo as neo  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ASTEROID-NEO output bundle schemas.")
    parser.add_argument("bundle", type=Path, help="Output directory containing report.json and generated table_*.csv files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    bundle = args.bundle.expanduser().resolve()
    report_path = bundle / "report.json"
    neo.validate_report_file(report_path)
    csv_paths = sorted(bundle.glob("table_*.csv"))
    if not csv_paths:
        raise neo.SchemaValidationError(f"No table_*.csv files found in {bundle}")
    for path in csv_paths:
        neo.validate_csv_table(path)
    print(f"validated {report_path}")
    print(f"validated {len(csv_paths)} CSV tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
