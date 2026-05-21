#!/usr/bin/env python3
"""Plan, run, and gate multi-NEO prediction benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkTarget:
    label: str
    target: str
    date_min: str
    date_max: str
    dist_max: str = "0.5"


@dataclass(frozen=True)
class BenchmarkProfile:
    label: str
    horizons_step: str
    refine_step: str
    refine_window_days: float
    integrator_max_step_days: float
    integrator_rtol: float
    integrator_atol: float
    uncertainty_samples: int
    state_refresh_days: float = 0.0
    prediction_mode: str = "single_arc_predictor"


@dataclass(frozen=True)
class BenchmarkThresholds:
    min_completed_targets: int = 5
    single_arc_validation_rmse_km_max: float = 500_000.0
    single_arc_nearest_cad_error_km_max: float = 5_000.0
    single_arc_cad_anchor_rmse_km_max: float = 25_000.0
    reconstruction_validation_rmse_km_max: float = 100.0
    reconstruction_nearest_cad_error_km_max: float = 5_000.0
    reconstruction_cad_anchor_rmse_km_max: float = 2_000.0
    sensitivity_rmse_relative_drift_max: float = 0.15
    sensitivity_cad_anchor_rmse_relative_drift_max: float = 0.10


DEFAULT_TARGETS = [
    BenchmarkTarget("apophis", "99942", "2026-04-27", "2030-12-31"),
    BenchmarkTarget("bennu", "101955", "2026-01-01", "2032-12-31"),
    BenchmarkTarget("didymos", "65803", "2026-01-01", "2032-12-31"),
    BenchmarkTarget("ryugu", "162173", "2026-01-01", "2032-12-31"),
    BenchmarkTarget("phaethon", "3200", "2026-01-01", "2032-12-31"),
    BenchmarkTarget("eros", "433", "2026-01-01", "2032-12-31"),
    BenchmarkTarget("itokawa", "25143", "2026-01-01", "2032-12-31"),
]


DEFAULT_PROFILES = [
    BenchmarkProfile(
        label="single_arc_fast",
        horizons_step="1d",
        refine_step="1h",
        refine_window_days=5.0,
        integrator_max_step_days=0.125,
        integrator_rtol=1e-11,
        integrator_atol=1e-13,
        uncertainty_samples=128,
    ),
    BenchmarkProfile(
        label="reconstruct_30d",
        horizons_step="1d",
        refine_step="1h",
        refine_window_days=5.0,
        integrator_max_step_days=0.125,
        integrator_rtol=1e-11,
        integrator_atol=1e-13,
        uncertainty_samples=128,
        state_refresh_days=30.0,
        prediction_mode="osculating_reconstruction",
    ),
]


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _pass_metric(value: float, limit: float) -> bool | str:
    if not _finite(value):
        return "n/a"
    return bool(value <= limit)


def _profile_command(target: BenchmarkTarget, profile: BenchmarkProfile, output_root: Path) -> tuple[Path, list[str]]:
    output_dir = output_root / _slug(target.label) / profile.label
    command = [
        sys.executable,
        str(ROOT / "astro.py"),
        "--target",
        target.target,
        "--date-min",
        target.date_min,
        "--date-max",
        target.date_max,
        "--dist-max",
        target.dist_max,
        "--dynamics",
        "--horizons-step",
        profile.horizons_step,
        "--refine-step",
        profile.refine_step,
        "--refine-window-days",
        f"{profile.refine_window_days:g}",
        "--uncertainty-samples",
        str(profile.uncertainty_samples),
        "--dynamics-frame",
        "barycentric",
        "--integrator-method",
        "DOP853",
        "--integrator-max-step-days",
        f"{profile.integrator_max_step_days:g}",
        "--integrator-rtol",
        f"{profile.integrator_rtol:.3g}",
        "--integrator-atol",
        f"{profile.integrator_atol:.3g}",
        "--plot-dir",
        str(output_dir),
        "--json",
        str(output_dir / "report.json"),
    ]
    if profile.state_refresh_days > 0.0:
        command.extend(["--state-refresh-days", f"{profile.state_refresh_days:g}"])
    return output_dir, command


def build_manifest(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in DEFAULT_TARGETS:
        for profile in DEFAULT_PROFILES:
            output_dir, command = _profile_command(target, profile, output_root)
            rows.append(
                {
                    "target_label": target.label,
                    "target": target.target,
                    "date_min": target.date_min,
                    "date_max": target.date_max,
                    "dist_max": target.dist_max,
                    "profile": profile.label,
                    "prediction_mode": profile.prediction_mode,
                    "state_refresh_days": profile.state_refresh_days,
                    "output_dir": str(output_dir),
                    "command": command,
                }
            )
    return rows


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "benchmark_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_root / "benchmark_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "target_label",
            "target",
            "date_min",
            "date_max",
            "dist_max",
            "profile",
            "prediction_mode",
            "state_refresh_days",
            "output_dir",
            "command",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_thresholds(path: Path | None) -> BenchmarkThresholds:
    if path is None:
        return BenchmarkThresholds()
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    valid = {field.name for field in BenchmarkThresholds.__dataclass_fields__.values()}
    unknown = sorted(set(data).difference(valid))
    if unknown:
        raise ValueError(f"unknown threshold key(s): {', '.join(unknown)}")
    return BenchmarkThresholds(**data)


def _report_row(bundle: Path) -> dict[str, Any]:
    report_path = bundle / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dynamics = report.get("dynamics", {})
    diagnostics = dynamics.get("numerical_diagnostics", {})
    state_refresh_count = _float_or_nan(diagnostics.get("state_refresh_count", 0.0))
    mode = str(diagnostics.get("prediction_mode") or "")
    if not mode:
        mode = "osculating_reconstruction" if state_refresh_count > 0.0 else "single_arc_predictor"
    return {
        "bundle": str(bundle),
        "target": str(report.get("target", "")),
        "object": str(report.get("object", {}).get("fullname", "")),
        "prediction_mode": mode,
        "n_samples": int(dynamics.get("n_samples", 0) or 0),
        "horizons_step": dynamics.get("horizons_step", ""),
        "state_refresh_count": state_refresh_count,
        "state_refresh_segment_days": _float_or_nan(diagnostics.get("state_refresh_segment_days")),
        "validation_mae_km": _float_or_nan(dynamics.get("validation_mae_km")),
        "validation_rmse_km": _float_or_nan(dynamics.get("validation_rmse_km")),
        "nearest_cad_error_km": _float_or_nan(dynamics.get("cad_validation_error_km")),
        "cad_anchor_rmse_km": _float_or_nan(diagnostics.get("cad_anchor_integrated_rmse_km")),
    }


def collect_report_rows(output_root: Path, bundles: list[Path]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for report_path in sorted(output_root.expanduser().glob("**/report.json")):
        bundle = report_path.parent.resolve()
        if bundle not in seen:
            seen.add(bundle)
            paths.append(bundle)
    for bundle in bundles:
        resolved = bundle.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)
    return [_report_row(path) for path in paths if (path / "report.json").exists()]


def _row_thresholds(row: dict[str, Any], thresholds: BenchmarkThresholds) -> list[dict[str, Any]]:
    mode = str(row["prediction_mode"])
    if mode == "osculating_reconstruction":
        checks = [
            ("reconstruction_error", row["validation_rmse_km"], thresholds.reconstruction_validation_rmse_km_max),
            ("reconstruction_cad_anchor_residual", row["cad_anchor_rmse_km"], thresholds.reconstruction_cad_anchor_rmse_km_max),
            ("reconstruction_nearest_cad_error", abs(row["nearest_cad_error_km"]), thresholds.reconstruction_nearest_cad_error_km_max),
        ]
    else:
        checks = [
            ("single_arc_forecast_error", row["validation_rmse_km"], thresholds.single_arc_validation_rmse_km_max),
            ("single_arc_cad_anchor_residual", row["cad_anchor_rmse_km"], thresholds.single_arc_cad_anchor_rmse_km_max),
            ("single_arc_nearest_cad_error", abs(row["nearest_cad_error_km"]), thresholds.single_arc_nearest_cad_error_km_max),
        ]
    out: list[dict[str, Any]] = []
    for name, value, limit in checks:
        out.append(
            {
                "scope": row["bundle"],
                "target": row["target"],
                "mode": mode,
                "check": name,
                "value": value,
                "limit": limit,
                "passed": _pass_metric(value, limit),
            }
        )
    return out


def _relative_spread(values: list[float]) -> float:
    finite = [abs(float(value)) for value in values if _finite(float(value))]
    if len(finite) < 2:
        return 0.0
    center = max(float(sorted(finite)[len(finite) // 2]), 1e-12)
    return (max(finite) - min(finite)) / center


def _drift_thresholds(rows: list[dict[str, Any]], thresholds: BenchmarkThresholds) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target"]), str(row["prediction_mode"])), []).append(row)

    checks: list[dict[str, Any]] = []
    for (target, mode), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        rmse_drift = _relative_spread([row["validation_rmse_km"] for row in group])
        anchor_drift = _relative_spread([row["cad_anchor_rmse_km"] for row in group])
        checks.extend(
            [
                {
                    "scope": f"{target}:{mode}",
                    "target": target,
                    "mode": mode,
                    "check": "sensitivity_rmse_relative_drift",
                    "value": rmse_drift,
                    "limit": thresholds.sensitivity_rmse_relative_drift_max,
                    "passed": bool(rmse_drift <= thresholds.sensitivity_rmse_relative_drift_max),
                },
                {
                    "scope": f"{target}:{mode}",
                    "target": target,
                    "mode": mode,
                    "check": "sensitivity_cad_anchor_rmse_relative_drift",
                    "value": anchor_drift,
                    "limit": thresholds.sensitivity_cad_anchor_rmse_relative_drift_max,
                    "passed": bool(anchor_drift <= thresholds.sensitivity_cad_anchor_rmse_relative_drift_max),
                },
            ]
        )
    return checks


def evaluate_thresholds(
    rows: list[dict[str, Any]],
    thresholds: BenchmarkThresholds,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    targets = sorted({str(row["target"]) for row in rows if row.get("target")})
    if not allow_partial:
        passed = len(targets) >= thresholds.min_completed_targets
        checks.append(
            {
                "scope": "benchmark_suite",
                "target": ",".join(targets),
                "mode": "all",
                "check": "completed_target_count",
                "value": len(targets),
                "limit": thresholds.min_completed_targets,
                "passed": bool(passed),
            }
        )
    for row in rows:
        checks.extend(_row_thresholds(row, thresholds))
    checks.extend(_drift_thresholds(rows, thresholds))
    failed = [check for check in checks if check["passed"] is False]
    return checks, not failed


def write_evaluation(output_root: Path, rows: list[dict[str, Any]], checks: list[dict[str, Any]], thresholds: BenchmarkThresholds) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "benchmark_summary.csv"
    checks_path = output_root / "threshold_report.csv"
    json_path = output_root / "threshold_report.json"
    md_path = output_root / "threshold_report.md"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if checks:
        with checks_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(checks[0].keys()))
            writer.writeheader()
            writer.writerows(checks)
    json_path.write_text(
        json.dumps(
            {
                "thresholds": asdict(thresholds),
                "rows": rows,
                "checks": checks,
                "passed": all(check["passed"] is not False for check in checks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# NEO Benchmark Threshold Report\n\n")
        fh.write(f"Overall: {'PASS' if all(check['passed'] is not False for check in checks) else 'FAIL'}\n\n")
        fh.write("| target | mode | check | value | limit | passed |\n")
        fh.write("| --- | --- | --- | --- | --- | --- |\n")
        for check in checks:
            fh.write(
                f"| {check['target']} | {check['mode']} | {check['check']} | "
                f"{check['value']} | {check['limit']} | {check['passed']} |\n"
            )
    print(f"wrote {summary_path}")
    print(f"wrote {checks_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ASTEROID-NEO across multiple NEOs and hard thresholds.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "neo_benchmarks")
    parser.add_argument("--thresholds", type=Path, help="Optional JSON file overriding benchmark thresholds.")
    parser.add_argument("--execute", action="store_true", help="Run the planned benchmark commands.")
    parser.add_argument("--dry-run", action="store_true", help="Write and print the benchmark manifest only.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate completed benchmark report.json files.")
    parser.add_argument("--bundle", action="append", type=Path, default=[], help="Additional output bundle to evaluate.")
    parser.add_argument("--include-local-outputs", action="store_true", help="Evaluate the existing local Apophis output bundles.")
    parser.add_argument("--allow-partial", action="store_true", help="Do not fail when fewer than the configured target count is complete.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing threshold reports.")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit planned/executed benchmark runs; 0 means all.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    thresholds = load_thresholds(args.thresholds)
    manifest = build_manifest(args.output_root)
    if args.max_runs > 0:
        manifest = manifest[: args.max_runs]

    if args.evaluate:
        bundles = list(args.bundle)
        if args.include_local_outputs:
            bundles.extend(
                [
                    ROOT / "outputs" / "predictor_full",
                    ROOT / "outputs" / "predictor_high_samples",
                    ROOT / "outputs" / "predictor_3x_samples",
                    ROOT / "outputs" / "predictor_refresh30",
                ]
            )
        rows = collect_report_rows(args.output_root, bundles)
        checks, passed = evaluate_thresholds(rows, thresholds, allow_partial=args.allow_partial)
        write_evaluation(args.output_root, rows, checks, thresholds)
        if not rows:
            print("no completed benchmark reports found", file=sys.stderr)
            return 2 if not args.no_fail else 0
        return 0 if (passed or args.no_fail) else 2

    write_manifest(args.output_root, manifest)
    print(f"planned {len(manifest)} benchmark runs under {args.output_root}")
    for row in manifest:
        print(" ".join(row["command"]))
    if not args.execute:
        return 0
    for row in manifest:
        print(f"\n=== running {row['target_label']} / {row['profile']} ===")
        subprocess.run(row["command"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
