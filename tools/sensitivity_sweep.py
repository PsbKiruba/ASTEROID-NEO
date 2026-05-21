#!/usr/bin/env python3
"""Prepare or run sensitivity sweeps for dynamics cadence and integrator settings."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepRun:
    name: str
    horizons_step: str
    refine_step: str
    refine_window_days: float
    integrator_max_step_days: float
    integrator_rtol: float
    integrator_atol: float
    uncertainty_samples: int
    state_refresh_days: float = 0.0
    post_encounter_reset_days: float = 0.0
    profile: str = "single_arc_predictor"

    def output_dir(self, root: Path) -> Path:
        return root / self.name

    def command(self, output_root: Path, target: str, date_min: str, date_max: str, dist_max: str) -> list[str]:
        out = self.output_dir(output_root)
        command = [
            sys.executable,
            str(ROOT / "astro.py"),
            "--target",
            target,
            "--date-min",
            date_min,
            "--date-max",
            date_max,
            "--dist-max",
            dist_max,
            "--dynamics",
            "--horizons-step",
            self.horizons_step,
            "--refine-step",
            self.refine_step,
            "--refine-window-days",
            f"{self.refine_window_days:g}",
            "--uncertainty-samples",
            str(self.uncertainty_samples),
            "--dynamics-frame",
            "barycentric",
            "--integrator-method",
            "DOP853",
            "--integrator-max-step-days",
            f"{self.integrator_max_step_days:g}",
            "--integrator-rtol",
            f"{self.integrator_rtol:.3g}",
            "--integrator-atol",
            f"{self.integrator_atol:.3g}",
            "--plot-dir",
            str(out),
            "--json",
            str(out / "report.json"),
        ]
        if self.state_refresh_days > 0.0:
            command.extend(["--state-refresh-days", f"{self.state_refresh_days:g}"])
        if self.post_encounter_reset_days > 0.0:
            command.extend(["--post-encounter-reset-days", f"{self.post_encounter_reset_days:g}"])
        return command


def build_runs(include_reconstruction: bool = False) -> list[SweepRun]:
    cadence_grid = [
        ("12h", "1h", 5.0),
        ("6h", "30m", 5.0),
        ("3h", "15m", 5.0),
    ]
    tolerance_grid = [
        (0.25, 1e-10, 1e-12),
        (0.125, 1e-11, 1e-13),
        (0.0625, 2e-12, 2e-14),
    ]
    uncertainty_samples = [384, 768, 1152]
    runs: list[SweepRun] = []
    for i, ((h_step, r_step, window), (max_step, rtol, atol), samples) in enumerate(
        itertools.product(cadence_grid, tolerance_grid, uncertainty_samples),
        start=1,
    ):
        runs.append(
            SweepRun(
                name=f"sweep_{i:02d}_{h_step}_{r_step}_tol{rtol:.0e}_u{samples}".replace("+", ""),
                horizons_step=h_step,
                refine_step=r_step,
                refine_window_days=window,
                integrator_max_step_days=max_step,
                integrator_rtol=rtol,
                integrator_atol=atol,
                uncertainty_samples=samples,
            )
        )
    if include_reconstruction:
        runs.extend(build_reconstruction_runs(start=len(runs) + 1))
    return runs


def build_reconstruction_runs(start: int = 1) -> list[SweepRun]:
    """Return declared multi-arc reconstruction profiles for broad residual control."""
    specs = [
        ("30d", 30.0, "1d", "1h", 5.0, 0.125, 1e-11, 1e-13, 128),
        ("90d", 90.0, "1d", "1h", 5.0, 0.125, 1e-11, 1e-13, 128),
        ("30d_dense", 30.0, "6h", "15m", 7.0, 0.0625, 5e-12, 5e-14, 256),
    ]
    runs: list[SweepRun] = []
    for offset, (label, refresh, h_step, r_step, window, max_step, rtol, atol, samples) in enumerate(specs):
        runs.append(
            SweepRun(
                name=f"reconstruct_{start + offset:02d}_{label}_{h_step}_{r_step}",
                horizons_step=h_step,
                refine_step=r_step,
                refine_window_days=window,
                integrator_max_step_days=max_step,
                integrator_rtol=rtol,
                integrator_atol=atol,
                uncertainty_samples=samples,
                state_refresh_days=refresh,
                profile="osculating_reconstruction",
            )
        )
    return runs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run ASTEROID-NEO sensitivity sweeps.")
    parser.add_argument("--target", default="99942")
    parser.add_argument("--date-min", default="2026-04-27")
    parser.add_argument("--date-max", default="2030-12-31")
    parser.add_argument("--dist-max", default="0.5")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "sensitivity_sweeps")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit sweep count; 0 means all configured runs.")
    parser.add_argument("--execute", action="store_true", help="Actually run jobs. Default is dry-run planning.")
    parser.add_argument("--dry-run", action="store_true", help="Print and write the plan without running jobs.")
    parser.add_argument(
        "--include-reconstruction",
        action="store_true",
        help="Include declared osculating-refresh reconstruction profiles in the sweep plan.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Summarize already-generated output bundles as a completed sensitivity sweep.",
    )
    parser.add_argument(
        "--existing-bundle",
        action="append",
        type=Path,
        default=[],
        help="Existing output bundle to include in --summarize-existing. Can be passed more than once.",
    )
    return parser.parse_args(argv)


def write_manifest(output_root: Path, runs: list[SweepRun], commands: list[list[str]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            **asdict(run),
            "output_dir": str(run.output_dir(output_root)),
            "command": command,
        }
        for run, command in zip(runs, commands)
    ]
    (output_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_root / "sweep_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)


def _safe_nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _float_or_blank(value: Any) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _summarize_bundle(bundle: Path) -> dict[str, Any]:
    report_path = bundle / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dynamics = report.get("dynamics", {})
    diagnostics = dynamics.get("numerical_diagnostics", {})
    return {
        "bundle": str(bundle),
        "n_samples": int(dynamics.get("n_samples", 0)),
        "horizons_step": dynamics.get("horizons_step", ""),
        "prediction_mode": diagnostics.get("prediction_mode", ""),
        "state_refresh_count": _float_or_blank(diagnostics.get("state_refresh_count")),
        "state_refresh_segment_days": _float_or_blank(diagnostics.get("state_refresh_segment_days")),
        "integrator": dynamics.get("integrator", diagnostics.get("integrator_method", "")),
        "integrator_rtol": _float_or_blank(diagnostics.get("integrator_rtol")),
        "integrator_atol": _float_or_blank(diagnostics.get("integrator_atol")),
        "integrator_substep_max_days": _float_or_blank(diagnostics.get("integrator_substep_max_days")),
        "validation_mae_km": _float_or_blank(dynamics.get("validation_mae_km")),
        "validation_rmse_km": _float_or_blank(dynamics.get("validation_rmse_km")),
        "nearest_cad_error_km": _float_or_blank(dynamics.get("cad_validation_error_km")),
        "cad_anchor_mae_km": _float_or_blank(diagnostics.get("cad_anchor_integrated_mae_km")),
        "cad_anchor_rmse_km": _float_or_blank(diagnostics.get("cad_anchor_integrated_rmse_km")),
        "covariance_width90_km_median": _float_or_blank(diagnostics.get("covariance_width90_km_median")),
        "gi_oi_cascade_width90_km_median": _float_or_blank(diagnostics.get("gi_oi_cascade_width90_km_median")),
        "global_gate_accepted": _safe_nested(diagnostics, "global_residual_gate_accepted", default="n/a"),
        "local_gate_accepted": _safe_nested(diagnostics, "local_gate_accepted", default="n/a"),
        "cad_reconstruction_gate_accepted": _safe_nested(
            diagnostics,
            "tensorflow_continuous_anchor_gate_accepted",
            default="n/a",
        ),
    }


def write_existing_summary(output_root: Path, bundles: list[Path]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [_summarize_bundle(bundle.expanduser().resolve()) for bundle in bundles]
    if not rows:
        raise FileNotFoundError("No existing bundles were provided for summary")
    rows.sort(key=lambda row: int(row["n_samples"]))
    json_path = output_root / "existing_bundle_sensitivity_summary.json"
    csv_path = output_root / "existing_bundle_sensitivity_summary.csv"
    md_path = output_root / "existing_bundle_sensitivity_summary.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Existing Bundle Sensitivity Summary\n\n")
        fh.write(
            "This table compares already-generated dynamics bundles across sample density, "
            "cadence/refinement settings, and integrator tolerances.\n\n"
        )
        columns = [
            "bundle",
            "n_samples",
            "prediction_mode",
            "state_refresh_count",
            "state_refresh_segment_days",
            "horizons_step",
            "integrator_rtol",
            "integrator_atol",
            "validation_rmse_km",
            "nearest_cad_error_km",
            "cad_anchor_rmse_km",
        ]
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            values = [str(row[column]) for column in columns]
            fh.write("| " + " | ".join(values) + " |\n")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.summarize_existing:
        bundles = args.existing_bundle or [
            ROOT / "outputs" / "predictor_full",
            ROOT / "outputs" / "predictor_high_samples",
            ROOT / "outputs" / "predictor_3x_samples",
        ]
        refresh_bundle = ROOT / "outputs" / "predictor_refresh30"
        if not args.existing_bundle and (refresh_bundle / "report.json").exists():
            bundles.append(refresh_bundle)
        write_existing_summary(args.output_root, bundles)
        return 0
    runs = build_runs(include_reconstruction=args.include_reconstruction)
    if args.max_runs > 0:
        runs = runs[: args.max_runs]
    commands = [run.command(args.output_root, args.target, args.date_min, args.date_max, args.dist_max) for run in runs]
    write_manifest(args.output_root, runs, commands)
    print(f"planned {len(runs)} sensitivity runs under {args.output_root}")
    for command in commands:
        print(" ".join(command))
    if not args.execute:
        return 0
    for run, command in zip(runs, commands):
        print(f"\n=== running {run.name} ===")
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
