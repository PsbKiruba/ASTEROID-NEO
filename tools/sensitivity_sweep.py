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

    def output_dir(self, root: Path) -> Path:
        return root / self.name

    def command(self, output_root: Path, target: str, date_min: str, date_max: str, dist_max: str) -> list[str]:
        out = self.output_dir(output_root)
        return [
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


def build_runs() -> list[SweepRun]:
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    runs = build_runs()
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
