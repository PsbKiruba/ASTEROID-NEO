from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import asteroid_neo as neo
import neo_infographic_viewer as viewer


FIXTURE_BUNDLE = Path(__file__).parent / "fixtures" / "dense_bundle"


def test_classifier_boundaries() -> None:
    assert neo.classify_neo_group(0.922, 0.746, 1.099) == "ATE"
    assert neo.classify_neo_group(1.0, 1.017, 1.5) == "APO"
    assert neo.classify_neo_group(1.1, 1.3, 1.8) == "AMO"
    assert neo.classify_neo_group(0.8, 0.6, 0.9) == "IEO"
    assert neo.classify_neo_group(0.9, 0.7, 0.983) == "NEO_UNCLASSIFIED_BOUNDARY"
    assert neo.classify_neo_group(1.4, 1.31, 2.0) == "NON_NEO"


def test_horizons_vector_csv_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "signature": {"version": "fixture", "source": "JPL"},
        "result": """
Header
$$SOE
2462240.500000, A.D. 2029-Apr-14 00:00:00.0000, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
bad,row
2462241.500000, A.D. 2029-Apr-15 00:00:00.0000, 1.5, 2.5, 3.5, 0.4, 0.5, 0.6,
$$EOE
Footer
""",
    }

    def fake_http_json(url: str, params: dict[str, object], timeout: float = 30.0) -> tuple[dict[str, object], str]:
        return payload, "https://fixture.invalid/horizons"

    monkeypatch.setattr(neo, "_http_json", fake_http_json)
    vectors = neo.fetch_horizons_vectors("99942", "500@399", "2029-04-13", "2029-04-16", "1d")
    assert vectors.center == "500@399"
    assert vectors.jd_tdb == [2462240.5, 2462241.5]
    assert vectors.calendar_tdb[0].startswith("A.D. 2029-Apr-14")
    assert vectors.state_au_d[1] == [1.5, 2.5, 3.5, 0.4, 0.5, 0.6]


def test_dense_bundle_loading_and_cad_anchor_matching() -> None:
    run = viewer.load_run(FIXTURE_BUNDLE)
    assert len(run.jd) == 5
    assert run.close_index == 2
    assert run.calendar[run.close_index].startswith("A.D. 2029-Apr-13 21:45")
    assert run.anchors[0].nearest_index == 2
    assert run.anchors[0].integrated_minus_cad_km == pytest.approx(42.0)
    assert run.earth_fixed_au.tolist() == pytest.approx(run.pos_au[2].tolist())


def test_output_schema_validation_accepts_fixture_bundle() -> None:
    report = json.loads((FIXTURE_BUNDLE / "report.json").read_text(encoding="utf-8"))
    neo.validate_report_dict(report)
    neo.validate_csv_table(FIXTURE_BUNDLE / "table_dynamical_integrator_timeseries.csv")
    neo.validate_csv_table(FIXTURE_BUNDLE / "table_dynamical_integrator_anchor_validation.csv")


def test_output_schema_validation_rejects_missing_required_column(tmp_path: Path) -> None:
    table = tmp_path / "table_dynamical_integrator_timeseries.csv"
    table.write_text("jd_tdb,calendar_tdb\n2462240.5,A.D. 2029-Apr-14\n", encoding="utf-8")
    with pytest.raises(neo.SchemaValidationError):
        neo.validate_csv_table(table)


def test_sensitivity_summary_writes_existing_bundle_report(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "tools" / "sensitivity_sweep.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--summarize-existing",
            "--existing-bundle",
            str(FIXTURE_BUNDLE),
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    summary = json.loads((tmp_path / "existing_bundle_sensitivity_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["n_samples"] == 5
    assert summary[0]["nearest_cad_error_km"] == pytest.approx(42.0)
    assert (tmp_path / "existing_bundle_sensitivity_summary.csv").exists()


def test_benchmark_thresholds_pass_fixture_bundle(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "tools" / "benchmark_neos.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--evaluate",
            "--bundle",
            str(FIXTURE_BUNDLE),
            "--output-root",
            str(tmp_path),
            "--allow-partial",
        ],
        check=True,
    )
    report = json.loads((tmp_path / "threshold_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert (tmp_path / "benchmark_summary.csv").exists()


def test_benchmark_thresholds_fail_hard_limit(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "tools" / "benchmark_neos.py"
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"single_arc_validation_rmse_km_max": 1.0}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--evaluate",
            "--bundle",
            str(FIXTURE_BUNDLE),
            "--output-root",
            str(tmp_path),
            "--thresholds",
            str(thresholds),
            "--allow-partial",
        ],
        check=False,
    )
    assert result.returncode == 2
    report = json.loads((tmp_path / "threshold_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
