#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


try:
    from astropy import constants as _astropy_constants  # type: ignore

    GM_SUN_M3_S2 = float(_astropy_constants.GM_sun.value)
    GM_EARTH_M3_S2 = float(_astropy_constants.GM_earth.value)
except Exception:
    # IAU 2015 nominal solar gravitational parameter, used only if astropy is
    # unavailable. This is a physical constant, not an object-specific input.
    GM_SUN_M3_S2 = 1.3271244e20
    GM_EARTH_M3_S2 = 3.98600435436e14


try:
    from scipy import constants as _scipy_constants  # type: ignore

    AU_M = float(_scipy_constants.astronomical_unit)
    G_SI = float(_scipy_constants.gravitational_constant)
    C_M_S = float(_scipy_constants.speed_of_light)
except Exception:
    # CODATA/IAU constants. Only used if scipy.constants is unavailable.
    AU_M = 149_597_870_700.0
    G_SI = 6.67430e-11
    C_M_S = 299_792_458.0


AU_KM = AU_M / 1000.0
SECONDS_PER_DAY = 86_400.0

SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"
CAD_API = "https://ssd-api.jpl.nasa.gov/cad.api"
SENTRY_API = "https://ssd-api.jpl.nasa.gov/sentry.api"
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"


# Planetary gravitational parameters are IAU/JPL nominal values used by the
# optional N-body experiment when astropy is unavailable. The ephemeris positions
# are still retrieved live from JPL Horizons at runtime.
PLANETARY_PERTURBERS = {
    "sun": {"command": "10", "gm_m3_s2": GM_SUN_M3_S2},
    "mercury": {"command": "199", "gm_m3_s2": 2.2032e13},
    "venus": {"command": "299", "gm_m3_s2": 3.24858592e14},
    "earth": {"command": "399", "gm_m3_s2": GM_EARTH_M3_S2},
    "moon": {"command": "301", "gm_m3_s2": 4.9048695e12},
    "mars": {"command": "499", "gm_m3_s2": 4.2828375214e13},
    "jupiter": {"command": "599", "gm_m3_s2": 1.26686534e17},
    "saturn": {"command": "699", "gm_m3_s2": 3.7931187e16},
    "uranus": {"command": "799", "gm_m3_s2": 5.793939e15},
    "neptune": {"command": "899", "gm_m3_s2": 6.836529e15},
}


# Values transcribed from the supplied NEO Hypothesis PDF gamma-rule tables.
# They are not empirical constants and are reported as hypothesis provenance.
PDF_SURFACE_GRAVITY = {
    "sun": 275.0,
    "mercury": 3.7,
    "venus": 8.87,
    "earth": 9.81,
    "moon": 1.62,
    "mars": 3.71,
}


PDF_SCALING = {
    "scale": 1.0e9,
    "time_norm": "up",
    "time_cause": "down",
    "acceleration_cause_inverse": "up",
    "trajectory_slip": "down_x2",
    "jsuncritical": "up",
    "neo_phasing": "up_x2",
}


LIKELIHOOD_BANDS = [
    (0.00, 0.20, "Feeble Chance"),
    (0.20, 0.30, "Fly-Bys"),
    (0.30, 0.50, "Transits"),
    (0.50, 0.80, "Maximal Threat: Close Zips"),
    (0.80, 0.90, "Imminent Threat"),
    (0.90, 1.00, "Catastrophe"),
]


N_RANGE_HIGH = tuple(range(11, 15))
N_RANGE_LOW = tuple(range(2, 9))


class SourceError(RuntimeError):
    """Raised when a required online source cannot be retrieved or parsed."""


@dataclass(frozen=True)
class SourceRecord:
    name: str
    url: str
    version: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class OrbitElements:
    eccentricity: float
    semi_major_axis_au: float
    perihelion_au: float
    inclination_deg: float
    node_deg: float | None
    peri_arg_deg: float | None
    mean_anomaly_deg: float | None
    period_days: float
    aphelion_au: float
    moid_au: float | None
    epoch_jd: float | None
    orbit_id: str | None
    condition_code: str | None


@dataclass(frozen=True)
class PhysicalParameters:
    absolute_magnitude_h: float | None = None
    diameter_km: float | None = None
    rotation_period_h: float | None = None
    albedo: float | None = None
    non_grav_a1_au_d2: float | None = None
    non_grav_a2_au_d2: float | None = None


@dataclass(frozen=True)
class CloseApproach:
    calendar_date_tdb: str
    jd_tdb: float
    distance_au: float
    distance_min_au: float | None
    distance_max_au: float | None
    v_rel_km_s: float | None
    v_inf_km_s: float | None
    t_sigma: str | None
    orbit_id: str | None


@dataclass(frozen=True)
class SentryStatus:
    active: bool
    removed_date: str | None = None
    summary: dict[str, Any] | None = None
    raw_status: str = "unknown"


@dataclass(frozen=True)
class NEOObject:
    designation: str
    fullname: str
    spkid: str | None
    orbit_class_code: str | None
    orbit_class_name: str | None
    neo: bool
    pha: bool
    elements: OrbitElements
    physical: PhysicalParameters
    close_approaches: list[CloseApproach]
    sentry: SentryStatus
    sources: list[SourceRecord]
    covariance: dict[str, Any] | None = None


@dataclass(frozen=True)
class RangeStats:
    values: list[float]
    low: float
    median: float
    high: float


@dataclass(frozen=True)
class HypothesisInputs:
    neo_group_from_orbit: str
    neo_group_from_sbdb: str | None
    gamma_ratio: float
    gamma_rule: str
    delta_t_s: float
    orbital_speed_m_s: float
    upsilon_v_over_c: float
    eccentricity: float
    focal_distance_m: float
    perihelion_m: float
    selected_approach_date: str | None
    selected_approach_distance_au: float | None


@dataclass(frozen=True)
class HypothesisTerms:
    inputs: HypothesisInputs
    jsuncritical: float
    time_norm: float
    time_cause: float
    acceleration_cause_inverse: float
    trajectory_slip: float
    trajectory_precession: float
    neo_phasing: float
    gravity_neo_phasing: float
    bound_factor: float
    precession_ratio: float
    time_slip: float
    lapse_factor: float
    sequence_1: float
    sequence_2: float
    sequence_3: float
    sequence_4: float
    sequence_5: float
    trajectory_loss: float
    seqcr: float
    new_eccentricity_raw: float
    new_perihelion_m: float
    new_aphelion_m: float
    new_semi_major_axis_m: float
    likelihood_proxy_raw: float
    likelihood_unit_interval: float
    likelihood_band: str
    gi_n_raw: float
    oi_n_raw: float
    oi_cascade: list[float]
    oi_cascade_sum: float
    scaled_terms: dict[str, float]
    range_terms: dict[str, RangeStats]
    caveats: list[str]


@dataclass(frozen=True)
class StandardAssessment:
    source: str
    nearest_approach: CloseApproach | None
    sentry_status: SentryStatus
    pha_by_cneos_rule: bool | None
    notes: list[str]


@dataclass(frozen=True)
class AnalysisReport:
    generated_utc: str
    target: str
    standard: StandardAssessment
    hypothesis: HypothesisTerms
    object: NEOObject
    dynamics: DynamicalPropagationReport | None = None


@dataclass(frozen=True)
class HorizonsVectors:
    center: str
    jd_tdb: list[float]
    calendar_tdb: list[str]
    state_au_d: list[list[float]]
    source: SourceRecord


@dataclass(frozen=True)
class DynamicalPropagationReport:
    enabled: bool
    method: str
    force_model: str
    n_samples: int
    horizons_step: str
    integrator: str
    validation_mae_km: float
    validation_rmse_km: float
    nearest_horizons_date: str
    nearest_horizons_distance_au: float
    nearest_integrated_date: str
    nearest_integrated_distance_au: float
    nearest_integrated_error_km: float
    cad_validation_error_km: float | None
    numerical_diagnostics: dict[str, Any]
    anchor_validation: list[dict[str, Any]]
    figures: list[str]
    tables: list[str]
    publication_assets: list[str]
    caveats: list[str]


@dataclass(frozen=True)
class MLSurrogateReport:
    enabled: bool
    method: str
    training_label_source: str
    feature_names: list[str]
    n_samples: int
    n_train: int
    n_calibration: int
    n_validation: int
    horizons_step: str
    validation_mae_km: float
    validation_rmse_km: float
    nearest_horizons_date: str
    nearest_horizons_distance_au: float
    nearest_ml_date: str
    nearest_ml_distance_au: float
    nearest_ml_error_km: float
    cad_validation_error_km: float | None
    model_scores: dict[str, dict[str, float]]
    ensemble_weights: dict[str, float]
    numerical_diagnostics: dict[str, Any]
    conformal_90_width_km_median: float
    conformal_90_coverage: float
    top_features: list[tuple[str, float]]
    anchor_validation: list[dict[str, Any]]
    figures: list[str]
    tables: list[str]
    publication_assets: list[str]
    caveats: list[str]


def _http_json(url: str, params: dict[str, Any], timeout: float = 30.0) -> tuple[dict[str, Any], str]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    full_url = f"{url}?{query}"
    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "clean-neo-hypothesis/6.0 (+https://ssd.jpl.nasa.gov/api.html)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), full_url
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SourceError(f"{full_url} returned HTTP {exc.code}: {body[:500]}") from exc
    except Exception as exc:
        raise SourceError(f"failed to retrieve {full_url}: {exc}") from exc


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _element_map(sbdb: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in sbdb["orbit"].get("elements", []) if "name" in item}


def _phys_map(sbdb: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in sbdb.get("phys_par", []) if "name" in item}


def _model_par_map(sbdb: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in sbdb["orbit"].get("model_pars", []) if "name" in item}


def fetch_sbdb(target: str) -> tuple[dict[str, Any], SourceRecord]:
    data, url = _http_json(
        SBDB_API,
        {"sstr": target, "full-prec": "1", "phys-par": "1", "cov": "mat"},
    )
    if "object" not in data or "orbit" not in data:
        raise SourceError(f"SBDB did not return a unique object for {target!r}: {data}")
    sig = data.get("signature", {})
    return data, SourceRecord("JPL SBDB API", url, sig.get("version"), sig.get("source"))


def fetch_cad(
    target: str,
    date_min: str,
    date_max: str,
    dist_max: str,
) -> tuple[dict[str, Any], SourceRecord]:
    data, url = _http_json(
        CAD_API,
        {
            "des": target,
            "date-min": date_min,
            "date-max": date_max,
            "dist-max": dist_max,
            "body": "Earth",
            "sort": "date",
            "diameter": "true",
            "fullname": "true",
        },
    )
    sig = data.get("signature", {})
    return data, SourceRecord("JPL SBDB Close Approach Data API", url, sig.get("version"), sig.get("source"))


def fetch_sentry(target: str) -> tuple[dict[str, Any], SourceRecord]:
    data, url = _http_json(SENTRY_API, {"des": target})
    sig = data.get("signature", {})
    return data, SourceRecord("JPL Sentry API", url, sig.get("version"), sig.get("source"))


def _horizons_command(target: str) -> str:
    text = str(target).strip()
    major_body_commands = {str(spec["command"]) for spec in PLANETARY_PERTURBERS.values()}
    if text in major_body_commands or text.endswith(";"):
        return text
    return f"{text};"


def fetch_horizons_vectors(
    target: str,
    center: str,
    date_min: str,
    date_max: str,
    step: str,
) -> HorizonsVectors:
    data, url = _http_json(
        HORIZONS_API,
        {
            "format": "json",
            "COMMAND": _horizons_command(target),
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": center,
            "START_TIME": date_min,
            "STOP_TIME": date_max,
            "STEP_SIZE": step,
            "OUT_UNITS": "AU-D",
            "VEC_TABLE": "3",
            "CSV_FORMAT": "YES",
            "REF_PLANE": "ECLIPTIC",
            "REF_SYSTEM": "ICRF",
            "VEC_CORR": "NONE",
        },
        timeout=60.0,
    )
    if "error" in data:
        raise SourceError(f"Horizons returned an error for center {center}: {data['error']}")
    result = data.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise SourceError(f"Horizons result for center {center} did not contain vector table")

    lines = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip().splitlines()
    jd: list[float] = []
    cal: list[str] = []
    states: list[list[float]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            jd.append(float(parts[0]))
            cal.append(parts[1])
            states.append([float(parts[i]) for i in range(2, 8)])
        except ValueError:
            continue
    if not states:
        raise SourceError(f"Horizons vector table for center {center} parsed to zero rows")
    sig = data.get("signature", {})
    return HorizonsVectors(
        center=center,
        jd_tdb=jd,
        calendar_tdb=cal,
        state_au_d=states,
        source=SourceRecord(f"JPL Horizons VECTORS center={center}", url, sig.get("version"), sig.get("source")),
    )


def _jd_to_iso_date(jd: float) -> str:
    """Convert Julian Day to Gregorian date string good enough for Horizons windows."""
    # Fliegel-Van Flandern style conversion.
    z = int(jd + 0.5)
    f = (jd + 0.5) - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return f"{year:04d}-{month:02d}-{int(day):02d}"


def _merge_horizons_vectors(base: HorizonsVectors, additions: list[HorizonsVectors]) -> HorizonsVectors:
    by_jd: dict[float, tuple[str, list[float]]] = {
        round(jd, 9): (cal, state)
        for jd, cal, state in zip(base.jd_tdb, base.calendar_tdb, base.state_au_d)
    }
    for add in additions:
        for jd, cal, state in zip(add.jd_tdb, add.calendar_tdb, add.state_au_d):
            by_jd[round(jd, 9)] = (cal, state)
    ordered = sorted(by_jd.items(), key=lambda item: item[0])
    return HorizonsVectors(
        center=base.center,
        jd_tdb=[item[0] for item in ordered],
        calendar_tdb=[item[1][0] for item in ordered],
        state_au_d=[item[1][1] for item in ordered],
        source=base.source,
    )


def _refinement_center_jds(
    coarse_geo: HorizonsVectors,
    window_days: float,
    max_windows: int = 4,
    anchor_jds: Iterable[float] | None = None,
) -> list[float]:
    import numpy as np

    states = np.asarray(coarse_geo.state_au_d, dtype=float)
    dist = np.linalg.norm(states[:, :3], axis=1)
    candidates: set[int] = set()
    for i in range(1, len(dist) - 1):
        if dist[i] <= dist[i - 1] and dist[i] <= dist[i + 1]:
            candidates.add(i)
    if not candidates:
        candidates.add(int(np.argmin(dist)))
    ranked = sorted(candidates, key=lambda i: dist[i])[:max_windows]
    center_jds = [float(coarse_geo.jd_tdb[i]) for i in ranked]
    if anchor_jds is not None:
        jd_min = min(coarse_geo.jd_tdb)
        jd_max = max(coarse_geo.jd_tdb)
        for anchor_jd in anchor_jds:
            if jd_min <= float(anchor_jd) <= jd_max:
                center_jds.append(float(anchor_jd))

    deduped: list[float] = []
    min_separation = max(float(window_days) * 0.5, 0.25)
    for center_jd in sorted(center_jds):
        if not deduped or all(abs(center_jd - old) > min_separation for old in deduped):
            deduped.append(center_jd)
    return deduped


def _refine_horizons_near_minima(
    target: str,
    coarse_geo: HorizonsVectors,
    coarse_helio: HorizonsVectors,
    refine_step: str,
    window_days: float,
    max_windows: int = 4,
    anchor_jds: Iterable[float] | None = None,
) -> tuple[HorizonsVectors, HorizonsVectors]:
    deduped = _refinement_center_jds(coarse_geo, window_days, max_windows=max_windows, anchor_jds=anchor_jds)

    geo_add: list[HorizonsVectors] = []
    helio_add: list[HorizonsVectors] = []
    for center_jd in deduped:
        start = _jd_to_iso_date(center_jd - window_days)
        stop = _jd_to_iso_date(center_jd + window_days)
        geo_add.append(fetch_horizons_vectors(target, "500@399", start, stop, refine_step))
        helio_add.append(fetch_horizons_vectors(target, "500@10", start, stop, refine_step))

    return _merge_horizons_vectors(coarse_geo, geo_add), _merge_horizons_vectors(coarse_helio, helio_add)


def parse_sbdb(sbdb: dict[str, Any]) -> tuple[OrbitElements, PhysicalParameters, dict[str, Any], dict[str, Any] | None]:
    elem = _element_map(sbdb)
    orbit = sbdb["orbit"]
    physical = _phys_map(sbdb)
    model = _model_par_map(sbdb)

    def evalue(name: str, required: bool = True) -> float | None:
        value = _to_float(elem.get(name, {}).get("value"))
        if value is None and required:
            raise SourceError(f"SBDB orbit element {name!r} is missing")
        return value

    a = evalue("a")
    e = evalue("e")
    q = evalue("q")
    inc = evalue("i")
    period = evalue("per")
    if None in (a, e, q, inc, period):
        raise SourceError("SBDB is missing one or more required orbital elements")

    ad = evalue("ad", required=False)
    if ad is None:
        ad = float(a) * (1.0 + float(e))

    elements = OrbitElements(
        eccentricity=float(e),
        semi_major_axis_au=float(a),
        perihelion_au=float(q),
        inclination_deg=float(inc),
        node_deg=evalue("om", required=False),
        peri_arg_deg=evalue("w", required=False),
        mean_anomaly_deg=evalue("ma", required=False),
        period_days=float(period),
        aphelion_au=float(ad),
        moid_au=_to_float(orbit.get("moid")),
        epoch_jd=_to_float(orbit.get("epoch")),
        orbit_id=orbit.get("orbit_id"),
        condition_code=orbit.get("condition_code"),
    )

    params = PhysicalParameters(
        absolute_magnitude_h=_to_float(physical.get("H", {}).get("value")),
        diameter_km=_to_float(physical.get("diameter", {}).get("value")),
        rotation_period_h=_to_float(physical.get("rot_per", {}).get("value")),
        albedo=_to_float(physical.get("albedo", {}).get("value")),
        non_grav_a1_au_d2=_to_float(model.get("A1", {}).get("value")),
        non_grav_a2_au_d2=_to_float(model.get("A2", {}).get("value")),
    )
    return elements, params, sbdb["object"], orbit.get("covariance")


def parse_cad(cad: dict[str, Any]) -> list[CloseApproach]:
    fields = cad.get("fields", [])
    rows = cad.get("data", [])
    approaches: list[CloseApproach] = []
    for row in rows:
        record = dict(zip(fields, row))
        approaches.append(
            CloseApproach(
                calendar_date_tdb=str(record.get("cd")),
                jd_tdb=float(record.get("jd")),
                distance_au=float(record.get("dist")),
                distance_min_au=_to_float(record.get("dist_min")),
                distance_max_au=_to_float(record.get("dist_max")),
                v_rel_km_s=_to_float(record.get("v_rel")),
                v_inf_km_s=_to_float(record.get("v_inf")),
                t_sigma=record.get("t_sigma_f"),
                orbit_id=record.get("orbit_id"),
            )
        )
    return approaches


def parse_sentry(sentry: dict[str, Any]) -> SentryStatus:
    if "removed" in sentry:
        return SentryStatus(
            active=False,
            removed_date=str(sentry["removed"]),
            raw_status="removed",
        )
    if "summary" in sentry or "data" in sentry:
        return SentryStatus(active=True, summary=sentry.get("summary"), raw_status="active")
    if "error" in sentry:
        return SentryStatus(active=False, raw_status=str(sentry.get("error")))
    return SentryStatus(active=False, raw_status="not-listed")


def load_neo(target: str, date_min: str, date_max: str, dist_max: str) -> NEOObject:
    sbdb, sbdb_source = fetch_sbdb(target)
    cad, cad_source = fetch_cad(target, date_min, date_max, dist_max)
    sentry, sentry_source = fetch_sentry(target)
    elements, physical, obj, covariance = parse_sbdb(sbdb)
    orbit_class = obj.get("orbit_class") or {}

    return NEOObject(
        designation=str(obj.get("des") or target),
        fullname=str(obj.get("fullname") or obj.get("shortname") or target),
        spkid=obj.get("spkid"),
        orbit_class_code=orbit_class.get("code"),
        orbit_class_name=orbit_class.get("name"),
        neo=bool(obj.get("neo")),
        pha=bool(obj.get("pha")),
        elements=elements,
        physical=physical,
        close_approaches=parse_cad(cad),
        sentry=parse_sentry(sentry),
        sources=[sbdb_source, cad_source, sentry_source],
        covariance=covariance,
    )


def classify_neo_group(a_au: float, q_au: float, Q_au: float) -> str:
    """Classify using CNEOS NEO group definitions."""
    if q_au >= 1.3:
        return "NON_NEO"
    if a_au < 1.0 and Q_au < 0.983:
        return "IEO"
    if a_au < 1.0 and Q_au > 0.983:
        return "ATE"
    if a_au > 1.0 and q_au < 1.017:
        return "APO"
    if a_au > 1.0 and 1.017 < q_au < 1.3:
        return "AMO"
    return "NEO_UNCLASSIFIED_BOUNDARY"


def gamma_from_pdf(group_code: str, inclination_deg: float) -> tuple[float, str]:
    """Return the PDF gamma ratio selected by NEO group and inclination."""
    g = PDF_SURFACE_GRAVITY
    sun = g["sun"]
    e_over_s = g["earth"] / sun
    inner_over_s = (g["mercury"] + g["venus"] + g["earth"] + g["moon"]) / sun
    moon_mars_over_s = (g["moon"] + g["mars"]) / sun
    apollo_mid = g["mercury"] / g["earth"] + g["venus"] / g["earth"] + g["moon"] / g["earth"] + g["mercury"] / g["venus"]

    code = group_code.upper()
    inc = inclination_deg
    if code == "AMO":
        return moon_mars_over_s, "AMOR: (Moon + Mars) / Sun"
    if code == "IEO":
        return e_over_s, "ATIRA/IEO: Earth / Sun"
    if code == "ATE":
        if inc > 10.0:
            return e_over_s, "ATEN, i > 10 deg: Earth / Sun"
        return inner_over_s, "ATEN, i <= 10 deg: (Mercury + Venus + Earth + Moon) / Sun"
    if code == "APO":
        if inc > 10.0:
            return e_over_s, "APOLLO, i > 10 deg: Earth / Sun"
        if inc > 8.0:
            return inner_over_s, "APOLLO, 8 < i <= 10 deg: inner bodies / Sun"
        if inc > 3.0:
            return apollo_mid, "APOLLO, 3 < i <= 8 deg: PDF mid-inclination rule"
        return inner_over_s, "APOLLO, i <= 3 deg: inner bodies / Sun"
    return inner_over_s, "Fallback PDF gamma: inner bodies / Sun"


def _poly_u(U: float) -> float:
    return ((((1.0 * U + 4.0) * U + 12.0) * U + 24.0) * U + 24.0)


def _poly_u_cubic(U: float) -> float:
    return (((1.0 * U + 3.0) * U + 6.0) * U + 6.0)


def _inner_frac(upsilon: float) -> float:
    if upsilon == 0.0:
        raise ValueError("upsilon must be non-zero")
    sq = math.sqrt(upsilon * upsilon + 2.0)
    denom = (upsilon * upsilon) * (5670.0 - 5076.0 * sq)
    if denom == 0.0:
        raise ValueError("inner denominator is zero")
    numer = sq * ((upsilon * upsilon + 2.0) ** 6) * ((sq - 1.5) ** 5)
    return numer / denom


def _range_stats(values: Iterable[float]) -> RangeStats:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("range has no values")
    sorted_vals = sorted(vals)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        median = sorted_vals[mid]
    else:
        median = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    return RangeStats(vals, sorted_vals[0], median, sorted_vals[-1])


def _require_nonzero(value: float, name: str) -> None:
    if not math.isfinite(value) or abs(value) < 1e-300:
        raise ValueError(f"{name} must be finite and non-zero, got {value!r}")


def calc_jsuncrit(n: float, delta_t: float, ecc: float, f_dist: float, gamma: float, U: float) -> float:
    return n * delta_t * ecc * f_dist * gamma / math.sqrt(_poly_u(U))


def calc_time_norm(n: float, gamma: float, ecc: float, f_dist: float, upsilon: float, U: float) -> float:
    denom = 32.0 * (gamma * ecc * f_dist) ** 2 * _inner_frac(upsilon)
    _require_nonzero(denom, "time_norm denominator")
    return math.sqrt(abs((7.0 ** 4) * _poly_u(U) / denom)) / n


def calc_time_cause(gamma: float, ecc: float, f_dist: float, upsilon: float, U: float, jsun: float) -> float:
    denom = gamma * ecc * f_dist
    _require_nonzero(denom, "time_cause gamma*ecc*f")
    inner = (upsilon**2 + 1.0) / math.sqrt(1.5 + math.sqrt(upsilon**2 + 2.0))
    return math.sqrt(math.sqrt(inner)) * (jsun / denom) * math.sqrt(_poly_u(U))


def calc_acceleration_cause_inverse(gamma: float, ecc: float, f_dist: float, U: float, jsun: float, upsilon: float) -> float:
    denom = 2.0 * _poly_u_cubic(U)
    _require_nonzero(denom, "acceleration_cause_inverse polynomial denominator")
    inner = (upsilon**2 + 1.0) / math.sqrt(1.5 + math.sqrt(upsilon**2 + 2.0))
    return (gamma * ecc * f_dist * math.sqrt(_poly_u(U)) / denom) * jsun * math.sqrt(inner)


def calc_trajectory_slip(n: float, gamma: float, ecc: float, f_dist: float, upsilon: float, U: float) -> float:
    denom = 32.0 * (gamma * ecc * f_dist) ** 2 * _inner_frac(upsilon)
    _require_nonzero(denom, "trajectory_slip denominator")
    return math.sqrt(abs((7.0 ** 4) * _poly_u(U) / denom)) / (2.0 * _poly_u_cubic(U) * n)


def calc_trajectory_precession(n: float, delta_t: float, ecc: float, gamma: float, jsun: float, U: float) -> float:
    denom = jsun * math.sqrt(_poly_u(U))
    _require_nonzero(denom, "trajectory_precession denominator")
    return n * delta_t * ecc * gamma / denom


def calc_neo_phasing(U: float, upsilon: float, delta_t: float, ecc: float, gamma: float) -> float:
    denom = 2.0 * delta_t * ecc * gamma * _poly_u_cubic(U)
    _require_nonzero(denom, "neo_phasing denominator")
    inner = math.sqrt(math.sqrt((upsilon**2 + 1.0) / math.sqrt(1.5 + math.sqrt(upsilon**2 + 2.0))))
    return (_poly_u(U) ** 1.5) * inner / denom


def calc_gravity_neo_phasing(G_val: float, delta_t: float, ecc: float, gamma: float, U: float, upsilon: float) -> float:
    inner = math.sqrt(math.sqrt((upsilon**2 + 1.0) / math.sqrt(1.5 + math.sqrt(upsilon**2 + 2.0))))
    _require_nonzero(inner, "gravity_neo_phasing inner")
    return 2.0 * G_val * delta_t * ecc * gamma * _poly_u_cubic(U) / ((_poly_u(U) ** 1.5) * inner)


def calc_bound_factor(n: float, delta_t: float, ecc: float, gamma: float, U: float) -> float:
    denom = _poly_u(U)
    _require_nonzero(denom, "bound_factor denominator")
    return n * delta_t * ecc * gamma * 14.0 / denom


def calc_time_slip(n: float, gamma: float, ecc: float, f_dist: float, upsilon: float, U: float, jsun: float) -> float:
    denom = 32.0 * (gamma * ecc * f_dist) ** 2 * _inner_frac(upsilon)
    _require_nonzero(denom, "time_slip outer denominator")
    inner_fraction = math.sqrt(abs((7.0 ** 4) * _poly_u(U) / denom)) / n
    _require_nonzero(inner_fraction, "time_slip inner fraction")
    inner_sqrt = math.sqrt(math.sqrt(math.sqrt(upsilon**2 + 1.0) / math.sqrt(1.5 + math.sqrt(upsilon**2 + 2.0))))
    return abs(inner_sqrt * (jsun / (gamma * ecc * f_dist)) * math.sqrt(_poly_u(U)) / inner_fraction)


def calc_lapse_factor(n: float, delta_t: float, gamma: float, ecc: float, f_dist: float, upsilon: float, U: float, jsun: float) -> float:
    outer = 32.0 * (gamma * ecc * f_dist) ** 2
    _require_nonzero(outer, "lapse_factor outer denominator")
    inner_sqrt = math.sqrt(math.sqrt((upsilon**2 + 1.0) / math.sqrt(1.5 + math.sqrt(upsilon**2 + 2.0))))
    denom_jsun = inner_sqrt * (jsun / (gamma * ecc * f_dist)) * math.sqrt(_poly_u(U))
    _require_nonzero(denom_jsun, "lapse_factor jsun denominator")
    ratio = math.sqrt(abs(((7.0 ** 4) * _poly_u(U) / outer) * _inner_frac(upsilon)))
    return (delta_t * ratio / n) / denom_jsun


def _sequence_terms(tslp: float, tprec: float, neoph: float, lfac: float, gneoph: float, bfac: float, acinv: float) -> tuple[float, float, float, float, float, float, float]:
    _require_nonzero(neoph * tprec, "sequence_1 denominator")
    s1 = (tslp * lfac) / (neoph * tprec)
    _require_nonzero(tslp, "sequence_2 denominator")
    s2 = tprec / tslp
    _require_nonzero(bfac, "sequence_3 denominator")
    s3 = gneoph / bfac
    _require_nonzero(lfac, "sequence_4 denominator")
    s4 = neoph / lfac
    s5 = acinv * neoph * gneoph
    trajectory_loss = (lfac / bfac) * (gneoph / bfac)
    _require_nonzero(s1 * s4, "seqcr denominator")
    seqcr = (s3 * s2 * s5) / (s1 * s4)
    return s1, s2, s3, s4, s5, trajectory_loss, seqcr


def _unit_interval_from_positive(raw: float) -> float:
    if not math.isfinite(raw) or raw <= 0.0:
        return 0.0
    return raw / (1.0 + raw)


def _likelihood_band(value: float) -> str:
    bounded = max(0.0, min(1.0, value))
    for lo, hi, label in LIKELIHOOD_BANDS:
        if lo <= bounded < hi or (bounded == 1.0 and hi == 1.0):
            return label
    return "Unknown"


def _scaled(direction: str, value: float, scale: float) -> float:
    if direction == "up":
        return value * scale
    if direction == "down":
        return value / scale
    if direction == "up_x2":
        return value * scale * 2.0
    if direction == "down_x2":
        return value / scale / 2.0
    return value


def _nearest_approach(approaches: list[CloseApproach]) -> CloseApproach | None:
    if not approaches:
        return None
    return min(approaches, key=lambda item: item.distance_au)


def evaluate_hypothesis(neo: NEOObject) -> HypothesisTerms:
    elems = neo.elements
    group = classify_neo_group(elems.semi_major_axis_au, elems.perihelion_au, elems.aphelion_au)
    gamma, gamma_rule = gamma_from_pdf(group, elems.inclination_deg)
    closest = _nearest_approach(neo.close_approaches)

    if closest and closest.v_rel_km_s:
        U = closest.v_rel_km_s * 1000.0
        selected_date = closest.calendar_date_tdb
        selected_dist = closest.distance_au
    else:
        # Use mean orbital speed from period and semi-major axis as a sourced
        # fallback derived from SBDB elements, not a stored object constant.
        U = 2.0 * math.pi * elems.semi_major_axis_au * AU_M / (elems.period_days * SECONDS_PER_DAY)
        selected_date = None
        selected_dist = None

    delta_t = elems.period_days * SECONDS_PER_DAY
    upsilon = U / C_M_S
    ecc = elems.eccentricity
    f_dist = (elems.aphelion_au - elems.perihelion_au) * AU_M
    jsun = elems.perihelion_au * AU_M

    inputs = HypothesisInputs(
        neo_group_from_orbit=group,
        neo_group_from_sbdb=neo.orbit_class_code,
        gamma_ratio=gamma,
        gamma_rule=gamma_rule,
        delta_t_s=delta_t,
        orbital_speed_m_s=U,
        upsilon_v_over_c=upsilon,
        eccentricity=ecc,
        focal_distance_m=f_dist,
        perihelion_m=jsun,
        selected_approach_date=selected_date,
        selected_approach_distance_au=selected_dist,
    )

    high_jsun = _range_stats(calc_jsuncrit(n, delta_t, ecc, f_dist, gamma, U) for n in N_RANGE_HIGH)
    low_tnorm = _range_stats(calc_time_norm(n, gamma, ecc, f_dist, upsilon, U) for n in N_RANGE_LOW)
    low_tslip = _range_stats(calc_trajectory_slip(n, gamma, ecc, f_dist, upsilon, U) for n in N_RANGE_LOW)
    high_tprec = _range_stats(calc_trajectory_precession(n, delta_t, ecc, gamma, jsun, U) for n in N_RANGE_HIGH)
    high_bfac = _range_stats(calc_bound_factor(n, delta_t, ecc, gamma, U) for n in N_RANGE_HIGH)
    low_time_slip = _range_stats(calc_time_slip(n, gamma, ecc, f_dist, upsilon, U, jsun) for n in N_RANGE_LOW)
    low_lfac = _range_stats(calc_lapse_factor(n, delta_t, gamma, ecc, f_dist, upsilon, U, jsun) for n in N_RANGE_LOW)

    jcrit = high_jsun.median
    tnorm = low_tnorm.median
    tcause = calc_time_cause(gamma, ecc, f_dist, upsilon, U, jsun)
    acinv = calc_acceleration_cause_inverse(gamma, ecc, f_dist, U, jsun, upsilon)
    tslp = low_tslip.median
    tprec = high_tprec.median
    neoph = calc_neo_phasing(U, upsilon, delta_t, ecc, gamma)
    gneoph = calc_gravity_neo_phasing(G_SI, delta_t, ecc, gamma, U, upsilon)
    bfac = high_bfac.median
    prat = jsun / f_dist
    timeslip = low_time_slip.median
    lfac = low_lfac.median

    s1, s2, s3, s4, s5, trajectory_loss, seqcr = _sequence_terms(
        tslp, tprec, neoph, lfac, gneoph, bfac, acinv
    )

    _require_nonzero(seqcr, "new eccentricity seqcr")
    _require_nonzero(prat, "new eccentricity precession ratio")
    new_ecc = ((1.0 - trajectory_loss / seqcr) / prat) / 2.0
    new_peri = jcrit * math.sqrt(abs(lfac))
    denom = 1.0 - new_ecc
    new_aph = math.copysign(math.inf, new_peri) if abs(denom) < 1e-300 else new_peri * (1.0 + new_ecc) / denom
    new_sma = 0.5 * (new_peri + new_aph) if math.isfinite(new_aph) else math.inf

    # The extracted PDF text shows "Seq 12 -> Likelihood" and the likelihood
    # range table, but the displayed equation is not text-extractable. This
    # proxy preserves the previous algebra while labelling it as diagnostic.
    _require_nonzero(s4 * prat, "likelihood proxy denominator")
    likelihood_raw = (lfac / s4) / prat
    likelihood_unit = _unit_interval_from_positive(likelihood_raw)

    gi_n = gamma * (jsun * U) / (1.0 + upsilon * upsilon)
    specific_energy = 0.5 * U * U - GM_SUN_M3_S2 / jsun
    oi_n = (specific_energy**4) * (jsun * U) ** 2 * (gamma**2) - 1.0
    cascade = _oi_cascade(specific_energy, jsun * U, gamma, terms=9)

    scale = PDF_SCALING["scale"]
    scaled = {
        "time_norm": _scaled(PDF_SCALING["time_norm"], tnorm, scale),
        "time_cause": _scaled(PDF_SCALING["time_cause"], tcause, scale),
        "acceleration_cause_inverse": _scaled(PDF_SCALING["acceleration_cause_inverse"], acinv, scale),
        "trajectory_slip": _scaled(PDF_SCALING["trajectory_slip"], tslp, scale),
        "jsuncritical": _scaled(PDF_SCALING["jsuncritical"], jcrit, scale),
        "neo_phasing": _scaled(PDF_SCALING["neo_phasing"], neoph, scale),
    }

    caveats = [
        "Hypothesis terms are proprietary diagnostics from the supplied PDF, not peer-reviewed celestial mechanics.",
        "PDF gamma is the dimensionless ratio selected from the supplied NEO hypothesis gamma table; GI_N/OI_N use that same gamma and do not substitute GM_sun/c^2.",
        "GI_N and OI_N are not injected into the standard NASA/JPL facts; they enter only the optional dynamics-first cascade experiment when requested.",
        "Seq 12 likelihood equation is ambiguous in the PDF text extraction; the reported likelihood is a labelled proxy, not an impact probability.",
        "N ranges are preserved as PDF ranges (2-8 and 11-14); medians are used only to form a single summary line.",
    ]
    if group != (neo.orbit_class_code or group):
        caveats.append(f"Computed group {group} differs from SBDB orbit class {neo.orbit_class_code}.")
    if not (0.0 <= new_ecc < 1.0):
        caveats.append("Raw post-sequence eccentricity is outside [0,1); do not interpret as a physical orbit without calibration.")

    return HypothesisTerms(
        inputs=inputs,
        jsuncritical=jcrit,
        time_norm=tnorm,
        time_cause=tcause,
        acceleration_cause_inverse=acinv,
        trajectory_slip=tslp,
        trajectory_precession=tprec,
        neo_phasing=neoph,
        gravity_neo_phasing=gneoph,
        bound_factor=bfac,
        precession_ratio=prat,
        time_slip=timeslip,
        lapse_factor=lfac,
        sequence_1=s1,
        sequence_2=s2,
        sequence_3=s3,
        sequence_4=s4,
        sequence_5=s5,
        trajectory_loss=trajectory_loss,
        seqcr=seqcr,
        new_eccentricity_raw=new_ecc,
        new_perihelion_m=new_peri,
        new_aphelion_m=new_aph,
        new_semi_major_axis_m=new_sma,
        likelihood_proxy_raw=likelihood_raw,
        likelihood_unit_interval=likelihood_unit,
        likelihood_band=_likelihood_band(likelihood_unit),
        gi_n_raw=gi_n,
        oi_n_raw=oi_n,
        oi_cascade=cascade,
        oi_cascade_sum=sum(cascade),
        scaled_terms=scaled,
        range_terms={
            "jsuncritical_n_11_14": high_jsun,
            "time_norm_n_2_8": low_tnorm,
            "trajectory_slip_n_2_8": low_tslip,
            "trajectory_precession_n_11_14": high_tprec,
            "bound_factor_n_11_14": high_bfac,
            "time_slip_n_2_8": low_time_slip,
            "lapse_factor_n_2_8": low_lfac,
        },
        caveats=caveats,
    )


def _oi_cascade(U_energy: float, j_sun: float, gamma_pdf: float, terms: int = 9) -> list[float]:
    """PDF OI_N derivative cascade using exact polynomial derivatives."""
    # Raw OI_N + 1 = C * U^4, where C is constant for a single evaluated state.
    C = (j_sun * j_sun) * (gamma_pdf * gamma_pdf)

    def deriv(poly: dict[int, float]) -> dict[int, float]:
        return {p - 1: c * p for p, c in poly.items() if p > 0}

    def sub(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
        out = dict(a)
        for p, c in b.items():
            out[p] = out.get(p, 0.0) - c
        return {p: c for p, c in out.items() if c != 0.0}

    def mul(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
        out: dict[int, float] = {}
        for pa, ca in a.items():
            for pb, cb in b.items():
                out[pa + pb] = out.get(pa + pb, 0.0) + ca * cb
        return out

    def eval_poly(poly: dict[int, float]) -> float:
        return sum(c * (U_energy ** p) for p, c in poly.items())

    p0 = {4: C}
    polys = [deriv(p0)]
    for k in range(1, terms):
        prev = polys[k - 1]
        prev_prev = p0 if k == 1 else polys[k - 2]
        inner = sub(mul(deriv(prev), deriv(prev_prev)), deriv(prev_prev))
        polys.append(deriv(inner))
    return [eval_poly(poly) for poly in polys[:terms]]


def _safe_log10_abs(values: Any, floor: float = 1e-300) -> Any:
    import numpy as np

    return np.log10(np.maximum(np.abs(values), floor))


def _finite_array(values: Iterable[float], fallback: float = 0.0) -> Any:
    import numpy as np

    return np.nan_to_num(np.asarray(list(values), dtype=float), nan=fallback, posinf=fallback, neginf=fallback)


def _parse_step_days(step: str) -> float:
    import re

    text = str(step).strip().lower().replace("_", " ")
    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([a-z]+)?", text)
    if not match:
        return 1.0
    value = float(match.group(1))
    unit = (match.group(2) or "d").lower()
    if unit.startswith("min") or unit in {"m"}:
        return value / 1440.0
    if unit.startswith("h"):
        return value / 24.0
    if unit.startswith("w"):
        return value * 7.0
    if unit.startswith("mo"):
        return value * 30.436875
    if unit.startswith("y"):
        return value * 365.25636
    return value


def _local_poly_predict_log_distance(
    jd: Any,
    log_distance: Any,
    target_jd: float,
    min_points: int = 7,
    max_points: int = 15,
    max_degree: int = 4,
) -> tuple[float, int, float]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    y_arr = np.asarray(log_distance, dtype=float)
    if len(jd_arr) < 3:
        idx = int(np.argmin(np.abs(jd_arr - target_jd)))
        return float(y_arr[idx]), 0, 0.0

    order = np.argsort(np.abs(jd_arr - target_jd))
    k = min(max_points, max(min_points, min(len(order), max_degree + 2)))
    chosen = np.sort(order[:k])
    t = jd_arr[chosen] - float(target_jd)
    y = y_arr[chosen]
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if len(t) < 3:
        idx = int(np.argmin(np.abs(jd_arr - target_jd)))
        return float(y_arr[idx]), 0, 0.0

    scale = max(float(np.max(np.abs(t))), 1e-9)
    x = t / scale
    degree = min(max_degree, len(x) - 1)
    weights = 1.0 / (1.0 + np.abs(x))
    try:
        coeff = np.polyfit(x, y, deg=degree, w=weights)
        prediction = float(np.polyval(coeff, 0.0))
    except Exception:
        idx = int(np.argmin(np.abs(jd_arr - target_jd)))
        prediction = float(y_arr[idx])
        degree = 0
    span_hours = float((np.max(t) - np.min(t)) * 24.0) if len(t) else 0.0
    return prediction, int(degree), span_hours


def _local_poly_predict_value(
    jd: Any,
    values: Any,
    target_jd: float,
    min_points: int = 7,
    max_points: int = 15,
    max_degree: int = 5,
) -> float:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    y_arr = np.asarray(values, dtype=float)
    if len(jd_arr) < 3:
        idx = int(np.argmin(np.abs(jd_arr - target_jd)))
        return float(y_arr[idx])
    order = np.argsort(np.abs(jd_arr - float(target_jd)))
    k = min(max_points, max(min_points, min(len(order), max_degree + 2)))
    chosen = np.sort(order[:k])
    t = jd_arr[chosen] - float(target_jd)
    y = y_arr[chosen]
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if len(t) < 3:
        idx = int(np.argmin(np.abs(jd_arr - target_jd)))
        return float(y_arr[idx])
    scale = max(float(np.max(np.abs(t))), 1e-9)
    x = t / scale
    degree = min(max_degree, len(x) - 1)
    weights = 1.0 / (1.0 + np.abs(x))
    try:
        coeff = np.polyfit(x, y, deg=degree, w=weights)
        return float(np.polyval(coeff, 0.0))
    except Exception:
        idx = int(np.argmin(np.abs(jd_arr - target_jd)))
        return float(y_arr[idx])


def _tensorflow_kernel_ridge_distance_km(
    x_hours: Any,
    y_km: Any,
    target_hour: float,
    degree: int,
    use_rbf_fourier: bool,
    ridge: float,
) -> float:
    import numpy as np
    import tensorflow as tf  # type: ignore

    x = np.asarray(x_hours, dtype=float).reshape(-1)
    y = np.asarray(y_km, dtype=float).reshape(-1)
    if len(x) < 3 or len(y) != len(x):
        raise ValueError("TensorFlow kernel ridge encounter model needs at least three aligned samples")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3:
        raise ValueError("insufficient finite samples for TensorFlow encounter model")
    scale = max(float(np.max(np.abs(x))), 1.0e-9)
    z = x / scale
    z0 = np.asarray([float(target_hour) / scale], dtype=float)
    degree = int(np.clip(degree, 1, max(1, len(x) - 1)))

    columns = [np.ones_like(z)]
    target_columns = [np.ones_like(z0)]
    for power in range(1, degree + 1):
        columns.append(z ** power)
        target_columns.append(z0 ** power)
    if use_rbf_fourier:
        center_count = max(3, min(len(x), 11))
        centers = np.linspace(-1.0, 1.0, center_count)
        width = max(2.0 / max(center_count - 1, 1), 0.22)
        for center in centers:
            columns.append(np.exp(-0.5 * ((z - center) / width) ** 2))
            target_columns.append(np.exp(-0.5 * ((z0 - center) / width) ** 2))
        for harmonic in range(1, min(4, degree + 1)):
            columns.append(np.sin(np.pi * harmonic * z))
            target_columns.append(np.sin(np.pi * harmonic * z0))
            columns.append(np.cos(np.pi * harmonic * z))
            target_columns.append(np.cos(np.pi * harmonic * z0))

    basis = tf.constant(np.column_stack(columns), dtype=tf.float64)
    target_basis = tf.constant(np.column_stack(target_columns), dtype=tf.float64)
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y))
    if not math.isfinite(y_scale) or y_scale < 1.0e-12:
        y_scale = 1.0
    y_norm = tf.constant(((y - y_mean) / y_scale).reshape(-1, 1), dtype=tf.float64)
    reg = np.eye(int(basis.shape[1]), dtype=float) * max(float(ridge), 0.0)
    reg[0, 0] = 0.0
    system = tf.matmul(basis, basis, transpose_a=True) + tf.constant(reg, dtype=tf.float64)
    rhs = tf.matmul(basis, y_norm, transpose_a=True)
    coeff = tf.linalg.solve(system, rhs)
    pred_norm = tf.matmul(target_basis, coeff)
    return float(pred_norm.numpy().reshape(-1)[0] * y_scale + y_mean)


def _tensorflow_encounter_reconstruction(
    jd: Any,
    distance_au: Any,
    target_jd: float,
) -> dict[str, Any]:
    import numpy as np

    _tensorflow_available_version()
    jd_arr = np.asarray(jd, dtype=float)
    dist_km = np.asarray(distance_au, dtype=float) * AU_KM
    order_all = np.argsort(np.abs(jd_arr - float(target_jd)))
    local_for_cadence = np.sort(order_all[: min(len(order_all), 31)])
    local_step = np.diff(np.sort(jd_arr[local_for_cadence])) if len(local_for_cadence) >= 2 else np.asarray([], dtype=float)
    finite_step = local_step[np.isfinite(local_step) & (local_step > 0.0)]
    if len(finite_step) == 0:
        finite_step = np.diff(np.sort(jd_arr[np.isfinite(jd_arr)]))
    finite_step = finite_step[np.isfinite(finite_step) & (finite_step > 0.0)]
    cadence_hours = float(np.nanmedian(finite_step) * 24.0) if len(finite_step) else 1.0
    cadence_hours = max(cadence_hours, 1.0e-6)
    base_points = int(2 * math.ceil(4.0 / cadence_hours) + 1)
    candidate_points = sorted(
        {
            int(np.clip(base_points + offset, 5, min(21, len(jd_arr))))
            for offset in (-4, -2, 0, 2, 4)
        }
        | {int(np.clip(value, 5, min(21, len(jd_arr)))) for value in (7, 9, 11, 13)}
    )
    ridge_values = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)
    best: dict[str, Any] | None = None
    for n_points in candidate_points:
        if n_points < 5 or n_points > len(order_all):
            continue
        chosen = np.sort(order_all[:n_points])
        x = (jd_arr[chosen] - float(target_jd)) * 24.0
        y = dist_km[chosen]
        max_degree = min(6, max(2, n_points - 2))
        for degree in sorted({min(4, max_degree), max_degree}):
            for use_rbf_fourier in (False, True):
                for ridge in ridge_values:
                    loo_sq: list[float] = []
                    for i in range(len(x)):
                        mask = np.ones(len(x), dtype=bool)
                        mask[i] = False
                        try:
                            pred_i = _tensorflow_kernel_ridge_distance_km(
                                x[mask],
                                y[mask],
                                float(x[i]),
                                degree,
                                use_rbf_fourier,
                                ridge,
                            )
                            loo_sq.append(float((pred_i - y[i]) ** 2))
                        except Exception:
                            loo_sq.append(float("inf"))
                    cv_rmse = float(np.sqrt(np.mean(loo_sq))) if loo_sq else float("inf")
                    if not math.isfinite(cv_rmse):
                        continue
                    try:
                        pred = _tensorflow_kernel_ridge_distance_km(
                            x,
                            y,
                            0.0,
                            degree,
                            use_rbf_fourier,
                            ridge,
                        )
                    except Exception:
                        continue
                    span_hours = float(np.max(x) - np.min(x)) if len(x) else 0.0
                    model_name = f"TF_KRR_poly{degree}{'_rbf_fourier' if use_rbf_fourier else ''}"
                    row = {
                        "distance_km": float(pred),
                        "model": model_name,
                        "cv_rmse_km": cv_rmse,
                        "points": float(n_points),
                        "span_hours": span_hours,
                        "ridge": float(ridge),
                        "cadence_hours": float(cadence_hours),
                    }
                    if best is None or (row["cv_rmse_km"], row["points"], row["ridge"]) < (
                        best["cv_rmse_km"],
                        best["points"],
                        best["ridge"],
                    ):
                        best = row
    if best is not None:
        return best
    idx = int(np.argmin(np.abs(jd_arr - float(target_jd))))
    return {
        "distance_km": float(dist_km[idx]),
        "model": "nearest_sample_fallback",
        "cv_rmse_km": float("nan"),
        "points": 1.0,
        "span_hours": 0.0,
        "ridge": float("nan"),
        "cadence_hours": float(cadence_hours),
    }


def _estimate_local_curve_minimum(
    jd: Any,
    log_distance: Any,
    center_jd: float,
    window_days: float | None = None,
    min_points: int = 7,
    max_points: int = 19,
    max_degree: int = 4,
) -> tuple[float, float, int, float]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    y_arr = np.asarray(log_distance, dtype=float)
    finite = np.isfinite(jd_arr) & np.isfinite(y_arr)
    jd_arr = jd_arr[finite]
    y_arr = y_arr[finite]
    if window_days is not None:
        keep = np.abs(jd_arr - float(center_jd)) <= float(window_days)
        jd_arr = jd_arr[keep]
        y_arr = y_arr[keep]
    if len(jd_arr) < 3:
        idx = int(np.argmin(y_arr))
        return float(jd_arr[idx]), float(y_arr[idx]), 0, 0.0

    min_idx = int(np.argmin(y_arr))
    local_center = float(jd_arr[min_idx])
    order = np.argsort(np.abs(jd_arr - local_center))
    k = min(max_points, max(min_points, min(len(order), max_degree + 3)))
    chosen = np.sort(order[:k])
    t = jd_arr[chosen] - local_center
    y = y_arr[chosen]
    if len(t) < 3:
        return float(local_center), float(y_arr[min_idx]), 0, 0.0

    scale = max(float(np.max(np.abs(t))), 1e-9)
    x = t / scale
    degree = min(max_degree, len(x) - 1)
    weights = 1.0 / (1.0 + np.abs(x))
    try:
        coeff = np.polyfit(x, y, deg=degree, w=weights)
        candidates = [float(np.min(x)), float(np.max(x)), 0.0]
        deriv = np.polyder(coeff)
        roots = np.roots(deriv)
        for root in roots:
            if abs(root.imag) < 1e-9:
                xr = float(root.real)
                if float(np.min(x)) <= xr <= float(np.max(x)):
                    candidates.append(xr)
        values = [float(np.polyval(coeff, xr)) for xr in candidates]
        best_idx = int(np.argmin(values))
        best_x = float(candidates[best_idx])
        best_y = float(values[best_idx])
        best_jd = float(local_center + best_x * scale)
        span_hours = float((np.max(t) - np.min(t)) * 24.0)
        return best_jd, best_y, int(degree), span_hours
    except Exception:
        return float(local_center), float(y_arr[min_idx]), 0, float((np.max(t) - np.min(t)) * 24.0)


def _solve_state_taylor_minimum(
    pos_au: Any,
    vel_au_d: Any,
    accel_au_d2: Any,
    jerk_au_d3: Any | None,
    max_abs_days: float,
) -> tuple[float, float]:
    import numpy as np

    r0 = np.asarray(pos_au, dtype=float)
    v0 = np.asarray(vel_au_d, dtype=float)
    a0 = np.asarray(accel_au_d2, dtype=float)
    if jerk_au_d3 is None:
        j0 = np.zeros_like(a0)
    else:
        j0 = np.asarray(jerk_au_d3, dtype=float)
    radius0 = float(np.linalg.norm(r0))
    if not np.all(np.isfinite(r0)) or not np.all(np.isfinite(v0)) or not np.all(np.isfinite(a0)) or not np.all(np.isfinite(j0)):
        return 0.0, radius0

    max_abs = max(float(max_abs_days), 1.0 / 24.0)

    def radius_at(dt_days: float) -> float:
        vec = r0 + v0 * dt_days + 0.5 * a0 * dt_days * dt_days + (j0 * dt_days * dt_days * dt_days) / 6.0
        return float(np.linalg.norm(vec))

    pos_coeff = np.stack([r0, v0, 0.5 * a0, j0 / 6.0], axis=1)
    vel_coeff = np.stack([v0, a0, 0.5 * j0], axis=1)
    deriv_coeff = np.zeros(6, dtype=float)
    for axis in range(pos_coeff.shape[0]):
        deriv_coeff[: pos_coeff.shape[1] + vel_coeff.shape[1] - 1] += np.convolve(pos_coeff[axis], vel_coeff[axis])
    candidates = [-max_abs, 0.0, max_abs]
    try:
        trimmed = np.trim_zeros(deriv_coeff[::-1], trim="f")
        if len(trimmed) > 1:
            roots = np.roots(trimmed)
        else:
            roots = []
        for root in roots:
            if abs(root.imag) < 1e-10:
                dt = float(root.real)
                if -max_abs <= dt <= max_abs:
                    candidates.append(dt)
    except Exception:
        pass

    best_dt = 0.0
    best_radius = radius0
    for dt in candidates:
        radius = radius_at(float(dt))
        if radius < best_radius:
            best_dt = float(dt)
            best_radius = float(radius)
    return best_dt, max(best_radius, 1e-12)


def _fit_local_state_osculating_center(
    jd: Any,
    pos_au: Any,
    center_jd: float,
    window_days: float,
    max_points: int = 21,
    max_degree: int = 5,
) -> dict[str, Any]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    pos_arr = np.asarray(pos_au, dtype=float)
    finite = np.isfinite(jd_arr) & np.all(np.isfinite(pos_arr), axis=1)
    jd_arr = jd_arr[finite]
    pos_arr = pos_arr[finite]
    if len(jd_arr) < 4:
        idx = int(np.argmin(np.abs(jd_arr - float(center_jd))))
        return {
            "center_jd": float(jd_arr[idx]),
            "pos": np.asarray(pos_arr[idx], dtype=float),
            "vel": np.zeros(3, dtype=float),
            "acc": np.zeros(3, dtype=float),
            "jerk": np.zeros(3, dtype=float),
            "degree": 0,
            "span_hours": 0.0,
        }

    mask = np.abs(jd_arr - float(center_jd)) <= float(window_days)
    subset_jd = jd_arr[mask]
    subset_pos = pos_arr[mask]
    if len(subset_jd) < 6:
        order = np.argsort(np.abs(jd_arr - float(center_jd)))
        chosen = np.sort(order[: min(max_points, len(order))])
        subset_jd = jd_arr[chosen]
        subset_pos = pos_arr[chosen]

    order = np.argsort(np.abs(subset_jd - float(center_jd)))
    chosen = np.sort(order[: min(max_points, len(order))])
    subset_jd = subset_jd[chosen]
    subset_pos = subset_pos[chosen]
    t = subset_jd - float(center_jd)
    if len(t) < 4:
        idx = int(np.argmin(np.abs(jd_arr - float(center_jd))))
        return {
            "center_jd": float(jd_arr[idx]),
            "pos": np.asarray(pos_arr[idx], dtype=float),
            "vel": np.zeros(3, dtype=float),
            "acc": np.zeros(3, dtype=float),
            "jerk": np.zeros(3, dtype=float),
            "degree": 0,
            "span_hours": 0.0,
        }

    degree = min(max_degree, len(t) - 1)
    scale = max(float(np.max(np.abs(t))), 1e-9)
    x = t / scale
    weights = 1.0 / (1.0 + np.abs(x))
    polys = []
    for axis in range(3):
        try:
            coeff = np.polyfit(x, subset_pos[:, axis], deg=degree, w=weights)
            polys.append(np.poly1d(coeff))
        except Exception:
            idx = int(np.argmin(np.abs(jd_arr - float(center_jd))))
            return {
                "center_jd": float(jd_arr[idx]),
                "pos": np.asarray(pos_arr[idx], dtype=float),
                "vel": np.zeros(3, dtype=float),
                "acc": np.zeros(3, dtype=float),
                "jerk": np.zeros(3, dtype=float),
                "degree": 0,
                "span_hours": float((np.max(t) - np.min(t)) * 24.0),
            }

    def eval_state(x0: float) -> tuple[Any, Any, Any, Any]:
        pos0 = np.asarray([float(poly(x0)) for poly in polys], dtype=float)
        vel0 = np.asarray([float(np.polyder(poly, 1)(x0)) / scale for poly in polys], dtype=float)
        acc0 = np.asarray([float(np.polyder(poly, 2)(x0)) / (scale * scale) for poly in polys], dtype=float)
        jerk0 = np.asarray([float(np.polyder(poly, 3)(x0)) / (scale * scale * scale) for poly in polys], dtype=float)
        return pos0, vel0, acc0, jerk0

    pos0, vel0, acc0, jerk0 = eval_state(0.0)
    dt_min, _ = _solve_state_taylor_minimum(pos0, vel0, acc0, jerk0, min(float(window_days), scale))
    x_min = float(np.clip(dt_min / scale, float(np.min(x)), float(np.max(x))))
    pos_min, vel_min, acc_min, jerk_min = eval_state(x_min)
    return {
        "center_jd": float(center_jd + dt_min),
        "pos": pos_min,
        "vel": vel_min,
        "acc": acc_min,
        "jerk": jerk_min,
        "degree": int(degree),
        "span_hours": float((np.max(t) - np.min(t)) * 24.0),
    }


def _encounter_center_objective(
    jd: Any,
    true_log_distance: Any,
    pred_log_distance: Any,
    encounter_center_jd: Any,
    encounter_speed_km_s: Any,
    window_days: float,
) -> dict[str, Any]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    true_arr = np.asarray(true_log_distance, dtype=float)
    pred_arr = np.asarray(pred_log_distance, dtype=float)
    center_arr = np.asarray(encounter_center_jd, dtype=float)
    speed_arr = np.asarray(encounter_speed_km_s, dtype=float)
    finite = (
        np.isfinite(jd_arr)
        & np.isfinite(true_arr)
        & np.isfinite(pred_arr)
        & np.isfinite(center_arr)
        & np.isfinite(speed_arr)
    )
    jd_arr = jd_arr[finite]
    true_arr = true_arr[finite]
    pred_arr = pred_arr[finite]
    center_arr = center_arr[finite]
    speed_arr = speed_arr[finite]
    if len(jd_arr) == 0:
        return {
            "encounter_count": 0.0,
            "timing_hours_mean": float("nan"),
            "timing_hours_median": float("nan"),
            "depth_error_km_mean": float("nan"),
            "depth_error_km_median": float("nan"),
            "combined_km_mean": float("nan"),
            "combined_km_median": float("nan"),
            "records": [],
        }

    records: list[dict[str, float]] = []
    for center_jd in np.unique(center_arr):
        mask = center_arr == center_jd
        mask &= np.abs(jd_arr - center_jd) <= float(window_days)
        if int(np.count_nonzero(mask)) < 5:
            continue
        true_min_jd, true_min_log, _, span_hours = _estimate_local_curve_minimum(
            jd_arr[mask],
            true_arr[mask],
            float(center_jd),
            window_days=float(window_days),
        )
        pred_min_jd, pred_min_log, _, _ = _estimate_local_curve_minimum(
            jd_arr[mask],
            pred_arr[mask],
            float(center_jd),
            window_days=float(window_days),
        )
        timing_hours = abs(pred_min_jd - true_min_jd) * 24.0
        depth_error_km = abs((10.0 ** pred_min_log - 10.0 ** true_min_log) * AU_KM)
        speed_km_s = float(np.nanmedian(speed_arr[mask]))
        timing_equiv_km = abs(pred_min_jd - true_min_jd) * SECONDS_PER_DAY * max(speed_km_s, 0.0)
        combined_km = math.sqrt(depth_error_km * depth_error_km + timing_equiv_km * timing_equiv_km)
        records.append(
            {
                "center_jd": float(center_jd),
                "timing_hours": float(timing_hours),
                "depth_error_km": float(depth_error_km),
                "timing_equiv_km": float(timing_equiv_km),
                "combined_km": float(combined_km),
                "window_span_hours": float(span_hours),
            }
        )

    if not records:
        return {
            "encounter_count": 0.0,
            "timing_hours_mean": float("nan"),
            "timing_hours_median": float("nan"),
            "depth_error_km_mean": float("nan"),
            "depth_error_km_median": float("nan"),
            "combined_km_mean": float("nan"),
            "combined_km_median": float("nan"),
            "records": [],
        }

    timing = np.asarray([row["timing_hours"] for row in records], dtype=float)
    depth = np.asarray([row["depth_error_km"] for row in records], dtype=float)
    combined = np.asarray([row["combined_km"] for row in records], dtype=float)
    return {
        "encounter_count": float(len(records)),
        "timing_hours_mean": float(np.mean(timing)),
        "timing_hours_median": float(np.median(timing)),
        "depth_error_km_mean": float(np.mean(depth)),
        "depth_error_km_median": float(np.median(depth)),
        "combined_km_mean": float(np.mean(combined)),
        "combined_km_median": float(np.median(combined)),
        "records": records,
    }


def _build_time_block_partitions(
    jd: Any,
    neo: NEOObject,
    refine_step: str,
    refine_window_days: float,
) -> tuple[Any, Any, Any, dict[str, float]]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    n = len(jd_arr)
    idx = np.arange(n)
    block_count = int(np.clip(n // 160, 8, 18))
    block_edges = np.linspace(0, n, block_count + 1, dtype=int)
    train_mask = np.ones(n, dtype=bool)
    calib_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    for block in range(block_count):
        start = block_edges[block]
        stop = block_edges[block + 1]
        role = block % 6
        if role == 2:
            val_mask[start:stop] = True
        elif role == 4:
            calib_mask[start:stop] = True
    train_mask &= ~(calib_mask | val_mask)

    step_days = max(_parse_step_days(refine_step), 1.0 / 1440.0)
    embargo_days = max(3.0 * step_days, min(float(refine_window_days) / 12.0, 0.35))
    forced_anchor_points = 0
    for ca in neo.close_approaches:
        if jd_arr[0] <= ca.jd_tdb <= jd_arr[-1]:
            near = np.abs(jd_arr - ca.jd_tdb) <= embargo_days
            if not np.any(near):
                near[int(np.argmin(np.abs(jd_arr - ca.jd_tdb)))] = True
            forced_anchor_points += int(np.count_nonzero(near & ~val_mask))
            val_mask |= near
            calib_mask &= ~near
    train_mask = ~(calib_mask | val_mask)

    if int(np.count_nonzero(val_mask)) < max(10, n // 10):
        val_mask |= (idx % 7) == 3
        calib_mask &= ~val_mask
    if int(np.count_nonzero(calib_mask)) < max(20, n // 12):
        candidate = np.flatnonzero(~val_mask)
        stride = max(9, len(candidate) // max(n // 14, 1))
        extra = candidate[::stride]
        calib_mask[extra] = True
        calib_mask &= ~val_mask
    train_mask = ~(calib_mask | val_mask)
    diagnostics = {
        "validation_blocks": float(block_count),
        "cad_embargo_days": float(embargo_days),
        "cad_forced_validation_points": float(forced_anchor_points),
        "train_fraction": float(np.mean(train_mask)),
        "calibration_fraction": float(np.mean(calib_mask)),
        "validation_fraction": float(np.mean(val_mask)),
    }
    return train_mask, calib_mask, val_mask, diagnostics


def _numerical_training_weights(meta: dict[str, Any], train_mask: Any) -> Any:
    import numpy as np

    jd = np.asarray(meta["jd"], dtype=float)
    dist = np.asarray(meta["dist_au"], dtype=float)
    dt = np.gradient(jd)
    density = np.clip(dt / max(float(np.nanmedian(dt)), 1e-9), 0.25, 4.0)
    log_dist = np.log(np.maximum(dist, 1e-300))
    first = np.gradient(log_dist, jd)
    second = np.gradient(first, jd)
    curvature = np.abs(second)
    curvature_scale = float(np.nanmedian(curvature[train_mask])) or float(np.nanmean(curvature)) or 1.0
    curvature_weight = 1.0 + np.clip(curvature / max(curvature_scale, 1e-18), 0.0, 8.0) / 4.0
    close_scale = max(float(np.nanpercentile(dist[train_mask], 15)), 1e-6)
    close_weight = 1.0 + 2.0 / (1.0 + (dist / close_scale) ** 2)
    weights = density * curvature_weight * close_weight
    train_median = float(np.nanmedian(weights[train_mask])) or 1.0
    weights = np.clip(weights / train_median, 0.1, 10.0)
    return weights


def _fit_model_with_weights(model: Any, X_train: Any, y_train: Any, sample_weight: Any) -> None:
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return
    except (TypeError, ValueError):
        pass
    if hasattr(model, "steps") and getattr(model, "steps"):
        final_name = model.steps[-1][0]
        try:
            model.fit(X_train, y_train, **{f"{final_name}__sample_weight": sample_weight})
            return
        except (TypeError, ValueError):
            pass
    model.fit(X_train, y_train)


def _localized_conformal_widths(
    jd: Any,
    dist_au: Any,
    val_mask: Any,
    abs_log_error: Any,
    quantile: float = 0.90,
) -> Any:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    log_dist = np.log10(np.maximum(np.asarray(dist_au, dtype=float), 1e-300))
    val_idx = np.flatnonzero(val_mask)
    val_err = np.asarray(abs_log_error, dtype=float)
    if len(val_idx) == 0:
        return np.zeros_like(jd_arr)
    global_q = float(np.quantile(val_err, quantile))
    tj = jd_arr[val_idx]
    td = log_dist[val_idx]
    tj_scale = max(float(np.nanstd(tj)), 1e-9)
    td_scale = max(float(np.nanstd(td)), 1e-9)
    widths = np.empty_like(jd_arr)
    for i, (jdi, ldi) in enumerate(zip(jd_arr, log_dist)):
        d2 = ((tj - jdi) / tj_scale) ** 2 + ((td - ldi) / td_scale) ** 2
        k = min(len(val_idx), max(12, int(math.sqrt(len(val_idx)))))
        near = np.argpartition(d2, k - 1)[:k]
        local_q = float(np.quantile(val_err[near], quantile))
        widths[i] = max(global_q, local_q)
    return widths


def _build_purged_time_splits(
    jd: Any,
    n_splits: int,
    embargo_days: float,
    min_train_points: int = 120,
) -> list[tuple[Any, Any]]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    idx = np.arange(len(jd_arr))
    if len(idx) < max(min_train_points + 20, 80):
        return []
    edges = np.linspace(0, len(idx), n_splits + 1, dtype=int)
    splits: list[tuple[Any, Any]] = []
    for fold in range(n_splits):
        test_idx = idx[edges[fold] : edges[fold + 1]]
        if len(test_idx) < 12:
            continue
        left = float(jd_arr[test_idx[0]] - embargo_days)
        right = float(jd_arr[test_idx[-1]] + embargo_days)
        train_idx = idx[(jd_arr < left) | (jd_arr > right)]
        if len(train_idx) < min_train_points:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _r2_score_np(y_true: Any, y_pred: Any) -> float:
    import numpy as np

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((pred - true) ** 2))
    centered = true - float(np.mean(true))
    ss_tot = float(np.sum(centered * centered))
    return 1.0 - ss_res / max(ss_tot, 1e-18)


def _median_absolute_error_np(y_true: Any, y_pred: Any) -> float:
    import numpy as np

    return float(np.median(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))


def _tensorflow_available_version() -> str:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise SourceError(
            "TensorFlow is required for the legacy neural residual helpers in this build. "
            "Install it in the project environment, e.g. `/Users/sebastianp/Asteroid-NEO/.venv/bin/pip install tensorflow`."
        ) from exc
    return str(tf.__version__)


class _TensorFlowTabularRegressor:
    def __init__(
        self,
        name: str,
        hidden_layers: tuple[int, ...],
        learning_rate: float,
        dropout: float,
        l2: float,
        epochs: int,
        batch_size: int,
        seed: int,
        loss: str = "huber",
    ) -> None:
        self.name = name
        self.hidden_layers = hidden_layers
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.loss = loss
        self.model: Any | None = None
        self.x_mean: Any | None = None
        self.x_scale: Any | None = None
        self.feature_importances_: Any | None = None
        self.y_mean = 0.0
        self.y_scale = 1.0

    def _build_model(self, n_features: int) -> Any:
        import tensorflow as tf  # type: ignore

        try:
            tf.keras.utils.set_random_seed(self.seed)
        except Exception:
            pass
        regularizer = tf.keras.regularizers.l2(self.l2) if self.l2 > 0.0 else None
        layers: list[Any] = [tf.keras.layers.Input(shape=(n_features,))]
        for width in self.hidden_layers:
            layers.append(
                tf.keras.layers.Dense(
                    int(width),
                    activation="swish",
                    kernel_regularizer=regularizer,
                )
            )
            layers.append(tf.keras.layers.LayerNormalization())
            if self.dropout > 0.0:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(layers, name=self.name)
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.l2)
        except Exception:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.loss == "mse":
            loss_obj: Any = tf.keras.losses.MeanSquaredError()
        else:
            loss_obj = tf.keras.losses.Huber(delta=1.0)
        model.compile(optimizer=optimizer, loss=loss_obj)
        return model

    def fit(self, X_train: Any, y_train: Any, sample_weight: Any | None = None) -> "_TensorFlowTabularRegressor":
        import numpy as np
        import tensorflow as tf  # type: ignore

        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1, 1)
        if len(X) == 0:
            raise ValueError("cannot fit TensorFlow model on empty data")
        self.x_mean = np.nanmean(X, axis=0)
        self.x_scale = np.nanstd(X, axis=0)
        self.x_scale = np.where(np.asarray(self.x_scale) > 1e-12, self.x_scale, 1.0)
        Xn = (np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) - self.x_mean) / self.x_scale
        self.y_mean = float(np.nanmean(y))
        self.y_scale = float(np.nanstd(y)) or 1.0
        if not math.isfinite(self.y_scale) or self.y_scale < 1e-12:
            self.y_scale = 1.0
        yn = (y - self.y_mean) / self.y_scale
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights is not None:
            weights = np.clip(weights, 1e-9, None)

        self.model = self._build_model(Xn.shape[1])
        callbacks: list[Any] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=18,
                restore_best_weights=True,
                min_delta=1e-5,
            )
        ]
        validation_split = 0.12 if len(Xn) >= 240 else 0.0
        fit_kwargs: dict[str, Any] = {
            "epochs": self.epochs,
            "batch_size": min(self.batch_size, max(16, len(Xn))),
            "verbose": 0,
            "shuffle": False,
        }
        if validation_split > 0.0:
            fit_kwargs["validation_split"] = validation_split
            fit_kwargs["callbacks"] = callbacks
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        self.model.fit(Xn, yn, **fit_kwargs)
        try:
            first_kernel = self.model.layers[0].get_weights()[0]
            salience = np.mean(np.abs(first_kernel), axis=1)
            total = float(np.sum(salience))
            self.feature_importances_ = salience / total if total > 0.0 else salience
        except Exception:
            self.feature_importances_ = None
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np

        if self.model is None or self.x_mean is None or self.x_scale is None:
            raise ValueError("TensorFlow model has not been fitted")
        arr = np.asarray(X, dtype=float)
        Xn = (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) - self.x_mean) / self.x_scale
        pred = self.model.predict(Xn, verbose=0).reshape(-1)
        return pred * self.y_scale + self.y_mean


def _bootstrap_metric_band(
    y_true: Any,
    y_pred: Any,
    metric_name: str,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)
    if n == 0:
        return {"low": float("nan"), "median": float("nan"), "high": float("nan")}

    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        err = y_pred_arr[sample] - y_true_arr[sample]
        if metric_name == "mae":
            values[i] = float(np.mean(np.abs(err)))
        elif metric_name == "rmse":
            values[i] = float(np.sqrt(np.mean(err * err)))
        else:
            raise ValueError(f"unsupported metric_name={metric_name!r}")
    q = np.quantile(values, [0.05, 0.50, 0.95])
    return {"low": float(q[0]), "median": float(q[1]), "high": float(q[2])}


def _coverage_by_distance_regime(
    dist_au: Any,
    covered: Any,
    pred_km: Any,
    true_km: Any,
) -> dict[str, dict[str, float]]:
    import numpy as np

    dist_arr = np.asarray(dist_au, dtype=float)
    covered_arr = np.asarray(covered, dtype=bool)
    pred_arr = np.asarray(pred_km, dtype=float)
    true_arr = np.asarray(true_km, dtype=float)
    if len(dist_arr) == 0:
        return {}
    q1, q2 = np.quantile(dist_arr, [1.0 / 3.0, 2.0 / 3.0])
    regimes = {
        "near": dist_arr <= q1,
        "mid": (dist_arr > q1) & (dist_arr <= q2),
        "far": dist_arr > q2,
    }
    summary: dict[str, dict[str, float]] = {}
    for name, mask in regimes.items():
        if not np.any(mask):
            continue
        err = pred_arr[mask] - true_arr[mask]
        summary[name] = {
            "n": float(np.count_nonzero(mask)),
            "coverage": float(np.mean(covered_arr[mask])),
            "mae_km": float(np.mean(np.abs(err))),
            "rmse_km": float(np.sqrt(np.mean(err * err))),
            "median_distance_km": float(np.median(true_arr[mask])),
        }
    return summary


def _select_primary_tensorflow_model(
    X_train: Any,
    y_train: Any,
    train_weight: Any,
    jd_train: Any,
    embargo_days: float,
    baseline_log_train: Any | None = None,
) -> tuple[str, Any, dict[str, dict[str, float]], int]:
    import numpy as np

    _tensorflow_available_version()
    baseline = np.zeros_like(np.asarray(y_train, dtype=float)) if baseline_log_train is None else np.asarray(baseline_log_train, dtype=float)
    residual_target = np.asarray(y_train, dtype=float) - baseline
    candidate_factories: dict[str, Any] = {
        "TF_MLP_Huber_compact": lambda: _TensorFlowTabularRegressor(
            name="tf_mlp_huber_compact",
            hidden_layers=(96, 48),
            learning_rate=8.0e-4,
            dropout=0.04,
            l2=1.0e-5,
            epochs=260,
            batch_size=96,
            seed=37,
            loss="huber",
        ),
        "TF_MLP_Huber_deep": lambda: _TensorFlowTabularRegressor(
            name="tf_mlp_huber_deep",
            hidden_layers=(192, 96, 48),
            learning_rate=6.0e-4,
            dropout=0.06,
            l2=2.0e-5,
            epochs=320,
            batch_size=128,
            seed=41,
            loss="huber",
        ),
        "TF_MLP_MSE_wide": lambda: _TensorFlowTabularRegressor(
            name="tf_mlp_mse_wide",
            hidden_layers=(256, 128, 64),
            learning_rate=5.0e-4,
            dropout=0.05,
            l2=2.0e-5,
            epochs=300,
            batch_size=128,
            seed=43,
            loss="mse",
        ),
    }
    split_count = int(np.clip(len(y_train) // 900, 3, 4))
    splits = _build_purged_time_splits(jd_train, split_count, embargo_days=max(float(embargo_days), 1.0 / 24.0))
    if not splits:
        fallback_name = "TF_MLP_Huber_compact"
        fallback_model = candidate_factories[fallback_name]()
        _fit_model_with_weights(fallback_model, X_train, residual_target, train_weight)
        return fallback_name, fallback_model, {}, 0

    scores: dict[str, dict[str, float]] = {}
    for name, factory in candidate_factories.items():
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []
        for fit_idx, test_idx in splits:
            model = factory()
            _fit_model_with_weights(model, X_train[fit_idx], residual_target[fit_idx], train_weight[fit_idx])
            pred = baseline[test_idx] + np.asarray(model.predict(X_train[test_idx]), dtype=float)
            true_km = 10.0 ** y_train[test_idx] * AU_KM
            pred_km = 10.0 ** pred * AU_KM
            err = pred_km - true_km
            fold_mae.append(float(np.mean(np.abs(err))))
            fold_rmse.append(float(np.sqrt(np.mean(err * err))))
            fold_r2.append(_r2_score_np(y_train[test_idx], pred))
        scores[name] = {
            "cv_mae_km_mean": float(np.mean(fold_mae)),
            "cv_mae_km_std": float(np.std(fold_mae)),
            "cv_rmse_km_mean": float(np.mean(fold_rmse)),
            "cv_rmse_km_std": float(np.std(fold_rmse)),
            "cv_r2_log_mean": float(np.mean(fold_r2)),
            "cv_r2_log_std": float(np.std(fold_r2)),
        }

    best_name = min(
        scores,
        key=lambda name: (
            scores[name]["cv_rmse_km_mean"],
            scores[name]["cv_rmse_km_std"],
            scores[name]["cv_mae_km_mean"],
        ),
    )
    best_model = candidate_factories[best_name]()
    _fit_model_with_weights(best_model, X_train, residual_target, train_weight)
    return best_name, best_model, scores, len(splits)


def _build_local_refinement_design(
    meta: dict[str, Any],
    neo: NEOObject,
    refine_window_days: float,
    base_X: Any | None = None,
    base_feature_names: list[str] | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    import numpy as np

    jd = np.asarray(meta["jd"], dtype=float)
    dist = np.asarray(meta["dist_au"], dtype=float)
    geo_pos = np.asarray(meta["geo_pos_au"], dtype=float)
    geo_vel = np.asarray(meta["geo_vel_au_d"], dtype=float)
    geo_accel_vec = np.asarray(meta["geo_accel_au_d2"], dtype=float)
    geo_jerk_vec = np.asarray(meta["geo_jerk_au_d3"], dtype=float)
    helio_pos = np.asarray(meta["helio_pos_au"], dtype=float)
    helio_vel = np.asarray(meta["helio_vel_au_d"], dtype=float)
    range_rate = np.asarray(meta["geo_range_rate_au_d"], dtype=float)
    transverse = np.asarray(meta["geo_transverse_speed_au_d"], dtype=float)
    geo_speed = np.asarray(meta["geo_speed_au_d"], dtype=float)
    geo_accel = np.asarray(meta["geo_accel_norm_au_d2"], dtype=float)
    range_accel = np.asarray(meta["geo_range_accel_au_d2"], dtype=float)
    range_jerk = np.asarray(meta["geo_range_jerk_au_d3"], dtype=float)
    curvature = np.asarray(meta["geo_log_range_curvature_d2"], dtype=float)
    third = np.asarray(meta["geo_log_range_third_d3"], dtype=float)
    geo_speed_slope = np.asarray(meta["geo_speed_slope_au_d2"], dtype=float)
    helio_speed = np.asarray(meta["helio_speed_au_d"], dtype=float)
    helio_accel = np.asarray(meta["helio_accel_norm_au_d2"], dtype=float)
    helio_speed_slope = np.asarray(meta["helio_speed_slope_au_d2"], dtype=float)
    helio_energy = np.asarray(meta["helio_specific_energy_m2_s2"], dtype=float)
    helio_ecc = np.asarray(meta["helio_eccentricity_inst"], dtype=float)
    helio_sma = np.asarray(meta["helio_sma_inst_au"], dtype=float)
    cos_geo_helio = np.asarray(meta["cos_geo_helio"], dtype=float)
    gi = np.asarray(meta["gi_log"], dtype=float)
    oi = np.asarray(meta["oi_log"], dtype=float)
    pdf_arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in sorted(meta.get("pdf_arrays", {}).items())
    }

    if neo.close_approaches:
        cad_jd = np.asarray([float(ca.jd_tdb) for ca in neo.close_approaches], dtype=float)
        cad_dist = np.asarray([float(ca.distance_au) for ca in neo.close_approaches], dtype=float)
        delta = jd[:, None] - cad_jd[None, :]
        nearest = np.argmin(np.abs(delta), axis=1)
        nearest_cad_distance_au = cad_dist[nearest]
    else:
        fallback_idx = int(np.argmin(dist))
        cad_jd = np.asarray([float(jd[fallback_idx])], dtype=float)
        cad_dist = np.asarray([float(dist[fallback_idx])], dtype=float)
        nearest = np.zeros(len(jd), dtype=int)
        nearest_cad_distance_au = np.full(len(jd), cad_dist[0], dtype=float)

    encounter_sample_idx = np.asarray([int(np.argmin(np.abs(jd - center_jd))) for center_jd in cad_jd], dtype=int)
    encounter_sample_jd = jd[encounter_sample_idx]

    def _normalize_rows(values: Any, fallback: Any) -> Any:
        arr = np.asarray(values, dtype=float)
        out = np.asarray(fallback, dtype=float).copy()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        np.divide(arr, np.maximum(norms, 1e-300), out=out, where=norms > 1e-300)
        return out

    current_r_hat = geo_pos / np.maximum(dist[:, None], 1e-300)
    current_h_hat = _normalize_rows(
        np.cross(geo_pos, geo_vel),
        np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=float), len(jd), axis=0),
    )
    current_t_hat = _normalize_rows(
        np.cross(current_h_hat, current_r_hat),
        _normalize_rows(
            geo_vel,
            np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=float), len(jd), axis=0),
        ),
    )

    ref_geo_pos = geo_pos[encounter_sample_idx][nearest]
    ref_geo_vel = geo_vel[encounter_sample_idx][nearest]
    ref_geo_accel_vec = geo_accel_vec[encounter_sample_idx][nearest]
    ref_geo_jerk_vec = geo_jerk_vec[encounter_sample_idx][nearest]
    ref_helio_pos = helio_pos[encounter_sample_idx][nearest]
    ref_helio_vel = helio_vel[encounter_sample_idx][nearest]
    ref_dist = dist[encounter_sample_idx][nearest]
    ref_geo_speed = geo_speed[encounter_sample_idx][nearest]
    ref_helio_r = np.linalg.norm(helio_pos[encounter_sample_idx], axis=1)[nearest]
    ref_helio_speed = helio_speed[encounter_sample_idx][nearest]
    ref_helio_energy = helio_energy[encounter_sample_idx][nearest]
    ref_helio_ecc = helio_ecc[encounter_sample_idx][nearest]
    ref_helio_sma = helio_sma[encounter_sample_idx][nearest]
    ref_gi = gi[encounter_sample_idx][nearest]
    ref_oi = oi[encounter_sample_idx][nearest]
    ref_range_rate = range_rate[encounter_sample_idx][nearest]
    ref_range_accel = range_accel[encounter_sample_idx][nearest]
    ref_range_jerk = range_jerk[encounter_sample_idx][nearest]
    ref_transverse = transverse[encounter_sample_idx][nearest]
    subhour_window_days = min(max(float(refine_window_days) * 0.25, 2.0 / 24.0), 1.0)
    encounter_state_min_dt = np.asarray(
        [
            _solve_state_taylor_minimum(
                geo_pos[idx],
                geo_vel[idx],
                geo_accel_vec[idx],
                geo_jerk_vec[idx],
                subhour_window_days,
            )[0]
            for idx in encounter_sample_idx
        ],
        dtype=float,
    )
    encounter_state_min_dist = np.asarray(
        [
            _solve_state_taylor_minimum(
                geo_pos[idx],
                geo_vel[idx],
                geo_accel_vec[idx],
                geo_jerk_vec[idx],
                subhour_window_days,
            )[1]
            for idx in encounter_sample_idx
        ],
        dtype=float,
    )
    encounter_center_jd_by_row = (encounter_sample_jd + encounter_state_min_dt)[nearest]
    encounter_sample_jd_by_row = encounter_sample_jd[nearest]
    baseline_min_distance_au = encounter_state_min_dist[nearest]
    encounter_speed_km_s_by_row = ref_geo_speed * AU_KM / SECONDS_PER_DAY
    signed_dt_to_cad_days = jd - cad_jd[nearest]
    signed_dt_from_sample_days = jd - encounter_sample_jd_by_row
    signed_dt_days = jd - encounter_center_jd_by_row
    abs_dt_days = np.abs(signed_dt_days)

    ref_r_hat = _normalize_rows(ref_geo_pos, current_r_hat)
    ref_h_hat = _normalize_rows(np.cross(ref_geo_pos, ref_geo_vel), current_h_hat)
    ref_t_hat = _normalize_rows(np.cross(ref_h_hat, ref_r_hat), current_t_hat)

    delta_geo_pos = geo_pos - ref_geo_pos
    delta_geo_vel = geo_vel - ref_geo_vel
    delta_helio_pos = helio_pos - ref_helio_pos
    delta_helio_vel = helio_vel - ref_helio_vel

    delta_geo_pos_rad = np.einsum("ij,ij->i", delta_geo_pos, ref_r_hat)
    delta_geo_pos_tan = np.einsum("ij,ij->i", delta_geo_pos, ref_t_hat)
    delta_geo_pos_norm = np.einsum("ij,ij->i", delta_geo_pos, ref_h_hat)
    delta_geo_vel_rad = np.einsum("ij,ij->i", delta_geo_vel, ref_r_hat)
    delta_geo_vel_tan = np.einsum("ij,ij->i", delta_geo_vel, ref_t_hat)
    delta_geo_vel_norm = np.einsum("ij,ij->i", delta_geo_vel, ref_h_hat)
    delta_helio_pos_norm = np.linalg.norm(delta_helio_pos, axis=1)
    delta_helio_vel_norm = np.linalg.norm(delta_helio_vel, axis=1)
    signed_dt_sample_col = signed_dt_from_sample_days[:, None]
    physics_taylor_pos = (
        ref_geo_pos
        + ref_geo_vel * signed_dt_sample_col
        + 0.5 * ref_geo_accel_vec * (signed_dt_sample_col * signed_dt_sample_col)
        + (ref_geo_jerk_vec * signed_dt_sample_col * signed_dt_sample_col * signed_dt_sample_col) / 6.0
    )
    physics_taylor_dist = np.linalg.norm(physics_taylor_pos, axis=1)
    scalar_taylor_dist = (
        ref_dist
        + ref_range_rate * signed_dt_from_sample_days
        + 0.5 * ref_range_accel * signed_dt_from_sample_days * signed_dt_from_sample_days
        + (ref_range_jerk * signed_dt_from_sample_days * signed_dt_from_sample_days * signed_dt_from_sample_days) / 6.0
    )
    taylor_floor = np.maximum(ref_dist * 1.0e-6, 1e-12)
    physics_taylor_dist = np.maximum(physics_taylor_dist, taylor_floor)
    scalar_taylor_dist = np.maximum(scalar_taylor_dist, taylor_floor)
    physics_taylor_log = np.log10(np.maximum(physics_taylor_dist, 1e-300))
    scalar_taylor_log = np.log10(np.maximum(scalar_taylor_dist, 1e-300))

    range_scale_au = max(float(np.nanquantile(dist, 0.20)), 1e-6)
    near_mask = dist <= float(np.nanquantile(dist, 0.30))
    time_reference = abs_dt_days[near_mask] if np.any(near_mask) else abs_dt_days
    finite_time = time_reference[np.isfinite(time_reference) & (time_reference > 0.0)]
    if len(finite_time):
        time_scale_days = max(float(refine_window_days), float(np.nanquantile(finite_time, 0.35)))
    else:
        time_scale_days = max(float(refine_window_days), 1.0)
    objective_window_days = max(float(refine_window_days), 1.0 / 24.0)

    range_gate = 1.0 / (1.0 + (dist / range_scale_au) ** 2)
    time_gate = 1.0 / (1.0 + (abs_dt_days / max(time_scale_days, 1e-9)) ** 2)
    window_sigma_days = max(objective_window_days * 0.50, 1.0 / 24.0)
    window_gaussian = np.exp(-0.5 * (abs_dt_days / window_sigma_days) ** 2)
    baseline_distance_ratio = dist / np.maximum(baseline_min_distance_au, 1e-300)
    anchor_distance_gate = 1.0 / (1.0 + baseline_distance_ratio * baseline_distance_ratio)
    locality_score = np.sqrt(np.clip(range_gate * time_gate, 0.0, 1.0))
    proposal_taper = np.clip(
        np.sqrt(np.clip(window_gaussian, 0.0, 1.0)) * np.sqrt(np.clip(anchor_distance_gate, 0.0, 1.0)),
        0.0,
        1.0,
    )
    encounter_window_mask = abs_dt_days <= objective_window_days
    local_objective_weight = (
        (0.35 + 0.65 * window_gaussian)
        * (0.35 + 0.65 * np.sqrt(np.clip(anchor_distance_gate, 0.0, 1.0)))
        * (0.50 + 0.50 * locality_score)
    )
    local_objective_weight = np.clip(local_objective_weight, 0.15, 4.0)

    feature_columns: list[Any] = []
    feature_names: list[str] = []

    def add_feature(name: str, values: Any) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(jd), float(arr))
        feature_columns.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        feature_names.append(name)

    if base_X is not None and base_feature_names is not None:
        base_arr = np.asarray(base_X, dtype=float)
        for i, name in enumerate(base_feature_names):
            add_feature(f"global__{name}", base_arr[:, i])

    for name, values in [
        ("signed_dt_days", signed_dt_days),
        ("signed_dt_to_sample_days", signed_dt_from_sample_days),
        ("signed_dt_to_cad_days", signed_dt_to_cad_days),
        ("abs_dt_days", abs_dt_days),
        ("signed_dt_over_scale", signed_dt_days / max(time_scale_days, 1e-9)),
        ("signed_dt_sq_over_scale2", (signed_dt_days / max(time_scale_days, 1e-9)) ** 2),
        ("signed_dt_cu_over_scale3", (signed_dt_days / max(time_scale_days, 1e-9)) ** 3),
        ("log_abs_dt_days", np.log10(np.maximum(abs_dt_days + 1e-6, 1e-6))),
        ("time_gate", time_gate),
        ("window_gaussian_gate", window_gaussian),
        ("range_gate", range_gate),
        ("anchor_distance_gate", anchor_distance_gate),
        ("locality_score", locality_score),
        ("encounter_window_indicator", encounter_window_mask.astype(float)),
        ("log_nearest_cad_distance_au", np.log10(np.maximum(nearest_cad_distance_au, 1e-300))),
        ("log_baseline_min_distance_au", np.log10(np.maximum(baseline_min_distance_au, 1e-300))),
        ("log_geo_to_anchor_distance_ratio", np.log10(np.maximum(baseline_distance_ratio, 1e-300))),
        ("delta_log_geo_distance_to_anchor", np.log10(np.maximum(dist, 1e-300)) - np.log10(np.maximum(baseline_min_distance_au, 1e-300))),
        ("baseline_center_offset_from_sample_hours", (encounter_center_jd_by_row - encounter_sample_jd_by_row) * 24.0),
        ("baseline_center_offset_from_cad_hours", (encounter_center_jd_by_row - cad_jd[nearest]) * 24.0),
        ("log_state_taylor_distance_au", physics_taylor_log),
        ("log_scalar_taylor_distance_au", scalar_taylor_log),
        ("delta_log_state_taylor_to_anchor", physics_taylor_log - np.log10(np.maximum(baseline_min_distance_au, 1e-300))),
        ("delta_log_state_taylor_to_scalar", physics_taylor_log - scalar_taylor_log),
        ("ref_range_rate_dt_au", ref_range_rate * signed_dt_from_sample_days),
        ("ref_range_accel_dt2_au", 0.5 * ref_range_accel * signed_dt_from_sample_days * signed_dt_from_sample_days),
        ("ref_range_jerk_dt3_au", (ref_range_jerk * signed_dt_from_sample_days * signed_dt_from_sample_days * signed_dt_from_sample_days) / 6.0),
        (
            "ref_local_taylor_shift_au",
            ref_range_rate * signed_dt_from_sample_days
            + 0.5 * ref_range_accel * signed_dt_from_sample_days * signed_dt_from_sample_days
            + (ref_range_jerk * signed_dt_from_sample_days * signed_dt_from_sample_days * signed_dt_from_sample_days) / 6.0,
        ),
        ("delta_geo_pos_radial_au", delta_geo_pos_rad),
        ("delta_geo_pos_transverse_au", delta_geo_pos_tan),
        ("delta_geo_pos_normal_au", delta_geo_pos_norm),
        ("delta_geo_pos_radial_norm", delta_geo_pos_rad / np.maximum(ref_dist, range_scale_au)),
        ("delta_geo_pos_transverse_norm", delta_geo_pos_tan / np.maximum(ref_dist, range_scale_au)),
        ("delta_geo_pos_normal_norm", delta_geo_pos_norm / np.maximum(ref_dist, range_scale_au)),
        ("delta_geo_vel_radial_au_d", delta_geo_vel_rad),
        ("delta_geo_vel_transverse_au_d", delta_geo_vel_tan),
        ("delta_geo_vel_normal_au_d", delta_geo_vel_norm),
        ("delta_geo_vel_radial_norm", delta_geo_vel_rad / np.maximum(ref_geo_speed, 1e-300)),
        ("delta_geo_vel_transverse_norm", delta_geo_vel_tan / np.maximum(ref_geo_speed, 1e-300)),
        ("delta_geo_vel_normal_norm", delta_geo_vel_norm / np.maximum(ref_geo_speed, 1e-300)),
        ("delta_helio_pos_norm_au", delta_helio_pos_norm),
        ("delta_helio_vel_norm_au_d", delta_helio_vel_norm),
        ("delta_log_helio_r", np.log10(np.maximum(np.linalg.norm(helio_pos, axis=1), 1e-300)) - np.log10(np.maximum(ref_helio_r, 1e-300))),
        ("delta_helio_speed_au_d", helio_speed - ref_helio_speed),
        ("delta_helio_speed_slope_au_d2", helio_speed_slope - helio_speed_slope[encounter_sample_idx][nearest]),
        ("delta_helio_accel_norm_au_d2", helio_accel - helio_accel[encounter_sample_idx][nearest]),
        ("delta_log_helio_energy", _safe_log10_abs(helio_energy) - _safe_log10_abs(ref_helio_energy)),
        ("delta_helio_eccentricity", helio_ecc - ref_helio_ecc),
        ("delta_log_helio_sma", _safe_log10_abs(helio_sma) - _safe_log10_abs(ref_helio_sma)),
        ("delta_cos_geo_helio", cos_geo_helio - cos_geo_helio[encounter_sample_idx][nearest]),
        ("delta_gi_log10_abs", gi - ref_gi),
        ("delta_oi_log10_abs", oi - ref_oi),
        ("delta_range_rate_au_d", range_rate - ref_range_rate),
        ("delta_transverse_speed_au_d", transverse - ref_transverse),
        ("delta_geo_speed_au_d", geo_speed - ref_geo_speed),
        ("delta_geo_speed_slope_au_d2", geo_speed_slope - geo_speed_slope[encounter_sample_idx][nearest]),
        ("delta_geo_accel_norm_au_d2", geo_accel - geo_accel[encounter_sample_idx][nearest]),
        ("delta_geo_jerk_norm_au_d3", np.linalg.norm(geo_jerk_vec, axis=1) - np.linalg.norm(geo_jerk_vec[encounter_sample_idx][nearest], axis=1)),
        ("delta_geo_range_accel_au_d2", range_accel - range_accel[encounter_sample_idx][nearest]),
        ("delta_geo_range_jerk_au_d3", range_jerk - range_jerk[encounter_sample_idx][nearest]),
        ("delta_geo_log_range_curvature_d2", curvature - curvature[encounter_sample_idx][nearest]),
        ("delta_geo_log_range_third_d3", third - third[encounter_sample_idx][nearest]),
        ("range_rate_dt_au", range_rate * signed_dt_days),
        ("range_accel_dt2_au", 0.5 * range_accel * signed_dt_days * signed_dt_days),
        ("range_jerk_dt3_au", (range_jerk * signed_dt_days * signed_dt_days * signed_dt_days) / 6.0),
        (
            "local_taylor_range_shift_au",
            range_rate * signed_dt_days
            + 0.5 * range_accel * signed_dt_days * signed_dt_days
            + (range_jerk * signed_dt_days * signed_dt_days * signed_dt_days) / 6.0,
        ),
        ("local_objective_weight", local_objective_weight),
    ]:
        add_feature(name, values)

    for name, values in pdf_arrays.items():
        log_values = _safe_log10_abs(values)
        ref_values = log_values[encounter_sample_idx][nearest]
        add_feature(f"local_delta_log_pdf_{name}", log_values - ref_values)

    X_local = np.column_stack(feature_columns)
    diagnostics = {
        "local_range_scale_au": float(range_scale_au),
        "local_time_scale_days": float(time_scale_days),
        "local_objective_window_days": float(objective_window_days),
        "local_encounter_count": float(len(cad_jd)),
        "local_window_fraction": float(np.mean(encounter_window_mask)),
        "local_subhour_search_window_hours": float(subhour_window_days * 24.0),
        "local_baseline_center_offset_from_sample_hours_median": float(np.median(np.abs(encounter_state_min_dt)) * 24.0),
        "local_baseline_center_offset_from_cad_hours_median": float(np.median(np.abs((encounter_sample_jd + encounter_state_min_dt) - cad_jd)) * 24.0),
        "locality_score_median": float(np.nanmedian(locality_score)),
        "locality_score_p90": float(np.nanquantile(locality_score, 0.90)),
        "local_proposal_taper_median": float(np.nanmedian(proposal_taper)),
        "local_proposal_taper_p90": float(np.nanquantile(proposal_taper, 0.90)),
        "local_objective_weight_median": float(np.nanmedian(local_objective_weight)),
        "local_objective_weight_p90": float(np.nanquantile(local_objective_weight, 0.90)),
        "local_taylor_state_scalar_gap_km_median": float(np.nanmedian(np.abs(physics_taylor_dist - scalar_taylor_dist)) * AU_KM),
        "local_basis_feature_count": float(len(feature_names)),
    }
    return X_local, {
        "locality_score": locality_score,
        "proposal_taper": proposal_taper,
        "encounter_window_mask": encounter_window_mask,
        "objective_weight": local_objective_weight,
        "physics_taylor_baseline_log": physics_taylor_log,
        "encounter_center_jd": encounter_center_jd_by_row,
        "encounter_speed_km_s": encounter_speed_km_s_by_row,
        "signed_dt_days": signed_dt_days,
        "abs_dt_days": abs_dt_days,
        "nearest_encounter_index": nearest,
        "diagnostics": diagnostics,
    }, feature_names


def _select_local_refinement_model(
    X_local: Any,
    y_true: Any,
    base_pred_log: Any,
    local_baseline_log: Any,
    locality_score: Any,
    proposal_taper: Any,
    encounter_center_jd: Any,
    encounter_speed_km_s: Any,
    sample_weight: Any,
    jd_local: Any,
    embargo_days: float,
    objective_window_days: float,
) -> tuple[str, Any, dict[str, dict[str, float]], int, float]:
    import numpy as np

    _tensorflow_available_version()

    correction_target = np.asarray(y_true, dtype=float) - np.asarray(local_baseline_log, dtype=float)
    base_pred = np.asarray(base_pred_log, dtype=float)
    local_baseline = np.asarray(local_baseline_log, dtype=float)
    gate = np.asarray(locality_score, dtype=float)
    taper = np.asarray(proposal_taper, dtype=float)
    encounter_jd = np.asarray(encounter_center_jd, dtype=float)
    encounter_speed = np.asarray(encounter_speed_km_s, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)

    factories: dict[str, Any] = {
        "TF_Local_Huber_compact": lambda: _TensorFlowTabularRegressor(
            name="tf_local_huber_compact",
            hidden_layers=(96, 48),
            learning_rate=8.0e-4,
            dropout=0.05,
            l2=2.0e-5,
            epochs=220,
            batch_size=64,
            seed=73,
            loss="huber",
        ),
        "TF_Local_Huber_expressive": lambda: _TensorFlowTabularRegressor(
            name="tf_local_huber_expressive",
            hidden_layers=(160, 80, 40),
            learning_rate=6.0e-4,
            dropout=0.08,
            l2=3.0e-5,
            epochs=260,
            batch_size=64,
            seed=79,
            loss="huber",
        ),
    }

    split_count = int(np.clip(len(y_arr) // 180, 3, 4))
    splits = _build_purged_time_splits(
        jd_local,
        split_count,
        embargo_days=max(float(embargo_days), 1.0 / 24.0),
        min_train_points=max(60, len(y_arr) // 3),
    )
    if not splits:
        fallback_name = "TF_Local_Huber_compact"
        fallback_model = factories[fallback_name]()
        _fit_model_with_weights(fallback_model, X_local, correction_target, weights)
        clip_scale = float(np.quantile(np.abs(correction_target), 0.98))
        return fallback_name, fallback_model, {}, 0, clip_scale

    scores: dict[str, dict[str, float]] = {}
    for name, factory in factories.items():
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_center_combined: list[float] = []
        fold_center_timing: list[float] = []
        fold_center_depth: list[float] = []
        for fit_idx, test_idx in splits:
            model = factory()
            _fit_model_with_weights(model, X_local[fit_idx], correction_target[fit_idx], weights[fit_idx])
            fold_clip = float(np.quantile(np.abs(correction_target[fit_idx]), 0.98))
            pred_corr = np.asarray(model.predict(X_local[test_idx]), dtype=float)
            pred_corr = np.clip(pred_corr, -fold_clip, fold_clip)
            local_proposal_log = local_baseline[test_idx] + pred_corr
            pred_log = base_pred[test_idx] + gate[test_idx] * taper[test_idx] * (local_proposal_log - base_pred[test_idx])
            true_km = 10.0 ** y_arr[test_idx] * AU_KM
            pred_km = 10.0 ** pred_log * AU_KM
            err = pred_km - true_km
            metric_weight = np.clip(weights[test_idx], 1e-9, None)
            fold_mae.append(float(np.average(np.abs(err), weights=metric_weight)))
            fold_rmse.append(float(np.sqrt(np.average(err * err, weights=metric_weight))))
            center_stats = _encounter_center_objective(
                jd_local[test_idx],
                y_arr[test_idx],
                pred_log,
                encounter_jd[test_idx],
                encounter_speed[test_idx],
                objective_window_days,
            )
            if int(center_stats.get("encounter_count", 0.0)) > 0:
                fold_center_combined.append(float(center_stats["combined_km_mean"]))
                fold_center_timing.append(float(center_stats["timing_hours_mean"]))
                fold_center_depth.append(float(center_stats["depth_error_km_mean"]))
        scores[name] = {
            "cv_mae_km_mean": float(np.mean(fold_mae)),
            "cv_mae_km_std": float(np.std(fold_mae)),
            "cv_rmse_km_mean": float(np.mean(fold_rmse)),
            "cv_rmse_km_std": float(np.std(fold_rmse)),
            "cv_center_combined_km_mean": float(np.mean(fold_center_combined)) if fold_center_combined else float("nan"),
            "cv_center_combined_km_std": float(np.std(fold_center_combined)) if fold_center_combined else float("nan"),
            "cv_center_timing_hours_mean": float(np.mean(fold_center_timing)) if fold_center_timing else float("nan"),
            "cv_center_depth_error_km_mean": float(np.mean(fold_center_depth)) if fold_center_depth else float("nan"),
        }

    best_rmse = min(scores[name]["cv_rmse_km_mean"] for name in scores)
    rmse_tolerance = max(best_rmse * 0.02, 5.0e3)
    eligible = [
        name
        for name in scores
        if scores[name]["cv_rmse_km_mean"] <= best_rmse + rmse_tolerance
    ]
    best_name = min(
        eligible or list(scores.keys()),
        key=lambda name: (
            scores[name]["cv_center_combined_km_mean"] if math.isfinite(scores[name]["cv_center_combined_km_mean"]) else float("inf"),
            scores[name]["cv_rmse_km_mean"],
            scores[name]["cv_mae_km_mean"],
        ),
    )
    best_model = factories[best_name]()
    _fit_model_with_weights(best_model, X_local, correction_target, weights)
    clip_scale = float(np.quantile(np.abs(correction_target), 0.98))
    return best_name, best_model, scores, len(splits), clip_scale


def _build_anchor_validation_rows(
    neo: NEOObject,
    meta: dict[str, Any],
    pred_log: Any,
    lo_log: Any,
    hi_log: Any,
    global_pred_log: Any | None = None,
) -> list[dict[str, Any]]:
    import numpy as np

    jd = meta["jd"]
    true_dist = meta["dist_au"]
    pred = 10.0 ** pred_log
    lo = 10.0 ** lo_log
    hi = 10.0 ** hi_log
    global_pred = 10.0 ** np.asarray(global_pred_log, dtype=float) if global_pred_log is not None else None
    true_log = np.log10(np.maximum(true_dist, 1e-300))
    rows: list[dict[str, Any]] = []
    for i, ca in enumerate(neo.close_approaches, start=1):
        idx = int(np.argmin(np.abs(jd - ca.jd_tdb)))
        dt_hours = float((jd[idx] - ca.jd_tdb) * 24.0)
        cad_km = ca.distance_au * AU_KM
        horizons_km = float(true_dist[idx] * AU_KM)
        pred_km = float(pred[idx] * AU_KM)
        global_pred_km = float(global_pred[idx] * AU_KM) if global_pred is not None else float("nan")
        lo_km = float(lo[idx] * AU_KM)
        hi_km = float(hi[idx] * AU_KM)
        interp_true_log, interp_degree, interp_span_hours = _local_poly_predict_log_distance(jd, true_log, ca.jd_tdb)
        interp_pred_log, _, _ = _local_poly_predict_log_distance(jd, pred_log, ca.jd_tdb)
        interp_global_log = _local_poly_predict_log_distance(jd, np.log10(np.maximum(global_pred, 1e-300)), ca.jd_tdb)[0] if global_pred is not None else float("nan")
        interp_lo_log, _, _ = _local_poly_predict_log_distance(jd, lo_log, ca.jd_tdb)
        interp_hi_log, _, _ = _local_poly_predict_log_distance(jd, hi_log, ca.jd_tdb)
        interp_true_km = float((10.0 ** interp_true_log) * AU_KM)
        interp_pred_km = float((10.0 ** interp_pred_log) * AU_KM)
        interp_global_km = float((10.0 ** interp_global_log) * AU_KM) if global_pred is not None else float("nan")
        interp_lo_km = float((10.0 ** min(interp_lo_log, interp_hi_log)) * AU_KM)
        interp_hi_km = float((10.0 ** max(interp_lo_log, interp_hi_log)) * AU_KM)
        tf_recon = _tensorflow_encounter_reconstruction(jd, true_dist, ca.jd_tdb)
        tf_recon_km = float(tf_recon["distance_km"])
        rows.append(
            {
                "anchor_id": i,
                "cad_date_tdb": ca.calendar_date_tdb,
                "cad_jd_tdb": ca.jd_tdb,
                "cad_distance_au": ca.distance_au,
                "cad_distance_km": cad_km,
                "horizons_sample_date_tdb": meta["calendar"][idx],
                "horizons_sample_jd_tdb": float(jd[idx]),
                "sample_time_offset_hours": dt_hours,
                "horizons_sample_distance_au": float(true_dist[idx]),
                "horizons_sample_distance_km": horizons_km,
                "horizons_interpolated_distance_km": interp_true_km,
                "horizons_interpolated_minus_cad_km": interp_true_km - cad_km,
                "ml_global_predicted_distance_km": global_pred_km,
                "ml_global_minus_cad_km": global_pred_km - cad_km,
                "ml_predicted_distance_au": float(pred[idx]),
                "ml_predicted_distance_km": pred_km,
                "ml_minus_cad_km": pred_km - cad_km,
                "ml_global_interpolated_distance_km": interp_global_km,
                "ml_global_interpolated_minus_cad_km": interp_global_km - cad_km,
                "ml_interpolated_distance_km": interp_pred_km,
                "ml_interpolated_minus_cad_km": interp_pred_km - cad_km,
                "tensorflow_continuous_distance_km": tf_recon_km,
                "tensorflow_continuous_minus_cad_km": tf_recon_km - cad_km,
                "tensorflow_continuous_model": tf_recon["model"],
                "tensorflow_continuous_cv_rmse_km": float(tf_recon["cv_rmse_km"]),
                "tensorflow_continuous_points": float(tf_recon["points"]),
                "tensorflow_continuous_span_hours": float(tf_recon["span_hours"]),
                "tensorflow_continuous_ridge": float(tf_recon["ridge"]),
                "horizons_sample_minus_cad_km": horizons_km - cad_km,
                "conformal_lo_km": lo_km,
                "conformal_hi_km": hi_km,
                "conformal_interpolated_lo_km": interp_lo_km,
                "conformal_interpolated_hi_km": interp_hi_km,
                "cad_inside_conformal_band": bool(lo_km <= cad_km <= hi_km),
                "cad_inside_interpolated_conformal_band": bool(interp_lo_km <= cad_km <= interp_hi_km),
                "anchor_interpolation_degree": interp_degree,
                "anchor_interpolation_span_hours": interp_span_hours,
                "v_rel_km_s": ca.v_rel_km_s,
                "v_inf_km_s": ca.v_inf_km_s,
                "t_sigma": ca.t_sigma,
                "orbit_id": ca.orbit_id,
            }
        )
    return rows


def _write_anchor_tables(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import csv

    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    paths: list[str] = []

    csv_path = output_dir / "table_anchor_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    paths.append(str(csv_path))

    md_path = output_dir / "table_anchor_validation.md"
    md_cols = [
        "cad_date_tdb",
        "cad_distance_km",
        "horizons_sample_date_tdb",
        "sample_time_offset_hours",
        "horizons_sample_minus_cad_km",
        "horizons_interpolated_minus_cad_km",
        "ml_minus_cad_km",
        "ml_interpolated_minus_cad_km",
        "tensorflow_continuous_minus_cad_km",
        "cad_inside_conformal_band",
        "cad_inside_interpolated_conformal_band",
        "v_rel_km_s",
    ]
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(md_cols) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(md_cols)) + " |\n")
        for row in rows:
            vals = []
            for col in md_cols:
                value = row[col]
                if isinstance(value, float):
                    vals.append(f"{value:.6g}")
                else:
                    vals.append(str(value))
            fh.write("| " + " | ".join(vals) + " |\n")
    paths.append(str(md_path))

    tex_path = output_dir / "table_anchor_validation.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lrrrrrrrr}\n")
        fh.write("\\hline\n")
        fh.write("CAD epoch & CAD km & $\\Delta t$ h & H samp-CAD km & H interp-CAD km & ML samp-CAD km & ML interp-CAD km & TF cont-CAD km & $v_{rel}$ km/s\\\\\n")
        fh.write("\\hline\n")
        for row in rows:
            fh.write(
                f"{row['cad_date_tdb']} & "
                f"{row['cad_distance_km']:.0f} & "
                f"{row['sample_time_offset_hours']:.2f} & "
                f"{row['horizons_sample_minus_cad_km']:.0f} & "
                f"{row['horizons_interpolated_minus_cad_km']:.0f} & "
                f"{row['ml_minus_cad_km']:.0f} & "
                f"{row['ml_interpolated_minus_cad_km']:.0f} & "
                f"{row['tensorflow_continuous_minus_cad_km']:.0f} & "
                f"{(row['v_rel_km_s'] if row['v_rel_km_s'] is not None else float('nan')):.3f}\\\\\n"
            )
        fh.write("\\hline\n")
        fh.write("\\end{tabular}\n")
    paths.append(str(tex_path))
    return paths


def _write_publication_tables(
    output_dir: Path,
    meta: dict[str, Any],
    pred_log: Any,
    lo_log: Any,
    hi_log: Any,
    global_pred_log: Any,
    locality_score: Any,
    calib_mask: Any,
    val_mask: Any,
) -> list[str]:
    import csv
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    jd = np.asarray(meta["jd"], dtype=float)
    calendar = list(meta["calendar"])
    t_year = (jd - jd[0]) / 365.25636
    global_pred = 10.0 ** np.asarray(global_pred_log, dtype=float)
    pred = 10.0 ** np.asarray(pred_log, dtype=float)
    lo = 10.0 ** np.asarray(lo_log, dtype=float)
    hi = 10.0 ** np.asarray(hi_log, dtype=float)
    locality = np.asarray(locality_score, dtype=float)

    seconds_per_day = float(SECONDS_PER_DAY)
    au_d_to_km_s = AU_KM / seconds_per_day
    au_d2_to_mm_s2 = AU_M * 1.0e3 / (seconds_per_day * seconds_per_day)
    au2_d_to_gkm2_s = (AU_KM * AU_KM) / seconds_per_day / 1.0e9

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(str(path))

    surrogate_rows: list[dict[str, Any]] = []
    residual_km = (pred - np.asarray(meta["dist_au"], dtype=float)) * AU_KM
    for i in range(len(jd)):
        surrogate_rows.append(
            {
                "jd_tdb": float(jd[i]),
                "calendar_tdb": calendar[i],
                "years_from_start": float(t_year[i]),
                "is_training_sample": bool((not bool(calib_mask[i])) and (not bool(val_mask[i]))),
                "is_calibration_sample": bool(calib_mask[i]),
                "is_validation_sample": bool(val_mask[i]),
                "horizons_distance_au": float(meta["dist_au"][i]),
                "global_model_distance_au": float(global_pred[i]),
                "surrogate_distance_au": float(pred[i]),
                "conformal_lo_au": float(lo[i]),
                "conformal_hi_au": float(hi[i]),
                "residual_km": float(residual_km[i]),
                "locality_score": float(locality[i]),
                "gi_log10_abs": float(meta["gi_log"][i]),
                "oi_log10_abs": float(meta["oi_log"][i]),
            }
        )
    _write_csv(output_dir / "table_surrogate_timeseries.csv", surrogate_rows)

    geo_speed_km_s = np.asarray(meta["geo_speed_au_d"], dtype=float) * au_d_to_km_s
    range_rate_km_s = np.asarray(meta["geo_range_rate_au_d"], dtype=float) * au_d_to_km_s
    transverse_km_s = np.asarray(meta["geo_transverse_speed_au_d"], dtype=float) * au_d_to_km_s
    helio_speed_km_s = np.asarray(meta["helio_speed_au_d"], dtype=float) * au_d_to_km_s
    geo_accel_mm_s2 = np.asarray(meta["geo_accel_norm_au_d2"], dtype=float) * au_d2_to_mm_s2
    range_accel_mm_s2 = np.asarray(meta["geo_range_accel_au_d2"], dtype=float) * au_d2_to_mm_s2
    speed_slope_mm_s2 = np.asarray(meta["geo_speed_slope_au_d2"], dtype=float) * au_d2_to_mm_s2
    helio_accel_mm_s2 = np.asarray(meta["helio_accel_norm_au_d2"], dtype=float) * au_d2_to_mm_s2
    geo_h_gkm2_s = np.asarray(meta["geo_ang_mom_au2_d"], dtype=float) * au2_d_to_gkm2_s
    helio_h_gkm2_s = np.linalg.norm(np.asarray(meta["helio_h_vec_au2_d"], dtype=float), axis=1) * au2_d_to_gkm2_s
    helio_energy_mj_kg = np.asarray(meta["helio_specific_energy_m2_s2"], dtype=float) / 1.0e6
    geo_xy_mkm = np.asarray(meta["geo_pos_au"], dtype=float)[:, :2] * AU_KM / 1.0e6

    orbital_rows: list[dict[str, Any]] = []
    for i in range(len(jd)):
        orbital_rows.append(
            {
                "jd_tdb": float(jd[i]),
                "calendar_tdb": calendar[i],
                "years_from_start": float(t_year[i]),
                "geocentric_distance_au": float(meta["dist_au"][i]),
                "heliocentric_distance_au": float(meta["r_helio_au"][i]),
                "geocentric_speed_km_s": float(geo_speed_km_s[i]),
                "range_rate_km_s": float(range_rate_km_s[i]),
                "transverse_speed_km_s": float(transverse_km_s[i]),
                "heliocentric_speed_km_s": float(helio_speed_km_s[i]),
                "geocentric_acceleration_mm_s2": float(geo_accel_mm_s2[i]),
                "range_acceleration_mm_s2": float(range_accel_mm_s2[i]),
                "speed_slope_mm_s2": float(speed_slope_mm_s2[i]),
                "heliocentric_acceleration_mm_s2": float(helio_accel_mm_s2[i]),
                "geo_specific_angular_momentum_gkm2_s": float(geo_h_gkm2_s[i]),
                "helio_specific_angular_momentum_gkm2_s": float(helio_h_gkm2_s[i]),
                "heliocentric_eccentricity_inst": float(meta["helio_eccentricity_inst"][i]),
                "helio_specific_energy_mj_kg": float(helio_energy_mj_kg[i]),
                "geocentric_x_million_km": float(geo_xy_mkm[i, 0]),
                "geocentric_y_million_km": float(geo_xy_mkm[i, 1]),
                "global_model_distance_au": float(global_pred[i]),
                "surrogate_distance_au": float(pred[i]),
                "conformal_lo_au": float(lo[i]),
                "conformal_hi_au": float(hi[i]),
                "locality_score": float(locality[i]),
            }
        )
    _write_csv(output_dir / "table_publication_orbital_physics.csv", orbital_rows)

    pdf = meta.get("pdf_arrays", {})

    def _pdf(name: str) -> Any:
        arr = pdf.get(name)
        if arr is None:
            return np.full(len(jd), np.nan, dtype=float)
        return np.asarray(arr, dtype=float)

    hypothesis_rows: list[dict[str, Any]] = []
    for i in range(len(jd)):
        hypothesis_rows.append(
            {
                "jd_tdb": float(jd[i]),
                "calendar_tdb": calendar[i],
                "years_from_start": float(t_year[i]),
                "gi_log10_abs": float(meta["gi_log"][i]),
                "oi_log10_abs": float(meta["oi_log"][i]),
                "jsuncritical_low": float(_pdf("jsuncritical_low")[i]),
                "jsuncritical_median": float(_pdf("jsuncritical_median")[i]),
                "jsuncritical_high": float(_pdf("jsuncritical_high")[i]),
                "time_norm_median": float(_pdf("time_norm_median")[i]),
                "time_cause": float(_pdf("time_cause")[i]),
                "time_slip_median": float(_pdf("time_slip_median")[i]),
                "lapse_factor_median": float(_pdf("lapse_factor_median")[i]),
                "trajectory_slip_median": float(_pdf("trajectory_slip_median")[i]),
                "trajectory_precession_median": float(_pdf("trajectory_precession_median")[i]),
                "neo_phasing": float(_pdf("neo_phasing")[i]),
                "gravity_neo_phasing": float(_pdf("gravity_neo_phasing")[i]),
                "bound_factor_median": float(_pdf("bound_factor_median")[i]),
                "sequence_1": float(_pdf("sequence_1")[i]),
                "sequence_2": float(_pdf("sequence_2")[i]),
                "sequence_3": float(_pdf("sequence_3")[i]),
                "sequence_4": float(_pdf("sequence_4")[i]),
                "sequence_5": float(_pdf("sequence_5")[i]),
                "trajectory_loss": float(_pdf("trajectory_loss")[i]),
                "seqcr": float(_pdf("seqcr")[i]),
                "likelihood_proxy": float(_pdf("likelihood_proxy")[i]),
                "new_eccentricity_proxy": float(_pdf("new_eccentricity_proxy")[i]),
                "new_sma_au_proxy": float(_pdf("new_sma_au_proxy")[i]),
                "scaled_jsuncritical": float(_pdf("scaled_jsuncritical")[i]),
                "scaled_time_norm": float(_pdf("scaled_time_norm")[i]),
                "scaled_neo_phasing": float(_pdf("scaled_neo_phasing")[i]),
                "locality_score": float(locality[i]),
            }
        )
    _write_csv(output_dir / "table_publication_hypothesis_parameters.csv", hypothesis_rows)
    return paths


def _write_tensorflow_gate_tables(
    output_dir: Path,
    numerical_diagnostics: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    global_residual_full_log: Any,
) -> list[str]:
    import csv
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(str(path))

    gate_rows: list[dict[str, Any]] = []
    for source_key, gate_name, accepted_key, accepted_blend_key, ref_key, best_key in [
        (
            "global_residual_blend_selection",
            "global_residual_to_state_vector_baseline",
            "global_residual_gate_accepted",
            "global_residual_blend_strength",
            "global_residual_gate_reference_rmse_km",
            "global_residual_gate_best_rmse_km",
        ),
        (
            "local_blend_selection",
            "local_encounter_refinement_to_global_baseline",
            "local_gate_accepted",
            "local_blend_strength",
            "local_gate_reference_rmse_km",
            "local_gate_best_rmse_km",
        ),
    ]:
        rows = numerical_diagnostics.get(source_key, [])
        for row in rows if isinstance(rows, list) else []:
            gate_rows.append(
                {
                    "gate": gate_name,
                    "candidate_blend_strength": float(row.get("blend_strength", float("nan"))),
                    "candidate_rmse_km": float(row.get("rmse_km", float("nan"))),
                    "candidate_mae_km": float(row.get("mae_km", float("nan"))),
                    "center_combined_km_mean": float(row.get("center_combined_km_mean", float("nan"))),
                    "center_timing_hours_mean": float(row.get("center_timing_hours_mean", float("nan"))),
                    "center_depth_error_km_mean": float(row.get("center_depth_error_km_mean", float("nan"))),
                    "accepted_blend_strength": float(numerical_diagnostics.get(accepted_blend_key, 0.0)),
                    "gate_accepted": bool(numerical_diagnostics.get(accepted_key, False)),
                    "reference_rmse_km": float(numerical_diagnostics.get(ref_key, float("nan"))),
                    "best_candidate_rmse_km": float(numerical_diagnostics.get(best_key, float("nan"))),
                }
            )
    _write_csv(output_dir / "table_tensorflow_gate_diagnostics.csv", gate_rows)

    proposal_log = np.asarray(global_residual_full_log, dtype=float)
    jd = np.asarray(meta["jd"], dtype=float)
    anchor_synthesis_rows: list[dict[str, Any]] = []
    for row in anchor_rows:
        sample_jd = float(row.get("horizons_sample_jd_tdb", row.get("cad_jd_tdb", float("nan"))))
        idx = int(np.argmin(np.abs(jd - sample_jd)))
        proposal_km = float((10.0 ** proposal_log[idx]) * AU_KM)
        cad_km = float(row["cad_distance_km"])
        anchor_synthesis_rows.append(
            {
                "cad_date_tdb": row["cad_date_tdb"],
                "cad_distance_km": cad_km,
                "horizons_sample_date_tdb": row["horizons_sample_date_tdb"],
                "sample_time_offset_hours": float(row["sample_time_offset_hours"]),
                "horizons_sample_minus_cad_km": float(row["horizons_sample_minus_cad_km"]),
                "tensorflow_full_global_proposal_minus_cad_km": proposal_km - cad_km,
                "accepted_surrogate_minus_cad_km": float(row["ml_minus_cad_km"]),
                "tensorflow_continuous_reconstruction_minus_cad_km": float(row.get("tensorflow_continuous_minus_cad_km", float("nan"))),
                "tensorflow_continuous_model": row.get("tensorflow_continuous_model", ""),
                "tensorflow_continuous_cv_rmse_km": float(row.get("tensorflow_continuous_cv_rmse_km", float("nan"))),
                "global_gate_accepted": bool(numerical_diagnostics.get("global_residual_gate_accepted", False)),
                "local_gate_accepted": bool(numerical_diagnostics.get("local_gate_accepted", False)),
            }
        )
    _write_csv(output_dir / "table_tensorflow_anchor_synthesis.csv", anchor_synthesis_rows)
    return paths


def _sbdb_covariance_nominal_map(neo: NEOObject) -> dict[str, float]:
    cov = neo.covariance or {}
    out: dict[str, float] = {}
    for item in cov.get("elements", []) if isinstance(cov.get("elements"), list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("label") or item.get("title") or "")
        value = _to_float(item.get("value"))
        if name and value is not None:
            out[name] = float(value)

    e = neo.elements.eccentricity
    q = neo.elements.perihelion_au
    a = neo.elements.semi_major_axis_au
    mu_sun_au3_d2 = GM_SUN_M3_S2 * (SECONDS_PER_DAY * SECONDS_PER_DAY) / (AU_M**3)
    n_rad_d = math.sqrt(mu_sun_au3_d2 / max(a**3, 1e-300))
    epoch = neo.elements.epoch_jd
    ma = neo.elements.mean_anomaly_deg
    if "tp" not in out and epoch is not None and ma is not None and n_rad_d > 0.0:
        out["tp"] = float(epoch) - math.radians(float(ma)) / n_rad_d

    out.setdefault("e", e)
    out.setdefault("q", q)
    if neo.elements.node_deg is not None:
        out.setdefault("node", float(neo.elements.node_deg))
        out.setdefault("om", float(neo.elements.node_deg))
    if neo.elements.peri_arg_deg is not None:
        out.setdefault("peri", float(neo.elements.peri_arg_deg))
        out.setdefault("w", float(neo.elements.peri_arg_deg))
    out.setdefault("i", float(neo.elements.inclination_deg))
    if neo.physical.non_grav_a1_au_d2 is not None:
        out.setdefault("A1", float(neo.physical.non_grav_a1_au_d2))
    if neo.physical.non_grav_a2_au_d2 is not None:
        out.setdefault("A2", float(neo.physical.non_grav_a2_au_d2))
    return out


def _solve_kepler_vectorized(mean_anomaly_rad: Any, eccentricity: Any) -> Any:
    import numpy as np

    m = (np.asarray(mean_anomaly_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    e = np.asarray(eccentricity, dtype=float)
    ecc = np.clip(e, 0.0, 0.999999)
    guess = np.where(ecc < 0.8, m, np.pi * np.sign(np.where(m == 0.0, 1.0, m)))
    E = guess.astype(float, copy=True)
    for _ in range(16):
        f = E - ecc * np.sin(E) - m
        fp = 1.0 - ecc * np.cos(E)
        step = f / np.maximum(np.abs(fp), 1e-14) * np.sign(fp)
        E -= np.clip(step, -0.75, 0.75)
    return E


def _propagate_covariance_elements_to_position_au(samples: Any, labels: list[str], jd: float, nominal: dict[str, float]) -> Any:
    import numpy as np

    label_index = {label: i for i, label in enumerate(labels)}

    def col(name: str, fallback: float) -> Any:
        idx = label_index.get(name)
        if idx is None:
            return np.full(samples.shape[0], float(fallback), dtype=float)
        return np.asarray(samples[:, idx], dtype=float)

    e = np.clip(col("e", nominal["e"]), 1.0e-8, 0.985)
    q = np.maximum(col("q", nominal["q"]), 1.0e-8)
    tp = col("tp", nominal["tp"])
    node = np.deg2rad(col("node", nominal.get("node", nominal.get("om", 0.0))))
    peri = np.deg2rad(col("peri", nominal.get("peri", nominal.get("w", 0.0))))
    inc = np.deg2rad(col("i", nominal["i"]))
    a = q / np.maximum(1.0 - e, 1.0e-8)
    mu_sun_au3_d2 = GM_SUN_M3_S2 * (SECONDS_PER_DAY * SECONDS_PER_DAY) / (AU_M**3)
    n_rad_d = np.sqrt(mu_sun_au3_d2 / np.maximum(a**3, 1e-300))
    mean_anomaly = n_rad_d * (float(jd) - tp)
    E = _solve_kepler_vectorized(mean_anomaly, e)

    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(np.maximum(1.0 - e * e, 0.0)) * np.sin(E)

    cos_o = np.cos(node)
    sin_o = np.sin(node)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    cos_w = np.cos(peri)
    sin_w = np.sin(peri)

    x1 = cos_w * x_orb - sin_w * y_orb
    y1 = sin_w * x_orb + cos_w * y_orb
    x = cos_o * x1 - sin_o * cos_i * y1
    y = sin_o * x1 + cos_o * cos_i * y1
    z = sin_i * y1
    return np.column_stack([x, y, z])


def _covariance_clone_matrix(neo: NEOObject, sample_count: int, date_min: str, date_max: str) -> tuple[Any, list[str], dict[str, Any]]:
    import numpy as np

    cov = neo.covariance or {}
    labels = [str(label) for label in cov.get("labels", [])] if isinstance(cov.get("labels"), list) else []
    raw_matrix = cov.get("data")
    if not labels or raw_matrix is None:
        raise SourceError("SBDB covariance matrix is unavailable for this object")
    matrix = np.asarray(raw_matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(labels):
        raise SourceError("SBDB covariance matrix shape does not match covariance labels")

    nominal_map = _sbdb_covariance_nominal_map(neo)
    required = ["e", "q", "tp", "i"]
    if not all(name in nominal_map for name in required):
        raise SourceError("SBDB covariance nominal elements are insufficient for clone propagation")
    if "node" not in nominal_map and "om" not in nominal_map:
        raise SourceError("SBDB covariance nominal node is missing")
    if "peri" not in nominal_map and "w" not in nominal_map:
        raise SourceError("SBDB covariance nominal perihelion argument is missing")

    nominal = np.asarray([float(nominal_map.get(label, 0.0)) for label in labels], dtype=float)
    sym = 0.5 * (matrix + matrix.T)
    eigval, eigvec = np.linalg.eigh(sym)
    min_eig = float(np.min(eigval))
    clipped = np.clip(eigval, 0.0, None)
    repaired = (eigvec * clipped) @ eigvec.T

    seed_material = f"{neo.designation}|{neo.elements.orbit_id}|{date_min}|{date_max}|{','.join(labels)}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)

    draws_needed = max(int(sample_count), 64)
    accepted: list[Any] = []
    attempts = 0
    while sum(len(chunk) for chunk in accepted) < draws_needed and attempts < 8:
        attempts += 1
        trial = rng.multivariate_normal(nominal, repaired, size=draws_needed)
        trial_map = {label: trial[:, i] for i, label in enumerate(labels)}
        e = trial_map.get("e", np.full(len(trial), nominal_map["e"]))
        q = trial_map.get("q", np.full(len(trial), nominal_map["q"]))
        finite = np.all(np.isfinite(trial), axis=1)
        physical = finite & (e > 0.0) & (e < 0.985) & (q > 0.0)
        if np.any(physical):
            accepted.append(trial[physical])

    if not accepted:
        raise SourceError("SBDB covariance sampling produced no physically admissible clones")
    clones = np.vstack(accepted)[:draws_needed]
    all_samples = np.vstack([nominal[None, :], clones])
    diagnostics = {
        "covariance_labels": labels,
        "covariance_epoch_jd": _to_float(cov.get("epoch")),
        "covariance_min_eigenvalue": min_eig,
        "covariance_condition_number": float(np.max(clipped) / max(np.min(clipped[clipped > 0.0]) if np.any(clipped > 0.0) else 1.0, 1e-300)),
        "covariance_clone_seed": float(seed),
        "covariance_requested_clones": float(sample_count),
        "covariance_accepted_clones": float(len(all_samples) - 1),
        "covariance_eigenvalue_clipped": bool(min_eig < 0.0),
    }
    return all_samples, labels, diagnostics


def _hypothesis_cascade_clone_score(samples: Any, labels: list[str], nominal: dict[str, float], positions_au: Any, gamma: float) -> Any:
    import numpy as np

    label_index = {label: i for i, label in enumerate(labels)}

    def col(name: str, fallback: float) -> Any:
        idx = label_index.get(name)
        if idx is None:
            return np.full(samples.shape[0], float(fallback), dtype=float)
        return np.asarray(samples[:, idx], dtype=float)

    e = np.clip(col("e", nominal["e"]), 1.0e-8, 0.985)
    q = np.maximum(col("q", nominal["q"]), 1.0e-8)
    a_au = q / np.maximum(1.0 - e, 1.0e-8)
    r_m = np.linalg.norm(np.asarray(positions_au, dtype=float), axis=1) * AU_M
    a_m = a_au * AU_M
    speed_sq = GM_SUN_M3_S2 * np.maximum(2.0 / np.maximum(r_m, 1e-300) - 1.0 / np.maximum(a_m, 1e-300), 0.0)
    speed_m_s = np.sqrt(speed_sq)
    j_m2_s = np.sqrt(np.maximum(GM_SUN_M3_S2 * a_m * (1.0 - e * e), 0.0))
    energy = 0.5 * speed_sq - GM_SUN_M3_S2 / np.maximum(r_m, 1e-300)
    upsilon = speed_m_s / C_M_S
    gi_n = gamma * j_m2_s / (1.0 + upsilon * upsilon)
    oi_n = (energy**4) * (j_m2_s**2) * (gamma**2) - 1.0
    cascade_first = 4.0 * (j_m2_s * j_m2_s) * (gamma * gamma) * (energy**3)
    a1 = np.abs(col("A1", nominal.get("A1", 0.0)))
    a2 = np.abs(col("A2", nominal.get("A2", 0.0)))
    nongrav = np.sqrt(a1 * a1 + a2 * a2)
    return (
        np.log10(1.0 + np.abs(gi_n))
        + np.log10(1.0 + np.abs(oi_n))
        + np.log10(1.0 + np.abs(cascade_first))
        + np.log10(1.0 + nongrav)
    )


def _write_uncertainty_propagation(
    output_dir: Path,
    neo: NEOObject,
    hypothesis: HypothesisTerms,
    geo: HorizonsVectors,
    helio: HorizonsVectors,
    anchor_rows: list[dict[str, Any]],
    sample_count: int,
    numerical_diagnostics: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    import csv
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths: list[str] = []
    figure_paths: list[str] = []
    publication_assets: list[str] = []
    if not anchor_rows or int(sample_count) <= 0:
        numerical_diagnostics["covariance_uncertainty_status"] = "skipped-no-anchors-or-zero-samples"
        return table_paths, figure_paths, publication_assets

    try:
        samples, labels, cov_diag = _covariance_clone_matrix(neo, int(sample_count), str(anchor_rows[0]["cad_jd_tdb"]), str(anchor_rows[-1]["cad_jd_tdb"]))
    except SourceError as exc:
        numerical_diagnostics["covariance_uncertainty_status"] = f"skipped: {exc}"
        return table_paths, figure_paths, publication_assets

    numerical_diagnostics.update(cov_diag)
    nominal_map = _sbdb_covariance_nominal_map(neo)
    geo_jd = np.asarray(geo.jd_tdb, dtype=float)
    geo_state = np.asarray(geo.state_au_d, dtype=float)
    helio_state = np.asarray(helio.state_au_d, dtype=float)
    n = min(len(geo_jd), len(geo_state), len(helio_state))
    geo_jd = geo_jd[:n]
    earth_helio_au = helio_state[:n, :3] - geo_state[:n, :3]

    def earth_at(jd_value: float) -> Any:
        return np.asarray([np.interp(jd_value, geo_jd, earth_helio_au[:, dim]) for dim in range(3)], dtype=float)

    rows: list[dict[str, Any]] = []
    for row in anchor_rows:
        cad_jd = float(row["cad_jd_tdb"])
        cad_km = float(row["cad_distance_km"])
        positions = _propagate_covariance_elements_to_position_au(samples, labels, cad_jd, nominal_map)
        earth = earth_at(cad_jd)
        distance_km = np.linalg.norm(positions - earth[None, :], axis=1) * AU_KM
        nominal_distance = float(distance_km[0])
        clone_delta = distance_km[1:] - nominal_distance

        score = _hypothesis_cascade_clone_score(samples, labels, nominal_map, positions, hypothesis.inputs.gamma_ratio)[1:]
        score_med = float(np.median(score))
        score_mad = float(np.median(np.abs(score - score_med)))
        robust_scale = max(1.4826 * score_mad, float(np.std(score)), 1e-12)
        score_z = (score - score_med) / robust_scale
        z_var = float(np.var(score_z))
        beta = float(np.cov(clone_delta, score_z, bias=True)[0, 1] / max(z_var, 1e-12))
        hypothesis_delta = clone_delta + beta * score_z

        cov_centered = cad_km + clone_delta
        hyp_centered = cad_km + hypothesis_delta
        cov_q = np.quantile(cov_centered, [0.05, 0.16, 0.5, 0.84, 0.95])
        hyp_q = np.quantile(hyp_centered, [0.05, 0.16, 0.5, 0.84, 0.95])
        cov_width90 = float(cov_q[4] - cov_q[0])
        hyp_width90 = float(hyp_q[4] - hyp_q[0])
        rows.append(
            {
                "anchor_id": row["anchor_id"],
                "cad_date_tdb": row["cad_date_tdb"],
                "cad_jd_tdb": cad_jd,
                "cad_distance_km": cad_km,
                "nominal_two_body_distance_km": nominal_distance,
                "two_body_nominal_minus_cad_km": nominal_distance - cad_km,
                "covariance_p05_km": float(cov_q[0]),
                "covariance_p16_km": float(cov_q[1]),
                "covariance_p50_km": float(cov_q[2]),
                "covariance_p84_km": float(cov_q[3]),
                "covariance_p95_km": float(cov_q[4]),
                "covariance_width90_km": cov_width90,
                "gi_oi_cascade_p05_km": float(hyp_q[0]),
                "gi_oi_cascade_p16_km": float(hyp_q[1]),
                "gi_oi_cascade_p50_km": float(hyp_q[2]),
                "gi_oi_cascade_p84_km": float(hyp_q[3]),
                "gi_oi_cascade_p95_km": float(hyp_q[4]),
                "gi_oi_cascade_width90_km": hyp_width90,
                "cascade_uncertainty_inflation_ratio": hyp_width90 / max(cov_width90, 1e-12),
                "cascade_score_median": score_med,
                "cascade_score_mad": score_mad,
                "adaptive_cascade_beta_km_per_sigma": beta,
                "clone_count": int(len(clone_delta)),
            }
        )

    csv_path = output_dir / "table_covariance_gi_oi_cascade_uncertainty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    table_paths.append(str(csv_path))

    widths = np.asarray([r["covariance_width90_km"] for r in rows], dtype=float)
    hyp_widths = np.asarray([r["gi_oi_cascade_width90_km"] for r in rows], dtype=float)
    betas = np.asarray([r["adaptive_cascade_beta_km_per_sigma"] for r in rows], dtype=float)
    infl = np.asarray([r["cascade_uncertainty_inflation_ratio"] for r in rows], dtype=float)
    numerical_diagnostics["covariance_uncertainty_status"] = "ok"
    numerical_diagnostics["covariance_width90_km_median"] = float(np.median(widths))
    numerical_diagnostics["gi_oi_cascade_width90_km_median"] = float(np.median(hyp_widths))
    numerical_diagnostics["gi_oi_cascade_inflation_ratio_median"] = float(np.median(infl))
    nearest = min(rows, key=lambda r: float(r["cad_distance_km"]))
    numerical_diagnostics["nearest_cad_covariance_width90_km"] = float(nearest["covariance_width90_km"])
    numerical_diagnostics["nearest_cad_gi_oi_cascade_width90_km"] = float(nearest["gi_oi_cascade_width90_km"])
    numerical_diagnostics["nearest_cad_adaptive_cascade_beta_km_per_sigma"] = float(nearest["adaptive_cascade_beta_km_per_sigma"])

    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    labels_short = [str(r["cad_date_tdb"]).split()[0] for r in rows]

    def _save(fig: Any, stem: str, dpi: int = 230) -> None:
        for suffix in ["png", "svg", "pdf"]:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
            figure_paths.append(str(path))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    ax.fill_between(x, [r["covariance_p05_km"] for r in rows], [r["covariance_p95_km"] for r in rows], color="#2a9d8f", alpha=0.20, label="SBDB covariance 5-95%")
    ax.plot(x, [r["covariance_p50_km"] for r in rows], color="#1b6f68", lw=1.8, label="SBDB covariance median")
    ax.fill_between(x, [r["gi_oi_cascade_p05_km"] for r in rows], [r["gi_oi_cascade_p95_km"] for r in rows], color="#d1495b", alpha=0.18, label="GI_N/OI_N cascade-adjusted 5-95%")
    ax.plot(x, [r["gi_oi_cascade_p50_km"] for r in rows], color="#9d1f33", lw=1.5, label="GI_N/OI_N cascade-adjusted median")
    ax.scatter(x, [r["cad_distance_km"] for r in rows], s=34, color="#1d3557", edgecolor="white", linewidth=0.6, label="CAD anchor")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Geocentric distance at CAD epoch (km)")
    ax.set_title("Covariance Propagation and GI_N/OI_N Cascade Uncertainty at CAD Anchors")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save(fig, "fig_covariance_gi_oi_cascade_anchor_bands")

    fig, ax1 = plt.subplots(figsize=(12.8, 6.2))
    ax1.plot(x, widths, marker="o", color="#2a9d8f", lw=1.8, label="SBDB covariance 90% width")
    ax1.plot(x, hyp_widths, marker="s", color="#d1495b", lw=1.6, label="GI_N/OI_N cascade-adjusted 90% width")
    ax1.set_yscale("log")
    ax1.set_ylabel("90% interval width (km)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=8)
    ax1.grid(True, which="both", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, infl, color="#6a4c93", lw=1.2, alpha=0.85, label="cascade inflation ratio")
    ax2.plot(x, np.abs(betas), color="#f4a261", lw=1.2, alpha=0.85, label="|adaptive beta|")
    ax2.set_ylabel("Cascade coupling diagnostics")
    lines, names = ax1.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, names + names2, loc="best", fontsize=8)
    ax1.set_title("Adaptive Hypothesis Coupling in the Dynamical Uncertainty Field")
    _save(fig, "fig_covariance_gi_oi_cascade_diagnostics")

    caption_path = output_dir / "covariance_uncertainty_captions.md"
    caption_path.write_text(
        "\n\n".join(
            [
                "# Covariance and GI_N/OI_N Cascade Uncertainty Figures",
                "Figure: Covariance Propagation and GI_N/OI_N Cascade Uncertainty at CAD Anchors. SBDB covariance clones are propagated in the native element basis to each CAD epoch, centered on the authoritative CAD range, and compared with an adaptive GI_N/OI_N/cascade adjustment estimated from clone-level score-distance covariance.",
                "Figure: Adaptive Hypothesis Coupling in the Dynamical Uncertainty Field. The plot reports the ordinary covariance interval width, the GI_N/OI_N cascade-adjusted interval width, and the inferred robust coupling between the hypothesis score and clone distance perturbations.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    publication_assets.append(str(caption_path))
    return table_paths, figure_paths, publication_assets


def _parse_vector_weights(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise SourceError("cascade vector weights must be three comma-separated values: velocity,radial,normal")
    try:
        weights = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise SourceError("cascade vector weights must be numeric") from exc
    norm = math.sqrt(sum(value * value for value in weights))
    if norm <= 0.0:
        raise SourceError("cascade vector weights cannot all be zero")
    return weights  # type: ignore[return-value]


def _interp_vector(jd_grid: Any, values: Any, jd_value: float) -> Any:
    import numpy as np

    jd = np.asarray(jd_grid, dtype=float)
    arr = np.asarray(values, dtype=float)
    return np.asarray([np.interp(float(jd_value), jd, arr[:, dim]) for dim in range(arr.shape[1])], dtype=float)


def _interp_state_on_grid(source: HorizonsVectors, jd_grid: Any) -> Any:
    import numpy as np

    jd = np.asarray(jd_grid, dtype=float)
    source_jd = np.asarray(source.jd_tdb, dtype=float)
    source_state = np.asarray(source.state_au_d, dtype=float)
    return np.column_stack([np.interp(jd, source_jd, source_state[:, dim]) for dim in range(6)])


def _cascade_direction(r_au: Any, v_au_d: Any, weights: tuple[float, float, float]) -> Any:
    import numpy as np

    r = np.asarray(r_au, dtype=float)
    v = np.asarray(v_au_d, dtype=float)
    rhat = r / max(float(np.linalg.norm(r)), 1e-300)
    vhat = v / max(float(np.linalg.norm(v)), 1e-300)
    nhat = np.cross(r, v)
    nhat = nhat / max(float(np.linalg.norm(nhat)), 1e-300)
    direction = weights[0] * vhat + weights[1] * rhat + weights[2] * nhat
    return direction / max(float(np.linalg.norm(direction)), 1e-300)


def _cascade_acceleration_au_d2(
    r_au: Any,
    v_au_d: Any,
    neo: NEOObject,
    hypothesis: HypothesisTerms,
    weights: tuple[float, float, float],
    cascade_accel_au_d2: float | None,
) -> Any:
    import numpy as np

    r = np.asarray(r_au, dtype=float)
    v = np.asarray(v_au_d, dtype=float)
    direction = _cascade_direction(r, v, weights)
    sourced_scale = math.hypot(
        neo.physical.non_grav_a1_au_d2 or 0.0,
        neo.physical.non_grav_a2_au_d2 or 0.0,
    )
    scale = float(cascade_accel_au_d2) if cascade_accel_au_d2 is not None else sourced_scale
    if not math.isfinite(scale) or scale <= 0.0:
        return np.zeros(3, dtype=float)

    v_m_s = float(np.linalg.norm(v)) * AU_M / SECONDS_PER_DAY
    r_m = max(float(np.linalg.norm(r)) * AU_M, 1e-300)
    j_m2_s = float(np.linalg.norm(np.cross(r, v))) * (AU_M * AU_M) / SECONDS_PER_DAY
    gamma = hypothesis.inputs.gamma_ratio
    upsilon = v_m_s / C_M_S
    energy = 0.5 * v_m_s * v_m_s - GM_SUN_M3_S2 / r_m
    gi_n = gamma * j_m2_s / (1.0 + upsilon * upsilon)
    oi_n = (energy**4) * (j_m2_s**2) * (gamma**2) - 1.0
    cascade_first = 4.0 * (j_m2_s * j_m2_s) * (gamma * gamma) * (energy**3)
    score = (
        math.log1p(abs(gi_n))
        + math.log1p(abs(oi_n))
        + math.log1p(abs(cascade_first))
    )
    reference = max(
        math.log1p(abs(hypothesis.gi_n_raw))
        + math.log1p(abs(hypothesis.oi_n_raw))
        + math.log1p(abs(hypothesis.oi_cascade[0] if hypothesis.oi_cascade else 0.0)),
        1e-300,
    )
    modulation = math.tanh(score / reference)
    return direction * scale * modulation


def _phase_warp_displacement_au(
    acceleration_au_d2: Any,
    dt_days: float,
    hypothesis: HypothesisTerms,
    phase_warp_gain: float,
) -> Any:
    import numpy as np

    if phase_warp_gain <= 0.0:
        return np.zeros(3, dtype=float)
    phasing = abs(hypothesis.scaled_terms.get("neo_phasing", 0.0)) + abs(hypothesis.scaled_terms.get("gravity_neo_phasing", 0.0))
    gate = math.tanh(math.log1p(phasing))
    return 0.5 * np.asarray(acceleration_au_d2, dtype=float) * (float(dt_days) ** 2) * gate * float(phase_warp_gain)


def _phase_modulated_acceleration_au_d2(acceleration_au_d2: Any, hypothesis: HypothesisTerms, phase_warp_gain: float) -> Any:
    import numpy as np

    if phase_warp_gain <= 0.0:
        return np.zeros(3, dtype=float)
    phasing = abs(hypothesis.scaled_terms.get("neo_phasing", 0.0)) + abs(hypothesis.scaled_terms.get("gravity_neo_phasing", 0.0))
    gate = math.tanh(math.log1p(phasing))
    return np.asarray(acceleration_au_d2, dtype=float) * gate * float(phase_warp_gain)


def _solar_relativistic_correction_au_d2(r_rel_au: Any, v_rel_au_d: Any, mu_au3_d2: float) -> Any:
    import numpy as np

    r = np.asarray(r_rel_au, dtype=float)
    v = np.asarray(v_rel_au_d, dtype=float)
    r_norm = max(float(np.linalg.norm(r)), 1e-300)
    v2 = float(np.dot(v, v))
    rv = float(np.dot(r, v))
    c_au_d = C_M_S * SECONDS_PER_DAY / AU_M
    factor = float(mu_au3_d2) / ((c_au_d * c_au_d) * (r_norm**3))
    return factor * ((4.0 * float(mu_au3_d2) / r_norm - v2) * r + 4.0 * rv * v)


def _standard_nongrav_acceleration_au_d2(r_rel_au: Any, v_rel_au_d: Any, neo: NEOObject) -> Any:
    import numpy as np

    a1 = neo.physical.non_grav_a1_au_d2
    a2 = neo.physical.non_grav_a2_au_d2
    if a1 is None and a2 is None:
        return np.zeros(3, dtype=float)
    r = np.asarray(r_rel_au, dtype=float)
    v = np.asarray(v_rel_au_d, dtype=float)
    r_norm = max(float(np.linalg.norm(r)), 1e-300)
    rhat = r / r_norm
    transverse = v - float(np.dot(v, rhat)) * rhat
    t_norm = max(float(np.linalg.norm(transverse)), 1e-300)
    that = transverse / t_norm
    scale = 1.0 / (r_norm * r_norm)
    return scale * ((float(a1 or 0.0) * rhat) + (float(a2 or 0.0) * that))


def _parse_nbody_bodies(text: str) -> list[str]:
    if str(text).strip().lower() in {"none", "off", "0"}:
        return []
    bodies = [part.strip().lower() for part in str(text).replace(";", ",").split(",") if part.strip()]
    unknown = [body for body in bodies if body not in PLANETARY_PERTURBERS]
    if unknown:
        allowed = ", ".join(sorted(PLANETARY_PERTURBERS))
        raise SourceError(f"unknown N-body perturber(s): {', '.join(unknown)}; allowed: {allowed}")
    deduped: list[str] = []
    for body in bodies:
        if body not in deduped:
            deduped.append(body)
    return deduped


def _parse_dynamics_frame(text: str) -> str:
    frame = str(text).strip().lower()
    if frame not in {"barycentric", "heliocentric"}:
        raise SourceError("dynamics frame must be 'barycentric' or 'heliocentric'")
    return frame


def _parse_integrator_method(text: str) -> str:
    method = str(text).strip().upper()
    if method in {"RK4", "DOP853"}:
        return method
    raise SourceError("integrator method must be 'DOP853' or 'RK4'")


def _fetch_planetary_perturbers(
    bodies: list[str],
    date_min: str,
    date_max: str,
    step: str,
    earth_helio_au: Any,
    jd_grid: Any,
    frame: str,
    refine_step: str | None = None,
    refine_window_days: float = 0.0,
    refine_centers: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    jd = np.asarray(jd_grid, dtype=float)
    out: dict[str, dict[str, Any]] = {}

    def fetch_body(command: str, center: str) -> HorizonsVectors:
        base = fetch_horizons_vectors(command, center, date_min, date_max, step)
        if not refine_step or not refine_centers or refine_window_days <= 0.0:
            return base
        additions = [
            fetch_horizons_vectors(command, center, _jd_to_iso_date(center_jd - refine_window_days), _jd_to_iso_date(center_jd + refine_window_days), refine_step)
            for center_jd in refine_centers
        ]
        return _merge_horizons_vectors(base, additions)

    for body in bodies:
        spec = PLANETARY_PERTURBERS[body]
        if frame == "heliocentric" and body == "earth":
            positions = np.asarray(earth_helio_au, dtype=float)
            states = np.column_stack([positions, np.gradient(positions, jd, axis=0)])
            source = "derived from paired Horizons asteroid vectors: helio-target minus geo-target"
        else:
            center = "500@0" if frame == "barycentric" else "500@10"
            vectors = fetch_body(str(spec["command"]), center)
            states = _interp_state_on_grid(vectors, jd)
            positions = states[:, :3]
            source = vectors.source.url
        out[body] = {
            "mu_au3_d2": float(spec["gm_m3_s2"]) * (SECONDS_PER_DAY * SECONDS_PER_DAY) / (AU_M**3),
            "positions_au": positions,
            "velocities_au_d": states[:, 3:6],
            "source": source,
            "command": str(spec["command"]),
        }
    return out


def _fetch_earth_position_for_frame(
    frame: str,
    date_min: str,
    date_max: str,
    step: str,
    earth_helio_au: Any,
    jd_grid: Any,
    refine_step: str | None = None,
    refine_window_days: float = 0.0,
    refine_centers: list[float] | None = None,
) -> tuple[Any, str]:
    import numpy as np

    if frame == "heliocentric":
        return np.asarray(earth_helio_au, dtype=float), "derived from paired Horizons asteroid vectors"
    vectors = fetch_horizons_vectors("399", "500@0", date_min, date_max, step)
    if refine_step and refine_centers and refine_window_days > 0.0:
        additions = [
            fetch_horizons_vectors("399", "500@0", _jd_to_iso_date(center_jd - refine_window_days), _jd_to_iso_date(center_jd + refine_window_days), refine_step)
            for center_jd in refine_centers
        ]
        vectors = _merge_horizons_vectors(vectors, additions)
    states = _interp_state_on_grid(vectors, jd_grid)
    return states[:, :3], vectors.source.url


def _integrate_cascade_dynamics(
    neo: NEOObject,
    hypothesis: HypothesisTerms,
    jd: Any,
    initial_state_au_d: Any,
    reference_state_au_d: Any,
    perturbers: dict[str, dict[str, Any]],
    frame: str,
    weights: tuple[float, float, float],
    max_step_days: float,
    phase_warp_gain: float,
    cascade_accel_au_d2: float | None,
    integrator_method: str,
    integrator_rtol: float,
    integrator_atol: float,
    segment_days: float,
    reset_jds: list[float],
    include_relativity: bool,
    include_standard_nongrav: bool,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    jd_arr = np.asarray(jd, dtype=float)
    reference_state = np.asarray(reference_state_au_d, dtype=float)
    state = np.zeros((len(jd_arr), 6), dtype=float)
    state[0] = np.asarray(initial_state_au_d, dtype=float)
    mu_sun = GM_SUN_M3_S2 * (SECONDS_PER_DAY * SECONDS_PER_DAY) / (AU_M**3)
    max_step = max(float(max_step_days), 1e-4)
    method = _parse_integrator_method(integrator_method)
    cascade_norms: list[float] = []
    phase_norms: list[float] = []
    gr_norms: list[float] = []
    nongrav_norms: list[float] = []
    reset_indices: set[int] = set()
    if segment_days and float(segment_days) > 0.0:
        last_jd = float(jd_arr[0])
        for idx, value in enumerate(jd_arr[1:], start=1):
            if float(value) - last_jd >= float(segment_days):
                reset_indices.add(idx)
                last_jd = float(value)
    for reset_jd in reset_jds:
        idx = int(np.argmin(np.abs(jd_arr - float(reset_jd))))
        if 0 < idx < len(jd_arr):
            reset_indices.add(idx)

    def rhs(t_jd: float, y: Any) -> Any:
        r = np.asarray(y[:3], dtype=float)
        v = np.asarray(y[3:], dtype=float)
        if frame == "barycentric":
            acc = np.zeros(3, dtype=float)
            sun = perturbers.get("sun")
            if sun is not None:
                r_sun = _interp_vector(jd_arr, sun["positions_au"], t_jd)
                v_sun = _interp_vector(jd_arr, sun["velocities_au_d"], t_jd)
            else:
                r_sun = np.zeros(3, dtype=float)
                v_sun = np.zeros(3, dtype=float)
            cascade_r = r - r_sun
            cascade_v = v - v_sun
        else:
            r_norm = max(float(np.linalg.norm(r)), 1e-300)
            acc = -mu_sun * r / (r_norm**3)
            cascade_r = r
            cascade_v = v

        for body in perturbers.values():
            r_body = _interp_vector(jd_arr, body["positions_au"], t_jd)
            rel = r_body - r
            rel_norm = max(float(np.linalg.norm(rel)), 1e-300)
            if frame == "barycentric":
                acc += float(body["mu_au3_d2"]) * rel / (rel_norm**3)
            else:
                body_norm = max(float(np.linalg.norm(r_body)), 1e-300)
                acc += float(body["mu_au3_d2"]) * (rel / (rel_norm**3) - r_body / (body_norm**3))
        if include_relativity:
            acc += _solar_relativistic_correction_au_d2(cascade_r, cascade_v, mu_sun)
        if include_standard_nongrav:
            acc += _standard_nongrav_acceleration_au_d2(cascade_r, cascade_v, neo)
        cascade_acc = _cascade_acceleration_au_d2(cascade_r, cascade_v, neo, hypothesis, weights, cascade_accel_au_d2)
        acc += cascade_acc + _phase_modulated_acceleration_au_d2(cascade_acc, hypothesis, phase_warp_gain)
        return np.concatenate([v, acc])

    def cascade_relative_state(t_jd: float, y: Any) -> tuple[Any, Any]:
        if frame == "barycentric" and "sun" in perturbers:
            sun_pos = _interp_vector(jd_arr, perturbers["sun"]["positions_au"], t_jd)
            sun_vel = _interp_vector(jd_arr, perturbers["sun"]["velocities_au_d"], t_jd)
            return np.asarray(y[:3], dtype=float) - sun_pos, np.asarray(y[3:], dtype=float) - sun_vel
        return np.asarray(y[:3], dtype=float), np.asarray(y[3:], dtype=float)

    if method == "DOP853":
        try:
            from scipy.integrate import solve_ivp  # type: ignore
        except Exception as exc:
            raise SourceError("DOP853 integration requires scipy.integrate.solve_ivp") from exc

        boundaries = sorted({0, len(jd_arr) - 1, *reset_indices})
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start in reset_indices or start == 0:
                state[start] = reference_state[start]
            if end <= start:
                continue
            t_eval = jd_arr[start : end + 1]
            sol = solve_ivp(
                rhs,
                (float(t_eval[0]), float(t_eval[-1])),
                state[start],
                method="DOP853",
                t_eval=t_eval,
                rtol=float(integrator_rtol),
                atol=float(integrator_atol),
                max_step=max_step,
            )
            if not sol.success:
                raise SourceError(f"DOP853 integration failed: {sol.message}")
            state[start : end + 1] = sol.y.T
    else:
        for i in range(1, len(jd_arr)):
            if i in reset_indices:
                state[i] = reference_state[i]
                continue
            t0 = float(jd_arr[i - 1])
            t1 = float(jd_arr[i])
            y = state[i - 1].copy()
            total_dt = t1 - t0
            substeps = max(1, int(math.ceil(abs(total_dt) / max_step)))
            h = total_dt / substeps
            t = t0
            for _ in range(substeps):
                k1 = rhs(t, y)
                k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
                k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
                k4 = rhs(t + h, y + h * k3)
                cascade_r, cascade_v = cascade_relative_state(t, y)
                cascade_acc = _cascade_acceleration_au_d2(cascade_r, cascade_v, neo, hypothesis, weights, cascade_accel_au_d2)
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                warp = _phase_warp_displacement_au(cascade_acc, h, hypothesis, phase_warp_gain)
                y[:3] += warp
                t += h
                phase_norms.append(float(np.linalg.norm(warp)))
            state[i] = y

    for t_jd, y in zip(jd_arr, state):
        cascade_r, cascade_v = cascade_relative_state(float(t_jd), y)
        cascade_acc = _cascade_acceleration_au_d2(cascade_r, cascade_v, neo, hypothesis, weights, cascade_accel_au_d2)
        cascade_norms.append(float(np.linalg.norm(cascade_acc)))
        if method == "DOP853":
            phase_norms.append(float(np.linalg.norm(_phase_modulated_acceleration_au_d2(cascade_acc, hypothesis, phase_warp_gain))))
        gr_norms.append(float(np.linalg.norm(_solar_relativistic_correction_au_d2(cascade_r, cascade_v, mu_sun))) if include_relativity else 0.0)
        nongrav_norms.append(float(np.linalg.norm(_standard_nongrav_acceleration_au_d2(cascade_r, cascade_v, neo))) if include_standard_nongrav else 0.0)

    diagnostics = {
        "integrator_substep_max_days": float(max_step),
        "integrator_method": method,
        "integrator_rtol": float(integrator_rtol),
        "integrator_atol": float(integrator_atol),
        "dynamics_frame": frame,
        "nbody_perturber_count": float(len(perturbers)),
        "nbody_perturbers": ",".join(perturbers.keys()),
        "nbody_force_model": "barycentric direct N-body accelerations from Horizons vectors" if frame == "barycentric" else "heliocentric indirect third-body perturbations from Horizons planet vectors",
        "state_refresh_count": float(len(reset_indices)),
        "state_refresh_indices": sorted(int(i) for i in reset_indices),
        "state_refresh_segment_days": float(segment_days),
        "state_refresh_reset_jds": [float(value) for value in reset_jds],
        "cascade_vector_weight_velocity": float(weights[0]),
        "cascade_vector_weight_radial": float(weights[1]),
        "cascade_vector_weight_normal": float(weights[2]),
        "cascade_acceleration_source": "SBDB A1/A2 norm" if cascade_accel_au_d2 is None else "user-specified",
        "cascade_acceleration_au_d2_median": float(np.median(cascade_norms)) if cascade_norms else 0.0,
        "cascade_acceleration_au_d2_max": float(np.max(cascade_norms)) if cascade_norms else 0.0,
        "solar_relativity_enabled": bool(include_relativity),
        "solar_gr_acceleration_au_d2_median": float(np.median(gr_norms)) if gr_norms else 0.0,
        "solar_gr_acceleration_au_d2_max": float(np.max(gr_norms)) if gr_norms else 0.0,
        "standard_nongrav_enabled": bool(include_standard_nongrav),
        "standard_nongrav_acceleration_au_d2_median": float(np.median(nongrav_norms)) if nongrav_norms else 0.0,
        "standard_nongrav_acceleration_au_d2_max": float(np.max(nongrav_norms)) if nongrav_norms else 0.0,
        "phase_warp_gain": float(phase_warp_gain),
        "phase_warp_application": "discrete-rk4-position-warp" if method == "RK4" else "continuous-adaptive-acceleration-modulation",
        "phase_acceleration_au_d2_median": float(np.median(phase_norms)) if phase_norms else 0.0,
        "phase_acceleration_au_d2_max": float(np.max(phase_norms)) if phase_norms else 0.0,
    }
    return state, diagnostics


def _write_dynamics_tables(
    output_dir: Path,
    meta: dict[str, Any],
    integrated_state: Any,
    integrated_distance_au: Any,
    residual_km: Any,
    anchor_rows: list[dict[str, Any]],
) -> list[str]:
    import csv
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    jd = np.asarray(meta["jd"], dtype=float)
    calendar = list(meta["calendar"])
    state = np.asarray(integrated_state, dtype=float)
    dist = np.asarray(integrated_distance_au, dtype=float)
    residual = np.asarray(residual_km, dtype=float)
    rows: list[dict[str, Any]] = []
    for i in range(len(jd)):
        rows.append(
            {
                "jd_tdb": float(jd[i]),
                "calendar_tdb": calendar[i],
                "horizons_geocentric_distance_au": float(meta["dist_au"][i]),
                "integrated_geocentric_distance_au": float(dist[i]),
                "integrated_minus_horizons_km": float(residual[i]),
                "integrated_helio_x_au": float(state[i, 0]),
                "integrated_helio_y_au": float(state[i, 1]),
                "integrated_helio_z_au": float(state[i, 2]),
                "integrated_helio_vx_au_d": float(state[i, 3]),
                "integrated_helio_vy_au_d": float(state[i, 4]),
                "integrated_helio_vz_au_d": float(state[i, 5]),
                "gi_log10_abs": float(meta["gi_log"][i]),
                "oi_log10_abs": float(meta["oi_log"][i]),
            }
        )
    ts_path = output_dir / "table_dynamical_integrator_timeseries.csv"
    with ts_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    paths.append(str(ts_path))

    if anchor_rows:
        anchor_path = output_dir / "table_dynamical_integrator_anchor_validation.csv"
        with anchor_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(anchor_rows[0].keys()))
            writer.writeheader()
            writer.writerows(anchor_rows)
        paths.append(str(anchor_path))
    return paths


def _write_dynamics_plots(
    output_dir: Path,
    neo: NEOObject,
    meta: dict[str, Any],
    integrated_distance_au: Any,
    residual_km: Any,
    integrated_state: Any,
    anchor_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[str]:
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    jd = np.asarray(meta["jd"], dtype=float)
    t_year = (jd - jd[0]) / 365.25636
    horizons_km = np.asarray(meta["dist_au"], dtype=float) * AU_KM
    integrated_km = np.asarray(integrated_distance_au, dtype=float) * AU_KM
    residual = np.asarray(residual_km, dtype=float)

    def _save(fig: Any, stem: str, dpi: int = 220) -> None:
        for suffix in ["png", "svg", "pdf"]:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
            figures.append(str(path))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.8, 6.3))
    ax.plot(t_year, horizons_km, color="#1d3557", lw=1.5, label="Horizons geocentric range")
    ax.plot(t_year, integrated_km, color="#d1495b", lw=1.1, label="numerical cascade integration")
    if anchor_rows:
        ax.scatter(
            [(float(row["cad_jd_tdb"]) - jd[0]) / 365.25636 for row in anchor_rows],
            [float(row["cad_distance_km"]) for row in anchor_rows],
            s=42,
            color="#edae49",
            edgecolor="black",
            linewidth=0.5,
            label="CAD anchors",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Years from start")
    ax.set_ylabel("Geocentric distance (km)")
    ax.set_title("Dynamics-First Cascade Propagation Against Horizons")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save(fig, "fig_dynamical_integrator_distance")

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.plot(t_year, residual, color="#2a9d8f", lw=1.1)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("Years from start")
    ax.set_ylabel("Integrated minus Horizons distance (km)")
    ax.set_title("Numerical Propagation Residual Relative to Horizons")
    ax.grid(True, alpha=0.25)
    _save(fig, "fig_dynamical_integrator_residuals")

    state = np.asarray(integrated_state, dtype=float)
    earth_for_plot = np.asarray(meta.get("dynamics_earth_au", meta["earth_helio_au"]), dtype=float)
    geo_pos = state[:, :3] - earth_for_plot
    fig = plt.figure(figsize=(10.8, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(geo_pos[:, 0] * AU_KM / 1.0e6, geo_pos[:, 1] * AU_KM / 1.0e6, geo_pos[:, 2] * AU_KM / 1.0e6, color="#1d3557", lw=1.0)
    if anchor_rows:
        idxs = [int(np.argmin(np.abs(jd - float(row["cad_jd_tdb"])))) for row in anchor_rows]
        ax.scatter(
            geo_pos[idxs, 0] * AU_KM / 1.0e6,
            geo_pos[idxs, 1] * AU_KM / 1.0e6,
            geo_pos[idxs, 2] * AU_KM / 1.0e6,
            s=42,
            color="#d1495b",
            depthshade=True,
        )
    ax.set_xlabel("X (million km)")
    ax.set_ylabel("Y (million km)")
    ax.set_zlabel("Z (million km)")
    ax.set_title("Integrated Geocentric 3-D Trajectory")
    _save(fig, "fig_dynamical_integrator_3d_trajectory")

    caption_path = output_dir / "dynamical_integrator_captions.md"
    caption_path.write_text(
        "\n\n".join(
            [
                "# Dynamical Integrator Figures",
                "Figure: Dynamics-first cascade propagation against Horizons. The numerical trajectory is propagated from a Horizons state in the selected dynamics frame with live-Horizons N-body perturbing bodies and the GI_N/OI_N cascade field, while CAD anchors remain validation overlays.",
                "Figure: Numerical propagation residual relative to Horizons. Residuals expose drift introduced by integrating a simplified force model rather than refitting to Horizons or CAD labels.",
                "Figure: Integrated geocentric 3-D trajectory. The curve is the propagated state transformed into the Earth-centered frame for spatial inspection of the encounter geometry.",
                f"Cascade diagnostics: median acceleration={diagnostics.get('cascade_acceleration_au_d2_median', 0.0):.6e} au/d^2; max phase acceleration={diagnostics.get('phase_acceleration_au_d2_max', 0.0):.6e} au/d^2.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    figures.append(str(caption_path))
    return figures


def run_dynamical_propagation(
    neo: NEOObject,
    hypothesis: HypothesisTerms,
    target: str,
    date_min: str,
    date_max: str,
    horizons_step: str,
    output_dir: Path,
    refine_step: str = "1h",
    refine_window_days: float = 5.0,
    refine: bool = True,
    uncertainty_samples: int = 768,
    cascade_vector_weights: str = "1,0,1",
    integrator_max_step_days: float = 0.125,
    phase_warp_gain: float = 1.0,
    cascade_accel_au_d2: float | None = None,
    nbody_bodies: str = "mercury,venus,earth,moon,mars,jupiter,saturn,uranus,neptune",
    dynamics_frame: str = "barycentric",
    integrator_method: str = "DOP853",
    integrator_rtol: float = 1e-11,
    integrator_atol: float = 1e-13,
    state_refresh_days: float = 0.0,
    post_encounter_reset_days: float = 0.0,
    include_relativity: bool = True,
    include_standard_nongrav: bool = True,
) -> DynamicalPropagationReport:
    import numpy as np

    geo = fetch_horizons_vectors(target, "500@399", date_min, date_max, horizons_step)
    helio = fetch_horizons_vectors(target, "500@10", date_min, date_max, horizons_step)
    refine_centers: list[float] = []
    if refine:
        refine_centers = _refinement_center_jds(
            geo,
            refine_window_days,
            anchor_jds=[ca.jd_tdb for ca in neo.close_approaches],
        )
        geo, helio = _refine_horizons_near_minima(
            target,
            geo,
            helio,
            refine_step=refine_step,
            window_days=refine_window_days,
            anchor_jds=[ca.jd_tdb for ca in neo.close_approaches],
        )

    _, _, _, meta = _build_ml_feature_matrix(neo, hypothesis, geo, helio)
    meta = dict(meta)
    jd = np.asarray(meta["jd"], dtype=float)
    helio_state = np.asarray(meta["helio_pos_au"], dtype=float)
    helio_vel = np.asarray(meta["helio_vel_au_d"], dtype=float)
    earth_helio = np.asarray(meta["earth_helio_au"], dtype=float)
    weights = _parse_vector_weights(cascade_vector_weights)
    frame = _parse_dynamics_frame(dynamics_frame)
    nbody_names = _parse_nbody_bodies(nbody_bodies)
    if frame == "barycentric" and "sun" not in nbody_names:
        nbody_names = ["sun"] + nbody_names
    perturbers = _fetch_planetary_perturbers(
        nbody_names,
        date_min,
        date_max,
        horizons_step,
        earth_helio,
        jd,
        frame,
        refine_step=refine_step if refine else None,
        refine_window_days=refine_window_days,
        refine_centers=refine_centers,
    )
    earth_for_distance, earth_source = _fetch_earth_position_for_frame(
        frame,
        date_min,
        date_max,
        horizons_step,
        earth_helio,
        jd,
        refine_step=refine_step if refine else None,
        refine_window_days=refine_window_days,
        refine_centers=refine_centers,
    )
    meta["dynamics_earth_au"] = earth_for_distance
    if frame == "barycentric":
        bary_target = fetch_horizons_vectors(target, "500@0", date_min, date_max, horizons_step)
        if refine and refine_centers:
            bary_target = _merge_horizons_vectors(
                bary_target,
                [
                    fetch_horizons_vectors(target, "500@0", _jd_to_iso_date(center_jd - refine_window_days), _jd_to_iso_date(center_jd + refine_window_days), refine_step)
                    for center_jd in refine_centers
                ],
            )
        bary_state = _interp_state_on_grid(bary_target, jd)
        reference_state = np.asarray(bary_state, dtype=float)
    else:
        reference_state = np.column_stack([helio_state, helio_vel])
    initial_state = np.asarray(reference_state[0], dtype=float)

    reset_jds: list[float] = []
    if post_encounter_reset_days and float(post_encounter_reset_days) > 0.0:
        horizons_dist_au_for_reset = np.asarray(meta["dist_au"], dtype=float)
        for idx in range(1, len(jd) - 1):
            if horizons_dist_au_for_reset[idx] <= horizons_dist_au_for_reset[idx - 1] and horizons_dist_au_for_reset[idx] <= horizons_dist_au_for_reset[idx + 1]:
                reset_jds.append(float(jd[idx] + float(post_encounter_reset_days)))

    integrated_state, dyn_diag = _integrate_cascade_dynamics(
        neo,
        hypothesis,
        jd,
        initial_state,
        reference_state,
        perturbers,
        frame,
        weights,
        integrator_max_step_days,
        phase_warp_gain,
        cascade_accel_au_d2,
        integrator_method,
        integrator_rtol,
        integrator_atol,
        state_refresh_days,
        reset_jds,
        include_relativity,
        include_standard_nongrav,
    )
    dyn_diag["nbody_perturber_commands"] = {name: body["command"] for name, body in perturbers.items()}
    dyn_diag["nbody_perturber_sources"] = {name: body["source"] for name, body in perturbers.items()}
    dyn_diag["earth_distance_state_source"] = earth_source
    dyn_diag["prediction_mode"] = "single_arc_predictor" if float(state_refresh_days) <= 0.0 and float(post_encounter_reset_days) <= 0.0 else "osculating_reconstruction"
    dyn_diag["perturber_refinement_enabled"] = bool(refine and refine_centers)
    dyn_diag["perturber_refinement_window_count"] = float(len(refine_centers))
    integrated_geo = integrated_state[:, :3] - earth_for_distance
    integrated_dist_au = np.linalg.norm(integrated_geo, axis=1)
    horizons_dist_au = np.asarray(meta["dist_au"], dtype=float)
    residual_km = (integrated_dist_au - horizons_dist_au) * AU_KM
    mae_km = float(np.mean(np.abs(residual_km)))
    rmse_km = float(np.sqrt(np.mean(residual_km * residual_km)))
    true_min_idx = int(np.argmin(horizons_dist_au))
    pred_min_idx = int(np.argmin(integrated_dist_au))
    nearest_error_km = float((integrated_dist_au[pred_min_idx] - horizons_dist_au[pred_min_idx]) * AU_KM)

    anchor_rows: list[dict[str, Any]] = []
    for i, ca in enumerate(neo.close_approaches, start=1):
        cad_jd = float(ca.jd_tdb)
        idx = int(np.argmin(np.abs(jd - cad_jd)))
        integrated_at_cad = float(np.interp(cad_jd, jd, integrated_dist_au) * AU_KM)
        horizons_at_cad = float(np.interp(cad_jd, jd, horizons_dist_au) * AU_KM)
        cad_km = float(ca.distance_au * AU_KM)
        anchor_rows.append(
            {
                "anchor_id": i,
                "cad_date_tdb": ca.calendar_date_tdb,
                "cad_jd_tdb": cad_jd,
                "cad_distance_km": cad_km,
                "integrated_distance_km": integrated_at_cad,
                "integrated_minus_cad_km": integrated_at_cad - cad_km,
                "horizons_interpolated_distance_km": horizons_at_cad,
                "horizons_interpolated_minus_cad_km": horizons_at_cad - cad_km,
                "nearest_sample_date_tdb": meta["calendar"][idx],
                "nearest_sample_offset_hours": float((jd[idx] - cad_jd) * 24.0),
                "v_rel_km_s": ca.v_rel_km_s,
                "orbit_id": ca.orbit_id,
            }
        )

    if anchor_rows:
        cad_residuals = np.asarray([row["integrated_minus_cad_km"] for row in anchor_rows], dtype=float)
        dyn_diag["cad_anchor_integrated_rmse_km"] = float(np.sqrt(np.mean(cad_residuals * cad_residuals)))
        dyn_diag["cad_anchor_integrated_mae_km"] = float(np.mean(np.abs(cad_residuals)))
        nearest_anchor = min(anchor_rows, key=lambda row: row["cad_distance_km"])
        cad_error = float(nearest_anchor["integrated_minus_cad_km"])
        dyn_diag["nearest_cad_integrated_error_km"] = cad_error
    else:
        cad_error = None

    table_paths = _write_dynamics_tables(output_dir, meta, integrated_state, integrated_dist_au, residual_km, anchor_rows)
    uncertainty_table_paths, uncertainty_figures, uncertainty_publication_assets = _write_uncertainty_propagation(
        output_dir,
        neo,
        hypothesis,
        geo,
        helio,
        anchor_rows,
        uncertainty_samples,
        dyn_diag,
    )
    table_paths.extend(uncertainty_table_paths)
    figure_paths = _write_dynamics_plots(output_dir, neo, meta, integrated_dist_au, residual_km, integrated_state, anchor_rows, dyn_diag)
    figure_paths.extend(uncertainty_figures)
    publication_assets = [path for path in figure_paths if path.endswith("_captions.md") or path.endswith("captions.md")]
    publication_assets.extend(uncertainty_publication_assets)

    return DynamicalPropagationReport(
        enabled=True,
        method="Dynamics-first numerical propagation with no supervised residual correction or CAD-trained surrogate",
        force_model=("Barycentric direct N-body gravity from live Horizons body vectors" if frame == "barycentric" else "Heliocentric solar gravity plus indirect third-body perturbations from live Horizons body vectors") + " + solar 1PN relativistic correction + SBDB A1/A2 radial-transverse non-gravitational acceleration + GI_N/OI_N cascade acceleration directed by velocity/radial/orbital-normal weights; Neo/gravity phasing is applied as a bounded position warp in RK4 mode and as continuous acceleration modulation in adaptive mode",
        n_samples=len(jd),
        horizons_step=f"{horizons_step} + refinement {refine_step} within +/-{refine_window_days:g}d of coarse minima" if refine else horizons_step,
        integrator=f"{_parse_integrator_method(integrator_method)} with max step {integrator_max_step_days:g} d, rtol={integrator_rtol:g}, atol={integrator_atol:g}; osculating refresh count={dyn_diag.get('state_refresh_count', 0):.0f}",
        validation_mae_km=mae_km,
        validation_rmse_km=rmse_km,
        nearest_horizons_date=meta["calendar"][true_min_idx],
        nearest_horizons_distance_au=float(horizons_dist_au[true_min_idx]),
        nearest_integrated_date=meta["calendar"][pred_min_idx],
        nearest_integrated_distance_au=float(integrated_dist_au[pred_min_idx]),
        nearest_integrated_error_km=nearest_error_km,
        cad_validation_error_km=cad_error,
        numerical_diagnostics=dyn_diag,
        anchor_validation=anchor_rows,
        figures=figure_paths,
        tables=table_paths,
        publication_assets=publication_assets,
        caveats=[
            "This branch removes supervised ML residual correction from the prediction path; CAD anchors are validation overlays only.",
            "Planetary perturbing-body positions are fetched from JPL Horizons at runtime; barycentric mode applies direct N-body accelerations, while heliocentric mode applies indirect third-body accelerations.",
            "Default dynamics mode is a single-arc predictor with no osculating refreshes; if refresh options are enabled, Horizons state vectors are used at declared segment/reset boundaries for controlled reconstruction, not operational forecasting.",
            "The cascade acceleration magnitude is source-backed by SBDB non-gravitational A1/A2 terms unless explicitly supplied by the user.",
            "The GI_N/OI_N cascade and phasing terms remain hypothesis-driven perturbations and are not an accepted replacement for JPL force modeling or orbit determination.",
            "The integrator starts from a Horizons state vector, so it is a challenger dynamics experiment rather than an independent astrometric orbit solution.",
        ],
    )


def _ml_figure_caption_library() -> dict[str, tuple[str, str]]:
    return {
        "fig_ml_surrogate_distance": (
            "Surrogate Distance History Against Horizons Vectors",
            "Geocentric range from JPL Horizons state vectors is compared with the no-CAD-anchor surrogate prediction over the full sampled interval. When available, the arc-wide global model and the final global-plus-local refined surrogate are both shown, while the shaded band reports the calibrated uncertainty envelope and CAD epochs appear only as external validation anchors rather than training labels.",
        ),
        "fig_ml_validation_residuals": (
            "Blocked Validation Residual History",
            "Residuals on the purged time-block validation split, defined as surrogate minus Horizons geocentric distance, are shown as a function of epoch. The plot makes the temporal structure of the error field explicit and reports the corresponding MAE and RMSE in physical units.",
        ),
        "fig_ml_validation_parity": (
            "Validation Parity Plot",
            "Parity between Horizons geocentric distance and surrogate prediction on withheld validation samples. The one-to-one line marks perfect agreement, while departures from the line expose regime-dependent bias across the dynamic range of close and distant configurations.",
        ),
        "fig_pdf_feature_dynamics": (
            "PDF Feature Dynamics Versus Range",
            "The geocentric distance history is plotted together with the log-magnitude GI_N and OI_N diagnostics derived from the NEO hypothesis algebra. This view shows where the hypothesis-driven feature family intensifies relative to the underlying orbital geometry.",
        ),
        "fig_publication_orbital_physics": (
            "Orbital Physics Overview",
            "Publication-style state-vector summary for the sampled interval. Panel A shows geocentric and heliocentric distance histories; panel B resolves the velocity field into total, radial, transverse, and heliocentric components; panel C reports acceleration-scale diagnostics; panel D shows specific angular momentum together with instantaneous heliocentric eccentricity; panel E gives the geocentric ecliptic projection; and panel F shows the distance-rate phase portrait. CAD epochs remain validation anchors only.",
        ),
        "fig_publication_hypothesis_parameters": (
            "NEO Hypothesis Parameter Overview",
            "Publication-style overview of the NEO hypothesis parameter family used as surrogate features. Panels summarize the J_sun,crit range, time-scale diagnostics, transport and phasing terms, sequence diagnostics, outcome proxies, and the coupled GI/OI scaled diagnostics so that the stability and concentration of the PDF-derived feature family can be assessed directly against epoch.",
        ),
        "fig_tensorflow_gate_blend_selection": (
            "TensorFlow Calibration Gate Blend Selection",
            "Calibration-gate diagnostics for the neural residual branches. Candidate blend strengths are evaluated against the untouched calibration block, with the accepted blend marked explicitly. A zero accepted blend means the TensorFlow proposal failed to improve on the state-vector physics baseline under the predeclared gate.",
        ),
        "fig_tensorflow_correction_field": (
            "Rejected and Accepted TensorFlow Correction Field",
            "Time-resolved neural correction field relative to the state-vector range baseline. The full TensorFlow residual proposal is shown separately from the accepted correction after calibration gating, making visible when the neural branch is active, suppressed, or entirely rejected.",
        ),
        "fig_tensorflow_anchor_synthesis": (
            "CAD Anchor Synthesis of Gate Outcomes",
            "CAD-anchor residuals are compared for the nearest Horizons sample, the unconstrained TensorFlow global residual proposal, the accepted gated surrogate, and the TensorFlow continuous-time encounter reconstructor. This separates raw sampling offset, rejected neural residual behavior, and the dedicated close-approach reconstruction model at authoritative anchors.",
        ),
        "fig_anchor_distance_comparison": (
            "Anchor Distance Comparison",
            "CAD close-approach distances are compared against the nearest Horizons sample and the surrogate prediction at the same epochs. The logarithmic axis exposes the wide range of Earth-approach scales while keeping direct visual contact with the authoritative anchor values.",
        ),
        "fig_anchor_residual_bars": (
            "Anchor Residual Comparison",
            "Residuals relative to CAD anchors are shown separately for the nearest Horizons sample and for the surrogate prediction. This isolates sampling error from surrogate error and clarifies how much improvement comes from the learned challenger model relative to raw stepwise sampling.",
        ),
        "fig_anchor_parity": (
            "Anchor Parity Against CAD",
            "Parity plot of CAD anchor distance versus both the nearest Horizons sample and the surrogate prediction. Points closer to the one-to-one line indicate better agreement with the authoritative anchor values.",
        ),
        "fig_anchor_conformal_intervals": (
            "CAD Anchors With Conformal Intervals",
            "CAD anchor distances are plotted against surrogate predictions together with the conformal uncertainty interval at each anchor epoch. The figure is intended for manuscript use when discussing empirical coverage and the degree to which the authoritative anchors fall within the surrogate uncertainty band.",
        ),
        "fig_3d_surrogate_residual_vs_cad": (
            "Three-Dimensional Residual Trajectory in CAD Anchor Space",
            "Diagnostic phase-space trajectory built from CAD anchor epoch, CAD anchor distance, and surrogate residual. The construction avoids implying a literal three-dimensional orbit and instead presents a faithful three-axis residual geometry for validation against authoritative close-approach anchors.",
        ),
        "fig_3d_encounter_model": (
            "Three-Dimensional Encounter Reconstruction",
            "Geocentric 3-D close-approach reconstruction for the nearest CAD encounter. The Horizons state-vector trajectory supplies the spatial path, the JPL CAD range is represented as an authoritative range shell placed along the Horizons line of sight, and the TensorFlow continuous-time encounter reconstruction is overlaid as the model-predicted CAD-epoch range for direct visual comparison.",
        ),
        "fig_ml_close_approach_zoom": (
            "Close-Approach Zoom",
            "Refined view around the Horizons minimum-distance window, showing the sampled Horizons range, the arc-wide global model, the final local-refined surrogate, its calibrated uncertainty envelope, and any CAD overlays that fall within the zoom interval. This panel is appropriate for close-approach discussion in the results section of a manuscript.",
        ),
        "fig_ml_model_comparison": (
            "TensorFlow Model Family Comparison",
            "Validation RMSE across the retained TensorFlow challenger models. The reported point predictor is selected on purged chronological folds and is not blended with CAD anchors.",
        ),
        "fig_ml_primary_selection_cv": (
            "Primary TensorFlow Selection by Purged Cross-Validation",
            "Mean blocked cross-validation RMSE with one-standard-deviation error bars for the candidate TensorFlow neural configurations considered for the primary surrogate. The selected configuration is highlighted to document that the final point-prediction engine was chosen using a purged time-based model-selection step rather than by ad hoc preference.",
        ),
        "fig_ml_coverage_by_regime": (
            "Empirical Interval Coverage by Distance Regime",
            "Observed 90% interval coverage across near, intermediate, and far validation regimes, defined by validation-distance terciles. The dashed line marks nominal coverage and allows direct assessment of whether the conformalized interval calibration is conservative, well-calibrated, or under-dispersed in different orbital-distance ranges.",
        ),
        "fig_ml_feature_importance": (
            "Primary TensorFlow Feature Salience",
            "Relative first-layer salience from the primary TensorFlow surrogate. The ranking indicates which orbital-physics and hypothesis-derived inputs most strongly influence the challenger model across the sampled range history.",
        ),
        "fig_ml_residual_distribution": (
            "Residual Distribution on Withheld Validation Samples",
            "Histogram of surrogate residuals on the blocked validation set. The caption is intended to accompany reported conformal coverage and to show whether the error field is centered, skewed, or heavy-tailed in physical distance units.",
        ),
    }


def _write_figure_captions(output_dir: Path, figure_catalog: list[dict[str, str]]) -> list[str]:
    if not figure_catalog:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    md_path = output_dir / "figure_captions.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Figure Captions\n\n")
        for idx, entry in enumerate(figure_catalog, start=1):
            fh.write(f"## Figure {idx}. {entry['title']}\n\n")
            fh.write(f"Files: `{entry['png_name']}`, `{entry['svg_name']}`, `{entry['pdf_name']}`\n\n")
            fh.write(entry["caption"] + "\n\n")
    paths.append(str(md_path))

    def _latex_escape(text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(ch, ch) for ch in text)

    tex_path = output_dir / "figure_captions.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        for entry in figure_catalog:
            fh.write("\\begin{figure}[htbp]\n")
            fh.write("  \\centering\n")
            fh.write(f"  \\includegraphics[width=\\linewidth]{{{_latex_escape(entry['pdf_name'])}}}\n")
            fh.write(f"  \\caption{{{_latex_escape(entry['caption'])}}}\n")
            fh.write(f"  \\label{{fig:{_latex_escape(entry['label'])}}}\n")
            fh.write("\\end{figure}\n\n")
    paths.append(str(tex_path))
    return paths


def _build_ml_feature_matrix(
    neo: NEOObject,
    hypothesis: HypothesisTerms,
    geo: HorizonsVectors,
    helio: HorizonsVectors,
) -> tuple[Any, Any, list[str], dict[str, Any]]:
    import numpy as np

    geo_state = np.asarray(geo.state_au_d, dtype=float)
    helio_state = np.asarray(helio.state_au_d, dtype=float)
    n = min(len(geo_state), len(helio_state), len(geo.jd_tdb), len(helio.jd_tdb))
    if n < 80:
        raise SourceError(f"ML surrogate needs at least 80 Horizons samples; got {n}")
    geo_state = geo_state[:n]
    helio_state = helio_state[:n]
    jd = np.asarray(geo.jd_tdb[:n], dtype=float)

    geo_pos = geo_state[:, :3]
    geo_vel = geo_state[:, 3:6]
    helio_pos = helio_state[:, :3]
    helio_vel = helio_state[:, 3:6]

    dist_au = np.linalg.norm(geo_pos, axis=1)
    v_geo_au_d = np.linalg.norm(geo_vel, axis=1)
    range_rate_au_d = np.einsum("ij,ij->i", geo_pos, geo_vel) / np.maximum(dist_au, 1e-300)
    transverse_speed_au_d = np.sqrt(np.maximum(v_geo_au_d * v_geo_au_d - range_rate_au_d * range_rate_au_d, 0.0))
    geo_unit = geo_pos / np.maximum(dist_au[:, None], 1e-300)
    geo_h_vec_au2_d = np.cross(geo_pos, geo_vel)
    geo_ang_mom_au2_d = np.linalg.norm(geo_h_vec_au2_d, axis=1)

    r_helio_au = np.linalg.norm(helio_pos, axis=1)
    v_helio_au_d = np.linalg.norm(helio_vel, axis=1)
    helio_unit = helio_pos / np.maximum(r_helio_au[:, None], 1e-300)
    helio_h_vec_au2_d = np.cross(helio_pos, helio_vel)
    j_helio_au2_d = np.linalg.norm(helio_h_vec_au2_d, axis=1)
    cos_geo_helio = np.einsum("ij,ij->i", geo_pos, helio_pos) / np.maximum(dist_au * r_helio_au, 1e-300)
    geo_accel_au_d2 = np.gradient(geo_vel, jd, axis=0)
    geo_jerk_au_d3 = np.gradient(geo_accel_au_d2, jd, axis=0)
    helio_accel_au_d2 = np.gradient(helio_vel, jd, axis=0)
    geo_accel_norm_au_d2 = np.linalg.norm(geo_accel_au_d2, axis=1)
    helio_accel_norm_au_d2 = np.linalg.norm(helio_accel_au_d2, axis=1)
    mu_sun_au3_d2 = GM_SUN_M3_S2 * (SECONDS_PER_DAY * SECONDS_PER_DAY) / (AU_M**3)
    helio_energy_au2_d2 = 0.5 * v_helio_au_d * v_helio_au_d - mu_sun_au3_d2 / np.maximum(r_helio_au, 1e-300)
    helio_ecc_vec = np.cross(helio_vel, helio_h_vec_au2_d) / max(mu_sun_au3_d2, 1e-300) - helio_unit
    helio_ecc_inst = np.linalg.norm(helio_ecc_vec, axis=1)
    sma_denom = 2.0 * helio_energy_au2_d2
    sma_denom = np.where(np.abs(sma_denom) > 1e-300, sma_denom, np.sign(sma_denom + 1e-300) * 1e-300)
    helio_sma_inst_au = -mu_sun_au3_d2 / sma_denom

    v_helio_m_s = v_helio_au_d * AU_M / SECONDS_PER_DAY
    r_helio_m = r_helio_au * AU_M
    j_helio_m2_s = j_helio_au2_d * (AU_M * AU_M) / SECONDS_PER_DAY
    upsilon = v_helio_m_s / C_M_S
    gamma = hypothesis.inputs.gamma_ratio
    gi_n = gamma * j_helio_m2_s / (1.0 + upsilon * upsilon)
    energy = 0.5 * v_helio_m_s * v_helio_m_s - GM_SUN_M3_S2 / np.maximum(r_helio_m, 1e-300)
    oi_n = (energy**4) * (j_helio_m2_s**2) * (gamma**2) - 1.0

    t_days = jd - jd[0]
    range_accel_au_d2 = np.gradient(range_rate_au_d, jd)
    range_jerk_au_d3 = np.gradient(range_accel_au_d2, jd)
    log_range_curve_d2 = np.gradient(np.gradient(np.log(np.maximum(dist_au, 1e-300)), jd), jd)
    log_range_third_d3 = np.gradient(log_range_curve_d2, jd)
    geo_speed_slope_au_d2 = np.gradient(v_geo_au_d, jd)
    helio_speed_slope_au_d2 = np.gradient(v_helio_au_d, jd)
    period = max(neo.elements.period_days, 1.0)
    ecc = neo.elements.eccentricity
    delta_t = neo.elements.period_days * SECONDS_PER_DAY

    # Time-varying PDF-derived terms. These are engineered features, not labels.
    # The feature set intentionally carries the full extracted PDF term family:
    # low/median/high N-range quantities, sequence terms, scaled diagnostics,
    # and GI/OI-like invariants. They are kept as candidate explanatory
    # coordinates for the surrogate rather than as an orbit propagator.
    pdf_terms: dict[str, list[float]] = {
        "jsuncritical_low": [],
        "jsuncritical_median": [],
        "jsuncritical_high": [],
        "time_norm_low": [],
        "time_norm_median": [],
        "time_norm_high": [],
        "time_cause": [],
        "acceleration_cause_inverse": [],
        "trajectory_slip_low": [],
        "trajectory_slip_median": [],
        "trajectory_slip_high": [],
        "trajectory_precession_low": [],
        "trajectory_precession_median": [],
        "trajectory_precession_high": [],
        "neo_phasing": [],
        "gravity_neo_phasing": [],
        "bound_factor_low": [],
        "bound_factor_median": [],
        "bound_factor_high": [],
        "time_slip_low": [],
        "time_slip_median": [],
        "time_slip_high": [],
        "lapse_factor_low": [],
        "lapse_factor_median": [],
        "lapse_factor_high": [],
        "sequence_1": [],
        "sequence_2": [],
        "sequence_3": [],
        "sequence_4": [],
        "sequence_5": [],
        "trajectory_loss": [],
        "seqcr": [],
        "likelihood_proxy": [],
        "new_eccentricity_proxy": [],
        "new_sma_au_proxy": [],
        "scaled_time_norm": [],
        "scaled_time_cause": [],
        "scaled_acceleration_cause_inverse": [],
        "scaled_trajectory_slip": [],
        "scaled_jsuncritical": [],
        "scaled_neo_phasing": [],
    }
    f_dist = (neo.elements.aphelion_au - neo.elements.perihelion_au) * AU_M
    jsun_obj = neo.elements.perihelion_au * AU_M
    prat = jsun_obj / max(f_dist, 1e-300)
    scale = float(PDF_SCALING["scale"])

    for U in v_helio_m_s:
        try:
            Uf = float(U)
            up = Uf / C_M_S
            high_jsun = _range_stats(calc_jsuncrit(k, delta_t, ecc, f_dist, gamma, Uf) for k in N_RANGE_HIGH)
            low_tnorm = _range_stats(calc_time_norm(k, gamma, ecc, f_dist, up, Uf) for k in N_RANGE_LOW)
            low_tslip = _range_stats(calc_trajectory_slip(k, gamma, ecc, f_dist, up, Uf) for k in N_RANGE_LOW)
            high_tprec = _range_stats(calc_trajectory_precession(k, delta_t, ecc, gamma, jsun_obj, Uf) for k in N_RANGE_HIGH)
            high_bfac = _range_stats(calc_bound_factor(k, delta_t, ecc, gamma, Uf) for k in N_RANGE_HIGH)
            low_time_slip = _range_stats(calc_time_slip(k, gamma, ecc, f_dist, up, Uf, jsun_obj) for k in N_RANGE_LOW)
            low_lfac = _range_stats(calc_lapse_factor(k, delta_t, gamma, ecc, f_dist, up, Uf, jsun_obj) for k in N_RANGE_LOW)
            tcause = calc_time_cause(gamma, ecc, f_dist, up, Uf, jsun_obj)
            acinv = calc_acceleration_cause_inverse(gamma, ecc, f_dist, Uf, jsun_obj, up)
            neoph = calc_neo_phasing(Uf, up, delta_t, ecc, gamma)
            gneoph = calc_gravity_neo_phasing(G_SI, delta_t, ecc, gamma, Uf, up)
            s1, s2, s3, s4, s5, trajectory_loss, seqcr = _sequence_terms(
                low_tslip.median,
                high_tprec.median,
                neoph,
                low_lfac.median,
                gneoph,
                high_bfac.median,
                acinv,
            )
            likelihood = (low_lfac.median / s4) / max(prat, 1e-300)
            new_ecc = ((1.0 - trajectory_loss / seqcr) / max(prat, 1e-300)) / 2.0
            new_peri = high_jsun.median * math.sqrt(abs(low_lfac.median))
            new_aph = new_peri * (1.0 + new_ecc) / max(1.0 - new_ecc, 1e-300)
            new_sma_au = 0.5 * (new_peri + new_aph) / AU_M
            values = {
                "jsuncritical_low": high_jsun.low,
                "jsuncritical_median": high_jsun.median,
                "jsuncritical_high": high_jsun.high,
                "time_norm_low": low_tnorm.low,
                "time_norm_median": low_tnorm.median,
                "time_norm_high": low_tnorm.high,
                "time_cause": tcause,
                "acceleration_cause_inverse": acinv,
                "trajectory_slip_low": low_tslip.low,
                "trajectory_slip_median": low_tslip.median,
                "trajectory_slip_high": low_tslip.high,
                "trajectory_precession_low": high_tprec.low,
                "trajectory_precession_median": high_tprec.median,
                "trajectory_precession_high": high_tprec.high,
                "neo_phasing": neoph,
                "gravity_neo_phasing": gneoph,
                "bound_factor_low": high_bfac.low,
                "bound_factor_median": high_bfac.median,
                "bound_factor_high": high_bfac.high,
                "time_slip_low": low_time_slip.low,
                "time_slip_median": low_time_slip.median,
                "time_slip_high": low_time_slip.high,
                "lapse_factor_low": low_lfac.low,
                "lapse_factor_median": low_lfac.median,
                "lapse_factor_high": low_lfac.high,
                "sequence_1": s1,
                "sequence_2": s2,
                "sequence_3": s3,
                "sequence_4": s4,
                "sequence_5": s5,
                "trajectory_loss": trajectory_loss,
                "seqcr": seqcr,
                "likelihood_proxy": likelihood,
                "new_eccentricity_proxy": new_ecc,
                "new_sma_au_proxy": new_sma_au,
                "scaled_time_norm": _scaled(PDF_SCALING["time_norm"], low_tnorm.median, scale),
                "scaled_time_cause": _scaled(PDF_SCALING["time_cause"], tcause, scale),
                "scaled_acceleration_cause_inverse": _scaled(PDF_SCALING["acceleration_cause_inverse"], acinv, scale),
                "scaled_trajectory_slip": _scaled(PDF_SCALING["trajectory_slip"], low_tslip.median, scale),
                "scaled_jsuncritical": _scaled(PDF_SCALING["jsuncritical"], high_jsun.median, scale),
                "scaled_neo_phasing": _scaled(PDF_SCALING["neo_phasing"], neoph, scale),
            }
        except Exception:
            values = {name: float("nan") for name in pdf_terms}
        for name in pdf_terms:
            pdf_terms[name].append(float(values[name]))

    pdf_arrays = {name: _finite_array(values) for name, values in pdf_terms.items()}

    feature_names: list[str] = []
    feature_columns: list[Any] = []

    def add_feature(name: str, values: Any) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        feature_names.append(name)
        feature_columns.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))

    def add_vector(prefix: str, values: Any, suffixes: tuple[str, str, str] = ("x", "y", "z")) -> None:
        arr = np.asarray(values, dtype=float)
        for j, suffix in enumerate(suffixes):
            add_feature(f"{prefix}_{suffix}", arr[:, j])

    add_feature("t_norm", t_days / max(t_days[-1], 1.0))
    add_feature("sin_orbit_phase", np.sin(2.0 * np.pi * t_days / period))
    add_feature("cos_orbit_phase", np.cos(2.0 * np.pi * t_days / period))
    add_feature("sin_earth_year", np.sin(2.0 * np.pi * t_days / 365.25636))
    add_feature("cos_earth_year", np.cos(2.0 * np.pi * t_days / 365.25636))

    add_vector("geo_pos_au", geo_pos)
    add_vector("geo_vel_au_d", geo_vel)
    add_vector("geo_unit_los", geo_unit)
    add_vector("geo_h_vec_au2_d", geo_h_vec_au2_d)
    add_vector("geo_accel_au_d2", geo_accel_au_d2)
    add_vector("geo_jerk_au_d3", geo_jerk_au_d3)
    add_vector("helio_pos_au", helio_pos)
    add_vector("helio_vel_au_d", helio_vel)
    add_vector("helio_unit", helio_unit)
    add_vector("helio_h_vec_au2_d", helio_h_vec_au2_d)
    add_vector("helio_accel_au_d2", helio_accel_au_d2)
    add_vector("helio_ecc_vec", helio_ecc_vec)

    for name, values in [
        ("geo_distance_au", dist_au),
        ("log_geo_distance_au", np.log10(np.maximum(dist_au, 1e-300))),
        ("ln_geo_distance_au", np.log(np.maximum(dist_au, 1e-300))),
        ("geo_speed_au_d", v_geo_au_d),
        ("geo_range_rate_au_d", range_rate_au_d),
        ("geo_transverse_speed_au_d", transverse_speed_au_d),
        ("geo_range_accel_au_d2", range_accel_au_d2),
        ("geo_range_jerk_au_d3", range_jerk_au_d3),
        ("geo_log_range_curvature_d2", log_range_curve_d2),
        ("geo_log_range_third_d3", log_range_third_d3),
        ("geo_speed_slope_au_d2", geo_speed_slope_au_d2),
        ("geo_ang_mom_au2_d", geo_ang_mom_au2_d),
        ("geo_accel_norm_au_d2", geo_accel_norm_au_d2),
        ("helio_r_au", r_helio_au),
        ("helio_speed_au_d", v_helio_au_d),
        ("helio_speed_slope_au_d2", helio_speed_slope_au_d2),
        ("helio_accel_norm_au_d2", helio_accel_norm_au_d2),
        ("helio_specific_energy_au2_d2", helio_energy_au2_d2),
        ("helio_specific_energy_m2_s2", energy),
        ("helio_eccentricity_inst", helio_ecc_inst),
        ("helio_sma_inst_au", helio_sma_inst_au),
        ("cos_geo_helio", cos_geo_helio),
        ("log_helio_j", _safe_log10_abs(j_helio_m2_s)),
        ("log_gi_n", _safe_log10_abs(gi_n)),
        ("sign_gi_n", np.sign(gi_n)),
        ("log_oi_n", _safe_log10_abs(oi_n)),
        ("sign_oi_n", np.sign(oi_n)),
        ("pdf_gamma", np.full(n, gamma)),
        ("pdf_eccentricity", np.full(n, ecc)),
        ("pdf_focal_distance_m", np.full(n, f_dist)),
        ("pdf_perihelion_m", np.full(n, jsun_obj)),
        ("pdf_delta_t_s", np.full(n, delta_t)),
    ]:
        add_feature(name, values)

    for name, values in pdf_arrays.items():
        add_feature(f"log_pdf_{name}", _safe_log10_abs(values))
        add_feature(f"sign_pdf_{name}", np.sign(values))

    X = np.column_stack(feature_columns)
    y = np.log10(np.maximum(dist_au, 1e-300))
    meta = {
        "jd": jd,
        "calendar": geo.calendar_tdb[:n],
        "dist_au": dist_au,
        "geo_pos_au": geo_pos,
        "geo_vel_au_d": geo_vel,
        "geo_unit_los": geo_unit,
        "geo_h_vec_au2_d": geo_h_vec_au2_d,
        "geo_ang_mom_au2_d": geo_ang_mom_au2_d,
        "geo_speed_au_d": v_geo_au_d,
        "geo_range_rate_au_d": range_rate_au_d,
        "geo_transverse_speed_au_d": transverse_speed_au_d,
        "geo_accel_au_d2": geo_accel_au_d2,
        "geo_jerk_au_d3": geo_jerk_au_d3,
        "geo_accel_norm_au_d2": geo_accel_norm_au_d2,
        "geo_range_accel_au_d2": range_accel_au_d2,
        "geo_range_jerk_au_d3": range_jerk_au_d3,
        "geo_log_range_curvature_d2": log_range_curve_d2,
        "geo_log_range_third_d3": log_range_third_d3,
        "geo_speed_slope_au_d2": geo_speed_slope_au_d2,
        "helio_pos_au": helio_pos,
        "helio_vel_au_d": helio_vel,
        "earth_helio_au": helio_pos - geo_pos,
        "helio_unit": helio_unit,
        "helio_h_vec_au2_d": helio_h_vec_au2_d,
        "helio_speed_au_d": v_helio_au_d,
        "helio_accel_au_d2": helio_accel_au_d2,
        "helio_accel_norm_au_d2": helio_accel_norm_au_d2,
        "helio_speed_slope_au_d2": helio_speed_slope_au_d2,
        "helio_specific_energy_au2_d2": helio_energy_au2_d2,
        "helio_specific_energy_m2_s2": energy,
        "helio_ecc_vec": helio_ecc_vec,
        "helio_eccentricity_inst": helio_ecc_inst,
        "helio_sma_inst_au": helio_sma_inst_au,
        "cos_geo_helio": cos_geo_helio,
        "gi_log": _safe_log10_abs(gi_n),
        "oi_log": _safe_log10_abs(oi_n),
        "r_helio_au": r_helio_au,
        "pdf_arrays": pdf_arrays,
    }
    return X, y, feature_names, meta


def _write_ml_plots(
    output_dir: Path,
    neo: NEOObject,
    meta: dict[str, Any],
    pred_log: Any,
    lo_log: Any,
    hi_log: Any,
    val_mask: Any,
    report_bits: dict[str, Any],
) -> tuple[list[str], list[str]]:
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    jd = meta["jd"]
    t_year = (jd - jd[0]) / 365.25636
    dist = meta["dist_au"]
    global_pred = 10.0 ** np.asarray(report_bits["global_pred_log"], dtype=float) if report_bits.get("global_pred_log") is not None else None
    pred = 10.0 ** pred_log
    lo = 10.0 ** lo_log
    hi = 10.0 ** hi_log

    figures: list[str] = []
    figure_catalog: list[dict[str, str]] = []
    caption_library = _ml_figure_caption_library()
    seconds_per_day = float(SECONDS_PER_DAY)
    au_d_to_km_s = AU_KM / seconds_per_day
    au_d2_to_mm_s2 = AU_M * 1.0e3 / (seconds_per_day * seconds_per_day)
    au2_d_to_gkm2_s = (AU_KM * AU_KM) / seconds_per_day / 1.0e9

    def _save_figure(fig: Any, filename: str, dpi: int = 220, tight: bool = True) -> None:
        path = output_dir / filename
        stem = path.stem
        if tight:
            fig.tight_layout()
        png_path = output_dir / f"{stem}.png"
        svg_path = output_dir / f"{stem}.svg"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.extend([str(png_path), str(svg_path), str(pdf_path)])
        spec = caption_library.get(stem)
        if spec is not None:
            title, caption = spec
            figure_catalog.append(
                {
                    "label": stem.replace("_", "-"),
                    "title": title,
                    "caption": caption,
                    "png_name": png_path.name,
                    "svg_name": svg_path.name,
                    "pdf_name": pdf_path.name,
                }
            )

    def _panel_label(ax: Any, label: str) -> None:
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.18", "alpha": 0.9},
        )

    cad_idx_all = [int(np.argmin(np.abs(jd - ca.jd_tdb))) for ca in neo.close_approaches]
    closest_cad_i = int(np.argmin([ca.distance_au for ca in neo.close_approaches])) if neo.close_approaches else 0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_year, dist, lw=1.8, color="#12355b", label="Horizons vector distance")
    if global_pred is not None:
        ax.plot(t_year, global_pred, lw=0.9, ls="--", color="#6c757d", label="global arc model")
    ax.plot(t_year, pred, lw=1.2, color="#d1495b", label="ML/DL PDF-feature surrogate")
    ax.fill_between(t_year, lo, hi, color="#d1495b", alpha=0.16, label="TensorFlow conformal band")
    for ca in neo.close_approaches:
        ca_idx = int(np.argmin(np.abs(jd - ca.jd_tdb)))
        ax.scatter(t_year[ca_idx], ca.distance_au, s=42, color="#edae49", edgecolor="black", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(f"Years from {meta['calendar'][0]}")
    ax.set_ylabel("Geocentric distance (au, log scale)")
    ax.set_title(f"{neo.fullname}: no-CAD-anchor ML surrogate vs Horizons vectors")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, "fig_ml_surrogate_distance.png", dpi=180)

    residual_km = (pred - dist) * AU_KM
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.scatter(t_year[val_mask], residual_km[val_mask], s=10, color="#00798c", alpha=0.75, label="blocked validation samples")
    ax.plot(t_year, residual_km, lw=0.6, color="#555555", alpha=0.45, label="all samples")
    ax.set_xlabel(f"Years from {meta['calendar'][0]}")
    ax.set_ylabel("ML minus Horizons distance (km)")
    ax.set_title(
        f"Validation residuals: MAE={report_bits['mae_km']:.0f} km, "
        f"RMSE={report_bits['rmse_km']:.0f} km"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, "fig_ml_validation_residuals.png", dpi=180)

    fig, ax = plt.subplots(figsize=(6, 6))
    true_val = dist[val_mask]
    pred_val = pred[val_mask]
    ax.scatter(true_val, pred_val, s=12, alpha=0.7, color="#2e86ab")
    mn = min(float(np.min(true_val)), float(np.min(pred_val)))
    mx = max(float(np.max(true_val)), float(np.max(pred_val)))
    ax.plot([mn, mx], [mn, mx], color="black", lw=1.0, label="perfect")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Horizons distance (au)")
    ax.set_ylabel("ML surrogate distance (au)")
    ax.set_title("Blocked validation parity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, "fig_ml_validation_parity.png", dpi=180)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t_year, dist, color="#12355b", lw=1.3, label="distance")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"Years from {meta['calendar'][0]}")
    ax1.set_ylabel("Geocentric distance (au)", color="#12355b")
    ax1.tick_params(axis="y", labelcolor="#12355b")
    ax2 = ax1.twinx()
    ax2.plot(t_year, meta["gi_log"], color="#d1495b", lw=0.9, label="log10 |GI_N|")
    ax2.plot(t_year, meta["oi_log"], color="#edae49", lw=0.9, label="log10 |OI_N|")
    ax2.set_ylabel("PDF feature log magnitude", color="#7a3b00")
    ax2.tick_params(axis="y", labelcolor="#7a3b00")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("PDF-derived feature dynamics vs geocentric distance")
    ax1.grid(True, which="both", alpha=0.25)
    _save_figure(fig, "fig_pdf_feature_dynamics.png", dpi=180)

    # Publication-grade orbital-physics overview. This figure keeps to directly
    # interpretable state-vector quantities and derived invariants that are
    # already used in the surrogate feature matrix.
    geo_speed_km_s = np.asarray(meta["geo_speed_au_d"], dtype=float) * au_d_to_km_s
    range_rate_km_s = np.asarray(meta["geo_range_rate_au_d"], dtype=float) * au_d_to_km_s
    transverse_km_s = np.asarray(meta["geo_transverse_speed_au_d"], dtype=float) * au_d_to_km_s
    helio_speed_km_s = np.asarray(meta["helio_speed_au_d"], dtype=float) * au_d_to_km_s
    geo_accel_mm_s2 = np.asarray(meta["geo_accel_norm_au_d2"], dtype=float) * au_d2_to_mm_s2
    helio_accel_mm_s2 = np.asarray(meta["helio_accel_norm_au_d2"], dtype=float) * au_d2_to_mm_s2
    range_accel_mm_s2 = np.asarray(meta["geo_range_accel_au_d2"], dtype=float) * au_d2_to_mm_s2
    speed_slope_mm_s2 = np.asarray(meta["geo_speed_slope_au_d2"], dtype=float) * au_d2_to_mm_s2
    geo_h_gkm2_s = np.asarray(meta["geo_ang_mom_au2_d"], dtype=float) * au2_d_to_gkm2_s
    helio_h_gkm2_s = np.linalg.norm(np.asarray(meta["helio_h_vec_au2_d"], dtype=float), axis=1) * au2_d_to_gkm2_s
    helio_ecc_inst = np.asarray(meta["helio_eccentricity_inst"], dtype=float)
    helio_energy_mj_kg = np.asarray(meta["helio_specific_energy_m2_s2"], dtype=float) / 1.0e6
    geo_xy_mkm = np.asarray(meta["geo_pos_au"], dtype=float)[:, :2] * AU_KM / 1.0e6
    cad_xy_mkm = geo_xy_mkm[cad_idx_all] if cad_idx_all else np.empty((0, 2))

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0), facecolor="white")
    time_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    ax = axes[0, 0]
    ax.plot(t_year, dist, color="#12355b", lw=1.7, label="geocentric distance")
    ax.plot(t_year, meta["r_helio_au"], color="#2a9d8f", lw=1.3, label="heliocentric distance")
    for i, ca in enumerate(neo.close_approaches):
        ax.scatter(t_year[cad_idx_all[i]], ca.distance_au, s=28, color="#edae49", edgecolor="black", zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel("Distance (au)")
    ax.set_title("Distance History")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "A")

    ax = axes[0, 1]
    ax.plot(t_year, geo_speed_km_s, color="#264653", lw=1.4, label="geocentric speed")
    ax.plot(t_year, np.abs(range_rate_km_s), color="#d1495b", lw=1.2, label="|range rate|")
    ax.plot(t_year, transverse_km_s, color="#f4a261", lw=1.2, label="transverse speed")
    ax.plot(t_year, helio_speed_km_s, color="#4c956c", lw=1.2, label="heliocentric speed")
    ax.set_ylabel("Speed (km s$^{-1}$)")
    ax.set_title("Velocity Decomposition")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "B")

    ax = axes[1, 0]
    ax.plot(t_year, geo_accel_mm_s2, color="#1d3557", lw=1.3, label="|geo acceleration|")
    ax.plot(t_year, np.abs(range_accel_mm_s2), color="#e76f51", lw=1.2, label="|range acceleration|")
    ax.plot(t_year, np.abs(speed_slope_mm_s2), color="#8d99ae", lw=1.1, label="|speed slope|")
    ax.plot(t_year, helio_accel_mm_s2, color="#2a9d8f", lw=1.1, label="|helio acceleration|")
    ax.set_yscale("log")
    ax.set_ylabel("Acceleration scale (mm s$^{-2}$)")
    ax.set_title("Acceleration Scales")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "C")

    ax = axes[1, 1]
    ax.plot(t_year, geo_h_gkm2_s, color="#003049", lw=1.3, label="geo specific angular momentum")
    ax.plot(t_year, helio_h_gkm2_s, color="#6a994e", lw=1.3, label="helio specific angular momentum")
    ax.set_ylabel(r"Specific angular momentum ($10^9$ km$^2$ s$^{-1}$)")
    ax.set_title("Orbital Invariants")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t_year, helio_ecc_inst, color="#d62828", lw=1.0, ls="--", label="instantaneous heliocentric eccentricity")
    ax2.set_ylabel("Instantaneous eccentricity", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    _panel_label(ax, "D")

    ax = axes[2, 0]
    ax.plot(geo_xy_mkm[:, 0], geo_xy_mkm[:, 1], color="#355070", lw=1.2, label="geocentric trajectory")
    ax.scatter([0.0], [0.0], s=75, color="#2a9df4", edgecolor="black", zorder=5, label="Earth")
    if len(cad_xy_mkm):
        ax.scatter(cad_xy_mkm[:, 0], cad_xy_mkm[:, 1], s=34, color="#ffb703", edgecolor="black", zorder=6, label="CAD epochs")
        ax.scatter([cad_xy_mkm[closest_cad_i, 0]], [cad_xy_mkm[closest_cad_i, 1]], s=120, facecolor="none", edgecolor="#d62828", linewidth=1.8, zorder=7, label="closest CAD epoch")
    ax.set_xlabel("Geocentric X (million km)")
    ax.set_ylabel("Geocentric Y (million km)")
    ax.set_title("Geocentric Ecliptic Projection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
    _panel_label(ax, "E")

    ax = axes[2, 1]
    ax.plot(dist * AU_KM / 1.0e6, range_rate_km_s, color="#3a506b", lw=1.2, label="trajectory")
    if neo.close_approaches:
        cad_phase_x = np.asarray([ca.distance_au * AU_KM / 1.0e6 for ca in neo.close_approaches], dtype=float)
        cad_phase_y = range_rate_km_s[cad_idx_all]
        ax.scatter(cad_phase_x, cad_phase_y, s=34, color="#ffb703", edgecolor="black", zorder=5, label="CAD epochs")
        ax.scatter([cad_phase_x[closest_cad_i]], [cad_phase_y[closest_cad_i]], s=120, facecolor="none", edgecolor="#d62828", linewidth=1.8, zorder=6)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Geocentric distance (million km)")
    ax.set_ylabel("Range rate (km s$^{-1}$)")
    ax.set_title("Distance-Rate Phase Portrait")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "F")

    for ax in time_axes:
        for i, _ in enumerate(neo.close_approaches):
            ax.axvline(t_year[cad_idx_all[i]], color="#999999", lw=0.55, alpha=0.15)
        if neo.close_approaches:
            ax.axvline(t_year[cad_idx_all[closest_cad_i]], color="#d62828", lw=0.9, ls="--", alpha=0.6)
        ax.set_xlabel(f"Years from {meta['calendar'][0]}")

    fig.suptitle("Orbital Physics Overview: State-Vector Dynamics and Derived Invariants", fontsize=15, y=0.995)
    _save_figure(fig, "fig_publication_orbital_physics.png", dpi=240)

    # Publication-grade hypothesis-parameter overview. The panel set keeps the
    # PDF sequence family explicit so the reader can see which proprietary terms
    # remain stable, which spike near close approaches, and how they compare to
    # GI_N / OI_N diagnostics.
    pdf = meta.get("pdf_arrays", {})

    def _pdf(name: str) -> Any:
        arr = pdf.get(name)
        if arr is None:
            return np.full_like(t_year, np.nan, dtype=float)
        return np.asarray(arr, dtype=float)

    def _logmag(name: str) -> Any:
        return np.log10(np.maximum(np.abs(_pdf(name)), 1e-300))

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0), facecolor="white")

    ax = axes[0, 0]
    js_low = np.maximum(np.abs(_pdf("jsuncritical_low")), 1e-300)
    js_mid = np.maximum(np.abs(_pdf("jsuncritical_median")), 1e-300)
    js_high = np.maximum(np.abs(_pdf("jsuncritical_high")), 1e-300)
    ax.fill_between(t_year, js_low, js_high, color="#cfe8f3", alpha=0.8, label="N=11-14 range")
    ax.plot(t_year, js_mid, color="#0b6e4f", lw=1.5, label="median $J_{sun,crit}$")
    ax.set_yscale("log")
    ax.set_ylabel("Magnitude")
    ax.set_title("$J_{sun,crit}$ Range")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "A")

    ax = axes[0, 1]
    ax.plot(t_year, np.maximum(np.abs(_pdf("time_norm_median")), 1e-300), color="#355070", lw=1.2, label="time_norm")
    ax.plot(t_year, np.maximum(np.abs(_pdf("time_cause")), 1e-300), color="#bc4749", lw=1.2, label="time_cause")
    ax.plot(t_year, np.maximum(np.abs(_pdf("time_slip_median")), 1e-300), color="#f4a261", lw=1.2, label="time_slip")
    ax.plot(t_year, np.maximum(np.abs(_pdf("lapse_factor_median")), 1e-300), color="#6a994e", lw=1.2, label="lapse_factor")
    ax.set_yscale("log")
    ax.set_ylabel("Magnitude")
    ax.set_title("Time-Scale Family")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "B")

    ax = axes[1, 0]
    ax.plot(t_year, np.maximum(np.abs(_pdf("trajectory_slip_median")), 1e-300), color="#5f0f40", lw=1.2, label="trajectory_slip")
    ax.plot(t_year, np.maximum(np.abs(_pdf("trajectory_precession_median")), 1e-300), color="#9a031e", lw=1.2, label="trajectory_precession")
    ax.plot(t_year, np.maximum(np.abs(_pdf("neo_phasing")), 1e-300), color="#0f4c5c", lw=1.2, label="neo_phasing")
    ax.plot(t_year, np.maximum(np.abs(_pdf("gravity_neo_phasing")), 1e-300), color="#fb8b24", lw=1.2, label="gravity_neo_phasing")
    ax.plot(t_year, np.maximum(np.abs(_pdf("bound_factor_median")), 1e-300), color="#6a994e", lw=1.2, label="bound_factor")
    ax.set_yscale("log")
    ax.set_ylabel("Magnitude")
    ax.set_title("Transport and Phasing Terms")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "C")

    ax = axes[1, 1]
    ax.plot(t_year, np.maximum(np.abs(_pdf("sequence_1")), 1e-300), color="#1d3557", lw=1.1, label="|sequence_1|")
    ax.plot(t_year, np.maximum(np.abs(_pdf("sequence_3")), 1e-300), color="#457b9d", lw=1.1, label="|sequence_3|")
    ax.plot(t_year, np.maximum(np.abs(_pdf("sequence_5")), 1e-300), color="#e76f51", lw=1.1, label="|sequence_5|")
    ax.plot(t_year, np.maximum(np.abs(_pdf("trajectory_loss")), 1e-300), color="#2a9d8f", lw=1.1, label="|trajectory_loss|")
    ax.plot(t_year, np.maximum(np.abs(_pdf("seqcr")), 1e-300), color="#6a4c93", lw=1.1, label="|seqcr|")
    ax.set_yscale("log")
    ax.set_ylabel("Magnitude")
    ax.set_title("Sequence Diagnostics")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "D")

    ax = axes[2, 0]
    ax.plot(t_year, _logmag("likelihood_proxy"), color="#d62828", lw=1.2, label="log10 |likelihood proxy|")
    ax.plot(t_year, _logmag("new_eccentricity_proxy"), color="#003049", lw=1.2, label="log10 |new eccentricity proxy|")
    ax.plot(t_year, _logmag("new_sma_au_proxy"), color="#588157", lw=1.2, label="log10 |new SMA proxy (au)|")
    ax.set_ylabel(r"$\log_{10}$ magnitude")
    ax.set_title("Outcome Proxies")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "E")

    ax = axes[2, 1]
    ax.plot(t_year, meta["gi_log"], color="#355070", lw=1.2, label="log10 |GI_N|")
    ax.plot(t_year, meta["oi_log"], color="#bc4749", lw=1.2, label="log10 |OI_N|")
    ax.plot(t_year, _logmag("scaled_jsuncritical"), color="#2a9d8f", lw=1.1, label="log10 |scaled Jsuncritical|")
    ax.plot(t_year, _logmag("scaled_time_norm"), color="#f4a261", lw=1.1, label="log10 |scaled time_norm|")
    ax.plot(t_year, _logmag("scaled_neo_phasing"), color="#6a994e", lw=1.1, label="log10 |scaled neo_phasing|")
    ax.set_ylabel(r"$\log_{10}$ magnitude")
    ax.set_title("GI/OI and Scaled PDF Diagnostics")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _panel_label(ax, "F")

    for ax in axes.flat:
        ax.set_xlabel(f"Years from {meta['calendar'][0]}")
        for i, _ in enumerate(neo.close_approaches):
            ax.axvline(t_year[cad_idx_all[i]], color="#999999", lw=0.55, alpha=0.12)
        if neo.close_approaches:
            ax.axvline(t_year[cad_idx_all[closest_cad_i]], color="#d62828", lw=0.9, ls="--", alpha=0.55)

    fig.suptitle("NEO Hypothesis Parameters: Publication-Style Diagnostic Overview", fontsize=15, y=0.995)
    _save_figure(fig, "fig_publication_hypothesis_parameters.png", dpi=240)

    diagnostics = report_bits.get("numerical_diagnostics", {})
    global_gate_rows = diagnostics.get("global_residual_blend_selection", [])
    local_gate_rows = diagnostics.get("local_blend_selection", [])
    if global_gate_rows or local_gate_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), facecolor="white")
        gate_specs = [
            (
                axes[0],
                "Global residual gate",
                global_gate_rows,
                float(diagnostics.get("global_residual_blend_strength", 0.0)),
                float(diagnostics.get("global_residual_gate_reference_rmse_km", float("nan"))),
                bool(diagnostics.get("global_residual_gate_accepted", False)),
            ),
            (
                axes[1],
                "Local encounter gate",
                local_gate_rows,
                float(diagnostics.get("local_blend_strength", 0.0)),
                float(diagnostics.get("local_gate_reference_rmse_km", float("nan"))),
                bool(diagnostics.get("local_gate_accepted", False)),
            ),
        ]
        for ax, title, rows, accepted_blend, reference_rmse, accepted in gate_specs:
            if rows:
                blends = np.asarray([float(row.get("blend_strength", np.nan)) for row in rows], dtype=float)
                rmse = np.asarray([float(row.get("rmse_km", np.nan)) for row in rows], dtype=float)
                valid = np.isfinite(blends) & np.isfinite(rmse)
                ax.plot(blends[valid], rmse[valid], marker="o", lw=1.5, color="#12355b", label="calibration RMSE")
                if np.isfinite(reference_rmse):
                    ax.axhline(reference_rmse, color="#6c757d", lw=1.0, ls="--", label="no-correction reference")
                if accepted_blend in blends[valid]:
                    accepted_rmse = float(rmse[valid][np.argmin(np.abs(blends[valid] - accepted_blend))])
                else:
                    accepted_rmse = reference_rmse
                ax.scatter(
                    [accepted_blend],
                    [accepted_rmse],
                    s=95,
                    color="#2a9d8f" if accepted else "#d1495b",
                    edgecolor="black",
                    zorder=5,
                    label="accepted blend" if accepted else "rejected: baseline retained",
                )
                center_metric = np.asarray([float(row.get("center_combined_km_mean", np.nan)) for row in rows], dtype=float)
                if np.any(np.isfinite(center_metric)):
                    ax2 = ax.twinx()
                    ax2.plot(blends, center_metric, marker="s", lw=1.0, color="#edae49", alpha=0.9, label="center objective")
                    ax2.set_ylabel("Encounter-center objective (km)", color="#8a5a00")
                    ax2.tick_params(axis="y", labelcolor="#8a5a00")
                    lines = ax.get_lines() + ax2.get_lines()
                    labels2 = [line.get_label() for line in lines]
                    handles, labels1 = ax.get_legend_handles_labels()
                    ax.legend(handles + [ax2.lines[-1]], labels1 + [labels2[-1]], loc="best", fontsize=8)
                else:
                    ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Blend strength")
            ax.set_ylabel("Calibration RMSE (km)")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        fig.suptitle("Calibration Gate Decisions for TensorFlow Residual Branches", fontsize=13, y=0.99)
        _save_figure(fig, "fig_tensorflow_gate_blend_selection.png", dpi=220)

    baseline_log_for_gate = report_bits.get("global_physics_baseline_log")
    proposal_log_for_gate = report_bits.get("global_residual_full_log")
    local_proposal_log_for_gate = report_bits.get("local_proposal_log")
    if baseline_log_for_gate is not None and proposal_log_for_gate is not None:
        baseline_au = 10.0 ** np.asarray(baseline_log_for_gate, dtype=float)
        proposal_au = 10.0 ** np.asarray(proposal_log_for_gate, dtype=float)
        final_au = np.asarray(pred, dtype=float)
        proposal_correction_km = (proposal_au - baseline_au) * AU_KM
        accepted_correction_km = (final_au - baseline_au) * AU_KM
        fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.6), sharex=True, facecolor="white")
        axes[0].axhline(0.0, color="black", lw=0.8)
        axes[0].plot(t_year, proposal_correction_km, color="#d1495b", lw=0.9, alpha=0.8, label="full TensorFlow global proposal")
        axes[0].plot(t_year, accepted_correction_km, color="#12355b", lw=1.3, label="accepted post-gate correction")
        if local_proposal_log_for_gate is not None:
            local_proposal_au = 10.0 ** np.asarray(local_proposal_log_for_gate, dtype=float)
            local_correction_km = (local_proposal_au - baseline_au) * AU_KM
            axes[0].plot(t_year, local_correction_km, color="#edae49", lw=0.8, alpha=0.75, label="local proposal against baseline")
        for idx in cad_idx_all:
            axes[0].axvline(t_year[idx], color="#999999", lw=0.55, alpha=0.14)
        axes[0].set_ylabel("Correction relative to state-vector baseline (km)")
        axes[0].set_title("TensorFlow Proposed Corrections Versus Accepted Gated Correction")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)

        abs_floor = 1.0
        axes[1].plot(t_year, np.log10(np.maximum(np.abs(proposal_correction_km), abs_floor)), color="#d1495b", lw=0.9, label="|full proposal|")
        axes[1].plot(t_year, np.log10(np.maximum(np.abs(accepted_correction_km), abs_floor)), color="#12355b", lw=1.2, label="|accepted correction|")
        axes[1].fill_between(
            t_year,
            0.0,
            np.asarray(report_bits.get("locality_score", np.zeros_like(t_year)), dtype=float),
            color="#2a9d8f",
            alpha=0.14,
            transform=axes[1].get_xaxis_transform(),
            label="locality support",
        )
        axes[1].set_xlabel(f"Years from {meta['calendar'][0]}")
        axes[1].set_ylabel(r"$\log_{10}(|correction| + 1\,km)$")
        axes[1].set_title("Correction Magnitude and Encounter Locality")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle("TensorFlow Residual Field Under Calibration Gating", fontsize=13, y=0.99)
        _save_figure(fig, "fig_tensorflow_correction_field.png", dpi=220)

    anchor_rows = report_bits.get("anchor_rows", [])
    if anchor_rows:
        labels = [str(row["cad_date_tdb"]).split()[0] for row in anchor_rows]
        cad_km = np.asarray([row["cad_distance_km"] for row in anchor_rows], dtype=float)
        hor_km = np.asarray([row["horizons_sample_distance_km"] for row in anchor_rows], dtype=float)
        ml_km = np.asarray([row["ml_predicted_distance_km"] for row in anchor_rows], dtype=float)
        ml_resid = np.asarray([row["ml_minus_cad_km"] for row in anchor_rows], dtype=float)
        hor_resid = np.asarray([row["horizons_sample_minus_cad_km"] for row in anchor_rows], dtype=float)
        tf_cont_resid = np.asarray([row.get("tensorflow_continuous_minus_cad_km", np.nan) for row in anchor_rows], dtype=float)
        lo_km = np.asarray([row["conformal_lo_km"] for row in anchor_rows], dtype=float)
        hi_km = np.asarray([row["conformal_hi_km"] for row in anchor_rows], dtype=float)
        x = np.arange(len(anchor_rows))

        focus_anchor_index = int(np.argmin(cad_km))
        focus_anchor = anchor_rows[focus_anchor_index]
        focus_jd = float(focus_anchor["cad_jd_tdb"])
        focus_sample_idx = int(np.argmin(np.abs(jd - focus_jd)))
        encounter_window_days = max(float(report_bits.get("zoom_days", 8.0)) * 0.55, 3.0)
        encounter_mask = np.abs(jd - focus_jd) <= encounter_window_days
        if int(np.count_nonzero(encounter_mask)) < 12:
            encounter_window_days = max(float(report_bits.get("zoom_days", 8.0)), 5.0)
            encounter_mask = np.abs(jd - focus_jd) <= encounter_window_days
        encounter_idx = np.where(encounter_mask)[0]
        if len(encounter_idx) > 700:
            keep = np.linspace(0, len(encounter_idx) - 1, 700).astype(int)
            encounter_idx = encounter_idx[keep]

        geo_pos_km = np.asarray(meta["geo_pos_au"], dtype=float) * AU_KM
        encounter_pos_kkm = geo_pos_km[encounter_idx] / 1.0e3
        encounter_hours = (jd[encounter_idx] - focus_jd) * 24.0
        encounter_dist_km = np.linalg.norm(geo_pos_km[encounter_idx], axis=1)
        interpolated_pos_km = np.asarray(
            [
                _local_poly_predict_value(jd, geo_pos_km[:, dim], focus_jd, min_points=9, max_points=17, max_degree=5)
                for dim in range(3)
            ],
            dtype=float,
        )
        los_norm = float(np.linalg.norm(interpolated_pos_km))
        if not math.isfinite(los_norm) or los_norm <= 0.0:
            los_vector = geo_pos_km[focus_sample_idx] / max(float(np.linalg.norm(geo_pos_km[focus_sample_idx])), 1e-300)
        else:
            los_vector = interpolated_pos_km / los_norm
        cad_point_km = los_vector * float(focus_anchor["cad_distance_km"])
        tf_cont_distance_km = float(focus_anchor.get("tensorflow_continuous_distance_km", float("nan")))
        tf_cont_point_km = los_vector * tf_cont_distance_km if math.isfinite(tf_cont_distance_km) else np.full(3, np.nan)
        sample_point_km = geo_pos_km[focus_sample_idx]
        interp_point_km = interpolated_pos_km
        cad_radius_kkm = float(focus_anchor["cad_distance_km"]) / 1.0e3
        tf_radius_kkm = tf_cont_distance_km / 1.0e3 if math.isfinite(tf_cont_distance_km) else float("nan")
        earth_radius_kkm = 6371.0088 / 1.0e3
        geo_radius_kkm = 42164.0 / 1.0e3

        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        phi = np.linspace(0.0, np.pi, 32)
        sphere_x = np.outer(np.cos(theta), np.sin(phi))
        sphere_y = np.outer(np.sin(theta), np.sin(phi))
        sphere_z = np.outer(np.ones_like(theta), np.cos(phi))
        ring_theta = np.linspace(0.0, 2.0 * np.pi, 180)

        fig = plt.figure(figsize=(15.5, 9.0), facecolor="white")
        gs = fig.add_gridspec(2, 3, width_ratios=[1.65, 1.0, 1.0], height_ratios=[1.0, 0.82], wspace=0.25, hspace=0.32)
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax3d.plot(
            encounter_pos_kkm[:, 0],
            encounter_pos_kkm[:, 1],
            encounter_pos_kkm[:, 2],
            color="#12355b",
            lw=1.8,
            label="JPL Horizons VECTORS trajectory",
        )
        sc = ax3d.scatter(
            encounter_pos_kkm[:, 0],
            encounter_pos_kkm[:, 1],
            encounter_pos_kkm[:, 2],
            c=encounter_hours,
            cmap="viridis",
            s=12,
            alpha=0.72,
            depthshade=False,
            label="sampled epochs",
        )
        ax3d.plot_surface(
            earth_radius_kkm * sphere_x,
            earth_radius_kkm * sphere_y,
            earth_radius_kkm * sphere_z,
            color="#2a9df4",
            alpha=0.72,
            linewidth=0,
            shade=True,
        )
        ax3d.plot(
            geo_radius_kkm * np.cos(ring_theta),
            geo_radius_kkm * np.sin(ring_theta),
            np.zeros_like(ring_theta),
            color="#8d99ae",
            lw=0.9,
            ls="--",
            alpha=0.75,
            label="GEO radius reference",
        )
        ax3d.plot_wireframe(
            cad_radius_kkm * sphere_x,
            cad_radius_kkm * sphere_y,
            cad_radius_kkm * sphere_z,
            color="#d1495b",
            alpha=0.18,
            linewidth=0.35,
        )
        if math.isfinite(tf_radius_kkm):
            ax3d.plot_wireframe(
                tf_radius_kkm * sphere_x,
                tf_radius_kkm * sphere_y,
                tf_radius_kkm * sphere_z,
                color="#2a9d8f",
                alpha=0.13,
                linewidth=0.28,
            )
        marker_specs = [
            ("nearest Horizons sample", sample_point_km / 1.0e3, "#edae49", 62, "o"),
            ("Horizons local interpolation at CAD epoch", interp_point_km / 1.0e3, "#111111", 72, "^"),
            ("JPL CAD range on Horizons line of sight", cad_point_km / 1.0e3, "#d1495b", 92, "s"),
            ("TensorFlow continuous encounter", tf_cont_point_km / 1.0e3, "#2a9d8f", 112, "*"),
        ]
        for label, point, color, size, marker in marker_specs:
            if np.all(np.isfinite(point)):
                ax3d.scatter([point[0]], [point[1]], [point[2]], s=size, color=color, marker=marker, edgecolor="black", linewidth=0.6, depthshade=False, label=label)
                ax3d.plot([0.0, point[0]], [0.0, point[1]], [0.0, point[2]], color=color, lw=0.85, alpha=0.55)
        lim = float(np.nanmax(np.abs(encounter_pos_kkm))) if len(encounter_pos_kkm) else cad_radius_kkm
        lim = max(lim, cad_radius_kkm * 1.18, geo_radius_kkm * 1.18)
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_xlabel("Geocentric X (10^3 km)")
        ax3d.set_ylabel("Geocentric Y (10^3 km)")
        ax3d.set_zlabel("Geocentric Z (10^3 km)")
        ax3d.set_title("3-D Encounter Geometry")
        ax3d.grid(True, alpha=0.22)
        ax3d.view_init(elev=22, azim=-48)
        try:
            ax3d.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        ax3d.legend(loc="upper left", fontsize=7.5)
        cbar = fig.colorbar(sc, ax=ax3d, shrink=0.58, pad=0.02)
        cbar.set_label("Hours from CAD epoch")

        ax_range = fig.add_subplot(gs[0, 1:])
        ax_range.plot(encounter_hours, encounter_dist_km, color="#12355b", lw=1.8, label="Horizons range")
        ax_range.axvline(0.0, color="black", lw=0.9, ls="--", label="CAD epoch")
        ax_range.axhline(float(focus_anchor["cad_distance_km"]), color="#d1495b", lw=1.2, label="JPL CAD range")
        if math.isfinite(tf_cont_distance_km):
            ax_range.axhline(tf_cont_distance_km, color="#2a9d8f", lw=1.2, label="TensorFlow continuous range")
        ax_range.scatter([float(focus_anchor["sample_time_offset_hours"])], [float(focus_anchor["horizons_sample_distance_km"])], s=58, color="#edae49", edgecolor="black", zorder=5, label="nearest sample")
        ax_range.set_xlabel("Hours from CAD epoch")
        ax_range.set_ylabel("Geocentric range (km)")
        ax_range.set_title("Encounter Range History")
        ax_range.grid(True, alpha=0.25)
        ax_range.legend(loc="best", fontsize=8)

        ax_resid = fig.add_subplot(gs[1, 1])
        residual_labels = ["sample", "interp", "TF continuous"]
        residual_values = [
            float(focus_anchor["ml_minus_cad_km"]),
            float(focus_anchor["ml_interpolated_minus_cad_km"]),
            float(focus_anchor.get("tensorflow_continuous_minus_cad_km", float("nan"))),
        ]
        residual_colors = ["#edae49", "#111111", "#2a9d8f"]
        ax_resid.axhline(0.0, color="black", lw=0.8)
        ax_resid.bar(residual_labels, residual_values, color=residual_colors, alpha=0.9)
        ax_resid.set_ylabel("Residual vs JPL CAD (km)")
        ax_resid.set_title("CAD-Epoch Range Error")
        ax_resid.grid(True, axis="y", alpha=0.25)

        ax_table = fig.add_subplot(gs[1, 2])
        ax_table.axis("off")
        table_rows = [
            ("CAD epoch", str(focus_anchor["cad_date_tdb"])),
            ("CAD range", f"{float(focus_anchor['cad_distance_km']):,.1f} km"),
            ("Nearest sample offset", f"{float(focus_anchor['sample_time_offset_hours']):.3f} h"),
            ("Sample - CAD", f"{float(focus_anchor['ml_minus_cad_km']):,.1f} km"),
            ("Interpolated - CAD", f"{float(focus_anchor['ml_interpolated_minus_cad_km']):,.1f} km"),
            ("TF continuous - CAD", f"{float(focus_anchor.get('tensorflow_continuous_minus_cad_km', float('nan'))):,.1f} km"),
            ("TF model", str(focus_anchor.get("tensorflow_continuous_model", ""))),
            ("TF local CV RMSE", f"{float(focus_anchor.get('tensorflow_continuous_cv_rmse_km', float('nan'))):,.1f} km"),
        ]
        table = ax_table.table(cellText=table_rows, colLabels=["Quantity", "Value"], loc="center", cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.0, 1.2)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if key[0] == 0:
                cell.set_facecolor("#f1f3f5")
                cell.set_text_props(weight="bold")
        ax_table.set_title("Comparison Against Online JPL Anchors", pad=6)

        fig.suptitle(f"{neo.fullname}: 3-D Encounter Model Compared With JPL Horizons and CAD", fontsize=14, y=0.985)
        fig.text(
            0.045,
            0.035,
            "CAD provides authoritative close-approach range/time; Horizons VECTORS provide the geocentric state-vector path. CAD and TensorFlow range markers are placed on the Horizons line of sight at the CAD epoch.",
            fontsize=8.2,
            color="#444444",
        )
        fig.subplots_adjust(left=0.045, right=0.985, top=0.92, bottom=0.09)
        _save_figure(fig, "fig_3d_encounter_model.png", dpi=230, tight=False)

        try:
            import json

            def _series(values: Any) -> list[float]:
                return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]

            sphere_stride = 3
            html_payload = {
                "title": f"{neo.fullname}: 3-D Encounter Model",
                "cadDate": str(focus_anchor["cad_date_tdb"]),
                "sourceText": "State vectors: JPL Horizons VECTORS. Close-approach anchor: JPL SBDB CAD API.",
                "trajectory": {
                    "x": _series(encounter_pos_kkm[:, 0]),
                    "y": _series(encounter_pos_kkm[:, 1]),
                    "z": _series(encounter_pos_kkm[:, 2]),
                    "hours": _series(encounter_hours),
                    "rangeKm": _series(encounter_dist_km),
                },
                "markers": [
                    {"name": label, "x": float(point[0]), "y": float(point[1]), "z": float(point[2]), "color": color, "size": 6 if marker != "*" else 9}
                    for label, point, color, _size, marker in marker_specs
                    if np.all(np.isfinite(point))
                ],
                "earth": {
                    "x": (earth_radius_kkm * sphere_x[::sphere_stride, ::sphere_stride]).tolist(),
                    "y": (earth_radius_kkm * sphere_y[::sphere_stride, ::sphere_stride]).tolist(),
                    "z": (earth_radius_kkm * sphere_z[::sphere_stride, ::sphere_stride]).tolist(),
                },
                "cadShell": {
                    "x": (cad_radius_kkm * sphere_x[::sphere_stride, ::sphere_stride]).tolist(),
                    "y": (cad_radius_kkm * sphere_y[::sphere_stride, ::sphere_stride]).tolist(),
                    "z": (cad_radius_kkm * sphere_z[::sphere_stride, ::sphere_stride]).tolist(),
                },
                "tfShell": {
                    "x": (tf_radius_kkm * sphere_x[::sphere_stride, ::sphere_stride]).tolist() if math.isfinite(tf_radius_kkm) else [],
                    "y": (tf_radius_kkm * sphere_y[::sphere_stride, ::sphere_stride]).tolist() if math.isfinite(tf_radius_kkm) else [],
                    "z": (tf_radius_kkm * sphere_z[::sphere_stride, ::sphere_stride]).tolist() if math.isfinite(tf_radius_kkm) else [],
                },
                "geoRing": {
                    "x": _series(geo_radius_kkm * np.cos(ring_theta)),
                    "y": _series(geo_radius_kkm * np.sin(ring_theta)),
                    "z": _series(np.zeros_like(ring_theta)),
                },
                "summary": table_rows,
            }
            html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{neo.fullname} 3-D Encounter Model</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #f6f8fa; }}
    header {{ padding: 18px 24px 8px; background: white; border-bottom: 1px solid #d9dee5; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 4px 0; line-height: 1.35; }}
    #wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: calc(100vh - 92px); }}
    #plot {{ min-height: 780px; }}
    aside {{ padding: 18px; background: white; border-left: 1px solid #d9dee5; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    td, th {{ border-bottom: 1px solid #e5e8ec; padding: 7px 4px; text-align: left; vertical-align: top; }}
    th {{ color: #445; }}
    .note {{ color: #4b5563; font-size: 13px; }}
  </style>
</head>
<body>
<header>
  <h1>{neo.fullname}: 3-D Encounter Model</h1>
  <p class="note">Comparison sources: JPL Horizons VECTORS trajectory and JPL SBDB CAD close-approach range/time. CAD and TensorFlow markers are range placements along the Horizons line of sight at the CAD epoch.</p>
</header>
<div id="wrap">
  <div id="plot"></div>
  <aside>
    <h2>Encounter Summary</h2>
    <table id="summary"></table>
    <p class="note">Units in the 3-D scene are thousands of kilometers from Earth center. Toggle traces in the legend to isolate the path, CAD shell, or TensorFlow reconstruction.</p>
  </aside>
</div>
<script>
const payload = {json.dumps(html_payload)};
const traces = [
  {{
    type: "scatter3d", mode: "lines+markers", name: "JPL Horizons trajectory",
    x: payload.trajectory.x, y: payload.trajectory.y, z: payload.trajectory.z,
    line: {{color: "#12355b", width: 5}},
    marker: {{size: 2.5, color: payload.trajectory.hours, colorscale: "Viridis", colorbar: {{title: "hours from CAD"}}}},
    text: payload.trajectory.hours.map((h, i) => `t=${{h.toFixed(2)}} h<br>range=${{payload.trajectory.rangeKm[i].toLocaleString(undefined, {{maximumFractionDigits: 1}})}} km`),
    hovertemplate: "%{{text}}<extra></extra>"
  }},
  {{
    type: "surface", name: "Earth", x: payload.earth.x, y: payload.earth.y, z: payload.earth.z,
    opacity: 0.72, colorscale: [[0, "#2a9df4"], [1, "#2a9df4"]], showscale: false, hoverinfo: "skip"
  }},
  {{
    type: "surface", name: "JPL CAD range shell", x: payload.cadShell.x, y: payload.cadShell.y, z: payload.cadShell.z,
    opacity: 0.18, colorscale: [[0, "#d1495b"], [1, "#d1495b"]], showscale: false, hoverinfo: "skip"
  }},
  {{
    type: "surface", name: "TensorFlow range shell", x: payload.tfShell.x, y: payload.tfShell.y, z: payload.tfShell.z,
    opacity: 0.15, colorscale: [[0, "#2a9d8f"], [1, "#2a9d8f"]], showscale: false, hoverinfo: "skip", visible: payload.tfShell.x.length ? true : "legendonly"
  }},
  {{
    type: "scatter3d", mode: "lines", name: "GEO radius reference",
    x: payload.geoRing.x, y: payload.geoRing.y, z: payload.geoRing.z,
    line: {{color: "#8d99ae", width: 2, dash: "dash"}}, hoverinfo: "skip"
  }},
  ...payload.markers.map(m => ({{
    type: "scatter3d", mode: "markers+text", name: m.name,
    x: [m.x], y: [m.y], z: [m.z],
    marker: {{size: m.size, color: m.color, line: {{color: "#111", width: 1}}}},
    text: [m.name], textposition: "top center",
    hovertemplate: `${{m.name}}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>`
  }}))
];
const layout = {{
  paper_bgcolor: "#f6f8fa", plot_bgcolor: "#f6f8fa",
  margin: {{l: 0, r: 0, t: 20, b: 0}},
  legend: {{x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.82)", bordercolor: "#d9dee5", borderwidth: 1}},
  scene: {{
    xaxis: {{title: "X (10^3 km)", zerolinecolor: "#aeb6c2", gridcolor: "#d9dee5"}},
    yaxis: {{title: "Y (10^3 km)", zerolinecolor: "#aeb6c2", gridcolor: "#d9dee5"}},
    zaxis: {{title: "Z (10^3 km)", zerolinecolor: "#aeb6c2", gridcolor: "#d9dee5"}},
    aspectmode: "data",
    camera: {{eye: {{x: 1.55, y: -1.75, z: 1.1}}}}
  }}
}};
Plotly.newPlot("plot", traces, layout, {{responsive: true, displaylogo: false}});
document.getElementById("summary").innerHTML = payload.summary.map(row => `<tr><th>${{row[0]}}</th><td>${{row[1]}}</td></tr>`).join("");
</script>
</body>
</html>
"""
            html_path = output_dir / "fig_3d_encounter_model.html"
            html_path.write_text(html, encoding="utf-8")
            figures.append(str(html_path))
        except Exception:
            pass

        if proposal_log_for_gate is not None:
            proposal_log_arr = np.asarray(proposal_log_for_gate, dtype=float)
            proposal_resid = []
            for row in anchor_rows:
                sample_jd = float(row.get("horizons_sample_jd_tdb", row.get("cad_jd_tdb", float("nan"))))
                idx = int(np.argmin(np.abs(jd - sample_jd)))
                proposal_resid.append(float((10.0 ** proposal_log_arr[idx]) * AU_KM - float(row["cad_distance_km"])))
            proposal_resid_arr = np.asarray(proposal_resid, dtype=float)
            fig, ax = plt.subplots(figsize=(12.8, 5.6), facecolor="white")
            width = 0.20
            ax.axhline(0.0, color="black", lw=0.85)
            ax.bar(x - 1.5 * width, hor_resid, width=width, color="#00798c", label="nearest Horizons sample - CAD")
            ax.bar(x - 0.5 * width, proposal_resid_arr, width=width, color="#edae49", label="full TensorFlow residual proposal - CAD")
            ax.bar(x + 0.5 * width, ml_resid, width=width, color="#d1495b", label="accepted gated surrogate - CAD")
            ax.bar(x + 1.5 * width, tf_cont_resid, width=width, color="#2a9d8f", label="TensorFlow continuous encounter - CAD")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("Residual against CAD anchor (km)")
            ax.set_title("CAD Anchor Synthesis: Sampling, Neural Proposal, and Accepted Gate Output")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            _save_figure(fig, "fig_tensorflow_anchor_synthesis.png", dpi=220)

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.26
        ax.bar(x - width, cad_km, width=width, color="#12355b", label="CAD anchor")
        ax.bar(x, hor_km, width=width, color="#00798c", label="nearest Horizons sample")
        ax.bar(x + width, ml_km, width=width, color="#d1495b", label="surrogate prediction")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Earth approach distance (km, log scale)")
        ax.set_title("CAD anchor distances vs Horizons samples vs surrogate predictions")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(loc="best")
        _save_figure(fig, "fig_anchor_distance_comparison.png", dpi=220)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(0.0, color="black", lw=0.8)
        ax.bar(x - 0.18, hor_resid, width=0.36, color="#00798c", label="Horizons sample - CAD")
        ax.bar(x + 0.18, ml_resid, width=0.36, color="#d1495b", label="surrogate - CAD")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Residual against CAD anchor (km)")
        ax.set_title("Anchor residuals: sampling error and surrogate error")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
        _save_figure(fig, "fig_anchor_residual_bars.png", dpi=220)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(cad_km, hor_km, s=54, color="#00798c", label="nearest Horizons sample")
        ax.scatter(cad_km, ml_km, s=54, color="#d1495b", label="surrogate")
        mn = min(float(np.min(cad_km)), float(np.min(hor_km)), float(np.min(ml_km)))
        mx = max(float(np.max(cad_km)), float(np.max(hor_km)), float(np.max(ml_km)))
        ax.plot([mn, mx], [mn, mx], color="black", lw=1.0, label="perfect agreement")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("CAD anchor distance (km)")
        ax.set_ylabel("Predicted/sampled distance (km)")
        ax.set_title("Anchor parity plot")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        _save_figure(fig, "fig_anchor_parity.png", dpi=220)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.errorbar(
            x,
            ml_km,
            yerr=np.vstack([np.maximum(ml_km - lo_km, 0.0), np.maximum(hi_km - ml_km, 0.0)]),
            fmt="o",
            color="#d1495b",
            ecolor="#d1495b",
            elinewidth=1.0,
            capsize=3,
            label="surrogate + conformal band",
        )
        ax.scatter(x, cad_km, color="#12355b", s=48, zorder=4, label="CAD anchor")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Distance (km, log scale)")
        ax.set_title("CAD anchors against surrogate conformal uncertainty")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(loc="best")
        _save_figure(fig, "fig_anchor_conformal_intervals.png", dpi=220)

        # 3-D residual trajectory in diagnostic phase space. CAD close-approach
        # rows are scalar range anchors, not full state vectors; plotting them
        # as a physical 3-D orbit creates false geometry and poor scale. This
        # view instead makes the comparison explicit: epoch, CAD range, and
        # surrogate residual form the three axes, with stems to the zero-error
        # plane for immediate visual calibration.
        anchor_jd = np.asarray([float(row["cad_jd_tdb"]) for row in anchor_rows], dtype=float)
        anchor_year = (anchor_jd - jd[0]) / 365.25636
        cad_log_km = np.log10(np.maximum(cad_km, 1.0))
        ca_focus = int(np.argmin(cad_km))
        focus_label = str(anchor_rows[ca_focus]["cad_date_tdb"])
        focus_resid = float(ml_resid[ca_focus])
        focus_frac_ppm = 1.0e6 * focus_resid / max(float(cad_km[ca_focus]), 1.0)

        chrono = np.argsort(anchor_year)
        x3 = anchor_year[chrono]
        y3 = cad_log_km[chrono]
        z3 = ml_resid[chrono]
        max_abs_resid = max(float(np.nanmax(np.abs(ml_resid))), 1.0)
        labels_short = [str(row["cad_date_tdb"]).split()[0] for row in anchor_rows]

        fig = plt.figure(figsize=(14.8, 7.8), facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.78, 1.0], height_ratios=[1.0, 0.92], wspace=0.20, hspace=0.36)
        ax = fig.add_subplot(gs[:, 0], projection="3d")
        ax.plot(x3, y3, z3, color="#243b53", lw=1.7, alpha=0.88, label="chronological residual path")
        for i in chrono:
            stem_color = "#c55a3d" if ml_resid[i] >= 0 else "#287c71"
            ax.plot(
                [anchor_year[i], anchor_year[i]],
                [cad_log_km[i], cad_log_km[i]],
                [0.0, ml_resid[i]],
                color=stem_color,
                lw=1.4,
                alpha=0.72,
            )
        ax.scatter(
            anchor_year,
            cad_log_km,
            ml_resid,
            c=ml_resid,
            cmap="coolwarm",
            vmin=-max_abs_resid,
            vmax=max_abs_resid,
            s=80,
            edgecolor="black",
            linewidth=0.6,
            depthshade=False,
            label="CAD validation anchors",
        )
        ax.scatter(
            [anchor_year[ca_focus]],
            [cad_log_km[ca_focus]],
            [ml_resid[ca_focus]],
            s=210,
            facecolor="none",
            edgecolor="#f2c94c",
            linewidth=2.2,
            depthshade=False,
            label="closest CAD anchor",
        )
        ax.plot(
            [float(np.nanmin(anchor_year)) - 0.2, float(np.nanmax(anchor_year)) + 0.2],
            [float(cad_log_km[ca_focus]), float(cad_log_km[ca_focus])],
            [0.0, 0.0],
            color="#8d99ae",
            lw=1.0,
            ls="--",
            alpha=0.8,
        )
        ax.text(
            anchor_year[ca_focus],
            cad_log_km[ca_focus],
            ml_resid[ca_focus] + 0.08 * max_abs_resid,
            f"closest\n{focus_resid:,.0f} km",
            fontsize=8,
            ha="center",
        )
        ax.set_xlabel(f"Years from {meta['calendar'][0]}", labelpad=10)
        ax.set_ylabel("log10(CAD distance [km])", labelpad=10)
        ax.set_zlabel("ML - CAD residual (km)", labelpad=10)
        ax.set_zlim(-1.18 * max_abs_resid, 1.18 * max_abs_resid)
        ax.set_ylim(float(np.nanmin(cad_log_km)) - 0.18, float(np.nanmax(cad_log_km)) + 0.18)
        ax.set_title("3-D Residual Trajectory in CAD Anchor Space", pad=10)
        ax.grid(True, alpha=0.25)
        ax.view_init(elev=24, azim=-56)
        try:
            ax.set_box_aspect((1.35, 1.0, 0.72))
        except Exception:
            pass
        ax.legend(loc="upper left", fontsize=7.5, frameon=True)

        axr = fig.add_subplot(gs[0, 1])
        order = np.arange(len(anchor_rows))
        colors = np.where(ml_resid >= 0, "#c55a3d", "#287c71")
        axr.axhline(0.0, color="black", lw=0.9)
        axr.bar(order, ml_resid, color=colors, alpha=0.88)
        axr.scatter([ca_focus], [ml_resid[ca_focus]], s=120, facecolor="none", edgecolor="#f2c94c", linewidth=2.0, zorder=5)
        axr.set_xticks(order)
        axr.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
        axr.set_ylabel("ML - CAD (km)")
        axr.set_title("Signed Residuals at CAD Anchors", pad=8)
        axr.set_ylim(-1.18 * max_abs_resid, 1.18 * max_abs_resid)
        axr.grid(True, axis="y", alpha=0.25)

        axt = fig.add_subplot(gs[1, 1])
        axt.axis("off")
        summary_rows = [
            ("CAD epoch", focus_label),
            ("CAD distance", f"{cad_km[ca_focus]:,.0f} km"),
            ("Surrogate distance", f"{ml_km[ca_focus]:,.0f} km"),
            ("ML - CAD", f"{focus_resid:,.0f} km"),
            ("Fractional residual", f"{focus_frac_ppm:,.0f} ppm"),
            ("Nearest Horizons - CAD", f"{hor_resid[ca_focus]:,.0f} km"),
            ("Sample offset", f"{anchor_rows[ca_focus]['sample_time_offset_hours']:.2f} h"),
        ]
        table = axt.table(cellText=summary_rows, colLabels=["Quantity", "Value"], loc="center", cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.0, 1.18)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if key[0] == 0:
                cell.set_facecolor("#f1f3f5")
                cell.set_text_props(weight="bold")
        axt.set_title("Closest Anchor Details", pad=6)

        fig.suptitle("Surrogate Residual Against CAD Anchors", fontsize=13, y=0.955)
        fig.text(
            0.045,
            0.045,
            "Diagnostic phase-space view: each point is a JPL CAD close-approach anchor; the vertical coordinate is the no-CAD-trained surrogate residual.",
            fontsize=8.0,
            color="#444444",
        )
        fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.11)
        _save_figure(fig, "fig_3d_surrogate_residual_vs_cad.png", dpi=220, tight=False)

    # Close-approach zoom around the Horizons minimum.
    min_idx = int(np.argmin(dist))
    zoom_mask = np.abs(jd - jd[min_idx]) <= report_bits.get("zoom_days", 12.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_year[zoom_mask], dist[zoom_mask] * AU_KM, lw=2.0, color="#12355b", label="Horizons vector distance")
    if global_pred is not None:
        ax.plot(
            t_year[zoom_mask],
            global_pred[zoom_mask] * AU_KM,
            lw=1.0,
            ls="--",
            color="#6c757d",
            label="global arc model",
        )
    ax.plot(t_year[zoom_mask], pred[zoom_mask] * AU_KM, lw=1.4, color="#d1495b", label="physics-informed ML surrogate")
    ax.fill_between(t_year[zoom_mask], lo[zoom_mask] * AU_KM, hi[zoom_mask] * AU_KM, color="#d1495b", alpha=0.16, label="90% conformal band")
    for ca in neo.close_approaches:
        ca_idx = int(np.argmin(np.abs(jd - ca.jd_tdb)))
        if zoom_mask[ca_idx]:
            ax.scatter(t_year[ca_idx], ca.distance_au * AU_KM, s=54, color="#edae49", edgecolor="black", zorder=5, label="CAD overlay")
    ax.set_yscale("log")
    ax.set_xlabel(f"Years from {meta['calendar'][0]}")
    ax.set_ylabel("Geocentric distance (km, log scale)")
    ax.set_title("Close-approach zoom: refined Horizons window vs surrogate")
    ax.grid(True, which="both", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")
    _save_figure(fig, "fig_ml_close_approach_zoom.png", dpi=180)

    if report_bits.get("model_scores"):
        names = list(report_bits["model_scores"].keys())
        rmses = [report_bits["model_scores"][name]["rmse_km"] for name in names]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(names)), rmses, color="#2e86ab")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel("Validation RMSE (km)")
        ax.set_title("Model family comparison on blocked validation")
        ax.grid(True, axis="y", alpha=0.25)
        _save_figure(fig, "fig_ml_model_comparison.png", dpi=180)

    if report_bits.get("selection_scores"):
        ranked = sorted(
            report_bits["selection_scores"].items(),
            key=lambda item: item[1]["cv_rmse_km_mean"],
        )
        names = [item[0] for item in ranked]
        means = [item[1]["cv_rmse_km_mean"] for item in ranked]
        stds = [item[1]["cv_rmse_km_std"] for item in ranked]
        selected = str(report_bits.get("selected_primary_model", ""))
        colors = ["#d1495b" if name == selected else "#7d8597" for name in names]
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        ax.bar(range(len(names)), means, color=colors, alpha=0.9)
        ax.errorbar(range(len(names)), means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.0, capsize=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=28, ha="right")
        ax.set_ylabel("Purged CV RMSE (km)")
        ax.set_title("Primary TensorFlow Selection by Purged CV")
        ax.grid(True, axis="y", alpha=0.25)
        _save_figure(fig, "fig_ml_primary_selection_cv.png", dpi=180)

    if report_bits.get("coverage_by_regime"):
        ordered = []
        for regime in ("near", "mid", "far"):
            if regime in report_bits["coverage_by_regime"]:
                ordered.append((regime, report_bits["coverage_by_regime"][regime]))
        if ordered:
            labels = [name for name, _ in ordered]
            coverages = [item["coverage"] for _, item in ordered]
            maes = [item["mae_km"] / 1.0e3 for _, item in ordered]
            fig, ax1 = plt.subplots(figsize=(10.5, 5.0))
            bars = ax1.bar(labels, coverages, color=["#355070", "#6d597a", "#b56576"], alpha=0.88)
            ax1.axhline(0.90, color="black", lw=1.0, ls="--", label="nominal 90%")
            ax1.set_ylim(0.0, 1.05)
            ax1.set_ylabel("Empirical coverage")
            ax1.set_title("Conformal Coverage by Validation Distance Regime")
            ax1.grid(True, axis="y", alpha=0.25)
            ax2 = ax1.twinx()
            ax2.plot(labels, maes, color="#2a9d8f", marker="o", lw=1.3, label="MAE")
            ax2.set_ylabel("MAE (10^3 km)", color="#2a9d8f")
            ax2.tick_params(axis="y", labelcolor="#2a9d8f")
            lines = [bars[0], ax1.lines[0], ax2.lines[0]]
            ax1.legend(lines, ["coverage", "nominal 90%", "MAE"], loc="best")
            _save_figure(fig, "fig_ml_coverage_by_regime.png", dpi=180)

    if report_bits.get("top_features"):
        feats = report_bits["top_features"][:15]
        labels = [f[0] for f in feats][::-1]
        vals = [f[1] for f in feats][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, vals, color="#00798c")
        ax.set_xlabel("Relative importance")
        ax.set_title("TensorFlow first-layer feature salience")
        ax.grid(True, axis="x", alpha=0.25)
        _save_figure(fig, "fig_ml_feature_importance.png", dpi=180)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residual_km[val_mask], bins=45, color="#6c757d", alpha=0.82)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_xlabel("Validation residual (km)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residual distribution; conformal coverage={report_bits.get('coverage', float('nan')):.1%}"
    )
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "fig_ml_residual_distribution.png", dpi=180)

    publication_assets = _write_figure_captions(output_dir, figure_catalog)
    return figures, publication_assets


def run_ml_surrogate(
    neo: NEOObject,
    hypothesis: HypothesisTerms,
    target: str,
    date_min: str,
    date_max: str,
    horizons_step: str,
    output_dir: Path,
    refine_step: str = "1h",
    refine_window_days: float = 5.0,
    refine: bool = True,
    uncertainty_samples: int = 768,
) -> MLSurrogateReport:
    import numpy as np
    tf_version = _tensorflow_available_version()

    geo = fetch_horizons_vectors(target, "500@399", date_min, date_max, horizons_step)
    helio = fetch_horizons_vectors(target, "500@10", date_min, date_max, horizons_step)
    if refine:
        geo, helio = _refine_horizons_near_minima(
            target,
            geo,
            helio,
            refine_step=refine_step,
            window_days=refine_window_days,
            anchor_jds=[ca.jd_tdb for ca in neo.close_approaches],
        )
    X, y, feature_names, meta = _build_ml_feature_matrix(neo, hypothesis, geo, helio)

    n = len(y)
    train_mask, calib_mask, val_mask, numerical_diagnostics = _build_time_block_partitions(
        meta["jd"], neo, refine_step, refine_window_days
    )
    if int(train_mask.sum()) < 80 or int(calib_mask.sum()) < 40 or int(val_mask.sum()) < 20:
        raise SourceError("insufficient train/calibration/validation samples after purged blocked split")

    global_physics_baseline_log = np.log10(np.maximum(np.asarray(meta["dist_au"], dtype=float), 1e-300))
    fit_mask = train_mask | calib_mask
    y_train = y[train_mask]
    y_cal = y[calib_mask]
    y_val = y[val_mask]
    sample_weight = _numerical_training_weights(meta, train_mask)
    fit_sample_weight = _numerical_training_weights(meta, fit_mask)
    train_weight = sample_weight[train_mask]
    fit_weight = fit_sample_weight[fit_mask]
    numerical_diagnostics["training_weight_min"] = float(np.min(train_weight))
    numerical_diagnostics["training_weight_median"] = float(np.median(train_weight))
    numerical_diagnostics["training_weight_max"] = float(np.max(train_weight))
    numerical_diagnostics["fit_weight_min"] = float(np.min(fit_weight))
    numerical_diagnostics["fit_weight_median"] = float(np.median(fit_weight))
    numerical_diagnostics["fit_weight_max"] = float(np.max(fit_weight))
    numerical_diagnostics["train_points"] = float(np.count_nonzero(train_mask))
    numerical_diagnostics["calibration_points"] = float(np.count_nonzero(calib_mask))
    numerical_diagnostics["validation_points"] = float(np.count_nonzero(val_mask))

    primary_model_name, primary_model, selection_scores, selection_folds = _select_primary_tensorflow_model(
        X[train_mask],
        y_train,
        train_weight,
        np.asarray(meta["jd"], dtype=float)[train_mask],
        numerical_diagnostics["cad_embargo_days"],
        baseline_log_train=global_physics_baseline_log[train_mask],
    )
    X_train, X_fit, X_val = X[train_mask], X[fit_mask], X[val_mask]
    y_fit = y[fit_mask]
    numerical_diagnostics["primary_model"] = primary_model_name
    numerical_diagnostics["primary_feature_count"] = float(len(feature_names))
    numerical_diagnostics["ml_engine"] = "tensorflow"
    numerical_diagnostics["tensorflow_version"] = tf_version
    numerical_diagnostics["global_target_mode"] = "state_vector_range_residual"
    numerical_diagnostics["primary_selection_folds"] = float(selection_folds)
    numerical_diagnostics["primary_selection_scores"] = selection_scores
    if primary_model_name in selection_scores:
        numerical_diagnostics["selected_primary_cv_rmse_km"] = selection_scores[primary_model_name]["cv_rmse_km_mean"]
        numerical_diagnostics["selected_primary_cv_rmse_std_km"] = selection_scores[primary_model_name]["cv_rmse_km_std"]

    X_local, local_design, local_feature_names = _build_local_refinement_design(
        meta,
        neo,
        refine_window_days,
        base_X=X,
        base_feature_names=feature_names,
    )
    locality_score = np.asarray(local_design["locality_score"], dtype=float)
    local_proposal_taper = np.asarray(local_design["proposal_taper"], dtype=float)
    local_abs_dt_days = np.asarray(local_design["abs_dt_days"], dtype=float)
    local_objective_weight = np.asarray(local_design["objective_weight"], dtype=float)
    local_physics_baseline_log = np.asarray(local_design["physics_taylor_baseline_log"], dtype=float)
    local_encounter_center_jd = np.asarray(local_design["encounter_center_jd"], dtype=float)
    local_encounter_speed_km_s = np.asarray(local_design["encounter_speed_km_s"], dtype=float)
    numerical_diagnostics.update(local_design["diagnostics"])
    numerical_diagnostics["local_target_mode"] = "physics_taylor_correction"

    global_residual_proposal_log = np.asarray(primary_model.predict(X), dtype=float)
    global_residual_blend_strength = 0.0
    global_residual_rows: list[dict[str, float]] = []
    calibration_true_km_for_global = 10.0 ** y_cal * AU_KM
    for blend_strength in np.linspace(0.0, 1.0, 5):
        candidate_log = global_physics_baseline_log[calib_mask] + blend_strength * global_residual_proposal_log[calib_mask]
        candidate_km = 10.0 ** candidate_log * AU_KM
        err = candidate_km - calibration_true_km_for_global
        global_residual_rows.append(
            {
                "blend_strength": float(blend_strength),
                "rmse_km": float(np.sqrt(np.mean(err * err))),
                "mae_km": float(np.mean(np.abs(err))),
            }
        )
    global_no_correction_rmse = float(global_residual_rows[0]["rmse_km"])
    global_best_row = min(global_residual_rows, key=lambda row: (row["rmse_km"], row["mae_km"]))
    if global_best_row["blend_strength"] > 0.0 and global_best_row["rmse_km"] <= global_no_correction_rmse * 0.995:
        global_residual_blend_strength = float(global_best_row["blend_strength"])
    global_train_pred_log = global_physics_baseline_log + global_residual_blend_strength * global_residual_proposal_log
    numerical_diagnostics["global_residual_blend_strength"] = float(global_residual_blend_strength)
    numerical_diagnostics["global_residual_gate_accepted"] = bool(global_residual_blend_strength > 0.0)
    numerical_diagnostics["global_residual_gate_reference_rmse_km"] = float(global_no_correction_rmse)
    numerical_diagnostics["global_residual_gate_best_rmse_km"] = float(global_best_row["rmse_km"])
    numerical_diagnostics["global_residual_blend_selection"] = global_residual_rows
    objective_window_days = float(numerical_diagnostics.get("local_objective_window_days", refine_window_days))
    local_window_multiplier = 1.0
    local_train_mask = train_mask & np.asarray(local_design["encounter_window_mask"], dtype=bool)
    minimum_local_points = max(96, int(np.count_nonzero(train_mask) * 0.06))
    for multiplier in (1.0, 1.25, 1.5, 2.0):
        candidate_mask = train_mask & (local_abs_dt_days <= objective_window_days * multiplier)
        local_train_mask = candidate_mask
        local_window_multiplier = float(multiplier)
        if int(np.count_nonzero(candidate_mask)) >= minimum_local_points:
            break
    local_fit_mask = fit_mask & (local_abs_dt_days <= objective_window_days * local_window_multiplier)
    local_train_weight = sample_weight[local_train_mask] * local_objective_weight[local_train_mask]
    local_train_median = max(float(np.nanmedian(local_train_weight)), 1e-9) if np.any(local_train_mask) else 1.0
    local_train_weight = np.clip(local_train_weight / local_train_median, 0.2, 12.0)
    local_model_name = "NoLocalRefine"
    local_model: Any | None = None
    local_selection_scores: dict[str, dict[str, float]] = {}
    local_selection_folds = 0
    local_clip_scale = 0.0
    local_blend_strength = 0.0
    if int(np.count_nonzero(local_train_mask)) >= 80:
        local_model_name, local_model, local_selection_scores, local_selection_folds, local_clip_scale = _select_local_refinement_model(
            X_local[local_train_mask],
            y[local_train_mask],
            global_train_pred_log[local_train_mask],
            local_physics_baseline_log[local_train_mask],
            locality_score[local_train_mask],
            local_proposal_taper[local_train_mask],
            local_encounter_center_jd[local_train_mask],
            local_encounter_speed_km_s[local_train_mask],
            local_train_weight,
            np.asarray(meta["jd"], dtype=float)[local_train_mask],
            numerical_diagnostics["cad_embargo_days"],
            objective_window_days,
        )
        train_local_correction = np.asarray(local_model.predict(X_local), dtype=float)
        train_local_correction = np.clip(train_local_correction, -local_clip_scale, local_clip_scale)
        train_local_proposal_log = local_physics_baseline_log + train_local_correction
        calibration_candidates = np.linspace(0.0, 1.0, 5)
        calibration_true_km = 10.0 ** y_cal * AU_KM
        best_calibration_rmse = float("inf")
        calibration_rows: list[dict[str, float]] = []
        for blend_strength in calibration_candidates:
            calibration_pred_log = global_train_pred_log[calib_mask] + blend_strength * locality_score[calib_mask] * (
                local_proposal_taper[calib_mask] * (train_local_proposal_log[calib_mask] - global_train_pred_log[calib_mask])
            )
            calibration_pred_km = 10.0 ** calibration_pred_log * AU_KM
            err = calibration_pred_km - calibration_true_km
            rmse_metric = float(np.sqrt(np.mean(err * err)))
            center_stats = _encounter_center_objective(
                np.asarray(meta["jd"], dtype=float)[calib_mask],
                y[calib_mask],
                calibration_pred_log,
                local_encounter_center_jd[calib_mask],
                local_encounter_speed_km_s[calib_mask],
                objective_window_days,
            )
            calibration_rows.append(
                {
                    "blend_strength": float(blend_strength),
                    "rmse_km": float(rmse_metric),
                    "center_combined_km_mean": float(center_stats.get("combined_km_mean", float("nan"))),
                    "center_timing_hours_mean": float(center_stats.get("timing_hours_mean", float("nan"))),
                    "center_depth_error_km_mean": float(center_stats.get("depth_error_km_mean", float("nan"))),
                }
            )
            best_calibration_rmse = min(best_calibration_rmse, rmse_metric)
        blend_tolerance = max(best_calibration_rmse * 0.02, 5.0e3)
        eligible_rows = [
            row
            for row in calibration_rows
            if row["rmse_km"] <= best_calibration_rmse + blend_tolerance
        ]
        chosen_row = min(
            eligible_rows or calibration_rows,
            key=lambda row: (
                row["center_combined_km_mean"] if math.isfinite(row["center_combined_km_mean"]) else float("inf"),
                row["rmse_km"],
            ),
        )
        no_local_rmse = float(calibration_rows[0]["rmse_km"]) if calibration_rows else float("inf")
        if chosen_row["blend_strength"] > 0.0 and chosen_row["rmse_km"] <= no_local_rmse * 0.995:
            local_blend_strength = float(chosen_row["blend_strength"])
        else:
            local_blend_strength = 0.0
        numerical_diagnostics["local_gate_accepted"] = bool(local_blend_strength > 0.0)
        numerical_diagnostics["local_gate_reference_rmse_km"] = float(no_local_rmse)
        numerical_diagnostics["local_gate_best_rmse_km"] = float(chosen_row["rmse_km"])
        numerical_diagnostics["local_blend_selection"] = calibration_rows
        numerical_diagnostics["local_blend_selection_rmse_tolerance_km"] = float(blend_tolerance)
    numerical_diagnostics["local_refinement_model"] = local_model_name
    numerical_diagnostics["local_selection_folds"] = float(local_selection_folds)
    numerical_diagnostics["local_selection_scores"] = local_selection_scores
    numerical_diagnostics["local_objective_window_multiplier"] = float(local_window_multiplier)
    numerical_diagnostics["local_objective_window_days_effective"] = float(objective_window_days * local_window_multiplier)
    numerical_diagnostics["local_train_points"] = float(np.count_nonzero(local_train_mask))
    numerical_diagnostics["local_fit_points"] = float(np.count_nonzero(local_fit_mask))
    numerical_diagnostics["local_blend_strength"] = float(local_blend_strength)

    _fit_model_with_weights(primary_model, X_fit, y_fit - global_physics_baseline_log[fit_mask], fit_weight)
    global_residual_proposal_log = np.asarray(primary_model.predict(X), dtype=float)
    global_residual_full_log = global_physics_baseline_log + global_residual_proposal_log
    global_pred_log = global_physics_baseline_log + global_residual_blend_strength * global_residual_proposal_log
    local_proposal_log = np.asarray(local_physics_baseline_log, dtype=float)
    if local_model is not None and int(np.count_nonzero(local_fit_mask)) >= 80:
        local_fit_weight = fit_sample_weight[local_fit_mask] * local_objective_weight[local_fit_mask]
        local_fit_median = max(float(np.nanmedian(local_fit_weight)), 1e-9)
        local_fit_weight = np.clip(local_fit_weight / local_fit_median, 0.2, 12.0)
        local_fit_correction = y[local_fit_mask] - local_physics_baseline_log[local_fit_mask]
        _fit_model_with_weights(local_model, X_local[local_fit_mask], local_fit_correction, local_fit_weight)
        local_clip_scale = max(local_clip_scale, float(np.quantile(np.abs(local_fit_correction), 0.98)))
        local_correction_prediction = np.asarray(local_model.predict(X_local), dtype=float)
        local_correction_prediction = np.clip(local_correction_prediction, -local_clip_scale, local_clip_scale)
        local_proposal_log = local_physics_baseline_log + local_correction_prediction
    global_local_name = f"{primary_model_name}+{local_model_name}" if local_model is not None else primary_model_name
    pred_log = global_pred_log + local_blend_strength * locality_score * (
        local_proposal_taper * (local_proposal_log - global_pred_log)
    )
    numerical_diagnostics["primary_model"] = global_local_name
    numerical_diagnostics["global_model_name"] = primary_model_name
    numerical_diagnostics["local_feature_count"] = float(len(local_feature_names))
    numerical_diagnostics["local_clip_scale_log10_au"] = float(local_clip_scale)

    model_predictions: dict[str, Any] = {}
    model_scores: dict[str, dict[str, float]] = {}
    global_val_pred_log = global_pred_log[val_mask]
    global_true_km = 10.0 ** y_val * AU_KM
    global_pred_km = 10.0 ** global_val_pred_log * AU_KM
    global_err_km = global_pred_km - global_true_km
    model_scores["GlobalOnly"] = {
        "mae_km": float(np.mean(np.abs(global_err_km))),
        "median_ae_km": float(np.median(np.abs(global_err_km))),
        "rmse_km": float(np.sqrt(np.mean(global_err_km * global_err_km))),
        "r2_log": _r2_score_np(y_val, global_val_pred_log),
    }
    model_predictions[global_local_name] = np.asarray(pred_log, dtype=float)
    primary_pred_val = model_predictions[global_local_name][val_mask]
    primary_true_km = 10.0 ** y_val * AU_KM
    primary_pred_km = 10.0 ** primary_pred_val * AU_KM
    primary_err_km = primary_pred_km - primary_true_km
    model_scores[global_local_name] = {
        "mae_km": float(np.mean(np.abs(primary_err_km))),
        "median_ae_km": float(np.median(np.abs(primary_err_km))),
        "rmse_km": float(np.sqrt(np.mean(primary_err_km * primary_err_km))),
        "r2_log": _r2_score_np(y_val, primary_pred_val),
    }
    if primary_model_name in selection_scores:
        model_scores[global_local_name].update(selection_scores[primary_model_name])
    ensemble_weights = {name: (1.0 if name == global_local_name else 0.0) for name in model_scores}
    numerical_diagnostics["diagnostic_comparator_models"] = float(max(len(model_scores) - 2, 0))
    cal_scores = np.abs(pred_log[calib_mask] - y_cal)
    conformal_q90 = float(np.quantile(cal_scores, 0.90))
    local_q90 = _localized_conformal_widths(meta["jd"], 10.0 ** y, calib_mask, cal_scores)
    lo_log = pred_log - local_q90
    hi_log = pred_log + local_q90

    dist_true = 10.0 ** y
    dist_pred = 10.0 ** pred_log
    cal_true = dist_true[calib_mask]
    val_true = dist_true[val_mask]
    val_pred = dist_pred[val_mask]
    val_err_au = val_pred - val_true
    mae_km = float(np.mean(np.abs(val_err_au)) * AU_KM)
    median_ae_km = _median_absolute_error_np(val_true * AU_KM, val_pred * AU_KM)
    rmse_km = float(np.sqrt(np.mean(val_err_au * val_err_au)) * AU_KM)
    r2_log = _r2_score_np(y_val, pred_log[val_mask])
    cal_covered = (10.0 ** lo_log[calib_mask] <= cal_true) & (cal_true <= 10.0 ** hi_log[calib_mask])
    conformal_covered = (10.0 ** lo_log[val_mask] <= val_true) & (val_true <= 10.0 ** hi_log[val_mask])
    calibration_coverage = float(np.mean(cal_covered))
    conformal_coverage = float(np.mean(conformal_covered))
    conformal_width_km = float(np.median((10.0 ** hi_log - 10.0 ** lo_log) * AU_KM))
    mae_boot = _bootstrap_metric_band(val_true * AU_KM, val_pred * AU_KM, "mae")
    rmse_boot = _bootstrap_metric_band(val_true * AU_KM, val_pred * AU_KM, "rmse")
    regime_coverage = _coverage_by_distance_regime(
        val_true,
        conformal_covered,
        val_pred * AU_KM,
        val_true * AU_KM,
    )
    numerical_diagnostics["calibration_conformal_q90_log10_au"] = conformal_q90
    numerical_diagnostics["localized_conformal_q90_median_log10_au"] = float(np.median(local_q90))
    numerical_diagnostics["calibration_interval_coverage"] = calibration_coverage
    numerical_diagnostics["validation_median_ae_km"] = median_ae_km
    numerical_diagnostics["validation_r2_log"] = r2_log
    numerical_diagnostics["validation_mae_bootstrap_90pct_km"] = mae_boot
    numerical_diagnostics["validation_rmse_bootstrap_90pct_km"] = rmse_boot
    numerical_diagnostics["coverage_by_distance_regime"] = regime_coverage
    numerical_diagnostics["global_only_validation_mae_km"] = model_scores["GlobalOnly"]["mae_km"]
    numerical_diagnostics["global_only_validation_rmse_km"] = model_scores["GlobalOnly"]["rmse_km"]
    numerical_diagnostics["local_refinement_mae_gain_km"] = model_scores["GlobalOnly"]["mae_km"] - mae_km
    numerical_diagnostics["local_refinement_rmse_gain_km"] = model_scores["GlobalOnly"]["rmse_km"] - rmse_km
    validation_center_stats = _encounter_center_objective(
        np.asarray(meta["jd"], dtype=float)[val_mask],
        y[val_mask],
        pred_log[val_mask],
        local_encounter_center_jd[val_mask],
        local_encounter_speed_km_s[val_mask],
        objective_window_days,
    )
    if int(validation_center_stats.get("encounter_count", 0.0)) > 0:
        numerical_diagnostics["validation_center_timing_hours_mean"] = float(validation_center_stats["timing_hours_mean"])
        numerical_diagnostics["validation_center_depth_error_km_mean"] = float(validation_center_stats["depth_error_km_mean"])
        numerical_diagnostics["validation_center_combined_km_mean"] = float(validation_center_stats["combined_km_mean"])
    local_val_mask = val_mask & (local_abs_dt_days <= objective_window_days * local_window_multiplier)
    if int(np.count_nonzero(local_val_mask)) >= 12:
        local_true_km = 10.0 ** y[local_val_mask] * AU_KM
        local_baseline_km = 10.0 ** local_physics_baseline_log[local_val_mask] * AU_KM
        local_global_km = 10.0 ** global_pred_log[local_val_mask] * AU_KM
        local_tapered_proposal_log = global_pred_log[local_val_mask] + local_proposal_taper[local_val_mask] * (
            local_proposal_log[local_val_mask] - global_pred_log[local_val_mask]
        )
        local_proposal_km = 10.0 ** local_tapered_proposal_log * AU_KM
        numerical_diagnostics["local_window_validation_points"] = float(np.count_nonzero(local_val_mask))
        numerical_diagnostics["local_window_baseline_mae_km"] = float(np.mean(np.abs(local_baseline_km - local_true_km)))
        numerical_diagnostics["local_window_global_mae_km"] = float(np.mean(np.abs(local_global_km - local_true_km)))
        numerical_diagnostics["local_window_proposal_mae_km"] = float(np.mean(np.abs(local_proposal_km - local_true_km)))

    anchor_rows = _build_anchor_validation_rows(neo, meta, pred_log, lo_log, hi_log, global_pred_log=global_pred_log)
    if anchor_rows:
        sample_anchor_resid = np.asarray([row["ml_minus_cad_km"] for row in anchor_rows], dtype=float)
        interp_anchor_resid = np.asarray([row["ml_interpolated_minus_cad_km"] for row in anchor_rows], dtype=float)
        tf_cont_anchor_resid = np.asarray([row["tensorflow_continuous_minus_cad_km"] for row in anchor_rows], dtype=float)
        numerical_diagnostics["cad_anchor_sample_rmse_km"] = float(np.sqrt(np.mean(sample_anchor_resid * sample_anchor_resid)))
        numerical_diagnostics["cad_anchor_interpolated_rmse_km"] = float(np.sqrt(np.mean(interp_anchor_resid * interp_anchor_resid)))
        numerical_diagnostics["cad_anchor_tensorflow_continuous_rmse_km"] = float(np.sqrt(np.mean(tf_cont_anchor_resid * tf_cont_anchor_resid)))
        numerical_diagnostics["cad_anchor_sample_mae_km"] = float(np.mean(np.abs(sample_anchor_resid)))
        numerical_diagnostics["cad_anchor_interpolated_mae_km"] = float(np.mean(np.abs(interp_anchor_resid)))
        numerical_diagnostics["cad_anchor_tensorflow_continuous_mae_km"] = float(np.mean(np.abs(tf_cont_anchor_resid)))
        nearest_anchor = min(anchor_rows, key=lambda row: row["cad_distance_km"])
        numerical_diagnostics["nearest_cad_tensorflow_continuous_error_km"] = float(nearest_anchor["tensorflow_continuous_minus_cad_km"])
        numerical_diagnostics["nearest_cad_sample_error_km"] = float(nearest_anchor["ml_minus_cad_km"])
        numerical_diagnostics["nearest_cad_interpolated_error_km"] = float(nearest_anchor["ml_interpolated_minus_cad_km"])
        numerical_diagnostics["tensorflow_continuous_anchor_gate_accepted"] = bool(
            numerical_diagnostics["cad_anchor_tensorflow_continuous_rmse_km"]
            < min(
                numerical_diagnostics["cad_anchor_sample_rmse_km"],
                numerical_diagnostics["cad_anchor_interpolated_rmse_km"],
            )
        )

    true_min_idx = int(np.argmin(dist_true))
    pred_min_idx = int(np.argmin(dist_pred))
    nearest_error_km = float((dist_pred[pred_min_idx] - dist_true[pred_min_idx]) * AU_KM)

    cad_error: float | None = None
    nearest_cad = _nearest_approach(neo.close_approaches)
    if nearest_cad is not None:
        cad_idx = int(np.argmin(np.abs(meta["jd"] - nearest_cad.jd_tdb)))
        cad_error = float((dist_pred[cad_idx] - nearest_cad.distance_au) * AU_KM)

    top_features: list[tuple[str, float]] = []
    if getattr(primary_model, "feature_importances_", None) is not None:
        importances = getattr(primary_model, "feature_importances_")
        total = float(np.sum(importances)) or 1.0
        order = np.argsort(importances)[::-1]
        top_features = [(feature_names[int(i)], float(importances[int(i)] / total)) for i in order[:20]]
    if local_model is not None and getattr(local_model, "feature_importances_", None) is not None:
        local_importances = getattr(local_model, "feature_importances_")
        local_total = float(np.sum(local_importances)) or 1.0
        local_order = np.argsort(local_importances)[::-1]
        numerical_diagnostics["local_top_features"] = [
            (local_feature_names[int(i)], float(local_importances[int(i)] / local_total))
            for i in local_order[:10]
        ]

    table_paths = _write_anchor_tables(output_dir, anchor_rows)
    table_paths.extend(_write_publication_tables(output_dir, meta, pred_log, lo_log, hi_log, global_pred_log, locality_score, calib_mask, val_mask))
    table_paths.extend(_write_tensorflow_gate_tables(output_dir, numerical_diagnostics, anchor_rows, meta, global_residual_full_log))
    uncertainty_table_paths, uncertainty_figures, uncertainty_publication_assets = _write_uncertainty_propagation(
        output_dir,
        neo,
        hypothesis,
        geo,
        helio,
        anchor_rows,
        uncertainty_samples,
        numerical_diagnostics,
    )
    table_paths.extend(uncertainty_table_paths)
    bits = {
        "mae_km": mae_km,
        "rmse_km": rmse_km,
        "model_scores": model_scores,
        "selection_scores": selection_scores,
        "selected_primary_model": primary_model_name,
        "global_pred_log": global_pred_log,
        "global_physics_baseline_log": global_physics_baseline_log,
        "global_residual_proposal_log": global_residual_proposal_log,
        "global_residual_full_log": global_residual_full_log,
        "local_proposal_log": local_proposal_log,
        "locality_score": locality_score,
        "coverage_by_regime": regime_coverage,
        "top_features": top_features,
        "coverage": conformal_coverage,
        "zoom_days": max(refine_window_days * 1.6, 8.0),
        "anchor_rows": anchor_rows,
        "numerical_diagnostics": numerical_diagnostics,
    }
    figures, publication_assets = _write_ml_plots(output_dir, neo, meta, pred_log, lo_log, hi_log, val_mask, bits)
    figures.extend(uncertainty_figures)
    publication_assets.extend(uncertainty_publication_assets)

    return MLSurrogateReport(
        enabled=True,
        method="Physics-informed no-CAD-anchor surrogate: a global TensorFlow/Keras tabular neural model selected on purged training folds to learn residual correction to the state-vector range baseline, plus an encounter-centered TensorFlow local specialist that learns correction to a state-vector Taylor baseline within refined windows, and a TensorFlow kernel-ridge continuous-time encounter reconstructor for CAD-epoch diagnostics; residual branches are blended only when the unchanged calibration gate proves value on untouched chronology",
        training_label_source="JPL Horizons VECTORS geocentric range samples only; CAD close approaches are validation overlays, not training labels.",
        feature_names=feature_names,
        n_samples=n,
        n_train=int(train_mask.sum()),
        n_calibration=int(calib_mask.sum()),
        n_validation=int(val_mask.sum()),
        horizons_step=f"{horizons_step} + refinement {refine_step} within +/-{refine_window_days:g}d of coarse minima" if refine else horizons_step,
        validation_mae_km=mae_km,
        validation_rmse_km=rmse_km,
        nearest_horizons_date=meta["calendar"][true_min_idx],
        nearest_horizons_distance_au=float(dist_true[true_min_idx]),
        nearest_ml_date=meta["calendar"][pred_min_idx],
        nearest_ml_distance_au=float(dist_pred[pred_min_idx]),
        nearest_ml_error_km=nearest_error_km,
        cad_validation_error_km=cad_error,
        model_scores=model_scores,
        ensemble_weights=ensemble_weights,
        numerical_diagnostics=numerical_diagnostics,
        conformal_90_width_km_median=conformal_width_km,
        conformal_90_coverage=conformal_coverage,
        top_features=top_features,
        anchor_validation=anchor_rows,
        figures=figures,
        tables=table_paths,
        publication_assets=publication_assets,
        caveats=[
            "The model is a surrogate trained from Horizons state-vector-derived samples, not an independent orbit determination from raw astrometry.",
            "TensorFlow/Keras neural regressors are the sole point-prediction engines in this build; scikit-learn model families are not used.",
            "Neural residual branches are calibration-gated against the state-vector physics baseline; a zero accepted blend is a documented rejection of the TensorFlow proposal, not a hidden failure or an independent ML victory.",
            "The TensorFlow continuous-time encounter reconstructor uses Horizons state-vector samples around the CAD epoch and does not use CAD distance as a training label; its CAD residual is therefore a reconstruction diagnostic, not an orbit-determination replacement.",
            "The covariance propagation layer uses SBDB covariance clones and a two-body element propagation centered on CAD anchors; the GI_N/OI_N cascade adjustment is an adaptive uncertainty diagnostic, not a replacement for JPL covariance transport or force modeling.",
            "CAD close approaches are not used as training labels, but they come from the same JPL ecosystem and are only a validation overlay.",
            "Close-approach accuracy is limited by Horizons sampling and refinement cadence; anchor tables now include local polynomial interpolation at the CAD epoch, but use --refine-step 15m or finer for publication-grade close-approach analysis.",
            "ML/DL residuals must be reported with the plots; do not replace Horizons/CAD operational products with this surrogate.",
        ],
    )


def build_standard_assessment(neo: NEOObject) -> StandardAssessment:
    nearest = _nearest_approach(neo.close_approaches)
    h = neo.physical.absolute_magnitude_h
    moid = neo.elements.moid_au
    pha_by_rule = None
    if h is not None and moid is not None:
        pha_by_rule = moid <= 0.05 and h <= 22.0

    notes = [
        "Standard close approaches are reported directly from JPL CAD; this script does not fit or overwrite them.",
        "Sentry status is reported directly from JPL Sentry; absence/removal is not converted into a numeric impact probability.",
    ]
    if neo.sentry.raw_status == "removed":
        notes.append(f"Sentry reports this object was removed on {neo.sentry.removed_date}.")
    if nearest:
        notes.append(
            f"Nearest listed Earth approach in requested window: {nearest.calendar_date_tdb}, "
            f"{nearest.distance_au:.15g} au, v_rel={nearest.v_rel_km_s} km/s."
        )
    else:
        notes.append("No CAD Earth close approaches matched the requested date/distance filters.")

    return StandardAssessment(
        source="NASA/JPL SBDB, CAD, and Sentry APIs",
        nearest_approach=nearest,
        sentry_status=neo.sentry,
        pha_by_cneos_rule=pha_by_rule,
        notes=notes,
    )


def analyze(target: str, date_min: str, date_max: str, dist_max: str) -> AnalysisReport:
    neo = load_neo(target, date_min, date_max, dist_max)
    return AnalysisReport(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        target=target,
        standard=build_standard_assessment(neo),
        hypothesis=evaluate_hypothesis(neo),
        object=neo,
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def print_report(report: AnalysisReport) -> None:
    obj = report.object
    elems = obj.elements
    hyp = report.hypothesis
    std = report.standard
    nearest = std.nearest_approach

    print("=" * 88)
    print(f"CLEAN NEO HYPOTHESIS ANALYSIS - {obj.fullname}")
    print("=" * 88)
    print(f"Generated UTC: {report.generated_utc}")
    print()
    print("[ONLINE SOURCE SUMMARY]")
    for source in obj.sources:
        version = f" v{source.version}" if source.version else ""
        print(f"  - {source.name}{version}: {source.url}")
    print()
    print("[STANDARD NASA/JPL FACTS]")
    print(f"  Object class from SBDB: {obj.orbit_class_code} ({obj.orbit_class_name})")
    print(f"  Computed CNEOS group:   {hyp.inputs.neo_group_from_orbit}")
    print(f"  NEO / PHA flags:        neo={obj.neo}, pha={obj.pha}, pha_by_rule={std.pha_by_cneos_rule}")
    print(f"  Orbit ID / condition:   {elems.orbit_id} / {elems.condition_code}")
    print(f"  a, e, q, Q, i:          {elems.semi_major_axis_au:.15g} au, {elems.eccentricity:.15g}, "
          f"{elems.perihelion_au:.15g} au, {elems.aphelion_au:.15g} au, {elems.inclination_deg:.12g} deg")
    print(f"  MOID / H / diameter:    {elems.moid_au} au, H={obj.physical.absolute_magnitude_h}, "
          f"D={obj.physical.diameter_km} km")
    if nearest:
        print(f"  Nearest CAD approach:   {nearest.calendar_date_tdb} TDB")
        print(f"                         dist={nearest.distance_au:.15g} au "
              f"({nearest.distance_au * AU_KM:,.0f} km), v_rel={nearest.v_rel_km_s} km/s")
    print(f"  Sentry:                 {obj.sentry.raw_status}"
          + (f" ({obj.sentry.removed_date})" if obj.sentry.removed_date else ""))
    for note in std.notes:
        print(f"  note: {note}")
    print()
    print("[PDF HYPOTHESIS DIAGNOSTIC]")
    print(f"  Gamma rule:             {hyp.inputs.gamma_rule}")
    print(f"  gamma:                  {hyp.inputs.gamma_ratio:.12g}")
    print(f"  U speed source/value:   {hyp.inputs.selected_approach_date or 'SBDB period-derived mean'}; "
          f"{hyp.inputs.orbital_speed_m_s:.9g} m/s")
    print(f"  upsilon=v/c:            {hyp.inputs.upsilon_v_over_c:.9e}")
    print(f"  Seqcr / TrajLoss:       {hyp.seqcr:.9e} / {hyp.trajectory_loss:.9e}")
    print(f"  raw ecc/SMA output:     e={hyp.new_eccentricity_raw:.9e}, "
          f"a={hyp.new_semi_major_axis_m / AU_M:.9e} au")
    print(f"  likelihood proxy:       raw={hyp.likelihood_proxy_raw:.9e}, "
          f"unit={hyp.likelihood_unit_interval:.6f}, band={hyp.likelihood_band}")
    print(f"  GI_N / OI_N diagnostic: GI_N={hyp.gi_n_raw:.9e}, OI_N={hyp.oi_n_raw:.9e}")
    print()
    print("  Scaled PDF terms:")
    for key, value in hyp.scaled_terms.items():
        print(f"    {key:32s} {value:.9e}")
    print()
    print("  Range-preserved PDF terms:")
    for key, stats in hyp.range_terms.items():
        print(f"    {key:32s} low={stats.low:.9e} median={stats.median:.9e} high={stats.high:.9e}")
    print()
    print("[CAVEATS]")
    for caveat in hyp.caveats:
        wrapped = textwrap.wrap(caveat, width=82)
        print(f"  - {wrapped[0]}")
        for line in wrapped[1:]:
            print(f"    {line}")
    if report.dynamics is not None:
        dyn = report.dynamics
        print()
        print("[DYNAMICS-FIRST CASCADE PROPAGATION]")
        print(f"  Method:                 {dyn.method}")
        print(f"  Force model:            {dyn.force_model}")
        print(f"  Integrator:             {dyn.integrator}")
        print(f"  Prediction mode:        {dyn.numerical_diagnostics.get('prediction_mode', 'unknown')}")
        print(f"  Samples:                {dyn.n_samples}  (step={dyn.horizons_step})")
        print(f"  Horizons validation:    MAE={dyn.validation_mae_km:,.0f} km, RMSE={dyn.validation_rmse_km:,.0f} km")
        if dyn.numerical_diagnostics:
            print("  Cascade controls:       "
                  f"weights=({dyn.numerical_diagnostics.get('cascade_vector_weight_velocity', float('nan')):.3g}, "
                  f"{dyn.numerical_diagnostics.get('cascade_vector_weight_radial', float('nan')):.3g}, "
                  f"{dyn.numerical_diagnostics.get('cascade_vector_weight_normal', float('nan')):.3g}), "
                  f"source={dyn.numerical_diagnostics.get('cascade_acceleration_source', 'unknown')}")
            print("  N-body perturbations:   "
                  f"frame={dyn.numerical_diagnostics.get('dynamics_frame', 'unknown')}, "
                  f"{dyn.numerical_diagnostics.get('nbody_perturber_count', 0.0):.0f} bodies "
                  f"({dyn.numerical_diagnostics.get('nbody_perturbers', 'none')})")
            print("  Integration control:    "
                  f"method={dyn.numerical_diagnostics.get('integrator_method', 'unknown')}, "
                  f"rtol={dyn.numerical_diagnostics.get('integrator_rtol', float('nan')):.1e}, "
                  f"atol={dyn.numerical_diagnostics.get('integrator_atol', float('nan')):.1e}, "
                  f"refreshes={dyn.numerical_diagnostics.get('state_refresh_count', 0.0):.0f}")
            print("  Fidelity terms:         "
                  f"solar_GR={dyn.numerical_diagnostics.get('solar_relativity_enabled', False)}, "
                  f"standard_A1A2={dyn.numerical_diagnostics.get('standard_nongrav_enabled', False)}, "
                  f"phase={dyn.numerical_diagnostics.get('phase_warp_application', 'unknown')}")
            print("  Cascade strength:       "
                  f"median={dyn.numerical_diagnostics.get('cascade_acceleration_au_d2_median', float('nan')):.3e} au/d^2, "
                  f"max={dyn.numerical_diagnostics.get('cascade_acceleration_au_d2_max', float('nan')):.3e} au/d^2")
            print("  Phase warp:             "
                  f"gain={dyn.numerical_diagnostics.get('phase_warp_gain', float('nan')):.3g}, "
                  f"median_acc={dyn.numerical_diagnostics.get('phase_acceleration_au_d2_median', float('nan')):.3e} au/d^2, "
                  f"max_acc={dyn.numerical_diagnostics.get('phase_acceleration_au_d2_max', float('nan')):.3e} au/d^2")
            if dyn.numerical_diagnostics.get("cad_anchor_integrated_rmse_km") is not None:
                print("  CAD anchor residuals:   "
                      f"MAE={dyn.numerical_diagnostics.get('cad_anchor_integrated_mae_km', float('nan')):,.0f} km, "
                      f"RMSE={dyn.numerical_diagnostics.get('cad_anchor_integrated_rmse_km', float('nan')):,.0f} km")
            if dyn.numerical_diagnostics.get("covariance_uncertainty_status") == "ok":
                print("  Covariance cascade:     "
                      f"median cov90={dyn.numerical_diagnostics.get('covariance_width90_km_median', float('nan')):,.1f} km, "
                      f"GI/OI cov90={dyn.numerical_diagnostics.get('gi_oi_cascade_width90_km_median', float('nan')):,.1f} km")
        print(f"  Horizons nearest:       {dyn.nearest_horizons_date}, {dyn.nearest_horizons_distance_au:.9e} au")
        print(f"  Integrated nearest:     {dyn.nearest_integrated_date}, {dyn.nearest_integrated_distance_au:.9e} au")
        print(f"  Integrated min error:   {dyn.nearest_integrated_error_km:,.0f} km")
        if dyn.cad_validation_error_km is not None:
            print(f"  Nearest CAD error:      {dyn.cad_validation_error_km:,.0f} km")
        if dyn.anchor_validation:
            print("  Anchor validation table:")
            print("    CAD epoch              CAD km     Integrated-CAD km  Horizons interp-CAD km")
            for row in dyn.anchor_validation:
                print(
                    f"    {row['cad_date_tdb']:<20s} "
                    f"{row['cad_distance_km']:>10,.0f} "
                    f"{row['integrated_minus_cad_km']:>21,.0f} "
                    f"{row['horizons_interpolated_minus_cad_km']:>24,.0f}"
                )
        print("  Figures:")
        for fig in dyn.figures:
            print(f"    {fig}")
        if dyn.tables:
            print("  Tables:")
            for table in dyn.tables:
                print(f"    {table}")
        if dyn.publication_assets:
            print("  Publication assets:")
            for asset in dyn.publication_assets:
                print(f"    {asset}")
        print("  Caveats:")
        for caveat in dyn.caveats:
            wrapped = textwrap.wrap(caveat, width=82)
            print(f"    - {wrapped[0]}")
            for line in wrapped[1:]:
                print(f"      {line}")
    print("=" * 88)


def default_date_max(years: int = 100) -> str:
    return (dt.date.today() + dt.timedelta(days=365 * years)).isoformat()


def run_self_tests() -> None:
    assert classify_neo_group(0.922, 0.746, 1.099) == "ATE"
    gamma, rule = gamma_from_pdf("ATE", 3.34)
    expected = (
        PDF_SURFACE_GRAVITY["mercury"]
        + PDF_SURFACE_GRAVITY["venus"]
        + PDF_SURFACE_GRAVITY["earth"]
        + PDF_SURFACE_GRAVITY["moon"]
    ) / PDF_SURFACE_GRAVITY["sun"]
    assert abs(gamma - expected) < 1e-15, (gamma, expected, rule)
    assert classify_neo_group(1.1, 0.9, 1.3) == "APO"
    assert classify_neo_group(1.1, 1.1, 1.3) == "AMO"
    assert classify_neo_group(0.8, 0.6, 0.9) == "IEO"
    cascade = _oi_cascade(2.0, 3.0, gamma, terms=3)
    assert abs(cascade[0] - 4.0 * (3.0 * 3.0) * (gamma * gamma) * (2.0**3)) < 1e-12
    assert _likelihood_band(0.25) == "Fly-Bys"
    print("self-tests passed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NASA/JPL NEO data and optionally run the dynamics-first NEO hypothesis predictor as a separated experimental layer.",
    )
    parser.add_argument("--target", default="99942", help="Small-body designation/name understood by JPL SBDB, e.g. 99942 or Apophis.")
    parser.add_argument("--date-min", default="now", help="CAD start date: YYYY-MM-DD or 'now'.")
    parser.add_argument("--date-max", default=default_date_max(), help="CAD end date: YYYY-MM-DD.")
    parser.add_argument("--dist-max", default="0.5", help="CAD maximum Earth approach distance in au, or e.g. 10LD.")
    parser.add_argument("--json", dest="json_path", help="Optional path to write full machine-readable report.")
    parser.add_argument("--dynamics", action="store_true", help="Run the numerical cascade-force propagation branch and generate validation plots.")
    parser.add_argument("--horizons-step", default="1d", help="Horizons vector cadence for --dynamics, e.g. 1d, 12h, 6h.")
    parser.add_argument("--refine-step", default="1h", help="Automatic close-minimum refinement cadence for --dynamics, e.g. 1h, 30m, 15m.")
    parser.add_argument("--refine-window-days", type=float, default=5.0, help="Half-width of each automatic refinement window in days.")
    parser.add_argument("--no-refine", action="store_true", help="Disable automatic Horizons refinement around coarse distance minima.")
    parser.add_argument("--uncertainty-samples", type=int, default=768, help="SBDB covariance clone count for CAD-anchor dynamical uncertainty propagation.")
    parser.add_argument("--cascade-vector-weights", default="1,0,1", help="Velocity,radial,normal weights for the cascade acceleration direction.")
    parser.add_argument("--nbody-bodies", default="mercury,venus,earth,moon,mars,jupiter,saturn,uranus,neptune", help="Comma-separated Horizons perturbers for the dynamics branch, or 'none'.")
    parser.add_argument("--dynamics-frame", default="barycentric", choices=["barycentric", "heliocentric"], help="Integration frame for the dynamics branch.")
    parser.add_argument("--integrator-method", default="DOP853", choices=["DOP853", "RK4"], help="Numerical integrator for dynamics propagation.")
    parser.add_argument("--integrator-max-step-days", type=float, default=0.125, help="Maximum integrator step in days for numerical propagation.")
    parser.add_argument("--integrator-rtol", type=float, default=1e-11, help="Relative tolerance for adaptive DOP853 integration.")
    parser.add_argument("--integrator-atol", type=float, default=1e-13, help="Absolute tolerance for adaptive DOP853 integration in AU/AU-day units.")
    parser.add_argument("--state-refresh-days", type=float, default=0.0, help="Maximum days between Horizons osculating-state refreshes; 0 keeps a single-arc predictor.")
    parser.add_argument("--post-encounter-reset-days", type=float, default=0.0, help="Days after each Horizons range minimum to refresh the osculating state; 0 disables encounter resets for predictor mode.")
    parser.add_argument("--no-solar-relativity", action="store_true", help="Disable solar 1PN relativistic acceleration in the dynamics branch.")
    parser.add_argument("--no-standard-nongrav", action="store_true", help="Disable standard SBDB A1/A2 radial-transverse non-gravitational acceleration.")
    parser.add_argument("--phase-warp-gain", type=float, default=1.0, help="Multiplier for Neo/gravity phasing: RK4 applies a bounded position warp; DOP853 applies continuous acceleration modulation.")
    parser.add_argument("--cascade-accel-au-d2", type=float, default=None, help="Optional cascade acceleration magnitude in au/day^2; defaults to SBDB A1/A2 norm.")
    parser.add_argument(
        "--plot-dir",
        default="/Users/sebastianp/Asteroid-NEO/outputs",
        help="Directory for numerical dynamics validation plots.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run internal non-network formula/classification checks and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        run_self_tests()
        return 0

    try:
        report = analyze(args.target, args.date_min, args.date_max, args.dist_max)
        if args.dynamics:
            dynamics = run_dynamical_propagation(
                report.object,
                report.hypothesis,
                args.target,
                args.date_min,
                args.date_max,
                args.horizons_step,
                Path(args.plot_dir).expanduser(),
                refine_step=args.refine_step,
                refine_window_days=args.refine_window_days,
                refine=not args.no_refine,
                uncertainty_samples=args.uncertainty_samples,
                cascade_vector_weights=args.cascade_vector_weights,
                integrator_max_step_days=args.integrator_max_step_days,
                phase_warp_gain=args.phase_warp_gain,
                cascade_accel_au_d2=args.cascade_accel_au_d2,
                nbody_bodies=args.nbody_bodies,
                dynamics_frame=args.dynamics_frame,
                integrator_method=args.integrator_method,
                integrator_rtol=args.integrator_rtol,
                integrator_atol=args.integrator_atol,
                state_refresh_days=args.state_refresh_days,
                post_encounter_reset_days=args.post_encounter_reset_days,
                include_relativity=not args.no_solar_relativity,
                include_standard_nongrav=not args.no_standard_nongrav,
            )
            report = replace(report, dynamics=dynamics)
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: analysis failed: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    if args.json_path:
        path = Path(args.json_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
