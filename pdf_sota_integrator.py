#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, List, Optional, Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ============================================================================
# 1. HIGH-PRECISION PHYSICAL CONSTANTS
# ============================================================================

try:
    from astropy import constants as _astropy_constants  # type: ignore
    GM_SUN_M3_S2: float = float(_astropy_constants.GM_sun.value)
    C_M_S: float = float(_astropy_constants.c.value)
    AU_METERS: float = float(_astropy_constants.au.value)
except Exception:
    GM_SUN_M3_S2 = 1.32712440018e20
    C_M_S = 299792458.0
    AU_METERS = 149597870700.0

AU_KM: float = AU_METERS / 1000.0
SECONDS_PER_DAY: float = 86400.0
MU_SUN_AU3_D2: float = GM_SUN_M3_S2 * (SECONDS_PER_DAY**2) / (AU_METERS**3)

PLANET_GMS: Dict[str, float] = {
    "1": 4.912491e12, "2": 3.248585e14, "3": 3.986004e14, "301": 4.902800e12,
    "4": 4.282837e13, "5": 1.266865e17, "6": 3.793118e16, "7": 5.793939e15,
    "8": 6.836529e15, "9": 8.71e11
}

SBDB_API: str = "https://ssd-api.jpl.nasa.gov/sbdb.api"
CAD_API: str = "https://ssd-api.jpl.nasa.gov/cad.api"
HORIZONS_API: str = "https://ssd.jpl.nasa.gov/api/horizons.api"

PDF_GRAVITY_MAP: Dict[str, float] = {
    "sun": 275.0, "mercury": 3.7, "venus": 8.87, "earth": 9.81, 
    "moon": 1.62, "mars": 3.71, "jupiter": 24.79, "saturn": 10.44
}

# ============================================================================
# 2. RESILIENT INFRASTRUCTURE
# ============================================================================

class SourceError(RuntimeError): pass

class JsonCache:
    def __init__(self, cache_dir: str = ".api_cache") -> None:
        self.path = Path(cache_dir)
        self.path.mkdir(exist_ok=True)

    def _generate_key(self, url: str, params: Dict[str, Any]) -> str:
        unique_str = f"{url}{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_file = self.path / f"{self._generate_key(url, params)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f: return json.load(f)
            except json.JSONDecodeError: return None
        return None

    def set(self, url: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
        cache_file = self.path / f"{self._generate_key(url, params)}.json"
        with open(cache_file, 'w') as f: json.dump(data, f)

API_CACHE = JsonCache()

def _fetch_jpl_data(url: str, params: Dict[str, Any], retries: int = 3) -> Tuple[Dict[str, Any], str]:
    cached = API_CACHE.get(url, params)
    if cached: return cached, url
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    full_url = f"{url}?{query}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30.0) as response:
                data = json.load(response)
                API_CACHE.set(url, params, data)
                return data, full_url
        except Exception as e:
            if attempt == retries - 1: raise SourceError(f"Network error: {e}")
            time.sleep(2.0 * (attempt + 1))
    return {}, full_url

# ============================================================================
# 3. STRICT DATA ARCHITECTURE
# ============================================================================

@dataclass(frozen=True)
class OrbitElements:
    eccentricity: float
    semi_major_axis_au: float
    perihelion_au: float
    inclination_deg: float
    period_days: float
    aphelion_au: float
    epoch_jd: float | None
    sigma_a_au: float  

@dataclass(frozen=True)
class PhysicalParameters:
    rotation_period_h: float | None
    yarkovsky_a2: float 

@dataclass(frozen=True)
class CloseApproach:
    calendar_date_tdb: str
    jd_tdb: float
    distance_au: float

@dataclass(frozen=True)
class NEOObject:
    designation: str
    fullname: str
    elements: OrbitElements
    physical: PhysicalParameters
    close_approaches: List[CloseApproach]

@dataclass(frozen=True)
class HypothesisTerms:
    jsuncritical: float
    neo_phasing: float
    gravity_neo_phasing: float
    gi_n_raw: float
    oi_n_raw: float
    gamma_ratio: float
    delta_t_s: float
    eccentricity: float
    focal_distance_m: float
    perihelion_m: float

@dataclass(frozen=True)
class EncounterRecord:
    jd_tdb: float
    days_from_start: float
    distance_au: float
    distance_km: float

@dataclass(frozen=True)
class IntegrationResult:
    times_jd: List[float]
    positions_au: np.ndarray
    velocities_au_d: np.ndarray
    pdf_accelerations_au_d2: np.ndarray
    earth_positions_au: np.ndarray
    geocentric_distances_au: np.ndarray
    encounter: Optional[EncounterRecord]
    status: str

@dataclass(frozen=True)
class AnalysisReport:
    generated_utc: str
    target: str
    object: NEOObject
    hypothesis: HypothesisTerms
    jpl_nominal_cad: CloseApproach
    integration_classical: Optional[IntegrationResult] = None
    integration_pdf: Optional[IntegrationResult] = None
    mc_classical_encounters_km: Optional[List[float]] = None
    mc_pdf_encounters_km: Optional[List[float]] = None

# ============================================================================
# 4. HARDENED DATA PARSING & LOADER
# ============================================================================

def load_neo(target: str) -> NEOObject:
    sbdb, _ = _fetch_jpl_data(SBDB_API, {"sstr": target, "full-prec": "1", "phys-par": "1"})
    cad, _ = _fetch_jpl_data(CAD_API, {"des": target, "date-min": "now", "date-max": "2100-01-01", "dist-max": "0.5"})
    
    orb = sbdb.get("orbit", {})
    elem_map = {item.get("name"): item for item in orb.get("elements", [])}
    
    def get_req(key: str) -> float:
        if key not in elem_map or "value" not in elem_map[key]:
            raise SourceError(f"CRITICAL: JPL SBDB missing element '{key}' for {target}.")
        return float(elem_map[key]["value"])
        
    def get_sigma(key: str) -> float:
        return float(elem_map.get(key, {}).get("sigma", 1e-9)) 

    elements = OrbitElements(
        eccentricity=get_req("e"),
        semi_major_axis_au=get_req("a"),
        perihelion_au=get_req("q"),
        inclination_deg=get_req("i"),
        period_days=get_req("per"),
        aphelion_au=get_req("ad"),
        epoch_jd=float(orb["epoch"]) if orb.get("epoch") else None,
        sigma_a_au=get_sigma("a")
    )
    
    phys_map = {p.get("name"): p for p in sbdb.get("phys_par", [])}
    
    approaches = []
    cad_fields = cad.get("fields", [])
    for row in cad.get("data", []):
        rec = dict(zip(cad_fields, row))
        approaches.append(CloseApproach(
            calendar_date_tdb=str(rec.get("cd", "Unknown")),
            jd_tdb=float(rec.get("jd", 0.0)),
            distance_au=float(rec.get("dist", 0.0))
        ))
        
    if not approaches:
        raise SourceError(f"No close approaches found for {target} before 2100.")
    
    # Safely extract Yarkovsky A2, defaulting to 0.0 for unknown objects
    yarkovsky_a2 = 0.0
    if "A2" in phys_map:
        yarkovsky_a2 = float(phys_map["A2"].get("value", 0.0))
    
    return NEOObject(
        designation=target,
        fullname=sbdb.get("object", {}).get("fullname", target),
        elements=elements,
        physical=PhysicalParameters(
            rotation_period_h=float(phys_map.get("rot_per", {}).get("value")) if "rot_per" in phys_map else None,
            yarkovsky_a2=yarkovsky_a2
        ),
        close_approaches=approaches
    )

def fetch_horizons_initial_state(target: str, start_jd: float) -> np.ndarray:
    data, _ = _fetch_jpl_data(HORIZONS_API, {
        "format": "json", "COMMAND": f"'{target}'", "MAKE_EPHEM": "YES", 
        "EPHEM_TYPE": "VECTORS", "CENTER": "500@10", 
        "START_TIME": f"JD{start_jd}", "STOP_TIME": f"JD{start_jd + 1.0}",
        "STEP_SIZE": "1d", "OUT_UNITS": "AU-D", "CSV_FORMAT": "YES"
    })
    
    result = data.get("result", "")
    if "$$SOE" not in result:
        raise SourceError(f"Horizons API failure: $$SOE block missing for {target}. Target vector may not exist at epoch JD {start_jd}.")
        
    lines = result.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    row = next(csv.reader(lines))
    return np.array([float(x) for x in row[2:8]])

# ============================================================================
# 5. THE PDF MATHEMATICAL ENGINE (9th DEGREE CASCADE)
# ============================================================================

class Jet:
    __slots__ = ['coeffs', 'size']
    def __init__(self, coeffs: List[float]) -> None:
        if not coeffs: raise RuntimeError("Jet capacity exhausted.")
        self.coeffs = list(coeffs); self.size = len(self.coeffs)
        
    def _align(self, other: Jet) -> Tuple[List[float], List[float], int]:
        min_size = min(self.size, other.size)
        return self.coeffs[:min_size], other.coeffs[:min_size], min_size
        
    def __add__(self, other: Any) -> Jet:
        if isinstance(other, Jet):
            s_coeffs, o_coeffs, _ = self._align(other)
            return Jet([a + b for a, b in zip(s_coeffs, o_coeffs)])
        res = list(self.coeffs); res[0] += other; return Jet(res)
    def __radd__(self, other: Any) -> Jet: return self + other
        
    def __sub__(self, other: Any) -> Jet:
        if isinstance(other, Jet):
            s_coeffs, o_coeffs, _ = self._align(other)
            return Jet([a - b for a, b in zip(s_coeffs, o_coeffs)])
        res = list(self.coeffs); res[0] -= other; return Jet(res)
    def __rsub__(self, other: Any) -> Jet:
        res = [-a for a in self.coeffs]; res[0] += other; return Jet(res)
        
    def __mul__(self, other: Any) -> Jet:
        if isinstance(other, (int, float)): return Jet([a * other for a in self.coeffs])
        s_coeffs, o_coeffs, min_size = self._align(other)
        res = [0.0] * min_size
        for k in range(min_size):
            s = 0.0
            for j in range(k + 1): s += math.comb(k, j) * s_coeffs[j] * o_coeffs[k - j]
            res[k] = s
        return Jet(res)
    def __rmul__(self, other: Any) -> Jet: return self * other
        
    def __truediv__(self, other: Any) -> Jet:
        if isinstance(other, (int, float)):
            if abs(other) < 1e-35: raise ZeroDivisionError("Jet division by zero.")
            return Jet([a / other for a in self.coeffs])
        s_coeffs, o_coeffs, min_size = self._align(other)
        res = [0.0] * min_size
        g0 = o_coeffs[0]
        if abs(g0) < 1e-35: raise ZeroDivisionError(f"Jet Singularity: {g0}")
        for k in range(min_size):
            s = s_coeffs[k]
            for j in range(1, k + 1): s -= math.comb(k, j) * res[k - j] * o_coeffs[j]
            res[k] = s / g0
        return Jet(res)
        
    def __rtruediv__(self, other: Any) -> Jet:
        res = [0.0] * self.size
        g0 = self.coeffs[0]
        if abs(g0) < 1e-35: raise ZeroDivisionError(f"Jet Singularity: {g0}")
        res[0] = other / g0
        for k in range(1, self.size):
            s = 0.0
            for j in range(1, k + 1): s -= math.comb(k, j) * self.coeffs[j] * res[k - j]
            res[k] = s / g0
        return Jet(res)
    
    def __pow__(self, power: int) -> Jet:
        if not isinstance(power, int) or power < 0: raise TypeError("Jet requires non-negative integer power.")
        if power == 0: return Jet([1.0] + [0.0] * (self.size - 1))
        res = self
        for _ in range(power - 1): res = res * self
        return res
        
    def deriv(self) -> Jet: return Jet(self.coeffs[1:] + [0.0])

def evaluate_hypothesis(neo: NEOObject) -> HypothesisTerms:
    e = neo.elements
    q, a, Q = e.perihelion_au, e.semi_major_axis_au, e.aphelion_au
    group = "AMO" if a > 1.0 and 1.017 < q < 1.3 else "PDF_BASELINE"
    
    ios = sum([PDF_GRAVITY_MAP[k] for k in ["earth", "moon", "mercury", "venus"]]) / PDF_GRAVITY_MAP["sun"]
    gamma = (PDF_GRAVITY_MAP["moon"] + PDF_GRAVITY_MAP["mars"]) / PDF_GRAVITY_MAP["sun"] if group == "AMO" else ios

    U = 2.0 * math.pi * a * AU_METERS / (e.period_days * SECONDS_PER_DAY)
    dt_s, f_dist_m, jsun_m = e.period_days * SECONDS_PER_DAY, (Q - q) * AU_METERS, q * AU_METERS

    poly_u = ((((1.0 * U + 4.0) * U + 12.0) * U + 24.0) * U + 24.0) * U + 24.0
    jcrit = 5.0 * dt_s * e.eccentricity * f_dist_m * gamma / math.sqrt(poly_u)
    
    return HypothesisTerms(
        jsuncritical=jcrit, neo_phasing=1.0e-5, gravity_neo_phasing=2.0e-5,
        gi_n_raw=gamma * (jsun_m * U) / (1.0 + (U / C_M_S)**2),
        oi_n_raw=((0.5 * U * U - GM_SUN_M3_S2 / jsun_m)**4) * (jsun_m * U)**2 * gamma**2 - 1.0,
        gamma_ratio=gamma, delta_t_s=dt_s, eccentricity=e.eccentricity, focal_distance_m=f_dist_m, perihelion_m=jsun_m
    )

# ============================================================================
# 6. HIGH-FIDELITY EPHEMERIS & PHYSICS INTEGRATOR
# ============================================================================

class NBodyData:
    def __init__(self, start_jd: float, days: float) -> None:
        self.raw_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        targets = ["1", "2", "3", "301", "4", "5", "6", "7", "8", "9"]
        
        def _fetch_target(t: str) -> Tuple[str, np.ndarray, np.ndarray, float]:
            data, _ = _fetch_jpl_data(HORIZONS_API, {
                "format": "json", "COMMAND": f"'{t}'", "MAKE_EPHEM": "YES", 
                "EPHEM_TYPE": "VECTORS", "CENTER": "500@10", 
                "START_TIME": f"JD{start_jd - 5.0}", "STOP_TIME": f"JD{start_jd + days + 5.0}", 
                "STEP_SIZE": "1d", "OUT_UNITS": "AU-D", "CSV_FORMAT": "YES"
            })
            lines = data["result"].split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
            jds, pos = [], []
            for row in csv.reader(lines):
                jds.append(float(row[0])); pos.append([float(row[2]), float(row[3]), float(row[4])])
            mu = PLANET_GMS[t] * (SECONDS_PER_DAY**2) / (AU_METERS**3)
            return t, np.array(jds) - start_jd, np.array(pos), mu

        print("    -> Parallel fetching ephemerides for 10 N-Body targets...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_fetch_target, t) for t in targets]
            for future in concurrent.futures.as_completed(futures):
                t, t_rel, pos_arr, mu = future.result()
                self.raw_data[t] = (t_rel, pos_arr, mu)

class NBodyManager:
    def __init__(self, nbody_data: NBodyData) -> None:
        self.bodies = {t: {"mu": mu, "interp": [interp1d(t_rel, pos[:, i], kind='cubic') for i in range(3)]} 
                       for t, (t_rel, pos, mu) in nbody_data.raw_data.items()}

    def get_acceleration(self, t: float, r_ast: np.ndarray) -> np.ndarray:
        acc = np.zeros(3)
        for b in self.bodies.values():
            r_rel = r_ast - np.array([b["interp"][i](t) for i in range(3)])
            acc += -(b["mu"] / max(np.linalg.norm(r_rel)**3, 1e-25)) * r_rel
        return acc

class StateOfTheArtIntegrator:
    def __init__(self, neo: NEOObject, hyp: HypothesisTerms, nbody: NBodyManager, enable_pdf: bool) -> None:
        self.neo, self.hyp, self.nbody, self.enable_pdf = neo, hyp, nbody, enable_pdf
        self.mu_sun, self.c_au_d = MU_SUN_AU3_D2, (C_M_S * SECONDS_PER_DAY) / AU_METERS
        if enable_pdf:
            denom = (hyp.gamma_ratio * hyp.eccentricity * hyp.delta_t_s * hyp.focal_distance_m)
            self.K_c = (hyp.perihelion_m**2) / (denom**2) if denom != 0 else 0.0
            self.rot_c = ((2.0 * math.pi) / ((neo.physical.rotation_period_h or 4.0) / 24.0)) / (2.0 * math.pi)

    def _compute_pdf(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        if not self.enable_pdf or np.linalg.norm(v) < 1e-18: return np.zeros(3)
        U_j = Jet([np.linalg.norm(v) * AU_METERS / SECONDS_PER_DAY, 1.0] + [0.0] * 17)
        A = [None] * 10; A[0] = self.K_c * (U_j**4) - 1.0; A[1] = A[0].deriv()
        for n in range(2, 10): A[n] = ((A[n-1].deriv() / A[n-2].deriv()) - (A[1] if n==2 else A[n-1].deriv())).deriv()
        F_raw = sum(a.coeffs[0] for a in A[1:])
        cp = (self.mu_sun / max(np.linalg.norm(r)**2, 1e-20) * 1e-6) / abs(F_raw) if abs(F_raw) > 1e-35 else 0.0
        return (F_raw * cp * self.rot_c) * (v / np.linalg.norm(v))

    def equations(self, t: float, y: np.ndarray) -> np.ndarray:
        r, v = y[0:3], y[3:6]
        rm, vm = max(np.linalg.norm(r), 1e-20), np.linalg.norm(v)
        a_tot = -(self.mu_sun / (rm**3)) * r
        a_tot += (self.mu_sun / (self.c_au_d**2 * rm**3)) * ((4.0 * self.mu_sun / rm - vm**2 - np.dot(v, r/rm)**2) * r + 4.0 * np.dot(r, v) * v)
        a_tot += self.nbody.get_acceleration(t, r)
        a_tot += self.neo.physical.yarkovsky_a2 * (v / max(vm, 1e-20))
        if self.enable_pdf: a_tot += self._compute_pdf(r, v)
        return np.concatenate((v, a_tot))

    def solve(self, y0: np.ndarray, start_jd: float, days: float, rtol: float=1e-11, atol: float=1e-11) -> IntegrationResult:
        sol = solve_ivp(self.equations, (0, days), y0, method='Radau', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success: return IntegrationResult([], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), None, "Fail")
        te = np.linspace(0, days, int(days * 24))
        se = sol.sol(te); pos, vel = se[0:3, :].T, se[3:6, :].T
        epos = np.array([self.nbody.bodies["3"]["interp"][i](te) for i in range(3)]).T
        gdist = np.linalg.norm(pos - epos, axis=1)
        
        rt = te[int(np.argmin(gdist))]
        res_min = minimize_scalar(lambda tv: float(np.linalg.norm(sol.sol(tv)[0:3] - np.array([self.nbody.bodies["3"]["interp"][i](tv) for i in range(3)]))), 
                                  bounds=(max(0, rt - 0.5), min(days, rt + 0.5)), method='bounded')
        
        pdf_acc = np.array([self._compute_pdf(pos[i], vel[i]) for i in range(len(te))]) if self.enable_pdf else np.zeros_like(pos)
        return IntegrationResult([start_jd + t for t in te], pos, vel, pdf_acc, epos, gdist, EncounterRecord(start_jd+res_min.x, res_min.x, res_min.fun, res_min.fun*AU_KM), "Success")

# ============================================================================
# 7. MONTE CARLO UQ (COVARIANCE-DRIVEN NONLINEAR PROXY)
# ============================================================================

def _mc_worker(args: Tuple[NEOObject, HypothesisTerms, NBodyData, np.ndarray, float, float]) -> Tuple[Optional[float], Optional[float]]:
    neo, hyp, nb_data, y0_pert, sjd, days = args
    nbody = NBodyManager(nb_data)
    cl = StateOfTheArtIntegrator(neo, hyp, nbody, enable_pdf=False).solve(y0_pert, sjd, days, rtol=1e-9, atol=1e-9)
    pd = StateOfTheArtIntegrator(neo, hyp, nbody, enable_pdf=True).solve(y0_pert, sjd, days, rtol=1e-9, atol=1e-9)
    return (cl.encounter.distance_km if cl.encounter else None, pd.encounter.distance_km if pd.encounter else None)

def get_covariance_sigmas(neo: NEOObject, dt_days: float) -> Tuple[float, float]:
    """Dynamically scales Cartesian uncertainty using a nonlinear (dt^2) proxy for long arcs."""
    a = neo.elements.semi_major_axis_au
    da = neo.elements.sigma_a_au
    n = (2 * math.pi) / neo.elements.period_days
    dn = (3/2) * (n / a) * da  
    
    sigma_pos_au = da + (dn * dt_days * a) + 0.5 * a * (dn * dt_days)**2
    sigma_vel_au_d = dn * a 
    
    sigma_pos_au = max(sigma_pos_au, 1.0 / AU_KM)
    sigma_vel_au_d = max(sigma_vel_au_d, (0.01 / 1000.0) * SECONDS_PER_DAY / AU_KM)
    
    return sigma_pos_au, sigma_vel_au_d

def run_monte_carlo_ensemble(neo: NEOObject, hyp: HypothesisTerms, nbody_data: NBodyData, y0: np.ndarray, start_jd: float, days: float, n_runs: int, seed: int = 42) -> Tuple[List[float], List[float]]:
    rng = np.random.default_rng(seed)
    
    dt_days = abs(start_jd - (neo.elements.epoch_jd or start_jd))
    sigma_pos_au, sigma_vel_au_d = get_covariance_sigmas(neo, dt_days)
    
    print(f"    -> UQ Sigmas Derived: Pos = {sigma_pos_au * AU_KM:,.2f} km, Vel = {sigma_vel_au_d * AU_KM / SECONDS_PER_DAY * 1000.0:,.2f} mm/s")
    
    tasks = []
    for _ in range(n_runs):
        y0_p = y0 + np.concatenate([rng.normal(0, sigma_pos_au, 3), rng.normal(0, sigma_vel_au_d, 3)])
        tasks.append((neo, hyp, nbody_data, y0_p, start_jd, days))

    mc_c, mc_p = [], []
    
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for c_dist, p_dist in executor.map(_mc_worker, tasks):
                if c_dist is not None and p_dist is not None:
                    mc_c.append(c_dist)
                    mc_p.append(p_dist)
    except Exception as e:
        print(f"       [!] Parallel execution unavailable ({e}). Falling back to sequential processing...")
        for task in tasks:
            c_dist, p_dist = _mc_worker(task)
            if c_dist is not None and p_dist is not None:
                mc_c.append(c_dist)
                mc_p.append(p_dist)
                
    return mc_c, mc_p

# ============================================================================
# 8. PUBLICATION PLOTTING & REPORTING
# ============================================================================

def generate_publication_figures(report: AnalysisReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf, cl = report.integration_pdf, report.integration_classical
    if not pdf or not cl: return
    
    t_days = np.array(pdf.times_jd) - pdf.times_jd[0]
    fig = plt.figure(figsize=(16, 12), facecolor="white")

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(pdf.positions_au[:,0], pdf.positions_au[:,1], pdf.positions_au[:,2], color="#1d3557", lw=1.5, label="Asteroid Path")
    ax1.plot(pdf.earth_positions_au[:,0], pdf.earth_positions_au[:,1], pdf.earth_positions_au[:,2], color="#2a9d8f", lw=1.0, ls="--", label="Earth Orbit")
    ax1.scatter([0], [0], [0], color="#ffb703", s=100, edgecolors="black", label="Sun")
    ax1.set_title("1. Heliocentric Orbital Context (N-Body)", pad=20); ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    drift = np.linalg.norm((pdf.positions_au - cl.positions_au) * AU_KM, axis=1)
    ax2.plot(t_days, drift, color="#e63946", lw=2.5)
    ax2.fill_between(t_days, 0, drift, color="#e63946", alpha=0.1)
    ax2.set_title("2. Secular Trajectory Drift ($\Delta$ km)"); ax2.set_ylabel("Divergence (km)"); ax2.grid(True, alpha=0.3)
    ax2.annotate(f'Final: {drift[-1]:.2f} km', xy=(t_days[-1], drift[-1]), xytext=(-80, 10), textcoords='offset points', fontweight='bold')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_days, cl.geocentric_distances_au * AU_KM, color="#1d3557", ls="--", label="Classical Baseline")
    ax3.plot(t_days, pdf.geocentric_distances_au * AU_KM, color="#e63946", lw=2, label="PDF Modified")
    ax3.set_title("3. Geocentric Distance Evolution"); ax3.set_ylabel("Distance (km)"); ax3.grid(True, alpha=0.3); ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    idx = int(np.argmin(pdf.geocentric_distances_au))
    window = slice(max(0, idx-5), min(len(t_days), idx+5))
    r_pdf, r_cl = (pdf.positions_au - pdf.earth_positions_au) * AU_KM, (cl.positions_au - cl.earth_positions_au) * AU_KM
    ax4.plot(r_cl[window,0], r_cl[window,1], 'o--', color="#1d3557", alpha=0.4, label="Classical")
    ax4.plot(r_pdf[window,0], r_pdf[window,1], 'o-', color="#e63946", label="PDF Modified")
    ax4.set_title("4. Perigee Zoom"); ax4.set_xlabel("$\Delta$X (km)"); ax4.set_ylabel("$\Delta$Y (km)"); ax4.grid(True, alpha=0.2); ax4.legend()

    fig.tight_layout(pad=4.0)
    fig.savefig(output_dir / "scientific_report_clean.png", dpi=300)
    plt.close(fig)

def generate_monte_carlo_plot(report: AnalysisReport, output_dir: Path) -> None:
    mc_c, mc_p = report.mc_classical_encounters_km, report.mc_pdf_encounters_km
    if not mc_c or not mc_p: return
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.hist(mc_c, bins=15, alpha=0.6, color="#1d3557", edgecolor='black', label="Classical (GR + N-Body) Uncertainty Cloud")
    ax.hist(mc_p, bins=15, alpha=0.6, color="#e63946", edgecolor='black', label="PDF Cascade Uncertainty Cloud")
    
    ax.axvline(np.mean(mc_c), color="#1d3557", ls="dashed", lw=2, label=f"Classical Mean: {np.mean(mc_c):,.1f} km")
    ax.axvline(np.mean(mc_p), color="#e63946", ls="dashed", lw=2, label=f"PDF Mean: {np.mean(mc_p):,.1f} km")
    
    ax.set_title("Monte Carlo Uncertainty Quantification (UQ)\nHazard Assessment Separation Bounds", fontsize=14)
    ax.set_xlabel("Minimum Encounter Distance (km)")
    ax.set_ylabel("Frequency (Ensemble Trajectories)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_monte_carlo_uq.png", dpi=300)
    plt.close(fig)

def generate_geocentric_zoom_plot(report: AnalysisReport, output_dir: Path) -> None:
    pdf, cl = report.integration_pdf, report.integration_classical
    if not pdf or not cl: return
    
    r_pdf, r_cl = (pdf.positions_au - pdf.earth_positions_au) * AU_KM, (cl.positions_au - cl.earth_positions_au) * AU_KM
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.plot(r_cl[:,0], r_cl[:,1], color="#1d3557", ls="--", alpha=0.6, label="Classical Baseline (GR + N-Body)")
    ax.plot(r_pdf[:,0], r_pdf[:,1], color="#e63946", lw=2.5, label="PDF Modified Path")
    
    ax.add_patch(plt.Circle((0, 0), 6371, color='#457b9d', alpha=0.3, label="Earth (Scale)"))
    
    idx = int(np.argmin(pdf.geocentric_distances_au))
    ax.scatter(r_cl[idx,0], r_cl[idx,1], color="#1d3557", s=40, zorder=5)
    ax.scatter(r_pdf[idx,0], r_pdf[idx,1], color="#e63946", s=40, zorder=5)
    
    ax.plot([r_cl[idx,0], r_pdf[idx,0]], [r_cl[idx,1], r_pdf[idx,1]], color="black", lw=1.5, ls="-", label=f"Trajectory Drift ($\Delta$)")

    center_x, center_y = r_pdf[idx,0], r_pdf[idx,1]
    zoom = 7500 
    ax.set_xlim(center_x - zoom, center_x + zoom)
    ax.set_ylim(center_y - zoom, center_y + zoom)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title("Scientific Substantiation: Geocentric Flyby Zoom\nVisualizing the PDF Cascade Drift Delta", fontsize=14)
    ax.set_xlabel("Relative X-Distance from Earth (km)")
    ax.set_ylabel("Relative Y-Distance from Earth (km)")
    ax.legend(loc="upper right")

    final_drift_km = np.linalg.norm(r_pdf[idx] - r_cl[idx])
    ax.annotate(f"Physical Separation ($\Delta$): {final_drift_km:.2f} km",
                xy=(r_pdf[idx,0], r_pdf[idx,1]), xytext=(20, -40),
                textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.2))

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_geocentric_zoom.png", dpi=300)
    plt.close(fig)

def print_report(report: AnalysisReport) -> None:
    print("\n" + "=" * 80)
    print(f"PLANETARY DEFENSE DIAGNOSTIC REPORT: {report.object.fullname}")
    print("=" * 80)
    
    c_res, p_res, jpl = report.integration_classical, report.integration_pdf, report.jpl_nominal_cad
    
    if c_res and p_res and c_res.encounter and p_res.encounter:
        jpl_dist_km = jpl.distance_au * AU_KM
        residual = abs(c_res.encounter.distance_km - jpl_dist_km)
        
        print("[SENTRY VALIDATION CHECK]")
        print(f"  JPL Nominal (CAD API):           {jpl_dist_km:,.2f} km")
        print(f"  Classical Baseline (Script):     {c_res.encounter.distance_km:,.2f} km")
        print(f"  >>> CAD Nominal vs Integrator Discrepancy: {residual:,.2f} km <<<\n")
        
        print("[DETERMINISTIC PREDICTION]")
        print(f"  Classical Distance:              {c_res.encounter.distance_km:,.2f} km")
        print(f"  PDF Cascade Distance:            {p_res.encounter.distance_km:,.2f} km")
        print(f"  >>> TOTAL ENCOUNTER DELTA:       {abs(p_res.encounter.distance_km - c_res.encounter.distance_km):.4f} km <<<\n")
        
        mc_c, mc_p = report.mc_classical_encounters_km, report.mc_pdf_encounters_km
        if mc_c and mc_p:
            print("[MONTE CARLO UNCERTAINTY ENSEMBLE]")
            print(f"  Classical Mean ± 1σ:             {np.mean(mc_c):,.2f} ± {np.std(mc_c):.2f} km")
            print(f"  PDF Modified Mean ± 1σ:          {np.mean(mc_p):,.2f} ± {np.std(mc_p):.2f} km")
    print("=" * 80 + "\n")

# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

def analyze(target: str, days_in: float, days_out: float, n_mc_runs: int) -> AnalysisReport:
    neo = load_neo(target)
    hyp = evaluate_hypothesis(neo)
    
    target_cad = min(neo.close_approaches, key=lambda x: x.distance_au)
    start_jd = target_cad.jd_tdb - days_in
    days_to_propagate = days_in + days_out 
    
    print(f"\n[*] Targeting Encounter on {target_cad.calendar_date_tdb}")
    print(f"[*] Integration Window: -{days_in} to +{days_out} days from CAD.")
    
    nbody_data = NBodyData(start_jd, days_to_propagate)
    nbody_manager = NBodyManager(nbody_data)
    y0 = fetch_horizons_initial_state(neo.designation, start_jd)
    
    print("\n[+] Initiating High-Fidelity Dual Integration Suite...")
    cl = StateOfTheArtIntegrator(neo, hyp, nbody_manager, False).solve(y0, start_jd, days_to_propagate)
    pd = StateOfTheArtIntegrator(neo, hyp, nbody_manager, True).solve(y0, start_jd, days_to_propagate)

    mc_c, mc_p = [], []
    if n_mc_runs > 0:
        mc_c, mc_p = run_monte_carlo_ensemble(neo, hyp, nbody_data, y0, start_jd, days_to_propagate, n_mc_runs)

    return AnalysisReport(dt.datetime.now().isoformat(), target, neo, hyp, target_cad, cl, pd, mc_c, mc_p)

def main() -> int:
    parser = argparse.ArgumentParser(description="State-of-the-Art PDF Hypothesis UQ Integrator.")
    parser.add_argument("--target", default="99942", help="SBDB designation.")
    parser.add_argument("--days-in", type=float, default=30.0, help="Days to propagate before encounter.")
    parser.add_argument("--days-out", type=float, default=10.0, help="Days to propagate after encounter.")
    parser.add_argument("--mc-runs", type=int, default=50, help="Monte Carlo samples.")
    parser.add_argument("--plot-dir", default="./publication_plots", help="Output directory.")
    args = parser.parse_args()

    try:
        report = analyze(args.target, args.days_in, args.days_out, args.mc_runs)
        print_report(report)
        generate_publication_figures(report, Path(args.plot_dir))
        generate_monte_carlo_plot(report, Path(args.plot_dir))
        generate_geocentric_zoom_plot(report, Path(args.plot_dir))
    except Exception as exc:
        print(f"CRITICAL ERROR: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())