# ASTEROID-NEO

ASTEROID-NEO is an experimental, GPL-licensed numerical study code for near-Earth-object close-approach analysis. The current implementation focuses on 99942 Apophis by default, but the command line accepts any small-body designation that is resolvable by the JPL Small-Body Database and Horizons services.

The code is deliberately framed as a challenger research model. It does not replace JPL Horizons, CNEOS Close Approach Data, Sentry, or any operational orbit-determination pipeline. Its scientific role is to start from authoritative online state data, propagate a transparent force model, overlay the supplied NEO Hypothesis diagnostics, and compare the resulting trajectory against untouched JPL/CAD anchors.

## License

This repository is distributed under the GNU General Public License. See [LICENSE](LICENSE).

## Main File

The active implementation is:

```text
Clude_gen-V5.py
```

The default supported prediction path is the dynamics-first numerical branch enabled by `--dynamics`. It uses no supervised CAD-trained residual correction in the active predictor flow.

## Quick Start

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the internal formula and parser sanity checks:

```bash
python Clude_gen-V5.py --self-test
```

Run the default Apophis diagnostic without the numerical propagator:

```bash
python Clude_gen-V5.py --target 99942 --date-min now --date-max 2030-12-31 --dist-max 0.5
```

Run the full dynamics-first predictor:

```bash
python Clude_gen-V5.py \
  --target 99942 \
  --date-min 2026-04-27 \
  --date-max 2030-12-31 \
  --dist-max 0.5 \
  --dynamics \
  --horizons-step 1d \
  --refine-step 1h \
  --refine-window-days 5 \
  --uncertainty-samples 384 \
  --cascade-vector-weights 1,0,1 \
  --dynamics-frame barycentric \
  --nbody-bodies mercury,venus,earth,moon,mars,jupiter,saturn,uranus,neptune \
  --integrator-method DOP853 \
  --integrator-max-step-days 0.125 \
  --integrator-rtol 1e-11 \
  --integrator-atol 1e-13 \
  --phase-warp-gain 1.0 \
  --plot-dir outputs/predictor_full \
  --json outputs/predictor_full/report.json
```

## Online Data Sources

The code retrieves authoritative input data at runtime:

| Source | Purpose |
|---|---|
| JPL SBDB API | Object identity, orbit elements, physical parameters, covariance, model parameters |
| JPL CAD API | Close-approach anchor table used only for validation overlays |
| JPL Sentry API | Risk-list status reporting |
| JPL Horizons API | Asteroid, Earth, Sun, planet, and Moon vectors for propagation and validation |

No object-specific close-approach values are hardcoded. Physical constants are taken from `astropy` and `scipy` where available, with documented IAU/CODATA fallback constants.

## Scientific Scope

The predictor combines four layers:

1. Authoritative online object state retrieval.
2. Standard NEO classification and close-approach reporting.
3. NEO Hypothesis diagnostics from the supplied hypothesis document.
4. Numerical propagation with barycentric N-body perturbations and optional hypothesis-driven cascade perturbation.

The CAD table is not used as a training label in the active dynamics branch. CAD appears only after propagation as a validation reference.

## Mathematical Background

### NEO Classification

The object group is derived from semimajor axis $a$, perihelion distance $q$, and aphelion distance $Q$:

```text
ATE: a < 1 au and Q > 0.983 au
APO: a >= 1 au and q <= 1.017 au
AMO: 1.017 au < q <= 1.3 au
IEO: Q < 0.983 au
```

The potentially hazardous asteroid rule is evaluated with the usual geometric proxy:

```math
\mathrm{PHA} \approx \left(\mathrm{NEO}\right) \land \left(H \le 22\right) \land \left(\mathrm{MOID} \le 0.05\,\mathrm{au}\right)
```

### PDF Gamma

The hypothesis gamma is not $GM/c^2$. It is a dimensionless ratio selected from the NEO Hypothesis gamma table. For the Apophis Aten, low-inclination case, the implemented rule is:

```math
\gamma =
\frac{g_\mathrm{Mercury} + g_\mathrm{Venus} + g_\mathrm{Earth} + g_\mathrm{Moon}}
     {g_\mathrm{Sun}}
```

where the $g$ values are the surface-gravity values transcribed from the supplied PDF.

### Speed Ratio

The close-approach relative speed is converted into:

```math
\upsilon = \frac{v}{c}
```

where $v$ is the selected CAD relative speed and $c$ is the speed of light.

### Hypothesis Diagnostic Terms

The code preserves the PDF-style diagnostic family as a separated layer. Representative terms include:

```math
J_{\odot,\mathrm{crit}} =
\frac{N\,\Delta T\,e\,f\,\gamma\,U}{\left(1 + \upsilon\right)^2}
```

```math
T_\mathrm{norm} =
\frac{N\,\gamma\,e\,f}{\upsilon\,U}
```

```math
T_\mathrm{cause} =
\frac{\gamma\,e\,f\,\upsilon\,U}{J_{\odot,\mathrm{crit}}}
```

```math
A_\mathrm{cause}^{-1} =
\frac{\gamma\,e\,f\,U}{J_{\odot,\mathrm{crit}}\,\upsilon}
```

```math
\Delta_\mathrm{slip} =
\frac{N\,\gamma\,e\,f\,\upsilon}{U}
```

```math
\Phi_\mathrm{NEO} =
\frac{U\,\upsilon\,\Delta T\,e}{\gamma}
```

The implementation reports range-preserved values over the PDF $N$ bands rather than collapsing every diagnostic into a single unqualified scalar.

### GI/OI Cascade

The code evaluates a GI/OI-style cascade diagnostic using the same PDF gamma. A compact representation of the cascade family is:

```math
\mathrm{OI}_k =
4\,J_\odot^2\,\gamma^2\,U^{k+1}
```

The cascade terms are not declared as accepted celestial mechanics. They are treated as hypothesis-driven diagnostics and, when `--dynamics` is enabled, as an optional perturbation experiment.

### Dynamics State

The numerical state is:

```math
\mathbf{y}(t) =
\begin{bmatrix}
\mathbf{r}(t) \\
\mathbf{v}(t)
\end{bmatrix}
```

with position in au and velocity in au/day.

The integrated equation is:

```math
\frac{d\mathbf{y}}{dt} =
\begin{bmatrix}
\mathbf{v} \\
\mathbf{a}_\mathrm{total}
\end{bmatrix}
```

### Barycentric N-body Gravity

In barycentric mode, the direct perturbing-body acceleration is:

```math
\mathbf{a}_\mathrm{Nbody}
=
\sum_i \mu_i
\frac{\mathbf{r}_i - \mathbf{r}}
     {\lVert \mathbf{r}_i - \mathbf{r} \rVert^3}
```

where each $\mathbf{r}_i$ is retrieved from JPL Horizons.

### Solar Relativistic Correction

The optional solar 1PN correction is applied relative to the Sun:

```math
\mathbf{a}_\mathrm{1PN}
=
\frac{\mu_\odot}{c^2 r^3}
\left[
\left(4\frac{\mu_\odot}{r} - v^2\right)\mathbf{r}
+ 4(\mathbf{r}\cdot\mathbf{v})\mathbf{v}
\right]
```

The code converts this into au/day units internally.

### Standard Non-gravitational A1/A2

When SBDB model parameters are available, radial and transverse non-gravitational terms are applied as:

```math
\mathbf{a}_\mathrm{NG}
=
A_1\,\hat{\mathbf{r}} + A_2\,\hat{\mathbf{t}}
```

where $\hat{\mathbf{r}}$ is the radial unit vector and $\hat{\mathbf{t}}$ is the along-track unit vector.

### Cascade Force Direction

The cascade direction uses user-selectable velocity, radial, and orbital-normal weights:

```math
\hat{\mathbf{d}} =
\frac{
w_v\hat{\mathbf{v}} +
w_r\hat{\mathbf{r}} +
w_n\hat{\mathbf{n}}
}
{
\left\lVert
w_v\hat{\mathbf{v}} +
w_r\hat{\mathbf{r}} +
w_n\hat{\mathbf{n}}
\right\rVert
}
```

The default is:

```math
(w_v,w_r,w_n)=(1,0,1)
```

so the perturbation combines along-velocity and orbital-normal structure.

### Cascade Acceleration

The cascade acceleration is:

```math
\mathbf{a}_\mathrm{cascade}
=
a_c\,\hat{\mathbf{d}}
```

By default, $a_c$ is source-backed by the SBDB A1/A2 acceleration norm. A user can override it with `--cascade-accel-au-d2`, but the recommended mode is to let the code derive it from online SBDB parameters.

### Neo/Gravity Phasing

The phasing term is implemented differently by integrator:

| Integrator | Phasing treatment |
|---|---|
| `RK4` | bounded position warp during the step |
| `DOP853` | continuous acceleration modulation |

The adaptive DOP853 path now applies:

```math
\mathbf{a}_\mathrm{phase}
=
g_\phi \, s_\phi \, \mathbf{a}_\mathrm{cascade}
```

where $g_\phi$ is `--phase-warp-gain` and $s_\phi$ is a bounded scalar derived from the NEO phasing and gravity-phasing diagnostics.

### Total Active Acceleration

For the supported predictor branch:

```math
\mathbf{a}_\mathrm{total}
=
\mathbf{a}_\mathrm{Nbody}
+ \mathbf{a}_\mathrm{1PN}
+ \mathbf{a}_\mathrm{NG}
+ \mathbf{a}_\mathrm{cascade}
+ \mathbf{a}_\mathrm{phase}
```

Terms can be disabled for sensitivity testing, but the full model is the recommended scientific run.

## Predictor Discipline

The default dynamics mode is:

```text
single_arc_predictor
```

That means the integration starts from a Horizons state vector and propagates forward without hidden osculating-state refreshes. The options `--state-refresh-days` and `--post-encounter-reset-days` are available for controlled reconstruction studies, but their default is `0`.

If either refresh option is enabled, the report labels the mode as:

```text
osculating_reconstruction
```

This distinction matters. A reconstruction can look very close to Horizons because it is periodically reset to Horizons. A predictor is the stricter test.

## Implementation Map

### Data Containers

| Name | Purpose |
|---|---|
| `SourceRecord` | Captures API provenance |
| `OrbitElements` | Stores parsed SBDB orbital elements |
| `PhysicalParameters` | Stores diameter, absolute magnitude, MOID, NEO/PHA flags |
| `CloseApproach` | Stores CAD close-approach rows |
| `SentryStatus` | Stores JPL Sentry status |
| `NEOObject` | Combines SBDB, CAD, Sentry, and source records |
| `HypothesisTerms` | Stores gamma, speed ratio, Seq/Traj diagnostics, GI/OI terms |
| `DynamicalPropagationReport` | Stores predictor metrics, plots, tables, caveats |
| `AnalysisReport` | Top-level report object serialized to JSON |

### Online Retrieval

| Subroutine | Annotation |
|---|---|
| `_http_json` | Shared JSON fetcher with API provenance and error reporting |
| `fetch_sbdb` | Retrieves SBDB orbit, physical, covariance, and model-parameter data |
| `fetch_cad` | Retrieves CNEOS close-approach anchors |
| `fetch_sentry` | Retrieves Sentry risk-list state |
| `_horizons_command` | Formats small-body and major-body Horizons commands correctly |
| `fetch_horizons_vectors` | Retrieves vector ephemerides from Horizons |
| `_merge_horizons_vectors` | Merges base and refined vector grids without duplicate epochs |
| `_refinement_center_jds` | Finds close-approach refinement centers from the coarse distance grid |
| `_refine_horizons_near_minima` | Requests high-cadence asteroid vectors around minima |

### Parsing and Object Construction

| Subroutine | Annotation |
|---|---|
| `parse_sbdb` | Converts SBDB JSON into typed orbit and physical records |
| `parse_cad` | Converts CAD rows into typed close-approach objects |
| `parse_sentry` | Converts Sentry JSON into a typed status object |
| `load_neo` | Builds the complete `NEOObject` from all online sources |
| `build_standard_assessment` | Creates the standard NASA/JPL-facing assessment |
| `analyze` | Runs the non-dynamics analysis pipeline |

### Hypothesis Diagnostics

| Subroutine | Annotation |
|---|---|
| `classify_neo_group` | Computes Aten/Apollo/Amor/IEO group |
| `gamma_from_pdf` | Selects the dimensionless PDF gamma rule |
| `calc_jsuncrit` | Computes the PDF-style solar critical diagnostic |
| `calc_time_norm` | Computes the normalized time diagnostic |
| `calc_time_cause` | Computes the time-causality diagnostic |
| `calc_acceleration_cause_inverse` | Computes the inverse acceleration-cause diagnostic |
| `calc_trajectory_slip` | Computes the trajectory-slip diagnostic |
| `calc_trajectory_precession` | Computes the trajectory-precession diagnostic |
| `calc_neo_phasing` | Computes the Neo phasing diagnostic |
| `calc_gravity_neo_phasing` | Computes the gravity phasing diagnostic |
| `calc_bound_factor` | Computes the bound-factor diagnostic |
| `calc_time_slip` | Computes the time-slip diagnostic |
| `calc_lapse_factor` | Computes the lapse-factor diagnostic |
| `_oi_cascade` | Builds the GI/OI cascade sequence |
| `evaluate_hypothesis` | Evaluates and labels all hypothesis diagnostics |

### Numerical Propagation

| Subroutine | Annotation |
|---|---|
| `_parse_vector_weights` | Parses cascade direction weights |
| `_interp_vector` | Interpolates vector ephemerides on the propagation grid |
| `_cascade_direction` | Builds the normalized velocity/radial/normal cascade direction |
| `_cascade_acceleration_au_d2` | Converts cascade strength and direction into acceleration |
| `_phase_warp_displacement_au` | Applies the bounded RK4 position warp |
| `_phase_modulated_acceleration_au_d2` | Applies continuous phasing acceleration for DOP853 |
| `_solar_relativistic_correction_au_d2` | Computes the solar 1PN acceleration |
| `_standard_nongrav_acceleration_au_d2` | Computes SBDB A1/A2 radial-transverse acceleration |
| `_fetch_planetary_perturbers` | Retrieves base and refined Sun/planet/Moon ephemerides |
| `_fetch_earth_position_for_frame` | Retrieves Earth vectors for distance validation |
| `_integrate_cascade_dynamics` | Runs RK4 or DOP853 propagation |
| `run_dynamical_propagation` | Orchestrates the full predictor branch |

### Numerical Utility Routines

| Subroutine | Annotation |
|---|---|
| `_safe_log10_abs` | Computes protected logarithms for wide-range diagnostics |
| `_finite_array` | Converts iterables to finite numerical arrays with fallback handling |
| `_parse_step_days` | Converts Horizons-style step strings into day units |
| `_local_poly_predict_log_distance` | Legacy local polynomial predictor for log-distance experiments |
| `_local_poly_predict_value` | Legacy local polynomial scalar predictor |
| `_estimate_local_curve_minimum` | Estimates a local minimum from sampled range curves |
| `_solve_state_taylor_minimum` | Solves a Taylor-state close-approach minimum estimate |
| `_fit_local_state_osculating_center` | Builds a two-sided local osculating-state fit around an encounter center |
| `_encounter_center_objective` | Scores encounter timing and offset candidates |
| `_build_time_block_partitions` | Creates chronological partitions for leakage-resistant validation experiments |
| `_build_purged_time_splits` | Builds purged chronological train/calibration/validation splits |
| `_r2_score_np` | Computes a NumPy R-squared metric |
| `_median_absolute_error_np` | Computes a NumPy median absolute error metric |
| `_bootstrap_metric_band` | Estimates bootstrap uncertainty bands for scalar metrics |
| `_coverage_by_distance_regime` | Summarizes coverage behavior by distance regime |

### Legacy ML Research Utilities

The active command-line predictor does not expose a supervised ML switch. The source file still contains legacy TensorFlow and feature-engineering routines from earlier research iterations so that old experiments remain auditable. They are not part of the default validated predictor path.

| Subroutine | Annotation |
|---|---|
| `MLSurrogateReport` | Legacy report container for old ML/DL surrogate runs |
| `_tensorflow_available_version` | Checks TensorFlow availability for legacy experiments |
| `_TensorFlowTabularRegressor` | Legacy Keras tabular regressor wrapper |
| `_tensorflow_kernel_ridge_distance_km` | Legacy TensorFlow kernel-ridge distance reconstructor |
| `_tensorflow_encounter_reconstruction` | Legacy continuous-time encounter reconstruction helper |
| `_select_primary_tensorflow_model` | Legacy model-selection helper |
| `_build_local_refinement_design` | Builds encounter-centered legacy feature matrices |
| `_select_local_refinement_model` | Selects legacy local refinement candidates |
| `_build_anchor_validation_rows` | Builds anchor comparison rows used by older synthesis tables |
| `_write_anchor_tables` | Writes anchor-validation tables |
| `_write_publication_tables` | Writes legacy manuscript tables |
| `_write_tensorflow_gate_tables` | Writes legacy TensorFlow gate diagnostics |
| `_ml_figure_caption_library` | Stores legacy ML figure-caption metadata |
| `_build_ml_feature_matrix` | Builds the legacy feature matrix from orbital and hypothesis terms |
| `_write_ml_plots` | Writes legacy ML/DL plots |
| `run_ml_surrogate` | Legacy supervised surrogate entry point retained for auditability, not wired into current CLI |

### Uncertainty and Tables

| Subroutine | Annotation |
|---|---|
| `_sbdb_covariance_nominal_map` | Maps SBDB covariance labels to nominal orbital values |
| `_covariance_clone_matrix` | Samples covariance clones for uncertainty propagation |
| `_propagate_covariance_elements_to_position_au` | Propagates sampled orbital elements to positions |
| `_hypothesis_cascade_clone_score` | Scores clone sensitivity under GI/OI cascade diagnostics |
| `_write_uncertainty_propagation` | Writes covariance and cascade uncertainty tables/plots |
| `_write_dynamics_tables` | Writes propagated time-series and anchor-validation CSVs |
| `_write_dynamics_plots` | Writes distance, residual, and 3D trajectory figures |
| `_write_figure_captions` | Writes manuscript-style figure captions |

### Command-line and Reporting

| Subroutine | Annotation |
|---|---|
| `print_report` | Prints the human-readable terminal report |
| `_json_default` | Serializes dataclasses and paths to JSON |
| `default_date_max` | Supplies default CAD horizon |
| `run_self_tests` | Runs non-network internal sanity tests |
| `parse_args` | Defines the command-line interface |
| `main` | Executes the pipeline and writes JSON if requested |

## Outputs

The dynamics run writes:

| Output | Purpose |
|---|---|
| `report.json` | Machine-readable full report |
| `table_dynamical_integrator_timeseries.csv` | Propagated distance time series |
| `table_dynamical_integrator_anchor_validation.csv` | CAD anchor comparison |
| `table_covariance_gi_oi_cascade_uncertainty.csv` | Covariance and cascade uncertainty summary |
| `fig_dynamical_integrator_distance.*` | Distance curve against Horizons |
| `fig_dynamical_integrator_residuals.*` | Residual diagnostics |
| `fig_dynamical_integrator_residual_structure.*` | Raw residual, linear trend, and detrended residual curvature |
| `fig_dynamical_integrator_close_approach_zoom.*` | Close-approach distance and residual zoom in lunar-distance units |
| `fig_dynamical_integrator_3d_trajectory.*` | 3D trajectory view |
| `fig_public_close_approach_timeline.*` | Public-facing CAD encounter timeline in lunar distances |
| `fig_public_close_approach_dashboard.*` | Four-panel encounter explainer with distance, range rate, geometry, and residuals |
| `fig_public_solar_system_context.*` | Ecliptic Sun-Earth-NEO context map for outreach and society talks |
| `fig_public_force_budget.*` | Acceleration-scale comparison for the active force terms |
| `fig_public_hypothesis_diagnostics.*` | Plain-language view of hypothesis diagnostic magnitudes along the arc |
| `fig_covariance_gi_oi_cascade_anchor_bands.*` | Anchor uncertainty bands |
| `fig_covariance_gi_oi_cascade_diagnostics.*` | GI/OI uncertainty diagnostics |
| `dynamical_integrator_captions.md` | Publication-style captions |
| `covariance_uncertainty_captions.md` | Uncertainty-plot captions |

Figures are saved in PNG, SVG, and PDF form for both inspection and manuscript workflows.

## Current Validation Snapshot

A full Apophis run on 2026-04-29 with barycentric N-body propagation, DOP853, 1-day Horizons cadence, 1-hour close-approach refinement, and no osculating refreshes produced:

| Metric | Value |
|---|---:|
| Prediction mode | `single_arc_predictor` |
| Samples | 2630 |
| Horizons validation MAE | 172,799 km |
| Horizons validation RMSE | 428,051 km |
| CAD anchor MAE | 9,306 km |
| CAD anchor RMSE | 17,838 km |
| 2029 CAD miss | 1,461 km |
| Integrated minimum error vs sampled Horizons minimum | 177 km |

These values are run-dependent because Horizons and SBDB are queried live. They should be reported as reproducibility notes, not as immutable constants.

## Scientific Limitations

This code is not an orbit-determination system. It starts from a Horizons state vector and therefore depends on an authoritative upstream solution.

The NEO Hypothesis terms are experimental diagnostics from the supplied hypothesis document. They are not accepted replacements for standard perturbation modeling, astrometric weighting, or covariance estimation.

CAD anchors are used for validation overlays only in the active predictor branch. They should not be interpreted as training labels or proof of independent operational accuracy.

Prediction quality for another asteroid depends on the availability and quality of SBDB covariance, Horizons vectors, non-gravitational model parameters, and close-approach geometry.

## Applying to Other NEOs

Use any SBDB/Horizons-resolvable designation:

```bash
python Clude_gen-V5.py \
  --target "101955 Bennu" \
  --date-min now \
  --date-max 2050-01-01 \
  --dist-max 0.5 \
  --dynamics \
  --plot-dir outputs/bennu \
  --json outputs/bennu/report.json
```

Recommended scientific practice:

1. Run the diagnostic report first without `--dynamics`.
2. Confirm SBDB orbit condition, covariance availability, NEO group, and Sentry status.
3. Run `--dynamics` with conservative tolerances.
4. Compare CAD anchors only after propagation.
5. Report the predictor mode, force model, step sizes, refinement window, and all caveats.

## Reproducibility Notes

Network access is required. The code depends on public JPL services at runtime.

If macOS or Python certificate handling blocks HTTPS calls, install `certifi` and run:

```bash
SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')" \
python Clude_gen-V5.py --self-test
```

Then use the same `SSL_CERT_FILE` prefix for network runs.

## Repository Hygiene

Generated outputs belong in `outputs/` and are ignored by Git. Commit source code, documentation, and reproducibility metadata. Avoid committing generated plots, API caches, virtual environments, or local notebook checkpoints.

## Citation and Use

When using this repository in a paper or appendix, cite:

1. JPL SBDB for orbit and physical inputs.
2. JPL CAD for close-approach validation anchors.
3. JPL Horizons for vector ephemerides.
4. This GPL repository for the experimental dynamics-first NEO Hypothesis implementation.

Always state that the model is a research challenger and not an operational replacement for NASA/JPL products.
