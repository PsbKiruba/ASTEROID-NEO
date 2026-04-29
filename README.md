# State-of-the-Art Planetary Defense Integrator

This repository contains a high-fidelity astrodynamics numerical integrator designed for deep-future close approaches of Near-Earth Objects (NEOs). It models the classical point-mass Newtonian baseline and compares it against the proprietary Patented Proportionality Derivative Field (PDF) Hypothesis cascade force.

## 🚀 Core Capabilities
* **N-Body Ephemeris Model:** Dynamically fetches and interpolates the state vectors of 8 planets, the Moon, and Pluto via JPL Horizons.
* **General Relativity (1st-Order PPN):** Computes exact metric curvature corrections for the Solar gravity well, rigorously accounting for the radial-velocity cross-term.
* **Yarkovsky Thermal Drag:** Applies transverse thermal radiation thrust based on JPL SBDB non-gravitational $A_2$ parameters.
* **Automatic Differentiation (Jet Engine):** Evaluates a 9th-degree non-linear PDF cascade derivative entirely in pure Python without expression swell or catastrophic cancellation.
* **Covariance-Driven Monte Carlo UQ:** Generates an uncertainty ensemble utilizing a non-linear secular anomaly runoff proxy ($dt^2$) derived from JPL's 1-sigma orbital uncertainties.

## 🔬 Mathematical Architecture

The integrator utilizes a highly stiff Radau Runge-Kutta solver to integrate the following equations of motion:

### 1. General Relativity (Post-Newtonian Metric)
The 1st-order PPN correction for the Sun's gravity well:
$$\mathbf{a}_{GR}=\frac{\mu_{\odot}}{c^2 r^3}\left[\left(\frac{4\mu_{\odot}}{r}-v^2-(\mathbf{v}\cdot\mathbf{\hat{r}})^2\right)\mathbf{r}+4(\mathbf{r}\cdot\mathbf{v})\mathbf{v}\right]$$

### 2. N-Body Perturbations
Standard Newtonian point-mass gravity for 10 major solar system bodies:
$$\mathbf{a}_{NBody}=-\sum_{i=1}^{10}\mu_i\frac{\mathbf{r}-\mathbf{r}_i}{|\mathbf{r}-\mathbf{r}_i|^3}$$

### 3. Yarkovsky Transverse Thermal Drag
Based on the SBDB unscaled definition, applied along the velocity vector:
$$\mathbf{a}_{Yarkovsky}=A_2\mathbf{\hat{v}}$$

### 4. Keplerian Covariance Proxy (Uncertainty Quantification)
To map the 1-sigma orbital uncertainty ($\Delta a$) to Cartesian coordinate ensembles over long arcs, the model uses a non-linear secular anomaly proxy to capture along-track $dt^2$ growth:
$$\sigma_{pos}\approx\Delta a+a(\Delta n\cdot t)+\frac{1}{2}a(\Delta n\cdot t)^2$$
Where $\Delta n = \frac{3}{2}\frac{n}{a}\Delta a$.

### 5. The Patented PDF Cascade Force
Evaluated using a 1D Truncated Taylor Polynomial Jet:
$$\mathbf{a}_{PDF}=\left(F_{raw}\cdot C_p\cdot \omega_{rot}\right)\mathbf{\hat{v}}$$
Where $F_{raw}$ is the scalar sum of the exact recursive derivatives $A_n = \frac{d}{dt}\left(\frac{A_{n-1}'}{A_{n-2}'}-A_{n-1}'\right)$.

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy scipy matplotlib astropy

Note: astropy is recommended for high-precision physical constants, but the script contains a fallback to CODATA standard constants if it is missing).
🖥️ Usage
Run the integrated pipeline via the CLI. The script automatically handles API fetching, data caching, and multiprocessing.
python pdf_sota_integrator.py --target "99942" --days-in 30.0 --days-out 10.0 --mc-runs 50

Arguments:
•	--target: The JPL SBDB designation of the asteroid (e.g., 99942 or Apophis).
•	--days-in: Days to propagate the integration before the nominal encounter (Default: 30.0).
•	--days-out: Days to propagate after the nominal encounter (Default: 10.0).
•	--mc-runs: The number of parallelized Monte Carlo ensemble universes to simulate for UQ (Default: 50).
•	--plot-dir: Output directory for the generated publication-grade figures (Default: ./publication_plots).