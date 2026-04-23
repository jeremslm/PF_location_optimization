# PF Coil Location Optimization

Optimization study comparing multi-start L-BFGS and Bayesian optimization for placing poloidal field (PF) coils in a tokamak. The test case is the DIII-D equilibrium (`g192185.02440`).

## Research Problem

Poloidal field coils generate the external magnetic field that shapes and maintains plasma equilibrium. Given a target fixed-boundary equilibrium, the question is: **where should the PF coils be placed to best reproduce the target boundary flux?**

The search space is the annular region between two plasma-shaped contours in the (R, Z) plane (inner limiter offset and outer vessel wall), parameterized by poloidal angle θ ∈ [10°, 170°] and radial ratio ρ ∈ [0, 1]. Each coil placed at (R, Z) is mirrored at (R, −Z) to enforce up-down symmetry.

### Objective Function

```
Cost = ||C · I − err||² + ω · Σ max(0, d_min − Δθ_ij)²
```

- **Flux error**: least-squares solve for coil currents `I` that minimize mismatch between computed and target boundary flux. The constraint matrix `C` is built from Green's function evaluations at boundary points. Tikhonov regularization with strength λ (`REG_IN`) is added to the diagonal to prevent unrealistically large currents.
- **Distance penalty**: penalizes pairs of coils with angular separation below `DIST_TH`, weighted by `OMEGA`.

Each coil is modeled as a 3×3 array of 9 filaments (radius `RFIL`) to represent finite coil thickness.

### Parameters Studied

| Parameter | Symbol | Values |
|-----------|--------|--------|
| Number of coil pairs | `NUM_COILS` | 2, 3, 4, 5, 6 |
| Regularization strength | λ (`REG_IN`) | 1e-5, 1e-6, 1e-7, 1e-8 |
| Distance penalty weight | ω (`OMEGA`) | 1e-7 (fixed) |
| Min. angular separation | `DIST_TH` | 5° (fixed) |
| Filament radius | `RFIL` | 0.01 m (fixed) |

---

## Methods

### Multi-start L-BFGS
L-BFGS-B with Sobol quasi-random initialization. Each start runs to local convergence. Stops when no improvement is observed over a window of `starts_window` consecutive completed starts, or when `MAX_EVALS` / `MAX_TIME` is reached.

### Bayesian Optimization
Gaussian process surrogate (Expected Improvement acquisition) via `scikit-optimize`. Phase 1 runs GP-guided sampling until stagnation. Phase 2 uses the fitted acquisition function to identify the most promising candidates and refines them with L-BFGS. Deduplication (Chebyshev distance in normalized parameter space) prevents redundant refinements.

### Brute Force Baseline
2^18 = 262,144 Sobol samples evaluated directly, providing a dense coverage baseline to assess the quality of optimized solutions.

---

## Repository Structure

```
opt_comp_convergence.py        Main comparison script (L-BFGS vs Bayesian, fixed-boundary)
opt_comp_asymmetric.py         Variant without up-down symmetry (θ ∈ [0°, 360°])
OFT_pf_coil_opt_fct.py        CoilPositionSpace abstraction and optimization utilities
coil_mapping.py                Parameter-to-geometry mappings (theta-radial, direct R/Z, polar)
helper_fct.py                  Geometry utilities (boundary generation, polygon offset, filament layout)
brute_force.py                 Brute force baseline runner
free_boundary.py               Free-boundary TokaMaker solve using optimized coil positions

notebooks/
  convergence_analysis.ipynb       Main results: cost comparison, sample efficiency, statistical tests
  acq_guided_analysis.ipynb        Acquisition function analysis (Phase 2 refinement)
  bayesian_kth_p_refined.ipynb     Effect of top-k L-BFGS refinement on Bayesian results
  DIIID_free_boundary.ipynb        Free-boundary equilibrium with optimized coil configurations
```

Results are saved under:
```
examples/comparisons/closed_boundary_DIIID/
  convergence/lambda:{REG_IN},coils:{NUM_COILS}/run_XX/
    results.json      convergence history, best parameters, coil positions, currents
    *.png             convergence plots (generated locally from JSON)
  brute_force/lambda:{REG_IN},coils:{NUM_COILS}/
    results.json      best cost from 2^18 Sobol samples
```

---

## Dependencies

- [OpenFUSIONToolkit](https://github.com/hansec/OpenFUSIONToolkit) — TokaMaker solver and Green's function evaluation
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `scikit-optimize` — Gaussian process optimizer

---

## Usage

```python
from opt_comp_convergence import main
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker

myOFT = OFT_env(nthreads=4)
mygs = TokaMaker(myOFT)
# ... mesh setup and equilibrium solve ...

comparison, summary = main(
    mygs=mygs,
    methods=["multistart_lbfgs", "bayesian"],
    NUM_COILS=4,
    REG_IN=1e-7,
    MAX_EVALS=2**18,
    N_RUNS=20,
)
```

Each run is saved to its own `run_XX/` folder. Running the script again with existing results automatically continues from the next index and uses a fresh random seed offset, so runs accumulate safely across multiple sessions.







12: 
export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export OPENBLAS_NUM_THREADS=1 export NUMEXPR_NUM_THREADS=1
python opt_comp_convergence_parallel.py --nprocs 12 --nthreads 2 --folder convergence_w5

open: 
export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export OPENBLAS_NUM_THREADS=1 export NUMEXPR_NUM_THREADS=1
python opt_comp_combined_boundary.py --nprocs 12 --nthreads 2 --ntrials 1 --folder convergence_w5_l_temp

w25: 
export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export OPENBLAS_NUM_THREADS=1 export NUMEXPR_NUM_THREADS=1
python opt_comp_convergence_parallel.py --nprocs 8 --nthreads 2 --folder convergence_w25




export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export OPENBLAS_NUM_THREADS=1 export NUMEXPR_NUM_THREADS=1
python opt_comp_convergence_parallel.py --nprocs 5 --nthreads 2 --folder convergence_w5


export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export OPENBLAS_NUM_THREADS=1 export NUMEXPR_NUM_THREADS=1
python opt_comp_convergence_parallel.py --nprocs 8 --nthreads 2 --folder convergence_w10