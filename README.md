# PF Coil Location Optimization

Optimization study comparing multi-start L-BFGS and Bayesian optimization for placing poloidal field (PF) coils in a tokamak. Test case: DIII-D equilibrium (`g192185.02440`).

## Research Problem

PF coils generate the external magnetic field that shapes and maintains plasma equilibrium. Given a target equilibrium, the question is: **where should the PF coils be placed to best reproduce it?**

Search space: annular region between two plasma-shaped contours in the (R, Z) plane (inner limiter offset, outer vessel wall), parameterized by poloidal angle θ ∈ [10°, 170°] and radial ratio ρ ∈ [0, 1]. Each coil at (R, Z) is mirrored at (R, −Z) to enforce up-down symmetry. Each coil = 3×3 array of 9 filaments (radius `RFIL`).

### Cost Variants

Three cost formulations, in increasing order of fidelity and expense:

**1. Fixed-boundary** (cheap, lstsq-only)
```
Cost = ||C·I − err||² + ω · Σ max(0, d_min − Δθ_ij)²
```
`C` from Green's function evaluations at boundary points, Tikhonov regularization λ on diagonal. Distance penalty on coil pairs with angular separation < `DIST_TH`.

**2. Free-boundary** (expensive, full GS solve)
Boundary distance from TokaMaker Grad-Shafranov solve at each evaluation (~1-2 s/eval).

**3. Combined boundary** (mixed)
```
Cost = (1 − α) · fixed_cost + α · free_cost + dist_penalty
```
Default α = 0.75.

### Parameters Studied

| Parameter | Symbol | Values |
|-----------|--------|--------|
| Number of coil pairs | `NUM_COILS` | 2, 3, 4, 5, 6 |
| Regularization | λ (`REG_IN`) | 1e-5, 1e-6, 1e-7, 1e-8 |
| Free/fixed weight | α | 0.75 (combined) |
| Distance penalty weight | ω (`OMEGA`) | 1e-2 (combined), 1e-7 (fixed) |
| Min angular separation | `DIST_TH` | 5° |
| Filament radius | `RFIL` | 0.01 m |

---

## Methods

### Multi-start L-BFGS
L-BFGS-B with Sobol quasi-random initialization. Each start runs to local convergence. Stops when no improvement over `starts_window` consecutive completed starts, or `MAX_EVALS` / `MAX_TIME` reached.

### Bayesian Optimization
Gaussian process surrogate via `scikit-optimize`. Phase 1: GP-guided sampling until stagnation. Phase 2: top-k candidates from acquisition function refined with L-BFGS. Deduplication by Euclidean distance in real (R, Z) space.

### Brute Force Baseline
2^18 = 262,144 Sobol samples evaluated directly.

---

## Repository Structure

```
opt_comp_convergence.py             Fixed-boundary L-BFGS vs Bayesian (single-process)
opt_comp_convergence_parallel.py    Fixed-boundary, parallel ntrials
opt_comp_free_boundary.py           Free-boundary cost optimization
opt_comp_combined_boundary.py       Combined cost (1-alpha)*fixed + alpha*free, parallel
opt_comp_combined_boundary_k.py     Combined cost variant, weight_fb x ncoils sweep
opt_acq_func_sweep.py               Bayesian acquisition function sweep
brute_force.py                      Sobol brute-force baseline
resume_lbfgs.py                     Resume incomplete L-BFGS runs from checkpoint

OFT_pf_coil_opt_fct.py              CoilPositionSpace + optimization utilities
coil_mapping.py                     Parameter -> geometry mappings (theta-radial, R/Z, polar)
helper_fct.py                       Boundary generation, polygon offset, filament layout
free_boundary_old.py                Legacy free-boundary code

notebooks/
  bay_vs_lbfgs.ipynb                Main comparison: cost, sample efficiency, stats
  bay_kth_pts_refined.ipynb         Effect of top-k L-BFGS refinement
  bay_no_refinement_comparison.ipynb
  acq_func_sweep_analysis.ipynb     Acquisition function comparison
  combined_boundary_analysis.ipynb  Combined-cost results (driven by opt_comp_combined_boundary.py)
  conv_window_analysis.ipynb        Convergence window sensitivity
  conv_window_comparison.ipynb
  curr_error_for_each_lambda.ipynb  Current/error vs lambda
  refinement_candidates.ipynb       Phase-2 candidate inspection
  checkpoint_cost_history.ipynb     Cost-history reconstruction from checkpoints
  alfven_analysis.ipynb
  PF_coil_opt_original.ipynb
```

Results layout:
```
examples/comparisons/
  closed_boundary_DIIID/
  free_boundary_DIIID/
  combined_boundary_DIIID/
    {folder}/lambda:{REG_IN},coils:{NUM_COILS}/run_XX/
      results.json    history, best params, coil positions, currents
      *.png           plots (regenerated from JSON)
examples/data/
  eqdsk/              g192185.02440 input
  mesh/               TokaMaker meshes
```

---

## Dependencies

- [OpenFUSIONToolkit](https://github.com/hansec/OpenFUSIONToolkit) - TokaMaker GS solver, Green's functions
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `scikit-optimize` - Gaussian process optimizer

OFT install path: `~/OpenFUSIONToolkit/install_release` (set via `OFT_ROOTPATH`).

---

## Usage

Single-process fixed-boundary comparison:
```python
from opt_comp_convergence import main
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker

myOFT = OFT_env(nthreads=4)
mygs = TokaMaker(myOFT)
# mesh setup + equilibrium solve

comparison, summary = main(
    mygs=mygs,
    methods=["multistart_lbfgs", "bayesian"],
    NUM_COILS=4,
    REG_IN=1e-7,
    MAX_EVALS=2**18,
    N_RUNS=20,
)
```

Parallel sweeps via CLI:
```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Fixed-boundary, convergence-window sweep
python opt_comp_convergence_parallel.py --nprocs 8 --nthreads 2 --folder convergence_w5

# Combined-boundary (alpha=0.75)
python opt_comp_combined_boundary.py --nprocs 4 --nthreads 2 --ntrials 1 --folder combined

# Acquisition function sweep
python opt_acq_func_sweep.py --acq_funcs EI LCB PI --coils 2 3 4 5 6 --lambdas 1e-8 1e-7 1e-6 1e-5
```

Each run saves to its own `run_XX/` folder. Re-running auto-continues from the next index with a fresh seed offset; runs accumulate safely across sessions.

open: 
export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export OPENBLAS_NUM_THREADS=1 export NUMEXPR_NUM_THREADS=1
python opt_comp_combined_boundary.py --nprocs 12 --nthreads 2 --ntrials 1 --folder convergence_w5_b_temp