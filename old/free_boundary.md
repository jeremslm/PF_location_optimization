# Free Boundary Combined Objective Plan

## Goal
Create `opt_comp_combined_boundary.py` with a blended objective:

```
cost = (1 - alpha) * fixed_boundary_cost + alpha * free_boundary_cost
```

Default `alpha = 0.75`. Both costs evaluated at every function call (Option B).

---

## Fixed-boundary cost
Same lstsq computation as in `opt_comp_convergence_parallel.py:951-1004`.
Returns `flux_error_squared` (residual of flux matching at boundary points).
Lambda (`REG_IN`) kept constant — sweep is over `ncoils` and `weight_fb`.

## Free-boundary cost
For each `params` evaluation:
1. `make_new_coils(params, nCoils, coil_center_cand1, coil_center_cand2)` → `scan_geom`
2. `make_mesh(DIIID_geom, scan_geom, savename='mesh_tmp.h5', lim, ...)` → build FEM mesh
3. `TokaMaker(myOFT_fb)` → load mesh → set isoflux + coil regularization (`weight_fb`) → solve GS
4. Save eqdsk → `boundary_distance(fixed_LCFS, free_LCFS, fixed_mag_axis)` → `os.remove`

Returns polar-coordinate radial distance sum between free and fixed LCFS.

Each worker is already in its own `tmp_dir`, so `mesh_tmp.h5` and `gTMP` don't collide across processes.

---

## Regularization parameters
- `REG_IN` (lambda): Tikhonov regularization on the lstsq fixed-boundary solve. Kept constant.
- `weight_fb`: TokaMaker GS regularization on coil currents. Swept alongside `ncoils`.
- These are independent — different solvers, different scales, tuned separately.
- From sweep: `weight_fb` in range `[1e-3, 1e-2]` before shape quality degrades significantly.

---

## Convergence
`starts_window = 5` for both L-BFGS and Bayesian (instead of 25).
Do NOT call `compare_all()` — call methods directly to pass `starts_window=5`.

---

## New file structure: `opt_comp_combined_boundary.py`

### Helpers (defined in new file)
- `boundary_distance(fixed_LCFS, free_LCFS, mag_axis)` — from notebook cell 36
- `make_new_coils(params, nCoils, coil_center_cand1, coil_center_cand2, dx, dy)` — boundaries passed as args
- `make_mesh(DIIID_geom, scan_geom, savename, lim, plasma_dx, coil_dx, vac_dx, vv_dx)`

### `OptimizationComparison`
Rewritten (not imported) with any tracking changes needed for the combined objective.
`objective.last_flux_err` stores `fixed_cost` for compatibility with `_track_objective`.

### `make_combined_objective(..., alpha, weight_fb, ...)`
Factory returning closure over both costs.

### `main(mygs, myOFT_fb, eqdsk, fixed_mag_axis, fixed_LCFS, DIIID_geom, lim, methods, **kwargs)`
Additional kwargs: `ALPHA=0.75`, `WEIGHT_FB=1e-2`.
Output: `examples/comparisons/combined_boundary_DIIID/{RUN_FOLDER}/alpha:{ALPHA},lambda:{REG_IN},coils:{NUM_COILS}/`

### `parallel_case(reg_in, num_coils, ntrials, run_folder, nthreads, alpha, weight_fb)`
Extends existing parallel_case with `alpha` and `weight_fb`. Same `myOFT` reused for free-boundary.

### CLI
Same as `opt_comp_convergence_parallel.py` + `--alpha` + `--weight_fb`.

---

## Sweep plan
Keep `REG_IN` fixed (pick from lambda sweep analysis — see `notebooks/lambda_current_sweep.ipynb`).
Sweep: `ncoils` × `weight_fb`.

---

## Verification
- `alpha=0.0` → identical to `opt_comp_convergence_parallel.py` results
- `alpha=1.0` → optimizer guided purely by free-boundary distance
- Print both cost components in first few evals to confirm both are computed
