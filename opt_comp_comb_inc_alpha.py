"""
Memory-efficient Bayesian sweep for combined-boundary optimization with an
increasing objective blend ALPHA.

Self-dispatching:
  --mode orchestrator (default): Pool of N watchdogs; each watchdog drives one
    (weight_fb, num_coils) case via repeated subprocess calls to itself in
    --mode chunk. Process exit between chunks reclaims OFT/Fortran heap so peak
    RAM stays bounded. Cross-case parallelism only; a case never runs two
    chunks at once.
  --mode chunk: Runs one segment for one case. The checkpoint 'phase' field
    decides which segment:
      bayesian: build a fresh skopt Optimizer from the checkpointed (X, y),
        run BAYES_ITERS_PER_CHUNK ask/tell iterations, save, exit.
      refine:   memory-efficient multistart L-BFGS over the best Bayesian
        acquisition candidates at the final ALPHA.

The objective blend ALPHA rises across the run (alpha_schedule). Because the
raw cost components (flux_err, fb_cost, dist_penalty) do not depend on ALPHA,
every past observation's blended y is recomputed under the current ALPHA when
the surrogate is rebuilt - no physics re-run. Rebuilding the Optimizer from
(X, recomputed y) is the surrogate reinitialization.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OMEGA = 1e-2
DIST_TH = 5.0
RFIL = 0.01
CONVERGENCE_THRESHOLD = 0.01
LBFGS_MAXFUN = 100
MAX_RETRIES = 3
CHUNK_TIMEOUT_S = 14400

BAYES_ITERS_PER_CHUNK = 50
STAGNATION_WINDOW = 25
N_INITIAL_PER_COIL = 25
N_SOBOL_POOL = 8192
ACQ_MULTIPLIER = 10
UNIQUE_REFINED_POINTS = 3
ACQ_DEDUP_TOL = 0.05
REFINE_WINDOW = 5
STARTS_PER_REFINE_CHUNK = 1

ALPHA_MIN = 0.0
ALPHA_MAX = 1.0


def alpha_schedule(n_evals):
    """Objective blend ALPHA for the current evaluation count.

    Called once per chunk; ALPHA is held constant within a chunk and bumped at
    each chunk boundary (the reinit cadence). Should rise from ALPHA_MIN toward
    ALPHA_MAX as n_evals grows.
    """
    # TODO(human): return the blend ALPHA as a function of n_evals.
    raise NotImplementedError("alpha_schedule not yet implemented")


def _case_base(folder, w, reg_in, c):
    return os.path.join(_BASE_DIR,
        f'examples/comparisons/combined_boundary_DIIID/{folder}/'
        f'weight:{w:.0e},lambda:{reg_in:.0e},coils:{c}')


def _find_or_create_run_dir(base):
    os.makedirs(base, exist_ok=True)
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isdir(d) and name.startswith('run_'):
            if os.path.exists(os.path.join(d, 'results.json')):
                continue
            ckpt = os.path.join(d, 'checkpoint.json')
            return d, os.path.exists(ckpt)
    idx = 1
    while os.path.exists(os.path.join(base, f'run_{idx:02d}')):
        idx += 1
    new_dir = os.path.join(base, f'run_{idx:02d}')
    os.makedirs(new_dir)
    return new_dir, False


def _has_completed_run(base):
    if not os.path.isdir(base):
        return False
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if os.path.isdir(d) and name.startswith('run_'):
            if os.path.exists(os.path.join(d, 'results.json')):
                return True
    return False


def _build_physics_isolated(nthreads, tmp_suffix):
    import shutil
    import numpy as np
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    from helper_fct import update_boundary

    eqdsk = read_eqdsk(os.path.join(_BASE_DIR, 'examples/data/eqdsk/DIIID_opt_3coil_symm'))
    LCFS_contour = eqdsk['rzout'].copy()
    fixed_LCFS = LCFS_contour
    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    tmp_dir = os.path.join(_BASE_DIR, 'tmp', f'inc_alpha_{tmp_suffix}')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)

    mesh_dx = 0.015
    gs_mesh = gs_Domain()
    gs_mesh.define_region('plasma', mesh_dx, 'plasma')
    gs_mesh.add_polygon(LCFS_contour, 'plasma')
    mesh_pts, mesh_lc, _ = gs_mesh.build_mesh()

    myOFT = OFT_env(nthreads=nthreads)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False
    F0 = eqdsk['rcentr'] * eqdsk['bcentr']
    mygs.setup(order=2, F0=F0)
    mygs.set_targets(Ip=eqdsk['ip'], pax=eqdsk['pres'][0])
    mygs.init_psi()
    mygs.solve()

    fixed_mag_axis = np.array([1.77764093, -0.04014656])
    os.chdir(_BASE_DIR)
    return myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim


def _dist_penalty(x, num_coils):
    import numpy as np
    thetas = np.asarray(x[:num_coils])
    return OMEGA * np.sum(np.maximum(DIST_TH - np.diff(np.sort(thetas)), 0.0) ** 2)


def _blend(fixed_raw, fb_raw, failed, dist, alpha, norm_fixed, norm_fb):
    """Recompute the blended objective y from raw components under any ALPHA."""
    norm_fixed_term = fixed_raw / norm_fixed if norm_fixed and norm_fixed > 0 else fixed_raw
    if failed:
        norm_fb_term = 1.0
    elif norm_fb and norm_fb > 0:
        norm_fb_term = fb_raw / norm_fb
    else:
        norm_fb_term = fb_raw
    return (1 - alpha) * norm_fixed_term + alpha * norm_fb_term + dist


def _problem_setup(c, reg_in, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, mygs, w):
    import numpy as np
    from helper_fct import resize_polygon, update_boundary
    from OFT_pf_coil_opt_fct import CoilPositionSpace
    from opt_comp_combined_boundary import make_combined_objective

    r_bnd, psi_bnd = mygs.get_vfixed()
    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    coil_space = CoilPositionSpace(inner_boundary=coil_center_cand1,
                                   outer_boundary=coil_center_cand2, method='coords')
    coil_space.set_bounds(angular_bounds=(10, 170), radial_bounds=(0, 1))
    theta_bounds, radial_bounds = coil_space.get_bounds()
    bounds = [theta_bounds] * c + [radial_bounds] * c

    theta_range = np.linspace(0, 180, len(coil_center_cand1) // 2)
    inner = coil_center_cand1[:len(coil_center_cand1) // 2]
    outer = coil_center_cand2[:len(coil_center_cand2) // 2]
    return (r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
            bounds, theta_range, inner, outer)


# ============================================
# Bayesian chunk
# ============================================

def bayesian_chunk(args, run_dir, ckpt_path, ckpt, physics):
    import numpy as np
    from scipy.stats import qmc
    from skopt import Optimizer
    from skopt.space import Real

    from opt_comp_combined_boundary import (
        OptimizationComparison, make_combined_objective, FBCostSlowException, FB_COST_MAX_S,
    )
    from OpenFUSIONToolkit.TokaMaker.util import eval_green

    w, c, reg_in = args.weight, args.coil, args.reg_in
    myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim = physics
    (r_bnd, psi_bnd, cc1, cc2, bounds, theta_range, inner, outer) = _problem_setup(
        c, reg_in, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, mygs, w)
    n_params = len(bounds)
    n_initial = N_INITIAL_PER_COIL * c
    random_state = args.random_state

    # restore or init state
    if ckpt is not None:
        bayes_x = [list(p) for p in ckpt['bayes_x']]
        bayes_fixed = list(ckpt['bayes_fixed'])
        bayes_fb = list(ckpt['bayes_fb'])
        bayes_failed = list(ckpt['bayes_failed'])
        bayes_dist = list(ckpt['bayes_dist'])
        bayes_alpha = list(ckpt['bayes_alpha'])
        norm_fixed = ckpt['norm_fixed']
        norm_fb = ckpt['norm_fb']
        cost_history = list(ckpt['cost_history'])
        times = list(ckpt['times'])
        flux_err_history = list(ckpt['flux_err_history'])
        fb_cost_history = list(ckpt['fb_cost_history'])
        fb_failures = ckpt['fb_failures']
        elapsed_offset = ckpt.get('elapsed', 0.0)
    else:
        bayes_x, bayes_fixed, bayes_fb, bayes_failed = [], [], [], []
        bayes_dist, bayes_alpha = [], []
        norm_fixed, norm_fb = None, None
        cost_history, times = [], []
        flux_err_history, fb_cost_history = [], []
        fb_failures = 0
        elapsed_offset = 0.0

    n_evals = len(bayes_x)
    start_time = time.time() - elapsed_offset
    norm_ready = norm_fixed is not None and norm_fb is not None
    alpha = alpha_schedule(n_evals)
    print(f"[bayes] {'resume' if ckpt else 'fresh'} {run_dir} n_evals={n_evals} alpha={alpha:.4f}", flush=True)

    objective = make_combined_objective(
        alpha, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, cc1, cc2, lim,
        r_bnd, psi_bnd, w, c, RFIL, reg_in, OMEGA, DIST_TH, theta_range, inner, outer)

    # uniform y for every point under the current chunk ALPHA
    def recompute_y():
        return [_blend(bayes_fixed[i], bayes_fb[i], bayes_failed[i], bayes_dist[i],
                       alpha, norm_fixed, norm_fb) for i in range(len(bayes_x))]

    space = [Real(low, high) for low, high in bounds]
    gp_opt = Optimizer(space, base_estimator='gp', n_initial_points=n_initial,
                       initial_point_generator='sobol', acq_func='EI',
                       random_state=random_state)
    if norm_ready and bayes_x:
        gp_opt.tell([list(p) for p in bayes_x], recompute_y())

    sampler = qmc.Sobol(d=n_params, scramble=True, seed=random_state)
    sobol_samples = sampler.random(N_SOBOL_POOL)
    sobol_starts = [[low + sobol_samples[i, j] * (high - low)
                     for j, (low, high) in enumerate(bounds)]
                    for i in range(N_SOBOL_POOL)]

    def evaluate(x):
        nonlocal fb_failures
        objective.norm_fixed = None
        objective.norm_fb = None
        objective.fb_failures = 0
        objective(np.asarray(x))
        fixed_raw = float(objective.last_flux_err)
        failed = objective.fb_failures > 0
        fb_raw = None if failed else float(objective.last_fb_cost)
        if failed:
            fb_failures += 1
        timing = objective.last_fb_timing
        if timing is not None and timing['total'] > FB_COST_MAX_S:
            raise FBCostSlowException(f"fb_cost total={timing['total']:.1f}s exceeds {FB_COST_MAX_S:.1f}s")
        return fixed_raw, fb_raw, failed

    stopped_by = None
    chunk_done = 0
    while chunk_done < BAYES_ITERS_PER_CHUNK:
        if time.time() - start_time > args.max_time:
            stopped_by = "exceeded wall time"
            break
        if n_evals >= args.max_evals:
            stopped_by = "max function calls"
            break
        if n_evals < n_initial or not norm_ready:
            x = list(sobol_starts[n_evals])
        else:
            x = gp_opt.ask()
        try:
            fixed_raw, fb_raw, failed = evaluate(x)
        except FBCostSlowException as e:
            stopped_by = "fb_cost_slow"
            print(f"[bayes] {e}; checkpoint and respawn", flush=True)
            break
        dist = float(_dist_penalty(x, c))
        bayes_x.append(list(x))
        bayes_fixed.append(fixed_raw)
        bayes_fb.append(fb_raw)
        bayes_failed.append(failed)
        bayes_dist.append(dist)
        bayes_alpha.append(alpha)
        flux_err_history.append(fixed_raw)
        fb_cost_history.append(fb_raw if not failed else norm_fb)
        n_evals += 1
        chunk_done += 1
        times.append(time.time() - start_time)

        if not norm_ready and not failed:
            norm_fixed = fixed_raw
            norm_fb = fb_raw
            norm_ready = True
            gp_opt.tell([list(p) for p in bayes_x], recompute_y())
            print(f"[bayes] norm set at eval {n_evals}: norm_fixed={norm_fixed:.4e} norm_fb={norm_fb:.4e}", flush=True)
        elif norm_ready:
            y = _blend(fixed_raw, fb_raw, failed, dist, alpha, norm_fixed, norm_fb)
            gp_opt.tell(list(x), y)

        if norm_ready:
            y_all = recompute_y()
            cost_history.append(y_all[-1])
            running_min = np.minimum.accumulate(y_all)
            if n_evals % 100 == 0:
                print(f"[bayes] eval={n_evals} alpha={alpha:.4f} best={running_min[-1]:.4e}", flush=True)
            if n_evals > n_initial + STAGNATION_WINDOW:
                old = running_min[-(STAGNATION_WINDOW + 1)]
                new = running_min[-1]
                if abs(old) > 0 and (old - new) / abs(old) < CONVERGENCE_THRESHOLD:
                    stopped_by = "bayesian_stagnation"
                    break
                if old == 0 and new == 0:
                    stopped_by = "bayesian_stagnation"
                    break
        else:
            cost_history.append(None)

    if stopped_by is None:
        stopped_by = "chunk_budget"

    # persist
    elapsed = time.time() - start_time
    state = {
        'phase': 'bayesian',
        'optimization_settings': {'num_coils': int(c), 'weight_fb': float(w),
                                  'reg_in': float(reg_in)},
        'n_evals': n_evals,
        'elapsed': elapsed,
        'random_state': random_state,
        'norm_fixed': norm_fixed,
        'norm_fb': norm_fb,
        'fb_failures': fb_failures,
        'bayes_x': bayes_x,
        'bayes_fixed': bayes_fixed,
        'bayes_fb': bayes_fb,
        'bayes_failed': bayes_failed,
        'bayes_dist': bayes_dist,
        'bayes_alpha': bayes_alpha,
        'cost_history': cost_history,
        'times': times,
        'flux_err_history': flux_err_history,
        'fb_cost_history': fb_cost_history,
    }

    if stopped_by in ("chunk_budget", "fb_cost_slow"):
        with open(ckpt_path, 'w') as f:
            json.dump(state, f)
        print(f"[bayes] stopped_by={stopped_by} n_evals={n_evals}; respawn", flush=True)
        return 0

    # terminal: stagnation -> hand off to refine; walltime/maxevals -> finalize now
    if stopped_by == "bayesian_stagnation":
        n_acq = ACQ_MULTIPLIER * UNIQUE_REFINED_POINTS
        raw_candidates = gp_opt.ask(n_points=n_acq, strategy='cl_min')
        cmp_dedup = OptimizationComparison(objective, bounds, NUM_COILS=c)
        cmp_dedup.set_problem_data(r_bnd, psi_bnd, cc1, cc2, mygs.o_point, eval_green)
        candidates = cmp_dedup._deduplicate_candidates(raw_candidates, tol=ACQ_DEDUP_TOL,
                                                       max_unique=UNIQUE_REFINED_POINTS)
        state['phase'] = 'refine'
        state['refinement_candidates'] = [list(c_) for c_ in candidates]
        state['refine_current'] = 0
        state['starts_completed'] = 0
        state['start_costs'] = []
        with open(ckpt_path, 'w') as f:
            json.dump(state, f)
        print(f"[bayes] converged; {len(candidates)} refine candidates; phase->refine", flush=True)
        return 0

    _finalize_from_state(state, stopped_by, run_dir, args, physics, alpha)
    return 0


# ============================================
# Refinement chunk
# ============================================

def refine_chunk(args, run_dir, ckpt_path, ckpt, physics):
    import numpy as np
    from scipy.optimize import minimize

    from opt_comp_combined_boundary import (
        OptimizationComparison, make_combined_objective, _check_starts_convergence,
        TimeoutException, MaxEvalsException, FBCostSlowException, FB_COST_MAX_S,
    )
    from OpenFUSIONToolkit.TokaMaker.util import eval_green

    w, c, reg_in = args.weight, args.coil, args.reg_in
    myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim = physics
    (r_bnd, psi_bnd, cc1, cc2, bounds, theta_range, inner, outer) = _problem_setup(
        c, reg_in, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, mygs, w)

    alpha = ALPHA_MAX
    objective = make_combined_objective(
        alpha, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, cc1, cc2, lim,
        r_bnd, psi_bnd, w, c, RFIL, reg_in, OMEGA, DIST_TH, theta_range, inner, outer)
    objective.norm_fixed = ckpt['norm_fixed']
    objective.norm_fb = ckpt['norm_fb']
    objective.fb_failures = ckpt['fb_failures']

    comparison = OptimizationComparison(
        objective, bounds, max_time=args.max_time, max_evals=args.max_evals,
        convergence_threshold=CONVERGENCE_THRESHOLD, NUM_COILS=c, OMEGA=OMEGA,
        DIST_TH=DIST_TH, REG_IN=reg_in, RFIL=RFIL, ALPHA=alpha, WEIGHT_FB=w, verbose=True)
    comparison.set_problem_data(r_bnd, psi_bnd, cc1, cc2, mygs.o_point, eval_green,
                                myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim)
    comparison.checkpoint_path = None
    comparison._reset_tracking()
    comparison._current_method = 'Bayesian'
    comparison.fb_cost_max_s = FB_COST_MAX_S

    # restore merged history from the Bayesian phase
    comparison._n_evals = ckpt['n_evals']
    comparison._history = list(ckpt['cost_history'])
    comparison._times = list(ckpt['times'])
    comparison._flux_err_history = list(ckpt['flux_err_history'])
    comparison._fb_cost_history = list(ckpt['fb_cost_history'])
    comparison._fb_failures = ckpt['fb_failures']
    comparison._initial_fixed_cost = ckpt['norm_fixed']
    comparison._initial_fb_cost = ckpt['norm_fb']
    bayes_y = [v for v in ckpt['cost_history'] if v is not None]
    comparison._convergence = list(np.minimum.accumulate(bayes_y)) if bayes_y else []
    comparison._best_cost = float(min(bayes_y)) if bayes_y else float('inf')
    comparison._best_params = np.array(ckpt['bayes_x'][int(np.argmin(bayes_y))]) if bayes_y else None
    elapsed_offset = ckpt.get('elapsed', 0.0)
    comparison._start_time = time.time() - elapsed_offset

    candidates = [np.array(c_) for c_ in ckpt['refinement_candidates']]
    cur = ckpt['refine_current']
    starts_completed = ckpt['starts_completed']
    start_costs = list(ckpt['start_costs'])

    stopped_by = None
    chunk_done = 0
    for cand in candidates[cur:]:
        if chunk_done >= STARTS_PER_REFINE_CHUNK:
            stopped_by = "chunk_budget"
            break
        try:
            cur += 1
            minimize(comparison._track_objective, cand, method='L-BFGS-B', bounds=bounds,
                     options={'ftol': 1e-9, 'gtol': 1e-6, 'maxfun': LBFGS_MAXFUN, 'disp': False})
            starts_completed += 1
            start_costs.append(comparison._best_cost)
            chunk_done += 1
            if _check_starts_convergence(start_costs, REFINE_WINDOW, CONVERGENCE_THRESHOLD):
                stopped_by = "converged"
                break
        except TimeoutException:
            stopped_by = "exceeded wall time"
            break
        except MaxEvalsException:
            stopped_by = "max function calls"
            break
        except FBCostSlowException as e:
            stopped_by = "fb_cost_slow"
            print(f"[refine] {e}; checkpoint and respawn", flush=True)
            break
    else:
        stopped_by = "all refinements completed"

    ckpt['refine_current'] = cur
    ckpt['starts_completed'] = starts_completed
    ckpt['start_costs'] = start_costs
    ckpt['n_evals'] = comparison._n_evals
    ckpt['cost_history'] = list(comparison._history)
    ckpt['times'] = list(comparison._times)
    ckpt['flux_err_history'] = list(comparison._flux_err_history)
    ckpt['fb_cost_history'] = list(comparison._fb_cost_history)
    ckpt['elapsed'] = time.time() - comparison._start_time

    if stopped_by in ("chunk_budget", "fb_cost_slow"):
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)
        print(f"[refine] stopped_by={stopped_by} starts={starts_completed}; respawn", flush=True)
        return 0

    _finalize_comparison(comparison, stopped_by, run_dir)
    return 0


# ============================================
# Finalization
# ============================================

def _finalize_from_state(state, stopped_by, run_dir, args, physics, alpha):
    """Finalize a Bayesian-only run (no refinement) directly from checkpoint state."""
    import numpy as np
    from opt_comp_combined_boundary import OptimizationComparison, make_combined_objective
    from OpenFUSIONToolkit.TokaMaker.util import eval_green

    w, c, reg_in = args.weight, args.coil, args.reg_in
    myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim = physics
    (r_bnd, psi_bnd, cc1, cc2, bounds, theta_range, inner, outer) = _problem_setup(
        c, reg_in, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, mygs, w)
    objective = make_combined_objective(
        alpha, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, cc1, cc2, lim,
        r_bnd, psi_bnd, w, c, RFIL, reg_in, OMEGA, DIST_TH, theta_range, inner, outer)
    comparison = OptimizationComparison(
        objective, bounds, max_time=args.max_time, max_evals=args.max_evals,
        convergence_threshold=CONVERGENCE_THRESHOLD, NUM_COILS=c, OMEGA=OMEGA,
        DIST_TH=DIST_TH, REG_IN=reg_in, RFIL=RFIL, ALPHA=alpha, WEIGHT_FB=w, verbose=True)
    comparison.set_problem_data(r_bnd, psi_bnd, cc1, cc2, mygs.o_point, eval_green,
                                myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim)
    comparison._reset_tracking()
    comparison._current_method = 'Bayesian'
    comparison._n_evals = state['n_evals']
    comparison._history = list(state['cost_history'])
    comparison._times = list(state['times'])
    comparison._flux_err_history = list(state['flux_err_history'])
    comparison._fb_cost_history = list(state['fb_cost_history'])
    comparison._fb_failures = state['fb_failures']
    comparison._initial_fixed_cost = state['norm_fixed']
    comparison._initial_fb_cost = state['norm_fb']
    bayes_y = [v for v in state['cost_history'] if v is not None]
    comparison._convergence = list(np.minimum.accumulate(bayes_y)) if bayes_y else []
    comparison._best_cost = float(min(bayes_y)) if bayes_y else float('inf')
    comparison._best_params = np.array(state['bayes_x'][int(np.argmin(bayes_y))]) if bayes_y else None
    _finalize_comparison(comparison, stopped_by, run_dir)


def _finalize_comparison(comparison, stopped_by, run_dir):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    flux = [v for v in comparison._flux_err_history if v is not None]
    fb = [v for v in comparison._fb_cost_history if v is not None]
    thetas, radials, coil_positions, coil_currents = comparison._extract_best_result()
    comparison.results['Bayesian'] = {
        'best_cost': comparison._best_cost,
        'best_flux_err': float(min(flux)) if flux else None,
        'best_fb_cost': float(min(fb)) if fb else None,
        'initial_fixed_cost': comparison._initial_fixed_cost,
        'initial_fb_cost': comparison._initial_fb_cost,
        'best_params': comparison._best_params,
        'n_evals': comparison._n_evals,
        'time': comparison._times[-1] if comparison._times else 0.0,
        'times': list(comparison._times),
        'stopping': stopped_by,
        'parameters': {'thetas': thetas, 'radials': radials},
        'coil_positions_top': coil_positions,
        'coil_currents': coil_currents,
        'convergence_history': list(comparison._convergence),
        'cost_history': list(comparison._history),
        'fb_failures': comparison._fb_failures,
        'flux_err_history': list(comparison._flux_err_history),
        'fb_cost_history': list(comparison._fb_cost_history),
    }
    comparison.save_results_to_json(os.path.join(run_dir, 'results.json'))
    fig = comparison.plot_result()
    time_fig = comparison.plot_convergence_vs_time(log_scale=True)
    fig.savefig(os.path.join(run_dir, 'convergence_plot.png'), dpi=150, bbox_inches='tight')
    time_fig.savefig(os.path.join(run_dir, 'convergence_vs_time_plot.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"[finalize] stopped_by={stopped_by} wrote results to {run_dir}", flush=True)


# ============================================
# Chunk dispatch
# ============================================

def chunk_main(args):
    import shutil

    base = _case_base(args.folder, args.weight, args.reg_in, args.coil)
    run_dir, resuming = _find_or_create_run_dir(base)
    ckpt_path = os.path.join(run_dir, 'checkpoint.json')

    import opt_comp_combined_boundary as _ocb
    _ocb._MEM_LOG_DIR = run_dir
    _ocb._get_mem_logger()

    ckpt = None
    if resuming and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
    phase = ckpt['phase'] if ckpt is not None else 'bayesian'

    tmp_suffix = f"{os.getpid()}_{args.weight:.0e}_{args.coil}"
    tmp_dir_path = os.path.join(_BASE_DIR, 'tmp', f'inc_alpha_{tmp_suffix}')
    physics = _build_physics_isolated(args.nthreads, tmp_suffix)
    try:
        if phase == 'bayesian':
            return bayesian_chunk(args, run_dir, ckpt_path, ckpt, physics)
        else:
            return refine_chunk(args, run_dir, ckpt_path, ckpt, physics)
    finally:
        shutil.rmtree(tmp_dir_path, ignore_errors=True)


# ============================================
# Orchestrator
# ============================================

def case_watchdog(args_tuple):
    w, c, ns = args_tuple
    base = _case_base(ns.folder, w, ns.reg_in, c)
    os.makedirs(base, exist_ok=True)
    log_path = os.path.join(base, 'chunks.log')
    consecutive_failures = 0
    chunk_idx = 0
    while True:
        if _has_completed_run(base):
            print(f"[watchdog w={w:.0e} c={c}] results.json present, done", flush=True)
            return (w, c, "complete")
        chunk_idx += 1
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--mode", "chunk",
            "--weight", str(w), "--coil", str(c),
            "--folder", ns.folder,
            "--lambda", str(ns.reg_in),
            "--nthreads", str(ns.nthreads),
            "--max-evals", str(ns.max_evals),
            "--max-time", str(ns.max_time),
            "--random-state", str(ns.random_state),
        ]
        print(f"[watchdog w={w:.0e} c={c}] launching chunk #{chunk_idx} -> {log_path}", flush=True)
        with open(log_path, 'a') as logf:
            logf.write(f"\n===== chunk #{chunk_idx} w={w:.0e} c={c} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            logf.flush()
            try:
                ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=CHUNK_TIMEOUT_S)
                rc = ret.returncode
            except subprocess.TimeoutExpired:
                logf.write(f"\n[watchdog] chunk #{chunk_idx} exceeded {CHUNK_TIMEOUT_S}s wall limit, killed\n")
                logf.flush()
                rc = -1
        if rc != 0:
            consecutive_failures += 1
            print(f"[watchdog w={w:.0e} c={c}] chunk failed ret={rc} "
                  f"consecutive_failures={consecutive_failures}", flush=True)
            if consecutive_failures > MAX_RETRIES:
                print(f"[watchdog w={w:.0e} c={c}] giving up after {MAX_RETRIES} retries", flush=True)
                return (w, c, "failed")
            time.sleep(5)
        else:
            consecutive_failures = 0


def orchestrator_main(ns):
    pairs = [(w, c) for w in ns.weights for c in ns.coils]
    print(f"[orchestrator] cases={len(pairs)} ncpus={ns.ncpus} folder={ns.folder}", flush=True)
    with Pool(processes=ns.ncpus) as pool:
        results = pool.map(case_watchdog, [(w, c, ns) for (w, c) in pairs])
    print("[orchestrator] all watchdogs returned:", flush=True)
    for w, c, status in results:
        print(f"  w={w:.0e} c={c}: {status}", flush=True)


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['orchestrator', 'chunk'], default='orchestrator')
    parser.add_argument('--weights', type=float, nargs='+', default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument('--coils', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--ncpus', type=int, default=16)
    parser.add_argument('--folder', type=str, default='inc_alpha_temp')
    parser.add_argument('--lambda', dest='reg_in', type=float, default=1e-6)
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--max-evals', type=int, default=2**18, dest='max_evals')
    parser.add_argument('--max-time', type=int, default=86400, dest='max_time')
    parser.add_argument('--random-state', type=int, default=1, dest='random_state')
    parser.add_argument('--weight', type=float, default=None, help='chunk mode: single weight_fb')
    parser.add_argument('--coil', type=int, default=None, help='chunk mode: single num_coils')
    args = parser.parse_args()

    if args.mode == 'chunk':
        if args.weight is None or args.coil is None:
            parser.error("--mode chunk requires --weight and --coil")
        sys.exit(chunk_main(args))
    else:
        orchestrator_main(args)


if __name__ == '__main__':
    main()
