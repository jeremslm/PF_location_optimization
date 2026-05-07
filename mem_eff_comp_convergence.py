"""
Memory-efficient LBFGS sweep for combined-boundary optimization.

Self-dispatching:
  --mode orchestrator (default): Pool of N workers; each worker handles one
    (weight_fb, num_coils) case via repeated subprocess calls to itself in
    --mode chunk. Process exit between chunks reclaims OFT/Fortran heap so
    peak RAM stays bounded.
  --mode chunk: Runs K Sobol-multistart LBFGS starts for one (w, c) case,
    saves checkpoint, exits without writing results.json unless terminal
    (converged / max_evals / max_time / sobol-exhausted).

LBFGS-only. Bayesian state cannot be cheaply checkpointed across processes.
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

N_STARTS = 262144
OMEGA = 1e-2
DIST_TH = 5.0
RFIL = 0.01
CONVERGENCE_THRESHOLD = 0.01
STARTS_WINDOW = 5
LBFGS_MAXFUN = 100
MAX_RETRIES = 3


def _case_base(folder, alpha, w, reg_in, c):
    return os.path.join(_BASE_DIR,
        f'examples/comparisons/combined_boundary_DIIID/{folder}/'
        f'alpha:{alpha},weight:{w:.0e},lambda:{reg_in:.0e},coils:{c}')


def _find_or_create_run_dir(base):
    os.makedirs(base, exist_ok=True)
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isdir(d) and name.startswith('run_'):
            res = os.path.join(d, 'results.json')
            if os.path.exists(res):
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


# ============================================
# Chunk mode
# ============================================

def _build_physics_isolated(nthreads, tmp_suffix):
    import shutil
    import numpy as np
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    from helper_fct import update_boundary

    eqdsk = read_eqdsk(os.path.join(_BASE_DIR, 'examples/data/eqdsk/g192185.02440'))
    LCFS_contour = eqdsk['rzout'].copy()
    fixed_LCFS = LCFS_contour
    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    tmp_dir = os.path.join(_BASE_DIR, 'tmp', f'mem_eff_{tmp_suffix}')
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


def chunk_main(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy.stats import qmc

    import opt_comp_combined_boundary as _ocb
    from opt_comp_combined_boundary import (
        OptimizationComparison, make_combined_objective,
        _check_starts_convergence, TimeoutException, MaxEvalsException,
    )
    from OpenFUSIONToolkit.TokaMaker.util import eval_green
    from helper_fct import resize_polygon, update_boundary
    from OFT_pf_coil_opt_fct import CoilPositionSpace

    w = args.weight
    c = args.coil
    K = args.starts_per_call
    alpha = args.alpha
    reg_in = args.reg_in
    folder = args.folder
    nthreads = args.nthreads
    max_evals = args.max_evals
    max_time = args.max_time
    random_state = args.random_state

    base = _case_base(folder, alpha, w, reg_in, c)
    run_dir, resuming = _find_or_create_run_dir(base)
    ckpt_path = os.path.join(run_dir, 'checkpoint.json')
    _ocb._MEM_LOG_DIR = run_dir
    _ocb._get_mem_logger()
    print(f"[chunk] {('resume' if resuming else 'fresh')} {run_dir}", flush=True)

    import shutil
    tmp_suffix = f"{os.getpid()}_{w:.0e}_{c}"
    tmp_dir_path = os.path.join(_BASE_DIR, 'tmp', f'mem_eff_{tmp_suffix}')
    myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim = _build_physics_isolated(nthreads, tmp_suffix)
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

    objective = make_combined_objective(
        alpha, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
        coil_center_cand1, coil_center_cand2, lim,
        r_bnd, psi_bnd, w, c, RFIL,
        reg_in, OMEGA, DIST_TH, theta_range, inner, outer,
    )

    comparison = OptimizationComparison(
        objective, bounds,
        max_time=max_time, max_evals=max_evals,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        NUM_COILS=c, OMEGA=OMEGA, DIST_TH=DIST_TH,
        REG_IN=reg_in, RFIL=RFIL, ALPHA=alpha, WEIGHT_FB=w,
        verbose=True,
    )
    comparison.set_problem_data(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                                mygs.o_point, eval_green)
    comparison.checkpoint_path = ckpt_path
    comparison._reset_tracking()
    comparison._current_method = 'L-BFGS'
    comparison._convergence_window = STARTS_WINDOW
    comparison._random_state = random_state
    comparison._maxiter = 1000000000
    comparison._lbfgs_maxfun = LBFGS_MAXFUN

    starts_bests = []
    if resuming:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        objective.norm_fixed = ckpt.get('initial_fixed_cost')
        objective.norm_fb = ckpt.get('initial_fb_cost')
        objective.fb_failures = ckpt.get('fb_failures', 0)
        comparison._n_evals = ckpt['n_evals']
        comparison._starts_completed = ckpt['starts_completed']
        comparison._current_start = ckpt['current_start']
        comparison._best_cost = ckpt['best_cost']
        comparison._best_flux_err = ckpt.get('best_flux_err')
        comparison._best_fb_cost = ckpt.get('best_fb_cost')
        comparison._best_params = np.array(ckpt['best_params'])
        comparison._initial_fixed_cost = ckpt.get('initial_fixed_cost')
        comparison._initial_fb_cost = ckpt.get('initial_fb_cost')
        comparison._fb_failures = ckpt.get('fb_failures', 0)
        comparison._history = list(ckpt['cost_history'])
        comparison._flux_err_history = list(ckpt['flux_err_history'])
        comparison._fb_cost_history = list(ckpt['fb_cost_history'])
        comparison._convergence = list(ckpt['convergence_history'])
        comparison._start_boundaries = list(ckpt['start_boundaries'])
        comparison._start_costs = list(ckpt['start_costs'])
        comparison._times = list(ckpt['times'])
        comparison._fb_mesh_times = list(ckpt.get('fb_mesh_times', []))
        comparison._fb_setup_times = list(ckpt.get('fb_setup_times', []))
        comparison._fb_solve_times = list(ckpt.get('fb_solve_times', []))
        comparison._fb_other_times = list(ckpt.get('fb_other_times', []))
        comparison._fb_total_times = list(ckpt.get('fb_total_times', []))
        comparison._fixed_times = list(ckpt.get('fixed_times', []))
        comparison._bayesian_ask_times = list(ckpt.get('bayesian_ask_times', []))
        comparison._bayesian_tell_times = list(ckpt.get('bayesian_tell_times', []))
        random_state = ckpt['random_state']
        comparison._random_state = random_state
        starts_bests = list(ckpt['start_costs'])
        elapsed_offset = ckpt.get('elapsed', 0.0)
        comparison._start_time = time.time() - elapsed_offset
        print(f"[chunk] restored: starts_completed={comparison._starts_completed} "
              f"current_start={comparison._current_start} best={comparison._best_cost:.4e}", flush=True)

    sampler = qmc.Sobol(d=comparison.n_params, scramble=True, seed=random_state)
    samples = sampler.random(N_STARTS)
    starts = [
        [low + samples[i, j] * (high - low) for j, (low, high) in enumerate(bounds)]
        for i in range(N_STARTS)
    ]

    chunk_done = 0
    stopped_by = None
    cur = comparison._current_start
    for x0 in starts[cur:]:
        if chunk_done >= K:
            stopped_by = "chunk_budget"
            break
        try:
            comparison._current_start += 1
            minimize(comparison._track_objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'ftol': 1e-9, 'gtol': 1e-6,
                              'maxiter': comparison._maxiter,
                              'maxfun': LBFGS_MAXFUN, 'disp': False})
            comparison._starts_completed += 1
            starts_bests.append(comparison._best_cost)
            comparison._start_boundaries.append(comparison._n_evals)
            comparison._start_costs.append(comparison._best_cost)
            comparison._save_checkpoint()
            chunk_done += 1
            if _check_starts_convergence(starts_bests, STARTS_WINDOW, CONVERGENCE_THRESHOLD):
                stopped_by = "converged"
                break
        except TimeoutException:
            stopped_by = "exceeded wall time"
            break
        except MaxEvalsException:
            stopped_by = "max function calls"
            break
    else:
        stopped_by = "all starts completed"

    print(f"[chunk] stopped_by={stopped_by} chunk_starts={chunk_done} "
          f"total_starts={comparison._starts_completed} evals={comparison._n_evals}", flush=True)

    if stopped_by == "chunk_budget":
        shutil.rmtree(tmp_dir_path, ignore_errors=True)
        return 0

    elapsed = time.time() - comparison._start_time
    thetas, radials, coil_positions, coil_currents = comparison._extract_best_result()
    comparison.results['Multi-start L-BFGS'] = {
        'best_cost': comparison._best_cost,
        'best_flux_err': comparison._best_flux_err,
        'best_fb_cost': comparison._best_fb_cost,
        'initial_fixed_cost': comparison._initial_fixed_cost,
        'initial_fb_cost': comparison._initial_fb_cost,
        'best_params': comparison._best_params,
        'n_evals': comparison._n_evals,
        'time': elapsed,
        'times': comparison._times.copy(),
        'stopping': stopped_by,
        'parameters': {'thetas': thetas, 'radials': radials},
        'coil_positions_top': coil_positions,
        'coil_currents': coil_currents,
        'convergence_history': list(comparison._convergence),
        'cost_history': list(comparison._history),
        'fb_failures': comparison._fb_failures,
        'starts_completed': comparison._starts_completed,
        'start_boundaries': comparison._start_boundaries,
        'start_costs': comparison._start_costs,
        'convergence_window': STARTS_WINDOW,
        'random_state': random_state,
        'flux_err_history': list(comparison._flux_err_history),
        'fb_cost_history': list(comparison._fb_cost_history),
        'maxiter': comparison._maxiter,
        'lbfgs_maxfun': LBFGS_MAXFUN,
    }
    comparison.save_results_to_json(os.path.join(run_dir, 'results.json'))
    fig = comparison.plot_result()
    time_fig = comparison.plot_convergence_vs_time(log_scale=True)
    fig.savefig(os.path.join(run_dir, 'convergence_plot.png'), dpi=150, bbox_inches='tight')
    time_fig.savefig(os.path.join(run_dir, 'convergence_vs_time_plot.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    shutil.rmtree(tmp_dir_path, ignore_errors=True)
    return 0


# ============================================
# Orchestrator mode
# ============================================

def case_watchdog(args_tuple):
    w, c, ns = args_tuple
    base = _case_base(ns.folder, ns.alpha, w, ns.reg_in, c)
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
            "--starts-per-call", str(ns.starts_per_call),
            "--folder", ns.folder,
            "--alpha", str(ns.alpha),
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
            ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            consecutive_failures += 1
            print(f"[watchdog w={w:.0e} c={c}] chunk failed ret={ret.returncode} "
                  f"consecutive_failures={consecutive_failures}", flush=True)
            if consecutive_failures > MAX_RETRIES:
                print(f"[watchdog w={w:.0e} c={c}] giving up after {MAX_RETRIES} retries", flush=True)
                return (w, c, "failed")
            time.sleep(5)
        else:
            consecutive_failures = 0


def orchestrator_main(ns):
    pairs = [(w, c) for w in ns.weights for c in ns.coils]
    print(f"[orchestrator] cases={len(pairs)} ncpus={ns.ncpus} "
          f"starts_per_call={ns.starts_per_call} folder={ns.folder}", flush=True)
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
    parser.add_argument('--starts-per-call', type=int, default=20, dest='starts_per_call')
    parser.add_argument('--folder', type=str, default='convergence_w5_l_temp')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lambda', dest='reg_in', type=float, default=1e-6)
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--max-evals', type=int, default=2**18, dest='max_evals')
    parser.add_argument('--max-time', type=int, default=86400, dest='max_time')
    parser.add_argument('--random-state', type=int, default=1, dest='random_state')
    parser.add_argument('--weight', type=float, default=None,
                        help='chunk mode: single weight_fb value')
    parser.add_argument('--coil', type=int, default=None,
                        help='chunk mode: single num_coils value')
    args = parser.parse_args()

    if args.mode == 'chunk':
        if args.weight is None or args.coil is None:
            parser.error("--mode chunk requires --weight and --coil")
        sys.exit(chunk_main(args))
    else:
        orchestrator_main(args)


if __name__ == '__main__':
    main()

