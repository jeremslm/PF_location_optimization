"""
Resume interrupted L-BFGS multistart runs from checkpoint files.

For each checkpoint found under SWEEP_DIR that has no results.json yet,
rebuilds the physics, restores all tracking state, continues the Sobol
sequence from current_start, and writes a fully-fledged results.json
identical in structure to a run that never crashed.

Usage:
    python resume_lbfgs.py [--nthreads N] [--sweep_dir PATH]
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path
tokamaker_python_path = os.getenv("OFT_ROOTPATH")
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, "python"))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, eval_green
from helper_fct import resize_polygon, update_boundary
from OFT_pf_coil_opt_fct import CoilPositionSpace

from opt_comp_combined_boundary import (
    OptimizationComparison,
    make_combined_objective,
    _check_starts_convergence,
    TimeoutException,
    MaxEvalsException,
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_checkpoints(sweep_dir):
    """Return list of checkpoint.json paths that have no results.json yet."""
    paths = []
    for config in sorted(os.listdir(sweep_dir)):
        config_dir = os.path.join(sweep_dir, config)
        if not os.path.isdir(config_dir):
            continue
        for run in sorted(os.listdir(config_dir)):
            run_dir = os.path.join(config_dir, run)
            ckpt = os.path.join(run_dir, 'checkpoint.json')
            result = os.path.join(run_dir, 'results.json')
            if os.path.exists(ckpt) and not os.path.exists(result):
                paths.append(ckpt)
    return paths


def build_physics(nthreads):
    """Mirrors parallel_case: build mesh, run fixed-boundary solve, return physics objects."""
    eqdsk = read_eqdsk(os.path.join(_BASE_DIR, 'examples/data/eqdsk/g192185.02440'))
    LCFS_contour = eqdsk['rzout'].copy()
    fixed_LCFS = LCFS_contour

    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    tmp_dir = os.path.join(_BASE_DIR, 'tmp', 'temp_resume_physics')
    try:
        shutil.rmtree(tmp_dir)
    except FileNotFoundError:
        pass
    os.makedirs(tmp_dir)
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


def resume_checkpoint(ckpt_path, myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim):
    with open(ckpt_path) as f:
        ckpt = json.load(f)

    run_dir = os.path.dirname(ckpt_path)
    s = ckpt['optimization_settings']
    NUM_COILS = s['num_coils']
    ALPHA = s['alpha']
    WEIGHT_FB = s['weight_fb']
    REG_IN = s['reg_in']
    RFIL = s['rfil']
    OMEGA = s['omega']
    DIST_TH = s['dist_th']
    MAX_TIME = s['max_time']
    MAX_EVALS = s['max_evals']
    CONVERGENCE_THRESHOLD = s['convergence_threshold']
    RANDOM_STATE = ckpt['random_state']
    STARTS_WINDOW = ckpt['convergence_window']
    CURRENT_START = ckpt['current_start']
    ELAPSED_OFFSET = ckpt['elapsed']

    print(f"\n{'='*60}")
    print(f"Resuming: {run_dir}")
    print(f"weight_fb={WEIGHT_FB:.0e}  coils={NUM_COILS}  alpha={ALPHA}")
    print(f"Already done: {ckpt['starts_completed']} starts, {ckpt['n_evals']} evals, {ELAPSED_OFFSET/3600:.1f}h")
    print(f"Resuming from Sobol index {CURRENT_START}")
    print('='*60)

    # mirrors main() exactly
    r_bnd, psi_bnd = mygs.get_vfixed()
    print(f"Found {len(r_bnd)} boundary points")

    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    coil_space = CoilPositionSpace(inner_boundary=coil_center_cand1,
                                   outer_boundary=coil_center_cand2, method='coords')
    coil_space.set_bounds(angular_bounds=(10, 170), radial_bounds=(0, 1))
    theta_bounds, radial_bounds = coil_space.get_bounds()
    bounds = [theta_bounds] * NUM_COILS + [radial_bounds] * NUM_COILS

    theta_range = np.linspace(0, 180, len(coil_center_cand1) // 2)
    inner = coil_center_cand1[:len(coil_center_cand1) // 2]
    outer = coil_center_cand2[:len(coil_center_cand2) // 2]

    objective = make_combined_objective(
        ALPHA, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
        coil_center_cand1, coil_center_cand2, lim,
        r_bnd, psi_bnd, WEIGHT_FB, NUM_COILS, RFIL,
        REG_IN, OMEGA, DIST_TH, theta_range, inner, outer
    )

    # restore normalization so costs are consistent with original run
    objective.norm_fixed = ckpt.get('initial_fixed_cost')
    objective.norm_fb = ckpt.get('initial_fb_cost')
    objective.fb_failures = ckpt['fb_failures']

    comparison = OptimizationComparison(
        objective, bounds,
        max_time=MAX_TIME, max_evals=MAX_EVALS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        NUM_COILS=NUM_COILS, OMEGA=OMEGA, DIST_TH=DIST_TH,
        REG_IN=REG_IN, RFIL=RFIL, ALPHA=ALPHA, WEIGHT_FB=WEIGHT_FB
    )
    comparison.set_problem_data(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                                mygs.o_point, eval_green)
    comparison.checkpoint_path = ckpt_path

    # restore all tracking state
    comparison._current_method = 'L-BFGS'
    comparison._convergence_window = STARTS_WINDOW
    comparison._random_state = RANDOM_STATE
    comparison._n_evals = ckpt['n_evals']
    comparison._starts_completed = ckpt['starts_completed']
    comparison._current_start = CURRENT_START
    comparison._best_cost = ckpt['best_cost']
    comparison._best_flux_err = ckpt['best_flux_err']
    comparison._best_fb_cost = ckpt['best_fb_cost']
    comparison._best_params = np.array(ckpt['best_params'])
    comparison._initial_fixed_cost = ckpt.get('initial_fixed_cost')
    comparison._initial_fb_cost = ckpt.get('initial_fb_cost')
    comparison._fb_failures = ckpt['fb_failures']
    comparison._history = list(ckpt['cost_history'])
    comparison._flux_err_history = list(ckpt['flux_err_history'])
    comparison._fb_cost_history = list(ckpt['fb_cost_history'])
    comparison._convergence = list(ckpt['convergence_history'])
    comparison._start_boundaries = list(ckpt['start_boundaries'])
    comparison._start_costs = list(ckpt['start_costs'])
    comparison._times = list(ckpt['times'])
    comparison._x_history = []  # not checkpointed, not in results — safe to start empty
    # offset start_time so new elapsed values continue the existing timeline
    comparison._start_time = time.time() - ELAPSED_OFFSET

    # regenerate same Sobol sequence, skip already-done starts
    n_starts = 262144
    sampler = qmc.Sobol(d=comparison.n_params, scramble=True, seed=RANDOM_STATE)
    samples = sampler.random(n_starts)
    starts = [
        [low + samples[i, j] * (high - low) for j, (low, high) in enumerate(bounds)]
        for i in range(n_starts)
    ]

    starts_bests = list(ckpt['start_costs'])
    stopped_by = "all starts completed"

    for x0 in starts[CURRENT_START:]:
        try:
            comparison._current_start += 1
            minimize(comparison._track_objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'ftol': 1e-9, 'gtol': 1e-6, 'disp': False})
            comparison._starts_completed += 1
            starts_bests.append(comparison._best_cost)
            comparison._start_boundaries.append(comparison._n_evals)
            comparison._start_costs.append(comparison._best_cost)
            if _check_starts_convergence(starts_bests, STARTS_WINDOW, CONVERGENCE_THRESHOLD):
                stopped_by = "converged"
                break
        except TimeoutException:
            stopped_by = "exceeded wall time"
            break
        except MaxEvalsException:
            stopped_by = "max function calls"
            break

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
        'random_state': RANDOM_STATE,
        'flux_err_history': list(comparison._flux_err_history),
        'fb_cost_history': list(comparison._fb_cost_history),
    }

    print(f"L-BFGS resumed: {comparison._n_evals} total evals, {elapsed/3600:.2f}h total, "
          f"{comparison._starts_completed} starts, stopped: {stopped_by}")

    comparison.save_results_to_json(os.path.join(run_dir, 'results.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--sweep_dir', type=str,
                        default=os.path.join(_BASE_DIR,
                            'examples/comparisons/combined_boundary_DIIID/convergence_w5_lbfgs'))
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.sweep_dir)
    if not checkpoints:
        print("No in-progress checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) to resume:")
    for c in checkpoints:
        print(f"  {c}")

    print("\nBuilding physics (shared across all runs)...")
    myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim = build_physics(args.nthreads)

    for ckpt_path in checkpoints:
        resume_checkpoint(ckpt_path, myOFT, mygs, eqdsk, fixed_LCFS, fixed_mag_axis, lim)

    print("\nAll runs resumed and saved.")


if __name__ == '__main__':
    main()
