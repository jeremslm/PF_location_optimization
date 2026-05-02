"""
Optimization Comparison for PF Coil Placement (Combined Boundary, Parallel)
============================================================================

Objective: (1 - alpha) * fixed_boundary_cost + alpha * free_boundary_cost + dist_penalty

Fixed-boundary cost: flux_error_squared from lstsq solve (cheap).
Free-boundary cost:  boundary_distance from TokaMaker GS solve (expensive, ~1-2s/eval).
Both evaluated at every function call. Convergence window = 5.

Sweep: weight_fb x ncoils, REG_IN = 1e-6 fixed, alpha = 0.75.
"""

import copy
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
import argparse

import psutil
from itertools import permutations as _iperms
from math import factorial
from multiprocessing import Pool
import psutil 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import qmc
from skopt import Optimizer
from skopt.space import Real

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path
tokamaker_python_path = os.getenv("OFT_ROOTPATH")
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, "python"))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh, load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, eval_green
from helper_fct import resize_polygon, update_boundary, place_points_pol_rad, make_3x3_thick
from OFT_pf_coil_opt_fct import CoilPositionSpace

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_crash_logger = logging.getLogger('fb_crashes')
_crash_logger.setLevel(logging.ERROR)

_MEM_LOG_DIR = None
_mem_loggers = {}

def _get_mem_logger():
    if _MEM_LOG_DIR is None:
        return None
    pid = os.getpid()
    cached = _mem_loggers.get((_MEM_LOG_DIR, pid))
    if cached is not None:
        return cached
    log = logging.getLogger(f'mem_{_MEM_LOG_DIR}_{pid}')
    log.setLevel(logging.INFO)
    log.propagate = False
    fh = logging.FileHandler(os.path.join(_MEM_LOG_DIR, 'memory.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log.addHandler(fh)
    _mem_loggers[(_MEM_LOG_DIR, pid)] = log
    return log


class TimeoutException(Exception):
    pass


class MaxEvalsException(Exception):
    pass


def _check_starts_convergence(starts_bests, window, threshold):
    n = len(starts_bests)
    if n <= window:
        return False
    old_best = starts_bests[n - window - 1]
    new_best = starts_bests[-1]
    if abs(old_best) > 0:
        rel_imp = (old_best - new_best) / abs(old_best)
        return rel_imp < threshold
    return new_best == 0


# ============================================
# Free-boundary helpers
# ============================================

def boundary_distance(fixed_LCFS, free_LCFS, mag_axis):
    R0, Z0 = mag_axis
    theta_fixed = np.arctan2(fixed_LCFS[:, 1] - Z0, fixed_LCFS[:, 0] - R0)
    r_fixed = np.sqrt((fixed_LCFS[:, 0] - R0)**2 + (fixed_LCFS[:, 1] - Z0)**2)
    theta_free = np.arctan2(free_LCFS[:, 1] - Z0, free_LCFS[:, 0] - R0)
    r_free = np.sqrt((free_LCFS[:, 0] - R0)**2 + (free_LCFS[:, 1] - Z0)**2)
    r_fixed_interp = np.interp(theta_free, theta_fixed, r_fixed, period=2 * np.pi)
    # return np.sum(np.abs(r_free - r_fixed_interp))
    return np.mean((r_free - r_fixed_interp) ** 2)


def make_new_coils(params, nCoils, coil_center_cand1, coil_center_cand2, dx=0.03, dy=0.03):
    thetas = params[:nCoils]
    radials = params[nCoils:2 * nCoils]
    inner = coil_center_cand1[:len(coil_center_cand1) // 2]
    outer = coil_center_cand2[:len(coil_center_cand2) // 2]
    _, locs = place_points_pol_rad(nCoils, inner, outer, thetas, radials)
    scan_geom = {"coils": {}}
    for i, loc in enumerate(locs):
        pts_top = np.array([
            [loc[0] - dx, loc[1] + dy],
            [loc[0] + dx, loc[1] + dy],
            [loc[0] + dx, loc[1] - dy],
            [loc[0] - dx, loc[1] - dy],
        ])
        pts_bot = pts_top * np.array([1, -1])
        scan_geom["coils"][f"F{i}A"] = {"pts": copy.deepcopy(pts_top), "nturns": 1.0}
        scan_geom["coils"][f"F{i}B"] = {"pts": copy.deepcopy(pts_bot), "nturns": 1.0}
    return scan_geom


def make_mesh(scan_geom, savename, lim,
              plasma_dx=0.08, coil_dx=0.02, vac_dx=0.04, vv_dx=0.04):
    gs_mesh = gs_Domain()
    gs_mesh.define_region("air", vac_dx, "boundary")
    gs_mesh.define_region("plasma", plasma_dx, "plasma")
    gs_mesh.define_region("vacuum", vv_dx, "vacuum", allow_xpoints=True)
    gs_mesh.define_region("vv", vv_dx, "conductor", eta=6e-7)
    for key, coil in scan_geom["coils"].items():
        gs_mesh.define_region(key, coil_dx, "coil", nTurns=coil["nturns"])
    gs_mesh.add_polygon(lim, "plasma", parent_name="vacuum")
    gs_mesh.add_annulus(resize_polygon(lim, 0.01), "vacuum", resize_polygon(lim, 0.05), "vv")
    gs_mesh.add_enclosed([1.75, 1.25], "vacuum")
    for key, coil in scan_geom["coils"].items():
        gs_mesh.add_polygon(coil["pts"], key, parent_name="air")
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    coil_dict = gs_mesh.get_coils()
    cond_dict = gs_mesh.get_conductors()
    save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, savename)
    return coil_dict, cond_dict


def _free_boundary_cost(params, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
                        coil_center_cand1, coil_center_cand2, lim,
                        weight_fb, nCoils, xpoint_index=55):
    pid = os.getpid()
    _tmp_dir = os.path.join(_BASE_DIR, 'tmp')
    os.makedirs(_tmp_dir, exist_ok=True)
    mesh_file = os.path.join(_tmp_dir, f"mesh_fb_{pid}.h5")
    eqdsk_tmp = os.path.join(_tmp_dir, f"gTMP_{pid}")
    try:
        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss / 1024 ** 2
        t0 = time.time()
        scan_geom = make_new_coils(params, nCoils, coil_center_cand1, coil_center_cand2)
        make_mesh(scan_geom, mesh_file, lim)
        t_mesh = time.time()

        mygs = TokaMaker(myOFT)

        mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_file)
        mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
        mygs.settings.free_boundary = True

        F0 = eqdsk["rcentr"] * eqdsk["bcentr"]
        mygs.setup(order=2, F0=F0)
        t_setup = time.time()

        R0_target = float(fixed_mag_axis[0])
        Z0_target = float(fixed_mag_axis[1])
        mygs.set_targets(Ip=eqdsk["ip"], R0=R0_target, V0=Z0_target)
        mygs.set_coil_bounds({key: [-6e8, 6e8] for key in mygs.coil_sets})

        isoflux_pts = eqdsk["rzout"].copy()
        w_iso = np.ones(len(isoflux_pts[:, 0]))
        w_iso[xpoint_index] = 1e4
        mygs.set_isoflux(isoflux_pts, weights=w_iso)

        reg_terms = [mygs.coil_reg_term({name: 1.0}, target=0.0, weight=weight_fb)
                     for name in mygs.coil_sets]
        mygs.set_coil_reg(reg_terms=reg_terms)
        mygs.init_psi(r0=1.8, z0=-0.040, a=0.45, kappa=1.547, delta=-0.288)
        mygs.solve()
        t_solve = time.time()

        mygs.save_eqdsk(eqdsk_tmp, truncate_eq=False)
        EQ_in = read_eqdsk(eqdsk_tmp)
        t_total = time.time()
        print(f"[fb_cost] mesh={t_mesh-t0:.2f}s setup={t_setup-t_mesh:.2f}s solve={t_solve-t_setup:.2f}s other={t_total-t_solve:.2f}s total={t_total-t0:.2f}s", flush=True)

        del mygs
        gc.collect()
        rss1 = proc.memory_info().rss / 1024 ** 2
        mem_log = _get_mem_logger()
        if mem_log is not None:
            mem_log.info(f"pid={os.getpid()} nCoils={nCoils} pre={rss0:.1f}MB post={rss1:.1f}MB delta={rss1-rss0:+.1f}MB")

        return boundary_distance(fixed_LCFS, EQ_in["rzout"], fixed_mag_axis)
    except Exception:
        _crash_logger.error(
            "pid=%d weight_fb=%.2e params=%s\n%s",
            os.getpid(), weight_fb, np.array2string(np.asarray(params), precision=4),
            traceback.format_exc()
        )
        return 1e6
    finally:
        for f in [mesh_file, eqdsk_tmp]:
            if os.path.exists(f):
                os.remove(f)


# ============================================
# OptimizationComparison
# ============================================

class OptimizationComparison:
    def __init__(self, objective_func, bounds, max_time=86400,
                 max_evals=None, convergence_threshold=0.001,
                 NUM_COILS=3, OMEGA=1e-7, DIST_TH=5.0, REG_IN=1e-6,
                 RFIL=0.01, ALPHA=0.75, WEIGHT_FB=1e-2):
        self.objective = objective_func
        self.bounds = bounds
        self.max_time = max_time
        self.max_evals = max_evals
        self.convergence_threshold = convergence_threshold
        self.n_params = len(bounds)
        self.num_coils = NUM_COILS
        self.omega = OMEGA
        self.dist_th = DIST_TH
        self.reg_in = REG_IN
        self.rfil = RFIL
        self.alpha = ALPHA
        self.weight_fb = WEIGHT_FB
        self.results = {}
        self.all_runs = {}
        self.r_bnd = None
        self.psi_bnd = None
        self.coil_center_cand1 = None
        self.coil_center_cand2 = None
        self.o_point = None
        self.eval_green = None
        self.brute_force_cost = None

    def set_problem_data(self, r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2, o_point, eval_green):
        self.r_bnd = r_bnd
        self.psi_bnd = psi_bnd
        self.coil_center_cand1 = coil_center_cand1
        self.coil_center_cand2 = coil_center_cand2
        self.o_point = o_point
        self.eval_green = eval_green

    def _reset_tracking(self, start_time=None):
        self._n_evals = 0
        self._history = []
        self._x_history = []
        self._times = []
        self._best_cost = float('inf')
        self._best_flux_err = None
        self._best_fb_cost = None
        self._initial_fixed_cost = None
        self._initial_fb_cost = None
        self._best_params = None
        self._start_time = start_time if start_time is not None else time.time()
        self._convergence = []
        self._stopped_reason = None
        self._current_method = None
        self._current_start = 0
        self._starts_completed = 0
        self._start_boundaries = []
        self._start_costs = []
        self._convergence_window = None
        self._random_state = None
        self._maxiter = None
        self._lbfgs_maxfun = None
        self._fb_failures = 0
        self._flux_err_history = []
        self._fb_cost_history = []
        self.objective.norm_fixed = None
        self.objective.norm_fb = None
        self.objective.fb_failures = 0

    def _save_checkpoint(self):
        if self.checkpoint_path is None or self._best_params is None:
            return
        thetas, radials, coil_positions, coil_currents = self._extract_best_result()
        data = {
            'optimization_settings': {
                'num_coils': int(self.num_coils),
                'max_evals': int(self.max_evals) if self.max_evals is not None else None,
                'max_time': float(self.max_time),
                'convergence_threshold': float(self.convergence_threshold),
                'omega': float(self.omega),
                'dist_th': float(self.dist_th),
                'reg_in': float(self.reg_in),
                'rfil': float(self.rfil),
                'alpha': float(self.alpha),
                'weight_fb': float(self.weight_fb),
                'maxiter': self._maxiter,
                'lbfgs_maxfun': self._lbfgs_maxfun,
            },
            'method': self._current_method,
            'stopping': 'in_progress',
            'n_evals': self._n_evals,
            'current_start': self._current_start,
            'elapsed': self._times[-1] if self._times else 0.0,
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_fb_cost': self._best_fb_cost,
            'initial_fixed_cost': self._initial_fixed_cost,
            'initial_fb_cost': self._initial_fb_cost,
            'best_norm_fixed': self._best_flux_err / self._initial_fixed_cost if self._initial_fixed_cost else None,
            'best_norm_fb': self._best_fb_cost / self._initial_fb_cost if self._initial_fb_cost else None,
            'best_params': self._best_params.tolist(),
            'fb_failures': self._fb_failures,
            'starts_completed': self._starts_completed,
            'convergence_window': self._convergence_window,
            'random_state': self._random_state,
            'parameters': {'thetas': thetas, 'radials': radials},
            'coil_positions_top': coil_positions,
            'coil_currents': coil_currents,
            'start_boundaries': list(self._start_boundaries),
            'start_costs': list(self._start_costs),
            'convergence_history': list(self._convergence),
            'cost_history': list(self._history),
            'times': list(self._times),
            'flux_err_history': list(self._flux_err_history),
            'fb_cost_history': list(self._fb_cost_history),
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f)

    def _track_objective(self, params):
        if time.time() - self._start_time > self.max_time:
            raise TimeoutException("Wall-clock time limit reached")
        if self.max_evals is not None and self._n_evals >= self.max_evals:
            raise MaxEvalsException("Maximum function evaluations reached")
        self._n_evals += 1
        params = np.asarray(params)
        cost = self.objective(params)
        elapsed = time.time() - self._start_time
        self._history.append(cost)
        self._x_history.append(params.copy())
        self._times.append(elapsed)
        flux_err = getattr(self.objective, 'last_flux_err', None)
        fb_cost = getattr(self.objective, 'last_fb_cost', None)
        self._fb_failures = getattr(self.objective, 'fb_failures', 0)
        self._flux_err_history.append(flux_err)
        self._fb_cost_history.append(fb_cost)
        if self._initial_fixed_cost is None and flux_err is not None:
            self._initial_fixed_cost = flux_err
        if self._initial_fb_cost is None and fb_cost is not None:
            self._initial_fb_cost = fb_cost
        if cost < self._best_cost:
            self._best_cost = cost
            self._best_params = params.copy()
            if flux_err is not None:
                self._best_flux_err = flux_err
            if fb_cost is not None:
                self._best_fb_cost = fb_cost
        self._convergence.append(self._best_cost)
        if self._n_evals % 3 == 0 and elapsed > 0:
            rate = self._n_evals / elapsed
            print(f"[{self._current_method}] eval={self._n_evals} start={self._current_start} best={self._best_cost:.4e} {1/rate:.2f}s/eval", flush=True)
            self._save_checkpoint()
        return cost

    def _compute_flux_for_params(self, params):
        num_coils = len(params) // 2
        thetas = params[:num_coils]
        radials = params[num_coils:]
        inner_arc = self.coil_center_cand1[:len(self.coil_center_cand1) // 2]
        outer_arc = self.coil_center_cand2[:len(self.coil_center_cand2) // 2]
        _, coil_locs = place_points_pol_rad(num_coils, inner_arc, outer_arc, thetas, radials)
        coil_centers_3x3 = []
        for loc in coil_locs:
            coil_centers_3x3.append(make_3x3_thick(loc, self.rfil))
            coil_centers_3x3.append(make_3x3_thick([loc[0], -loc[1]], self.rfil))
        n_bnd = self.psi_bnd.shape[0]
        n_coils_total = len(coil_centers_3x3)
        con = np.zeros((n_bnd - 1 + n_coils_total, n_coils_total))
        for i, filament_set in enumerate(coil_centers_3x3):
            flux_tmp = np.zeros((n_bnd,))
            for fil in filament_set:
                flux_tmp += self.eval_green(self.r_bnd, fil)
            con[:n_bnd - 1, i] = flux_tmp[1:] - flux_tmp[0]
            con[n_bnd - 1 + i, i] = self.reg_in
        err = np.zeros((n_bnd - 1 + n_coils_total,))
        err[:n_bnd - 1] = self.psi_bnd[1:] - self.psi_bnd[0]
        currs, _, _, _ = np.linalg.lstsq(con, err, rcond=None)
        psi_computed = np.dot(con, currs)[:n_bnd - 1]
        return coil_locs, psi_computed, currs

    def _get_permuted_points(self, x, cost, max_perms=None):
        n = self.num_coils
        thetas = x[:n]
        radials = x[n:]
        all_perms = list(_iperms(range(n)))
        if max_perms is not None and len(all_perms) > max_perms:
            all_perms = random.sample(all_perms, max_perms)
        xs, ys = [], []
        for perm in all_perms:
            xs.append([thetas[i] for i in perm] + [radials[i] for i in perm])
            ys.append(cost)
        return xs, ys

    def _deduplicate_candidates(self, candidates, tol=0.05, max_unique=None):
        if not candidates:
            return []
        if self.coil_center_cand1 is None or self.coil_center_cand2 is None:
            raise RuntimeError("set_problem_data() must be called before _deduplicate_candidates")
        inner = self.coil_center_cand1[:len(self.coil_center_cand1) // 2]
        outer = self.coil_center_cand2[:len(self.coil_center_cand2) // 2]
        theta_range = np.linspace(0, 180, len(inner))

        def _to_rz(params):
            thetas = params[:self.num_coils]
            radials = params[self.num_coils:]
            coords = []
            for theta, rho in zip(thetas, radials):
                R_inner = np.interp(theta, theta_range, inner[:, 0])
                Z_inner = np.interp(theta, theta_range, inner[:, 1])
                R_outer = np.interp(theta, theta_range, outer[:, 0])
                Z_outer = np.interp(theta, theta_range, outer[:, 1])
                coords.append((1 - rho) * R_inner + rho * R_outer)
                coords.append((1 - rho) * Z_inner + rho * Z_outer)
            return np.array(coords)

        unique = [candidates[0]]
        unique_rz = [_to_rz(candidates[0])]
        for c in candidates[1:]:
            if max_unique is not None and len(unique) >= max_unique:
                break
            crz = _to_rz(c)
            if min(np.linalg.norm(crz - u) for u in unique_rz) > tol:
                unique.append(c)
                unique_rz.append(crz)
        return unique

    def _extract_best_result(self):
        best_params = self._best_params
        num_coils = len(best_params) // 2
        thetas = best_params[:num_coils].tolist()
        radials = best_params[num_coils:].tolist()
        if self.coil_center_cand1 is not None:
            coil_locs, _, currents = self._compute_flux_for_params(best_params)
            coil_positions = [[float(loc[0]), float(loc[1])] for loc in coil_locs]
            coil_currents = currents.tolist()
        else:
            coil_positions = None
            coil_currents = None
        return thetas, radials, coil_positions, coil_currents

    METHOD_COLORS = {
        'Multi-start L-BFGS': '#2ca02c',
        'Bayesian': '#1f77b4',
    }

    def _get_sorted_methods_and_colors(self):
        sorted_methods = sorted(self.results.items(), key=lambda x: x[1]['best_cost'])
        colors = [self.METHOD_COLORS.get(m, '#7f7f7f') for m, _ in sorted_methods]
        return sorted_methods, colors

    def _plot_position_space_boundaries(self, ax):
        ax.plot(self.coil_center_cand1[:, 0], self.coil_center_cand1[:, 1],
                'k--', alpha=0.3, linewidth=1, label='Position space')
        ax.plot(self.coil_center_cand2[:, 0], self.coil_center_cand2[:, 1],
                'k--', alpha=0.3, linewidth=1)

    def run_multistart_lbfgs(self, n_starts=262144, ftol=1e-9, gtol=1e-6,
                             starts_window=5, random_state=42, maxiter=1000000000, lbfgs_maxfun=1000000000,
                             start_time=None):
        self._reset_tracking(start_time=start_time)
        self._current_method = 'L-BFGS'
        self._convergence_window = starts_window
        self._maxiter = maxiter
        self._lbfgs_maxfun = lbfgs_maxfun
        self._random_state = random_state
        sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=random_state)
        samples = sampler.random(n_starts)
        starts = []
        for i in range(n_starts):
            point = [low + samples[i, j] * (high - low) for j, (low, high) in enumerate(self.bounds)]
            starts.append(point)
        stopped_by = "all starts completed"
        starts_bests = []
        for x0 in starts:
            try:
                self._current_start += 1
                minimize(self._track_objective, x0, method='L-BFGS-B', bounds=self.bounds,
                         options={'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter, 'maxfun': lbfgs_maxfun, 'disp': False})
                self._starts_completed += 1
                starts_bests.append(self._best_cost)
                self._start_boundaries.append(self._n_evals)
                self._start_costs.append(self._best_cost)
                if _check_starts_convergence(starts_bests, starts_window, self.convergence_threshold):
                    stopped_by = "converged"
                    break
            except TimeoutException:
                stopped_by = "exceeded wall time"
                break
            except MaxEvalsException:
                stopped_by = "max function calls"
                break
        elapsed = time.time() - self._start_time
        thetas, radials, coil_positions, coil_currents = self._extract_best_result()
        self.results['Multi-start L-BFGS'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_fb_cost': self._best_fb_cost,
            'initial_fixed_cost': self._initial_fixed_cost,
            'initial_fb_cost': self._initial_fb_cost,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'stopping': stopped_by,
            'parameters': {'thetas': thetas, 'radials': radials},
            'coil_positions_top': coil_positions,
            'coil_currents': coil_currents,
            'convergence_history': list(self._convergence),
            'cost_history': list(self._history),
            'fb_failures': self._fb_failures,
            'starts_completed': self._starts_completed,
            'start_boundaries': self._start_boundaries,
            'start_costs': self._start_costs,
            'convergence_window': starts_window,
            'random_state': random_state,
            'flux_err_history': list(self._flux_err_history),
            'fb_cost_history': list(self._fb_cost_history),
            'maxiter': maxiter,
            'lbfgs_maxfun': lbfgs_maxfun,
        }
        print(f"L-BFGS: {self._n_evals} evals, {elapsed:.1f}s, "
              f"{self._starts_completed} starts, stopped by: {stopped_by}")
        return self.results['Multi-start L-BFGS']

    def run_bayesian(self, n_initial=None, acq_func='EI',
                     bayesian_stagnation_window=5,
                     local_optimize=True, refinement_window=5,
                     max_perms=None, acq_multiplier=10,
                     acq_dedup_tol=0.05, unique_refined_points=1,
                     random_state=1, maxiter=1000000000, lbfgs_maxfun=1000000000,
                     start_time=None):
        if n_initial is None:
            n_initial = int(round(25 * self.num_coils))
        if max_perms is None:
            max_perms = self.num_coils
        self._reset_tracking(start_time=start_time)
        self._maxiter = maxiter
        self._lbfgs_maxfun = lbfgs_maxfun
        self._current_method = 'Bayesian'
        space = [Real(low, high) for low, high in self.bounds]

        def stopping_callback():
            n = self._n_evals
            if n <= n_initial + bayesian_stagnation_window:
                return False
            running_min = np.minimum.accumulate(self._history)
            old_best = running_min[-(bayesian_stagnation_window + 1)]
            new_best = running_min[-1]
            if abs(old_best) > 0:
                rel_imp = (old_best - new_best) / abs(old_best)
                if rel_imp < self.convergence_threshold:
                    self._stopped_reason = "bayesian_stagnation"
                    return True
            elif new_best == 0:
                self._stopped_reason = "bayesian_stagnation"
                return True
            return False

        n_perms = min(max_perms, factorial(self.num_coils))
        gp_opt = Optimizer(space, base_estimator='gp',
                           n_initial_points=n_initial * n_perms,
                           initial_point_generator="sobol",
                           acq_func=acq_func, random_state=random_state)
        bayesian_stopped_by = "completed"
        try:
            flag = True
            while flag:
                x = gp_opt.ask()
                cost = self._track_objective(x)
                xs_perm, ys_perm = self._get_permuted_points(x, cost, max_perms=max_perms)
                gp_opt.tell(xs_perm, ys_perm)
                if stopping_callback():
                    flag = False
        except TimeoutException:
            bayesian_stopped_by = "exceeded wall time"
        except MaxEvalsException:
            bayesian_stopped_by = "max function calls"
        if self._stopped_reason:
            bayesian_stopped_by = self._stopped_reason
        bayesian_evals = self._n_evals
        elapsed_bayesian = time.time() - self._start_time
        print(f"Bayesian phase: {bayesian_evals} evals, {elapsed_bayesian:.1f}s, "
              f"stopped by: {bayesian_stopped_by}")

        pts_refined = 0
        refinement_stopped_by = None
        refinement_bests = []
        refinement_evals = []
        refinement_times = []
        refinement_costs = []
        bayesian_convergence = list(self._convergence)
        refinement_convergence = []
        n_acq_candidates = acq_multiplier * unique_refined_points
        n_acq_unique = None
        candidates = []
        if local_optimize and bayesian_stopped_by not in ("exceeded wall time", "max function calls"):
            self._convergence = []
            self._stopped_reason = None
            raw_candidates = gp_opt.ask(n_points=n_acq_candidates, strategy='cl_min')
            candidates = self._deduplicate_candidates(raw_candidates, tol=acq_dedup_tol,
                                                      max_unique=unique_refined_points)
            n_acq_unique = len(candidates)
            print(f"Acq candidates: {n_acq_candidates} raw -> {n_acq_unique} unique "
                  f"(target={unique_refined_points}, tol={acq_dedup_tol} m)")
            for cand in candidates:
                evals_before = self._n_evals
                time_before = time.time() - self._start_time
                try:
                    minimize(self._track_objective, np.array(cand), method='L-BFGS-B',
                             bounds=self.bounds, options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': maxiter, 'maxfun': lbfgs_maxfun, 'disp': False})
                    pts_refined += 1
                    refinement_bests.append(self._best_cost)
                    refinement_evals.append(self._n_evals - evals_before)
                    refinement_times.append(time.time() - self._start_time - time_before)
                    refinement_costs.append(self._best_cost)
                    if _check_starts_convergence(refinement_bests, refinement_window,
                                                 self.convergence_threshold):
                        refinement_stopped_by = "converged"
                        break
                except TimeoutException:
                    refinement_stopped_by = "exceeded wall time"
                    break
                except MaxEvalsException:
                    refinement_stopped_by = "max function calls"
                    break
            refinement_convergence = list(self._convergence)
            if refinement_stopped_by is None:
                refinement_stopped_by = "all refinements completed"

        elapsed = time.time() - self._start_time
        stopped_by = refinement_stopped_by or bayesian_stopped_by
        thetas, radials, coil_positions, coil_currents = self._extract_best_result()
        self.results['Bayesian'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_fb_cost': self._best_fb_cost,
            'initial_fixed_cost': self._initial_fixed_cost,
            'initial_fb_cost': self._initial_fb_cost,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'stopping': stopped_by,
            'parameters': {'thetas': thetas, 'radials': radials},
            'coil_positions_top': coil_positions,
            'coil_currents': coil_currents,
            'convergence_history': bayesian_convergence + refinement_convergence,
            'bayesian_convergence_history': bayesian_convergence,
            'refinement_convergence_history': refinement_convergence,
            'cost_history': list(self._history),
            'fb_failures': self._fb_failures,
            'n_initial': n_initial,
            'n_perms': n_perms,
            'n_bayesian_evals': bayesian_evals,
            'n_gp_observations': bayesian_evals * n_perms,
            'time_bayesian_phase': elapsed_bayesian,
            'pts_refined': pts_refined,
            'refinement_evals': refinement_evals,
            'refinement_times': refinement_times,
            'refinement_costs': refinement_costs,
            'bayesian_stopping': bayesian_stopped_by,
            'acq_multiplier': acq_multiplier,
            'n_acq_candidates': n_acq_candidates,
            'acq_dedup_tol': acq_dedup_tol,
            'unique_refined_points': unique_refined_points,
            'n_acq_unique': n_acq_unique,
            'refinement_candidates': [list(c) for c in candidates],
            'convergence_window': bayesian_stagnation_window,
            'refinement_window': refinement_window,
            'refinement_stopping': refinement_stopped_by,
            'random_state': random_state,
            'flux_err_history': list(self._flux_err_history),
            'fb_cost_history': list(self._fb_cost_history),
            'maxiter': maxiter,
            'lbfgs_maxfun': lbfgs_maxfun,
        }
        print(f"Total: {self._n_evals} evals, {elapsed:.1f}s, "
              f"refined {pts_refined} pts, stopped by: {stopped_by}")
        return self.results['Bayesian']

    def run_multiple(self, method, n_runs=1, base_seed=1, **kwargs):
        key_map = {'bayesian': 'Bayesian', 'multistart_lbfgs': 'Multi-start L-BFGS'}
        if method not in key_map:
            raise ValueError(f"method must be one of {list(key_map)}")
        key = key_map[method]
        run_fn = self.run_bayesian if method == 'bayesian' else self.run_multistart_lbfgs
        runs = []
        for i in range(n_runs):
            seed = base_seed + i
            print(f"\n[{method}] run {i+1}/{n_runs}, random_state={seed}, coils={self.num_coils}, weight_fb={self.weight_fb:.0e}")
            result = run_fn(random_state=seed, **kwargs)
            runs.append(dict(result))
        self.all_runs[key] = runs
        best = min(runs, key=lambda r: r['best_cost'])
        self.results[key] = best
        print(f"\n{key}: best cost {best['best_cost']:.4e} "
              f"over {n_runs} runs")
        return runs

    def summary(self):
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPARISON RESULTS")
        print("=" * 70)
        data = []
        for method, res in self.results.items():
            data.append({
                'Method': method,
                'Best Cost': res['best_cost'],
                'Evals Used': res['n_evals'],
                'Time (s)': res['time'],
                'Stopping': res['stopping'],
            })
        df = pd.DataFrame(data)
        df = df.sort_values('Best Cost')
        print(df.to_string(index=False))
        best = df.iloc[0]
        print(f"\nWinner: {best['Method']} with cost {best['Best Cost']:.6e}")
        return df

    def plot_result(self, log_scale=True, figsize=(16, 10)):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        sorted_methods, colors = self._get_sorted_methods_and_colors()
        for (method, res), color in zip(sorted_methods, colors):
            conv = res['convergence_history']
            ax1.plot(range(1, len(conv) + 1), conv,
                     label=f"{method} ({res['best_cost']:.2e})", color=color, linewidth=2)
        if self.brute_force_cost is not None:
            ax1.axhline(y=self.brute_force_cost, color='gray', linestyle='--',
                        linewidth=1.5, alpha=0.7,
                        label=f'Brute force ({self.brute_force_cost:.2e})')
        ax1.set_xlabel('Function Evaluations', fontsize=12)
        ax1.set_ylabel('Best Cost Found', fontsize=12)
        ax1.set_title('Convergence Speed', fontsize=14)
        if log_scale:
            ax1.set_yscale('log')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        params_text = '\n'.join([
            f'NUM_COILS = {self.num_coils}',
            f'MAX_EVALS = {self.max_evals}',
            f'ALPHA = {self.alpha}',
            f'OMEGA = {self.omega:.0e}',
            f'REG_IN = {self.reg_in:.0e}',
            f'WEIGHT_FB = {self.weight_fb:.0e}',
        ])
        ax1.text(0.8, 0.5, params_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        methods_names = [m for m, _ in sorted_methods]
        costs = np.array([r['best_cost'] for _, r in sorted_methods])
        ax2.barh(methods_names, costs, color=colors)
        ax2.set_xlabel('Final Cost', fontsize=12)
        ax2.set_title('Final Cost by Method', fontsize=14)
        ax2.axvline(x=costs[0], color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        ax2.grid(True, alpha=0.3, axis='x')
        if self.coil_center_cand1 is not None and self.coil_center_cand2 is not None:
            self._plot_position_space_boundaries(ax3)
            for (method, res), color in zip(sorted_methods, colors):
                coil_locs, _, _ = self._compute_flux_for_params(res['best_params'])
                for loc in coil_locs:
                    ax3.add_patch(plt.Rectangle((loc[0]-0.035, loc[1]-0.035), 0.07, 0.07,
                                                facecolor=color, edgecolor=color, alpha=0.6))
                    ax3.add_patch(plt.Rectangle((loc[0]-0.035, -loc[1]-0.035), 0.07, 0.07,
                                                facecolor=color, edgecolor=color, alpha=0.6))
            ax3.set_xlabel('R [m]', fontsize=12)
            ax3.set_ylabel('Z [m]', fontsize=12)
            ax3.set_title('Coil Placement (All Methods)', fontsize=14)
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
        if self.r_bnd is not None and self.psi_bnd is not None:
            theta = np.arctan2(self.r_bnd[:, 1], self.r_bnd[:, 0] - self.o_point[0])
            psi_desired = self.psi_bnd[1:] - self.psi_bnd[0]
            ax4.plot(theta[1:], psi_desired, 'ko', markersize=3, label='Desired', alpha=0.5)
            for (method, res), color in zip(sorted_methods, colors):
                _, psi_computed, _ = self._compute_flux_for_params(res['best_params'])
                ax4.plot(theta[1:], psi_computed, '+', color=color, markersize=4,
                         label=method, alpha=0.7)
            ax4.set_xlabel(r'$\theta$ [rad]', fontsize=12)
            ax4.set_ylabel(r'$\psi_{boundary}$', fontsize=12)
            ax4.set_title('Desired vs Computed Flux', fontsize=14)
            ax4.legend(fontsize=8, loc='best')
            ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_convergence_vs_time(self, log_scale=True, figsize=(12, 6)):
        if not self.results:
            print("No results to plot")
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sorted_methods, colors = self._get_sorted_methods_and_colors()
        for (method, res), color in zip(sorted_methods, colors):
            ax.plot(res['times'], res['convergence_history'],
                    label=f"{method} ({res['best_cost']:.2e})", color=color, linewidth=2, alpha=0.8)
        if self.brute_force_cost is not None:
            ax.axhline(y=self.brute_force_cost, color='gray', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'Brute force ({self.brute_force_cost:.2e})')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Best Cost Found', fontsize=12)
        ax.set_title('Convergence vs Time', fontsize=14)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def save_results_to_json(self, filename):
        if not self.results:
            print("No results to save")
            return
        save_data = {
            'optimization_settings': {
                'num_coils': int(self.num_coils),
                'max_evals': int(self.max_evals) if self.max_evals is not None else None,
                'max_time': float(self.max_time),
                'convergence_threshold': float(self.convergence_threshold),
                'omega': float(self.omega),
                'dist_th': float(self.dist_th),
                'reg_in': float(self.reg_in),
                'rfil': float(self.rfil),
                'alpha': float(self.alpha),
                'weight_fb': float(self.weight_fb),
                'maxiter': self._maxiter,
                'lbfgs_maxfun': self._lbfgs_maxfun,
            },
            'methods': {}
        }
        if self.brute_force_cost is not None:
            save_data['brute_force_cost'] = float(self.brute_force_cost)
        for method, res in self.results.items():
            method_data = {
                'best_cost': float(res['best_cost']),
                'best_flux_err': float(res['best_flux_err']) if res['best_flux_err'] is not None else None,
                'best_fb_cost': float(res['best_fb_cost']) if res.get('best_fb_cost') is not None else None,
                'initial_fixed_cost': float(res['initial_fixed_cost']) if res.get('initial_fixed_cost') is not None else None,
                'initial_fb_cost': float(res['initial_fb_cost']) if res.get('initial_fb_cost') is not None else None,
                'n_evals': int(res['n_evals']),
                'time': float(res['time']),
                'stopping': res['stopping'],
            }
            for key in ['fb_failures', 'starts_completed', 'convergence_window', 'random_state',
                        'n_initial', 'n_perms', 'n_bayesian_evals', 'n_gp_observations',
                        'pts_refined', 'n_acq_candidates', 'n_acq_unique',
                        'unique_refined_points', 'refinement_window', 'acq_multiplier']:
                if key in res and res[key] is not None:
                    method_data[key] = int(res[key])
            for key in ['time_bayesian_phase', 'acq_dedup_tol']:
                if key in res and res[key] is not None:
                    method_data[key] = float(res[key])
            for key in ['bayesian_stopping', 'refinement_stopping']:
                if key in res and res[key] is not None:
                    method_data[key] = res[key]
            method_data['parameters'] = res['parameters']
            method_data['coil_positions_top'] = res['coil_positions_top']
            method_data['coil_currents'] = res['coil_currents']
            for key in ['start_boundaries', 'refinement_evals']:
                if key in res:
                    method_data[key] = [int(x) for x in res[key]]
            for key in ['start_costs', 'refinement_times', 'refinement_costs']:
                if key in res:
                    method_data[key] = [float(x) for x in res[key]]
            for key in ['bayesian_convergence_history', 'refinement_convergence_history',
                        'refinement_candidates']:
                if key in res:
                    method_data[key] = res[key]
            method_data['convergence_history'] = res['convergence_history']
            method_data['cost_history'] = [float(c) for c in res['cost_history']]
            method_data['times'] = [float(t) for t in res['times']]
            method_data['flux_err_history'] = res['flux_err_history']
            method_data['fb_cost_history'] = res['fb_cost_history']
            save_data['methods'][method] = method_data
        if self.all_runs:
            save_data['all_runs'] = {}
            for method_key, runs in self.all_runs.items():
                save_data['all_runs'][method_key] = [
                    {k: (v.tolist() if hasattr(v, 'tolist') else v)
                     for k, v in run.items() if k != 'best_params'}
                    for run in runs
                ]
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved results to {filename}")


# ============================================
# Combined objective factory
# ============================================

def make_combined_objective(alpha, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
                            coil_center_cand1, coil_center_cand2, lim,
                            r_bnd, psi_bnd, weight_fb, NUM_COILS, RFIL,
                            REG_IN, OMEGA, DIST_TH, theta_range, inner, outer):
    def objective(params):
        thetas = params[:NUM_COILS]
        radials = params[NUM_COILS:]

        locs = []
        for theta, rho in zip(thetas, radials):
            R_pos = (1 - rho) * np.interp(theta, theta_range, inner[:, 0]) + rho * np.interp(theta, theta_range, outer[:, 0])
            Z_pos = (1 - rho) * np.interp(theta, theta_range, inner[:, 1]) + rho * np.interp(theta, theta_range, outer[:, 1])
            locs.append([R_pos, Z_pos])

        coil_centers_3x3 = []
        for loc in locs:
            centers_top = [[loc[0] + 2*RFIL*dx, loc[1] + 2*RFIL*dy]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            centers_bot = [[loc[0] + 2*RFIL*dx, -loc[1] + 2*RFIL*dy]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            coil_centers_3x3.append(centers_top)
            coil_centers_3x3.append(centers_bot)

        n_bnd = psi_bnd.shape[0]
        n_coils_total = len(coil_centers_3x3)
        con = np.zeros((n_bnd - 1 + n_coils_total, n_coils_total))
        for i, filament_set in enumerate(coil_centers_3x3):
            flux_tmp = np.zeros((n_bnd,))
            for fil in filament_set:
                flux_tmp += eval_green(r_bnd, fil)
            con[:n_bnd - 1, i] = flux_tmp[1:] - flux_tmp[0]
            con[n_bnd - 1 + i, i] = REG_IN
        err = np.zeros((n_bnd - 1 + n_coils_total,))
        err[:n_bnd - 1] = psi_bnd[1:] - psi_bnd[0]
        currs, residuals, _, _ = np.linalg.lstsq(con, err, rcond=None)
        if len(residuals) > 0:
            fixed_cost = residuals[0]
        else:
            fixed_cost = np.linalg.norm(np.dot(con, currs) - err) ** 2

        objective.last_flux_err = fixed_cost

        fb_cost_raw = _free_boundary_cost(params, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
                                          coil_center_cand1, coil_center_cand2, lim,
                                          weight_fb, NUM_COILS)
        failed = fb_cost_raw >= 1e6

        if objective.norm_fixed is None:
            objective.norm_fixed = fixed_cost
        if objective.norm_fb is None and not failed:
            objective.norm_fb = fb_cost_raw

        if failed:
            objective.fb_failures += 1
            fb_cost = objective.norm_fb if objective.norm_fb is not None else 1e6
        else:
            fb_cost = fb_cost_raw
        objective.last_fb_cost = fb_cost

        norm_fixed = fixed_cost / objective.norm_fixed if objective.norm_fixed > 0 else fixed_cost
        norm_fb = fb_cost / objective.norm_fb if objective.norm_fb is not None and objective.norm_fb > 0 else fb_cost

        dist_penalty = OMEGA * np.sum(np.maximum(DIST_TH - np.diff(np.sort(thetas)), 0.0) ** 2)

        return (1 - alpha) * norm_fixed + alpha * norm_fb + dist_penalty

    objective.norm_fixed = None
    objective.norm_fb = None
    objective.fb_failures = 0
    return objective


# ============================================
# Main function
# ============================================

def main(mygs, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim,
         methods=None, **kwargs):
    NUM_COILS = kwargs.get('NUM_COILS', 4)
    MAX_EVALS = kwargs.get('MAX_EVALS', 2**18)
    MAX_TIME = kwargs.get('MAX_TIME', 86400)
    # CONVERGENCE_THRESHOLD = kwargs.get('CONVERGENCE_THRESHOLD', 0.001)
    CONVERGENCE_THRESHOLD = kwargs.get('CONVERGENCE_THRESHOLD', 0.01)
    OMEGA = kwargs.get('OMEGA', 1e-2)
    DIST_TH = kwargs.get('DIST_TH', 5.0)
    REG_IN = kwargs.get('REG_IN', 1e-6)
    RFIL = kwargs.get('RFIL', 0.01)
    N_RUNS = kwargs.get('N_RUNS', 1)
    RUN_FOLDER = kwargs.get('RUN_FOLDER', 'combined')
    ALPHA = kwargs.get('ALPHA', 0.75)
    WEIGHT_FB = kwargs.get('WEIGHT_FB', 1e-2)

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

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPARISON (COMBINED BOUNDARY)")
    print("=" * 60)
    print(f"Coils:{NUM_COILS} alpha:{ALPHA} weight_fb:{WEIGHT_FB:.0e} reg_in:{REG_IN:.0e}")
    print(f"Max evals:{MAX_EVALS} threshold:{CONVERGENCE_THRESHOLD}")
    print("1 eval = lstsq + TokaMaker GS solve (mesh rebuild per eval)")
    print("=" * 60 + "\n")

    comparison = OptimizationComparison(
        objective, bounds,
        max_time=MAX_TIME, max_evals=MAX_EVALS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        NUM_COILS=NUM_COILS, OMEGA=OMEGA, DIST_TH=DIST_TH,
        REG_IN=REG_IN, RFIL=RFIL, ALPHA=ALPHA, WEIGHT_FB=WEIGHT_FB
    )
    comparison.set_problem_data(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                                mygs.o_point, eval_green)

    base = os.path.join(_BASE_DIR, f'examples/comparisons/combined_boundary_DIIID/{RUN_FOLDER}/'
                        f'alpha:{ALPHA},weight:{WEIGHT_FB:.0e},lambda:{REG_IN:.0e},coils:{NUM_COILS}')

    existing_runs = 1
    while os.path.exists(os.path.join(base, f'run_{existing_runs:02d}', 'results.json')):
        existing_runs += 1

    seed_offset = existing_runs

    if N_RUNS == 1:
        run_idx = existing_runs
        while os.path.exists(os.path.join(base, f'run_{run_idx:02d}', 'results.json')):
            run_idx += 1
        foldername_pre = os.path.join(base, f'run_{run_idx:02d}')
        os.makedirs(foldername_pre, exist_ok=True)
        comparison.checkpoint_path = os.path.join(foldername_pre, 'checkpoint.json')
    else:
        os.makedirs(base, exist_ok=True)
        comparison.checkpoint_path = os.path.join(base, f'checkpoint_{os.getpid()}.json')

    if methods is None:
        methods = ['multistart_lbfgs', 'bayesian']

    PROCESS_START = kwargs.get('PROCESS_START', None)
    if 'multistart_lbfgs' in methods:
        print(f"Running Multi-start L-BFGS... coils={NUM_COILS}, weight_fb={WEIGHT_FB:.0e}")
        if N_RUNS > 1:
            comparison.run_multiple('multistart_lbfgs', n_runs=N_RUNS,
                                    base_seed=seed_offset, 
                                    starts_window=5, 
                                    lbfgs_maxfun=100)
        else:
            comparison.run_multistart_lbfgs(starts_window=5, 
                                            random_state=seed_offset, 
                                            lbfgs_maxfun=100,
                                            start_time=PROCESS_START)

    if 'bayesian' in methods:
        print(f"Running Bayesian Optimization... coils={NUM_COILS}, weight_fb={WEIGHT_FB:.0e}")
        if N_RUNS > 1:
            comparison.run_multiple('bayesian', n_runs=N_RUNS, 
                                    base_seed=seed_offset,
                                    bayesian_stagnation_window=25, 
                                    unique_refined_points=3, 
                                    acq_multiplier=10,
                                    lbfgs_maxfun=100)
        else:
            comparison.run_bayesian(bayesian_stagnation_window=25,
                                    random_state=seed_offset, 
                                    unique_refined_points=3, 
                                    acq_multiplier=10,
                                    lbfgs_maxfun=100,
                                    start_time=PROCESS_START,)

    summary = comparison.summary()

    if comparison.all_runs:
        n_individual = max(len(runs) for runs in comparison.all_runs.values())
        orig_results = comparison.results
        orig_all_runs = comparison.all_runs
        for i in range(n_individual):
            run_idx = existing_runs
            while os.path.exists(os.path.join(base, f'run_{run_idx:02d}', 'results.json')):
                run_idx += 1
            foldername = os.path.join(base, f'run_{run_idx:02d}')
            os.makedirs(foldername, exist_ok=True)
            comparison.results = {k: runs[i] for k, runs in orig_all_runs.items() if i < len(runs)}
            comparison.all_runs = {}
            fig_i = comparison.plot_result()
            time_fig_i = comparison.plot_convergence_vs_time(log_scale=True)
            fig_i.savefig(f'{foldername}/convergence_plot.png', dpi=150, bbox_inches='tight')
            time_fig_i.savefig(f'{foldername}/convergence_vs_time_plot.png', dpi=150, bbox_inches='tight')
            comparison.save_results_to_json(f'{foldername}/results.json')
            plt.close('all')
            print(f"Saved run {i} to: {foldername}/")
        comparison.results = orig_results
        comparison.all_runs = orig_all_runs 
    else:
        foldername = foldername_pre
        fig = comparison.plot_result()
        time_fig = comparison.plot_convergence_vs_time(log_scale=True)
        fig.savefig(f'{foldername}/convergence_plot.png', dpi=150, bbox_inches='tight')
        time_fig.savefig(f'{foldername}/convergence_vs_time_plot.png', dpi=150, bbox_inches='tight')
        comparison.save_results_to_json(f'{foldername}/results.json')
        plt.close('all')
        print(f"Saved all plots and results to: {foldername}/")

    return comparison, summary


# ============================================
# Parallel worker
# ============================================

def parallel_case(weight_fb, num_coils, ntrials, run_folder, nthreads, alpha):
    t0 = time.time()
    global _MEM_LOG_DIR
    _MEM_LOG_DIR = os.path.join(_BASE_DIR,
        f'examples/comparisons/combined_boundary_DIIID/{run_folder}/'
        f'alpha:{alpha},weight:{weight_fb:.0e},lambda:1e-06,coils:{num_coils}')
    os.makedirs(_MEM_LOG_DIR, exist_ok=True)
    tmp_dir = os.path.join(_BASE_DIR, 'tmp', f'temp_combined_{weight_fb}_{num_coils}')
    try:
        shutil.rmtree(tmp_dir)
    except FileNotFoundError:
        pass
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    eqdsk = read_eqdsk(os.path.join(_BASE_DIR, 'examples/data/eqdsk/g192185.02440'))
    LCFS_contour = eqdsk['rzout'].copy()
    fixed_LCFS = LCFS_contour

    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    mesh_dx = 0.015
    gs_mesh = gs_Domain()
    gs_mesh.define_region('plasma', mesh_dx, 'plasma')
    gs_mesh.add_polygon(LCFS_contour, 'plasma')
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()

    myOFT = OFT_env(nthreads=nthreads)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False

    F0 = eqdsk['rcentr'] * eqdsk['bcentr']
    mygs.setup(order=2, F0=F0)
    mygs.set_targets(Ip=eqdsk['ip'], pax=eqdsk['pres'][0])

    print(f"Solving fixed-boundary EQ for num_coils={num_coils}, weight_fb={weight_fb}...")
    mygs.init_psi()
    mygs.solve()

    # fixed_mag_axis = np.array(mygs.o_point)
    fixed_mag_axis = np.array([1.77764093, -0.04014656])

    os.chdir(_BASE_DIR)

    comparison, summary = main(
        mygs=mygs,
        myOFT=myOFT,
        eqdsk=eqdsk,
        fixed_mag_axis=fixed_mag_axis,
        fixed_LCFS=fixed_LCFS,
        lim=lim,
        # methods=["multistart_lbfgs", "bayesian"],
        methods = ["multistart_lbfgs"], 
        # methods = ["bayesian"], 
        NUM_COILS=num_coils,
        REG_IN=1e-6,
        ALPHA=alpha,
        WEIGHT_FB=weight_fb,
        MAX_EVALS=2**18,
        MAX_TIME=3*86400,
        N_RUNS=ntrials,
        RUN_FOLDER=run_folder,
        PROCESS_START=t0,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return summary


# ============================================
# Per-case run logging
# ============================================

def _make_case_logger(base, weight_fb, num_coils):
    name = f'run_{weight_fb}_{num_coils}'
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = logging.FileHandler(os.path.join(base, 'run.log'))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def logged_parallel_case(weight_fb, num_coils, ntrials, run_folder, nthreads, alpha):
    base = os.path.join(_BASE_DIR,
        f'examples/comparisons/combined_boundary_DIIID/{run_folder}/'
        f'alpha:{alpha},weight:{weight_fb:.0e},lambda:1e-06,coils:{num_coils}')
    os.makedirs(base, exist_ok=True)

    # redirect fb_crashes logger to per-case file
    crash_log = logging.getLogger('fb_crashes')
    for h in crash_log.handlers[:]:
        crash_log.removeHandler(h)
    crash_handler = logging.FileHandler(os.path.join(base, 'fb_crashes.log'))
    crash_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    crash_log.addHandler(crash_handler)

    log = _make_case_logger(base, weight_fb, num_coils)
    params = f"weight_fb={weight_fb:.0e} num_coils={num_coils} ntrials={ntrials} alpha={alpha}"
    t0 = time.time()
    log.info(f"START {params}")
    try:
        result = parallel_case(weight_fb, num_coils, ntrials, run_folder, nthreads, alpha)
        log.info(f"DONE {params} elapsed={time.time()-t0:.1f}s")
        return result
    except Exception as e:
        log.error(f"FAILED {params} elapsed={time.time()-t0:.1f}s error={e}")
        raise


# ============================================
# Entry point
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='combined',
                        help='Output folder under examples/comparisons/combined_boundary_DIIID/')
    parser.add_argument('--nprocs', type=int, default=20,
                        help='Number of parallel processes')
    parser.add_argument('--ntrials', type=int, default=20,
                        help='Optimization trials (N_RUNS) per case')
    parser.add_argument('--nthreads', type=int, default=2,
                        help='OFT threads per process')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Blending weight for free-boundary cost')
    args = parser.parse_args()

    weights = [1e-4, 1e-3, 1e-2, 1e-1]
    coils = [3]

    # weights = [1e-4, 1e-3, 1e-2, 1e-1]
    # coils = [3]

    pool = Pool(processes=args.nprocs)
    async_results = {}
    for w in weights:
        for nc in coils:
            async_results[(w, nc)] = pool.apply_async(
                logged_parallel_case, args=(w, nc, args.ntrials, args.folder, args.nthreads, args.alpha)
            )

    pool.close()
    pool.join()

    for (w, nc), result in async_results.items():
        try:
            result.get()
            print(f"weight_fb={w}, num_coils={nc}: done")
        except Exception as e:
            print(f"weight_fb={w}, num_coils={nc}: failed - {e}")
