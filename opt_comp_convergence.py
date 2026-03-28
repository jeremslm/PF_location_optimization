"""
Optimization Comparison for PF Coil Placement
==============================================

Compare multi-start L-BFGS and Bayesian optimization methods for PF coil
placement. Stopping criteria are per-start (not per function call):
after N completed starts without improvement, the method stops.

Uses brute force parameters: OMEGA=1e-7, DIST_TH=5.0, eqdsk=g192185.02440 (DIIID).
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
# from itertools import permutations as _iperms
from math import factorial
from scipy.optimize import minimize
from scipy.stats import qmc
from skopt import Optimizer
from skopt.space import Real
import os
import sys
import json

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path

tokamaker_python_path = os.getenv("OFT_ROOTPATH")
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, 'python'))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, eval_green
from helper_fct import resize_polygon, update_boundary, place_points_pol_rad, make_3x3_thick
from OFT_pf_coil_opt_fct import CoilPositionSpace


class TimeoutException(Exception):
    pass


class MaxEvalsException(Exception):
    pass


def _check_starts_convergence(starts_bests, window, threshold):
    """Check if best cost has improved over last `window` completed starts.

    Returns True if converged (no meaningful improvement).
    """
    n = len(starts_bests)
    if n <= window:
        return False
    old_best = starts_bests[n - window - 1]
    new_best = starts_bests[-1]
    if old_best > 0:
        rel_imp = (old_best - new_best) / abs(old_best)
        return rel_imp < threshold
    return False


class OptimizationComparison:
    """
    Compare optimization methods with per-start convergence stopping.

    Each method lets individual L-BFGS runs converge naturally (scipy handles
    that). Stopping is based on whether the global best improves across
    completed starts/refinements, plus safety limits on wall-clock time
    and total function evaluations.
    """

    def __init__(self, objective_func, bounds, max_time=86400,
                 max_evals=None, convergence_threshold=0.001,
                 NUM_COILS=3, OMEGA=1e-7, DIST_TH=5.0, REG_IN=1e-7, RFIL=0.01):
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

        self.results = {}
        self.all_runs = {}

        # Problem data for plotting (set via set_problem_data)
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

    def _reset_tracking(self):
        self._n_evals = 0
        self._history = []
        self._x_history = []
        self._times = []
        self._best_cost = float('inf')
        self._best_flux_err = None
        self._best_params = None
        self._start_time = time.time()
        self._convergence = []
        self._stopped_reason = None

    def _track_objective(self, params):
        """Evaluate objective, track history, enforce time/eval limits."""
        if time.time() - self._start_time > self.max_time:
            raise TimeoutException("Wall-clock time limit reached")

        if self.max_evals is not None and self._n_evals >= self.max_evals:
            raise MaxEvalsException("Maximum function evaluations reached")

        self._n_evals += 1
        params = np.asarray(params)
        cost = self.objective(params)
        self._history.append(cost)
        self._x_history.append(params.copy())
        self._times.append(time.time() - self._start_time)

        flux_err = getattr(self.objective, 'last_flux_err', None)

        if cost < self._best_cost:
            self._best_cost = cost
            self._best_params = params.copy()
            if flux_err is not None:
                self._best_flux_err = flux_err

        self._convergence.append(self._best_cost)
        return cost

    def _compute_flux_for_params(self, params):
        """Compute flux and coil positions for given parameters."""
        num_coils = len(params) // 2
        thetas = params[:num_coils]
        radials = params[num_coils:]

        inner_arc = self.coil_center_cand1[:len(self.coil_center_cand1)//2]
        outer_arc = self.coil_center_cand2[:len(self.coil_center_cand2)//2]

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
            con[:n_bnd-1, i] = flux_tmp[1:] - flux_tmp[0]
            con[n_bnd-1+i, i] = self.reg_in

        err = np.zeros((n_bnd - 1 + n_coils_total,))
        err[:n_bnd-1] = self.psi_bnd[1:] - self.psi_bnd[0]
        currs, _, _, _ = np.linalg.lstsq(con, err, rcond=None)
        psi_computed = np.dot(con, currs)[:n_bnd - 1]

        return coil_locs, psi_computed, currs

    def _get_permuted_points(self, x, cost, max_perms=None):
        """Return permutations of coil ordering for x with the same cost.

        For N coils, params are [theta_0..theta_{N-1}, rho_0..rho_{N-1}].
        Swapping coil i and j simultaneously in both halves leaves the cost
        unchanged. If max_perms is set, a random subset is returned instead
        of all N! to keep GP training set size manageable.
        """
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

    def _extract_best_result(self):
        """Extract parameters, coil positions, and currents from best result."""
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

    # ========================================
    # Plotting helpers
    # ========================================

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

    def _plot_coils_on_axis(self, ax, coil_locs, color, dx=0.035, dy=0.035):
        for loc in coil_locs:
            rect_top = plt.Rectangle((loc[0]-dx, loc[1]-dy), 2*dx, 2*dy,
                                    facecolor=color, edgecolor='black',
                                    alpha=0.7, linewidth=0.5)
            ax.add_patch(rect_top)
            rect_bot = plt.Rectangle((loc[0]-dx, -loc[1]-dy), 2*dx, 2*dy,
                                    facecolor=color, edgecolor='black',
                                    alpha=0.7, linewidth=0.5)
            ax.add_patch(rect_bot)

    # ========================================
    # Optimization methods
    # ========================================

    def run_multistart_lbfgs(self, n_starts=262144, ftol=1e-9, gtol=1e-6,
                             starts_window=50, random_state=42):
        """
        Multi-start L-BFGS-B with Sobol sampling.

        Each L-BFGS start runs to its own local convergence. Stops when
        `starts_window` consecutive completed starts show no improvement,
        or when max evals / wall-clock time is hit.
        """
        self._reset_tracking()

        sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=random_state)
        samples = sampler.random(n_starts)

        starts = []
        for i in range(n_starts):
            point = []
            for j, (low, high) in enumerate(self.bounds):
                point.append(low + samples[i, j] * (high - low))
            starts.append(point)

        starts_completed = 0
        stopped_by = "all starts completed"
        starts_bests = []
        start_boundaries = []   # cumulative n_evals at end of each start
        start_costs = []        # best cost at end of each start

        for x0 in starts:
            try:
                minimize(
                    self._track_objective, x0,
                    method='L-BFGS-B', bounds=self.bounds,
                    options={'ftol': ftol, 'gtol': gtol, 'disp': False}
                )
                starts_completed += 1
                starts_bests.append(self._best_cost)
                start_boundaries.append(self._n_evals)
                start_costs.append(self._best_cost)

                if _check_starts_convergence(starts_bests, starts_window,
                                             self.convergence_threshold):
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
            'starts_completed': starts_completed,
            'start_boundaries': start_boundaries,
            'start_costs': start_costs,
            'random_state': random_state,
        }

        return self.results['Multi-start L-BFGS']

    def _deduplicate_candidates(self, candidates, tol=1e-2, max_unique=None):
        """Remove near-duplicate candidates in normalized parameter space.

        Uses Chebyshev (L-inf) distance in [0,1]^d space.  A candidate is
        a duplicate only if it agrees with an existing unique point in EVERY
        dimension within tol — it is kept if it differs meaningfully in at
        least one parameter.  tol=0.05 means a 5% range difference in any
        single parameter is enough to keep the candidate.

        Parameters
        ----------
        candidates : list of array-like
        tol : float
            Chebyshev threshold in normalized space.
        max_unique : int or None
            If set, return at most this many unique candidates (taken in
            order, so the highest-acquisition ones come first).
        """
        if not candidates:
            return []
        lows = np.array([lo for lo, _ in self.bounds])
        ranges = np.array([hi - lo for lo, hi in self.bounds])
        ranges[ranges == 0] = 1.0

        def _norm(c):
            return (np.asarray(c) - lows) / ranges

        unique = [candidates[0]]
        unique_norm = [_norm(candidates[0])]
        for c in candidates[1:]:
            if max_unique is not None and len(unique) >= max_unique:
                break
            cn = _norm(c)
            if all(np.max(np.abs(cn - u)) > tol for u in unique_norm):
                unique.append(c)
                unique_norm.append(cn)
        return unique

    def run_bayesian(self, n_initial=None, acq_func='EI',
                     bayesian_stagnation_window=50,
                     local_optimize=True, refinement_window=50,
                     max_perms=None, acq_multiplier=10,
                     acq_dedup_tol=1e-2, unique_refined_points=5,
                     random_state=1):
        """
        Bayesian Optimization with GP, then L-BFGS refinement.

        Phase 1 (Bayesian): runs gp_minimize. Stops when the best value
        hasn't improved in `bayesian_stagnation_window` GP-guided iterations
        (each iteration = 1 point).

        Phase 2 (L-BFGS refinement): asks the fitted GP's acquisition
        function for n_acq_candidates = acq_multiplier * unique_refined_points
        points via the constant-liar batch strategy, deduplicates them with
        Chebyshev tolerance acq_dedup_tol, and refines up to
        unique_refined_points unique candidates on the real cost.
        unique_refined_points is a cap — if fewer unique candidates survive
        deduplication, only those are refined.  Stops when `refinement_window`
        consecutive completed refinements
        show no improvement, or when max evals / wall-clock time is hit.
        """
        if n_initial is None:
            n_initial = int(round(25 * self.num_coils ** 1.5))

        if max_perms is None:
            max_perms = self.num_coils

        self._reset_tracking()
        space = [Real(low, high) for low, high in self.bounds]

        def stopping_callback():
            n = self._n_evals
            if n <= n_initial + bayesian_stagnation_window:
                return False
            running_min = np.minimum.accumulate(self._history)
            old_best = running_min[-(bayesian_stagnation_window + 1)]
            new_best = running_min[-1]
            if old_best > 0:
                rel_imp = (old_best - new_best) / abs(old_best)
                if rel_imp < self.convergence_threshold:
                    self._stopped_reason = "bayesian_stagnation"
                    return True
            return False

        n_perms = min(max_perms, factorial(self.num_coils))
        gp_opt = Optimizer(
            space,
            base_estimator='gp',
            n_initial_points=n_initial * n_perms,
            initial_point_generator="sobol",
            acq_func=acq_func,
            random_state=random_state,
        )

        # Phase 1: Bayesian exploration
        bayesian_stopped_by = "completed"
        try:
            flag = True 
            while flag:
                x = gp_opt.ask()
                cost = self._track_objective(x)

                # Tell permutations — same cost by coil symmetry
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
        print(f"  Bayesian phase: {bayesian_evals} evals, "
              f"{elapsed_bayesian:.1f}s, stopped by: {bayesian_stopped_by}")

        # Phase 2: L-BFGS refinement
        pts_refined = 0
        refinement_stopped_by = None
        refinement_bests = []
        refinement_evals = []        # function calls per refined point
        refinement_times = []        # wall time per refined point
        refinement_costs = []        # best cost at end of each refinement
        bayesian_convergence = list(self._convergence)
        refinement_convergence = []
        n_acq_candidates = acq_multiplier * unique_refined_points
        n_acq_unique = None
        if local_optimize and bayesian_stopped_by not in ("exceeded wall time", "max function calls"):
            self._convergence = []
            self._stopped_reason = None

            raw_candidates = gp_opt.ask(n_points=n_acq_candidates, strategy='cl_min')
            candidates = self._deduplicate_candidates(
                raw_candidates, tol=acq_dedup_tol, max_unique=unique_refined_points
            )
            n_acq_unique = len(candidates)
            print(f"  Acq candidates: {n_acq_candidates} raw -> {n_acq_unique} unique "
                  f"(target={unique_refined_points}, tol={acq_dedup_tol})")

            for cand in candidates:
                evals_before = self._n_evals
                time_before = time.time() - self._start_time
                try:
                    start = np.array(cand)
                    minimize(
                        self._track_objective, start,
                        method='L-BFGS-B', bounds=self.bounds,
                        options={'ftol': 1e-9, 'gtol': 1e-6, 'disp': False}
                    )
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
            'refinement_stopping': refinement_stopped_by,
            'random_state': random_state,
        }

        print(f"  Total: {self._n_evals} evals, {elapsed:.1f}s, "
              f"refined {pts_refined} pts, stopped by: {stopped_by}")

        return self.results['Bayesian']

    # ========================================
    # Comparison driver
    # ========================================

    def run_multiple(self, method, n_runs=1, base_seed=1, **kwargs):
        """Run the given method n_runs times with different random states.

        Each run uses random_state = base_seed + i.  All individual results
        are stored in self.all_runs[key].  self.results[key] is set to the
        best run by best_cost.

        Parameters
        ----------
        method : str
            'bayesian' or 'multistart_lbfgs'
        n_runs : int
        base_seed : int
        **kwargs : passed through to the underlying run method
        """
        key_map = {'bayesian': 'Bayesian', 'multistart_lbfgs': 'Multi-start L-BFGS'}
        if method not in key_map:
            raise ValueError(f"method must be one of {list(key_map)}")
        key = key_map[method]
        run_fn = self.run_bayesian if method == 'bayesian' else self.run_multistart_lbfgs

        runs = []
        for i in range(n_runs):
            seed = base_seed + i
            print(f"\n  [{method}] run {i+1}/{n_runs}, random_state={seed}")
            result = run_fn(random_state=seed, **kwargs)
            runs.append(dict(result))

        self.all_runs[key] = runs
        best = min(runs, key=lambda r: r['best_cost'])
        self.results[key] = best
        print(f"\n  {key}: best cost {best['best_cost']:.4e} "
              f"(seed={best['random_state']}) over {n_runs} runs")
        return runs

    def compare_all(self, methods=None, n_runs=1, base_seed=42):
        if methods is None:
            methods = ['multistart_lbfgs', 'bayesian']

        print(f"Comparing methods: max_evals={self.max_evals}, "
              f"conv_threshold={self.convergence_threshold}, n_runs={n_runs}")
        print("=" * 60)

        if 'multistart_lbfgs' in methods:
            print("Running Multi-start L-BFGS...")
            if n_runs > 1:
                self.run_multiple('multistart_lbfgs', n_runs=n_runs, base_seed=base_seed)
            else:
                self.run_multistart_lbfgs()

        if 'bayesian' in methods:
            print("Running Bayesian Optimization...")
            if n_runs > 1:
                self.run_multiple('bayesian', n_runs=n_runs, base_seed=base_seed,
                                  max_perms=1, unique_refined_points=5)
            else:
                self.run_bayesian(
                    max_perms=1,
                    unique_refined_points=5)

        return self.summary()

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

    # ========================================
    # Plotting
    # ========================================

    def plot_result(self, log_scale=True, figsize=(16, 10)):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        sorted_methods, colors = self._get_sorted_methods_and_colors()

        # ax1: Convergence curves
        for (method, res), color in zip(sorted_methods, colors):
            conv = res['convergence_history']
            label = f"{method} ({res['best_cost']:.2e})"
            ax1.plot(range(1, len(conv) + 1), conv, label=label, color=color, linewidth=2)

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

        params_text = [
            f'NUM_COILS = {self.num_coils}',
            f'MAX_EVALS = {self.max_evals}',
            f'OMEGA = {self.omega:.0e}',
            f'DIST_TH = {self.dist_th}',
            f'REG_IN = {self.reg_in:.0e}',
            f'RFIL = {self.rfil}',
        ]
        textstr = '\n'.join(params_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.8, 0.5, textstr, transform=ax1.transAxes, fontsize=8,
                 verticalalignment='top', bbox=props)

        # ax2: Bar chart of final costs
        methods_names = [m for m, _ in sorted_methods]
        costs = np.array([r['best_cost'] for _, r in sorted_methods])

        ax2.barh(methods_names, costs, color=colors)
        ax2.set_xlabel('Final Cost', fontsize=12)
        ax2.set_title('Final Cost by Method', fontsize=14)
        ax2.axvline(x=costs[0], color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        ax2.grid(True, alpha=0.3, axis='x')

        # ax3: Coil placement
        if self.coil_center_cand1 is not None and self.coil_center_cand2 is not None:
            self._plot_position_space_boundaries(ax3)

            for (method, res), color in zip(sorted_methods, colors):
                coil_locs, _, _ = self._compute_flux_for_params(res['best_params'])
                for loc in coil_locs:
                    rect_top = plt.Rectangle((loc[0]-0.035, loc[1]-0.035), 0.07, 0.07,
                                            facecolor=color, edgecolor=color, alpha=0.6)
                    ax3.add_patch(rect_top)
                    rect_bot = plt.Rectangle((loc[0]-0.035, -loc[1]-0.035), 0.07, 0.07,
                                            facecolor=color, edgecolor=color, alpha=0.6)
                    ax3.add_patch(rect_bot)

            ax3.set_xlabel('R [m]', fontsize=12)
            ax3.set_ylabel('Z [m]', fontsize=12)
            ax3.set_title('Coil Placement (All Methods)', fontsize=14)
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)

        # ax4: Flux comparison
        if self.r_bnd is not None and self.psi_bnd is not None:
            theta = np.arctan2(self.r_bnd[:,1], self.r_bnd[:,0] - self.o_point[0])
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
            conv = res['convergence_history']
            times = res['times']
            label = f"{method} ({res['best_cost']:.2e})"
            ax.plot(times, conv, label=label, color=color, linewidth=2, alpha=0.8)

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

    # ========================================
    # JSON save
    # ========================================

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
                'rfil': float(self.rfil)
            },
            'methods': {}
        }

        if self.brute_force_cost is not None:
            save_data['brute_force_cost'] = float(self.brute_force_cost)

        for method, res in self.results.items():
            method_data = {
                'best_cost': float(res['best_cost']),
                'best_flux_err': float(res['best_flux_err']) if res['best_flux_err'] is not None else None,
                'n_evals': int(res['n_evals']),
                'time': float(res['time']),
                'times': [float(t) for t in res['times']],
                'stopping': res['stopping'],
                'parameters': res['parameters'],
                'coil_positions_top': res['coil_positions_top'],
                'coil_currents': res['coil_currents'],
                'convergence_history': res['convergence_history'],
                'cost_history': [float(c) for c in res['cost_history']],
            }

            if 'starts_completed' in res:
                method_data['starts_completed'] = int(res['starts_completed'])
            if 'start_boundaries' in res:
                method_data['start_boundaries'] = [int(x) for x in res['start_boundaries']]
            if 'start_costs' in res:
                method_data['start_costs'] = [float(x) for x in res['start_costs']]
            if 'n_initial' in res:
                method_data['n_initial'] = int(res['n_initial'])
            if 'n_perms' in res:
                method_data['n_perms'] = int(res['n_perms'])
            if 'n_bayesian_evals' in res:
                method_data['n_bayesian_evals'] = int(res['n_bayesian_evals'])
            if 'n_gp_observations' in res:
                method_data['n_gp_observations'] = int(res['n_gp_observations'])
            if 'time_bayesian_phase' in res:
                method_data['time_bayesian_phase'] = float(res['time_bayesian_phase'])
            if 'pts_refined' in res:
                method_data['pts_refined'] = int(res['pts_refined'])
            if 'refinement_evals' in res:
                method_data['refinement_evals'] = [int(x) for x in res['refinement_evals']]
            if 'refinement_times' in res:
                method_data['refinement_times'] = [float(x) for x in res['refinement_times']]
            if 'refinement_costs' in res:
                method_data['refinement_costs'] = [float(x) for x in res['refinement_costs']]
            if 'bayesian_convergence_history' in res:
                method_data['bayesian_convergence_history'] = res['bayesian_convergence_history']
            if 'refinement_convergence_history' in res:
                method_data['refinement_convergence_history'] = res['refinement_convergence_history']
            if 'bayesian_stopping' in res:
                method_data['bayesian_stopping'] = res['bayesian_stopping']
            if 'acq_multiplier' in res and res['acq_multiplier'] is not None:
                method_data['acq_multiplier'] = int(res['acq_multiplier'])
            if 'n_acq_candidates' in res and res['n_acq_candidates'] is not None:
                method_data['n_acq_candidates'] = int(res['n_acq_candidates'])
            if 'acq_dedup_tol' in res and res['acq_dedup_tol'] is not None:
                method_data['acq_dedup_tol'] = float(res['acq_dedup_tol'])
            if 'n_acq_unique' in res and res['n_acq_unique'] is not None:
                method_data['n_acq_unique'] = int(res['n_acq_unique'])
            if 'unique_refined_points' in res and res['unique_refined_points'] is not None:
                method_data['unique_refined_points'] = int(res['unique_refined_points'])
            if 'refinement_stopping' in res and res['refinement_stopping'] is not None:
                method_data['refinement_stopping'] = res['refinement_stopping']
            if 'random_state' in res and res['random_state'] is not None:
                method_data['random_state'] = int(res['random_state'])

            save_data['methods'][method] = method_data

        if self.all_runs:
            save_data['all_runs'] = {}
            for method_key, runs in self.all_runs.items():
                save_data['all_runs'][method_key] = [
                    {k: (v.tolist() if hasattr(v, 'tolist') else v)
                     for k, v in run.items()
                     if k != 'best_params'}
                    for run in runs
                ]

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Saved results to {filename}")


# ============================================
# Main function
# ============================================

def main(mygs, methods=None, **kwargs):
    NUM_COILS = kwargs.get('NUM_COILS', 4)
    MAX_EVALS = kwargs.get('MAX_EVALS', 2**18)
    MAX_TIME = kwargs.get('MAX_TIME', 86400)
    CONVERGENCE_THRESHOLD = kwargs.get('CONVERGENCE_THRESHOLD', 0.001)
    OMEGA = kwargs.get('OMEGA', 1e-3)
    DIST_TH = kwargs.get('DIST_TH', 5.0)
    REG_IN = kwargs.get('REG_IN', 1e-7)
    RFIL = kwargs.get('RFIL', 0.01)
    N_RUNS = kwargs.get('N_RUNS', 1)

    r_bnd, psi_bnd = mygs.get_vfixed()
    print(f"  Found {len(r_bnd)} boundary points")

    # Coil position space (DIIID)
    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    # Bounds
    coil_space = CoilPositionSpace(
        inner_boundary=coil_center_cand1,
        outer_boundary=coil_center_cand2,
        method='coords'
    )
    coil_space.set_bounds(angular_bounds=(10, 170), radial_bounds=(0, 1))

    bounds = []
    theta_bounds, radial_bounds = coil_space.get_bounds()
    for _ in range(NUM_COILS):
        bounds.append(theta_bounds)
    for _ in range(NUM_COILS):
        bounds.append(radial_bounds)

    # Objective function
    theta_range = np.linspace(0, 180, len(coil_center_cand1) // 2)
    inner = coil_center_cand1[:len(coil_center_cand1) // 2]
    outer = coil_center_cand2[:len(coil_center_cand2) // 2]

    def objective(params):
        thetas = params[:NUM_COILS]
        radials = params[NUM_COILS:]

        locs = []
        for theta, rho in zip(thetas, radials):
            R_inner = np.interp(theta, theta_range, inner[:, 0])
            Z_inner = np.interp(theta, theta_range, inner[:, 1])
            R_outer = np.interp(theta, theta_range, outer[:, 0])
            Z_outer = np.interp(theta, theta_range, outer[:, 1])
            R_pos = (1 - rho) * R_inner + rho * R_outer
            Z_pos = (1 - rho) * Z_inner + rho * Z_outer
            locs.append([R_pos, Z_pos])

        coil_centers_3x3 = []
        for loc in locs:
            centers_top = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    centers_top.append([loc[0] + 2*RFIL*dx, loc[1] + 2*RFIL*dy])
            coil_centers_3x3.append(centers_top)
            centers_bot = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    centers_bot.append([loc[0] + 2*RFIL*dx, -loc[1] + 2*RFIL*dy])
            coil_centers_3x3.append(centers_bot)

        n_bnd = psi_bnd.shape[0]
        n_coils_total = len(coil_centers_3x3)
        con = np.zeros((n_bnd - 1 + n_coils_total, n_coils_total))

        for i, filament_set in enumerate(coil_centers_3x3):
            flux_tmp = np.zeros((n_bnd,))
            for fil in filament_set:
                flux_tmp += eval_green(r_bnd, fil)
            con[:n_bnd-1, i] = flux_tmp[1:] - flux_tmp[0]
            con[n_bnd-1+i, i] = REG_IN

        err = np.zeros((n_bnd - 1 + n_coils_total,))
        err[:n_bnd-1] = psi_bnd[1:] - psi_bnd[0]
        currs, residuals, _, _ = np.linalg.lstsq(con, err, rcond=None)

        if len(residuals) > 0:
            flux_error_squared = residuals[0]
        else:
            flux_error_squared = np.linalg.norm(np.dot(con, currs) - err) ** 2

        objective.last_flux_err = flux_error_squared

        dist_angles = np.diff(np.sort(thetas))
        pen_terms = np.maximum(DIST_TH - dist_angles, 0.0) ** 2
        dist_penalty = OMEGA * np.sum(pen_terms)

        return flux_error_squared + dist_penalty

    # Run comparison
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPARISON")
    print("=" * 60)
    print(f"  Coils: {NUM_COILS}  |  Max evals: {MAX_EVALS}  |  "
          f"Threshold: {CONVERGENCE_THRESHOLD}")
    print(f"  omega={OMEGA}  |  reg_in={REG_IN}  |  dist_th={DIST_TH}")
    print("=" * 60 + "\n")

    comparison = OptimizationComparison(
        objective, bounds,
        max_time=MAX_TIME,
        max_evals=MAX_EVALS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        NUM_COILS=NUM_COILS,
        OMEGA=OMEGA,
        DIST_TH=DIST_TH,
        REG_IN=REG_IN,
        RFIL=RFIL
    )

    comparison.set_problem_data(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                                mygs.o_point, eval_green)

    # Load brute force baseline
    bf_path = f'examples/comparisons/closed_boundary_DIIID/brute_force/lambda:{REG_IN},coils:{NUM_COILS}/results.json'
    if os.path.exists(bf_path):
        with open(bf_path, 'r') as f:
            bf_data = json.load(f)
        comparison.brute_force_cost = bf_data['best_cost']
        print(f"Brute force baseline: {comparison.brute_force_cost:.6e}")
    else:
        print(f"No brute force baseline found at {bf_path}")

    base = f'examples/comparisons/closed_boundary_DIIID/convergence/lambda:{REG_IN},coils:{NUM_COILS}'

    existing_runs = 0
    while os.path.exists(os.path.join(base, f'run_{existing_runs:02d}', 'results.json')):
        existing_runs += 1

    summary = comparison.compare_all(methods=methods, n_runs=N_RUNS, base_seed=42 + existing_runs)

    if comparison.all_runs:
        # Save each individual seed run to its own run_XX folder
        n_individual = max(len(runs) for runs in comparison.all_runs.values())
        orig_results = comparison.results
        orig_all_runs = comparison.all_runs

        for i in range(n_individual):
            run_idx = 0
            while os.path.exists(os.path.join(base, f'run_{run_idx:02d}', 'results.json')):
                run_idx += 1
            foldername = os.path.join(base, f'run_{run_idx:02d}')
            os.makedirs(foldername, exist_ok=True)

            comparison.results = {
                key: runs[i]
                for key, runs in orig_all_runs.items()
                if i < len(runs)
            }
            comparison.all_runs = {}

            print(f"\nGenerating plots for run {i}...")
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
        # Single run
        run_idx = 0
        while os.path.exists(os.path.join(base, f'run_{run_idx:02d}', 'results.json')):
            run_idx += 1
        foldername = os.path.join(base, f'run_{run_idx:02d}')
        os.makedirs(foldername, exist_ok=True)

        print("\nGenerating plots...")
        fig = comparison.plot_result()
        time_fig = comparison.plot_convergence_vs_time(log_scale=True)

        fig.savefig(f'{foldername}/convergence_plot.png', dpi=150, bbox_inches='tight')
        time_fig.savefig(f'{foldername}/convergence_vs_time_plot.png', dpi=150, bbox_inches='tight')
        comparison.save_results_to_json(f'{foldername}/results.json')
        plt.close('all')

        print(f"Saved all plots and results to: {foldername}/")

    return comparison, summary

if __name__ == "__main__":
    eqdsk = read_eqdsk('examples/data/eqdsk/g192185.02440')
    LCFS_contour = eqdsk['rzout'].copy()
    mesh_dx = 0.015

    gs_mesh = gs_Domain()
    gs_mesh.define_region('plasma', mesh_dx, 'plasma')
    gs_mesh.add_polygon(LCFS_contour, 'plasma')
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)

    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False

    F0 = eqdsk['rcentr'] * eqdsk['bcentr']
    mygs.setup(order=2, F0=F0)

    Ip_target = eqdsk['ip']
    pres_target = eqdsk['pres'][0]
    mygs.set_targets(Ip=Ip_target, pax=pres_target)

    print("Solving fixed-boundary equilibrium...")
    mygs.init_psi()
    mygs.solve()

    methods = ["multistart_lbfgs", "bayesian"]

    # 2,3,4,5,6
    # 1e-5, 1e-6, 1e-7, 1e-8
    
    for num_coils in [2]:
        for reg_in in [1e-8]:
            print(f"\n{'='*60}")
            print(f"NUM_COILS={num_coils}, REG_IN={reg_in}")
            print(f"{'='*60}")

            try:
                comparison, summary = main(
                    mygs=mygs,
                    methods=methods,
                    NUM_COILS=num_coils,
                    REG_IN=reg_in,
                    MAX_EVALS=2**18,
                    N_RUNS=5
                )
            except Exception as e:
                print(f"\nFailed for NUM_COILS={num_coils}, REG_IN={reg_in}")
                print(f"Error: {e}")
                continue
    