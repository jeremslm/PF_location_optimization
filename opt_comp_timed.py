"""
Optimization Method Comparison Framework
=========================================

Compare different optimization methods to find which reaches the global minimum fastest.

Key metrics:
- Best cost at fixed evaluation budget
- Number of evaluations to reach target cost
- Convergence rate
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping
from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import qmc
import os
import sys

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
from helper_fct import resize_polygon, update_boundary
from OFT_pf_coil_opt_fct import CoilPositionSpace


class TimeoutException(Exception):
    """Raised when optimization time limit is reached."""
    pass


class OptimizationComparison:
    """
    Compare optimization methods on speed to global minimum.

    All methods are given the same time budget for fair comparison.
    """

    def __init__(self, objective_func, bounds, max_time=120.0,
                 NUM_COILS=3, OMEGA=1e-5, DIST_TH=5, REG_IN=1e-5, RFIL=0.01):
        """
        Initialize comparison.

        Parameters
        ----------
        objective_func : callable
            Function to minimize: f(params) -> cost
        bounds : list of tuples
            [(min, max), ...] for each parameter
        max_time : float
            Maximum time in seconds per method (default: 60s = 1 minute)
        NUM_COILS : int, optional
            Number of coils (for display in plots)
        OMEGA : float, optional
            Distance penalty weight (for display in plots)
        DIST_TH : float, optional
            Minimum distance threshold (for display in plots)
        REG_IN : float, optional
            Current regularization (for display in plots)
        RFIL : float, optional
            Coil filament radius (for display in plots)
        """
        self.objective = objective_func
        self.bounds = bounds
        self.max_time = max_time
        self.n_params = len(bounds)

        # Store optimization parameters for display
        self.num_coils = NUM_COILS
        self.omega = OMEGA
        self.dist_th = DIST_TH
        self.reg_in = REG_IN
        self.rfil = RFIL

        # Results storage
        self.results = {}

        # Problem data for plotting (set later via set_problem_data)
        self.r_bnd = None
        self.psi_bnd = None
        self.coil_center_cand1 = None
        self.coil_center_cand2 = None
        self.o_point = None
        self.eval_green = None

    def set_problem_data(self, r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2, o_point, eval_green):
        """
        Store problem-specific data needed for plotting.

        Parameters
        ----------
        r_bnd : ndarray
            Boundary points (R, Z coordinates)
        psi_bnd : ndarray
            Required flux at boundary points
        coil_center_cand1 : ndarray
            Inner coil position space boundary
        coil_center_cand2 : ndarray
            Outer coil position space boundary
        o_point : ndarray
            Magnetic axis position [R, Z]
        eval_green : callable
            Function to evaluate Green's function
        """
        self.r_bnd = r_bnd
        self.psi_bnd = psi_bnd
        self.coil_center_cand1 = coil_center_cand1
        self.coil_center_cand2 = coil_center_cand2
        self.o_point = o_point
        self.eval_green = eval_green

    def _compute_flux_for_params(self, params):
        """
        Compute flux and coil positions for given parameters.

        Parameters
        ----------
        params : array-like
            Coil parameters [angles, radials]

        Returns
        -------
        coil_locs : ndarray
            Coil locations (R, Z)
        psi_computed : ndarray
            Computed flux at boundary
        currs : ndarray
            Coil currents
        """
        from helper_fct import place_points_pol_rad, make_3x3_thick

        num_coils = len(params) // 2
        thetas = params[:num_coils]
        radials = params[num_coils:]

        # Get inner and outer arcs (top half only)
        inner_arc = self.coil_center_cand1[:len(self.coil_center_cand1)//2]
        outer_arc = self.coil_center_cand2[:len(self.coil_center_cand2)//2]

        # Place coils
        _, coil_locs = place_points_pol_rad(num_coils, inner_arc, outer_arc, thetas, radials)

        # Create 3x3 thick coils (top + bottom)
        coil_centers_3x3 = []
        for loc in coil_locs:
            # Top coil
            thick_centers = make_3x3_thick(loc, self.rfil)
            coil_centers_3x3.append(thick_centers)
            # Bottom coil (mirror)
            thick_centers_bot = make_3x3_thick([loc[0], -loc[1]], self.rfil)
            coil_centers_3x3.append(thick_centers_bot)

        # Build least squares system
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

        # Compute flux
        psi_computed = np.dot(con, currs)[:n_bnd - 1]

        return coil_locs, psi_computed, currs

    # ========================================
    # Plotting helper methods
    # ========================================

    def _get_sorted_methods_and_colors(self):
        """Get methods sorted by cost and assign colors."""
        sorted_methods = sorted(
            self.results.items(),
            key=lambda x: x[1]['best_cost']
        )
        if(len(sorted_methods)!=2):
            colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_methods)))

        colors = plt.cm.tab10(np.array([0.2,0.6]))
        return sorted_methods, colors

    def _plot_position_space_boundaries(self, ax):
        """Plot position space boundaries on given axis."""
        ax.plot(self.coil_center_cand1[:, 0], self.coil_center_cand1[:, 1],
               'k--', alpha=0.3, linewidth=1, label='Position space')
        ax.plot(self.coil_center_cand2[:, 0], self.coil_center_cand2[:, 1],
               'k--', alpha=0.3, linewidth=1)

    def _plot_coils_on_axis(self, ax, coil_locs, color, dx=0.035, dy=0.035):
        """Plot coils (top and bottom) on given axis."""
        for loc in coil_locs:
            # Top coil
            rect_top = plt.Rectangle((loc[0]-dx, loc[1]-dy), 2*dx, 2*dy,
                                    facecolor=color, edgecolor='black',
                                    alpha=0.7, linewidth=0.5)
            ax.add_patch(rect_top)
            # Bottom coil
            rect_bot = plt.Rectangle((loc[0]-dx, -loc[1]-dy), 2*dx, 2*dy,
                                    facecolor=color, edgecolor='black',
                                    alpha=0.7, linewidth=0.5)
            ax.add_patch(rect_bot)

    def _track_objective(self, params):
        """Wrapper that tracks evaluations and checks time limit."""
        # Check time limit
        if time.time() - self._start_time > self.max_time:
            raise TimeoutException("Time limit reached")

        self._n_evals += 1
        cost = self.objective(params)
        self._history.append(cost)
        self._x_history.append(params.copy())  # Track parameter vector
        self._times.append(time.time() - self._start_time)

        # Read flux_err from objective function attribute if it exists
        flux_err = getattr(self.objective, 'last_flux_err', None)

        if cost < self._best_cost:
            self._best_cost = cost
            self._best_params = params.copy()
            if flux_err is not None:
                self._best_flux_err = flux_err

        return cost

    def _reset_tracking(self):
        """Reset tracking variables."""
        self._n_evals = 0
        self._history = []
        self._x_history = []  # Track all parameter vectors
        self._times = []  # Track time of each evaluation
        self._best_cost = float('inf')
        self._best_flux_err = None  # Track best flux error (without distance penalty)
        self._best_params = None
        self._start_time = time.time()

    def run_lbfgs(self, x0, ftol=1e-9, gtol=1e-6):
        """
        L-BFGS-B optimization.

        Stopping: gradient < gtol OR function change < ftol OR time limit
        """
        self._reset_tracking()

        try:
            minimize(
                self._track_objective, 
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={
                    'ftol': ftol,
                    'gtol': gtol,
                    'maxfun': 100000,  # High limit, time controls stopping
                    'maxiter': 100000,
                    'disp': False
                }
            )
            stopped_by = 'converged'
        except TimeoutException:
            stopped_by = 'time limit'

        elapsed = time.time() - self._start_time

        self.results['L-BFGS-B'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{stopped_by} ({elapsed:.1f}s)'
        }

        return self.results['L-BFGS-B']

    def run_multistart_lbfgs(self, n_starts=1000, ftol=1e-9, gtol=1e-6):
        """
        Multi-start L-BFGS-B with Sobol sequence sampling.

        Stopping: time limit reached
        """

        self._reset_tracking()

        # Generate starting points with Sobol
        sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=42)
        samples = sampler.random(n_starts)

        # Scale to bounds
        starts = []
        for i in range(n_starts):
            point = []
            for j, (low, high) in enumerate(self.bounds):
                point.append(low + samples[i, j] * (high - low))
            starts.append(point)

        starts_completed = 0

        for x0 in starts:
            try:
                minimize(
                    self._track_objective, x0,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={
                        'ftol': ftol,
                        'gtol': gtol,
                        # 'maxfun': 1000,
                        # 'maxiter': 1000,
                        'disp': False
                    }
                )
                starts_completed += 1
            except TimeoutException:
                break

        elapsed = time.time() - self._start_time

        self.results['Multi-start L-BFGS'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{starts_completed}/{n_starts} starts ({elapsed:.1f}s)'
        }

        return self.results['Multi-start L-BFGS']

    def run_differential_evolution(self, popsize=1000, tol=1e-7):
        """
        Differential Evolution (global optimizer).

        Stopping: population converged (tol) OR time limit
        """
        self._reset_tracking()

        try:
            differential_evolution(
                self._track_objective,
                self.bounds,
                maxiter=10000,  # High limit, time controls stopping
                tol=tol,
                popsize=popsize,
                seed=42,
                disp=False,
                polish=True
            )
            stopped_by = 'converged'
        except TimeoutException:
            stopped_by = 'time limit'

        elapsed = time.time() - self._start_time

        self.results['Differential Evolution'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{stopped_by} ({elapsed:.1f}s)'
        }

        return self.results['Differential Evolution']

    def run_dual_annealing(self, initial_temp=5230.0, restart_temp_ratio=2e-5, visit=2.62):
        """
        Dual Annealing (simulated annealing + local search).

        Stopping: temperature cooled OR time limit
        """
        self._reset_tracking()

        try:
            dual_annealing(
                self._track_objective,
                self.bounds,
                maxfun=1000000,  # High limit, time controls stopping
                initial_temp=initial_temp,
                restart_temp_ratio=restart_temp_ratio,
                seed=42,
                no_local_search=False,
                visit=visit
            )
            stopped_by = 'converged'
        except TimeoutException:
            stopped_by = 'time limit'

        elapsed = time.time() - self._start_time

        self.results['Dual Annealing'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{stopped_by} ({elapsed:.1f}s)'
        }

        return self.results['Dual Annealing']

    def run_bayesian(self, n_initial=50, acq_func='EI', local_optimize=True, n_local_refine=100):
        """
        Bayesian Optimization with Gaussian Process.

        Parameters
        ----------
        n_initial : int
            Number of random initial points before GP modeling
        acq_func : str
            Acquisition function ('EI', 'LCB', 'PI')
        local_optimize : bool
            If True, refine top results with L-BFGS after Bayesian optimization
        n_local_refine : int
            Number of top Bayesian results to refine with L-BFGS

        Stopping: time limit reached
        """
        self._reset_tracking()

        # Adjust time limit for two-phase optimization
        if local_optimize:
            bayesian_time_limit = self.max_time * 0.5  # x% for Bayesian exploration
            original_max_time = self.max_time
            self.max_time = bayesian_time_limit

        space = [Real(low, high) for low, high in self.bounds]

        # Phase 1: Bayesian optimization
        try:
            gp_minimize(
                self._track_objective,
                space,
                n_calls=10000,  # High limit, time controls stopping
                n_initial_points=n_initial,
                acq_func=acq_func,
                random_state=42,
                verbose=False
            )
            stopped_by = 'completed'
        except TimeoutException:
            stopped_by = 'time limit'

        # Save Bayesian-only result
        elapsed_bayesian = time.time() - self._start_time

        self.results['Bayesian'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed_bayesian,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{stopped_by} ({elapsed_bayesian:.1f}s)'
        }
        bayesian_evals = self._n_evals

        # Phase 2: L-BFGS refinement (if enabled)
        if local_optimize:
            # Restore full time budget
            self.max_time = original_max_time

            # Get top N indices from Bayesian results
            top_indices = np.argsort(self._history)[:n_local_refine]

            # Refine each with L-BFGS
            refinements_completed = 0
            for idx in top_indices:
                try:
                    start = np.array(self._x_history[idx])
                    minimize(
                        self._track_objective,
                        start,
                        method='L-BFGS-B',
                        bounds=self.bounds,
                        options={'ftol': 1e-9, 'gtol': 1e-6, 'disp': False}
                    )
                    refinements_completed += 1
                except TimeoutException:
                    break  # Stop refining if time runs out
                except Exception:
                    continue  # Skip failed refinement

            # Save refined result
            elapsed_total = time.time() - self._start_time
            refinement_evals = self._n_evals  # Evals used during refinement phase

            # Calculate how many points were attempted vs completed
            n_attempted = len(top_indices)

            self.results['Bayesian'] = {
                'best_cost': self._best_cost,
                'best_flux_err': self._best_flux_err,
                'best_params': self._best_params,
                'n_evals': bayesian_evals + refinement_evals,
                'time': elapsed_total,
                'times': self._times.copy(),
                'history': self._history.copy(),
                'convergence': np.minimum.accumulate(self._history),
                'stopping': f'{stopped_by} ({elapsed_bayesian:.1f}s)',
                'n_bayesian_evals': bayesian_evals,
                'pts_refined': refinements_completed
            }
 
        elif local_optimize:
            # Restore max time even if we didn't refine
            self.max_time = original_max_time

        return self.results['Bayesian']

    def run_basin_hopping(self, x0, T=1.0, stepsize=0.5, niter=10000):
        """
        Basin Hopping (global + local).

        Stopping: time limit reached
        """
        self._reset_tracking()

        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds,
            # 'options': {'maxfun': 50}
        }

        try:
            
            basinhopping(
                self._track_objective, x0,
                niter=niter,  # High limit, time controls stopping
                T=T,
                stepsize=stepsize,
                minimizer_kwargs=minimizer_kwargs,
                seed=42
            )
            stopped_by = 'completed'
        except TimeoutException:
            stopped_by = 'time limit'

        elapsed = time.time() - self._start_time

        self.results['Basin Hopping'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{stopped_by} ({elapsed:.1f}s)'
        }

        return self.results['Basin Hopping']

    def run_multi_basin_hopping(self, T=1.0, stepsize=0.5, niter=1000, n_starts=100000):
        """
        Multi-start Basin Hopping with Sobol sequence sampling.

        Stopping: time limit reached
        """

        self._reset_tracking()

        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds,
            # 'options': {'maxfun': 50}
        }

        # Generate starting points with Sobol sequence sampling
        sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=42)
        samples = sampler.random(n_starts)

        # Scale to bounds
        starts = []
        for i in range(n_starts):
            point = []
            for j, (low, high) in enumerate(self.bounds):
                point.append(low + samples[i, j] * (high - low))
            starts.append(point)

        starts_completed = 0

        for x0 in starts:
            try:
                basinhopping(
                    self._track_objective, x0,
                    niter=niter,
                    T=T,
                    stepsize=stepsize,
                    minimizer_kwargs=minimizer_kwargs,
                    seed=42
                )
                starts_completed += 1
            except TimeoutException:
                break

        elapsed = time.time() - self._start_time

        self.results['Multi-start Basin Hopping'] = {
            'best_cost': self._best_cost,
            'best_flux_err': self._best_flux_err,
            'best_params': self._best_params,
            'n_evals': self._n_evals,
            'time': elapsed,
            'times': self._times.copy(),
            'history': self._history.copy(),
            'convergence': np.minimum.accumulate(self._history),
            'stopping': f'{starts_completed}/{n_starts} starts ({elapsed:.1f}s)'
        }

        return self.results['Multi-start Basin Hopping']

        

    def compare_all(self, x0=None, methods=None):
        """
        Run all methods and compare.

        Parameters
        ----------
        x0 : array-like, optional
            Starting point for methods that need it. Random if None.
        methods : list, optional
            Which methods to run. Default: all.

        Returns
        -------
        summary : DataFrame
            Comparison results sorted by best cost
        """
        if x0 is None:
            x0 = []
            np.random.seed(42)
            for low, high in self.bounds:
                x0.append(np.random.uniform(low, high))
            x0 = np.array(x0)

        if methods is None:
            methods = ['lbfgs', 'multistart_lbfgs', 'differential_evolution', 'dual_annealing', 'bayesian', 'basin_hopping', 'multi_basin_hopping']

        print(f"Comparing methods with max {self.max_time:.0f}s time limit each")
        print("=" * 60)

        if 'lbfgs' in methods:
            print("Running L-BFGS-B...")
            self.run_lbfgs(x0)

        if 'multistart_lbfgs' in methods:
            print("Running Multi-start L-BFGS...")
            self.run_multistart_lbfgs()

        if 'differential_evolution' in methods:
            print("Running Differential Evolution...")
            self.run_differential_evolution()

        if 'dual_annealing' in methods:
            print("Running Dual Annealing...")
            self.run_dual_annealing()

        if 'bayesian' in methods:
            print("Running Bayesian Optimization...")
            self.run_bayesian()

        if 'basin_hopping' in methods:
            print("Running Basin Hopping...")
            self.run_basin_hopping(x0)

        if 'multi_basin_hopping' in methods:
            print("Running Multi-start Basin Hopping...")
            self.run_multi_basin_hopping()

        return self.summary()

    def summary(self):
        """Generate comparison summary."""
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPARISON RESULTS")
        print("=" * 70)

        data = []

        for method, res in self.results.items():
            row = {
                'Method': method,
                'Best Cost': res['best_cost'],
                'Evals Used': res['n_evals'],
                'Time (s)': res['time'],
                'Stopping': res['stopping'],
            }

            # Add Bayesian-specific fields if they exist
            if 'n_bayesian_evals' in res:
                row['Bayesian Evals'] = res['n_bayesian_evals']
            if 'pts_refined' in res:
                row['Pts Refined'] = res['pts_refined']

            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('Best Cost')

        print(df.to_string(index=False))

        # Announce winner
        best = df.iloc[0]
        print(f"\n🏆 Winner: {best['Method']} with cost {best['Best Cost']:.6e}")

        return df

    def plot_result(self, log_scale=True, figsize=(16, 10), each_method_flag=True):
        """
        Plot 4 subplots: convergence curves, final costs, coil placement, and flux comparison.
        """
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Get sorted methods and colors
        sorted_methods, colors = self._get_sorted_methods_and_colors()

        # ===== ax1: Convergence curves =====
        for (method, res), color in zip(sorted_methods, colors):
            conv = res['convergence']
            label = f"{method} ({res['best_cost']:.2e})"
            ax1.plot(range(1, len(conv) + 1), conv, label=label, color=color, linewidth=2)

        ax1.set_xlabel('Function Evaluations', fontsize=12)
        ax1.set_ylabel('Best Cost Found', fontsize=12)
        ax1.set_title('Convergence Speed', fontsize=14)
        if log_scale:
            ax1.set_yscale('log')
            # Add more tick labels on y-axis
            from matplotlib.ticker import LogLocator, NullFormatter
            ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
            ax1.yaxis.set_minor_formatter(NullFormatter())
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Add parameter text box
        if any([self.num_coils, self.omega, self.dist_th, self.reg_in, self.rfil]):
            params_text = []
            if self.num_coils is not None:
                params_text.append(f'NUM_COILS = {self.num_coils}')
            if self.max_time is not None:
                params_text.append(f'MAX_TIME = {self.max_time:.0f}')
            if self.omega is not None:
                params_text.append(f'OMEGA = {self.omega:.0e}')
            if self.dist_th is not None:
                params_text.append(f'DIST_TH = {self.dist_th}')
            if self.reg_in is not None:
                params_text.append(f'REG_IN = {self.reg_in:.0e}')
            if self.rfil is not None:
                params_text.append(f'RFIL = {self.rfil}')

            textstr = '\n'.join(params_text)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.8, 0.5, textstr, transform=ax1.transAxes, fontsize=8,
                     verticalalignment='top', bbox=props)
            
        # ===== ax2: Bar chart of final costs =====
        methods_names = [m for m, _ in sorted_methods]
        costs = np.array([r['best_cost'] for _, r in sorted_methods])
        best_cost = costs[0]  # sorted_methods is already sorted by cost

        ax2.barh(methods_names, costs, color=colors)
        ax2.set_xlabel('Final Cost', fontsize=12)
        ax2.set_title('Final Cost by Method', fontsize=14)
        ax2.axvline(x=best_cost, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        ax2.grid(True, alpha=0.3, axis='x')

        max_cost = np.max(costs)
        ax2.set_xlim(best_cost * 0.95, max_cost * 1.15)

        for i, cost in enumerate(costs):
            pct_increase = ((cost - best_cost) / best_cost) * 100
            if pct_increase < 0.1:
                ax2.text(cost, i, f' Best', va='center', fontsize=9, ha='left', weight='bold')
            else:
                ax2.text(cost, i, f' +{pct_increase:.1f}%', va='center', fontsize=9, ha='left')
            
        # ===== ax3: Coil placement for all methods =====
        if self.coil_center_cand1 is not None and self.coil_center_cand2 is not None:
            self._plot_position_space_boundaries(ax3)

            # Plot coils for each method
            for (method, res), color in zip(sorted_methods, colors):
                coil_locs, _, _ = self._compute_flux_for_params(res['best_params'])
                # Use lighter alpha for combined plot
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

        # ===== ax4: Flux comparison for all methods =====
        if self.r_bnd is not None and self.psi_bnd is not None:
            # Compute theta and desired flux
            theta = np.arctan2(self.r_bnd[:,1], self.r_bnd[:,0] - self.o_point[0])
            psi_desired = self.psi_bnd[1:] - self.psi_bnd[0]

            # Plot desired flux
            ax4.plot(theta[1:], psi_desired, 'ko', markersize=3, label='Desired', alpha=0.5)

            # Plot computed flux for each method
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

        # Create separate figures if requested
        coil_fig = None
        error_fig = None
        if each_method_flag:
            coil_fig = self.plot_each_method_coils()
            error_fig = self.plot_each_method_error()

        return fig, coil_fig, error_fig

    def plot_early_convergence(self, n_evals=200, log_scale=True, figsize=(12, 6)):
        """
        Plot convergence for early evaluations only to see initial exploration.

        Parameters
        ----------
        n_evals : int
            Number of initial evaluations to plot (default: 200)
        log_scale : bool
            Use log scale for y-axis (default: False for linear scale)
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with early convergence plot
        """
        if not self.results:
            print("No results to plot")
            return None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get sorted methods and colors
        sorted_methods, colors = self._get_sorted_methods_and_colors()

        # Plot early convergence for each method
        for (method, res), color in zip(sorted_methods, colors):
            conv = res['convergence']
            n_plot = min(n_evals, len(conv))
            label = f"{method} ({res['best_cost']:.2e})"
            ax.plot(range(1, n_plot + 1), conv[:n_plot], label=label,
                   color=color, linewidth=2, marker='o', markersize=3, alpha=0.7)

        ax.set_xlabel('Function Evaluations', fontsize=12)
        ax.set_ylabel('Best Cost Found', fontsize=12)
        ax.set_title(f'Early Convergence (First {n_evals} Evaluations)', fontsize=14)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_convergence_vs_time(self, log_scale=True, figsize=(12, 6)):
        """
        Plot convergence vs time instead of evaluations.

        Parameters
        ----------
        log_scale : bool
            Use log scale for y-axis (default: True)
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with convergence vs time plot
        """
        if not self.results:
            print("No results to plot")
            return None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get sorted methods and colors
        sorted_methods, colors = self._get_sorted_methods_and_colors()

        # Plot convergence vs time for each method
        for (method, res), color in zip(sorted_methods, colors):
            conv = res['convergence']
            times = res['times']
            label = f"{method} ({res['best_cost']:.2e})"
            ax.plot(times, conv, label=label, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Best Cost Found', fontsize=12)
        ax.set_title('Convergence vs Time', fontsize=14)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_each_method_coils(self, figsize=(16,10)):
        """
        Create separate figure with subplots showing each method's coil placement.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with coil placement subplots for each method
        """
        if not self.results:
            print("No results to plot")
            return None

        if self.coil_center_cand1 is None or self.coil_center_cand2 is None:
            print("Position space data not set. Call set_problem_data() first.")
            return None

        # Get sorted methods and colors
        sorted_methods, colors = self._get_sorted_methods_and_colors()
        n_methods = len(sorted_methods)

        # Calculate subplot grid (roughly square)
        ncols = int(np.ceil(np.sqrt(n_methods)))
        nrows = int(np.ceil(n_methods / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, ((method, res), color) in enumerate(zip(sorted_methods, colors)):
            ax = axes[idx]

            # Plot position space boundaries
            self._plot_position_space_boundaries(ax)

            # Plot coils
            coil_locs, _, _ = self._compute_flux_for_params(res['best_params'])
            self._plot_coils_on_axis(ax, coil_locs, color)

            ax.set_xlabel('R [m]', fontsize=10)
            ax.set_ylabel('Z [m]', fontsize=10)
            ax.set_title(f'{method}\nCost: {res["best_cost"]:.2e}', fontsize=11)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Coil Placement by Method', fontsize=14, y=0.995)
        plt.tight_layout()

        return fig

    def plot_each_method_error(self, figsize=(16,10)):
        """
        Create separate figure with subplots showing each method's flux error.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with flux error subplots for each method
        """
        if not self.results:
            print("No results to plot")
            return None

        if self.r_bnd is None or self.psi_bnd is None:
            print("Boundary data not set. Call set_problem_data() first.")
            return None

        # Get sorted methods and colors
        sorted_methods, colors = self._get_sorted_methods_and_colors()
        n_methods = len(sorted_methods)

        # Calculate subplot grid (roughly square)
        ncols = int(np.ceil(np.sqrt(n_methods)))
        nrows = int(np.ceil(n_methods / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Compute theta and desired flux
        theta = np.arctan2(self.r_bnd[:,1], self.r_bnd[:,0] - self.o_point[0])
        psi_desired = self.psi_bnd[1:] - self.psi_bnd[0]

        for idx, ((method, res), color) in enumerate(zip(sorted_methods, colors)):
            ax = axes[idx]

            # Plot desired flux
            ax.plot(theta[1:], psi_desired, 'ko', markersize=3,
                   label='Desired', alpha=0.5)

            # Plot computed flux
            _, psi_computed, _ = self._compute_flux_for_params(res['best_params'])
            ax.plot(theta[1:], psi_computed, '+', color=color, markersize=5,
                   label='Computed', alpha=0.8)

            # Calculate error metrics
            error = psi_computed - psi_desired
            rmse = np.sqrt(np.mean(error**2))
            max_error = np.max(np.abs(error))

            ax.set_xlabel(r'$\theta$ [rad]', fontsize=10)
            ax.set_ylabel(r'$\psi_{boundary}$', fontsize=10)
            ax.set_title(f'{method}\nRMSE: {rmse:.2e}, Max: {max_error:.2e}',
                        fontsize=10)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Flux Error by Method', fontsize=14, y=0.995)
        plt.tight_layout()

        return fig

    def evals_to_target(self, target_cost):
        """
        Find how many evaluations each method needed to reach target cost.

        Returns
        -------
        dict : {method: n_evals or None if never reached}
        """
        result = {}
        for method, res in self.results.items():
            conv = res['convergence']
            reached = np.where(conv <= target_cost)[0]
            if len(reached) > 0:
                result[method] = reached[0] + 1  # +1 for 1-indexing
            else:
                result[method] = None

        print(f"\nEvaluations to reach cost ≤ {target_cost:.2e}:")
        for method, n in sorted(result.items(), key=lambda x: x[1] if x[1] else float('inf')):
            if n is not None:
                print(f"  {method}: {n} evals")
            else:
                print(f"  {method}: did not reach target")

        return result

    def save_results_to_json(self, filename):
        """
        Save optimization results to JSON file.

        Saves: method, parameters (thetas/radials), coil positions (R,Z),
        best cost, n_evals, time, and optimization settings.

        Parameters
        ----------
        filename : str
            Path to JSON file to save
        """
        import json

        if not self.results:
            print("No results to save")
            return

        # Prepare data for JSON serialization
        save_data = {
            'optimization_settings': {
                'num_coils': int(self.num_coils),
                'max_time': float(self.max_time),
                'omega': float(self.omega),
                'dist_th': float(self.dist_th),
                'reg_in': float(self.reg_in),
                'rfil': float(self.rfil)
            },
            'methods': {}
        }

        for method, res in self.results.items():
            # Extract parameters
            best_params = res['best_params']
            num_coils = len(best_params) // 2
            thetas = best_params[:num_coils].tolist()
            radials = best_params[num_coils:].tolist()

            # Compute coil positions (top half only)
            if self.coil_center_cand1 is not None:
                coil_locs, _, currents = self._compute_flux_for_params(best_params)
                coil_positions = [[float(loc[0]), float(loc[1])] for loc in coil_locs]
                coil_currents = currents.tolist()
            else:
                coil_positions = None
                coil_currents = None

            # Store method results
            method_data = {
                'best_cost': float(res['best_cost']),
                'flux_err': float(res['best_flux_err']) if res['best_flux_err'] is not None else None,
                'n_evals': int(res['n_evals']),
                'time': float(res['time']),
                'stopping': res['stopping'],
                'parameters': {
                    'thetas': thetas,  # Poloidal angles in degrees
                    'radials': radials  # Radial positions (0=inner, 1=outer)
                },
                'coil_positions_top': coil_positions,  # (R, Z) positions of top coils
                'coil_currents': coil_currents,  # Currents in each coil
                'convergence_history': res['convergence'].tolist()
            }

            # Add Bayesian-specific fields if they exist
            if 'n_bayesian_evals' in res:
                method_data['n_bayesian_evals'] = int(res['n_bayesian_evals'])
            if 'pts_refined' in res:
                method_data['pts_refined'] = int(res['pts_refined'])

            save_data['methods'][method] = method_data

        # Save to file
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Saved results to {filename}")


# ============================================
# Main function to run comparison
# ============================================

def main(mygs, methods=None, **kwargs):
    """
    Run optimization method comparison for PF coil placement.

    Sets up TokaMaker, computes fixed-boundary equilibrium, then compares
    optimization methods on finding the best coil positions.

    Parameters
    ----------
    methods : list of str, optional
        Which optimization methods to run. If None, runs all methods.
    """
    # ========================================
    # Optimization parameters
    # ========================================

    NUM_COILS = kwargs.get('NUM_COILS', 6)
    MAX_TIME = kwargs.get('MAX_TIME', 120) # seconds per method
    OMEGA = kwargs.get('OMEGA', 1e-5) # Distance penalty weight
    DIST_TH = kwargs.get('DIST_TH', 10) # Minimum distance threshold (degrees)
    REG_IN = kwargs.get('REG_IN', 1e-5) # Current regularization
    RFIL = kwargs.get('RFIL', 0.01) # Coil filament radius

    # Get boundary flux
    r_bnd, psi_bnd = mygs.get_vfixed()
    print(f"  Found {len(r_bnd)} boundary points")

    # ========================================
    # Define coil position space
    # ========================================
    print("Setting up coil position space...")

    # # Inner boundary - DIIID
    # lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    # coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    # # Outer boundary - DIIID
    # lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    # coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    lim1 
    lim2

    # ========================================
    # Create objective function
    # ========================================
    print("Creating objective function...")

    # Create CoilPositionSpace and extract bounds
    coil_space = CoilPositionSpace(
        inner_boundary=coil_center_cand1,
        outer_boundary=coil_center_cand2,
        method='coords'
    )
    coil_space.set_bounds(angular_bounds=(10,170), radial_bounds=(0, 1))

    # Generate bounds from coil space
    bounds = []
    theta_bounds, radial_bounds = coil_space.get_bounds()
    for i in range(NUM_COILS):
        bounds.append(theta_bounds)
    for i in range(NUM_COILS):
        bounds.append(radial_bounds)

    def objective(params):
        """PF coil objective function."""
        thetas = params[:NUM_COILS]
        radials = params[NUM_COILS:]

        # Place coils
        theta_range = np.linspace(0, 180, len(coil_center_cand1) // 2)
        inner = coil_center_cand1[:len(coil_center_cand1) // 2]
        outer = coil_center_cand2[:len(coil_center_cand2) // 2]

        locs = []
        for theta, rho in zip(thetas, radials):
            R_inner = np.interp(theta, theta_range, inner[:, 0])
            Z_inner = np.interp(theta, theta_range, inner[:, 1])
            R_outer = np.interp(theta, theta_range, outer[:, 0])
            Z_outer = np.interp(theta, theta_range, outer[:, 1])

            R_pos = (1 - rho) * R_inner + rho * R_outer
            Z_pos = (1 - rho) * Z_inner + rho * Z_outer
            locs.append([R_pos, Z_pos])

        # Create 3x3 thick coils (top + bottom)
        coil_centers_3x3 = []
        for loc in locs:
            # Top coil
            centers = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    centers.append([loc[0] + 2*RFIL*dx, loc[1] + 2*RFIL*dy])
            coil_centers_3x3.append(centers)

            # Bottom coil (mirror)
            centers = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    centers.append([loc[0] + 2*RFIL*dx, -loc[1] + 2*RFIL*dy])
            coil_centers_3x3.append(centers)

        # Build least squares system
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

        # Extract flux error (sum of squared residuals from least-squares fit)
        if len(residuals) > 0:
            flux_error_squared = residuals[0]
        else:
            # If lstsq doesn't return residuals (underdetermined system), compute manually
            flux_error_squared = np.linalg.norm(np.dot(con, currs) - err) ** 2

        # Store flux_err as function attribute for tracking (this is the first term, without distance penalty)
        objective.last_flux_err = flux_error_squared

        # Distance penalty (second term in cost function)
        dist_angles = np.diff(np.sort(thetas))
        pen_terms = np.maximum(DIST_TH - dist_angles, 0.0) ** 2
        dist_penalty = OMEGA * np.sum(pen_terms)

        total_cost = flux_error_squared + dist_penalty

        return total_cost

    # ========================================
    # Run comparison
    # ========================================
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION METHOD COMPARISON")
    print("=" * 60)
    print(f"Number of coils: {NUM_COILS}")
    print(f"Time limit per method: {MAX_TIME:.0f}s")
    print(f"Regularization: omega={OMEGA}, reg_in={REG_IN}")
    print(f"Min coil distance: {DIST_TH} degrees")
    print("=" * 60 + "\n")

    # Create comparison object
    comparison = OptimizationComparison(objective, bounds, MAX_TIME, NUM_COILS, OMEGA, DIST_TH, REG_IN, RFIL)

    # Set problem data for plotting
    comparison.set_problem_data(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2, mygs.o_point, eval_green)

    # Run all methods
    summary = comparison.compare_all(methods=methods)

    # Plot results
    print("\nGenerating convergence plot...")
    fig, coil_fig, err_fig = comparison.plot_result()
    early_fig = comparison.plot_early_convergence(n_evals=200, log_scale=True)
    time_fig = comparison.plot_convergence_vs_time(log_scale=True)

    # Save results
    foldername_end = "lambda:" + str(REG_IN) + ",coils:" + str(NUM_COILS)
    foldername = f'examples/comparisons/{foldername_end}'
    os.makedirs(foldername, exist_ok=True)

    # Save plots
    fig.savefig(f'{foldername}/convergence_plot.png', dpi=150, bbox_inches='tight')
    coil_fig.savefig(f'{foldername}/coil_placement_plot.png', dpi=150, bbox_inches='tight')
    err_fig.savefig(f'{foldername}/flux_error_plot.png', dpi=150, bbox_inches='tight')
    early_fig.savefig(f'{foldername}/early_convergence_plot.png', dpi=150, bbox_inches='tight')
    time_fig.savefig(f'{foldername}/convergence_vs_time_plot.png', dpi=150, bbox_inches='tight')

    # Save results to JSON
    comparison.save_results_to_json(f'{foldername}/results.json')

    print(f'Saved all plots and results to: {foldername}/')

    return comparison, summary


if __name__ == "__main__":
    # Get user input for which methods to run
    # print("Available methods: lbfgs, multistart_lbfgs, differential_evolution, dual_annealing, bayesian, basin_hopping, multi_basin_hopping")
    
    # methods_input = input("Enter the methods to run (comma separated, or press Enter for all): ").strip()

    # if methods_input == "":
    #     methods = ['lbfgs', 'multistart_lbfgs', 'differential_evolution', 'dual_annealing', 'bayesian', 'basin_hopping', 'multi_basin_hopping']  # Run all methods
    # else:
    #     methods = [m.strip() for m in methods_input.split(",")]

    methods = ["multistart_lbfgs","bayesian"]

    eqdsk = read_eqdsk('examples/data/eqdsk/MANTA_posCS_final')
    # eqdsk = read_eqdsk('examples/data/eqdsk/g192185.02440')
    LCFS_contour = eqdsk['rzout'].copy()
    mesh_dx = 0.015

    # Create mesh
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

    # Set targets
    Ip_target = eqdsk['ip']
    pres_target = eqdsk['pres'][0]
    mygs.set_targets(Ip=Ip_target, pax=pres_target)

    # Solve
    print("Solving fixed-boundary equilibrium...")
    mygs.init_psi()
    mygs.solve()

    # Run comparison
    for num_coils in [2,3,4,5,6,7,8]:
        for reg_in in [1e-8,1e-7,1e-6,5*1e-6,1e-5]:
            try:
                comparison, summary = main(mygs=mygs, methods=methods, NUM_COILS=num_coils, REG_IN=reg_in, MAX_TIME=120)
            except Exception as e:
                print(f"\n Failed for NUM_COILS={num_coils}, REG_IN={reg_in}")
                print(f"\n Error: {e}")
                continue

    # comparison, summary = main(mygs=mygs, methods=methods, NUM_COILS=5, REG_IN=1e-5, MAX_TIME=10)
