"""
PF Coil Optimization Module
============================

Core module for optimizing poloidal field coil locations in tokamak equilibria.
"""

import os
import sys
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import qmc
import pandas as pd
import matplotlib as plt

# Import helper functions from existing module
try:
    from .helper_fct import (
        resize_polygon,
        resize_polygon_MANTA,
        update_boundary,
        plot_coil,
        place_points
    )
except ImportError:
    from helper_fct import (
        resize_polygon,
        resize_polygon_MANTA,
        update_boundary,
        plot_coil,
        place_points,
        smoothen,
    )

# Optional dependencies

from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence

# Importing OFT 
home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path

tokamaker_python_path = os.getenv("OFT_ROOTPATH")

if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path,'python'))
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, eval_green

class CoilPositionSpace:
    """
    Define search space for coil positions.
    
    Represents region bounded by two curves (inner/outer).
    Coil positions parameterized by:
    - Poloidal angle (theta): 0-180 degrees for top half
    - Radial ratio (rho): 0=inner boundary, 1=outer boundary
    """
    
    def __init__(self, inner_boundary=None, outer_boundary=None, method='coords',
                 angular_bounds=None, radial_bounds=None, 
                 single_curve=None, dx_inner=None, dx_outer=None, smoothen_window = 5):
        """
        Initialize coil position space.

        Parameters
        ----------
        inner_boundary: ndarray (N, 2) or dict or callable
            - If method='coords': (R, Z) array
            - If method='parametric': dict with keys {r0, z0, a, kappa, delta, squar, npts, offset}
            - If method='function': callable(theta) -> (R, Z)
            - If method='single_curve': pass in a curve like a limiter
                - dx_inner = offset of curve which defines the inner boundary
                - dx_outer = offset of the curve which defines the outer boundary
        outer_boundary: same as inner_boundary
        method: str
            'coords', 'parametric', 'function', 'single_curve' 
        angular_bounds: tuple (float, float), optional
            (min, max) poloidal angle in degrees. Default: (0, 180)
        radial_bounds: tuple (float, float), optional
            (min, max) radial ratio. Default: (0, 1)
        """
        self.method = method

        # Set bounds (use defaults if not provided)
        self.angular_bounds = angular_bounds if angular_bounds is not None else (0, 180)
        self.radial_bounds = radial_bounds if radial_bounds is not None else (0, 1)

        if method == 'coords':
            # Direct (R, Z) arrays
            if inner_boundary is None or outer_boundary is None:
                raise ValueError("For method 'coords', inner boundary and outer boundary curves must not be None")

            self.inner_curve = np.asarray(inner_boundary, dtype=float)
            self.outer_curve = np.asarray(outer_boundary, dtype=float)

            if self.inner_curve.shape[1] != 2 or self.outer_curve.shape[1] != 2:
                raise ValueError("Boundary curves must have shape (N, 2)")
            if len(self.inner_curve) != len(self.outer_curve):
                raise ValueError("Inner and outer curves must have same number of points")

        elif method == 'parametric':
            # Generate from plasma shape parameters (like your notebook!)
            if inner_boundary is None or outer_boundary is None:
                raise ValueError("For method 'parametric', inner boundary and outer boundary curves must not be None")

            try:
                # Inner boundary
                lim_inner = update_boundary(
                    r0=inner_boundary['r0'],
                    z0=inner_boundary.get('z0', 0),
                    a0=inner_boundary['a'],
                    kappa=inner_boundary['kappa'],
                    delta=inner_boundary['delta'],
                    squar=inner_boundary['squar'],
                    npts=inner_boundary.get('npts', 1700)
                )
                self.inner_curve = resize_polygon(lim_inner, dx=inner_boundary['offset'])

                # Outer boundary
                lim_outer = update_boundary(
                    r0=outer_boundary['r0'],
                    z0=outer_boundary.get('z0', 0),
                    a0=outer_boundary['a'],
                    kappa=outer_boundary['kappa'],
                    delta=outer_boundary['delta'],
                    squar=outer_boundary['squar'],
                    npts=outer_boundary.get('npts', 1700)
                )
                self.outer_curve = resize_polygon(lim_outer, dx=outer_boundary['offset'])

            except KeyError as e:
                raise KeyError(f"Missing required key for method 'parametric': {e}. "
                             f"Required: r0, a, kappa, delta, squar, offset. "
                             f"Optional: z0 (default=0), npts (default=1700)")

        elif method == 'function':
            # User-provided functions
            if not callable(inner_boundary) or not callable(outer_boundary):
                raise TypeError("For method 'function', boundaries must be callable: func(theta) -> (R, Z)")

            self.inner_func = inner_boundary
            self.outer_func = outer_boundary

            # Pre-compute curves for standard interpolation (sample at 0-180 degrees)
            npts = 1700
            theta_samples = np.linspace(0, 180, npts)

            inner_pts = []
            outer_pts = []
            for theta in theta_samples:
                R_in, Z_in = self.inner_func(theta)
                R_out, Z_out = self.outer_func(theta)
                inner_pts.append([R_in, Z_in])
                outer_pts.append([R_out, Z_out])

                # to-do !!! mapping from parameters to R,Z positions array = defined by a fct that user provides 
                # would have to change the way we define CoilPositionSpace

            self.inner_curve = np.array(inner_pts)
            self.outer_curve = np.array(outer_pts)
        
        elif method == 'single_curve':
            if single_curve is None or dx_inner is None or dx_outer is None:
                raise ValueError("For method 'single_curve', single_curve, dx_inner, and dx_outer must not be None")
            
            if(dx_outer<dx_inner):
               raise ValueError("For method 'single_curve, dx_outer must be greater than dx_inner")

            if(dx_outer <= 0 or dx_inner <= 0):
                raise ValueError("For method 'single_curve, dx_outer and dx_inner must be greater 0")
        
            smooth_curve = smoothen(single_curve, smoothen_window)
            self.outer_curve = resize_polygon(smooth_curve, dx_outer)
            self.inner_curve = resize_polygon(smooth_curve, dx_inner)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'coords', 'single_curve', 'parametric', or 'function'")
        

    def interpolate(self, theta, radial):
        """Get (R, Z) position from poloidal angle and radial ratio."""
        theta_range = np.linspace(0, 180, len(self.inner_curve))
        
        R_inner = np.interp(theta, theta_range, self.inner_curve[:, 0])
        Z_inner = np.interp(theta, theta_range, self.inner_curve[:, 1])
        R_outer = np.interp(theta, theta_range, self.outer_curve[:, 0])
        Z_outer = np.interp(theta, theta_range, self.outer_curve[:, 1])
        
        R = (1 - radial) * R_inner + radial * R_outer
        Z = (1 - radial) * Z_inner + radial * Z_outer
        
        return R, Z
    
    def get_bounds(self):
        """
        Get bounds for optimization parameters.

        Returns
        -------
        theta_bounds : tuple (float, float)
            (min, max) poloidal angle in degrees
        radial_bounds : tuple (float, float)
            (min, max) radial ratio
        """
        return self.angular_bounds, self.radial_bounds

    def set_bounds(self, angular_bounds=None, radial_bounds=None):
        """
        Set bounds for optimization parameters.

        Parameters
        ----------
        angular_bounds : tuple (float, float), optional
            (min, max) poloidal angle in degrees
        radial_bounds : tuple (float, float), optional
            (min, max) radial ratio

        Examples
        --------
        >>> space = CoilPositionSpace(inner, outer, method='curve')
        >>> space.set_bounds(angular_bounds=(10, 170), radial_bounds=(0.1, 0.9))
        """
        if angular_bounds is not None:
            self.angular_bounds = angular_bounds
        if radial_bounds is not None:
            self.radial_bounds = radial_bounds


class PerCoilPositionSpace:
    """Define per-coil search spaces."""
    
    def __init__(self, coil_specs, global_space=None):
        """Initialize per-coil position spaces."""
        self.coil_spaces = []
        self.global_space = global_space
        
        for i, spec in enumerate(coil_specs):
            if spec == 'global':
                if global_space is None:
                    raise ValueError(f"Coil {i} uses 'global' but no global_space provided")
                self.coil_spaces.append(None)
            elif isinstance(spec, CoilPositionSpace):
                self.coil_spaces.append(spec)
            else:
                raise NotImplementedError(f"Coil spec type {type(spec)} not yet implemented")
    
    def get_bounds_for_coil(self, coil_index):
        """Get bounds for specific coil."""
        space = self.coil_spaces[coil_index]
        if space is None:
            space = self.global_space
        return space.get_bounds()
    
    def interpolate_for_coil(self, coil_index, theta, radial):
        """Get (R, Z) position for specific coil."""
        space = self.coil_spaces[coil_index]
        if space is None:
            space = self.global_space
        return space.interpolate(theta, radial)


class OptimizationResult:
    """Container for optimization results."""
    
    def __init__(self, success, method, angles, radials, positions, currents,
                 coil_geometry, cost_outer, cost_inner, flux_error,
                 n_iterations, message):
        """Initialize optimization result."""
        self.success = success
        self.method = method
        self.angles = angles
        self.radials = radials
        self.positions = positions
        self.currents = currents
        self.coil_geometry = coil_geometry
        self.cost_outer = cost_outer
        self.cost_inner = cost_inner
        self.flux_error = flux_error
        self.n_iterations = n_iterations
        self.message = message
    
    def __repr__(self):
        """String representation."""
        s = f"OptimizationResult(method='{self.method}', success={self.success})\n"
        s += f"  Cost (outer): {self.cost_outer:.6e}\n"
        s += f"  Cost (inner): {self.cost_inner:.6e}\n"
        s += f"  Flux error: {self.flux_error:.6e}\n"
        s += f"  Iterations: {self.n_iterations}\n"
        s += f"  Message: {self.message}\n"
        return s


def _make_coils_from_params(params, ncoils, position_space, coil_dx=0.08, coil_dy=0.08):
    """Generate coil geometry from optimization parameters."""
    thetas = params[:ncoils]
    radials = params[ncoils:2*ncoils]
    
    # Get (R, Z) locations for top-side coils
    if isinstance(position_space, CoilPositionSpace):
        locs = np.array([position_space.interpolate(theta, radial)
                        for theta, radial in zip(thetas, radials)])
    elif isinstance(position_space, PerCoilPositionSpace):
        locs = np.array([position_space.interpolate_for_coil(i, theta, radial)
                        for i, (theta, radial) in enumerate(zip(thetas, radials))])
    else:
        raise TypeError(f"Unknown position_space type: {type(position_space)}")
    
    # Create coil geometry dictionary
    coil_geometry = {"coils": {}}
    
    for i, loc in enumerate(locs):
        pts_top = np.array([
            [loc[0] - coil_dx, loc[1] + coil_dy],
            [loc[0] + coil_dx, loc[1] + coil_dy],
            [loc[0] + coil_dx, loc[1] - coil_dy],
            [loc[0] - coil_dx, loc[1] - coil_dy]
        ])
        pts_bot = pts_top * np.array([1, -1])
        
        coil_geometry["coils"][f'F{i}A'] = {'pts': copy.deepcopy(pts_top), 'nturns': 1.0}
        coil_geometry["coils"][f'F{i}B'] = {'pts': copy.deepcopy(pts_bot), 'nturns': 1.0}
    
    return coil_geometry


def _compute_coil_centers(coil_geometry):
    """Compute center of each coil."""
    coil_centers = []
    for coil_name in coil_geometry["coils"]:
        pts = np.array(coil_geometry["coils"][coil_name]["pts"])
        center = np.mean(pts, axis=0)
        coil_centers.append(np.asarray([center]))
    return coil_centers


def _make_3x3_thick(center, R):
    """Generate centers of 9 filaments in 3×3 arrangement."""
    R0, Z0 = center
    offsets = [-1, 0, 1]
    fil_centers = []
    for dx in offsets:
        for dy in offsets:
            fil_centers.append([R0 + 2 * R * dx, Z0 + 2 * R * dy])
    return fil_centers


def _objective_function(params, tokamaker_solver, position_space, n_coils,
                       r_bnd, psi_bnd, omega, dist_th, reg_in, Rfil):
    """Objective function for coil position optimization."""
    # Generate coil geometry
    coil_geometry = _make_coils_from_params(params, n_coils, position_space)
    coil_centers = _compute_coil_centers(coil_geometry)
    
    # Add 3×3 thick coil model
    coil_centers_3x3 = []
    for center in coil_centers:
        thick_centers = _make_3x3_thick(center[0], Rfil)
        coil_centers_3x3.append(thick_centers)
    
    n_bnd = psi_bnd.shape[0]
    n_coils_total = len(coil_centers_3x3)
    
    # Build constraint matrix
    con = np.zeros((n_bnd - 1 + n_coils_total, n_coils_total))
    
    for i, filament_set in enumerate(coil_centers_3x3):
        flux_tmp = np.zeros((n_bnd,))
        for fil in filament_set:
            flux_tmp += eval_green(r_bnd, fil)
        con[:n_bnd - 1, i] = flux_tmp[1:] - flux_tmp[0]
        con[n_bnd - 1 + i, i] = reg_in
    
    # Right-hand side
    err = np.zeros((n_bnd - 1 + n_coils_total,))
    err[:n_bnd - 1] = psi_bnd[1:] - psi_bnd[0]
    
    # Solve least-squares
    currents, residuals, _, _ = np.linalg.lstsq(con, err, rcond=None)
    
    # Inner cost
    if len(residuals) > 0:
        inner_cost = residuals[0]
    else:
        inner_cost = np.linalg.norm(np.dot(con, currents) - err) ** 2
    
    # Distance penalty
    thetas = params[:n_coils]
    dist_angles = np.diff(np.sort(thetas))
    pen_terms = np.maximum(dist_th - dist_angles, 0.0) ** 2
    dist_penalty = omega * np.sum(pen_terms)
    
    total_cost = inner_cost + dist_penalty
    
    return total_cost, currents, inner_cost


def _optimize_lbfgs(initial_params, bounds, objective_args, verbose=True):
    """Run L-BFGS-B optimization."""
    final_currents = None
    inner_cost = None
    
    def objective_wrapper(params):
        nonlocal final_currents, inner_cost
        cost, currents, inner = _objective_function(params, **objective_args)
        final_currents = currents
        inner_cost = inner
        return cost
    
    result = minimize(
        objective_wrapper,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': verbose}
    )
    
    return result, final_currents, inner_cost

def _optimize_multi_start_lbfgs(initial_params, bounds, objective_args, verbose=True, n_starts=10):
    """Run multi-start L-BFGS-B optimization."""

    final_currents = None
    inner_cost = None

    def objective_wrapper(params):
        nonlocal final_currents, inner_cost
        cost, currents, inner = _objective_function(params, **objective_args)
        final_currents = currents
        inner_cost = inner
        return cost

    # Sobol sequence sampling
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=42)
    samples = sampler.random(n_starts)

    # Scale to bounds
    starts = []
    for i in range(n_starts):
        point = []
        for j, (low, high) in enumerate(bounds):
            point.append(low + samples[i, j] * (high - low))
        starts.append(point)

    results = []
    for start in starts:
        try:
            result = minimize(
                objective_wrapper,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': verbose}
            )
            results.append(result)
        except Exception as e:
            print(f"Error optimizing with start {start}: {e}")
            continue

    if len(results) == 0:
        return None, None, None

    return results, final_currents, inner_cost

def _optimize_bayesian(bounds, objective_args, n_calls=100, n_initial_points=50,
                       acq_func='EI', random_state=42, n_jobs=1,
                       local_optimize=False, n_local_refine=5, verbose=True):
    """
    Run Bayesian optimization using Gaussian Process surrogate model.

    Parameters
    ----------
    bounds : list of tuples
        Parameter bounds [(low, high), ...]
    objective_args : dict
        Arguments for _objective_function
    n_calls : int
        Total number of evaluations
    n_initial_points : int
        Number of random initial points before using GP
    acq_func : str
        Acquisition function: 'EI' (Expected Improvement), 'LCB', 'PI'
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs
    local_optimize : bool
        If True, run L-BFGS on top results after Bayesian optimization
    n_local_refine : int
        Number of top Bayesian results to refine with L-BFGS
    verbose : bool
        Print progress information

    Returns
    -------
    result : OptimizeResult
        Bayesian optimization result (or best L-BFGS result if local_optimize)
    final_currents : ndarray
        Optimized coil currents
    inner_cost : float
        Inner loop cost (flux matching error)
    """
    final_currents = None
    inner_cost = None

    def objective_wrapper(params):
        nonlocal final_currents, inner_cost
        cost, currents, inner = _objective_function(params, **objective_args)
        final_currents = currents
        inner_cost = inner
        return cost

    # Build search space from bounds
    space = []
    n_params = len(bounds)
    n_coils = n_params // 2
    for i in range(n_coils):
        space.append(Real(bounds[i][0], bounds[i][1], name=f'angle_{i}'))
    for i in range(n_coils):
        space.append(Real(bounds[n_coils + i][0], bounds[n_coils + i][1], name=f'radial_{i}'))

    if verbose:
        print(f"Running Bayesian optimization with {n_calls} calls, {n_initial_points} initial points")

    # Run Bayesian optimization
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func=acq_func,
        random_state=random_state,
        initial_point_generator='sobol',
        verbose=verbose,
        acq_optimizer='lbfgs',
        n_jobs=n_jobs,
    )

    if local_optimize and n_local_refine > 0:
        if verbose:
            print(f"\nRefining top {n_local_refine} results with L-BFGS...")

        # Find indices of top N results
        min_indices = np.argsort(result.func_vals)[:n_local_refine]
        best_starts = [result.x_iters[idx] for idx in min_indices]

        # Run L-BFGS from each of the top results
        best_cost = float('inf')
        best_result = None

        for i, start in enumerate(best_starts):
            if verbose:
                print(f"  Refining result {i+1}/{n_local_refine} (cost: {result.func_vals[min_indices[i]]:.6e})")

            try:
                lbfgs_result, curr, inner = _optimize_lbfgs(
                    np.array(start), bounds, objective_args, verbose=False
                )

                if lbfgs_result.fun < best_cost:
                    best_cost = lbfgs_result.fun
                    best_result = lbfgs_result
                    final_currents = curr
                    inner_cost = inner

                if verbose:
                    print(f"    → Refined cost: {lbfgs_result.fun:.6e}")

            except Exception as e:
                if verbose:
                    print(f"    → Failed: {e}")
                continue

        if best_result is not None:
            result = best_result
        else:
            # Recompute final values for best Bayesian result
            objective_wrapper(result.x)
    else:
        # Recompute final values for best result
        objective_wrapper(result.x)

    return result, final_currents, inner_cost

def pf_coil_optimize(tokamaker_solver, reg_current, reg_distance, min_coil_distance, coil_space, n_coils=5,
                     coil_dx=0.08, coil_dy=0.08, coil_filament_radius=0.01,
                     method='lbfgs', local_optimize=False,
                     initial_angles=None, initial_radials=None, bounds=None,
                     
                     # Bayesian optimization parameters
                     n_calls=100, n_initial_points=50, acq_func='EI',
                     random_state=42, n_jobs=1, n_local_refine=5,
                    
                     # Multi-start parameters
                     n_starts=10,
                     verbose=True, plot_result=False):
    """
    Optimize PF coil locations for fixed-boundary equilibrium.
    
    Parameters
    ----------
    tokamaker_solver : TokaMaker instance
        Solver with completed fixed-boundary equilibrium
    coil_space : CoilPositionSpace or PerCoilPositionSpace
        Search space for coil positions
    n_coils : int
        Number of coil pairs (top/bottom)
    method : str
        Optimization method ('lbfgs', 'multi_start_lbfgs', 'bayesian')
    
    Returns
    -------
    result : OptimizationResult
        Optimization results
    """
    # Validate inputs
    if method not in ['lbfgs', 'multi_start_lbfgs', 'bayesian']:
        raise ValueError(f"Unknown method '{method}'")
    
    # Extract boundary flux from TokaMaker
    if verbose:
        print("Extracting boundary flux from TokaMaker...")
    
    r_bnd, psi_bnd = tokamaker_solver.get_vfixed()
    
    if verbose:
        print(f"  Found {len(r_bnd)} boundary points")
    
    # Generate bounds
    if bounds is None:
        if verbose:
            print("Generating parameter bounds from coil space...")
        bounds = []
        for i in range(n_coils):
            if isinstance(coil_space, CoilPositionSpace):
                theta_bounds, radial_bounds = coil_space.get_bounds()
            else:
                theta_bounds, radial_bounds = coil_space.get_bounds_for_coil(i)
            bounds.append(theta_bounds)
        for i in range(n_coils):
            if isinstance(coil_space, CoilPositionSpace):
                theta_bounds, radial_bounds = coil_space.get_bounds()
            else:
                theta_bounds, radial_bounds = coil_space.get_bounds_for_coil(i)
            bounds.append(radial_bounds)
    
    # Generate initial guess
    if initial_angles is None or initial_radials is None:
        if verbose:
            print("Generating initial guess...")
        if initial_angles is None:
            initial_angles = np.linspace(10, 170, n_coils)
        if initial_radials is None:
            initial_radials = np.linspace(0.2, 0.8, n_coils)
    
    initial_params = np.concatenate([initial_angles, initial_radials])
    
    if verbose:
        print(f"\nStarting optimization with method '{method}'...")
        print(f"  n_coils: {n_coils}")
        print(f"  reg_current: {reg_current:.1e}")
        print(f"  reg_distance: {reg_distance:.1e}")
        print(f"  min_coil_distance: {min_coil_distance} degrees")
    
    # Prepare objective function arguments
    try: 
        objective_args = {
            'tokamaker_solver': tokamaker_solver,
            'position_space': coil_space,
            'n_coils': n_coils,
            'r_bnd': r_bnd,
            'psi_bnd': psi_bnd,
            'omega': reg_distance,
            'dist_th': min_coil_distance,
            'reg_in': reg_current,
            'Rfil': coil_filament_radius
        }
    except Exception as e:
        raise ValueError(f"Error preparing objective function arguments: {e}")
    
    # Run optimization
    if method == 'lbfgs':
        opt_result, final_currents, inner_cost = _optimize_lbfgs(
            initial_params, bounds, objective_args, verbose=verbose
        )
        opt_params = opt_result.x
        success = opt_result.success
        n_iterations = opt_result.nit
        message = opt_result.message
        outer_cost = opt_result.fun

    elif method == 'multi_start_lbfgs':
        results, final_currents, inner_cost = _optimize_multi_start_lbfgs(
            initial_params, bounds, objective_args, verbose=verbose, n_starts=n_starts
        )
        if results is None:
            raise RuntimeError("All multi-start optimizations failed")
        # Find best result
        opt_result = min(results, key=lambda x: x.fun)
        opt_params = opt_result.x
        success = opt_result.success
        n_iterations = sum(r.nit for r in results)
        message = f"Best of {len(results)} starts"
        outer_cost = opt_result.fun
        # Recompute final currents for best result
        _, final_currents, inner_cost = _objective_function(opt_params, **objective_args)

    elif method == 'bayesian':
        opt_result, final_currents, inner_cost = _optimize_bayesian(
            bounds, objective_args, n_calls=n_calls, n_initial_points=n_initial_points,
            acq_func=acq_func, random_state=random_state, n_jobs=n_jobs,
            local_optimize=local_optimize, n_local_refine=n_local_refine, verbose=verbose
        )
        opt_params = np.array(opt_result.x)
        # Handle both scipy and skopt result types
        success = getattr(opt_result, 'success', True)
        n_iterations = getattr(opt_result, 'nit', getattr(opt_result, 'nfev', n_calls))
        message = getattr(opt_result, 'message', f"Bayesian optimization completed with {n_calls} calls")
        outer_cost = opt_result.fun
    
    # Extract results
    opt_angles = opt_params[:n_coils]
    opt_radials = opt_params[n_coils:]
    
    # Generate final coil geometry
    final_geometry = _make_coils_from_params(opt_params, n_coils, coil_space, coil_dx, coil_dy)
    
    # Get all coil positions
    all_positions = []
    for coil_name in sorted(final_geometry["coils"].keys()):
        pts = final_geometry["coils"][coil_name]["pts"]
        center = np.mean(pts, axis=0)
        all_positions.append(center)
    all_positions = np.array(all_positions)
    
    # Compute flux error
    coil_centers = _compute_coil_centers(final_geometry)
    coil_centers_3x3 = [_make_3x3_thick(c[0], coil_filament_radius) for c in coil_centers]
    
    # Compute flux from coils
    psi_computed = np.zeros_like(psi_bnd)
    for i, filament_set in enumerate(coil_centers_3x3):
        for fil in filament_set:
            psi_computed += final_currents[i] * eval_green(r_bnd, fil)
    
    # Flux error (relative)
    psi_rel_target = psi_bnd[1:] - psi_bnd[0]
    psi_rel_computed = psi_computed[1:] - psi_computed[0]
    flux_error = np.linalg.norm(psi_rel_target - psi_rel_computed) ** 2
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"  Success: {success}")
        print(f"  Outer cost: {outer_cost:.6e}")
        print(f"  Inner cost: {inner_cost:.6e}")
        print(f"  Flux error: {flux_error:.6e}")
        print(f"  Iterations: {n_iterations}")
    
    # Create result object
    result = OptimizationResult(
        success=success,
        method=method,
        angles=opt_angles,
        radials=opt_radials,
        positions=all_positions,
        currents=final_currents,
        coil_geometry=final_geometry,
        cost_outer=outer_cost,
        cost_inner=inner_cost,
        flux_error=flux_error,
        n_iterations=n_iterations,
        message=message
    )
    
    if plot_result:
        print("\nPlotting not yet implemented in Sprint 1")
    
    return result
