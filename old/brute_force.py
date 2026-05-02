"""
Brute Force Sobol Sampling for PF Coil Optimization
=====================================================

Evaluates the objective function at Sobol-sampled points to find the best
coil placement without any optimization. Serves as a baseline comparison.
"""

import numpy as np
import json
import time
import os
import sys

from scipy.stats import qmc

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

# ========================================
# Fixed parameters
# ========================================
N_SAMPLES = 2 ** 18
OMEGA = 1e-7
DIST_TH = 5.0
RFIL = 0.01
ANGULAR_BOUNDS = (10, 170)
RADIAL_BOUNDS = (0, 1)


def make_objective(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                   num_coils, reg_in):
    """Create objective function for given parameters."""

    theta_range = np.linspace(0, 180, len(coil_center_cand1) // 2)
    inner = coil_center_cand1[:len(coil_center_cand1) // 2]
    outer = coil_center_cand2[:len(coil_center_cand2) // 2]
    n_bnd = psi_bnd.shape[0]

    def objective(params):
        thetas = params[:num_coils]
        radials = params[num_coils:]

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

        # Build least squares system
        n_coils_total = len(coil_centers_3x3)
        con = np.zeros((n_bnd - 1 + n_coils_total, n_coils_total))

        for i, filament_set in enumerate(coil_centers_3x3):
            flux_tmp = np.zeros((n_bnd,))
            for fil in filament_set:
                flux_tmp += eval_green(r_bnd, fil)
            con[:n_bnd-1, i] = flux_tmp[1:] - flux_tmp[0]
            con[n_bnd-1+i, i] = reg_in

        err = np.zeros((n_bnd - 1 + n_coils_total,))
        err[:n_bnd-1] = psi_bnd[1:] - psi_bnd[0]
        currs, residuals, _, _ = np.linalg.lstsq(con, err, rcond=None)

        if len(residuals) > 0:
            flux_error_squared = residuals[0]
        else:
            flux_error_squared = np.linalg.norm(np.dot(con, currs) - err) ** 2

        # Distance penalty
        dist_angles = np.diff(np.sort(thetas))
        pen_terms = np.maximum(DIST_TH - dist_angles, 0.0) ** 2
        dist_penalty = OMEGA * np.sum(pen_terms)

        total_cost = flux_error_squared + dist_penalty

        return total_cost, flux_error_squared, locs, currs

    return objective


def run_brute_force(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                    num_coils, reg_in):
    """Run brute force Sobol sampling for given parameters."""

    n_params = 2 * num_coils
    bounds_low = np.array([ANGULAR_BOUNDS[0]] * num_coils + [RADIAL_BOUNDS[0]] * num_coils)
    bounds_high = np.array([ANGULAR_BOUNDS[1]] * num_coils + [RADIAL_BOUNDS[1]] * num_coils)

    # Generate Sobol samples
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=42)
    samples = sampler.random(N_SAMPLES)
    scaled_samples = bounds_low + samples * (bounds_high - bounds_low)

    obj = make_objective(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                         num_coils, reg_in)

    best_cost = float('inf')
    best_flux_err = None
    best_params = None
    best_locs = None
    best_currs = None

    start_time = time.time()

    for i, params in enumerate(scaled_samples):
        cost, flux_err, locs, currs = obj(params)
        if cost < best_cost:
            best_cost = cost
            best_flux_err = flux_err
            best_params = params.copy()
            best_locs = locs
            best_currs = currs

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{N_SAMPLES} evaluated, best cost: {best_cost:.6e}")

    elapsed = time.time() - start_time

    return {
        'best_cost': best_cost,
        'best_flux_err': best_flux_err,
        'best_params': best_params,
        'best_locs': best_locs,
        'best_currs': best_currs,
        'n_evals': N_SAMPLES,
        'time': elapsed
    }


def save_results(result, num_coils, reg_in, output_dir):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    best_params = result['best_params']
    thetas = best_params[:num_coils].tolist()
    radials = best_params[num_coils:].tolist()
    coil_positions = [[float(loc[0]), float(loc[1])] for loc in result['best_locs']]
    coil_currents = result['best_currs'].tolist()

    save_data = {
        'optimization_settings': {
            'method': 'brute_force_sobol',
            'num_coils': int(num_coils),
            'n_samples': N_SAMPLES,
            'omega': OMEGA,
            'dist_th': DIST_TH,
            'reg_in': float(reg_in),
            'rfil': RFIL
        },
        'best_cost': float(result['best_cost']),
        'best_flux_err': float(result['best_flux_err']),
        'n_evals': int(result['n_evals']),
        'time': float(result['time']),
        'parameters': {
            'thetas': thetas,
            'radials': radials
        },
        'coil_positions_top': coil_positions,
        'coil_currents': coil_currents
    }

    filepath = os.path.join(output_dir, 'results.json')
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"  Saved results to {filepath}")


if __name__ == "__main__":
    # ========================================
    # TokaMaker setup (DIIID)
    # ========================================
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

    r_bnd, psi_bnd = mygs.get_vfixed()
    print(f"Found {len(r_bnd)} boundary points")

    # ========================================
    # Define coil position space (DIIID)
    # ========================================
    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)

    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    # ========================================
    # Run brute force for all combinations
    # ========================================

    for num_coils in [2, 3, 4, 5, 6, 7, 8]:
        for reg_in in [1e-5, 1e-6, 5e-6, 1e-7, 1e-8]:
            print(f"\n{'='*60}")
            print(f"NUM_COILS={num_coils}, REG_IN={reg_in}")
            print(f"{'='*60}")

            try:
                result = run_brute_force(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2,
                                         num_coils, reg_in)

                foldername = f'examples/comparisons/closed_boundary_DIIID/brute_force/lambda:{str(reg_in)},coils:{num_coils}'
                save_results(result, num_coils, reg_in, foldername)

                print(f"  Best cost: {result['best_cost']:.6e}")
                print(f"  Best flux err: {result['best_flux_err']:.6e}")
                print(f"  Time: {result['time']:.1f}s")

            except Exception as e:
                print(f"  Failed: {e}")
                continue