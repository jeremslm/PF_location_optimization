"""
Cost-landscape Sobol scan for DIII-D 3-coil case.

Constraints: theta1 = 20 deg, mu1 = mu2 = mu3 = 0.
Sobol-samples (theta2, theta3). One scan per regularization value.

Modes:
  fixed: fixed-boundary cost (cheap lstsq, ~ms/eval). Sweep over LAMBDAS.
  free:  free-boundary cost via TokaMaker GS solve (~30s/eval). Sweep over WEIGHTS_FB.

Saves npz per scan for plotting in notebooks/landscape_smoothness.ipynb.
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
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
from opt_comp_combined_boundary import _free_boundary_cost

NUM_COILS = 3
N_SAMPLES_FIXED = 512 * 512
N_SAMPLES_FREE = 720
LAMBDAS = [1e-5, 1e-6, 1e-7, 1e-8]
WEIGHTS_FB = [1e-4, 1e-3, 1e-2, 1e-1]
THETA1_FIXED = 20.0
OMEGA = 1e-3
DIST_TH = 5.0
RFIL = 0.01
ANGULAR_BOUNDS = (10, 170)
SOBOL_SEED = 42
OUT_DIR_FIXED = "examples/comparisons/closed_boundary_DIIID/landscape/coils:3"
OUT_DIR_FREE = "examples/comparisons/free_boundary_DIIID/landscape/coils:3"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_objective(r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2, num_coils, reg_in, omega):
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
            locs.append([(1 - rho) * R_inner + rho * R_outer,
                         (1 - rho) * Z_inner + rho * Z_outer])

        coil_centers_3x3 = []
        for loc in locs:
            centers_top = [[loc[0] + 2*RFIL*dx, loc[1] + 2*RFIL*dy]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            centers_bot = [[loc[0] + 2*RFIL*dx, -loc[1] + 2*RFIL*dy]
                           for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            coil_centers_3x3.append(centers_top)
            coil_centers_3x3.append(centers_bot)

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

        dist_angles = np.diff(np.sort(thetas))
        pen_terms = np.maximum(DIST_TH - dist_angles, 0.0) ** 2
        dist_penalty = omega * np.sum(pen_terms)

        return flux_error_squared + dist_penalty, flux_error_squared

    return objective


def setup_diiid_fixed():
    eqdsk = read_eqdsk("examples/data/eqdsk/g192185.02440")
    LCFS_contour = eqdsk["rzout"].copy()
    mesh_dx = 0.015

    gs_mesh = gs_Domain()
    gs_mesh.define_region("plasma", mesh_dx, "plasma")
    gs_mesh.add_polygon(LCFS_contour, "plasma")
    mesh_pts, mesh_lc, _ = gs_mesh.build_mesh()

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False

    F0 = eqdsk["rcentr"] * eqdsk["bcentr"]
    mygs.setup(order=2, F0=F0)
    mygs.set_targets(Ip=eqdsk["ip"], pax=eqdsk["pres"][0])

    print("solving fixed-boundary equilibrium")
    mygs.init_psi()
    mygs.solve()

    r_bnd, psi_bnd = mygs.get_vfixed()
    print(f"boundary points: {len(r_bnd)}")

    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    return r_bnd, psi_bnd, coil_center_cand1, coil_center_cand2


def setup_diiid_free():
    """Setup for free-boundary mode. Mirrors parallel_case in opt_comp_combined_boundary.py."""
    tmp_dir = os.path.join(_BASE_DIR, "tmp", "landscape_free")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    cwd_orig = os.getcwd()
    os.chdir(tmp_dir)

    eqdsk = read_eqdsk(os.path.join(_BASE_DIR, "examples/data/eqdsk/g192185.02440"))
    LCFS_contour = eqdsk["rzout"].copy()
    fixed_LCFS = LCFS_contour
    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    mesh_dx = 0.015
    gs_mesh = gs_Domain()
    gs_mesh.define_region("plasma", mesh_dx, "plasma")
    gs_mesh.add_polygon(LCFS_contour, "plasma")
    mesh_pts, mesh_lc, _ = gs_mesh.build_mesh()

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False
    F0 = eqdsk["rcentr"] * eqdsk["bcentr"]
    mygs.setup(order=2, F0=F0)
    mygs.set_targets(Ip=eqdsk["ip"], pax=eqdsk["pres"][0])
    print("solving fixed-boundary equilibrium for magnetic axis reference")
    mygs.init_psi()
    mygs.solve()

    fixed_mag_axis = np.array([1.77764093, -0.04014656])

    os.chdir(cwd_orig)

    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    return myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, coil_center_cand1, coil_center_cand2


def main_fixed():
    r_bnd, psi_bnd, cand1, cand2 = setup_diiid_fixed()
    os.makedirs(OUT_DIR_FIXED, exist_ok=True)

    th_lo, th_hi = ANGULAR_BOUNDS
    sampler = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED)
    samples_unit = sampler.random(N_SAMPLES_FIXED)
    samples = th_lo + samples_unit * (th_hi - th_lo)

    for lam in LAMBDAS:
        lam_key = f"{lam:.0e}"
        print(f"lambda={lam_key} theta1={THETA1_FIXED} mu_all=0 omega={OMEGA} N={N_SAMPLES_FIXED}")

        obj = make_objective(r_bnd, psi_bnd, cand1, cand2, NUM_COILS, lam, OMEGA)

        cost = np.zeros(N_SAMPLES_FIXED)
        flux = np.zeros(N_SAMPLES_FIXED)
        t0 = time.time()
        for k in range(N_SAMPLES_FIXED):
            th2, th3 = samples[k]
            params = np.array([THETA1_FIXED, th2, th3, 0.0, 0.0, 0.0])
            cost[k], flux[k] = obj(params)
            if (k + 1) % 8192 == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed
                eta = (N_SAMPLES_FIXED - k - 1) / rate
                print(f"  {k+1}/{N_SAMPLES_FIXED} elapsed={elapsed:.1f}s rate={rate:.0f}/s eta={eta:.0f}s")

        idx_min = int(np.argmin(cost))
        min_theta2 = float(samples[idx_min, 0])
        min_theta3 = float(samples[idx_min, 1])
        min_cost = float(cost[idx_min])
        print(f"  min: theta2={min_theta2:.3f} theta3={min_theta3:.3f} cost={min_cost:.4e}")

        out_path = os.path.join(OUT_DIR_FIXED, f"lambda:{lam_key}.npz")
        np.savez(
            out_path,
            samples=samples,
            cost=cost,
            flux_err=flux,
            theta1_anchor=THETA1_FIXED,
            mu_anchor=0.0,
            min_theta2=min_theta2,
            min_theta3=min_theta3,
            min_cost=min_cost,
            lam=lam,
            omega=OMEGA,
            num_coils=NUM_COILS,
            angular_bounds=np.array(ANGULAR_BOUNDS),
            n_samples=N_SAMPLES_FIXED,
            sobol_seed=SOBOL_SEED,
        )
        print(f"  saved {out_path} ({time.time()-t0:.1f}s total)")


def main_free(n_samples=N_SAMPLES_FREE, weights=WEIGHTS_FB):
    myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, cand1, cand2 = setup_diiid_free()
    os.makedirs(OUT_DIR_FREE, exist_ok=True)

    th_lo, th_hi = ANGULAR_BOUNDS
    sampler = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED)
    samples_unit = sampler.random(n_samples)
    samples = th_lo + samples_unit * (th_hi - th_lo)

    for w in weights:
        w_key = f"{w:.0e}"
        print(f"weight_fb={w_key} theta1={THETA1_FIXED} mu_all=0 N={n_samples}")

        cost = np.zeros(n_samples)
        failed = np.zeros(n_samples, dtype=bool)
        t0 = time.time()
        for k in range(n_samples):
            th2, th3 = samples[k]
            params = np.array([THETA1_FIXED, th2, th3, 0.0, 0.0, 0.0])
            c = _free_boundary_cost(params, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
                                    cand1, cand2, lim, w, NUM_COILS)
            cost[k] = c
            failed[k] = c >= 1e6
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (n_samples - k - 1) / rate
            print(f"  {k+1}/{n_samples} cost={c:.4e} elapsed={elapsed:.1f}s rate={rate:.3f}/s eta={eta:.0f}s")

        valid = ~failed
        if valid.any():
            idx_min = int(np.argmin(np.where(valid, cost, np.inf)))
            min_theta2 = float(samples[idx_min, 0])
            min_theta3 = float(samples[idx_min, 1])
            min_cost = float(cost[idx_min])
        else:
            idx_min = -1
            min_theta2 = float("nan")
            min_theta3 = float("nan")
            min_cost = float("nan")
        print(f"  min: theta2={min_theta2:.3f} theta3={min_theta3:.3f} cost={min_cost:.4e} failures={failed.sum()}/{n_samples}")

        out_path = os.path.join(OUT_DIR_FREE, f"weight:{w_key}.npz")
        np.savez(
            out_path,
            samples=samples,
            cost=cost,
            failed=failed,
            theta1_anchor=THETA1_FIXED,
            mu_anchor=0.0,
            min_theta2=min_theta2,
            min_theta3=min_theta3,
            min_cost=min_cost,
            weight_fb=w,
            num_coils=NUM_COILS,
            angular_bounds=np.array(ANGULAR_BOUNDS),
            n_samples=n_samples,
            sobol_seed=SOBOL_SEED,
        )
        print(f"  saved {out_path} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "free"], default="fixed")
    parser.add_argument("--n", type=int, default=None, help="N samples (free mode default 256, fixed mode default 262144)")
    args = parser.parse_args()

    if args.mode == "fixed":
        main_fixed()
    else:
        n = args.n if args.n is not None else N_SAMPLES_FREE
        main_free(n_samples=n)


# python landscape_scan_diiid.py --mode free --n 64
