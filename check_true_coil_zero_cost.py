"""
Sanity check: evaluate _free_boundary_cost at TRUE_COILS_TOP (converted to
theta/rho params), confirm fb_cost matches the 1e-12 baseline in
notebooks/target=free_diiid_3_coil/fb_candidate_cost_results.json.
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.optimize import minimize

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path
if oft_root_path:
    sys.path.append(os.path.join(oft_root_path, "python"))

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
from helper_fct import resize_polygon, update_boundary
from opt_comp_combined_boundary import _free_boundary_cost

TRUE_COILS_TOP = np.array([
    [2.6127, 0.4377],
    [2.3744, 1.1156],
    [1.6883, 1.5868],
])
NUM_COILS = 3
WEIGHT_FB = 1e-4


def rz_to_theta_rho(R_true, Z_true, inner, outer):
    theta_range = np.linspace(0, 180, len(inner))
    def _cost(x):
        theta, rho = x
        R_pos = (1 - rho) * np.interp(theta, theta_range, inner[:, 0]) + rho * np.interp(theta, theta_range, outer[:, 0])
        Z_pos = (1 - rho) * np.interp(theta, theta_range, inner[:, 1]) + rho * np.interp(theta, theta_range, outer[:, 1])
        return (R_true - R_pos) ** 2 + (Z_true - Z_pos) ** 2
    res = minimize(_cost, x0=[90.0, 0.5], bounds=[(10, 170), (0, 1)], method="L-BFGS-B",
                   options={"ftol": 1e-20, "gtol": 1e-12})
    recon_err = np.sqrt(res.fun)
    return res.x[0], res.x[1], recon_err


def main():
    json_path = os.path.join(_BASE_DIR, "notebooks", "target=free_diiid_3_coil", "fb_candidate_cost_results.json")
    with open(json_path) as f:
        expected = json.load(f)
    expected_cost = expected[f"{WEIGHT_FB:.0e}"]
    print(f"expected fb_cost from JSON: {expected_cost:.6e}")

    tmp_dir = os.path.join(_BASE_DIR, "tmp", "true_coil_check")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    eqdsk = read_eqdsk(os.path.join(_BASE_DIR, "examples/data/eqdsk/DIIID_opt_3coil_symm"))
    _target = np.load(os.path.join(_BASE_DIR, "notebooks", "fb_lcfs_target.npz"))
    fixed_LCFS = _target["lcfs"]
    eqdsk["rzout"] = fixed_LCFS
    fixed_mag_axis = _target["mag_axis"]
    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    gs_mesh = gs_Domain()
    gs_mesh.define_region("plasma", 0.015, "plasma")
    gs_mesh.add_polygon(fixed_LCFS, "plasma")
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False
    mygs.setup(order=2, F0=eqdsk["rcentr"] * eqdsk["bcentr"])
    mygs.set_targets(Ip=eqdsk["ip"], pax=eqdsk["pres"][0])
    mygs.init_psi()
    mygs.solve()

    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

    inner = coil_center_cand1[:len(coil_center_cand1) // 2]
    outer = coil_center_cand2[:len(coil_center_cand2) // 2]
    xpoint_index = int(np.argmin(fixed_LCFS[:, 1]))

    os.chdir(_BASE_DIR)

    thetas, rhos = [], []
    print("inverting R,Z -> theta,rho:")
    for R_true, Z_true in TRUE_COILS_TOP:
        theta, rho, err = rz_to_theta_rho(R_true, Z_true, inner, outer)
        thetas.append(theta)
        rhos.append(rho)
        print(f"  ({R_true}, {Z_true}) -> theta={theta:.4f} rho={rho:.4f} recon_err={err:.2e} m")

    params = np.array(thetas + rhos)
    print(f"params: {params}")

    fb_cost, _, _ = _free_boundary_cost(
        params, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
        coil_center_cand1, coil_center_cand2, lim,
        WEIGHT_FB, NUM_COILS, xpoint_index=xpoint_index,
    )
    print(f"fb_cost={fb_cost:.6e}  expected={expected_cost:.6e}  ratio={fb_cost/expected_cost:.3f}")
    assert fb_cost < 1e-10, f"FAIL: fb_cost={fb_cost:.6e} exceeds 1e-10"
    print("PASS")


if __name__ == "__main__":
    main()
