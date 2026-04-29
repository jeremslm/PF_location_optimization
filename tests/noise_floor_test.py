"""
Measure objective noise floor by evaluating the same point N times per weight.
Reports fixed_cost (deterministic) and fb_cost (noisy) separately.
"""

import os
import sys
import shutil
import numpy as np

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
from opt_comp_combined_boundary import make_combined_objective, _BASE_DIR

N_REPEATS = 5
NUM_COILS = 3
ALPHA = 0.75
WEIGHTS = [1e-4, 1e-3, 1e-2, 1e-1]
TEST_PARAMS = np.array([40.0, 90.0, 140.0, 0.3, 0.5, 0.7])

tmp_dir = os.path.join(_BASE_DIR, 'tmp', 'noise_floor_test')
shutil.rmtree(tmp_dir, ignore_errors=True)
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

myOFT = OFT_env(nthreads=2)
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

lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
coil_center_cand1 = resize_polygon(lim1, dx=0.1)
lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
coil_center_cand2 = resize_polygon(lim2, dx=0.15)

r_bnd, psi_bnd = mygs.get_vfixed()
theta_range = np.linspace(0, 180, len(coil_center_cand1) // 2)
inner = coil_center_cand1[:len(coil_center_cand1) // 2]
outer = coil_center_cand2[:len(coil_center_cand2) // 2]

print(f"test_params={TEST_PARAMS}")
print(f"n_repeats={N_REPEATS} per weight\n")

for weight_fb in WEIGHTS:
    obj = make_combined_objective(
        ALPHA, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
        coil_center_cand1, coil_center_cand2, lim,
        r_bnd, psi_bnd, weight_fb, NUM_COILS, 0.01,
        1e-6, 1e-3, 5.0, theta_range, inner, outer
    )

    fixed_costs = []
    fb_costs = []
    combined_costs = []

    for i in range(N_REPEATS):
        obj.norm_fixed = None
        obj.norm_fb = None
        c = obj(TEST_PARAMS.copy())
        fixed_costs.append(obj.last_flux_err)
        fb_costs.append(obj.last_fb_cost)
        combined_costs.append(c)
        print(f"weight={weight_fb:.0e} rep={i+1} fixed={obj.last_flux_err:.6e} fb={obj.last_fb_cost:.6e} combined={c:.6e}")

    fixed_costs = np.array(fixed_costs)
    fb_costs = np.array(fb_costs)
    combined_costs = np.array(combined_costs)
    print(f"  fixed:    mean={fixed_costs.mean():.6e} std={fixed_costs.std():.2e} range={np.ptp(fixed_costs):.2e}")
    print(f"  fb:       mean={fb_costs.mean():.6e} std={fb_costs.std():.2e} range={np.ptp(fb_costs):.2e}")
    print(f"  combined: mean={combined_costs.mean():.6e} std={combined_costs.std():.2e} range={np.ptp(combined_costs):.2e}")
    print()

shutil.rmtree(tmp_dir, ignore_errors=True)