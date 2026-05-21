import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from helper_fct import resize_polygon, update_boundary

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, 'examples/comparisons/diiid_3_coil')
SWEEPS = ['convergence_w5_b_temp', 'convergence_w5_l_temp']
HALF = 0.035

# real DIII-D F6/F7/F9 coil centers from DIIID_mesh.h5, the physical truth (coils:3 only)
TRUE_COILS = [
    (2.6127, 0.4377), (2.3744, 1.1156), (1.6883, 1.5868),
    (2.6129, -0.4387), (2.3807, -1.1167), (1.6882, -1.5786),
]

METHOD_COLORS = {'Multi-start L-BFGS': '#2ca02c', 'Bayesian': '#1f77b4'}

# candidate position-space arcs, mirrors opt_comp_combined_boundary.py:1221-1224
lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
CAND1 = resize_polygon(lim1, dx=0.1)
CAND2 = resize_polygon(lim2, dx=0.15)


def plot_config(run_dir, ncoils):
    d = json.load(open(os.path.join(run_dir, 'results.json')))
    method, inner = next(iter(d['methods'].items()))
    color = METHOD_COLORS.get(method, '#7f7f7f')
    top = np.array(inner['coil_positions_top'], dtype=float)
    algo = np.vstack([top, top * np.array([1.0, -1.0])])

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(CAND1[:, 0], CAND1[:, 1], 'k--', alpha=0.3, linewidth=1, label='Position space')
    ax.plot(CAND2[:, 0], CAND2[:, 1], 'k--', alpha=0.3, linewidth=1)

    for R, Z in algo:
        ax.add_patch(plt.Rectangle((R - HALF, Z - HALF), 2 * HALF, 2 * HALF,
                                   facecolor=color, edgecolor=color, alpha=0.6))
    for R, Z in TRUE_COILS:
        ax.add_patch(plt.Rectangle((R - HALF, Z - HALF), 2 * HALF, 2 * HALF,
                                   facecolor='none', edgecolor='red', linewidth=2.0))

    handles = [Patch(facecolor=color, alpha=0.6, label=method),
               Patch(facecolor='none', edgecolor='red', linewidth=2.0, label='true (DIII-D)')]
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('Coil Placement: ' + os.path.basename(os.path.dirname(run_dir)), fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'coil_comp.png'), dpi=130)
    plt.close(fig)


for sweep in SWEEPS:
    sweep_dir = os.path.join(DATA_DIR, sweep)
    for config in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, config, 'run_01')
        if not os.path.exists(os.path.join(run_dir, 'results.json')):
            continue
        ncoils = int(config.split('coils:')[1])
        plot_config(run_dir, ncoils)
        print(f"{sweep}/{config} done")
