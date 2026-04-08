"""
Acquisition Function Sweep
==========================

Runs Bayesian optimization with each of the three main acquisition
functions (EI, PI, LCB) to compare solution quality and convergence.

Each (acq_func, coils, lambda) result is saved to:
  examples/comparisons/closed_boundary_DIIID/
    acq_sweep/acq:{ACQ_FUNC}/lambda:{REG_IN},coils:{NUM_COILS}/run_{N:02d}/
"""

import argparse
import os
import matplotlib.pyplot as plt

from opt_comp_convergence import (
    OFT_env, TokaMaker, gs_Domain,
    read_eqdsk, eval_green,
    resize_polygon, update_boundary,
    CoilPositionSpace,
    OptimizationComparison,
    np, json,
)

ACQ_FUNCS = ['EI', 'gp_hedge', 'LCB']

def main(mygs, acq_func, methods=None, **kwargs):
    NUM_COILS = kwargs.get('NUM_COILS', 4)
    MAX_EVALS = kwargs.get('MAX_EVALS', 2**18)
    MAX_TIME = kwargs.get('MAX_TIME', 86400)
    CONVERGENCE_THRESHOLD = kwargs.get('CONVERGENCE_THRESHOLD', 0.001)
    OMEGA = kwargs.get('OMEGA', 1e-3)
    DIST_TH = kwargs.get('DIST_TH', 5.0)
    REG_IN = kwargs.get('REG_IN', 1e-7)
    RFIL = kwargs.get('RFIL', 0.01)
    RUN_FOLDER = kwargs.get('RUN_FOLDER', f'acq_sweep/acq:{acq_func}')

    if methods is None:
        methods = ['bayesian']

    r_bnd, psi_bnd = mygs.get_vfixed()
    print(f"Found {len(r_bnd)} boundary points")

    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_cand2 = resize_polygon(lim2, dx=0.15)

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

    print("\n" + "=" * 60)
    print(f"ACQ_FUNC={acq_func} | Coils={NUM_COILS} | Max evals={MAX_EVALS}")
    print(f"omega={OMEGA} | reg_in={REG_IN} | dist_th={DIST_TH}")
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

    bf_path = f'examples/comparisons/closed_boundary_DIIID/brute_force/lambda:{REG_IN},coils:{NUM_COILS}/results.json'
    if os.path.exists(bf_path):
        with open(bf_path, 'r') as f:
            bf_data = json.load(f)
        comparison.brute_force_cost = bf_data['best_cost']
        print(f"Brute force baseline: {comparison.brute_force_cost:.6e}")
    else:
        print(f"No brute force baseline found at {bf_path}")

    if 'bayesian' in methods:
        print(f"Running Bayesian Optimization (acq_func={acq_func})...")
        comparison.run_bayesian(acq_func=acq_func, bayesian_stagnation_window=25)

    base = f'examples/comparisons/closed_boundary_DIIID/{RUN_FOLDER}/lambda:{REG_IN},coils:{NUM_COILS}'
    run_idx = 1
    while os.path.exists(os.path.join(base, f'run_{run_idx:02d}', 'results.json')):
        run_idx += 1
    foldername = os.path.join(base, f'run_{run_idx:02d}')
    os.makedirs(foldername, exist_ok=True)

    fig = comparison.plot_result()
    time_fig = comparison.plot_convergence_vs_time(log_scale=True)
    fig.savefig(f'{foldername}/convergence_plot.png', dpi=150, bbox_inches='tight')
    time_fig.savefig(f'{foldername}/convergence_vs_time_plot.png', dpi=150, bbox_inches='tight')
    results_path = f'{foldername}/results.json'
    comparison.save_results_to_json(results_path)
    with open(results_path) as f:
        saved = json.load(f)
    saved['optimization_settings']['acq_func'] = acq_func
    with open(results_path, 'w') as f:
        json.dump(saved, f, indent=2)
    plt.close('all')
    print(f"Saved to: {foldername}/")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acq_funcs', type=str, nargs='+', default=ACQ_FUNCS,
                        choices=ACQ_FUNCS, help='Acquisition functions to sweep')
    parser.add_argument('--coils', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                        help='Coil counts to run')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1e-8, 1e-7, 1e-6, 1e-5],
                        help='REG_IN values to run')
    args = parser.parse_args()

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

    print("Solving fixed-boundary equilibrium...")
    mygs.init_psi()
    mygs.solve()

    for acq_func in args.acq_funcs:
        for num_coils in args.coils:
            for reg_in in args.lambdas:
                print(f"\n{'='*60}")
                print(f"ACQ_FUNC={acq_func} | NUM_COILS={num_coils} | REG_IN={reg_in}")
                print(f"{'='*60}")
                try:
                    main(
                        mygs=mygs,
                        acq_func=acq_func,
                        NUM_COILS=num_coils,
                        REG_IN=reg_in,
                        MAX_EVALS=2**18,
                        RUN_FOLDER=f'acq_sweep/acq:{acq_func}',
                    )
                except Exception as e:
                    print(f"\nFailed: ACQ_FUNC={acq_func}, NUM_COILS={num_coils}, REG_IN={reg_in}")
                    print(f"Error: {e}")
                    continue
