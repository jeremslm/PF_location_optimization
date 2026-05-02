"""
Free-Boundary Analysis for PF Coil Optimization
================================================

This script analyzes optimized coil configurations from fixed-boundary optimization
by running free-boundary TokaMaker solves and computing actual currents and flux errors.

Workflow:
1. Load optimized coil positions from test_general/ (fixed-boundary results)
2. For each configuration, run free-boundary TokaMaker solve
3. Extract actual free-boundary currents and compute flux error
4. Generate plots: Total Current vs Flux Error vs Lambda (regularization parameter)
"""

import os
import sys
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from helper_fct import (
    resize_polygon, place_points, update_boundary, plot_coil, place_points_pol_rad
)

plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path

tokamaker_python_path = os.getenv("OFT_ROOTPATH")

if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path,'python'))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh, load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk


# ============================================
# Helper Functions from Notebook
# ============================================

def make_new_coils(params, nCoils, coil_center_cand1, coil_center_cand2, dx=0.03, dy=0.03):
    """
    Generate PF coil geometry from optimized parameters.

    Based on notebook cells 13-14.

    Parameters
    ----------
    params : array-like
        Coil parameters [thetas..., radials...]
    nCoils : int
        Number of coils (top-side only)
    coil_center_cand1 : ndarray
        Inner boundary of coil position space
    coil_center_cand2 : ndarray
        Outer boundary of coil position space
    dx, dy : float
        Coil half-widths

    Returns
    -------
    scan_geom : dict
        Dictionary with coil geometry
    """
    thetas = params[:nCoils]
    radials = params[nCoils:2*nCoils]

    scan_geom = {"coils": {}}

    # Place coils using parametric positions
    inds, locs = place_points_pol_rad(
        nCoils,
        coil_center_cand1[:len(coil_center_cand1)//2,:],
        coil_center_cand2[:len(coil_center_cand2)//2,:],
        thetas, radials
    )

    # Create coil pairs (top + bottom)
    for i, loc in enumerate(locs):
        pts_top = np.array([
            [loc[0]-dx, loc[1]+dy],
            [loc[0]+dx, loc[1]+dy],
            [loc[0]+dx, loc[1]-dy],
            [loc[0]-dx, loc[1]-dy]
        ])
        pts_bot = pts_top * np.array([1, -1])

        scan_geom["coils"][f'F{i}A'] = {'pts': copy.deepcopy(pts_top), 'nturns': 1.0}
        scan_geom["coils"][f'F{i}B'] = {'pts': copy.deepcopy(pts_bot), 'nturns': 1.0}

    return scan_geom


def make_mesh(DIIID_geom, scan_geom, lim, plasma_dx=0.01, coil_dx=0.005, vac_dx=0.04, vv_dx=0.04, savename='temp_mesh.h5'):
    """
    Create TokaMaker mesh for free-boundary solve.

    Based on notebook cell 25.

    Parameters
    ----------
    DIIID_geom : dict
        Machine geometry
    scan_geom : dict
        Coil geometry from make_new_coils()
    lim : ndarray
        Limiter/plasma boundary
    plasma_dx, coil_dx, vac_dx, vv_dx : float
        Mesh resolutions
    savename : str
        HDF5 file to save mesh

    Returns
    -------
    coil_dict, cond_dict : dict
        TokaMaker coil and conductor dictionaries
    """
    gs_mesh = gs_Domain()
    gs_mesh.define_region('air', vac_dx, 'boundary')
    gs_mesh.define_region('plasma', plasma_dx, 'plasma')
    gs_mesh.define_region('vacuum', vv_dx, 'vacuum', allow_xpoints=True)
    gs_mesh.define_region('vv', vv_dx, 'conductor', eta=6e-07)

    # Define PF coils
    for key, coil in scan_geom["coils"].items():
        if key.startswith('ECOIL'):
            for i, subcoil in enumerate(coil):
                gs_mesh.define_region(f'{key}_{i}', coil_dx, 'coil', coil_set=key, nTurns=subcoil["nturns"])
        else:
            gs_mesh.define_region(key, coil_dx, 'coil', nTurns=coil["nturns"])

    # Define geometry - plasma and vacuum vessel
    gs_mesh.add_polygon(lim, 'plasma', parent_name='vacuum')
    gs_mesh.add_annulus(resize_polygon(lim, 0.01), 'vacuum', resize_polygon(lim, 0.05), 'vv')

    gs_mesh.add_enclosed([1.75,1.25],'vacuum')

    # Add coil polygons
    for key, coil in scan_geom["coils"].items():
        gs_mesh.add_polygon(coil["pts"], key, parent_name='air')

    # Build mesh
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    coil_dict = gs_mesh.get_coils()
    cond_dict = gs_mesh.get_conductors()

    # Save mesh
    save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, savename)

    return coil_dict, cond_dict


def set_regularization(mygs, solenoid_target, weight_curr=1.E-2):
    """
    Set regularization terms for TokaMaker.

    Based on notebook cell 52.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker object
    solenoid_target : float
        Target current for solenoids
    weight_curr : float
        Regularization weight for PF coils
    """
    regularization_terms = []
    for name, coil in mygs.coil_sets.items():
        if name.find('E0') >= 0 or name.find('E1') >= 0:
            # Fix solenoid currents
            regularization_terms.append(
                mygs.coil_reg_term({name: 1.0}, target=solenoid_target, weight=1.E4)
            )
        else:
            # Regularize PF coils to zero
            regularization_terms.append(
                mygs.coil_reg_term({name: 1.0}, target=0.0, weight=weight_curr)
            )

    mygs.set_coil_reg(reg_terms=regularization_terms)


# ============================================
# Main Analysis Functions
# ============================================

def analyze_free_boundary(coil_params, num_coils, eqdsk_file, DIIID_geom, myOFT,
                          weight_curr=3.E-2, equilibrium_name='g192185.02440', verbose=False):
    """
    Run free-boundary analysis for a single coil configuration.

    This function takes optimized coil positions from fixed-boundary optimization
    and runs a full free-boundary TokaMaker solve to compute actual currents and flux error.

    Parameters
    ----------
    coil_params : dict
        Dictionary with 'thetas' and 'radials' keys
    num_coils : int
        Number of coils
    eqdsk_file : str
        Path to EQDSK file
    DIIID_geom : dict
        Machine geometry
    myOFT : OFT_env
        OpenFUSION Toolkit environment
    weight_curr : float
        Regularization weight for coil currents
    equilibrium_name : str
        Name of equilibrium (determines fixed magnetic axis)
    verbose : bool
        Print progress messages

    Returns
    -------
    results : dict
        Dictionary containing:
        - total_current: Sum of absolute coil currents (excluding solenoids)
        - flux_error: RMS flux error at boundary
        - currents: Dictionary of individual coil currents
        - success: Boolean indicating if solve succeeded
    """
    try:
        # Extract parameters
        thetas = coil_params['thetas']
        radials = coil_params['radials']
        params = thetas + radials

        if verbose:
            print(f"  Analyzing {num_coils} coils configuration...")

        # Load eqdsk
        eqdsk = read_eqdsk(eqdsk_file)
        LCFS_contour = eqdsk['rzout'].copy()

        # Geometry
        with open('DIII_D_orig/DIIID_geom.json','r') as fid:
            DIIID_geom = json.load(fid)

        # Get the machine limiter
        lim0 = np.array(DIIID_geom['limiter'])
        # Setup coil position space (from notebook cells 11-12)
        lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
        lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
        coil_center_cand1 = resize_polygon(lim1, dx=0.1)
        lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
        coil_center_cand2 = resize_polygon(lim2, dx=0.15)

        # Create coil geometry
        scan_geom_DIIID = make_new_coils(params, num_coils, coil_center_cand1, coil_center_cand2, dx=0.03, dy=0.03)

        # Add solenoids (from notebook cell 27)
        scan_geom_DIIID['coils']['E0'] = {
            'pts': np.array([[0.658165, -1.57925], [0.716835, -1.57925], [0.716835, 1.57925], [0.658165, 1.57925]]),
            'nturns': 1.0
        }
        scan_geom_DIIID['coils']['E1'] = {
            'pts': np.array([[0.728165, -1.57925], [0.786835, -1.57925], [0.786835, 1.57925], [0.728165, 1.57925]]),
            'nturns': 1.0
        }

        # Create mesh
        mesh_savename = f'temp_mesh_{num_coils}coils.h5'
        coil_dict, cond_dict = make_mesh(DIIID_geom, scan_geom_DIIID, lim, savename=mesh_savename)

        # Setup TokaMaker

        mygs = TokaMaker(myOFT)
        mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_savename)
        mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
        mygs.settings.free_boundary = True

        F0 = eqdsk['rcentr'] * eqdsk['bcentr']
        mygs.setup(order=2, F0=F0)

        # Set targets (from notebook cells 43, 47, 49)
        # Equilibrium-specific magnetic axis
        mag_axis_dict = {
            'g192185.02440': np.array([1.77764093, -0.04014656]),
            'g174864.02500': np.array([1.74028708, 0.02673819]),
            'g173630.03000': np.array([1.79456678, -0.04926767])
        }
        fixed_mag_axis = mag_axis_dict.get(equilibrium_name, np.array([1.77764093, -0.04014656]))

        Ip_target = eqdsk['ip']
        R0_target = fixed_mag_axis[0].item()
        Z0 = fixed_mag_axis[1].item()
        mygs.set_targets(Ip=Ip_target, R0=R0_target, V0=Z0)

        coil_bounds = {key: [-6.E8, 6.E8] for key in mygs.coil_sets}
        mygs.set_coil_bounds(coil_bounds)

        # X-point (equilibrium-specific, from notebook cell 47)
        xpoint_index_dict = {
            'g192185.02440': 55,
            'g174864.02500': 68,
            'g173630.03000': 69
        }
        index = xpoint_index_dict.get(equilibrium_name, 55)

        isoflux_pts = eqdsk['rzout'].copy()
        weights = np.ones(len(isoflux_pts[:,0]))
        weights[index] = 1e4
        mygs.set_isoflux(isoflux_pts, weights=weights)

        # Solenoid currents (from notebook cell 53)
        target_solenoid1 = -0.977888676757812E6
        target_solenoid2 = -0.962711173828125E6

        # Set regularization and solve
        set_regularization(mygs, target_solenoid1, weight_curr)

        # Initialize and solve (from notebook cell 56)
        # Equilibrium-specific initialization
        init_params_dict = {
            'g192185.02440': {'r0': 1.8, 'z0': -0.040, 'a': 0.45, 'kappa': 1.547, 'delta': -0.288},
            'g174864.02500': {'r0': 1.8, 'z0': -0.040, 'a': 0.45, 'kappa': 1.679, 'delta': 0.318},
            'g173630.03000': {'r0': 1.8, 'z0': -0.040, 'a': 0.45, 'kappa': 1.821, 'delta': 0.467}
        }
        init_params = init_params_dict.get(equilibrium_name, {'r0': 1.8, 'z0': -0.040, 'a': 0.45, 'kappa': 1.547, 'delta': -0.288})

        mygs.init_psi(**init_params)
        err_flag = mygs.solve()

        if err_flag != 0:
            if verbose:
                print(f"    Warning: Solver returned error flag {err_flag}")

        # Extract currents
        currents, currents_reg = mygs.get_coil_currents()

        # Compute total current (excluding solenoids)
        excluded_keys = {'E0', 'E1'}
        total_current = sum(abs(val) for key, val in currents.items() if key not in excluded_keys)

        # Clean up temporary mesh file
        if os.path.exists(mesh_savename):
            os.remove(mesh_savename)

        if verbose:
            print(f"    Total current: {total_current:.2e} A")

        return {
            'total_current': total_current,
            'currents': currents,
            'success': True
        }

    except Exception as e:
        if verbose:
            print(f"    Error during analysis: {e}")
        return {
            'total_current': np.nan,
            'currents': {},
            'success': False,
            'error': str(e)
        }


def scan(results_dir='examples/comparisons/closed_boundary_DIIID/test_general/',
                      eqdsk_file='examples/data/eqdsk/g192185.02440',
                      DIIID_geom_file='DIII_D_orig/DIIID_geom.json',
                      myOFT=None,
                      methods=None,
                      verbose=True):
    """
    Scan all configurations in test_general and run free-boundary analysis.

    Parameters
    ----------
    results_dir : str
        Directory containing optimization results
    eqdsk_file : str
        Path to EQDSK file
    DIIID_geom_file : str
        Path to DIII-D geometry JSON file
    myOFT : OFT_env, optional
        OpenFUSION Toolkit environment (created if None)
    methods : list of str, optional
        Methods to analyze (default: ['Multi-start L-BFGS', 'Bayesian'])
    verbose : bool
        Print progress messages

    Returns
    -------
    results_data : dict
        Organized results grouped by (num_coils, lambda, method)
    """
    if methods is None:
        methods = ['Multi-start L-BFGS', 'Bayesian']

    if myOFT is None:
        myOFT = OFT_env(nthreads=2)

    # Load DIII-D geometry
    with open(DIIID_geom_file, 'r') as fid:
        DIIID_geom = json.load(fid)

    results_data = {}
    results_path = Path(results_dir)

    # Scan all JSON files
    json_files = sorted(results_path.rglob('*.json'))
    if verbose:
        print(f"Found {len(json_files)} result files to process")

    for json_file in json_files:
        try:
            # Load fixed-boundary optimization results
            with open(json_file, 'r') as f:
                opt_results = json.load(f)

            # Extract optimization settings
            opt_settings = opt_results['optimization_settings']
            num_coils = opt_settings['num_coils']
            lambda_val = opt_settings['reg_in']

            if verbose:
                print(f"\nProcessing: num_coils={num_coils}, lambda={lambda_val:.2e}")

            # Analyze each method
            for method_name in methods:
                if method_name not in opt_results['methods']:
                    continue

                method_data = opt_results['methods'][method_name]
                coil_params = method_data['parameters']

                # Extract fixed-boundary flux error from JSON (if available)
                fixed_boundary_flux_err = method_data.get('flux_err', None)

                if verbose:
                    print(f"  Method: {method_name}")
                    if fixed_boundary_flux_err is not None:
                        print(f"    Fixed-boundary flux error: {fixed_boundary_flux_err:.2e}")

                # Run free-boundary analysis
                fb_results = analyze_free_boundary(
                    coil_params, num_coils, eqdsk_file, DIIID_geom, myOFT,
                    weight_curr=3.E-2, equilibrium_name='g192185.02440', verbose=verbose
                )

                # Store results
                key = (num_coils, lambda_val, method_name)
                results_data[key] = {
                    'num_coils': num_coils,
                    'lambda': lambda_val,
                    'method': method_name,
                    'coil_params': coil_params,
                    'total_current': fb_results['total_current'],
                    'flux_error_fixed_boundary': fixed_boundary_flux_err,
                    'currents': fb_results['currents'],
                    'success': fb_results['success']
                }

        except Exception as e:
            if verbose:
                print(f"  Error processing {json_file}: {e}")
            continue

    return results_data


def plot_coil_currents_comparison(results_data, num_coils_filter, method='Multi-start L-BFGS',
                                   save_path=None, figsize=(10, 6)):
    """
    Plot individual coil currents for different lambda values.

    Creates a plot showing current in each coil (F0A, F1A, ..., F0B, F1B, ...)
    with different colored lines for each lambda value.

    Parameters
    ----------
    results_data : dict
        Results from scan()
    num_coils_filter : int
        Number of coils to plot
    method : str
        Method name to plot
    save_path : str, optional
        Path to save plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Created figure
    """
    # Filter data by num_coils and method
    filtered_data = {k: v for k, v in results_data.items()
                    if v['num_coils'] == num_coils_filter and v['method'] == method and v['success']}

    if len(filtered_data) == 0:
        print(f"No data found for num_coils={num_coils_filter}, method={method}")
        return None

    # Get unique lambda values and sort them
    lambda_values = sorted(set(v['lambda'] for v in filtered_data.values()))

    # Define color map for different lambda values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambda_values)))

    # Get coil names (excluding solenoids)
    sample_currents = next(iter(filtered_data.values()))['currents']
    excluded_keys = {'E0', 'E1'}
    coil_names = sorted([key for key in sample_currents.keys() if key not in excluded_keys])

    # Sort coil names: top coils (A) first, then bottom coils (B)
    top_coils = sorted([c for c in coil_names if c.endswith('A')])
    bottom_coils = sorted([c for c in coil_names if c.endswith('B')])
    ordered_coil_names = top_coils + bottom_coils

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each lambda value
    for i, lambda_val in enumerate(lambda_values):
        # Find data for this lambda
        data_for_lambda = [v for v in filtered_data.values() if v['lambda'] == lambda_val]

        if len(data_for_lambda) == 0:
            continue

        # Get currents for this lambda
        currents = data_for_lambda[0]['currents']
        current_values = [currents.get(coil, 0.0) for coil in ordered_coil_names]

        # Convert to millions of Amperes for readability
        current_values_ma = [c / 1e6 for c in current_values]

        # Plot
        ax.plot(range(len(ordered_coil_names)), current_values_ma,
                'o-', color=colors[i], linewidth=2, markersize=8,
                label=f'λ = {lambda_val:.0e}')

    # Customize plot
    ax.set_xticks(range(len(ordered_coil_names)))
    ax.set_xticklabels(ordered_coil_names, fontsize=12, fontweight='bold')
    ax.set_xlabel('Coil', fontsize=14, fontweight='bold')
    ax.set_ylabel('Current [MA]', fontsize=14, fontweight='bold')
    ax.set_title(f'Coil Currents vs Lambda\n({num_coils_filter} coils, {method})',
                fontsize=12, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(labelsize=11)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


def plot_current_vs_lambda(results_data, num_coils_filter, method='Multi-start L-BFGS',
                           save_path=None, figsize=(8, 6)):
    """
    Create dual-axis plot: Total Current vs Flux Error vs Lambda.

    Parameters
    ----------
    results_data : dict
        Results from scan()
    num_coils_filter : int
        Number of coils to plot
    method : str
        Method name to plot
    save_path : str, optional
        Path to save plot
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
        Created figure
    """
    # Filter data
    filtered_data = {k: v for k, v in results_data.items()
                    if v['num_coils'] == num_coils_filter and v['method'] == method and v['success']}

    if len(filtered_data) == 0:
        print(f"No data found for num_coils={num_coils_filter}, method={method}")
        return None

    # Extract data
    lambda_values = np.array([v['lambda'] for v in filtered_data.values()])
    total_currents = np.array([v['total_current'] for v in filtered_data.values()])
    flux_errors = np.array([v['flux_error_fixed_boundary'] for v in filtered_data.values()])

    # Sort by lambda
    sorted_indices = np.argsort(lambda_values)
    lambda_values = lambda_values[sorted_indices]
    total_currents = total_currents[sorted_indices]
    flux_errors = flux_errors[sorted_indices]

    # Create plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Left y-axis: Total current (blue)
    color_current = 'tab:blue'
    ax1.set_xlabel(r'$\lambda$', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'$I_T$ [A]', color=color_current, fontsize=14, fontweight='bold')
    line1 = ax1.plot(lambda_values, total_currents, 'o-', color=color_current,
                     linewidth=2, markersize=8, label='Total Current')
    ax1.tick_params(axis='y', labelcolor=color_current, labelsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Flux error (red)
    ax2 = ax1.twinx()
    color_error = 'tab:red'
    ax2.set_ylabel(r'$\chi^2$', color=color_error, fontsize=14, fontweight='bold')
    line2 = ax2.plot(lambda_values, flux_errors, 's--', color=color_error,
                     linewidth=2, markersize=8, label='Flux Error')
    ax2.tick_params(axis='y', labelcolor=color_error, labelsize=12)

    # Title
    plt.title(f'Total Current and Flux Error vs Regularization\n({num_coils_filter} coils, {method})',
              fontsize=12, fontweight='bold', pad=20)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("FREE-BOUNDARY ANALYSIS FOR PF COIL OPTIMIZATION")
    print("=" * 70)

    # Initialize OFT environment
    myOFT = OFT_env(nthreads=2)

    # Scan all configurations and run free-boundary analysis
    results_data = scan(
        results_dir='examples/comparisons/closed_boundary_DIIID/test_flux_err/',
        eqdsk_file='examples/data/eqdsk/g192185.02440',
        DIIID_geom_file='DIII_D_orig/DIIID_geom.json',
        myOFT=myOFT,
        methods=['Multi-start L-BFGS', 'Bayesian'],
        verbose=True
    )

    # Save results to JSON
    output_dir = Path('examples/comparisons/free_boundary_analysis/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"free_boundary_results.json"
    # Convert results to JSON-serializable format
    json_results = {}
    for key, val in results_data.items():
        key_str = f"{val['num_coils']}_coils_lambda_{val['lambda']:.2e}_{val['method']}"
        json_results[key_str] = {
            'num_coils': val['num_coils'],
            'lambda': val['lambda'],
            'method': val['method'],
            'total_current': float(val['total_current']),
            'flux_error_fixed_boundary': float(val['flux_error_fixed_boundary']) if val['flux_error_fixed_boundary'] is not None else None,
            'coil_params': val['coil_params'],
            'success': val['success']
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Generate plots for each num_coils value
    unique_num_coils = sorted(set(v['num_coils'] for v in results_data.values()))

    for num_coils in unique_num_coils:
        for method in ['Multi-start L-BFGS', 'Bayesian']:
            # Dual-axis plot: Total Current vs Flux Error vs Lambda
            plot_path = output_dir / f'current_vs_lambda_{num_coils}coils_{method.replace(" ", "_")}.png'
            fig = plot_current_vs_lambda(results_data, num_coils, method, save_path=str(plot_path))
            if fig is not None:
                plt.close(fig)

            # Coil currents comparison plot
            coil_plot_path = output_dir / f'coil_currents_{num_coils}coils_{method.replace(" ", "_")}.png'
            fig = plot_coil_currents_comparison(results_data, num_coils, method, save_path=str(coil_plot_path))
            if fig is not None:
                plt.close(fig)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)