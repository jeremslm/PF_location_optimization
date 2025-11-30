#!/usr/bin/env python3
"""
Test script for OFT_pf_coil_optimize module.

Tests L-BFGS optimization with DIII-D geometry.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setup OFT paths
home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path

if os.getenv("OFT_ROOTPATH") is not None:
    sys.path.append(os.path.join(os.getenv("OFT_ROOTPATH"), 'python'))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

from OFT_pf_coil_opt_fct import (
    CoilPositionSpace,
    pf_coil_optimize
)
from helper_fct import update_boundary, resize_polygon


def test_lbfgs_optimization():
    """Test L-BFGS optimization with DIII-D geometry."""
    print("=" * 60)
    print("Testing L-BFGS Optimization")
    print("=" * 60)

    # Load EQDSK
    eqdsk = read_eqdsk('g192185.02440')
    LCFS_contour = eqdsk['rzout'].copy()

    # Create mesh
    mesh_dx = 0.015
    gs_mesh = gs_Domain()
    gs_mesh.define_region('plasma', mesh_dx, 'plasma')
    gs_mesh.add_polygon(LCFS_contour, 'plasma')
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()

    # Setup TokaMaker
    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False

    F0 = eqdsk['rcentr'] * eqdsk['bcentr']
    mygs.setup(order=2, F0=F0)

    # Set targets and solve
    Ip_target = eqdsk['ip']
    pres_target = eqdsk['pres'][0]
    mygs.set_targets(Ip=Ip_target, pax=pres_target)

    err_flag = mygs.init_psi()
    err_flag = mygs.solve()

    # Create coil position space
    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_inner = resize_polygon(lim1, dx=0.1)

    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_outer = resize_polygon(lim2, dx=0.15)

    coil_space = CoilPositionSpace(
        inner_boundary=coil_center_inner,
        outer_boundary=coil_center_outer,
        method='coords'
    )

    # Run L-BFGS optimization
    print("\nRunning L-BFGS optimization...")
    result = pf_coil_optimize(
        tokamaker_solver=mygs,
        reg_current=1e-3,
        reg_distance=1e-5,
        min_coil_distance=5,
        coil_space=coil_space,
        n_coils=5,
        method='lbfgs',
        verbose=True
    )

    print("\n" + "=" * 60)
    print("L-BFGS Result:")
    print(result)
    print("Optimized angles:", result.angles)
    print("Optimized radials:", result.radials)

    return result


def test_bayesian_optimization():
    """Test Bayesian optimization with DIII-D geometry."""
    print("\n" + "=" * 60)
    print("Testing Bayesian Optimization")
    print("=" * 60)

    # Load EQDSK
    eqdsk = read_eqdsk('g192185.02440')
    LCFS_contour = eqdsk['rzout'].copy()

    # Create mesh
    mesh_dx = 0.015
    gs_mesh = gs_Domain()
    gs_mesh.define_region('plasma', mesh_dx, 'plasma')
    gs_mesh.add_polygon(LCFS_contour, 'plasma')
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()

    # Setup TokaMaker
    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)
    mygs.setup_mesh(mesh_pts, mesh_lc)
    mygs.settings.free_boundary = False

    F0 = eqdsk['rcentr'] * eqdsk['bcentr']
    mygs.setup(order=2, F0=F0)

    # Set targets and solve
    Ip_target = eqdsk['ip']
    pres_target = eqdsk['pres'][0]
    mygs.set_targets(Ip=Ip_target, pax=pres_target)

    err_flag = mygs.init_psi()
    err_flag = mygs.solve()

    # Create coil position space
    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    coil_center_inner = resize_polygon(lim1, dx=0.1)

    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    coil_center_outer = resize_polygon(lim2, dx=0.15)

    coil_space = CoilPositionSpace(
        inner_boundary=coil_center_inner,
        outer_boundary=coil_center_outer,
        method='coords'
    )

    # Run Bayesian optimization (smaller for testing)
    print("\nRunning Bayesian optimization...")
    result = pf_coil_optimize(
        tokamaker_solver=mygs,
        reg_current=1e-3,
        reg_distance=1e-5,
        min_coil_distance=5,
        coil_space=coil_space,
        n_coils=5,
        method='bayesian',
        n_calls=20,  # Small for testing
        n_initial_points=10,
        acq_func='EI',
        local_optimize=True,
        n_local_refine=3,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Bayesian Result:")
    print(result)
    print("Optimized angles:", result.angles)
    print("Optimized radials:", result.radials)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test OFT_pf_coil_optimize')
    parser.add_argument('--method', choices=['lbfgs', 'bayesian', 'all'],
                        default='lbfgs', help='Optimization method to test')
    args = parser.parse_args()

    if args.method == 'lbfgs' or args.method == 'all':
        result_lbfgs = test_lbfgs_optimization()

    if args.method == 'bayesian' or args.method == 'all':
        result_bayesian = test_bayesian_optimization()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
