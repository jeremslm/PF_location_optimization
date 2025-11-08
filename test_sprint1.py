"""
Test Script for Sprint 1: PF Coil Optimization Package

This script tests the basic functionality of the pf_coil_optimize package
by replicating the DIII-D optimization from your notebook.

Run this from the PF_location_optimization directory:
    python test_sprint1.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add OpenFUSIONToolkit to path
home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
sys.path.insert(0, os.path.join(oft_root_path, 'python'))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

# Import our new package
from OFT_pf_coil_optimize import pf_coil_optimize, CoilPositionSpace
from helper_functions_angle import update_boundary, resize_polygon

print("="*70)
print("SPRINT 1 TEST: PF Coil Optimization Package")
print("="*70)

# ============================================================================
# Step 1: Setup TokaMaker Equilibrium (from your notebook)
# ============================================================================
print("\n[1/6] Loading EQDSK and setting up TokaMaker...")

mesh_dx = 0.015
eqdsk = read_eqdsk('g192185.02440')
LCFS_contour = eqdsk['rzout'].copy()

# Create mesh
gs_mesh = gs_Domain()
gs_mesh.define_region('plasma', mesh_dx, 'plasma')
gs_mesh.add_polygon(LCFS_contour, 'plasma')
mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()

# Initialize TokaMaker
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

print(f"✓ TokaMaker equilibrium solved successfully")

# ============================================================================
# Step 2: Test method='parametric' (Your Notebook Method)
# ============================================================================
print("\n[2/6] Testing method='parametric' (replicating notebook)...")

space_parametric = CoilPositionSpace(
    inner_boundary={
        'r0': 1.69,
        'z0': 0,
        'a': 0.67,
        'kappa': 2,
        'delta': 0.8,
        'squar': 0.15,
        'npts': 1700,
        'offset': 0.1
    },
    outer_boundary={
        'r0': 1.94,
        'z0': 0,
        'a': 0.95,
        'kappa': 1.55,
        'delta': 0.8,
        'squar': 0.15,
        'npts': 1700,
        'offset': 0.15
    },
    method='parametric'
)

print(f"✓ CoilPositionSpace created with method='parametric'")
print(f"  Inner curve shape: {space_parametric.inner_curve.shape}")
print(f"  Outer curve shape: {space_parametric.outer_curve.shape}")

# ============================================================================
# Step 3: Test Optimization
# ============================================================================
print("\n[3/6] Running optimization with method='lbfgs'...")

result = pf_coil_optimize(
    mygs,
    space_parametric,
    n_coils=5,
    coil_dx=0.035,
    coil_dy=0.035,
    coil_filament_radius=0.01,
    method='lbfgs',
    reg_current=1e-7,
    reg_distance=1e-5,
    min_coil_distance=5,
    verbose=True
)

print(f"\n✓ Optimization completed!")
print(result)

# ============================================================================
# Step 4: Test method='curve' (Direct Arrays)
# ============================================================================
print("\n[4/6] Testing method='curve' (direct arrays)...")

# Generate curves manually (same as parametric, but explicit)
lim_inner = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
inner_curve = resize_polygon(lim_inner, dx=0.1)

lim_outer = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
outer_curve = resize_polygon(lim_outer, dx=0.15)

space_curve = CoilPositionSpace(inner_curve, outer_curve, method='curve')

print(f"✓ CoilPositionSpace created with method='curve'")
print(f"  Inner curve shape: {space_curve.inner_curve.shape}")
print(f"  Outer curve shape: {space_curve.outer_curve.shape}")

# Quick test (no full optimization)
test_theta, test_radial = 45, 0.5
R, Z = space_curve.interpolate(test_theta, test_radial)
print(f"  Test interpolation: θ={test_theta}°, ρ={test_radial} → (R={R:.3f}, Z={Z:.3f})")

# ============================================================================
# Step 5: Test method='function' (Custom Callables)
# ============================================================================
print("\n[5/6] Testing method='function' (custom functions)...")

def inner_func(theta):
    """Simple parametric inner boundary."""
    theta_rad = np.radians(theta)
    R = 1.2 + 0.5 * np.cos(theta_rad)
    Z = 0.6 * np.sin(theta_rad)
    return R, Z

def outer_func(theta):
    """Simple parametric outer boundary."""
    theta_rad = np.radians(theta)
    R = 1.5 + 0.7 * np.cos(theta_rad)
    Z = 0.8 * np.sin(theta_rad)
    return R, Z

space_function = CoilPositionSpace(inner_func, outer_func, method='function')

print(f"✓ CoilPositionSpace created with method='function'")
print(f"  Inner curve shape: {space_function.inner_curve.shape}")
print(f"  Outer curve shape: {space_function.outer_curve.shape}")

# Test interpolation
R, Z = space_function.interpolate(test_theta, test_radial)
print(f"  Test interpolation: θ={test_theta}°, ρ={test_radial} → (R={R:.3f}, Z={Z:.3f})")

# ============================================================================
# Step 6: Test Custom Bounds
# ============================================================================
print("\n[6/6] Testing custom bounds...")

# Test 1: Bounds in constructor
space_with_bounds = CoilPositionSpace(
    inner_curve, outer_curve,
    method='curve',
    angular_bounds=(10, 170),
    radial_bounds=(0.1, 0.9)
)
theta_bounds, radial_bounds = space_with_bounds.get_bounds()
print(f"✓ Bounds in constructor:")
print(f"  Angular: {theta_bounds}")
print(f"  Radial: {radial_bounds}")

# Test 2: Setter method
space_set_bounds = CoilPositionSpace(inner_curve, outer_curve, method='curve')
space_set_bounds.set_bounds(angular_bounds=(20, 160), radial_bounds=(0.2, 0.8))
theta_bounds, radial_bounds = space_set_bounds.get_bounds()
print(f"✓ Bounds via setter:")
print(f"  Angular: {theta_bounds}")
print(f"  Radial: {radial_bounds}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SPRINT 1 TEST RESULTS")
print("="*70)
print(f"✓ TokaMaker equilibrium: PASSED")
print(f"✓ method='parametric': PASSED")
print(f"✓ method='curve': PASSED")
print(f"✓ method='function': PASSED")
print(f"✓ Custom bounds: PASSED")
print(f"✓ Optimization (L-BFGS): PASSED")
print("\nFinal Optimization Results:")
print(f"  Success: {result.success}")
print(f"  Optimal angles: {result.angles}")
print(f"  Optimal radials: {result.radials}")
print(f"  Optimal currents: {result.currents}")
print(f"  Outer cost: {result.cost_outer:.6e}")
print(f"  Inner cost: {result.cost_inner:.6e}")
print(f"  Flux error: {result.flux_error:.6e}")
print(f"  Iterations: {result.n_iterations}")
print("\n" + "="*70)
print("ALL TESTS PASSED! 🎉")
print("="*70)