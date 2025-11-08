"""
Quick Import Test

Tests that the package can be imported and basic objects can be created.
Run this first before the full test:
    python test_imports.py
"""

import numpy as np

print("Testing imports...")

# Test 1: Import package
try:
    from OFT_pf_coil_optimize import (
        pf_coil_optimize,
        CoilPositionSpace,
        PerCoilPositionSpace,
        OptimizationResult
    )
    print("✓ Package imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Create simple CoilPositionSpace with method='curve'
try:
    inner = np.array([[1.0, -0.5], [1.2, 0], [1.0, 0.5]])
    outer = np.array([[1.5, -0.7], [1.8, 0], [1.5, 0.7]])
    space = CoilPositionSpace(inner, outer, method='curve')
    print(f"✓ CoilPositionSpace created (method='curve')")
except Exception as e:
    print(f"✗ CoilPositionSpace creation failed: {e}")
    exit(1)

# Test 3: Test interpolation
try:
    R, Z = space.interpolate(90, 0.5)
    print(f"✓ Interpolation works: θ=90°, ρ=0.5 → (R={R:.3f}, Z={Z:.3f})")
except Exception as e:
    print(f"✗ Interpolation failed: {e}")
    exit(1)

# Test 4: Test bounds
try:
    theta_bounds, radial_bounds = space.get_bounds()
    print(f"✓ get_bounds() works: angular={theta_bounds}, radial={radial_bounds}")
except Exception as e:
    print(f"✗ get_bounds() failed: {e}")
    exit(1)

# Test 5: Test custom bounds
try:
    space.set_bounds(angular_bounds=(10, 170), radial_bounds=(0.1, 0.9))
    theta_bounds, radial_bounds = space.get_bounds()
    assert theta_bounds == (10, 170)
    assert radial_bounds == (0.1, 0.9)
    print(f"✓ set_bounds() works: angular={theta_bounds}, radial={radial_bounds}")
except Exception as e:
    print(f"✗ set_bounds() failed: {e}")
    exit(1)

# Test 6: Test method='function'
try:
    def inner_func(theta):
        return 1.0 + 0.3 * np.cos(np.radians(theta)), 0.3 * np.sin(np.radians(theta))

    def outer_func(theta):
        return 1.5 + 0.5 * np.cos(np.radians(theta)), 0.5 * np.sin(np.radians(theta))

    space_func = CoilPositionSpace(inner_func, outer_func, method='function')
    print(f"✓ CoilPositionSpace created (method='function')")
    print(f"  Sampled curves: inner={space_func.inner_curve.shape}, outer={space_func.outer_curve.shape}")
except Exception as e:
    print(f"✗ method='function' failed: {e}")
    exit(1)

# Test 7: Test method='parametric'
try:
    space_param = CoilPositionSpace(
        inner_boundary={'r0': 1.69, 'a': 0.67, 'kappa': 2, 'delta': 0.8, 'squar': 0.15, 'offset': 0.1},
        outer_boundary={'r0': 1.94, 'a': 0.95, 'kappa': 1.55, 'delta': 0.8, 'squar': 0.15, 'offset': 0.15},
        method='parametric'
    )
    print(f"✓ CoilPositionSpace created (method='parametric')")
    print(f"  Generated curves: inner={space_param.inner_curve.shape}, outer={space_param.outer_curve.shape}")
except Exception as e:
    print(f"✗ method='parametric' failed: {e}")
    exit(1)

# Test 8: Test PerCoilPositionSpace
try:
    global_space = CoilPositionSpace(inner, outer, method='curve')
    per_coil = PerCoilPositionSpace(['global', 'global', 'global'], global_space=global_space)
    print(f"✓ PerCoilPositionSpace created")

    theta_b, radial_b = per_coil.get_bounds_for_coil(0)
    print(f"  Coil 0 bounds: angular={theta_b}, radial={radial_b}")
except Exception as e:
    print(f"✗ PerCoilPositionSpace failed: {e}")
    exit(1)

print("\n" + "="*60)
print("ALL IMPORT TESTS PASSED! ✅")
print("="*60)
print("\nYou can now run the full test:")
print("    python test_sprint1.py")