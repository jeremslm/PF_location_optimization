# PF Coil Optimization Package - Implementation Summary

## ✅ Sprint 1 Complete!

We've successfully created a generalized `pf_coil_optimize` package with three methods for defining coil position spaces.

---

## Files Created

1. **`__init__.py`** - Package initialization with exports
2. **`pf_coil_optimize.py`** - Main module with all classes and functions

---

## CoilPositionSpace Methods Implemented

### 1. **method='coords'** (Direct Arrays)
```python
inner_curve = np.array([[R1, Z1], [R2, Z2], ...])
outer_curve = np.array([[R1, Z1], [R2, Z2], ...])
space = CoilPositionSpace(inner_curve, outer_curve, method='coords')
```

### 2. **method='parametric'** (Plasma Shape Parameters)
```python
space = CoilPositionSpace(
    inner_boundary={'r0': 1.69, 'z0': 0, 'a': 0.67, 'kappa': 2,
                   'delta': 0.8, 'squar': 0.15, 'npts': 1700, 'offset': 0.1},
    outer_boundary={'r0': 1.94, 'z0': 0, 'a': 0.95, 'kappa': 1.55,
                   'delta': 0.8, 'squar': 0.15, 'npts': 1700, 'offset': 0.15},
    method='parametric'
)
```
**How it works:**
- Calls `update_boundary()` to create two different plasma shapes
- Calls `resize_polygon()` to offset each shape by specified distance
- Exactly matches your notebook workflow

### 3. **method='function'** (User-Defined Functions)
```python
def inner_func(theta):
    # theta in degrees (0-180)
    R = 1.0 + 0.5 * np.cos(np.radians(theta))
    Z = 0.5 * np.sin(np.radians(theta))
    return R, Z

def outer_func(theta):
    R = 1.5 + 0.7 * np.cos(np.radians(theta))
    Z = 0.7 * np.sin(np.radians(theta))
    return R, Z

space = CoilPositionSpace(inner_func, outer_func, method='function')
```
**How it works:**
- Accepts callable functions: `func(theta_degrees) -> (R, Z)`
- Pre-samples at 1700 points from 0-180 degrees
- Stores as curves for efficient interpolation

### 4. **method='single_curve'** (From Limiter/LCFS)
```python
# From EQDSK limiter
eqdsk = read_eqdsk('g192185.02440')
lim = eqdsk['rzlim']
space = CoilPositionSpace(
    single_curve=lim,
    dx_inner=0.1,      # Inner offset (meters)
    dx_outer=0.3,      # Outer offset (meters)
    smoothen_window=5, # Smoothing window (optional, default=5)
    method='single_curve'
)
```
**How it works:**
- Takes single reference curve (LCFS, limiter, or wall geometry)
- Applies `smoothen()` using uniform filter to avoid offset artifacts
- Applies `resize_polygon()` twice with different offsets
- `dx_inner` creates inner boundary (closer to plasma)
- `dx_outer` creates outer boundary (farther from plasma)
- Both offsets must be positive, with `dx_outer > dx_inner`

**Validation:**
- Checks `dx_outer > dx_inner > 0`
- Validates smoothing window: `1 ≤ window ≤ len(curve)`

---

## PerCoilPositionSpace - Per-Coil Search Spaces

`PerCoilPositionSpace` allows you to define **different search spaces for each coil**. This is useful when:
- Different coils have different geometric constraints
- Some coils are limited to specific regions (e.g., divertor vs midplane)
- You want to mix global and custom spaces

### Basic Usage

```python
# Define a global space for most coils
global_space = CoilPositionSpace(
    single_curve=limiter,
    dx_inner=0.1,
    dx_outer=0.3,
    method='single_curve'
)

# Define custom space for one coil (e.g., limited to upper region)
upper_space = CoilPositionSpace(
    inner_boundary=upper_inner,
    outer_boundary=upper_outer,
    method='coords',
    angular_bounds=(120, 180),  # Only upper angles
    radial_bounds=(0.3, 0.7)     # Restricted radial range
)

# Create per-coil specification
per_coil_space = PerCoilPositionSpace(
    coil_specs=[
        'global',      # Coil 0: use global space
        'global',      # Coil 1: use global space
        upper_space,   # Coil 2: use custom upper space
        'global',      # Coil 3: use global space
        'global'       # Coil 4: use global space
    ],
    global_space=global_space
)

# Use in optimization
result = pf_coil_optimize(
    mygs,
    per_coil_space,  # Pass PerCoilPositionSpace instead of CoilPositionSpace
    n_coils=5
)
```

### How It Works

- `coil_specs`: List of length `n_coils`, one entry per coil
  - `'global'`: Use the `global_space` for this coil
  - `CoilPositionSpace` object: Use custom space for this coil
- `global_space`: Fallback space when spec is `'global'`

### Internal Methods

```python
# Get bounds for specific coil
theta_bounds, radial_bounds = per_coil_space.get_bounds_for_coil(2)

# Get (R, Z) position for specific coil
R, Z = per_coil_space.interpolate_for_coil(2, theta=135, radial=0.5)
```

---

## Usage Examples

### Example 1: Parametric (Replicating Your Notebook)
```python
from pf_coil_optimize import pf_coil_optimize, CoilPositionSpace

# Define space exactly like your notebook
space = CoilPositionSpace(
    inner_boundary={'r0': 1.69, 'a': 0.67, 'kappa': 2, 'delta': 0.8,
                   'squar': 0.15, 'offset': 0.1},
    outer_boundary={'r0': 1.94, 'a': 0.95, 'kappa': 1.55, 'delta': 0.8,
                   'squar': 0.15, 'offset': 0.15},
    method='parametric'
)

# Run optimization
result = pf_coil_optimize(
    mygs,  # Your TokaMaker solver
    space,
    n_coils=5,
    method='lbfgs',
    reg_current=1e-7,
    reg_distance=1e-5,
    min_coil_distance=5
)

print(result)
print(f"Optimal angles: {result.angles}")
print(f"Optimal currents: {result.currents}")
```

### Example 2: From LCFS
```python
# Extract LCFS from EQDSK
eqdsk = read_eqdsk('g192185.02440')
space = CoilPositionSpace.from_single_curve(eqdsk['rzout'], 0.1, 0.3)

result = pf_coil_optimize(mygs, space, n_coils=5)
```

### Example 3: Custom Functions
```python
def my_inner_boundary(theta):
    # Custom shape for inner boundary
    return R, Z

def my_outer_boundary(theta):
    # Custom shape for outer boundary
    return R, Z

space = CoilPositionSpace(my_inner_boundary, my_outer_boundary, method='function')
result = pf_coil_optimize(mygs, space, n_coils=5)
```

---

## What's Next?

**Not yet implemented (for Sprint 2):**
- `method='multi_start'` - Latin Hypercube Sampling
- `method='bayesian'` - Bayesian optimization
- `PerCoilPositionSpace` - Per-coil boundaries with dict specifications
- Plotting in `OptimizationResult`
- Save/load functionality

---

## Key Design Decisions

1. **Composition over Inheritance** - `PerCoilPositionSpace` contains `CoilPositionSpace` objects
2. **Factory Methods** - Clean API with `from_single_curve()` instead of complex constructor
3. **Flexible Parametric** - Uses `.get()` for optional parameters (z0, npts)
4. **Pre-computation** - Function-based boundaries sampled once at initialization
5. **Reuses Helpers** - Leverages existing `update_boundary()` and `resize_polygon()`

---

## Testing Checklist

- [ ] Test `method='curve'` with direct arrays
- [ ] Test `method='parametric'` matching notebook (cell 72d94a7f)
- [ ] Test `method='function'` with custom callables
- [ ] Test `from_single_curve()` with EQDSK LCFS
- [ ] Verify optimization runs end-to-end
- [ ] Compare results with original notebook
