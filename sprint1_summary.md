# PF Coil Optimization Package - Implementation Summary

## ✅ Sprint 1 Complete!

We've successfully created a generalized `pf_coil_optimize` package with three methods for defining coil position spaces.

---

## Files Created

1. **`__init__.py`** - Package initialization with exports
2. **`pf_coil_optimize.py`** - Main module with all classes and functions

---

## CoilPositionSpace Methods Implemented

### 1. **method='curve'** (Direct Arrays)
```python
inner_curve = np.array([[R1, Z1], [R2, Z2], ...])
outer_curve = np.array([[R1, Z1], [R2, Z2], ...])
space = CoilPositionSpace(inner_curve, outer_curve, method='curve')
```

### 2. **method='parametric'** (Like Your Notebook!)
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
- Exactly matches your notebook workflow (cell 72d94a7f)

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

### 4. **from_single_curve()** (Class Method)
```python
eqdsk = read_eqdsk('g192185.02440')
lcfs = eqdsk['rzout']
space = CoilPositionSpace.from_single_curve(lcfs, inner_offset=0.1, outer_offset=0.3)
```
**How it works:**
- Takes single reference curve (LCFS or limiter)
- Applies `resize_polygon()` twice with different offsets
- Returns `CoilPositionSpace` with method='curve'

---

##Usage Examples

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
