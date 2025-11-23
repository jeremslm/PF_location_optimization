# PF Coil Location Optimization

A Python package for optimizing poloidal field (PF) coil locations in tokamak equilibria using the OpenFUSION Toolkit's TokaMaker solver.

## Overview

This package provides a flexible framework for finding optimal PF coil placements that best reproduce a target fixed-boundary plasma equilibrium. It supports multiple optimization methods, customizable coil position search spaces, and comprehensive analysis tools.

### Key Features

- **Flexible Search Spaces**: Define coil position spaces using coordinates, parametric shapes, functions, or single curves
- **Multiple Optimization Methods**: L-BFGS-B, multi-start L-BFGS, Bayesian optimization with Gaussian processes
- **Per-Coil Customization**: Set individual search spaces and bounds for each coil
- **3×3 Thick Coil Model**: Accurate representation of finite-thickness coils using 9 filaments
- **Optimization Comparison Framework**: Compare performance of different optimization algorithms
- **Comprehensive Visualization**: Built-in plotting for coil placements, flux errors, and convergence

## Installation

### Prerequisites

- Python 3.7+
- OpenFUSION Toolkit (TokaMaker)
- Standard scientific Python stack: NumPy, SciPy, Matplotlib, pandas
- scikit-optimize (for Bayesian optimization)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd PF_location_optimization

# Ensure OpenFUSION Toolkit is installed and OFT_ROOTPATH is set
export OFT_ROOTPATH=/path/to/OpenFUSIONToolkit/install_release
```

## Quick Start

```python
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OFT_pf_coil_optimize import CoilPositionSpace, pf_coil_optimize

# Setup and solve fixed-boundary equilibrium with TokaMaker
myOFT = OFT_env(nthreads=2)
mygs = TokaMaker(myOFT)
# ... configure and solve equilibrium ...

# Define coil position search space
coil_space = CoilPositionSpace(
    inner_boundary=inner_curve,  # (N, 2) array of (R, Z) coordinates
    outer_boundary=outer_curve,  # (N, 2) array of (R, Z) coordinates
    method='coords',
    angular_bounds=(10, 170),    # Poloidal angle range (degrees)
    radial_bounds=(0.0, 1.0)     # Radial position (0=inner, 1=outer)
)

# Optimize coil positions
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=coil_space,
    n_coils=6,
    reg_current=1e-5,       # Current regularization
    reg_distance=1e-5,      # Distance penalty weight
    min_coil_distance=10,   # Minimum angular separation (degrees)
    method='bayesian',
    n_calls=100,
    local_optimize=True,    # Refine with L-BFGS
    verbose=True
)

# Access results
print(result)
print(f"Optimized angles: {result.angles}")
print(f"Optimized positions: {result.positions}")
print(f"Coil currents: {result.currents}")
```

## Core Components

### 1. OFT_pf_coil_optimize.py

Main optimization module providing:

#### CoilPositionSpace
Defines the search space for coil positions as a region bounded by two curves.

**Methods of initialization:**
- `'coords'`: Direct (R, Z) coordinate arrays
- `'parametric'`: Generated from plasma shape parameters (R₀, a, κ, δ, squareness)
- `'function'`: User-provided callable functions
- `'single_curve'`: Offset from a single curve (e.g., limiter)

```python
# Example: Parametric method
inner_params = {
    'r0': 1.69, 'a': 0.67, 'kappa': 2.0,
    'delta': 0.8, 'squar': 0.15, 'offset': 0.1
}
outer_params = {
    'r0': 1.94, 'a': 0.95, 'kappa': 1.55,
    'delta': 0.8, 'squar': 0.15, 'offset': 0.15
}

space = CoilPositionSpace(
    inner_boundary=inner_params,
    outer_boundary=outer_params,
    method='parametric'
)
```

#### PerCoilPositionSpace
Enables per-coil custom search spaces.

```python
# Define custom spaces for each coil
coil_specs = [
    'global',                    # Use global space
    custom_space_1,             # Custom CoilPositionSpace
    'global',
    custom_space_2
]

per_coil_space = PerCoilPositionSpace(
    coil_specs=coil_specs,
    global_space=default_space
)
```

#### pf_coil_optimize()
Main optimization function with configurable parameters.

**Key parameters:**
- `tokamaker_solver`: TokaMaker instance with solved equilibrium
- `coil_space`: CoilPositionSpace or PerCoilPositionSpace
- `n_coils`: Number of coil pairs (top/bottom)
- `method`: `'lbfgs'`, `'multi_start_lbfgs'`, or `'bayesian'`
- `reg_current`: Regularization weight for coil currents
- `reg_distance`: Penalty weight for coil spacing
- `min_coil_distance`: Minimum angular separation (degrees)

**Bayesian optimization parameters:**
- `n_calls`: Total evaluations
- `n_initial_points`: Random samples before GP modeling
- `acq_func`: Acquisition function (`'EI'`, `'LCB'`, `'PI'`)
- `local_optimize`: Refine results with L-BFGS
- `n_local_refine`: Number of top results to refine

#### OptimizationResult
Container for optimization results with attributes:
- `success`: Optimization success flag
- `method`: Method used
- `angles`: Optimized poloidal angles
- `radials`: Optimized radial ratios
- `positions`: Final coil (R, Z) positions
- `currents`: Computed coil currents
- `coil_geometry`: Full coil geometry dictionary
- `cost_outer`: Total cost (with penalties)
- `cost_inner`: Flux matching cost only
- `flux_error`: L2 norm of flux reproduction error
- `n_iterations`: Number of iterations

### 2. helper_functions.py

Utility functions for geometry manipulation:

- **`resize_polygon(points, dx)`**: Offset a polygon outward by distance dx
- **`resize_polygon_MANTA(points, dx)`**: MANTA-specific polygon offsetting
- **`update_boundary(...)`**: Generate plasma boundary from shape parameters
- **`place_points(npoints, arc, pol_angles)`**: Place coils at specified angles
- **`place_points_pol_rad(...)`**: Place coils using angle + radial offset
- **`smoothen(curve, window)`**: Smooth a curve with moving average
- **`compute_coil_centers(coil_pts_dict)`**: Compute centers from coil geometry
- **`make_3x3_thick(center, R)`**: Generate 3×3 filament arrangement

### 3. optimization_comparison.py

Framework for comparing optimization methods:

```python
from optimization_comparison import OptimizationComparison

# Create comparison object
comparison = OptimizationComparison(
    objective_func=my_objective,
    bounds=parameter_bounds,
    max_time=120,  # seconds per method
    NUM_COILS=10
)

# Run all methods
comparison.compare_all(x0=initial_guess)

# Generate visualizations
comparison.plot_result()
comparison.plot_each_method_coils()
comparison.plot_each_method_error()
```

**Supported methods:**
- L-BFGS-B
- Multi-start L-BFGS (Sobol sampling)
- Differential Evolution
- Dual Annealing
- Bayesian Optimization (with optional L-BFGS refinement)
- Basin Hopping
- Multi-start Basin Hopping

## Files

### Core Modules
- **`OFT_pf_coil_optimize.py`** - Main optimization module
- **`helper_functions.py`** - Geometry utility functions
- **`optimization_comparison.py`** - Optimization method benchmarking framework

### Test and Example Files
- **`test_oft_optimize.py`** - Unit tests for optimization module
- **`PF_coil_opt_original.ipynb`** - Original notebook implementation
- **`test_files/MIT_CPSFR25_withpsidt.ipynb`** - MIT test case

### Data Files
- **`g192185.02440`** - DIII-D equilibrium EQDSK file
- **`DIIID_geom.json`** - DIII-D geometric data
- **`DIIID_mesh*.h5`** - DIII-D mesh files

### Documentation
- **`CLAUDE.md`** - Development sprint plan
- **`comparisons/`** - Saved optimization comparison results

## Optimization Methods

### L-BFGS-B
Gradient-based local optimizer. Fast but sensitive to initial conditions.

```python
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=space,
    n_coils=6,
    method='lbfgs',
    initial_angles=np.linspace(10, 170, 6)
)
```

### Multi-start L-BFGS
Runs L-BFGS from multiple starting points (Sobol sequence). Better global search.

```python
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=space,
    n_coils=6,
    method='multi_start_lbfgs',
    n_starts=50  # Number of random starts
)
```

### Bayesian Optimization
Gaussian process surrogate model with intelligent sampling. Best for expensive objectives.

```python
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=space,
    n_coils=6,
    method='bayesian',
    n_calls=100,
    n_initial_points=20,
    local_optimize=True,    # Refine with L-BFGS
    n_local_refine=5
)
```

## Objective Function

The optimization minimizes:

```
Cost = Flux_error + ω × Distance_penalty + λ × Current_regularization
```

Where:
- **Flux_error**: L2 norm of boundary flux reproduction error
- **Distance_penalty**: Penalty for coils closer than `min_coil_distance`
- **Current_regularization**: Tikhonov regularization on coil currents

The flux error is computed by solving a least-squares problem for coil currents that best match the target boundary flux from the TokaMaker equilibrium.

## Examples

### Example 1: Simple optimization with default settings

```python
from OFT_pf_coil_optimize import CoilPositionSpace, pf_coil_optimize

# Define search space
space = CoilPositionSpace(
    inner_boundary=inner_curve,
    outer_boundary=outer_curve,
    method='coords'
)

# Optimize
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=space,
    n_coils=5,
    reg_current=1e-5,
    reg_distance=1e-5,
    min_coil_distance=10
)
```

### Example 2: Per-coil custom search spaces

```python
# Create custom spaces for specific coils
upper_space = CoilPositionSpace(...)  # For divertor coils
lower_space = CoilPositionSpace(...)  # For vertical field coils

coil_specs = [upper_space, 'global', 'global', 'global', lower_space]

space = PerCoilPositionSpace(
    coil_specs=coil_specs,
    global_space=default_space
)

result = pf_coil_optimize(mygs, space, n_coils=5)
```

### Example 3: Comparison of optimization methods

```python
from optimization_comparison import main

# Run comparison with custom settings
comparison, summary = main(
    methods=['lbfgs', 'multistart_lbfgs', 'bayesian']
)

# Results automatically saved to comparisons/ directory
```

## Output and Visualization

Optimization comparison generates three plots:
1. **Convergence plot**: Cost vs. iterations for each method
2. **Coil placement**: Optimized coil positions for each method
3. **Flux error**: Target vs. computed boundary flux

Results are saved to timestamped folders in `comparisons/`.

## Development Status

**Current**: Sprint 2 - Full Features
- ✅ Core optimization module
- ✅ CoilPositionSpace with multiple initialization methods
- ✅ PerCoilPositionSpace for custom per-coil spaces
- ✅ Multiple optimization algorithms (L-BFGS, multi-start, Bayesian)
- ✅ Optimization comparison framework
- ✅ Active learning and adaptive refinement
- 🚧 Advanced plotting in OptimizationResult
- 🚧 Comprehensive documentation

**Next**: Sprint 3 - Polish
- Comprehensive docstrings
- Error handling improvements
- Extended test coverage

## Citation

If you use this code in your research, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

[Contact information to be added]
