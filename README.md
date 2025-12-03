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
from OFT_pf_coil_opt_fct import CoilPositionSpace, pf_coil_optimize

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

# Optimize coil positions using active learning (Bayesian + L-BFGS)
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=coil_space,
    n_coils=6,
    reg_current=1e-5,       # Current regularization
    reg_distance=1e-5,      # Distance penalty weight
    min_coil_distance=10,   # Minimum angular separation (degrees)
    method='bayesian',
    n_calls=100,
    n_initial_points=50,
    local_optimize=True,    # Enable active learning refinement
    n_local_refine=5,       # Refine top 5 results with L-BFGS
    verbose=True
)

# Access results
print(result)
print(f"Optimized angles: {result.angles}")
print(f"Optimized positions: {result.positions}")
print(f"Coil currents: {result.currents}")
print(f"Final cost: {result.cost_outer:.6e}")
print(f"Flux error: {result.flux_error:.6e}")
```

For a complete working example, see `examples/test_OFT_pf_coil_optimize.py`.

## Core Components

### 1. OFT_pf_coil_opt_fct.py

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
- `n_calls`: Total evaluations (default: 100)
- `n_initial_points`: Random Sobol samples before GP modeling (default: 50)
- `acq_func`: Acquisition function - `'EI'` (Expected Improvement), `'LCB'` (Lower Confidence Bound), `'PI'` (Probability of Improvement)
- `local_optimize`: Enable active learning - refine top results with L-BFGS (default: False)
- `n_local_refine`: Number of top Bayesian results to refine with L-BFGS (default: 5)
- `random_state`: Random seed for reproducibility
- `n_jobs`: Number of parallel jobs for GP optimization

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

### 2. helper_fct.py

Utility functions for geometry manipulation:

- **`resize_polygon(points, dx)`**: Offset a polygon outward by distance dx
- **`resize_polygon_MANTA(points, dx)`**: MANTA-specific polygon offsetting
- **`update_boundary(...)`**: Generate plasma boundary from shape parameters
- **`place_points(npoints, arc, pol_angles)`**: Place coils at specified angles
- **`place_points_pol_rad(...)`**: Place coils using angle + radial offset
- **`smoothen(curve, window)`**: Smooth a curve with moving average
- **`compute_coil_centers(coil_pts_dict)`**: Compute centers from coil geometry
- **`make_3x3_thick(center, R)`**: Generate 3×3 filament arrangement

### 3. opt_comparison.py

Framework for comparing optimization methods with comprehensive visualization:

```python
from opt_comparison import OptimizationComparison

# Create comparison object
comparison = OptimizationComparison(
    objective_func=my_objective,
    bounds=parameter_bounds,
    max_time=120,  # seconds per method
    NUM_COILS=10,
    OMEGA=1e-5,
    DIST_TH=10,
    REG_IN=1e-5,
    RFIL=0.01
)

# Set problem data for plotting
comparison.set_problem_data(r_bnd, psi_bnd, coil_center_cand1,
                           coil_center_cand2, o_point, eval_green)

# Run all methods
comparison.compare_all(x0=initial_guess)

# Generate visualizations
fig, coil_fig, err_fig = comparison.plot_result()  # 2x2 overview plot
comparison.plot_each_method_coils()  # Separate coil placement plots
comparison.plot_each_method_error()  # Separate flux error plots
```

**Supported methods:**
- L-BFGS-B
- Multi-start L-BFGS (Sobol sampling for better coverage)
- Differential Evolution
- Dual Annealing
- Bayesian Optimization with Gaussian Process
- Bayesian + L-BFGS (hybrid active learning approach)
- Basin Hopping
- Multi-start Basin Hopping

**Key features:**
- Time-limited optimization for fair comparison
- Comprehensive tracking of convergence history
- Three-tier visualization system:
  - Combined 2x2 overview (convergence, costs, placements, flux)
  - Per-method coil placement comparison
  - Per-method flux error analysis
- Results automatically saved to `examples/comparisons/` with timestamps

## Files

### Core Modules
- **`OFT_pf_coil_opt_fct.py`** - Main optimization module with CoilPositionSpace and pf_coil_optimize
- **`helper_fct.py`** - Geometry utility functions for polygon operations and coil placement
- **`opt_comparison.py`** - Optimization method benchmarking framework with comprehensive plotting

### Examples Directory
- **`examples/test_OFT_pf_coil_optimize.py`** - Example script demonstrating L-BFGS and Bayesian optimization
- **`examples/comparisons/`** - Saved optimization comparison results with timestamps

### Notebooks
- **`PF_coil_opt_original.ipynb`** - Original notebook implementation and development

### Data Files
- **`g192185.02440`** - DIII-D equilibrium EQDSK file
- **`DIIID_geom.json`** - DIII-D geometric data
- **`DIIID_mesh*.h5`** - DIII-D mesh files

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

**Active Learning (Hybrid) Approach:**
When `local_optimize=True`, uses a two-phase strategy:
1. **Phase 1 (50% time)**: Bayesian optimization explores the space using GP-guided acquisition
2. **Phase 2 (50% time)**: L-BFGS refinement of top N results for precise convergence

```python
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=space,
    n_coils=6,
    reg_current=1e-5,
    reg_distance=1e-5,
    min_coil_distance=10,
    method='bayesian',
    n_calls=100,            # Total GP evaluations
    n_initial_points=50,    # Sobol samples before modeling
    acq_func='EI',          # Expected Improvement
    local_optimize=True,    # Enable active learning
    n_local_refine=5,       # Refine top 5 results
    random_state=42
)
```

**Pure Bayesian (No Refinement):**
```python
result = pf_coil_optimize(
    tokamaker_solver=mygs,
    coil_space=space,
    n_coils=6,
    method='bayesian',
    n_calls=100,
    local_optimize=False    # Pure GP optimization
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

### Example 3: Running the test script

The `examples/` directory contains a ready-to-run test script:

```bash
# Run L-BFGS optimization test
python examples/test_OFT_pf_coil_optimize.py --method lbfgs

# Run Bayesian optimization test
python examples/test_OFT_pf_coil_optimize.py --method bayesian

# Run all tests
python examples/test_OFT_pf_coil_optimize.py --method all
```

### Example 4: Comparison of optimization methods

```python
from opt_comparison import main

# Setup TokaMaker solver (mygs) first...
# (see examples/test_OFT_pf_coil_optimize.py for full setup)

# Run comparison with custom settings
comparison, summary = main(
    mygs=mygs,
    methods=['lbfgs', 'multistart_lbfgs', 'bayesian'],
    NUM_COILS=6,
    MAX_TIME=120,
    OMEGA=1e-7,
    DIST_TH=10,
    REG_IN=1e-5,
    RFIL=0.01
)

# Results automatically saved to examples/comparisons/{timestamp}/
# - convergence_plot.png (2x2 overview)
# - coil_placement_plot.png (per-method placements)
# - flux_error_plot.png (per-method flux errors)
```

The comparison framework generates three comprehensive plots:
1. **2x2 Overview**: Convergence curves, cost comparison bar chart, combined coil placements, flux reproduction
2. **Per-Method Coil Placement**: Individual subplots showing optimal coil positions for each method
3. **Per-Method Flux Error**: Individual subplots showing desired vs. computed flux for each method

## Output and Visualization

### Optimization Results
The `pf_coil_optimize()` function returns an `OptimizationResult` object containing:
- Optimized coil angles and radial positions
- Final (R, Z) coil positions
- Computed coil currents
- Complete coil geometry dictionary
- Cost metrics (outer cost, inner cost, flux error)
- Optimization metadata (iterations, success flag, method used)

### Comparison Visualization
The `opt_comparison.py` framework generates three comprehensive visualization plots:

1. **2x2 Overview Plot** (`convergence_plot.png`):
   - Top-left: Convergence curves showing cost vs. evaluations for all methods
   - Top-right: Bar chart comparing final costs with percentage differences
   - Bottom-left: Combined coil placement showing all methods' optimal positions
   - Bottom-right: Flux reproduction showing desired vs. computed boundary flux

2. **Per-Method Coil Placement** (`coil_placement_plot.png`):
   - Individual subplots for each optimization method
   - Shows coil positions (top and bottom) with position space boundaries
   - Displays final cost for each method

3. **Per-Method Flux Error** (`flux_error_plot.png`):
   - Individual subplots comparing desired and computed flux
   - Includes RMSE and max error metrics for each method
   - Helps diagnose where flux reproduction is weakest

Results are saved to timestamped folders in `examples/comparisons/{YYYYMMDD_HHMMSS}/`.

## Development Status

**Current**: Sprint 2 - Full Features ✅
- ✅ Core optimization module with three methods
- ✅ CoilPositionSpace with 4 initialization methods (coords, parametric, function, single_curve)
- ✅ PerCoilPositionSpace for custom per-coil spaces
- ✅ Multiple optimization algorithms (L-BFGS, multi-start L-BFGS, Bayesian)
- ✅ Active learning (Bayesian + L-BFGS hybrid approach)
- ✅ Comprehensive optimization comparison framework
- ✅ Three-tier visualization system (overview, per-method coils, per-method flux)
- ✅ Example scripts and test cases
- ✅ Sobol sampling for multi-start methods

**Future Enhancements**:
- Extended test coverage and unit tests
- Additional optimization methods (DIRECT, CMA-ES)
- Interactive plotting capabilities
- Constraint handling for engineering limits

## Citation

If you use this code in your research, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

[Contact information to be added]
