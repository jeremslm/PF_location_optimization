"""
PF Coil Optimization Package
=============================

Generalized optimization framework for poloidal field (PF) coil placement in tokamaks.

This package provides tools to optimize PF coil locations in fixed-boundary equilibria,
minimizing coil currents while matching target boundary flux conditions.

Main Components
---------------
- pf_coil_optimize: Main optimization function
- CoilPositionSpace: Define global coil search space boundaries
- PerCoilPositionSpace: Define per-coil search space boundaries
- OptimizationResult: Container for optimization results with visualization

"""

from .OFT_pf_coil_opt_fct import (
    pf_coil_optimize,
    CoilPositionSpace,
    PerCoilPositionSpace,
    OptimizationResult
)

from .opt_comparison import (
    OptimizationComparison
)

__version__ = '1.0.2'
__all__ = [
    'OFT_pf_coil_opt_fct',
    'CoilPositionSpace',
    'PerCoilPositionSpace',
    'OptimizationResult'
    'OptimizationComparison'
]