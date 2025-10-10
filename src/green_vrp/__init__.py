"""
Green Vehicle Routing Problem Optimization Package

This package provides tools for solving multi-objective green vehicle routing problems
with mixed fleets of conventional fuel vehicles (CFVs) and hybrid gas-electric vehicles (HGEVs).

Main Components:
    - config: Configuration and parameters
    - utils: Data loading and utility functions
    - optimized_solver: MILP solver with indicator constraints (Gurobi)
    - ortools_solver: CP-SAT solver with OR-Tools (baseline for research)
    - alns_solver: Adaptive Large Neighborhood Search metaheuristic
    - visualize: Visualization and reporting tools
"""

__version__ = "1.0.0"
__author__ = "bazylhorsey"

from .config import ModelParams
from .utils import load_data, expand_network, validate_solution
from .optimized_solver import GreenVRPModel
from .alns_solver import ALNSSolver
from .visualize import plot_routes, plot_pareto_frontier, create_full_report

# Optional: OR-Tools solver (may not be installed)
try:
    from .ortools_solver import ORToolsGreenVRP
    __all_solvers__ = ["GreenVRPModel", "ORToolsGreenVRP", "ALNSSolver"]
except ImportError:
    ORToolsGreenVRP = None
    __all_solvers__ = ["GreenVRPModel", "ALNSSolver"]

__all__ = [
    "ModelParams",
    "load_data",
    "expand_network",
    "validate_solution",
    "GreenVRPModel",
    "ORToolsGreenVRP",
    "ALNSSolver",
    "plot_routes",
    "plot_pareto_frontier",
    "create_full_report",
]
