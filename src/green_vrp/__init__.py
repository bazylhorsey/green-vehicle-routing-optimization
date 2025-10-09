"""
Green Vehicle Routing Problem Optimization Package

This package provides tools for solving multi-objective green vehicle routing problems
with mixed fleets of conventional fuel vehicles (CFVs) and hybrid gas-electric vehicles (HGEVs).

Main Components:
    - config: Configuration and parameters
    - utils: Data loading and utility functions
    - optimized_solver: MILP solver with indicator constraints
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

__all__ = [
    "ModelParams",
    "load_data",
    "expand_network",
    "validate_solution",
    "GreenVRPModel",
    "ALNSSolver",
    "plot_routes",
    "plot_pareto_frontier",
    "create_full_report",
]
