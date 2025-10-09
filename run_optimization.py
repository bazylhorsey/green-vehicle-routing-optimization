"""
Main execution script for Green Vehicle Routing Optimization
Provides convenient interface for running different optimization modes
"""

import sys
import argparse
from pathlib import Path

from config import create_default_config, ModelParams
from optimized_solver import GreenVRPModel
from visualize import create_full_report


def run_cost_minimization(config: ModelParams):
    """Run cost minimization"""
    print("\n" + "="*80)
    print("COST MINIMIZATION MODE")
    print("="*80)
    
    config.run_mode = 'cost_min'
    
    model = GreenVRPModel(config)
    model.build_model()
    success, results = model.solve()
    
    if success:
        print(f"\n✓ Cost minimization complete!")
        print(f"Optimal Cost: ${results['total_cost']:,.2f}")
        print(f"Corresponding Emissions: {results['total_emissions']:.4f} metric tons CO2e")
        
        # Save normalization values
        print(f"\nSave these values for weighted optimization:")
        print(f"COST_MIN = {results['total_cost']}")
        print(f"EMISS_AT_COST_MIN = {results['total_emissions']}")
        
        # Generate visualizations
        create_full_report(results, output_dir='results/cost_min')
        
        return results
    else:
        print("\n❌ Cost minimization failed")
        return None


def run_emissions_minimization(config: ModelParams):
    """Run emissions minimization"""
    print("\n" + "="*80)
    print("EMISSIONS MINIMIZATION MODE")
    print("="*80)
    
    config.run_mode = 'emissions_min'
    
    model = GreenVRPModel(config)
    model.build_model()
    success, results = model.solve()
    
    if success:
        print(f"\n✓ Emissions minimization complete!")
        print(f"Optimal Emissions: {results['total_emissions']:.4f} metric tons CO2e")
        print(f"Corresponding Cost: ${results['total_cost']:,.2f}")
        
        # Save normalization values
        print(f"\nSave these values for weighted optimization:")
        print(f"EMISS_MIN = {results['total_emissions']}")
        print(f"COST_AT_EMISS_MIN = {results['total_cost']}")
        
        # Generate visualizations
        create_full_report(results, output_dir='results/emissions_min')
        
        return results
    else:
        print("\n❌ Emissions minimization failed")
        return None


def run_weighted_optimization(config: ModelParams, cost_weight: float = 0.5):
    """Run weighted multi-objective optimization"""
    print("\n" + "="*80)
    print(f"WEIGHTED OPTIMIZATION MODE (Cost Weight: {cost_weight})")
    print("="*80)
    
    # Check normalization bounds
    if (config.cost_min is None or config.emiss_min is None or
        config.emiss_at_cost_min is None or config.cost_at_emiss_min is None):
        print("\n❌ ERROR: Normalization bounds not set!")
        print("Please run cost_min and emissions_min modes first, then update config.")
        return None
    
    config.run_mode = 'weighted'
    config.weight_cost = cost_weight
    
    model = GreenVRPModel(config)
    model.build_model()
    success, results = model.solve()
    
    if success:
        print(f"\n✓ Weighted optimization complete!")
        print(f"Total Cost: ${results['total_cost']:,.2f}")
        print(f"Total Emissions: {results['total_emissions']:.4f} metric tons CO2e")
        
        # Generate visualizations
        create_full_report(results, output_dir=f'results/weighted_{int(cost_weight*100)}')
        
        return results
    else:
        print("\n❌ Weighted optimization failed")
        return None


def run_pareto_analysis(config: ModelParams, weights: list = None):
    """Run multiple weighted optimizations to generate Pareto frontier"""
    print("\n" + "="*80)
    print("PARETO FRONTIER ANALYSIS")
    print("="*80)
    
    if weights is None:
        weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    # Check normalization bounds
    if (config.cost_min is None or config.emiss_min is None):
        print("\n❌ ERROR: Normalization bounds not set!")
        print("Running cost and emissions minimization first...")
        
        results_cost = run_cost_minimization(config)
        if results_cost:
            config.cost_min = results_cost['total_cost']
            config.emiss_at_cost_min = results_cost['total_emissions']
        else:
            return None
        
        results_emiss = run_emissions_minimization(config)
        if results_emiss:
            config.emiss_min = results_emiss['total_emissions']
            config.cost_at_emiss_min = results_emiss['total_cost']
        else:
            return None
    
    # Run weighted optimizations
    solutions = []
    for weight in weights:
        print(f"\n{'='*80}")
        print(f"Running with cost weight: {weight}")
        print(f"{'='*80}")
        
        result = run_weighted_optimization(config, weight)
        if result:
            solutions.append((result['total_cost'], result['total_emissions']))
    
    # Plot Pareto frontier
    if solutions:
        from visualize import plot_pareto_frontier
        plot_pareto_frontier(solutions, filename='results/pareto_frontier.png')
        
        print("\n✓ Pareto analysis complete!")
        print(f"Generated {len(solutions)} solutions")
    
    return solutions


def run_alns_metaheuristic(config: ModelParams):
    """Run ALNS metaheuristic"""
    print("\n" + "="*80)
    print("ALNS METAHEURISTIC MODE")
    print("="*80)
    
    try:
        from alns_solver import ALNSSolver
        from utils import load_data, expand_network
        
        # Load data
        d, t, v_avg, p, D, depot = load_data(
            config.demand_file,
            config.distance_file,
            config.time_file
        )
        
        # Expand network
        N_expanded, d_exp, t_exp, v_avg_exp = expand_network(
            D, depot, config.vmax, d, t, v_avg
        )
        
        # Create vehicle lists
        K_c_list = [f"CFV{i}" for i in range(1, config.num_cfv + 1)]
        K_e_list = [f"HGEV{i}" for i in range(1, config.num_hgev + 1)]
        K_list = K_c_list + K_e_list
        
        # Build vehicle parameter dictionaries
        Q = {}
        h = {}
        R = {}
        O = {}
        for k in K_list:
            params = config.get_vehicle_params('CFV' if 'CFV' in k else 'HGEV')
            Q[k] = params.capacity
            h[k] = params.energy_consumption
            R[k] = params.battery_capacity
            O[k] = params.operating_cost
        
        # Prepare problem data
        problem_data = {
            'd_exp': d_exp,
            't_exp': t_exp,
            'v_avg_exp': v_avg_exp,
            'p': p,
            'D': D,
            'depot': depot,
            'K_list': K_list,
            'K_c_list': K_c_list,
            'K_e_list': K_e_list,
            'Q': Q,
            'h': h,
            'R': R,
            'O': O
        }
        
        # Run ALNS
        solver = ALNSSolver(problem_data, config)
        best_solution = solver.solve()
        
        # Plot convergence
        from visualize import plot_alns_convergence
        plot_alns_convergence(solver.history, filename='results/alns_convergence.png')
        
        print(f"\n✓ ALNS complete!")
        print(f"Best Cost: ${best_solution.cost:,.2f}")
        print(f"Emissions: {best_solution.emissions:.4f} metric tons CO2e")
        
        return best_solution
        
    except Exception as e:
        print(f"\n❌ ALNS failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Green Vehicle Routing Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run cost minimization
  python run_optimization.py --mode cost
  
  # Run emissions minimization
  python run_optimization.py --mode emissions
  
  # Run weighted optimization (70% cost, 30% emissions)
  python run_optimization.py --mode weighted --weight 0.7
  
  # Generate Pareto frontier
  python run_optimization.py --mode pareto
  
  # Run ALNS metaheuristic
  python run_optimization.py --mode alns
  
  # Custom configuration
  python run_optimization.py --mode cost --vmax 2 --time-limit 3600
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['cost', 'emissions', 'weighted', 'pareto', 'alns'],
                       help='Optimization mode')
    parser.add_argument('--weight', type=float, default=0.5,
                       help='Cost weight for weighted mode (0-1)')
    parser.add_argument('--vmax', type=int, default=1,
                       help='Maximum visits per customer')
    parser.add_argument('--time-limit', type=int, default=1800,
                       help='Solver time limit in seconds')
    parser.add_argument('--num-cfv', type=int, default=14,
                       help='Number of CFVs available')
    parser.add_argument('--num-hgev', type=int, default=14,
                       help='Number of HGEVs available')
    parser.add_argument('--mip-gap', type=float, default=0.01,
                       help='MIP optimality gap')
    parser.add_argument('--alns-iterations', type=int, default=10000,
                       help='Number of ALNS iterations')
    
    args = parser.parse_args()
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Create configuration
    config = create_default_config()
    config.vmax = args.vmax
    config.time_limit = args.time_limit
    config.num_cfv = args.num_cfv
    config.num_hgev = args.num_hgev
    config.mip_gap = args.mip_gap
    config.alns_iterations = args.alns_iterations
    
    # Run selected mode
    if args.mode == 'cost':
        run_cost_minimization(config)
    
    elif args.mode == 'emissions':
        run_emissions_minimization(config)
    
    elif args.mode == 'weighted':
        # For weighted mode, need to set normalization bounds manually
        # These should be obtained from cost and emissions runs
        print("\n⚠ WARNING: Make sure normalization bounds are set in config.py!")
        print("Run 'cost' and 'emissions' modes first to get these values.")
        run_weighted_optimization(config, args.weight)
    
    elif args.mode == 'pareto':
        run_pareto_analysis(config)
    
    elif args.mode == 'alns':
        run_alns_metaheuristic(config)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print("Check the 'results/' directory for output files and visualizations.")


if __name__ == "__main__":
    main()
