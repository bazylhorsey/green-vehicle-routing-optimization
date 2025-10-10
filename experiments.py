"""
Comprehensive Solver Comparison for Research Paper
Compares Gurobi MILP, OR-Tools, and ALNS methods
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

from src.green_vrp.config import create_default_config, ModelParams
from src.green_vrp.optimized_solver import GreenVRPModel
from src.green_vrp.alns_solver import ALNSSolver

# Try to import OR-Tools
try:
    from src.green_vrp.ortools_solver import ORToolsGreenVRP
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠️  OR-Tools not available. Install with: pip install ortools")


def run_comparison_experiment(
    config: ModelParams,
    methods: List[str] = None,
    output_dir: str = "results/comparison"
) -> Dict:
    """
    Run comprehensive comparison of all available methods
    
    Args:
        config: Model configuration
        methods: List of methods to compare ['gurobi', 'ortools', 'alns']
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    
    if methods is None:
        methods = ['alns']  # Default to ALNS since no license issues
        if ORTOOLS_AVAILABLE:
            methods.append('ortools')
    
    print("="*80)
    print("COMPREHENSIVE SOLVER COMPARISON")
    print("="*80)
    print(f"\nMethods to compare: {', '.join(methods)}")
    print(f"Problem: {config.num_cfv} CFVs + {config.num_hgev} HGEVs")
    print(f"Mode: {config.run_mode}")
    print(f"Time limit: {config.time_limit}s per method")
    
    results = {
        'config': {
            'num_cfv': config.num_cfv,
            'num_hgev': config.num_hgev,
            'vmax': config.vmax,
            'run_mode': config.run_mode,
            'time_limit': config.time_limit
        },
        'methods': {}
    }
    
    # Run each method
    for method in methods:
        print(f"\n{'='*80}")
        print(f"RUNNING: {method.upper()}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            if method == 'gurobi':
                success, method_results = run_gurobi(config)
            elif method == 'ortools':
                success, method_results = run_ortools(config)
            elif method == 'alns':
                success, method_results = run_alns(config)
            else:
                print(f"Unknown method: {method}")
                continue
            
            method_results['wall_time'] = time.time() - start_time
            method_results['success'] = success
            
            results['methods'][method] = method_results
            
            if success:
                print(f"\n✓ {method.upper()} completed successfully")
                print(f"  Cost: ${method_results.get('total_cost', 0):,.2f}")
                print(f"  Emissions: {method_results.get('total_emissions', 0):.2f} kg CO2e")
                print(f"  Time: {method_results['wall_time']:.2f}s")
            else:
                print(f"\n❌ {method.upper()} failed")
                
        except Exception as e:
            print(f"\n❌ {method.upper()} crashed: {str(e)}")
            results['methods'][method] = {
                'success': False,
                'error': str(e),
                'wall_time': time.time() - start_time
            }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path / 'comparison_results.json'}")
    
    # Create comparison visualizations
    create_comparison_plots(results, output_dir)
    
    return results


def run_gurobi(config: ModelParams):
    """Run Gurobi MILP solver"""
    try:
        model = GreenVRPModel(config)
        model.build_model()
        success, results = model.solve()
        return success, results
    except Exception as e:
        print(f"Gurobi error: {str(e)}")
        return False, {'error': str(e)}


def run_ortools(config: ModelParams):
    """Run OR-Tools solver"""
    if not ORTOOLS_AVAILABLE:
        return False, {'error': 'OR-Tools not installed'}
    
    try:
        solver = ORToolsGreenVRP(config)
        solver.setup_problem()
        solver.create_model()
        success, results = solver.solve(time_limit=config.time_limit)
        return success, results
    except Exception as e:
        print(f"OR-Tools error: {str(e)}")
        return False, {'error': str(e)}


def run_alns(config: ModelParams):
    """Run ALNS metaheuristic"""
    try:
        solver = ALNSSolver(config)
        solver.setup_problem()
        
        # Run ALNS with appropriate iterations based on time limit
        # Approximate: 1000 iterations ~= 1 minute
        max_iterations = int(config.time_limit * 16)  # ~16 iterations per second
        
        success, results = solver.solve(
            max_iterations=max_iterations,
            time_limit=config.time_limit
        )
        return success, results
    except Exception as e:
        print(f"ALNS error: {str(e)}")
        return False, {'error': str(e)}


def create_comparison_plots(results: Dict, output_dir: str):
    """Create comparison visualizations"""
    output_path = Path(output_dir)
    
    # Extract successful methods
    methods = []
    costs = []
    emissions = []
    times = []
    
    for method, data in results['methods'].items():
        if data.get('success', False):
            methods.append(method.upper())
            costs.append(data.get('total_cost', 0))
            emissions.append(data.get('total_emissions', 0))
            times.append(data.get('wall_time', 0))
    
    if not methods:
        print("No successful methods to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Solver Comparison', fontsize=16, fontweight='bold')
    
    # Cost comparison
    axes[0, 0].bar(methods, costs, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0, 0].set_ylabel('Total Cost ($)', fontweight='bold')
    axes[0, 0].set_title('Cost Comparison')
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    for i, (m, c) in enumerate(zip(methods, costs)):
        axes[0, 0].text(i, c, f'${c:,.0f}', ha='center', va='bottom')
    
    # Emissions comparison
    axes[0, 1].bar(methods, emissions, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0, 1].set_ylabel('Total Emissions (kg CO2e)', fontweight='bold')
    axes[0, 1].set_title('Emissions Comparison')
    for i, (m, e) in enumerate(zip(methods, emissions)):
        axes[0, 1].text(i, e, f'{e:.1f}', ha='center', va='bottom')
    
    # Time comparison
    axes[1, 0].bar(methods, times, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1, 0].set_ylabel('Computation Time (s)', fontweight='bold')
    axes[1, 0].set_title('Solution Time Comparison')
    for i, (m, t) in enumerate(zip(methods, times)):
        axes[1, 0].text(i, t, f'{t:.1f}s', ha='center', va='bottom')
    
    # Cost vs Emissions scatter
    axes[1, 1].scatter(costs, emissions, s=200, c=['#2E86AB', '#A23B72', '#F18F01'][:len(methods)])
    for i, m in enumerate(methods):
        axes[1, 1].annotate(m, (costs[i], emissions[i]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontweight='bold')
    axes[1, 1].set_xlabel('Total Cost ($)', fontweight='bold')
    axes[1, 1].set_ylabel('Total Emissions (kg CO2e)', fontweight='bold')
    axes[1, 1].set_title('Cost vs Emissions Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'solver_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to {output_path / 'solver_comparison.png'}")
    plt.close()
    
    # Create summary table
    df = pd.DataFrame({
        'Method': methods,
        'Cost ($)': costs,
        'Emissions (kg CO2e)': emissions,
        'Time (s)': times
    })
    
    # Calculate relative performance
    if len(methods) > 1:
        df['Cost Gap (%)'] = ((df['Cost ($)'] - df['Cost ($)'].min()) / df['Cost ($)'].min() * 100).round(2)
        df['Emissions Gap (%)'] = ((df['Emissions (kg CO2e)'] - df['Emissions (kg CO2e)'].min()) / 
                                   df['Emissions (kg CO2e)'].min() * 100).round(2)
        df['Speedup'] = (df['Time (s)'].max() / df['Time (s)']).round(2)
    
    df.to_csv(output_path / 'comparison_table.csv', index=False)
    print(f"✓ Comparison table saved to {output_path / 'comparison_table.csv'}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))


def run_pareto_analysis(
    config: ModelParams,
    method: str = 'alns',
    num_points: int = 10,
    output_dir: str = "results/pareto"
):
    """
    Generate Pareto frontier for multi-objective optimization
    
    Args:
        config: Model configuration
        method: Solver to use ('gurobi', 'ortools', 'alns')
        num_points: Number of Pareto points to generate
        output_dir: Directory to save results
    """
    print("="*80)
    print(f"PARETO FRONTIER ANALYSIS - {method.upper()}")
    print("="*80)
    print(f"Generating {num_points} Pareto points...")
    
    pareto_points = []
    
    for i, weight_cost in enumerate(np.linspace(0, 1, num_points)):
        print(f"\n[{i+1}/{num_points}] Weight_cost = {weight_cost:.2f}")
        
        # Update config
        config.run_mode = 'weighted'
        config.weight_cost = weight_cost
        
        # Run solver
        if method == 'gurobi':
            success, results = run_gurobi(config)
        elif method == 'ortools':
            success, results = run_ortools(config)
        elif method == 'alns':
            success, results = run_alns(config)
        else:
            print(f"Unknown method: {method}")
            continue
        
        if success:
            pareto_points.append({
                'weight_cost': weight_cost,
                'cost': results.get('total_cost', 0),
                'emissions': results.get('total_emissions', 0),
                'time': results.get('solve_time', results.get('wall_time', 0))
            })
    
    # Save and visualize
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(pareto_points)
    df.to_csv(output_path / f'pareto_{method}.csv', index=False)
    
    # Plot Pareto frontier
    plt.figure(figsize=(10, 8))
    plt.scatter(df['cost'], df['emissions'], s=100, c=df['weight_cost'], 
               cmap='viridis', edgecolors='black', linewidth=2)
    plt.colorbar(label='Cost Weight')
    plt.xlabel('Total Cost ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Emissions (kg CO2e)', fontsize=12, fontweight='bold')
    plt.title(f'Pareto Frontier - {method.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate extremes
    plt.annotate('Cost-optimal', xy=(df['cost'].min(), df.loc[df['cost'].idxmin(), 'emissions']),
                xytext=(-50, 20), textcoords='offset points', 
                arrowprops=dict(arrowstyle='->', color='red'),
                fontweight='bold', color='red')
    plt.annotate('Emissions-optimal', xy=(df.loc[df['emissions'].idxmin(), 'cost'], df['emissions'].min()),
                xytext=(50, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(output_path / f'pareto_{method}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Pareto frontier saved to {output_path / f'pareto_{method}.png'}")
    
    return df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run solver comparison experiments')
    parser.add_argument('--mode', choices=['comparison', 'pareto'], default='comparison',
                       help='Experiment mode')
    parser.add_argument('--methods', nargs='+', choices=['gurobi', 'ortools', 'alns'],
                       default=['ortools', 'alns'],
                       help='Methods to compare')
    parser.add_argument('--pareto-points', type=int, default=10,
                       help='Number of Pareto points')
    parser.add_argument('--time-limit', type=int, default=300,
                       help='Time limit per run (seconds)')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    config.time_limit = args.time_limit
    config.vmax = 1  # Start with single visit
    
    if args.mode == 'comparison':
        results = run_comparison_experiment(
            config,
            methods=args.methods,
            output_dir=f"{args.output_dir}/comparison"
        )
    elif args.mode == 'pareto':
        for method in args.methods:
            if method == 'gurobi':
                print("\n⚠️  Skipping Gurobi for Pareto (license limitations)")
                continue
            results = run_pareto_analysis(
                config,
                method=method,
                num_points=args.pareto_points,
                output_dir=f"{args.output_dir}/pareto"
            )


if __name__ == "__main__":
    import numpy as np
    main()
