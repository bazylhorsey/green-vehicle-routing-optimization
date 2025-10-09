"""
Visualization tools for Green Vehicle Routing Problem solutions
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


def plot_routes(routes: Dict[str, List[List]], coordinates: Dict, 
                filename: str = 'routes_map.png'):
    """
    Plot vehicle routes on a 2D map
    
    Args:
        routes: Dictionary {vehicle_id: [[route1], [route2], ...]}
        coordinates: Dictionary {node: (x, y)} with node coordinates
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors for different vehicles
    colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
    
    # Plot depot
    depot_node = (22, 0)
    if depot_node in coordinates:
        depot_x, depot_y = coordinates[depot_node]
        ax.plot(depot_x, depot_y, 'k*', markersize=20, label='Depot', zorder=5)
    
    # Plot customer nodes
    customer_coords = {node: coord for node, coord in coordinates.items() 
                      if node != depot_node}
    if customer_coords:
        cx = [coord[0] for coord in customer_coords.values()]
        cy = [coord[1] for coord in customer_coords.values()]
        ax.scatter(cx, cy, c='gray', s=100, alpha=0.6, zorder=3, label='Customers')
    
    # Plot routes for each vehicle
    for idx, (vehicle, vehicle_routes) in enumerate(sorted(routes.items())):
        if not vehicle_routes:
            continue
        
        color = colors[idx]
        
        for route_idx, route in enumerate(vehicle_routes):
            # Get coordinates for route
            route_coords = [coordinates.get(node, (0, 0)) for node in route]
            rx = [coord[0] for coord in route_coords]
            ry = [coord[1] for coord in route_coords]
            
            # Plot route
            label = f'{vehicle}' if route_idx == 0 else None
            ax.plot(rx, ry, '-', color=color, linewidth=2, alpha=0.7, 
                   label=label, zorder=2)
            
            # Add arrows to show direction
            for i in range(len(rx) - 1):
                dx = rx[i+1] - rx[i]
                dy = ry[i+1] - ry[i]
                if dx != 0 or dy != 0:
                    ax.arrow(rx[i], ry[i], dx*0.3, dy*0.3, 
                            head_width=0.5, head_length=0.3, 
                            fc=color, ec=color, alpha=0.6, zorder=1)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Vehicle Routes', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Routes map saved to {filename}")
    plt.close()


def plot_pareto_frontier(solutions: List[Tuple[float, float]], 
                         filename: str = 'pareto_frontier.png'):
    """
    Plot Pareto frontier of cost vs emissions
    
    Args:
        solutions: List of (cost, emissions) tuples
        filename: Output filename
    """
    costs = [s[0] for s in solutions]
    emissions = [s[1] for s in solutions]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot all solutions
    ax.scatter(costs, emissions, s=100, alpha=0.6, c='blue', 
              edgecolors='black', linewidths=1, zorder=3)
    
    # Identify Pareto frontier
    pareto_points = []
    for i, (c, e) in enumerate(solutions):
        is_pareto = True
        for j, (c2, e2) in enumerate(solutions):
            if i != j:
                if c2 <= c and e2 <= e and (c2 < c or e2 < e):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append((c, e))
    
    # Plot Pareto frontier
    if pareto_points:
        pareto_points.sort()
        px = [p[0] for p in pareto_points]
        py = [p[1] for p in pareto_points]
        ax.plot(px, py, 'r-', linewidth=2, label='Pareto Frontier', zorder=4)
        ax.scatter(px, py, s=150, c='red', marker='*', 
                  edgecolors='black', linewidths=2, zorder=5)
    
    ax.set_xlabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Emissions (metric tons CO2e)', fontsize=12, fontweight='bold')
    ax.set_title('Cost vs Emissions Trade-off', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Pareto frontier saved to {filename}")
    plt.close()


def plot_alns_convergence(history: Dict, filename: str = 'alns_convergence.png'):
    """
    Plot ALNS convergence over iterations
    
    Args:
        history: Dictionary with 'iteration', 'cost', 'best_cost', 'temperature'
        filename: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    iterations = history['iteration']
    costs = history['cost']
    best_costs = history['best_cost']
    temps = history['temperature']
    
    # Plot costs
    ax1.plot(iterations, costs, 'b-', alpha=0.3, linewidth=0.5, label='Current Cost')
    ax1.plot(iterations, best_costs, 'r-', linewidth=2, label='Best Cost')
    ax1.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('ALNS Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature
    ax2.plot(iterations, temps, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ ALNS convergence plot saved to {filename}")
    plt.close()


def plot_fleet_composition(solution: Dict, filename: str = 'fleet_composition.png'):
    """
    Plot fleet composition pie chart
    
    Args:
        solution: Solution dictionary with 'used' key
        filename: Output filename
    """
    used = solution.get('used', {})
    
    cfv_count = sum(1 for k, v in used.items() if 'CFV' in k and v > 0.5)
    hgev_count = sum(1 for k, v in used.items() if 'HGEV' in k and v > 0.5)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if cfv_count + hgev_count > 0:
        labels = ['CFVs', 'HGEVs']
        sizes = [cfv_count, hgev_count]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90,
               textprops={'fontsize': 14, 'fontweight': 'bold'})
        ax.axis('equal')
        
        title = f'Fleet Composition\nTotal Vehicles: {cfv_count + hgev_count}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with counts
        legend_labels = [f'CFVs: {cfv_count}', f'HGEVs: {hgev_count}']
        ax.legend(legend_labels, loc='upper right', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No vehicles used', 
               ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Fleet composition chart saved to {filename}")
    plt.close()


def plot_cost_breakdown(cost_breakdown: Dict, filename: str = 'cost_breakdown.png'):
    """
    Plot cost breakdown bar chart
    
    Args:
        cost_breakdown: Dictionary {cost_type: value}
        filename: Output filename
    """
    # Filter out zero or very small values
    filtered = {k: v for k, v in cost_breakdown.items() if v > 100}
    
    if not filtered:
        print("No significant costs to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = list(filtered.keys())
    values = list(filtered.values())
    
    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:,.0f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Cost breakdown chart saved to {filename}")
    plt.close()


def plot_emissions_breakdown(emissions_breakdown: Dict, 
                            filename: str = 'emissions_breakdown.png'):
    """
    Plot emissions breakdown pie chart
    
    Args:
        emissions_breakdown: Dictionary {emission_type: value}
        filename: Output filename
    """
    # Filter out zero or very small values
    filtered = {k: v for k, v in emissions_breakdown.items() if v > 0.01}
    
    if not filtered:
        print("No significant emissions to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(filtered.keys())
    sizes = list(filtered.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    
    explode = [0.05] * len(labels)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.axis('equal')
    
    total = sum(sizes)
    title = f'Emissions Breakdown\nTotal: {total:.2f} metric tons CO2e'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Emissions breakdown chart saved to {filename}")
    plt.close()


def generate_coordinates_from_distances(distance_matrix: np.ndarray, 
                                        method: str = 'mds') -> Dict:
    """
    Generate 2D coordinates from distance matrix using dimensionality reduction
    
    Args:
        distance_matrix: Square distance matrix
        method: 'mds' (Multi-Dimensional Scaling) or 'tsne'
    
    Returns:
        Dictionary {node_index: (x, y)}
    """
    try:
        if method == 'mds':
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(distance_matrix)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, metric='precomputed', random_state=42)
            coords = tsne.fit_transform(distance_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create coordinate dictionary
        coordinates = {}
        for i in range(len(coords)):
            if i == len(coords) - 1:
                # Last node is depot
                coordinates[(i, 0)] = tuple(coords[i])
            else:
                # Customer nodes (with visit index 1)
                coordinates[(i, 1)] = tuple(coords[i])
        
        return coordinates
    
    except ImportError:
        print("Warning: scikit-learn not installed. Using random coordinates.")
        # Fallback to random coordinates
        coordinates = {}
        for i in range(len(distance_matrix)):
            if i == len(distance_matrix) - 1:
                coordinates[(i, 0)] = (0, 0)  # Depot at origin
            else:
                coordinates[(i, 1)] = (np.random.rand() * 100, np.random.rand() * 100)
        return coordinates


def create_full_report(results: Dict, output_dir: str = '.'):
    """
    Create a full visualization report
    
    Args:
        results: Solution results dictionary
        output_dir: Directory to save plots
    """
    print("\nGenerating visualization report...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot fleet composition
    if 'solution' in results:
        plot_fleet_composition(
            results['solution'], 
            filename=f'{output_dir}/fleet_composition.png'
        )
    
    # Plot cost breakdown
    if 'cost_breakdown' in results and results['cost_breakdown']:
        plot_cost_breakdown(
            results['cost_breakdown'], 
            filename=f'{output_dir}/cost_breakdown.png'
        )
    
    # Plot emissions breakdown
    if 'emissions_breakdown' in results and results['emissions_breakdown']:
        plot_emissions_breakdown(
            results['emissions_breakdown'], 
            filename=f'{output_dir}/emissions_breakdown.png'
        )
    
    print(f"\n✓ Visualization report complete! Files saved to {output_dir}/")


if __name__ == "__main__":
    print("Visualization module for Green Vehicle Routing Problem")
    print("Use with solution results from optimized_solver.py or alns_solver.py")
