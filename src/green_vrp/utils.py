"""
Utility functions for Green Vehicle Routing Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, List
import sys


def load_data(demand_file: str, distance_file: str, time_file: str) -> Tuple[Dict, Dict, Dict, Dict, Set, int]:
    """
    Load and process input data files
    
    Returns:
        d: Distance dictionary {(i, j): distance}
        t: Time dictionary {(i, j): time}
        v_avg: Velocity dictionary {(i, j): velocity}
        p: Demand dictionary {i: demand}
        D: Set of customer nodes
        depot: Depot node index
    """
    try:
        # Load data
        demand_df = pd.read_csv(demand_file, header=None)
        distance_df = pd.read_csv(distance_file, header=None)
        time_df = pd.read_csv(time_file, header=None)
        
        # Extract dimensions
        num_customers = len(demand_df) - 1
        D = set(range(num_customers))
        depot = num_customers
        
        # Convert to arrays
        distance_array = distance_df.values
        time_array = time_df.values
        demand_array = demand_df.values
        
        # Build dictionaries
        d = {}
        t = {}
        v_avg = {}
        
        # Ghost pairs handling (special case for charging station duplicates)
        ghost_pairs = {(0, 20), (11, 21), (20, 0), (21, 11)}
        
        N_original = D.union({depot})
        for i in N_original:
            for j in N_original:
                if i == j:
                    continue
                
                # Extract distance and time
                if i < distance_array.shape[0] and j < distance_array.shape[1]:
                    d[(i, j)] = distance_array[i, j]
                if i < time_array.shape[0] and j < time_array.shape[1]:
                    t[(i, j)] = time_array[i, j]
                
                # Calculate velocity
                dist_ij = d.get((i, j), 0)
                time_ij = t.get((i, j), 0)
                
                if (i, j) in ghost_pairs and time_ij == 0:
                    time_ij = 0.1  # Avoid division by zero
                
                if time_ij > 0 and dist_ij >= 0:
                    v_avg[(i, j)] = dist_ij / time_ij
        
        # Build demand dictionary
        p = {}
        for i in range(len(N_original)):
            if i < demand_array.shape[0]:
                p[i] = float(demand_array[i, 0])
            else:
                p[i] = 0.0
        
        print(f"Loaded data: {num_customers} customers, depot at node {depot}")
        print(f"Total demand: {sum(p.values()):.2f} kg")
        
        return d, t, v_avg, p, D, depot
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def expand_network(D: Set[int], depot: int, Vmax: int, 
                   d: Dict, t: Dict, v_avg: Dict) -> Tuple[List, Dict, Dict, Dict]:
    """
    Expand the network to allow multiple visits per customer
    
    Returns:
        N_expanded: List of expanded nodes
        d_exp: Expanded distance dictionary
        t_exp: Expanded time dictionary
        v_avg_exp: Expanded velocity dictionary
    """
    # Create expanded customer set
    D_expanded = {(i, v) for i in D for v in range(1, Vmax + 1)}
    
    # Create expanded node set
    N_expanded = sorted({(depot, 0)}.union(D_expanded), key=lambda x: (x[0], x[1]))
    
    # Build expanded dictionaries
    d_exp = {}
    t_exp = {}
    v_avg_exp = {}
    
    for i_node in N_expanded:
        for j_node in N_expanded:
            if i_node == j_node:
                continue
            
            i_orig = i_node[0]
            j_orig = j_node[0]
            
            # If either is depot, use original data
            if i_orig == depot or j_orig == depot:
                d_exp[(i_node, j_node)] = d.get((i_orig, j_orig), 0)
                t_exp[(i_node, j_node)] = t.get((i_orig, j_orig), 0)
                v_avg_exp[(i_node, j_node)] = v_avg.get((i_orig, j_orig), 0)
            else:
                # If same customer, distance is 0
                if i_orig == j_orig:
                    d_exp[(i_node, j_node)] = 0
                    t_exp[(i_node, j_node)] = 0
                    v_avg_exp[(i_node, j_node)] = 0
                else:
                    d_exp[(i_node, j_node)] = d.get((i_orig, j_orig), 0)
                    t_exp[(i_node, j_node)] = t.get((i_orig, j_orig), 0)
                    v_avg_exp[(i_node, j_node)] = v_avg.get((i_orig, j_orig), 0)
    
    print(f"Network expanded: {len(N_expanded)} nodes (Vmax={Vmax})")
    
    return N_expanded, d_exp, t_exp, v_avg_exp


def calculate_fuel_consumption(W_load: float, d_ij: float, t_ij: float, v_ij: float,
                               phi: float, sigma: float, epsilon: float, lambda_param: float,
                               W_c: float, f_param: float, I_param: float, E_param: float) -> float:
    """
    Calculate fuel consumption for a CFV on an arc
    
    Args:
        W_load: Load weight (kg)
        d_ij: Distance (m)
        t_ij: Time (s)
        v_ij: Velocity (m/s)
        phi, sigma, epsilon, lambda_param: Model parameters
        W_c, f_param, I_param, E_param: Vehicle parameters
    
    Returns:
        Fuel consumption in liters
    """
    # Curb weight component
    fuel_curb = (W_c * phi * sigma * d_ij + 
                 f_param * I_param * E_param * t_ij + 
                 epsilon * phi * d_ij * (v_ij ** 2)) * lambda_param
    
    # Load weight component
    fuel_load = phi * sigma * lambda_param * d_ij * W_load
    
    return fuel_curb + fuel_load


def reconstruct_routes(x_solution: Dict, K_list: List[str], N_expanded: List, depot: int) -> Dict:
    """
    Reconstruct vehicle routes from solution
    
    Args:
        x_solution: Dictionary of {(k, i, j): value} for arcs used
        K_list: List of vehicle IDs
        N_expanded: List of expanded nodes
        depot: Depot node index
    
    Returns:
        routes: Dictionary {vehicle_id: [list of tours]}
    """
    routes = {}
    
    for k in K_list:
        # Get arcs for this vehicle
        vehicle_arcs = {(i, j): val for (veh, i, j), val in x_solution.items() 
                       if veh == k and val > 0.5}
        
        if not vehicle_arcs:
            continue
        
        tours = []
        remaining_arcs = set(vehicle_arcs.keys())
        
        # Find all tours starting from depot
        while remaining_arcs:
            # Find arc from depot
            start_arc = None
            for i, j in remaining_arcs:
                if i == (depot, 0):
                    start_arc = (i, j)
                    break
            
            if start_arc is None:
                break
            
            # Follow the tour
            tour = [start_arc[0], start_arc[1]]
            remaining_arcs.remove(start_arc)
            current = start_arc[1]
            
            # Follow path until returning to depot
            max_steps = len(N_expanded)  # Safety limit
            steps = 0
            while current != (depot, 0) and steps < max_steps:
                steps += 1
                found = False
                for next_i, next_j in list(remaining_arcs):
                    if next_i == current:
                        tour.append(next_j)
                        remaining_arcs.remove((next_i, next_j))
                        current = next_j
                        found = True
                        break
                if not found:
                    break
            
            tours.append(tour)
        
        if tours:
            routes[k] = tours
    
    return routes


def print_solution_summary(obj_val: float, total_cost: float, total_emissions: float,
                          cost_breakdown: Dict, emissions_breakdown: Dict,
                          routes: Dict, run_mode: str):
    """Print a formatted summary of the solution"""
    
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    
    print(f"\nRun Mode: {run_mode}")
    print(f"Objective Value: {obj_val:.4f}")
    print(f"\nTotal Cost: ${total_cost:,.2f}")
    print(f"Total Emissions: {total_emissions:.4f} metric tons CO2e")
    
    print("\n" + "-"*80)
    print("COST BREAKDOWN")
    print("-"*80)
    for key, value in cost_breakdown.items():
        print(f"  {key:.<50} ${value:>15,.2f}")
    
    print("\n" + "-"*80)
    print("EMISSIONS BREAKDOWN")
    print("-"*80)
    for key, value in emissions_breakdown.items():
        print(f"  {key:.<50} {value:>15.4f} metric tons")
    
    print("\n" + "-"*80)
    print("VEHICLE ROUTES")
    print("-"*80)
    for vehicle, tours in sorted(routes.items()):
        print(f"\n{vehicle}:")
        for idx, tour in enumerate(tours, 1):
            tour_str = " -> ".join(str(node) for node in tour)
            print(f"  Trip {idx}: {tour_str}")
    
    print("\n" + "="*80)


def validate_solution(routes: Dict, p: Dict, delivered: Dict, 
                     K_list: List[str], D_expanded: List, Vmax: int) -> bool:
    """
    Validate that the solution meets all constraints
    
    Returns:
        True if valid, False otherwise
    """
    valid = True
    
    # Check demand satisfaction
    print("\nValidating demand satisfaction...")
    for i in set(node[0] for node in D_expanded):
        demand_i = p.get(i, 0)
        if demand_i == 0:
            continue
        
        delivered_i = sum(delivered.get((k, (i, v)), 0) 
                         for k in K_list 
                         for v in range(1, Vmax + 1))
        
        if abs(demand_i - delivered_i) > 1e-3:
            print(f"  ⚠ Customer {i}: Demand={demand_i:.2f}, Delivered={delivered_i:.2f}")
            valid = False
    
    if valid:
        print("  ✓ All demands satisfied")
    
    return valid


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test data loading
    d, t, v_avg, p, D, depot = load_data(
        "data/Demand.csv", 
        "data/Distances.csv", 
        "data/Times.csv"
    )
    
    print(f"✓ Data loaded: {len(D)} customers")
    
    # Test network expansion
    N_expanded, d_exp, t_exp, v_avg_exp = expand_network(D, depot, 1, d, t, v_avg)
    print(f"✓ Network expanded: {len(N_expanded)} nodes")
    
    print("\nAll utility functions working correctly!")
