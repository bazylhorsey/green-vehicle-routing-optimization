"""
OR-Tools Solver for Green Vehicle Routing Problem
Provides exact/near-optimal baseline solutions for research comparison
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from .config import ModelParams, create_default_config
from .utils import load_data


class ORToolsGreenVRP:
    """Green VRP solver using Google OR-Tools"""
    
    def __init__(self, config: ModelParams = None):
        """Initialize OR-Tools solver"""
        self.config = config if config else create_default_config()
        self.data = {}
        self.manager = None
        self.routing = None
        self.solution = None
        
    def setup_problem(self):
        """Load data and setup problem"""
        print("="*80)
        print("OR-TOOLS GREEN VRP SOLVER - SETUP")
        print("="*80)
        
        # Load data
        d, t, v_avg, p, D, depot = load_data(
            self.config.demand_file,
            self.config.distance_file,
            self.config.time_file
        )
        
        # Convert to OR-Tools format
        customers = sorted(list(D))
        num_locations = len(customers) + 1  # +1 for depot
        
        # Create distance matrix
        distance_matrix = np.zeros((num_locations, num_locations))
        time_matrix = np.zeros((num_locations, num_locations))
        demands = [0]  # Depot has no demand
        
        # Map nodes to indices - handle multi-visit for customers exceeding capacity
        vehicle_capacity = 15000  # kg per vehicle
        node_to_idx = {depot: 0}
        actual_customers = [depot]  # Start with depot
        multi_visit_map = {}  # Track which nodes are copies of the same customer
        original_customer_map = {0: depot}  # Map index back to original customer
        
        current_idx = 1
        for customer in customers:
            customer_demand = p[customer]
            visits_needed = max(1, int(np.ceil(customer_demand / vehicle_capacity)))
            
            if visits_needed == 1:
                # Single visit sufficient
                node_to_idx[customer] = current_idx
                actual_customers.append(customer)
                original_customer_map[current_idx] = customer
                demands.append(int(customer_demand))
                current_idx += 1
            else:
                # Multiple visits needed - split demand across visits
                demand_per_visit = customer_demand / visits_needed
                for visit_num in range(visits_needed):
                    # Map this index to the original customer
                    multi_visit_map[current_idx] = (customer, visit_num, visits_needed)
                    actual_customers.append(customer)
                    original_customer_map[current_idx] = customer
                    demands.append(int(demand_per_visit))
                    current_idx += 1
        
        num_locations = len(actual_customers)
        print(f"Loaded data: {len(customers)} customers, depot at node {depot}")
        print(f"Total demand: {sum(p.values()):.2f} kg")
        if multi_visit_map:
            print(f"Multi-visit nodes: {len(multi_visit_map)} (customers with demand > capacity)")
        
        # Create expanded matrices
        distance_matrix = np.zeros((num_locations, num_locations))
        time_matrix = np.zeros((num_locations, num_locations))
        
        # Fill matrices - use same distances for multi-visit copies
        # Convert distances from km to meters for OR-Tools
        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    # Map back to original customers for distance/time lookup
                    orig_i = original_customer_map[i]
                    orig_j = original_customer_map[j]
                    distance_matrix[i, j] = d.get((orig_i, orig_j), 0) * 1000  # km to meters
                    time_matrix[i, j] = t.get((orig_i, orig_j), 0)
        
        # Setup data model
        total_vehicles = self.config.num_cfv + self.config.num_hgev
        self.data['distance_matrix'] = distance_matrix.astype(int)
        self.data['time_matrix'] = time_matrix.astype(int)
        self.data['demands'] = demands
        self.data['num_vehicles'] = total_vehicles
        self.data['depot'] = 0
        self.data['node_to_idx'] = node_to_idx
        self.data['customers'] = customers
        self.data['actual_customers'] = actual_customers
        self.data['multi_visit_map'] = multi_visit_map
        self.data['original_customer_map'] = original_customer_map
        
        # Vehicle specifications
        cfv_params = self.config.get_vehicle_params('CFV')
        hgev_params = self.config.get_vehicle_params('HGEV')
        
        # Vehicle capacities (all same for now)
        self.data['vehicle_capacities'] = [cfv_params.capacity] * total_vehicles
        
        # Vehicle costs (operating costs, not acquisition)
        # Using simplified cost structure from data files
        vehicle_fixed_costs = []
        vehicle_km_costs = []
        vehicle_emissions = []
        
        # CFV: $1000 fixed + $10/km variable from cost file
        for i in range(self.config.num_cfv):
            vehicle_fixed_costs.append(1000)  # Operating fixed cost
            vehicle_km_costs.append(10)  # $/km
            # CFV emissions: 1100 g CO2e/km from cost file
            vehicle_emissions.append(1100)
        
        # HGEV: $1000 fixed + $11/km variable from cost file  
        for i in range(self.config.num_hgev):
            vehicle_fixed_costs.append(1000)  # Operating fixed cost
            vehicle_km_costs.append(11)  # $/km
            # HGEV emissions: 11 g CO2e/km from cost file
            vehicle_emissions.append(11)
        
        self.data['vehicle_fixed_costs'] = vehicle_fixed_costs
        self.data['vehicle_km_costs'] = vehicle_km_costs
        self.data['vehicle_emissions'] = vehicle_emissions
        
        unique_customers = len(customers)
        total_nodes = len(actual_customers) - 1  # Exclude depot
        print(f"\n✓ Problem loaded:")
        print(f"  Customers: {unique_customers}")
        print(f"  Total nodes (with multi-visit): {total_nodes}")
        print(f"  Total demand: {sum(demands):.0f} kg")
        print(f"  Vehicles: {self.config.num_cfv} CFVs + {self.config.num_hgev} HGEVs")
        print(f"  Capacity per vehicle: {cfv_params.capacity:.0f} kg")
        
    def create_model(self):
        """Create OR-Tools routing model"""
        print("\nCreating OR-Tools model...")
        
        # Create routing index manager
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.data['distance_matrix']),
            self.data['num_vehicles'],
            self.data['depot']
        )
        
        # Create routing model
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            """Returns the distance between two nodes"""
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        
        # Register per-vehicle arc cost evaluators
        # Each vehicle has different $/km costs
        for vehicle_id in range(self.data['num_vehicles']):
            vehicle_cost_per_km = self.data['vehicle_km_costs'][vehicle_id]
            
            # Create a closure to capture vehicle_id
            def make_cost_callback(vid):
                def vehicle_cost_callback(from_index, to_index):
                    """Returns the cost for arc on this vehicle"""
                    from_node = self.manager.IndexToNode(from_index)
                    to_node = self.manager.IndexToNode(to_index)
                    distance_m = self.data['distance_matrix'][from_node][to_node]
                    distance_km = distance_m / 1000
                    return int(distance_km * self.data['vehicle_km_costs'][vid])
                return vehicle_cost_callback
            
            cost_callback_index = self.routing.RegisterTransitCallback(make_cost_callback(vehicle_id))
            self.routing.SetArcCostEvaluatorOfVehicle(cost_callback_index, vehicle_id)
        
        # Add capacity constraints
        def demand_callback(from_index):
            """Returns the demand of the node"""
            from_node = self.manager.IndexToNode(from_index)
            return self.data['demands'][from_node]
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data['vehicle_capacities'],
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add time constraints
        def time_callback(from_index, to_index):
            """Returns the travel time between two nodes"""
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.data['time_matrix'][from_node][to_node]
        
        time_callback_index = self.routing.RegisterTransitCallback(time_callback)
        
        self.routing.AddDimension(
            time_callback_index,
            int(self.config.tstop),  # Stop time slack
            int(self.config.tmax),   # Maximum time per vehicle
            True,  # start cumul to zero
            'Time'
        )
        
        # Allow dropping nodes with penalty (for infeasibility analysis)
        penalty = 1000000000  # High penalty to prefer serving all customers
        for node in range(1, len(self.data['distance_matrix'])):
            self.routing.AddDisjunction([self.manager.NodeToIndex(node)], penalty)
        
        # Add vehicle fixed costs
        for vehicle_id in range(self.data['num_vehicles']):
            self.routing.SetFixedCostOfVehicle(
                self.data['vehicle_fixed_costs'][vehicle_id],
                vehicle_id
            )
        
        print("  ✓ Model created with capacity and time constraints")
        
    def solve(self, time_limit: int = None) -> Tuple[bool, Dict]:
        """
        Solve the routing problem
        
        Args:
            time_limit: Time limit in seconds (uses config if None)
            
        Returns:
            success: True if solution found
            results: Dictionary with solution details
        """
        print("\n" + "="*80)
        print("SOLVING WITH OR-TOOLS")
        print("="*80)
        
        if time_limit is None:
            time_limit = self.config.time_limit
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # First solution strategy
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Local search metaheuristic
        if self.config.run_mode == 'cost_min':
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
        elif self.config.run_mode == 'emissions_min':
            # Use different strategy for emissions
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
            )
        else:
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
            )
        
        # Time limit
        search_parameters.time_limit.seconds = time_limit
        
        # Enable logging
        search_parameters.log_search = True
        
        # Solve
        start_time = time.time()
        self.solution = self.routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time
        
        # Process results
        results = {
            'success': False,
            'obj_val': None,
            'total_cost': None,
            'total_emissions': None,
            'total_distance': None,
            'routes': {},
            'num_vehicles_used': 0,
            'solve_time': solve_time,
            'dropped_nodes': []
        }
        
        if self.solution:
            results['success'] = True
            results['obj_val'] = self.solution.ObjectiveValue()
            results['total_cost'] = self.solution.ObjectiveValue()
            
            # Extract routes
            total_distance = 0
            total_load = 0
            total_time = 0
            num_vehicles_used = 0
            
            for vehicle_id in range(self.data['num_vehicles']):
                index = self.routing.Start(vehicle_id)
                route = []
                route_distance = 0
                route_load = 0
                route_time = 0
                
                while not self.routing.IsEnd(index):
                    node = self.manager.IndexToNode(index)
                    route.append(node)
                    
                    previous_index = index
                    index = self.solution.Value(self.routing.NextVar(index))
                    
                    # Get actual distance (in meters from matrix)
                    from_node = self.manager.IndexToNode(previous_index)
                    to_node = self.manager.IndexToNode(index)
                    route_distance += self.data['distance_matrix'][from_node][to_node] / 1000  # meters to km
                    route_load += self.data['demands'][node]
                
                # Add final depot
                node = self.manager.IndexToNode(index)
                route.append(node)
                
                if len(route) > 2:  # More than just depot-depot
                    vehicle_type = 'CFV' if vehicle_id < self.config.num_cfv else 'HGEV'
                    vehicle_name = f"{vehicle_type}{vehicle_id + 1 - (0 if vehicle_type == 'CFV' else self.config.num_cfv)}"
                    
                    results['routes'][vehicle_name] = {
                        'route': route,
                        'distance': route_distance,
                        'load': route_load,
                        'time': route_time
                    }
                    
                    total_distance += route_distance
                    total_load += route_load
                    num_vehicles_used += 1
            
            results['total_distance'] = total_distance
            results['num_vehicles_used'] = num_vehicles_used
            
            # Calculate emissions (simplified)
            total_emissions = 0
            for vehicle_name, route_data in results['routes'].items():
                # Parse vehicle ID based on type
                if 'CFV' in vehicle_name:
                    vehicle_num = int(vehicle_name.replace('CFV', ''))
                    vehicle_id = vehicle_num - 1
                else:  # HGEV
                    vehicle_num = int(vehicle_name.replace('HGEV', ''))
                    vehicle_id = self.config.num_cfv + vehicle_num - 1
                
                distance_km = route_data['distance'] / 1000
                emissions = distance_km * self.data['vehicle_emissions'][vehicle_id] / 1000
                total_emissions += emissions
            
            results['total_emissions'] = total_emissions
            
            # Check for dropped nodes
            dropped_nodes = []
            for node in range(1, len(self.data['distance_matrix'])):
                if self.solution.Value(self.routing.NextVar(self.manager.NodeToIndex(node))) == node:
                    dropped_nodes.append(node)
            results['dropped_nodes'] = dropped_nodes
            
            # Print summary
            print(f"\n✓ Solution found!")
            print(f"  Objective value: ${results['obj_val']:,.0f}")
            print(f"  Total distance: {total_distance/1000:.2f} km")
            print(f"  Vehicles used: {num_vehicles_used}/{self.data['num_vehicles']}")
            print(f"  Total emissions: {total_emissions:.2f} kg CO2e")
            print(f"  Solve time: {solve_time:.2f} seconds")
            
            if dropped_nodes:
                print(f"  ⚠️  Dropped nodes: {dropped_nodes}")
        else:
            print("\n❌ No solution found")
            print(f"  Search status: {self.routing.status()}")
            print(f"  Time limit: {time_limit} seconds")
        
        return results['success'], results
    
    def print_solution(self):
        """Print detailed solution"""
        if not self.solution:
            print("No solution to print")
            return
        
        print("\n" + "="*80)
        print("DETAILED SOLUTION")
        print("="*80)
        
        for vehicle_id in range(self.data['num_vehicles']):
            index = self.routing.Start(vehicle_id)
            route_nodes = []
            
            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                route_nodes.append(node)
                index = self.solution.Value(self.routing.NextVar(index))
            
            route_nodes.append(self.manager.IndexToNode(index))
            
            if len(route_nodes) > 2:
                vehicle_type = 'CFV' if vehicle_id < self.config.num_cfv else 'HGEV'
                vehicle_name = f"{vehicle_type}{vehicle_id + 1 - (0 if vehicle_type == 'CFV' else self.config.num_cfv)}"
                
                print(f"\n{vehicle_name}:")
                print(f"  Route: {' → '.join(map(str, route_nodes))}")
                
                # Calculate load
                load = sum(self.data['demands'][node] for node in route_nodes[:-1])
                print(f"  Load: {load:.0f} kg / {self.data['vehicle_capacities'][vehicle_id]:.0f} kg")


def main():
    """Test OR-Tools solver"""
    config = create_default_config()
    config.run_mode = 'cost_min'
    config.time_limit = 300  # 5 minutes
    
    solver = ORToolsGreenVRP(config)
    solver.setup_problem()
    solver.create_model()
    success, results = solver.solve()
    
    if success:
        solver.print_solution()


if __name__ == "__main__":
    main()
