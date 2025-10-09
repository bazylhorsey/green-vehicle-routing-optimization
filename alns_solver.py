"""
Adaptive Large Neighborhood Search (ALNS) Metaheuristic
for Green Vehicle Routing Problem

Implements removal and repair operators with adaptive weight selection
"""

import random
import math
import copy
from typing import List, Dict, Tuple, Set, Callable
import numpy as np

from config import ModelParams


class Solution:
    """Represents a VRP solution"""
    
    def __init__(self, routes: Dict[str, List[List]], cost: float = float('inf'), 
                 emissions: float = float('inf')):
        """
        Args:
            routes: Dictionary {vehicle_id: [[route1], [route2], ...]}
            cost: Total cost of solution
            emissions: Total emissions of solution
        """
        self.routes = routes
        self.cost = cost
        self.emissions = emissions
        self.objective = cost  # Can be changed to weighted objective
    
    def copy(self):
        """Create a deep copy of the solution"""
        return Solution(
            routes=copy.deepcopy(self.routes),
            cost=self.cost,
            emissions=self.emissions
        )
    
    def get_all_customers(self) -> Set[int]:
        """Get set of all customers visited in solution"""
        customers = set()
        for vehicle_routes in self.routes.values():
            for route in vehicle_routes:
                for node in route:
                    if isinstance(node, tuple) and node[0] != 22:  # Not depot
                        customers.add(node[0])
        return customers
    
    def remove_customer(self, customer: int):
        """Remove all visits to a customer from solution"""
        for vehicle in self.routes:
            new_routes = []
            for route in self.routes[vehicle]:
                new_route = [node for node in route 
                           if not (isinstance(node, tuple) and node[0] == customer)]
                # Only keep route if it has more than just depot
                if len(new_route) > 2:
                    new_routes.append(new_route)
            self.routes[vehicle] = new_routes
    
    def remove_route(self, vehicle: str):
        """Remove all routes for a vehicle"""
        self.routes[vehicle] = []
    
    def is_feasible(self) -> bool:
        """Check if solution structure is valid"""
        for vehicle_routes in self.routes.values():
            for route in vehicle_routes:
                if len(route) < 2:
                    return False
                # Check starts and ends at depot
                if route[0] != (22, 0) or route[-1] != (22, 0):
                    return False
        return True


class ALNSSolver:
    """Adaptive Large Neighborhood Search for Green VRP"""
    
    def __init__(self, problem_data: Dict, config: ModelParams):
        """
        Initialize ALNS solver
        
        Args:
            problem_data: Dictionary with d_exp, t_exp, p, Q, h, R, etc.
            config: Model configuration
        """
        self.data = problem_data
        self.config = config
        self.depot = (problem_data['depot'], 0)
        
        # Operator weights (will be updated adaptively)
        self.removal_weights = {
            'random_customer': 1.0,
            'random_route': 1.0,
            'worst_cost': 1.0,
            'worst_emissions': 1.0
        }
        
        self.repair_weights = {
            'random': 1.0,
            'lowest_cost': 1.0,
            'lowest_emissions': 1.0,
            'regret2_cost': 1.0,
            'regret2_emissions': 1.0
        }
        
        # Operator scores
        self.removal_scores = {k: 0 for k in self.removal_weights}
        self.repair_scores = {k: 0 for k in self.repair_weights}
        self.removal_uses = {k: 0 for k in self.removal_weights}
        self.repair_uses = {k: 0 for k in self.repair_weights}
        
        # Best solutions
        self.best_solution = None
        self.current_solution = None
        
        # Statistics
        self.iteration = 0
        self.temperature = config.alns_start_temp
        self.history = {
            'iteration': [],
            'cost': [],
            'emissions': [],
            'best_cost': [],
            'temperature': []
        }
    
    def construct_greedy_solution(self) -> Solution:
        """
        Construct initial solution using greedy heuristic
        Assign customers to vehicles based on nearest neighbor
        """
        print("Constructing initial greedy solution...")
        
        routes = {k: [] for k in self.data['K_list']}
        unserved = set(self.data['D'])
        
        for vehicle in self.data['K_list']:
            if not unserved:
                break
            
            route = [self.depot]
            current_load = 0
            current_pos = self.depot
            current_energy = self.data['R'][vehicle]  # Full battery
            
            while unserved:
                # Find nearest unserved customer that fits capacity
                best_customer = None
                best_distance = float('inf')
                
                for customer in unserved:
                    customer_node = (customer, 1)  # First visit
                    demand = self.data['p'][customer]
                    
                    # Check capacity
                    if current_load + demand > self.data['Q'][vehicle]:
                        continue
                    
                    # Check energy (for HGEVs)
                    if 'HGEV' in vehicle:
                        dist_to_customer = self.data['d_exp'].get((current_pos, customer_node), 0)
                        dist_to_depot = self.data['d_exp'].get((customer_node, self.depot), 0)
                        energy_needed = self.data['h'][vehicle] * (dist_to_customer + dist_to_depot)
                        energy_needed += self.config.ms  # Safety margin
                        
                        if energy_needed > current_energy:
                            continue
                    
                    # Calculate distance
                    distance = self.data['d_exp'].get((current_pos, customer_node), 0)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer
                
                if best_customer is None:
                    break  # No more customers fit
                
                # Add customer to route
                customer_node = (best_customer, 1)
                route.append(customer_node)
                unserved.remove(best_customer)
                
                # Update state
                current_load += self.data['p'][best_customer]
                if 'HGEV' in vehicle:
                    energy_consumed = self.data['h'][vehicle] * self.data['d_exp'].get((current_pos, customer_node), 0)
                    current_energy -= energy_consumed
                current_pos = customer_node
            
            # Return to depot
            if len(route) > 1:
                route.append(self.depot)
                routes[vehicle] = [route]
        
        # Create solution
        solution = Solution(routes=routes)
        
        # Evaluate solution
        solution.cost, solution.emissions = self.evaluate_solution(solution)
        solution.objective = solution.cost
        
        print(f"  Initial solution: {len(solution.get_all_customers())} customers served")
        print(f"  Cost: ${solution.cost:,.2f}, Emissions: {solution.emissions:.2f} tons")
        
        return solution
    
    def evaluate_solution(self, solution: Solution) -> Tuple[float, float]:
        """
        Evaluate solution cost and emissions
        Returns: (total_cost, total_emissions)
        """
        total_cost = 0
        total_emissions = 0
        
        cfg = self.config
        
        for vehicle, vehicle_routes in solution.routes.items():
            if not vehicle_routes:
                continue
            
            # Vehicle acquisition cost if used
            vehicle_params = cfg.get_vehicle_params('CFV' if 'CFV' in vehicle else 'HGEV')
            total_cost += vehicle_params.acquisition_cost - vehicle_params.subsidy
            
            for route in vehicle_routes:
                route_load = 0
                
                for i in range(len(route) - 1):
                    curr_node = route[i]
                    next_node = route[i + 1]
                    
                    # Get arc data
                    distance = self.data['d_exp'].get((curr_node, next_node), 0)
                    time = self.data['t_exp'].get((curr_node, next_node), 0)
                    velocity = self.data['v_avg_exp'].get((curr_node, next_node), 0)
                    
                    # Add delivery if at customer
                    if isinstance(curr_node, tuple) and curr_node[0] != self.data['depot']:
                        route_load += self.data['p'][curr_node[0]]
                    
                    if 'CFV' in vehicle:
                        # CFV costs and emissions
                        # Diesel fuel consumption
                        fuel_curb = (cfg.w_c * cfg.phi_param * cfg.sigma_param * distance +
                                   cfg.f_param * cfg.i_param * cfg.e_param * time +
                                   cfg.epsilon_param * cfg.phi_param * distance * (velocity ** 2)) * cfg.lambda_param
                        
                        fuel_load = cfg.phi_param * cfg.sigma_param * cfg.lambda_param * distance * route_load
                        
                        fuel_total = fuel_curb + fuel_load
                        
                        # Fuel cost
                        total_cost += cfg.f_fuel * fuel_total
                        
                        # Emissions
                        total_emissions += cfg.omega * fuel_total / 1000  # Convert to tonnes
                        
                        # Carbon tax
                        total_cost += cfg.beta * cfg.omega * fuel_total / 1000
                    
                    else:  # HGEV
                        # Energy consumption
                        energy = self.data['h'][vehicle] * distance
                        
                        # Recharging cost
                        total_cost += cfg.r * energy
                        
                        # Emissions from electricity
                        total_emissions += cfg.mu * energy / 1000  # Convert to tonnes
                    
                    # Operating costs
                    total_cost += self.data['O'][vehicle] * distance
        
        # HGEV infrastructure costs (simplified)
        hgev_used = any('HGEV' in v and solution.routes[v] for v in solution.routes)
        if hgev_used:
            total_cost += cfg.hgev_initiation_cost
        
        return total_cost, total_emissions
    
    # ========== REMOVAL OPERATORS ==========
    
    def remove_random_customer(self, solution: Solution, num_remove: int) -> List[int]:
        """Remove random customers from solution"""
        customers = list(solution.get_all_customers())
        num_remove = min(num_remove, len(customers))
        removed = random.sample(customers, num_remove)
        
        for customer in removed:
            solution.remove_customer(customer)
        
        return removed
    
    def remove_random_route(self, solution: Solution, num_remove: int) -> List[int]:
        """Remove random routes from solution"""
        removed_customers = []
        
        # Get non-empty vehicles
        vehicles_with_routes = [v for v, routes in solution.routes.items() if routes]
        
        if not vehicles_with_routes:
            return removed_customers
        
        num_remove = min(num_remove, len(vehicles_with_routes))
        vehicles_to_clear = random.sample(vehicles_with_routes, num_remove)
        
        for vehicle in vehicles_to_clear:
            # Get customers from this vehicle's routes
            for route in solution.routes[vehicle]:
                for node in route:
                    if isinstance(node, tuple) and node[0] != self.data['depot']:
                        if node[0] not in removed_customers:
                            removed_customers.append(node[0])
            
            # Clear vehicle routes
            solution.remove_route(vehicle)
        
        return removed_customers
    
    def remove_worst_cost(self, solution: Solution, num_remove: int) -> List[int]:
        """Remove customers with highest cost contribution"""
        customer_costs = {}
        
        # Calculate cost contribution for each customer
        for customer in solution.get_all_customers():
            # Create solution without this customer
            temp_solution = solution.copy()
            temp_solution.remove_customer(customer)
            
            # Calculate cost difference
            original_cost = solution.cost
            new_cost, _ = self.evaluate_solution(temp_solution)
            savings = original_cost - new_cost
            
            customer_costs[customer] = -savings  # Negative savings = high cost
        
        # Remove customers with highest cost
        sorted_customers = sorted(customer_costs.items(), key=lambda x: x[1], reverse=True)
        num_remove = min(num_remove, len(sorted_customers))
        removed = [c for c, _ in sorted_customers[:num_remove]]
        
        for customer in removed:
            solution.remove_customer(customer)
        
        return removed
    
    def remove_worst_emissions(self, solution: Solution, num_remove: int) -> List[int]:
        """Remove customers with highest emissions contribution"""
        customer_emissions = {}
        
        # Calculate emissions contribution for each customer
        for customer in solution.get_all_customers():
            temp_solution = solution.copy()
            temp_solution.remove_customer(customer)
            
            original_emissions = solution.emissions
            _, new_emissions = self.evaluate_solution(temp_solution)
            savings = original_emissions - new_emissions
            
            customer_emissions[customer] = -savings
        
        # Remove customers with highest emissions
        sorted_customers = sorted(customer_emissions.items(), key=lambda x: x[1], reverse=True)
        num_remove = min(num_remove, len(sorted_customers))
        removed = [c for c, _ in sorted_customers[:num_remove]]
        
        for customer in removed:
            solution.remove_customer(customer)
        
        return removed
    
    # ========== REPAIR OPERATORS ==========
    
    def repair_random(self, solution: Solution, removed_customers: List[int]):
        """Insert customers in random positions"""
        for customer in removed_customers:
            # Get all possible insertion positions
            positions = []
            
            for vehicle in self.data['K_list']:
                if not solution.routes[vehicle]:
                    # Create new route
                    positions.append((vehicle, -1, 0))
                else:
                    for route_idx, route in enumerate(solution.routes[vehicle]):
                        for pos in range(1, len(route)):
                            positions.append((vehicle, route_idx, pos))
            
            if positions:
                vehicle, route_idx, pos = random.choice(positions)
                self._insert_customer(solution, customer, vehicle, route_idx, pos)
    
    def repair_lowest_cost(self, solution: Solution, removed_customers: List[int]):
        """Insert customers in positions that minimize cost increase"""
        for customer in removed_customers:
            best_insertion = None
            best_cost_increase = float('inf')
            
            # Try all possible insertions
            for vehicle in self.data['K_list']:
                if not solution.routes[vehicle]:
                    # Try creating new route
                    temp_solution = solution.copy()
                    new_route = [self.depot, (customer, 1), self.depot]
                    temp_solution.routes[vehicle] = [new_route]
                    
                    new_cost, _ = self.evaluate_solution(temp_solution)
                    cost_increase = new_cost - solution.cost
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_insertion = (vehicle, -1, 0)
                else:
                    for route_idx, route in enumerate(solution.routes[vehicle]):
                        for pos in range(1, len(route)):
                            # Try insertion
                            temp_solution = solution.copy()
                            self._insert_customer(temp_solution, customer, vehicle, route_idx, pos)
                            
                            if temp_solution.is_feasible():
                                new_cost, _ = self.evaluate_solution(temp_solution)
                                cost_increase = new_cost - solution.cost
                                
                                if cost_increase < best_cost_increase:
                                    best_cost_increase = cost_increase
                                    best_insertion = (vehicle, route_idx, pos)
            
            if best_insertion:
                vehicle, route_idx, pos = best_insertion
                self._insert_customer(solution, customer, vehicle, route_idx, pos)
    
    def repair_lowest_emissions(self, solution: Solution, removed_customers: List[int]):
        """Insert customers in positions that minimize emissions increase"""
        for customer in removed_customers:
            best_insertion = None
            best_emissions_increase = float('inf')
            
            for vehicle in self.data['K_list']:
                if not solution.routes[vehicle]:
                    temp_solution = solution.copy()
                    new_route = [self.depot, (customer, 1), self.depot]
                    temp_solution.routes[vehicle] = [new_route]
                    
                    _, new_emissions = self.evaluate_solution(temp_solution)
                    emissions_increase = new_emissions - solution.emissions
                    
                    if emissions_increase < best_emissions_increase:
                        best_emissions_increase = emissions_increase
                        best_insertion = (vehicle, -1, 0)
                else:
                    for route_idx, route in enumerate(solution.routes[vehicle]):
                        for pos in range(1, len(route)):
                            temp_solution = solution.copy()
                            self._insert_customer(temp_solution, customer, vehicle, route_idx, pos)
                            
                            if temp_solution.is_feasible():
                                _, new_emissions = self.evaluate_solution(temp_solution)
                                emissions_increase = new_emissions - solution.emissions
                                
                                if emissions_increase < best_emissions_increase:
                                    best_emissions_increase = emissions_increase
                                    best_insertion = (vehicle, route_idx, pos)
            
            if best_insertion:
                vehicle, route_idx, pos = best_insertion
                self._insert_customer(solution, customer, vehicle, route_idx, pos)
    
    def repair_regret2_cost(self, solution: Solution, removed_customers: List[int]):
        """Insert customers using regret-2 heuristic based on cost"""
        while removed_customers:
            best_regret = -float('inf')
            best_customer = None
            best_insertion = None
            
            for customer in removed_customers:
                # Find two best insertion positions
                insertions = []
                
                for vehicle in self.data['K_list']:
                    if not solution.routes[vehicle]:
                        temp_solution = solution.copy()
                        new_route = [self.depot, (customer, 1), self.depot]
                        temp_solution.routes[vehicle] = [new_route]
                        new_cost, _ = self.evaluate_solution(temp_solution)
                        cost_increase = new_cost - solution.cost
                        insertions.append((cost_increase, vehicle, -1, 0))
                    else:
                        for route_idx, route in enumerate(solution.routes[vehicle]):
                            for pos in range(1, len(route)):
                                temp_solution = solution.copy()
                                self._insert_customer(temp_solution, customer, vehicle, route_idx, pos)
                                if temp_solution.is_feasible():
                                    new_cost, _ = self.evaluate_solution(temp_solution)
                                    cost_increase = new_cost - solution.cost
                                    insertions.append((cost_increase, vehicle, route_idx, pos))
                
                if len(insertions) >= 2:
                    insertions.sort()
                    regret = insertions[1][0] - insertions[0][0]
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_customer = customer
                        best_insertion = insertions[0][1:]  # vehicle, route_idx, pos
            
            if best_customer and best_insertion:
                vehicle, route_idx, pos = best_insertion
                self._insert_customer(solution, best_customer, vehicle, route_idx, pos)
                removed_customers.remove(best_customer)
            else:
                break
    
    def repair_regret2_emissions(self, solution: Solution, removed_customers: List[int]):
        """Insert customers using regret-2 heuristic based on emissions"""
        while removed_customers:
            best_regret = -float('inf')
            best_customer = None
            best_insertion = None
            
            for customer in removed_customers:
                insertions = []
                
                for vehicle in self.data['K_list']:
                    if not solution.routes[vehicle]:
                        temp_solution = solution.copy()
                        new_route = [self.depot, (customer, 1), self.depot]
                        temp_solution.routes[vehicle] = [new_route]
                        _, new_emissions = self.evaluate_solution(temp_solution)
                        emissions_increase = new_emissions - solution.emissions
                        insertions.append((emissions_increase, vehicle, -1, 0))
                    else:
                        for route_idx, route in enumerate(solution.routes[vehicle]):
                            for pos in range(1, len(route)):
                                temp_solution = solution.copy()
                                self._insert_customer(temp_solution, customer, vehicle, route_idx, pos)
                                if temp_solution.is_feasible():
                                    _, new_emissions = self.evaluate_solution(temp_solution)
                                    emissions_increase = new_emissions - solution.emissions
                                    insertions.append((emissions_increase, vehicle, route_idx, pos))
                
                if len(insertions) >= 2:
                    insertions.sort()
                    regret = insertions[1][0] - insertions[0][0]
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_customer = customer
                        best_insertion = insertions[0][1:]
            
            if best_customer and best_insertion:
                vehicle, route_idx, pos = best_insertion
                self._insert_customer(solution, best_customer, vehicle, route_idx, pos)
                removed_customers.remove(best_customer)
            else:
                break
    
    def _insert_customer(self, solution: Solution, customer: int, 
                        vehicle: str, route_idx: int, pos: int):
        """Helper to insert customer into solution"""
        customer_node = (customer, 1)
        
        if route_idx == -1:
            # Create new route
            new_route = [self.depot, customer_node, self.depot]
            solution.routes[vehicle] = [new_route]
        else:
            # Insert into existing route
            route = solution.routes[vehicle][route_idx]
            route.insert(pos, customer_node)
    
    def select_operator(self, weights: Dict) -> str:
        """Select operator using roulette wheel selection"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return random.choice(list(weights.keys()))
        
        rand = random.uniform(0, total_weight)
        cumsum = 0
        
        for operator, weight in weights.items():
            cumsum += weight
            if rand <= cumsum:
                return operator
        
        return list(weights.keys())[-1]
    
    def accept_solution(self, new_solution: Solution, current_solution: Solution) -> bool:
        """Simulated annealing acceptance criterion"""
        if new_solution.objective <= current_solution.objective:
            return True
        
        delta = new_solution.objective - current_solution.objective
        probability = math.exp(-delta / self.temperature)
        
        return random.random() < probability
    
    def update_weights(self):
        """Update operator weights based on recent performance"""
        segment_size = self.config.alns_segment_size
        decay = self.config.alns_decay
        
        if self.iteration % segment_size == 0 and self.iteration > 0:
            # Update removal operator weights
            for operator in self.removal_weights:
                if self.removal_uses[operator] > 0:
                    avg_score = self.removal_scores[operator] / self.removal_uses[operator]
                    self.removal_weights[operator] = (decay * self.removal_weights[operator] + 
                                                     (1 - decay) * avg_score)
                self.removal_scores[operator] = 0
                self.removal_uses[operator] = 0
            
            # Update repair operator weights
            for operator in self.repair_weights:
                if self.repair_uses[operator] > 0:
                    avg_score = self.repair_scores[operator] / self.repair_uses[operator]
                    self.repair_weights[operator] = (decay * self.repair_weights[operator] + 
                                                    (1 - decay) * avg_score)
                self.repair_scores[operator] = 0
                self.repair_uses[operator] = 0
    
    def solve(self, initial_solution: Solution = None) -> Solution:
        """
        Run ALNS algorithm
        
        Args:
            initial_solution: Starting solution (if None, construct greedy)
        
        Returns:
            Best solution found
        """
        print("\n" + "="*80)
        print("ADAPTIVE LARGE NEIGHBORHOOD SEARCH")
        print("="*80)
        
        # Initialize
        if initial_solution is None:
            self.current_solution = self.construct_greedy_solution()
        else:
            self.current_solution = initial_solution.copy()
        
        self.best_solution = self.current_solution.copy()
        
        print(f"\nRunning ALNS for {self.config.alns_iterations} iterations...")
        print(f"Initial objective: ${self.best_solution.cost:,.2f}\n")
        
        # Main loop
        for iteration in range(self.config.alns_iterations):
            self.iteration = iteration
            
            # Select operators
            removal_op = self.select_operator(self.removal_weights)
            repair_op = self.select_operator(self.repair_weights)
            
            # Apply removal
            new_solution = self.current_solution.copy()
            num_remove = random.randint(1, min(5, len(new_solution.get_all_customers())))
            
            if removal_op == 'random_customer':
                removed = self.remove_random_customer(new_solution, num_remove)
            elif removal_op == 'random_route':
                removed = self.remove_random_route(new_solution, 1)
            elif removal_op == 'worst_cost':
                removed = self.remove_worst_cost(new_solution, num_remove)
            elif removal_op == 'worst_emissions':
                removed = self.remove_worst_emissions(new_solution, num_remove)
            
            # Apply repair
            if repair_op == 'random':
                self.repair_random(new_solution, removed)
            elif repair_op == 'lowest_cost':
                self.repair_lowest_cost(new_solution, removed)
            elif repair_op == 'lowest_emissions':
                self.repair_lowest_emissions(new_solution, removed)
            elif repair_op == 'regret2_cost':
                self.repair_regret2_cost(new_solution, removed)
            elif repair_op == 'regret2_emissions':
                self.repair_regret2_emissions(new_solution, removed)
            
            # Evaluate new solution
            new_solution.cost, new_solution.emissions = self.evaluate_solution(new_solution)
            new_solution.objective = new_solution.cost
            
            # Update operator scores
            self.removal_uses[removal_op] += 1
            self.repair_uses[repair_op] += 1
            
            score = 0
            if new_solution.objective < self.best_solution.objective:
                score = self.config.alns_w1  # New best
                self.best_solution = new_solution.copy()
            elif new_solution.objective < self.current_solution.objective:
                score = self.config.alns_w2  # Better than current
            elif self.accept_solution(new_solution, self.current_solution):
                score = self.config.alns_w3  # Accepted
            else:
                score = self.config.alns_w4  # Rejected
            
            self.removal_scores[removal_op] += score
            self.repair_scores[repair_op] += score
            
            # Accept/reject solution
            if self.accept_solution(new_solution, self.current_solution):
                self.current_solution = new_solution
            
            # Update weights
            self.update_weights()
            
            # Update temperature
            self.temperature *= self.config.alns_cooling_rate
            
            # Track history
            self.history['iteration'].append(iteration)
            self.history['cost'].append(self.current_solution.cost)
            self.history['emissions'].append(self.current_solution.emissions)
            self.history['best_cost'].append(self.best_solution.cost)
            self.history['temperature'].append(self.temperature)
            
            # Progress update
            if iteration % 100 == 0:
                print(f"Iteration {iteration:5d}: "
                      f"Current=${self.current_solution.cost:,.0f}, "
                      f"Best=${self.best_solution.cost:,.0f}, "
                      f"T={self.temperature:.2f}")
        
        print(f"\n✓ ALNS complete!")
        print(f"Best solution: ${self.best_solution.cost:,.2f}")
        print(f"Emissions: {self.best_solution.emissions:.2f} tons CO2e")
        
        return self.best_solution


if __name__ == "__main__":
    print("ALNS Metaheuristic Module")
    print("Use this module with optimized_solver.py or standalone")
