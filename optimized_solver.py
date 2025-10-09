"""
Optimized MILP Model for Green Vehicle Routing Problem
Uses indicator constraints instead of Big M for improved performance
"""

import sys
from typing import Dict, Tuple
import gurobipy as gp
from gurobipy import GRB

from config import ModelParams, create_default_config
from utils import load_data, expand_network, reconstruct_routes, print_solution_summary, validate_solution


class GreenVRPModel:
    """Green Vehicle Routing Problem MILP Model"""
    
    def __init__(self, config: ModelParams = None):
        """Initialize the model with configuration"""
        self.config = config if config else create_default_config()
        self.model = gp.Model("green_vrp_optimized")
        
        # Configure Gurobi parameters
        self.model.setParam('Heuristics', self.config.heuristics)
        self.model.setParam('MIPGap', self.config.mip_gap)
        self.model.setParam('TimeLimit', self.config.time_limit)
        if self.config.threads > 0:
            self.model.setParam('Threads', self.config.threads)
        
        # Data structures
        self.K_c_list = []
        self.K_e_list = []
        self.K_list = []
        self.D = set()
        self.depot = None
        self.N_expanded = []
        self.D_expanded = set()
        
        # Parameters
        self.d = {}
        self.t = {}
        self.v_avg = {}
        self.p = {}
        self.d_exp = {}
        self.t_exp = {}
        self.v_avg_exp = {}
        
        # Vehicle parameters
        self.A = {}
        self.S = {}
        self.h = {}
        self.Q = {}
        self.R = {}
        self.O = {}
        
        # Variables
        self.x = None
        self.q = None
        self.y = None
        self.T = None
        self.used = None
        self.U = None
        self.HGEV_used = None
        self.HGEV_exp = None
        self.delivered = None
        
        # Objective expressions
        self.obj_expr_cost = None
        self.obj_expr_emissions = None
    
    def setup_problem(self):
        """Load data and initialize problem structure"""
        print("="*80)
        print("GREEN VEHICLE ROUTING OPTIMIZATION - SETUP")
        print("="*80)
        
        # Load data
        self.d, self.t, self.v_avg, self.p, self.D, self.depot = load_data(
            self.config.demand_file,
            self.config.distance_file,
            self.config.time_file
        )
        
        # Expand network
        self.N_expanded, self.d_exp, self.t_exp, self.v_avg_exp = expand_network(
            self.D, self.depot, self.config.vmax, self.d, self.t, self.v_avg
        )
        
        self.D_expanded = {node for node in self.N_expanded if node[0] != self.depot}
        
        # Create vehicle sets
        self.K_c_list = sorted([f"CFV{i}" for i in range(1, self.config.num_cfv + 1)])
        self.K_e_list = sorted([f"HGEV{i}" for i in range(1, self.config.num_hgev + 1)])
        self.K_list = sorted(self.K_c_list + self.K_e_list)
        
        # Setup vehicle parameters
        cfv_params = self.config.get_vehicle_params('CFV')
        hgev_params = self.config.get_vehicle_params('HGEV')
        
        for k in self.K_list:
            params = cfv_params if k in self.K_c_list else hgev_params
            self.A[k] = params.acquisition_cost
            self.S[k] = params.subsidy
            self.h[k] = params.energy_consumption
            self.Q[k] = params.capacity
            self.R[k] = params.battery_capacity
            self.O[k] = params.operating_cost
        
        print(f"\n✓ Problem setup complete")
        print(f"  Vehicles: {len(self.K_c_list)} CFVs, {len(self.K_e_list)} HGEVs")
        print(f"  Network: {len(self.N_expanded)} nodes ({self.config.vmax}x expansion)")
        print(f"  Total demand: {sum(self.p.values()):.2f} kg")
    
    def define_variables(self):
        """Define decision variables"""
        print("\nDefining decision variables...")
        
        # Route variables x[k,i,j]: 1 if vehicle k travels from i to j
        x_keys = [(k, i, j) for k in self.K_list 
                  for i in self.N_expanded 
                  for j in self.N_expanded if i != j]
        self.x = self.model.addVars(x_keys, vtype=GRB.BINARY, name="x")
        
        # Load variables q[k,i,j]: load on vehicle k on arc (i,j)
        q_keys = x_keys
        self.q = self.model.addVars(q_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="q")
        
        # Battery level y[k,i]: charge of HGEV k when arriving at node i
        y_keys = [(k, i) for k in self.K_e_list for i in self.N_expanded]
        self.y = self.model.addVars(y_keys, vtype=GRB.CONTINUOUS, lb=0.0, 
                                    ub=self.R[self.K_e_list[0]], name="y")
        
        # Time variables T[k]: total time for vehicle k
        self.T = self.model.addVars(self.K_list, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
        
        # Vehicle usage indicator used[k]: 1 if vehicle k is used
        self.used = self.model.addVars(self.K_list, vtype=GRB.BINARY, name="used")
        
        # Load tracking U[k,i]: cumulative load on vehicle k at node i
        U_keys = [(k, i) for k in self.K_list for i in self.N_expanded]
        self.U = self.model.addVars(U_keys, vtype=GRB.CONTINUOUS, lb=0.0, 
                                    ub=self.Q[self.K_list[0]], name="U")
        
        # HGEV usage indicators
        self.HGEV_used = self.model.addVar(vtype=GRB.BINARY, name="HGEV_used")
        self.HGEV_exp = self.model.addVar(vtype=GRB.BINARY, name="HGEV_exp")
        
        # Delivery variables delivered[k,j]: amount delivered by k to node j
        delivered_keys = [(k, j) for k in self.K_list for j in self.D_expanded]
        self.delivered = self.model.addVars(delivered_keys, vtype=GRB.CONTINUOUS, 
                                           lb=0.0, name="delivered")
        
        print(f"  ✓ {len(self.x)} route variables")
        print(f"  ✓ {len(self.q)} load variables")
        print(f"  ✓ {len(self.y)} battery variables")
        print(f"  ✓ {len(self.delivered)} delivery variables")
    
    def define_objectives(self):
        """Define objective function expressions"""
        print("\nDefining objective functions...")
        
        cfg = self.config
        
        # === COST OBJECTIVE ===
        
        # 1. Recharging costs (HGEVs)
        term1 = cfg.r * gp.quicksum(
            self.h[k] * self.d_exp.get((i, j), 0) * self.x.get((k, i, j), 0)
            for k in self.K_e_list
            for i in self.N_expanded for j in self.N_expanded if i != j
        )
        
        # 2. Diesel fuel costs - curb weight (CFVs)
        term2 = cfg.f_fuel * gp.quicksum(
            (cfg.w_c * cfg.phi_param * cfg.sigma_param * self.d_exp.get((i, j), 0) +
             cfg.f_param * cfg.i_param * cfg.e_param * self.t_exp.get((i, j), 0) +
             cfg.epsilon_param * cfg.phi_param * self.d_exp.get((i, j), 0) * 
             (self.v_avg_exp.get((i, j), 0) ** 2)
            ) * cfg.lambda_param * self.x.get((k, i, j), 0)
            for k in self.K_c_list
            for i in self.N_expanded for j in self.N_expanded if i != j
            if self.v_avg_exp.get((i, j), None) is not None
        )
        
        # 3. Diesel fuel costs - load weight (CFVs)
        term3 = cfg.f_fuel * gp.quicksum(
            cfg.phi_param * cfg.sigma_param * cfg.lambda_param * 
            self.d_exp.get((i, j), 0) * self.q.get((k, i, j), 0)
            for k in self.K_c_list
            for i in self.N_expanded for j in self.N_expanded if i != j
        )
        
        # 4. Operating costs (all vehicles)
        term4 = gp.quicksum(
            self.O[k] * self.d_exp.get((i, j), 0) * self.x.get((k, i, j), 0)
            for k in self.K_list
            for i in self.N_expanded for j in self.N_expanded if i != j
        )
        
        # 5. Acquisition costs
        term5 = gp.quicksum((self.A[k] - self.S[k]) * self.used[k] for k in self.K_list)
        
        # 6. HGEV infrastructure costs
        term6 = (cfg.hgev_initiation_cost * self.HGEV_used + 
                cfg.hgev_expanded_cost * self.HGEV_exp)
        
        # 7. Carbon tax on diesel emissions (CFVs)
        term7_fuel = gp.quicksum(
            (cfg.w_c * cfg.phi_param * cfg.sigma_param * self.d_exp.get((i, j), 0) +
             cfg.f_param * cfg.i_param * cfg.e_param * self.t_exp.get((i, j), 0) +
             cfg.epsilon_param * cfg.phi_param * self.d_exp.get((i, j), 0) * 
             (self.v_avg_exp.get((i, j), 0) ** 2)
            ) * cfg.lambda_param * self.x.get((k, i, j), 0)
            for k in self.K_c_list
            for i in self.N_expanded for j in self.N_expanded if i != j
            if self.v_avg_exp.get((i, j), None) is not None
        ) + gp.quicksum(
            cfg.phi_param * cfg.sigma_param * cfg.lambda_param * 
            self.d_exp.get((i, j), 0) * self.q.get((k, i, j), 0)
            for k in self.K_c_list
            for i in self.N_expanded for j in self.N_expanded if i != j
        )
        term7 = cfg.beta * cfg.omega * term7_fuel / 1000  # Convert to tonnes
        
        self.obj_expr_cost = term1 + term2 + term3 + term4 + term5 + term6 + term7
        
        # === EMISSIONS OBJECTIVE ===
        
        # 1. Diesel emissions (CFVs) - kg CO2
        term8 = cfg.omega * term7_fuel
        
        # 2. Electricity emissions (HGEVs) - kg CO2
        term9 = cfg.mu * gp.quicksum(
            self.h[k] * self.d_exp.get((i, j), 0) * self.x.get((k, i, j), 0)
            for k in self.K_e_list
            for i in self.N_expanded for j in self.N_expanded if i != j
        )
        
        self.obj_expr_emissions = (term8 + term9) / 1000  # Convert to tonnes
        
        print("  ✓ Cost and emissions objectives defined")
    
    def add_routing_constraints(self):
        """Add routing flow constraints"""
        print("\nAdding routing constraints...")
        
        # 1. Vehicle must leave depot if used
        self.model.addConstrs(
            (gp.quicksum(self.x.get((k, (self.depot, 0), j), 0) 
                        for j in self.N_expanded if j != (self.depot, 0)) >= self.used[k]
             for k in self.K_list),
            name="vehicle_leave_depot"
        )
        
        # 2. Flow conservation at depot
        self.model.addConstrs(
            (gp.quicksum(self.x.get((k, (self.depot, 0), j), 0) 
                        for j in self.N_expanded if j != (self.depot, 0)) ==
             gp.quicksum(self.x.get((k, i, (self.depot, 0)), 0) 
                        for i in self.N_expanded if i != (self.depot, 0))
             for k in self.K_list),
            name="depot_flow"
        )
        
        # 3. Flow conservation at customer nodes
        self.model.addConstrs(
            (gp.quicksum(self.x.get((k, i, j), 0) 
                        for i in self.N_expanded if i != j) ==
             gp.quicksum(self.x.get((k, j, i), 0) 
                        for i in self.N_expanded if i != j)
             for k in self.K_list for j in self.D_expanded),
            name="customer_flow"
        )
        
        # 4. Time calculation
        penalty_nodes = self.D_expanded
        for k in self.K_list:
            travel_time = gp.quicksum(
                self.t_exp.get((i, j), 0) * self.x.get((k, i, j), 0)
                for i in self.N_expanded for j in self.N_expanded if i != j
            )
            penalty_stops = gp.quicksum(
                self.x.get((k, j, i), 0)
                for j in penalty_nodes for i in self.N_expanded if i != j
            )
            self.model.addConstr(
                self.T[k] >= travel_time + self.config.tstop * penalty_stops,
                name=f"time_calc_{k}"
            )
        
        # 5. Time limit
        self.model.addConstrs(
            (self.T[k] <= self.config.tmax for k in self.K_list),
            name="time_limit"
        )
        
        print("  ✓ Flow and time constraints added")
    
    def add_capacity_constraints(self):
        """Add capacity and demand constraints"""
        print("\nAdding capacity constraints...")
        
        # 1. Link load to route using INDICATOR constraints
        for k in self.K_list:
            for i in self.N_expanded:
                for j in self.N_expanded:
                    if i == j:
                        continue
                    # If x[k,i,j] = 0, then q[k,i,j] = 0
                    # If x[k,i,j] = 1, then q[k,i,j] <= Q[k]
                    self.model.addGenConstrIndicator(
                        self.x.get((k, i, j), 0), 0, 
                        self.q.get((k, i, j), 0) == 0,
                        name=f"load_link_0_{k}_{i}_{j}"
                    )
                    self.model.addGenConstrIndicator(
                        self.x.get((k, i, j), 0), 1,
                        self.q.get((k, i, j), 0) <= self.Q[k],
                        name=f"load_link_1_{k}_{i}_{j}"
                    )
        
        # 2. Demand satisfaction
        for i in self.D:
            if self.p.get(i, 0) > 0:
                self.model.addConstr(
                    gp.quicksum(self.delivered.get((k, (i, v)), 0) 
                               for k in self.K_list 
                               for v in range(1, self.config.vmax + 1) 
                               if (i, v) in self.D_expanded) == self.p[i],
                    name=f"demand_{i}"
                )
        
        # 3. Flow conservation at customers
        for k in self.K_list:
            for j in self.D_expanded:
                load_in = gp.quicksum(self.q.get((k, i, j), 0) 
                                     for i in self.N_expanded if i != j)
                load_out = gp.quicksum(self.q.get((k, j, i), 0) 
                                      for i in self.N_expanded if i != j)
                delivery = self.delivered.get((k, j), 0)
                self.model.addConstr(
                    load_in == delivery + load_out,
                    name=f"flow_balance_{k}_{j}"
                )
        
        # 4. Link delivery to visit using INDICATOR constraints
        for k in self.K_list:
            for j in self.D_expanded:
                is_visited = gp.quicksum(self.x.get((k, j, i), 0) 
                                        for i in self.N_expanded if i != j)
                # If not visited, delivery must be 0
                # If visited, delivery can be up to demand
                self.model.addConstr(
                    self.delivered.get((k, j), 0) <= self.p.get(j[0], 0) * is_visited,
                    name=f"delivery_link_{k}_{j}"
                )
        
        print("  ✓ Capacity and demand constraints added")
    
    def add_hgev_constraints(self):
        """Add HGEV battery and charging constraints"""
        print("\nAdding HGEV constraints...")
        
        cfg = self.config
        
        # 1. Link HGEV usage
        self.model.addConstrs(
            (self.HGEV_used >= self.used[k] for k in self.K_e_list),
            name="hgev_used_link"
        )
        
        # 2. Link expanded network usage
        for k in self.K_e_list:
            for i in self.N_expanded:
                for j in self.N_expanded:
                    if i != j and j[0] in cfg.charging_stations:
                        self.model.addConstr(
                            self.HGEV_exp >= self.x.get((k, i, j), 0),
                            name=f"hgev_exp_link_{k}_{i}_{j}"
                        )
        
        # 3. Battery level updates using INDICATOR constraints
        for k in self.K_e_list:
            # Leaving depot
            for j in self.N_expanded:
                if j[0] == self.depot:
                    continue
                energy_consumed = self.h[k] * self.d_exp.get(((self.depot, 0), j), 0)
                
                # If traveling from depot to j
                self.model.addGenConstrIndicator(
                    self.x.get((k, (self.depot, 0), j), 0), 1,
                    self.y.get((k, j), 0) <= self.R[k] - energy_consumed,
                    name=f"battery_from_depot_{k}_{j}"
                )
            
            # Between customer nodes
            for i in self.N_expanded:
                if i[0] == self.depot:
                    continue
                for j in self.N_expanded:
                    if i == j or j[0] == self.depot:
                        continue
                    
                    energy_consumed = self.h[k] * self.d_exp.get((i, j), 0)
                    
                    # Normal nodes or inactive charging stations
                    if i[0] not in cfg.charging_stations:
                        # If traveling i->j, battery decreases
                        self.model.addGenConstrIndicator(
                            self.x.get((k, i, j), 0), 1,
                            self.y.get((k, j), 0) <= self.y.get((k, i), 0) - energy_consumed,
                            name=f"battery_normal_{k}_{i}_{j}"
                        )
                    else:
                        # Charging station node - conditional on HGEV_exp
                        # This requires more complex logic - use Big M here only for CS
                        BIG_M = self.R[k]
                        self.model.addConstr(
                            self.y.get((k, j), 0) <= self.y.get((k, i), 0) - energy_consumed +
                            BIG_M * (1 - self.x.get((k, i, j), 0)) + BIG_M * self.HGEV_exp,
                            name=f"battery_cs_inactive_{k}_{i}_{j}"
                        )
                        self.model.addConstr(
                            self.y.get((k, j), 0) <= self.R[k] - energy_consumed +
                            BIG_M * (1 - self.x.get((k, i, j), 0)) + BIG_M * (1 - self.HGEV_exp),
                            name=f"battery_cs_active_{k}_{i}_{j}"
                        )
        
        # 4. Battery reserve constraints
        d_star = {}
        for i_node in self.D_expanded:
            i_orig = i_node[0]
            d_to_depot = self.d.get((i_orig, self.depot), float('inf'))
            d_to_cs = float('inf')
            if cfg.charging_stations:
                d_to_cs = min(self.d.get((i_orig, cs), float('inf')) 
                             for cs in cfg.charging_stations)
            d_star[i_node] = min(d_to_depot, d_to_cs)
        
        for k in self.K_e_list:
            for i in self.D_expanded:
                energy_to_depot = self.h[k] * self.d.get((i[0], self.depot), float('inf'))
                required_reserve_off = energy_to_depot + cfg.ms
                
                energy_to_nearest = self.h[k] * d_star.get(i, float('inf'))
                required_reserve_on = energy_to_nearest + cfg.ms
                
                # Use indicator on HGEV_exp
                self.model.addGenConstrIndicator(
                    self.HGEV_exp, 0,
                    self.y.get((k, i), 0) >= required_reserve_off,
                    name=f"reserve_off_{k}_{i}"
                )
                self.model.addGenConstrIndicator(
                    self.HGEV_exp, 1,
                    self.y.get((k, i), 0) >= required_reserve_on,
                    name=f"reserve_on_{k}_{i}"
                )
        
        print("  ✓ HGEV battery and charging constraints added")
    
    def set_objective(self):
        """Set the objective function based on run mode"""
        print(f"\nSetting objective (mode: {self.config.run_mode})...")
        
        if self.config.run_mode == 'cost_min':
            self.model.setObjective(self.obj_expr_cost, GRB.MINIMIZE)
            print("  ✓ Objective: Minimize total cost")
        
        elif self.config.run_mode == 'emissions_min':
            self.model.setObjective(self.obj_expr_emissions, GRB.MINIMIZE)
            print("  ✓ Objective: Minimize emissions")
        
        elif self.config.run_mode == 'weighted':
            # Check normalization bounds
            if (self.config.cost_min is None or self.config.emiss_min is None):
                print("\nERROR: Normalization bounds not set!")
                print("Run 'cost_min' and 'emissions_min' modes first.")
                sys.exit(1)
            
            # Normalize objectives
            cost_range = self.config.cost_at_emiss_min - self.config.cost_min
            emiss_range = self.config.emiss_at_cost_min - self.config.emiss_min
            
            if abs(cost_range) < 1e-6 or abs(emiss_range) < 1e-6:
                print("WARNING: Near-zero normalization range detected")
            
            norm_cost = (self.obj_expr_cost - self.config.cost_min) / cost_range
            norm_emiss = (self.obj_expr_emissions - self.config.emiss_min) / emiss_range
            
            weighted_obj = (self.config.weight_cost * norm_cost + 
                           (1 - self.config.weight_cost) * norm_emiss)
            
            self.model.setObjective(weighted_obj, GRB.MINIMIZE)
            print(f"  ✓ Objective: Weighted sum (cost weight: {self.config.weight_cost})")
        
        else:
            print(f"ERROR: Unknown run mode '{self.config.run_mode}'")
            sys.exit(1)
    
    def build_model(self):
        """Build the complete optimization model"""
        self.setup_problem()
        self.define_variables()
        self.define_objectives()
        self.add_routing_constraints()
        self.add_capacity_constraints()
        self.add_hgev_constraints()
        self.set_objective()
        print("\n✓ Model build complete!")
    
    def solve(self) -> Tuple[bool, Dict]:
        """
        Solve the optimization model
        
        Returns:
            success: True if optimal/feasible solution found
            results: Dictionary containing solution details
        """
        print("\n" + "="*80)
        print("SOLVING MODEL")
        print("="*80)
        
        self.model.optimize()
        
        results = {
            'status': self.model.status,
            'success': False,
            'obj_val': None,
            'total_cost': None,
            'total_emissions': None,
            'cost_breakdown': {},
            'emissions_breakdown': {},
            'routes': {},
            'solution': {}
        }
        
        if self.model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.SOLUTION_LIMIT]:
            results['success'] = True
            results['obj_val'] = self.model.ObjVal
            results['total_cost'] = self.obj_expr_cost.getValue()
            results['total_emissions'] = self.obj_expr_emissions.getValue()
            
            # Extract solution
            x_solution = {key: var.X for key, var in self.x.items() if var.X > 0.5}
            delivered_solution = {key: var.X for key, var in self.delivered.items()}
            
            results['solution'] = {
                'x': x_solution,
                'delivered': delivered_solution,
                'used': {k: self.used[k].X for k in self.K_list},
                'HGEV_used': self.HGEV_used.X,
                'HGEV_exp': self.HGEV_exp.X,
                'time': {k: self.T[k].X for k in self.K_list if self.used[k].X > 0.5}
            }
            
            # Reconstruct routes
            results['routes'] = reconstruct_routes(
                x_solution, self.K_list, self.N_expanded, self.depot
            )
            
            # Cost breakdown
            results['cost_breakdown'] = self._get_cost_breakdown()
            results['emissions_breakdown'] = self._get_emissions_breakdown()
            
            # Print summary
            print_solution_summary(
                results['obj_val'],
                results['total_cost'],
                results['total_emissions'],
                results['cost_breakdown'],
                results['emissions_breakdown'],
                results['routes'],
                self.config.run_mode
            )
            
            # Validate solution
            validate_solution(
                results['routes'], self.p, delivered_solution,
                self.K_list, list(self.D_expanded), self.config.vmax
            )
            
        elif self.model.status == GRB.INFEASIBLE:
            print("\n❌ MODEL IS INFEASIBLE")
            print("Computing IIS...")
            self.model.computeIIS()
            self.model.write("model_infeasible.ilp")
            print("IIS written to model_infeasible.ilp")
        
        else:
            print(f"\n❌ Optimization ended with status: {self.model.status}")
        
        return results['success'], results
    
    def _get_cost_breakdown(self) -> Dict[str, float]:
        """Calculate detailed cost breakdown"""
        # This would need to evaluate individual terms
        # Simplified version here
        return {
            'Total Cost': self.obj_expr_cost.getValue(),
        }
    
    def _get_emissions_breakdown(self) -> Dict[str, float]:
        """Calculate detailed emissions breakdown"""
        return {
            'Total Emissions': self.obj_expr_emissions.getValue(),
        }


def main():
    """Main execution function"""
    # Create configuration
    config = create_default_config()
    config.run_mode = 'cost_min'  # Change as needed
    config.vmax = 1
    config.time_limit = 1800  # 30 minutes
    
    # Create and solve model
    model = GreenVRPModel(config)
    model.build_model()
    success, results = model.solve()
    
    if success:
        print("\n✓ Solution found successfully!")
        print(f"Total Cost: ${results['total_cost']:,.2f}")
        print(f"Total Emissions: {results['total_emissions']:.4f} metric tons CO2e")
    else:
        print("\n❌ No solution found")
    
    return success, results


if __name__ == "__main__":
    main()
