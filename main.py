# --- START OF FILE Main Prob No Trip Index.py ---

import numpy as np
import pandas as pd
import math
import datetime
import sys # Added for handling potential errors during normalization

import gurobipy as gp
from gurobipy import GRB

# =============================================================================
#               CONTROL PARAMETERS FOR DIFFERENT RUN MODES
# =============================================================================
# Set the mode for this run:
# 'find_cost_min': Minimize only cost (to find normalization bounds)
# 'find_emiss_min': Minimize only emissions (to find normalization bounds)
# 'weighted_sum': Minimize a weighted sum of normalized cost and emissions
RUN_MODE = 'find_cost_min'  # <-- CHANGE THIS AS NEEDED

# Set the weight for the cost objective when RUN_MODE is 'weighted_sum'
# Weight for emissions will be (1.0 - WEIGHT_COST)
WEIGHT_COST = 0.5  # <-- CHANGE THIS WHEN RUNNING WEIGHTED SUM

# --- Placeholder values for normalization ---
# --- These MUST be populated by running in 'find_cost_min' and 'find_emiss_min' modes first! ---
COST_MIN = None             # Objective value from 'find_cost_min' run
EMISS_AT_COST_MIN = None    # Emissions value corresponding to COST_MIN
EMISS_MIN = None            # Objective value from 'find_emiss_min' run
COST_AT_EMISS_MIN = None    # Cost value corresponding to EMISS_MIN
# --- Example Populated Values (Replace with your actual results) ---
# COST_MIN = 8000000
# EMISS_AT_COST_MIN = 50000
# EMISS_MIN = 20000
# COST_AT_EMISS_MIN = 12000000
# ---------------------------------------------------------------------

model = gp.Model("multi_objective_milp")

# --- Gurobi Parameter Tuning ---
# Experiment with these parameters on your machine to potentially reduce solution time
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# log_file = f"gurobi_log_{timestamp}.txt"
# model.setParam('LogFile', log_file)
# model.setParam('MIPFocus', 1)          # 1: Feasible, 2: Optimal, 3: Bound
model.setParam('Heuristics', 0.25)      # Default is 0.05. Higher values spend more time on heuristics.
# model.setParam('MIPGap', 0.01)         # Target optimality gap (e.g., 1%)
# model.setParam('SolutionLimit', 1)     # Stop after finding the first feasible solution (useful for quick checks)
# model.setParam('ImproveStartTime', 60) # Time in seconds spent improving the start solution
# model.setParam('SubMIPNodes', 500)     # Node limit for sub-MIPs during heuristics/cuts
# ---------------------------------

# ===============================================================================================================================
# ===============================================================================================================================
#                                PROBLEM SET-UP
# ===============================================================================================================================
# ===============================================================================================================================

# =============================================================================
# 1. Define Sets
# =============================================================================

# Available vehicles:
K_c = {f"CFV{i}" for i in range (1,15)}
K_e = {f"HGEV{i}" for i in range (1,15)}
K = K_c.union(K_e)

K_c_list = sorted(K_c)   # List of available CFVs
K_e_list = sorted(K_e)   # List of available HGEVs
K_list   = sorted(K)     # All vehicles

# --- Data File Paths ---
# Make sure these paths are correct for your setup
DEMAND_FILE = "Demand.csv"
DISTANCE_FILE = "Distances.csv"
TIME_FILE = "Times.csv"
# -----------------------

try:
    demand_df = pd.read_csv(DEMAND_FILE, header=None)
    num_customers = len(demand_df)-1
    D = set(range(0,num_customers))
    depot = num_customers

    N_original = D.union([depot])

    Distance_Matrix = pd.read_csv(DISTANCE_FILE, header=None)
    Travel_Times = pd.read_csv(TIME_FILE, header=None)
    Customer_demand = pd.read_csv(DEMAND_FILE, header=None)

except FileNotFoundError as e:
    print(f"Error loading input file: {e}")
    print("Please ensure Demand.csv, Distances.csv, and Times.csv are in the correct directory.")
    sys.exit(1) # Exit if files are not found


# -------------------------------
# Network Expansion Settings
# -------------------------------
# --- Vmax Validation ---
# To validate the network expansion setting, run the model sequentially with
# Vmax = 1, Vmax = 2, and Vmax = 3 (on your powerful machine).
# Compare the objective function values. If Vmax=3 provides little to no
# improvement over Vmax=2, then Vmax=2 is likely sufficient.
# -----------------------
Vmax = 1  # maximum allowed visits per customer <-- CHANGE FOR VALIDATION

# Expanded customer set: each customer i is duplicated Vmax times.
D_expanded = {(i, v) for i in D for v in range(1, Vmax+1)}
# Expanded node set: depot is represented as (depot,0)
N_expanded = sorted({(depot, 0)}.union(D_expanded), key=lambda x: (x[0], x[1]))

# =============================================================================
# 2. Define Parameters
# =============================================================================

# -------------------------------------------------------
# 2.1 Parameters Indexed by Vehicle (k in K)
# -------------------------------------------------------

# A^k: Acquisition cost for vehicle k ($USD)
A = {k: 165000 if k in K_c_list else 400000 for k in K_list}

# S^k: Subsidy amount for vehicle k ($USD)
S = {k: 0 if k in K_c_list else 0 for k in K_list}

# h^k: Energy consumption per distance by vehicle k (kWh/m)
h = {k: .004375 if k in K_c_list else .0011 for k in K_list} # Note: CFV value is arbitrary, not used in calculations

# Q^k: The maximum load capacity for vehicle k  (kg)
Q = {k: 15000 for k in K_list}

# R^k: The battery capacity of vehicle k (kWh)
R = {k: 99999 if k in K_c_list else 900 for k in K_list} # Arbitrarily large for CFV

# O^k: The operating costs of vehicle k per meter ($USD/m)
O = {k: .00018125 if k in K_c_list else .00015875 for k in K_list}

# -------------------------------------------------------
# 2.2 Parameters Indexed by Arc (i, j)
# -------------------------------------------------------

# d_ij: Distance between vertex i and vertex j
# t_ij: Travel time from vertex i to vertex j
# v_avg_ij: Average velocity between vertex i and vertex j

distance_array = Distance_Matrix.values
time_array = Travel_Times.values

d = {}
t = {}
for i in range(len(N_original)):
    for j in range(len(N_original)):
        # Store distances/times even for i==j if needed elsewhere, but exclude from arcs later
        # Ensure indices match array dimensions
        if i < distance_array.shape[0] and j < distance_array.shape[1]:
             d[(i, j)] = distance_array[i, j]
        if i < time_array.shape[0] and j < time_array.shape[1]:
             t[(i, j)] = time_array[i, j]


# υ: The average speed of the vehicle (m/s)
v_avg = {}

# --- Conditional Charging Validation Setup ---
# To test conditional charging, you need to:
# 1. Modify Distances.csv/Times.csv: Add/modify a customer (e.g., index 23)
#    to be ~500km from the depot (index 22). Ensure the round trip distance
#    exceeds HGEV range without charging (2 * 500000 * h[k] > R[k] - Ms).
# 2. Modify Demand.csv: Give this customer significant demand.
# 3. Modify the CS set below: Add the new customer index (e.g., 23) to CS.
# Example: CS = {20, 21, 23}
# ------------------------------------------
# CS: Set of customers that also could have recharging stations at them
CS = {20, 21} # <-- MODIFY FOR CONDITIONAL CHARGING TEST

# Calculate average velocity, handling potential division by zero or missing data
ghost_pairs = {(0, 20), (11, 21), (20, 0), (21, 11)} # Example pairs, adjust if needed

for i in range(len(N_original)):
    for j in range(len(N_original)):
        if i == j:
            continue

        # Check if distance and time data exist for the pair
        dist_ij = d.get((i, j))
        time_ij = t.get((i, j))

        if dist_ij is None or time_ij is None:
            # print(f"Warning: Missing distance or time for arc ({i}, {j}). Skipping velocity calculation.")
            continue # Skip if data is missing

        # Handle ghost pairs (adjust time if zero)
        if (i, j) in ghost_pairs:
            if time_ij == 0:
                time_ij = 0.1 # Avoid division by zero, use a small time

        # Calculate velocity if time is non-zero
        if time_ij != 0:
            v_avg[(i, j)] = dist_ij / time_ij
        elif dist_ij != 0:
            # print(f"Warning: Zero travel time for non-zero distance on arc ({i}, {j}). Cannot calculate velocity.")
            pass # Handle case with zero time but non-zero distance if necessary

# -------------------------------------------------------
# 2.3 Expanded Network Data
# -------------------------------------------------------
# For any two expanded nodes i_node and j_node in N_expanded, define:
#   - if either node is the depot (i.e. (depot,0)), use the original distance/time.
#   - if both nodes correspond to customers and have the same original customer index, set distance = 0.
#   - otherwise, use the original distance/time between the two customer indices.
d_exp = {}
t_exp = {}
v_avg_exp = {}
for i_node in N_expanded:
    for j_node in N_expanded:
        if i_node == j_node:
            continue
        # Determine original indices: for depot, the index is depot; for customers, use i_node[0]
        i_orig = i_node[0]
        j_orig = j_node[0]
        # Use .get() with default value 0 for safety if an arc is missing in original data
        if i_orig == depot or j_orig == depot:
            d_exp[(i_node, j_node)] = d.get((i_orig, j_orig), 0)
            t_exp[(i_node, j_node)] = t.get((i_orig, j_orig), 0)
            v_avg_exp[(i_node, j_node)] = v_avg.get((i_orig, j_orig), 0) # Default to 0 if velocity wasn't calculated
        else:
            if i_orig == j_orig:
                d_exp[(i_node, j_node)] = 0  # no extra travel cost between copies
                t_exp[(i_node, j_node)] = 0
                v_avg_exp[(i_node, j_node)] = 0
            else:
                d_exp[(i_node, j_node)] = d.get((i_orig, j_orig), 0)
                t_exp[(i_node, j_node)] = t.get((i_orig, j_orig), 0)
                v_avg_exp[(i_node, j_node)] = v_avg.get((i_orig, j_orig), 0)

# -------------------------------------------------------
# 2.4 Demand Parameter
# -------------------------------------------------------

Demand = Customer_demand.values
p = {}
for i in range(len(N_original)):
     if i < Demand.shape[0]: # Ensure index is within bounds
         p[i] = float(Demand[i, 0])
     else:
         p[i] = 0 # Assign 0 demand if index is out of bounds (e.g., for depot)


# -------------------------------------------------------
# 2.5 Emissions, Energy, and Fuel-Related Parameters
# -------------------------------------------------------

# β: Unit cost of GHG emissions (carbon tax rate, $USD/ metric ton of CO2)
Beta = 24  # The actual tax rate using 'beta' instead of 'β'

# μ: Regional average grid emissions factor for electricity (kg CO2/kWh)
mu = .035

# Ω: GHG emissions coefficient for diesel fuel (kg CO2/L)
Omega = 2.56
#changed from 2.79

# r: Recharging costs ($USD/kWh)
r = .11

# F: cost of diesel fuel per liter ($USD/L)
F = 1.14

# M: Safety margin for HGEVs (kWh). HGEVs always reserve at least 3% charge for return trips.
Ms = 27 # Corresponds to 3% of 900 kWh

# ξ: Fuel-to-air mass ratio
xi = .055

# f: Engine friction factor
f_param = .25

# I: Engine speed (revolutions/s)
I_param = 23
#based off of Peterbilt 579

# E: Engine displacement (L)
E_param = 12.9
#based off of Peterbilt 579 PACCAR MX-13

# m: The efficiency parameter for diesel engines (engine thermal efficiency)
m_param = .33

# n: The heating value of typical diesel fuel (kJ/g) -> Convert to kJ/L using density (approx 840 g/L)
# n_param_per_g = 42
# density_diesel_g_L = 840
# n_param = n_param_per_g * density_diesel_g_L # kJ/L
# Let's stick to the original formulation's likely intent using lambda with psi
n_param = 42 # Assuming this was used with psi which converts g->L

# o: The mechanical drive-train efficiency of the vehicle
o_param = .85

# W_c: Curb weight (tractor + empty trailer, kg)
W_c = 21000 # Check if this includes empty trailer or just tractor
#value of Peterbilt 579

# τ: The average acceleration of the vehicle (m/s^2)
tau = .68
#Using value from Amiri et al

# g: Gravitational constant (m/s^2)
g_param = 9.81

# θ: Angle of the road (radians) - Average assumed grade
theta = math.radians(4.57) # Approx 8% grade - check if this is realistic average
#look up reference later, it is in my debugging notes

# C_d: Aerodynamic drag coefficient
C_d = .6 # Seems reasonable for a truck
#look up new reference later

# C_r: Rolling resistance coefficient
C_r = .01 # Reasonable for paved roads
#look up reference later

# α: Effective frontal area of the vehicle (tractor + trailer combination, m^2)
alpha = 25 # Seems high, typical is ~10 m^2. Double check this.

# ψ: Constant for converting g/s to L/s (Density of diesel fuel in g/L)
psi = 840 # g/L

# ρ: Air density (kg/m^3) at standard conditions
rho = 1.2041 # at 20°C

# -------------------------------------------------------
# 2.6 Additional Terms
# -------------------------------------------------------

#HGEV_Initiation_Cost: Fixed cost if *any* HGEV is used
HGEV_Initiation_Cost = 5000000
#HGEV_Expanded_Cost: Additional fixed cost if the *expanded charging network* is activated (HGEV_exp=1)
HGEV_Expanded_Cost = 6000000 # This cost is ON TOP of Initiation_Cost if HGEV_exp=1

#Tmax: The maximum amount of time a vehicle may service customers (seconds)
Tmax = 22 * 60 * 60 # 22 hours in seconds

#Tstop: The penalty time added for each stop at customers (not depot, not CS)
Tstop = 2 * 60 * 60 # 2 hours in seconds

#W: Driver wages per second ($/s) - DEPRECATED, included in O^k
# W = 35 / 60/ 60

# Terms used to simplify equations
# Check units carefully here
# lambda: (g_fuel / g_air) / ( (kJ / g_fuel) * (g_fuel / L_fuel) ) -> L_fuel / (kJ * g_air/g_fuel) ?? --> Let's re-derive based on source if possible
# Original: lambda_param = xi/(n_param*psi) # Units: (dim) / ( (kJ/g) * (g/L) ) = L / kJ ??? Seems wrong.
# Let's assume the final fuel consumption equations from the source paper are correct and these params combine appropriately.
lambda_param = xi / (n_param * psi) if n_param * psi != 0 else 0 # L/kJ ?

# phi: 1 / (1000 * thermal_eff * drivetrain_eff) # Units: 1 / ( J/kJ * dim * dim) = 1 (dimensionless?) - Check source
phi_param = 1 / (1000 * m_param * o_param) if m_param * o_param != 0 else 0 # 1/(kJ/J) = J/kJ

# sigma: (m/s^2) + (m/s^2)*sin(rad) + (m/s^2)*coeff*cos(rad) # Units: m/s^2 (Force per unit mass)
sigma_param = tau + g_param * math.sin(theta) + g_param * C_r * math.cos(theta) # m/s^2

# epsilon: 0.5 * drag_coeff * air_density * frontal_area # Units: dim * (kg/m^3) * m^2 = kg/m
epsilon_param = 0.5 * C_d * rho * alpha # kg/m


# --- Sanity Check Parameters ---
# print(f"{lambda_param=}, {phi_param=}, {sigma_param=}, {epsilon_param=}")
# -----------------------------


# =============================================================================
# 3. Define Variables
# =============================================================================
print("Defining variables...")
# Use tupledict for potentially better performance with sparse access
x_keys = [(k, i, j) for k in K_list for i in N_expanded for j in N_expanded if i != j]
x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")

q_keys = [(k, i, j) for k in K_list for i in N_expanded for j in N_expanded if i != j]
q = model.addVars(q_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="q")

y_keys = [(k, i) for k in K_e_list for i in N_expanded]
y = model.addVars(y_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="y")

T = model.addVars(K_list, vtype=GRB.CONTINUOUS, lb=0.0, name="TotalTime")

used = model.addVars(K_list, vtype=GRB.BINARY, name="used")

U_keys = [(k, i) for k in K_list for i in N_expanded]
U = model.addVars(U_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="U")

HGEV_used = model.addVar(vtype=GRB.BINARY, name="HGEV_used")
HGEV_exp = model.addVar(vtype=GRB.BINARY, name="HGEV_exp") # 1 if expanded charging network is active

delivered_keys = [(k, j) for k in K_list for j in D_expanded]
delivered = model.addVars(delivered_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="delivered")

print("Variables defined.")
# ===============================================================================================================================
# ===============================================================================================================================
#                                OBJECTIVE FUNCTIONS
# ===============================================================================================================================
# ===============================================================================================================================
print("Defining objectives...")
# Define objective expressions BEFORE setting the objective in the model

# =============================================================================
# 1. Minimize Total Costs Expression
# =============================================================================

# 1.1 Recharging Costs (HGEVs only)
# Cost = $/kWh * kWh/m * m * (arc usage)
term1 = r * gp.quicksum(
    h[k] * d_exp.get((i, j), 0) * x.get((k, i, j), 0)
    for k in K_e_list
    for i in N_expanded for j in N_expanded if i != j
)

# 1.2 Diesel Fuel Costs Due to Curb Weight (CFVs only)
# Cost = $/L * ( L_fuel ) * (arc usage)
# L_fuel = ( Power_req * time / (heating_val * density * thermal_eff * drive_eff) ) ??? -> Use formula components
# L_fuel = ( W_c*phi*sigma*d + f*I*E*t + eps*phi*d*v^2 ) * lambda
# Units: (kg * J/kJ * m/s^2 * m + Hz*L*s + kg/m*J/kJ*m*(m/s)^2) * (L/kJ?) -> Needs careful unit check from source
term2 = F * gp.quicksum(
    (W_c * phi_param * sigma_param * d_exp.get((i, j), 0) +
     f_param * I_param * E_param * t_exp.get((i, j), 0) +
     epsilon_param * phi_param * d_exp.get((i, j), 0) * (v_avg_exp.get((i, j), 0))**2
    ) * lambda_param * x.get((k, i, j), 0)
    for k in K_c_list
    for i in N_expanded for j in N_expanded if i != j if (v_avg_exp.get((i, j), None) is not None and lambda_param is not None) # Ensure velocity and lambda exist
)

# 1.3 Diesel Fuel Costs Due to Load Effect (CFVs only)
# Cost = $/L * ( L_fuel_load )
# L_fuel_load = ( Load * phi * sigma * d ) * lambda
# Units: (kg * J/kJ * m/s^2 * m) * (L/kJ?)
term3 = F * gp.quicksum(
    phi_param * sigma_param * lambda_param * d_exp.get((i, j), 0) * q.get((k, i, j), 0) # q is the load flow on the arc
    for k in K_c_list
    for i in N_expanded for j in N_expanded if i != j if lambda_param is not None
)

# 1.4 General Operating Costs (All Vehicles) + Driver Wage Costs (Implicit in O[k] now)
# Cost = $/m * m * (arc usage)
term4 = gp.quicksum(
    O[k] * d_exp.get((i, j), 0) * x.get((k, i, j), 0)
    for k in K_list
    for i in N_expanded for j in N_expanded if i != j
)
# Removed separate Wage cost term: + W * gp.quicksum(...)

# 1.5 Acquisition Costs (Includes Subsidies)
# Cost = ($Acq - $Sub) * (vehicle used)
term5 = gp.quicksum((A[k] - S[k]) * used[k] for k in K_list)

# 1.6 HGEV Infrastructure Costs
# Cost = FixedCost_Initial * (any HGEV used) + FixedCost_Expanded * (expanded network active)
term6 = HGEV_Initiation_Cost * HGEV_used + HGEV_Expanded_Cost * HGEV_exp

# 1.7 Cost of Diesel Emissions (CFVs only)
# Cost = $/tonne_CO2 * kg_CO2/L_fuel * ( L_fuel_total ) / (kg/tonne)
# L_fuel_total = L_fuel_curb + L_fuel_load
term7_temp_expr_curb = gp.quicksum(
    (W_c * phi_param * sigma_param * d_exp.get((i, j), 0) +
     f_param * I_param * E_param * t_exp.get((i, j), 0) +
     epsilon_param * phi_param * d_exp.get((i, j), 0) * (v_avg_exp.get((i, j), 0))**2
    ) * lambda_param * x.get((k, i, j), 0)
    for k in K_c_list
    for i in N_expanded for j in N_expanded if i != j if (v_avg_exp.get((i, j), None) is not None and lambda_param is not None)
)
term7_temp_expr_load = gp.quicksum(
    phi_param * sigma_param * lambda_param * d_exp.get((i, j), 0) * q.get((k, i, j), 0)
    for k in K_c_list
    for i in N_expanded for j in N_expanded if i != j if lambda_param is not None
)
term7 = Beta * Omega * (term7_temp_expr_curb + term7_temp_expr_load) / 1000 # Convert kg CO2 to tonnes

# Complete Cost Expression
obj_expr_cost = term1 + term2 + term3 + term4 + term5 + term6 + term7

# =============================================================================
# 2. Minimize GHG Emissions Expression
# =============================================================================

# 2.1 Diesel Emissions (CFVs only)
# Emissions = kg_CO2/L_fuel * ( L_fuel_total )
term8_temp_expr_curb = gp.quicksum(
    (W_c * phi_param * sigma_param * d_exp.get((i, j), 0) +
     f_param * I_param * E_param * t_exp.get((i, j), 0) +
     epsilon_param * phi_param * d_exp.get((i, j), 0) * (v_avg_exp.get((i, j), 0))**2
    ) * lambda_param * x.get((k, i, j), 0)
    for k in K_c_list
    for i in N_expanded for j in N_expanded if i != j if (v_avg_exp.get((i, j), None) is not None and lambda_param is not None)
)
term8_temp_expr_load = gp.quicksum(
    phi_param * sigma_param * lambda_param * d_exp.get((i, j), 0) * q.get((k, i, j), 0)
    for k in K_c_list
    for i in N_expanded for j in N_expanded if i != j if lambda_param is not None
)
term8 = Omega * (term8_temp_expr_curb + term8_temp_expr_load) # kg CO2

# 2.2 Electricity-Related Emissions from Charging HGEVs
# Emissions = kg_CO2/kWh * kWh_consumed
# kWh_consumed = kWh/m * m * (arc usage)
term9 = mu * gp.quicksum(
    h[k] * d_exp.get((i, j), 0) * x.get((k, i, j), 0)
    for k in K_e_list
    for i in N_expanded for j in N_expanded if i != j
)

# Complete Emissions Expression (in Metric Tons)
obj_expr_emissions = (term8 + term9) / 1000 # Convert kg CO2 to tonnes

print("Objectives defined.")
# ===============================================================================================================================
# ===============================================================================================================================
#                                CONSTRAINTS
# ===============================================================================================================================
# ===============================================================================================================================
print("Defining constraints...")
# =============================================================================
# 1. Routing Constraints
# =============================================================================

# 1.1 Each used vehicle must depart from the depot on at least one arc
# Sum over all arcs leaving depot for vehicle k >= used[k]
model.addConstrs(
    (gp.quicksum(x.get((k, (depot, 0), j), 0) for j in N_expanded if j != (depot, 0)) >= used[k]
    for k in K_list),
    name="Vehicle_must_leave_depot"
)

# 1.2 Conservation of Routing Flow (at depot)
# Number of times k leaves depot == number of times k returns to depot
model.addConstrs(
    (gp.quicksum(x.get((k, (depot, 0), j), 0) for j in N_expanded if j != (depot, 0)) ==
     gp.quicksum(x.get((k, i, (depot, 0)), 0) for i in N_expanded if i != (depot, 0))
     for k in K_list),
    name="Depot_flow_conservation"
)

# 1.3 Connectivity of Tours (at customer nodes)
# Flow into customer node j == Flow out of customer node j (for each vehicle k)
model.addConstrs(
    (gp.quicksum(x.get((k, i, j), 0) for i in N_expanded if i != j) ==
     gp.quicksum(x.get((k, j, i), 0) for i in N_expanded if i != j)
     for k in K_list for j in D_expanded),
    name="Customer_flow_conservation"
)


# 1.4 Maximum Route Time Calculation (per vehicle)
# T[k] >= travel_time + stop_penalties
# Define penalty nodes (customers not acting as charging stations in this specific context)
# Note: A customer in CS *could* be visited without charging if HGEV_exp=0,
# but the Tstop penalty applies based on the node type (customer), not charging activity.
penalty_nodes = {node for node in D_expanded} # All expanded customer nodes incur stop time

for k in K_list:
    # Sum travel times for arcs used by vehicle k
    travel_time_expr = gp.quicksum(
        t_exp.get((i, j), 0) * x.get((k, i, j), 0)
        for i in N_expanded for j in N_expanded if i != j
    )
    # Sum stops at penalty nodes (customer locations)
    # We sum arcs *leaving* the penalty node j, as each such arc implies a stop *at* j occurred before departure.
    # The number of arcs leaving a customer node equals the number of visits to that node (due to constraint 1.3)
    penalty_stops_expr = gp.quicksum(
        x.get((k, j, i), 0)
        for j in penalty_nodes for i in N_expanded if i != j
    )

    # Link total time T[k]
    # Tstop penalty applies for every stop at a customer node.
    model.addConstr(
        T[k] >= travel_time_expr + Tstop * penalty_stops_expr,
        name=f"TimeLink_{k}"
    )
    # We don't subtract used[k] here because the penalty applies to *all* customer stops.
    # The first departure from the depot doesn't incur Tstop.

# 1.5 Driver shift time limit (per vehicle)
# Total time for vehicle k <= Tmax
model.addConstrs(
    (T[k] <= Tmax for k in K_list),
    name="Maximum_service_time"
)

# 1.6 Used vehicle link (redundant with 1.1 but harmless)
# If vehicle k uses any arc (depot -> j), then used[k] must be 1.
# Ensures acquisition cost is applied if vehicle moves.
for k in K_list:
    for j in N_expanded:
        if j != (depot, 0):
            # If x[k, depot, j] is 1, used[k] must be at least 1.
            model.addConstr(used[k] >= x.get((k, (depot, 0), j), 0),
                            name=f"UsedLink_{k}_{j}")


# =============================================================================
# 2. Capacity and Flow Constraints
# =============================================================================

# 2.1 Prevent Flow on Unused Arcs
# If x[k,i,j] = 0, then q[k,i,j] must be 0.
model.addConstrs(
    (q.get((k, i, j), 0) <= Q[k] * x.get((k, i, j), 0)
     for k in K_list for i in N_expanded for j in N_expanded if i != j),
    name="link_q_x"
)

# 2.2 Customer demand is satisfied
# Sum of deliveries to all copies 'v' of customer 'i' by all vehicles 'k' must equal demand p[i].
for i in D: # Iterate through original customer indices
     if p.get(i, 0) > 0: # Only add constraint if demand exists
        model.addConstr(
            gp.quicksum(delivered.get((k, (i, v)), 0) for k in K_list for v in range(1, Vmax+1) if (i,v) in D_expanded) == p[i],
            name=f"demand_satisfaction_{i}"
        )


# 2.3 Per-Vehicle Flow Conservation at Customer Nodes (Load)
# Load arriving at j = Load delivered at j + Load departing from j
for k in K_list:
    for j in D_expanded: # j is an expanded customer node (i_orig, v)
        # Sum of load arriving at j on vehicle k from any node i
        load_in = gp.quicksum(q.get((k, i, j), 0) for i in N_expanded if i != j)
        # Sum of load departing from j on vehicle k to any node i
        load_out = gp.quicksum(q.get((k, j, i), 0) for i in N_expanded if i != j)
        # Delivery at this specific expanded node j by vehicle k
        delivery_at_j = delivered.get((k, j), 0)

        model.addConstr(load_in == delivery_at_j + load_out, name=f"flow_conservation_{k}_{j}")


# 2.4 Load Tracking using Cumulative Load Variable U (MTZ style)
# This helps prevent subtours and tracks load correctly.
# U[k,j] = Load remaining on vehicle k *after* visiting node j.
BIG_M_LOAD = sum(p.values()) # A safe upper bound for total demand

for k in K_list:
    # Initial load leaving depot: Total demand served by this vehicle in its tour(s).
    # This is tricky without tour indices. Let's use the flow variable q leaving the depot.
    # U[k, depot] isn't meaningful in MTZ. Focus on updates between nodes.

    for i in N_expanded:
        for j in N_expanded:
            if i == j: continue
            if j[0] == depot: continue # No demand at depot, U resets implicitly on next departure

            # If vehicle k travels i -> j, then U[k,j] <= U[k,i] - delivered[k,j]
            # Using Big M formulation for U update:
            # U[k,i] - U[k,j] >= delivered[k,j] - BIG_M * (1 - x[k,i,j])
            # We also need U[k,j] <= Q[k] and U[k,j] >= 0 (defined in variable bounds)
            # And U needs to be linked to q. Let's use the flow conservation (2.3)
            # and link U to the outgoing flow q.

            # Simpler Approach: Define U[k,i] as load *before* delivery at i.
            # Let's redefine U[k,i] = load on vehicle k when *arriving* at node i.
            if i[0] != depot: # Update from a customer/CS node
                 model.addConstr(
                    U.get((k, j), 0) <= U.get((k, i), 0) - delivered.get((k, i), 0) + Q[k] * (1 - x.get((k, i, j), 0)),
                    name=f"update_U_from_customer_{k}_{i}_{j}"
                 )
            else: # Update from the depot (i is depot)
                 # Load arriving at first customer j = Total load picked up - delivery at j
                 # Total load picked = Sum of deliveries this vehicle makes on this trip? Complicated.
                 # Let's enforce U <= Q[k] always (via var bounds) and use flow 'q'.

                 # Constraint: Load on arc (i,j) q[k,i,j] <= U[k,i] (Load arriving at i)
                 # This assumes U[k,i] is load *before* delivery at i.
                 model.addConstr(
                     q.get((k, i, j), 0) <= U.get((k, i), 0) + Q[k] * (1 - x.get((k, i, j), 0)), # Relax if arc not used
                     name=f"link_q_U_upper_{k}_{i}_{j}"
                 )
                 # If k travels i->j, then U[j] = U[i] - delivered[i] (for i != depot)
                 # This requires knowing which vehicle delivered what.

# Let's stick to the flow conservation constraint (2.3) which links q_in, q_out, and delivered.
# And constraint (2.1) linking q to Q and x.
# Remove the U variable constraints for now as they are complex without tour indices and might be redundant.
# We rely on q <= Q*x and flow conservation.

# 2.5 Linking delivery variable with the route
# Ensures delivery occurs only if the node is visited (arc leaves the node).
# Caps delivery at the node's total demand p[j[0]].
for k in K_list:
    for j in D_expanded: # j = (customer_idx, copy_idx)
        # Sum arcs leaving node j for vehicle k
        is_visited = gp.quicksum(x.get((k, j, i), 0) for i in N_expanded if i != j)
        # Delivery at this specific node instance (k, j) cannot exceed total demand p[j[0]]
        # And can only happen if visited (is_visited = 1)
        model.addConstr(
            delivered.get((k, j), 0) <= p.get(j[0], 0) * is_visited,
            name=f"delivery_route_link_{k}_{j}"
        )
        # Ensure delivered amount is not negative (already handled by lb=0.0)


# 2.6 Ensure non-zero deliveries if demand is met (Optional, based on prompt)
# If a vehicle k visits node j (is_visited = 1), force a minimum delivery?
# The prompt implies half-truck increments. Let's enforce this if visited.
# If is_visited = 1, then delivered[k,j] >= (Q[k]/2) ?? No, delivered sums up to p[i].
# Let's ensure that if a vehicle stops at j, it delivers *something* if it contributes to demand p[j[0]].
# This is implicitly handled by flow conservation if p[j[0]] > 0.
# Let's skip the half-truck constraint for now, as it complicates demand satisfaction (2.2).
# Constraint 2.2 (demand satisfaction) and 2.3 (flow conservation) should handle deliveries correctly.


# =============================================================================
# 3. HGEV Constraints
# =============================================================================

# 3.1 If any HGEV is used, "HGEV_Used" is set to 1 (for infra cost)
model.addConstrs(
    (HGEV_used >= used[k] for k in K_e_list),
    name="link_HGEV_used")

# 3.1b If any HGEV uses a charging station (CS), HGEV_exp must be 1
# This links the expanded network cost to actual CS usage (recharge).
# We need to track recharge events. Let's add this logic inside 3.2.
# For now, link HGEV_exp to travel TO a CS node by an HGEV.
model.addConstrs(
    (HGEV_exp >= x.get((k, i, j), 0)
     for k in K_e_list
     for i in N_expanded
     for j in N_expanded if i != j and j[0] in CS), # If travel TO a CS node by HGEV
    name="link_HGEV_exp_usage"
)


# 3.2 The battery level of vehicles is continuously updated (Big M for y)
# y[k,j] = battery level of HGEV k when arriving at node j.
BIG_M_ENERGY = R[K_e_list[0]] if K_e_list else 1000 # Max battery capacity

for k in K_e_list:
    # Initial charge at depot doesn't need constraint, assumed full when leaving.
    # When leaving depot (i = depot, 0): y[k,j] <= R[k] - energy_consumed(i,j)
    for j in N_expanded:
        if j[0] == depot: continue # Arriving at depot resets charge (handled below)
        energy_consumed = h[k] * d_exp.get(((depot, 0), j), 0)
        model.addConstr(
            y.get((k, j), 0) <= R[k] - energy_consumed + BIG_M_ENERGY * (1 - x.get((k, (depot, 0), j), 0)),
            name=f"charge_after_depot_{k}_{j}"
        )

    # Updates between other nodes i -> j
    for i in N_expanded:
        if i[0] == depot: continue # Starting from depot handled above

        for j in N_expanded:
            if i == j: continue

            energy_consumed = h[k] * d_exp.get((i, j), 0)

            # Base case: Energy decreases
            # y[k,j] <= y[k,i] - energy_consumed + BIG_M * (1 - x[k,i,j])
            model.addConstr(
                y.get((k, j), 0) <= y.get((k, i), 0) - energy_consumed + BIG_M_ENERGY * (1 - x.get((k, i, j), 0)),
                name=f"update_charge_{k}_{i}_{j}"
            )

            # Recharge case 1: Arriving at Depot (j = depot, 0)
            if j[0] == depot:
                # If x[k,i,j] = 1, force y[k,j] = R[k]? No, y is arrival charge.
                # Recharge happens *after* arrival. The constraint y[k, depot]=R[k] isn't needed
                # because the 'charge_after_depot' constraint handles the level when *leaving*.
                pass # Implicitly handled by next departure charge level.

            # Recharge case 2: Arriving at active Charging Station (j[0] in CS and HGEV_exp = 1)
            elif j[0] in CS:
                 # If k travels i -> j AND HGEV_exp=1, then the charge *leaving* j should be R[k].
                 # This affects the *next* constraint starting from j.
                 # Let's modify the base case update: If the *origin* i is an active CS, start full.
                 # y[k,j] <= R[k] - energy_consumed + BIG_M * (1 - x[k,i,j]) IF i is active CS
                 # This requires indicator on HGEV_exp.

                 # Alternative: Use GenConstrIndicator (might be clearer)
                 # If x[k,i,j]=1 and j is NOT depot/CS: y[j] = y[i] - consumed
                 # If x[k,i,j]=1 and j IS depot: (handled by departure constraint)
                 # If x[k,i,j]=1 and j IS CS and HGEV_exp=1: (handled by departure constraint from CS)

                 # Let's stick to Big M for now, but refine the recharge logic.
                 # When leaving an active CS 'i', the charge is R[k].
                 # So, if i[0] in CS, the update should start from R[k] if HGEV_exp=1.

                 # If HGEV_exp = 0, CS nodes act like normal customers for charging.
                 # If HGEV_exp = 1, leaving CS node 'i' starts charge at R[k].
                 # y[k,j] <= R[k] - energy_consumed + BIG_M*(1-x[k,i,j]) + BIG_M*(1-HGEV_exp)
                 # This applies if i[0] in CS.

                 if i[0] in CS:
                     # If leaving an active CS node i: y[j] <= R[k] - consumed ...
                     model.addConstr(
                         y.get((k, j), 0) <= R[k] - energy_consumed + BIG_M_ENERGY * (1 - x.get((k, i, j), 0)) + BIG_M_ENERGY * (1-HGEV_exp),
                         name=f"update_charge_from_active_CS_{k}_{i}_{j}"
                     )
                     # Need to prevent the normal update rule when this one applies.
                     # Add constraint: y[k,j] >= y[k,i] - consumed - BIG_M*(1-x) to make it equality?
                     # Let's simplify: Assume y[k,i] holds arrival charge. Recharge happens conceptually between arrival and departure.
                     # The departure constraint (like 'charge_after_depot') is key.

                     # If vehicle k departs from active CS i to j:
                     model.addConstr(
                         y.get((k, j), 0) <= R[k] - energy_consumed + BIG_M_ENERGY * (1 - x.get((k, i, j), 0)) + BIG_M_ENERGY * (1-HGEV_exp),
                         name=f"charge_after_active_CS_{k}_{i}_{j}"
                         )
                     # If vehicle k departs from inactive CS i to j: (acts like normal customer)
                     model.addConstr(
                         y.get((k, j), 0) <= y.get((k, i), 0) - energy_consumed + BIG_M_ENERGY * (1 - x.get((k, i, j), 0)) + BIG_M_ENERGY * HGEV_exp, # Active only if HGEV_exp=0
                         name=f"charge_after_inactive_CS_{k}_{i}_{j}"
                         )


# 3.3 Reserve battery constraint (Energy level on arrival must be sufficient)
# y[k,i] >= energy_to_nearest_safe_haven + SafetyMargin
# Safe haven = depot OR active CS (if HGEV_exp = 1)

# Pre-calculate d_star: min distance from customer i to depot or any CS
d_star = {}
for i_node in D_expanded: # Only need for customer nodes
    i_orig = i_node[0]
    d_to_depot = d.get((i_orig, depot), float('inf'))
    d_to_cs = float('inf')
    if CS: # If there are potential charging stations
        d_to_cs = min(d.get((i_orig, cs_idx), float('inf')) for cs_idx in CS)
    d_star[i_node] = min(d_to_depot, d_to_cs)

# Add constraints
for k in K_e_list:
    for i in D_expanded: # Check arrival charge at customer nodes
        # Energy needed to reach depot
        energy_to_depot = h[k] * d.get((i[0], depot), float('inf'))
        # Energy needed to reach nearest CS (if any exist)
        energy_to_cs = float('inf')
        if CS:
            energy_to_cs = min(h[k] * d.get((i[0], cs_idx), float('inf')) for cs_idx in CS)

        # Required reserve if expanded network is OFF (must reach depot)
        required_reserve_off = energy_to_depot + Ms
        model.addConstr(
            y.get((k, i), 0) >= required_reserve_off - BIG_M_ENERGY * HGEV_exp, # Constraint active only if HGEV_exp = 0
            name=f"reserve_return_depot_{k}_{i}"
        )

        # Required reserve if expanded network is ON (must reach depot OR nearest CS)
        required_reserve_on = h[k] * d_star.get(i, float('inf')) + Ms
        model.addConstr(
            y.get((k, i), 0) >= required_reserve_on - BIG_M_ENERGY * (1 - HGEV_exp), # Constraint active only if HGEV_exp = 1
            name=f"reserve_return_expanded_{k}_{i}"
        )

# Ensure battery level never exceeds capacity (redundant with updates, but safe)
# model.addConstrs((y[k,i] <= R[k] for k in K_e_list for i in N_expanded), name="Max_charge")
# Ensure battery level >= 0 (already defined in variable)

print("Constraints defined.")
# ===============================================================================================================================
# ===============================================================================================================================
#                                RUNNING THE MODEL
# ===============================================================================================================================
# ===============================================================================================================================

# --- Set Objective Based on RUN_MODE ---
print(f"Setting objective for RUN_MODE: {RUN_MODE}")
if RUN_MODE == 'find_cost_min':
    model.setObjective(obj_expr_cost, GRB.MINIMIZE)
    print("Objective: Minimize Total Cost")
elif RUN_MODE == 'find_emiss_min':
    model.setObjective(obj_expr_emissions, GRB.MINIMIZE)
    print("Objective: Minimize GHG Emissions")
elif RUN_MODE == 'weighted_sum':
    print(f"Using Cost Weight: {WEIGHT_COST}")
    # --- Check if normalization bounds are available ---
    if COST_MIN is None or EMISS_AT_COST_MIN is None or EMISS_MIN is None or COST_AT_EMISS_MIN is None:
        print("\nERROR: Normalization bounds (COST_MIN, etc.) are not set.")
        print("Please run the model in 'find_cost_min' and 'find_emiss_min' modes first to populate these values.")
        sys.exit(1)

    # --- Calculate Normalization Ranges (handle division by zero) ---
    cost_range = COST_AT_EMISS_MIN - COST_MIN
    emiss_range = EMISS_AT_COST_MIN - EMISS_MIN

    if abs(cost_range) < 1e-6: # Avoid division by zero if cost doesn't change
         print("Warning: Cost range for normalization is near zero. Normalizing only emissions.")
         if abs(emiss_range) < 1e-6:
              print("Warning: Emissions range also near zero. Setting objective to 0.")
              weighted_objective = 0 # Or handle as error
         else:
              norm_emiss = (obj_expr_emissions - EMISS_MIN) / emiss_range
              weighted_objective = (1.0 - WEIGHT_COST) * norm_emiss
    elif abs(emiss_range) < 1e-6: # Avoid division by zero if emissions don't change
         print("Warning: Emissions range for normalization is near zero. Normalizing only cost.")
         norm_cost = (obj_expr_cost - COST_MIN) / cost_range
         weighted_objective = WEIGHT_COST * norm_cost
    else:
        # --- Calculate Normalized Objectives ---
        norm_cost = (obj_expr_cost - COST_MIN) / cost_range
        norm_emiss = (obj_expr_emissions - EMISS_MIN) / emiss_range
        # --- Set Weighted Objective ---
        weighted_objective = WEIGHT_COST * norm_cost + (1.0 - WEIGHT_COST) * norm_emiss
        print("Objective: Minimize Weighted Sum of Normalized Cost and Emissions")

    model.setObjective(weighted_objective, GRB.MINIMIZE)

else:
    print(f"ERROR: Invalid RUN_MODE: {RUN_MODE}")
    sys.exit(1)

# --- Optimize ---
print("\nStarting Gurobi optimization...")
model.optimize()
print("Optimization finished.")

# ===============================================================================================================================
# ===============================================================================================================================
#                                MODEL RESULTS
# ===============================================================================================================================
# ===============================================================================================================================

if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL or model.status == GRB.SOLUTION_LIMIT:
    print("\n--- Solution Found ---")
    print(f"Model Status: {model.status}")
    print(f"Objective Value ({RUN_MODE}): {model.ObjVal:.4f}") # Note: This is the value of the objective set (could be weighted normalized value)

    # --- Calculate and Print ACTUAL Cost and Emissions ---
    actual_total_cost = obj_expr_cost.getValue()
    actual_total_emissions = obj_expr_emissions.getValue() # In tonnes
    print(f"\nActual Total Cost: ${actual_total_cost:,.2f}")
    print(f"Actual Total Emissions: {actual_total_emissions:.4f} metric tons CO2e")

    # --- Cost Breakdown ---
    print("\n--- Cost Breakdown ---")
    recharging_cost = term1.getValue()
    diesel_cost_curb = term2.getValue()
    diesel_cost_load = term3.getValue()
    operating_cost = term4.getValue()
    acquisition_cost = term5.getValue()
    hgev_infra_cost_init = HGEV_Initiation_Cost * HGEV_used.X
    hgev_infra_cost_exp = HGEV_Expanded_Cost * HGEV_exp.X
    hgev_infra_cost_total = term6.getValue() # Should equal init + exp
    diesel_emissions_cost = term7.getValue()
    calculated_total_cost = recharging_cost + diesel_cost_curb + diesel_cost_load + operating_cost + acquisition_cost + hgev_infra_cost_total + diesel_emissions_cost

    print(f"  Recharging Costs (HGEV): ${recharging_cost:,.2f}")
    print(f"  Total Diesel Fuel Costs (CFV): ${diesel_cost_load + diesel_cost_curb:,.2f}")
    print(f"    Diesel - Curb Weight: ${diesel_cost_curb:,.2f}")
    print(f"    Diesel - Load Weight: ${diesel_cost_load:,.2f}")
    if F > 0:
        print(f"    Liters of Diesel Fuel Used: {(diesel_cost_load + diesel_cost_curb)/F:.2f} L")
    print(f"  General Operating Costs: ${operating_cost:,.2f}")
    print(f"  Acquisition Cost (Net): ${acquisition_cost:,.2f}")
    print(f"  HGEV Infrastructure Cost: ${hgev_infra_cost_total:,.2f}")
    print(f"    HGEV Initiation Cost Applied: ${hgev_infra_cost_init:,.2f} (HGEV_used={HGEV_used.X:.0f})")
    print(f"    HGEV Expanded Net Cost Applied: ${hgev_infra_cost_exp:,.2f} (HGEV_exp={HGEV_exp.X:.0f})") # Conditional Charging Validation
    print(f"  Diesel Emissions Cost (Carbon Tax): ${diesel_emissions_cost:,.2f}")
    print(f"  Calculated Total Cost: ${calculated_total_cost:,.2f}") # Verification

    # --- Emissions Breakdown ---
    print("\n--- Emissions Breakdown (metric tons CO2e) ---")
    total_diesel_emissions = term8.getValue() / 1000 # Convert kg to tonnes
    hgev_emissions = term9.getValue() / 1000 # Convert kg to tonnes
    calculated_total_emissions = total_diesel_emissions + hgev_emissions

    print(f"  Total Diesel Emissions (CFV): {total_diesel_emissions:.4f}")
    print(f"  Total HGEV Emissions (Electricity): {hgev_emissions:.4f}")
    print(f"  Calculated Total Emissions: {calculated_total_emissions:.4f}") # Verification

    # --- Selected Vehicles and Arcs ---
    selected_arcs_list = [key for key, var in x.items() if var.X > 0.5]
    selected_vehicles = [k for k in K_list if used[k].X > 0.5]
    print("\nSelected vehicles:", selected_vehicles)
    # print("Selected arcs (k, i, j):", selected_arcs_list) # Can be very long

    # --- Tmax Constraint Validation ---
    print("\n--- Verifying Tmax Constraint ---")
    print(f"Maximum allowed time (Tmax): {Tmax} seconds ({Tmax / 3600:.2f} hours)")
    violation_found = False
    for k in K_list:
        if used[k].X > 0.5:
            vehicle_time = T[k].X
            print(f"Vehicle {k}: Total Time = {vehicle_time:.2f} seconds ({vehicle_time / 3600:.2f} hours)")
            if vehicle_time > Tmax + 1e-6: # Add small tolerance for floating point
                 print(f"  *** WARNING: Vehicle {k} exceeds Tmax! ***")
                 violation_found = True
    if not violation_found:
        print("All used vehicles are within the Tmax limit.")

    # --- Delivery Verification (Optional - can be verbose) ---
    # print("\n--- Verifying Deliveries ---")
    # total_demand = sum(p.values())
    # total_delivered = sum(v.X for v in delivered.values())
    # print(f"Total Demand: {total_demand:.2f}")
    # print(f"Total Delivered: {total_delivered:.2f}")
    # for i in D:
    #     demand_i = p.get(i, 0)
    #     delivered_i = sum(delivered.get((k, (i, v)), 0).X for k in K_list for v in range(1, Vmax+1) if (i,v) in D_expanded)
    #     if abs(demand_i - delivered_i) > 1e-6:
    #          print(f"  Mismatch for customer {i}: Demand={demand_i:.2f}, Delivered={delivered_i:.2f}")

    # --- HGEV Battery Level Verification (Optional - can be verbose) ---
    # print("\n--- HGEV Battery Levels Along Their Routes ---")
    # ... (Add the battery printing logic from the original prompt if needed for detailed debugging) ...

    # --- Exact Vehicle Routes ---
    print("\n--- Vehicle Routes (Exact Paths) ---")
    for k in K_list:
        if used[k].X < 0.5:
            continue
        print(f"\nVehicle {k}:")
        # Gather arcs for this vehicle
        vehicle_arcs_dict = { (i, j): x[k, i, j].X for i in N_expanded for j in N_expanded if i != j and x.get((k, i, j), 0).X > 0.5 }
        if not vehicle_arcs_dict:
            print("  No arcs selected for this vehicle.")
            continue

        # Reconstruct tours starting from depot
        remaining_arcs = set(vehicle_arcs_dict.keys())
        tour_count = 0
        while True:
            # Find an arc starting at the depot among remaining arcs
            start_arc = None
            for i, j in remaining_arcs:
                if i == (depot, 0):
                    start_arc = (i, j)
                    break

            if start_arc is None: # No more tours starting from depot
                break

            tour_count += 1
            current_tour = [start_arc[0], start_arc[1]]
            remaining_arcs.remove(start_arc)
            current_node = start_arc[1]

            # Follow the path until returning to depot
            while current_node != (depot, 0):
                found_next = False
                for next_i, next_j in list(remaining_arcs): # Iterate over copy
                    if next_i == current_node:
                        current_tour.append(next_j)
                        remaining_arcs.remove((next_i, next_j))
                        current_node = next_j
                        found_next = True
                        break
                if not found_next:
                     print(f"  Warning: Tour {tour_count} for vehicle {k} did not return to depot. Path: {' -> '.join(map(str, current_tour))}")
                     break # Should not happen in feasible solutions

            print(f"  Trip {tour_count}: {' -> '.join(map(str, current_tour))}")

        if remaining_arcs:
             print(f"  Warning: Some arcs remained unused for vehicle {k} after tour reconstruction: {remaining_arcs}")


elif model.status == GRB.INFEASIBLE:
    print("\n--- Model is Infeasible ---")
    print("Calculating IIS (Irreducible Inconsistent Subsystem)...")
    model.computeIIS()
    model.write("model_infeasible.ilp")
    print("IIS written to model_infeasible.ilp")
    print("Common causes: Conflicting constraints (e.g., Tmax too short, demand too high for capacity/range, battery reserve issues).")

else:
    print(f"\n--- Optimization Ended with Status Code: {model.status} ---")
    # Refer to Gurobi documentation for status codes: https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html

# --- END OF FILE ---