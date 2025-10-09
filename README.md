# Green Vehicle Routing Optimization

A comprehensive optimization framework for solving the Green Vehicle Routing Problem (G-VRP) with mixed fleet of Conventional Fuel Vehicles (CFVs) and Heavy-Duty Electric Vehicles (HGEVs). This project implements both exact MILP methods and metaheuristic approaches to minimize operational costs and greenhouse gas emissions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Green Vehicle Routing Problem addresses the challenge of designing delivery routes that balance economic efficiency with environmental sustainability. This implementation considers:

- **Mixed fleet management**: CFVs and HGEVs with different operating characteristics
- **Multi-objective optimization**: Simultaneous consideration of cost and emissions
- **Charging infrastructure**: Strategic placement and use of charging stations for HGEVs
- **Capacity and time constraints**: Realistic operational limitations
- **Battery management**: Range anxiety mitigation with safety margins
- **Multiple visits**: Customers can be visited multiple times if beneficial

## ✨ Features

### Optimization Approaches

1. **MILP Solver** (`optimized_solver.py`)
   - Exact optimization using Gurobi
   - **Indicator constraints** instead of Big M formulation for improved performance
   - Three run modes: cost minimization, emissions minimization, weighted sum
   - Handles complex constraints efficiently

2. **ALNS Metaheuristic** (`alns_solver.py`)
   - Adaptive Large Neighborhood Search
   - Multiple removal operators: random customer, random route, worst cost, worst emissions
   - Multiple repair operators: random, lowest cost, lowest emissions, regret-2
   - Adaptive weight adjustment based on operator performance
   - Simulated annealing acceptance criterion

### Key Improvements Over Previous Versions

✅ **Indicator constraints** replace Big M formulation → Faster solution times  
✅ **Modular architecture** with separate config, utils, and solver modules  
✅ **Comprehensive data validation** with error handling  
✅ **Flexible configuration** system for easy parameter tuning  
✅ **ALNS metaheuristic** for large-scale problems  
✅ **Solution visualization** capabilities  
✅ **Detailed cost and emissions breakdown**  

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Gurobi Optimizer (with valid license)
- Required Python packages

### Setup

```bash
# Clone the repository
git clone https://github.com/bazylhorsey/green-vehicle-routing-optimization.git
cd green-vehicle-routing-optimization

# Install required packages
pip install gurobipy pandas numpy matplotlib

# Ensure Gurobi license is configured
# Follow instructions at: https://www.gurobi.com/documentation/
```

### Directory Structure

```
green-vehicle-routing-optimization/
├── data/                          # Input data files
│   ├── Demand.csv                # Customer demand (kg)
│   ├── Distances.csv             # Distance matrix (meters)
│   ├── Times.csv                 # Travel time matrix (seconds)
│   ├── CFV_Operating_Costs.csv   # CFV cost parameters
│   ├── HGEV_Operating_Costs.csv  # HGEV cost parameters
│   └── General.csv               # General parameters
├── config.py                      # Configuration and parameters
├── utils.py                       # Utility functions
├── optimized_solver.py            # MILP solver with indicator constraints
├── alns_solver.py                 # ALNS metaheuristic
├── visualize.py                   # Visualization tools
├── main.py                        # Legacy implementation
└── README.md                      # This file
```

## 📖 Usage

### Quick Start - MILP Solver

```python
from config import create_default_config
from optimized_solver import GreenVRPModel

# Create configuration
config = create_default_config()
config.run_mode = 'cost_min'  # or 'emissions_min', 'weighted'
config.vmax = 1  # Maximum visits per customer
config.time_limit = 1800  # Solver time limit (seconds)

# Create and solve model
model = GreenVRPModel(config)
model.build_model()
success, results = model.solve()

if success:
    print(f"Total Cost: ${results['total_cost']:,.2f}")
    print(f"Emissions: {results['total_emissions']:.2f} tons CO2e")
```

### Multi-Objective Optimization

To perform proper multi-objective optimization:

```python
# Step 1: Find cost minimum
config.run_mode = 'cost_min'
model = GreenVRPModel(config)
model.build_model()
_, results_cost = model.solve()

# Save values
COST_MIN = results_cost['total_cost']
EMISS_AT_COST_MIN = results_cost['total_emissions']

# Step 2: Find emissions minimum
config.run_mode = 'emissions_min'
model = GreenVRPModel(config)
model.build_model()
_, results_emiss = model.solve()

# Save values
EMISS_MIN = results_emiss['total_emissions']
COST_AT_EMISS_MIN = results_emiss['total_cost']

# Step 3: Run weighted optimization
config.run_mode = 'weighted'
config.cost_min = COST_MIN
config.emiss_at_cost_min = EMISS_AT_COST_MIN
config.emiss_min = EMISS_MIN
config.cost_at_emiss_min = COST_AT_EMISS_MIN
config.weight_cost = 0.7  # 70% cost, 30% emissions

model = GreenVRPModel(config)
model.build_model()
success, results = model.solve()
```

### ALNS Metaheuristic

For large-scale problems where MILP is too slow:

```python
from config import create_default_config
from alns_solver import ALNSSolver
from utils import load_data, expand_network

# Load data
config = create_default_config()
d, t, v_avg, p, D, depot = load_data(
    config.demand_file, 
    config.distance_file, 
    config.time_file
)
N_expanded, d_exp, t_exp, v_avg_exp = expand_network(D, depot, config.vmax, d, t, v_avg)

# Prepare problem data
problem_data = {
    'd_exp': d_exp,
    't_exp': t_exp,
    'v_avg_exp': v_avg_exp,
    'p': p,
    'D': D,
    'depot': depot,
    'K_list': [f"CFV{i}" for i in range(1, 15)] + [f"HGEV{i}" for i in range(1, 15)],
    # ... add other parameters
}

# Run ALNS
solver = ALNSSolver(problem_data, config)
best_solution = solver.solve()

print(f"Best cost: ${best_solution.cost:,.2f}")
print(f"Emissions: {best_solution.emissions:.2f} tons")
```

## 🔬 Methodology

### Problem Formulation

The Green VRP is formulated as a mixed-integer linear program (MILP) with the following components:

#### Decision Variables

- `x[k,i,j]`: Binary, 1 if vehicle k travels from node i to j
- `q[k,i,j]`: Continuous, load on vehicle k on arc (i,j)
- `y[k,i]`: Continuous, battery charge of HGEV k at node i
- `delivered[k,j]`: Continuous, amount delivered by vehicle k to customer j
- `used[k]`: Binary, 1 if vehicle k is used
- `HGEV_used`: Binary, 1 if any HGEV is used
- `HGEV_exp`: Binary, 1 if expanded charging network is activated

#### Objectives

**Cost Minimization:**
```
minimize: recharging_cost + fuel_cost + operating_cost + 
          acquisition_cost + infrastructure_cost + carbon_tax
```

**Emissions Minimization:**
```
minimize: diesel_emissions + electricity_emissions
```

**Weighted Sum:**
```
minimize: w_cost × normalized_cost + w_emiss × normalized_emissions
```

#### Key Constraints

1. **Flow Conservation**: Vehicles leave and return to depot; flow is balanced at all nodes
2. **Demand Satisfaction**: All customer demand must be met
3. **Capacity**: Vehicle load cannot exceed capacity
4. **Time Windows**: Total route time ≤ Tmax
5. **Battery Management** (HGEVs):
   - Battery level updated based on energy consumption
   - Safety margin maintained for return trip
   - Charging at depot or designated stations
6. **Fuel Consumption** (CFVs): Based on load, distance, velocity, and road conditions

### Fuel Consumption Model

For CFVs, fuel consumption considers:
- **Curb weight**: Base vehicle weight consumption
- **Load weight**: Additional consumption due to cargo
- **Road grade**: Angle of inclination
- **Aerodynamic drag**: Velocity-dependent resistance
- **Rolling resistance**: Tire-road friction

Formula:
```
Fuel = λ × [W_c × φ × σ × d + f × I × E × t + ε × φ × d × v² + φ × σ × d × Q]
```

Where:
- λ, φ, σ, ε: Derived parameters from vehicle/environmental characteristics
- W_c: Curb weight, Q: Load weight
- d: Distance, t: Time, v: Velocity
- f, I, E: Engine parameters

### ALNS Algorithm

The Adaptive Large Neighborhood Search iteratively improves solutions through:

1. **Destruction**: Remove customers/routes using selected operator
2. **Repair**: Reinsert removed customers using selected operator
3. **Acceptance**: Accept solution based on simulated annealing criterion
4. **Adaptation**: Update operator weights based on performance

**Removal Operators:**
- Random customer removal
- Random route removal
- Worst cost removal (removes high-cost customers)
- Worst emissions removal (removes high-emission customers)

**Repair Operators:**
- Random insertion
- Lowest cost insertion (greedy on cost)
- Lowest emissions insertion (greedy on emissions)
- Regret-2 cost (considers opportunity cost)
- Regret-2 emissions (considers opportunity cost for emissions)

## ⚙️ Configuration

Key parameters in `config.py`:

### Vehicle Parameters
```python
num_cfv = 14              # Number of CFVs available
num_hgev = 14             # Number of HGEVs available
```

### Network Parameters
```python
vmax = 1                  # Max visits per customer (1 = no revisits)
charging_stations = {20, 21}  # Customers with charging capability
```

### Constraints
```python
tmax = 22 * 60 * 60       # Maximum route time (seconds)
tstop = 2 * 60 * 60       # Stop penalty (seconds)
ms = 27                   # Battery safety margin (kWh)
```

### Costs and Emissions
```python
beta = 24                 # Carbon tax ($/ton CO2)
mu = 0.035               # Grid emissions (kg CO2/kWh)
omega = 2.56             # Diesel emissions (kg CO2/L)
r = 0.11                 # Electricity cost ($/kWh)
f_fuel = 1.14            # Diesel cost ($/L)
```

### Solver Parameters
```python
mip_gap = 0.01           # 1% optimality gap
time_limit = 3600        # Max solve time (seconds)
heuristics = 0.25        # Gurobi heuristics effort
```

### ALNS Parameters
```python
alns_iterations = 10000   # Number of ALNS iterations
alns_start_temp = 10000   # Initial temperature
alns_cooling_rate = 0.9999  # Temperature decay rate
alns_w1 = 33             # Score for new best solution
alns_w2 = 9              # Score for improvement
alns_w3 = 13             # Score for accepted solution
```

## 📊 Results

### Expected Outputs

The solver provides:

1. **Optimal/Best Routes**: Detailed vehicle itineraries
2. **Cost Breakdown**:
   - Recharging costs (HGEV)
   - Fuel costs (CFV)
   - Operating costs
   - Acquisition costs
   - Infrastructure costs
   - Carbon tax
3. **Emissions Breakdown**:
   - Diesel emissions (CFV)
   - Electricity emissions (HGEV)
4. **Performance Metrics**:
   - Solution time
   - Optimality gap
   - Number of vehicles used
   - Total distance traveled

### Visualization

Use `visualize.py` to create:
- Route maps
- Cost vs emissions Pareto frontier
- ALNS convergence plots
- Fleet composition charts

## 🔍 Troubleshooting

### Model is Infeasible

If the solver returns infeasible:

1. **Check time constraints**: Increase `tmax` or reduce `tstop`
2. **Check battery capacity**: Increase `R[k]` or reduce safety margin `ms`
3. **Check vehicle availability**: Increase `num_cfv` or `num_hgev`
4. **Review demand**: Ensure total demand ≤ total capacity
5. **Examine IIS**: Check `model_infeasible.ilp` for conflicting constraints

### Slow Solution Times

1. **Use ALNS** for large instances (>50 customers)
2. **Reduce Vmax**: Set to 1 unless multiple visits are essential
3. **Adjust MIP gap**: Increase to 2-5% for faster solutions
4. **Set time limit**: Limit solver time with `time_limit`
5. **Warm start**: Provide initial solution from ALNS to MILP

### Memory Issues

1. **Reduce problem size**: Fewer vehicles or customers
2. **Increase system memory** or use cloud computing
3. **Use ALNS**: More memory-efficient than MILP

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional metaheuristic algorithms (Tabu Search, Genetic Algorithm)
- More sophisticated fuel consumption models
- Real-world data integration (traffic, weather)
- Multi-depot extensions
- Time-varying charging rates
- Stochastic demand

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

Key papers and resources:

1. Demir, E., Bektaş, T., & Laporte, G. (2014). "A review of recent research on green road freight transportation"
2. Schneider, M., Stenger, A., & Goeke, D. (2014). "The Electric Vehicle Routing Problem with Time Windows and Recharging Stations"
3. Ropke, S., & Pisinger, D. (2006). "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows"
4. Bektaş, T., & Laporte, G. (2011). "The Pollution-Routing Problem"

## 👥 Authors

- Bazyl Horsey
- Griffin Wilson

## 📧 Contact

For questions or collaboration:
- GitHub: [@bazylhorsey](https://github.com/bazylhorsey)
- Email: [Contact through GitHub]

---

**Note**: This is an academic/research project. Real-world deployment requires additional validation, testing, and integration with operational systems.
