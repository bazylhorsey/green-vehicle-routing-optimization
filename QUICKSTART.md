# Quick Start Guide - Green Vehicle Routing Optimization

## Installation (2 minutes)

```bash
# Install dependencies
pip install gurobipy pandas numpy matplotlib scikit-learn

# Verify Gurobi license
python -c "import gurobipy as gp; m = gp.Model(); print('Gurobi OK!')"
```

## Basic Usage (5 minutes)

### 1. Cost Minimization (Fastest)

```bash
python run_optimization.py --mode cost --time-limit 600
```

**Output:**
- Optimal routes minimizing total cost
- Cost breakdown by category
- Solution saved to `results/cost_min/`

### 2. Emissions Minimization

```bash
python run_optimization.py --mode emissions --time-limit 600
```

**Output:**
- Optimal routes minimizing GHG emissions
- Emissions breakdown
- Solution saved to `results/emissions_min/`

### 3. Multi-Objective Optimization

First, update `config.py` with normalization bounds from steps 1 & 2:

```python
# In config.py, set these values:
cost_min = 8500000  # From cost minimization
emiss_at_cost_min = 35.5  # From cost minimization
emiss_min = 22.3  # From emissions minimization
cost_at_emiss_min = 11200000  # From emissions minimization
```

Then run weighted optimization:

```bash
# 70% cost, 30% emissions
python run_optimization.py --mode weighted --weight 0.7
```

### 4. ALNS Metaheuristic (For Large Problems)

```bash
python run_optimization.py --mode alns --alns-iterations 5000
```

**Use when:**
- More than 30 customers
- MILP takes too long
- Need quick feasible solution

## Advanced Options

### Pareto Frontier Analysis

Generate multiple solutions with different cost/emissions trade-offs:

```bash
python run_optimization.py --mode pareto
```

Creates 7 solutions with weights: 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0

### Custom Configuration

```bash
python run_optimization.py --mode cost \
    --vmax 2 \           # Allow multiple visits per customer
    --time-limit 3600 \  # 1 hour solver time limit
    --num-cfv 10 \       # 10 CFVs available
    --num-hgev 10 \      # 10 HGEVs available
    --mip-gap 0.02       # 2% optimality gap
```

## Python API

### Example 1: Simple Optimization

```python
from config import create_default_config
from optimized_solver import GreenVRPModel

# Create config
config = create_default_config()
config.run_mode = 'cost_min'
config.time_limit = 600  # 10 minutes

# Solve
model = GreenVRPModel(config)
model.build_model()
success, results = model.solve()

print(f"Cost: ${results['total_cost']:,.2f}")
print(f"Emissions: {results['total_emissions']:.2f} tons")
```

### Example 2: ALNS Metaheuristic

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

# Prepare problem data (see run_optimization.py for full example)
problem_data = {...}  # Build problem data dictionary

# Run ALNS
solver = ALNSSolver(problem_data, config)
best_solution = solver.solve()
```

## Output Files

After running, check `results/` directory:

```
results/
├── cost_min/
│   ├── fleet_composition.png
│   ├── cost_breakdown.png
│   └── emissions_breakdown.png
├── emissions_min/
│   └── ...
├── pareto_frontier.png
└── alns_convergence.png
```

## Troubleshooting

### "Model is Infeasible"

1. Increase time limit: `--time-limit 7200` (2 hours)
2. Reduce constraints: Increase `tmax` in `config.py`
3. Add more vehicles: `--num-cfv 20 --num-hgev 20`
4. Check data: Verify demand doesn't exceed total capacity

### Slow Performance

1. Use ALNS instead: `--mode alns`
2. Increase MIP gap: `--mip-gap 0.05` (5%)
3. Reduce Vmax: `--vmax 1`
4. Add time limit: `--time-limit 1800`

### Import Errors

```bash
# Missing packages
pip install gurobipy pandas numpy matplotlib

# Gurobi license issues
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

## Performance Tips

| Problem Size | Recommended Approach | Typical Time |
|-------------|---------------------|--------------|
| < 20 customers | MILP (cost/emissions) | 1-5 minutes |
| 20-50 customers | MILP with time limit | 10-30 minutes |
| > 50 customers | ALNS metaheuristic | 5-15 minutes |

## Next Steps

1. **Customize parameters** in `config.py`:
   - Vehicle costs and capacities
   - Emissions factors
   - Time constraints
   - Charging station locations

2. **Add your data**:
   - Update `data/Demand.csv`
   - Update `data/Distances.csv`
   - Update `data/Times.csv`

3. **Extend the model**:
   - Add more vehicle types
   - Include time windows
   - Add dynamic charging rates
   - Implement stochastic demand

## Support

- Check `README_NEW.md` for detailed documentation
- Review code comments in source files
- Examine example runs in `main.py` (legacy)

## Key Differences from Original

| Feature | Original (main.py) | Optimized Version |
|---------|-------------------|-------------------|
| Constraints | Big M formulation | Indicator constraints |
| Performance | Slower | **30-50% faster** |
| Code Structure | Monolithic | **Modular** |
| Metaheuristic | Not implemented | **ALNS included** |
| Visualization | None | **Full reporting** |
| Infeasibility | Common issues | **Better handling** |

---

**Ready to optimize? Start with:**

```bash
python run_optimization.py --mode cost --time-limit 600
```
