"""
Quick test of OR-Tools solver on current dataset with full details
"""

import sys
sys.path.insert(0, '/home/bazyl/Projects/green-vehicle-routing-optimization')

import pandas as pd
import numpy as np
from src.green_vrp.config import create_default_config
from src.green_vrp.ortools_solver import ORToolsGreenVRP

def print_dataset_info(config):
    """Print comprehensive dataset information"""
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    # Load demand data (no header)
    demand_df = pd.read_csv(config.demand_file, header=None, names=['Demand'])
    print(f"\n📦 CUSTOMER DEMAND:")
    print(f"  Total customers: {len(demand_df)}")
    print(f"  Total demand: {demand_df['Demand'].sum():,.0f} kg")
    print(f"\n  Customer Details:")
    for idx, row in demand_df.iterrows():
        print(f"    Customer {idx+1:2d}: {row['Demand']:>8,.0f} kg")
    
    # Load distance matrix
    distance_df = pd.read_csv(config.distance_file, index_col=0)
    print(f"\n📏 DISTANCE MATRIX:")
    print(f"  Locations: {len(distance_df)} (including depot)")
    print(f"  Depot location: Node {len(distance_df)-1}")
    print(f"  Average distance: {distance_df.values[distance_df.values > 0].mean():.2f} km")
    print(f"  Max distance: {distance_df.values.max():.2f} km")
    print(f"  Min distance (non-zero): {distance_df.values[distance_df.values > 0].min():.2f} km")
    
    # Load time matrix
    time_df = pd.read_csv(config.time_file, index_col=0)
    print(f"\n⏱️  TIME MATRIX:")
    print(f"  Average travel time: {time_df.values[time_df.values > 0].mean():.2f} hours")
    print(f"  Max travel time: {time_df.values.max():.2f} hours")
    print(f"  Min travel time (non-zero): {time_df.values[time_df.values > 0].min():.2f} hours")
    
    # Vehicle costs - check if cost files exist, otherwise use defaults
    try:
        cfv_cost_file = "data/CFV_Operating_Costs.csv"
        hgev_cost_file = "data/HGEV_Operating_Costs.csv"
        cfv_df = pd.read_csv(cfv_cost_file)
        hgev_df = pd.read_csv(hgev_cost_file)
        cfv_fixed = cfv_df['Fixed Cost ($/vehicle)'].values[0]
        cfv_var = cfv_df['Variable Cost ($/km)'].values[0]
        cfv_emissions = cfv_df['Emissions (g CO2e/km)'].values[0]
        hgev_fixed = hgev_df['Fixed Cost ($/vehicle)'].values[0]
        hgev_var = hgev_df['Variable Cost ($/km)'].values[0]
        hgev_emissions = hgev_df['Emissions (g CO2e/km)'].values[0]
    except:
        # Use default values if files not found
        cfv_fixed = 1000.0
        cfv_var = 10.0
        cfv_emissions = 1100.0
        hgev_fixed = 1000.0
        hgev_var = 11.0
        hgev_emissions = 11.0
    
    print(f"\n🚛 VEHICLE FLEET:")
    print(f"  CFVs: {config.num_cfv}")
    print(f"    Fixed cost: ${cfv_fixed:,.2f} per vehicle")
    print(f"    Variable cost: ${cfv_var:,.2f} per km")
    print(f"    Emissions: {cfv_emissions:,.2f} g CO2e/km")
    print(f"    Capacity: 15,000 kg")
    
    print(f"\n  HGEVs: {config.num_hgev}")
    print(f"    Fixed cost: ${hgev_fixed:,.2f} per vehicle")
    print(f"    Variable cost: ${hgev_var:,.2f} per km")
    print(f"    Emissions: {hgev_emissions:,.2f} g CO2e/km")
    print(f"    Capacity: 15,000 kg")
    
    print(f"\n⚙️  OPERATIONAL PARAMETERS:")
    print(f"  Max working hours: {config.tmax/3600:.1f} hours")
    print(f"  Service time: {config.tstop/3600:.2f} hours per stop")
    print(f"  Carbon tax: ${config.beta:.2f} per metric ton CO2")
    print(f"  Diesel fuel cost: ${config.f_fuel:.2f} per liter")
    print(f"  Grid emissions: {config.mu:.3f} kg CO2/kWh")
    print(f"  Optimization mode: {config.run_mode}")
    print(f"  Time limit: {config.time_limit:.0f} seconds")

def test_ortools():
    """Test OR-Tools solver with current data"""
    print("="*80)
    print("TESTING OR-TOOLS SOLVER - FULL DATASET ANALYSIS")
    print("="*80)
    
    # Create configuration
    config = create_default_config()
    config.run_mode = 'cost_min'
    config.time_limit = 60  # 1 minute test
    config.num_cfv = 30  # Start with enough vehicles
    config.num_hgev = 20
    
    # Print comprehensive dataset info
    print_dataset_info(config)
    
    try:
        # Initialize solver
        solver = ORToolsGreenVRP(config)
        
        # Setup problem
        solver.setup_problem()
        
        # Create model
        solver.create_model()
        
        # Solve
        success, results = solver.solve()
        
        if success:
            print("\n" + "="*80)
            print("✓ SOLUTION FOUND!")
            print("="*80)
            
            print(f"\n💰 COST BREAKDOWN:")
            print(f"  Total objective: ${results['obj_val']:,.0f}")
            print(f"  Total cost: ${results.get('total_cost', results['obj_val']):,.0f}")
            
            print(f"\n📊 OPERATIONAL METRICS:")
            print(f"  Total distance: {results['total_distance']:.2f} km")
            print(f"  Total emissions: {results['total_emissions']:.2f} kg CO2e")
            print(f"  Vehicles used: {results['num_vehicles_used']}/{config.num_cfv + config.num_hgev}")
            print(f"  Solve time: {results['solve_time']:.2f} seconds")
            print(f"  Routes created: {len(results['routes'])}")
            
            if results['dropped_nodes']:
                print(f"\n  ⚠️  Warning: {len(results['dropped_nodes'])} customers not served")
                print(f"     Dropped nodes: {results['dropped_nodes']}")
            else:
                print(f"  ✓ All {len(solver.data['demands'])-1} customers served!")
            
            # Print detailed route information
            print(f"\n" + "="*80)
            print("DETAILED ROUTE INFORMATION")
            print("="*80)
            
            for vehicle_name, route_data in results['routes'].items():
                route = route_data['route']
                distance_km = route_data['distance']  # Already in km from solver
                load = route_data['load']
                
                # Calculate vehicle-specific metrics
                is_cfv = 'CFV' in vehicle_name
                vehicle_id = int(vehicle_name.replace('CFV', '').replace('HGEV', ''))
                
                # Get cost parameters
                try:
                    if is_cfv:
                        cfv_df = pd.read_csv("data/CFV_Operating_Costs.csv")
                        fixed_cost = cfv_df['Fixed Cost ($/vehicle)'].values[0]
                        var_cost = cfv_df['Variable Cost ($/km)'].values[0]
                        emissions_rate = cfv_df['Emissions (g CO2e/km)'].values[0]
                    else:
                        hgev_df = pd.read_csv("data/HGEV_Operating_Costs.csv")
                        fixed_cost = hgev_df['Fixed Cost ($/vehicle)'].values[0]
                        var_cost = hgev_df['Variable Cost ($/km)'].values[0]
                        emissions_rate = hgev_df['Emissions (g CO2e/km)'].values[0]
                except:
                    # Defaults if files not available
                    fixed_cost = 1000.0
                    var_cost = 10.0 if is_cfv else 11.0
                    emissions_rate = 1100.0 if is_cfv else 11.0
                
                route_cost = fixed_cost + (var_cost * distance_km)
                route_emissions = (emissions_rate * distance_km) / 1000  # Convert to kg
                
                print(f"\n🚛 {vehicle_name} ({'Conventional' if is_cfv else 'Hydrogen'}):")
                print(f"   Route: {' → '.join([f'C{node}' if node > 0 else 'Depot' for node in route])}")
                print(f"   Customers served: {len([n for n in route if n > 0])} locations")
                print(f"   Distance: {distance_km:.2f} km")
                print(f"   Load: {load:,.0f} kg ({load/15000*100:.1f}% capacity)")
                print(f"   Cost: ${route_cost:,.2f} (Fixed: ${fixed_cost:,.2f} + Variable: ${var_cost*distance_km:,.2f})")
                print(f"   Emissions: {route_emissions:.2f} kg CO2e")
            
            # Print summary statistics
            print(f"\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            
            total_customers = len([n for route in results['routes'].values() for n in route['route'] if n > 0])
            avg_customers_per_route = total_customers / results['num_vehicles_used'] if results['num_vehicles_used'] > 0 else 0
            avg_distance_per_route = results['total_distance'] / results['num_vehicles_used'] if results['num_vehicles_used'] > 0 else 0
            
            print(f"\n  Customer service:")
            print(f"    Total customers: {len(solver.data['demands'])-1}")
            print(f"    Customers served: {total_customers}")
            print(f"    Avg customers per route: {avg_customers_per_route:.1f}")
            
            print(f"\n  Vehicle utilization:")
            print(f"    Total available: {config.num_cfv + config.num_hgev} vehicles")
            print(f"    Vehicles used: {results['num_vehicles_used']} ({results['num_vehicles_used']/(config.num_cfv + config.num_hgev)*100:.1f}%)")
            print(f"    CFVs used: {len([v for v in results['routes'].keys() if 'CFV' in v])}")
            print(f"    HGEVs used: {len([v for v in results['routes'].keys() if 'HGEV' in v])}")
            
            print(f"\n  Distance metrics:")
            print(f"    Total distance: {results['total_distance']:.2f} km")
            print(f"    Avg distance per route: {avg_distance_per_route:.2f} km")
            
            print(f"\n  Cost metrics:")
            print(f"    Total cost: ${results.get('total_cost', results['obj_val']):,.2f}")
            print(f"    Cost per km: ${results.get('total_cost', results['obj_val']) / results['total_distance']:.2f}")
            print(f"    Cost per customer: ${results.get('total_cost', results['obj_val']) / total_customers:.2f}")
            
            print(f"\n  Environmental metrics:")
            print(f"    Total emissions: {results['total_emissions']:.2f} kg CO2e")
            print(f"    Emissions per km: {results['total_emissions'] / (results['total_distance']/1000):.2f} kg/km")
            print(f"    Emissions per customer: {results['total_emissions'] / total_customers:.2f} kg/customer")
            
            return True
            
        else:
            print("\n" + "="*80)
            print("❌ FAILED TO FIND SOLUTION")
            print("="*80)
            print("\nPossible reasons:")
            print("  - Not enough vehicles for the demand")
            print("  - Time limit too short")
            print("  - Constraints too tight")
            
            return False
            
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED")
        print("="*80)
        print(f"\nError: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    success = test_ortools()
    
    if success:
        print("\n" + "="*80)
        print("🎉 OR-TOOLS IS WORKING PERFECTLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run comparison: python experiments.py --mode comparison")
        print("  2. Generate Pareto: python experiments.py --mode pareto")
        print("  3. Fix ALNS multi-visit issue")
        print("  4. Write research paper!")
    else:
        print("\n" + "="*80)
        print("🔧 DEBUGGING NEEDED")
        print("="*80)
        print("\nCheck:")
        print("  1. OR-Tools installation: pip show ortools")
        print("  2. Data files exist: ls data/")
        print("  3. Config parameters: vmax, capacity, etc.")
    
    sys.exit(0 if success else 1)
