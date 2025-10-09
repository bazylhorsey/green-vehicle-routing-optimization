"""
Configuration module for Green Vehicle Routing Optimization
Contains all parameters and settings for the optimization model
"""

import math
from dataclasses import dataclass
from typing import Dict, Set, Tuple

@dataclass
class VehicleParams:
    """Vehicle-specific parameters"""
    acquisition_cost: float  # $ USD
    subsidy: float  # $ USD
    energy_consumption: float  # kWh/m
    capacity: float  # kg
    battery_capacity: float  # kWh
    operating_cost: float  # $/m


@dataclass
class ModelParams:
    """General model parameters"""
    # Run mode options: 'cost_min', 'emissions_min', 'weighted', 'alns'
    run_mode: str = 'cost_min'
    weight_cost: float = 0.5  # Weight for cost in weighted objective
    
    # Network expansion
    vmax: int = 1  # Maximum visits per customer
    
    # Vehicle fleet sizes
    num_cfv: int = 14  # Number of conventional fuel vehicles available
    num_hgev: int = 14  # Number of heavy-duty electric vehicles available
    
    # Time constraints
    tmax: float = 22 * 60 * 60  # Maximum service time (seconds) - 22 hours
    tstop: float = 2 * 60 * 60  # Stop penalty time (seconds) - 2 hours
    
    # Emissions and fuel parameters
    beta: float = 24  # Carbon tax rate ($/metric ton CO2)
    mu: float = 0.035  # Grid emissions factor (kg CO2/kWh)
    omega: float = 2.56  # Diesel emissions coefficient (kg CO2/L)
    r: float = 0.11  # Recharging cost ($/kWh)
    f_fuel: float = 1.14  # Diesel fuel cost ($/L)
    ms: float = 27  # Safety margin for HGEVs (kWh) - 3% of 900 kWh
    
    # HGEV infrastructure costs
    hgev_initiation_cost: float = 5_000_000  # Fixed cost if any HGEV used
    hgev_expanded_cost: float = 6_000_000  # Additional cost for expanded network
    
    # Diesel engine parameters
    xi: float = 0.055  # Fuel-to-air mass ratio
    f_param: float = 0.25  # Engine friction factor
    i_param: float = 23  # Engine speed (rev/s)
    e_param: float = 12.9  # Engine displacement (L)
    m_param: float = 0.33  # Engine thermal efficiency
    n_param: float = 42  # Heating value of diesel (kJ/g)
    o_param: float = 0.85  # Drivetrain efficiency
    
    # Vehicle physical parameters
    w_c: float = 21000  # Curb weight (kg)
    tau: float = 0.68  # Average acceleration (m/s^2)
    g_param: float = 9.81  # Gravitational constant (m/s^2)
    theta_deg: float = 4.57  # Road angle (degrees)
    c_d: float = 0.6  # Aerodynamic drag coefficient
    c_r: float = 0.01  # Rolling resistance coefficient
    alpha: float = 25  # Frontal area (m^2)
    psi: float = 840  # Diesel density (g/L)
    rho: float = 1.2041  # Air density (kg/m^3)
    
    # Gurobi solver parameters
    mip_gap: float = 0.01  # 1% optimality gap
    time_limit: float = 3600  # Solver time limit (seconds)
    heuristics: float = 0.25  # Heuristics effort (0-1)
    threads: int = 0  # 0 = use all available threads
    
    # ALNS parameters
    alns_iterations: int = 10000
    alns_segment_size: int = 100
    alns_decay: float = 0.95
    alns_w1: float = 33  # Score for new best solution
    alns_w2: float = 9   # Score for better solution
    alns_w3: float = 13  # Score for accepted solution
    alns_w4: float = 0   # Score for rejected solution
    alns_start_temp: float = 10000
    alns_end_temp: float = 0.01
    alns_cooling_rate: float = 0.9999
    
    # Normalization bounds (to be populated after initial runs)
    cost_min: float = None
    emiss_at_cost_min: float = None
    emiss_min: float = None
    cost_at_emiss_min: float = None
    
    # Data file paths
    demand_file: str = "data/Demand.csv"
    distance_file: str = "data/Distances.csv"
    time_file: str = "data/Times.csv"
    
    # Charging station indices
    charging_stations: Set[int] = None
    
    def __post_init__(self):
        """Initialize computed parameters"""
        self.theta = math.radians(self.theta_deg)
        
        # Computed terms for fuel consumption
        if self.n_param * self.psi != 0:
            self.lambda_param = self.xi / (self.n_param * self.psi)
        else:
            self.lambda_param = 0
            
        if self.m_param * self.o_param != 0:
            self.phi_param = 1 / (1000 * self.m_param * self.o_param)
        else:
            self.phi_param = 0
            
        self.sigma_param = (self.tau + 
                           self.g_param * math.sin(self.theta) + 
                           self.g_param * self.c_r * math.cos(self.theta))
        
        self.epsilon_param = 0.5 * self.c_d * self.rho * self.alpha
        
        # Default charging stations if not set
        if self.charging_stations is None:
            self.charging_stations = {20, 21}
    
    def get_vehicle_params(self, vehicle_type: str) -> VehicleParams:
        """Get parameters for a vehicle type"""
        if vehicle_type == 'CFV':
            return VehicleParams(
                acquisition_cost=165000,
                subsidy=0,
                energy_consumption=0.004375,  # Arbitrary for CFV
                capacity=15000,
                battery_capacity=99999,  # Arbitrarily large
                operating_cost=0.00018125
            )
        elif vehicle_type == 'HGEV':
            return VehicleParams(
                acquisition_cost=400000,
                subsidy=0,
                energy_consumption=0.0011,
                capacity=15000,
                battery_capacity=900,
                operating_cost=0.00015875
            )
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")


def create_default_config() -> ModelParams:
    """Create a default configuration"""
    return ModelParams()


if __name__ == "__main__":
    # Test configuration
    config = create_default_config()
    print(f"Configuration loaded successfully")
    print(f"Run mode: {config.run_mode}")
    print(f"Lambda parameter: {config.lambda_param:.6e}")
    print(f"Phi parameter: {config.phi_param:.6e}")
    print(f"Sigma parameter: {config.sigma_param:.6f}")
    print(f"Epsilon parameter: {config.epsilon_param:.6f}")
