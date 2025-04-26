from dataclasses import dataclass


@dataclass
class Config:
    """
    Holds all global constants and simulation settings.
    """
    # Physical constants
    g: float = 9.80665         # gravitational acceleration (m/s²)
    rho: float = 1.293         # air density at sea level (kg/m³)
    mass: float = 0.5          # rocket mass (kg)

    # Air-brake geometry & aerodynamics
    A_min: float = 0.004       # minimum brake area (m²)
    A_max: float = 5 * A_min   # maximum brake area (m²)
    Cd_min: float = 0.4        # minimum drag coefficient
    Cd_max: float = 0.6        # maximum drag coefficient

    # Target & tolerance
    target_apogee: float = 250.0  # desired apogee (m)
    tol: float = 1.0              # ± tolerance for success (m)

    # Simulation time settings
    dt: float = 0.01         # time step (s)
    t_max: float = 6.0       # total simulation duration (s)

    # Monte Carlo settings
    N: int = 100                 # number of Monte Carlo trials
    target_min: float = 200.0    # lower bound for random target (m)
    target_max: float = 300.0    # upper bound for random target (m)

    # Output toggles
    save_results: bool = True      # write results to CSV
    plot_all_true: bool = False     # overplot all true trajectories
    plot_each_run: bool = False     # generate per-run 3D plots
    vary_target: bool = False       # enable random target sampling
    plot_input_space: bool = False  # show input-space scatter

    # Filename to save to
    filename: str = "Simulation_Results.csv"


# Shared instance for import
cfg = Config()
