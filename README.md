# Dynamic Autonomous Air Brake System

**Ben Mamut’s STEM Capstone Project**  
Precision altitude control for model rockets using onboard sensor fusion and control.

## Overview

This repository contains a Python simulation of a dynamic autonomous air brake system designed to hit a target apogee within ±1 m for model rocketry competitions (e.g., the American Rocketry Challenge). It integrates:

- A discrete-time physics engine modeling gravity, drag, and wind (`Rocket_Physics.py`)
- Simulated barometric altimeter and inertial accelerometer sensors (`Sensors.py`)
- A 9-state Extended Kalman Filter for onboard state estimation (`EKF.py`)
- A PID controller for dynamic brake deployment based on drag estimates (`PID.py`)
- A drag solver to compute required vs. experienced drag constants (`Drag_Solver.py`)
- A quaternion utility for attitude updates (`Quaternion_Utils.py`)
- A Monte Carlo framework to validate performance (`Monte_Carlo.py`)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BenMamut/CapstoneRepository.git
   cd CapstoneRepository
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate     # On Linux/Mac
   .venv\Scripts\activate        # On Windows
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Run Instructions

To run the main simulation and Monte Carlo testing:
```bash
python Monte_Carlo.py
```
Simulation results will be output as a CSV file (`Simulation_Results.csv`) for further analysis.

---

## File Tree

```
CapstoneRepository/
├── Rocket_Physics.py         # Simulates rocket flight physics
├── Sensors.py                # Models sensor outputs
├── EKF.py                    # Extended Kalman Filter for state estimation
├── PID.py                    # PID controller for air-brake adjustment
├── Drag_Solver.py            # Solves for drag constants dynamically
├── Quaternion_Utils.py       # Quaternion math utilities
├── Monte_Carlo.py            # Runs 10,000 flight simulations
├── Simulation_Results.csv    # (generated) Monte Carlo results
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Dependencies

- Python 3.13
- NumPy 2.2.5
- Matplotlib 3.10.1

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
