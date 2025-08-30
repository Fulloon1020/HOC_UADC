# HOC_UADC


This repository contains the official implementation for the paper: **"[Your Paper Title Here]"**. The paper is available at **[Journal/Conference Name or ArXiv link here]**.



## Overview



!(placeholder_image.png)

This project tackles the large-scale, non-convex Mixed-Integer Non-Linear Program (MINLP) that arises from optimizing a **fleet of Unmanned Aerial Vehicles (UAVs)** for **time-sensitive data collection** in complex urban environments. The primary goal is to ensure mission success while respecting vehicle dynamics and constraints on **deadlines** and **energy**. Due to the problem's computational intractability, we propose a novel **Hierarchical Optimal Control for UAV-assisted Data Collection (HOC-UADC)** framework.

This hierarchical strategy decouples the problem into two main components operating on different timescales and levels of abstraction, making it solvable in real-time without pre-trained models:

1. **Global Mission Planner (GMP)**: The high-level component responsible for long-term, fleet-wide strategic effectiveness. It confronts the **combinatorial complexity** of multi-UAV **task allocation** and **coarse routing**. This is achieved by formulating and solving a tractable **Mixed-Integer Linear Program (MILP)** approximation within a **Receding Horizon Control (RHC)** framework for adaptive, closed-loop re-planning.
2. **Local Trajectory Optimizer (LTO)**: The low-level component that, guided by the GMP, handles the **non-linear dynamics** and continuous control for each individual UAV. It generates **physically feasible and locally optimal trajectories** by solving a **Non-linear Model Predictive Control (NMPC)** problem. This ensures safe, precise, and energy-aware execution in the immediate environment.

This repository contains the complete implementation of the HOC-UADC framework, the simulation engine, and several baseline algorithms (e.g., Genetic Algorithm, Particle Swarm Optimization) used for comprehensive performance evaluation.



## Setup & Getting Started





### Prerequisites



- Python 3.8 or newer



### Dependencies



All Python dependencies are listed in the `requirements.txt` file. Key dependencies include:

- numpy
- pandas
- cvxpy
- casadi
- matplotlib
- pyproj



### Installation



We highly recommend using a virtual environment to ensure a clean setup.

1. **Clone this repository and navigate into the directory:**

   ```
   git clone [Your Repository URL]
   cd [Your Repository Folder]
   ```

2. **Create and activate a Python virtual environment:**

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate    # On Windows
   ```

3. **Install all dependencies:**

   ```
   pip install -r requirements.txt
   ```



## How to Run Experiments



All experiments are configured and launched by modifying and running the `main.py` file. The **core configuration area** is located within the **`EXPERIMENT CONTROL PANEL`** code block in this file.



### 1. Main Performance & Scalability Experiment



This experiment reproduces the comparative analysis figures in the paper (e.g., Figs. 1, 2, 3) regarding algorithm performance and scalability.

1. **Configure the Experiment Scenario**:

   - In the control panel of `main.py`, set the desired testing ranges for `N_values` (number of tasks) and `M_values` (number of UAVs).
   - In the `config.py` file, modify the `CITY_MAP_PATH` variable to select the scenario: `city_map_dense.json` (Dense Urban) or `city_map_sparse.json` (Sparse Suburban).
   - In `main.py`, ensure that `HOC_UADCSolver` and all `other_solvers` to be compared (e.g., `GASolver`, `PSOSolver`) are active.

2. **Run the Script**:
   
   ```
   python main.py
   ```

   The results will be automatically saved to `experiment_results_[timestamp]/performance_results.csv`.



### 2. Ablation Study



This experiment validates the contribution of each component within the HOC-UADC framework (e.g., Fig. 4).

1. Configure Ablation Modes:

   In the control panel of main.py, configure the hoc_ablation_modes list to include 'full' (the complete model) and all its variants (e.g., 'no_nmpc', 'no_strategic_layer').

2. **Run the Script**:


   ```
   python main.py
   ```



### 3. Parameter Sensitivity Analysis



This experiment investigates the impact of key parameters, such as the strategic planning period (T_H) and the control cycle period (T_S), on algorithm performance.

1. Activate the Analysis Module:

   Navigate to the end of the main() function in main.py and uncomment the corresponding sensitivity analysis code block (e.g., "TH&TS joint analysis").

2. **Run the Script**:


   ```
   python main.py
   ```

   The results will be saved to a separate CSV file, such as `sensitivity_combined_parameters.csv`.



## Results and Outputs



!(placeholder_image_2.png)

!(placeholder_image_3.png)

The simulation results are intended to demonstrate the superiority of the HOC-UADC framework in generating efficient, safe, and constraint-compliant mission plans. Key outputs include:

- **Comprehensive Performance Metrics**: The `performance_results.csv` file records the core metrics for each run, including **maximum completion time (T_max)**, **task success rate (η_succ)**, **average energy consumption**, and **computation time**. This data is the foundation for plotting all performance comparison figures in the paper.
- **Optimized UAV Trajectories**: For specific scenarios, a `visualization_logs/` directory will be created, containing `trajectory_uav_X.csv` files with detailed 4D state information (x, y, z, t) for each UAV.
- **Task Schedules**: The `visualization_logs/schedule_log.csv` file provides the raw data needed to generate a **Gantt chart**, clearly illustrating the task assignments and role transitions for each UAV over time.
- **Visualizations**: The data in `visualization_logs` can be used to render the key qualitative figures from the paper, such as the **3D trajectory and relay topology plots** and the **UAV task schedule Gantt chart**, which provide intuitive insight into the algorithm's intelligent decision-making process.
