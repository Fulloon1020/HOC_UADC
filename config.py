# file: config.py

import os
import numpy as np
from pathlib import Path

# --- Core Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

# --- Data File Paths ---
CITY_MAP_PATH = DATA_DIR / 'city_map_dense.json'
TASKS_PATH = DATA_DIR / 'tasks_dense.csv'
UAV_CONFIG_PATH = DATA_DIR / 'uav_config_dense.json'
SCENARIO_PATH = DATA_DIR / 'scenario.json'

# --- Physical Constants ---
SPEED_OF_LIGHT = 3e8  # c (m/s)
CARRIER_FREQUENCY = 2.4e9  # fc (Hz)

# --- HOC-UADC LTO (NMPC/SCP) Hyperparameters ---
# Weights for LTO objective function J_m in Eq. (32)
WEIGHT_TRACKING_ERROR = 1.0    # Weight for ||p_m(k) - p_ref||^2
WEIGHT_COMM_RATE = 1e-8        # w_R for R_link
WEIGHT_CONTROL_EFFORT = 0.1    # Weight for ||u_m(k)||^2
WEIGHT_TERMINAL_ERROR = 10.0   # Weight for terminal cost
ROLE_SWITCHING_PENALTY = 100.0 # eta_s in Eq. (33)

def db_to_linear(db):
    return 10**(db / 10)

def linear_to_db(linear):
    return 10 * np.log10(linear)