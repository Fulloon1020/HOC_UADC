# file: solvers/base_solver.py

from abc import ABC, abstractmethod
from typing import Dict
from problem_def import Problem, Solution


class BaseSolver(ABC):
    """
    An abstract base class for all solvers. It handles common functionalities
    like parameter extraction and adaptation to ensure consistency.
    """
    algorithm_name: str = "BaseSolver"
    is_stochastic: bool = False

    def __init__(self, problem_instance: Problem):
        self.problem = problem_instance
        self.params = self._extract_and_prepare_params()
        self.uav_spec = self.params['uav_spec']
        self.scenario = self.params['scenario']

    def _extract_and_prepare_params(self) -> Dict:
        """
        Extracts, converts, and validates all necessary parameters from the
        problem object, adapting to the user's specific data files.
        """
        uav_params = self.problem.uav_config['default_params']

        spec = {
            'max_velocity_ms': uav_params['v_max_mps'],
            'min_altitude_m': uav_params['h_min_m'],
            'max_altitude_m': uav_params['h_max_m'],
            'hover_altitude_m': uav_params.get('hover_altitude_m', 100),
            'battery_capacity_J': uav_params['energy_max_kj'] * 1000.0,
            'hover_power_W': uav_params['power_hover_kw'] * 1000.0,
            'prop_power_coeff': uav_params['power_move_coeff'],
            'max_acceleration_ms2': uav_params.get('a_max_ms2', 5.0),
            'safety_margin_m': uav_params.get('safety_margin_m', 5.0),
            'prop_energy_coeff_J_per_m': uav_params.get('prop_energy_coeff_J_per_m', 55.0),
            'max_tx_power_W': uav_params.get('tx_power_max_w', 10.0) , # 默认10W
            'channel_bandwidth_B_mhz':uav_params.get('channel_bandwidth_B_mhz',1e6)
        }
        return {'uav_spec': spec, 'scenario': self.problem.scenario}

    def run(self) -> Solution:
        """
        The main entry point for the solver. This method should be implemented
        by all subclasses.
        """
        return self._solve()

    @abstractmethod
    def _solve(self) -> Solution:
        """
        Contains the core algorithm logic. Must be implemented by subclasses.
        """
        raise NotImplementedError