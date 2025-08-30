# file: evaluation.py

import numpy as np
import pandas as pd
from problem_def import Problem, Solution
from typing import Dict, Tuple


class Evaluator:
    """
    评估器，负责计算所有性能指标。
    """

    def __init__(self, problem: Problem, solution: Solution, history: Dict, final_states: Dict, completion_times: Dict):
        self.problem = problem
        self.solution = solution
        self.history = history
        self.final_states = final_states
        self.completion_times = completion_times

        # 【修正】内部实现参数标准化映射，确保与仿真器和求解器一致
        uav_params = self.problem.uav_config['default_params']
        self.uav_spec = {
            'max_velocity_ms': uav_params['v_max_mps'],
            'min_altitude_m': uav_params['h_min_m'],
            'max_altitude_m': uav_params['h_max_m'],
            'hover_altitude_m': uav_params.get('hover_altitude_m', 100),
            'battery_capacity_J': uav_params['energy_max_kj'] * 1000.0,
            'hover_power_W': uav_params['power_hover_kw'] * 1000.0,
            'prop_power_coeff': uav_params['power_move_coeff'],
            'max_acceleration_ms2': uav_params.get('a_max_ms2', 5.0),
            'safety_margin_m': uav_params.get('safety_margin_m', 5.0)
        }

    def calculate_all_metrics(self) -> Dict:
        """计算并返回所有指标的字典。"""
        t_max = self._calculate_tmax()
        eta_succ = self._calculate_success_ratio()
        avg_energy, energy_variance = self._calculate_energy_metrics()

        return {
            'T_max': t_max,
            'eta_succ': eta_succ,
            'avg_energy': avg_energy,
            'energy_variance': energy_variance,
            'computation_time': self.solution.computation_time_s
        }

    def _calculate_tmax(self) -> float:
        """计算最大任务完成时间。"""
        valid_times = [t for t in self.completion_times.values() if t != float('inf')]
        return max(valid_times) if valid_times else float('inf')

    def _calculate_success_ratio(self) -> float:
        """计算数据交付成功率。"""
        successful_tasks = 0
        for task in self.problem.tasks:
            task_id = task['task_id']
            deadline = task.get('deadline_s', float('inf'))
            if self.completion_times[task_id] <= deadline:
                successful_tasks += 1
        return successful_tasks / self.problem.num_tasks if self.problem.num_tasks > 0 else 0

    def _calculate_energy_metrics(self) -> Tuple[float, float]:
        """计算平均能耗和能耗方差。"""
        energy_consumed = []
        initial_energy = self.uav_spec['battery_capacity_J']

        for uav_id in self.problem.uav_ids:
            if uav_id in self.final_states:
                final_energy = self.final_states[uav_id].get('energy', initial_energy)
                # 【修正】能耗计算，并确保其为正数
                consumed = initial_energy - final_energy
                energy_consumed.append(max(0, consumed)) # 确保能耗不为负

        avg_e = np.mean(energy_consumed) if energy_consumed else 0
        var_e = np.var(energy_consumed) if energy_consumed else 0

        return avg_e, var_e

    def display_all(self):
        """在控制台打印所有评估指标。"""
        metrics = self.calculate_all_metrics()
        print("\n" + "-" * 30 + " EVALUATION RESULTS " + "-" * 30)
        for key, value in metrics.items():
            print(f"  - {key:<20}: {value:.4f}")
        print("-" * 78)