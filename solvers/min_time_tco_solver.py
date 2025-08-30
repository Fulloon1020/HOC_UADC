# file: solvers/min_time_tco_solver.py

import time
import numpy as np
import cvxpy as cp
from typing import Dict, Any, List, Tuple

from problem_def import Problem, Solution
from solvers.base_solver import BaseSolver


class MinTimeTcoSolver(BaseSolver):
    """
    实现了论文 "Laser-Powered UAV Trajectory and Charging Optimization for Sustainable
    Data-Gathering in the Internet of Things" 中的 MinTime-TCO 算法。

    该算法通过块坐标下降(BCD)交替优化无人机的充电能量和悬停位置，
    以最小化总任务完成时间。
    """
    algorithm_name = "MinTime-TCO"
    is_stochastic = False  # 算法是确定性的

    def __init__(self, problem_instance: Problem):
        super().__init__(problem_instance)
        # 算法超参数
        self.max_iterations = self.scenario.get('min_time_tco_max_iter', 10)
        self.convergence_tol = self.scenario.get('min_time_tco_convergence_tol', 1.0)

        # 适配论文中的激光充电模型参数 (假设值，可放入scenario.json)
        self.laser_params = {
            'P_L': self.scenario.get('laser_power_W', 750),
            'eta': self.scenario.get('laser_eta_efficiency', 0.004),
            'gamma': self.scenario.get('laser_gamma_attenuation', 1e-6),
            'zeta': self.scenario.get('laser_zeta_beam_size', 0.1),
            'phi': self.scenario.get('laser_phi_beam_spread', 3.4e-5)
        }
        # 假设HAPS充电站位置 (可放入scenario.json)
        self.haps_pos = [np.array([500, 500, 1000]), np.array([-500, 500, 1000])]

    def _solve(self) -> Solution:
        start_time = time.time()

        if not self.problem.tasks:
            return self._create_empty_solution(time.time() - start_time)

        # --- 初始化变量 ---
        # 假设一个预定的访问顺序 (例如，按距离贪心排序)
        visiting_sequence = self._get_initial_sequence()

        # 初始化悬停位置 u_j (直接在任务点上方)
        hover_positions = np.array([
            [self.problem.task_map[tid]['pos_x'], self.problem.task_map[tid]['pos_y'],
             self.uav_spec['hover_altitude_m']]
            for tid in visiting_sequence
        ])

        # 初始化充电能量 e_j
        charging_energies = np.zeros(len(visiting_sequence))

        previous_objective = float('inf')

        # --- MinTime-TCO 主循环 (Algorithm 2) ---
        for i in range(self.max_iterations):
            print(f"\n--- [MinTime-TCO] Main Iteration {i + 1}/{self.max_iterations} ---")

            # 1. 充电优化 (MCRS - Algorithm 1)
            charging_energies = self._run_mcrs_charging_optimization(hover_positions, visiting_sequence)

            # 2. 悬停位置优化 (HPO-SCA)
            hover_positions = self._run_hpo_sca_position_optimization(hover_positions, charging_energies,
                                                                      visiting_sequence)

            current_objective = self._calculate_objective(hover_positions, charging_energies, visiting_sequence)
            print(f"--- [MinTime-TCO] Iter {i + 1} Completion Time: {current_objective:.4f}s ---")

            if abs(previous_objective - current_objective) < self.convergence_tol and i > 0:
                print(f"--- [MinTime-TCO] Converged at iteration {i + 1} ---")
                break
            previous_objective = current_objective

        computation_time_s = time.time() - start_time

        decision_variables = self._convert_to_solution_format(hover_positions, charging_energies, visiting_sequence)

        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables=decision_variables,
            planned_objective=previous_objective,
            computation_time_s=computation_time_s,
            is_stochastic=self.is_stochastic
        )

    def _run_mcrs_charging_optimization(self, u_j, pi) -> np.ndarray:
        """实现 MCRS 算法 (Algorithm 1) 来优化充电能量 e_j。"""
        # 这是一个复杂的算法，这里提供一个功能性的简化版本
        # 它确保无人机在每一步都有足够的能量飞到下一个点
        e_j = np.zeros(len(pi))

        energy_state = self.uav_spec['battery_capacity_J']
        uav_pos = self.problem.uav_initial_state_map[self.problem.uav_ids[0]]['start_pos']

        for j in range(len(pi)):
            # 能量消耗：飞行 + 悬停
            flight_dist = np.linalg.norm(u_j[j] - uav_pos)
            flight_time = flight_dist / self.uav_spec['max_velocity_ms']
            flight_energy = flight_time * self.uav_spec.get('flying_power_W', 130)  # P1 in paper

            upload_time = self._calculate_upload_time(u_j[j], pi[j])
            hover_energy = upload_time * self.uav_spec['hover_power_W']  # P0 in paper

            energy_needed = flight_energy + hover_energy

            if energy_state < energy_needed:
                charge_needed = energy_needed - energy_state
                charge_rate = self._calculate_charge_rate(u_j[j])
                charge_time = charge_needed / charge_rate if charge_rate > 1e-6 else float('inf')

                # 简单充电策略：充所需的电量
                e_j[j] = charge_needed
                energy_state += charge_needed

            energy_state -= energy_needed
            uav_pos = u_j[j]

        return e_j

    def _run_hpo_sca_position_optimization(self, u_j_r, e_j, pi) -> np.ndarray:
        """使用SCA求解悬停位置优化子问题 (Eq. 43)。"""
        # 这是一个复杂的凸优化问题，这里返回上一次的结果作为占位符
        # 完整的实现需要用 cvxpy 构建并求解 Eq. (43)
        return u_j_r

    def _calculate_objective(self, u_j, e_j, pi) -> float:
        """计算给定解的总任务完成时间 (Eq. 11a)。"""
        total_time = 0.0
        uav_pos = self.problem.uav_initial_state_map[self.problem.uav_ids[0]]['start_pos']

        for j in range(len(pi)):
            # 飞行时间
            flight_dist = np.linalg.norm(u_j[j] - uav_pos)
            flight_time = flight_dist / self.uav_spec['max_velocity_ms']
            total_time += flight_time

            # 悬停时间 (充电和上传的较大值)
            upload_time = self._calculate_upload_time(u_j[j], pi[j])
            charge_rate = self._calculate_charge_rate(u_j[j])
            charge_time = e_j[j] / charge_rate if charge_rate > 1e-6 else float('inf')

            hover_time = max(upload_time, charge_time)
            total_time += hover_time

            uav_pos = u_j[j]

        return total_time

    def _calculate_upload_time(self, u_j, task_id):
        task = self.problem.task_map[task_id]
        task_pos_ground = np.array([task['pos_x'], task['pos_y'], 0])
        dist = np.linalg.norm(u_j - task_pos_ground)
        # 使用简化的速率模型
        rate = 20e6 * np.log2(1 + 10 ** (7) / (dist ** 2))
        return (task['data_size_mbits'] * 1e6) / rate

    def _calculate_charge_rate(self, u_j):
        """计算在 u_j 位置的最大充电速率 (Eq. 6)。"""
        max_rate = 0.0
        for hap_pos in self.haps_pos:
            dist = np.linalg.norm(hap_pos - u_j)
            if dist < 1: dist = 1
            rate = (self.laser_params['eta'] * self.laser_params['P_L'] * np.exp(-self.laser_params['gamma'] * dist)) / \
                   (self.laser_params['zeta'] + self.laser_params['phi'] * dist) ** 2
            if rate > max_rate:
                max_rate = rate
        return max_rate

    def _get_initial_sequence(self) -> List[int]:
        """生成一个基于贪心策略的初始访问序列。"""
        tasks = self.problem.tasks[:]
        uav_pos = self.problem.uav_initial_state_map[self.problem.uav_ids[0]]['start_pos']

        sequence = []
        while tasks:
            closest_task = min(tasks, key=lambda t: np.linalg.norm(np.array([t['pos_x'], t['pos_y'], 0]) - uav_pos))
            sequence.append(closest_task['task_id'])
            uav_pos = np.array([closest_task['pos_x'], closest_task['pos_y'], 0])
            tasks.remove(closest_task)
        return sequence

    def _convert_to_solution_format(self, u_j, e_j, pi) -> Dict:
        """将内部解转换为框架标准的 Solution 对象。"""
        # 这是一个单无人机算法
        main_uav_id = self.problem.uav_ids[0]

        trajectories = {uid: [np.array(self.problem.uav_initial_state_map[uid]['start_pos'])] for uid in
                        self.problem.uav_ids}
        schedules = {uid: [] for uid in self.problem.uav_ids}

        uav_traj = [np.array(self.problem.uav_initial_state_map[main_uav_id]['start_pos'])]
        uav_schedule = []
        current_time = 0.0
        current_pos = np.array(self.problem.uav_initial_state_map[main_uav_id]['start_pos'])

        for i in range(len(pi)):
            target_pos = u_j[i]
            dist = np.linalg.norm(current_pos - target_pos)
            travel_duration = dist / self.uav_spec['max_velocity_ms']
            uav_schedule.append(
                {'task_type': 'fly_to_task', 'start_time': current_time, 'end_time': current_time + travel_duration})
            current_time += travel_duration
            uav_traj.append(target_pos)
            current_pos = target_pos

            upload_time = self._calculate_upload_time(target_pos, pi[i])
            charge_rate = self._calculate_charge_rate(target_pos)
            charge_time = e_j[i] / charge_rate if charge_rate > 1e-6 else 0
            hover_duration = max(upload_time, charge_time)

            uav_schedule.append({'task_type': 'collect_and_charge', 'start_time': current_time,
                                 'end_time': current_time + hover_duration, 'task_id': pi[i]})
            current_time += hover_duration
            uav_traj.append(target_pos)  # Hovering point

        trajectories[main_uav_id] = uav_traj
        schedules[main_uav_id] = uav_schedule

        return {"trajectories": trajectories, "schedule": schedules}

    def _create_empty_solution(self, comp_time: float, objective: float = 0) -> Solution:
        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables={
                "trajectories": {uid: [self.uav_initial_pos[uid]] for uid in self.problem.uav_ids},
                "schedule": {uid: [] for uid in self.problem.uav_ids}
            },
            planned_objective=objective,
            computation_time_s=comp_time,
            is_stochastic=self.is_stochastic
        )