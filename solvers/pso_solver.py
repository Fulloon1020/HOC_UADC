# file: solvers/pso_solver.py

import time
import numpy as np
import pyswarms as ps
from typing import Dict, Any, List, Tuple

from problem_def import Problem, Solution
from simulation import SystemModels


class PSOSolver:
    """
    使用粒子群优化（PSO）算法解决无人机数据收集问题。
    该求解器确定静态任务分配和每架无人机的简单轨迹。
    """
    algorithm_name = "PSO"
    is_stochastic = True

    def __init__(self, problem_instance: Problem):
        self.problem = problem_instance

        # --- 内部实现参数处理，模仿 GASolver ---
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
            'safety_margin_m': uav_params.get('safety_margin_m', 5.0),
            'prop_energy_coeff_J_per_m': uav_params.get('prop_energy_coeff_J_per_m', 55.0)
        }
        self.scenario = self.problem.scenario
        self.tasks = self.problem.tasks
        self.uav_ids = self.problem.uav_ids

        # 创建 SystemModels 实例以用于适应度函数
        self.models = SystemModels(problem_instance)

        # PSO 超参数
        self.n_particles = 30
        self.n_iterations = 100
        self.options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    def run(self) -> Solution:
        """
        执行 PSO 算法以找到一个近似最优解。
        """
        start_time = time.time()

        n_tasks = self.problem.num_tasks
        n_uavs = self.problem.num_uavs

        if n_tasks == 0:
            return self._create_empty_solution(time.time() - start_time)

        dimensions = n_tasks
        bounds = (np.zeros(dimensions), np.full(dimensions, n_uavs))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=dimensions,
            options=self.options,
            bounds=bounds
        )

        # 传递 problem, models, uav_spec 等实例给适应度函数
        cost, pos = optimizer.optimize(
            objective_func=self._fitness_function,
            iters=self.n_iterations,
            problem=self.problem,
            models=self.models,
            uav_spec=self.uav_spec
        )

        computation_time_s = time.time() - start_time

        # 解码最佳位置到规划方案
        decision_variables, planned_objective = self._decode_position_to_plan(pos, cost)

        print(f"PSO 算法完成，最佳成本为： {cost:.2f}")

        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables=decision_variables,
            planned_objective=planned_objective,
            computation_time_s=computation_time_s,
            is_stochastic=self.is_stochastic
        )

    @staticmethod
    def _fitness_function(particle_positions: np.ndarray, **kwargs) -> np.ndarray:
        """
        评估每个粒子的“适应度”（成本）。成本越低，解越好。
        :param particle_positions: 形状为 (n_particles, n_dimensions) 的 numpy 数组。
        :param kwargs: 必须包含 'problem' 和 'models' 对象。
        :return: 每个粒子的成本数组。
        """
        problem: Problem = kwargs['problem']
        models: SystemModels = kwargs['models']
        uav_spec: Dict = kwargs['uav_spec']
        n_particles = particle_positions.shape[0]
        costs = np.zeros(n_particles)

        tasks = problem.tasks
        uav_ids = problem.uav_ids

        for i in range(n_particles):
            assignments_indices = np.floor(particle_positions[i]).astype(int)
            uav_task_lists = {uid: [] for uid in uav_ids}
            for task_idx, uav_idx in enumerate(assignments_indices):
                if uav_idx < len(uav_ids):
                    uav_id = uav_ids[uav_idx]
                    uav_task_lists[uav_id].append(tasks[task_idx])

            uav_completion_times = []
            total_energy_ok = True

            for uav_id, assigned_tasks in uav_task_lists.items():
                if not assigned_tasks:
                    uav_completion_times.append(0)
                    continue

                current_pos = np.array(problem.uav_initial_state_map[uav_id]['start_pos'])
                sorted_tasks = sorted(assigned_tasks, key=lambda t: np.linalg.norm(
                    current_pos[:2] - np.array([t['pos_x'], t['pos_y']])))

                uav_time = 0.0
                uav_energy_spent = 0.0

                for task in sorted_tasks:
                    task_pos = np.array([task['pos_x'], task['pos_y'], uav_spec['hover_altitude_m']])
                    distance = np.linalg.norm(current_pos - task_pos)

                    velocity = np.array([uav_spec['max_velocity_ms'], 0, 0])
                    travel_time = distance / uav_spec['max_velocity_ms']
                    travel_energy = models.calculate_uav_power_W(velocity, 'move') * travel_time

                    current_pos = task_pos

                    task_pos_ground = np.array([task['pos_x'], task['pos_y'], 0])
                    rate_bps = models.calculate_rate_bps(
                        task_pos, task_pos_ground, {}, is_a2a=False)

                    if rate_bps < 1e-9:
                        service_time = float('inf')
                    else:
                        service_time = (task['data_size_mbits'] * 1e6) / rate_bps

                    hover_power = models.calculate_uav_power_W(np.array([0, 0, 0]), 'hover')
                    hover_energy = hover_power * service_time if service_time != float('inf') else float('inf')

                    uav_time += travel_time + service_time
                    uav_energy_spent += travel_energy + hover_energy

                uav_completion_times.append(uav_time)

                if uav_energy_spent > uav_spec['battery_capacity_J'] * 1000:
                    total_energy_ok = False

            if any(t == float('inf') for t in uav_completion_times):
                makespan = float('inf')
            else:
                makespan = max(uav_completion_times)

            if not total_energy_ok:
                makespan = float('inf')

            costs[i] = makespan if makespan != float('inf') else 1e12

        return costs

    def _decode_position_to_plan(self, best_pos: np.ndarray, cost: float) -> Tuple[Dict, float]:
        """
        将 PSO 找到的最佳粒子位置转换为一个完整的规划方案（轨迹和调度）。
        """
        assignments_indices = np.floor(best_pos).astype(int)

        uav_task_lists = {uid: [] for uid in self.uav_ids}
        for task_idx, uav_idx in enumerate(assignments_indices):
            if uav_idx < len(self.uav_ids):
                uav_id = self.uav_ids[uav_idx]
                uav_task_lists[uav_id].append(self.tasks[task_idx])

        trajectories = {}
        schedule = {}
        planned_completion_times = []

        for uav_id, assigned_tasks in uav_task_lists.items():
            current_pos = np.array(self.problem.uav_initial_state_map[uav_id]['start_pos'])
            sorted_tasks = sorted(assigned_tasks,
                                  key=lambda t: np.linalg.norm(current_pos[:2] - np.array([t['pos_x'], t['pos_y']])))

            uav_trajectory = [current_pos]
            uav_schedule = []
            current_time = 0.0

            for task in sorted_tasks:
                target_pos = np.array([task['pos_x'], task['pos_y'], self.uav_spec['hover_altitude_m']])
                distance = np.linalg.norm(current_pos - target_pos)
                travel_duration = distance / self.uav_spec['max_velocity_ms']

                uav_schedule.append({
                    'task_type': 'fly_to_task',
                    'start_time': current_time,
                    'end_time': current_time + travel_duration,
                    'task_id': task['task_id']
                })
                current_time += travel_duration
                uav_trajectory.append(target_pos)
                current_pos = target_pos

                rate_bps = self.models.calculate_rate_bps(
                    target_pos, np.array([task['pos_x'], task['pos_y'], 0]), {}, is_a2a=False)

                service_duration = (task['data_size_mbits'] * 1e6) / rate_bps if rate_bps > 1e-9 else float('inf')

                uav_schedule.append({
                    'task_type': 'collect_data',
                    'start_time': current_time,
                    'end_time': current_time + service_duration,
                    'task_id': task['task_id']
                })
                current_time += service_duration
                uav_trajectory.append(target_pos)

            trajectories[uav_id] = uav_trajectory
            schedule[uav_id] = uav_schedule
            planned_completion_times.append(current_time)

        final_makespan = max(planned_completion_times) if planned_completion_times else 0

        return {"trajectories": trajectories, "schedule": schedule}, final_makespan

    def _create_empty_solution(self, comp_time: float) -> Solution:
        """为没有任务的情况创建解决方案。"""
        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables={
                "trajectories": {uid: [np.array(self.problem.uav_initial_state_map[uid]['start_pos'])] for uid in
                                 self.uav_ids},
                "schedule": {uid: [] for uid in self.uav_ids}
            },
            planned_objective=0,
            computation_time_s=comp_time,
            is_stochastic=self.is_stochastic
        )