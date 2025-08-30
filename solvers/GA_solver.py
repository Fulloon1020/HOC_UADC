# file: solvers/ga_solver.py

import time
import numpy as np
from typing import Dict, List, Any

from problem_def import Problem, Solution
from solvers.base_solver import BaseSolver
# 【新增】导入 SystemModels 类
from simulation import SystemModels


class GASolver(BaseSolver):
    """
    使用遗传算法解决多无人机任务分配和路径规划问题。
    继承自 BaseSolver 以确保参数处理的一致性。
    """
    algorithm_name = "Genetic_Algorithm"
    is_stochastic = True

    def __init__(self, problem_instance: Problem, ga_params: Dict = None):
        # 调用父类构造函数来处理所有参数设置
        super().__init__(problem_instance)

        # --- GA 超参数 ---
        default_ga_params = {
            'pop_size': 50,
            'generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'elite_size': 5,
            'tournament_size': 5
        }
        self.ga_params = ga_params if ga_params is not None else default_ga_params

        # --- 预先计算好的问题数据，以提高效率 ---
        self.tasks = self.problem.tasks
        self.task_map = self.problem.task_map
        self.uav_ids = self.problem.uav_ids

        self.uav_initial_pos = {uid: np.array(self.problem.uav_initial_state_map[uid]['start_pos']) for uid in
                                self.uav_ids}

        self.task_pos = {
            t['task_id']: np.array([t['pos_x'], t['pos_y'], self.uav_spec['hover_altitude_m']])
            for t in self.tasks
        }

        # 【新增】在求解器中初始化 SystemModels，用于精确的成本计算
        self.models = SystemModels(problem_instance)
        # 【新增】飞行速度（假设为最大速度）
        self.velocity = np.array([self.uav_spec['max_velocity_ms'], 0, 0])

    def _solve(self) -> Solution:
        """执行遗传算法优化循环。"""
        start_time = time.time()

        if not self.tasks:
            return self._create_empty_solution(time.time() - start_time)

        population = self._initialize_population()

        for _ in range(self.ga_params['generations']):
            fitness_scores = self._evaluate_fitness(population)
            elites = self._select_elites(population, fitness_scores)
            parents = self._selection(population, fitness_scores)
            offspring = self._crossover(parents)
            mutated_offspring = self._mutate(offspring)
            population = elites + mutated_offspring

        final_fitness = self._evaluate_fitness(population)
        best_chromosome = self._get_best_individual(population, final_fitness)

        decision_variables, planned_objective = self._decode_to_plan(best_chromosome)

        computation_time_s = time.time() - start_time

        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables=decision_variables,
            planned_objective=planned_objective,
            computation_time_s=computation_time_s,
            is_stochastic=self.is_stochastic
        )

    # ########################## 遗传算法核心方法 ##########################

    def _initialize_population(self) -> List[np.ndarray]:
        """创建随机初始种群。"""
        num_uavs = self.problem.num_uavs
        num_tasks = self.problem.num_tasks
        return [np.random.randint(0, num_uavs, size=num_tasks) for _ in range(self.ga_params['pop_size'])]

    def _evaluate_fitness(self, population: List[np.ndarray]) -> List[float]:
        """评估每个染色体的适应度。成本越低，适应度越高（取负值）。"""
        return [-self._calculate_cost(chromosome) for chromosome in population]

    def _calculate_cost(self, chromosome: np.ndarray) -> float:
        """为给定的任务分配染色体计算总成本（Makespan）。"""
        task_assignments = {uid: [] for uid in self.uav_ids}
        for task_idx, uav_idx in enumerate(chromosome):
            task_assignments[self.uav_ids[uav_idx]].append(self.tasks[task_idx])

        uav_completion_times = []
        total_energy_penalty = 0

        for uav_id, tasks_for_uav in task_assignments.items():
            if not tasks_for_uav:
                uav_completion_times.append(0)
                continue

            current_pos = self.uav_initial_pos[uav_id]
            # 简单的最近邻排序
            tasks_for_uav.sort(key=lambda t: np.linalg.norm(current_pos - np.array([t['pos_x'], t['pos_y'], 0])))

            uav_time = 0.0
            uav_energy_spent = 0.0

            for task in tasks_for_uav:
                task_pos = np.array([task['pos_x'], task['pos_y'], self.uav_spec['hover_altitude_m']])
                distance = np.linalg.norm(current_pos - task_pos)

                # 【修正】使用 SystemModels 计算更精确的飞行时间和能耗
                # 假设匀速飞行
                travel_time = distance / self.uav_spec['max_velocity_ms']
                travel_energy = self.models.calculate_uav_power_W(self.velocity, 'move') * travel_time

                # 【修正】使用 SystemModels 计算更精确的服务时间
                # 假设无人机在任务点上方盘旋
                service_rate_bps = self.models.calculate_rate_bps(
                    task_pos,
                    np.array([task['pos_x'], task['pos_y'], 0]),
                    {},
                    is_a2a=False
                )

                # 如果速率为0或接近0，施加巨大惩罚
                if service_rate_bps < 1e-6:
                    # 使用一个大的惩罚值来代替无穷大，以防止计算问题
                    service_time = 1e6
                    service_energy = self.models.calculate_uav_power_W(np.array([0, 0, 0]), 'hover') * service_time
                    total_energy_penalty += 1e9  # 增加一个巨大的惩罚
                else:
                    service_time = (task['data_size_mbits'] * 1e6) / service_rate_bps
                    service_energy = self.models.calculate_uav_power_W(np.array([0, 0, 0]), 'hover') * service_time

                uav_time += travel_time + service_time
                uav_energy_spent += travel_energy + service_energy
                current_pos = task_pos

            uav_completion_times.append(uav_time)

            if uav_energy_spent > self.uav_spec['battery_capacity_J'] * 1000:
                total_energy_penalty += 1e9  # 增加一个巨大的惩罚

        makespan = max(uav_completion_times) if uav_completion_times else 0
        return makespan + total_energy_penalty

    def _selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        """使用锦标赛选择法选择父代。"""
        parents = []
        pop_size = len(population)
        num_to_select = pop_size - self.ga_params['elite_size']
        for _ in range(num_to_select):
            indices = np.random.choice(pop_size, self.ga_params['tournament_size'], replace=False)
            winner_idx = max(indices, key=lambda i: fitness_scores[i])
            parents.append(population[winner_idx])
        return parents

    def _crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """对父代进行两点交叉。"""
        offspring = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i + 1] if i + 1 < len(parents) else parents[0]
            if np.random.rand() < self.ga_params['crossover_rate'] and len(p1) > 2:
                pt1, pt2 = sorted(np.random.choice(range(1, len(p1)), 2, replace=False))
                c1 = np.concatenate((p1[:pt1], p2[pt1:pt2], p1[pt2:]))
                c2 = np.concatenate((p2[:pt1], p1[pt1:pt2], p2[pt2:]))
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1.copy(), p2.copy()])
        return offspring

    def _mutate(self, offspring: List[np.ndarray]) -> List[np.ndarray]:
        """对每个染色体进行随机重置变异。"""
        num_uavs = self.problem.num_uavs
        for chromosome in offspring:
            for i in range(len(chromosome)):
                if np.random.rand() < self.ga_params['mutation_rate']:
                    chromosome[i] = np.random.randint(0, num_uavs)
        return offspring

    def _select_elites(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        """选择适应度最高的个体作为精英。"""
        elite_indices = np.argsort(fitness_scores)[-self.ga_params['elite_size']:]
        return [population[i].copy() for i in elite_indices]

    def _get_best_individual(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """从最终种群中找到最佳个体。"""
        return population[np.argmax(fitness_scores)]

    def _decode_to_plan(self, chromosome: np.ndarray) -> tuple[Dict, float]:
        """将最终染色体解码为轨迹和调度。"""
        task_assignments = {uid: [] for uid in self.uav_ids}
        for task_idx, uav_idx in enumerate(chromosome):
            task_assignments[self.uav_ids[uav_idx]].append(self.tasks[task_idx])

        trajectories = {}
        schedule = {}
        uav_completion_times = []

        for uav_id, tasks_for_uav in task_assignments.items():
            current_pos = self.uav_initial_pos[uav_id]
            tasks_for_uav.sort(key=lambda t: np.linalg.norm(current_pos - np.array([t['pos_x'], t['pos_y'], 0])))

            uav_trajectory = [current_pos]
            uav_schedule = []
            current_time = 0.0

            # 【新增】解码时也考虑电量
            current_energy = self.uav_spec['battery_capacity_J'] * 1000

            for task in tasks_for_uav:
                target_pos = np.array([task['pos_x'], task['pos_y'], self.uav_spec['hover_altitude_m']])
                distance = np.linalg.norm(current_pos - target_pos)
                travel_duration = distance / self.uav_spec['max_velocity_ms']
                travel_energy = self.models.calculate_uav_power_W(self.velocity, 'move') * travel_duration

                # 检查是否还有足够的电量飞行
                if current_energy < travel_energy:
                    travel_duration = float('inf')  # 任务无法完成

                uav_schedule.append({'task_type': 'fly_to_task', 'start_time': current_time,
                                     'end_time': current_time + travel_duration})
                current_time += travel_duration
                uav_trajectory.append(target_pos)
                current_pos = target_pos
                current_energy -= travel_energy

                # 如果无法到达，则剩余任务也无法完成
                if current_time == float('inf'):
                    break

                service_rate_bps = self.models.calculate_rate_bps(
                    target_pos,
                    np.array([task['pos_x'], task['pos_y'], 0]),
                    {},
                    is_a2a=False
                )

                if service_rate_bps < 1e-6:
                    service_duration = float('inf')
                else:
                    service_duration = (task['data_size_mbits'] * 1e6) / service_rate_bps

                service_energy = self.models.calculate_uav_power_W(np.array([0, 0, 0]), 'hover') * service_duration

                if current_energy < service_energy:
                    service_duration = float('inf')

                uav_schedule.append({'task_type': 'collect_data', 'start_time': current_time,
                                     'end_time': current_time + service_duration, 'task_id': task['task_id']})
                current_time += service_duration
                uav_trajectory.append(target_pos)
                current_energy -= service_energy

            trajectories[uav_id] = uav_trajectory
            schedule[uav_id] = uav_schedule
            uav_completion_times.append(current_time)

        planned_makespan = max(uav_completion_times) if uav_completion_times else 0
        decision_vars = {"trajectories": trajectories, "schedule": schedule}

        return decision_vars, planned_makespan

    def _create_empty_solution(self, comp_time: float) -> Solution:
        """为没有任务的情况创建解决方案。"""
        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables={
                "trajectories": {uid: [self.uav_initial_pos[uid]] for uid in self.uav_ids},
                "schedule": {uid: [] for uid in self.uav_ids}
            },
            planned_objective=0,
            computation_time_s=comp_time,
            is_stochastic=self.is_stochastic
        )