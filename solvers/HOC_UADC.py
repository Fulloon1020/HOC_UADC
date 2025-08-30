# file: solvers/HOC_UADC.py

import time
import numpy as np
import pulp
import casadi as ca
from typing import Dict, Any, List, Tuple

from problem_def import Problem, Solution
from simulation import SystemModels
from solvers.base_solver import BaseSolver


# ##############################################################################
# SECTION 2: ALGORITHM CORE COMPONENTS (GMP and LTO)
# ##############################################################################

class GlobalMissionPlanner:
    """
    顶层任务规划器 (GMP)。
    使用 MILP (混合整数线性规划) 为无人机分配任务，目标是最小化任务集的最大完成时间。
    """

    def __init__(self, problem: Problem, params: Dict, system_models: SystemModels):
        self.problem = problem
        self.params = params
        self.uav_spec = self.params['uav_spec']
        self.scenario = self.params['scenario']
        self.models = system_models

    def run(self, uav_states: Dict, tasks_status: Dict) -> Dict:
        """运行MILP求解器，进行全局任务分配。"""
        prob = pulp.LpProblem("GMP_Makespan_Minimization", pulp.LpMinimize)
        uav_ids = self.problem.uav_ids
        active_tasks = [t for t in self.problem.tasks if
                        not tasks_status.get(t['task_id'], {}).get('is_complete', False)]

        if not active_tasks:
            return {uid: [] for uid in uav_ids}
        task_ids = [t['task_id'] for t in active_tasks]

        x = pulp.LpVariable.dicts("x", (task_ids, uav_ids), cat='Binary')
        T_max = pulp.LpVariable("T_max", lowBound=0)
        prob += T_max, "Minimize_Makespan"

        for tid in task_ids:
            prob += pulp.lpSum(x[tid][uid] for uid in uav_ids) == 1, f"task_assignment_{tid}"

        for uid in uav_ids:
            uav_pos = uav_states[uid]['position']
            total_time_for_uav = 0
            total_energy_for_uav = 0

            for tid in task_ids:
                task = self.problem.task_map[tid]
                task_hover_pos = np.array([task['pos_x'], task['pos_y'], self.uav_spec['hover_altitude_m']])
                distance = np.linalg.norm(uav_pos - task_hover_pos)
                travel_time = distance / self.uav_spec['max_velocity_ms']
                task_pos_ground = np.array([task['pos_x'], task['pos_y'], 0])
                rate_bps = self.models.calculate_rate_bps(p_tx=task_pos_ground, p_rx=task_hover_pos, world_state={},
                                                          is_a2a=False)
                service_time = (task['data_size_mbits'] * 1e6) / rate_bps if rate_bps > 1e-9 else float('inf')
                travel_power = self.models.calculate_uav_power_W(np.array([self.uav_spec['max_velocity_ms'], 0, 0]),
                                                                 'move')
                service_power = self.models.calculate_uav_power_W(np.array([0, 0, 0]), 'hover')
                prop_energy = travel_power * travel_time
                hover_energy = service_power * service_time
                total_time_for_uav += x[tid][uid] * (travel_time + service_time)
                total_energy_for_uav += x[tid][uid] * (prop_energy + hover_energy)

                if 'deadline_s' in task and task['deadline_s'] is not None:
                    # 估算完成该任务的时刻 = 从UAV当前位置出发的旅行时间 + 服务时间
                    # 注意：这是一个简化的估算，因为它没有考虑无人机完成其他任务的序列
                    # 但对于高层规划来说，这是一个常用且有效的约束
                    estimated_completion_time = travel_time + service_time

                    # 添加约束：如果任务tid分配给无人机uid (即 x[tid][uid] == 1),
                    # 那么其估算完成时间必须小于等于其截止时间。
                    prob += estimated_completion_time * x[tid][uid] <= task[
                        'deadline_s'], f"deadline_constraint_{tid}_{uid}"
                # --- 修改结束 ---

            prob += T_max >= total_time_for_uav, f"makespan_constraint_{uid}"
            prob += total_energy_for_uav <= uav_states[uid]['energy_J'], f"energy_constraint_{uid}"

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.params['scenario'].get('gmp_time_limit_s', 20)))

        plan = {uid: [] for uid in uav_ids}
        if pulp.LpStatus[prob.status] == 'Optimal':
            for uid in uav_ids:
                assigned_tasks_obj = [self.problem.task_map[tid] for tid in task_ids if x[tid][uid].value() > 0.5]
                uav_pos = uav_states[uid]['position']
                assigned_tasks_obj.sort(key=lambda t: np.linalg.norm(uav_pos - np.array([t['pos_x'], t['pos_y'], 0])))
                plan[uid] = [t['task_id'] for t in assigned_tasks_obj]
        else:
            unassigned_tasks = task_ids[:]
            if unassigned_tasks:
                for uid in uav_ids:
                    if unassigned_tasks:
                        plan[uid].append(unassigned_tasks.pop(0))
        return plan


class LocalTrajectoryOptimizer:
    """局部轨迹优化器 (LTO)。"""

    def __init__(self, uav_id: str, problem: Problem, params: Dict, system_models: SystemModels,
                 ablation_mode: str = 'full'):
        self.uav_id = uav_id
        self.problem = problem
        self.params = params
        self.uav_spec = self.params['uav_spec']
        self.scenario = self.params['scenario']
        self.dt = self.scenario['simulation_time_step_s']
        self.Kp = int(self.scenario['lto_prediction_horizon_s'] / self.dt)
        self.models = system_models
        # 【新增】存储当前的消融实验模式
        self.ablation_mode = ablation_mode

    def run(self, uav_state: Dict, directive: List[str]) -> Tuple[np.ndarray, str]:
        if not directive:
            return np.zeros(3), 'hover'

        target_task_id = directive[0]
        target_task = self.problem.task_map[target_task_id]
        p_ref = np.array([target_task['pos_x'], target_task['pos_y'], self.uav_spec['hover_altitude_m']])

        # --- 【消融实验逻辑切换】 ---
        if self.ablation_mode == 'no_nmpc':
            return self._run_simplified_lto(uav_state, p_ref)

        dist_to_target = np.linalg.norm(uav_state['position'][:2] - p_ref[:2])
        candidate_roles = ['fly_to_task']
        if dist_to_target < self.scenario.get('collection_radius_m', 50):
            candidate_roles.append('collect_data')

        # 如果是“无角色优化”模式，则强制只使用第一个候选角色
        if self.ablation_mode == 'no_role_optimization':
            candidate_roles = [candidate_roles[0]]

        best_cost, best_role, best_control = float('inf'), 'hover', np.zeros(3)
        for role in candidate_roles:
            cost, sol, u_var = self._solve_nmpc_for_role(uav_state, p_ref, target_task, role)
            total_cost = cost if cost is not None else float('inf')
            if role != uav_state['role']:
                total_cost += self.scenario.get('role_switching_penalty', 100)
            if total_cost < best_cost:
                best_cost, best_role = total_cost, role
                if sol:
                    best_control = sol.value(u_var)[:, 0]
        return best_control, best_role

    def _run_simplified_lto(self, uav_state: Dict, p_ref: np.ndarray) -> Tuple[np.ndarray, str]:
        """【新增】为 'no_nmpc' 模式实现的简化版LTO。"""
        direction = p_ref - uav_state['position']
        dist = np.linalg.norm(direction)

        # 以恒定速度飞向目标
        target_velocity = direction / dist * self.uav_spec['max_velocity_ms'] if dist > 1e-6 else np.zeros(3)

        # 计算所需的加速度（简化控制）
        control_input = (target_velocity - uav_state['velocity']) / self.dt
        control_norm = np.linalg.norm(control_input)
        if control_norm > self.uav_spec['max_acceleration_ms2']:
            control_input = control_input / control_norm * self.uav_spec['max_acceleration_ms2']

        role = 'collect_data' if dist < self.scenario.get('collection_radius_m', 50) else 'fly_to_task'
        return control_input, role

    # ... (其他LTO函数 _solve_nmpc_for_role, _build_objective 等保持不变) ...
    def _solve_nmpc_for_role(self, uav_state: Dict, p_ref: np.ndarray, target_task: Dict, role: str) -> Tuple[
        float, Any, Any]:
        opti = ca.Opti()
        p, v, u = opti.variable(3, self.Kp + 1), opti.variable(3, self.Kp + 1), opti.variable(3, self.Kp)
        cost = self._build_objective(p, u, p_ref, target_task, role)
        opti.minimize(cost)
        self._add_dynamics_constraints(opti, p, v, u, uav_state)
        self._add_kinematic_constraints(opti, p, v, u)
        self._add_energy_constraint(opti, v, uav_state)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver('ipopt', opts)
        try:
            sol = opti.solve()
            return sol.value(cost), sol, u
        except:
            return float('inf'), None, None

    def _build_objective(self, p, u, p_ref, target_task, role):
        cost = 0;
        Q = np.diag(self.scenario['lto_weights']['tracking']);
        R = np.diag(self.scenario['lto_weights']['control']);
        w_R = self.scenario['lto_weights']['rate']
        for k in range(self.Kp):
            cost += ca.mtimes([(p[:, k] - p_ref).T, Q, (p[:, k] - p_ref)]);
            cost += ca.mtimes([u[:, k].T, R, u[:, k]])
            if role == 'collect_data':
                p_ground = np.array([target_task['pos_x'], target_task['pos_y'], 0]);
                dist = ca.sqrt(ca.sumsqr(p[:, k] - p_ground) + 1e-6)
                fspl_db = 20 * ca.log10(dist) + 20 * ca.log10(self.scenario['carrier_frequency_hz']) + 20 * ca.log10(
                    4 * np.pi / 299792458.0);
                pl_db = fspl_db + self.scenario['eta_los']
                h_gain = 10 ** (-pl_db / 10.0);
                sinr = (self.scenario['iot_tx_power_watt'] * h_gain) / self.scenario['noise_power_watt'];
                rate = self.scenario['a2g_bandwidth_hz'] * (ca.log(1 + sinr) / ca.log(2))
                cost -= w_R * rate
        return cost

    def _add_dynamics_constraints(self, opti, p, v, u, uav_state):
        for k in range(self.Kp): opti.subject_to(p[:, k + 1] == p[:, k] + v[:, k] * self.dt); opti.subject_to(
            v[:, k + 1] == v[:, k] + u[:, k] * self.dt)
        opti.subject_to(p[:, 0] == uav_state['position']);
        opti.subject_to(v[:, 0] == uav_state['velocity'])

    def _add_kinematic_constraints(self, opti, p, v, u):
        for k in range(self.Kp + 1): opti.subject_to(
            ca.sumsqr(v[:, k]) <= self.uav_spec['max_velocity_ms'] ** 2); opti.subject_to(
            opti.bounded(self.uav_spec['min_altitude_m'], p[2, k], self.uav_spec['max_altitude_m']))
        for k in range(self.Kp): opti.subject_to(ca.sumsqr(u[:, k]) <= self.uav_spec['max_acceleration_ms2'] ** 2)

    def _add_energy_constraint(self, opti, v, uav_state):
        total_energy_consumption = 0
        for k in range(self.Kp): speed_sq = ca.sumsqr(v[:, k]); prop_power = self.uav_spec['hover_power_W'] + \
                                                                             self.uav_spec[
                                                                                 'prop_power_coeff'] * speed_sq; total_energy_consumption += prop_power * self.dt
        opti.subject_to(total_energy_consumption <= uav_state['energy_J'])


# ##############################################################################
# SECTION 3: 主求解器类
# ##############################################################################

class HOC_UADCSolver(BaseSolver):
    algorithm_name = "HOC-UADC"
    is_stochastic = False

    def __init__(self, problem_instance: Problem, ablation_mode: str = 'full'):
        """【新增】构造函数增加 ablation_mode 参数。"""
        super().__init__(problem_instance)
        self.models = SystemModels(problem_instance)
        self.ablation_mode = ablation_mode

        # 根据消融模式，决定是否实例化完整的GMP和LTO
        if self.ablation_mode != 'no_strategic_layer':
            self.gmp = GlobalMissionPlanner(self.problem, self.params, self.models)

        self.ltos = {
            uav_id: LocalTrajectoryOptimizer(uav_id, self.problem, self.params, self.models, self.ablation_mode)
            for uav_id in self.problem.uav_ids
        }

        # 如果是消融实验，修改算法名称以便在结果中区分
        if self.ablation_mode != 'full':
            self.algorithm_name = f"HOC-UADC-({ablation_mode})"

    def _solve(self) -> Solution:
        start_time = time.time()
        uav_states = self._initialize_planning_states()
        tasks_status = {t['task_id']: {'data_collected_bits': 0, 'is_complete': False} for t in self.problem.tasks}
        planned_trajectories = {uid: [state['position']] for uid, state in uav_states.items()}
        planned_schedule_events = {uid: [] for uid in uav_states.keys()}
        dt = self.params['scenario']['simulation_time_step_s']
        mission_duration = self.params['scenario']['mission_duration_s']
        strategic_interval = self.params['scenario']['strategic_interval_s']

        # --- 【消融实验逻辑切换】 ---
        # 模式: 'no_receding_horizon'，只在开始时规划一次
        if self.ablation_mode == 'no_receding_horizon':
            print("[ABLATION MODE] No Receding Horizon: Running GMP once at t=0.")
            gmp_plan = self.gmp.run(uav_states, tasks_status)
            for uid in self.problem.uav_ids:
                uav_states[uid]['gmp_directive'] = gmp_plan.get(uid, [])

        for t_step in range(int(mission_duration / dt)):
            global_time = t_step * dt

            # --- GMP 执行 ---
            if self.ablation_mode == 'no_receding_horizon':
                pass  # 在此模式下，循环内不执行GMP
            elif self.ablation_mode == 'no_strategic_layer':
                if t_step % int(strategic_interval / dt) == 0:
                    print(f"[ABLATION MODE] No Strategic Layer: Running Greedy GMP at t={global_time:.1f}s...")
                    gmp_plan = self._run_gmp_greedy(uav_states, tasks_status)
                    for uid in self.problem.uav_ids:
                        uav_states[uid]['gmp_directive'] = gmp_plan.get(uid, [])
            else:  # 完整模式
                if t_step % int(strategic_interval / dt) == 0:
                    print(f"[t={global_time:.1f}s] Running Global Mission Planner (GMP)...")
                    gmp_plan = self.gmp.run(uav_states, tasks_status)
                    for uid in self.problem.uav_ids:
                        uav_states[uid]['gmp_directive'] = gmp_plan.get(uid, [])

            if all(s.get('is_complete', False) for s in tasks_status.values()):
                break

            for uid, state in uav_states.items():
                if state['energy_J'] <= 0:
                    planned_trajectories[uid].append(state['position'])
                    continue

                control_input, role = self.ltos[uid].run(state, state.get('gmp_directive', []))

                # ... (后续的状态更新和日志记录逻辑保持不变) ...
                state['velocity'] += control_input * dt
                speed = np.linalg.norm(state['velocity'])
                if speed > self.uav_spec['max_velocity_ms']:
                    state['velocity'] *= self.uav_spec['max_velocity_ms'] / speed
                state['position'] += state['velocity'] * dt
                state['position'][2] = np.clip(state['position'][2], self.uav_spec['min_altitude_m'],
                                               self.uav_spec['max_altitude_m'])
                power = self.models.calculate_uav_power_W(state['velocity'], role)
                state['energy_J'] -= power * dt
                if state['role'] != role:
                    if planned_schedule_events[uid]: planned_schedule_events[uid][-1]['end_time'] = global_time
                    planned_schedule_events[uid].append({'task_type': role, 'start_time': global_time, 'end_time': -1})
                state['role'] = role
                planned_trajectories[uid].append(state['position'].copy())
                if role == 'collect_data' and state.get('gmp_directive'):
                    task_id = state['gmp_directive'][0]
                    task = self.problem.task_map[task_id]
                    if not tasks_status[task_id]['is_complete']:
                        task_pos_ground = np.array([task['pos_x'], task['pos_y'], 0])
                        rate = self.models.calculate_rate_bps(p_tx=task_pos_ground, p_rx=state['position'],
                                                              world_state={}, is_a2a=False)
                        tasks_status[task_id]['data_collected_bits'] += rate * dt
                        if tasks_status[task_id]['data_collected_bits'] >= task['data_size_mbits'] * 1e6:
                            tasks_status[task_id]['is_complete'] = True

        computation_time_s = time.time() - start_time
        final_schedule = {}
        for uid, events in planned_schedule_events.items():
            if events:
                if events[-1]['end_time'] == -1:
                    events[-1]['end_time'] = mission_duration
            final_schedule[uid] = events

        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables={"trajectories": planned_trajectories, "schedule": final_schedule},
            computation_time_s=computation_time_s,
            is_stochastic=self.is_stochastic
        )

    def _run_gmp_greedy(self, uav_states: Dict, tasks_status: Dict) -> Dict:
        """【新增】为 'no_strategic_layer' 模式实现的贪心任务分配。"""
        plan = {uid: [] for uid in self.problem.uav_ids}
        active_tasks = [t for t in self.problem.tasks if
                        not tasks_status.get(t['task_id'], {}).get('is_complete', False)]

        # 为每个当前没有任务的无人机分配一个最近的未分配任务
        unassigned_tasks = list(active_tasks)
        for uid, state in uav_states.items():
            if not state.get('gmp_directive'):  # 如果无人机当前空闲
                if not unassigned_tasks: break

                uav_pos = state['position']
                closest_task = min(
                    unassigned_tasks,
                    key=lambda t: np.linalg.norm(uav_pos - np.array([t['pos_x'], t['pos_y'], 0]))
                )
                plan[uid] = [closest_task['task_id']]
                unassigned_tasks.remove(closest_task)
        return plan

    def _initialize_planning_states(self) -> Dict:
        states = {}
        spec = self.params['uav_spec']
        for uid, uav_info in self.problem.uav_initial_state_map.items():
            states[uid] = {
                'position': np.array(uav_info['start_pos'], dtype=float),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'energy_J': spec['battery_capacity_J'],
                'role': 'hover',
                'gmp_directive': []
            }
        return states