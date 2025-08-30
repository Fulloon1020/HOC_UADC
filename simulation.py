# file: simulation.py

import numpy as np
from problem_def import Problem, Solution
from typing import Dict, List, Tuple


class SystemModels:
    """
    实现了论文中描述的高保真物理模型。
    这个类代表了模拟世界的“地面真理”。
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.uav_params = problem.uav_config['default_params']
        self.comm_params = self.uav_params['comm_params']
        self.buildings = problem.city_map
        self.scenario = problem.scenario

    def check_los(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """实现了基于 Eq. (8) 的确定性LoS检查。"""
        line_start, line_end = p1, p2
        if np.linalg.norm(line_end - line_start) < 1e-6:
            return True

        for building in self.buildings:
            footprint = np.array(building['footprint_xy'])
            min_coords = np.append(np.min(footprint, axis=0), 0)
            max_coords = np.append(np.max(footprint, axis=0), building['height_m'])

            if self._line_intersects_aabb(line_start, line_end, min_coords, max_coords):
                return False  # NLoS (非视距)
        return True  # LoS (视距)

    def _line_intersects_aabb(self, p1, p2, box_min, box_max):
        """辅助函数：3D线段与AABB（轴对齐包围盒）的相交检测。"""
        line_dir = p2 - p1
        inv_dir = 1.0 / (line_dir + 1e-9)

        t_min = (box_min - p1) * inv_dir
        t_max = (box_max - p1) * inv_dir

        t_enter = np.max(np.minimum(t_min, t_max))
        t_exit = np.min(np.maximum(t_min, t_max))

        return t_enter < t_exit and t_exit > 0 and t_enter < 1

    def get_path_loss_db(self, p1: np.ndarray, p2: np.ndarray, is_a2a: bool) -> float:
        """实现路径损耗模型。A2G（空对地）对应Eq.(10)，A2A（空对空）对应Eq.(16)。"""
        dist = np.linalg.norm(p1 - p2)
        if dist < 1.0:
            dist = 1.0

        fc = self.scenario['carrier_frequency_hz']
        c = 299792458.0

        fspl_db = 20 * np.log10(dist) + 20 * np.log10(fc) + 20 * np.log10(4 * np.pi / c)

        if is_a2a:
            return fspl_db
        else:
            los = self.check_los(p1, p2)
            eta_db = self.scenario['eta_los'] if los else self.scenario['eta_nlos']
            return fspl_db + eta_db

    def calculate_rate_bps(self, p_tx: np.ndarray, p_rx: np.ndarray, world_state: dict, is_a2a: bool) -> float:
        """实现基于SINR的数据速率。A2G对应Eq.(12)，A2A对应Eq.(17)。"""
        path_loss_db = self.get_path_loss_db(p_tx, p_rx, is_a2a)
        path_gain_linear = 10 ** (-path_loss_db / 10.0)

        comm = self.comm_params
        if is_a2a:
            tx_power = self.uav_params['power_transmit_kw'] * 1000
            tx_gain_linear = 10 ** (comm['gain_uav_dBi'] / 10.0)
            rx_gain_linear = 10 ** (comm['gain_uav_dBi'] / 10.0)
            bandwidth = comm['bandwidth_a2a_hz']
            noise_power = comm['noise_power_watt']
        else:
            tx_power = comm['P_iot_watt']
            tx_gain_linear = 10 ** (comm['gain_iot_dBi'] / 10.0)
            rx_gain_linear = 10 ** (comm['gain_uav_dBi'] / 10.0)
            bandwidth = comm['bandwidth_a2g_hz']
            noise_power = comm['noise_power_watt']

        signal_power = tx_power * tx_gain_linear * rx_gain_linear * path_gain_linear
        interference = self._calculate_interference(p_rx, world_state)
        sinr = signal_power / (noise_power + interference)
        return bandwidth * np.log2(1 + sinr) if sinr > 0 else 0.0

    def _calculate_interference(self, p_rx: np.ndarray, world_state: dict) -> float:
        """辅助函数：计算干扰（目前设为0）。"""
        return 0.0

    def calculate_uav_power_W(self, velocity: np.ndarray, role: str) -> float:
        """实现无人机总功耗 (W)。对应 Eq. (18) 和 (19)。"""
        propulsion_power_W = (self.uav_params['power_hover_kw'] * 1000) + \
                             self.uav_params['power_move_coeff'] * np.linalg.norm(velocity) ** 2

        comm_power_W = self.uav_params['power_transmit_kw'] * 1000 if 'transmit' in role else 0.0
        return propulsion_power_W + comm_power_W


# --- SimulationEngine class with corrected logic ---
class SimulationEngine:
    """
    高保真模拟器，负责执行一个规划方案。
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.scenario = problem.scenario

        # 【修改】内部实现参数标准化映射
        uav_params = problem.uav_config['default_params']
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

        # 【修正】初始化 SystemModels 实例
        self.models = SystemModels(problem)
        self.delta_t = self.scenario['simulation_time_step_s']

    def run_simulation(self, solution: Solution) -> Tuple[Dict, Dict, Dict]:
        """
        执行模拟循环。
        """
        # --- 初始化模拟状态 ---
        uav_states = {
            uav_id: {
                'pos': np.array(self.problem.uav_initial_state_map[uav_id]['start_pos']),
                'energy': self.uav_spec['battery_capacity_J'],
                'is_active': True,  # 【新增】添加无人机活跃状态标志
                'data_buffer': {task['task_id']: 0 for task in self.problem.tasks}
            } for uav_id in self.problem.uav_ids
        }

        trajectories = self._interpolate_trajectories(solution.decision_variables.get("trajectories", {}))

        tasks_data_collected = {task['task_id']: 0 for task in self.problem.tasks}
        completion_times = {task['task_id']: float('inf') for task in self.problem.tasks}

        history = {uav_id: [] for uav_id in self.problem.uav_ids}

        # --- 模拟主循环 ---
        max_steps = int(self.scenario['mission_duration_s'] / self.delta_t)
        for t_step in range(max_steps):
            current_time = t_step * self.delta_t

            for uav_id in self.problem.uav_ids:
                state = uav_states[uav_id]

                # 【修正】如果无人机电量耗尽或不活跃，跳过所有步骤
                if not state['is_active']:
                    continue

                # 1. 更新无人机位置
                target_pos = trajectories[uav_id][t_step]
                distance_moved = np.linalg.norm(target_pos - state['pos'])
                velocity_vector = (target_pos - state['pos']) / self.delta_t
                state['pos'] = target_pos

                # 2. 更新能耗
                # 【修正】使用 SystemModels 实例来计算能耗
                power_consumed = self.models.calculate_uav_power_W(velocity_vector, 'move')
                energy_consumed = power_consumed * self.delta_t
                state['energy'] -= energy_consumed

                # 【新增】检查电量是否耗尽
                if state['energy'] <= 0:
                    state['is_active'] = False
                    continue  # 停止该无人机后续所有动作

                # 3. 执行通信任务
                for task in self.problem.tasks:
                    task_id = task['task_id']
                    if completion_times[task_id] != float('inf'):
                        continue

                    task_pos = np.array([task['pos_x'], task['pos_y'], 0])
                    dist_to_task = np.linalg.norm(state['pos'][:2] - task_pos[:2])

                    # --- START: 添加调试代码 ---
                    # 为了避免刷屏，我们可以只在距离很近的时候打印信息
                    if dist_to_task < self.scenario.get('collection_radius_m', 50) + 20:  # 靠近时就打印
                        print(f"\n[SIM DEBUG t={current_time:.2f}] UAV {uav_id} is close to Task {task_id}")
                        print(f"  UAV Pos: {np.round(state['pos'], 2)}")
                        print(f"  Task Pos: {np.round(task_pos, 2)}")
                        print(
                            f"  Distance: {dist_to_task:.2f}m / Collection Radius: {self.scenario['collection_radius_m']}m")
                    # --- END: 添加调试代码 ---

                    if dist_to_task < self.scenario['collection_radius_m']:

                        # --- START: 添加调试代码 ---
                        is_los = self.models.check_los(state['pos'], task_pos)
                        rate = self.models.calculate_rate_bps(state['pos'], task_pos, {}, is_a2a=False)
                        print(f"  >>> IN RANGE! <<<")
                        print(f"  LoS Status: {is_los}")
                        print(f"  Calculated Rate: {rate:.4f} bps")
                        # --- END: 添加调试代码 ---

                        data_transmitted = rate * self.delta_t
                        tasks_data_collected[task_id] += data_transmitted

                        if tasks_data_collected[task_id] >= task['data_size_mbits'] * 1e6:
                            completion_times[task_id] = current_time + self.delta_t

                # 4. 记录历史状态
                history[uav_id].append({
                    'time': current_time,
                    'x': state['pos'][0], 'y': state['pos'][1], 'z': state['pos'][2],
                    'energy': state['energy']
                })

        # 【修正】确保 final_states 包含最新状态
        final_states = uav_states

        return history, final_states, completion_times

    def _interpolate_trajectories(self, trajectories: Dict) -> Dict:
        """确保轨迹点数量与模拟步数一致。"""
        max_steps = int(self.scenario['mission_duration_s'] / self.delta_t)
        interp_trajs = {}
        for uav_id in self.problem.uav_ids:
            initial_pos = self.problem.uav_initial_state_map[uav_id]['start_pos']
            planned_traj = np.array(trajectories.get(uav_id, [initial_pos]))
            if len(planned_traj) < 2:
                planned_traj = np.array([planned_traj[0], planned_traj[0]])

            time_points_orig = np.linspace(0, self.scenario['mission_duration_s'], len(planned_traj))
            time_points_new = np.linspace(0, self.scenario['mission_duration_s'], max_steps)

            x_interp = np.interp(time_points_new, time_points_orig, planned_traj[:, 0])
            y_interp = np.interp(time_points_new, time_points_orig, planned_traj[:, 1])
            z_interp = np.interp(time_points_new, time_points_orig, planned_traj[:, 2])

            interp_trajs[uav_id] = np.vstack([x_interp, y_interp, z_interp]).T
        return interp_trajs