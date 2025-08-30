# file: solvers/aoi_eato_solver.py

import time
import numpy as np
import random
from typing import Dict, Tuple

# 导入您自己框架的标准数据结构
from problem_def import Problem, Solution
from solvers.base_solver import BaseSolver

# 【修正】从 aoi_eato_lib 导入所有必需的模块和数据类
from aoi_eato_lib.model.environment import Environment
from aoi_eato_lib.model.uav import UAV
from aoi_eato_lib.model.trajectory import TrajectorySolution
from aoi_eato_lib.optimization.speed_optimization import SpeedOptimizerSCA
from aoi_eato_lib.optimization.visiting_seq_optimization import VisitingSequenceOptimizerGA
from aoi_eato_lib.optimization.hovering_pos_optimization import HoveringPositionOptimizerSCA
from aoi_eato_lib.configs.config_loader import get_simulation_config
from aoi_eato_lib.configs.data_classes import (
    EnvironmentConfig, UAVConfig,
    SpeedParams, EnergyParams, PropulsionPowerModelParams,
    SensingParams, CommunicationParams, LosParams
)


class AoiEatoSolver(BaseSolver):
    """
    将 AoI-EaTO 算法封装为一个求解器，以便接入您的HOC-UADC实验框架。
    """
    algorithm_name = "AoI-EaTO"
    is_stochastic = True  # 因其内部使用遗传算法

    def __init__(self, problem_instance: Problem):
        super().__init__(problem_instance)

        # 加载 AoI-EaTO 自身的仿真参数
        self.sim_cfg_aoi_eato = get_simulation_config()

        # 【修正】添加缺失的 uav_ids 和 uav_initial_pos 属性
        self.uav_ids = self.problem.uav_ids
        self.uav_initial_pos = {uid: np.array(self.problem.uav_initial_state_map[uid]['start_pos']) for uid in
                                self.uav_ids}

    def _solve(self) -> Solution:
        """
        实现了 AoI-EaTO 的主逻辑 (Algorithm 4 from its paper)。
        """
        start_time = time.time()

        if self.problem.num_tasks == 0:
            return self._create_empty_solution(time.time() - start_time)

        # 1. 数据适配：将您的 Problem 对象转换为 AoI-EaTO 的内部模型
        env, uav = self._adapt_problem_to_aoi_eato_models()

        # 2. 初始化轨迹
        trajectory = self._initialize_trajectory(env)

        # 3. 运行 AoI-EaTO 的主迭代循环 (交替优化)
        max_outer_iterations = self.sim_cfg_aoi_eato.aoi_eato.max_iterations
        convergence_tol = self.sim_cfg_aoi_eato.aoi_eato.convergence_threshold_seconds

        previous_objective = float('inf')
        best_trajectory = trajectory
        best_mission_time = float('inf')

        for i in range(max_outer_iterations):
            print(f"\n--- [AoI-EaTO] Main Iteration {i + 1}/{max_outer_iterations} ---")

            # Stage 1: 速度优化 (SCA)
            speed_opt = SpeedOptimizerSCA(uav, env, self.sim_cfg_aoi_eato, trajectory)
            v_opt, slacks_opt, _ = speed_opt.optimize_speeds()
            if v_opt is not None:
                trajectory.speeds_v_mps = v_opt
                if slacks_opt is not None: trajectory.slacks_lambda_m = slacks_opt

            # Stage 2a: 访问序列优化 (GA)
            seq_opt_ga = VisitingSequenceOptimizerGA(uav, env, self.sim_cfg_aoi_eato, trajectory)
            seq_opt, _ = seq_opt_ga.optimize_sequence()
            if seq_opt is not None:
                trajectory.visiting_sequence_pi = seq_opt

            # Stage 2b: 悬停位置优化 (SCA)
            hover_opt = HoveringPositionOptimizerSCA(uav, env, self.sim_cfg_aoi_eato, trajectory)
            q_opt, p_opt, _ = hover_opt.optimize_hovering_positions()
            if q_opt is not None and p_opt is not None:
                trajectory.hover_positions_q_tilde_meters = q_opt
                trajectory.hover_positions_p_meters = p_opt

            current_mission_time, _, _ = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)

            if current_mission_time < best_mission_time:
                best_mission_time = current_mission_time
                best_trajectory = trajectory

            if abs(previous_objective - current_mission_time) < convergence_tol and i > 0:
                print(f"--- [AoI-EaTO] Converged at iteration {i + 1} ---")
                break
            previous_objective = current_mission_time

        computation_time_s = time.time() - start_time

        # 4. 结果转换：将 AoI-EaTO 的解转换为您的标准 Solution 对象
        decision_variables = self._convert_trajectory_to_decision_vars(best_trajectory, env, uav)

        return Solution(
            algorithm_name=self.algorithm_name,
            decision_variables=decision_variables,
            planned_objective=best_mission_time,
            computation_time_s=computation_time_s,
            is_stochastic=self.is_stochastic
        )

    def _adapt_problem_to_aoi_eato_models(self) -> Tuple[Environment, UAV]:
        """
        适配器函数：严格按照 data_classes.py 的定义来创建 AoI-EaTO 的模型对象。
        """
        # --- 1. 为 AoI-EaTO 创建 EnvironmentConfig 实例 ---
        initial_uav_pos_3d = self.problem.uav_initial_state_map[self.problem.uav_ids[0]]['start_pos']

        env_config_for_aoi_eato = EnvironmentConfig(
            num_areas=self.problem.num_tasks,
            area_radius_ra_meters=10.0,
            area_positions_wk_meters=[[t['pos_x'], t['pos_y']] for t in self.problem.tasks],
            gbs_altitude_hb_meters=self.uav_spec.get('gbs_altitude_m', 20.0),
            initial_uav_position_s0_meters=initial_uav_pos_3d[:2]
        )
        env = Environment(config=env_config_for_aoi_eato, sim_config=self.sim_cfg_aoi_eato)

        # --- 2. 为 AoI-EaTO 创建 UAVConfig 实例 (自下而上构建) ---
        speed_cfg = SpeedParams(
            max_vmax_mps=self.uav_spec['max_velocity_ms'],
            min_init_vmin_mps=1.0
        )

        propulsion_cfg = PropulsionPowerModelParams(
            P0_watts=self.uav_spec.get('P0_W', 79.86), P1_watts=self.uav_spec.get('P1_W', 88.63),
            v_tip_mps=120, v0_hover_mps=4.03, d0_ratio=0.6,
            rho_kg_per_m3=1.225, s_solidity=0.05, A_disc_m2=0.503
        )
        energy_cfg = EnergyParams(
            max_elimit_joule=self.uav_spec['battery_capacity_J'],
            propulsion_power_model=propulsion_cfg
        )

        sensing_cfg = SensingParams(
            zeta_parameter=0.01, p_th_probability=0.99, t_int_seconds=1.0,
            uav_monitoring_range_ru_meters=26.8, data_collection_time_te_seconds=10.0
        )

        los_params_cfg = LosParams(a=10, b=0.6)

        # 【修正】在创建 CommunicationParams 实例时，补上缺失的 kappa_nlos 参数
        comm_cfg = CommunicationParams(
            pu_transmit_power_dbm=20, beta0_db=-60,
            plos_approx_probability=1.0, path_loss_alpha=2.2,
            noise_power_sigma2_dbm=-110, snr_gap_gamma_db=8.2,
            snr_min_db=5.0, channel_bandwidth_B_mhz=self.scenario['a2g_bandwidth_hz'] / 1e6,
            d_min_comm_meters=10.0,
            data_packet_size_Sk_mbits=self.problem.tasks[0]['data_size_mbits'] if self.problem.tasks else 30.0,
            los_params=los_params_cfg,
            kappa_nlos=0.2  # 补上这个参数，0.2是论文中常用的一个值
        )

        uav_config_for_aoi_eato = UAVConfig(
            altitude_hu_meters=self.uav_spec['max_altitude_m'],
            speed=speed_cfg,
            energy=energy_cfg,
            sensing=sensing_cfg,
            communication=comm_cfg
        )
        uav = UAV(config=uav_config_for_aoi_eato, environment=env)

        return env, uav

    def _initialize_trajectory(self, env: Environment) -> TrajectorySolution:
        """Creates a valid initial TrajectorySolution object."""
        num_areas = env.get_total_monitoring_areas()
        initial_sequence = list(range(num_areas))
        random.shuffle(initial_sequence)

        return TrajectorySolution(
            visiting_sequence_pi=initial_sequence,
            speeds_v_mps=np.full(num_areas + 1, self.uav_spec['max_velocity_ms'] / 2.0),
            hover_positions_q_tilde_meters=env.get_all_area_positions_wk(),
            hover_positions_p_meters=env.get_all_area_positions_wk(),
            slacks_lambda_m=np.ones(num_areas + 1) * 0.1
        )

    def _convert_trajectory_to_decision_vars(self, trajectory: TrajectorySolution, env: Environment, uav: UAV) -> Dict:
        """将AoI-EaTO的解转换为您框架的标准 decision_variables 字典。"""
        # AoI-EaTO是单无人机算法，我们将计划分配给第一个无人机
        main_uav_id = self.uav_ids[0]

        # 从TrajectorySolution的辅助函数中获取路径分段信息
        path_segments = trajectory.get_path_segment_details(env)

        # 1. 构建轨迹点列表
        uav_waypoints = []
        if path_segments:
            # 添加起点
            uav_waypoints.append(np.append(path_segments[0]['start_pos'], self.uav_spec['max_altitude_m']))
            # 添加所有分段的终点
            for segment in path_segments:
                uav_waypoints.append(np.append(segment['end_pos'], self.uav_spec['max_altitude_m']))

        # 2. 构建调度事件列表
        uav_schedule = []
        current_time = 0.0

        # 模拟AoI-EaTO的时间计算过程来生成调度
        total_time_T, _, aois = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
        # 由于我们无法直接获取每个事件的精确时长，这里生成一个简化的调度
        uav_schedule.append({'task_type': 'mission', 'start_time': 0, 'end_time': total_time_T})

        decision_vars = {
            "trajectories": {main_uav_id: uav_waypoints},
            "schedule": {main_uav_id: uav_schedule}
        }

        # 为其他未使用的无人机分配空计划
        for other_uav_id in self.uav_ids[1:]:
            decision_vars["trajectories"][other_uav_id] = [self.uav_initial_pos[other_uav_id]]
            decision_vars["schedule"][other_uav_id] = []

        return decision_vars

    def _create_empty_solution(self, comp_time: float) -> Solution:
        """在没有任务时创建一个空的解。"""
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