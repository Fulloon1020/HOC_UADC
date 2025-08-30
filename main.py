# file: main.py

import json
import pandas as pd
import numpy as np
import config
import pyproj
import os
from datetime import datetime
from typing import Type, Dict, Any
import copy

# --- 导入问题定义和所有求解器 ---
from problem_def import Problem, Solution
from simulation import SimulationEngine
from evaluation import Evaluator

# --- 导入本文算法和其他对比算法 ---
from solvers.HOC_UADC import HOC_UADCSolver
from solvers.GA_solver import GASolver
from solvers.pso_solver import PSOSolver
from solvers.aoi_eato_solver import AoiEatoSolver
from solvers.ao_ctm_solver import AocTmSolver
#from solvers.ee_noma_solver import EeNomaScaSolver
from solvers.hts_vnd_solver import HtsVndSolver
from solvers.min_time_tco_solver import MinTimeTcoSolver
from solvers.maco_solver import MacoSolver


def load_base_problem() -> Problem:
    """从文件加载一次“母版”问题实例，包含所有数据。"""
    with open(config.UAV_CONFIG_PATH, 'r') as f:
        uav_config = json.load(f)
    with open(config.CITY_MAP_PATH, 'r') as f:
        city_map = json.load(f)
    tasks = pd.read_csv(config.TASKS_PATH).to_dict('records')
    with open(config.SCENARIO_PATH, 'r') as f:
        scenario = json.load(f)

    print(
        f"--- Base data loaded from files (Tasks: {len(tasks)}, UAVs: {len(uav_config.get('initial_state', []))}) ---")

    # --- 坐标转换逻辑 (对母版数据进行一次性转换) ---
    print("--- Converting Lon/Lat coordinates to local metric (meter) coordinates ---")

    all_lon_lat_points = []
    for task in tasks:
        all_lon_lat_points.append((task['pos_x'], task['pos_y']))
    for uav_start in uav_config['initial_state']:
        all_lon_lat_points.append((uav_start['start_pos'][0], uav_start['start_pos'][1]))

    if not all_lon_lat_points:
        return Problem(uav_config=uav_config, tasks=tasks, city_map=city_map, scenario=scenario)

    center_lon, center_lat = np.mean(all_lon_lat_points, axis=0)
    source_crs = pyproj.CRS("EPSG:4326")
    target_crs = pyproj.CRS(proj='aeqd', lat_0=center_lat, lon_0=center_lon, datum='WGS84', units='m')
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    for task in tasks:
        lon, lat = task['pos_x'], task['pos_y']
        x_m, y_m = transformer.transform(lon, lat)
        task['pos_x'], task['pos_y'] = x_m, y_m
    for uav_start in uav_config['initial_state']:
        lon, lat = uav_start['start_pos'][0], uav_start['start_pos'][1]
        x_m, y_m = transformer.transform(lon, lat)
        uav_start['start_pos'][0], uav_start['start_pos'][1] = x_m, y_m
    for building in city_map:
        converted_footprint = []
        for lon, lat in building['footprint_xy']:
            x_m, y_m = transformer.transform(lon, lat)
            converted_footprint.append([x_m, y_m])
        building['footprint_xy'] = converted_footprint

    print("--- Base coordinate conversion complete ---")
    return Problem(uav_config=uav_config, tasks=tasks, city_map=city_map, scenario=scenario)


def create_problem_instance(base_problem: Problem, num_tasks: int, num_uavs: int) -> Problem | None:
    """从“母版”问题中抽取指定数量的任务和无人机，创建一个新的问题实例。"""
    if num_tasks > base_problem.num_tasks:
        print(f"Warning: Requested {num_tasks} tasks, but only {base_problem.num_tasks} are available. Skipping.")
        return None
    if num_uavs > base_problem.num_uavs:
        print(f"Warning: Requested {num_uavs} UAVs, but only {base_problem.num_uavs} are available. Skipping.")
        return None

    new_uav_config = copy.deepcopy(base_problem.uav_config)
    new_tasks = copy.deepcopy(base_problem.tasks[:num_tasks])

    new_uav_config['initial_state'] = new_uav_config['initial_state'][:num_uavs]
    if 'num_uavs' in new_uav_config:
        new_uav_config['num_uavs'] = num_uavs

    return Problem(
        uav_config=new_uav_config,
        tasks=new_tasks,
        city_map=copy.deepcopy(base_problem.city_map),
        scenario=copy.deepcopy(base_problem.scenario)
    )


def run_single_experiment(SolverClass: Type, problem_instance: Problem, solver_kwargs: Dict[str, Any], run_id: int = 1):
    # 动态生成一个临时的算法名称用于打印，实际名称由求解器内部决定
    temp_solver_name = SolverClass.algorithm_name
    if 'ablation_mode' in solver_kwargs and solver_kwargs['ablation_mode'] != 'full':
        temp_solver_name = f"{temp_solver_name}-({solver_kwargs['ablation_mode']})"

    print(
        f"\n--- Running Experiment: {temp_solver_name}, N={problem_instance.num_tasks}, M={problem_instance.num_uavs}, Run ID: {run_id} ---")

    # 使用 solver_kwargs 来实例化求解器
    solver = SolverClass(problem_instance, **solver_kwargs)
    planned_solution: Solution = solver.run()

    engine = SimulationEngine(problem_instance)
    history, final_states, completion_times = engine.run_simulation(planned_solution)

    evaluator = Evaluator(problem_instance, planned_solution, history, final_states, completion_times)
    metrics = evaluator.calculate_all_metrics()

    result_row = {
        "algorithm": planned_solution.algorithm_name,  # 使用求解器内部生成的、包含消融信息的名称
        "N_iot": problem_instance.num_tasks,
        "M_uav": problem_instance.num_uavs,
        "run_id": run_id,
        **metrics
    }
    planned_solution.display()
    return result_row, history, planned_solution


def save_visualization_data(solver_name: str, problem: Problem, history: Dict, planned_solution: Solution,
                            output_dir: str):
    """将单次运行的详细日志保存到文件中，用于绘图。"""
    # 清理算法名称中的括号以便创建有效的文件夹名
    safe_solver_name = solver_name.replace('(', '').replace(')', '')
    vis_dir = os.path.join(output_dir, "visualization_logs",
                           f"{safe_solver_name}_N{problem.num_tasks}_M{problem.num_uavs}")
    os.makedirs(vis_dir, exist_ok=True)

    for uav_id, uav_history in history.items():
        pd.DataFrame(uav_history).to_csv(os.path.join(vis_dir, f"trajectory_{uav_id}.csv"), index=False)

    schedule = planned_solution.decision_variables.get("schedule", {})
    if schedule:
        schedule_log = []
        for uav_id, tasks in schedule.items():
            for task in tasks:
                schedule_log.append({
                    "uav_id": uav_id,
                    "task_type": task.get('task_type', 'UNKNOWN'),
                    "start_time": task.get('start_time', -1),
                    "end_time": task.get('end_time', -1)
                })
        pd.DataFrame(schedule_log).to_csv(os.path.join(vis_dir, 'schedule_log.csv'), index=False)

    print(f"--- Saved detailed visualization data to: {vis_dir} ---")


def main():
    """主实验工作流，作为实验的总控中心。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiment_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- All results will be saved in: {output_dir} ---")

    base_problem = load_base_problem()
    all_performance_results = []

    # ######################################################################
    # ####################### EXPERIMENT CONTROL PANEL #####################
    # ######################################################################

    # 【配置点 1】选择要运行的求解器和消融模式
    hoc_ablation_modes = [
        'full',
        #'no_nmpc',
        #'no_role_optimization',
        #'no_strategic_layer',
        #'no_receding_horizon'
    ]
    other_solvers = [
        GASolver,
        PSOSolver,
        AoiEatoSolver,
        MinTimeTcoSolver,
        MacoSolver
    ]

    # 【配置点 2】设置全局的截止时间缩放因子 (alpha)
    # 1.0 = 使用原始截止时间 (默认, 无影响)
    # 0.8 = 所有截止时间收紧20% (更紧急)
    # 1.2 = 所有截止时间放松20% (更宽松)
    DEADLINE_SCALE_FACTOR = 1.2

    # 定义实验规模
    N_values = [50] # 任务数
    M_values = [8]  # 无人机数目
    num_runs_stochastic = 5

    # ######################################################################
    # ######################## END OF CONTROL PANEL ########################
    # ######################################################################

    # --- 自动构建求解器运行列表 ---
    solvers_to_run_config = []
    for mode in hoc_ablation_modes:
        solvers_to_run_config.append({'class': HOC_UADCSolver, 'kwargs': {'ablation_mode': mode}})
    for solver_class in other_solvers:
        solvers_to_run_config.append({'class': solver_class, 'kwargs': {}})

    # --- 实验主循环 ---
    for N in N_values:
        for M in M_values:
            problem_instance = create_problem_instance(base_problem, num_tasks=N, num_uavs=M)
            if problem_instance is None:
                continue

            # 如果设置了缩放因子，则动态修改所有任务的截止时间
            if DEADLINE_SCALE_FACTOR != 1.0:
                print(f"--- Applying Deadline Scale Factor: {DEADLINE_SCALE_FACTOR} to all tasks ---")
                original_deadlines = {task['task_id']: task['deadline_s'] for task in base_problem.tasks}
                for task in problem_instance.tasks:
                    if 'deadline_s' in task and task['task_id'] in original_deadlines:
                        original_deadline = original_deadlines[task['task_id']]
                        task['deadline_s'] = original_deadline * DEADLINE_SCALE_FACTOR

            for config in solvers_to_run_config:
                SolverClass = config['class']
                solver_kwargs = config['kwargs']

                temp_solver = SolverClass(problem_instance, **solver_kwargs)
                num_runs = num_runs_stochastic if getattr(temp_solver, 'is_stochastic', False) else 1

                for i in range(1, num_runs + 1):
                    result_row, history, planned_solution = run_single_experiment(
                        SolverClass,
                        problem_instance,
                        solver_kwargs=solver_kwargs,
                        run_id=i
                    )

                    # 将缩放因子记录到结果中，方便分析
                    result_row['deadline_scale_factor'] = DEADLINE_SCALE_FACTOR
                    all_performance_results.append(result_row)

                    if SolverClass == HOC_UADCSolver and N == 50 and M == 5 and i == 1:
                        save_visualization_data(
                            planned_solution.algorithm_name,
                            problem_instance,
                            history,
                            planned_solution,
                            output_dir
                        )

    # --- 保存并打印最终结果 ---
    performance_df = pd.DataFrame(all_performance_results)
    results_filepath = os.path.join(output_dir, 'performance_results.csv')
    performance_df.to_csv(results_filepath, index=False)

    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Data saved to: {results_filepath}")
    print("=" * 70)
    print(performance_df.to_string())

    # 电池容量敏感性分析
    # print("\n" + "=" * 70)
    # print("RUNNING EXPERIMENT C.1: SENSITIVITY TO 'energy_max_kj'")
    # print("=" * 70)
    #
    # # 定义一个围绕M300/M350 RTK真实容量的、合理的测试范围
    # energy_max_kj_values = [1800, 2000, 2255.4, 2500, 2800]
    # battery_sensitivity_results = []
    #
    # problem_instance_base = create_problem_instance(base_problem, num_tasks=50, num_uavs=3)
    #
    # for energy_val_kj in energy_max_kj_values:
    #     print(f"\n--- Testing with energy_max_kj: {energy_val_kj} kJ ---")
    #
    #     problem_instance = copy.deepcopy(problem_instance_base)
    #
    #     problem_instance.uav_config['default_params']['energy_max_kj'] = energy_val_kj
    #
    #     # --- 运行实验 ---
    #     config = solvers_to_run_config[0]
    #     result_row, _, _ = run_single_experiment(
    #         config['class'],
    #         problem_instance,
    #         solver_kwargs=config['kwargs']
    #     )
    #
    #     result_row['energy_max_kj'] = energy_val_kj
    #     battery_sensitivity_results.append(result_row)
    #
    # # 保存结果...
    # sensitivity_df = pd.DataFrame(battery_sensitivity_results)
    # results_filepath = os.path.join(output_dir, 'sensitivity_energy_max_kj.csv')
    # sensitivity_df.to_csv(results_filepath, index=False)
    # print(f"\nSensitivity results for 'energy_max_kj' saved to: {results_filepath}")
    # print(sensitivity_df)


    #任务截止时间敏感性分析
    # print("\n" + "=" * 70)
    # print("RUNNING EXPERIMENT: SENSITIVITY TO TASK DEADLINE")
    # print("=" * 70)
    #
    # # 定义要测试的截止时间宽松度因子 alpha
    # # alpha < 1.0: 收紧截止时间 (更紧急)
    # # alpha = 1.0: 原始截止时间
    # # alpha > 1.0: 放松截止时间 (更宽松)
    # alpha_values = [0.3,0.5, 2, 3]
    # deadline_sensitivity_results = []
    #
    # # 从最原始加载的问题中，提取出原始的截止时间作为基准
    # original_deadlines = {task['task_id']: task['deadline_s'] for task in base_problem.tasks}
    #
    # problem_instance_base = create_problem_instance(base_problem, num_tasks=50, num_uavs=3)
    #
    # for alpha in alpha_values:
    #     print(f"\n--- Testing with Deadline Factor alpha: {alpha} ---")
    #
    #     problem_instance = copy.deepcopy(problem_instance_base)
    #
    #     # 【核心】动态修改当前实验实例中所有任务的截止时间
    #     for task in problem_instance.tasks:
    #         task_id = task['task_id']
    #         original_deadline = original_deadlines[task_id]
    #         # 根据alpha因子缩放截止时间
    #         task['deadline_s'] = original_deadline * alpha
    #
    #     # --- 运行实验 ---
    #     # (这里的运行逻辑与电池实验完全相同)
    #     config = solvers_to_run_config[0]  # 假设只运行 HOC-UADC-full
    #     result_row, _, _ = run_single_experiment(
    #         config['class'],
    #         problem_instance,
    #         solver_kwargs=config['kwargs']
    #     )
    #
    #     result_row['alpha'] = alpha
    #     deadline_sensitivity_results.append(result_row)
    #
    # # 保存该实验的结果到独立的CSV文件
    # sensitivity_df = pd.DataFrame(deadline_sensitivity_results)
    # results_filepath = os.path.join(output_dir, 'sensitivity_deadline.csv')
    # sensitivity_df.to_csv(results_filepath, index=False)
    # print(f"\nDeadline sensitivity results saved to: {results_filepath}")
    # print(sensitivity_df)

    #战略规划周期敏感性分析
    # print("\n" + "=" * 70)
    # print("RUNNING EXPERIMENT C.3.1: SENSITIVITY TO 'strategic_interval_s' (TH)")
    # print("=" * 70)
    #
    # # 定义要测试的战略规划周期 (单位: 秒)
    # # TH 越小，重规划越频繁
    # strategic_interval_s_values = [11.7,12,12.2,12.3,12.5]
    # TH_sensitivity_results = []
    #
    # # 固定一个基准问题实例
    # problem_instance_base = create_problem_instance(base_problem, num_tasks=50, num_uavs=5)
    # # 确保 TS 在此实验中是固定的，例如 0.2 秒
    # problem_instance_base.scenario['simulation_time_step_s'] = 0.9
    #
    # for interval_s in strategic_interval_s_values:
    #     print(f"\n--- Testing with strategic_interval_s (TH): {interval_s}s ---")
    #     problem_instance = copy.deepcopy(problem_instance_base)
    #
    #     # 【核心】动态修改场景参数
    #     problem_instance.scenario['strategic_interval_s'] = interval_s
    #
    #     # --- 运行实验 ---
    #     config = solvers_to_run_config[0]  # 假设只运行 HOC-UADC-full
    #     result_row, _, _ = run_single_experiment(
    #         config['class'],
    #         problem_instance,
    #         solver_kwargs=config['kwargs']
    #     )
    #
    #     result_row['strategic_interval_s'] = interval_s
    #     TH_sensitivity_results.append(result_row)
    #
    # # 保存该实验的结果到独立的CSV文件
    # sensitivity_df = pd.DataFrame(TH_sensitivity_results)
    # results_filepath = os.path.join(output_dir, 'sensitivity_strategic_interval.csv')
    # sensitivity_df.to_csv(results_filepath, index=False)
    # print(f"\nSensitivity results for 'strategic_interval_s' saved to: {results_filepath}")
    # print(sensitivity_df)

    # #控制/模拟周期敏感性分析
    # print("\n" + "=" * 70)
    # print("RUNNING EXPERIMENT C.3.2: SENSITIVITY TO 'simulation_time_step_s' (TS)")
    # print("=" * 70)
    #
    # # 定义要测试的控制/模拟周期 (单位: 秒)
    # # TS 越小，控制越精细，但模拟步数越多
    # simulation_time_step_s_values = [1.0, 0.5, 0.2, 0.1]
    # TS_sensitivity_results = []
    #
    # # 固定一个基准问题实例
    # problem_instance_base = create_problem_instance(base_problem, num_tasks=50, num_uavs=3)
    # # 确保 TH 在此实验中是固定的，例如 20 秒
    # problem_instance_base.scenario['strategic_interval_s'] = 20
    #
    # for time_step_s in simulation_time_step_s_values:
    #     print(f"\n--- Testing with simulation_time_step_s (TS): {time_step_s}s ---")
    #     problem_instance = copy.deepcopy(problem_instance_base)
    #
    #     # 【核心】动态修改场景参数
    #     problem_instance.scenario['simulation_time_step_s'] = time_step_s
    #
    #     # --- 运行实验 ---
    #     config = solvers_to_run_config[0]
    #     result_row, _, _ = run_single_experiment(
    #         config['class'],
    #         problem_instance,
    #         solver_kwargs=config['kwargs']
    #     )
    #
    #     result_row['simulation_time_step_s'] = time_step_s
    #     TS_sensitivity_results.append(result_row)
    #
    # # 保存该实验的结果到独立的CSV文件
    # sensitivity_df = pd.DataFrame(TS_sensitivity_results)
    # results_filepath = os.path.join(output_dir, 'sensitivity_simulation_timestep.csv')
    # sensitivity_df.to_csv(results_filepath, index=False)
    # print(f"\nSensitivity results for 'simulation_time_step_s' saved to: {results_filepath}")
    # print(sensitivity_df)

    # print("\n" + "=" * 70)
    # print("RUNNING EXPERIMENT C.3: SENSITIVITY TO 'strategic_interval_s' AND 'simulation_time_step_s'")
    # print("=" * 70)

    # TH&TS联合分析  定义要测试的参数组合
    # strategic_interval_s_values = [8, 9, 10,11,12,13,15,18,20,23]  # 战略规划周期 (TH)
    # simulation_time_step_s_values = [0.5,0.7,0.8,0.9,1,1.1,1.2,1.3,1.5,2]  # 控制/模拟周期 (TS)
    # combined_sensitivity_results = []

    # # 创建基准问题实例
    # problem_instance_base = create_problem_instance(base_problem, num_tasks=50, num_uavs=8)

    # # 双重循环测试所有参数组合
    # for interval_s in strategic_interval_s_values:
    #     for time_step_s in simulation_time_step_s_values:
    #         print(
    #             f"\n--- Testing with strategic_interval_s (TH): {interval_s}s, simulation_time_step_s (TS): {time_step_s}s ---")
    #         problem_instance = copy.deepcopy(problem_instance_base)

    #         # 同时修改两个参数
    #         problem_instance.scenario['strategic_interval_s'] = interval_s
    #         problem_instance.scenario['simulation_time_step_s'] = time_step_s

    #         # 运行实验
    #         config = solvers_to_run_config[0]  # 假设只运行 HOC-UADC-full
    #         result_row, _, _ = run_single_experiment(
    #             config['class'],
    #             problem_instance,
    #             solver_kwargs=config['kwargs']
    #         )

    #         # 记录参数和结果
    #         result_row['strategic_interval_s'] = interval_s
    #         result_row['simulation_time_step_s'] = time_step_s
    #         combined_sensitivity_results.append(result_row)

    # # 保存结果到CSV文件
    # sensitivity_df = pd.DataFrame(combined_sensitivity_results)
    # results_filepath = os.path.join(output_dir, 'sensitivity_combined_parameters.csv')
    # sensitivity_df.to_csv(results_filepath, index=False)
    # print(f"\nCombined sensitivity results saved to: {results_filepath}")
    # print(sensitivity_df)

if __name__ == "__main__":
    main()