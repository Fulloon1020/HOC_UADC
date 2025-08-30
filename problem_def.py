# file: problem_def.py

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Problem:
    """封装一个仿真问题实例的所有静态信息。"""
    uav_config: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    city_map: List[Dict[str, Any]]
    scenario: Dict[str, Any]

    def __post_init__(self):
        """在对象初始化后自动计算一些常用属性。"""
        self.task_map = {task['task_id']: task for task in self.tasks}
        self.uav_ids = [f"UAV{uav['uav_id']}" for uav in self.uav_config['initial_state']]
        self.uav_initial_state_map = {uid: uav for uid, uav in zip(self.uav_ids, self.uav_config['initial_state'])}

    @property
    def num_tasks(self) -> int:
        """返回任务（物联网设备）的数量。"""
        return len(self.tasks)

    @property
    def num_uavs(self) -> int:
        """返回无人机的数量。"""
        return len(self.uav_ids)


@dataclass
class Solution:
    """封装一个算法求解后输出的完整规划方案。"""
    algorithm_name: str
    decision_variables: Dict[str, Any] = field(default_factory=dict)
    planned_objective: float = -1.0
    computation_time_s: float = 0.0

    # 一个属性，用于告知主循环本算法是否是随机的
    is_stochastic: bool = False

    def display(self):
        """清晰地打印解的关键决策变量，方便人工校验。"""
        print("\n" + "=" * 70)
        print(f"ALGORITHM '{self.algorithm_name}' - PLANNED SOLUTION SUMMARY")
        print("=" * 70)
        print(f"  - Computation Time: {self.computation_time_s:.4f} seconds")
        print(f"  - Planned Objective (e.g., estimated T_max): {self.planned_objective:.2f}")

        # 从 decision_variables 中获取调度信息，并增加健壮性检查
        schedule = self.decision_variables.get("schedule", {})
        if schedule:
            print("\n  DECISION VARIABLE: Schedule")
            print("  ---------------------------")
            for uav_id in sorted(schedule.keys()):
                tasks = schedule[uav_id]
                if tasks:
                    # 假设 tasks 是一个包含任务字典的列表
                    task_str = " -> ".join([str(t.get('task_id', 'WAYPOINT')) for t in tasks])
                    print(f"     - {uav_id}: {task_str}")
                else:
                    print(f"     - {uav_id}: No tasks assigned.")
        else:
            print("\n  - No schedule/assignment data found in solution.")

        trajectories = self.decision_variables.get("trajectories", {})
        if trajectories:
            print("\n  DECISION VARIABLE: Trajectory")
            print("  -----------------------------")
            print("    - Trajectory waypoints have been generated.")
        else:
            print("\n  - No trajectory data found in solution.")

        print("=" * 70 + "\n")