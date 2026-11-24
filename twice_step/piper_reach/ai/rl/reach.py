"""Run the learned Piper policy for one Cartesian target in MuJoCo."""

import argparse
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

if __package__:
    from .plant import Plant
    from .policy import load_policy
else:
    from plant import Plant
    from policy import load_policy

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "runs" / "best.pt"


def run_episode(plant, target, policy, *, real_time=False):
    plant.reset(target)
    initial_error = best_error = float(plant.distance)
    first_reach_step = 0 if best_error < plant.tolerance else None
    previous_action = None
    started = perf_counter()

    while plant.steps < plant.max_steps and not plant.success and not plant.collision:
        step_started = perf_counter()
        action = policy(plant.state(), target, previous_action)
        plant.step(action)
        previous_action = action
        best_error = min(best_error, float(plant.distance))
        if first_reach_step is None and plant.distance < plant.tolerance:
            first_reach_step = int(plant.steps)
        if real_time:
            sleep(max(0.0, plant.dt - (perf_counter() - step_started)))

    success = bool(plant.success and not plant.collision)
    failure_reason = None
    if not success:
        if plant.collision:
            failure_reason = "collision"
        elif first_reach_step is None:
            failure_reason = "timeout"
        else:
            failure_reason = "did_not_hold"

    return {
        "target_m": np.asarray(target).tolist(),
        "endpoint_m": np.asarray(plant.position).tolist(),
        "initial_error_m": initial_error,
        "best_error_m": best_error,
        "final_error_m": float(plant.distance),
        "success": success,
        "collision": bool(plant.collision),
        "steps": int(plant.steps),
        "first_reach_step": first_reach_step,
        "first_reach_time_s": None if first_reach_step is None else float(first_reach_step * plant.dt),
        "sim_time_s": float(plant.steps * plant.dt),
        "wall_time_s": perf_counter() - started,
        "failure_reason": failure_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="运行 Piper 纯学习策略，目标为仿真世界坐标（米）")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="策略权重文件")
    parser.add_argument("--render", action="store_true", help="打开 MuJoCo 窗口，按控制周期回放")
    args = parser.parse_args()
    target = np.asarray([args.x, args.y, args.z], dtype=np.float64)
    if not np.isfinite(target).all():
        parser.error("目标 XYZ 必须为有限数值")

    policy = load_policy(args.weights)
    plant = Plant(render=args.render)
    try:
        result = run_episode(plant, target, policy, real_time=args.render)
    except KeyboardInterrupt:
        return 130
    finally:
        plant.close()
    status = "到位并稳定保持" if result["success"] else f"未完成（{result['failure_reason']}）"
    print(f"{status} | {result['steps']} 步 | 仿真用时 {result['sim_time_s']:.2f} s")
    print("目标 XYZ（m） " + " ".join(f"{value:.4f}" for value in result["target_m"]))
    print("末端 XYZ（m） " + " ".join(f"{value:.4f}" for value in result["endpoint_m"]))
    print(f"最终误差 {result['final_error_m'] * 1000:.2f} mm | 最近误差 {result['best_error_m'] * 1000:.2f} mm")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# 本文件主要代码由 AI 助手完成。
