"""Evaluate the learned Piper policy on fixed-home, held-out targets."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

if __package__:
    from .plant import Plant, make_goals
    from .policy import load_policy
    from .reach import DEFAULT_WEIGHTS, run_episode
else:
    from plant import Plant, make_goals
    from policy import load_policy
    from reach import DEFAULT_WEIGHTS, run_episode


def run_benchmark(plant, goals, policy):
    """Apply one deterministic policy to every supplied target."""
    goals = np.asarray(goals, dtype=np.float64)
    if goals.ndim != 2 or goals.shape[1] != 3 or not len(goals) or not np.isfinite(goals).all():
        raise ValueError("目标必须是非空的 N×3 有限坐标数组，单位为米")

    episodes = [run_episode(plant, target, policy) for target in goals]
    successes = [row for row in episodes if row["success"]]
    final_errors = [row["final_error_m"] for row in episodes]
    failures = Counter(row["failure_reason"] for row in episodes if not row["success"])
    return {
        "protocol": {
            "start": "home",
            "position_tolerance_m": float(plant.tolerance),
            "hold_steps": int(plant.hold_steps),
            "max_steps": int(plant.max_steps),
            "control_dt_s": float(plant.dt),
        },
        "summary": {
            "episodes": len(episodes),
            "successes": len(successes),
            "success_rate": len(successes) / len(episodes),
            "entered_tolerance": sum(row["first_reach_step"] is not None for row in episodes),
            "collision_episodes": sum(row["collision"] for row in episodes),
            "mean_best_error_m": float(np.mean([row["best_error_m"] for row in episodes])),
            "mean_final_error_m": float(np.mean(final_errors)),
            "median_final_error_m": float(np.median(final_errors)),
            "p90_final_error_m": float(np.quantile(final_errors, 0.90)),
            "max_final_error_m": float(np.max(final_errors)),
            "mean_success_time_s": float(np.mean([row["sim_time_s"] for row in successes])) if successes else None,
            "total_wall_s": float(sum(row["wall_time_s"] for row in episodes)),
            "failures": dict(failures),
        },
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="评估 Piper 纯学习策略")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="策略权重文件")
    parser.add_argument("--episodes", type=int, default=100, help="未指定目标文件时的采样局数")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--goals", type=Path, help="三维目标数组，支持 JSON 或 NPY，单位为米")
    parser.add_argument("--report", type=Path, help="写入逐局统计 JSON")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("--episodes 必须大于 0，--seed 不能为负数")
    if args.report is not None and args.report.exists():
        parser.error(f"报告文件已存在 {args.report}")

    policy = load_policy(args.weights)
    plant = Plant()
    try:
        if args.goals is None:
            goals = make_goals(plant, args.episodes, args.seed)
            origin = {"sample_seed": args.seed}
        elif args.goals.suffix.lower() == ".npy":
            goals = np.load(args.goals, allow_pickle=False)
            origin = {"goal_file": str(args.goals)}
        else:
            goals = json.loads(args.goals.read_text(encoding="utf-8"))
            origin = {"goal_file": str(args.goals)}
        report = run_benchmark(plant, goals, policy)
    finally:
        plant.close()
    report["protocol"].update(origin, weights=str(args.weights))

    summary = report["summary"]
    print(f"持续到位 {summary['successes']}/{summary['episodes']} "
          f"({summary['success_rate']:.1%}) | 曾进入容差 {summary['entered_tolerance']} | "
          f"碰撞 {summary['collision_episodes']}")
    print(f"平均末距 {summary['mean_final_error_m'] * 1000:.2f} mm | "
          f"P90 末距 {summary['p90_final_error_m'] * 1000:.2f} mm | "
          f"平均最近距离 {summary['mean_best_error_m'] * 1000:.2f} mm")
    if summary["failures"]:
        print("失败类型 " + json.dumps(summary["failures"], ensure_ascii=False))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()


# 本文件主要代码由 AI 助手完成。
