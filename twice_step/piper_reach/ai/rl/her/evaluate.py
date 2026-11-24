"""Run a trained SAC+HER policy from home on XYZ goals in CPU MuJoCo."""

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

if __package__:
    from .policy import LearnedPolicy
else:
    from policy import LearnedPolicy
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from twice_step.piper_reach.ppo.env import PiperReachEnv


def evaluate(weights, count, seed, goals=None, render=False):
    policy = LearnedPolicy(weights)
    env = PiperReachEnv(fixed_start=True)
    env.joint_step = policy.joint_step
    report = {"target_count": count, "seed": seed, "checkpoint_steps": policy.training_steps,
              "joint_step_rad": env.joint_step, "successes": 0, "single_hits": 0,
              "collisions": 0, "horizon": env.max_steps, "hold_steps": 10, "episodes": []}
    try:
        for episode in range(count):
            env.reset(seed=seed + episode)
            if goals is not None:
                env.target_position = goals[episode]
                env.data.mocap_pos[env.target_mocap_id] = env.target_position
                mujoco.mj_forward(env.model, env.data)
            if render:
                env.render_mode = "human"
                env.render()
            target = env.target_position
            closest = float(np.linalg.norm(env.data.site("reach_point").xpos - target))
            hold = 0
            status = "timeout"
            for _ in range(env.max_steps):
                started = time.monotonic()
                _, _, _, _, info = env.step(policy.action(env))
                closest = min(closest, float(info["distance"]))
                collision = any(contact.dist < -0.001 for contact in env.data.contact)
                if collision:
                    status = "collision"
                    report["collisions"] += 1
                    break
                hold = hold + 1 if info["distance"] < 0.01 else 0
                if hold >= 10:
                    status = "success"
                    report["successes"] += 1
                    break
                if render:
                    if not env.viewer.is_running():
                        raise KeyboardInterrupt
                    time.sleep(max(0., env.sim_timestep * env.frame_skip - (time.monotonic() - started)))
            report["single_hits"] += closest < 0.01
            report["episodes"].append({"target_m": target.tolist(), "final_distance_m": float(info["distance"]),
                                       "endpoint_m": env.data.site("reach_point").xpos.tolist(),
                                       "closest_distance_m": closest, "steps": env.steps, "status": status})
    finally:
        env.close()
    report["mean_final_distance_m"] = float(np.mean([row["final_distance_m"] for row in report["episodes"]]))
    report["mean_closest_distance_m"] = float(np.mean([row["closest_distance_m"] for row in report["episodes"]]))
    return report


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, nargs="?", default=Path(__file__).with_name("actor.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target", "--goal", dest="goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
    group.add_argument("--goals", type=Path, help="JSON array of XYZ world positions in metres")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    goals = None
    if args.goal is not None:
        goals = np.asarray([args.goal], dtype=np.float64)
    if args.goals is not None:
        goals = np.asarray(json.loads(args.goals.read_text()), dtype=np.float64)
    if goals is not None and (goals.ndim != 2 or not len(goals) or goals.shape[1] != 3 or not np.isfinite(goals).all()):
        parser.error("goals must be a nonempty array of finite XYZ coordinates")
    torch.set_num_threads(1)
    try:
        report = evaluate(args.weights, len(goals) if goals is not None else args.episodes, args.seed, goals, args.render)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, ensure_ascii=False))
    if args.goal is not None:
        print(json.dumps(report["episodes"][0], ensure_ascii=False))
    if args.report is not None:
        with args.report.open("x") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
    if args.goal is not None and not report["successes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


# 本文件主要代码由 AI 助手完成。
