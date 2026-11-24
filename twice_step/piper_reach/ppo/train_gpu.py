# 本文件主要代码由 AI 助手完成。
# GPU 批量 Piper 环境与 PPO；每个环境的 GAE 独立计算。
from pathlib import Path
from datetime import datetime
import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twice_step.ppo_pendulum.networks import Ppo_net
from twice_step.ppo_cartpole.advantages import compute_gae
from twice_step.ppo_cartpole.losses import computer_ppo_loss


if __name__ == "__main__":
    # 每批 1024×16=16384 条转移，4 个 epoch、每小批 2048 条，共 32 次更新。
    seed = 42
    num_envs = 1024
    rollout_steps = 16
    num_batches = 200
    update_epochs = 4
    minibatch_size = 2048
    hidden_dimension = [512, 256, 128]
    activation = "elu"
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.0  # 现有网络返回的是原始高斯熵。
    max_grad_norm = 0.5
    render_interval = 0.5  # 1024 台的窗口更新有开销，可按需调整。

    parser = argparse.ArgumentParser(description="GPU-batched PPO training for Piper Reach")
    parser.add_argument("--num-envs", type=int, default=num_envs)
    parser.add_argument("--rollout-steps", type=int, default=rollout_steps)
    parser.add_argument("--batches", type=int, default=num_batches)
    parser.add_argument("--epochs", type=int, default=update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=minibatch_size)
    parser.add_argument("--no-render", action="store_true", help="关闭 MuJoCo 窗口")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if min(args.num_envs, args.rollout_steps, args.batches, args.epochs, args.minibatch_size) < 1:
        parser.error("环境数、步数、批数、训练轮数和小批量大小必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA GPU；不会自动退回 CPU 训练。")

    from twice_step.piper_reach.ppo.gpu_env import PiperReachGPU

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f"CUDA: {torch.cuda.get_device_name(device)} | 可用显存 {free_bytes / 2**30:.2f}/"
          f"{total_bytes / 2**30:.2f} GiB | 环境数 {args.num_envs}", flush=True)
    print(f"每批 {args.num_envs * args.rollout_steps} 条转移；"
          f"每批最多 {args.epochs * math.ceil(args.num_envs * args.rollout_steps / args.minibatch_size)} 次更新。",
          flush=True)

    started_training = time.perf_counter()
    env = PiperReachGPU(num_envs=args.num_envs, device="cuda")
    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    model = Ppo_net(state_dimension, action_dimension, hidden_dimension, activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    viewer = None
    total_env_steps = 0
    total_updates = 0
    episode_returns = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    total_render_seconds = 0.0
    window_closed = False
    overflow_detected = False
    stop_reason = "completed"

    try:
        states_now = env.reset(seed=seed)
        if not args.no_render:
            # 可视化器只镜像状态，不推进或修改 GPU 仿真。
            render_start = time.perf_counter()
            from twice_step.piper_reach.ppo.gpu_grid_viewer import GPUGridViewer
            viewer = GPUGridViewer(args.num_envs)
            viewer.sync(env.qpos.detach().cpu().numpy(), env.targets.detach().cpu().numpy())
            total_render_seconds += time.perf_counter() - render_start
            last_render = time.perf_counter()

        for batch_index in range(1, args.batches + 1):
            states_list, raw_actions_list, rewards_list = [], [], []
            terminated_list, truncated_list = [], []
            old_log_probs_list, values_list, next_values_list = [], [], []
            completed_returns_list, completed_distances_list, completed_successes_list = [], [], []

            for _ in range(args.rollout_steps):
                if viewer is not None and not viewer.is_running():
                    window_closed = True
                    break
                with torch.no_grad():
                    raw_actions, raw_log_probs, values, _ = model(states_now)
                    actions = raw_actions.tanh()
                    log_jacobian = 2 * (math.log(2) - raw_actions - F.softplus(-2 * raw_actions))
                    old_log_probs = raw_log_probs - log_jacobian.sum(-1)
                    next_states, rewards, terminated, truncated, info = env.step(actions)
                    # 发现接触或约束容量溢出时不能继续使用该批样本。
                    if "overflow" not in info:
                        raise RuntimeError("GPU 环境未提供 overflow 标志，无法检查接触或约束容量。")
                    if torch.any(info["overflow"] != 0).item():
                        overflow_detected = True
                        stop_reason = "physics_overflow"
                        break
                    # 截断需要终局状态自举，因此必须在重置完成的环境之前计算。
                    next_values = model.critic_net(next_states).squeeze(-1)
                    done = terminated | truncated

                    states_list.append(states_now.clone())
                    raw_actions_list.append(raw_actions.clone())
                    rewards_list.append(rewards.clone())
                    terminated_list.append(terminated.clone())
                    truncated_list.append(truncated.clone())
                    old_log_probs_list.append(old_log_probs.clone())
                    values_list.append(values.clone())
                    next_values_list.append(next_values.clone())
                    episode_returns += rewards
                    completed_returns_list.append(episode_returns.clone())
                    completed_distances_list.append(info["distance"].clone())
                    completed_successes_list.append(info["is_success"].clone())
                    episode_returns = torch.where(done, torch.zeros_like(episode_returns), episode_returns)
                    states_now = env.reset_done(done)
                    total_env_steps += args.num_envs

                if viewer is not None and time.perf_counter() - last_render >= render_interval:
                    render_start = time.perf_counter()
                    viewer.sync(env.qpos.detach().cpu().numpy(), env.targets.detach().cpu().numpy())
                    total_render_seconds += time.perf_counter() - render_start
                    last_render = time.perf_counter()

            if overflow_detected:
                print("检测到物理接触或约束容量溢出，停止训练并保存上一次完整更新的模型。", flush=True)
                break
            if terminated_list:
                done = torch.stack(terminated_list) | torch.stack(truncated_list)
                # 一次性将结束回合的统计传到 CPU，避免逐环境打印和同步。
                episode_rewards.extend(torch.stack(completed_returns_list)[done].cpu().tolist())
                episode_distances.extend(torch.stack(completed_distances_list)[done].cpu().tolist())
                episode_successes.extend(torch.stack(completed_successes_list)[done].cpu().tolist())
            if window_closed:
                stop_reason = "window_closed"
                print("窗口已关闭，将保存上一次完整更新的模型。", flush=True)
                break

            # 保留 [时间步, 环境] 结构，在每个环境内部独立计算 GAE。
            states = torch.stack(states_list)
            raw_actions = torch.stack(raw_actions_list)
            rewards = torch.stack(rewards_list)
            terminated = torch.stack(terminated_list)
            truncated = torch.stack(truncated_list)
            old_log_probs = torch.stack(old_log_probs_list)
            values = torch.stack(values_list)
            next_values = torch.stack(next_values_list)
            _, advantages, value_targets = compute_gae(
                rewards, values, next_values, terminated, truncated, gamma, gae_lambda
            )

            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            states = states.reshape(-1, state_dimension)
            raw_actions = raw_actions.reshape(-1, action_dimension)
            old_log_probs = old_log_probs.reshape(-1)
            advantages = advantages.reshape(-1)
            value_targets = value_targets.reshape(-1)

            for _ in range(args.epochs):
                indices = torch.randperm(len(states), device=device)
                for start in range(0, len(states), args.minibatch_size):
                    mb_indices = indices[start:start + args.minibatch_size]
                    mb_raw = raw_actions[mb_indices]
                    _, new_raw_log_probs, new_values, entropy = model(states[mb_indices], mb_raw)
                    log_jacobian = 2 * (math.log(2) - mb_raw - F.softplus(-2 * mb_raw))
                    new_log_probs = new_raw_log_probs - log_jacobian.sum(-1)
                    loss = computer_ppo_loss(
                        new_log_probs, old_log_probs[mb_indices], advantages[mb_indices],
                        new_values, value_targets[mb_indices], entropy,
                        clip_epsilon, value_coef, entropy_coef,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(-5, 2)
                    total_updates += 1

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started_training
            if episode_successes:
                recent_successes = episode_successes[-100:]
                recent_distances = episode_distances[-100:]
                result = (f"最近{len(recent_successes)}局成功率 {np.mean(recent_successes):.1%} | "
                          f"平均最终距离 {np.mean(recent_distances):.4f} 米")
            else:
                result = "暂无完整回合"
            print(f"批次 {batch_index}/{args.batches} | 环境步数 {total_env_steps} | "
                  f"更新次数 {total_updates} | 完成局数 {len(episode_rewards)} | {result} | "
                  f"端到端总速度（含初始化） {total_env_steps / elapsed:.0f} 步/秒 | "
                  f"窗口同步及复制耗时（含等待） {total_render_seconds:.2f} 秒", flush=True)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\n训练已中断，将保存当前模型。", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"ppo_piper_gpu_{datetime.now():%Y%m%d_%H%M%S_%f}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "network_config": {"state_dimension": state_dimension, "action_dimension": action_dimension,
                           "hidden_dimension": hidden_dimension, "activation": activation},
        "algorithm": "PPO with tanh-squashed Gaussian actions",
        "total_env_steps": total_env_steps,
        "total_updates": total_updates,
        "stop_reason": stop_reason,
        "episode_rewards": episode_rewards,
        "episode_distances": episode_distances,
        "episode_successes": episode_successes,
        "environment_config": {key: getattr(env, key) for key in (
            "frame_skip", "max_steps", "success_distance", "joint_step", "min_goal_distance",
            "linear_speed_cost", "angular_speed_cost",
        )},
        "physics_capacity": {"nconmax": env.nconmax, "njmax": env.njmax},
        "training_config": {"seed": seed, "num_envs": args.num_envs, "num_batches": args.batches,
                            "rollout_steps_per_env": args.rollout_steps, "update_epochs": args.epochs,
                            "minibatch_size": args.minibatch_size, "gamma": gamma, "gae_lambda": gae_lambda,
                            "learning_rate": learning_rate, "clip_epsilon": clip_epsilon,
                            "value_coef": value_coef, "entropy_coef": entropy_coef,
                            "max_grad_norm": max_grad_norm},
    }, checkpoint_path)
    print(f"模型已保存：{checkpoint_path}", flush=True)
