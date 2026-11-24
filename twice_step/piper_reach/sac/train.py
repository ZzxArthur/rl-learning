import argparse
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twice_step.piper_reach.sac.networks import Actor, Critics


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.states = torch.empty((capacity, state_dim), device=device)
        self.actions = torch.empty((capacity, action_dim), device=device)
        self.rewards = torch.empty(capacity, device=device)
        self.next_states = torch.empty_like(self.states)
        self.terminated = torch.empty(capacity, device=device, dtype=torch.bool)

    def add(self, states, actions, rewards, next_states, terminated):
        count = len(states)
        indices = (torch.arange(count, device=states.device) + self.position) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.terminated[indices] = terminated
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size):
        indices = torch.randint(self.size, (batch_size,), device=self.states.device)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.terminated[indices]


if __name__ == "__main__":
    seed = 42
    num_envs = 1024
    total_steps = 10_000_000
    replay_capacity = 100_000
    warmup_steps = 10_240
    batch_size = 1024
    updates_per_tick = 2
    hidden_dims = [512, 512, 256, 128]
    gamma = 0.99
    tau = 0.005
    learning_rate = 3e-4

    parser = argparse.ArgumentParser(description="Train SAC on GPU Piper Reach")
    parser.add_argument("--num-envs", type=int, default=num_envs)
    parser.add_argument("--steps", type=int, default=total_steps)
    parser.add_argument("--warmup-steps", type=int, default=warmup_steps)
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help="采样预热结束后，固定 Actor 和 alpha，仅更新 Critic 的次数")
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--initial-alpha", type=float, default=0.2, help="初始熵系数")
    parser.add_argument("--init-actor", type=Path, help="用已有权重初始化 Actor，并用它采集预热数据")
    parser.add_argument("--precision-weight", type=float, default=0.0, help="近目标距离惩罚的权重")
    parser.add_argument("--fixed-start", action=argparse.BooleanOptionalAction, default=True,
                        help="每局从 home 姿态出发，目标仍随机")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    if not 1 <= args.num_envs <= replay_capacity or args.steps < 1:
        parser.error("环境数须在经验池容量内，步数须为正整数")
    if not 1 <= args.batch_size <= replay_capacity or not 0 <= args.warmup_steps < args.steps:
        parser.error("检查采样批量和预热步数")
    if args.critic_warmup_updates < 0:
        parser.error("Critic 预热更新次数须为非负整数")
    if not math.isfinite(args.initial_alpha) or args.initial_alpha <= 0:
        parser.error("初始熵系数须为有限正数")
    if not math.isfinite(args.precision_weight) or args.precision_weight < 0:
        parser.error("近目标距离惩罚权重须为有限非负数")
    if not torch.cuda.is_available():
        raise RuntimeError("SAC 训练需要 CUDA")

    from twice_step.piper_reach.ppo.gpu_env import PiperReachGPU

    torch.manual_seed(seed)
    env = PiperReachGPU(num_envs=args.num_envs, fixed_start=args.fixed_start,
                        precision_weight=args.precision_weight)
    device = env.device
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim, hidden_dims).to(device)
    if args.init_actor is not None:
        checkpoint = torch.load(args.init_actor, map_location="cpu", weights_only=True)
        actor.load_state_dict(checkpoint["actor_state_dict"])
    critics = Critics(state_dim, action_dim, hidden_dims).to(device)
    target_critics = deepcopy(critics).requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, fused=True)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=learning_rate, fused=True)
    log_alpha = torch.tensor(math.log(args.initial_alpha), device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=learning_rate, fused=True)
    target_entropy = -action_dim
    replay = ReplayBuffer(replay_capacity, state_dim, action_dim, device)
    episodes, successes, steps, updates = 0, 0, 0, 0
    critic_only_updates = 0
    simulated_steps = 0
    latest_distance = None
    stop_reason = "completed"
    started = time.monotonic()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "sac_piper.pt"
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    print(f"CUDA {torch.cuda.get_device_name(device)} | {args.num_envs} 环境 | {args.steps:,} 步 | 经验池 {replay_capacity:,} 条", flush=True)
    print(f"Actor / Q 隐藏层 {hidden_dims} | ELU", flush=True)
    print(f"起点 {'固定 home' if args.fixed_start else '随机姿态'} | 目标随机", flush=True)
    print(f"近目标权重 {env.precision_weight:g} | 初始 alpha {args.initial_alpha:g}", flush=True)
    if args.init_actor is not None:
        print(f"Actor 初始权重 {args.init_actor}", flush=True)
    try:
        states = env.reset(seed=seed)
        for tick in range(1, math.ceil(args.steps / args.num_envs) + 1):
            with torch.no_grad():
                if steps < args.warmup_steps and args.init_actor is None:
                    actions = torch.rand((args.num_envs, action_dim), device=device) * 2 - 1
                else:
                    actions, _ = actor.sample(states)
                next_states, rewards, terminated, truncated, info = env.step(actions)
                simulated_steps += args.num_envs
                if info["overflow"].any().item():
                    stop_reason = "physics_overflow"
                    print("物理容量溢出，停止采样。", flush=True)
                    break
                # 时间上限截断仍要自举，因此这里保存重置前的末状态与真正终止标志。
                # GPU 每次推进所有环境，最后一轮只收录预算内的转移。
                count = min(args.num_envs, args.steps - steps)
                replay.add(states[:count], actions[:count], rewards[:count], next_states[:count], terminated[:count])
                done = terminated[:count] | truncated[:count]
                if done.any().item():
                    episodes += int(done.sum().item())
                    successes += int(terminated[:count].sum().item())
                    latest_distance = float(info["distance"][:count][done].mean().item())
                steps += count
                if steps < args.steps:
                    states = env.reset_done(done)

            if steps >= args.warmup_steps and replay.size >= args.batch_size:
                warmup_updates = args.critic_warmup_updates if updates == 0 else 0
                if warmup_updates:
                    print(f"采样预热结束 | 环境步数 {steps:,} | 成功 {successes}/{episodes} | "
                          f"用经验池 {replay.size:,} 条转移预热 Critic {warmup_updates:,} 次", flush=True)
                for batch_index in range(warmup_updates + updates_per_tick):
                    s, a, r, next_s, terminal = replay.sample(args.batch_size)
                    alpha = log_alpha.detach().exp()
                    with torch.no_grad():
                        next_a, next_log_prob = actor.sample(next_s)
                        target_q1, target_q2 = target_critics(next_s, next_a)
                        target_q = r + gamma * (~terminal) * (torch.minimum(target_q1, target_q2) - alpha * next_log_prob)

                    q1, q2 = critics(s, a)
                    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_optimizer.step()
                    with torch.no_grad():
                        for target, param in zip(target_critics.parameters(), critics.parameters()):
                            target.lerp_(param, tau)

                    if batch_index < warmup_updates:
                        critic_only_updates += 1
                        continue
                    if warmup_updates and batch_index == warmup_updates:
                        print(f"Critic 预热完成 | 单独更新 {critic_only_updates:,} 次 | 开始更新 Actor 和 alpha", flush=True)

                    critics.requires_grad_(False)
                    policy_actions, log_prob = actor.sample(s)
                    q1, q2 = critics(s, policy_actions)
                    actor_loss = (alpha * log_prob - torch.minimum(q1, q2)).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_optimizer.step()
                    critics.requires_grad_(True)

                    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    updates += 1

            if tick == 1 or tick % 50 == 0 or steps == args.steps:
                loss_text = "预热" if updates == 0 else f"Q损失 {critic_loss.item():.4f} | alpha {log_alpha.detach().exp().item():.4f}"
                distance_text = "暂无完整回合" if latest_distance is None else f"最近结束的一组平均末距 {latest_distance:.4f} 米"
                print(f"步数 {steps:,}/{args.steps:,} | 更新 {updates:,} | 经验池 {replay.size:,} | "
                      f"成功 {successes}/{episodes} | {distance_text} | {loss_text} | {steps / (time.monotonic() - started):.0f} 步/秒", flush=True)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        env.close()

    torch.save({
        "algorithm": "SAC", "actor_state_dict": actor.state_dict(),
        "network_config": {"state_dimension": state_dim, "action_dimension": action_dim, "hidden_dimension": hidden_dims},
        "environment_config": {
            **{key: getattr(env, key) for key in (
                "frame_skip", "max_steps", "success_distance", "joint_step", "min_goal_distance",
                "linear_speed_cost", "angular_speed_cost", "precision_weight", "precision_scale",
            )},
            "fixed_start": args.fixed_start,
        },
        "training_config": {"seed": seed, "num_envs": args.num_envs, "replay_capacity": replay_capacity,
                            "planned_env_steps": args.steps,
                            "warmup_steps": args.warmup_steps, "batch_size": args.batch_size,
                            "critic_warmup_updates": args.critic_warmup_updates,
                            "updates_per_tick": updates_per_tick, "gamma": gamma, "tau": tau,
                            "learning_rate": learning_rate, "initial_alpha": args.initial_alpha,
                            "init_actor": str(args.init_actor) if args.init_actor is not None else None},
        "total_env_steps": steps, "simulated_env_steps": simulated_steps, "total_updates": updates,
        "critic_only_updates": critic_only_updates,
        "stop_reason": stop_reason,
        "episodes": episodes, "successes": successes,
    }, checkpoint_path)
    print(f"模型已保存 {checkpoint_path} | {stop_reason}", flush=True)
    if stop_reason == "physics_overflow":
        sys.exit(1)


# 本文件主要代码由 AI 助手完成。
