"""Goal-conditioned SAC with same-episode future hindsight replay on MJWarp."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import warp as wp

if __package__:
    from .policy import Actor, features
else:
    from policy import Actor, features
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from twice_step.piper_reach.ppo.gpu_env import PiperReachGPU


class TrainingEnv(PiperReachGPU):
    def capture_dynamics(self):
        self.reset(seed=0)
        self.step(torch.zeros((self.num_envs, 6), device=self.device))
        self.warp_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(self.warp_stream):
            with wp.ScopedCapture(stream=self.warp_stream) as capture:
                for _ in range(self.frame_skip):
                    mjw.step(self.model, self.data)
                mjw.forward(self.model, self.data)
        self.graph = capture.graph

    def advance(self, action):
        self.ctrl[:, :6] = (self.qpos[:, :6] + self.joint_step * action).clamp(self.limits[:, 0], self.limits[:, 1])
        wp.capture_launch(self.graph, stream=self.warp_stream)
        linear_velocity, _ = self._reach_velocity()
        return torch.cat((self.qpos[:, :6], self.qvel[:, :6],
                          self.site_xpos[:, self.reach_site_id], linear_velocity), dim=-1)


class Replay:
    def __init__(self, capacity, worlds, horizon, device):
        self.capacity = capacity
        self.worlds = worlds
        self.horizon = horizon
        self.device = device
        self.cursor = self.size = 0
        self.states = torch.empty((capacity, horizon + 1, 18), device=device)
        self.actions = torch.empty((capacity, horizon, 6), device=device)
        self.collisions = torch.empty((capacity, horizon), dtype=torch.bool, device=device)
        self.goals = torch.empty((capacity, 3), device=device)
        self.indices = torch.arange(worlds, device=device)

    def add(self, states, actions, collisions, goals):
        indices = (self.cursor + self.indices) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.collisions[indices] = collisions
        self.goals[indices] = goals
        self.cursor = (self.cursor + self.worlds) % self.capacity
        self.size = min(self.size + self.worlds, self.capacity)

    def sample(self, count, hindsight_ratio):
        rows = torch.randint(self.size, (count,), device=self.device)
        step = torch.randint(self.horizon, (count,), device=self.device)
        future = step + 1 + ((self.horizon - step) * torch.rand(count, device=self.device).square()).long()
        goals = torch.where(torch.rand((count, 1), device=self.device) < hindsight_ratio,
                            self.states[rows, future, 12:15], self.goals[rows])
        next_states = self.states[rows, step + 1]
        action = self.actions[rows, step]
        collision = self.collisions[rows, step]
        distance = (goals - next_states[:, 12:15]).norm(dim=-1)
        speed = next_states[:, 15:18].norm(dim=-1)
        reward = (-(4 * distance) + (-distance / 0.045).exp() + 0.55 * (-distance / 0.011).exp()
                  - 0.08 * speed - 3.0 * collision - 0.008 * action.square().mean(dim=-1))
        return features(self.states[rows, step], goals), action, reward, features(next_states, goals)


def goal_pool(env, seed, size=4096):
    rng = np.random.default_rng(seed)
    cpu = env.cpu_env
    cpu.reset(seed=seed)
    home = cpu.data.qpos.copy()
    origin = cpu.data.site("reach_point").xpos.copy()
    lower, upper = cpu.model.actuator_ctrlrange[:6].T
    data = cpu.goal_data
    regular, near = [], []
    while len(regular) < size:
        cpu._sample_pose(data)
        position = data.site_xpos[cpu.reach_site_id].copy()
        if np.linalg.norm(position - origin) >= 0.04:
            regular.append(position)
    while len(near) < size // 2:
        data.qpos[:] = home
        offset = rng.standard_normal(6) * rng.uniform(0.08, 0.55)
        data.qpos[:6] = np.clip(home[:6] + offset, lower, upper)
        mujoco.mj_fwdPosition(cpu.model, data)
        position = data.site_xpos[cpu.reach_site_id].copy()
        if 0.035 <= np.linalg.norm(position - origin) <= 0.35 and all(c.dist >= -1e-5 for c in data.contact):
            near.append(position)
    return tuple(torch.as_tensor(np.asarray(pool), dtype=torch.float32, device=env.device)
                 for pool in (regular, near))


def q_network(width):
    return nn.Sequential(nn.Linear(27, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))


def train(args):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    env = TrainingEnv(num_envs=args.worlds, fixed_start=True)
    env.joint_step = args.joint_step
    env.capture_dynamics()
    device = env.device
    actor = Actor(args.width).to(device)
    critics = nn.ModuleList([q_network(args.width), q_network(args.width)]).to(device)
    target_critics = deepcopy(critics).requires_grad_(False)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr, fused=True)
    opt_critic = torch.optim.Adam(critics.parameters(), lr=args.lr, fused=True)
    log_alpha = torch.tensor(-3.5, device=device, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr, fused=True)
    replay = Replay(max(args.worlds, 4096), args.worlds, args.horizon, device)
    full_pool, near_pool = goal_pool(env, args.seed)
    block = torch.empty((args.worlds, args.horizon + 1, 18), device=device)
    actions = torch.empty((args.worlds, args.horizon, 6), device=device)
    collisions = torch.empty((args.worlds, args.horizon), dtype=torch.bool, device=device)
    contact_dist = wp.to_torch(env.data.contact.dist)
    contact_world = wp.to_torch(env.data.contact.worldid)
    contact_count = wp.to_torch(env.data.nacon)
    contact_ids = torch.arange(len(contact_dist), device=device)
    penetration = torch.zeros(args.worlds, device=device)
    args.run_dir.mkdir(parents=True)
    updates = total_steps = 0
    started = time.monotonic()
    print(f"SAC+HER worlds={args.worlds} joint_step={args.joint_step} horizon={args.horizon}", flush=True)
    try:
        while total_steps < args.steps:
            observations = env.reset(seed=args.seed + total_steps)  # All worlds begin at the home keyframe.
            p_near = 0.20 + 0.65 * max(0., 1. - total_steps / args.curriculum_steps)
            is_near = torch.rand((args.worlds, 1), device=device) < p_near
            goals = torch.where(is_near, near_pool[torch.randint(len(near_pool), (args.worlds,), device=device)],
                                full_pool[torch.randint(len(full_pool), (args.worlds,), device=device)])
            env.targets.copy_(goals)
            env.mocap_pos[:, env.target_mocap_id] = goals
            block[:, 0] = torch.cat((observations[:, :15], observations[:, 18:21]), dim=-1)
            noise = torch.zeros((args.worlds, 6), device=device)
            hold = torch.zeros(args.worlds, dtype=torch.int32, device=device)
            success = torch.zeros_like(hold, dtype=torch.bool)
            nearest = (block[:, 0, 12:15] - goals).norm(dim=-1)
            for tick in range(args.horizon):
                with torch.no_grad():
                    noise.mul_(0.94).add_(torch.randn_like(noise), alpha=0.3412)
                    if replay.size:
                        action, _ = actor.sample(features(block[:, tick], goals))
                        action = (action + 0.12 * noise).clamp(-1, 1)
                    else:
                        action = (0.7 * noise).clamp(-1, 1)
                    block[:, tick + 1] = env.advance(action)
                    actions[:, tick] = action
                    depth = torch.where(contact_ids < contact_count[0], -contact_dist, 0.)
                    penetration.zero_().scatter_reduce_(0, contact_world.long().clamp(0, args.worlds - 1),
                                                        depth, reduce="amax")
                    collisions[:, tick] = penetration > 0.001
                    distance = (block[:, tick + 1, 12:15] - goals).norm(dim=-1)
                    nearest = torch.minimum(nearest, distance)
                    hold = torch.where((distance < 0.01) & ~collisions[:, tick], hold + 1, 0)
                    success |= hold >= 10
                if not replay.size:
                    continue
                for _ in range(args.updates_per_tick):
                    obs, act, rew, nxt = replay.sample(args.batch_size, args.her_ratio)
                    alpha = log_alpha.detach().exp()
                    with torch.no_grad():
                        n_act, n_log_prob = actor.sample(nxt)
                        n_pair = torch.cat((nxt, n_act), dim=-1)
                        target = torch.minimum(*[net(n_pair).squeeze(-1) for net in target_critics])
                        target = rew + args.gamma * (target - alpha * n_log_prob)
                    pair = torch.cat((obs, act), dim=-1)
                    critic_loss = sum(F.mse_loss(net(pair).squeeze(-1), target) for net in critics)
                    opt_critic.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    opt_critic.step()
                    critics.requires_grad_(False)
                    pa, pl = actor.sample(obs)
                    pp = torch.cat((obs, pa), dim=-1)
                    q_val = torch.minimum(*[net(pp).squeeze(-1) for net in critics])
                    actor_loss = (alpha * pl - q_val).mean()
                    opt_actor.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    opt_actor.step()
                    critics.requires_grad_(True)
                    opt_alpha.zero_grad(set_to_none=True)
                    (-(log_alpha * (pl.detach() - 6)).mean()).backward()
                    opt_alpha.step()
                    with torch.no_grad():
                        for target_param, param in zip(target_critics.parameters(), critics.parameters(), strict=True):
                            target_param.lerp_(param, 0.005)
                    updates += 1
            if hasattr(env.data, "overflow") and wp.to_torch(env.data.overflow).any().item():
                raise RuntimeError("MuJoCo Warp contact allocation overflow")
            replay.add(block, actions, collisions, goals)
            total_steps += args.worlds * args.horizon
            full_success = success[~is_near[:, 0]].float().mean().item()
            print(f"steps={total_steps} updates={updates} near_fraction={p_near:.3f} "
                  f"success={success.float().mean().item():.3f} full_success={full_success:.3f} "
                  f"nearest_m={nearest.mean().item():.5f} contact_steps={collisions.float().mean().item():.3f} "
                  f"alpha={log_alpha.exp().item():.4f} steps_per_s={total_steps/(time.monotonic()-started):.0f}", flush=True)
            pending = args.run_dir / "latest.tmp"
            torch.save({"actor": actor.state_dict(), "width": args.width, "joint_step": args.joint_step,
                        "steps": total_steps, "updates": updates, "seed": args.seed}, pending)
            pending.replace(args.run_dir / "latest.pt")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_400_000)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--joint-step", type=float, default=0.12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--updates-per-tick", type=int, default=4)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--her-ratio", type=float, default=0.78)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--curriculum-steps", type=int, default=650_000)
    args = parser.parse_args()
    with torch.cuda.stream(torch.cuda.Stream()):
        train(args)


# 本文件主要代码由 AI 助手完成。
