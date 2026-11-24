"""Goal-conditioned model-based policy training using MuJoCo rollouts."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .plant import Plant
    from .policy import Dynamics, FeedbackActor, GoalActor, Outcome
else:
    from plant import Plant
    from policy import Dynamics, FeedbackActor, GoalActor, Outcome


class ForwardPosition(nn.Module):
    """Differentiable copy of the simulator's forward position, used in the training reward."""

    def __init__(self, plant):
        super().__init__()
        model = plant.model
        matrices, offsets = [], []
        for axis in range(6):
            joint = model.joint(f"joint{axis + 1}")
            body = model.body(int(joint.bodyid[0]))
            assert np.allclose(joint.pos, 0) and np.allclose(joint.axis, [0, 0, 1])
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, body.quat)
            matrices.append(matrix.reshape(3, 3))
            offsets.append(body.pos.copy())
        for key, value in (("matrix", matrices), ("offset", offsets), ("scale", plant.scale),
                           ("center", plant.offset), ("tip", model.site_pos[plant.site_id])):
            self.register_buffer(key, torch.as_tensor(np.asarray(value), dtype=torch.float32))

    def forward(self, q):
        radians = self.center + self.scale * q
        c, s = radians.cos(), radians.sin()
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        z_rotation = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), dim=-1)
        z_rotation = z_rotation.reshape(*q.shape, 3, 3)
        rotation = self.matrix[0] @ z_rotation[..., 0, :, :]
        position = self.offset[0]
        for axis in range(1, 6):
            position = position + (rotation @ self.offset[axis])
            rotation = rotation @ self.matrix[axis] @ z_rotation[..., axis, :, :]
        return position + (rotation @ self.tip)


def collect(plant, rng, poses, trajectories):
    model, data = plant.model, plant.data
    q = rng.uniform(-1, 1, (poses, 6)).astype(np.float32)
    # Most plans retain the home wrist roll. Sample its neighborhood as well as the full range.
    q[::2, 5] *= 0.15
    xyz = np.empty((poses, 3), np.float32)
    bad = np.empty(poses, np.float32)
    for i, command in enumerate(q):
        data.qpos[:6] = plant.offset + command * plant.scale
        mujoco.mj_fwdPosition(model, data)
        xyz[i] = data.site_xpos[plant.site_id]
        bad[i] = any(c.dist < -0.001 for c in data.contact)

    valid = q[bad == 0]
    state = np.empty((trajectories, 4, 12), np.float32)
    actions = np.empty((trajectories, 3, 6), np.float32)
    for i in range(trajectories):
        start = valid[rng.integers(len(valid))]
        mujoco.mj_resetDataKeyframe(model, data, plant.home_id)
        data.qpos[:6] = plant.offset + start * plant.scale
        data.qvel[:6] = 0.8 * rng.standard_normal(6)
        data.ctrl[:6] = data.qpos[:6]
        mujoco.mj_forward(model, data)
        for tick in range(4):
            state[i, tick] = plant.state()[:12]
            if tick == 3:
                break
            radius = 0.1 if i % 3 else 0.65
            action = np.clip(start + radius * rng.standard_normal(6), -1.0, 1.0)
            actions[i, tick] = action
            plant.advance(action)
    return q, xyz, bad, state, actions


def step(optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--poses", type=int, default=160000)
    parser.add_argument("--trajectories", type=int, default=16000)
    parser.add_argument("--outcome-updates", type=int, default=3800)
    parser.add_argument("--actor-updates", type=int, default=8500)
    parser.add_argument("--feedback-updates", type=int, default=2400)
    args = parser.parse_args()
    if args.poses < 100 or args.trajectories < 100 or min(args.outcome_updates, args.actor_updates, args.feedback_updates) < 1:
        parser.error("采样和更新预算过小")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    plant = Plant()
    try:
        arrays = collect(plant, rng, args.poses, args.trajectories)
        fk = ForwardPosition(plant).to(device)
        home = torch.as_tensor((plant.home[:6] - plant.offset) / plant.scale, dtype=torch.float32, device=device)
    finally:
        plant.close()
    q, xyz, contact, states, actions = [torch.as_tensor(a, device=device) for a in arrays]
    torch.testing.assert_close(fk(q[:1000]), xyz[:1000], atol=2e-6, rtol=0)
    print(f"采集姿态 {len(q):,} | 碰撞占比 {contact.mean().item():.2%} | 动态转移 {3 * len(states):,} | {device}", flush=True)
    model_split = int(0.95 * len(q))
    outcome = Outcome().to(device)
    dynamics = Dynamics().to(device)
    outcome_optim = torch.optim.AdamW(outcome.parameters(), lr=0.001)
    dynamics_optim = torch.optim.AdamW(dynamics.parameters(), lr=0.001)
    for update in range(1, args.outcome_updates + 1):
        idx = torch.randint(model_split, (1024,), device=device)
        pose_loss = ((2 * (outcome.positions(q[idx]) - xyz[idx])).square()).mean()
        risk_loss = F.binary_cross_entropy_with_logits(outcome.contact(q[idx]).squeeze(-1), contact[idx])
        step(outcome_optim, 30 * pose_loss + 0.25 * risk_loss)
        idx = torch.randint(int(len(states) * 0.95), (512,), device=device)
        predicted = states[idx, 0]
        losses = []
        for tick in range(3):
            predicted = dynamics(predicted, actions[idx, tick])
            error = predicted - states[idx, tick + 1]
            losses.append(400 * error[:, :6].square().mean() + error[:, 6:].square().mean())
        step(dynamics_optim, torch.stack(losses).mean())
        if update % 500 == 0 or update == args.outcome_updates:
            print(f"模型更新 {update}/{args.outcome_updates} | xyz {pose_loss.item():.6f} | 接触 {risk_loss.item():.4f}", flush=True)

    with torch.no_grad():
        pose_errors = (xyz[model_split:] - outcome.positions(q[model_split:])).norm(dim=-1)
        prediction_p90_mm = (1000 * torch.quantile(pose_errors, 0.9)).item()
    print(f"独立姿态末端预测 P90 {prediction_p90_mm:.2f} mm", flush=True)
    outcome.requires_grad_(False)
    dynamics.requires_grad_(False)
    valid_goals = xyz[:model_split][contact[:model_split] == 0]
    actor = GoalActor().to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.001)
    anchors = torch.tensor([[0, 0.15, 0.05, 0, 0, 0], [0, -0.25, -0.1, 0, 0.3, 0],
                            [0, 0.1, -0.25, 0, -0.3, 0], [0, -0.4, 0.25, 0, 0, 0],
                            [0, 0.4, -0.4, 0, 0.15, 0], [0, 0.1, 0.35, 0, -0.3, 0]], device=device)
    fractions = torch.linspace(0.15, 1.0, 5, device=device)[:, None]

    for update in range(1, args.actor_updates + 1):
        if update == args.actor_updates // 2:
            optimizer.param_groups[0]["lr"] = 0.0004
        if update == int(args.actor_updates * 0.8):
            optimizer.param_groups[0]["lr"] = 0.00015
        goal = valid_goals[torch.randint(len(valid_goals), (256,), device=device)]
        command = actor(goal)
        distance = ((fk(command) - goal[:, None]).square().sum(dim=-1) + 1e-6).sqrt()
        path = home + fractions * (command.unsqueeze(-2) - home)
        risk = outcome.risk(path).amax(dim=-1)
        preference = (command - anchors).square().mean(dim=-1)
        cost = distance + 0.08 * risk + 0.014 * preference + 0.007 * command[..., 5].square()
        loss = cost.amin(dim=-1).mean() + 0.35 * cost.mean()
        step(optimizer, loss)
        if update % 1000 == 0 or update == args.actor_updates:
            print(f"目标 Actor 更新 {update}/{args.actor_updates} | 候选最小误差 {distance.amin(dim=-1).mean().item() * 1000:.1f} mm", flush=True)

    checkpoint = {"actor": actor.state_dict(), "outcome": outcome.state_dict(), "home": home, "feedback": None}
    torch.save(checkpoint, args.output / "coarse.pt")
    feedback = FeedbackActor().to(device)
    optimizer = torch.optim.AdamW(feedback.parameters(), lr=0.0005)
    for update in range(1, args.feedback_updates + 1):
        indices = torch.randint(len(states), (384,), device=device)
        state = states[indices, 3]
        initial = state[:, :6]
        goal_pose = (initial + 0.070 * torch.randn_like(initial)).clamp(-1, 1)
        goal = fk(goal_pose)
        command = (initial + 0.018 * torch.randn_like(initial)).clamp(-1, 1)
        costs = []
        for _ in range(8):
            previous = command
            command = feedback(state, goal - fk(state[:, :6]), previous)
            state = dynamics(state, command)
            error = (goal - fk(state[:, :6])).square().sum(dim=-1)
            velocity = state[:, 6:].square().mean(dim=-1)
            effort = (command - previous).square().mean(dim=-1)
            costs.append((1000 * error + 0.08 * velocity + 20 * effort).mean())
        loss = torch.stack(costs).mean() + costs[-1]
        step(optimizer, loss)
        if update % 400 == 0 or update == args.feedback_updates:
            print(f"反馈 Actor 更新 {update}/{args.feedback_updates} | 想象代价 {loss.item():.4f}", flush=True)
            checkpoint["feedback"] = feedback.state_dict()
            torch.save(checkpoint, args.output / f"feedback_{update:04d}.pt")
    report = {"seed": args.seed, "poses": args.poses, "dynamic_transitions": 3 * args.trajectories,
              "prediction_p90_mm": prediction_p90_mm, "outcome_updates": args.outcome_updates,
              "goal_actor_updates": args.actor_updates, "feedback_actor_updates": args.feedback_updates}
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()


# 本文件主要代码由 AI 助手完成。
