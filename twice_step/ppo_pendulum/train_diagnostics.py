"""独立重训与诊断；只向 diagnostics_run（或 --output-dir）写入实验结果。

默认保持当前 train.py 的算法设置：无优势标准化，全局梯度裁剪 0.5。
梯度按每次 minibatch 更新记录，局数通过采样批次关联；不逐次保存完整梯度张量。
KL / clip_fraction 的统计定义参考 SB3 PPO，不引入 KL 提前停止或其他算法改动：
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twice_step.ppo_pendulum.networks import Ppo_net  # Ppo_net(state_dimension, action_dimension, hidden_dimension) -> model
from twice_step.ppo_pendulum.rollout import collect_rollout  # collect_rollout(env, model, state, num_steps) -> transitions, state
from twice_step.ppo_pendulum.rollout import random_split  # random_split(states, actions, old_log_probs, advantages, values_target, mb_indices) -> minibatch tensors
from twice_step.ppo_cartpole.advantages import compute_gae  # compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda) -> td_errors, advantages, value_targets
from twice_step.ppo_cartpole.losses import computer_ppo_loss  # computer_ppo_loss(new_log_probs, old_log_probs, advantages, new_values, value_targets, entropy, clip_epsilon, value_coef, entropy_coef) -> loss


@torch.inference_mode()
def evaluate_policy(model, seeds, env_id):
    # 均值动作不抽样；独立环境不会消耗训练环境或 Torch 的随机数。
    rewards = []
    with gym.make(env_id) as env:
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            total = 0.0
            while True:
                mean = model.actor_net(torch.as_tensor(state, dtype=torch.float32))
                action = np.clip(mean.numpy(), env.action_space.low, env.action_space.high)
                state, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            rewards.append(total)
    return rewards


def save_checkpoint(path, model, optimizer, config, batch, updates, rewards, score):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "network_config": config["network_config"],
        "training_config": config["training_config"],
        "env_id": config["env_id"],
        "batch": batch,
        "total_env_steps": batch * config["training_config"]["num_steps"],
        "total_updates": updates,
        "episode_rewards": list(rewards),
        "validation_mean_reward": score,
        "validation_seeds": config["validation_seeds"],
        "torch_rng_state": torch.get_rng_state(),
        "source_hashes": config["source_hashes"],
    }, path)


def plot_diagnostics(output_dir, episodes, batches, evaluations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    rewards = np.array([row["reward"] for row in episodes])
    ends = [min(start + 10, len(rewards)) for start in range(0, len(rewards), 10)]
    means = [rewards[start:start + 10].mean() for start in range(0, len(rewards), 10)]
    axes[0, 0].scatter(np.arange(1, len(rewards) + 1), rewards,
                       s=2, color="0.65", alpha=0.35, label="Every episode")
    axes[0, 0].plot(ends, means, color="#0072B2", label="Mean per 10 episodes")
    axes[0, 0].set(xlabel="Training episode", ylabel="Undiscounted reward",
                   title="Sampled training actions")
    validation = [row for row in evaluations if row["split"] == "validation"]
    axes[0, 1].errorbar([row["env_steps"] for row in validation],
                        [row["mean_reward"] for row in validation],
                        yerr=[row["reward_std"] for row in validation],
                        fmt="o-", markersize=3, linewidth=1, color="#0072B2",
                        label="Fixed validation seeds: mean +/- episode SD")
    axes[0, 1].set(xlabel="Training environment steps", ylabel="Undiscounted reward",
                   title="Deterministic mean-action evaluation")
    steps = [row["env_steps"] for row in batches]
    axes[1, 0].plot(steps, [row["std_after"] for row in batches], color="#0072B2",
                    label="Std at end of each batch")
    axes[1, 0].set(xlabel="Training environment steps", ylabel="Action std",
                   title="Learned Gaussian standard deviation")
    for field, label, color, style in [
        ("grad_total_max", "Total max", "#0072B2", "-"),
        ("grad_actor_max", "Actor max", "#D55E00", "--"),
        ("grad_critic_max", "Critic max", "#009E73", ":"),
    ]:
        axes[1, 1].plot(steps, [row[field] for row in batches],
                        label=label, color=color, linestyle=style, linewidth=1)
    axes[1, 1].axhline(0.5, color="0.35", linestyle="-.", label="Global clip threshold")
    axes[1, 1].set(xlabel="Training environment steps", ylabel="L2 norm (log scale)",
                   yscale="log", title="Pre-clip gradient: maximum within each batch")
    axes[2, 0].plot(steps, [row["kl_old_to_new"] for row in batches],
                    color="#0072B2", label="Exact Gaussian KL(old || batch-end)")
    axes[2, 0].set(xlabel="Training environment steps", ylabel="KL (symlog scale)",
                   title="Policy change on all rollout states")
    axes[2, 0].set_yscale("symlog", linthresh=1e-4)
    axes[2, 1].plot(steps, [row["raw_action_clip_fraction"] for row in batches],
                    color="#0072B2", label="Sampled raw action outside [-2, 2]")
    axes[2, 1].plot(steps, [row["actor_tanh2_saturation"] for row in batches],
                    color="#D55E00", linestyle="--", label="Actor Tanh2: |activation| > 0.99")
    axes[2, 1].set(xlabel="Training environment steps", ylabel="Fraction", ylim=(-0.02, 1.02),
                   title="Batch-end saturation and rollout action clipping")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Pendulum PPO diagnostics: one training seed; all raw values retained in CSV")
    fig.savefig(output_dir / "diagnostics.png", dpi=160)
    plt.close(fig)


def run_experiment(output_dir, num_batches=500, eval_every=5, eval_episodes=20, test_episodes=100):
    if min(num_batches, eval_every, eval_episodes, test_episodes) < 1:
        raise ValueError("训练批数、评估间隔和评估局数必须大于0")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    # 小型 MLP 在 CPU 单线程运行，避免大量线程调度；不改变模型或更新公式。
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = dict(seed=42, num_batches=num_batches, num_steps=1024, update_epochs=10,
                    minibatch_size=64, gamma=0.99, gae_lambda=0.95, lr=0.001,
                    optimizer="Adam", clip_epsilon=0.2, value_coef=0.5,
                    entropy_coef=0.0, max_grad_norm=0.5, normalize_advantage=False)
    network = dict(state_dimension=3, action_dimension=1, hidden_dimension=[256, 256])
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(), root / "twice_step/ppo_pendulum/train.py",
                    root / "twice_step/ppo_pendulum/networks.py",
                    root / "twice_step/ppo_pendulum/rollout.py",
                    root / "twice_step/ppo_cartpole/advantages.py",
                    root / "twice_step/ppo_cartpole/losses.py"]
    source_text = {str(path.relative_to(root)): path.read_text() for path in source_paths}
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    seed_pool = np.random.default_rng(20261117).choice(
        1_000_000, eval_episodes + test_episodes, replace=False).tolist()
    config = dict(env_id="Pendulum-v1", network_config=network, training_config=training,
                  device="cpu", torch_threads=1, torch_version=str(torch.__version__),
                  gymnasium_version=gym.__version__, validation_every_batches=eval_every,
                  validation_seeds=seed_pool[:eval_episodes], test_seeds=seed_pool[eval_episodes:],
                  source_hashes=source_hashes, source_snapshots=source_text,
                  metrics_note="Gradients are from total PPO loss; critic gradients include value_coef. "
                  "Actor gradient norms cover actor_net only; log_std gradients are logged separately. "
                  "Per-parameter gradients are pre-clip L2 norms and absolute maxima, not full tensors. "
                  "Each update is linked to its rollout episodes, not an independent per-episode gradient. "
                  "KL is for raw Gaussians before environment clipping. Validation selects best.pt; "
                  "disjoint test seeds are used only after training. Re-running overwrites these experiment outputs.")
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    model = Ppo_net(**network)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    named_parameters = dict(model.named_parameters())
    actor_names = [name for name in named_parameters if name.startswith("actor_net")]
    critic_names = [name for name in named_parameters if name.startswith("critic_net")]
    episodes, batches, evaluations = [], [], []
    episode_rewards = []
    updates = env_steps = episode_steps = 0
    episode_reward = episode_std_sum = 0.0
    episode_first_batch = 1
    episode_initial_std = float(model.log_std.exp().item())
    best_score, best_batch = -math.inf, 0
    last_validation = None

    with ExitStack() as stack:
        env = stack.enter_context(gym.make(config["env_id"]))
        state, _ = env.reset(seed=training["seed"])
        files = {name: stack.enter_context((output_dir / f"{name}.csv").open("w", newline=""))
                 for name in ["episodes", "updates", "batches", "evaluations"]}
        writers = {}

        def record(name, row):
            if name not in writers:
                writers[name] = csv.DictWriter(files[name], fieldnames=list(row))
                writers[name].writeheader()
            writers[name].writerow(row)

        def validate(batch):
            nonlocal best_score, best_batch, last_validation
            rewards = evaluate_policy(model, config["validation_seeds"], config["env_id"])
            score = float(np.mean(rewards))
            row = dict(split="validation", batch=batch, env_steps=env_steps, updates=updates,
                       mean_reward=score, reward_std=float(np.std(rewards)),
                       min_reward=min(rewards), max_reward=max(rewards),
                       seeds=json.dumps(config["validation_seeds"]), rewards=json.dumps(rewards))
            record("evaluations", row)
            evaluations.append(row)
            last_validation = score
            if score > best_score:
                best_score, best_batch = score, batch
                save_checkpoint(output_dir / "best.pt", model, optimizer, config,
                                batch, updates, episode_rewards, score)
            print(f"评估批次{batch}: {eval_episodes}局均值{score:.2f}, "
                  f"当前最佳{best_score:.2f}（批次{best_batch}）", flush=True)

        validate(0)
        for batch in range(1, num_batches + 1):
            rollout_first_episode = len(episodes) + 1
            rollout_std = float(model.log_std.exp().item())
            transitions, state = collect_rollout(env, model, state, training["num_steps"])
            for _, _, reward, _, terminated, truncated, _, _ in transitions:
                if episode_steps == 0:
                    episode_first_batch = batch
                    episode_initial_std = rollout_std
                env_steps += 1
                episode_steps += 1
                episode_reward += float(reward)
                episode_std_sum += rollout_std
                if terminated or truncated:
                    row = dict(episode=len(episodes) + 1, reward=episode_reward, steps=episode_steps,
                               env_steps=env_steps, first_batch=episode_first_batch, last_batch=batch,
                               updates_before_rollout=updates, std_at_start=episode_initial_std,
                               std_at_end=rollout_std, std_mean=episode_std_sum / episode_steps)
                    record("episodes", row)
                    episodes.append(row)
                    episode_rewards.append(episode_reward)
                    episode_reward = episode_std_sum = 0.0
                    episode_steps = 0
                    episode_first_batch = batch
                    episode_initial_std = rollout_std
            rollout_last_episode = len(episodes) + int(episode_steps > 0)
            states, actions, rewards, next_states, terminated, truncated, old_log_probs, values = zip(*transitions)
            states = torch.as_tensor(np.stack(states), dtype=torch.float32)
            actions = torch.stack(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
            terminated = torch.tensor(terminated, dtype=torch.bool)
            truncated = torch.tensor(truncated, dtype=torch.bool)
            old_log_probs = torch.stack(old_log_probs)
            values = torch.stack(values)
            with torch.no_grad():
                old_means = model.actor_net(states).clone()
                old_std = model.log_std.exp().clone()
                next_values = model.critic_net(next_states).squeeze(-1)
            td_errors, advantages, targets = compute_gae(
                rewards, values, next_values, terminated, truncated,
                training["gamma"], training["gae_lambda"])
            update_rows = []

            for epoch in range(training["update_epochs"]):
                indices = torch.randperm(training["num_steps"], device=states.device)
                for start in range(0, training["num_steps"], training["minibatch_size"]):
                    mb_indices = indices[start:start + training["minibatch_size"]]
                    mb_states, mb_actions, mb_old_lp, mb_adv, mb_targets = random_split(
                        states, actions, old_log_probs, advantages, targets, mb_indices)
                    _, new_lp, new_values, entropy = model(mb_states, mb_actions)
                    loss = computer_ppo_loss(new_lp, mb_old_lp, mb_adv, new_values, mb_targets,
                                            entropy, training["clip_epsilon"],
                                            training["value_coef"], training["entropy_coef"])
                    with torch.no_grad():
                        log_ratio = new_lp - mb_old_lp
                        ratio = log_ratio.exp()
                        actor_loss = -torch.minimum(ratio * mb_adv,
                            ratio.clamp(1 - training["clip_epsilon"], 1 + training["clip_epsilon"]) * mb_adv).mean()
                        critic_loss = (new_values - mb_targets).square().mean()
                        mean_before = model.actor_net(mb_states)
                        std_before = model.log_std.exp().clone()
                        params_before = {name: p.clone() for name, p in named_parameters.items()}
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norms = {name: float(p.grad.norm().item()) for name, p in named_parameters.items()}
                    total_grad = math.sqrt(sum(value ** 2 for value in grad_norms.values()))
                    actor_grad = math.sqrt(sum(grad_norms[name] ** 2 for name in actor_names))
                    critic_grad = math.sqrt(sum(grad_norms[name] ** 2 for name in critic_names))
                    std_grad = float(model.log_std.grad.item())
                    layer_metrics = {}
                    for name, p in named_parameters.items():
                        layer_metrics[f"{name}.grad_norm_pre"] = grad_norms[name]
                        layer_metrics[f"{name}.grad_abs_max_pre"] = float(p.grad.abs().max().item())
                    if not all(math.isfinite(v) for v in [loss.item(), total_grad]):
                        (output_dir / "failure.json").write_text(json.dumps(dict(
                            batch=batch, update=updates + 1, reason="Non-finite loss or gradient")))
                        raise FloatingPointError(f"批次{batch}、更新{updates + 1}出现非有限loss/梯度，已停止")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"],
                                                   error_if_nonfinite=True)
                    post_norms = {name: float(p.grad.norm().item())
                                  for name, p in named_parameters.items()}
                    grad_post = math.sqrt(sum(v ** 2 for v in post_norms.values()))
                    optimizer.step()
                    updates += 1
                    with torch.no_grad():
                        std_after = model.log_std.exp()
                        mean_after = model.actor_net(mb_states)
                        kl_old = (torch.log(std_after / old_std) +
                            (old_std.square() + (old_means[mb_indices] - mean_after).square()) /
                            (2 * std_after.square()) - 0.5).sum(-1).mean()
                        kl_step = (torch.log(std_after / std_before) +
                            (std_before.square() + (mean_before - mean_after).square()) /
                            (2 * std_after.square()) - 0.5).sum(-1).mean()
                        deltas = {name: float((p - params_before[name]).norm().item())
                                  for name, p in named_parameters.items()}
                        for name, p in named_parameters.items():
                            layer_metrics[f"{name}.param_abs_max_post"] = float(p.abs().max().item())
                    row = dict(update=updates, batch=batch, epoch=epoch + 1,
                               minibatch=start // training["minibatch_size"] + 1,
                               env_steps=env_steps, rollout_episode_first=rollout_first_episode,
                               rollout_episode_last=rollout_last_episode, loss=float(loss.item()),
                               actor_loss=float(actor_loss.item()), critic_loss=float(critic_loss.item()),
                               entropy=float(entropy.mean().item()), advantage_mean=float(mb_adv.mean().item()),
                               advantage_std=float(mb_adv.std(unbiased=False).item()),
                               grad_total_pre=total_grad, grad_actor_pre=actor_grad,
                               grad_critic_pre=critic_grad, log_std_grad_pre=std_grad,
                               grad_total_post=grad_post,
                               grad_actor_post=math.sqrt(sum(post_norms[n] ** 2 for n in actor_names)),
                               grad_critic_post=math.sqrt(sum(post_norms[n] ** 2 for n in critic_names)),
                               grad_scale=grad_post / total_grad if total_grad else 1.0,
                               log_std_before=float(params_before["log_std"].item()),
                               log_std_after=float(model.log_std.item()),
                               std_before=float(std_before.item()), std_after=float(std_after.item()),
                               param_delta_total=math.sqrt(sum(v ** 2 for v in deltas.values())),
                               param_delta_actor=math.sqrt(sum(deltas[n] ** 2 for n in actor_names)),
                               param_delta_critic=math.sqrt(sum(deltas[n] ** 2 for n in critic_names)),
                               approx_kl_pre=float((ratio - 1 - log_ratio).mean().item()),
                               exact_kl_old_to_new_post=float(kl_old.item()),
                               exact_kl_single_step=float(kl_step.item()),
                               ratio_clip_fraction=float(((ratio - 1).abs() > training["clip_epsilon"]).float().mean().item()),
                               learning_rate=optimizer.param_groups[0]["lr"], **layer_metrics)
                    record("updates", row)
                    update_rows.append(row)
                    if not all(math.isfinite(v) for v in row.values()) or row["std_after"] <= 0:
                        files["updates"].flush()
                        (output_dir / "failure.json").write_text(json.dumps(dict(
                            batch=batch, update=updates, reason="Non-finite update diagnostics or invalid std")))
                        raise FloatingPointError(f"更新{updates}后出现非有限数值或无效std，已停止")

            with torch.no_grad():
                current_means = model.actor_net(states)
                current_std = model.log_std.exp()
                kl = (torch.log(current_std / old_std) +
                    (old_std.square() + (old_means - current_means).square()) /
                    (2 * current_std.square()) - 0.5).sum(-1).mean()
                batch_row = dict(batch=batch, env_steps=env_steps, updates=updates,
                    episodes_completed=len(episodes), rollout_episode_first=rollout_first_episode,
                    rollout_episode_last=rollout_last_episode,
                    reward_last10=float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0,
                    std_before=rollout_std, std_after=float(current_std.item()),
                    std_min=min(min(r["std_before"], r["std_after"]) for r in update_rows),
                    std_max=max(max(r["std_before"], r["std_after"]) for r in update_rows),
                    raw_action_clip_fraction=float(((actions < -2) | (actions > 2)).float().mean().item()),
                    mean_action_clip_fraction=float(((current_means < -2) | (current_means > 2)).float().mean().item()),
                    kl_old_to_new=float(kl.item()), advantage_mean=float(advantages.mean().item()),
                    advantage_std=float(advantages.std(unbiased=False).item()),
                    td_abs_mean=float(td_errors.abs().mean().item()),
                    target_mean=float(targets.mean().item()), target_std=float(targets.std(unbiased=False).item()),
                    critic_mse_after=float((model.critic_net(states).squeeze(-1) - targets).square().mean().item()),
                    grad_total_max=max(r["grad_total_pre"] for r in update_rows),
                    grad_actor_max=max(r["grad_actor_pre"] for r in update_rows),
                    grad_critic_max=max(r["grad_critic_pre"] for r in update_rows),
                    grad_total_post_max=max(r["grad_total_post"] for r in update_rows),
                    kl_step_max=max(r["exact_kl_single_step"] for r in update_rows),
                    ratio_clip_fraction_mean=float(np.mean([r["ratio_clip_fraction"] for r in update_rows])))
                for branch in ["actor", "critic"]:
                    features = states
                    tanh_index = 0
                    for layer in getattr(model, f"{branch}_net"):
                        features = layer(features)
                        if isinstance(layer, torch.nn.Tanh):
                            tanh_index += 1
                            batch_row[f"{branch}_tanh{tanh_index}_saturation"] = float(
                                (features.abs() > 0.99).float().mean().item())
            record("batches", batch_row)
            batches.append(batch_row)
            if batch % eval_every == 0 or batch == num_batches:
                validate(batch)
            for file in files.values():
                file.flush()
            if batch % 5 == 0 or batch == 1 or batch == num_batches:
                print(f"批次{batch}/{num_batches}, 局数{len(episodes)}, 步数{env_steps}, 更新{updates}, "
                      f"最近10局{batch_row['reward_last10']:.2f}, std={current_std.item():.4f}, "
                      f"Actor/Critic梯度最大={batch_row['grad_actor_max']:.2f}/{batch_row['grad_critic_max']:.2f}, "
                      f"耗时{perf_counter() - started:.1f}s", flush=True)

        save_checkpoint(output_dir / "final.pt", model, optimizer, config,
                        num_batches, updates, episode_rewards, last_validation)
        for label in ["final", "best"]:
            checkpoint = torch.load(output_dir / f"{label}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            rewards = evaluate_policy(model, config["test_seeds"], config["env_id"])
            row = dict(split=f"test_{label}", batch=checkpoint["batch"],
                       env_steps=checkpoint["total_env_steps"], updates=checkpoint["total_updates"],
                       mean_reward=float(np.mean(rewards)), reward_std=float(np.std(rewards)),
                       min_reward=min(rewards), max_reward=max(rewards),
                       seeds=json.dumps(config["test_seeds"]), rewards=json.dumps(rewards))
            record("evaluations", row)
            evaluations.append(row)
            print(f"独立测试 {label}: {test_episodes}局均值{row['mean_reward']:.2f} "
                  f"± {row['reward_std']:.2f}（局间标准差）", flush=True)

    plot_diagnostics(output_dir, episodes, batches, evaluations)
    summary = dict(total_env_steps=env_steps, total_updates=updates, episodes=len(episodes),
                   best_batch=best_batch, best_validation_mean=best_score,
                   final_validation_mean=last_validation,
                   test_results={r["split"]: dict(mean=r["mean_reward"], std=r["reward_std"])
                                 for r in evaluations if r["split"].startswith("test_")},
                   elapsed_seconds=perf_counter() - started,
                   max_actor_grad=max(r["grad_actor_max"] for r in batches),
                   max_critic_grad=max(r["grad_critic_max"] for r in batches),
                   max_total_grad_post=max(r["grad_total_post_max"] for r in batches),
                   std_min=min(r["std_min"] for r in batches),
                   std_max=max(r["std_max"] for r in batches))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"实验完成：{output_dir}\n{json.dumps(summary, ensure_ascii=False, indent=2)}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("diagnostics_run"))
    parser.add_argument("--num-batches", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=100)
    args = parser.parse_args()
    run_experiment(args.output_dir, args.num_batches, args.eval_every,
                   args.eval_episodes, args.test_episodes)

# 本文件主要代码由 AI 助手完成。
