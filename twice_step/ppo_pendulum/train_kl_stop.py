"""从零训练 Pendulum PPO，验证每个 rollout 内的 KL 提前停止。"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twice_step.ppo_cartpole.advantages import compute_gae  # compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda) -> td_errors, advantages, value_targets
from twice_step.ppo_cartpole.losses import computer_ppo_loss  # computer_ppo_loss(new_log_probs, old_log_probs, advantages, new_values, value_targets, entropy, clip_epsilon, value_coef, entropy_coef) -> loss
from twice_step.ppo_pendulum.networks import Ppo_net  # Ppo_net(state_dimension, action_dimension, hidden_dimension) -> model
from twice_step.ppo_pendulum.rollout import collect_rollout  # collect_rollout(env, model, state, num_steps) -> transitions, next_state
from twice_step.ppo_pendulum.rollout import random_split  # random_split(states, actions, old_log_probs, advantages, value_targets, indices) -> minibatch tensors
from twice_step.ppo_pendulum.train_diagnostics import evaluate_policy  # evaluate_policy(model, seeds, env_id) -> episode_rewards
from twice_step.ppo_pendulum.train_diagnostics import save_checkpoint  # save_checkpoint(path, model, optimizer, config, batch, updates, rewards, score) -> None


def run(output_dir, num_batches, target_kl):
    if num_batches < 1 or not math.isfinite(target_kl) or target_kl <= 0:
        raise ValueError("num_batches 和 target_kl 必须大于 0，且 target_kl 必须有限")
    output_dir.mkdir(parents=True)
    started = perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = dict(seed=42, num_batches=num_batches, num_steps=1024, update_epochs=10,
                    minibatch_size=64, gamma=0.99, gae_lambda=0.95, lr=0.001,
                    optimizer="Adam", clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.0,
                    max_grad_norm=0.5, normalize_advantage=False, target_kl=target_kl)
    network = dict(state_dimension=3, action_dimension=1, hidden_dimension=[256, 256])
    seed_pool = np.random.default_rng(20261117).choice(1_000_000, 120, replace=False).tolist()
    sources = [__name__, collect_rollout.__module__, Ppo_net.__module__, compute_gae.__module__,
               computer_ppo_loss.__module__, evaluate_policy.__module__]
    root = Path(__file__).resolve().parents[2]
    paths = [Path(sys.modules[name].__file__).resolve() for name in sources]
    config = dict(env_id="Pendulum-v1", network_config=network, training_config=training,
                  validation_seeds=seed_pool[:20], test_seeds=seed_pool[20:],
                  torch_version=str(torch.__version__), gymnasium_version=gym.__version__,
                  torch_threads=1, device="cpu", validation_every_batches=5,
                  stop_threshold=1.5 * target_kl,
                  source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    model = Ppo_net(**network)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    num_steps = training["num_steps"]
    minibatch_size = training["minibatch_size"]
    updates = stops = 0
    episode_reward = 0.0
    episode_rewards = []
    evaluations = []
    best_score = -math.inf
    best_batch = 0

    def validate(batch):
        nonlocal best_score, best_batch
        scores = evaluate_policy(model, config["validation_seeds"], config["env_id"])
        score = float(np.mean(scores))
        evaluations.append(dict(batch=batch, env_steps=batch * num_steps, updates=updates,
                                mean_reward=score, reward_std=float(np.std(scores)), rewards=scores))
        if score > best_score:
            best_batch, best_score = batch, score
            save_checkpoint(output_dir / "best.pt", model, optimizer, config,
                            batch, updates, episode_rewards, score)
        print(f"批次 {batch}/{num_batches} | 验证 {score:.2f} | 最佳 {best_score:.2f}（第 {best_batch} 批）"
              f" | 已更新 {updates} 次 | 提前停止 {stops} 批 | {perf_counter() - started:.1f}s", flush=True)

    validate(0)
    fields = ["batch", "env_steps", "updates", "batch_updates", "kl_stop", "max_minibatch_approx_kl",
              "exact_batch_end_kl", "max_pre_clip_grad_norm", "std", "episodes", "mean_last_10_rewards"]
    with gym.make(config["env_id"]) as env, (output_dir / "batches.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        state, _ = env.reset(seed=training["seed"])
        for batch in range(1, num_batches + 1):
            transitions, state = collect_rollout(env, model, state, num_steps)
            for _, _, reward, _, terminated, truncated, _, _ in transitions:
                episode_reward += float(reward)
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0

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
                old_means = model.actor_net(states)
                old_std = model.log_std.exp()
                next_values = model.critic_net(next_states).squeeze(-1)
            _, advantages, targets = compute_gae(rewards, values, next_values, terminated, truncated,
                                                 training["gamma"], training["gae_lambda"])
            stopped = False
            batch_updates = 0
            max_approx_kl = max_grad_norm = 0.0
            for _ in range(training["update_epochs"]):
                indices = torch.randperm(num_steps, device=states.device)
                for start in range(0, num_steps, minibatch_size):
                    mb = indices[start:start + minibatch_size]
                    mb_states, mb_actions, old_lp, mb_advantages, mb_targets = random_split(
                        states, actions, old_log_probs, advantages, targets, mb)
                    _, new_lp, new_values, entropy = model(mb_states, mb_actions)

                    # old_lp 始终来自采集本批数据的策略。超过阈值时先跳过本次更新。
                    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
                    with torch.no_grad():
                        log_ratio = new_lp - old_lp
                        approx_kl = (log_ratio.exp() - 1 - log_ratio).mean().item()
                    if math.isnan(approx_kl):
                        raise FloatingPointError(f"第 {batch} 批出现无效 KL")
                    max_approx_kl = max(max_approx_kl, approx_kl)
                    if approx_kl > config["stop_threshold"]:
                        stopped = True
                        break

                    loss = computer_ppo_loss(new_lp, old_lp, mb_advantages, new_values, mb_targets,
                                             entropy, training["clip_epsilon"], training["value_coef"],
                                             training["entropy_coef"])
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"],
                                                               error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    batch_updates += 1
                    max_grad_norm = max(max_grad_norm, grad_norm.item())
                if stopped:
                    break  # 同时退出当前 rollout 的剩余 epoch；外层采样循环继续。

            stops += int(stopped)
            with torch.no_grad():
                new_std = model.log_std.exp()
                mean_delta = old_means - model.actor_net(states)
                exact_kl = (torch.log(new_std / old_std) +
                            (old_std.square() + mean_delta.square()) / (2 * new_std.square()) - 0.5)
            writer.writerow(dict(batch=batch, env_steps=batch * num_steps, updates=updates,
                                 batch_updates=batch_updates, kl_stop=stopped,
                                 max_minibatch_approx_kl=max_approx_kl,
                                 exact_batch_end_kl=exact_kl.sum(-1).mean().item(),
                                 max_pre_clip_grad_norm=max_grad_norm, std=new_std.item(),
                                 episodes=len(episode_rewards), mean_last_10_rewards=np.mean(episode_rewards[-10:])))
            file.flush()
            if batch % config["validation_every_batches"] == 0 or batch == num_batches:
                validate(batch)

    final_score = evaluations[-1]["mean_reward"]
    save_checkpoint(output_dir / "final.pt", model, optimizer, config, num_batches,
                    updates, episode_rewards, final_score)
    tests = {}
    for name in ["final", "best"]:
        checkpoint = torch.load(output_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        scores = evaluate_policy(model, config["test_seeds"], config["env_id"])
        tests[name] = dict(batch=checkpoint["batch"], mean=float(np.mean(scores)),
                           std=float(np.std(scores)), rewards=scores)
        print(f"{name}.pt | 100 局测试均值 {tests[name]['mean']:.2f} ± {tests[name]['std']:.2f}")
    summary = dict(total_env_steps=num_batches * num_steps, episodes=len(episode_rewards),
                   updates=updates, batches_stopped_early=stops, best_batch=best_batch,
                   best_validation_mean=best_score, final_validation_mean=final_score,
                   elapsed_seconds=perf_counter() - started, validation=evaluations, tests=tests)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"结果目录 {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("kl_stop_run"))
    parser.add_argument("--num-batches", type=int, default=500)
    parser.add_argument("--target-kl", type=float, default=0.02)
    args = parser.parse_args()
    run(args.output_dir, args.num_batches, args.target_kl)

# 本文件主要代码由 AI 助手完成。
