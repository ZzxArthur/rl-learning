from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actor import Actor_net  # Actor_net(state_space, action_space) -> actor


if __name__ == "__main__":
    checkpoint_path = Path(__file__).with_name("sac_checkpoint.pt")
    seed = 42
    observation_steps = (10, 25, 50)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor = Actor_net(checkpoint["state_dimension"], checkpoint["action_dimension"])
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()

    with gym.make(checkpoint["env_id"]) as env, torch.no_grad():
        state, _ = env.reset(seed=seed)
        distance = np.linalg.norm(state[-2:])
        print(f"第  0 步距离 {distance:.6f} 米")

        for step in range(1, 51):
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, _ = actor(state_tensor)
            action = torch.tanh(mean).squeeze(0).cpu().numpy()
            state, reward, terminated, truncated, info = env.step(action)

            if step in observation_steps:
                distance = np.linalg.norm(state[-2:])
                print(f"第 {step:2d} 步距离 {distance:.6f} 米")
            if terminated or truncated:
                break

# 本文件主要代码由 AI 助手完成。
