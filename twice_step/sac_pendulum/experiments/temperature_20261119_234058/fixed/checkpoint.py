"""保存 SAC 结果，并加载带配置的检查点或旧版 Actor 权重。"""

from pathlib import Path

import torch

from networks import Actornet  # Actornet(**actor_config) -> actor


ACTOR_PATH = Path(__file__).with_name("sac_actor.pt")


def save_actor(actor, path=ACTOR_PATH):
    path = Path(path)
    torch.save(actor.state_dict(), path)
    return path


def save_checkpoint(actor, path, *, env_id, env_kwargs, network_config,
                    training_config, episode_rewards, total_updates):
    path = Path(path)
    torch.save({
        "model_state_dict": actor.state_dict(),
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "network_config": network_config,
        "training_config": training_config,
        "episode_rewards": episode_rewards,
        "total_env_steps": training_config["total_steps"],
        "total_updates": total_updates,
    }, path)
    return path


def load_actor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        actor = Actornet(**checkpoint["network_config"]["actor"])
        parameters = checkpoint["model_state_dict"]
        env_id = checkpoint["env_id"]
        env_kwargs = checkpoint["env_kwargs"]
    else:
        # 旧 sac_actor.pt 只保存参数，使用当时的默认网络和环境。
        actor = Actornet()
        parameters = checkpoint
        env_id = "Pendulum-v1"
        env_kwargs = {}
    actor.load_state_dict(parameters)
    return actor.eval(), env_id, env_kwargs


# 本文件主要代码由 AI 助手完成。
