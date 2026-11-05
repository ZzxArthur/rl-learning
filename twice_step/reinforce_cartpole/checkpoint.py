# 保存策略参数，并将参数恢复到新的策略模型。
import torch
from policy import PolicyNetwork


def save_model_weights(policy, save_path):

    torch.save(policy.state_dict(), save_path)


def load_model(model_path):
    parameters = torch.load(model_path, map_location="cpu", weights_only=True)
    policy = PolicyNetwork()
    policy.load_state_dict(parameters)
    return policy
    