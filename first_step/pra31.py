# 练习31：从 pra30 存档恢复模型，用 torch.equal 核对权重和偏置。
import torch


checkpoint = torch.load("/home/hatepp/piper/RL/first_step_model/pra30_model.pt", map_location="cpu", weights_only=True)
model = torch.nn.Linear(in_features=4, out_features=2)
model.load_state_dict(checkpoint["model_state_dict"])
saved_parameters = checkpoint["model_state_dict"]
print("weight 是否一致：", torch.equal(model.weight, saved_parameters["weight"]))
print("bias 是否一致：", torch.equal(model.bias, saved_parameters["bias"]))
