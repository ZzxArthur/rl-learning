# 检查折扣回报的数值、形状和数据类型。
import torch
from returns import compute_returns  # compute_returns(trajectory, gamma) -> returns


def check_returns(trajectory, gamma, expected):
    actual = compute_returns(trajectory, gamma)
    assert actual.shape == expected.shape, f"shape: {actual.shape} != {expected.shape}"
    assert actual.dtype == torch.float32, f"dtype: {actual.dtype}"
    assert torch.allclose(actual, expected), f"returns: {actual} != {expected}"
    print(f"gamma={gamma}: {actual}，检查通过")


if __name__ == "__main__":
    trajectory = [
        {"reward": 1.0, "terminated": False, "truncated": False},
        {"reward": 2.0, "terminated": False, "truncated": False},
        {"reward": 3.0, "terminated": True, "truncated": False},
    ]
    check_returns(trajectory, 0.9, torch.tensor([5.23, 4.7, 3.0], dtype=torch.float32))
    check_returns(trajectory, 0.0, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
