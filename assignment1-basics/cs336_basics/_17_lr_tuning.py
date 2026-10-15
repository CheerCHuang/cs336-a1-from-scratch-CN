"""
Problem (learning_rate_tuning)：调节学习率 (1 point)

学习率是影响训练最大的超参数之一。在上面的 toy 例子中实际看一下：分别用另外三个学习率
1e1、1e2、1e3 运行 SGD 例子，只训练 10 个 iteration。每个学习率下 loss 发生了什么？
是衰减更快、更慢，还是发散（即随训练进行而增大）？

交付物：一到两句话的回答，描述你观察到的行为。

说明：
- 下面的 SGD 实现与最小训练循环来自作业 PDF 4.2.1（课程教学示例）。
- SGD 是带学习率衰减的变体：θ_{t+1} = θ_t − α/√(t+1) · ∇L(θ_t)
- toy 目标：loss = (weights**2).mean()，weights = 5 * randn((10, 10))
- 实验壳子：固定随机种子（保证三个 lr 从同一组初始 weights 出发，公平对比），
  各跑 10 步，打印每一步的 loss 轨迹。

运行：uv run python cs336_basics/_17_lr_tuning.py
"""

import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def run(lr, num_iterations=10, seed=0):
    """用给定学习率跑 toy 例子 num_iterations 步，返回每一步的 loss。"""
    torch.manual_seed(seed)  # 固定初始化：三个 lr 从同一组 weights 出发
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []
    for t in range(num_iterations):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        losses.append(loss.item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
    return losses


if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        print(f"lr = {lr:g}")
        for t, loss in enumerate(run(lr)):
            print(f"  step {t:2d}: {loss:.4e}")
        print()
