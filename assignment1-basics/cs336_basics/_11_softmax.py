"""
Problem (softmax)：实现 softmax (1 point)

交付物：编写一个对张量应用 softmax 的**函数**（不是类，不用继承 nn.Module）。

接口：
- def softmax(x: torch.Tensor, dim: int) -> torch.Tensor

对输入张量的第 dim 维做 softmax，输出张量与输入**形状相同**，
但其第 dim 维变成归一化的概率分布（和为 1）。

数值稳定（关键）：
    softmax(x)_i = exp(x_i - max) / Σ_j exp(x_j - max)
其中 max = 该维的最大值。
减去最大值在数学上等价（分子分母同时约掉 exp(max)），
但能把指数的输入压到 ≤ 0，彻底避免 exp(大数) 溢出成 inf。

不要用 torch.softmax / F.softmax（本题要自己实现）。

测试：实现 [adapters.run_softmax]，运行 uv run pytest -k test_softmax_matches_pytorch
"""
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x-x_max)
    return exp_x/exp_x.sum(dim=dim,keepdim=True)
    
