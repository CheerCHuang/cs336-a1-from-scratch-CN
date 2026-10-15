"""
Problem (linear)：实现 linear 模块 (1 point)

交付物：实现一个继承自 torch.nn.Module 的 Linear 类，执行线性变换。
实现应遵循 PyTorch 内置 nn.Linear 模块的接口，但没有 bias 参数。

推荐接口：
- def __init__(self, in_features, out_features, device=None, dtype=None)
- def forward(self, x: torch.Tensor) -> torch.Tensor

注意：
- 继承 nn.Module；调用父类构造函数
- 把参数构造并存储为 W（不是 Wᵀ），放进 nn.Parameter
- 用 torch.nn.init.trunc_normal_ 初始化权重 截断正态分布 𝒩(𝜇=0, 𝜎²=2/(𝑑_in+𝑑_out))，截断区间为 [−3𝜎, 3𝜎]。
- 不要使用 nn.Linear 或 nn.functional.linear

初始化参数（std / a / b）见作业 PDF 讲初始化的那一节。

测试：实现 [adapters.run_linear]，运行 uv run pytest -k test_linear
"""
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self,in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))   # 分配一个未初始化张量
        sigma = (2/(out_features+in_features))**0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x,self.weight," ... d_in ,d_out d_in-> ... d_out")

