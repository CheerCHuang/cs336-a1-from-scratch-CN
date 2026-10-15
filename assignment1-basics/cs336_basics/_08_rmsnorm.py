"""
Problem (rmsnorm)：Root Mean Square Layer Normalization (1 point)

交付物：把 RMSNorm 实现为 torch.nn.Module。

推荐接口：
- def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
- def forward(self, x: torch.Tensor) -> torch.Tensor

处理形状为 (batch_size, sequence_length, d_model) 的输入，返回**同形状**张量。

计算（对最后一维 d_model 做）：
- RMS = sqrt( mean(x²) + eps )
- 输出 = x / RMS * gain

注意：
- gain 是可学习参数，初始化为**全 1 向量** 𝟙（3.3.1）
- 归一化前把输入 upcast 到 torch.float32，算完再 downcast 回原 dtype
- 是逐位置操作：只对最后一维算，前面的 batch / sequence 维原样保留
- 可配合 einops.reduce 表达"对最后一维求均值"

测试：实现 [adapters.run_rmsnorm]，运行 uv run pytest -k test_rmsnorm
"""

 
import torch
import torch.nn as nn
from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm...
        mean_sq = reduce(x ** 2, "... d_model -> ... 1", "mean")
        rms = torch.sqrt(mean_sq + self.eps)
        result = (x/rms) * self.weight 
        # Return the result in the original dtype
        return result.to(in_dtype)