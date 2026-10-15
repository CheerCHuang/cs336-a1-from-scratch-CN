"""
Problem (rope)：实现 RoPE (2 points)

交付物：实现 RotaryPositionalEmbedding 类，对输入张量应用 RoPE。

推荐接口（注意参数顺序）：
- def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
- def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor

处理形状为 (..., seq_len, d_k) 的输入，返回**同形状**张量，要支持任意数量的 batch 维。
token_positions 形状 (..., seq_len)，指定 x 沿序列维的 token 位置。

原理（旋转，不是加法）：
- 频率：freq_i = 1 / theta^(2i/d_k)，i = 0, 1, ..., d_k/2 - 1
- 角度：angle = position * freq
- 把 d_k 个维度**两两配对** (x[2i], x[2i+1])，每组旋转 angle_i：
      x1' = x1 * cos - x2 * sin
      x2' = x1 * sin + x2 * cos

注意：
- **没有可学习参数**——不需要 nn.Parameter（这点和前几个模块不同）
- cos / sin 可以预计算到 max_seq_len 缓存起来，用 register_buffer 存
  （buffer 不参与梯度，但会跟着 .to(device) 走）
- forward 里用 token_positions 沿序列维切片（索引）预计算好的 cos / sin

测试：实现 [adapters.run_rope]，运行 uv run pytest -k test_rope
"""

import torch
import torch.nn as nn
from einops import einsum, rearrange


class RoPE(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    super().__init__()
    self.theta = theta
    self.d_k = d_k
    self.max_seq_len = max_seq_len

    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    positions = torch.arange(0,max_seq_len,device=device).float()
    angles = einsum(positions, freqs, "seq, half -> seq half")
    # self.register_buffer("cos_cached", torch.cos(angles))
    # self.register_buffer("sin_cached", torch.sin(angles))
    self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
    self.register_buffer("sin_cached", torch.sin(angles), persistent=False)


  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    cos = self.cos_cached[token_positions]
    sin = self.sin_cached[token_positions]

    x_pair = rearrange(x,"... (half two) -> ... half two", two = 2)
    x1 = x_pair[...,0]
    x2 = x_pair[...,1]
    x1r = x1 * cos - x2 * sin
    x2r = x1 * sin + x2 * cos
    out = torch.stack([x1r, x2r], dim=-1)

    return rearrange(out,"... half two -> ... (half two)", two = 2)