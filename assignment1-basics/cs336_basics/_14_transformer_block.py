"""
Problem (transformer_block)：实现 Transformer block (3 points)

交付物：按 3.4 节和图 2 实现**预归一化（pre-norm）** Transformer block。

构造参数（适配器会传这些）：
- d_model, num_heads, d_ff
- max_seq_len, theta   ← block 内的注意力**要带 RoPE**，所以需要这两个

结构（pre-norm：归一化在子层之前）：

    h   = x + attn( ln1(x) )
    out = h + ffn ( ln2(h) )

组件（属性名要和适配器 weights 的 key 对上）：
- self.ln1 / self.ln2 : RMSNorm(d_model)
- self.attn           : MultiHeadAttention(d_model, num_heads, max_seq_len, theta)
- self.ffn            : PositionWiseFFN(d_model, d_ff)

⚠️ weights 的 key 格式问题（重要）：
官方参考实现的 state_dict 用的是「子模块 + .weight」的形式：
    "attn.q_proj.weight" / "attn.output_proj.weight" / "ln1.weight" / "ffn.w1.weight" ...
而我们前面各模块里参数就是 nn.Parameter，state_dict 的 key 是：
    "attn.q_proj" / "attn.o_proj" / "ln1.weight" / "ffn.weight1" ...
两处差异：① 有没有 `.weight` 后缀；② 输出投影叫 `output_proj` 还是 `o_proj`
→ 直接 `load_state_dict(weights)` 会报 key 不匹配，需要在适配器里做一次 key 映射。

测试：实现 [adapters.run_transformer_block]，运行 uv run pytest -k test_transformer_block
"""

import torch
import torch.nn as nn
from cs336_basics._08_rmsnorm import RMSNorm
from cs336_basics._13_multihead_attention import MultiHeadAttention
from cs336_basics._09_positionwise_feedforward import PositionWiseFFN

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta,norm_type="pre",pos_encoding="rope", ffn_type="swiglu"):
        super().__init__()
        self.d_model=d_model
        self.num_heads = num_heads
        self.d_ff =d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.norm_type = norm_type          # forward 里要用，必须存成属性,用于组件消融实验
        self.pos_encoding = pos_encoding
        self.ffn_type = ffn_type

        self.ln1 = RMSNorm(d_model) if norm_type != "none" else nn.Identity()# 消融
        self.ln2 = RMSNorm(d_model) if norm_type != "none" else nn.Identity()

        self.attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,theta=theta)
        self.ffn = PositionWiseFFN(d_model=d_model,d_ff=d_ff, ffn_type=ffn_type)
    
    def forward(self,x):
        token_positions = torch.arange(0, x.shape[-2], device=x.device) if self.pos_encoding == "rope" else None

        if self.norm_type == "post":          # 先过子层 + 加残差，最后才归一化
            x = self.ln1(x + self.attn(x=x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
        else:                                  # "pre"（也兼容 "none"，此时 ln 是 Identity）
            x = x + self.attn(x=self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))

        return x

