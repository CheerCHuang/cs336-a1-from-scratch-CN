"""
Problem (multihead_self_attention)：实现 causal 多头自注意力 (5 points)

交付物：把 causal 多头自注意力实现为 torch.nn.Module。

接口：
- def __init__(self, d_model: int, num_heads: int, device=None, dtype=None)
- def forward(self, x: torch.Tensor) -> torch.Tensor

- 输入 x: (..., sequence_length, d_model)
- 返回: 同形状 (..., sequence_length, d_model)
- d_k = d_v = d_model / num_heads

四个投影矩阵（形状都是 d_model × d_model，适配器按这些 key 灌权重）：
- q_proj / k_proj / v_proj / o_proj

流程：
1. Q = x W_qᵀ, K = x W_kᵀ, V = x W_vᵀ        （einsum）
2. 拆头：(..., seq, d_model) → (..., head, seq, d_k)
        rearrange(..., "... seq (h d) -> ... h seq d", h=num_heads)
3. 造因果掩码：下三角（含对角线）为 True 的 (seq, seq) 布尔矩阵
4. 调用 scaled_dot_product_attention(Q, K, V, mask)   ← 复用 _12 的成果
5. 合并头：(..., head, seq, d_k) → (..., seq, d_model)
        rearrange(..., "... h seq d -> ... seq (h d)")
6. 输出投影：结果 @ W_oᵀ

因果掩码约定：
- mask[i][j] = True  表示位置 i 能看到位置 j
- 所以是**下三角**为 True（j <= i），上三角（未来）为 False
- 生成：torch.tril(torch.ones(seq, seq)).bool()
- forward 里按当前 seq_len 动态生成即可（本题没有 max_seq_len 参数）

初始化：线性层权重规则 𝒩(0, 𝜎²=2/(𝑑_in+𝑑_out))（3.3.1）

测试：实现 [adapters.run_multihead_self_attention]，运行 uv run pytest -k test_multihead_self_attention
"""
import torch
import torch.nn as nn
from einops import einsum,rearrange
from cs336_basics._10_rope import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None,max_seq_len=None, theta=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta, d_model//num_heads, max_seq_len, device=device)

        self.q_proj = nn.Parameter(torch.empty(d_model,d_model,device=device,dtype=dtype))
        self.k_proj = nn.Parameter(torch.empty(d_model,d_model,device=device,dtype=dtype))
        self.v_proj = nn.Parameter(torch.empty(d_model,d_model,device=device,dtype=dtype))
        self.o_proj = nn.Parameter(torch.empty(d_model,d_model,device=device,dtype=dtype))
        sigma = (2/(d_model+d_model))**0.5
        nn.init.trunc_normal_(self.q_proj ,mean=0,std= sigma,a=-3*sigma,b=3*sigma)
        nn.init.trunc_normal_(self.k_proj ,mean=0,std= sigma,a=-3*sigma,b=3*sigma)
        nn.init.trunc_normal_(self.v_proj ,mean=0,std= sigma,a=-3*sigma,b=3*sigma)
        nn.init.trunc_normal_(self.o_proj ,mean=0,std= sigma,a=-3*sigma,b=3*sigma)
    def forward(self, x: torch.Tensor,token_positions=None) -> torch.Tensor:
        q = einsum(x,self.q_proj,"... sequence_length d_in,d_out d_in -> ... sequence_length d_out")
        k = einsum(x,self.k_proj,"... sequence_length d_in,d_out d_in -> ... sequence_length d_out")
        v = einsum(x,self.v_proj,"... sequence_length d_in,d_out d_in -> ... sequence_length d_out")
        q = rearrange(q,"... seq (n_head d_k) -> ... n_head seq d_k",n_head=self.num_heads)
        k = rearrange(k,"... seq (n_head d_k) -> ... n_head seq d_k",n_head=self.num_heads)
        v = rearrange(v,"... seq (n_head d_k) -> ... n_head seq d_k",n_head=self.num_heads)

        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        d_k = q.shape[-1]
        seq = q.shape[-2]
        attn_scores = einsum(q,k,"... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
        attn_scores = attn_scores / d_k**0.5
        mask = torch.tril(torch.ones(seq,seq,device = x.device,dtype=bool))
        attn_scores = attn_scores.masked_fill(~mask,float('-inf'))
        attn = torch.softmax(attn_scores,dim=-1)
        values = einsum(attn,v,"... seq_q seq_k, ... seq_k d_k -> ... seq_q d_k")
        values = rearrange(values,"... n_head seq_q d_k-> ... seq_q (n_head d_k)")
        output = einsum(values,self.o_proj,"... seq_q d_in,d_out d_in -> ... seq_q d_out")
        return output


