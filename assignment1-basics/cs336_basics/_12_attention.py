"""
Problem (scaled_dot_product_attention)：实现 scaled dot-product attention (5 points)

交付物：编写计算 scaled dot-product attention 的函数。

接口：
- def scaled_dot_product_attention(Q, K, V, mask=None) -> torch.Tensor

- Q: (..., queries, d_k)
- K: (..., keys, d_k)
- V: (..., keys, d_v)      注意 values 的个数 = keys 的个数
- mask: (..., queries, keys) 可选，布尔张量
- 返回: (..., queries, d_v)

公式：
    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

四步：
1. scores = Q Kᵀ            （einsum，缩并 d_k 那一维）
2. scores = scores / √d_k    （scale，防止点积过大把 softmax 推到饱和区）
3. 若有 mask：把 False 的位置填成 -inf（softmax 后自然变 0）
4. 对最后一维 softmax，再乘 V

mask 约定（注意方向！）：
- mask == True  的位置 → 参与注意力（保留）
- mask == False 的位置 → 注意力概率为 0（填 -inf）
- 因果掩码里，True 通常在**下三角**（含对角线）

测试：实现 [adapters.run_scaled_dot_product_attention]；
uv run pytest -k test_scaled_dot_product_attention   （三阶输入）
uv run pytest -k test_4d_scaled_dot_product_attention （四阶输入）
"""
import torch
from einops import einsum
from cs336_basics._11_softmax import softmax

def scaled_dot_product_attention(Q, K, V, mask=None) -> torch.Tensor:
    d_k = Q.shape[-1]
    attention_score = einsum(Q,K,"... queries d_k,... keys d_k -> ... queries keys")
    attention_score = attention_score / d_k**0.5
    if mask is not None:
        attention_score = attention_score.masked_fill(~mask,float("-inf"))
    attn = softmax(attention_score, dim=-1) 
    return einsum(attn,V,"... queries keys,... keys d_v -> ... queries d_v")

