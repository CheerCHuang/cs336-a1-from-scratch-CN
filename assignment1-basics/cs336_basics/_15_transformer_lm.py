"""
Problem (transformer_lm)：实现 Transformer LM (3 points)

按 3.1 节和图 1 组装完整的 Transformer 语言模型。

================================================================
一、来自题面（PDF）
================================================================
除 Transformer block 的全部构造参数外，还需接受：

- vocab_size: int
    词表大小，决定 token embedding 矩阵的维度
- context_length: int
    最大上下文长度，决定 RoPE sin/cos buffer 的维度
- num_layers: int
    Transformer block 的堆叠层数

流程（3.1 节描述）：
    token ID → token embedding → num_layers 个 block → 最终层归一化
             → 一个可学习的线性投影（"output embedding" / "LM head"）→ logits

测试：实现 [adapters.run_transformer_lm]，运行 uv run pytest -k test_transformer_lm
"""

import torch
import torch.nn as nn

from cs336_basics._06_linear import Linear
from cs336_basics._07_embedding import Embedding
from cs336_basics._08_rmsnorm import RMSNorm
from cs336_basics._14_transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,vocab_size,context_length,num_layer,norm_type="pre",pos_encoding="rope",ffn_type="swiglu"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layer = num_layer
        
        
        self.token_embeddings = Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.layers = nn.ModuleList([TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=theta,
            norm_type=norm_type,
            pos_encoding=pos_encoding,
            ffn_type=ffn_type) for _ in range(num_layer)])
        self.ln_final = RMSNorm(d_model) if norm_type != "none" else nn.Identity()# pre/post 都保留，只有 none 才去掉
        self.lm_head = Linear(d_model,vocab_size)

    def forward(self,x):
        x = self.token_embeddings(x)
        for block in self.layers:# ModuleList 可以直接迭代
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        
        return x






"""
================================================================
二、来自 adapters.py（课程提供的代码）
================================================================
完整构造参数：
    vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta

weights 字典的全部 key（官方参考实现的 state_dict）：
    token_embeddings.weight                    (vocab_size, d_model)
    layers.{i}.attn.q_proj.weight              (d_model, d_model)
    layers.{i}.attn.k_proj.weight              (d_model, d_model)
    layers.{i}.attn.v_proj.weight              (d_model, d_model)
    layers.{i}.attn.output_proj.weight         (d_model, d_model)
    layers.{i}.ln1.weight                      (d_model,)
    layers.{i}.ln2.weight                      (d_model,)
    layers.{i}.ffn.w1.weight                   (d_ff, d_model)
    layers.{i}.ffn.w2.weight                   (d_model, d_ff)
    layers.{i}.ffn.w3.weight                   (d_ff, d_model)
    ln_final.weight                            (d_model,)
    lm_head.weight                             (vocab_size, d_model)
（{i} = 0 .. num_layers-1）

→ 属性名须与这些 key 对齐：token_embeddings / layers / ln_final / lm_head

================================================================
三、补充提醒（非课程原文，自己整理）
================================================================
1. context_length 就是传给每个 block 的 max_seq_len（题面只说它用于 RoPE buffer）

2. self.layers 必须用 nn.ModuleList，不能用普通 list
   （普通 list 里的模块不会被注册，parameters() 和 load_state_dict 都找不到）

3. weights 的 key 与我们前面各模块的命名有差异，适配器里需映射：
       能直接用：token_embeddings.weight / ln_final.weight / lm_head.weight
                 layers.{i}.ln1.weight / layers.{i}.ln2.weight
       需改名：  layers.{i}.attn.q_proj.weight      → layers.{i}.attn.q_proj
                layers.{i}.attn.k_proj.weight      → layers.{i}.attn.k_proj
                layers.{i}.attn.v_proj.weight      → layers.{i}.attn.v_proj
                layers.{i}.attn.output_proj.weight → layers.{i}.attn.o_proj
                layers.{i}.ffn.w1.weight           → layers.{i}.ffn.weight1
                layers.{i}.ffn.w2.weight           → layers.{i}.ffn.weight2
                layers.{i}.ffn.w3.weight           → layers.{i}.ffn.weight3
   映射时用循环拼 f"layers.{i}.attn.q_proj" 这类字符串最省事

4. load_state_dict 要 strict=False（跳过每层 RoPE 的 cos/sin buffer）
"""
