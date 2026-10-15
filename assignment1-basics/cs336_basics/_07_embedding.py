"""
Problem (embedding)：实现 embedding 模块 (1 point)

交付物：实现继承自 torch.nn.Module 的 Embedding 类，执行 embedding 查表。
实现应遵循 PyTorch 内置 nn.Embedding 模块的接口。

推荐接口：
- def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
- def forward(self, token_ids: torch.Tensor) -> torch.Tensor

注意：
- 继承 nn.Module；调用父类构造函数
- 把 embedding 矩阵初始化为 nn.Parameter
- 存储时让 d_model 是最后一个维度
- 不要使用 nn.Embedding 或 nn.functional.embedding

初始化（3.3.1）：截断正态分布 𝒩(𝜇=0, 𝜎²=1)，截断区间为 [−3, 3]
（注意：和 Linear 的 𝜎²=2/(d_in+d_out) 不同）

测试：实现 [adapters.run_embedding]，运行 uv run pytest -k test_embedding
"""

import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings,embedding_dim,device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight,mean = 0,std=1,a=-3, b=3)

    def forward(self, token_ids:torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]
