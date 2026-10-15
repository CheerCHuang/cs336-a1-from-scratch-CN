"""
Problem (cross_entropy)：实现 cross-entropy (1 point)

交付物：编写一个计算 cross-entropy loss 的函数，输入预测 logits（o_i）和目标（x_{i+1}），
计算 cross-entropy ℓ_i = −log softmax(o_i)[x_{i+1}]。

带入有 ℓ = −(o_t − m) + log( Σ_j exp(o_j − m) )


函数应处理以下几点：
- 为数值稳定性减去最大元素。
- 尽可能消去 log 和 exp。
- 处理任意额外的 batch 维度，并返回跨 batch 的平均值。与 3.2 节一样，假设 batch 维度
  总是排在最前面（在词表大小维度之前）。

接口说明（见 tests/adapters.py::run_cross_entropy）：
- inputs  : Float[Tensor, "batch_size vocab_size"]  模型输出的未归一化 logits
- targets : Int[Tensor, "batch_size"]               每个位置正确的下一个 token ID
- 返回    : Float[Tensor, ""]                       一个标量（所有位置的平均 loss）

关于 batch 维度的说明：测试里可能出现 (batch, seq, vocab) 这类形状，即"任意额外的
batch 维度"。实现时不要写死两维。

测试：先实现 [adapters.run_cross_entropy]（把调用转发到这里的函数），
然后运行 uv run pytest -k test_cross_entropy
"""

import torch
from einops import reduce

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 实现：数值稳定 + 消去 log/exp + 对前面所有 batch 维取平均
    max_voc = reduce(inputs,"... vocab -> ... 1","max")
    inputs_stable = inputs - max_voc
    targets_voc = targets.unsqueeze(-1) #（batch_size，1）
    z_t = torch.gather(inputs_stable,-1,targets_voc) #对inputs的最后一个维度去targets_voc的下标
    z_t = z_t.squeeze(-1)   #（batch_size) 对应真实所需ID的稳定logit

    denominator = torch.exp(inputs_stable)
    denominator = reduce(denominator,"... vocab -> ...","sum")

    total = - z_t + torch.log(denominator)
    loss = reduce(total,"... ->","mean")#mean避免loss随batch增长爆炸
    return loss    

    

    
