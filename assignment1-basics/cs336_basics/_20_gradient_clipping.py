"""
Problem (gradient_clipping)：实现梯度裁剪 (1 point)

编写一个实现梯度裁剪的函数。函数应接受参数列表和最大 ℓ2-范数。
它应就地（in place）修改每个参数的梯度。使用 ε = 10⁻⁶（PyTorch 默认值）。

接口（见 tests/adapters.py::run_gradient_clipping）：
- parameters  : Iterable[torch.nn.Parameter]，可训练参数的集合
- max_l2_norm : float，允许的最大 ℓ2 范数
- 返回 None（直接改 parameter.grad）

注意：范数是对**所有参数梯度拼在一起**算的整体范数，不是每个参数各自算。

测试：
1. 先实现 [adapters.run_gradient_clipping]（把调用转发到这里的函数）
2. uv run pytest -k test_gradient_clipping
"""

import torch
from einops import reduce

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += (p.grad**2).sum()
    total_norm = total ** 0.5
    if total_norm>max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            ratio = max_l2_norm / (total_norm + 1e-6)
            p.grad *= ratio
    
    return None
