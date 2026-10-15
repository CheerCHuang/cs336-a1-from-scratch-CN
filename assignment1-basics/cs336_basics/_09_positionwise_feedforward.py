"""
Problem (positionwise_feedforward)：实现 position-wise 前馈网络 (2 points)

交付物：把 SiLU 激活函数和 GLU 组成 SwiGLU 前馈网络。

公式：
    SwiGLU(x) = W2 ( SiLU(W1 x) ⊙ (W3 x) )
    其中 SiLU(z) = z * sigmoid(z)

参数（形状）：
- w1: (d_ff, d_model)
- w2: (d_model, d_ff)
- w3: (d_ff, d_model)

注意：
- 继承 nn.Module；调用父类构造函数
- 三个权重都放进 nn.Parameter（适配器按 w1 / w2 / w3 三个 key 灌权重）
- d_ff ≈ 8/3 × d_model，且取 64 的倍数（向上取整，充分利用硬件）
- 允许使用 torch.sigmoid 保证数值稳定性
- 初始化：线性层权重规则 𝒩(0, 𝜎²=2/(𝑑_in+𝑑_out))（3.3.1）

测试：实现 [adapters.run_swiglu]，运行 uv run pytest -k test_swiglu
"""
import torch
import torch.nn as nn
from einops import einsum

class PositionWiseFFN(nn.Module):
    def __init__(self,d_model,d_ff,ffn_type="swiglu",device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.ffn_type = ffn_type

        sigma = (2/(d_ff+d_model))**0.5
        self.weight1 = nn.Parameter(torch.ones(d_ff,d_model,device=device,dtype=dtype))
        self.weight2 = nn.Parameter(torch.ones(d_model,d_ff,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight1,mean = 0,std = sigma,a=-3*sigma,b=3*sigma)
        nn.init.trunc_normal_(self.weight2,mean = 0,std = sigma,a=-3*sigma,b=3*sigma)

        if ffn_type == "swiglu":
            self.weight3 = nn.Parameter(torch.ones(d_ff,d_model,device=device,dtype=dtype))
            nn.init.trunc_normal_(self.weight3,mean = 0,std = sigma,a=-3*sigma,b=3*sigma)

        
        
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x,self.weight1,"... d_model,d_ff d_model -> ... d_ff")
        silu_w1x = w1_x * torch.sigmoid(w1_x)

        if self.ffn_type == "swiglu":
            w3_x = einsum(x,self.weight3,"... d_model,d_ff d_model -> ... d_ff")
            gated = silu_w1x * w3_x
        else:
            gated = silu_w1x
            
        return einsum(gated,self.weight2,"... d_ff,d_model d_ff -> ... d_model")
