"""
Problem (adamw)：实现 AdamW (2 points)

交付物：把 AdamW optimizer 实现为 torch.optim.Optimizer 的子类。
你的类应在 __init__ 中接受学习率 α，以及 β、ε、λ 超参数。
为了保存状态，基类 Optimizer 提供了字典 self.state，它把 nn.Parameter 对象映射到
存储该参数所需信息的字典（对 AdamW 来说就是矩估计量）。

接口（测试用关键字参数调用，见 tests/test_optimizer.py）：
- AdamW(params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

算法见作业 PDF Algorithm 1（已存为 notes_cache/figure/image.png）。
骨架可参考同目录 _17_lr_tuning.py 里的 SGD 实现（同继承 torch.optim.Optimizer）。

测试：
1. 先实现 [adapters.get_adamw_cls]（直接 return 这里的 AdamW 类）
2. uv run pytest -k test_adamw
3. 之后 uv run pytest -k test_checkpointing 也会用到它
"""

import math
from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults ={"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            #先提取本组超参数
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            #接下来遍历传入的tensor的list的每个tensor参数
            for p in group["params"]:
                if p.grad is None:
                    continue
                #每个参数都有自己的state，对于adamW是一阶二阶矩
                state = self.state[p]
                if len(state) == 0:#首次初始化
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 1
                m,v,t = state["m"],state["v"],state["t"]
                grad = p.grad.data
                alpha = lr * ((1-betas[1]**t)**0.5) /(1-betas[0]**t)
                p.data -= lr * weight_decay * p.data
                m = betas[0]*m +(1-betas[0])*grad
                v = betas[1]*v +(1-betas[1])*grad**2
                #m.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                #v.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])更节约显存不用多个

                t += 1
                p.data -= alpha * m/(v**0.5+eps)
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss

            



