"""
Problem (learning_rate_schedule)：实现带 warmup 的余弦学习率调度 (1 point)

编写一个函数，输入 t、α_max、α_min、T_w 和 T_c，按上述调度器返回学习率 α_t：

- warm-up：t < T_w 时，α_t = t/T_w · α_max
- 余弦退火：T_w ≤ t ≤ T_c 时，α_t = α_min + ½(1 + cos((t−T_w)/(T_c−T_w) · π))(α_max − α_min)
- 退火后：t > T_c 时，α_t = α_min

接口（见 tests/adapters.py::run_get_lr_cosine_schedule）：
- it                  : t，当前迭代数
- max_learning_rate   : α_max
- min_learning_rate   : α_min
- warmup_iters        : T_w
- cosine_cycle_iters  : T_c

测试：
1. 先实现 [adapters.get_lr_cosine_schedule]（把调用转发到这里的函数）
2. uv run pytest -k test_get_lr_cosine_schedule
"""

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        lr = it/warmup_iters * max_learning_rate
    elif it >= warmup_iters and it < cosine_cycle_iters:
        lr=min_learning_rate +0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        lr=min_learning_rate
    return lr
