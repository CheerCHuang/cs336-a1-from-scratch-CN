"""
Problem (checkpointing)：实现模型 checkpointing (1 point)

实现下面两个保存/加载 checkpoint 的函数：

- save_checkpoint(model, optimizer, iteration, out)
  把 model、optimizer 和 iteration 的全部状态存入文件对象 out。
  用两者的 state_dict() 拿到状态，用 torch.save(obj, out) 写盘。
  典型做法是把 obj 做成一个字典（也可以是任何你之后能读回来的格式）。

- load_checkpoint(src, model, optimizer)
  从 src（路径或文件对象）加载 checkpoint，恢复 model 和 optimizer 的状态。
  返回 checkpoint 中保存的 iteration 数。
  用 torch.load(src) 读回，用两者的 load_state_dict() 恢复。

接口（见 tests/adapters.py）：
- out / src : str | os.PathLike | BinaryIO | IO[bytes]（路径或文件对象都支持）
- 返回      : load_checkpoint 返回 int（iteration）；save_checkpoint 不返回

测试：
1. 先实现 [adapters.run_save_checkpoint] 和 [adapters.run_load_checkpoint]
2. uv run pytest -k test_checkpointing
"""

import os
from typing import BinaryIO, IO
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
  model_dict = model.state_dict()#权重
  optimzer_state = optimizer.state_dict()#优化器状态

  ckpt={
"model":model_dict,
"optimzer":optimzer_state,
"iteration":iteration
  
  }
  torch.save(ckpt,out)
  return None

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
  ckpt = torch.load(src)
  model.load_state_dict(ckpt["model"])
  optimizer.load_state_dict(ckpt["optimzer"])
  return ckpt["iteration"]
