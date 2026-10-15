"""
Problem (data_loading)：实现数据加载 (2 points)

交付物：编写一个函数，输入 numpy 数组 x（token ID 整数数组）、batch_size、
context_length 和 PyTorch 设备字符串（如 'cpu' 或 'cuda:0'），返回一对张量：
采样出的输入序列和对应的 next-token 目标。

要求：
- 两个张量形状都是 (batch_size, context_length)，内容是 token ID
- 两个张量都要放到指定的 device 上
- 返回 torch.LongTensor（embedding 层要求索引是 int64）

接口（见 tests/adapters.py::run_get_batch）：
- dataset       : 1D numpy 数组，token ID
- batch_size    : B，一批抽几条序列
- context_length: m，每条序列长度
- device        : 设备字符串

测试：
1. 先实现 [adapters.run_get_batch]（把调用转发到这里的函数）
2. uv run pytest -k test_get_batch
"""

import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    #每段数据为(batch_size,context_length)的输入和(batch_size,context_length)的标签
    input_datas = []
    labels = []
    starts = np.random.randint(0,len(dataset)-context_length,size=batch_size)
    for start in starts:
        input_data = dataset[start:start+context_length]
        label = dataset[start+1:start+context_length+1]
        input_datas.append(input_data)
        labels.append(label)
    input_datas = np.stack(input_datas)# 列表b个m维array变成1个array【b*m】
    labels = np.stack(labels)

    return torch.tensor(input_datas,device=device,dtype=torch.long),torch.tensor(labels,device=device,dtype=torch.long)
