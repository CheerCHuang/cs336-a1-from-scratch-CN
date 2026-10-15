"""
Problem (decoding)：解码 (3 points)

交付物：实现从语言模型解码的函数。建议支持以下功能：

- 为用户提供的 prompt 生成补全（输入 x_{1...t}，采样补全直到遇到 <|endoftext|> token）
- 允许用户控制生成的最大 token 数
- 给定温度值，在采样前对预测的 next-token 分布应用 softmax 温度缩放
- Top-p 采样（nucleus sampling），给定用户指定的阈值

说明：
- 本题**没有适配器/单元测试**，靠第 7 章 Problem (generate) 的生成结果来检验
- 自回归：每次把当前序列喂给模型，取**最后一个位置**的 logits 作为下一个 token 的分布
- 温度 τ：logits / τ 再 softmax（τ→0 接近贪心，τ 大则更随机）
- top-p：把概率降序排列，保留累积概率达到 p 的最小集合，其余置 0，重新归一化后采样
- 生成到 <|endoftext|>（token ID）或达到 max_new_tokens 时停止

建议接口（可自行调整）：

    def generate(
        model,               
        prompt,              
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor       

测试：跑一个小脚本，喂一段 prompt，打印 decode 出来的文本看是否通顺。
"""

#注意这里batch只支持1

import torch

@torch.no_grad()
def generate(
    model, # TransformerLM
    prompt: torch.Tensor, # (1, t) 或 (t,) 的 token ID 张量，放在 model 所在 device
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
):
    #prompt为单个的话会扩维
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    for _ in range(max_new_tokens):
        if prompt.size(1) > model.context_length:
            prompt = prompt[:, -model.context_length:]     # 只保留最后 context_length 个
        logits = model(prompt) #b,s,v
        next_logits = logits[:,-1,:]#B,V
        probs = torch.softmax(next_logits / temperature, dim=-1)#因为porbs就是词表所以排序前下表就是id
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs,dim=-1,descending=True)#得到排序prob和这些值在原数组里的下标
            cumsum = torch.cumsum(sorted_probs,dim=-1)#b,v

            to_remove = cumsum > top_p #逐元素比较
            to_remove[..., 1:] = to_remove[..., :-1].clone()## 右移一位：保留"踩线"
            to_remove[..., 0] = False#保留概率最高 
            sorted_probs = sorted_probs.masked_fill(to_remove,0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            pos = torch.multinomial(sorted_probs, num_samples=1)#B,1 对最后维度采样B次，得到采样到的prob的索引
            next_id = torch.gather(input=sorted_idx, dim=-1, index=pos)#pos是采样到的sorted_probs索引，sorted_idx是排序前对应的下表，所以要用pos查id，得到排序前数组的下标
        else:
            next_id = torch.multinomial(probs,num_samples=1)#得到(B,1)张量

        prompt = torch.cat([prompt,next_id],dim=1)#B,S+1
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return prompt
