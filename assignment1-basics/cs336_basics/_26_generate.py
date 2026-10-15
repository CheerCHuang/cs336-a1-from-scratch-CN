"""
Problem (generate)：生成文本 (1 point)

用你的解码器和训练好的 checkpoint，报告模型生成的文本。你可能需要调整解码器参数
（temperature、top-p 等）以获得流畅的输出。

交付物：至少 256 个 token 的文本 dump（或直到第一个 <|endoftext|> token），
并简要评论输出的流畅度，以及**至少两个**影响输出好坏的因素。

--------------------------------------------------------------------------------
关键参数（别猜，按这几个来）
- 词表 / merges：tinystories_vocab.pkl、tinystories_merges.pkl（第 2 章训练产出）
- special_tokens：["<|endoftext|>"]，其 ID = **256**
  （依据 _01_train_bpe：vocab[0..255] 是单字节，special token 从 256 开始分配）
  ⚠️ 但你最好自己验证一次：用 tokenizer.encode("<|endoftext|>") 看返回的 ID
- 模型超参必须和训练时一致：d_model 512 / num_layer 4 / num_heads 16 /
  d_ff 1344 / vocab_size 10000 / context_length 256 / rope theta 10000
  错了会导致 load_checkpoint 报 shape mismatch（这其实是好事，能立刻发现）
- max_new_tokens：>= 256（交付物要求）
- prompt：建议用 "Once upon a time"（对齐题面示例，便于对比流畅度）
-------------------------------------------------------------------------------
--------------------------------------------------------------------------------
两个坑
1. load_checkpoint 的签名是 (src, model, optimizer)，**必须传 optimizer**。
   所以要先建一个 AdamW（参数用 model.parameters()），哪怕你不做训练。
2. 必须先构造一个**结构完全相同**的 TransformerLM，再把权重 load 进去；
   load_checkpoint 不会帮你建模型。
--------------------------------------------------------------------------------
流程
--------------------------------------------------------------------------------
1. 建 tokenizer：Tokenizer.from_files(vocab, merges, special_tokens=["<|endoftext|>"])
2. prompt 编码成 ID 列表 → 转成 (1, t) 的 tensor 放到 device
3. 建模型 + optimizer → load_checkpoint
4. generate(...) 拿到完整序列（含 prompt）
5. tokenizer.decode(...) 打印
6. 调 temperature / top_p，对比几组输出，挑最流畅的
--------------------------------------------------------------------------------
交付物要求：至少两个影响输出质量的因素
可以写的方向（挑你有数据支撑的）：
- temperature：太高 → 用词随机、语法崩；太低 → 重复循环、缺乏变化
- top_p：截断长尾能显著提升流畅度（小模型尤其明显）
- 训练 token 量：40M（低资源）vs 327M，题面自己给了对比——
  40M 的输出"像英语但不流畅"，这是最有力的一个因素
- 验证 loss：1.647 的模型 vs 更差的 checkpoint，输出质量差异
- prompt 长度 / 是否以 "Once upon a time" 开头（TinyStories 的强先验）
--------------------------------------------------------------------------------
低资源参考（题面给的 40M token 模型的输出水平）
--------------------------------------------------------------------------------
"Once upon a time, there was a little girl named Sue. Sue had a tooth that she
loved very much. It was his best head. ..." —— 像英语，但逻辑混乱、代词错乱。
你的输出达到或接近这个水平即达标。
"""

from cs336_basics._04_tokenizer import Tokenizer
from cs336_basics._15_transformer_lm import TransformerLM
from cs336_basics._18_adamw import AdamW
from cs336_basics._22_checkpointing import load_checkpoint
from cs336_basics._24_decoding import generate

import torch
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, default="Once upon a time", help="提示词")
    #分词器、模型权重
    parser.add_argument("--vocab-path", type=str, default="tinystories_vocab.pkl", help="分词器词表路径")
    parser.add_argument("--merges-path", type=str, default="tinystories_merges.pkl", help="分词器合并路径")
    parser.add_argument("--ckpt-path", type=str, default="ckpt/bs16-lr1e-3/ckpt_final.pt", help="权重加载路径")
    #输出配置
    parser.add_argument("--max-new-tokens", type=int, default=300, help="最大输出长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top-p", type=float, default=None, help="top-p采样")
    #模型setting
    parser.add_argument("--vocab-size", type=int, default=10000, help="词表大小")
    parser.add_argument("--num-layers", type=int, default=4, help="模型层数")
    parser.add_argument("--num-heads", type=int, default=16, help="注意力头")
    parser.add_argument("--d-model", type=int, default=512, help="hidden维度")
    parser.add_argument("--d-ff", type=int, default=1344, help="FFN hidden维度")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="rope角度")
    parser.add_argument("--context-length", type=int, default=256, help="上下文长度")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="mps", help="设备")

    args = parser.parse_args()
    torch.manual_seed(args.seed)   # generate 内部用 torch.multinomial 采样，必须设种子才可复现

    # 加载 tokenizer → 建模型 → load_checkpoint → generate → decode → 打印
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    print(tokenizer.encode("<|endoftext|>"))

    model = TransformerLM(
    d_model = args.d_model,
    num_heads = args.num_heads,
    d_ff = args.d_ff,
    theta = args.rope_theta,
    vocab_size = args.vocab_size,
    context_length = args.context_length,
    num_layer = args.num_layers
    )
    model = model.to(args.device)
    
    optimizer = AdamW(model.parameters())          # 空壳，只为接状态优化器
    it = load_checkpoint(args.ckpt_path, model, optimizer)
    print(f"loaded ckpt from iteration {it}")       # ← 这个打印很有价值
    model.eval()
    
    seq = generate(model, ids, max_new_tokens=args.max_new_tokens,
               temperature=args.temperature, top_p=args.top_p, eos_token_id=tokenizer.encode("<|endoftext|>")[0])#得到(1,T+N)tensor
    output = tokenizer.decode(seq[0].tolist())# t+n list
    print(output)

if __name__ == "__main__":
    main()
