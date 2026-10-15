"""
第 7 章实验用的训练脚本（argparse 版 + 后面接 wandb）

说明：
- 这是给第 7 节（learning_rate / batch_size_experiment / 消融 / main_experiment /
  leaderboard）跑实验用的脚本，_23 保持原样作为第 5 章 training_together 的交付物
- 循环逻辑照搬 _23，只把模块级常量换成命令行参数，扫超参时不用改代码
- 低资源提示：cpu/mps 上总 token 数降到 40,000,000，验证 loss 目标放宽到 2.00

--------------------------------------------------------------------------------
7.2.1 题面给的基线（这些是"官方起点"，不是要调的）
--------------------------------------------------------------------------------

vocab-size 10000 / context-length 256 / d-model 512 / d-ff 1344（≈8/3×512，取 64 倍数）/
rope-theta 10000.0 / num-layers 4 / num-heads 16  → 约 17M 参数

--------------------------------------------------------------------------------
7.2.1 要你自己试出来的（题目点名：trial and error）
--------------------------------------------------------------------------------

learning rate / lr warmup / AdamW 的 β1 β2 ε / weight decay

--------------------------------------------------------------------------------
7.2.1 要你 sweep 观察的（看两个指标：per-iteration runtime + final perplexity）
--------------------------------------------------------------------------------

- vocab size：⚠️ 改它要**重训 tokenizer**（第 2 章代码，merge 数跟着变），成本最高，放最后做
- context length：观察单步耗时怎么变、最终 ppl 怎么变；后面 OWT 可能需要加长
- perplexity = exp(loss)，所以每个 run 顺手记一下 valid_ppl = exp(valid_loss)

--------------------------------------------------------------------------------
两个 token 预算别搞混
--------------------------------------------------------------------------------
- 标准版（B200）：327,680,000 = 128 × 10000 × 256
- 低资源版（cpu/mps，我们走这个）：约 40,000,000，官方参考 32 × 5000 × 256 = 40,960,000
  在 M4 Max + mps 上 36 分钟，第 5000 步 valid loss 1.80
--------------------------------------------------------------------------------
当前 default（低资源实验配置 32 × 5000 × 256，不是 _23 的旧值）
--------------------------------------------------------------------------------

模型：d-model 512 / num-layers 4 / num-heads 16 / d-ff 1344 /
      vocab-size 10000 / context-length 256 / rope-theta 10000.0
数据：train-path ts_train.bin / valid-path ts_valid.bin /
      data-dtype uint16(str) / batch-size 32
优化：lr-max 1e-3（待扫描）/ lr-min 1e-4 / warmup-iters 250(5%) /
      cosine-cycle-iters None / beta1 0.9 / beta2 0.999 / eps 1e-8 /
      weight-decay 0.01 / grad-clip 1.0
训练：max-iters 5000 / max-tokens None / eval-interval 200 / eval-batches 5 /
      ckpt-interval 1000 / ckpt-dir ckpt / seed 42 / device mps
日志：run-name / wandb-project / wandb-mode（已接入）

每步记录：train_loss / lr / tokens_seen / grad_norm（裁剪前）
评估时记录：valid_loss / valid_ppl
墙钟由 wandb 自动记（_runtime），画 loss vs time 直接在网页切横轴

--------------------------------------------------------------------------------
已经拍板的决策（原来的三个坑已有答案）
--------------------------------------------------------------------------------

1. cosine-cycle-iters 默认 None，在 main() 里 fallback 到 max_iters
   → 保证余弦衰减正好在第 N 步到最低 lr（题面明确要求）
2. beta 拆成 --beta1 / --beta2 两个 float，不用 tuple
3. max-tokens 给了就覆盖 max_iters：max_iters = tokens // (batch × context)

--------------------------------------------------------------------------------
7.2.2 预期耗时 / 瓶颈排查
--------------------------------------------------------------------------------

- 官方参考：这套超参在 1 B200 上约 20–30 分钟
- 若你的耗时远超预期，题面点名三个可疑瓶颈：
  dataloading（是否 np.memmap）、checkpointing（是否写太频繁）、
  validation loss 计算（是否进了 no_grad / 是否 batch 化）
  另外确认实现是**正确 batched** 的
- 低资源版：mps 36 分钟 / cpu 1 小时 22 分（M4 Max 36GB）

--------------------------------------------------------------------------------
7.2.3 调试建议（模型有问题时按这个顺序查）
--------------------------------------------------------------------------------

1. **先过拟合单个 minibatch**：实现正确的话，反复喂同一批数据应该能迅速把
   train loss 打到接近 0。做不到 → 实现有 bug，别急着跑长训练
2. 在模型各组件打断点，检查中间张量 shape 是否符合预期
3. 监控 activation / weight / gradient 的 norm，看有没有爆炸或消失
   （这条和 learning_rate (b) 的 edge of stability 直接相关）
4. 优先用 IDE debugger 而不是 print

--------------------------------------------------------------------------------
Problem (learning_rate) 的判据
--------------------------------------------------------------------------------

(a) lr 扫描 + 最终 loss（发散的要标注"发散"）
    目标：TinyStories 验证 loss(per-token) ≤ 1.45
    低资源（我们在 mps）：总 token 降到 40,000,000，目标放宽到 ≤ 2.00
    官方参考：32×5000×256，第 5000 步 valid loss 1.80
(b) edge of stability：递増 lr 直到至少有一条**发散**的曲线，
    分析发散点与最佳 lr 的关系（最佳 lr 通常略低于发散阈值）

--------------------------------------------------------------------------------
MPS 专属注意事项（题面明说，别踩）
--------------------------------------------------------------------------------

- ❌ 不要 torch.set_float32_matmul_precision('high')：torch 2.9.0 + mps 会用到
  静默出错的 TF32 kernel，训练不稳定（很可能是"莫名发散"的元凶）
- 想加速再试 torch.compile(model, backend="aot_eager")（mps）/ torch.compile(model)（cpu）；
  先跑通不加 compile 的版本，确认 loss 正常后再加，避免引入新变量
- 余弦衰减要正好在第 N 步到最低 lr（N = 总步数）

--------------------------------------------------------------------------------
run 的留痕（wandb config / 或自己存 json）至少要有
--------------------------------------------------------------------------------

- 模型超参 + 消融开关（norm_type / pos_encoding / ffn_type，做到消融题时再加）
- 优化超参、batch_size、seed、device、参数总量
- 数据集路径与总 token 数、起始时间
判据：三天后的你能照着它原样复现这个 run。

--------------------------------------------------------------------------------
每个 run 要能画出两条曲线
--------------------------------------------------------------------------------

- 横轴 step：train / valid loss
- 横轴 wall-clock：valid loss（leaderboard 题要求横轴 < 45 分钟）
"""

from cs336_basics._15_transformer_lm import TransformerLM
from cs336_basics._16_cross_entropy import cross_entropy
from cs336_basics._18_adamw import AdamW
from cs336_basics._19_learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics._20_gradient_clipping import gradient_clipping
from cs336_basics._21_data_loading import get_batch
from cs336_basics._22_checkpointing import save_checkpoint, load_checkpoint

import argparse
import numpy as np
import torch
import os
import wandb
import math

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--vocab-size", type=int, default=10000, help="词表大小")
  parser.add_argument("--context-length", type=int, default=256, help="上下文长度")
  parser.add_argument("--d-model", type=int, default=512, help="hidden维度")
  parser.add_argument("--d-ff", type=int, default=1344, help="FFN维度")
  parser.add_argument("--rope-theta", type=float, default=10000.0, help="ROPE的维度")
  parser.add_argument("--num-layers", type=int, default=4, help="模型层数")
  parser.add_argument("--n-heads", type=int, default=16, help="注意力头")

  parser.add_argument("--train-path", type=str, default="ts_train.bin", help="训练集")
  parser.add_argument("--valid-path", type=str, default="ts_valid.bin", help="验证集")
  parser.add_argument("--data-dtype", type=str, default="uint16", help="数据集类型，用 getattr(np, args.data_dtype) 取类型")
  parser.add_argument("--batch-size", type=int, default=32, help="批大小")

  parser.add_argument("--lr-max", type=float, default=1e-3, help="最大学习率")
  parser.add_argument("--lr-min", type=float, default=1e-4, help="最小学习率")
  parser.add_argument("--warmup-iters", type=int, default=250	, help="5000 步的 0.05")
  parser.add_argument("--cosine-cycle-iters", type=int, default=None, help="None 时取 max_iters")
  parser.add_argument("--beta1", type=float, default=	0.9, help="Adam一阶超参数")
  parser.add_argument("--beta2", type=float, default=	0.999, help="Adam二阶超参数")
  parser.add_argument("--eps", type=float, default=	1e-8, help="epison")
  parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
  parser.add_argument("--grad-clip", type=float, default=	1.0, help="梯度裁剪")

  parser.add_argument("--max-iters", type=int, default=	5000, help="迭代步数")
  parser.add_argument("--max-tokens", type=int, default=None, help="训练总token预算")

  parser.add_argument("--eval-interval", type=int, default=	200	, help="验证间隔")
  parser.add_argument("--eval-batches", type=int, default= 5, help="eval batch")
  parser.add_argument("--ckpt-interval", type=int, default= 1000, help="权重保存间隔")
  parser.add_argument("--ckpt-dir", type=str, default="ckpt", help="权重保存路径")
  parser.add_argument("--no-save-ckpt", action="store_false", dest="save_ckpt",
                    help="不保存任何 checkpoint（过拟合测试/调试用）")


  parser.add_argument("--seed", type=int, default=42, help="随机种子")
  parser.add_argument("--device", type=str, default="mps", help="设备")

  parser.add_argument("--run-name", type=str, default=None, help="wandb名字，建议扫 lr 时手动传 lr1e-3 这种，事后好认")
  parser.add_argument("--wandb-project", type=str, default="cs336-a1", help="所有 run 放同一个 project 里才能叠加对比")
  parser.add_argument("--wandb-mode", type=str, default="online", help="想离线跑改 offline；想临时完全关掉改 disabled")
  
  parser.add_argument("--overfit-batch", action="store_true", help="反复训练同一个 batch，用于验证实现正确性")

  #消融
  parser.add_argument("--norm-type", type=str, default="pre", help="pre=pre-norm rmsnorm（基线），post=post-norm，none=移除所有 rmsnorm")
  parser.add_argument("--pos-encoding", type=str, default="rope", help="rope=启动旋转位置编码，none=NoPE没有位置编码")
  parser.add_argument("--ffn-type", type=str, default="swiglu", help="swiglu=门控（基线），silu=无门控（d_ff 自动设为 4*d_model）")


  args = parser.parse_args()
  
  if args.ffn_type == "silu":
    args.d_ff = 4 * args.d_model      # 2048，保证参数量匹配,不启动门控

  # 过拟合测试本来就用不到权重，自动关闭保存，省 520MB/run
  if args.overfit_batch:
    args.save_ckpt = False

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  dtype = getattr(np, args.data_dtype)

  if args.max_tokens is not None:#启动token计算配额而不是迭代步配额
    args.max_iters = args.max_tokens // (args.batch_size * args.context_length)
  if args.cosine_cycle_iters is None:
    args.cosine_cycle_iters = args.max_iters

  train_data = np.memmap(args.train_path, dtype=dtype, mode="r")
  valid_data = np.memmap(args.valid_path, dtype=dtype, mode="r")
  # print(len(train_data) )
  # print(train_data.max(), "词表", args.vocab_size)   # 应该 < args.vocab_size
  # print(args.max_iters,args.cosine_cycle_iters,args.batch_size * args.max_iters * args.context_length)
  model = TransformerLM(
    d_model = args.d_model,
    num_heads = args.n_heads,
    d_ff = args.d_ff,
    theta = args.rope_theta,
    vocab_size = args.vocab_size,
    context_length = args.context_length,
    num_layer = args.num_layers,
    norm_type=args.norm_type,
    pos_encoding=args.pos_encoding,
    ffn_type=args.ffn_type
  )
  model = model.to(args.device)
  # print(sum(p.numel() for p in model.parameters()))


  optimizer = AdamW(
    params = model.parameters(),
    lr = args.lr_max,#之后会被覆盖
    betas = (args.beta1, args.beta2),
    eps = args.eps,
    weight_decay = args.weight_decay
  )
  wandb.init(
    project=args.wandb_project,
    name=args.run_name,
    config=vars(args),
    mode=args.wandb_mode,
  )

  # wandb 启用时用 run 名做子目录，避免不同 run 的权重互相覆盖
  # 注意：必须在 wandb.init() 之后，否则拿不到 wandb.run.name
  if args.wandb_mode != "disabled":
    ckpt_dir = f"{args.ckpt_dir}/{wandb.run.name}"
  else:
    ckpt_dir = args.ckpt_dir
  if args.save_ckpt:
    os.makedirs(ckpt_dir, exist_ok=True)#checkpoint文件夹

  fixed_inputs, fixed_targets = get_batch(train_data, args.batch_size, args.context_length, args.device)#debug用过拟合

  for it in range(args.max_iters):
    #1.学习率调度
    lr = get_lr_cosine_schedule(it, args.lr_max,args.lr_min,args.warmup_iters, args.cosine_cycle_iters)
    for group in optimizer.param_groups:#其实只有一个组，可改为optimizer.param_groups[0]["lr"] = lr
      group["lr"] = lr
    #2.取数据 (batch,context_length)
    if args.overfit_batch: #debug用过拟合
      inputs, targets = fixed_inputs, fixed_targets
    else:
      inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

    #3.前向
    logits = model(inputs)
    #4.交叉熵loss
    loss = cross_entropy(logits,targets)
    #5.反向
    loss.backward()

    grad_norm = 0
    for p in model.parameters():
      if p.grad is None:
        continue
      grad_norm += p.grad.detach().pow(2).sum().item()
    grad_norm = grad_norm**0.5

    #6裁剪
    gradient_clipping(model.parameters(), args.grad_clip)
    #7更新
    optimizer.step()
    optimizer.zero_grad()

    
    wandb.log(
    {"train_loss": loss.item(), "lr": lr, "tokens_seen": (it+1) * args.batch_size * args.context_length,"grad_norm":grad_norm},
    step=it,
    )

    #8.验证集loss
    if it % args.eval_interval == 0:
        total = 0.0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                vinputs,vtargets = get_batch(valid_data, args.batch_size, args.context_length, args.device)
                vlogits = model(vinputs)
                total += cross_entropy(vlogits,vtargets).item() 
        avg = total / args.eval_batches
        print(f"step {it} | train {loss.item():.4f} | valid {avg:.4f}")
        valid_ppl = math.exp(avg) if avg < 20 else float("inf")
        wandb.log({"valid_loss": avg, "valid_ppl": valid_ppl}, step=it)


    #CKPT保存
    if args.save_ckpt and it % args.ckpt_interval == 0 and it > 0:# it=0 是随机初始化，不存
        save_checkpoint(model, optimizer, it, f"{ckpt_dir}/ckpt_{it}.pt")
  if args.save_ckpt:
    save_checkpoint(model, optimizer, args.max_iters, f"{ckpt_dir}/ckpt_final.pt")

  wandb.finish()

if __name__ == "__main__":
    main()
