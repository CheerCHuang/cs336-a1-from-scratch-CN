"""
Problem (training_together)：组装到一起 (4 points)

交付物：编写一个脚本，运行训练循环，在用户提供的输入上训练模型。
建议至少支持：

- 配置和控制各种模型和 optimizer 超参数
- 用 np.memmap 内存高效地加载大型训练和验证数据集
- 把 checkpoint 序列化到用户提供的路径
- 周期性记录训练和验证性能（输出到控制台和/或 Weights & Biases 等外部服务）

要用到的零件（都已实现）：
cs336_basics._15_transformer_lm      TransformerLM
cs336_basics._16_cross_entropy       cross_entropy
cs336_basics._18_adamw               AdamW
cs336_basics._19_learning_rate_schedule   get_lr_cosine_schedule
cs336_basics._20_gradient_clipping   gradient_clipping
cs336_basics._21_data_loading        get_batch
cs336_basics._22_checkpointing       save_checkpoint / load_checkpoint

参考循环（顺序有讲究）：
    for it in range(max_iters):
        1. 用调度器算出这一步的 lr，写进 optimizer.param_groups
        2. get_batch 取一批 (inputs, targets)
        3. logits = model(inputs)            # (B, m, vocab_size)
        4. loss = cross_entropy(logits, targets)   # 已支持任意 batch 维，不用 view
        5. loss.backward()
        6. gradient_clipping(model.parameters(), max_l2_norm)
        7. optimizer.step() → optimizer.zero_grad()
        8. 定期：算 valid loss、打印、save_checkpoint

运行示例：
    uv run python cs336_basics/_23_training_together.py \
        --train-path data/TinyStoriesV2-GPT4-train.bin ...
"""

from cs336_basics._15_transformer_lm import TransformerLM
from cs336_basics._16_cross_entropy import cross_entropy
from cs336_basics._18_adamw import AdamW
from cs336_basics._19_learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics._20_gradient_clipping import gradient_clipping
from cs336_basics._21_data_loading import get_batch
from cs336_basics._22_checkpointing import save_checkpoint, load_checkpoint
import numpy as np
import torch 

#DATA
TRAIN_PATH = "ts_train.bin"
VALID_PATH = "ts_valid.bin"
DATA_DTYPE = np.uint16

#model
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS =16
D_FF=1344
ROPE_THETA = 10000.0

#TRAIN
BATCH_SIZE = 8
MAX_ITERS = 100
LR_MAX =1e-3
LR_MIN=1e-4
WARMUP_ITERS = 10
COSINE_CYCLE_ITERS = 100
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8
GRAD_CLIP =1.0
DEVICE = "mps"
EVAL_BATCHES =5
EVAL_INTERVAL = 20     
CKPT_INTERVAL = 50      
CKPT_PATH = "ckpt"
        


def main():
    #1️.加载数据
    data = np.memmap(TRAIN_PATH,dtype=DATA_DTYPE,mode="r")
    print("数据长度:",len(data),"最大index", data.max())

    data_valid = np.memmap(VALID_PATH,dtype=DATA_DTYPE,mode="r")

    #2.建立模型
    model = TransformerLM(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        theta=ROPE_THETA,
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        num_layer=NUM_LAYERS, # ⚠️ 单数！
    )

    model = model.to(DEVICE)

    #3.优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LR_MAX,                   # 初值，之后每步被调度器覆盖
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    for it in range(MAX_ITERS):
        #1.学习率调度
        lr = get_lr_cosine_schedule(it, LR_MAX,LR_MIN,WARMUP_ITERS, COSINE_CYCLE_ITERS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        
        #2.取数据 (batch,context_length)
        inputs, targets = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)

        #3.前向
        logits = model(inputs)

        #4.交叉熵loss
        loss = cross_entropy(logits,targets)


        #5.反向
        loss.backward()

        #6裁剪
        gradient_clipping(model.parameters(), GRAD_CLIP)

        #7更新
        optimizer.step()
        optimizer.zero_grad()

        #8.验证集loss
        if it % EVAL_INTERVAL == 0:
            total = 0.0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    vinputs,vtargets = get_batch(data_valid, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
                    vlogits = model(vinputs)
                    total += cross_entropy(vlogits,vtargets).item() 
            avg = total / EVAL_BATCHES
            print(f"step {it} | train {loss.item():.4f} | valid {avg:.4f}")
        
        #CKPT保存
        if it % CKPT_INTERVAL == 0:
            save_checkpoint(model, optimizer, it, f"{CKPT_PATH}_{it}.pt")


if __name__ == "__main__":
    main()
