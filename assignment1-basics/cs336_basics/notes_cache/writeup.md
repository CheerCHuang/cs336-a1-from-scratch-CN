# A1 Basics — Writeup（题目中文版 + 我的答案）

> 题目由 `cs336_assignment1_basics.pdf` 翻译为中文，术语保留英文。答案区留空，自己做完再填。
> 交作业时最终 writeup 需导出为 `writeup.pdf`。

---


---



---

## 4. 训练 Transformer LM

### Problem (cross_entropy)：实现 cross-entropy (1 point)

交付物：编写计算 cross-entropy loss 的函数，输入预测 logits（o_i）和目标（x_{i+1}），计算 cross-entropy ℓ_i = −log softmax(o_i)[x_{i+1}]。函数应处理：

- 为数值稳定性减去最大元素。
- 尽可能消去 log 和 exp。
- 处理任意额外的 batch 维度并返回跨 batch 的平均值。与 3.2 节一样，假设 batch 维度总是排在最前面（在词表大小维度之前）。

实现 [adapters.run_cross_entropy]，然后运行 `uv run pytest -k test_cross_entropy`。

**我的回答：**

---

### Problem (learning_rate_tuning)：调节学习率 (1 point)

如我们将看到的，学习率是影响训练最大的超参数之一。在上面的 toy 例子中实际看一下：分别用另外三个学习率 1e1、1e2、1e3 运行 SGD 例子，只训练 10 个 iteration。每个学习率下 loss 发生了什么？是衰减更快、更慢，还是发散（即随训练进行而增大）？

交付物：一到两句话的回答，描述你观察到的行为。

**我的回答：**

当学习率 lr=1e1 时，损失平滑下降（从 26.3 降至 3.5）；当 lr=1e2 时，第一步损失保持不变（更新因子恰好为 -1，因此权重仅改变符号），之后到第 4 步时损失骤降至约 1e-16。当 lr=1e3 时，损失从 26.3 激增至 2.4e18，即发生发散——一旦每步更新因子的绝对值超过 1，优化过程就会崩溃。

---

### Problem (adamw)：实现 AdamW (2 points)

交付物：把 AdamW optimizer 实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 α，以及 β、ε、λ 超参数。为了保存状态，基类 Optimizer 提供了字典 `self.state`，它把 `nn.Parameter` 对象映射到存储该参数所需信息的字典（对 AdamW 来说就是矩估计量）。实现 [adapters.get_adamw_cls]，确保通过 `uv run pytest -k test_adamw`。

**我的回答：**

---

### Problem (adamw_accounting)：使用 AdamW 训练的资源核算 (2 points)

我们来计算运行 AdamW 需要多少内存和计算量。假设所有张量都用 float32。

(a) 运行 AdamW 需要多少峰值内存？基于参数（parameters）、激活（activations）、梯度（gradients）和 optimizer state 的内存使用分别给出分解，用 batch_size 和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表达答案。假设 d_ff = 8/3 × d_model。为简单起见，计算激活内存时只考虑以下组件：
- Transformer block：RMSNorm(s)；多头自注意力子层（QKV 投影、QKᵀ 矩阵乘、softmax、values 加权和、输出投影）；position-wise 前馈（SwiGLU）：W1、W2、门控分支上的 SiLU、逐元素乘积、W3
- 最终 RMSNorm
- 输出 embedding
- logits 上的 cross-entropy

交付物：分别对 parameters、activations、gradients 和 optimizer state 给出代数表达式，以及总量。

(b) 用 GPT-2 XL 尺寸实例化你的答案，得到只依赖 batch_size 的表达式。在 80GB 内存限制下你能用的最大 batch size 是多少？
交付物：形如 a·batch_size + b 的表达式（a、b 为数值），以及最大 batch size 数值。

(c) 运行一步 AdamW 需要多少 FLOPs？
交付物：代数表达式，附简要说明。

(d) 模型 FLOPs 利用率（MFU）定义为观测吞吐量（tokens per second）相对硬件理论峰值 FLOP 吞吐的比值。NVIDIA H100 GPU 的 "float32"（实际是 TensorFloat-32）理论峰值为 495 teraFLOP/s。假设你能达到 50% MFU，在单张 H100 上以 batch size 1024 训练 GPT-2 XL 400K 步需要多长时间？假设 backward pass 的 FLOPs 是 forward pass 的两倍。
交付物：训练所需小时数，附简要说明。

**我的回答：**

(a)


> 符号：$n$=层数，$d$=d_model，$h$=num_heads，$v$=vocab_size，$s$=context_length，$b$=batch_size，$d_{\text{ff}}=\frac{8}{3}d$

---

参数（Parameters）

| 组件 | 表达式 | 数量 |
|---|---|---|
| 每层 2×RMSNorm weight | $2d$ | |
| 每层 Q/K/V/O 投影 | $4d^2$ | |
| 每层 FFN（W₁/W₂/W₃） | $3d \cdot d_{\text{ff}} = 8d^2$ | |
| **每层小计** | $2d + 12d^2$ | $\times n$ |
| 最终 RMSNorm | $d$ | |
| 输入 + 输出 embedding | $2dv$ | |

$$N_{\text{param}} = n(2d + 12d^2) + d + 2dv$$

$$\text{Mem}_{\text{param}} = 4 \cdot N_{\text{param}}$$



激活（Activations）

| 组件（每层） | 形状 | 元素数 |
|---|---|---|
| 2×RMSNorm 输出 | $(b,s,d)$ | $2bsd$ |
| Q/K/V 投影 | $(b,s,d)$ | $3bsd$ |
| QKᵀ 注意力分数 | $(b,h,s,s)$ | $bhs^2$ |
| softmax 输出 | $(b,h,s,s)$ | $bhs^2$ |
| attn·V 加权和 | $(b,s,d)$ | $bsd$ |
| O 投影输出 | $(b,s,d)$ | $bsd$ |
| W₁ + W₃ 升维投影 | $(b,s,d_{\text{ff}})$ | $2bsd_{\text{ff}}$ |
| SiLU 输出 | $(b,s,d_{\text{ff}})$ | $bsd_{\text{ff}}$ |
| 逐元素乘积 | $(b,s,d_{\text{ff}})$ | $bsd_{\text{ff}}$ |
| W₂ 降维输出 | $(b,s,d)$ | $bsd$ |

每层小计：$8bsd + 2bhs^2 + 4bsd_{\text{ff}} = \frac{56}{3}bsd + 2bhs^2$

| 全局组件 | 元素数 |
|---|---|
| 最终 RMSNorm | $bsd$ |
| logits（输出 embedding） | $bsv$ |
| cross-entropy | $bs$ |

$$N_{\text{act}} = n\left(\frac{56}{3}bsd + 2bhs^2\right) + bs(d + v + 1)$$

$$\text{Mem}_{\text{act}} = 4 \cdot N_{\text{act}}$$

---

梯度（Gradients）

形状与参数一一对应：

$$\text{Mem}_{\text{grad}} = 4 \cdot N_{\text{param}}$$

优化器状态（Optimizer State）

每个参数维护 $m$（一阶矩）和 $v$（二阶矩）各一份：

$$\text{Mem}_{\text{opt}} = 8 \cdot N_{\text{param}}$$



总峰值内存

$$\boxed{\text{Peak Memory} = 16 \cdot N_{\text{param}} + 4 \cdot N_{\text{act}}}$$

$$= 16\bigl[n(2d+12d^2)+d+2dv\bigr] + 4\left[n\left(\frac{56}{3}bsd+2bhs^2\right)+bs(d+v+1)\right] \quad \text{(元素数)}$$

实际字节数再乘 4（float32）。

(b)

用 **GB（十进制，$1\,\text{GB}=10^9\,\text{bytes}$）** 口径重新算一遍：## 问答题精简版（GB 十进制口径）

**配置：** $v{=}50257,\ s{=}1024,\ n{=}48,\ d{=}1600,\ h{=}25,\ d_{\text{ff}}{=}4288$，float32，$1\,\text{GB}=10^9\,\text{bytes}$。

**参数总数**（与 batch 无关）：

$$N_p = n(2d + 4d^2 + 3dd_{\text{ff}}) + d + 2dv = 1{,}640{,}452{,}800$$

**每 batch 激活元素数**：

$$N_a = n(8sd + 2hs^2 + 4sd_{\text{ff}}) + s(d + v + 1) = 4{,}041{,}885{,}696$$

**总峰值内存** = 参数($4N_p$) + 梯度($4N_p$) + optimizer($8N_p$) + 激活($4bN_a$)：

$$\text{Peak} = 16N_p + 4bN_a$$

$$\boxed{\text{Peak} = 16{,}167{,}542{,}784\,b + 26{,}247{,}244{,}800\ \text{bytes} \approx 16.17b + 26.25\ \text{GB}}$$

**最大 batch size**（80 GB 限制）：

$$b_{\max} = \left\lfloor\frac{80 \times 10^9 - 26{,}247{,}244{,}800}{16{,}167{,}542{,}784}\right\rfloor = \lfloor 3.32 \rfloor = \boxed{3}$$

> $b{=}3$ 时约 74.75 GB，$b{=}4$ 时约 90.92 GB 超限。

(c)

 AdamW 单步 FLOPs

对每个参数元素（共 $N$ 个可训练参数），逐行分解：

| 步骤 | 公式 | 每元素 FLOPs |
|---|---|---|
| 权重衰减 | $\theta \leftarrow \theta - \alpha\lambda\theta$ | 2（1乘 + 1减） |
| 一阶矩更新 | $m \leftarrow \beta_1 m + (1-\beta_1)g$ | 3（2乘 + 1加） |
| 二阶矩更新 | $v \leftarrow \beta_2 v + (1-\beta_2)g^2$ | 4（3乘 + 1加，含 $g^2$） |
| 参数更新 | $\theta \leftarrow \theta - \alpha_t \dfrac{m}{\sqrt{v}+\varepsilon}$ | 5（1 sqrt + 1加 + 1除 + 1乘 + 1减） |
| **合计** | | **14** |

$$\boxed{\text{FLOPs per AdamW step} = 14 \cdot N_{\text{param}}}$$

**理由：**
- 所有更新都是逐元素（element-wise）操作，每个参数元素固定 14 次浮点运算，与张量形状无关。
- $\alpha_t = \alpha\sqrt{1-\beta_2^t}/(1-\beta_1^t)$、$t$ 递增、$\beta^t$ 递推都是**标量运算**，每步仅常数次，与 $N$ 无关，忽略不计。
- 梯度 $g$ 由反向传播产生，**不算在 optimizer step 内**。
- 若课程只统计乘法（MACs），则约为 $8N$（权重衰减1 + m更新2 + v更新3 + 更新2）；完整 FLOPs（含加减除 sqrt）为 $14N$。

代入你的 GPT-2 XL（$N_{\text{param}}=1{,}640{,}452{,}800$）：单步约 $2.30 \times 10^{10}$ FLOPs（23 GFLOPs），与一次前向的 ~3.5 TFLOPs 相比可以忽略。

(d)

单步前向 FLOPs（batch_size=1024），再算总训练时间。训练时间计算

**已知：**
- 单样本前向 FLOPs（seq=1024）：$3.517 \times 10^{12}$
- 反向 = 2 × 前向，故单步总 = 3 × 前向
- 步数：400,000，batch_size=1024
- H100 峰值：495 TFLOP/s，MFU=50% → 有效算力 247.5 TFLOP/s

**单步前向（batch=1024）：**

$$3.517 \times 10^{12} \times 1024 = 3.601 \times 10^{15}\ \text{FLOPs}$$

**单步总（前向+反向）：**

$$3 \times 3.601 \times 10^{15} = 1.080 \times 10^{16}\ \text{FLOPs}$$

**400K 步总 FLOPs：**

$$400{,}000 \times 1.080 \times 10^{16} = 4.321 \times 10^{21}\ \text{FLOPs}$$

**训练时间：**

$$t = \frac{4.321 \times 10^{21}}{247.5 \times 10^{12}} = 1.746 \times 10^7\ \text{秒}$$

$$\boxed{t \approx 4850\ \text{小时} \approx 202\ \text{天}}$$

**简要理由：** 每步前向约 3.6 PFLOPs，加反向共约 10.8 PFLOPs；单卡有效算力 \(247.5\) TFLOP/s，每步耗时约 43.6 秒；40 万步累计约 1746 万秒，折合 4850 小时（约 202 天）。optimizer step 的 FLOPs（~23 GFLOPs/步）相对前向+反向可忽略。

---

### Problem (learning_rate_schedule)：实现带 warmup 的余弦学习率调度 (1 point)

编写一个函数，输入 t、α_max、α_min、T_w 和 T_c，按上述调度器返回学习率 α_t（warm-up：t < T_w 时 α_t = t/T_w · α_max；余弦退火：T_w ≤ t ≤ T_c 时 α_t = α_min + ½(1 + cos((t−T_w)/(T_c−T_w)·π))(α_max − α_min)；退火后：t > T_c 时 α_t = α_min）。然后实现 [adapters.get_lr_cosine_schedule]，确保通过 `uv run pytest -k test_get_lr_cosine_schedule`。

**我的回答：**

---

### Problem (gradient_clipping)：实现梯度裁剪 (1 point)

编写实现梯度裁剪的函数。函数应接受参数列表和最大 ℓ2-范数。它应就地（in place）修改每个参数的梯度。使用 ε = 10⁻⁶（PyTorch 默认值）。然后实现适配器 [adapters.run_gradient_clipping]，确保通过 `uv run pytest -k test_gradient_clipping`。

**我的回答：**

---

## 5. 训练循环

### Problem (data_loading)：实现数据加载 (2 points)

交付物：编写一个函数，输入 numpy 数组 x（token ID 整数数组）、batch_size、context_length 和 PyTorch 设备字符串（如 'cpu' 或 'cuda:0'），返回一对张量：采样出的输入序列和对应的 next-token 目标。两个张量形状都应为 (batch_size, context_length)，包含 token ID，且都放在指定设备上。测试：先实现测试适配器 [adapters.run_get_batch]，然后运行 `uv run pytest -k test_get_batch`。

**我的回答：**

---

### Problem (checkpointing)：实现模型 checkpointing (1 point)

实现以下两个保存/加载 checkpoint 的函数：

- `def save_checkpoint(model, optimizer, iteration, out)`：把 model、optimizer 和 iteration 的全部状态存入文件对象 out。
  - model: torch.nn.Module；optimizer: torch.optim.Optimizer；iteration: int；out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
- `def load_checkpoint(src, model, optimizer)`：从 src（路径或文件对象）加载 checkpoint，然后恢复 model 和 optimizer 的状态。函数应返回 checkpoint 中保存的 iteration 数。
  - src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

实现 [adapters.run_save_checkpoint] 和 [adapters.run_load_checkpoint] 适配器，确保通过 `uv run pytest -k test_checkpointing`。

**我的回答：**

---

### Problem (training_together)：组装到一起 (4 points)

交付物：编写一个脚本，运行训练循环，在用户提供的输入上训练模型。我们建议你的训练脚本至少支持：

- 配置和控制各种模型和 optimizer 超参数。
- 用 `np.memmap` 内存高效地加载大型训练和验证数据集。
- 把 checkpoint 序列化到用户提供的路径。
- 周期性记录训练和验证性能（如输出到控制台和/或 Weights and Biases 等外部服务）。

**我的回答：**

---

## 6. 生成文本

### Problem (decoding)：解码 (3 points)

交付物：实现从语言模型解码的函数。我们建议支持以下功能：

- 为用户提供的 prompt 生成补全（即输入 x_{1...t}，采样补全直到遇到 `<|endoftext|>` token）。
- 允许用户控制生成的最大 token 数。
- 给定温度值，在采样前对预测的 next-token 分布应用 softmax 温度缩放。
- Top-p 采样（又称 nucleus sampling），给定用户指定的阈值。

**我的回答：**

---

## 7. 实验

### Problem (experiment_log)：实验记录 (3 points)

为你的训练和评估代码创建实验跟踪基础设施，使你能跟踪实验及 loss 曲线相对梯度步数和 wall-clock time 的变化。

交付物：实验的日志基础设施代码，以及针对本节下列作业问题的实验记录（一份记录你尝试过的所有事情的文档）。

**我的回答：**

---

### Problem (learning_rate)：调节学习率 (2 B200 hrs) (3 points)

学习率是最重要的超参数之一。基于你训练的基础模型，回答以下问题：

(a) 对学习率做超参数扫描，报告最终 loss（若 optimizer 发散则注明发散）。
交付物：多个学习率对应的 learning curves。解释你的超参数搜索策略。
交付物：一个模型在 TinyStories 上的验证 loss（per-token）不超过 1.45。

（低资源提示：在 cpu/mps 上把处理的总 token 数减少到 40,000,000；可以把目标验证 loss 放宽到 2.00。）

(b) 经验之谈是最佳学习率处于"稳定边缘"（edge of stability）。研究学习率发散点与你的最佳学习率之间的关系。
交付物：递增学习率的 learning curves，其中至少包含一次发散的运行，并分析这与收敛速度的关系。

**我的回答：**

(a)

(b)

---

### Problem (batch_size_experiment)：batch size 变化 (1 B200 hr) (1 point)

把 batch size 从 1 一直变到 GPU 内存上限。中间至少试几个 batch size，包括典型值 64 和 128。

交付物：不同 batch size 的运行的 learning curves。如有必要学习率要重新优化。
交付物：用几句话讨论你对 batch size 及其对训练影响的发现。

**我的回答：**

---

### Problem (generate)：生成文本 (1 point)

使用你的解码器和训练好的 checkpoint，报告你的模型生成的文本。你可能需要调整解码器参数（temperature、top-p 等）以获得流畅的输出。

交付物：至少 256 个 token 的文本 dump（或直到第一个 `<|endoftext|>` token），并简要评论输出的流畅度，以及至少两个影响输出好坏的因素。

**我的回答：**

---

### Problem (layer_norm_ablation)：移除 RMSNorm 并训练 (0.5 B200 hrs) (1 point)

移除 Transformer 中的所有 RMSNorm 并训练。在之前的最优学习率下会发生什么？用更低的学习率能获得稳定性吗？

交付物：移除 RMSNorm 训练时的 learning curve，以及最优学习率下的 learning curve。
交付物：几句话评论 RMSNorm 的影响。

**我的回答：**

---

### Problem (pre_norm_ablation)：实现 post-norm 并训练 (0.5 B200 hrs) (1 point)

把你的 pre-norm Transformer 实现修改为 post-norm。用 post-norm 模型训练，看看会发生什么。

交付物：post-norm Transformer 的 learning curve，与 pre-norm 对比。

**我的回答：**

---

### Problem (no_pos_emb)：实现 NoPE (0.5 B200 hrs) (1 point)

修改你的带 RoPE 的 Transformer 实现，完全移除位置编码信息，看看会发生什么。

交付物：对比 RoPE 与 NoPE 性能的 learning curve。

**我的回答：**

---

### Problem (swiglu_ablation)：SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)

测试前馈网络中门控机制的重要性：对比 SwiGLU 前馈网络与使用 SiLU 激活但无门控线性单元（GLU）的前馈网络：FFN_SiLU(x) = W2·SiLU(W1·x)。在这个消融基线中，你的 FFN_SiLU 实现应设 d_ff = 4 × d_model，以近似匹配默认 SwiGLU 前馈网络的参数量。

交付物：参数量近似匹配条件下，SwiGLU 与 SiLU 前馈网络性能对比的 learning curve。
交付物：几句话讨论你的发现。

**我的回答：**

---

### Problem (main_experiment)：在 OWT 上实验 (2 B200 hrs) (2 points)

用与 TinyStories 相同的模型架构和总训练迭代数在 OpenWebText 上训练你的语言模型。这个模型表现如何？

交付物：你的语言模型在 OpenWebText 上的 learning curve。描述与 TinyStories 上 loss 的差异——我们应该如何解释这些 loss？
交付物：OpenWebText LM 生成的文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？尽管模型和算力预算与 TinyStories 相同，为什么输出质量更差？

**我的回答：**

---

### Problem (leaderboard)：排行榜 (10 B200 hrs) (6 points)

在上述排行榜规则下（运行时间：B200 上最多 45 分钟；数据：只能使用提供的 OpenWebText 训练数据集）训练模型，目标是在 0.75 B200-hours 内最小化语言模型的验证 loss。

交付物：记录到的最终验证 loss、清楚显示 wall-clock time 横轴小于 45 分钟的 learning curve，以及你做了什么的描述。排行榜提交至少应超过 5.0 的 naive baseline。

**我的回答：**
