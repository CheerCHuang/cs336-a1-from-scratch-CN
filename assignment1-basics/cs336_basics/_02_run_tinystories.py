# 问题 (train_bpe_tinystories)：在 TinyStories 数据集上训练字节级 BPE 分词器（2 分）

# (a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories 的 <|endoftext|> 特殊标记添加到词汇表中。将生成的词汇表和合并规则序列化到磁盘以便进一步检查。训练花费了多少时间和内存？词汇表中最长的标记是什么？这合理吗？

# 资源要求：≤30 分钟（无需 GPU），≤30 GB 内存

# 提示：通过在预分词时使用多进程以及以下两个事实，你应该能够将 BPE 训练时间控制在 2 分钟以内：
# (a) <|endoftext|> 标记用于界定数据文件中的文档。
# (b) <|endoftext|> 标记在应用 BPE 合并之前作为特殊情况处理。

# 交付成果：一到两句话的回答。

# (b) 对你的代码进行性能分析。分词器训练过程中哪个部分耗时最长？

# 交付成果：一到两句话的回答。


import cProfile
import pickle
import pstats
import resource
import time

from cs336_basics.train_bpe import train_bpe

INPUT = "data/TinyStoriesV2-GPT4-train.txt"   # ← 最后全量时用
VOCAB_SIZE = 10000
SPECIAL = ["<|endoftext|>"]

t0 = time.perf_counter()
profiler = cProfile.Profile()
profiler.enable()
vocab, merges = train_bpe(INPUT, VOCAB_SIZE, SPECIAL)
profiler.disable()
t1 = time.perf_counter()

# 峰值内存（macOS 上 ru_maxrss 单位是字节）
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

# 最长 token
longest = max(vocab.values(), key=len)

# 序列化到磁盘
with open("tinystories_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("tinystories_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

print("用时(秒):", round(t1 - t0, 1))
print("峰值内存(GB):", round(peak_gb, 2))
print("最长 token:", longest, "共", len(longest), "字节")
print("merge 次数:", len(merges))

# (b) profile：按累计耗时排序，前 15 个函数
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(15)
