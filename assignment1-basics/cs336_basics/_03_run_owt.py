"""Problem (train_bpe_expts_owt): 在 OpenWebText 上训练 BPE (2 points)

跑法：cd 到 assignment1-basics 根目录，uv run python cs336_basics/03_run_owt.py
"""
import pickle
import resource
import time

from cs336_basics._01_train_bpe import train_bpe

INPUT = "data/owt_bpe_train.txt"
VOCAB_SIZE = 32000
# ⚠️ 跑之前先打开 data/owt_train.txt 看开头几十行，确认文档之间的分隔符！
# OWT 是网页文本，分隔符可能不是 <|endoftext|>（比如双换行 \n\n）。
# special_tokens 传错 → 切分失效 → 文档边界处发生合并 → 结果错误。
SPECIAL = ["<|endoftext|>"]

t0 = time.perf_counter()
vocab, merges = train_bpe(INPUT, VOCAB_SIZE, SPECIAL)
t1 = time.perf_counter()

# 峰值内存（macOS 上 ru_maxrss 单位是字节）
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

# (a) 最长 token
longest = max(vocab.values(), key=len)

# (a) 序列化到磁盘（文件名区分，别覆盖 TinyStories 的结果）
with open("owt_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("owt_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

print("用时(秒):", round(t1 - t0, 1))
print("峰值内存(GB):", round(peak_gb, 2))
print("最长 token:", longest, "共", len(longest), "字节")
print("merge 次数:", len(merges))
