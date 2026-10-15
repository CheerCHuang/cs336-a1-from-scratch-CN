from cs336_basics._04_tokenizer import Tokenizer
import random
import time
import numpy as np

random.seed(42)
ts_tokenizer = Tokenizer.from_files("tinystories_vocab.pkl", "tinystories_merges.pkl",["<|endoftext|>"])

owt_tokenizer = Tokenizer.from_files("owt_vocab.pkl", "owt_merges.pkl",["<|endoftext|>"])

print(len(ts_tokenizer.vocab), len(owt_tokenizer.vocab))



def sample_documents(filepath, k):
    with open(filepath, "r") as f:
        documents = f.read().split("<|endoftext|>")
        text_list = []
        for document in documents:
            if document.strip() != "":
                text_list.append(document)
        documents = text_list
    return random.sample(documents, k)

ts_documents = sample_documents("data/TinyStoriesV2-GPT4-valid.txt", 10)


owt_documents = sample_documents("data/owt_valid.txt", 10)


# print(len(ts_documents), len(owt_documents))

# print(repr(ts_documents[0][:200]))


# 压缩比：这些文档的总字节数 ÷ 编码出的总 token 数
def compression_ratio(tokenizer, documents):
    total_tokens = 0
    byte_num = 0
    for document in documents:
        byte_num += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))
    return byte_num/total_tokens
#题目a看压缩比
print(compression_ratio(ts_tokenizer, ts_documents), compression_ratio(owt_tokenizer, owt_documents))
#题目b不类型数据压缩比
print(compression_ratio(owt_tokenizer, ts_documents), compression_ratio(ts_tokenizer, owt_documents))

#题目c估算时间
with open("data/owt_valid.txt", "r") as f:
    documents_clip = f.read()[:3000000]

t1 = time.perf_counter()
owtid_list = owt_tokenizer.encode(documents_clip)
t2 = time.perf_counter()
print("编码耗时(秒):",t2 - t1)

owt_byte_num = len(documents_clip.encode("utf-8"))

print("吞吐(bytes/秒):",owt_byte_num/(t2-t1))

pile_bytes = 825 * 1024 * 1024 * 1024
print("目标文档所需天数:",pile_bytes/(owt_byte_num/(t2-t1))/3600/24)


#题目d存词表
def encode_file(tokenizer, input_path, output_path):
    ids = []
    with open(input_path, "r") as f, open(output_path, "wb") as out:
        # 这个缩进块里：f 能读，out 能写，两个文件都开着
        for id in tokenizer.encode_iterable(f):
            ids.append(id)
            if len(ids) == 1000000:
                np.array(ids, dtype=np.uint16).tofile(out)
                ids = []
        if len(ids) > 0:
            np.array(ids, dtype=np.uint16).tofile(out)

# encode_file(ts_tokenizer, "data/TinyStoriesV2-GPT4-valid.txt", "ts_valid.bin")
# arr = np.memmap("ts_valid.bin", dtype=np.uint16, mode="r")
# print(len(arr), arr.dtype)
# print(repr(ts_tokenizer.decode(list(arr[:100]))))
encode_file(owt_tokenizer, "data/owt_valid.txt", "owt_valid.bin")
# encode_file(ts_tokenizer, "data/TinyStoriesV2-GPT4-train.txt", "ts_train.bin")
encode_file(owt_tokenizer, "data/owt_train.txt", "owt_train.bin")