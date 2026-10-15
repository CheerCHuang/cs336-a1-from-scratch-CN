# 问题（tokenizer）：实现分词器（15 分）
# 交付内容：实现一个 Tokenizer 类，给定词表和 merges 列表，把文本编码成整数 ID 序列，
# 以及把整数 ID 解码回文本。还要支持用户提供的 special tokens（如果它们不在词表中则追加进词表）。
# 接口：
#   __init__(self, vocab, merges, special_tokens=None)
#   from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)   ← classmethod
#   encode(self, text: str) -> list[int]
#   encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]          ← 惰性生成器
#   decode(self, ids: list[int]) -> str
# 测试：先实现 tests/adapters.py 的 get_tokenizer，然后 uv run pytest tests/test_tokenizer.py
# 注意：decode 时可能遇到不合法的 UTF-8 字节序列 → 用 U+FFFD（Unicode replacement character）替换
import pickle
from typing import Iterable, Iterator

import regex as re


class Tokenizer:
    """字节级 BPE 分词器：由训练得到的词表与 merges 驱动，负责编码/解码。"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """用给定的词表、merges 列表和（可选的）special tokens 列表构建分词器。

        参数:
            vocab: dict[int, bytes]           token ID → token 字节
            merges: list[tuple[bytes, bytes]] 按创建顺序排列的合并规则
            special_tokens: list[str] | None  特殊 token（永不拆分，编码时作为整体映射为单个 ID）
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        #建立一个字典用于byte查询ID
        self.bytes2ID = {}
        for ID,byte in self.vocab.items():
            self.bytes2ID[byte] = ID

        # 建立bytepair的排名字典
        self.pair2rank = {}
        for i, (a, b) in enumerate(merges):
            self.pair2rank[(a, b)] = i

        # 给词表加入special_token
        for special_token in self.special_tokens:
            special_token_utf8 = special_token.encode("utf-8")
            if special_token_utf8 in self.bytes2ID:
                continue
            new_ID = len(self.vocab)
            self.vocab[new_ID] = special_token_utf8
            self.bytes2ID[special_token_utf8]= new_ID

         # GPT-2 的预分词规则
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")



    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None) -> "Tokenizer":
        """从序列化文件构建 Tokenizer（格式与 train_bpe 输出的一致，即训练时 pickle 存的那两个文件）。"""

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """把输入文本编码成 token ID 序列。"""

        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            segment_list = re.split(pattern, text)
        else:
            segment_list = [text]

        id_list = []
        for segment in segment_list:
            if segment in self.special_tokens:
                id_list.append(self.bytes2ID[segment.encode("utf-8")])
                continue
            for m in self.PAT.finditer(segment):
                word = m.group()

                utf8word = word.encode("utf-8")
                byte_list = []
                for i in range(len(utf8word)):
                    byte_list.append(utf8word[i:i+1])

                while True:
                    pair_list = []
                    for i in range(len(byte_list)-1):
                        pair_list.append((byte_list[i],byte_list[i+1]))
                    rank = float("inf")
                    pair2merge = None
                    for pair in pair_list:
                        if pair in self.pair2rank:
                            if self.pair2rank[pair]<rank:
                                rank = self.pair2rank[pair]
                                pair2merge = pair
                    if pair2merge is None:
                        break
                    new_bytes_list = []
                    i = 0
                    while i< len(byte_list):
                        if i < len(byte_list) - 1 and (byte_list[i],byte_list[i+1])== pair2merge:
                            new_bytes_list.append(pair2merge[0]+pair2merge[1])
                            i += 2
                        else:
                            new_bytes_list.append(byte_list[i])
                            i += 1
                    byte_list = new_bytes_list
                #合并完的byte_list转为ID
                for byte in byte_list:
                    id_list.append(self.bytes2ID[byte])
                    
                    
        return id_list                     


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """给定可迭代的字符串（如文件句柄），惰性地逐个产出 token ID。

        用于无法整体载入内存的大文件的省内存分词。
        """
        for text in iterable:              # 传文件句柄时，这里逐行给
            for ID in self.encode(text):    # 复用你写好的 encode，得到 ID 列表
                yield ID                      # 一个一个交出去


    def decode(self, ids: list[int]) -> str:
        """把 token ID 序列解码回文本。非法 UTF-8 字节序列用 U+FFFD 替换。"""
        text_list = []
        for id in ids:
            text_list.append(self.vocab[id])
        
        str_byte = b"".join(text_list)
        text = str_byte.decode("utf-8", errors="replace")
        return text

