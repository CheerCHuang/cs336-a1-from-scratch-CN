# 问题（train_bpe）：BPE分词器训练（15分）
# 提交内容：编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE分词器。你的BPE训练函数应至少处理以下输入参数：
# 输入
# input_path: str 包含BPE分词器训练数据的文本文件路径。
# vocab_size: int 一个正整数，定义最终词汇表的最大大小（包括初始字节词汇表、通过合并产生的词汇项以及任何特殊标记）。
# special_tokens: list[str] 要添加到词汇表中的字符串列表。在训练期间，将它们视为硬边界，防止跨其跨度的合并，但在计算合并统计信息时不要包含它们。
# 你的BPE训练函数应返回生成的词汇表和合并操作：
# 输出
# vocab: dict[int, bytes] 分词器词汇表，从整数（词汇表中的标记ID）到字节（标记字节）的映射。
# merges: list[tuple[bytes, bytes]] 训练产生的BPE合并操作列表。每个列表项是一个字节元组（<token1>, <token2>），表示<token1>与<token2>合并。合并应按创建顺序排序。 
# 为了使用我们提供的测试来测试你的BPE训练函数，你首先需要在[adapters.run_train_bpe]处实现测试适配器。然后运行uv run pytest tests/test_train_bpe.py。
# 你的实现应该能够通过所有测试。可选地（这可能需要大量时间投入），你可以使用某种系统语言（例如C++（考虑cppyy或nanobind）或Rust（使用PyO3））来实现训练方法的关键部分。
# 如果你这样做，请注意哪些操作需要从Python内存复制而不是直接读取，并确保留下构建说明，或确保仅使用pyproject.toml即可构建。
# 另请注意，大多数正则表达式引擎对GPT-2正则表达式的支持不佳，并且在大多数支持它的引擎中速度会太慢。
# 我们已经验证Oniguruma速度相当快并支持负向前瞻，但Python中的regex包实际上甚至更快。
import regex as re



def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")   
    #训练词库
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        docs = re.split(pattern, text)
    else:
        docs = [text]

    # GPT-2 的预分词规则
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    word_tuple_counts = {}
    memo = {}
    for doc in docs:
        for m in PAT.finditer(doc): # 文档内逐个 pre-token
            per_token =m.group()    # 拿到文本，如 "Once"、","、" the"
            if per_token in memo:
                tuple_word = memo[per_token]
            else:
                utf8_token = per_token.encode("utf-8")
                list_word = []
                for i in range(len(utf8_token)):
                    list_word.append(utf8_token[i:i+1])
                tuple_word = tuple(list_word)
                memo[per_token] = tuple_word
            word_tuple_counts[tuple_word] = word_tuple_counts.get(tuple_word,0) + 1


    # 预分词后: 把语料变成字节元组，然后统计字节元组(代表一个单词)的出现频次组成字典 word_tuple_counts
    # word_tuple_counts = {} # key是字节元组，value是出现频次 如{(b'l', b'o', b'w'): 5, (b'l', b'o', b'w', b'e', b'r'): 2, (b'w', b'i', b'd', b'e', b's', b't'): 3, (b'n', b'e', b'w', b'e', b's', b't'): 6}
    # for name, count in testdict.items():
    #     utf8_word = name.encode("utf-8")
    #     list_word = []
    #     for i in range(len(utf8_word)):
    #         list_word.append(utf8_word[i:i+1])
    #     tuple_word = tuple(list_word)
    #     word_tuple_counts[tuple_word] = count
    # print(word_tuple_counts)



    # ① 统计字节对。需求：从N对里选出频次最高的；平局时选字典序更大的。首次需要完整遍历，后续只需微调
    byte_pairs_counts = {}  # 已有每个单词的字节元组，逐单词、每个单词再两两成对统计算出所有相邻字节对的出现频次。 byte_pairs_counts
    pair_to_words = {} #每个字节对被哪些单词包含，方便用于更新字节对表

    for name, count in word_tuple_counts.items():
        for i in range(len(name)-1):
            pair = (name[i],name[i+1])
            if pair not in byte_pairs_counts:
                byte_pairs_counts[pair] = count
                pair_to_words[pair] = set([name])
            else:
                byte_pairs_counts[pair] += count
                pair_to_words[pair].add(name)

    
    merges = []

    # 循环训练
    while len(vocab) < vocab_size:

        if len(merges) % 50 == 0:
            print(f"第 {len(merges)} 轮 merge")
        # ① 统计字节对。需求：从N对里选出频次最高的；平局时选字典序更大的

        if len(byte_pairs_counts) == 0:
            break
        
        # ② 选最大 + 记录 + 登记词表
        best_pair = max(byte_pairs_counts, key=lambda p: (byte_pairs_counts[p],p))
        # print(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token


        # ③ 找到本次合并影响的word，在字节对中抛去对应计数；重新构成新的word并重计算字节对计数并补回。对第一版优化了一下。使其每次只对受影响word更新
        affected = list(pair_to_words[best_pair]) 
        for word in affected:
            count = word_tuple_counts.pop(word)
            for i in range(len(word)-1):
                pair = (word[i],word[i+1])
                left = byte_pairs_counts[pair] - count
                if left <= 0:
                    del byte_pairs_counts[pair]
                    del pair_to_words[pair]
                else:
                    byte_pairs_counts[pair] = left
                    pair_to_words[pair].discard(word)

            #重新构成新的word并
            new_word_list = []
            i = 0
            while i < len(word):
                if i< len(word) -1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word_list.append(new_token)
                    i += 2
                    continue
                new_word_list.append(word[i])
                i += 1
            new_word = tuple(new_word_list)

            # 重计算新word字节对计数并补回
            for j in range(len(new_word)-1):
                pair2 = (new_word[j],new_word[j+1])
                byte_pairs_counts[pair2] = byte_pairs_counts.get(pair2,0) + count
                pair_to_words.setdefault(pair2, set()).add(new_word)
            word_tuple_counts[new_word] = word_tuple_counts.get(new_word,0) + count



        # ③ 合并所有新词(更新预分词表，再重新计算字节对)，更新通过新建list并得到对应字典实现
        # new_counts = {}
        # for word_tuple,count in word_tuple_counts.items():
        #     new_word_list = []
        #     i = 0
        #     while i <len(word_tuple):
        #         if i< len(word_tuple) -1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
        #             new_word_list.append(new_token)
        #             i += 2
        #             continue
        #         new_word_list.append(word_tuple[i])
        #         i += 1
        #     new_word = tuple(new_word_list)
        #     new_counts[new_word] = count
        # word_tuple_counts = new_counts


        merges.append(best_pair)
 

    return vocab, merges


