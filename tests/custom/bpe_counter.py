from collections import defaultdict, Counter
import regex as re
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def _merge_token_optimized(token_tuple, merge_pair):
    """优化的token合并函数"""
    merge_byte1, merge_byte2 = merge_pair
    result = []
    i = 0
    
    while i < len(token_tuple):
        if (i < len(token_tuple) - 1 and 
            token_tuple[i] == merge_byte1 and 
            token_tuple[i + 1] == merge_byte2):
            # 合并两个相邻的bytes
            merged_bytes = token_tuple[i] + token_tuple[i + 1]
            result.append(merged_bytes)
            i += 2
        else:
            result.append(token_tuple[i])
            i += 1
    
    return tuple(result)

class BPECounter:
    """高效的字节级 BPE 计数 / 训练器。

    为了与 GPT-2 的参考实现及测试保持一致，我们在内部完全使用 *bytes* 作为 token
    单元，而不是 Python 字符串。这避免了字符串/字节之间的类型不匹配问题，并且能让
    最终输出的 `vocab` 和 `merges` 与测试夹中的参考文件对齐。

    参数
    ------
    tokens: list[list[bytes]]
        语料经过预分词（pre-tokenization）后得到的 token 序列，每个 token 再被拆分为
        字节序列。例如单词 b"hello" 会被表示为

        [b"h", b"e", b"l", b"l", b"o"].

    vocab_size: int
        目标词表大小（包含特殊 token 与 256 个单字节 token）。

    special_tokens: list[str]
        需要保留且禁止被进一步合并的特殊 token（形如 "<|endoftext|>"）。这些 token
        会直接以其 UTF-8 编码整体加入词表。
    """

    def __init__(self, tokens: list[list[bytes]], vocab_size: int, special_tokens: list[str]):
        self.word_freq_counter: dict[tuple[bytes, ...], int] = {}
        self.max_vocab_size = vocab_size
        
        # 优化的索引结构
        self.pair_freq: Counter[tuple[bytes, bytes]] = Counter()
        self.pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
        
        self._init_count(tokens)
        self._init_vocab(special_tokens)
        self._build_indexes()
        self.merges: list[tuple[bytes, bytes]] = []

        # 记录特殊 token 相关的前缀，用于后续合并时的过滤。
        # 目前只需要阻止出现形如 b"<|" 的前缀（但允许完整特殊 token 本身）。
        self._special_tokens_bytes = {tok.encode("utf-8") for tok in special_tokens}
        self._forbidden_prefixes = {tok_bytes[:2] for tok_bytes in self._special_tokens_bytes if len(tok_bytes) >= 2}

    def _init_count(self, tokens: list[list[bytes]]):
        """根据预处理后的字节 token 序列统计频次。"""
        for token in tokens:
            token_tuple = tuple(token)
            if token_tuple in self.word_freq_counter:
                self.word_freq_counter[token_tuple] += 1
            else:
                self.word_freq_counter[token_tuple] = 1

    def _init_vocab(self, special_tokens: list[str]):
        """构建初始词表，包含特殊 token 与 256 个单字节 token。"""
        self.vocab: dict[int, bytes] = {}
        next_id = 0

        # 先加入特殊 token（整体作为一个 bytes 序列）
        for tok in special_tokens:
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        # 紧接着加入 0-255 单字节 token
        for i in range(256):
            self.vocab[next_id] = bytes([i])
            next_id += 1

        self.vocab_size = next_id
        assert self.vocab_size <= self.max_vocab_size, "目标词表大小小于 256+|special_tokens|"

    def _build_indexes(self):
        """构建bigram索引"""
        for word, freq in self.word_freq_counter.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_freq[pair] += freq
                if pair in self.pair_to_words:
                    self.pair_to_words[pair].add(word)
                else:
                    self.pair_to_words[pair] = {word}

    def _get_best_pair(self):
        """获取频次最高的pair，保持与原始算法相同的tie-breaking规则"""
        if not self.pair_freq:
            return None
        
        # 使用与原始算法相同的逻辑确保正确性
        best_pair = max(self.pair_freq, key=lambda x: (self.pair_freq[x], x))
        return best_pair

    def _update_word_indexes(self, old_word, new_word, freq):
        """更新单词的索引信息"""
        # 移除旧单词的pairs
        if old_word and len(old_word) > 1:
            for i in range(len(old_word) - 1):
                pair = (old_word[i], old_word[i + 1])
                self.pair_freq[pair] -= freq
                if self.pair_freq[pair] == 0:
                    # 完全移除pair
                    del self.pair_freq[pair]
                    del self.pair_to_words[pair]
                else:
                    # 只更新pair_to_words
                    self.pair_to_words[pair].discard(old_word)
        
        # 添加新单词的pairs
        if new_word and len(new_word) > 1:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                self.pair_freq[pair] += freq
                if pair in self.pair_to_words:
                    self.pair_to_words[pair].add(new_word)
                else:
                    self.pair_to_words[pair] = {new_word}

    def _update_vocab(self, pair: tuple[bytes, bytes]):
        """将新合并得到的 token 写入词表。"""
        self.vocab[self.vocab_size] = pair[0] + pair[1]
        self.vocab_size += 1

    def merge_one_step(self):
        """执行一次最频繁 pair 的合并。"""
        while True:
            pair = self._get_best_pair()
            if not pair:
                return

            merged_candidate = pair[0] + pair[1]

            # 若合并结果以任何禁止前缀开头，且并非合法的特殊 token，则跳过该 pair
            if any(merged_candidate.startswith(pref) for pref in self._forbidden_prefixes) and merged_candidate not in self._special_tokens_bytes:
                # 将该 pair 从计数中移除，避免再次选择
                del self.pair_freq[pair]
                if pair in self.pair_to_words:
                    del self.pair_to_words[pair]
                # 继续寻找下一个最佳 pair
                continue
            break

        # 找到包含这个pair的所有单词（拷贝避免迭代时修改）
        words_to_update = list(self.pair_to_words[pair])
        
        # 批量更新
        updates = []  # (old_word, new_word, freq)
        
        for word in words_to_update:
            freq = self.word_freq_counter.get(word)
            if freq is None:
                continue
                
            # 使用优化的token合并
            new_word_tuple = _merge_token_optimized(word, pair)
            if new_word_tuple != word:
                updates.append((word, new_word_tuple, freq))
        
        # 批量应用更新
        for old_word, new_word_tuple, freq in updates:
            # 更新单词频次
            del self.word_freq_counter[old_word]
            self.word_freq_counter[new_word_tuple] = self.word_freq_counter.get(new_word_tuple, 0) + freq
            
            # 更新索引
            self._update_word_indexes(old_word, new_word_tuple, freq)

        # 更新词表与合并记录
        self._update_vocab(pair)
        self.merges.append(pair)
    
    def merge(self):
        """持续合并直至词表达到目标大小。"""
        while self.vocab_size < self.max_vocab_size:
            self.merge_one_step()

        return self.vocab, self.merges


def _find_chunk_boundaries(
    file_path: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> list[int]:
    """
    将文件分块，每个块可以独立处理。
    参考pretokenization_example.py的实现。
    """
    with open(file_path, "rb") as file:
        # 获取文件总大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size <= desired_num_chunks * 1024:  # 文件太小，不分块
            return [0, file_size]

        chunk_size = file_size // desired_num_chunks
        
        # 初始边界猜测
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            
            while True:
                mini_chunk = file.read(mini_chunk_size)
                
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # 寻找换行符作为分割点（比special token更常见）
                found_at = mini_chunk.find(b"\n")
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at + 1
                    break
                initial_position += mini_chunk_size

        return sorted(set(chunk_boundaries))

def _process_text_chunk(args):
    """并行处理文本块的工作函数"""
    text_chunk, special_tokens = args
    # 构造优先匹配特殊 token 的正则：先匹配任何特殊 token，其次再匹配 GPT-2 拆分规则。
    special_regex = "|".join(re.escape(tok) for tok in special_tokens)
    PAT = rf"({special_regex})|'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"

    tokens: list[list[bytes]] = []
    special_set = set(special_tokens)

    # 直接在完整文本块上运行正则，**保留换行符等空白字符**，从而生成与 GPT-2 参考实现一致的 token
    for tok in re.findall(PAT, text_chunk):
        if tok in special_set:
            tokens.append([tok.encode("utf-8")])
        else:
            tokens.append([bytes([b]) for b in tok.encode("utf-8")])

    return tokens

def read_tokens(input_path: str | os.PathLike, special_tokens: list[str], num_workers: int = None) -> list[list[bytes]]:
    """读取文本文件并执行并行化的预分词，返回字节 token 序列。

    使用与 GPT-2 相同的正则表达式，将文本拆分为 *字符串* token，随后将每个 token
    UTF-8 编码并拆分为单字节列表。通过并行化处理提升大文件的处理速度。
    """
    if num_workers is None:
        num_workers = min(4, max(1, mp.cpu_count()))
    
    # 获取文件大小，决定是否使用并行处理
    file_size = os.path.getsize(input_path)
    if file_size < 1024 * 1024 or num_workers == 1:  # 小于1MB的文件或单进程，使用串行处理
        return _read_tokens_serial(input_path, special_tokens)
    
    try:
        # 并行处理大文件
        boundaries = _find_chunk_boundaries(input_path, num_workers)
        
        # 读取各个文本块
        text_chunks = []
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                text_chunk = chunk_bytes.decode("utf-8", errors="ignore")
                text_chunks.append((text_chunk, special_tokens))
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_text_chunk, text_chunks))
        
        # 合并结果
        all_tokens = []
        for chunk_tokens in results:
            all_tokens.extend(chunk_tokens)
        
        return all_tokens
    
    except:
        # 并行处理失败，回退到串行处理
        return _read_tokens_serial(input_path, special_tokens)

def _read_tokens_serial(input_path: str | os.PathLike, special_tokens: list[str]) -> list[list[bytes]]:
    """串行版本的token读取（原始实现）"""
    special_regex = "|".join(re.escape(tok) for tok in special_tokens)
    PAT = rf"({special_regex})|'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"

    tokens: list[list[bytes]] = []
    special_set = set(special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 直接在完整文本上运行正则，保留换行符（"\n"）等空白字符，
    # 这样与 GPT-2 参考实现保持一致，能够生成诸如 b"\n\n" 的 token
    for tok in re.findall(PAT, text):
        # 若该 token 恰好是特殊 token，则整体加入
        if tok in special_set:
            tokens.append([tok.encode("utf-8")])
        else:
            # 否则按字节拆分
            tokens.append([bytes([b]) for b in tok.encode("utf-8")])

    return tokens