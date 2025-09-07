import json
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab  # token_id -> bytes
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 创建反向词汇表：bytes -> token_id
        self.byte_to_token = {v: k for k, v in vocab.items()}

        # 添加特殊token到词汇表
        self._add_special_tokens_to_vocab()

        # 创建合并规则的字典，用于快速查找
        self.merge_dict = {}
        for i, (token1, token2) in enumerate(merges):
            self.merge_dict[(token1, token2)] = i

        # GPT-2风格的预分词正则表达式
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = json.load(f)
        with open(merges_filepath, "rb") as f:
            merges = [tuple(line.split()) for line in f.readlines()]
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer

    def _add_special_tokens_to_vocab(self) -> None:
        """添加特殊token到词汇表"""
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_token:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token[token_bytes] = new_id

    def _get_pairs(self, word: list[bytes]) -> set[tuple[bytes, bytes]]:
        """获取word中所有相邻的字节对"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe_encode_word(self, word_bytes: bytes) -> list[int]:
        """对单个word应用BPE编码"""
        # 将word转换为字节列表
        word = [bytes([b]) for b in word_bytes]

        if len(word) == 1:
            # 单字节直接查找
            return [self.byte_to_token.get(word[0], self.byte_to_token.get(b'\x00', 0))]

        # 应用BPE合并规则
        while len(word) > 1:
            pairs = self._get_pairs(word)

            # 找到最早的合并规则
            bigram = min(pairs, key=lambda pair: self.merge_dict.get(pair, float('inf')))

            # 如果没有找到可合并的pair，停止
            if bigram not in self.merge_dict:
                break

            # 执行合并
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    # 合并这两个token
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        # 将最终的字节序列转换为token ID
        result = []
        for token_bytes in word:
            token_id = self.byte_to_token.get(token_bytes)
            if token_id is not None:
                result.append(token_id)
            else:
                # 如果找不到，尝试拆分为单字节
                for b in token_bytes:
                    single_byte = bytes([b])
                    single_token_id = self.byte_to_token.get(single_byte, 0)
                    result.append(single_token_id)

        return result

    def encode(self, text: str) -> list[int]:
        """编码文本为token ID列表"""
        if not text:
            return []

        # 处理特殊token
        if self.special_tokens:
            # 构建特殊token的正则表达式
            special_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            # 分割文本，保留特殊token
            parts = re.split(f'({special_pattern})', text)
        else:
            parts = [text]

        result = []
        for part in parts:
            if not part:
                continue

            # 检查是否是特殊token
            if part in self.special_tokens:
                special_bytes = part.encode("utf-8")
                token_id = self.byte_to_token.get(special_bytes)
                if token_id is not None:
                    result.append(token_id)
                continue

            # 对普通文本应用预分词
            for match in self.pat.finditer(part):
                word = match.group(0)
                word_bytes = word.encode("utf-8")
                word_tokens = self._bpe_encode_word(word_bytes)
                result.extend(word_tokens)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """编码可迭代的文本"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """解码token ID列表为文本"""
        if not ids:
            return ""

        # 将token ID转换为字节序列
        byte_sequences = []
        for token_id in ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is not None:
                byte_sequences.append(token_bytes)

        # 连接所有字节序列
        combined_bytes = b"".join(byte_sequences)

        # 解码为UTF-8字符串
        try:
            return combined_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # 如果解码失败，使用错误处理
            return combined_bytes.decode("utf-8", errors="replace")