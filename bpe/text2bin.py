import json
import pickle
import sys
from pathlib import Path

import numpy as np

# 当从 bpe/ 目录直接运行脚本时，补充工程根目录到 sys.path，便于导入 tests.custom.*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tests.custom.tokenizer import Tokenizer

vocab = json.load(open("bpe/bpe_results/tinystories_vocab.json"))
merges = pickle.load(open("bpe/bpe_results/tinystories_merges.pkl", "rb"))
tokenizer = Tokenizer({int(k): bytes(v) if isinstance(v, list) else v.encode("utf-8") for k, v in vocab.items()},
                       [(bytes(a), bytes(b)) for a, b in merges],
                       ["<|endoftext|>"])

def text_to_bin(txt_path, out_path):
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(tokenizer.encode(line))
            ids.extend(tokenizer.encode("<|endoftext|>"))
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(out_path)

text_to_bin("data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-train.bin")
text_to_bin("data/TinyStoriesV2-GPT4-valid.txt", "data/TinyStoriesV2-GPT4-valid.bin")
