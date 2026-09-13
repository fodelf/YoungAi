#!/usr/bin/env python3
"""v41_engram_consts.py — 把 V4.1 engram 哈希的【常量】从 tokenizer 算出来落一个二进制, 给 v41_to_gguf 嵌进 GGUF。

为什么用 Python 算这一份: 常量来自 tokenizer 的文本归一化(NFKC/NFD/去重音/小写/空白折叠, 官方
engram.py build_compressed_token_map)与 numpy 的 PCG64 随机乘子 —— 是字符串处理和 RNG 复现, 不是
数值链, 与 vocab/merges 同类; 在 C 里重写 Unicode 归一化只会引入"不报错只出错"的差异。算一次进文件,
引擎永远只读文件, 与官方实现逐位同源(sympy 素数 / numpy 乘子 / tokenizers 归一化全是官方同一份代码路径)。

输出格式(全 little-endian):
  [u32 'EGRC'][u32 ver=1][u32 vocab][u32 compressed_vocab][u32 n_layer][u32 max_ngram][u32 n_heads][u32 pad_id_compressed]
  [i32 token_map[vocab]]
  [i64 multipliers[n_layer][max_ngram]]
  [i64 primes[n_layer][max_ngram-1][n_heads]]
  [i64 offsets[n_layer][(max_ngram-1)*n_heads]]
用法: v41_engram_consts.py <hf-dir> <out.bin>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

hf, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(hf / "inference"))
from engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers  # noqa: E402
from model import ModelArgs  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

cfg = json.loads((hf / "inference" / "config.json").read_text())
args = ModelArgs(**{k: v for k, v in cfg.items() if k in ModelArgs.__dataclass_fields__})
tok = AutoTokenizer.from_pretrained(str(hf), trust_remote_code=True)
layout = EngramLayout.from_args(args)
token_map, cvocab = build_compressed_token_map(tok)
assert cvocab == args.engram_compressed_vocab_size, (cvocab, args.engram_compressed_vocab_size)
mult = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, cvocab).numpy().astype(np.int64)
primes = np.array(layout.primes, dtype=np.int64)                      # [L][ngram-1][heads]
flat = primes.reshape(len(layout.layer_ids), -1)
offsets = np.stack([np.cumsum(np.concatenate([[0], row[:-1]])) for row in flat]).astype(np.int64)
pad_id = token_map[args.engram_pad_id]
with out.open("wb") as f:
    f.write(b"EGRC")
    f.write(struct.pack("<7I", 1, len(token_map), cvocab, len(layout.layer_ids), layout.max_ngram_size, layout.n_heads, pad_id))
    np.asarray(token_map, dtype=np.int32).tofile(f)
    mult.tofile(f); primes.tofile(f); offsets.tofile(f)
print(f"engram 常量 → {out}: vocab {len(token_map)} 压缩词表 {cvocab} 层 {list(layout.layer_ids)} "
      f"n-gram {layout.max_ngram_size} heads {layout.n_heads} pad {args.engram_pad_id}→{pad_id}; "
      f"乘子 {mult.tolist()}; 素数首行 {primes[0, 0, :3].tolist()}…; 偏移末 {offsets[:, -1].tolist()}")
