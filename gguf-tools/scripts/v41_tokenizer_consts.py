#!/usr/bin/env python3
"""v41_tokenizer_consts.py — 把 V4.1 tokenizer.json 的词表与合并表落成二进制, 给 v41_to_gguf 写进 GGUF。

引擎(core_bpe.c vocab_load)只认两个键: tokenizer.ggml.tokens(按 id 排的字串表, GPT-2 字节→unicode 形式,
与 tokenizer.json 的 vocab 键逐字同)与 tokenizer.ggml.merges("a b" 字串表, 顺序即 rank); 特殊 id 按字串查。
V4 量化器是从模板 GGUF 抄 KV 段的, V4.1 没有模板, 所以从自己的 tokenizer.json 生成 —— 实测它与 V4 只差
几个 added_tokens 的名字(128799 `<｜System｜>` 等), 不能抄 V4 的。
输出: [u32 'TOKC'][u32 ver=1][u32 n_tokens][u32 n_merges] 然后 n_tokens 个 [u32 len][bytes], 再 n_merges 个同形。
用法: v41_tokenizer_consts.py <hf-dir> <out.bin>
"""
import json
import struct
import sys
from pathlib import Path

hf, out = Path(sys.argv[1]), Path(sys.argv[2])
t = json.loads((hf / "tokenizer.json").read_text(encoding="utf-8"))
vocab, merges, added = t["model"]["vocab"], t["model"]["merges"], t["added_tokens"]
n = max(max(vocab.values()), max(a["id"] for a in added)) + 1
toks = [None] * n
for s, i in vocab.items():
    toks[i] = s
clash = 0
for a in added:                      # added 覆盖同 id 的 base(实测 3 个重叠, 内容相同)
    if toks[a["id"]] is not None and toks[a["id"]] != a["content"]:
        clash += 1
    toks[a["id"]] = a["content"]
missing = [i for i, s in enumerate(toks) if s is None]
assert not missing, f"id 有洞: {missing[:10]}"
assert clash == 0, f"added_tokens 与 base 同 id 不同字: {clash}"
if isinstance(merges[0], list):      # 新版 tokenizers 把 merges 存成 [a, b] 对
    merges = [f"{a} {b}" for a, b in merges]
with out.open("wb") as f:
    f.write(b"TOKC" + struct.pack("<3I", 1, n, len(merges)))
    for s in toks:
        b = s.encode("utf-8"); f.write(struct.pack("<I", len(b)) + b)
    for s in merges:
        b = s.encode("utf-8"); f.write(struct.pack("<I", len(b)) + b)
sp = {s: i for i, s in enumerate(toks)}
need = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "｜DSML｜"]
print(f"tokenizer 常量 → {out}: {n} token / {len(merges)} merges; 特殊 id " + " ".join(f"{k}={sp[k]}" for k in need))
