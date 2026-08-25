#!/usr/bin/env python3
"""g4c_cells.py — v5 语料按 section 切块供分块 FP 捕获(2026-07-28)。
锚 RAM=NTOK×~4MB → 4699 tok 一次建锚=19G 爆红线 → 按 section 边界切 ≤CHUNK_TOK 块。
产出: /tmp/g4c_chunks/chunk_NN.ids(每行一 token id) + /tmp/g4c_cells.tsv
      (section  chunk  global_start  global_end; global=跨块累积行号=拼接 X 行号)。
边界代价: 块首 section 失去跨块上文(块界=section 界, v5 格间本就独立)。
用法(M1): python3 g4c_cells.py [corpus_txt] [chunk_tok=620]
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "corpus", "calib_prog_v5.txt")
CHUNK_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 620
OUT = "/tmp/g4c_chunks"

from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base/tokenizer.json")
enc = lambda t: tok.encode(t, add_special_tokens=False).ids

# section 解析: "// ==== name ====" 开头直到下一 header
txt = open(CORPUS).read()
parts = re.split(r"(?m)^(// ==== \S+ ====)$", txt)
secs = []                                            # [(name, text_with_header)]
for i in range(1, len(parts), 2):
    name = parts[i].split()[2]
    secs.append((name, parts[i] + parts[i+1]))
assert len(secs) >= 28, f"section 数 {len(secs)} 异常"

os.makedirs(OUT, exist_ok=True)
tsv, chunk_id, g = [], 0, 0
cur = []                                             # [(name, text)]
def flush():
    global chunk_id, g, cur
    if not cur: return
    ids, pos = [], []
    body = ""
    for name, t in cur:
        body += t
        n = len(enc(body))                           # 增量前缀 → 精确边界
        pos.append((name, n))
    ids = enc(body)
    with open(f"{OUT}/chunk_{chunk_id:02d}.ids", "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")
    prev = 0
    for name, n in pos:
        tsv.append(f"{name}\t{chunk_id}\t{g+prev}\t{g+n}")
        prev = n
    g += len(ids); chunk_id += 1; cur = []

acc = 0
for name, t in secs:
    n = len(enc(t))
    if cur and acc + n > CHUNK_TOK: flush(); acc = 0
    cur.append((name, t)); acc += n
flush()

open("/tmp/g4c_cells.tsv", "w").write("\n".join(tsv) + "\n")
print(f"chunks={chunk_id} total_tok={g} sections={len(secs)} → {OUT}/ + /tmp/g4c_cells.tsv", file=sys.stderr)
