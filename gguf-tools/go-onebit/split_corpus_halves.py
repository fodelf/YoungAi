#!/usr/bin/env python3
"""split_corpus_halves.py — 把一份开源语料按交替块真正切成两半(2026-08-21 用户令
"真正的切一半, 一半量化全量q2, 一半做放大器, 两边都必须是全场景且原始语料里就有代码")。

切法 = 沿文件顺序切成 2N 个等长块(按行边界对齐), 偶数块 -> even 半, 奇数块 -> odd 半。
交替保证两半各自沿整份语料均匀铺开, 场景/语种/代码比例与原始语料一致, 且零重叠。

产物:
  <out>/v5_even.txt          偶半原文(给 imatrix 收集 / 量化)
  <out>/v5_odd.txt           奇半原文(给放大器锚)
  <out>/v5_even_imatrix.txt  偶半渲染成 DS4 imatrix 数据集(===== DS4_IMATRIX_PROMPT 分隔 + BOS)

用法: python3 split_corpus_halves.py <corpus.txt> <outdir> [块字符数=4000]
"""
import os, sys

BOS = "<｜begin▁of▁sentence｜>"

def main():
    src, out = sys.argv[1], sys.argv[2]
    chunk_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    os.makedirs(out, exist_ok=True)
    lines = open(src, encoding="utf-8", errors="replace").read().splitlines(keepends=True)

    # 按行边界攒块, 每块 >= chunk_chars 即封块
    chunks, buf, n = [], [], 0
    for ln in lines:
        buf.append(ln); n += len(ln)
        if n >= chunk_chars:
            chunks.append("".join(buf)); buf, n = [], 0
    if buf: chunks.append("".join(buf))

    even = [c for i, c in enumerate(chunks) if i % 2 == 0]
    odd  = [c for i, c in enumerate(chunks) if i % 2 == 1]

    open(os.path.join(out, "v5_even.txt"), "w", encoding="utf-8").write("".join(even))
    open(os.path.join(out, "v5_odd.txt"),  "w", encoding="utf-8").write("".join(odd))

    # imatrix 数据集: 每块一条 prompt, base 续写口径(裸 BOS, 不套 chat 模板)
    with open(os.path.join(out, "v5_even_imatrix.txt"), "w", encoding="utf-8") as f:
        for i, c in enumerate(even):
            f.write("===== DS4_IMATRIX_PROMPT v5even-%04d raw =====\n" % i)
            f.write(BOS + c.strip() + "\n\n")

    print("块数=%d (each %d chars) even=%d块/%d字符 odd=%d块/%d字符" % (
        len(chunks), chunk_chars, len(even), sum(map(len, even)), len(odd), sum(map(len, odd))))

main()
