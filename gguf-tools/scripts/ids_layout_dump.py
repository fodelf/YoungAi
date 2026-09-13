#!/usr/bin/env python3
"""ids_layout_dump.py — 把一份 ids 切片按 .layout 还原成文字, 看【内容分布】(2026-09-12)。

stage_idshalf 切出的 8192 token 是"域连续铺 + 域内 128 宽等距窗", 光看 ids 数字看不出抽到了
什么; 这里按 .layout 逐窗解码, 每窗打一行: 域 / 窗号 / 字类占比 / 开头文字。只做显示, 不参与
任何数值链(tokenizer 解码是展示用, 不是判决口径)。

用法: ids_layout_dump.py <hf-dir> <x.ids> [每窗显示字符数=60] [--full 整窗全文]
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer


def cls(s):
    """字类占比: 中日韩 / 数字 / ASCII 字母 / 标点空白其他"""
    n = max(len(s), 1)
    cjk = sum(1 for c in s if 0x4E00 <= ord(c) <= 0x9FFF)
    dig = sum(1 for c in s if c.isdigit())
    asc = sum(1 for c in s if c.isascii() and c.isalpha())
    return cjk / n, dig / n, asc / n


def main():
    hf, ids_path = sys.argv[1], Path(sys.argv[2])
    head = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 60
    full = "--full" in sys.argv
    tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
    ids = [int(x) for x in ids_path.read_text().split()]
    win, layout = 128, []
    for ln in (ids_path.parent / (ids_path.name + ".layout")).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        if p[0] == "win":
            win = int(p[1])
        else:
            layout.append((p[0], int(p[1]), int(p[2])))
    print(f"{ids_path.name}: {len(ids)} token, 窗宽 {win}, {len(layout)} 域")
    tot = [0.0, 0.0, 0.0, 0]
    for dom, off, n in layout:
        seg = ids[off:off + n]
        txt = tok.decode(seg)
        c, d, a = cls(txt)
        tot[0] += c * len(txt); tot[1] += d * len(txt); tot[2] += a * len(txt); tot[3] += len(txt)
        print(f"\n== {dom}: {n} token = {n // win} 窗, 解码 {len(txt)} 字, 汉字 {c:.0%} 数字 {d:.0%} 英文 {a:.0%}")
        for k in range(n // win):
            w = tok.decode(seg[k * win:(k + 1) * win]).replace("\n", "⏎")
            c, d, a = cls(w)
            body = w if full else (w[:head] + ("…" if len(w) > head else ""))
            print(f"  [{k:02d}] 汉{c:.0%} 数{d:.0%} 英{a:.0%} | {body}")
    if tot[3]:
        print(f"\n全份: 汉字 {tot[0] / tot[3]:.0%} 数字 {tot[1] / tot[3]:.0%} 英文 {tot[2] / tot[3]:.0%}, 解码 {tot[3]} 字 / {len(ids)} token = {tot[3] / len(ids):.2f} 字/token")


if __name__ == "__main__":
    main()
