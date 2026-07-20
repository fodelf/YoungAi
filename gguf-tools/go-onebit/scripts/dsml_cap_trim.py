#!/usr/bin/env python3
"""dsml_cap_trim.py — 捕获文件截断到各 kind 一致行数 (P2 capture 后置修复)。

背景: capture 段间收割用 pkill 杀 worker, SIGTERM 不走 atexit → cap 句柄
stdio 缓冲尾巴 (~2.3KB/文件) 丢失, 五类文件行数不齐 (route 最痛: 12B/行)。
追加顺序固定 (ffn_in→ffn_out→logits→w→route 逐 chunk) → 所有文件都是
对齐前缀, 截到公共最短行数即恢复行↔行对齐; 丢失 ~1.3% 尾部行, 校准无感。

用法: python3 dsml_cap_trim.py CAPDIR LAYERS(lo-hi)   # 就地截断 seg0..3
"""
import os
import sys

STRIDE = {"raw_ffn_in": 8192, "raw_ffn_out": 8192,
          "raw_route_logits": 512, "raw_route_w": 12, "raw_route": 12}


def main():
    cap, layers = sys.argv[1], sys.argv[2]
    lo, hi = map(int, layers.split("-"))
    for i in range(4):
        for L in range(lo, hi + 1):
            paths = {k: f"{cap}/seg{i}/{k}_L{L}" for k in STRIDE}
            if not all(os.path.isfile(p) for p in paths.values()):
                print(f"seg{i} L{L}: 缺文件, 跳过"); continue
            n = min(os.path.getsize(p) // STRIDE[k] for k, p in paths.items())
            for k, p in paths.items():
                want = n * STRIDE[k]
                have = os.path.getsize(p)
                if have != want:
                    with open(p, "r+b") as f:
                        f.truncate(want)
            print(f"seg{i} L{L}: -> {n} 行 (各 kind 就地对齐)")


if __name__ == "__main__":
    main()
