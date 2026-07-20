#!/usr/bin/env python3
"""build_go_trie.py — Go 语料 → token 级 n-gram trie 词典 (ds4 无专家投机 drafter 的离线产物).

用途: ds4 单机 greedy 投机解码的第二提议源 (DS4_GO_TRIE / --go-trie). 纯语料统计,
零训练: 统计语料里长度 ≤ depth 的所有 token n-gram 计数, 剪掉低频, 铺成紧凑二进制
trie. 运行时在生成上下文尾部做最深后缀匹配, 沿最高频子链一次提议 3-8 token, 由主
模型批验证 (greedy 无损, 提议错只浪费一个 batch, 免费门拦掉大多数错误提议).

用法:
  /tmp/go_venv/bin/python build_go_trie.py \
      --corpus ../gocorpus_big.txt --out /path/go_trie.bin \
      [--depth 8] [--min-count 2] [--tokenizer <HF>/tokenizer.json] [--stats]

二进制格式 v1 (little-endian, 32B header + 16B/node):
  0  char[4]  magic   "GTRI"
  4  u32      version 1
  8  u32      depth        (最长 n-gram 长度; 上下文最深 depth-1)
  12 u32      min_count    (构建期剪枝阈值, 仅记录)
  16 u32      n_nodes
  20 u32      root_off     (根的子块起点: nodes[root_off .. root_off+root_n))
  24 u32      root_n
  28 u32      n_vocab_hint (tokenizer 词表大小, 加载端 sanity)
  32 node[n_nodes]: { i32 token; u32 count; u32 child_off; u32 child_n }
每个子块内按 token id 升序 (运行时二分). count = 该 n-gram 在语料中的出现次数;
child.count 之和 ≤ node.count (尾部截断 + 剪枝), 置信度 = child.count/node.count 保守成立.
"""
import argparse
import os
import struct
import sys
from collections import Counter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEF_TOKENIZER = os.path.join(
    os.environ.get("DS4_HF", os.path.join(_REPO, "hf", "DeepSeek-V4-Flash-Base")),
    "tokenizer.json")

MAGIC = b"GTRI"
VERSION = 1


def tokenize(paths, tok_path):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    ids = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        enc = tk.encode(text, add_special_tokens=False)
        ids.extend(enc.ids)
        print(f"  corpus {p}: {len(text)} bytes -> {len(enc.ids)} tokens")
    return ids, tk.get_vocab_size(), tk


def count_ngrams(ids, depth, min_count):
    """逐深度 Apriori 计数: d+1 gram 只在其 d 前缀存活时计数, 峰值内存小."""
    by_depth = []  # by_depth[d-1] = {tuple: count}
    prev = None
    n = len(ids)
    for d in range(1, depth + 1):
        c = Counter()
        if d == 1:
            for t in ids:
                c[(t,)] += 1
        else:
            for i in range(n - d + 1):
                if tuple(ids[i:i + d - 1]) in prev:
                    c[tuple(ids[i:i + d])] += 1
        # 剪枝: count < min_count 的分支不可能有更频繁的后代 (count 单调不增)
        c = {g: k for g, k in c.items() if k >= min_count}
        by_depth.append(c)
        print(f"  depth {d}: {len(c)} surviving n-grams (min_count={min_count})")
        if not c:
            break
        prev = c
    return [lvl for lvl in by_depth if lvl]


def build_nodes(by_depth):
    """BFS 铺平: 每个节点的子块连续且按 token id 升序 (运行时二分)."""
    # 每层预索引 prefix -> [child tokens], 避免每个节点扫全层
    child_idx = [None]
    for grams in by_depth[1:]:
        m = {}
        for g in grams:
            m.setdefault(g[:-1], []).append(g[-1])
        child_idx.append(m)

    nodes = []      # [token, count, child_off, child_n]
    node_of = {}
    order1 = sorted(by_depth[0].items(), key=lambda kv: kv[0][0])
    for g, cnt in order1:
        node_of[g] = len(nodes)
        nodes.append([g[0], cnt, 0, 0])
    queue = [g for g, _ in order1]
    qi = 0
    while qi < len(queue):
        g = queue[qi]
        qi += 1
        d = len(g)
        if d >= len(by_depth):
            continue
        toks = sorted(child_idx[d].get(g, []))
        if not toks:
            continue
        ni = node_of[g]
        nodes[ni][2] = len(nodes)
        nodes[ni][3] = len(toks)
        lvl = by_depth[d]
        for tok in toks:
            child = g + (tok,)
            node_of[child] = len(nodes)
            nodes.append([tok, lvl[child], 0, 0])
            queue.append(child)
    return nodes, 0, len(order1)


def write_trie(path, nodes, root_off, root_n, depth, min_count, n_vocab):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<7I", VERSION, depth, min_count,
                            len(nodes), root_off, root_n, n_vocab))
        buf = bytearray()
        for tok, cnt, off, cn in nodes:
            buf += struct.pack("<iIII", tok, min(cnt, 0xFFFFFFFF), off, cn)
        f.write(buf)
    return 32 + len(nodes) * 16


def show_stats(nodes, root_off, root_n, tk, top=15):
    """打印最高频深链 (肉眼核对是不是 Go 样板)."""
    print("== top deep chains (greedy max-count walk from top roots) ==")
    roots = sorted(range(root_off, root_off + root_n),
                   key=lambda i: -nodes[i][1])[:top]
    for r in roots:
        chain = [nodes[r][0]]
        cur = r
        while nodes[cur][3] > 0:
            off, cn = nodes[cur][2], nodes[cur][3]
            best = max(range(off, off + cn), key=lambda i: nodes[i][1])
            chain.append(nodes[best][0])
            cur = best
        print(f"  root_count={nodes[r][1]:>7} len={len(chain)} {tk.decode(chain)!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", action="append", required=True,
                    help="语料文本文件 (可重复)")
    ap.add_argument("--out", required=True, help="输出 trie 二进制路径")
    ap.add_argument("--depth", type=int, default=8,
                    help="最长 n-gram 长度 (默认 8; 上下文最深 depth-1)")
    ap.add_argument("--min-count", type=int, default=2,
                    help="剪枝阈值: 出现次数 < 此值的 n-gram 丢弃 (默认 2)")
    ap.add_argument("--tokenizer", default=DEF_TOKENIZER,
                    help=f"tokenizer.json 路径 (默认 {DEF_TOKENIZER})")
    ap.add_argument("--stats", action="store_true", help="打印高频链样例")
    args = ap.parse_args()

    if args.depth < 2 or args.depth > 16:
        sys.exit("depth must be in [2,16]")
    print(f"tokenizer: {args.tokenizer}")
    ids, n_vocab, tk = tokenize(args.corpus, args.tokenizer)
    print(f"total tokens: {len(ids)}  vocab: {n_vocab}")

    by_depth = count_ngrams(ids, args.depth, args.min_count)
    if not by_depth:
        sys.exit("empty corpus / everything pruned")
    nodes, root_off, root_n = build_nodes(by_depth)
    size = write_trie(args.out, nodes, root_off, root_n,
                      args.depth, args.min_count, n_vocab)
    print(f"wrote {args.out}: {len(nodes)} nodes, {size / 1e6:.2f} MB "
          f"(depth={args.depth} min_count={args.min_count})")
    if args.stats:
        show_stats(nodes, root_off, root_n, tk)


if __name__ == "__main__":
    main()
