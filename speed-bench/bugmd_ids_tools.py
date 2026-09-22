#!/usr/bin/env python3
"""bugmd_ids_tools.py — bug.md §5/§6 对拍的取料与比对(纯编排/比对, 不参与任何数值计算; 2026-09-21)。

子命令:
  extract <trace> <请求序号> <out前缀>   从 ds4-server --trace 文件抽第 k 个请求: 渲染提示 → <out>.prompt.txt,
                                          提示+生成的 token id(一行一个) → <out>.ids, 生成 id 单独 → <out>.gen.ids,
                                          并打印 生成里第一处 4-gram 死循环起点(按 300 字段, ≥95% 且连三段 ≥90%)。
  cut <prompt.txt> <砍掉多少字> <out>     把渲染提示的正文尾巴砍掉 N 个字(保留末尾的 "Thought:<｜Assistant｜></think>" 帧),
                                          造一条 token 数 mod 512 落在 [1,127] 的短尾提示(G3 用; 砍多少要试, 引擎日志报 N)。
  cmp-emit <a.err> <b.err>                两份 --emit-trace 日志的 [emit] 序列: 同的位数 / 第一处不同的绝对位置与两边 id。
  cmp-topk <topk.bin> <ids> <n_prompt>    --score-topk 小文件(ETGD 格式)的 top-1 vs 真吐 id: 逐位一致率、第一处不同、
                                          按生成段每 1000 位的一致率。位置 i 的 top-1 预测的是 ids[i+1]。
  eos-scan <topk.bin> <ids> <n_prompt> [tok=1]
                                          生成段逐位找 EOS 的排名与概率: 分清"模型想停引擎没听"和"模型自己不想停"。
"""
import struct
import sys


def read_trace_request(path, k):
    t = open(path, encoding="utf-8", errors="replace").read()
    parts = t.split("===== request ")
    if k == "last": k = len(parts) - 1   # 最后一个请求(冒烟在前, 真实请求在后)
    k = int(k)
    if k >= len(parts) or k < 1:
        sys.exit("trace 里只有 %d 个请求" % (len(parts) - 1))
    return parts[k]


def loop_start_segments(text):
    c = [ch for ch in text if not ch.isspace()]
    seen, cnt, tot, segs = set(), 0, 0, []
    for i in range(max(len(c) - 3, 0)):
        g = "".join(c[i:i + 4]); tot += 1
        if g in seen: cnt += 1
        seen.add(g)
        if tot == 300: segs.append(100 * cnt // 300); cnt = tot = 0
    for si in range(len(segs)):
        if segs[si] >= 95 and all(x >= 90 for x in segs[si:si + 3]):
            return si, len(segs), segs
    return None, len(segs), segs


def cmd_extract(trace, k, out):
    r = read_trace_request(trace, k)
    prompt = r.split("--- rendered prompt ---\n", 1)[1].split("\n\n--- generated text ---", 1)[0]
    gen = r.split("--- trace: prefill done", 1)[1].split("\n", 1)[1].split("\n--- parsed message", 1)[0]
    ids_sec = r.split("--- token ids:", 1)[1]
    p_line = ids_sec.split("prompt:", 1)[1].split("\n", 1)[0].split()
    g_line = ids_sec.split("generated:", 1)[1].split("\n", 1)[0].split()
    open(out + ".prompt.txt", "w", encoding="utf-8").write(prompt)
    open(out + ".ids", "w").write("\n".join(p_line + g_line) + "\n")
    open(out + ".gen.ids", "w").write("\n".join(g_line) + "\n")
    si, nseg, segs = loop_start_segments(gen)
    print("prompt %d token, generated %d token(含 EOS 与否看末 id), 提示 %d 字, 生成 %d 字" % (len(p_line), len(g_line), len(prompt), len(gen)))
    print("死循环起于 300 字段 %s / %s (≈ 生成第 %d 字)" % (si, nseg, (si or 0) * 300 * len(gen) // max(1, sum(1 for ch in gen if not ch.isspace()))))
    print("末 5 个生成 id:", g_line[-5:])


def cmd_cut(prompt_path, ncut, out):
    p = open(prompt_path, encoding="utf-8").read()
    tail_mark = "Thought:<｜Assistant｜></think>"
    if not p.endswith(tail_mark):
        sys.exit("提示不以 %r 结尾, 不知道帧在哪" % tail_mark)
    body = p[: -len(tail_mark)]
    n = int(ncut)
    open(out, "w", encoding="utf-8").write(body[: len(body) - n] + tail_mark)
    print("砍 %d 字 → %d 字, 写到 %s" % (n, len(body) - n + len(tail_mark), out))


def emit_ids(err_path):
    ids = []
    for line in open(err_path, encoding="utf-8", errors="replace"):
        if line.startswith("[emit] "):
            _, pos, tid = line.split()
            ids.append((int(pos), int(tid)))
    return ids


def cmd_cmp_emit(a, b):
    ia, ib = emit_ids(a), emit_ids(b)
    n = min(len(ia), len(ib))
    same = 0
    for i in range(n):
        if ia[i] != ib[i]:
            print("同 %d 位; 第一处不同: 位置 %d: %d vs %d (a %d 位 / b %d 位)" % (same, ia[i][0], ia[i][1], ib[i][1], len(ia), len(ib)))
            return
        same += 1
    print("全同 %d 位 ✓ (a %d 位 / b %d 位)" % (same, len(ia), len(ib)))


def cmd_cmp_topk(topk_path, ids_path, n_prompt):
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt = int(n_prompt)
    f = open(topk_path, "rb")
    magic, K, S, V = struct.unpack("<4I", f.read(16))
    assert magic == 0x44475445, "不是 ETGD 文件"
    top1 = {}
    row = struct.Struct("<Iiff" + "%di" % K + "%df" % K)
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf)
        i = v[0]; tid = v[4:4 + K]; pr = v[4 + K:4 + 2 * K]
        j = max(range(K), key=lambda q: pr[q])
        top1[i] = (tid[j], pr[j])
    agree = 0; first = None; buckets = {}
    total = 0
    diffs = []
    for i in range(n_prompt - 1, len(ids) - 1):   # 位置 i 预测 ids[i+1]; 只看生成段(含预测第一个生成 id 的那一位)
        if i not in top1: continue
        total += 1
        b = (i - (n_prompt - 1)) // 1000
        buckets.setdefault(b, [0, 0]); buckets[b][1] += 1
        if top1[i][0] == ids[i + 1]:
            agree += 1; buckets[b][0] += 1
        else:
            diffs.append((i, top1[i][1]))
            if first is None: first = (i, ids[i + 1], top1[i][0], top1[i][1])
    print("预填路(全 40 层) top-1 vs 解码路真吐 id: 生成段 %d 位, 一致 %d (%.2f%%)" % (total, agree, 100.0 * agree / max(1, total)))
    if first: print("第一处不同: 位置 %d, 真吐 %d, 预填路 top-1 %d (p=%.3f)" % first)
    # ★分歧的"底气"分布★(2026-09-22): 两条路在平局位上翻面是浮点次序的正常代价; 真病的指纹是
    # "预填路很有把握却被解码路选了别的"。09-22 修 VQ 位平面之前 141 处分歧里这一档占 66 处(47%)。
    if diffs:
        bands = [("<0.35 近平局", 0.0, 0.35), ("0.35~0.70", 0.35, 0.70), ("★>0.70 预填路明确偏好★", 0.70, 1.01)]
        print("分歧 %d 处, 按预填路 top-1 概率分档:" % len(diffs))
        for name, lo, hi in bands:
            sel = [d for d in diffs if lo <= d[1] < hi]
            print("  %-24s %4d 处 (%.0f%%)%s" % (name, len(sel), 100.0 * len(sel) / len(diffs),
                  "  位置: " + " ".join(str(d[0]) for d in sel[:12]) if sel and hi > 1.0 else ""))
    for b in sorted(buckets):
        a, t = buckets[b]
        print("  生成第 %5d~%5d 位: %.2f%% (%d/%d)" % (b * 1000, b * 1000 + 999, 100.0 * a / t, a, t))


def cmd_eos_scan(topk_path, ids_path, n_prompt, tok="1"):
    """--score-topk(ETGD)里逐位找"模型想不想停": 生成段每一位上 EOS 排第几、概率多少。

    为什么要它(2026-09-22): "写完了却不停"有两种完全不同的病 ——
      EOS 是 top-1 却还在往下写 ⇒ 引擎没听模型的(引擎 bug);
      EOS 从头到尾排不进 top-10 ⇒ 模型自己就不想停(提示词/贪心的事, 改引擎没用)。
    这一位的文本对齐很麻烦(id→文字要 tokenizer), 所以不挑位置, 整段扫: 报 EOS 进过 top-K 的
    所有位置 + 全段 EOS 概率最高的那一位。不需要知道"报告写完是第几个 token"也能判。"""
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, tok = int(n_prompt), int(tok)
    f = open(topk_path, "rb")
    magic, K, S, V = struct.unpack("<4I", f.read(16))
    assert magic == 0x44475445, "不是 ETGD 文件"
    row = struct.Struct("<Iiff" + "%di" % K + "%df" % K)
    hits, best = [], None
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf)
        i = v[0]
        if i < n_prompt - 1: continue
        tid, pr = v[4:4 + K], v[4 + K:4 + 2 * K]
        order = sorted(range(K), key=lambda q: -pr[q])
        for rank, q in enumerate(order):
            if tid[q] == tok:
                hits.append((i, rank + 1, pr[q], tid[order[0]], pr[order[0]]))
                if best is None or pr[q] > best[2]: best = hits[-1]
                break
    print("生成段 EOS(id %d) 进过 top-%d 的位置: %d 处" % (tok, K, len(hits)))
    for i, rank, p, t1, p1 in hits[:40]:
        print("  生成第 %6d 位(绝对 %6d): EOS 排第 %d p=%.4f; top-1 是 %d p=%.4f%s" %
              (i - (n_prompt - 1), i, rank, p, t1, p1, "  ★EOS 就是 top-1★" if rank == 1 else ""))
    if len(hits) > 40: print("  …… 还有 %d 处" % (len(hits) - 40))
    if best:
        print("全段最想停的一位: 生成第 %d 位, EOS 排第 %d p=%.4f (top-1 %d p=%.4f)" %
              (best[0] - (n_prompt - 1), best[1], best[2], best[3], best[4]))
    else:
        print("★整段生成里 EOS 一次都没进过 top-%d —— 模型自己不想停, 不是引擎压着它★" % K)
    print("真吐序列末 id = %d (%s)" % (ids[-1], "EOS, 模型自己收的口" if ids[-1] == tok else "不是 EOS, 顶到上限"))


def cmd_rows(rows_path, ids_path, n_prompt, entry_pos, tokenizer_json="", window="12"):
    """anchor_metrics --row-out(位置 kld smin same ref命中 stu命中 ref_argmax stu_argmax) 的读法:
    生成段按 1000 位分桶的 老师命中率 / 学生命中率 / same top; 入口 entry_pos 前后 window 位逐行列出并解码 token 文本。"""
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, entry_pos, window = int(n_prompt), int(entry_pos), int(window)
    rows = {}
    for line in open(rows_path):
        f = line.split()
        if len(f) < 8: continue
        rows[int(f[0])] = (float(f[1]), float(f[2]), int(f[3]), int(f[4]), int(f[5]), int(f[6]), int(f[7]))
    dec = None
    if tokenizer_json:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tokenizer_json)
        dec = lambda t: tk.decode([t]).replace("\n", "⏎")
    buckets = {}
    for p, r in rows.items():
        if p < n_prompt - 1: continue
        b = (p - (n_prompt - 1)) // 1000
        a = buckets.setdefault(b, [0, 0, 0, 0, 0.0])
        a[0] += 1; a[1] += r[3] == 1; a[2] += r[4] == 1; a[3] += r[2] == 1; a[4] += r[0]
    print("生成段每 1000 位: 老师命中(FP argmax==真吐) / 学生命中 / same top / 平均 KLD")
    for b in sorted(buckets):
        n, rh, sh, st, kl = buckets[b]
        print("  %5d~%5d: 老师 %5.1f%%  学生 %5.1f%%  same %5.1f%%  KLD %.3f  (n=%d)" % (b * 1000, b * 1000 + 999, 100.0 * rh / n, 100.0 * sh / n, 100.0 * st / n, kl / n, n))
    print("入口 %d 前后 %d 位(位置 | 真吐 | 老师 argmax | 学生 argmax | kld):" % (entry_pos, window))
    for p in range(entry_pos - window, entry_pos + window + 1):
        if p not in rows or p + 1 >= len(ids): continue
        r = rows[p]; nx = ids[p + 1]
        txt = (lambda t: ("%d(%s)" % (t, dec(t))) if dec else str(t))
        print("  %6d | %-22s | %-22s | %-22s | %.3f %s" % (p, txt(nx), txt(r[5]), txt(r[6]), r[0], "" if r[5] == nx else "★老师不同"))


if __name__ == "__main__":
    cmds = {"extract": cmd_extract, "cut": cmd_cut, "cmp-emit": cmd_cmp_emit, "cmp-topk": cmd_cmp_topk,
            "eos-scan": cmd_eos_scan, "rows": cmd_rows}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__ + "  rows <rows.txt> <ids> <n_prompt> <入口位置> [tokenizer.json] [窗口=12]   --row-out 的分桶命中率 + 入口逐行解码\n")
    cmds[sys.argv[1]](*sys.argv[2:])
