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
  loopstat <gen.ids> [think.txt]          采样样本的复读环形态(按 token id, 不按字): 入环位 / 周期 / 逃出窗数 / 末段去重 / 挣扎收尾短语数。
  faith <topk.bin> <ids> <n_prompt> [入环位=0]
                                          采样序列上判"采样器忠不忠"(Σp(top-1) vs 真吐==top-1 次数, 每 4096 位一桶)与
                                          "模型想不想收口"(</think>·EOS 的 Σp 对观测抽到次数); 采样路没有逐字节门, 这是它的门。
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


def cmd_loopstat(gen_ids_path, think_txt=""):
    """采样样本的复读环形态(2026-09-30, 601069 seed 1): 每 256 位窗口取最佳周期(1~64)的逐位复读率 mean(ids[i]==ids[i-p]) →
    入环位 = 首个 ≥0.9 的窗, 逃出 = 入环后复读率 <0.5 的窗数; 再报末 8192 位去重数与前三 id 占比、每 2048 位桶的复读率曲线与桶首 512 位去重数。
    为什么按 id 不按字: 环是 3~9 个 token 的且有变体([e]/[E]/[Produce] 轮着出), 字级 4-gram 尺(loop_start_segments)判它是"无死循环"。
    给 think.txt 时再数"挣扎收尾"短语(Now finish / Halt. / Produce final answer …): 这是模型知道该停却闭合不了思考块的指纹。"""
    import re
    from collections import Counter
    ids = [int(x) for x in open(gen_ids_path).read().split()]
    W, PMAX = 256, 64
    prof = []
    for lo in range(0, len(ids) - W + 1, W):
        best = (0.0, 0)
        for p in range(1, min(PMAX, lo) + 1):
            fr = sum(1 for i in range(lo, lo + W) if ids[i] == ids[i - p]) / W
            if fr > best[0]: best = (fr, p)
        prof.append((lo, best[0], best[1]))
    locked = [w for w in prof if w[1] >= 0.9]
    entry = locked[0][0] if locked else None
    escapes = sum(1 for w in prof if entry is not None and w[0] > entry and w[1] < 0.5)
    tail = Counter(ids[-8192:]); top3 = tail.most_common(3)
    print("生成 %d 位, 特殊 token(≥128000) %d 个, 末 id %d; 锁死窗(复读率≥0.9) %d/%d, 入环位 %s, 入环后逃出窗 %d; 末 8192 位去重 %d, 前三 id 占 %.0f%%" % (
        len(ids), sum(1 for t in ids if t >= 128000), ids[-1] if ids else -1, len(locked), len(prof), entry, escapes, len(tail),
        100.0 * sum(n for _, n in top3) / max(1, min(8192, len(ids)))))
    for b0 in range(0, len(ids), 2048):
        ws = [w for w in prof if b0 <= w[0] < b0 + 2048]
        if not ws: continue
        mx = max(ws, key=lambda w: w[1])
        print("  %6d~%6d  复读率 max %.2f(周期 %2d) min %.2f  桶首 512 位去重 %d" % (b0, b0 + 2047, mx[1], mx[2], min(w[1] for w in ws), len(set(ids[b0:b0 + 512]))))
    if think_txt:
        t = open(think_txt, encoding="utf-8", errors="replace").read()
        pat = re.compile(r"(Now finish|Let's finish|must finish|I must stop|Stop\.|Halt\.|[Pp]roduce final answer|Enough[.!]|stuck in a loop|trapped|runaway)")
        ms = list(pat.finditer(t))
        print("  思考 %d 字, 挣扎收尾短语 %d 句, 首现字符 %s" % (len(t), len(ms), ms[0].start() if ms else "无"))


def cmd_faith(topk_path, ids_path, n_prompt, entry="0"):
    """采样序列上判"采样器忠不忠 / 模型想不想收口"(2026-09-30, 601069 seed 1 思考段复读环; 采样路没有逐字节门):
      ① 每 4096 生成位: Σp(top-1) vs 真吐==top-1 次数 —— 忠实采样只差二项噪声 σ=√Σp(1−p), 超 3σ 就是采样器没按分布抽;
      ② </think>(128822) 与 EOS(1) 在生成段的 进 top-K 次数 / Σp / 单位最大 p, 对观测抽到次数 —— 观测 0 次时忠实采样下的概率 = exp(−Σp);
      ③ 入环位(生成段序号)前后 p(真吐)/p(top-1) 的分位数与 ≥1e-4 候选数 —— 锁死的是不是模型自己的分布。
    只算进 top-K 的质量(K=32 平均覆盖 0.985): Σp 是下界, "K 外"是没算进去的位数。位置 i 的行预测 ids[i+1]。"""
    import math
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, entry = int(n_prompt), int(entry)
    f = open(topk_path, "rb")
    magic, K, S, V = struct.unpack("<4I", f.read(16))
    assert magic == 0x44475445, "不是 ETGD 文件"
    row = struct.Struct("<Iiff" + "%di" % K + "%df" % K)
    g0 = n_prompt - 1
    buckets = {}; close = {128822: [0, 0.0, 0.0, -1], 1: [0, 0.0, 0.0, -1]}   # 进 top-K 次数, Σp, 最大 p, 最大 p 的生成位
    after, pre = [], []
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf); i = v[0]
        if i < g0 or i + 1 >= len(ids): continue
        tid, pr = v[4:4 + K], v[4 + K:4 + 2 * K]
        j = max(range(K), key=lambda q: pr[q]); p1 = pr[j]; y = ids[i + 1]
        py = next((pr[q] for q in range(K) if tid[q] == y), None)
        gpos = i - g0
        b = buckets.setdefault(gpos // 4096, [0.0, 0, 0, 0, 0.0])   # Σp1, hit1, n, K外, Σp1(1−p1)
        b[0] += p1; b[2] += 1; b[1] += tid[j] == y; b[3] += py is None; b[4] += p1 * (1.0 - p1)
        for t, c in close.items():
            for q in range(K):
                if tid[q] == t:
                    c[0] += 1; c[1] += pr[q]
                    if pr[q] > c[2]: c[2], c[3] = pr[q], gpos
        nk = sum(1 for q in range(K) if pr[q] >= 1e-4)
        (after if gpos >= entry else pre).append((py if py is not None else 0.0, p1, nk))
    print("ETGD K=%d; 生成 %d 位(提示 %d), 入环位 %d" % (K, len(ids) - n_prompt, n_prompt, entry))
    print("① 每 4096 生成位: Σp(top-1) | 真吐==top-1 | σ=√Σp(1−p) | 真吐落 top-K 外")
    for b in sorted(buckets):
        s, h, n, out, var = buckets[b]
        sigma = math.sqrt(max(var, 1e-9))
        print("  %6d~%6d  Σp1=%8.1f  hit1=%6d  σ=%5.1f  偏 %+5.1fσ  K外=%4d%s" % (b * 4096, b * 4096 + 4095, s, h, sigma, (h - s) / max(sigma, 1.0), out,
              "  ★超 3σ★" if abs(h - s) > 3 * max(sigma, 1.0) else ""))
    for t, name in ((128822, "</think>"), (1, "EOS")):
        c = close[t]
        print("② %-9s 进 top-K %4d 位, Σp=%.4f, 最大 p=%.5f(生成第 %d 位); 观测抽到 %d 次; 忠实采样下抽到 0 次的概率 exp(−Σp)=%.3f" % (
            name, c[0], c[1], c[2], c[3], sum(1 for x in ids[n_prompt:] if x == t), math.exp(-c[1])))
    def q(xs, k, fr):
        xs = sorted(x[k] for x in xs); return xs[min(len(xs) - 1, int(fr * len(xs)))] if xs else float("nan")
    for name, xs in (("入环后", after), ("入环前", pre)):
        if xs: print("③ %s %d 位: p(真吐) 分位 10/50/90 = %.3f/%.3f/%.3f; p(top-1) = %.3f/%.3f/%.3f; ≥1e-4 候选数中位 %d" % (
            name, len(xs), q(xs, 0, .1), q(xs, 0, .5), q(xs, 0, .9), q(xs, 1, .1), q(xs, 1, .5), q(xs, 1, .9), q(xs, 2, .5)))


def cmd_anchor_faith(anchor_path, ids_path, n_prompt, entry="0", tok="128822"):
    """FP 教师锚(anchor_metrics 的 <i32 S><i32 V><f32 logits[S][V]>)上做与 faith 同一张表(2026-09-30):
    每 4096 生成位: FP argmax == 真吐 的命中率 / mean p_FP(真吐) / Σp_FP(收口 token) / 最大 p; 全段 Σp、最大 p 与位置。
    为什么要它: 学生(引擎 top-32)上 </think> 全程 Σp 只有 0.39, 要分清"量化压没了"还是"FP 也不给", 只能读 FP 在同一条序列同一批位置的概率。
    逐行 seek 读, 不把 24 GB 锚整份装进内存。位置 i 的行预测 ids[i+1]。"""
    import math
    import numpy as np
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, entry, tok = int(n_prompt), int(entry), int(tok)
    f = open(anchor_path, "rb")
    S, V = struct.unpack("<ii", f.read(8))
    g0 = n_prompt - 1
    print("锚 S=%d V=%d; 生成段可比 %d 位(提示 %d), 入环位 %d, 收口 token %d" % (S, V, min(S, len(ids) - 1) - g0, n_prompt, entry, tok))
    buckets = {}; tot = [0.0, 0.0, -1]; after_p1 = []; pre_p1 = []
    for i in range(g0, min(S, len(ids) - 1)):
        f.seek(8 + i * V * 4)
        row = np.frombuffer(f.read(V * 4), dtype=np.float32).astype(np.float64)
        m = row.max(); ex = np.exp(row - m); Z = ex.sum()
        p = ex / Z
        y = ids[i + 1]; a = int(row.argmax()); gpos = i - g0
        b = buckets.setdefault(gpos // 4096, [0, 0, 0.0, 0.0, 0.0, 0.0])   # n, hit, Σp(y), Σp(tok), max p(tok), Σp(argmax)
        b[0] += 1; b[1] += a == y; b[2] += p[y]; b[3] += p[tok]; b[4] = max(b[4], p[tok]); b[5] += p[a]
        tot[0] += p[tok]
        if p[tok] > tot[1]: tot[1], tot[2] = p[tok], gpos
        (after_p1 if gpos >= entry else pre_p1).append(p[a])
    print("每 4096 生成位: FP argmax==真吐 | mean p_FP(真吐) | mean p_FP(argmax) | Σp_FP(收口) | 最大 p_FP(收口)")
    for k in sorted(buckets):
        n, h, sy, st, mx, sa = buckets[k]
        print("  %6d~%6d  命中 %5.1f%%  p(y) %.3f  p(top1) %.3f  Σp(收口) %.4f  max %.5f" % (k * 4096, k * 4096 + 4095, 100.0 * h / n, sy / n, sa / n, st, mx))
    print("全段 Σp_FP(收口 %d) = %.4f, 最大 %.5f(生成第 %d 位); 忠实采样下抽到 0 次的概率 exp(−Σp)=%.3f" % (tok, tot[0], tot[1], tot[2], math.exp(-tot[0])))
    def q(xs, fr):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(fr * len(xs)))] if xs else float("nan")
    for name, xs in (("入环后", after_p1), ("入环前", pre_p1)):
        if xs: print("  %s %d 位: FP p(top-1) 分位 10/50/90 = %.3f/%.3f/%.3f" % (name, len(xs), q(xs, .1), q(xs, .5), q(xs, .9)))


def cmd_tokprobe(anchor_path, ids_path, tok="1", n_prompt="0"):
    """决策 token 的定向压制尺(2026-09-30, 用户"先 1": 证明'决策 token 被量化压掉'不是思考专属):
    在 anchor 格式(<i32 S><i32 V><f32 logits>)上, 对【真值下一个 token 就是 tok】的每个位置(如整篇文档末尾的 EOS、思考段末尾的 </think>)
    打印 p(tok) 与名次; 汇总几何均值 / 中位; 再报其余位置(真值不是 tok)上 p(tok) 的均值 = 假停质量。
    同一份 ids 对 FP 锚与学生锚各跑一遍, 两份几何均值之比 = 该决策 token 被压了几倍(FP/学生)。"""
    import math
    import numpy as np
    ids = [int(x) for x in open(ids_path).read().split()]
    tok, n_prompt = int(tok), int(n_prompt)
    f = open(anchor_path, "rb")
    S, V = struct.unpack("<ii", f.read(8))
    hits, other_sum, other_n = [], 0.0, 0
    for i in range(max(0, n_prompt - 1), min(S, len(ids) - 1)):
        y = ids[i + 1]
        if y != tok and (i % 8) != 0: continue        # 非命中位每 8 位抽一个算假停质量, 省读盘
        f.seek(8 + i * V * 4)
        row = np.frombuffer(f.read(V * 4), dtype=np.float32).astype(np.float64)
        m = row.max(); ex = np.exp(row - m); p = ex / ex.sum()
        if y == tok:
            rank = int((p > p[tok]).sum()) + 1
            hits.append((i, float(p[tok]), rank, int(row.argmax()), float(p[row.argmax()])))
        else:
            other_sum += p[tok]; other_n += 1
    print("锚 %s: S=%d; 真值==%d 的位置 %d 处; 其余位置(抽样 %d)上 p(%d) 均值 %.2e" % (anchor_path.split("/")[-1], S, tok, len(hits), other_n, tok, other_sum / max(other_n, 1)))
    for i, pt, rank, a, pa in hits:
        print("  位置 %6d  p(tok)=%.4f  名次 %3d  argmax=%d(p=%.3f)%s" % (i, pt, rank, a, pa, "  ★tok 就是 argmax★" if a == tok else ""))
    if hits:
        ps = [h[1] for h in hits]
        gm = math.exp(sum(math.log(max(x, 1e-12)) for x in ps) / len(ps))
        print("  几何均值 p(tok) = %.4f; 中位 %.4f; tok 是 argmax 的位置 %d/%d" % (gm, sorted(ps)[len(ps) // 2], sum(1 for h in hits if h[3] == tok), len(hits)))


if __name__ == "__main__":
    cmds = {"extract": cmd_extract, "cut": cmd_cut, "cmp-emit": cmd_cmp_emit, "cmp-topk": cmd_cmp_topk,
            "eos-scan": cmd_eos_scan, "rows": cmd_rows, "loopstat": cmd_loopstat, "faith": cmd_faith,
            "anchor-faith": cmd_anchor_faith, "tokprobe": cmd_tokprobe}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__ + "  rows <rows.txt> <ids> <n_prompt> <入口位置> [tokenizer.json] [窗口=12]   --row-out 的分桶命中率 + 入口逐行解码\n")
    cmds[sys.argv[1]](*sys.argv[2:])
