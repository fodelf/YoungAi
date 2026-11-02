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
  minp <topk.bin> <ids> <n_prompt> [入环位=0] [min_p=0.02] [tok=128822]
                                          同一份 top-K 按产品 min_p 门重算: 偏离 top-1 与出口 tok 的期望次数, 原始对截断后;
                                          分清"环是截断造的吸收态"还是"模型自己锁死"。
  cyclescan <最短段长> <trace...>         按 token 精确周期找死环(P ≤ 256, 段长 ≥ max(3P, 最短段长)), 扫全部历史请求数误报/漏报。
  anchor2ptf <锚.bin> <ids> <out.ptf> [K=64] [前缀长度,…]  FP 全 logits 锚 → PTF1 top-K 表(训练器 teacher=fp: 的格式); 可另写几条前缀。
  ptfpick <api.ptf> <out.seqs> [每隔=10]  从段 B 一轮的官方 API 表里隔条取序列, 给 v41_teacher.py --seqs 打 FP 表。
  ptfcmp <api.ptf> <fp.ptf> [kl0.tsv]     同一批序列上官方 API top-20(训练靶)对 FP 原件(尺子参考)逐位比; 给 kl0.tsv 并排报学生。
"""
import json
import os
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
    after, pre, kout = [], [], []
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf); i = v[0]
        if i < g0 or i + 1 >= len(ids): continue
        tid, pr = v[4:4 + K], v[4 + K:4 + 2 * K]
        j = max(range(K), key=lambda q: pr[q]); p1 = pr[j]; y = ids[i + 1]
        py = next((pr[q] for q in range(K) if tid[q] == y), None)
        gpos = i - g0
        b = buckets.setdefault(gpos // 4096, [0.0, 0, 0, 0, 0.0, 0.0])   # Σp1, hit1, n, K外, Σp1(1−p1), 期望 K外=Σ(1−Σp_topK)
        b[0] += p1; b[2] += 1; b[1] += tid[j] == y; b[3] += py is None; b[4] += p1 * (1.0 - p1); b[5] += max(0.0, 1.0 - sum(pr))
        # ★K 外逐位存档★(2026-10-12 鹈鹕乱码): 真吐落 top-K 外的位置, 记 p(top-1) 与 top-K 里最小的 p(= p(真吐) 的上界),
        # 乱码 token("#fant"/"排队"/"q?")是不是全在这里, 以及它们的上界有多小 —— 忠实采样下 K 外总数 ≈ 期望 K外, 远超就是采样器没按分布抽
        if py is None: kout.append((gpos, y, p1, min(pr), tid[j]))
        for t, c in close.items():
            for q in range(K):
                if tid[q] == t:
                    c[0] += 1; c[1] += pr[q]
                    if pr[q] > c[2]: c[2], c[3] = pr[q], gpos
        nk = sum(1 for q in range(K) if pr[q] >= 1e-4)
        (after if gpos >= entry else pre).append((py if py is not None else 0.0, p1, nk))
    print("ETGD K=%d; 生成 %d 位(提示 %d), 入环位 %d" % (K, len(ids) - n_prompt, n_prompt, entry))
    print("① 每 4096 生成位: Σp(top-1) | 真吐==top-1 | σ=√Σp(1−p) | 真吐落 top-K 外(观测 / 期望=Σ(1−Σp_topK))")
    for b in sorted(buckets):
        s, h, n, out, var, eout = buckets[b]
        sigma = math.sqrt(max(var, 1e-9))
        print("  %6d~%6d  Σp1=%8.1f  hit1=%6d  σ=%5.1f  偏 %+5.1fσ  K外=%4d/期望 %6.1f%s" % (b * 4096, b * 4096 + 4095, s, h, sigma, (h - s) / max(sigma, 1.0), out, eout,
              "  ★超 3σ★" if abs(h - s) > 3 * max(sigma, 1.0) else ""))
    if kout:
        print("①b K 外逐位(生成位 | 真吐 id | p(top-1) | top-K 最小 p = p(真吐)上界 | top-1 id), 共 %d 位, 按上界从小到大前 40:" % len(kout))
        for gpos, y, p1, pmin, t1 in sorted(kout, key=lambda x: x[3])[:40]:
            print("  %6d | %6d | %.3f | <%.1e | %d" % (gpos, y, p1, pmin, t1))
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


def cmd_ids2seqs(ids_path, manifest_path, out_path):
    """docend 这类"几篇拼成一条"的 ids + manifest(篇号 子域 源行号 起点 EOS位置 篇长) → 一行一篇的序列文件(v41_teacher.py --seqs 的输入, 2026-10-22)。
    每篇 = ids[起点 .. EOS位置](含 BOS 与 EOS)。"""
    ids = [int(x) for x in open(ids_path).read().split()]
    n = 0
    with open(out_path, "w") as f:
        for ln in open(manifest_path, encoding="utf-8"):
            if ln.startswith("#") or not ln.strip(): continue
            _, _, _, b0, e0, _ = ln.split()[:6]
            f.write(" ".join(str(t) for t in ids[int(b0):int(e0) + 1]) + "\n"); n += 1
    print("%s: %d 篇 → %s" % (manifest_path.split("/")[-1], n, out_path))


def cmd_ptfcheck(anchor_path, ptf_path, seq_index, row0, kk="64"):
    """★batch 维 FP 路的对拍门(2026-10-22)★: PTF1 top-K 表(v41_teacher_seqs.py, 多条序列一次前向)里第 seq_index 条 vs 全 logits 锚(单条流式前向)从第 row0 行起的同一批位置。
    用法: 把 docend 的 25 篇按篇拆成序列(ids2seqs)一起打 top-K; 第 0 篇在流里从位置 0 起、前面没有别的篇 ⇒ 两条路必须逐位同(top-1 全同, |Δp| 在 bf16 噪声内);
    第 k 篇(k ≥ 1)在流里带着前面 k 篇的上下文 ⇒ 差多少就是"拼接当一条"对教师分布的影响。
    报: 位数 / top-1 同 / 表上 top-K 的 |Δp| 均值与最大 / 表的 top-1 概率均值 vs 锚的。"""
    import math
    import numpy as np
    seq_index, row0, kk = int(seq_index), int(row0), int(kk)
    f = open(ptf_path, "rb")
    recs = []
    while True:
        hd = f.read(12)
        if len(hd) < 12: break
        magic, K, N = struct.unpack("<III", hd)
        assert magic == 0x31465450, "不是 PTF1 段"
        for _ in range(N):
            L = struct.unpack("<I", f.read(4))[0]
            ids = np.frombuffer(f.read(L * 4), dtype=np.int32)
            ti = np.frombuffer(f.read((L - 1) * K * 4), dtype=np.int32).reshape(L - 1, K)
            tp = np.frombuffer(f.read((L - 1) * K * 4), dtype=np.float32).reshape(L - 1, K)
            rest = np.frombuffer(f.read((L - 1) * 4), dtype=np.float32)
            recs.append((ids, ti, tp, rest))
    ids, ti, tp, rest = recs[seq_index]
    a = open(anchor_path, "rb")
    S, V = struct.unpack("<ii", a.read(8))
    n = ids.shape[0] - 1
    same = 0; dsum = 0.0; dmax = 0.0; p1t = 0.0; p1a = 0.0; kls = 0.0
    for i in range(n):
        a.seek(8 + (row0 + i) * V * 4)
        row = np.frombuffer(a.read(V * 4), dtype=np.float32).astype(np.float64)
        m = row.max(); ex = np.exp(row - m); p = ex / ex.sum()
        same += int(row.argmax() == ti[i, 0])
        d = np.abs(p[ti[i]] - tp[i]); dsum += float(d.mean()); dmax = max(dmax, float(d.max()))
        p1t += float(tp[i, 0]); p1a += float(p.max())
        q = np.clip(tp[i].astype(np.float64), 1e-12, 1.0); pk = np.clip(p[ti[i]], 1e-12, 1.0)
        kls += float((pk * np.log(pk / q)).sum())   # 锚 vs 表在表的 top-K 上的 KL 项(余量略), 只作量级
    print("ptfcheck 表第 %d 条(%d 位) vs 锚第 %d 行起: top-1 同 %d/%d = %.2f%%; 表 top-K 上 |Δp| 均值 %.2e 最大 %.2e; top-1 p 表 %.4f / 锚 %.4f; KL(top-K 项) %.2e/位"
          % (seq_index, n, row0, same, n, 100.0 * same / max(n, 1), dsum / max(n, 1), dmax, p1t / max(n, 1), p1a / max(n, 1), kls / max(n, 1)))


def read_ptf(path):
    """PTF1 表(可多段首尾相接) → [(ids, top_id[L-1,K], top_p[L-1,K], rest[L-1])]"""
    import numpy as np
    f = open(path, "rb"); recs = []
    while True:
        hd = f.read(12)
        if len(hd) < 12: break
        magic, K, N = struct.unpack("<III", hd)
        assert magic == 0x31465450, "不是 PTF1 段"
        for _ in range(N):
            L = struct.unpack("<I", f.read(4))[0]
            ids = np.frombuffer(f.read(L * 4), dtype=np.int32)
            ti = np.frombuffer(f.read((L - 1) * K * 4), dtype=np.int32).reshape(L - 1, K)
            tp = np.frombuffer(f.read((L - 1) * K * 4), dtype=np.float32).reshape(L - 1, K)
            recs.append((ids, ti, tp, np.frombuffer(f.read((L - 1) * 4), dtype=np.float32)))
    return recs


def cmd_ptfhit(ptf_path):
    """教师表自检(11-02): 每条记录里 top-1 == 下一个真 token 的比例。FP 锚对本模型自然语料的 top-1 命中约 70%(v41_judge 参考 PPL 行);
    远低于 50% = 锚和 ids 不是同一串(配错了文件), 训练器照样能配上表、只是教的全是别的位置的分布, 不报错 —— 所以 prep 必须先过这一关。"""
    recs = read_ptf(ptf_path); hit = 0; n = 0
    for ids, ti, tp, rest in recs:
        n += len(ids) - 1; hit += int((ti[:, 0] == ids[1:]).sum())
    print("ptfhit %s: %d 条 %d 位, 教师 top-1 命中下一 token %.1f%%" % (ptf_path.split("/")[-1], len(recs), n, 100.0 * hit / max(n, 1)))
    if n and hit / n < 0.5: sys.exit("★命中率过低: 锚与 ids 不是同一串★")


def cmd_anchor2ptf(anchor_path, ids_path, out_path, kk="64", prefixes=""):
    """FP 全 logits 锚(<i32 S><i32 V><f32 S×V>, v41_judge.sh 的 teacher_*.bin)→ PTF1 top-K 表(训练器 teacher=fp: 吃的格式)。
    10-28 段 B 天花板: 尺子自己的 FP 读数当训练靶, 不另跑一趟原件。ids 文件 = 锚对应的那串 id(空白分隔), 长度必须等于 S, 否则停。
    prefixes = "33,…": 另写几条"整串的前 k 位"(因果前向, 前缀的各行与整串同一批行)。为什么要单独一条: 训练器按整条序列的哈希查表
    (core_ptrain_fp.c pt_fpmap_slot), 一道题的序列是尺子的前缀也查不到, 实撞"教师表里没有第 0 题的序列(33 位)"。"""
    import numpy as np
    K = int(kk); ids = np.array(open(ids_path).read().split(), dtype=np.int32); pre = [int(x) for x in prefixes.split(",") if x]
    with open(anchor_path, "rb") as f: S, V = struct.unpack("<ii", f.read(8))
    if S != len(ids): sys.exit("★锚 %d 行 ≠ ids %d 个, 不是同一串★" % (S, len(ids)))
    lg = np.memmap(anchor_path, dtype=np.float32, mode="r", offset=8, shape=(S, V))
    ti = np.zeros((S - 1, K), np.int32); tp = np.zeros((S - 1, K), np.float32); rest = np.zeros(S - 1, np.float32)
    for i0 in range(0, S - 1, 512):
        x = np.asarray(lg[i0:min(S - 1, i0 + 512)], dtype=np.float64); x -= x.max(1, keepdims=True); p = np.exp(x); p /= p.sum(1, keepdims=True)
        top = np.argpartition(-p, K, axis=1)[:, :K]; pt = np.take_along_axis(p, top, 1); o = np.argsort(-pt, 1)
        n = len(x); ti[i0:i0 + n] = np.take_along_axis(top, o, 1); tp[i0:i0 + n] = np.take_along_axis(pt, o, 1); rest[i0:i0 + n] = np.clip(1 - tp[i0:i0 + n].sum(1), 0, 1)
    if any(k < 2 or k > S for k in pre): sys.exit("★前缀长度要在 2..%d★" % S)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x31465450, K, 1 + len(pre)))
        for L in [S] + pre:
            f.write(struct.pack("<I", L)); f.write(ids[:L].tobytes())
            f.write(ti[:L - 1].tobytes()); f.write(tp[:L - 1].tobytes()); f.write(rest[:L - 1].tobytes())
    print("%s → %s: 整串 %d 位%s, top-%d; top-1 均 %.4f, 余量均 %.4f" % (anchor_path.split("/")[-1], out_path, S, "".join(" + 前 %d 位" % k for k in pre), K, tp[:, 0].mean(), rest.mean()))


def cmd_ptfpick(ptf_path, out_path, every="10"):
    """从一份 PTF1 表(段 B 一轮的 teacher.ptf = 官方 API top-20)里每隔 every 条取一条, 写成 v41_teacher.py --seqs 的输入(一行一条 id 串)。
    均匀隔条取是为了各种语言都摊到(① 的种子按语言连着排)。"""
    recs = read_ptf(ptf_path); every = int(every)
    pick = recs[::every]
    with open(out_path, "w") as f:
        for ids, *_ in pick: f.write(" ".join(map(str, ids.tolist())) + "\n")
    print("%s: %d 条里取 %d 条(每 %d 条一条), 共 %d 位 → %s" % (ptf_path.split("/")[-1], len(recs), len(pick), every, sum(len(r[0]) for r in pick), out_path))


def cmd_ptfcmp(api_ptf, fp_ptf, kl0_tsv=""):
    """★训练靶 vs 尺子参考(2026-10-28)★: 同一批序列上, 官方 API 的 top-20 表(段 B 的训练靶)对参考前向 FP 原件的 top-K 表(判决尺的参考)逐位比。
    为什么: 段 B 往 API 练、拿 FP 判; 两者在代码裸续写上若差得不小, "训练料 KL 降、编程尺不动" 就有了解释。10-26 只在一条对话思考轨迹上量过(KL 0.0065)。
    口径与训练器同(ds4_gpu_bwd_kl_topk_tensor): 在 API 的 top-20 + 余量桶上算 KL(API‖FP); API 候选不在 FP top-K 里时 FP 概率取它 top-K 的最小值(上界, KL 只会少算), 单报这类位数。
    API 表的 one-hot 行(种子开头 / 切分对不上的位, 不是分布)跳过。给了 kl0.tsv(同一轮 --restore-b 第 0 步逐位 KL = KL(API‖学生))就在同一批位上并排报学生。"""
    import math
    import numpy as np
    A = read_ptf(api_ptf); Fm = {tuple(r[0].tolist()): r for r in read_ptf(fp_ptf)}
    kid = kkl = None
    if kl0_tsv:
        z = np.loadtxt(kl0_tsv, dtype=np.float64); kid = z[:, 0].astype(np.int64); kkl = z[:, 1]
    same = 0; kl_af = []; kl_fa = []; smin = []; miss_n = 0; miss_m = 0.0; stu = []; nseq = nstu = 0
    sxy = sxx = 0.0; top1a = []; top1f = []   # 温度: 同一位上两边都有的候选, 去均值后 log p_API 对 log p_FP 的斜率(= T_FP / T_API, 1 = 同温)
    for ids, ai, ap, ar in A:
        fr = Fm.get(tuple(ids.tolist()))
        if fr is None: continue
        nseq += 1; _, fi, fp, frr = fr
        onehot = (ap[:, 0] > 0.999) & (ap[:, 1:].sum(1) < 1e-6) & (ar < 1e-6)
        rows = [i for i in range(len(ids) - 1) if not onehot[i]]
        if not rows: continue
        ks = None
        if kid is not None:   # 学生: kl0.tsv 是训练答案位的目标 id 流, 找这条从第一个分布行起的目标 id 串
            t0 = rows[0]; tgt = ids[t0 + 1:].astype(np.int64); w = tgt[:24]
            cand = np.flatnonzero(kid[:len(kid) - len(tgt) + 1] == w[0])
            for c in cand:
                if np.array_equal(kid[c:c + len(w)], w) and np.array_equal(kid[c:c + len(tgt)], tgt): ks = c - t0; break
            nstu += ks is not None
        for i in rows:
            pa = ap[i].astype(np.float64); fmap = dict(zip(fi[i].tolist(), fp[i].astype(np.float64).tolist())); fmin = float(fp[i].min())
            pf = np.array([fmap.get(t, -1.0) for t in ai[i].tolist()])
            miss = pf < 0; miss_n += int(miss.sum()); miss_m += float(pa[miss].sum()); pf[miss] = fmin
            ra = max(float(ar[i]), 1e-12); rf = max(1.0 - float(pf.sum()), 1e-12)
            pa_c = np.clip(pa, 1e-12, 1); pf_c = np.clip(pf, 1e-12, 1)
            kl_af.append(float((pa_c * np.log(pa_c / pf_c)).sum() + ra * math.log(ra / rf)))
            kl_fa.append(float((pf_c * np.log(pf_c / pa_c)).sum() + rf * math.log(rf / ra)))
            smin.append(float(np.minimum(pa, pf).sum() + min(ra, rf)))
            same += int(ai[i][int(pa.argmax())] == fi[i][int(fp[i].argmax())])
            both = ~miss & (pa > 0)
            if both.sum() >= 2:
                x = np.log(pf_c[both]); y = np.log(pa_c[both]); x -= x.mean(); y -= y.mean(); sxy += float(x @ y); sxx += float(x @ x)
            top1a.append(float(pa.max())); top1f.append(float(fp[i].max()))
            stu.append(kkl[ks + i] if ks is not None else float("nan"))
    kl_af = np.array(kl_af); kl_fa = np.array(kl_fa); smin = np.array(smin); stu = np.array(stu); n = len(kl_af)
    q = lambda x: "均 %.5f 中位 %.5f p95 %.5f" % (x.mean(), np.median(x), np.percentile(x, 95))
    print("API 表 %d 条, 对上 FP 表 %d 条, 分布位 %d" % (len(A), nseq, n))
    print("  Same top(API top-1 == FP top-1) %.2f%%; Σmin(top-20∪余量) %s" % (100 * same / n, q(smin)))
    print("  KL(API‖FP) %s  ← 与训练器同方向同口径" % q(kl_af))
    print("  KL(FP‖API) %s" % q(kl_fa))
    print("  API 候选不在 FP top-%d 里: %d 个(%.2f 个/位), 这些候选的 API 概率合计均 %.4f/位" % (fp.shape[1], miss_n, miss_n / n, miss_m / n))
    print("  温度: 位内去均值 log p_API 对 log p_FP 斜率 %.4f(1 = 同温; >1 = API 更尖); top-1 概率均 API %.4f / FP %.4f"
          % (sxy / max(sxx, 1e-12), np.mean(top1a), np.mean(top1f)))
    if kid is not None:
        ok = ~np.isnan(stu); s = stu[ok]; t = kl_af[ok]
        print("  学生(kl0.tsv, KL(API‖学生)) 对上 %d/%d 条 %d 位: %s;  同一批位 KL(API‖FP) 均 %.5f ⇒ 学生是 FP 的 %.1f 倍" % (nstu, nseq, len(s), q(s), t.mean(), s.mean() / max(t.mean(), 1e-12)))
        for th in (0.05, 0.2):
            m = t > th
            print("  KL(API‖FP) > %.2f 的位: 占 %.1f%%, 学生 KL 在这些位上占 %.1f%%" % (th, 100 * m.mean(), 100 * s[m].sum() / max(s.sum(), 1e-12)))


def cmd_ptfstat(ptf_path, think_end="128822", eos="1", qmark="33"):
    """每份 FP 教师落盘就看教师在决策位给了多少(2026-10-22 用户: "每一份结束能看下指标, 早点发现问题"): 不用 GPU, 只读 PTF1 top-K 表。
    每条序列: 末位(目标 EOS)的 p / 含 </think> 的条上 </think> 位的 p / 非决策位上 EOS·</think> 的尾巴(top-K 里可见的 p, 不可见记 0 = 下界) / 问号位。
    读法: 这是 ③ 要学的靶 —— 教师在 </think>·EOS 位给得高(docend 锚上 FP 真结尾几何均值 0.0629, 思考收口位 FP 常 0.2~0.9)说明渲染/尾段切法没毛病;
    给得很低先怀疑料: 尾段切坏、前缀渲染错、补齐污染。目标不在 top-K 的位单列("不可见"), 几何均值把它按第 K 名的 p 记(上界)。"""
    import math
    import numpy as np
    think_end, eos, qmark = int(think_end), int(eos), int(qmark)
    f = open(ptf_path, "rb")
    recs = []
    while True:
        hd = f.read(12)
        if len(hd) < 12: break
        magic, K, N = struct.unpack("<III", hd)
        assert magic == 0x31465450, "不是 PTF1 段"
        for _ in range(N):
            L = struct.unpack("<I", f.read(4))[0]
            ids = np.frombuffer(f.read(L * 4), dtype=np.int32)
            ti = np.frombuffer(f.read((L - 1) * K * 4), dtype=np.int32).reshape(L - 1, K)
            tp = np.frombuffer(f.read((L - 1) * K * 4), dtype=np.float32).reshape(L - 1, K)
            f.read((L - 1) * 4)
            recs.append((ids, ti, tp))

    def p_of(ti_row, tp_row, tok):   # tok 在这一位 top-K 里的 p; 不在 → None
        w = np.nonzero(ti_row == tok)[0]
        return float(tp_row[w[0]]) if len(w) else None

    # 四类序列分开报: 轨迹(<think>…</think>, 它的 </think> 是模型自己要停的收口位) / 中段(只有 <think> 没有 </think>: 思考开头一段当"这里别收"的负样本,
    # 10-22 晚; 末位那个 EOS 是训练器补的, 教师在那儿应 ≈0, 单列不混进真结尾) / 问答(chat 模板 <｜Assistant｜></think>答案: 那个 </think> 是模板硬塞的,
    # 模型从不采样, FP 在那里只给 ~3e-4, 不是决策位, 不计) / 文档(整篇, 只有末位 EOS)
    think_start = 128821
    def kind_of(ids):
        has_s, has_e = (ids == think_start).any(), (ids == think_end).any()
        if has_s and has_e: return "轨迹"
        if has_s: return "中段"
        if has_e: return "问答"
        return "文档"
    kinds = {}
    def add(st, name, p, is_top, kth):
        s = st.setdefault(name, {"v": [], "miss": 0, "bound": [], "top": 0})
        if p is None: s["miss"] += 1; s["bound"].append(kth)
        else: s["v"].append(p); s["top"] += int(is_top)
    hit = tot = 0
    for ids, ti, tp in recs:
        n = len(ids) - 1; tgt = ids[1:]
        hit += int((ti[:, 0] == tgt).sum()); tot += n
        kd = kind_of(ids)
        kk = kinds.setdefault(kd, {"stats": {}, "tails": {"EOS 非末位": [], "</think> 非收口位": [], "'?' 非问号位": []}, "n": 0, "rows": 0, "hit": 0})
        kk["n"] += 1; kk["rows"] += n; kk["hit"] += int((ti[:, 0] == tgt).sum())
        st, tails = kk["stats"], kk["tails"]
        for i in range(n):
            y = int(tgt[i]); kth = float(tp[i, -1]); top = int(ti[i, 0])
            if y == eos and i == n - 1:
                add(st, "EOS 末位" if kd != "中段" else "EOS 末位(补的,应≈0)", p_of(ti[i], tp[i], eos), top == eos, kth)
            elif y == think_end and kd == "轨迹":
                add(st, "</think> 收口位", p_of(ti[i], tp[i], think_end), top == think_end, kth)
            elif y == qmark:
                add(st, "'?' 问号位", p_of(ti[i], tp[i], qmark), top == qmark, kth)
            if y != eos: tails["EOS 非末位"].append(p_of(ti[i], tp[i], eos) or 0.0)
            if kd in ("轨迹", "中段") and y != think_end: tails["</think> 非收口位"].append(p_of(ti[i], tp[i], think_end) or 0.0)
            if y != qmark: tails["'?' 非问号位"].append(p_of(ti[i], tp[i], qmark) or 0.0)
    print("[ptfstat] %s: %d 条, %d 行, 教师 argmax 命中 %.2f%%; 分类 %s" % (ptf_path.split("/")[-1], len(recs), tot, 100.0 * hit / max(tot, 1),
          ", ".join("%s %d" % (k, v["n"]) for k, v in kinds.items())))
    for kd, kk in kinds.items():
        print("  [%s] %d 条 %d 行, 命中 %.1f%%" % (kd, kk["n"], kk["rows"], 100.0 * kk["hit"] / max(kk["rows"], 1)))
        for name in ("EOS 末位", "EOS 末位(补的,应≈0)", "</think> 收口位", "'?' 问号位"):
            s = kk["stats"].get(name)
            if not s: continue
            allv = s["v"] + s["bound"]; n = len(allv)
            gm = math.exp(sum(math.log(max(v, 1e-9)) for v in allv) / n)
            print("    %-18s n %4d  几何均值 p %.4f  中位 %.4f  argmax 占比 %d/%d  不在 top-K %d"
                  % (name, n, gm, float(np.median(allv)), s["top"], n, s["miss"]))
        for name, v in kk["tails"].items():
            if v: print("    %-18s 位数 %6d  均值 p %.2e(top-K 里可见的, 下界)" % (name, len(v), float(np.mean(v))))


def cmd_minp(topk_path, ids_path, n_prompt, entry="0", min_p="0.02", tok="128822"):
    """采样截断是不是把出口切掉了(2026-10-16, 助理思考段 4.6 万 token 逐 id 复读、`</think>` 一次不出):
    faith 的 ② 只给 p(</think>) 的原始总和, 而产品采样在抽之前先按 min_p 砍(p < min_p·p(top-1) 的词不参与)。
    这里把同一份 top-K 按产品门再算一遍, 入环前/入环后分开报:
      偏离 = 不抽 top-1 的概率: 原始 Σ(1−p1) 对 截断后 Σ(1−p1/保留质量), 对观测"真吐≠top-1"次数;
      出口 = tok 的概率: 原始 Σp 对 截断后 Σp/保留质量(被砍的位记 0), 以及 exp(−Σ) = 一次都抽不到的概率。
    原始大、截断后≈0 ⇒ 环是截断造的吸收态; 两个都≈0 ⇒ 模型自己锁死, 改采样没用。
    tok 不在 top-K 的位: 只知道 p ≤ top-K 最小 p, 按上界计入原始 Σ, 上界若也低于门就确定被砍。
    门在 top-K 里不一定够得着(K 个都 ≥ 门): 保留质量只算 K 内, 是下界, 计数报"门在 K 外"位数。"""
    import math
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, entry, min_p, tok = int(n_prompt), int(entry), float(min_p), int(tok)
    f = open(topk_path, "rb")
    magic, K, S, V = struct.unpack("<4I", f.read(16))
    assert magic == 0x44475445, "不是 ETGD 文件"
    row = struct.Struct("<Iiff" + "%di" % K + "%df" % K)
    g0 = n_prompt - 1
    seg = {k: dict(n=0, dev_raw=0.0, dev_cut=0.0, dev_obs=0, out_raw=0.0, out_cut=0.0, in_k=0, kept=0, cut_sure=0, gate_out=0, obs=0, best=[])
           for k in ("入环前", "入环后")}
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf); i = v[0]
        if i < g0 or i + 1 >= len(ids): continue
        tid, pr = v[4:4 + K], v[4 + K:4 + 2 * K]
        p1 = max(pr); gate = min_p * p1
        s = seg["入环后" if i - g0 >= entry else "入环前"]
        kept = sum(p for p in pr if p >= gate)
        s["n"] += 1; s["gate_out"] += min(pr) >= gate
        s["dev_raw"] += 1.0 - p1; s["dev_cut"] += 1.0 - p1 / kept
        s["dev_obs"] += tid[pr.index(p1)] != ids[i + 1]; s["obs"] += ids[i + 1] == tok
        pt = next((pr[q] for q in range(K) if tid[q] == tok), None)
        if pt is None:
            ub = min(pr); s["out_raw"] += ub; s["cut_sure"] += ub < gate
            continue
        s["in_k"] += 1; s["out_raw"] += pt
        if pt >= gate:
            s["kept"] += 1; s["out_cut"] += pt / kept
        s["best"].append((pt, i - g0, p1, pt >= gate))
    print("ETGD K=%d; 生成 %d 位, 入环位 %d, min_p %.3f, 出口 tok %d" % (K, len(ids) - n_prompt, entry, min_p, tok))
    for name, s in seg.items():
        if not s["n"]: continue
        print("== %s %d 位(门在 top-K 外 %d 位, 这些位保留质量是下界)" % (name, s["n"], s["gate_out"]))
        print("   偏离 top-1: 原始期望 %.1f 次 | min_p 后期望 %.1f 次 | 观测 %d 次" % (s["dev_raw"], s["dev_cut"], s["dev_obs"]))
        print("   出口 %d: 进 top-K %d 位, 其中过门 %d 位; K 外且上界也低于门(确定被砍) %d 位" % (tok, s["in_k"], s["kept"], s["cut_sure"]))
        print("   出口 Σp: 原始 %.4f(K 外按上界) → 一次不出概率 exp(−Σ)=%.3g | min_p 后 %.4f → exp(−Σ)=%.3g | 观测 %d 次" % (
            s["out_raw"], math.exp(-s["out_raw"]), s["out_cut"], math.exp(-s["out_cut"]), s["obs"]))
        for pt, gpos, p1, ok in sorted(s["best"], reverse=True)[:8]:
            print("     生成第 %6d 位 p(出口)=%.5f p(top-1)=%.3f 门 %.5f %s" % (gpos, pt, p1, min_p * p1, "过门" if ok else "★被砍★"))


def cmd_agree(topk_path, ids_path, n_prompt, bucket="1000"):
    """解码路(真吐)对预填路(top-K 重打分)的逐位一致率, 按位置分桶(2026-10-20, 用户"引擎里有魔法数字"):
    喂一条温 0 的真实轨迹 —— 真吐 = 解码路(投机验证批/纯解码)在该位的 argmax, 重打分 top-1 = 预填路(块 512)同一位的 argmax。
    两条路状态都对就只剩浮点近平局的分歧, 一致率应当平着不随位置掉; 某个偏移起掉下去 / 周期性塌 = 解码路把状态写坏了, 偏移就是魔法数字。
    每桶报: 一致率、分歧里预填 p(top-1) 的中位(近平局应 <0.6)、"预填 p1>0.7 却被解码路选别的"(09-22 位平面 bug 的指纹)计数。"""
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, bucket = int(n_prompt), int(bucket)
    f = open(topk_path, "rb")
    magic, K, S, V = struct.unpack("<4I", f.read(16))
    assert magic == 0x44475445, "不是 ETGD 文件"
    row = struct.Struct("<Iiff" + "%di" % K + "%df" % K)
    g0 = n_prompt - 1
    b = {}
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf); i = v[0]
        if i < g0 or i + 1 >= len(ids): continue
        tid, pr = v[4:4 + K], v[4 + K:4 + 2 * K]
        p1 = max(pr); top = tid[pr.index(p1)]; y = ids[i + 1]
        k = (i - g0) // bucket
        s = b.setdefault(k, [0, 0, [], 0, 0])   # n, same, 分歧时 p1 列表, p1>0.7 分歧数, 真吐不在 top-K
        s[0] += 1
        if top == y: s[1] += 1
        else:
            s[2].append(p1); s[3] += p1 > 0.7; s[4] += y not in tid
    print("解码路真吐 vs 预填路 top-1, 每 %d 生成位: 一致率 | 分歧数 | 分歧处预填 p1 中位 | p1>0.7 被翻 | 真吐不在 top-%d" % (bucket, K))
    tot = [0, 0, 0]
    for k in sorted(b):
        n, same, dl, hi, miss = b[k]; med = sorted(dl)[len(dl) // 2] if dl else float("nan")
        tot[0] += n; tot[1] += same; tot[2] += hi
        print("  %6d~%6d: %6.2f%% | %4d | %.3f | %3d | %3d" % (k * bucket, k * bucket + bucket - 1, 100.0 * same / n, n - same, med, hi, miss))
    print("全段 %d 位: 一致 %.2f%%, p1>0.7 被翻 %d" % (tot[0], 100.0 * tot[1] / max(tot[0], 1), tot[2]))


def cmd_cyclescan(min_run="256", *traces):
    """思考段死环检测器的离线误报/漏报账(2026-10-17): 判据 = 生成段里出现"周期 P ≤ 256、逐 token 与 P 位前相同"的连续段,
    段长(含第一圈) ≥ max(3P, min_run) 就算触发。为什么按 token 精确周期: 饿了么 r316 进环后 p(top-1)=1.000、逐 id 周期 43,
    这时再想下去不产生任何新 token —— 截断不丢信息; 措辞会变的环(鹈鹕/r331)这个判据故意不抓, 那种跟正常的反复推敲分不开。
    逐行流式读 trace(2.9 GB 的那份不整份进内存); 每个请求报: 最长周期段(P, 段长, 起点), 在思考段里还是正文里, finish。
    思考段 = 提示以 <think> 收尾时, 生成段到第一个 </think>(128822) 为止(没有就是整段)。"""
    import numpy as np
    min_run = int(min_run)
    rows = []
    for path in traces:
        req = finish = None; pend = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("===== request "):
                    req = line.strip("= \n"); finish = None
                elif line.startswith("finish: "):
                    finish = line.split(None, 1)[1].strip()
                elif line.startswith("prompt: ") and req:
                    pend = line.split()[-1]
                elif line.startswith("generated: ") and req and pend is not None:
                    g = np.array(line.split()[1:], dtype=np.int64); think_open = pend; pend = None
                    n = len(g)
                    if n < 8: continue
                    close = np.flatnonzero(g == 128822); close = int(close[0]) if len(close) else -1
                    best = (0, 0, 0)   # 段长, P, 起点
                    for P in range(1, min(256, n // 3) + 1):
                        eq = np.concatenate(([0], (g[P:] == g[:-P]).astype(np.int8), [0]))
                        d = np.diff(eq); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
                        if not len(st): continue
                        k = int(np.argmax(en - st)); L = int(en[k] - st[k]) + P
                        if L > best[0]: best = (L, P, int(st[k]))
                    L, P, s0 = best
                    rows.append((path.split("/")[-1], req, n, think_open, close, L, P, s0, finish,
                                 L >= max(3 * P, min_run)))
    trig = [r for r in rows if r[9]]
    print("扫 %d 个请求, 门槛 段长 ≥ max(3P, %d): 触发 %d 个" % (len(rows), min_run, len(trig)))
    print("trace | 请求 | 生成数 | 提示末 id | 首个 </think> 位 | 最长周期段 段长/P/起点 | 段在 | finish")
    for r in sorted(rows, key=lambda r: -r[5])[:40]:
        where = "思考" if (r[4] < 0 or r[7] < r[4]) and r[3] == "128821" else "正文"
        print("%s %s | %6d | %s | %6d | %6d / %3d / %6d | %s | %s%s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], where, r[8],
              "  ★触发★" if r[9] else ""))


def cmd_temper(topk_path, ids_path, n_prompt, entry="0", temps="1.0,1.2,1.5,2.0", min_p="0.02", tok="128822"):
    """温度/min_p 能不能把出口抽出来(2026-10-18, 用户"之前配置的温度和 min 参数再看看"): 同一份 top-K, 对每个 T 把 p 换成 p^(1/T) 在 K 内归一
    (K 外质量不算 ⇒ 出口概率略偏高, 上界口径), 再按 min_p·p(top-1) 砍, 报入环前/后两段的出口 Σp 与"一次都抽不到"概率 exp(−Σ)。
    代价一侧同一张表: 温度抬高后 top-1 平均概率(掉得越多正文越飘), 以及 K 外尾巴质量的上界 Σ_tail p^(1/T) ≤ (1−mass)·p_min^(1/T−1)
    (尾巴每个 p ≤ p_min; 这是 10-12 "每 160 token 抽一个垃圾词"那种乱码的来源)。判读: 出口 Σ 到 3 以上(exp(−3)=5%)才算能靠采样逃环;
    同一 T 下尾巴上界若也到 1e-2/位量级, 就是拿正文乱码换出口。"""
    import math
    ids = [int(x) for x in open(ids_path).read().split()]
    n_prompt, entry, min_p, tok = int(n_prompt), int(entry), float(min_p), int(tok)
    Ts = [float(t) for t in temps.split(",")]
    f = open(topk_path, "rb")
    magic, K, S, V = struct.unpack("<4I", f.read(16))
    assert magic == 0x44475445, "不是 ETGD 文件"
    row = struct.Struct("<Iiff" + "%di" % K + "%df" % K)
    g0 = n_prompt - 1
    seg = {k: {T: dict(n=0, out=0.0, top1=0.0, tail=0.0, tail_n=0) for T in Ts} for k in ("入环前", "入环后")}
    while True:
        buf = f.read(row.size)
        if len(buf) < row.size: break
        v = row.unpack(buf); i = v[0]
        if i < g0 or i + 1 >= len(ids): continue
        tid, pr = v[4:4 + K], v[4 + K:4 + 2 * K]
        mass = v[3]; pmin = min(pr)
        s = seg["入环后" if i - g0 >= entry else "入环前"]
        for T in Ts:
            q = [p ** (1.0 / T) for p in pr]; Z = sum(q)
            q1 = max(q) / Z; gate = min_p * q1
            kept = sum(x for x in q if x / Z >= gate) / Z
            st = s[T]; st["n"] += 1; st["top1"] += q1
            # 尾巴上界: 每个 K 外 p ≤ pmin ⇒ p^(1/T) ≤ p·pmin^(1/T−1); 归一分母至少是 Z
            if mass < 1.0 and pmin > 0:
                st["tail"] += (1.0 - mass) * pmin ** (1.0 / T - 1.0) / Z; st["tail_n"] += 1
            for q_, t_ in zip(q, tid):
                if t_ == tok and q_ / Z >= gate: st["out"] += (q_ / Z) / kept
    print("ETGD K=%d; 生成 %d 位, 入环位 %d, min_p %.3f, 出口 tok %d; 温度 %s" % (K, len(ids) - n_prompt, entry, min_p, tok, temps))
    for name, byT in seg.items():
        if not byT[Ts[0]]["n"]: continue
        print("== %s %d 位" % (name, byT[Ts[0]]["n"]))
        print("   %5s | %9s %10s | %8s | %s" % ("T", "出口Σp", "一次不出", "均top-1", "K外尾巴/位 上界"))
        for T in Ts:
            st = byT[T]
            print("   %5.2f | %9.4f %10.3g | %8.3f | %.2e" % (T, st["out"], math.exp(-st["out"]), st["top1"] / st["n"],
                  st["tail"] / max(1, st["tail_n"])))


def cmd_entq(tsv_path, out_dir):
    """实体特征知识召回尺的题单(2026-11-01, fable5 "不像鹈鹕 = 实体特征知识召回位被压"): 实体表(类别 中文 英文) → 三份 q.jsonl(api_teacher.py collect 的输入)。
    留出 = bird 整类 + 其余每类类内序号 % 5 == 4 的实体, 永不进料, 只量; 训练 = 其余实体 × 三问(中文外形 / 英文外形 / 中文"简笔画必须画对什么");
    训练探针 = 训练实体里类内序号 % 5 == 0 的, 只问中文外形 —— 和留出一起量, 分清"只有练过的实体动"与"没练过的也动"。
    id = 类别|中文|语言|问号, ptfprobe 靠它分组; 题面里不出现"鸟"字以外的禁词由 entity_probe_spark.sh 用 evo_ban.txt 再筛一遍。"""
    ents = []
    for ln in open(tsv_path, encoding="utf-8"):
        if not ln.strip() or ln.startswith("#"): continue
        cat, zh, en = ln.rstrip("\n").split("\t"); ents.append((cat, zh, en))
    q_zh = "请描述{zh}的外形特征：先说它最有辨识度的部位和各部位的大致比例，再说它和容易混淆的近似事物的区别。不超过 200 字。"
    q_en = "Describe the physical appearance of a {en}: first its most distinctive parts and their rough proportions, then how it differs from things it is commonly confused with. Under 150 words."
    q_dr = "如果要用简笔画画一个{zh}，必须画对哪几个特征才能一眼认出来？按重要性列出，并说明每个特征的大小比例。不超过 200 字。"
    idx = {}; out = {"train": [], "hold": [], "ptrain": []}
    for cat, zh, en in ents:
        k = idx.get(cat, 0); idx[cat] = k + 1
        hold = cat == "bird" or k % 5 == 4
        mk = lambda lang, qk, q: {"id": f"{cat}|{zh}|{lang}|{qk}", "q": q.format(zh=zh, en=en)}
        if hold:
            out["hold"] += [mk("zh", "q1", q_zh), mk("en", "q2", q_en)]
        else:
            out["train"] += [mk("zh", "q1", q_zh), mk("en", "q2", q_en), mk("zh", "q3", q_dr)]
            if k % 5 == 0: out["ptrain"].append(mk("zh", "q1", q_zh))
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in out.items():
        with open(os.path.join(out_dir, f"{name}_q.jsonl"), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nh = sum(1 for c, _, _ in ents if c == "bird"); print("entq: 实体 %d(鸟 %d); 训练题 %d / 留出题 %d / 训练探针题 %d → %s" % (len(ents), nh, len(out["train"]), len(out["hold"]), len(out["ptrain"]), out_dir))


def cmd_ptfhead(ptf_in, ptf_out, n="40"):
    """PTF1 表取前 n 条另存(2026-11-01 测量分叉排查: 训练器报留出损失 −57%、引擎挂同一 ③ 在同类行上 KL 不动, 拿【训练过的行】本身在引擎上量一遍)。"""
    import numpy as np
    recs = read_ptf(ptf_in)[:int(n)]
    with open(ptf_out, "wb") as f:
        f.write(struct.pack("<III", 0x31465450, recs[0][1].shape[1], len(recs)))
        for ids, ti, tp, rest in recs:
            f.write(struct.pack("<I", len(ids))); np.asarray(ids, np.int32).tofile(f); np.asarray(ti, np.int32).tofile(f); np.asarray(tp, np.float32).tofile(f); np.asarray(rest, np.float32).tofile(f)
    print("ptfhead: %s 前 %d 条(%d 位) → %s" % (ptf_in, len(recs), sum(len(r[0]) for r in recs), ptf_out))


def cmd_entlong(rows_path, ok_path, out_path, n_rows="100", n_ctx="14", seed="7", step="3"):
    """实体训练行加长上下文前缀(2026-11-01 晚, fable5 "修正只在流的前 ~1000 位起作用"): 从 train_rows.jsonl 每 3 行取 1 行当靶(每实体的中文外形题),
    前缀 = n_ctx 个【别的实体】的问答按部署多轮格式拼起来(<｜User｜>题<｜Assistant｜></think>答<｜end▁of▁sentence｜>…)再接原题头, text 不变。
    靶的 token 串与短行逐位同(前缀以特殊 token 收尾, BPE 不跨界), 教师表用 ptfprefix 从短行的表平移, 不重打 API。
    n_ctx 可以是逗号表(11-02, 如 0,12,40): 第 i 行取第 i % 表长 个 —— 短/中/长三档各占 1/3。为什么混: 只训 7k 长行(ent04)短上下文知识题退了(探针 大 0.80 → 0.55),
    "训多长管多长"反过来也成立, 部署里短问答和 10k agent 上下文都有, 料得两头都盖; 0 = 不加前缀(与短行逐位同, ptfprefix 当 P=1 平移)。"""
    import random
    import re
    USER, ASST, TE, EOS = "<｜User｜>", "<｜Assistant｜>", "</think>", "<｜end▁of▁sentence｜>"
    def ent_of(s):
        for pat in (r"请描述(.+?)的外形", r"appearance of a (.+?):", r"简笔画画一个(.+?)，"):
            m = re.search(pat, s)
            if m: return m.group(1)
        return s[:20]
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8") if l.strip()]
    oks = [json.loads(l) for l in open(ok_path, encoding="utf-8") if l.strip()]
    oks = [(o["id"].split("|")[1], USER + o["q"] + ASST + TE + o["content"].rstrip(" \r\n") + EOS) for o in oks if o.get("ok") and o.get("content")]
    rng = random.Random(int(seed)); n = 0; nctxs = [int(x) for x in str(n_ctx).split(",")]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows[::int(step)][:int(n_rows)]:
            nctx = nctxs[n % len(nctxs)]
            ent = ent_of(r["prefix"]); pool = [t for e, t in oks if e != ent]; rng.shuffle(pool)
            f.write(json.dumps({"prefix": "".join(pool[:nctx]) + r["prefix"], "text": r["text"], **({"eos": False} if r.get("eos") is False else {})}, ensure_ascii=False) + "\n"); n += 1
    print("entlong: %d 行, 每行前缀 %s 个别的实体问答(轮流) → %s" % (n, "/".join(map(str, nctxs)), out_path))


def cmd_ptfprefix(short_ptf, long_seqs, out_ptf):
    """短行教师表 → 长前缀行教师表: 长序列(simids 导出)以某条短记录去掉 BOS 后的 id 串结尾时, 记录 = 长序列 ids + 前缀段 one-hot + 短记录的行原样平移。
    为什么不重打 api_teacher.py ptf: 它按内容块找轨迹, 前缀里别的实体的答案会被先认领走(一条轨迹只认领一条序列), 长行自己反而配不上。
    对不上后缀的长序列直接停(训练器按整条序列哈希查表, 少一条就是少一道题的教师, 不许静默)。"""
    import numpy as np
    recs = read_ptf(short_ptf); K = recs[0][1].shape[1]
    by_tail = {}
    for ids, ti, tp, rest in recs: by_tail.setdefault(tuple(ids.tolist()[1:]), []).append((ids, ti, tp, rest))
    lens = sorted({len(k) for k in by_tail}, reverse=True)
    lines = [[int(x) for x in ln.split()] for ln in open(long_seqs, encoding="utf-8") if ln.strip()]
    out = []; tot = 0
    for seq in lines:
        hit = None
        for L in lens:
            if L < len(seq) and tuple(seq[-L:]) in by_tail: hit = (L, by_tail[tuple(seq[-L:])][0]); break
        if not hit: sys.exit("★长序列(%d 位)的尾巴在短表里找不到: 前缀与题头的分词跨界了, 或行序不对★" % len(seq))
        L, (sids, sti, stp, srest) = hit; P = len(seq) - L   # 长序列 = seq[:P](BOS + 前缀) + 短记录 ids[1:]
        n = len(seq) - 1
        ti = np.full((n, K), -1, np.int32); tp = np.zeros((n, K), np.float32); rest = np.zeros(n, np.float32)
        ti[:, 0] = seq[1:]; tp[:, 0] = 1.0   # 前缀段 one-hot(不是分布, 训练器只当上下文; ptfprobe 跳过)
        ti[P - 1:] = sti; tp[P - 1:] = stp; rest[P - 1:] = srest   # 短记录第 i 行预测 ids[i+1]; 长序列里 ids[P + i] = 短 ids[i + 1] ⇒ 行号 P − 1 + i
        out.append((np.asarray(seq, np.int32), ti, tp, rest)); tot += len(seq)
    with open(out_ptf, "wb") as f:
        f.write(struct.pack("<III", 0x31465450, K, len(out)))
        for ids, ti, tp, rest in out:
            f.write(struct.pack("<I", len(ids))); ids.tofile(f); ti.tofile(f); tp.tofile(f); rest.tofile(f)
    print("ptfprefix: %d 条长序列全部配上短表, 共 %d 位(均 %d) → %s" % (len(out), tot, tot // max(len(out), 1), out_ptf))


def cmd_ptfprobe(anchor_path, tokenizer_json, out_tsv, *ptfs):
    """★实体特征知识召回探针尺(2026-11-01)★: 学生全 logits 锚(--score-ids 跑 ptf 覆盖序列首尾相接的流) vs 官方 API top-20 表(ptf, 按给的顺序接起来就是流的顺序)。
    选位 = 教师给真值 ≥ 0.7 且真值 token 是内容词(含汉字, 或 ≥ 3 个字母)的位 —— 这些位是"实体长什么样"的事实位(鹈鹕序列上的 长/喉 就是这种位);
    报每条序列(实体 × 语言)上: 选位数 / 教师几何均值 / 学生几何均值 / 学生÷教师 / 选位上 top-1 同 / 全部分布位 KL(教师‖学生, top-20 + 余量桶, 训练器同口径)。
    再按 组(留出 / 训练探针) × 语言 汇总; 组别从题面里的实体名反查(ptf 里只有 ids, 没有标签): 第 5 个参数起每个 ptf 可写成 组名=路径。
    判读: 留出组 学生÷教师 往 1 靠 = 还原的是知识召回; 只有训练组动 = 低秩只会背。"""
    import math
    import re
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_json)
    f = open(anchor_path, "rb"); S, V = struct.unpack("<ii", f.read(8))
    cjk = re.compile("[一-鿿]")
    def content_word(t):
        s = tok.decode([int(t)])
        return bool(cjk.search(s)) or len(re.findall("[A-Za-z]", s)) >= 3
    user_id, asst_id = tok.token_to_id("<｜User｜>"), tok.token_to_id("<｜Assistant｜>")
    rows = []; off = 0; worst = []
    for spec in ptfs:
        grp, path = spec.split("=", 1) if "=" in spec else ("seq", spec)
        for ids, ti, tp, rest in read_ptf(path):
            L = len(ids); idl = ids.tolist()
            try: u, a = idl.index(user_id), idl.index(asst_id)
            except ValueError: u, a = 0, min(L, 32)
            prompt = tok.decode(idl[u + 1:a]); lang = "zh" if cjk.search(prompt) else "en"
            ent = prompt.split("描述")[1].split("的外形")[0] if "描述" in prompt and "的外形" in prompt else (prompt.split("appearance of a ")[1].split(":")[0] if "appearance of a " in prompt else prompt[:24])
            onehot = (tp[:, 0] > 0.999) & (tp[:, 1:].sum(1) < 1e-6) & (rest < 1e-6)
            if off + L > S: sys.exit("★锚只有 %d 行, 流到第 %d 行就超了: ptf 顺序或覆盖清单与打分的 ids 流不是同一串★" % (S, off + L))
            sel_t, sel_s, same, kl, nkl = [], [], 0, 0.0, 0
            for i in range(L - 1):
                if onehot[i]: continue
                y = idl[i + 1]; j = np.flatnonzero(ti[i] == y)
                f.seek(8 + (off + i) * V * 4); row = np.frombuffer(f.read(V * 4), dtype=np.float32).astype(np.float64)
                m = row.max(); ex = np.exp(row - m); p = ex / ex.sum()
                q = np.clip(tp[i].astype(np.float64), 1e-12, 1); ps = np.clip(p[ti[i]], 1e-12, 1)
                rq = max(float(rest[i]), 1e-12); rp = max(1.0 - float(p[ti[i]].sum()), 1e-12)
                kl += float((q * np.log(q / ps)).sum() + rq * math.log(rq / rp)); nkl += 1
                if len(j) == 0 or tp[i, j[0]] < 0.7 or not content_word(y): continue
                pt, pst = float(tp[i, j[0]]), float(p[y]); sel_t.append(pt); sel_s.append(pst); same += int(row.argmax() == y)
                worst.append((pst / pt, grp, ent, tok.decode([y]), pt, pst))
            off += L
            gm = lambda xs: math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs)) if xs else float("nan")
            rows.append((grp, lang, ent, len(sel_t), gm(sel_t), gm(sel_s), gm(sel_s) / gm(sel_t) if sel_t else float("nan"), same / max(len(sel_t), 1), kl / max(nkl, 1), nkl))
    with open(out_tsv, "w", encoding="utf-8") as o:
        o.write("组\t语言\t实体\t选位\t教师几何均值\t学生几何均值\t学生/教师\t选位top1同\t全位KL\t分布位\n")
        for r in rows: o.write("%s\t%s\t%s\t%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.4f\t%d\n" % r)
        o.write("# 汇总 组×语言: 序列数 / 选位合计 / 学生÷教师(按选位合并的几何均值比) / 选位 top-1 同 / 全位 KL 均\n")
        for grp in dict.fromkeys(r[0] for r in rows):
            for lang in ("zh", "en"):
                rs = [r for r in rows if r[0] == grp and r[1] == lang and r[3] > 0]
                if not rs: continue
                n = sum(r[3] for r in rs); lt = sum(math.log(r[4]) * r[3] for r in rs) / n; ls = sum(math.log(r[5]) * r[3] for r in rs) / n
                line = "%s\t%s\t%d 条\t%d 位\t%.3f\t%.3f\t%.4f" % (grp, lang, len(rs), n, math.exp(ls - lt), sum(r[7] * r[3] for r in rs) / n, sum(r[8] * r[9] for r in rs) / max(sum(r[9] for r in rs), 1))
                o.write("# " + line + "\n"); print(line)
        worst.sort()
        o.write("# 被压最狠的 15 位(学生/教师 组 实体 token 教师p 学生p):\n")
        for w in worst[:15]: o.write("#   %.3f\t%s\t%s\t%r\t%.3f\t%.4f\n" % w)
    print("ptfprobe: %d 条序列, 流 %d 行(锚 %d 行) → %s" % (len(rows), off, S, out_tsv))


if __name__ == "__main__":
    cmds = {"extract": cmd_extract, "cut": cmd_cut, "cmp-emit": cmd_cmp_emit, "cmp-topk": cmd_cmp_topk, "entq": cmd_entq, "ptfprobe": cmd_ptfprobe, "ptfhead": cmd_ptfhead, "entlong": cmd_entlong, "ptfprefix": cmd_ptfprefix,
            "eos-scan": cmd_eos_scan, "rows": cmd_rows, "loopstat": cmd_loopstat, "faith": cmd_faith,
            "anchor-faith": cmd_anchor_faith, "tokprobe": cmd_tokprobe, "minp": cmd_minp, "agree": cmd_agree,
            "ids2seqs": cmd_ids2seqs, "ptfcheck": cmd_ptfcheck, "ptfstat": cmd_ptfstat, "ptfpick": cmd_ptfpick, "ptfcmp": cmd_ptfcmp, "anchor2ptf": cmd_anchor2ptf, "ptfhit": cmd_ptfhit,
            "cyclescan": cmd_cyclescan, "temper": cmd_temper}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__ + "  rows <rows.txt> <ids> <n_prompt> <入口位置> [tokenizer.json] [窗口=12]   --row-out 的分桶命中率 + 入口逐行解码\n")
    cmds[sys.argv[1]](*sys.argv[2:])
