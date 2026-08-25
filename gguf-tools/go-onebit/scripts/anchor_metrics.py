#!/usr/bin/env python3
"""anchor_metrics.py — 锚(FP 参考)与学生 logits 的标准指标(R30 起, 2026-08-02 用户令:
本轮必须加入 PPL / Mean KLD / RMS Δp / Same top token; Bit-exact 由 bitexact_check.py 管)。

参考分布 = ds4quant_run 锚文件里的 logits[S][VOCAB](0731 HF 全 FP 前向, 判决基线)。
学生分布 = DS4_DUMP_LOGITS 落盘的 [S][VOCAB](量化域)——两者同 ids 同口径, 可逐位置对齐。

指标口径(与 llama.cpp --kl-divergence 报表同义):
  PPL        exp(mean NLL), teacher-forced, 预测 ids[i+1]
  Mean KLD   mean_i Σ_v p_ref·(ln p_ref − ln p_stu)
  RMS Δp     sqrt(mean_{i,v} (p_ref − p_stu)²)  ——报告 ×100(百分点)
  Same top   mean_i [argmax ref == argmax stu]
另报 top-token 概率差 Δp(top) 的均值±分位(退化最灵敏: 复读/塌缩会把 top 概率推向 1)。

用法:
  参考自检(锚冒烟判决): anchor_metrics.py --ref anchor.bin --ids ids.txt
  五指标(带学生):       anchor_metrics.py --ref anchor.bin --ids ids.txt --student stu.bin [--fit 933]
      --fit N: 前 N 位置为拟合段, 只报 held(N..S-1)段与全段两行(校准过拟合会在 held 现形)
"""
import argparse, struct, sys
import numpy as np


def read_anchor_logits(path):
    with open(path, 'rb') as f:
        hd = struct.unpack('<8I', f.read(32))
        if hd[0] != 0x32415144:
            sys.exit(f'{path}: 不是 DQA2 锚')
        _, S, HCM, DIM, NL, VOCAB, NACT, _ = hd
        f.read(8)  # idh
        # 跳 fin/ridx/rw/H, 直取 logits
        skip = (np.int64(NL) * S * DIM * 4 + np.int64(NL) * S * NACT * 4 * 2
                + np.int64(NL) * S * HCM * DIM * 4)
        f.seek(skip, 1)
        lg = np.fromfile(f, dtype=np.float32, count=S * VOCAB)
        if lg.size != S * VOCAB:
            sys.exit(f'{path}: logits 截断 ({lg.size} != {S}x{VOCAB})')
    return lg.reshape(S, VOCAB), dict(S=S, HCM=HCM, DIM=DIM, NL=NL, VOCAB=VOCAB, NACT=NACT)


def read_student_logits(path):
    with open(path, 'rb') as f:
        S, V = struct.unpack('<2i', f.read(8))
        lg = np.fromfile(f, dtype=np.float32, count=S * V)
    if lg.size != S * V:
        sys.exit(f'{path}: 学生 logits 截断')
    return lg.reshape(S, V)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def ppl_block(lg, ids, lo, hi):
    """teacher-forced: 位置 i 预测 ids[i+1], i∈[lo,hi)∩[0,S-1)"""
    hi = min(hi, len(ids) - 1)
    if hi <= lo:
        return float('nan'), 0
    ls = log_softmax(lg[lo:hi])
    nll = -ls[np.arange(hi - lo), ids[lo + 1:hi + 1]]
    return float(np.exp(nll.mean())), hi - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', help='DQA2 锚文件(全量)')
    ap.add_argument('--ref-raw', help='裸 [S,VOCAB] logits(锚被裁剪后的 ref_logits.bin; 与学生 dump 同格式)')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--student')
    ap.add_argument('--fit', type=int, default=0)
    ap.add_argument('--tail', type=int, default=0, help='尾部报表: 列 held 段 ΔNLL 最差 N token+集中度(PPL 差归因)')
    a = ap.parse_args()
    if not a.ref and not a.ref_raw:
        sys.exit('需 --ref 或 --ref-raw')

    ids = np.array([int(t) for t in open(a.ids).read().split()], dtype=np.int64)
    if a.ref_raw:
        ref = read_student_logits(a.ref_raw)
        meta = dict(S=ref.shape[0], VOCAB=ref.shape[1], NL='-', HCM='-', NACT='-')
    else:
        ref, meta = read_anchor_logits(a.ref)
    S = meta['S']
    ids = ids[:S]
    print(f"锚: S={S} VOCAB={meta['VOCAB']} NL={meta['NL']} HCM={meta['HCM']} NACT={meta['NACT']}")

    segs = [('全段', 0, S)]
    if a.fit and a.fit < S:
        segs = [('held', a.fit, S), ('全段', 0, S)]

    for nm, lo, hi in segs:
        p, n = ppl_block(ref, ids, lo, hi)
        print(f"参考 PPL[{nm} n={n}] = {p:.4f}")
    rt = ref.argmax(axis=1)
    acc = float((rt[:-1] == ids[1:S]).mean())
    print(f"参考 next-token top-1 命中 = {acc*100:.1f}%   (FP 模型对本语料的自然可预测度)")

    if not a.student:
        # 冒烟判决: FP 参考 PPL 在正常语言模型范围 = reader/前向对 0731 正确
        p_all, _ = ppl_block(ref, ids, 0, S)
        ok = np.isfinite(p_all) and 1.0 < p_all < 30.0
        print('★冒烟判决: PASS(前向/读权重正确)★' if ok else f'★冒烟判决: FAIL(PPL={p_all})★')
        sys.exit(0 if ok else 1)

    stu = read_student_logits(a.student)
    if stu.shape != ref.shape:
        sys.exit(f'形状不齐: ref{ref.shape} stu{stu.shape}')

    for nm, lo, hi in segs:
        n = hi - lo
        kld = np.empty(n); rms = np.empty(n); same = np.empty(n, bool); dtop = np.empty(n)
        for k in range(lo, hi, 256):     # 分块: 129280 vocab 全展开一次吃 1.3G, 按 256 位置流式
            j = min(k + 256, hi)
            lr = log_softmax(ref[k:j]); lsu = log_softmax(stu[k:j])
            pr = np.exp(lr); ps = np.exp(lsu)
            kld[k - lo:j - lo] = (pr * (lr - lsu)).sum(axis=1)
            # ★RMS Δp 改 top-K 联合窗口(2026-08-03 用户纠: 全词表均方被 12.9 万近零项冲稀无信息量):
            #   每 token 取 ref∪stu 各 top32 的并集, 在窗口内求均方 — 量的是"分布头部的形变"。
            d2 = np.empty(j - k)
            for t in range(j - k):
                idx = np.union1d(np.argpartition(pr[t], -32)[-32:], np.argpartition(ps[t], -32)[-32:])
                d2[t] = ((pr[t, idx] - ps[t, idx]) ** 2).mean()
            rms[k - lo:j - lo] = d2
            ir = pr.argmax(axis=1); iu = ps.argmax(axis=1)
            same[k - lo:j - lo] = ir == iu
            dtop[k - lo:j - lo] = np.abs(pr[np.arange(j - k), ir] - ps[np.arange(j - k), ir])
        sp, _ = ppl_block(stu, ids, lo, hi)
        rp, _ = ppl_block(ref, ids, lo, hi)
        print(f"\n== {nm} (n={n}) ==")
        print(f"  PPL(student)    = {sp:.4f}   (ref {rp:.4f}, 比值 {sp/rp:.3f})")
        print(f"  Mean KLD        = {kld.mean():.5f}   (中位 {np.median(kld):.5f}, p95 {np.quantile(kld,0.95):.5f})")
        print(f"  RMS Δp(top32窗) = {np.sqrt(rms.mean())*100:.4f}%")
        print(f"  Same top token  = {same.mean()*100:.2f}%")
        print(f"  Δp(ref top tok) = {dtop.mean():.4f} (p95 {np.quantile(dtop,0.95):.4f})")
    if a.tail:
        lo, hi = (a.fit, S) if a.fit and a.fit < S else (0, S)
        hi2 = min(hi, len(ids) - 1)
        lr = log_softmax(ref[lo:hi2]); lsu = log_softmax(stu[lo:hi2])
        tgt = ids[lo + 1:hi2 + 1]
        nr = -lr[np.arange(hi2 - lo), tgt]; ns = -lsu[np.arange(hi2 - lo), tgt]
        d = ns - nr
        order = np.argsort(-d)[:a.tail]
        gap = float(d.sum())
        print(f"\n== 尾部报表(held ΔNLL=stu−ref, 总log差={gap:.2f}, 均值={gap/(hi2-lo):.4f}) ==")
        print(f"  top{a.tail} token 承担全部 PPL 差的 {float(d[order].sum())/max(gap,1e-9)*100:.0f}%")
        for i in order:
            rt = int(lr[i].argmax()); st = int(lsu[i].argmax())
            print(f"  pos={lo+int(i)} tgt={int(tgt[i])} dNLL={d[i]:+.3f} (ref{nr[i]:.2f}->stu{ns[i]:.2f}) refTop={rt} stuTop={st}{chr(32)+chr(9733) if ns[i]>6 else chr(32)}")


if __name__ == '__main__':
    main()
