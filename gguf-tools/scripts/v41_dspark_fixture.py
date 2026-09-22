#!/usr/bin/env python3
"""v41_dspark_fixture.py — DSpark 草稿器的金标夹具(2026-09-18)。

【干什么】引擎 `--dspark-capture <out>` 顺带落的 `<out>.fix` 里, 每一对都存着草稿器当时吃的
main_hidden(主模型 L37/38/39 注意力输入的四路均值, 拼 15360 维)、块首位 token、它出的 5 位草稿
与 5 个 conf。本夹具只载**官方** mtp.* 三塔 + embed + head(FP8/BF16/FP4 原件 dequant 成 bf16,
与 v41_teacher.py 同口径), 按官方 `Transformer.forward_spec` 喂**同一份** main_hidden 逐位置重放,
逐位比 5 个草稿 id。

【判什么】
  ① 实现: 首位草稿 id 两边一致 ≥ 97%(近平局翻面之外) ⇒ 引擎三塔实现无错。09-18 实测 0.92, 错位全在
     半个 logit 以内, 出口隐态 cos 0.998 ⇒ 判"实现无错"; 官方三塔吃同一份量化隐态命中底座也只有 0.52。
  ② 输入(要 --fp-mainh, 教师 --dump-mainh 的产物): 同一批位置, 官方三塔分别喂 FP 隐态 / 量化隐态 /
     逐通道仿射校正后的量化隐态, 各自命中 FP argmax(--teacher)与底座 argmax(--pairs)多少;
     再给 cos(FP, 量化) 分档 —— 分"低维形状(能校)"与"高维噪声(只能重训塔)"。
为什么非要它: 修过口径的三个一致率(fable5 09-18)说草稿器在最好猜的位置也只中 76%(底座同档 97%),
而"塔算错了"与"输入把塔带偏了"两种病在 p1 上长得一模一样, 只有同一份输入喂两套实现才分得开。

【定位】金标夹具生成器 —— 全仓零 Python 裁决的例外; 不进任何数值链, 产物只是一份报告。

【口径】第 p 对 = 主模型位置 p+1: 引擎用 main_hidden(p) + tok = ids[p+1] 出草稿, 块首位坐在位置 p+1
  ⇒ 官方 forward_spec(input_ids=[[tok]], main_hidden(p), start_pos=p)。p = 0 只种窗口(官方
  start_pos==0 分支, 返回 None), 从 p = 1 起比 —— 与引擎在线路 v41_draft_step(tok, pos_main=p) 一字对应。
  教师 logits 第 r 行 = 位置 r 的 logits(预测 r+1) ⇒ 第 p 对的 FP 靶 = 第 p+1 行的 argmax。
  温度写死 0(引擎草稿是 argmax)。窗口(main_kv 环)靠逐位置顺序调用自然维护, 所以**不许跳位置**。

用法: v41_dspark_fixture.py <hf-dir> --fix <pairs.bin.fix> [--pairs <pairs.bin>] [--fp-mainh <f32 [S][15360]>]
      [--teacher <S><V><logits>] [--n N] [--split 0.7] [--out report.txt]
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v41_hf_io import FP4_TABLE, build_index, load_raw, dq_fp8, dq_fp4   # noqa: E402
import v41_patches as P                                                    # noqa: E402


def build_net(hf, dev, nt, d, blk, N):
    """官方模型: meta 建整模, 只实体化 mtp.* + embed + head(主干 40 层永远不碰)。"""
    idx = build_index(hf)
    import model as M
    P.patch_linear(M)          # dequant + fp32 matmul, 与教师同一条前向路
    P.patch_sparse_attn(M)     # 官方 kernel 要 141 KB shared, GB10 跑不了
    cfg = json.loads((hf / "inference" / "config.json").read_text())
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})
    args.temperature = 0.0                       # 引擎草稿 = argmax
    if args.max_seq_len < N + 2 + blk:           # freqs_cis 要盖住 start_pos + 1 + block
        args.max_seq_len = N + 2 + blk
    if args.dspark_block_size != blk or len(args.dspark_target_layer_ids) != nt or args.dim != d:
        raise SystemExit(f"config 与料对不上: block {args.dspark_block_size}/{blk} 目标层 "
                         f"{len(args.dspark_target_layer_ids)}/{nt} dim {args.dim}/{d}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf), trust_remote_code=True)
    with torch.device("meta"):
        net = M.Transformer(args, tok)
    nfix, nfreq = P.fix_buffers(M, net, args, tok, dev)
    print(f"[建模] meta + 修 {nfix} 个 buffer({nfreq} 个 freqs_cis)", flush=True)
    tbl = torch.tensor(FP4_TABLE, dtype=torch.float32, device=dev)
    nload = 0
    for full, _ in list(net.named_parameters()):
        if not full.startswith(("mtp.", "embed.", "head.")) or full.endswith(".scale"):
            continue
        if full not in idx:
            raise SystemExit(f"HF 里没有 {full}(mtp 该在第 44~46 分片)")
        t = load_raw(idx, full, dev)
        sn = full.rsplit(".", 1)[0] + ".scale"
        was_q = sn in idx
        if was_q:      # fp4 专家 / fp8 投影: dequant 成 bf16, 与教师 materialize 同一段逻辑(无损)
            sc = load_raw(idx, sn, dev)
            t = dq_fp4(t, sc, tbl) if idx[full][2] == "I8" else dq_fp8(t, sc)
            t = t.to(torch.bfloat16)
        sub, _, leaf = full.rpartition(".")
        owner = net.get_submodule(sub) if sub else net
        old = getattr(owner, leaf)
        if old.dtype in (torch.float32, torch.bfloat16, torch.float16):
            t = t.to(old.dtype)                  # ParallelHead 声明 f32, 其余 bf16 —— 保持模型声明的 dtype
        par = torch.nn.Parameter(t, requires_grad=False)
        par._was_quant = was_q                   # linear_dq 据此按官方 kernel 口径出 bf16
        setattr(owner, leaf, par)
        nload += 1
    torch.cuda.synchronize()
    print(f"[装载] mtp.*/embed/head 共 {nload} 个张量", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf", help="原始 HF 目录(要 FP8/BF16 原件的 mtp.*, 不是量化目录)")
    ap.add_argument("--fix", required=True, help="引擎 --dspark-capture 落的 <out>.fix")
    ap.add_argument("--pairs", default="", help="同一次取料的 <out>(DCAP): 底座 argmax + 引擎出口隐态 X")
    ap.add_argument("--fp-mainh", default="", help="教师 --dump-mainh 的 f32[S][nt*d]: 同一批位置的 FP main_hidden")
    ap.add_argument("--teacher", default="", help="教师 --out 的 <S><V><logits>: FP argmax 靶")
    ap.add_argument("--n", type=int, default=0, help="只重放前 N 对(0=全部)")
    ap.add_argument("--split", type=float, default=0.7, help="逐通道仿射校正: 前 split 拟合, 其余留出评估")
    ap.add_argument("--fp-only", action="store_true",
                    help="只重放 --fp-mainh 那一种输入(取法验证: 同一份料换四种 main_hidden 候选各跑一遍, 不必每次再跑量化/校正两趟)")
    ap.add_argument("--out", default="", help="报告落盘路径(不给只打屏)")
    a = ap.parse_args()

    hf = Path(a.hf)
    sys.path.insert(0, str(hf / "inference"))
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)   # 与官方 generate.py / v41_teacher.py 同
    dev = "cuda"

    # ---- 料 ----
    with open(a.fix, "rb") as f:
        magic = f.read(4)
        nt, d, blk, n = struct.unpack("<IIII", f.read(16))
    if magic != b"DFIX":
        raise SystemExit(f"{a.fix} 不是夹具料(头 {magic!r})")
    rec = nt * d * 4 + 4 + blk * 4 + blk * 4
    mm = np.memmap(a.fix, dtype=np.uint8, mode="r", offset=20)
    if mm.shape[0] < n * rec:
        raise SystemExit(f"夹具料截断: 头写 {n} 对, 文件只够 {mm.shape[0] // rec} 对")
    N = min(n, a.n) if a.n else n
    print(f"[料] {a.fix}: {n} 对, main_hidden {nt}×{d}, 块 {blk}; 重放前 {N} 对", flush=True)
    DM = nt * d

    def rec_at(p):
        b = mm[p * rec:(p + 1) * rec].tobytes()
        o = DM * 4
        mh = np.frombuffer(b[:o], dtype=np.float32)
        tok = int(np.frombuffer(b[o:o + 4], dtype=np.int32)[0])
        ids = np.frombuffer(b[o + 4:o + 4 + blk * 4], dtype=np.int32)
        conf = np.frombuffer(b[o + 4 + blk * 4:], dtype=np.float32)
        return mh, tok, ids, conf

    qmh = np.stack([rec_at(p)[0] for p in range(N)]).astype(np.float64)      # 量化 main_hidden [N][DM]
    toks = np.array([rec_at(p)[1] for p in range(N)], dtype=np.int64)
    eids = np.stack([rec_at(p)[2] for p in range(N)]).astype(np.int64)      # 引擎草稿 [N][blk]
    econf = np.stack([rec_at(p)[3] for p in range(N)]).astype(np.float64)

    mtok = xmm = None
    if a.pairs:   # 同一次取料的 DCAP: 尾部底座 argmax + 每对的引擎出口隐态 X
        with open(a.pairs, "rb") as f:
            pm = f.read(4)
            pd_, pn, ppos0 = struct.unpack("<III", f.read(12))
            if pm != b"DCAP" or pd_ != d or pn != n:
                raise SystemExit(f"{a.pairs} 与夹具料不是同一次取料(D {pd_}/{d} n {pn}/{n})")
            f.seek(16 + pn * d * 2 * 4)
            mtok = np.frombuffer(f.read(pn * 4), dtype=np.int32).astype(np.int64)
        xmm = np.memmap(a.pairs, dtype=np.float32, mode="r", offset=16, shape=(pn, 2 * d))
        print(f"[料] {a.pairs}: 底座 argmax {pn} 个, pos0 {ppos0}", flush=True)
    fpmh = None
    if a.fp_mainh:
        fpmh = np.fromfile(a.fp_mainh, dtype=np.float32)
        if fpmh.size < (N + 1) * DM:
            raise SystemExit(f"{a.fp_mainh} 只有 {fpmh.size // DM} 位置, 要 ≥ {N + 1}")
        fpmh = fpmh.reshape(-1, DM)[:N].astype(np.float64)   # 第 p 行 = 位置 p = 第 p 对吃的那份
        print(f"[料] {a.fp_mainh}: FP main_hidden 前 {N} 位置", flush=True)
    ftok = None
    if a.teacher:
        with open(a.teacher, "rb") as f:
            S, V = struct.unpack("<ii", f.read(8))
        if S < N + 1:
            raise SystemExit(f"{a.teacher} 只有 {S} 行, 要 ≥ {N + 1}")
        lg = np.memmap(a.teacher, dtype=np.float32, mode="r", offset=8, shape=(S, V))
        ftok = np.array([int(np.argmax(lg[p + 1])) for p in range(N)], dtype=np.int64)   # 第 p 对的靶 = 第 p+1 行
        print(f"[料] {a.teacher}: FP argmax 前 {N} 对(第 1..{N} 行)", flush=True)

    t0 = time.time()
    net = build_net(hf, dev, nt, d, blk, N)
    print(f"[装载] {time.time()-t0:.1f}s", flush=True)
    cap = {}   # 官方出口隐态(norm 之后、head 之前, 与引擎取料的 X 同一个取点)
    net.mtp[-1].norm.register_forward_hook(lambda _m, _i, o: cap.__setitem__("xn", o))
    torch.set_default_device(dev)   # 官方代码里有不带 device 的 arange/zeros

    def replay(src, tag):
        """按顺序重放前 N 对, src[p] 是第 p 对喂的 main_hidden(f64 [DM])。返回 got[N][blk](第 0 对无效=-1)、
        conf[N][blk]、logits 第 0 行的 (top1, 引擎选的那个) 差、官方出口隐态。"""
        for mod in net.mtp:   # 窗口 main_kv 环清零: 换一份输入就是另一条序列
            mod.attn.window_kv_cache.zero_()
        got = np.full((N, blk), -1, dtype=np.int64)
        conf = np.zeros((N, blk))
        margin = np.zeros(N)
        xo = np.zeros((N, d), dtype=np.float32)
        t1 = time.time()
        for p in range(N):
            main_hidden = torch.from_numpy(src[p].astype(np.float32)).to(dev).to(torch.bfloat16).view(1, 1, DM)
            input_ids = torch.tensor([[int(toks[p])]], device=dev, dtype=torch.long)
            res = net.forward_spec(input_ids, main_hidden, p)
            if res is None:
                continue                              # p == 0: 官方只种窗口
            out_ids, logits, cf = res
            got[p] = np.array(out_ids[0, 1:1 + blk].tolist())
            conf[p] = cf[0].float().cpu().numpy()[:blk]
            lg0 = logits[0, 0].float()
            margin[p] = float(lg0[int(got[p, 0])] - lg0[int(eids[p, 0])])
            xo[p] = cap["xn"][0, 0].float().cpu().numpy()
            if (p + 1) % 256 == 0:
                print(f"[重放·{tag}] {p+1} 对 ({time.time()-t1:.0f}s)", flush=True)
        torch.cuda.synchronize()
        return got, conf, margin, xo

    def rate(x, y, sel):
        return float(np.mean(x[sel] == y[sel])) if sel.any() else float("nan")

    valid = np.arange(N) >= 1
    lines = []
    # FP 间隔分档(与 dspark_agree 同口径): 最易档(≥ 6 logit)的命中最能说明"塔本身准不准"
    easy = None
    if a.teacher:
        with open(a.teacher, "rb") as f:
            S, V = struct.unpack("<ii", f.read(8))
        lg = np.memmap(a.teacher, dtype=np.float32, mode="r", offset=8, shape=(S, V))
        mg = np.array([float(np.subtract(*np.sort(np.asarray(lg[p + 1]))[-2:][::-1])) for p in range(N)])
        easy = valid & (mg >= 6.0)

    def bucket_line(name, g):
        s = f"      {name:14s} ↔FP {rate(g[:, 0], ftok, valid):.4f} / ↔底座 {rate(g[:, 0], mtok[:N], valid):.4f}"
        if easy is not None:
            s += f"   最易档(≥6, {int(easy.sum())} 对): ↔FP {rate(g[:, 0], ftok, easy):.4f} / ↔底座 {rate(g[:, 0], mtok[:N], easy):.4f}"
        return s

    if a.fp_only:
        if fpmh is None or ftok is None or mtok is None:
            raise SystemExit("--fp-only 要同时给 --fp-mainh / --teacher / --pairs")
        gf, _, _, _ = replay(fpmh, "候选隐态")
        lines.append(f"[取法验证] 官方三塔喂 {a.fp_mainh}, {int(valid.sum())} 对:")
        lines.append(bucket_line("候选隐态", gf))
        lines.append(bucket_line("引擎草稿(量化)", eids))
        rep = "\n".join(lines)
        print(rep, flush=True)
        if a.out:
            Path(a.out).write_text(rep + "\n")
        return

    # ---- ① 实现: 官方 vs 引擎, 同一份量化隐态 ----
    gq, cq, mq, xo = replay(qmh, "量化隐态")
    lines.append(f"[fixture] 官方三塔 vs 引擎三塔, 同一份(量化)main_hidden, {int(valid.sum())} 对")
    for j in range(blk):
        lines.append(f"  第 {j+1} 位草稿 id 一致率 = {rate(gq[:, j], eids[:, j], valid):.4f}")
    lines.append(f"  conf 平均绝对差 = {np.abs(cq[valid] - econf[valid]).mean():.4f}(logit 口径)")
    mis = valid & (gq[:, 0] != eids[:, 0])
    if mis.any():
        lines.append(f"  首位错位 {int(mis.sum())} 对: 官方 top1 比引擎所选高 中位 {np.median(mq[mis]):.2f} / "
                     f"p90 {np.percentile(mq[mis], 90):.2f} logit; 差 > 2 的 {int((mq[mis] > 2).sum())} 对(近平局翻面该在 1 以内)")
    if xmm is not None:
        xe = np.asarray(xmm[:N, :d], dtype=np.float64)
        cs = np.einsum("ij,ij->i", xo.astype(np.float64), xe) / (np.linalg.norm(xo, axis=1) * np.linalg.norm(xe, axis=1) + 1e-30)
        lines.append(f"  出口隐态 cos(官方, 引擎): 均值 {cs[valid].mean():.4f} / p10 {np.percentile(cs[valid], 10):.4f} / min {cs[valid].min():.4f}")
    if mtok is not None:
        lines.append(f"  ★首位 ↔ 底座 argmax: 官方三塔 {rate(gq[:, 0], mtok[:N], valid):.4f} / 引擎三塔 {rate(eids[:, 0], mtok[:N], valid):.4f}★")
    if ftok is not None:
        lines.append(f"  ★首位 ↔ FP argmax(教师): 官方三塔 {rate(gq[:, 0], ftok, valid):.4f} / 引擎三塔 {rate(eids[:, 0], ftok, valid):.4f}"
                     f" / 底座 ↔ FP {rate(mtok[:N], ftok, valid) if mtok is not None else float('nan'):.4f}★")

    # ---- ② 输入: FP 隐态 / 逐通道仿射校正 ----
    if fpmh is not None:
        dq = qmh - fpmh
        cos_pos = np.einsum("ij,ij->i", qmh, fpmh) / (np.linalg.norm(qmh, axis=1) * np.linalg.norm(fpmh, axis=1) + 1e-30)
        rel = np.linalg.norm(dq, axis=1) / (np.linalg.norm(fpmh, axis=1) + 1e-30)
        lines.append(f"[输入] cos(量化 main_hidden, FP): 均值 {cos_pos[valid].mean():.4f} / p10 {np.percentile(cos_pos[valid], 10):.4f};"
                     f" 相对误差 ‖q−fp‖/‖fp‖ 中位 {np.median(rel[valid]):.4f}")
        for s in range(nt):   # 分目标层看: 越靠后的层偏得越多?
            sl = slice(s * d, (s + 1) * d)
            c = np.einsum("ij,ij->i", qmh[:, sl], fpmh[:, sl]) / (np.linalg.norm(qmh[:, sl], axis=1) * np.linalg.norm(fpmh[:, sl], axis=1) + 1e-30)
            lines.append(f"    目标层槽 {s}: cos 均值 {c[valid].mean():.4f}")
        # 误差的"形状": 逐通道能量占比 —— 前 1% 通道占了多少误差能量(高 = 巨值通道那种低维病)
        ech = (dq[valid] ** 2).sum(0)
        top = np.sort(ech)[::-1]
        lines.append(f"    误差能量: 前 1% 通道占 {top[:DM//100].sum()/ech.sum():.3f}, 前 10% 占 {top[:DM//10].sum()/ech.sum():.3f}")
        # 逐通道仿射校正: 前 split 拟合 fp[c] ≈ a_c·q[c] + b_c, 留出评估
        nfit = int(N * a.split)
        fit = np.arange(N) < nfit
        held = valid & ~fit
        qf, ff = qmh[fit], fpmh[fit]
        qm, fm = qf.mean(0), ff.mean(0)
        var = ((qf - qm) ** 2).mean(0) + 1e-12
        acoef = ((qf - qm) * (ff - fm)).mean(0) / var
        bcoef = fm - acoef * qm
        corr = qmh * acoef + bcoef
        r2 = 1.0 - ((corr[held] - fpmh[held]) ** 2).sum() / ((fpmh[held] - fpmh[held].mean(0)) ** 2).sum()
        cos_c = np.einsum("ij,ij->i", corr, fpmh) / (np.linalg.norm(corr, axis=1) * np.linalg.norm(fpmh, axis=1) + 1e-30)
        lines.append(f"    逐通道仿射校正(前 {nfit} 对拟合, 留出 {int(held.sum())} 对): 留出 cos {cos_pos[held].mean():.4f} → {cos_c[held].mean():.4f}, "
                     f"留出 R² {r2:.4f}(1 = 全是逐通道形状, ≈0 = 高维噪声)")
        gf, _, _, _ = replay(fpmh, "FP 隐态")
        gc, _, _, _ = replay(corr, "校正隐态")
        lines.append(f"[输入] 官方三塔在留出 {int(held.sum())} 对上的首位命中(同一批位置三种输入):")
        hdr = "      %-14s %10s %10s" % ("输入", "↔FP argmax", "↔底座 argmax")
        lines.append(hdr)
        for nm, g in (("量化隐态", gq), ("FP 隐态", gf), ("逐通道校正", gc)):
            a_fp = rate(g[:, 0], ftok, held) if ftok is not None else float("nan")
            a_bs = rate(g[:, 0], mtok[:N], held) if mtok is not None else float("nan")
            lines.append("      %-14s %10.4f %10.4f" % (nm, a_fp, a_bs))
        lines.append("      读法: FP 隐态那一行 = 出厂草稿器的真实水平(官方说的那个 0.8 档); 量化隐态那一行掉多少就是输入的账;"
                     " 校正那一行追回多少 = 输入侧闭式校正这条路的肉。")
    rep = "\n".join(lines)
    print(rep, flush=True)
    if a.out:
        Path(a.out).write_text(rep + "\n")
        print(f"  → {a.out}")


if __name__ == "__main__":
    main()
