#!/usr/bin/env python3
"""emit_coding_imatrix.py — 从 HF 前向编程 token 产 legacy .dat imatrix (编程域激活 E[x²])。

deepseek4-quantize --imatrix 用它做 go1b_blk_quantize_imat 的激活感知 per-block scale
(s* = Σ ew_j·|w_j| / Σ ew_j = 输出最优), 即 full_optimize 79% 那条 per-block 输出最优的
可部署对应版。每层三张量:
  blk.L.ffn_gate_exps.weight  ew=E[Fin²]   (输入=层输入 Fin, 全专家共享, 忠实)
  blk.L.ffn_up_exps.weight    ew=E[Fin²]
  blk.L.ffn_down_exps.weight  ew=E[hf²]    (输入=per-expert hf, 取激发专家平均, 近似)
.dat 格式: i32 n_entries; 每条 i32 name_len/name/i32 n_call/i32 n_val/f32 values[](累积和, load 时 /n_call)。

用法(M1): DS4_HF=... python3 emit_coding_imatrix.py --ids CODE.ids [--ntok 128] --out coding.imatrix
"""
import argparse, os, struct, sys, time
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "calib", "pyfwd"))
sys.path.insert(0, os.path.join(_here, "..", "quant"))
import dsv4_fwd as F   # noqa: E402
from full_optimize import layer_attn_to_fin, NACT   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    ids = np.array([int(x) for x in open(a.ids) if x.strip()], dtype=np.int64)[:a.ntok]
    Ids = ids
    emb = F.R.get("embed.weight"); HCM = F.HCM
    H = np.repeat(emb[ids][:, None, :], HCM, 1).astype(np.float32)

    entries = []   # (name, n_call, values sum)
    print(f"前向 {hi+1} 层累积编程 E[x²]...", file=sys.stderr, flush=True)
    for L in range(hi + 1):
        W = F.load_layer(L); t0 = time.time()
        Fin, ctx = layer_attn_to_fin(H, W, L)
        T = Fin.shape[0]
        # gate/up: 输入 = Fin (全专家共享)
        fin_sq = (Fin.astype(np.float64) ** 2).sum(0).astype(np.float32)   # sum over tokens
        entries.append((f"blk.{L}.ffn_gate_exps.weight", T, fin_sq.copy()))
        entries.append((f"blk.{L}.ffn_up_exps.weight", T, fin_sq.copy()))
        # down: 输入 = per-expert hf, 取激发专家平均
        idx, wt, _ = F.gate_route(Fin, W, Ids)
        fe = idx.reshape(-1); tok = np.repeat(np.arange(T), NACT)
        hid_sq = None; hid_n = 0
        for e in np.unique(fe[fe >= 0]):
            m = fe == e; t = tok[m]; Xt = Fin[t]
            w1, w3 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3))
            gf = Xt @ w1.T; uf = Xt @ w3.T
            hf = (gf / (1 + np.exp(-gf))) * uf
            s = (hf.astype(np.float64) ** 2).sum(0)
            hid_sq = s if hid_sq is None else hid_sq + s
            hid_n += hf.shape[0]
        if hid_sq is not None and hid_n > 0:
            entries.append((f"blk.{L}.ffn_down_exps.weight", hid_n, hid_sq.astype(np.float32)))
        # 推进 H (fp8 routed, 与教师一致)
        share = F.expert_fp(Fin, W['s1'], W['s3'], W['s2'])
        out = share.copy(); fw = wt.reshape(-1)
        for e in np.unique(fe[fe >= 0]):
            m = fe == e; t = tok[m]; ww = fw[m][:, None]
            w1, w3, w2 = (F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w{k}.weight") for k in (1, 3, 2))
            np.add.at(out, t, F.expert_fp(Fin[t], w1, w3, w2, ww))
        H = F.hc_post(out.astype(np.float32), *ctx)
        print(f"[imat] L{L:2d} {time.time()-t0:5.1f}s ({len([e for e in entries])} entries)", file=sys.stderr, flush=True)
        del W

    with open(a.out, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, ncall, vals in entries:
            nb = name.encode()
            f.write(struct.pack("<i", len(nb))); f.write(nb)
            f.write(struct.pack("<i", int(ncall)))
            f.write(struct.pack("<i", len(vals)))
            f.write(vals.astype("<f4").tobytes())
        ds = b"coding_hard"
        f.write(struct.pack("<i", 1)); f.write(struct.pack("<i", len(ds))); f.write(ds)
    print(f"编程 imatrix 写出 {a.out}: {len(entries)} entries", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
