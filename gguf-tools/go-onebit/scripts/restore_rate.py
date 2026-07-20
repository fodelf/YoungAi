#!/usr/bin/env python3
"""restore_rate.py — 地基: 顺序量化前向 + top-1 一致率 (整体还原率, 纯量化零训练)。

对齐设计: 逐层前向天然顺序 → 量化某层专家自动喂进下游 → 误差层层传播。
测法: 跑 fp8 前向 (QUANT_HOOK=None) 拿 fp8 预测; 跑量化前向 (QUANT_HOOK 量化配置层)
拿量化预测; top-1 一致率 = mean(argmax 相同)。这是全局最优的总目标, ②敏感度③精修都挂它。

QUANT_HOOK(L,e,w1,w3,w2,Xin) 用该层真实传播输入 Xin 校准量化 (顺序=自洽)。
配置: QLAYERS(哪些层量化) × QBIT(go1b/go2b)。z 后续挂 (地基先只 base)。

用法(M1): DS4_HF=... python3 restore_rate.py --code CODE.txt --bit go1b --layers 0-42 [--ntok 200]
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "calib", "pyfwd"))
sys.path.insert(0, os.path.join(_here, "..", "quant"))
import dsv4_fwd as F   # noqa: E402
from go2b_encode import encode_go2b, decode_go2b   # noqa: E402


def go1b_q(W, X):
    S = np.sign(W).astype(np.float32); S[S == 0] = 1
    Pt = X @ W.T; Ps = X @ S.T
    num = (Pt * Ps).sum(0); den = (Ps * Ps).sum(0)
    s = np.where(den > 0, num / np.maximum(den, 1e-12), np.abs(W).mean(1))
    return S * s[:, None].astype(np.float32)


def make_hook(qlayers, bit):
    def hook(L, e, w1, w3, w2, Xin):
        if L not in qlayers:
            return w1, w3, w2
        Xh = Xin if len(Xin) >= 32 else np.repeat(Xin, 8, 0)[:256]
        if bit == "go1b":
            q1 = go1b_q(w1, Xh); q3 = go1b_q(w3, Xh)
            g = Xh @ q1.T; u = Xh @ q3.T; h = (g / (1 + np.exp(-g))) * u
            q2 = go1b_q(w2, h)
        else:
            os.environ["DS4_GO2B_ACT_SCALE"] = "1"
            b1, _ = encode_go2b(w1, Xh); b3, _ = encode_go2b(w3, Xh)
            q1 = decode_go2b(b1, 4096); q3 = decode_go2b(b3, 4096)
            g = Xh @ q1.T; u = Xh @ q3.T; h = (g / (1 + np.exp(-g))) * u
            b2, _ = encode_go2b(w2, h); q2 = decode_go2b(b2, 2048)
            os.environ.pop("DS4_GO2B_ACT_SCALE", None)
        return q1, q3, q2
    return hook


def run_forward(ids):
    for g in ("PRED", "LOGITS"):
        F.__dict__.pop(g, None)
    F.run([ids], [ids], "/tmp/rr_cap", capture=False)
    return np.concatenate(F.__dict__["PRED"], 0), np.concatenate(F.__dict__["LOGITS"], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--bit", default="go1b")
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--ntok", type=int, default=200)
    a = ap.parse_args()
    lo, hi = (map(int, a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers)))
    qlayers = set(range(lo, hi + 1))

    # 预分词 ids 文件 (M1 无 tokenizers; 用 M4 ds4 --dump-tokens 产)。每行一个 id。
    if a.code.endswith(".ids"):
        ids = np.array([int(x) for x in open(a.code) if x.strip()], dtype=np.int64)[:a.ntok]
    else:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(F.R.HF, "tokenizer.json"))
        ids = np.array(tk.encode(open(a.code).read()).ids, dtype=np.int64)[:a.ntok]
    print(f"tokens={len(ids)} 量化层={a.layers}({len(qlayers)}) bit={a.bit}", file=sys.stderr, flush=True)

    F.__dict__["QUANT_HOOK"] = None
    print("跑 fp8 前向 (teacher)...", file=sys.stderr, flush=True)
    fp8_pred, fp8_logits = run_forward(ids)

    F.__dict__["QUANT_HOOK"] = make_hook(qlayers, a.bit)
    print("跑量化前向 (顺序传播)...", file=sys.stderr, flush=True)
    q_pred, q_logits = run_forward(ids)

    agree = float(np.mean(fp8_pred == q_pred))
    # KL(fp8 ‖ quant) 近似
    def softmax(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    pf, pq = softmax(fp8_logits), softmax(q_logits)
    kl = float((pf * (np.log(pf + 1e-9) - np.log(pq + 1e-9))).sum(1).mean())
    print(f"RESTORE bit={a.bit} 层={a.layers}: top-1一致率={agree:.4f} KL={kl:.4f} (n={len(fp8_pred)})")


if __name__ == "__main__":
    main()
