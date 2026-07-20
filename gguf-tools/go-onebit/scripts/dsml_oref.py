#!/usr/bin/env python3
"""dsml_oref.py — P2 ref 步逐层工作器 (在 M1 上跑, HF shard 全在 M1 本地)。

O_REF_L = Σᵢ wᵢ·Expert_HF(x̂)  —— error-feedback 语义: 引擎轨迹的输入 x̂ +
引擎实际用的路由 (ids/gate 权重) + HF 原始 fp8 专家权重。这样 O_REF−O_BASE
恰好只含专家权重量化误差 (路由/输入完全同源), 就是 corr 侧车要拟合的 R。
不需要完整 HF 前向 (那是 55h 级); 数值全部复用 dsv4_fwd.expert_fp +
ds4reader 的 e4m3+block-scale dequant, 本文件零新数值代码。

用法 (M1):
  DS4_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base \
  python3 dsml_oref.py --cap /tmp/oref --layer 25 --out /tmp/oref/o_ref_L25

输入 (引擎捕获 raw 格式, 见 ds4.c cap_batch_layer 头注):
  {cap}/raw_ffn_in_L{L}   f16 [n,4096]  post-RMSNorm 专家输入 x̂
  {cap}/raw_route_L{L}    i16 [n,6]     选中专家 id (pre-remap = HF 编号)
  {cap}/raw_route_w_L{L}  f16 [n,6]     实际施加的 gate 权重 (含 route scale)
输出: f16 [n,4096] raw (与 raw_ffn_out 同构, solve 步直接减)。
"""
import argparse
import os
import sys
import time

import numpy as np

# repo 布局 (scripts/../calib/pyfwd) 与 M1 部署布局 (脚本旁 pyfwd/) 都支持
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "calib", "pyfwd"),
              os.path.join(_here, "pyfwd")):
    if os.path.isfile(os.path.join(_cand, "dsv4_fwd.py")):
        sys.path.insert(0, _cand)
        break
import dsv4_fwd as F   # noqa: E402  (import 安全: 顶层只读 config)

D = 4096
NACT = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = a.layer

    x = np.fromfile(f"{a.cap}/raw_ffn_in_L{L}", dtype="<f2").reshape(-1, D).astype(np.float32)
    ids = np.fromfile(f"{a.cap}/raw_route_L{L}", dtype="<i2").reshape(-1, NACT)
    wts = np.fromfile(f"{a.cap}/raw_route_w_L{L}", dtype="<f2").reshape(-1, NACT).astype(np.float32)
    n = x.shape[0]
    assert ids.shape[0] == n and wts.shape[0] == n, \
        f"行数不齐: x={n} ids={ids.shape[0]} w={wts.shape[0]}"

    out = np.zeros((n, D), dtype=np.float32)
    flat_e = ids.reshape(-1)
    flat_w = wts.reshape(-1)
    tok = np.repeat(np.arange(n), NACT)
    uniq = np.unique(flat_e)
    t0 = time.time()
    for k, e in enumerate(uniq):
        sel = flat_e == e
        t = tok[sel]
        ww = flat_w[sel][:, None]
        w1 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w1.weight")
        w3 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w3.weight")
        w2 = F.R.read_weight(f"layers.{L}.ffn.experts.{e}.w2.weight")
        np.add.at(out, t, F.expert_fp(x[t], w1, w3, w2, ww))
        if (k + 1) % 32 == 0 or k + 1 == len(uniq):
            el = time.time() - t0
            print(f"[oref] L{L} expert {k+1}/{len(uniq)} "
                  f"({el:.0f}s, ETA {el/(k+1)*(len(uniq)-k-1):.0f}s)",
                  file=sys.stderr, flush=True)
    out.astype("<f2").tofile(a.out)
    print(f"[oref] L{L} done n={n} -> {a.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
