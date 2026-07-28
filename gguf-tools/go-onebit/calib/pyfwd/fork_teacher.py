#!/usr/bin/env python3
# fork_teacher.py — 原始模型在"躲题分叉点"的裁决(2026-07-27 Go 躲题后训练 P0)。
# 前向 BOS+prompt(+尾换行)+"   "(体缩进), 打印最后两位置 top-K logits:
#   ①缩进位选择 ②缩进后首 token = 注释(' //') vs 代码 fork。
# 前向环逐行镜像 teacher_nll.py(官方 numpy 前向, 流式 fp8)。
# 用法: DS4_HF=... python fork_teacher.py <prompt.txt> [topk=10]
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4reader as R
import dsv4_fwd as F
from tokenizers import Tokenizer

path = sys.argv[1]
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
tk = Tokenizer.from_file(os.path.join(R.HF, "tokenizer.json"))
text = open(path).read()
if not text.endswith("\n"):
    text += "\n"
ids_l = tk.encode("<｜begin▁of▁sentence｜>" + text + "   ").ids
ids = np.array(ids_l, dtype=np.int64)
print(f"fork_teacher: {len(ids)} tokens ({path})", flush=True)

_route_store = {}
emb = R.get("embed.weight")
h = np.repeat(emb[ids][:, None, :], F.HCM, 1).astype(np.float32)
for L in range(F.NL):
    W = F.load_layer(L)
    resid = h
    y, post, comb = F.hc_pre(h, W['hc_attn_fn'], W['hc_attn_scale'], W['hc_attn_base'])
    a = F.attention(F.rms(y, W['an']), W, L)
    h = F.hc_post(a, resid, post, comb)
    resid2 = h
    y2, post2, comb2 = F.hc_pre(h, W['hc_ffn_fn'], W['hc_ffn_scale'], W['hc_ffn_base'])
    fin = F.rms(y2, W['fn'])
    # 行为校准 Step A: 存每层最后两位置的路由(idx/raw logits) → npz
    if os.environ.get("FORK_ROUTE_OUT"):
        _idx, _w, _raw = F.gate_route(fin[-2:].reshape(2, -1), W, ids[-2:])
        _route_store[f"idx_L{L}"] = _idx
        _route_store[f"raw_L{L}"] = _raw.astype(np.float32)
    Fout, _ = F.moe_all(fin, ids, W, L)
    h = F.hc_post(Fout, resid2, post2, comb2)
    print(f"L{L} done", flush=True)
    del W

hcfn = R.get("hc_head_fn"); hcb = R.get("hc_head_base"); hcs = R.get("hc_head_scale")
fnorm = R.get("norm.weight"); hw = R.get("head.weight")
Sn = h.shape[0]; x = h.reshape(Sn, -1)
rsq = np.reciprocal(np.sqrt(np.mean(x * x, -1, keepdims=True) + F.EPS))
mixes = (x @ hcfn.T) * rsq
pre = F.sigmoid(mixes * hcs + hcb) + F.HCEPS
y = (pre[:, :, None] * h).sum(1)
y = F.rms(y, fnorm)
logits = (y @ hw.T).astype(np.float64)
if os.environ.get("FORK_ROUTE_OUT"):
    np.savez(os.environ["FORK_ROUTE_OUT"], **_route_store)
    print(f"route saved -> {os.environ['FORK_ROUTE_OUT']}", flush=True)

for pos_name, pos in (("缩进位(预测体首token)", len(ids) - 2), ("fork位(缩进后)", len(ids) - 1)):
    row = logits[pos]
    top = np.argsort(row)[::-1][:topk]
    print(f"== {pos_name} pos={pos} ==", flush=True)
    for t in top:
        print(f"  logit={row[t]:9.3f}  id={int(t):6d}  {tk.decode([int(t)])!r}", flush=True)
