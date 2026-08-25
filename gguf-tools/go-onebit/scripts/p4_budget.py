#!/usr/bin/env python3
"""p4_budget.py — spark(GB10) decode 理论速度账(2026-08-20 用户令"计算理论最大速度")。
每 token 必读字节 = 6/256 路由专家 + 全部骨架2D权重(每token各读一遍) + 输出头 + zchain U/V。
理论 t/s = 带宽 / 每token字节。带宽档: 273(LPDDR5x标称) / 239(nsys实测q2 kernel) / 200(保守)。
用法: p4_budget.py <model.gguf> [amp_layers=42] [amp_k=512]
"""
import sys
from gguf import GGUFReader

mdl = sys.argv[1]
amp_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 42
amp_k = int(sys.argv[3]) if len(sys.argv) > 3 else 512
r = GGUFReader(mdl)
cat = {}
for t in r.tensors:
    n, b = t.name, int(t.n_bytes)
    if len(t.shape) < 2:
        continue                      # 1D norm/bias 忽略
    if "_exps." in n:
        key = "routed_experts"
        b = b * 6 // 256              # 每 token 只读 6/256
    elif "shexp" in n:
        key = "shared_experts"
    elif n == "output.weight":
        key = "output_head"
    elif n == "token_embd.weight":
        continue                      # 只读1行, 忽略
    elif "indexer" in n or "compressor" in n:
        key = "indexer_compressor"
    elif "attn" in n:
        key = "attention"
    elif "ffn_gate_inp" in n:
        key = "router"
    else:
        key = "other2d"
    cat[key] = cat.get(key, 0) + b

zc = amp_layers * (2 * 4096 * amp_k + amp_k) * 2
tot_bare = sum(cat.values())
print(f"== {mdl.split('/')[-1]} 每token必读字节账 ==")
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} {v/1e6:9.1f} MB   ({v/tot_bare*100:4.1f}%)")
print(f"  {'合计(裸)':18s} {tot_bare/1e6:9.1f} MB")
print(f"  {'zchain U/V/z':18s} {zc/1e6:9.1f} MB   (+amp{amp_layers}层 k={amp_k})")
tot_amp = tot_bare + zc
for bw in (273, 239, 200):
    print(f"  BW={bw}GB/s: 裸理论 {bw*1e9/tot_bare:6.1f} t/s | +amp理论 {bw*1e9/tot_amp:6.1f} t/s")
