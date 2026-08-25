#!/bin/bash
# p1_amp_parity.sh — 第1针: 引擎 zchain kernel vs python 复算 逐位对账(2026-08-20)。
# 原理: 批路捕获点在 zchain 应用之后(ds4.c:15545), 故 单层链(L0)两跑:
#   裸跑捕 x̂/O_pre, 链跑捕 O_post(上游无侧车层→x̂ 逐位同) → python 用 ds4_zchain.c
#   同式复算 y_py=apply(x̂,O_pre) 与 O_post 对账。type6(加性+信任域)/type7(乘性tanh)双式支持。
set -uo pipefail
ROOT="$HOME/ds4-main"
R30="$ROOT/gguf/go-onebit/r30"
cd "$ROOT"
LOG(){ echo "[p1 $(date +%H:%M:%S)] $*"; }
python3 -c 'ids=open("gguf/go-onebit/g7/wt2.ids").read().split(); import os
open("/tmp/p1.ids","w").write(" ".join(ids[:int(os.environ.get("P1_N","512"))]))'
rm -rf /tmp/p1_zc /tmp/p1_cap_bare /tmp/p1_cap_z
mkdir -p /tmp/p1_zc /tmp/p1_cap_bare /tmp/p1_cap_z
L="${P1_L:-00}"
LN=$((10#$L))
export P1_L="$L" P1_LN=$LN
cp "$R30/c86/layers/zrec_L$L.bin" /tmp/p1_zc/
python3 gguf-tools/go-onebit/zlever/zrec_to_zchain.py /tmp/p1_zc /tmp/p1_L0.zchain 43
LOG "裸跑(捕 x̂/O_pre)"
env DS4_CAP_DIR=/tmp/p1_cap_bare DS4_CAP_LAYERS=$LN-$LN DS4_CUDA_NO_TOKEN_GRAPH=1 \
    timeout --foreground 1200 ./ds4 --cuda -m gguf/ds4-cal12.gguf \
    --score-ids /tmp/p1.ids --score-out /tmp/p1_bare.bin </dev/null 2>&1 | grep -aE "完成" | tail -1
LOG "单层链跑(捕 O_post)"
env DS4_CAP_DIR=/tmp/p1_cap_z DS4_CAP_LAYERS=$LN-$LN DS4_CUDA_NO_TOKEN_GRAPH=1 \
    timeout --foreground 1200 ./ds4 --cuda -m gguf/ds4-cal12.gguf --zchain /tmp/p1_L0.zchain \
    --score-ids /tmp/p1.ids --score-out /tmp/p1_z.bin </dev/null 2>&1 | grep -aE "zchain|完成" | tail -2
LOG "python 复算对账"
python3 - <<'PY'
import struct
import numpy as np
def rd(p, d=4096):
    a = np.fromfile(p, dtype=np.float16)
    return a.reshape(-1, d)
xb = rd("/tmp/p1_cap_bare/raw_ffn_in_L"+__import__("os").environ.get("P1_LN","0"))
xz = rd("/tmp/p1_cap_z/raw_ffn_in_L"+__import__("os").environ.get("P1_LN","0"))
ob = rd("/tmp/p1_cap_bare/raw_ffn_out_L"+__import__("os").environ.get("P1_LN","0")).astype(np.float64)
oz = rd("/tmp/p1_cap_z/raw_ffn_out_L"+__import__("os").environ.get("P1_LN","0")).astype(np.float64)
print("tokens:", xb.shape[0], " x̂裸==x̂链 逐位:", np.array_equal(xb, xz))
raw = open("/tmp/p1_zc/zrec_L"+__import__("os").environ.get("P1_L","00")+".bin","rb").read()
off = 0
rec = None
while off + 116 <= len(raw):
    nm = raw[off:off+16].split(b"\0")[0].decode()
    psz, = struct.unpack_from("<Q", raw, off+88)
    pay = raw[off+116:off+116+psz]
    off += 116 + psz
    if nm.startswith("zl."):
        rec = (nm, pay)
if rec is None:
    raise SystemExit("★zrec 无 zl 记录★")
nm, pay = rec
k, tr, din, dout = struct.unpack_from("<IfII", pay, 0)
hz = np.frombuffer(pay, dtype=np.float16, count=k, offset=16).astype(np.float64)
d = 4096
hU = np.frombuffer(pay, dtype=np.float16, count=d*k, offset=16+2*k).astype(np.float64).reshape(d, k)
hV = np.frombuffer(pay, dtype=np.float16, count=din*k, offset=16+2*k+2*d*k).astype(np.float64).reshape(din, k)
x = xb.astype(np.float64)
if din == 3*d:
    nrm = np.sqrt((x**2).mean(axis=1, keepdims=True)) + 1e-6
    xin = np.concatenate([x, x*x/nrm, np.maximum(x, 0)], axis=1)
else:
    xin = x
a = xin @ hV                      # [n,k]
mul = "AMP" in nm
print(f"记录={nm} k={k} tr={tr:.4g} din={din} 乘性={mul}")
if mul:
    pv = np.tanh(a / (tr if tr > 0 else 1.0)) * hz
    ua = pv @ hU.T
    ypy = ob * (1.0 + ua)
else:
    pv = a * hz
    ua = pv @ hU.T
    nd = np.linalg.norm(ua, axis=1)
    nr = np.linalg.norm(ob, axis=1)
    cap = tr * nr
    s = np.where((nd > cap) & (nd > 0), cap/np.maximum(nd, 1e-30), 1.0)
    ypy = ob + s[:, None]*ua
delta_eng = oz - ob
delta_py = ypy - ob
den = np.linalg.norm(delta_eng, axis=1) + 1e-12
rel = np.linalg.norm(delta_py - delta_eng, axis=1) / den
cos = (delta_py*delta_eng).sum(1) / (np.linalg.norm(delta_py,axis=1)*den + 1e-30)
print(f"Δ幅度: 引擎中位 {np.median(np.linalg.norm(delta_eng,axis=1)):.4g}  python中位 {np.median(np.linalg.norm(delta_py,axis=1)):.4g}")
print(f"Δ对账: rel_L2 中位 {np.median(rel):.4g} p95 {np.percentile(rel,95):.4g} max {rel.max():.4g}")
print(f"Δ余弦: 中位 {np.median(cos):.6f} min {cos.min():.6f}")
full_rel = np.linalg.norm(ypy-oz, axis=1)/(np.linalg.norm(oz,axis=1)+1e-12)
print(f"y整体: rel_L2 中位 {np.median(full_rel):.4g} max {full_rel.max():.4g}")
PY
LOG "p1 收官"
