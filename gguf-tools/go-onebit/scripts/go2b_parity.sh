#!/usr/bin/env bash
# go2b_parity.sh — go2b C 编码器 ↔ Python go2b_encode.py 数值对齐探针(M1 跑, HF 在 M1)。
# 长跑前机制审计铁律: C 集成投产前, 同一 W/X 两端各编一次, 输出级 relL2/cos 必须对齐(≤2% 差)。
# 用法: go2b_parity.sh [L] [expert]   (默认 L20 + 该层热表第1个专家)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
L=${1:-20}
CAP=${CAP:-/tmp/cap_algo43}
log(){ echo "[g2parity] $*" >&2; }

scp -q "$ROOT/gguf-tools/go-onebit/quant/go2b_qc.h" \
       "$ROOT/gguf-tools/go-onebit/quant/go2b_parity.c" \
       "$M1:$M1DIR/gguf-tools/go-onebit/quant/" || { log "scp 失败"; exit 1; }
scp -q "$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt" "$M1:/tmp/prog_active.txt"

ssh "$M1" "cd $M1DIR/gguf-tools/go-onebit/quant && \
  cc -O3 -ffast-math -lm -framework Accelerate go2b_parity.c -o /tmp/go2b_parity -lpthread && \
  E=\${E:-\$(awk -F'[: ]+' '/^L$L:/{print \$2}' /tmp/prog_active.txt)} && \
  echo \"[g2parity] L$L expert=\$E\" >&2 && \
  DS4_HF=$M1DIR/hf/DeepSeek-V4-Flash-Base DS4_GO2B_ACT_SCALE=1 PYL=$L PYE=\$E CAP=$CAP python3 - <<'PYEOF'
import os,sys,numpy as np
sys.path.insert(0,os.getcwd()); sys.path.insert(0,'../calib/pyfwd')
import ds4reader as R
from go2b_encode import encode_go2b
L=int(os.environ['PYL']); E=int(os.environ['PYE']); cap=os.environ['CAP']
W=R.read_weight(f'layers.{L}.ffn.experts.{E}.w1.weight').astype(np.float32)
X=np.fromfile(f'{cap}/raw_ffn_in_L{L}',dtype='<f2').reshape(-1,4096).astype(np.float32)[:256]
W.tofile('/tmp/g2p_W.f32'); X.tofile('/tmp/g2p_X.f32')
blk,wq=encode_go2b(W,Xh=X,mode='nf')
Ot=X@W.T; Oq=X@wq.T
rel=np.linalg.norm(Ot-Oq)/np.linalg.norm(Ot)
cos=(Ot.flatten()@Oq.flatten())/(np.linalg.norm(Ot)*np.linalg.norm(Oq)+1e-9)
print(f'PY-go2b 输出级 relL2={rel:.4f} cos={cos:.4f} (nact={len(X)})')
wrel=np.linalg.norm(W-wq)/np.linalg.norm(W)
print(f'PY-go2b 权重级 relL2={wrel:.4f}')
PYEOF
  /tmp/go2b_parity /tmp/g2p_W.f32 2048 4096 /tmp/g2p_X.f32 256" 2>&1
log "判据: C 与 PY 输出级 relL2/cos 差≤2%, 且 C 往返自检 PASS"
