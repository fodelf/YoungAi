#!/usr/bin/env bash
# go2b_requant_act.sh — 最小验证: 16 个 GO2B 层用联合 act-scale 重量化 (在 M1 跑, HF 本地)。
# 产 go2b_ef_L{L}.bin, 供 splice_go2b_inplace 塞回 pristine mono 测生成。
# DS4_GO2B_ACT_SCALE=1 触发 encode_go2b 的联合(输出最优d1,d2 + 重分配码)迭代。
# 用法(M1): go2b_requant_act.sh /tmp/capcodeF /tmp/go2b_act   [层清单默认16个GO2B]
set -uo pipefail
CAP=${1:-/tmp/capcodeF}; OUT=${2:-/tmp/go2b_act}
LAYERS=${3:-"2 23 24 25 26 27 28 29 30 31 32 33 34 36 38 40"}
ROOT=/Users/fodelf/ds4-main
QUANT=$ROOT/gguf-tools/go-onebit/quant
mkdir -p "$OUT" /tmp/go2b_npy
for L in $LAYERS; do
  [ -f "$OUT/go2b_ef_L$L.bin" ] && { echo "L$L 已有, 跳过"; continue; }
  # raw f16/i16 → npy (ef_layer 要 npy)
  python3 -c "
import numpy as np
X=np.fromfile('$CAP/raw_ffn_in_L$L',dtype='<f2').reshape(-1,4096).astype(np.float32)
r=np.fromfile('$CAP/raw_route_L$L',dtype='<i2').reshape(-1,6).astype(np.int64)
ok=np.isfinite(X).all(1); X,r=X[ok],r[ok]
np.save('/tmp/go2b_npy/X_L$L.npy',X); np.save('/tmp/go2b_npy/r_L$L.npy',r)
print('L$L npy:',X.shape)
"
  echo "[requant] L$L 重量化 (联合 act-scale)..."
  DS4_HF=$ROOT/hf/DeepSeek-V4-Flash-Base DS4_GO2B_ACT_SCALE=1 \
    python3 "$QUANT/ef_layer.py" --hf-direct --layer "$L" --pack /tmp/pk_L$L \
      --ffn-in /tmp/go2b_npy/X_L$L.npy --route /tmp/go2b_npy/r_L$L.npy \
      --out "$OUT/go2b_ef_L$L.bin" 2>&1 | grep -E "EF-LAYER-OK|Error|Traceback" | tail -2
done
echo "[requant] 完成: $(ls $OUT/*.bin 2>/dev/null | wc -l) 层 -> $OUT"
