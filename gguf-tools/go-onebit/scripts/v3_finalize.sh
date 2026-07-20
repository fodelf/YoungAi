#!/bin/sh
# v3_finalize.sh — v3 收尾编排 (跑在 M4, 后台):
#   A. 等 L0-5 六层重写标记 + shard-26 重下完成
#   B. 校验新 shard (坏专家区非零) → 换入 (坏件保留 .CORRUPT) → fix_l24_requant → L24 GPTQ 重跑
#   C. v3 全量回传 M4 → nll_gate 终判
set -u
M1=192.168.1.2
M1ROOT=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
SSH="ssh -o BatchMode=yes $M1"
DONE=$M1ROOT/zdump/v3_gptq1_done
SHARD=model-00026-of-00046.safetensors
SZ=6611519072

echo "[A] wait L0-5 markers + shard download $(date +%H:%M:%S)"
while :; do
  N=$($SSH "ls $DONE 2>/dev/null | grep -cE '^L[0-5]$'")
  D=$($SSH "stat -f%z /tmp/shard26_fresh.safetensors 2>/dev/null || echo 0")
  echo "  L0-5 done=$N/6 shard=$((D/1048576))MiB/6305MiB $(date +%H:%M:%S)"
  [ "$N" = 6 ] && [ "$D" = "$SZ" ] && break
  sleep 120
done

echo "[B] verify fresh shard $(date +%H:%M:%S)"
$SSH "cd $M1ROOT && python3 - <<'EOF'
import json, struct, numpy as np
p='/tmp/shard26_fresh.safetensors'
f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n)); ds=8+n
for nm in ('layers.24.ffn.experts.2.w1.weight','layers.24.ffn.experts.223.w1.weight','layers.24.ffn.experts.122.w1.weight'):
    o0,o1=hdr[nm]['data_offsets']; f.seek(ds+o0); b=np.frombuffer(f.read(min(1048576,o1-o0)),dtype=np.uint8)
    nz=float((b!=0).mean()); print(nm.split('.')[3], 'nonzero', round(nz,4)); assert nz>0.5, 'fresh shard still zero!'
print('FRESH-SHARD-VERIFIED')
EOF" || { echo "FRESH-SHARD-BAD, abort"; exit 1; }

echo "[B] swap + requant L24 $(date +%H:%M:%S)"
$SSH "cd $M1ROOT/hf/DeepSeek-V4-Flash-Base && mv $SHARD $SHARD.CORRUPT && mv /tmp/shard26_fresh.safetensors $SHARD && ls -la $SHARD" || exit 1
$SSH "cd $M1ROOT && DS4_HF=$M1ROOT/hf/DeepSeek-V4-Flash-Base python3 gguf-tools/go-onebit/quant/fix_l24_requant.py --gguf gguf/ds4-go1b-v3.gguf" || exit 1
$SSH "rm -f $DONE/L24"
LO=24 HI=24 CAPDIRS="$M1ROOT/cap_l05,$M1ROOT/cap_v2r2,$M1ROOT/cap_v2r1_deep" sh "$ROOT/gguf-tools/go-onebit/scripts/m4_lane.sh" 3
$SSH "ls $DONE | grep -q '^L24$'" || { echo "L24-GPTQ-FAILED"; exit 1; }

echo "[C] re-ship v3 to M4 + final NLL $(date +%H:%M:%S)"
scp -o BatchMode=yes -q "$M1:$M1ROOT/gguf/ds4-go1b-v3.gguf" "$ROOT/gguf/ds4-go1b-v3.gguf" || exit 1
cd "$ROOT"
MODEL=gguf/ds4-go1b-v3.gguf sh gguf-tools/go-onebit/scripts/nll_gate.sh - 2>&1 | tail -1
echo "V3-FINALIZE-DONE $(date +%H:%M:%S)"
