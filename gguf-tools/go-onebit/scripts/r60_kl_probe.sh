#!/bin/bash
# r60_kl_probe.sh — 引擎执行口径三 KL 终审(2026-08-06 用户令"修好bug"):
# 链版/空链版 各跑单机全层 EVAL_IDS(批8 保底速)→ anchor_metrics 对锚账 → 三数字判决。
set -uo pipefail
ROOT=/Users/fodelf/git/ds4-main
cd $ROOT
IDS=gguf/go-onebit/g7_ids_v5mini.ids
ZL=gguf/go-onebit/ds4-r30.gguf.zchain.bin
LOG(){ echo "[klprobe $(date +%H:%M:%S)] $*"; }
EVAL_ENV="DS4_EVAL_NO_BOS=1 DS4_EVAL_CHUNK=8 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=8 DS4_VQ_GPU=1 DS4_MEM_BUDGET_MB=11500"

rm -f "$ZL"
LOG "链版(内嵌 op)起跑 ~7h"
env DS4_EVAL_IDS=$IDS DS4_EVAL_LOGITS=/tmp/eval_chain.bin $EVAL_ENV \
    ./ds4 -m gguf/go-onebit/ds4-r30.gguf --metal -c 4096 --nothink > /tmp/eval_kl_chain.log 2>&1 \
    || { LOG "★链版失败★"; tail -3 /tmp/eval_kl_chain.log; exit 3; }
LOG "链版 ✓ $(ls -l /tmp/eval_chain.bin | awk '{print $5}')B"

python3 -c "
import struct
b=struct.pack('<II',0x325A5144,43)+b''.join(struct.pack('<II',L,0) for L in range(43))
open('$ZL','wb').write(b)"
LOG "空链版起跑 ~7h"
env DS4_ZCHAIN=$ZL DS4_EVAL_IDS=$IDS DS4_EVAL_LOGITS=/tmp/eval_bare.bin $EVAL_ENV \
    ./ds4 -m gguf/go-onebit/ds4-r30.gguf --metal -c 4096 --nothink > /tmp/eval_kl_bare.log 2>&1 \
    || { LOG "★空链版失败★"; tail -3 /tmp/eval_kl_bare.log; exit 3; }
rm -f "$ZL"
LOG "空链版 ✓"

LOG "===== 三 KL 终审(量化器 VERDICT 参照 KL=0.387)====="
for v in chain bare; do
  echo "--- 引擎口径 [$v] ---"
  python3 gguf-tools/go-onebit/scripts/anchor_metrics.py --ref-raw /tmp/ref_logits.bin \
      --ids $IDS --student /tmp/eval_$v.bin --fit 933 2>&1 | /usr/bin/grep -E "PPL|KLD|top"
done
LOG "★三 KL 落定★"
