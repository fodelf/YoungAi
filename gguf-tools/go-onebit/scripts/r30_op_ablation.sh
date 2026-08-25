#!/bin/bash
# R30 op 族消融回放(2026-08-04): 定位"过程态1.853 vs 学生回放2.001"分叉元凶。
# 三组: 跳TREF(4) / 跳z^L(6) / 跳缩放族(1,2,3); 基线(全应用)=2.0012 已有。
# 用 ds4quant_run.ablate(带 DS4_REPLAY_SKIP_TYPES 门), 在 M1 跑, 每组 ~4 分钟。
set -u
R30=/Users/fodelf/ds4-main/gguf/go-onebit/r30
Q=/Users/fodelf/ds4-main/gguf-tools/go-onebit/quant
run_one(){ # $1=skip列表 $2=标签
  echo "[ablate $(date +%H:%M:%S)] 起 skip={$1}"
  env -u DS4_TUNE -u DS4_MINVOL -u DS4_MV_BASELINE -u DS4_VQ_RPLAN -u DS4_BWD \
    DS4_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-0731 \
    DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_CALIB_FULLSET=1 \
    DS4_ANCHOR="$R30/anchor_r30_s1716.bin" DS4_NFIT=933 DS4_THREADS=8 \
    DS4_LAYER_DIR="$R30/full/layers" DS4_LCFG=$(printf 'g%.0s' $(seq 1 43)) \
    DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_COADAPT=1 \
    DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE=/Users/fodelf/ds4-main/gguf-tools/go-onebit/corpus/prog_active_top64.txt \
    DS4_ZFILE="$R30/full/zfile.bin" DS4_ZCHAIN=/tmp/zchain_ablate.bin \
    DS4_ROUTE_BIAS="$R30/full/route_bias_r30.bin" DS4_ROUTE_BIAS_ALPHA=2.5 DS4_ROUTE_BIAS_MINCNT=8 \
    DS4_ANCHOR_ROUTE=1 \
    DS4_REPLAY_SKIP_TYPES="$1" \
    "$Q/ds4quant_run.ablate" /Users/fodelf/ds4-main/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids 1716 \
    > "/tmp/ablate_$2.log" 2>&1
  V=$(grep '^VERDICT' "/tmp/ablate_$2.log" | tail -1)
  echo "ABLATE skip={$1} $V"
}
run_one 4     T4
run_one 6     T6
run_one 1,2,3 T123
echo "[ablate] 三组收官"
