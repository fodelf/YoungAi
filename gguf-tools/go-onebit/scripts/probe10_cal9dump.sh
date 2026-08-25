#!/bin/bash
# probe10_cal9dump.sh — 10分钟针数据料①: M10 全量层(en86/layers)对 cal9 语料回放,
# dump student logits(/tmp/cal9_student.bin, <ii>头+fp32[n×V])供 probe_klsolve.py 行KL权重。
# 在 M1(192.168.1.2)上跑; log /tmp/p10a.log, 完成标记 CAL9DUMP_DONE。
set -e
export LC_ALL=en_US.UTF-8
R30=/Users/fodelf/ds4-main/gguf/go-onebit/r30
G7=/Users/fodelf/ds4-main/gguf/go-onebit/g7
LCFG=$(printf "g%.0s" $(seq 1 43))
cd /Users/fodelf/ds4-main/gguf-tools/go-onebit/quant
env -u DS4_TUNE -u DS4_MINVOL -u DS4_VQ_RPLAN -u DS4_ZCHAIN \
  DS4_HF=/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-0731 DS4_GSWEEP=0 DS4_BF_TERMINAL=0 \
  DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 \
  DS4_ANCHOR=$R30/anchor_wtcal9_s2906.bin DS4_NFIT=1 DS4_THREADS=8 \
  DS4_LAYER_DIR=$R30/en86/layers DS4_LCFG=$LCFG DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
  DS4_DUMP_LOGITS=/tmp/cal9_student.bin \
  ./ds4quant_run.dchunk $G7/wt2train_cal9.ids 8000 >/dev/null 2>&1
echo "CAL9DUMP_DONE $(date +%T)"
