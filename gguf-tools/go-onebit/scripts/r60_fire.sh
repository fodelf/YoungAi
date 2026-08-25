#!/bin/bash
# r60_fire.sh — 2026-08-06 用户令: 60GB 落地战役(热72+冷热双通道+RB, 坐标下降提速版)。
# 链: top72 表→plan(HOT=72)→量化→冷热反修(+RB在场+GSWEEP)→合并(opt内嵌+烘焙+60闸)→回传→双针。
set -uo pipefail
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
SC_M1=$M1DIR/gguf-tools/go-onebit/scripts
LOG(){ echo "[r60 $(date +%H:%M:%S)] $*"; }

# ★发车前机器自检(2026-08-06 用户铁律): 全绿才放行★
bash $ROOT/gguf-tools/go-onebit/scripts/r60_preflight.sh || { LOG "★preflight 拒发车★"; exit 9; }

pgrep -f ds4-server >/dev/null && { pkill -f ds4-server; sleep 2; }
ssh $M1 "pkill -f ds4-server; pkill -f ds4quant_run" 2>/dev/null; sleep 1
LOG "清场 ✓"

ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh anchor" || { LOG "★锚段失败★"; exit 3; }
ssh $M1 "cd $M1DIR && python3 $SC_M1/anchor_top_experts.py \
    gguf/go-onebit/r30/anchor_r30_s1716.bin 72 gguf-tools/go-onebit/corpus/prog_active_top72.txt \
    && head -1 gguf-tools/go-onebit/corpus/prog_active_top72.txt >/dev/null && echo top72ok" | tail -1
LOG "top72 热表 ✓"
ssh $M1 "cd $M1DIR && PLAN_HOT=72 bash $SC_M1/r30_campaign.sh plan" || { LOG "★plan 失败★"; exit 3; }
ssh $M1 "cd $M1DIR && HOT_TABLE=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top72.txt \
         RESUME_QUANT=${RESUME_QUANT:-} VOL_BUDGET=37.0 \
         bash $SC_M1/r30_campaign.sh quant" || { LOG "★量化失败★"; exit 3; }
LOG "量化 43 层 ✓"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         HOT_TABLE=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top72.txt \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=1 DS4_GS_CONV_PCT=0 \
         bash $SC_M1/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
LOG "冷热反修+回扫 ✓"
ssh $M1 "pkill -9 -f 'ds4 '" 2>/dev/null
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh merge" || { LOG "★合并失败★"; exit 3; }
LOG "合并+烘焙 ✓"
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf $ROOT/gguf/go-onebit/ds4-r30.gguf || { LOG "★回传失败★"; exit 4; }
LOG "回传 ✓ $(ls -l $ROOT/gguf/go-onebit/ds4-r30.gguf | awk '{printf "%.2f GB", $5/1e9}')"
cd $ROOT
MODEL=gguf/go-onebit/ds4-r30.gguf SUITE=humaneval TAG=r60 OFFSETS=0 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Py 针异常★"
MODEL=gguf/go-onebit/ds4-r30.gguf SUITE=humaneval-x-go TAG=r60 OFFSETS=1 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Go 针异常★"
pkill -f ds4-server 2>/dev/null
LOG "★r60 战役收官(双针出)★"
