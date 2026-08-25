#!/bin/bash
# r55_hotcold_run.sh — 冷热双通道步A(GLhc)全链: 保全→反修(RB在场+GLhc)→合并(折ge)→回传→17题+Go/1
set -uo pipefail
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
OUTF=$M1DIR/gguf/go-onebit/r30/full
SC=$M1DIR/gguf-tools/go-onebit/scripts
LOG(){ echo "[hc $(date +%H:%M:%S)] $*"; }

ssh $M1 "rm -rf $OUTF/hc_backup && mkdir -p $OUTF/hc_backup && cp -c $OUTF/layers/dql_ops_L*.bin $OUTF/zchain.bin $OUTF/hc_backup/ 2>/dev/null; ls $OUTF/hc_backup | wc -l" | { read N; LOG "保全 $N 文件"; }
LOG "冷热反修起跑(GLhc 阶段1.5 + RB 在场 + GSWEEP)"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=1 DS4_GS_CONV_PCT=0 \
         bash $SC/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
LOG "反修 ✓"
ssh $M1 "pkill -9 -f 'ds4 '; pkill -9 -f ds4-server" 2>/dev/null; sleep 1
ssh $M1 "cd $M1DIR && bash $SC/r30_campaign.sh merge" || { LOG "★合并失败★"; exit 3; }
LOG "合并+烘焙 ✓"
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf $ROOT/gguf/go-onebit/ds4-r30.gguf || { LOG "★回传失败★"; exit 4; }
LOG "回传 ✓"
cd $ROOT
MODEL=gguf/go-onebit/ds4-r30.gguf SUITE=humaneval TAG=r55hc OFFSETS=0 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Py 针异常★"
MODEL=gguf/go-onebit/ds4-r30.gguf SUITE=humaneval-x-go TAG=r55hc OFFSETS=1 bash gguf-tools/go-onebit/scripts/pubbench_serial.sh || LOG "★Go 针异常★"
pkill -f ds4-server 2>/dev/null
LOG "★冷热步A 全链收官★"
