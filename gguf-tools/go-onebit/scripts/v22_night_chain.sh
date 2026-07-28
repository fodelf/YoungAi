#!/bin/bash
# v22_night_chain.sh — 今夜收官链(2026-07-25): 反修收官 → rr S=305 终判 → merge。M4 侧驱动。
set -uo pipefail
M1=192.168.1.2; M1ROOT=/Users/fodelf/ds4-main
log(){ echo "[night $(date +%H:%M:%S)] $*"; }
# 1) 等反修退出
while ssh -o ConnectTimeout=10 $M1 'pgrep -f ds4quant_run >/dev/null' 2>/dev/null; do sleep 120; done
log "反修进程退出; 终版 VERDICT:"
ssh $M1 'grep -a "VERDICT.*S=530" /tmp/quant_all.out | tail -1'
# 2) rr S=305 终判(回放判据, VQ 感知二进制); 锚缺则 FP_ONLY 自建
ANCH=/tmp/ds4quant_anchor_code_s305_rr.bin
EXP=$((305*(43*81968+517120)+40))
ASZ=$(ssh $M1 "stat -f%z $ANCH 2>/dev/null || echo 0")
if [ "$ASZ" != "$EXP" ]; then
  log "建 rr 锚(S=305, ~10-15min)"
  ssh $M1 "cd $M1ROOT/gguf-tools/go-onebit/quant && env DS4_ANCHOR=$ANCH DS4_NL=43 DS4_FP_ONLY=1 ./ds4quant_run /tmp/rr_code.ids 305" >/tmp/v22_rranchor.log 2>&1 || { log "rr 锚构建失败"; exit 2; }
fi
log "起 rr S=305 终判"
ssh $M1 "cd $M1ROOT/gguf-tools/go-onebit/quant && env DS4_VQ=1 DS4_GO2B_HOT=1 \
  DS4_GO2B_HOT_TABLE=$M1ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt \
  DS4_ANCHOR=/tmp/ds4quant_anchor_code_s305_rr.bin DS4_NL=43 DS4_LCFG=g DS4_COADAPT=1 \
  DS4_BF_ONLY=1 DS4_BF_MEMGB=8 DS4_BF_TERMINAL=0 DS4_GSWEEP=0 DS4_BWD=0 DS4_BF_TERM_MAXP=0 \
  DS4_LAYER_DIR=$M1ROOT/gguf/go-onebit/layers DS4_ZFILE=/tmp/rrz.bin DS4_ZCHAIN=/tmp/rrzc.bin \
  ./ds4quant_run /tmp/rr_code.ids 305" > /tmp/v22_rr.out 2>/tmp/v22_rr.log || log "rr 步 rc=$? 见 /tmp/v22_rr.log"
grep -a "VERDICT" /tmp/v22_rr.out | tail -2
log "夜链前半完成(merge 等引擎侧确认后走)"
