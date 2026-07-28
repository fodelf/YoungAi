#!/bin/bash
# g2_signref_blk_ab.sh — G2 判决: SIGNREF_BLK / TGT_ALPHA 三层小样四侧 A/B(2026-07-25 v2)。
# ★产物入指定项目目录(用户裁决 07-25): gguf/go-onebit/g2ab/<side>/, /tmp 只留日志+锚缓存★
# 在 M1 运行(HF 本地)。生产同参: LCFG=g MU=10 COADAPT=1 GO2B_HOT=1 S=530 编程语料; BACKFIT_INCR=0。
# 四侧: base | blk(SIGNREF_BLK) | blk_a05(+α0.5) | blk_a10(+α1.0); 判据=g2_compare.py 逐层表。
set -uo pipefail
M1ROOT=/Users/fodelf/ds4-main
QDIR=$M1ROOT/gguf-tools/go-onebit/quant
HOTTAB=$M1ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt
OUTROOT=$M1ROOT/gguf/go-onebit/g2ab
IDS=/tmp/rr_calib_prog_v1.ids; NTOK=530; NL=${NL:-3}
ANCH=/tmp/g2_anchor_L${NL}.bin
cd "$QDIR"
log(){ echo "[g2-ab $(date +%H:%M:%S)] $*" >&2; }

if [ ! -s "$ANCH" ]; then
  log "建 ${NL} 层 FP 锚 S=$NTOK"
  DS4_ANCHOR="$ANCH" DS4_NL=$NL DS4_FP_ONLY=1 ./ds4quant_run "$IDS" $NTOK \
      >/tmp/g2_anchor.out 2>/tmp/g2_anchor.log || { log "建锚失败"; exit 2; }
fi
log "锚 OK: $(stat -f%z "$ANCH") bytes"

run_side(){ # $1=tag $2=extra_env("" 允许)
  local D=$OUTROOT/$1
  local FREE; FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  [ "$FREE" -ge 8 ] || { log "★盘闸: 仅 ${FREE}G <8G, 停★"; exit 6; }
  rm -rf "$D"; mkdir -p "$D"
  log "起跑 $1 (dir=$D extra='${2:-}' free=${FREE}G)"
  ( export DS4_ANCHOR="$ANCH" DS4_NL=$NL DS4_LCFG=g DS4_SIGNREF_MU=10 DS4_COADAPT=1 \
           DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$HOTTAB" DS4_BACKFIT_INCR=0 \
           DS4_ZFILE="$D/zfile.bin" DS4_ZCHAIN="$D/zchain.bin" \
           DS4_LAYER_DIR="$D" DS4_THREADS=${DS4_THREADS:-6}
    [ -n "${2:-}" ] && export $2
    exec ./ds4quant_run "$IDS" $NTOK ) >"$D/all.out" 2>"/tmp/g2_$1.log"
  log "$1 完成 rc=$? 层文件 $(ls "$D"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')"
}

run_side base    ""
run_side blk     "DS4_SIGNREF_BLK=1"
run_side blk_a05 "DS4_SIGNREF_BLK=1 DS4_TGT_ALPHA=0.5"
run_side blk_a10 "DS4_SIGNREF_BLK=1 DS4_TGT_ALPHA=1.0"

log "==== 四侧判决表 ===="
python3 "$M1ROOT/gguf-tools/go-onebit/scripts/g2_compare.py" || true
