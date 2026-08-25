#!/bin/bash
# probe_domz.sh — 分域z裁决针(2026-08-16 用户令"试一下/你验证一下")。
# v2(用户令"不要维基文章这个语料就不对, 重新跑"): 真域语料+专属锚版。
# 用法: probe_domz.sh [EVR] [TAG] [ANCHOR] [NTOK] [IDS] [name:fitranges ...]
#   缺参默认=cal9三臂(shared/prog/fin维基代理, 历史阶段1/2口径)。
#   锚缺失且给了 IDS → 自动 FP 捕锚(stage_anchor 同款: DS4_FP_ONLY=1 + 盘闸 + anchor_metrics 实读闸)。
#   TAG 非空 → cal9 zcache 先挪保/跑完还原(zcache 与语料绑定, 防陈旧缓存事故)。
# 口径: zlayer ADDON模式(M10记录之上), 零注入(zrec 逐臂挪 /tmp/zrec_probe_bak, 目录还原)。
set -e
export LC_ALL=en_US.UTF-8
ROOT=/Users/fodelf/ds4-main
[ -x "$ROOT/gguf-tools/amp/zlayer" ] || make -C "$ROOT/gguf-tools" zlayer   # C 反修解算器(zlayer.py 已删)
R30=$ROOT/gguf/go-onebit/r30
LD=$R30/en86/layers
DS4_HF=$ROOT/hf/DeepSeek-V4-Flash-0731
EVR=${1:-1590:1662,1899:1971}
TAG=${2:+_$2}
ANCHOR=${3:-$R30/anchor_wtcal9_s2906.bin}
NTOK=${4:-2906}
IDS=${5:-}
ARMS=("${@:6}")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(
  "shared:0:975,1141:1590,1668:1899,1971:2429,2577:2836"
  "prog:1141:1590"
  "fin:1668:1899" )
SUM=/tmp/domz_summary$TAG.log
BK=/tmp/zrec_probe_bak; mkdir -p $BK
cd $ROOT
: > $SUM

if [ ! -f "$ANCHOR" ]; then
  [ -n "$IDS" ] || { echo "锚缺失且无IDS, 无法捕锚" >> $SUM; exit 3; }
  FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  [ "$FREE" -ge 15 ] || { echo "★盘闸 free ${FREE}G <15G 停★" >> $SUM; exit 6; }
  echo "捕锚 $ANCHOR S=$NTOK $(date +%T)" >> $SUM
  ( cd $ROOT/gguf-tools/amp && \
    env DS4_HF=$DS4_HF DS4_FP_ONLY=1 DS4_ANCHOR="$ANCHOR" DS4_THREADS=8 \
      ./ds4quant_run.dchunk "$IDS" "$NTOK" >/tmp/domz_anchor$TAG.log 2>&1 )
  "$(dirname "$0")/../bench/anchor_metrics" --ref "$ANCHOR" --ids "$IDS" >/dev/null 2>&1 \
    || { echo "★锚完整性闸失败★" >> $SUM; exit 3; }
  echo "锚 ✓ $(ls -l "$ANCHOR" | awk '{printf "%.2f GiB",$5/1073741824}') $(date +%T)" >> $SUM
fi

if [ -n "$TAG" ]; then           # 语料切换: 既有 zcache 挪保
  for Lz in 00 30; do
    [ -f $LD/zcache_L$Lz.npz ] && mv $LD/zcache_L$Lz.npz $BK/zcache_pre$TAG.L$Lz.npz
  done
fi
for Lz in 00 30; do
  [ -f $LD/zrec_L$Lz.bin ] && mv $LD/zrec_L$Lz.bin $BK/zrec_L$Lz.pre$TAG.bin
done

run_arm() { # $1=layer $2=臂名 $3=fit范围
  Lz=$(printf "%02d" "$1"); lg=/tmp/domz_${2}_L$Lz$TAG.log
  env VECLIB_MAXIMUM_THREADS=4 DS4_ZL_NTOK=$NTOK DS4_ZL_FIT_RANGES="$3" \
    DS4_ZL_EV_RANGE="$EVR" DS4_ZL_SWLIM=60 \
    gguf-tools/amp/zlayer \
    "$DS4_HF" "$LD" "$ANCHOR" "$1" 1024 2 >"$lg" 2>&1
  grep -E "分域组合|held挽回" "$lg" | sed "s/^/[$2 L$Lz] /" >> $SUM
  mv "$LD/zrec_L$Lz.bin" "$BK/zrec_L$Lz.$2$TAG.bin" 2>/dev/null || true
  echo "ARM_DONE $2 L$Lz $(date +%T)" >> $SUM
}
for L in 0 30; do
  for a in "${ARMS[@]}"; do
    run_arm $L "${a%%:*}" "${a#*:}"
  done
done

if [ -n "$TAG" ]; then           # 还原: 本语料缓存归档, cal9 缓存回位
  for Lz in 00 30; do
    [ -f $LD/zcache_L$Lz.npz ] && mv $LD/zcache_L$Lz.npz $BK/zcache$TAG.L$Lz.npz
    [ -f $BK/zcache_pre$TAG.L$Lz.npz ] && mv $BK/zcache_pre$TAG.L$Lz.npz $LD/zcache_L$Lz.npz
  done
fi
echo "DOMZ_DONE $(date +%T)" >> $SUM
