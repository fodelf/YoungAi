#!/bin/bash
# v41_dynbits_probe_spark.sh — 动态位宽(逐层逐专家体积)量化, ★纯权重零语料★(2026-09-21, 用户令"不要用语料跑量化, 直接根据权重跑动态量化")。
#
# 【干什么】专家位宽不再一刀切: v41_nc_alloc 扫 HF 原始权重算每个专家的能量 Σw²(精确逐元素), 率失真反注水
#   分配每个专家的码本大小(默认 9~14 位), 总字节 = 全层均匀 12 位(码本按 GGUF 三份计);
#   --full = 全 40 层 + common 一份预算全局分(层体积与专家体积一起动), 量完直接文件态三尺(金融 j 8192 / 八域 j 8192 / wt2 512),
#   对照 = 零语料均匀 12 位的文件态历史读数(同一条 Python 路, 同一份教师锚; 金融 69.86% / wt2 76.95%, fable5 09-20 晚)。
#   不带 --full = 十层针: fincal 基底上只换 L[a,b), ref(均匀 12 位) 与 dyn 成对同趟判(留作单变量对照用)。
# 【零语料】分配只看权重, 不读任何 ids/取料/激活 —— 与"全域平权"一致, 判决语料仍只在尺上出现。
# 【为什么 --force(十层针)】--base 把基底分片软链进来后, 量化器"按层跳过已完成"会把软链当成品跳过; --force 只重做 --layers 的层。
# 【体积核对】十层针: 两目录真分片字节差 > 0.2% 停。全量: 分配器自己对账(用量 ≤ 预算), 目录字节由量化脚本打出, 转 GGUF 时再核 113.68。
# 用法(spark 本机, nohup): v41_dynbits_probe_spark.sh --full [--bmin 9] [--bmax 14] [--tag dynw] [--from ALLOC|DYN]
#                         v41_dynbits_probe_spark.sh [--layers 0:10] [--bmin 9] [--bmax 14] [--tag dyn] [--from ALLOC|REF|DYN|JUDGE]
# 停车规则: 任一段非 0 停在那一段; 尾行 PROBE_EXIT <rc>。看门狗: 量化与判决各自的脚本自带(available < 8 GB 杀)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
LAY="0:10"; BMIN=9; BMAX=14; TAG=""; FROM="ALLOC"; FULL=0
while [ $# -gt 0 ]; do case "$1" in
  --full) FULL=1; shift;;
  --layers) LAY="${2:?--layers 要 a:b}"; shift 2;;
  --bmin) BMIN="${2:?}"; shift 2;;
  --bmax) BMAX="${2:?}"; shift 2;;
  --tag) TAG="${2:?}"; shift 2;;
  --from) FROM="${2:?--from 要段名}"; shift 2;;
  *) echo "★不认识的参数 $1★"; exit 2;;
esac; done
HF="$ROOT/hf/DeepSeek-V4.1-Flash"
BASE="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-q4k-fincal"
if [ "$FULL" = 1 ]; then
  LAY="0:40"; [ -n "$TAG" ] || TAG="dynw"
  DYN="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8dyn-q4k-$TAG"; REF=""
else
  [ -n "$TAG" ] || TAG="dyn"
  IFS=: read -r A B <<<"$LAY"
  REF="$ROOT/gguf/v41/probe-dynbits-ref-L$A-$B"; DYN="$ROOT/gguf/v41/probe-dynbits-$TAG-L$A-$B"
fi
TABLE="$DYN/nc_table.txt"
FINJ="$ROOT/gguf/go-onebit/vqfin41/vqhalf_j.ids"; WT2="$ROOT/gguf/go-onebit/g7/wt2.ids"
Q="$ROOT/gguf-tools/scripts/v41_quantize_spark.sh"; J="$ROOT/gguf-tools/scripts/v41_judge.sh"
M(){ echo "[probe $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "PROBE_EXIT 1"; exit 1; }
# 段序比较: 段 $1 在 --from 指定的段之后(含)才跑
stage_ge(){ local order="ALLOC REF DYN JUDGE"; local a=${order%%$1*}; local b=${order%%$FROM*}; [ ${#a} -ge ${#b} ]; }
[ -d "$HF" ] || die "原始权重 $HF 不在"
[ "$FULL" = 1 ] || [ -d "$BASE" ] || die "基底 $BASE 不在"
# 实例锁: 学生 mmap 100 GB, 对面有模型/量化进程就不发(模式用 [x] 防 pgrep 自匹配)
if pgrep -f "^\./ds4 -m |v41_teacher[.]py|quantize/v41_quantiz[e]" >/dev/null; then die "有模型/量化进程在跑, 不发"; fi
mkdir -p "$DYN"
exec > >(tee -a "$DYN/probe_log.txt") 2>&1
M "动态位宽: 层 [$LAY) 位宽 [$BMIN,$BMAX] $([ "$FULL" = 1 ] && echo '全局预算(全 40 层 + common)' || echo "逐层预算, 基底 $(basename "$BASE")"); 产物 $(basename "$DYN")"

if stage_ge ALLOC; then
  M "ALLOC 编 v41_nc_alloc + 分配表 → $TABLE"
  make -C "$ROOT/gguf-tools" v41_nc_alloc 2>&1 | tail -2; [ -x "$ROOT/gguf-tools/quantize/v41_nc_alloc" ] || die "编 v41_nc_alloc 失败"
  GA=(); [ "$FULL" = 1 ] && GA=(--global)
  "$ROOT/gguf-tools/quantize/v41_nc_alloc" "$HF" "$LAY" "$TABLE" --nc-ref 4096 --bmin "$BMIN" --bmax "$BMAX" --cb-copies 3 --threads 8 "${GA[@]+"${GA[@]}"}" || die "分配表失败"
fi
if [ "$FULL" = 1 ]; then
  if stage_ge DYN; then
    M "DYN 全量: 40 层 + common(骨架 q4_K, 三塔 VQ 12 位), 逐专家位宽表; 量完文件态三尺"
    bash "$Q" --out "$DYN" --vq-nc 4096 --skel q4k --mtp-vq --vq-nc-table "$TABLE" --judge-set all || die "全量量化/判决失败"
  fi
  M "收工"; echo "PROBE_EXIT 0"; exit 0
fi
if stage_ge REF; then
  M "REF 参照: 零语料均匀 12 位 L[$LAY) 于基底"
  bash "$Q" --out "$REF" --base "$BASE" --layers "$LAY" --no-common --no-judge --force --vq-nc 4096 --skel q4k || die "REF 量化失败"
fi
if stage_ge DYN; then
  M "DYN 动态: 逐专家位宽表 L[$LAY) 于基底"
  bash "$Q" --out "$DYN" --base "$BASE" --layers "$LAY" --no-common --no-judge --force --vq-nc 4096 --skel q4k --vq-nc-table "$TABLE" || die "DYN 量化失败"
  vol(){ find "$1" -maxdepth 1 -type f -name 'model-layer??.safetensors' -printf '%s\n' | awk '{s+=$1} END{print s+0}'; }   # 只算真分片, 软链基底不算
  VR=$(vol "$REF"); VD=$(vol "$DYN")
  M "体积: ref $VR B / dyn $VD B (dyn/ref = $(awk -v a="$VR" -v b="$VD" 'BEGIN{printf "%.5f", b/a}'))"
  awk -v a="$VR" -v b="$VD" 'BEGIN{d=(b-a)/a; if (d<0) d=-d; exit (d>0.002)}' || die "体积差 > 0.2%, 不是等体积"
fi
if stage_ge JUDGE; then
  for spec in "$FINJ 8192" "$WT2 512"; do
    set -- $spec
    M "JUDGE $(basename "$(dirname "$1")")/$(basename "$1") $2: file:ref file:dyn 同趟"
    bash "$J" "$1" "$2" "file:$REF" "file:$DYN" || die "判决失败 $1"
  done
fi
M "收工"; echo "PROBE_EXIT 0"
