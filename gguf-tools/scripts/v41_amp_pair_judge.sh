#!/bin/bash
# v41_amp_pair_judge.sh — 把【几个反修侧车目录】放在同一个二进制、同一趟里, 用两把尺成对判(2026-09-21)。
#
# 【为什么要它】反修的收益经常是零点几个 pp, 而"换了二进制"或"隔了几小时"本身就能造出这个量级的差。
# 历史读数只能当参照, 定性必须同二进制同趟成对跑 —— 这个脚本就干这一件事, 不解算、不改模型。
#
# 【--strip-rb 在做什么】把一个【路由+增益】目录里的路由件摘掉, 只留增益, 生成一个探针目录。
# 用处: 分清"路由件本身有没有用"和"路由件让增益解到了更好的点" —— 后者摘掉 rb 仍然保留。
# ★软链必须写绝对路径★(09-20 实撞): 引擎 fopen 失败只当"没这个文件", 于是你以为挂了其实没挂, 不报错只出假数。
# manifest.txt 要一起拷 —— v41_judge.sh 靠它的 "# 完成" 判目录是不是半成品。
#
# 用法: v41_amp_pair_judge.sh <gguf> [--rulers <ids>:<ntok>[,…]] [--strip-rb <路由+增益目录>] <侧车目录>...
#   例: v41_amp_pair_judge.sh <v3.gguf> --strip-rb <…-grrb-…-engine> <…-gr-…-engine>
#       ⇒ 判三个态: grrb / grrb摘rb探针 / gr-only, 各跑 金融 j 8192 + wt2 512(默认两把尺)
#       --rulers gguf/go-onebit/vqhalf/vqhalf_j.ids:8192   ⇒ 只跑八域那把(补判时不重跑已有的)
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
J="$ROOT/gguf-tools/scripts/v41_judge.sh"
# 默认两把: 金融判决料(主尺) + wt2(英文通用守门)。八域 gguf/go-onebit/vqhalf/vqhalf_j.ids 按需用 --rulers 加。
RULERS="gguf/go-onebit/vqfin41/vqhalf_j.ids:8192,gguf/go-onebit/g7/wt2.ids:512"
GG="${1:?gguf}"; shift
M(){ echo "[pair $(date '+%m-%d %H:%M:%S')] $*"; }
DIRS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rulers) RULERS="${2:?--rulers 要 <ids>:<ntok>[,…]}"; shift 2 ;;
    --strip-rb)
      SRC="${2:?--strip-rb 要目录}"; shift 2
      [ -s "$SRC/manifest.txt" ] || { M "★$SRC 没有 manifest★"; exit 1; }
      P="${SRC%-engine}-norb-probe-engine"
      rm -rf "$P"; mkdir -p "$P"
      n=0; for f in "$SRC"/gr_L*.bin; do ln -s "$(readlink -f "$f")" "$P/$(basename "$f")"; n=$((n+1)); done
      cp "$SRC/manifest.txt" "$P/manifest.txt"
      printf '探针目录: %s 摘掉全部 rb_Lnn.bin, 只留 %d 个 gr_Lnn.bin(绝对软链)。\n不是部署件 —— 它回答"路由件本身值多少", 增益仍是挂着路由解出来的那一份。\n' "$SRC" "$n" > "$P/README_probe.txt"
      [ "$n" -gt 0 ] || { M "★$SRC 里一个 gr_Lnn.bin 都没有★"; exit 1; }
      M "探针 $P: $n 个增益件, 0 个路由件"
      DIRS+=("$P") ;;
    *) DIRS+=("$1"); shift ;;
  esac
done
[ "${#DIRS[@]}" -gt 0 ] || { M "★没给侧车目录★"; exit 2; }
for d in "${DIRS[@]}"; do [ -s "$d/manifest.txt" ] || { M "★$d 不是反修目录★"; exit 1; }; done

rc=0
# 尺在外层、目录在内层: 同一把尺的几个态**连着跑**, 中间不换尺 —— 成对读数要挨在一起才好比。
IFS=',' read -r -a RL <<< "$RULERS"
for r in "${RL[@]}"; do
  ids="${r%:*}"; n="${r##*:}"
  [ -s "$ids" ] || { M "★尺 $ids 不存在★"; rc=1; continue; }
  for d in "${DIRS[@]}"; do
    M "$(basename "$ids" .ids) n=$n ← $(basename "$d")"
    "$J" "$ids" "$n" "engine:${GG}:${d}" || rc=1
  done
done
M "收工 rc=$rc"; exit $rc
