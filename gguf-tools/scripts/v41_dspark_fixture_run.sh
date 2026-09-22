#!/bin/bash
# v41_dspark_fixture_run.sh — 在一段 ids 上把 DSpark 草稿器的判决三件套跑完(2026-09-18, 在 spark 上跑)。
#
# 干什么: ①引擎教师强制取料(修过口径的 --dspark-capture: 三个一致率的料 + 夹具料 .fix)
#         ②官方 FP 教师前向(同一段 ids: FP logits + FP main_hidden)
#         ③三个一致率分档(dspark_agree) + 夹具三输入重放(v41_dspark_fixture.py)
# 为什么要一个脚本: 三步串起来要 25 分钟, 手敲三条命令中间一停就断; 而且教师与引擎都吃整机内存, 必须串行。
# 用法: gguf-tools/scripts/v41_dspark_fixture_run.sh <tag> <ids文件(一行一个 id, 至少 ntok 个)> [ntok=1536] [--gguf 文件] [--zchain 目录|none]
#       --gguf / --zchain 默认现役对(2026-09-22 起 = vq8sh14-q4k-mtpnative + grrb); 换配方要成对给(反修是对某一份量化文件解的,
#       挂错不报错只出假数), 裸跑给 --zchain none。2026-09-20 加: 量 100 GB 配方(三塔 VQ blob)的草稿一致率。
#       产物全在 /tmp/dcap7/<tag>/: pairs.bin(.fix/.hdiag) teacher.bin fp_mainh.bin agree.txt fixture.txt 与各 log
# 出错会怎样: 任一步失败直接退出并留下该步的 log; 取料的"首位一致率"与 agree.txt 的 ① 必须一致, 不一致说明料错位。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
TAG="${1:?用法: $0 <tag> <ids> [ntok] [--gguf 文件] [--zchain 目录|none]}"
IDS="${2:?用法: $0 <tag> <ids> [ntok] [--gguf 文件] [--zchain 目录|none]}"
NTOK=1536
MODEL=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
ZCHAIN=gguf/v41/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine
shift 2
[ $# -gt 0 ] && [ "${1#--}" = "$1" ] && { NTOK="$1"; shift; }   # 第 3 个位置参数(不以 -- 开头)= ntok
while [ $# -gt 0 ]; do
    case "$1" in
        --gguf) MODEL="${2:?--gguf 后面要跟文件}"; shift 2;;
        --zchain) ZCHAIN="${2:?--zchain 后面要跟目录或 none}"; shift 2;;
        *) echo "★不认识的参数 $1★"; exit 2;;
    esac
done
[ -f "$MODEL" ] || { echo "★GGUF 不存在: $MODEL★"; exit 2; }
ZARG=(); if [ "$ZCHAIN" != none ]; then [ -d "$ZCHAIN" ] || { echo "★反修目录不存在: $ZCHAIN★"; exit 2; }; ZARG=(--zchain "$ZCHAIN"); fi
HF=hf/DeepSeek-V4.1-Flash
PY=~/v41env/bin/python
OUT=/tmp/dcap7/$TAG
mkdir -p "$OUT"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }
[ -x ./ds4 ] || { echo "★没有 ./ds4, 先 make cuda-spark★"; exit 1; }
[ -x ./gguf-tools/bench/dspark_agree ] || make -C gguf-tools dspark_agree || exit 1
[ "$(wc -l < "$IDS")" -ge "$NTOK" ] || { echo "★$IDS 不足 $NTOK 个 id★"; exit 1; }
# 实例锁: 引擎与教师都吃整机内存, 对面有本仓的模型进程在跑就不发
# ★锚定命令行开头★: 排队等着的 `sh -c "until …; ./ds4 -m …"` 的 argv 里也含 "ds4 -m", 不锚定会把等待壳当成模型进程(实撞)
if pgrep -f "^\./ds4 -m |^[^ ]*python[^ ]* gguf-tools/scripts/v41_teacher\.py" >/dev/null; then echo "★有模型进程在跑, 不发★"; exit 1; fi

LOG "① 引擎取料 $IDS → $OUT/pairs.bin"
echo "模型 $MODEL  反修 $ZCHAIN"
./ds4 -m "$MODEL" "${ZARG[@]+"${ZARG[@]}"}" --score-ids "$IDS" --dspark-capture "$OUT/pairs.bin" --decoder-full \
    > "$OUT/cap.out" 2> "$OUT/cap.err" || { echo "★取料失败, 看 $OUT/cap.err★"; exit 2; }
grep -a "一致率 =" "$OUT/cap.err"

LOG "② 教师前向 $NTOK 位 → teacher.bin + fp_mainh.bin"
$PY gguf-tools/scripts/v41_teacher.py "$HF" --ids "$IDS" --ntok "$NTOK" --out "$OUT/teacher.bin" --dump-mainh "$OUT/fp_mainh.bin" \
    > "$OUT/teacher.log" 2>&1 || { echo "★教师失败, 看 $OUT/teacher.log★"; exit 3; }
grep -a "PPL" "$OUT/teacher.log"

LOG "③ 三个一致率(分档)"
./gguf-tools/bench/dspark_agree "$OUT/pairs.bin" "$OUT/teacher.bin" > "$OUT/agree.txt" 2>&1 || { echo "★agree 失败★"; exit 4; }
grep -a -A7 "分档" "$OUT/agree.txt"; grep -a "①\|②\|③" "$OUT/agree.txt" | tail -3

LOG "④ 夹具三输入重放"
$PY gguf-tools/scripts/v41_dspark_fixture.py "$HF" --fix "$OUT/pairs.bin.fix" --pairs "$OUT/pairs.bin" \
    --fp-mainh "$OUT/fp_mainh.bin" --teacher "$OUT/teacher.bin" --out "$OUT/fixture.txt" > "$OUT/fixture.log" 2>&1 \
    || { echo "★夹具失败, 看 $OUT/fixture.log★"; exit 5; }
cat "$OUT/fixture.txt"
LOG "完成 $TAG"
