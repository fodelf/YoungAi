#!/bin/bash
# v41_dspark_variants.sh — DSpark 草稿器 main_hidden 取法验证(2026-09-18, 在 spark 上跑)。
#
# 为什么: 官方参考实现(model.py)的取法是"目标层输入的 hc 四路均值"; 可官方三塔喂教师算的这份 FP 隐态, 在 PPL 2 的
# 公告文本上也只中 FP 的 0.6(最易档 0.7, 底座同档 0.96)。一个生产级 MTP 头不该这样 ⇒ 怀疑参考实现的取法与训练口径不同。
# 做法: 教师前向一趟, 目标层各落四种候选(层输入均值/层输入按 pre_mix 折叠/层输出均值/层输出折叠), 逐个喂夹具(--fp-only),
# 看哪一种让最易档命中抬到 0.9 档 —— 抬得起来的那一种就是训练口径, 引擎照它改 hc_mean 的取点。
# 前置: v41_dspark_fixture_run.sh <tag> 已跑完(要 pairs.bin/.fix 与 teacher.bin)。
# 用法: gguf-tools/scripts/v41_dspark_variants.sh <tag> <ids> [ntok=1536]   产物: /tmp/dcap7/<tag>/var_<候选>.txt
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
TAG="${1:?用法: $0 <tag> <ids> [ntok]}"
IDS="${2:?}"
NTOK="${3:-1536}"
HF=hf/DeepSeek-V4.1-Flash
PY=~/v41env/bin/python
OUT=/tmp/dcap7/$TAG
LOG() { echo "[$(date +%H:%M:%S)] $*"; }
for f in "$OUT/pairs.bin" "$OUT/pairs.bin.fix" "$OUT/teacher.bin"; do [ -f "$f" ] || { echo "★缺 $f: 先跑 v41_dspark_fixture_run.sh $TAG★"; exit 1; }; done
# ★锚定命令行开头★(2026-09-18 实撞): 排队等着的 `sh -c "until …; ./ds4 -m …"` 的 argv 里也含 "ds4 -m", 不锚定就把
# 一个在睡觉的等待壳当成模型进程, 链拒发退出(exit 1), 而它写的 DONE 标记又把后面排队的任务放行 —— 顺序全乱。
if pgrep -f "^\./ds4 -m |^[^ ]*python[^ ]* gguf-tools/scripts/v41_teacher\.py" >/dev/null; then echo "★有模型进程在跑, 不发★"; exit 1; fi

LOG "① 教师前向: 四种候选 main_hidden → $OUT/mhv.*.bin"
$PY gguf-tools/scripts/v41_teacher.py "$HF" --ids "$IDS" --ntok "$NTOK" --dump-mainh-variants "$OUT/mhv" \
    > "$OUT/teacher_var.log" 2>&1 || { echo "★教师失败, 看 $OUT/teacher_var.log★"; exit 2; }
grep -a "取法验证\|PPL" "$OUT/teacher_var.log"

for k in inmean inpre outmean outpre; do
    LOG "② 夹具喂 $k"
    $PY gguf-tools/scripts/v41_dspark_fixture.py "$HF" --fix "$OUT/pairs.bin.fix" --pairs "$OUT/pairs.bin" \
        --fp-mainh "$OUT/mhv.$k.bin" --teacher "$OUT/teacher.bin" --fp-only --out "$OUT/var_$k.txt" \
        > "$OUT/var_$k.log" 2>&1 || { echo "★夹具($k)失败, 看 $OUT/var_$k.log★"; exit 3; }
    cat "$OUT/var_$k.txt"
done
LOG "完成 $TAG 取法验证"
