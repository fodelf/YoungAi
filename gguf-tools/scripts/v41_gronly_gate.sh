#!/bin/bash
# v41_gronly_gate.sh — q4k 骨架 + 纯增益反修(gr-only)的三道门(2026-09-20 夜, 用户定"不要路由反修, 只要增益反修")。
#
# 为什么要三道门而不是一道: 五指标是 teacher-forced(每一步喂 FP 的正确前文, 只考一步再平均), 天生看不见
# 温 0 自由生成的复读 —— 09-20 夜实撞: grrb 在五指标上 +0.9 pp, 在金融提示上却从 1500 字起整段死循环。
#   ① 金融判决料 j 8192 五指标: v41_judge.sh engamp …:gr 一趟 = 解 gr-only + 挂上判(先例 ~65 min)
#   ② wt2 512 守门("不忘老本事"): 裸 / +gr-only 同趟
#   ③ 打转尺: 金融提示 + 英文 readme, 温 0 1400 token(d1_kv_ring_gate.sh loop)
# 可选收尾 D1(给一个反修目录, 一般是 grrb): 用 loop 模式在金融提示上重生成一遍拿引擎自己打的精确 ids(含套模板的提示),
#   落 教师锚 + 学生 logits, 再按复读周期分遍看"老师在复读位选谁"(d1_kv_ring_gate.sh d1) ——
#   回答"路由反修指标变好为什么结果变差"里那条最该先查的解释。
# ★ids 要拷成短名再喂 judge★(09-20 实撞): judge 拿 ids 文件名 + 模型名 + 反修名拼学生 logits 文件名, loop 模式的
#   长 tag 拼出来 >255 字节, fopen 直接失败, 教师那 15 分钟白跑。
# 出错会怎样: 任一段 ★ 就停在那一段(exit 非 0), 不往下跑 —— 半成品反修目录挂上去不报错只出假数。
# 用法(spark 本机): v41_gronly_gate.sh <gguf> [D1 学生要挂的反修目录]
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
GG="${1:?gguf}"; D1AMP="${2:-}"
J="$ROOT/gguf-tools/scripts/v41_judge.sh"; L="$ROOT/speed-bench/d1_kv_ring_gate.sh"; OUT=/tmp/d1-kv-ring
FIT=gguf/go-onebit/vqfin41/vqhalf_a.ids; JUD=gguf/go-onebit/vqfin41/vqhalf_j.ids; WT2=gguf/go-onebit/g7/wt2.ids
for f in "$GG" "$FIT" "$FIT.layout" "$JUD" "$WT2"; do [ -s "$f" ] || { echo "★缺 $f★"; exit 1; }; done
ARM="gguf/v41/$(basename "$GG" .gguf)-gr-vqfin41_vqhalf_a_n8192-engine"   # 与 v41_judge.sh engamp 的命名逐字同
M(){ echo "[gate $(date '+%m-%d %H:%M:%S')] $*"; }

M "① 解 gr-only + 金融 j 8192 五指标 → $ARM"
"$J" "$JUD" 8192 "engamp:$FIT:8192:$GG:0:40:gr"
grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null || { M "★反修目录没有完成标记, 停★"; exit 3; }
M "② wt2 512 守门: 裸 / +gr-only"
"$J" "$WT2" 512 "engine:$GG" "engine:$GG:$ARM"
M "③ 打转尺: 金融提示 + readme, +gr-only"
mkdir -p "$OUT"; head -c 8000 speed-bench/readme_en_x80.txt > "$OUT/p2k.txt"
"$L" none "$GG" 1400 loop "$ARM"
"$L" none "$GG" 1400 loop "$ARM" "$OUT/p2k.txt"

if [ -n "$D1AMP" ]; then
  [ -d "$D1AMP" ] || { M "★D1 反修目录不存在: $D1AMP★"; exit 1; }
  M "D1: 金融提示上重生成($(basename "$D1AMP"))拿精确 ids"
  "$L" none "$GG" 1400 loop "$D1AMP"
  SRC="$OUT/loop_$(basename "$GG" .gguf)_$(basename "$D1AMP")_fin_chat_prompt"
  [ -s "$SRC.ids" ] && [ "$(cat "$SRC.np")" -gt 0 ] || { M "★ids 没落好($SRC.ids / .np), 二进制是否带 [ptok] 打点?★"; exit 2; }
  cp "$SRC.ids" "$OUT/d1.ids"; cp "$SRC.np" "$OUT/d1.np"
  N=$(wc -l < "$OUT/d1.ids"); TAGD="d1-kv-ring_d1_n$N"
  M "D1: 教师锚 + 学生 logits, n=$N(提示 $(cat "$OUT/d1.np"))"
  "$J" "$OUT/d1.ids" "$N" "engine:$GG:$D1AMP" || { M "★D1 的 judge 失败★"; exit 2; }
  TB="gguf/v41judge/teacher_$TAGD.bin"; SB="gguf/v41judge/stu_${TAGD}_eng_$(basename "$GG" .gguf)_amp_$(basename "$D1AMP").bin"
  [ -s "$TB" ] && [ -s "$SB" ] || { M "★D1 产物缺: $TB / $SB★"; ls gguf/v41judge/ | grep "$TAGD"; exit 2; }
  M "D1: 老师在复读位选谁"
  "$L" none "$GG" 0 d1 none "$OUT/d1.ids" "$TB" "$SB"
fi
M "GATE_DONE"
