#!/bin/bash
# entity_overnight_spark.sh — 实体知识召回 ③ 的部署量级长上下文夜跑(2026-11-01 深夜, fable5 "修正只在训练见过的上下文长度内起作用, 训多长管多长")。
#
# 做什么: 315 行实体训练行全上, 每行前面拼 40 个别的实体问答当上下文(≈ 8.6k 位, dsh agent 上下文的量级), 教师表从短表平移(ptfprefix), ③ 训 3 轮;
#   训完在停模型包装里过三把尺: ① 实体留出尺(entity_probe_spark.sh score, 16.8k 流) ② 编程锁门 + wt2(v41_judge.sh 两臂: 现役 ② / ②+③)
#   ③ 鹈鹕规划位探针(plan_adjective_probe_spark.sh)。选轮 = 训练器留出损失最低的 ckpt。结束写 /tmp/bugmd/ent_overnight.done(第一行 = 选中的 ③ 目录),
#   Mac 侧 tests/asst/pel_overnight_mac.sh 等这个文件, 挂 ③ 画两趟鹈鹕再卸。
# 用法(spark 仓库根): nohup bash gguf-tools/scripts/entity_overnight_spark.sh [标签=ent04] [料=long|merge] > /tmp/bugmd/ent_overnight.log 2>&1 &
#   料 long  = 315 行实体长前缀(ent04 那份, 11-01 夜);
#   料 merge = 长前缀 315 行 + v002 的 92 个 agent 请求体头行(gguf/go-onebit/evo/v002, 质感"抬标准"料, 10k 位接力) + 短行每 3 行取 1(105 行)。
#     为什么合: ent04 结论(fable5 11-02 05:05)—— 知识 ③ 和质感 ③ 是两份各管一半, 且只训长行短上下文知识题退了(探针 大 0.80 → 0.55), 料要短长混;
#     教师表直接 cat(PTF1 可多段首尾相接, 训练器按整条 ids 精确配, 多出来的记录不碍事); 长/短行 text 同串 ⇒ 留出按 text 哈希分桶, 同一实体两种形态落同一桶, 不漏。
#     ★11-02 07:1x 用户叫停(500 行 × 3 轮账上 5 小时): 本地一轮 ≤ 1 小时, 不许拿语料和时长堆★ —— merge 留着当记录, 别再发。
#   料 mix   = 315 行实体行, 前缀 0 / 12 / 40 个问答轮流(≈ 0.1k / 2.2k / 7k 位, 均 ≈ 3.1k), 2 轮, 训练器 budget_s 2400 硬闸(40 分钟训 + 装载/评估 ≈ 55 分钟内)。
#     为什么: 只用昨晚验出的杠杆(实体描述 × 官方 top-20 × 部署量级前缀, 留出实体 +10~12%), 短/中/长三档盖住部署两头(短问答 / 10k agent), 不加别的料。
# 判读(早上看): /tmp/bugmd/ent_overnight.log 末尾三段 "== 实体尺" / "== 编程尺" / "== 探针"; 线上自始至终是纯 c5f2(训练/判决都在停模型包装里, 结束按 serve_pick 装回)。
# 出错会怎样: 任一步失败打 ★ 并继续下一把尺(能量多少量多少), done 文件照写(第一行可能是空 = 没有 ③ 可挂, Mac 侧看到空就不画直接退)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
W=gguf/go-onebit/evo/entity; PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; TAG="${1:-ent04}"; MODE="${2:-long}"; NCTX=40
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"
M(){ echo "[night $(date '+%m-%d %H:%M:%S')] $*"; }
DONE=/tmp/bugmd/ent_overnight.done; rm -f "$DONE"

if [ ! -s "$W/long${NCTX}.ptf" ]; then
    M "① 料: 315 行 × 前缀 $NCTX 个问答"
    "$PY" speed-bench/bugmd_ids_tools.py entlong "$W/train_rows.jsonl" "$W/train_ok.jsonl" "$W/long${NCTX}_rows.jsonl" 400 "$NCTX" 7 1 || { echo "" > "$DONE"; exit 1; }
    "$PY" gguf-tools/scripts/api_teacher.py simids --hf "$HF" --rows "$W/long${NCTX}_rows.jsonl" --out "$W/long${NCTX}.seqs" | tail -1
    "$PY" speed-bench/bugmd_ids_tools.py ptfprefix "$W/train.ptf" "$W/long${NCTX}.seqs" "$W/long${NCTX}.ptf" || { echo "" > "$DONE"; exit 1; }
    awk '{n+=NF} END{printf "长行 %d 条 %d 位, 均 %d\n", NR, n, n/NR}' "$W/long${NCTX}.seqs"
fi
DATA=$W/long${NCTX}_rows.jsonl; TPTF=$W/long${NCTX}.ptf; EP=3; BUDGET=0
if [ "$MODE" = mix ]; then
    MIX=0,12,40; DATA=$W/mix_rows.jsonl; TPTF=$W/mix.ptf; EP=2; BUDGET=2400
    M "① 料: 315 行 × 前缀 $MIX 个问答轮流"
    "$PY" speed-bench/bugmd_ids_tools.py entlong "$W/train_rows.jsonl" "$W/train_ok.jsonl" "$DATA" 400 "$MIX" 7 1 || { echo "" > "$DONE"; exit 1; }
    "$PY" gguf-tools/scripts/api_teacher.py simids --hf "$HF" --rows "$DATA" --out "$W/mix.seqs" | tail -1
    "$PY" speed-bench/bugmd_ids_tools.py ptfprefix "$W/train.ptf" "$W/mix.seqs" "$TPTF" || { echo "" > "$DONE"; exit 1; }
    awk '{n+=NF; if(NF<512)a++; else if(NF<4000)b++; else c++} END{printf "混合行 %d 条 %d 位, 均 %d; 短/中/长 = %d/%d/%d\n", NR, n, n/NR, a, b, c}' "$W/mix.seqs"
elif [ "$MODE" = merge ]; then
    V=gguf/go-onebit/evo/v002; DATA=$W/merge_rows.jsonl; TPTF=$W/merge.ptf
    [ -s "$V/rows.jsonl" ] && [ -s "$V/teacher.ptf" ] || { echo "★没有 v002 的 agent 料($V)★"; echo "" > "$DONE"; exit 1; }
    { cat "$W/long${NCTX}_rows.jsonl" "$V/rows.jsonl"; awk 'NR % 3 == 1' "$W/train_rows.jsonl"; } > "$DATA"
    cat "$W/long${NCTX}.ptf" "$V/teacher.ptf" "$W/train.ptf" > "$TPTF"
    M "① 合料: 长前缀 $(wc -l < "$W/long${NCTX}_rows.jsonl") + agent $(wc -l < "$V/rows.jsonl") + 短 $(awk 'NR % 3 == 1' "$W/train_rows.jsonl" | wc -l) = $(wc -l < "$DATA") 行; 教师表 $(du -m "$TPTF" | cut -f1) MB"
fi

bash gguf-tools/scripts/with_model_stopped_spark.sh bash -c '
set -u; W='"$W"'; TAG='"$TAG"'; HF='"$HF"'; GG='"$GG"'; ARM='"$ARM"'; DATA='"$DATA"'; TPTF='"$TPTF"'; EP='"$EP"'; BUDGET='"$BUDGET"'
M(){ echo "[night $(date "+%m-%d %H:%M:%S")] $*"; }
M "② 训 ③ $EP 轮, 预算 $BUDGET s(料 $DATA)"
gguf-tools/scripts/entity_probe_spark.sh train "$TAG" "$DATA" "$TPTF" "$EP" "$BUDGET"
# 选轮: 训练器每轮末 "★轮 N 末(… 留出损失 X(第 0 步 …" 里留出损失最低的那轮
BEST=$(grep -a "★轮 [0-9]* 末" /tmp/bugmd/ent_train_$TAG.log | sed -E "s/.*★轮 ([0-9]+) 末.*留出损失 ([0-9.]+).*/\2 \1/" | sort -n | head -1 | awk "{print \$2}")
CK=""; [ -n "$BEST" ] && CK=gguf/v41/posttrain/entity-$TAG/ckpt_e$(printf %02d "$BEST")
[ -d "$CK" ] || { echo "★没有可用 ckpt(BEST=$BEST)★"; echo "" > '"$DONE"'; exit 1; }
M "选轮 e$BEST → $CK"
M "== 实体尺(16.8k 流, 留出 + 训练探针; 对照 c5f2 0.364/0.353 KL 0.970/0.975, ent04 0.401/0.396 KL 0.897/0.878)"
gguf-tools/scripts/entity_probe_spark.sh score "$TAG" "$CK" || echo "★实体尺失败★"
M "== 编程尺(judge.ids 15360 + wt2 512; 臂 1 现役 ② / 臂 2 ②+③; 对照 ent04 Same 84.90 KLD 0.22625)"
ai=0
for arm in "engine:$GG:$ARM" "engine:$GG:$ARM::$CK"; do
    ai=$((ai + 1)); an=$([ $ai = 1 ] && echo c5f2 || echo "c5f2+③")
    for ids in "gguf/go-onebit/code/judge.ids 15360" "gguf/go-onebit/g7/wt2.ids 512"; do
        set -- $ids; lg=/tmp/bugmd/night_judge_${TAG}_$(basename "$1" .ids)_arm$ai.log
        bash gguf-tools/scripts/v41_judge.sh "$1" "$2" --hf "$HF" "$arm" > "$lg" 2>&1 \
            && grep -E "Same top|Mean KLD|Σmin|PPL\(student" "$lg" | sed "s|^ *|  $an $(basename "$1" .ids): |" || echo "★判决 $an $1 失败(看 $lg)★"
    done
done
rm -f gguf/v41judge/stu_*_eng_*_pt_*entity-$TAG*.bin gguf/v41judge/stu_*_eng_*_pt_*entity-$TAG*.bin.*   # ③ 臂学生 logits(编程一份 7.9 GB)用完即删
M "== 探针(鹈鹕规划位 大/长/喉; 对照 adj_c5f2.tsv)"
bash gguf-tools/scripts/plan_adjective_probe_spark.sh "$TAG" "$CK" || echo "★探针失败★"
printf "%s\n" "$CK" > '"$DONE"'
M "夜跑判决段结束"
'
[ -s "$DONE" ] || echo "" > "$DONE"
M "OVERNIGHT_DONE $(cat "$DONE")"
