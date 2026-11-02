#!/bin/bash
# vstep_overnight_spark.sh — V 步两段过夜链(2026-11-02 深夜, 用户 "你自己想办法解决一下, 明天早上我要看到编程指标提升, 鹈鹕效果提升"):
#   段 A = 从 v11/ckpt_v07(编程判决份 −1.7%, 四尺 KLD 0.22510 对 c5f2 0.22712, 一小时预算到点时还在降)续训 L10-19, 料同 v11(三条编程流 + 实体短行, 知识留出按实体),
#          预算 2400 s —— 编程指标再往下走;
#   段 B = 从段 A 选中的 ckpt 起, 叠 L20-29, 料加官方 API 对齐的 agent 头行(evo/v002, 留出按任务), 预算 2400 s —— 规划段的风格 / 细节决策位往官方靠,
#          编程由训练器的配对门守(每轮判决份不显著比起点差才留), 选轮再要求判决份 ≤ 起点。
#   每段完过四把尺(vstep_spark.sh judge, ~10 分钟); 末了按编程尺选一个给 Mac 画鹈鹕(PICK 行):
#          B 过门(judge KLD ≤ A + 0.0002 且 Same ≥ A − 0.2 pp)用 B; 否则用 A(A 的 KLD 还得 < c5f2 0.22712, 否则退回 v11/ckpt_v07)。
# 为什么拆两段不合一段: v15 实撞, agent 头行与编程流在 L10-19 的码字上对着拉(四份全同意砍到 1/4), 合一段编程退 1.7σ; 天花板诊断 c5f2 的 177 个尾位
#   L20-29 换原件能修 39 个、L10-19 只 12 个 ⇒ 风格 / 细节的决策位放 L20-29, 编程均值留在 L10-19; 两段各自 ≤ 1 小时, 不堆料。
# 用法(spark 仓库根): nohup bash gguf-tools/scripts/vstep_overnight_spark.sh v16 v17 > /tmp/bugmd/vstep_overnight.log 2>&1 &
#   Mac 侧: nohup bash tests/asst/pel_overnight_mac.sh v16 > /tmp/bugmd/pel_overnight_v16.log 2>&1 &   (等 /tmp/bugmd/ent_overnight.done, 挂选中的 ckpt 画两趟, 画完卸)
# 判读(早上): 本日志 "== 段 A 判决" / "== 段 B 判决" 两段四把尺 + "PICK" 行; 鹈鹕 gguf/dsh/runs/local-pel-v16{a,b}/ws/index.html。
# 出错会怎样: 段 A 起点补丁挂不上 / 第 0 步判决份不比裸态低(续训没接上) ⇒ 整链停, done 文件第一行空, Mac 不画; 段 B 失败只是不用 B。
#   线上自始至终纯 c5f2(训练 / 判决都在停模型包装里, 包装退出按 serve_pick 装回)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
S=gguf-tools/scripts/vstep_spark.sh; V11=gguf/v41/posttrain/vstep-v11/ckpt_v07; AG=gguf/go-onebit/evo/v002
DONE=/tmp/bugmd/ent_overnight.done; PICKF=/tmp/bugmd/vstep_overnight.pick
M(){ echo "[vnight $(date '+%m-%d %H:%M:%S')] $*"; }
num(){ grep -a "$2" "$1" | tail -1 | sed -E 's/.*= *([0-9.]+).*/\1/'; }   # 判决日志里 "judge: Mean KLD = X" / "judge: Same top token = X%" 的数

if [ "${1:-}" = inner ]; then   # 停模型包装里的那段
    A="$2"; B="$3"; : > "$PICKF"
    M "== 段 A: prep(三条编程流 + 实体短行, 无 agent)"; $S prep | tail -2
    M "== 段 A: run $A L10-19 起点 $V11"; $S run "$A" 10-19 12 2400 4 0.0005 24 5 8 64 ent 0 4 "$V11" | tail -14
    K0=$(grep -a "第 0 步泛化留出(判决份)" "/tmp/bugmd/vstep_$A.log" | tail -1 | sed -E 's/.*判决份\) ([0-9.]+).*/\1/')
    grep -a "起点 = " "/tmp/bugmd/vstep_$A.log" | tail -1
    if [ -z "$K0" ] || awk "BEGIN{exit !($K0 > 0.1825)}"; then echo "★段 A 第 0 步判决份 '${K0:-空}' 不比裸态 0.1831 低: 起点补丁没接上, 整链停★"; exit 1; fi
    CKA=$($S pick "gguf/v41/posttrain/vstep-$A" 0); [ -n "$CKA" ] || { echo "★段 A 没有可用 ckpt★"; exit 1; }
    M "== 段 A 判决 $CKA(对照 v11 0.8367 / 0.22510 / 85.04, c5f2 0.8340 / 0.22712 / 85.07, 尾 123/177/634)"
    $S judge "$A" "$CKA" 2>&1 | tee "/tmp/bugmd/vstep_judge_$A.log" | grep -aE "judge|wt2|尾位|hold|★|DONE"
    KA=$(num "/tmp/bugmd/vstep_judge_$A.log" "judge: Mean KLD"); SA=$(num "/tmp/bugmd/vstep_judge_$A.log" "judge: Same top")
    M "== 段 B: prep agent=$AG"; $S prep agent="$AG" | tail -3
    M "== 段 B: run $B L20-29 起点 $CKA(agent 留出按任务, 短题 16/轮)"; $S run "$B" 20-29 12 2400 4 0.0005 16 5 8 64 agent 0 4 "$CKA" | tail -14
    CKB=$($S pick "gguf/v41/posttrain/vstep-$B" 1); KB=""; SB=""
    if [ -n "$CKB" ]; then
        M "== 段 B 判决 $CKB(判决份 ≤ 起点的轮里最低)"
        $S judge "$B" "$CKB" 2>&1 | tee "/tmp/bugmd/vstep_judge_$B.log" | grep -aE "judge|wt2|尾位|hold|★|DONE"
        KB=$(num "/tmp/bugmd/vstep_judge_$B.log" "judge: Mean KLD"); SB=$(num "/tmp/bugmd/vstep_judge_$B.log" "judge: Same top")
    else M "段 B 没有判决份 ≤ 起点的 ckpt, 不判"; fi
    PICK=""
    if [ -n "$KB" ] && [ -n "$KA" ] && awk "BEGIN{exit !($KB <= $KA + 0.0002 && $SB >= $SA - 0.2)}"; then PICK=$CKB
    elif [ -n "$KA" ] && awk "BEGIN{exit !($KA < 0.22712)}"; then PICK=$CKA
    else PICK=$V11; fi
    M "PICK $PICK  (A $CKA: KLD ${KA:-?} Same ${SA:-?};  B ${CKB:-无}: KLD ${KB:-?} Same ${SB:-?})"
    echo "$PICK" > "$PICKF"
    exit 0
fi

A="${1:-v16}"; B="${2:-v17}"; rm -f "$DONE" "$PICKF"
[ -s "$V11/base.fnv" ] && [ -s "$AG/rows.jsonl" ] || { echo "★起点 $V11 或 agent 料 $AG 缺★"; echo "" > "$DONE"; exit 1; }
bash gguf-tools/scripts/with_model_stopped_spark.sh bash "$0" inner "$A" "$B"
PICK=$(cat "$PICKF" 2>/dev/null); echo "${PICK:-}" > "$DONE"
M "OVERNIGHT_DONE '${PICK:-空}'(线上已按 serve_pick 装回纯 c5f2; Mac 侧看到 done 后挂它画鹈鹕)"
