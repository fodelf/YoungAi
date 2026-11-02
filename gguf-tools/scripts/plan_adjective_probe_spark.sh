#!/bin/bash
# plan_adjective_probe_spark.sh — 规划段"尺寸形容词"探针尺(2026-11-01)。鹈鹕像不像鸭, 定在规划开头官方写"长喙 / 大喉囊 / 又长又大"的那几位:
#   10-27 量过原件给 0.74~1.00、量化后 0.00~0.28(fable5 10-27 "FP 探针定位")。这把尺把那次手算固化: 同一批序列上, 在【真值下一个 token 就是 大/长/喉/large/long】的位置,
#   比 FP 锚与学生锚各给多少(bugmd_ids_tools.py tokprobe 的几何均值), 学生 / FP 的比值 = 这类决策被压了几倍。
# 序列(10-27 存在 gguf/go-onebit/code/peldist/, 锚在 gguf/v41judge/): 官方单轮鹈鹕 4 份(前 1536 位, 提示 48)+ 知识题"鹈鹕长什么样"中英两份(提示 43)。
#   ★这些序列只拿来量, 不进任何料★(鹈鹕在禁词表里)。
# 用法(spark 仓库根, 包在停模型包装里): with_model_stopped_spark.sh gguf-tools/scripts/plan_adjective_probe_spark.sh <标签> [③ 后训练目录]
#   学生臂 = serve_pick 的底座 + 侧车(+ 可选 ③), 学生锚算完即删(每份 1536 × 词表 ≈ 2.4 GB), 教师锚是缓存的不动。
# 判读: 每行 "序列 token FP几何均值 学生几何均值 比值 命中数"; 比值 < 1 = 被压; 一轮训练前后各跑一次, 比值往 1 靠才算修到这一位。
# 出错会怎样: 缺教师锚直接退(10-27 的锚被清了就要先 v41_judge 教师臂重打, 要装原件, 另算); v41_judge 失败的序列跳过并打 ★。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签}"; PT="${2:-}"; HF=hf/DeepSeek-V4.1-Flash-1017; HB=$(basename "$HF")
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"; GB=$(basename "$GG" .gguf); AB=$(basename "$ARM")
TOOLS=speed-bench/bugmd_ids_tools.py; OUT=gguf/go-onebit/evo/probe/adj_$TAG.tsv; mkdir -p "$(dirname "$OUT")"
ARMSPEC="engine:$GG:$ARM"; SFX="_amp_$AB"
[ -z "$PT" ] || { ptn="${PT%/}"; ptn="${ptn#"$ROOT"/}"; ARMSPEC="$ARMSPEC::$PT"; SFX="${SFX}_pt_${ptn//\//_}"; }
declare -A NP=([know_zh]=43 [know_en]=43 [peld_00]=48 [peld_01]=48 [peld_02]=48 [peld_03]=48)
declare -A TOK=([大]=547 [长]=1122 [喉]=23331 [large]=3226 [long]=1606)
printf '# %s 臂 %s %s\n序列\ttoken\tFP几何均值\t学生几何均值\t学生/FP\t命中\n' "$(date '+%m-%d %H:%M')" "$ARMSPEC" "$(date +%s)" > "$OUT"
for seq in know_zh know_en peld_00 peld_01 peld_02 peld_03; do
    case $seq in know_*) ids=gguf/go-onebit/code/peldist/know/$seq.ids; n=$(awk -v s="$seq" '$1 == s {print $2}' gguf/go-onebit/code/peldist/know/list.txt); tag="know_${seq}_n$n";;
                 *)      ids=gguf/go-onebit/code/peldist/api/$seq.ids; n=1536; tag="api_${seq}_n$n";; esac
    T="gguf/v41judge/teacher_${tag}_$HB.bin"; S="gguf/v41judge/stu_${tag}_eng_${GB}${SFX}.bin"
    [ -s "$T" ] || { echo "★缺教师锚 $T★"; exit 1; }
    bash gguf-tools/scripts/v41_judge.sh "$ids" "$n" --hf "$HF" "$ARMSPEC" > "/tmp/bugmd/adj_${TAG}_$seq.log" 2>&1 || { echo "★$seq 学生臂失败, 看 /tmp/bugmd/adj_${TAG}_$seq.log★"; continue; }
    [ -s "$S" ] || { echo "★$seq 学生锚 $S 没落下来★"; continue; }
    for w in 大 长 喉 large long; do
        ft=$(python3 "$TOOLS" tokprobe "$T" "$ids" "${TOK[$w]}" "${NP[$seq]}" 2>/dev/null | tail -1); st=$(python3 "$TOOLS" tokprobe "$S" "$ids" "${TOK[$w]}" "${NP[$seq]}" 2>/dev/null | tail -1)
        # tokprobe 末行形如 "  几何均值 p(tok) = 0.1234; 中位 0.2; tok 是 argmax 的位置 k/n"(bugmd_ids_tools.py cmd_tokprobe)
        fg=$(grep -o '几何均值 p(tok) = [0-9.e-]*' <<<"$ft" | awk '{print $NF}'); sg=$(grep -o '几何均值 p(tok) = [0-9.e-]*' <<<"$st" | awk '{print $NF}'); hit=$(grep -o '位置 [0-9]*/[0-9]*' <<<"$ft" | awk -F/ '{print $2}')
        [ -n "$fg" ] && [ -n "$sg" ] || continue   # 这份序列里没有该 token 的真吐位
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$w" "$fg" "$sg" "$(awk -v a="$fg" -v b="$sg" 'BEGIN{printf "%.3f", (a > 0 ? b / a : 0)}')" "$hit" | tee -a "$OUT"
    done
    rm -f "$S" "$S".*   # 学生锚用完即删(教师锚留)
done
echo "ADJ_DONE $OUT"
