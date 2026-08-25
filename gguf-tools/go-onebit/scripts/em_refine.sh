#!/bin/bash
# em_refine.sh — VQ-EM 码本精修一条龙(2026-08-23 铁律: 一切脚本化·改动即全量重跑)。
# 段: clean(恢复干净输入态) → em(全43层, 单一二进制) → merge(重合并GGUF) → judge(wt2 Σmin)
# 用法: em_refine.sh [all|clean|em|merge|judge]
# all=代码/参数变更后全量重跑; 上游产物完好仅下游段失败时, 用分段入口续跑(merge/judge), 别无脑 all 毁产物(2026-08-23教训: all 的 clean 先删产物)
# 输入态: vq86h_noz/layers = 量化半纯量化产物(保全区, 只读)。
# 判决对表: 裸 vq86h Σmin 0.7338(批量路) / 官方 q2 0.7187 / 目标 ≥0.90。
set -uo pipefail
ROOT="$HOME/ds4-main"
D2="$ROOT/gguf/go-onebit/vqhalf"
GT="$ROOT/gguf-tools"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
ITERS=3   # iters 甜点实测: 3 轮=93% 收益 40% 时间(L20: -12.26%@42s vs -13.14%@107s)
LAYERS_ARG="${2:-0-42}"   # 位置参数2=层范围(探针入口, 如 18-22); 默认全 43 层
ESTEP_ARG="${3:-gptvq}"   # 位置参数3=E步(gptvq|em); 对照探针解耦 ④升格 与 ⑤补偿指派
LOG(){ echo "[em_refine $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }

stage_clean(){
    [ -d "$D2/vq86h_noz/layers" ] || DIE "干净输入态缺($D2/vq86h_noz/layers)"
    N=$(ls "$D2/vq86h_noz/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)
    [ "$N" = 43 ] || DIE "干净态不齐 $N/43"
    LOG "①恢复干净输入态(全量重跑铁律)"
    rm -rf "$D2/vq86h_em"
    mkdir -p "$D2/vq86h_em"
    cp -r "$D2/vq86h_noz/layers" "$D2/vq86h_em/layers" || DIE "拷贝失败"
    rm -f "$D2/vq86h_em/layers"/*.pre_em "$D2/vq86h_em/layers"/zrec_*.bin
    rm -f "$ROOT/gguf/ds4-vq86h-em.gguf"   # 陈旧合并产物必清, 否则 merge 幂等跳过→终判量旧模型(2026-08-23事故)
    LOG "①就绪: vq86h_em/layers (纯量化态)"
}

stage_em(){
    [ -x "$GT/vq_em" ] || DIE "vq_em 二进制缺"
    N=$(ls "$D2/vq86h_em/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)
    [ "$N" = 43 ] || DIE "输入层不齐 $N/43, 先跑 clean"
    LOG "②GPTVQ 层$LAYERS_ARG(iters=$ITERS, 单一二进制 md5=$(md5sum $GT/vq_em | cut -c1-8))"
    "$GT/vq_em" --hf "$HF" --dql "$D2/vq86h_em/layers" \
        --anchor "$D2/anchor_vqhalf_q_s8192.bin" --layers $LAYERS_ARG --iters $ITERS --estep $ESTEP_ARG \
        || DIE "EM 失败"
    LOG "②收官"
}

stage_merge(){
    M86_LAYERS="$D2/vq86h_em/layers" M86_MDL="$ROOT/gguf/ds4-vq86h-em.gguf" \
    M86_SKEL="$R30/r30_skeleton.gguf" M86_ZCH= \
        bash "$ROOT/gguf-tools/go-onebit/scripts/merge_base86p.sh" || DIE "合并失败"
}

stage_judge(){
    cd "$ROOT"
    LOG "④终判 wt2(批量路, 对表: 裸 0.7338 / 官方 0.7187)"
    timeout --foreground 3000 ./ds4 --cuda -m "$ROOT/gguf/ds4-vq86h-em.gguf" \
        --eval-ids "$G7/wt2.ids" --eval-logits /tmp/wt2_em_ev.bin </dev/null 2>&1 | tail -1
    "$GT/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --stu-raw /tmp/wt2_em_ev.bin 2>&1 | grep -E "Σmin|PPL\(stu|Mean KLD|Same top"
    LOG "④收官"
}

ST="${1:-all}"
case "$ST" in
  clean) stage_clean;; em) stage_em;; merge) stage_merge;; judge) stage_judge;;
  all) stage_clean; stage_em; stage_merge; stage_judge;;
  *) echo "段: clean em merge judge all"; exit 2;;
esac
LOG "段 $ST 完成"
