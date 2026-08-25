#!/bin/bash
# dspark_quant.sh — DSpark drafter 独立量化脚本(2026-08-20 用户设计: 四文件架构之第3文件)。
# 产物 = 3 模块(mtp.0/1/2) drafter-only gguf, 引擎经 DS4_DRAFT_GGUF 挂载主模型旁。
# 档位铁律(引擎 kernel 类型假设): attn 矩阵 q8_0 / hc_*_fn f16 / hc_head_fn f32 /
#   router(gate_inp) f16 / 1D f32(1d-guard) / markov q8_0; 仅 exps 随 TIER 降档。
# 用法: TIER=q2|q4 [HF=...] [OUT=...] bash dspark_quant.sh
set -u
TIER="${TIER:-q2}"
HF="${HF:-$HOME/ds4-main/hf/DeepSeek-V4-Flash-DSpark}"
OUT="${OUT:-$HOME/ds4-main/gguf/ds4-dspark-drafter3-$TIER.gguf}"
case "$TIER" in
    q2) G=iq2_xxs; U=iq2_xxs; D=q2_k ;;
    q4) G=q4_k;    U=q4_k;    D=q4_k ;;
    *) echo "TIER 只认 q2|q4"; exit 1 ;;
esac
EXPS=""
for N in 0 1 2; do
    EXPS="$EXPS --tensor-type mtp.$N.ffn_gate_exps.weight=$G"
    EXPS="$EXPS --tensor-type mtp.$N.ffn_up_exps.weight=$U"
    EXPS="$EXPS --tensor-type mtp.$N.ffn_down_exps.weight=$D"
done
HARD=""
for N in 0 1 2; do
    HARD="$HARD --tensor-type mtp.$N.ffn_gate_inp.weight=f16"
    HARD="$HARD --tensor-type mtp.$N.hc_attn_fn.weight=f16"
    HARD="$HARD --tensor-type mtp.$N.hc_ffn_fn.weight=f16"
done
exec env HF="$HF" OUT="$OUT" bash "$(dirname "$0")/quant_allq2_spark.sh" \
    --mtp-append 3 --mtp-only \
    --tensor-type mtp.2.hc_head_fn.weight=f32 \
    $EXPS $HARD --tensor-type mtp.=q8_0
