#!/bin/bash
# dspark_quant.sh — DSpark drafter(mtp.0/1/2 三模块)独立量化(2026-09-07 改: 全参数化, 直呼量化器,
# 底座默认 Vision-Exp —— 与部署骨架/专家同底座, 见 fable5 09-06 铁律"量化只许 Vision-Exp 底座")。
# 产物 = 仅 mtp.* 张量的独立 gguf, 引擎 --draft-gguf 挂主模型旁(--spec 投机解码用)。
# 档位铁律(引擎 drafter kernel 的类型假设, 08-20 定罪, 错档=NaN/越界):
#   attn 矩阵 q8_0 / hc_attn_fn,hc_ffn_fn f16 / hc_head_fn f32 / router(gate_inp) f16 / 1D f32 / markov q8_0;
#   仅三家 exps 随档位: q2=iq2_xxs+q2_k / q4=q4_k / q8=q8_0。override 先到先得 ⇒ 具体条目必须在 mtp.= 之前。
# 用法: dspark_quant.sh <q2|q4|q8> [HF 目录=hf/DeepSeek-V4-Flash-Vision-Exp] [产物=gguf/ds4-dspark-<底座>-<档>.gguf]
# 体积账: 每模块专家 6.4B 参数 ×3 ⇒ q2 ≈6.0 GB / q4 ≈11.4 GB / q8 ≈20.5 GB(+attn/胶水 ~0.4 GB)。
set -uo pipefail
ROOT="$HOME/ds4-main"
TIER="${1:?档位 q2|q4|q8}"
HF="${2:-$ROOT/hf/DeepSeek-V4-Flash-Vision-Exp}"
ATTN="${4:-q8_0}"   # attn 矩阵档(q8_0|q4_k, 09-07): q8_0 在 5 行小批只能走 cuBLAS f16 影子路(每轮 ~4 ms, 离字节墙 3×); q4_k 走稠密 tile 多 token 核
case "$(basename "$HF")" in
    *Vision-Exp*) BASE=ve ;; *0731*) BASE=0731 ;; *DSpark*) BASE=dspark ;; *) BASE=$(basename "$HF") ;;
esac
SUF=""; [ "$ATTN" = q8_0 ] || SUF="-a${ATTN%_k}"
OUT="${3:-$ROOT/gguf/ds4-dspark-$BASE-$TIER$SUF.gguf}"
TMPL="$ROOT/gguf/go-onebit/r30/template_head.gguf"
Q="$ROOT/gguf-tools/deepseek4-quantize"
case "$TIER" in
    q2) G=iq2_xxs; U=iq2_xxs; D=q2_k ;;
    q4) G=q4_k;    U=q4_k;    D=q4_k ;;
    q8) G=q8_0;    U=q8_0;    D=q8_0 ;;
    *) echo "档位只认 q2|q4|q8"; exit 1 ;;
esac
LOG(){ echo "[dspark_quant $(date '+%m-%d %H:%M:%S')] $*"; }
[ -x "$Q" ] || { LOG "★量化器缺 $Q★"; exit 2; }
[ -s "$TMPL" ] || { LOG "★模版缺 $TMPL★"; exit 2; }
[ -s "$HF/model.safetensors.index.json" ] || { LOG "★HF 缺 $HF★"; exit 2; }
grep -q '"mtp\.2\.' "$HF/model.safetensors.index.json" || { LOG "★$HF 没有 mtp.2.* 张量(不是带 drafter 的底座)★"; exit 2; }
FREE=$(df -BG --output=avail "$(dirname "$OUT")" | sed -n 2p | tr -dc 0-9)
[ "${FREE:-0}" -ge 30 ] || { LOG "★盘 ${FREE}G < 30G★"; exit 4; }
# 内存闸: 量化器逐张量流式, drafter 三模块 RSS 应远低于 40 GB; 越线即杀, 别把 121 GB 机器拖假死。
( while true; do
    P=$(pgrep -nf "deepseek4-quantize .*mtp-only" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(awk '/VmRSS/{print int($2/1024)}' "/proc/$P/status" 2>/dev/null || true)
    [ -n "${MB:-}" ] && [ "$MB" -gt 40000 ] && { echo "[dspark_quant][wdog] RSS ${MB}MB > 40000MB 杀" >&2; kill -9 "$P"; }
    sleep 5; done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
EXPS=""; HARD=""
for N in 0 1 2; do
    EXPS="$EXPS --tensor-type mtp.$N.ffn_gate_exps.weight=$G --tensor-type mtp.$N.ffn_up_exps.weight=$U --tensor-type mtp.$N.ffn_down_exps.weight=$D"
    HARD="$HARD --tensor-type mtp.$N.ffn_gate_inp.weight=f16 --tensor-type mtp.$N.hc_attn_fn.weight=f16 --tensor-type mtp.$N.hc_ffn_fn.weight=f16"
    for T in attn_q_a attn_q_b attn_kv attn_output_a attn_output_b; do HARD="$HARD --tensor-type mtp.$N.$T.weight=$ATTN"; done
done
LOG "drafter ← $HF 档 $TIER → $OUT"
"$Q" --hf "$HF" --template "$TMPL" --out "$OUT" \
    --mtp-append 3 --mtp-only \
    --tensor-type mtp.2.hc_head_fn.weight=f32 \
    $EXPS $HARD --tensor-type mtp.=q8_0 \
    --threads 20 --overwrite || { LOG "★量化失败★"; exit 5; }
LOG "产物 $OUT $(ls -l "$OUT" | awk '{printf "%.2f GB", $5/1e9}')"
