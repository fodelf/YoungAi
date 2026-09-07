#!/bin/bash
# q2k_amp_spark.sh — Q2_K 骨架部署态的反修(用户设计: 学生 = 引擎捕获真值) + sweep + 引擎路五指标(2026-09-06)。
#
# 为什么要单独一条链: 冠军配方 amp_clean_full.sh 是 FP-x 口径(zlayer 不带 XCAP; --gguf 只换专家权重),
# 所以任何现有解算路里骨架都是 FP —— 部署骨架换成 Q2_K 时反修一无所见, 重跑只会复现冠军 zchain。
# 用户 08-22 定案的设计"学生 = 引擎捕获真值(raw_ffn_in / raw_ffn_out)"(amp_campaign.sh 头注)是唯一能让
# 骨架误差进解算的口径 = zlayer 第 7 位 XCAP。本链把冠军配方(K64 / INJ=1 dql 注入 / q 份 gate-anchor)
# 原样保留, 只把 x 与被乘量换成 Q2_K 裸模型的引擎真值。
# 链: ① 合并【裸】Q2_K 模型(层件 = champ86/layers_quant 纯量化态, 无内嵌 op, 否则捕获里带着冠军放大器)
#     ② 引擎捕获 a 份 8192 行 ×2 遍逐位复现(解码路 --score-ids + --cap-dir, 铁律)
#     ③ L0 先过十分钟闸 ④ 43 层 ⑤ dql_to_zchain 导出 ⑥ 引擎路 wt2 五指标: 裸 / +新链 / +冠军链(同模型对表)
#     ⑦ sweep(champbf full = 冠军 sweep 入口, 参考前向侧车路) → 再导出 → 再判
# 用法: q2k_amp_spark.sh   (零参数; 配方常量在下面, 改了记 fable5)
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
SC="$ROOT/gguf-tools/scripts"; D2="$ROOT/gguf/go-onebit/vqhalf"; R30="$ROOT/gguf/go-onebit/r30"; G7="$ROOT/gguf/go-onebit/g7"
HF="$ROOT/hf/DeepSeek-V4-Flash-Vision-Exp"
SKEL="$ROOT/gguf/go-onebit/skel/q2k_skeleton.gguf"
SRC="$D2/champ86/layers_quant"                 # 纯量化态(反修唯一还原点)
WS=champ86q2kamp; W="$D2/$WS"
MDL="$ROOT/gguf/ds4-champ86q2k_bare.gguf"
AIDS="$D2/vqhalf_a.ids"; AANC="$D2/anchor_a_clean_s8192.bin"; QANC="$D2/anchor_vqhalf_q_s8192.bin"
S=8192; K=64
ZLB="$ROOT/gguf-tools/amp/zlayer"
LOG(){ echo "[q2kamp $(date '+%m-%d %H:%M:%S')] $*"; }
DIE(){ LOG "★$*★"; exit 1; }
for f in "$SKEL" "$AIDS" "$AANC" "$AANC.layout" "$QANC" "$ZLB" "$ROOT/gguf-tools/amp/dql_to_zchain"; do [ -e "$f" ] || DIE "缺 $f"; done
[ "$(ls "$SRC"/dql_vq_L*.bin 2>/dev/null | wc -l)" = 43 ] || DIE "纯量化态层件不齐 $SRC"
BUSY=$(for p in ds4 ds4-bench ds4quant_run zlayer vq_merge_v4 deepseek4-quantize; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || DIE "机器非空 $BUSY"
WD=""
wdog(){ ( while true; do A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "${A:-0}" -lt 6000 ] && { echo "[wdog] avail ${A}MB ★杀★"; pkill -9 -x ds4; pkill -9 -x zlayer; pkill -9 -x ds4quant_run; break; }; sleep 5; done ) & WD=$!; }
wdog_stop(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null; WD=""; }
trap 'wdog_stop' EXIT

# ① 裸 Q2_K 模型
if [ ! -s "$MDL" ]; then
    LOG "① 合并裸 Q2_K 模型(纯量化态层件, 无内嵌 op)"
    M86_LAYERS="$SRC" M86_MDL="$MDL" M86_SKEL="$SKEL" M86_ZCH= M86_RB=/dev/null bash "$SC/merge_base86p.sh" || DIE "合并失败"
fi
LOG "裸模型 $(ls -l "$MDL" | awk '{printf "%.2f GB",$5/1e9}')"

# ② 工作区(dql 真拷贝 + dql_vq 硬链只读; layers_quant 一个字节不动)
if [ ! -d "$W/layers" ]; then
    mkdir -p "$W/layers"
    ( cd "$SRC" && for f in dql_vq_L*.bin; do ln -f "$f" "$W/layers/$f" 2>/dev/null || cp "$f" "$W/layers/"; done && cp dql_L*.bin "$W/layers/" )
    LOG "② 工作区 $W/layers ($(ls "$W/layers" | wc -l) 文件)"
fi

# ③ 引擎真值捕获 ×2
CAP="$W/cap"; CAP2="$W/cap2"
if [ ! -s "$CAP/raw_ffn_in_L0" ]; then
    LOG "③ 引擎真值捕获 ×2(解码路 --score-ids a 份 $S 行, 裸 Q2_K 模型)"
    rm -rf "$CAP" "$CAP2"; mkdir -p "$CAP" "$CAP2"; wdog
    for c in "$CAP" "$CAP2"; do
        timeout --foreground 7200 ./ds4 --cuda -m "$MDL" --mem-budget-mb 110000 --score-ids "$AIDS" \
            --score-out "$W/capscore_$(basename "$c").bin" --cap-dir "$c" > "$W/capture_$(basename "$c").log" 2>&1 </dev/null \
            || { tail -3 "$W/capture_$(basename "$c").log"; DIE "捕获失败"; }
    done
    wdog_stop
    for L in 0 16 32 42; do
        cmp "$CAP/raw_ffn_in_L$L" "$CAP2/raw_ffn_in_L$L" || DIE "L$L raw_ffn_in 不复现"
        [ -e "$CAP/raw_ffn_out_L$L" ] && { cmp "$CAP/raw_ffn_out_L$L" "$CAP2/raw_ffn_out_L$L" || DIE "L$L raw_ffn_out 不复现"; }
    done
    LOG "③ 复现 ✓(L0/16/32/42 逐位一致) cap $(du -sh "$CAP" | cut -f1) 文件: $(ls "$CAP" | sed 's/_L[0-9]*$//' | sort -u | tr '\n' ' ')"
    rm -rf "$CAP2"
else LOG "③ 捕获已在, 跳过"; fi

# ④⑤ zlayer 冠军配方 + XCAP(x/被乘量 = 引擎真值), L0 先过十分钟闸
zl(){ "$ZLB" "$HF" "$W/layers" "$AANC" "$1" "$K" 1 "$CAP" - --ntok "$S" --gate-anchor "$QANC" 2>&1 \
        | grep -aE "XCAP|跨语料|k曲线|纯z|终判|注入|Error|assert|★" | sed 's/^/    /'; }
MAN="$W/layers/zinject_manifest.txt"
DONE=$(grep -c '' "$MAN" 2>/dev/null || echo 0)
if [ "$DONE" -lt 1 ]; then
    LOG "④ L0 十分钟闸(XCAP 口径)"; wdog; zl 0 || DIE "L0 失败"; wdog_stop
    LOG "   L0 账本行数 $(grep -c '' "$MAN" 2>/dev/null || echo 0)(0 = 闸拒/未注入, 看上面读数)"
fi
LOG "⑤ 43 层 XCAP 反修"
wdog
for L in $(seq 0 42); do
    grep -q "^L$L " "$MAN" 2>/dev/null && continue
    [ "$L" -eq 0 ] && [ "$DONE" -ge 1 ] && continue
    zl "$L" || DIE "L$L 失败"
    rm -f "$W"/layers/zcache_L*.npz
    LOG "  L$L ✓ 账本 $(grep -c '' "$MAN" 2>/dev/null || echo 0)/43"
done
wdog_stop

# ⑥ 导出 + 引擎路五指标(同一裸模型三行: 裸 / +新链 / +冠军链)
ZC="$W/zchain_q2kamp.bin"
"$ROOT/gguf-tools/amp/dql_to_zchain" "$W/layers" "$ZC" 43 || DIE "zchain 导出失败"
judge(){ local tag="$1"; shift
    timeout --foreground 3600 ./ds4 --cuda -m "$MDL" --mem-budget-mb 110000 "$@" --score-ids "$G7/wt2.ids" \
        --score-out "$W/wt2_$tag.bin" > "$W/wt2_$tag.log" 2>&1 </dev/null || { tail -2 "$W/wt2_$tag.log"; DIE "判决 $tag 失败"; }
    echo "══ 引擎路 wt2 $tag ══" | tee -a "$W/metrics.txt"
    "$ROOT/gguf-tools/bench/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --student "$W/wt2_$tag.bin" 2>&1 | grep -v "^$" | tail -8 | tee -a "$W/metrics.txt"; }
: > "$W/metrics.txt"; wdog
judge bare
judge q2kamp --zchain "$ZC"
judge champzc --zchain "$D2/champ86amp/zchain.bin"
wdog_stop
LOG "⑥ 反修态收官: $ZC; 指标 $W/metrics.txt"; LOG "Q2KAMP_AMP_DONE"

# ⑦ sweep(冠军 sweep 入口 champbf full: 参考前向侧车路) → 再导出 → 再判
LOG "⑦ sweep(champbf $WS full)"
bash "$SC/amp_campaign.sh" champbf "$WS" full || LOG "★sweep 段失败★"
if "$ROOT/gguf-tools/amp/dql_to_zchain" "$W/layers" "$W/zchain_q2kamp_sweep.bin" 43; then
    wdog; judge q2kamp_sweep --zchain "$W/zchain_q2kamp_sweep.bin"; wdog_stop
fi
LOG "Q2KAMP_ALL_DONE"
