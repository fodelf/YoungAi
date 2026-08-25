#!/bin/bash
# zside_base86p.sh — 平权基座反修, 4-lane 并行。
# ★2026-08-19 用户令: 反修只留 z变量+四损失+感知(均在纯z闭式解内); GE/ERF/ftA 默认关。
# ★2026-08-20 单遍双解+择优(用户"为什么不是反修脚本一次跑完"): ZS_AMP=1 时每层
#   zcache 建一次 → 加性 zlayer 与乘性 amp_solve 并行同解(zcache 原子换名后 amp 即开跑)
#   → held 高者定案写 zrec_L(amp 胜=zl.AMP 记录; 双闸=116B 终态标记)。
# ★口径: 默认 FP 锚(自洽不动点, 在线正迁移实证 XZC +3.6%); DS4_ZL_XANCHOR 由上层显式给
#   才走链态锚(2026-08-20 判决: 链态口径自我拆台, 链上复利为负, 已弃)。
# 用法: bash zside_base86p.sh [lane数=4]
set -uo pipefail
ROOT="$HOME/ds4-main"
[ -x "$ROOT/gguf-tools/amp/zlayer" ] || make -C "$ROOT/gguf-tools" zlayer   # C 反修解算器(zlayer.py 已删)
LAYERS="${ZS_LAYERS:-$ROOT/gguf/go-onebit/r30/base86p/layers}"
ANCHOR="${ZS_ANCHOR:-$ROOT/gguf/go-onebit/r30/anchor_cal9_s2906.bin}"
ZS_NTOK="${ZS_NTOK:-2906}"
ZS_FIT="${ZS_FIT:-0:975,1141:1590,1668:1899,1971:2429,2577:2836}"
ZS_EV="${ZS_EV:-975:1141,1590:1668,1899:1971,2429:2577,2836:2906}"
NLANE="${1:-4}"
export DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
LOG(){ echo "[zside86p $(date +%H:%M:%S)] $*"; }

ZS_INJ="${ZS_INJ:-1}"   # 2=zrec 外挂模式(dql 不改, 断点续跑由 zlayer 自理)
ZS_AMP="${ZS_AMP:-0}"   # 1=单遍双解+择优(加性 zlayer ∥ 乘性 amp_solve)
lane(){  # $@=层列表
    local L NN ZR ZCF AMPREC AMPLOG AMPPID HA HM W
    for L in "$@"; do
        NN=$(printf %02d $L)
        ZR="$LAYERS/zrec_L$NN.bin"
        # 已注入跳过(dql 里已有 zl.RRR 记录的层; INJ=2 时 zlayer 自查 zrec)
        [ "$ZS_INJ" = 2 ] || python3 - "$LAYERS" "$L" <<'PY' && [ "$ZS_INJ" != 2 ] && { echo "L$L 已注入, 跳过"; continue; }
import struct,sys,os
p=os.path.join(sys.argv[1],f"dql_L{int(sys.argv[2]):02d}.bin")
raw=open(p,"rb").read(12); nr,=struct.unpack_from("<I",raw,8)
raw=open(p,"rb").read()
off=12
for _ in range(nr):
    nm=raw[off:off+16].split(b"\0")[0].decode("ascii","replace")
    psz,=struct.unpack_from("<Q",raw,off+88); off+=116+psz
    if "zl.RRR" in nm: sys.exit(0)
sys.exit(1)
PY
        ZCF="$LAYERS/zcache_L$NN.npz"
        rm -f "$ZCF"
        AMPPID=""
        if [ "$ZS_AMP" = 1 ] && [ ! -s "$ZR" ]; then
            rm -f "$ZR"   # 空 zrec=中断残留, 重解; 非空(含 116B 标记)=终态由 zlayer 自跳
            AMPREC="$LAYERS/zrec_amp_L$NN.bin"; AMPLOG="$LAYERS/amp_L$NN.out"
            rm -f "$AMPREC" "$AMPLOG"
            ( for _i in $(seq 1 600); do [ -f "$ZCF" ] && break; sleep 2; done
              [ -f "$ZCF" ] || { echo "★L$L amp 等 zcache 超时★"; exit 9; }
              exec "$(dirname "$0")/../legacy/amp_solve_zc" \
                  "$ANCHOR" "$ZCF" "$AMPREC" "${ZS_AMPNFIT:-1638}" ) > "$AMPLOG" 2>&1 &
            AMPPID=$!
        fi
        env DS4_ZL_NTOK="$ZS_NTOK" \
            DS4_ZL_FIT_RANGES="$ZS_FIT" \
            DS4_ZL_EV_RANGE="$ZS_EV" \
            DS4_ZL_GE="${DS4_ZL_GE:-0}" DS4_ZL_FTA="${DS4_ZL_FTA:-0}" DS4_ZL_ERF="${DS4_ZL_ERF:-0}" \
            DS4_ZL_ERF_R=16 DS4_ZL_ERF_BAR=0.01 DS4_ZL_SWLIM=60 DS4_ZL_GE_LAM=1e-3 \
            "$ROOT/gguf-tools/amp/zlayer" \
            "$DS4_HF" "$LAYERS" "$ANCHOR" $L 1024 "$ZS_INJ" 2>&1 | grep -E "★|失败" \
            | tee "$LAYERS/zl_L$NN.out" \
            || { LOG "★L$L 失败★"; echo "$L" >> "$FAILF"; }
        if [ -n "$AMPPID" ]; then
            wait "$AMPPID" || { LOG "★L$L amp 失败★"; echo "$L" >> "$FAILF"; }
            grep -a "★" "$AMPLOG" 2>/dev/null || true
            HA=$(sed -n 's/.*组合 \([0-9.]*\)%.*/\1/p' "$LAYERS/zl_L$NN.out" 2>/dev/null | head -1); : "${HA:=0}"
            HM=$(sed -n 's/.*held行为挽回 \([0-9.]*\)%.*/\1/p' "$AMPLOG" 2>/dev/null | head -1); : "${HM:=0}"
            # 择优单选(同残差双重修正禁): amp 胜且有实载荷(>116B)才换; 双闸=amp 116B 标记定格
            if awk -v a="$HA" -v m="$HM" 'BEGIN{exit !(m>a)}' \
               && [ -f "$AMPREC" ] && [ "$(stat -c %s "$AMPREC" 2>/dev/null || echo 0)" -gt 116 ]; then
                mv -f "$AMPREC" "$ZR"; W=amp
            elif [ ! -s "$ZR" ] && [ -f "$AMPREC" ]; then
                mv -f "$AMPREC" "$ZR"; W=none   # 双闸: 116B 终态标记防重解
            else
                rm -f "$AMPREC"; W=add
            fi
            LOG "L$L 择优: add ${HA}% vs amp ${HM}% → $W"
            rm -f "$AMPLOG" "$LAYERS/zl_L$NN.out"
        fi
        rm -f "$ZCF"
    done
}

FAILF="$(mktemp /tmp/zside_fail.XXXXXX)"   # lane 失败聚合: 假"收官"兜底禁(2026-08-20 事故)
PIDS=()
for i in $(seq 0 $((NLANE-1))); do
    LS=$(seq $i $NLANE 42)
    lane $LS &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
if [ -s "$FAILF" ]; then
    LOG "★反修失败层: $(sort -n "$FAILF" | tr '\n' ' ')★"
    rm -f "$FAILF"; exit 1
fi
rm -f "$FAILF"
LOG "反修 43 层收官"
