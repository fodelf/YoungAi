#!/bin/bash
# quant_verify.sh — 锚定-累积-快验驱动: 逐层量化配置 → 最终输出判决 (Σmin/KL/PPL 主判据)。
#
# 方法学(2026-07-10): 逐层局部 relL2 不再当判决 —— ①样本少的 joint-LS 是"没找到"不是"层差";
# ②每层独立最优≠43层堆叠最优。判决一律 = 配置传播 43 层后的最终输出质量(vs FP 锚定)。
#
# 用法 (在任一台有 HF 的机器, 通常 M1; DS4_HF 指 HF 目录):
#   ./quant_verify.sh build                      # 只编译
#   ./quant_verify.sh anchor [ids] [ntok]        # 预生成 FP 锚定(一次性, 之后所有验证复用)
#   ./quant_verify.sh selftest [ids] [ntok]      # 全F自检: VERDICT 必须 ratio=1.0000/Σmin≈1
#   ./quant_verify.sh cfg <LCFG> [ids] [ntok]    # 单配置判决 (LCFG=43字符 或 1字符广播;
#                                                #   档位 F=不量化 n=朴素1b 1=joint1b z=1b+低秩 2=1b+Q2 3=1b+Q3)
#   ./quant_verify.sh inject [qchar] [ids] [ntok]# 43×单层注入: 只该层量化(其余F) → 对最终输出的真实伤害排序
#   ./quant_verify.sh spare  [qchar] [ids] [ntok]# 43×单层豁免: 全量化只该层F → 该层升位的边际收益排序
# env: DS4_HF DS4_THREADS DS4_NL DS4_ANCHOR DS4_ABLATE DS4_LOCAL_Q DS4_NFIT
# 默认判决文本 = /tmp/rr_hard.ids (还原率铁律: 硬多样高PPL文本), ntok 默认 64。
set -euo pipefail
cd "$(dirname "$0")/../amp"
ROOT="$(cd ../.. && pwd)"
BIN=./ds4quant_run
NLDEF=${DS4_NL:-43}

build(){ cc -O3 -Wall -Wextra -Wno-unused-parameter -lm -framework Accelerate \
            -I"$ROOT" -o "$BIN" ds4quant_run.c -lpthread; echo "[build] $BIN ✓" >&2; }

sort_by_kl(){ awk '{k=""; for(i=1;i<=NF;i++) if($i~/^kl=/){split($i,a,"=");k=a[2]} if(k!="")print k"\t"$0}' "$1" | sort -rn | cut -f2-; }

MODE="${1:-}"
case "$MODE" in
build) build ;;
anchor) build; DS4_FP_ONLY=1 "$BIN" "${2:-/tmp/rr_hard.ids}" "${3:-64}" ;;
selftest) build; DS4_LCFG=F "$BIN" "${2:-/tmp/rr_hard.ids}" "${3:-64}" ;;
cfg)
    [ -n "${2:-}" ] || { echo "用法: $0 cfg <LCFG> [ids] [ntok]" >&2; exit 1; }
    build; DS4_LCFG="$2" "$BIN" "${3:-/tmp/rr_hard.ids}" "${4:-64}" ;;
inject|spare)
    Q="${2:-1}"; IDS="${3:-/tmp/rr_hard.ids}"; NTOK="${4:-64}"
    build
    OUT=/tmp/quant_verify_${MODE}_q${Q}_$(basename "$IDS" .ids)_S${NTOK}.csv
    : > "$OUT"
    DS4_FP_ONLY=1 "$BIN" "$IDS" "$NTOK"    # 锚定就位(缓存命中则秒过)
    for L in $(seq 0 $((NLDEF-1))); do
        echo "[$MODE L=$L $((L+1))/$NLDEF $(date +%H:%M:%S)]" >&2
        if [ "$MODE" = inject ]; then
            DS4_INJECT=$L DS4_QCHAR="$Q" "$BIN" "$IDS" "$NTOK" 2>/dev/null \
                | grep '^VERDICT' | sed "s/^VERDICT/L=$L/" | tee -a "$OUT"
        else
            DS4_SPARE=$L DS4_QCHAR="$Q" "$BIN" "$IDS" "$NTOK" 2>/dev/null \
                | grep '^VERDICT' | sed "s/^VERDICT/L=$L/" | tee -a "$OUT"
        fi
    done
    echo
    if [ "$MODE" = inject ]; then
        echo "=== 单层注入按 KL 降序 (最终输出伤害最大→最小; 取代旧局部 relL2 排序表) ==="
    else
        echo "=== 单层豁免按 KL 降序 (KL 仍大=该层升位收益小; KL 掉最多=该层最该升位) ==="
    fi
    sort_by_kl "$OUT"
    echo "[saved] $OUT"
    ;;
*)  sed -n '2,20p' "$0"; exit 1 ;;
esac
