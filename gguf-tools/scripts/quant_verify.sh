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
#   ./quant_verify.sh inject [qchar] [ids] [ntok]# ★停用: 单层注入钩(原 DS4_INJECT)已随 env 大扫除删除
#   ./quant_verify.sh spare  [qchar] [ids] [ntok]# ★停用: 单层豁免钩(原 DS4_SPARE)已随 env 大扫除删除
# 脚本间 env 接口(转成 flag 递给二进制): DS4_HF DS4_THREADS DS4_NL DS4_ANCHOR DS4_NFIT
# (DS4_ABLATE/DS4_LOCAL_Q 死名已删)
# 默认判决文本 = /tmp/rr_hard.ids (还原率铁律: 硬多样高PPL文本), ntok 默认 64。
set -euo pipefail
cd "$(dirname "$0")/../amp"
ROOT="$(cd ../.. && pwd)"
BIN=./ds4quant_run
NLDEF=${DS4_NL:-43}
HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"   # 原二进制 DS4_HF 写死回落=Flash-Base, 现显式 --hf 同值
# 调用方 env 旋钮 → flag 条件转接(不设=二进制默认, 与原 env 语义一致)
FW=(--hf "$HF" --nl "$NLDEF")
[ -n "${DS4_THREADS:-}" ] && FW+=(--threads "$DS4_THREADS")
[ -n "${DS4_ANCHOR:-}" ] && FW+=(--anchor "$DS4_ANCHOR")
[ -n "${DS4_NFIT:-}" ] && FW+=(--nfit "$DS4_NFIT")

build(){ cc -O3 -Wall -Wextra -Wno-unused-parameter -lm -framework Accelerate \
            -I"$ROOT" -o "$BIN" ds4quant_run.c -lpthread; echo "[build] $BIN ✓" >&2; }

sort_by_kl(){ awk '{k=""; for(i=1;i<=NF;i++) if($i~/^kl=/){split($i,a,"=");k=a[2]} if(k!="")print k"\t"$0}' "$1" | sort -rn | cut -f2-; }

MODE="${1:-}"
case "$MODE" in
build) build ;;
anchor) build; "$BIN" "${2:-/tmp/rr_hard.ids}" "${3:-64}" "${FW[@]}" --fp-only ;;
selftest) build; "$BIN" "${2:-/tmp/rr_hard.ids}" "${3:-64}" "${FW[@]}" --lcfg F ;;
cfg)
    [ -n "${2:-}" ] || { echo "用法: $0 cfg <LCFG> [ids] [ntok]" >&2; exit 1; }
    build; "$BIN" "${3:-/tmp/rr_hard.ids}" "${4:-64}" "${FW[@]}" --lcfg "$2" ;;
inject|spare)
    # ★停用(2026-08-31 env 大扫除)★: 单层注入/豁免钩(原 DS4_INJECT/DS4_SPARE/DS4_QCHAR)
    # 已从 C 拔死(零读者), 无 flag 承接 — 响亮失败, 不静默跑成整模判决冒充单层排序。
    echo "[$MODE] ★停用: DS4_INJECT/DS4_SPARE 机制已随 env 大扫除删除, 无 flag 承接★" >&2
    exit 2
    ;;
*)  sed -n '2,20p' "$0"; exit 1 ;;
esac
