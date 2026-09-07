#!/bin/bash
# skel_from_hf.sh — 从 HF 原始 Base 抽【自产骨架】(2026-08-01 用户令"骨架使用原始模型作为
#   骨架源, 不要再用冠军的骨架")。
#
# 为什么必须自产: r28_skeleton.gguf 抄自冠军 v4bf, 它的 exp_probs_b 里烘着 2.5·Δb_champ
# (skel_bias_probe 实测 k=+2.27 r=+0.94)。用它合并 ⇒ 新模型背着别人的路由药方, 得先"减
# 冠军 Δb"才能回到干净基准, 多一步、多一个出错点; fable5 另有实测迁移他人 Δb 是净负。
#
# 做法(不必新写工具, 现成的就够):
#   deepseek4-quantize --experts-hole  = 骨架模式: routed 专家"留洞"(不算不写),
#                                        非专家张量全部从 HF 原始重新量化
#   vq_merge_v4.c --extract-skeleton  = 把留洞文件压成紧凑骨架(丢掉洞和 down)
#
# --template 只提供 GGUF 的 KV 元数据(tokenizer/超参)和张量顺序, **不提供任何权重**
# —— 权重 100% 来自 --hf 指的原始 Base。这与"禁用 q2 当模版/backbone/源"的铁律不冲突:
# 那条禁的是拿量化模型当【权重来源】。模版取自 published GGUF 头部(57.8 MiB, 足够解析
# 出 n_tensors=1328 / n_kv=62)。
#
# 用法: skel_from_hf.sh [out_skeleton.gguf] [threads]
set -uo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_portable.sh"   # 盘闸/看门狗/stat 的跨平台形态
ROOT="$HOME/ds4-main"
# ★R30(2026-08-02)★: 源模型换代为 0731(用户裁决), 老 Base 已挪 hf-base(铁律不删)。
# SKEL_HF/SKEL_TMPL 可 env 覆盖 —— r30_campaign.sh 显式传, 裸跑用默认。
HF="${SKEL_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
TMPL="${SKEL_TMPL:-$ROOT/gguf/go-onebit/r30/template_head.gguf}"
Q="$ROOT/gguf-tools/deepseek4-quantize"
OUT="${1:-$ROOT/gguf/go-onebit/r29/r29_skeleton.gguf}"
TH="${2:-3}"
HOLE="$(dirname "$OUT")/skel_hole.gguf"   # 中间留洞文件跟随输出目录(2026-08-03: r29 硬编码修)
LOG(){ echo "[skel $(date +%H:%M:%S)] $*" >&2; }

[ -d "$HF" ]   || { LOG "HF 原始缺: $HF"; exit 2; }
[ -f "$TMPL" ] || { LOG "模版缺: $TMPL"; exit 2; }
[ -x "$Q" ]    || { LOG "量化器缺: $Q"; exit 2; }

# ★盘闸★ 留洞文件是稀疏的(APFS 支持), 但保守按骨架 ~9 GiB + 紧凑骨架 ~9 GiB 算
# SKEL_FREE_GB / SKEL_WDOG_MB / SKEL_WDOG_FREE_GB: 阈值全部可 env 覆盖。默认值是 16G Mac
# 的档位(内存 6G / 留 10G 盘), 直接搬到 121G 内存的机器上会**误杀** —— 多线程量化正常
# 就能过 6G。跨机跑前按机器改, 别让为小机器定的护栏变成大机器上的绊脚石。
FREE=$(disk_free_gb "$(dirname "$OUT")")
[ "${FREE:-0}" -ge "${SKEL_FREE_GB:-20}" ] || { LOG "★盘闸 free ${FREE}G < ${SKEL_FREE_GB:-20}G 停★"; exit 6; }

# ★内存闸 + 运行中盘闸★
# 盘闸的必要性(dry-run 实测): 留洞文件表观 86.7 GB, 只有 APFS 稀疏真生效时才实占 ~9 GiB。
# 若稀疏没生效(或量化产物已吃掉盘), 它会一路写满磁盘, 连带把还在跑的量化/反修拖死。
# 所以不能只在起跑前查一次 free —— 必须运行中持续盯, 低于 10 GiB 立刻杀。
( while true; do
    P=$(pgrep -nf "deepseek4-quantize .*experts-hole" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(proc_mem_mb "$P" || true)
    [ -n "${MB:-}" ] && [ "$MB" -gt "${SKEL_WDOG_MB:-6000}" ] && {
        echo "[skel][wdog] ${MB}MB >${SKEL_WDOG_MB:-6000}MB 杀" >&2; kill -9 "$P"; }
    FG=$(disk_free_gb "$(dirname "$OUT")")
    [ -n "${FG:-}" ] && [ "$FG" -lt "${SKEL_WDOG_FREE_GB:-10}" ] && {
        echo "[skel][wdog] free ${FG}G <10G — 稀疏可能没生效, 杀骨架保住量化" >&2
        kill -9 "$P"; rm -f "$HOLE" 2>/dev/null; }
    sleep 5; done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT

LOG "阶段1: HF 原始 → 留洞骨架(专家不算不写, ${TH} 线程; 额外量化旗标: ${*:3})"
# $3 起透传给量化器 = 骨架精度档(2026-09-05 骨架降字节战役): 不给 = 量化器默认(Q8 骨架, r30 同款);
# 给 `--attention-proj q2_k --attention q2_k --shared q2_k --output q2_k --dense q2_k --embedding q2_k`
# = allq2 同款全 Q2_K 骨架。精度档只在这一处决定, 合并/判决脚本不再各自猜。
"$Q" --hf "$HF" --template "$TMPL" --out "$HOLE" --experts-hole --threads "$TH" --overwrite "${@:3}" \
    || { LOG "★留洞骨架失败★"; exit 3; }
APP=$(file_apparent_bytes "$HOLE"); REAL=$(file_real_bytes "$HOLE")
LOG "留洞文件 表观 $(awk -v v=$APP 'BEGIN{printf "%.1f GiB",v/2^30}') / 实占 $(awk -v v=$REAL 'BEGIN{printf "%.1f GiB",v/2^30}')"
awk -v a=$APP -v r=$REAL 'BEGIN{exit !(r > a*0.5)}' && LOG "★稀疏未生效(实占>表观一半) — 后续轮次应改为流式抽取★"

LOG "阶段2: 压成紧凑骨架"
"$(dirname "$0")/../quantize/vq_merge_v4" --extract-skeleton \
    --base "$HOLE" --out "$OUT" || { LOG "★抽骨架失败★"; exit 4; }
rm -f "$HOLE"
LOG "自产骨架 $OUT $(ls -l "$OUT" | awk '{printf "%.2f GiB", $5/1073741824}')"

# ★干净度自检★: 自产骨架对任何他人 Δb 的相关应 ≈0
LOG "干净度自检(对冠军 Δb 的相关应 ≈0):"
bash "$ROOT/gguf-tools/scripts/preflight_skeleton.sh" "$OUT" 0.30 || \
    LOG "★自检未过 — 骨架仍带他人路由偏置, 需人工看★"
