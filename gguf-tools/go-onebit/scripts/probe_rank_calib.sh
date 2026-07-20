#!/bin/bash
# probe_rank_calib.sh — 小探针排序标定(2026-07-16 流程修正①落地):
# 用三套已有满档真值锚(26%/62%/100% 代码占比 → rr_code 0.2653/0.2972/0.5104)各跑
# NL=6 fast 量化 + NL=6 固定裁判回放, 验证分钟级小探针的【排序】是否复现满档排序。
# 一致 → 未来一切构成/配方扫描先过小探针; 不一致 → 小探针不可作筛选器(也是判决)。
# 跑在 M1。产出: /tmp/probe_rank.report (三行 VERDICT 对照)
set -uo pipefail
cd "$(dirname "$0")/../../.."
REPORT=/tmp/probe_rank.report
: > "$REPORT"
NL=6

for pair in "code62:/tmp/rr_calib_s512b.ids" "code26:/tmp/rr_calib_s512.ids"; do
    tag="${pair%%:*}"; ids="${pair##*:}"
    [ -f "$ids" ] || { echo "[$tag] 锚缺失 $ids, 跳过" | tee -a "$REPORT"; continue; }
    echo "[calib] $tag: NL=$NL fast 量化..." >&2
    DS4_CORPUS="$ids" DS4_FAST_LAYERS=$NL DS4_FAST_NTOK=64 DS4_SKIP_RRVERDICT=1 \
        ./quant_layer.sh fast > "/tmp/prc_${tag}.out" 2>&1
    # fast 会 exec backfit 接段(诚实反修) — 标定只要推进段 dql, backfit 段任其失败/短路均可;
    # 保险: 杀干净再判
    pkill -9 -f ds4quant_run 2>/dev/null; pkill -f "quant_layer" 2>/dev/null; sleep 2
    N_HAVE=$(ls gguf/go-onebit/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    echo "[calib] $tag: dql=$N_HAVE, NL=$NL 固定裁判回放..." >&2
    DS4_NL=$NL bash gguf-tools/go-onebit/scripts/rr_verdict.sh /tmp/rr_code.ids 305 \
        > "/tmp/prc_${tag}_verdict.out" 2>&1   # 全输出落盘(吞stderr教训)
    V=$(grep -aE '^VERDICT' "/tmp/prc_${tag}_verdict.out" | tail -1)
    echo "[$tag] ${V:-无VERDICT, 见 /tmp/prc_${tag}_verdict.out}" | tee -a "$REPORT"
done
echo "[calib] 真值参照: code26=0.2653 code62=0.2972 code100=0.5104 (满档 rr_code smin)" | tee -a "$REPORT"
echo "[calib] 完成 → $REPORT" >&2
