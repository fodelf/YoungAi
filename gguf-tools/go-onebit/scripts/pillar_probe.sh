#!/bin/bash
# pillar_probe.sh — 四支柱快判集探针: v2+热残差, BOS 裸续写 temp0, 原始输出逐条落盘。
# 复用 code1b_smoke.sh(自带内存看门狗+超时); 每条探针一次进程(冷加载~2min/条)。
# 用法: [PROBES=first|all] [NPRED=28] [MODEL=...] [RESID=...] ./pillar_probe.sh
#   PROBES=first(默认) = 每支柱第1条(共4条, 分钟级档); all = 全12条(~25min)。
# 输出: /tmp/pillar_probe.report(判读留给人; 我的判读只作参考——每次给看原始输出铁律)
set -euo pipefail
cd "$(dirname "$0")/.."     # → go-onebit/
MODEL="${MODEL:-gguf/go-onebit/ds4-code1b-v2.gguf}"
# RESID="" 显式=无侧车(裸底座); 未设才回 v2 默认(- 与 :- 语义之差, 07-15 12针空跑教训)
RESID="${RESID-gguf/sidecars/code-hot-res-v2.gguf}"
if [ -n "$RESID" ]; then export DS4_RESIDUAL="$RESID"; else unset DS4_RESIDUAL; fi
NPRED="${NPRED:-28}"
SEL="${PROBES:-first}"
REPORT=/tmp/pillar_probe.report
: > "$REPORT"

# 解析 pillar_probes.txt: --- 分块, 去 # 注释行; first 档取块 1,4,7,10(每支柱第1条)
mapfile_blocks() {
    awk 'BEGIN{RS="---\n"; n=0}
         { blk=""; nl = split($0, L, "\n");
           for (i = 1; i <= nl; i++) if (L[i] !~ /^#/) blk = blk (blk==""?"":"\n") L[i];
           gsub(/\n+$/, "", blk); sub(/^\n+/, "", blk);
           if (blk != "") { n++; printf "%s\x1e", blk } }' corpus/pillar_probes.txt
}
IFS=$'\x1e' read -r -a BLOCKS -d '' < <(mapfile_blocks; printf '\0') || true
PICK=(0 3 6 9); [ "$SEL" = all ] && PICK=($(seq 0 $((${#BLOCKS[@]}-1))))

echo "[probe] ${#PICK[@]} 条 (model=$MODEL resid=$RESID npred=$NPRED)" >&2
i=0
for k in "${PICK[@]}"; do
    i=$((i+1))
    frag="${BLOCKS[$k]}"
    echo "[probe $i/${#PICK[@]}] 块$((k+1)): ${frag%%$'\n'*}" >&2
    {   echo "════ PROBE $((k+1)) ════"
        echo "── 片段:"; printf '%s\n' "$frag"
        echo "── 续写(原始):"
    } >> "$REPORT"
    PROMPT="<｜begin▁of▁sentence｜>${frag}" MODEL="$MODEL" NPRED="$NPRED" TIMEOUT_S=240 \
      ./scripts/code1b_smoke.sh >> "$REPORT" 2>/tmp/pillar_probe.err \
      || echo "[探针失败, 见 /tmp/pillar_probe.err 尾部]" >> "$REPORT"
    echo >> "$REPORT"
done
echo "[probe] 完成 → $REPORT" >&2
