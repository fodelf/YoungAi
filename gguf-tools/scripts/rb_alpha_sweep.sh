#!/bin/bash
# rb_alpha_sweep.sh — 路由偏置 α 网格扫(2026-08-01 用户令"找到那个对的路由反修")。
#
# 方法与冠军当年定 α=2.5 同源(fable5 2026-07-28: code α网格 0.5/1.0→77.6 平, 1.5→80.3,
# 2.0→81.6, 2.5→84.2 峰, 3.0→81.6 回落)—— 用【只读回放判决器】扫, 不靠反修试错。
#   DS4_BF_ONLY=1 + DS4_BF_TERM_MAXP=0 : 跳过 SEARCH、不进 sweep、不改写任何层文件
#   DS4_ROUTE_BIAS=<Δb> + _ALPHA=<α>   : 每层 gate 打分 = gbias + α·Δb(= 部署态路由)
# 判据: VERDICT 行的 agree/smin/kl/ratio + 逐层"路由一致%"。
#
# 背景: v7 直接套冠军 α=2.5 后 L03 路由一致 89.7%→78.6%(反而更差), 说明 α 不可跨模型
# 移植 —— 两份 Δb 的语料/档族/量级都不同, 必须自己扫。
#
# 用法: rb_alpha_sweep.sh "<α列表>" [语料ids] [ntok]
#   例: rb_alpha_sweep.sh "0 0.5 1.0 1.5 2.5" /path/rr_hard.ids 64
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
G7="$ROOT/gguf/go-onebit/g7"
# ★2026-08-19 参数化(cal12 路由反修): RBS_* env 覆盖, 默认=原 r28v2 语义; RBS_PLAIN=1 关 GO2B/ZCHAIN
OUTF="${RBS_OUTF:-$ROOT/gguf/go-onebit/r28v2/full}"
RB="${RBS_RB:-$OUTF/route_bias_r28.bin}"
RBS_ANCHOR="${RBS_ANCHOR:-$G7/ds4quant_anchor_v5mini_s1716.bin}"
RBS_LAYERS="${RBS_LAYERS:-$OUTF/layers}"
ALPHAS="${1:-0 0.5 1.0 1.5 2.5}"
IDS="${2:-$G7/rr_hard.ids}"
NTOK="${3:-64}"
OUT="$OUTF/alpha_sweep"
mkdir -p "$OUT"
LOG(){ echo "[αscan $(date +%H:%M:%S)] $*" >&2; }

[ -f "$RB" ] || { LOG "Δb 文件缺: $RB"; exit 2; }
[ -f "$IDS" ] || { LOG "语料缺: $IDS"; exit 2; }
N=$(ls "$RBS_LAYERS"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
[ "$N" = 43 ] || { LOG "层文件 $N/43 不齐"; exit 2; }
pgrep -x ds4quant_run >/dev/null && { LOG "已有量化进程, 拒并发"; exit 3; }

cd "$ROOT/gguf-tools/amp"
printf 'α\tagree\tΣmin\tKL\tratio\t路由一致均值\n' > "$OUT/summary.tsv"
for A in $ALPHAS; do
    LOG "α=$A 回放中…"
    EXTRAE=()
    [ -z "${RBS_PLAIN:-}" ] && EXTRAE=(DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/data/corpus/prog_active_top64.txt" DS4_ZCHAIN="$OUTF/zchain.bin")
    env -u DS4_TUNE -u DS4_MINVOL -u DS4_MV_BASELINE -u DS4_VQ_RPLAN -u DS4_ANCHOR_ROUTE \
        -u DS4_ROUTE_SEQ -u DS4_ROUTE_BIAS_FIT -u DS4_BWD_FINAL -u DS4_GSWEEP \
        DS4_ANCHOR="$RBS_ANCHOR" \
        DS4_LAYER_DIR="$RBS_LAYERS" DS4_LCFG=$(printf 'g%.0s' $(seq 1 43)) \
        DS4_VQ=1 DS4_COADAPT=1 DS4_THREADS="${DS4_THREADS:-6}" \
        ${EXTRAE[@]+"${EXTRAE[@]}"} \
        DS4_BF_ONLY=1 DS4_BF_TERM_MAXP=0 \
        DS4_ROUTE_BIAS="$RB" DS4_ROUTE_BIAS_ALPHA="$A" DS4_ROUTE_BIAS_MINCNT=8 \
        ./ds4quant_run "$IDS" "$NTOK" > "$OUT/a$A.out" 2>&1
    V=$(grep -a "VERDICT" "$OUT/a$A.out" | tail -1)
    AG=$(echo "$V" | grep -oE 'agree=[0-9.]+' | cut -d= -f2)
    SM=$(echo "$V" | grep -oE 'smin=[0-9.]+' | cut -d= -f2)
    KL=$(echo "$V" | grep -oE 'kl=[0-9.]+' | cut -d= -f2)
    RT=$(echo "$V" | grep -oE 'ratio=[0-9.]+' | cut -d= -f2)
    RC=$(grep -a "路由一致" "$OUT/a$A.out" | grep -oE '路由一致=[ ]*[0-9.]+' | grep -oE '[0-9.]+' \
         | awk '{s+=$1;n++} END{if(n)printf "%.1f",s/n; else print "-"}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$A" "${AG:--}" "${SM:--}" "${KL:--}" "${RT:--}" "${RC:--}" >> "$OUT/summary.tsv"
    LOG "α=$A → agree=${AG:--} Σmin=${SM:--} KL=${KL:--} 路由一致均值=${RC:--}%"
done
LOG "扫完 — 汇总:"
column -t "$OUT/summary.tsv" >&2
echo >&2
LOG "冠军对照(rr判决榜 code S=305): agree 82.9 | Σmin 0.7680 | KL 0.4506 | ratio 1.4230"
