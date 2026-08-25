#!/bin/bash
# r60_deploy_route_replay.sh — 部署态路由口径验证遍 (2026-08-06 0.46 缺口定案后)。
# 与 campaign 评分段唯一差异: 去掉 DS4_ANCHOR_ROUTE=1 ⇒ 回放走学生自路由
# (dq_gate_route_topk + RB α2.5 偏置 = 引擎部署态同口径)。
# 判决: student_logits_deploy.bin 前 32 KL vs 教师 ≈ 引擎链版 1.082 ⇒ 口径对齐实锤,
#       反修段随后切部署态重拟合 op(教师口径判决文件 student_logits.bin 不动)。
# 跑在 M1 (锚/layers 所在机); 前置: svc worker 已停 (回放峰值 ~10G, 12G 红线)。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
R30="$ROOT/gguf/go-onebit/r30"; OUTF="$R30/full"
ANCHOR="$R30/anchor_r30_s1716.bin"
IDS="$ROOT/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids"
QBIN="$ROOT/gguf-tools/go-onebit/quant/ds4quant_run"
LOG(){ echo "[deploy-replay $(date +%H:%M:%S)] $*" >&2; }

pgrep -f "ds4 -m .*--role worker" >/dev/null && { LOG "worker 还在跑 — 先停 svc"; exit 3; }
[ -f "$ANCHOR" ] || { LOG "锚缺"; exit 2; }
N=$(ls "$OUTF"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
[ "$N" = 43 ] || { LOG "层文件 $N/43 不齐"; exit 2; }
cd "$ROOT/gguf-tools/go-onebit/quant"
LOG "部署态回放遍(学生自路由 + RB α2.5, dump student_logits_deploy.bin)"
env -u DS4_TUNE -u DS4_MINVOL -u DS4_MV_BASELINE -u DS4_VQ_RPLAN -u DS4_BWD \
    -u DS4_ANCHOR_ROUTE \
    DS4_HF="${DS4_HF:-$HOME/ds4-main/hf/DeepSeek-V4-Flash-0731}" \
    DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 \
    DS4_CALIB_FULLSET=1 \
    DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-8}" \
    DS4_LAYER_DIR="$OUTF/layers" DS4_LCFG=$(printf 'g%.0s' $(seq 1 43)) \
    DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_COADAPT=1 \
    DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top72.txt" \
    DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin" \
    DS4_ROUTE_BIAS="$OUTF/route_bias_r30.bin" DS4_ROUTE_BIAS_ALPHA="${RB_ALPHA:-2.5}" DS4_ROUTE_BIAS_MINCNT=8 \
    DS4_DUMP_LOGITS="$OUTF/student_logits_deploy.bin" \
    "$QBIN" "$IDS" 1716 || { LOG "★部署态回放失败★"; exit 3; }
[ -f "$OUTF/student_logits_deploy.bin" ] || { LOG "★deploy logits 没落盘★"; exit 3; }
LOG "完成 → $OUTF/student_logits_deploy.bin"
