#!/bin/bash
# q2z_pipeline.sh — zside 收官后的串行流水线: 质量验证(metrics 回放) → 合并 → v4 量化。
# 教训(2026-08-18): tmux 内嵌多层转义命令串秒退, 一切编排必须落脚本。
set -uo pipefail
SC="$HOME/ds4-main/gguf-tools/go-onebit/scripts"
LAYERS="$HOME/ds4-main/gguf/go-onebit/r30/full/layers"
LOG(){ echo "[pipeline $(date +%H:%M:%S)] $*"; }

while [ "$(wc -l < "$LAYERS/zinject_manifest.txt" 2>/dev/null || echo 0)" -lt 43 ]; do sleep 300; done
LOG "① metrics 回放验证"
bash "$SC/q2z_spark.sh" metrics > /tmp/q2z_metrics.log 2>&1
LOG "metrics rc=$? (指标在 /tmp/q2z_metrics.log)"
tail -5 /tmp/q2z_metrics.log

LOG "② 合并"
bash "$SC/q2z_spark.sh" merge > /tmp/q2z_merge_stage.log 2>&1 || { LOG "★合并失败★"; tail -5 /tmp/q2z_merge_stage.log; }

LOG "③ v4 量化(drafter q4 判别)"
EXTRA=""
for L in 0 1 2; do for W in gate up down; do EXTRA="$EXTRA --tensor-type mtp.$L.ffn_${W}_exps.weight=q4_k"; done; done
cd "$HOME/ds4-main"
AO=q4_k OUT="$HOME/ds4-main/gguf/ds4-final86v4.gguf" \
    bash gguf-tools/go-onebit/scripts/quant_final86_spark.sh --mtp-append 3 $EXTRA > /tmp/quant_v4.log 2>&1
LOG "流水线收官 v4 rc=$?"
