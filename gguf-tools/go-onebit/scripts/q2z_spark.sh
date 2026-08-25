#!/bin/bash
# q2z_spark.sh — q2z 战役的 Spark(GB10/aarch64) 驱动: M1 离线, 锚/层件在 spark 重造。
# 与 q2z_campaign.sh 同参数同语义, 差异只有: ①QBIN=spark 编译的 ds4quant_run(openblas)
# ②阶段可从 anchor 起。用法: bash q2z_spark.sh [anchor|quant|zside|all]
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
OUTF="$R30/full"
LAYERS="$OUTF/layers"
QBIN="$ROOT/gguf-tools/go-onebit/quant/ds4quant_run"
ANCHOR="$R30/anchor_r30_s1716.bin"
IDS="$ROOT/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids"
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
LOG(){ echo "[q2z-spark $(date +%H:%M:%S)] $*" >&2; }
NL=43

stage_anchor(){
    [ -f "$ANCHOR" ] && { LOG "锚已在"; return 0; }
    cd "$ROOT/gguf-tools/go-onebit/quant"
    DS4_FP_ONLY=1 DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-20}" \
        "$QBIN" "$IDS" 1716 || { LOG "★锚失败 rc=$?★"; exit 3; }
    [ -f "$ANCHOR" ] || { LOG "★锚没落盘★"; exit 3; }
    LOG "锚 ✓ $(ls -l "$ANCHOR" | awk '{printf "%.2f GiB",$5/1073741824}')"
}

quant_env(){  # q2z_campaign.sh 原样配方(平权 VQ + 锚路由 + FULLSET)
    # 外层专家循环已 pthread 并行; scipy-openblas 内层再开线程=20×20 锁争用(实测 CPU
    # 只吃 11.5/20 核) → BLAS 钉单线程
    export OPENBLAS_NUM_THREADS=1
    export DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-20}" DS4_CALIB_FULLSET=1
    export DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_TUNE=1 DS4_PURE_VQ=1
    export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$OUTF/rplan.txt" DS4_VOL_BUDGET_GIB=72
    export DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top49.txt"
    export DS4_ANCHOR_ROUTE=1 DS4_BF_GAIN_GATE=0.05
    export DS4_PLAN="$OUTF/plan.txt" DS4_CKPT_DIR="$OUTF/ckpt" DS4_LAYER_DIR="$LAYERS"
    export DS4_ZFILE=/tmp/zfile_junk.bin DS4_ZCHAIN=/tmp/zchain_junk.bin
    export DS4_BF_MEMGB="${DS4_BF_MEMGB:-60}"   # spark 121G: 放宽 fp16 缓存驱逐阈
    unset DS4_COADAPT DS4_MV_COAD_BASE DS4_BF_ONLY DS4_GSWEEP DS4_FP_ONLY 2>/dev/null || true
}

stage_quant(){
    mkdir -p "$LAYERS" "$OUTF/ckpt"
    quant_env
    # rplan: q2z 平权计划(原战役产物); 缺则从 q2_plan.json 生成或停
    [ -f "$OUTF/rplan.txt" ] || { LOG "★rplan 缺: $OUTF/rplan.txt — 看 r30/plan 或重生成★"; exit 2; }
    cd "$ROOT/gguf-tools/go-onebit/quant"
    "$QBIN" "$IDS" 1716 || { LOG "★量化批量失败 rc=$?★"; exit 2; }
    LOG "量化批量完: $(ls "$LAYERS"/dql_vq_L*.bin 2>/dev/null | wc -l | tr -d ' ')/43 层"
}

stage_zside(){
    cd "$ROOT"
    for L in $(seq 0 $((NL-1))); do
        F="$LAYERS/$(printf 'dql_vq_L%02d.bin' $L)"
        [ -f "$F" ] || { LOG "★L$L 未量化, 停★"; exit 2; }
        python3 -u "$ROOT/gguf-tools/go-onebit/zlever/zlayer.py" "$DS4_HF" "$LAYERS" "$ANCHOR" "$L" 1024 1 \
            2>&1 | tee -a /tmp/q2z_zside.log | grep '★' || true
    done
    LOG "z 侧车段收官"
}

stage_merge(){   # 合并 GGUF(q2z_campaign stage_merge 的 spark 适配): skeleton+blob --no-down + zchain 外挂
    MDL="$ROOT/gguf/go-onebit/ds4-q2z.gguf"
    MAN="$LAYERS/manifest.txt"
    [ -f "$MAN" ] || { LOG "★manifest 缺★"; exit 4; }
    python3 "$ROOT/gguf-tools/go-onebit/zlever/dql_to_zchain.py" "$LAYERS" "$OUTF/zchain_q2z.bin" 43 \
        || { LOG "★zchain 抽取失败★"; exit 5; }
    FREE=$(df -BG --output=avail "$ROOT/gguf" | sed -n 2p | tr -dc 0-9)
    [ "${FREE:-0}" -ge 90 ] || { LOG "★盘不足 90G 停★"; exit 7; }
    LOG "起合并(全VQ形态, 非消费式)"
    python3 "$ROOT/gguf-tools/go-onebit/quant/vq_merge_v4.py" --merge \
        --skeleton "$R30/r30_skeleton.gguf" \
        --blob-sizes "$MAN" --no-down \
        --dql-host 127.0.0.1 --dql-dir "$LAYERS" \
        --out "$MDL" > /tmp/q2z_merge.log 2>&1 \
        || { LOG "★合并失败★"; tail -5 /tmp/q2z_merge.log >&2; exit 6; }
    LOG "合并完: $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}')  zchain=$OUTF/zchain_q2z.bin"
}

stage_metrics(){  # 五指标回放(量化+z侧车 vs FP 锚, held=位置1287..1716) — q2z_campaign 同口径
    cd "$ROOT/gguf-tools/go-onebit/quant"
    env -u DS4_TUNE -u DS4_MINVOL -u DS4_VQ_RPLAN -u DS4_ZCHAIN \
        OPENBLAS_NUM_THREADS=1 \
        DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
        DS4_EXPORT_BYTES=0 DS4_ANCHOR="$ANCHOR" DS4_NFIT=1287 DS4_THREADS=20 \
        DS4_LAYER_DIR="$LAYERS" DS4_LCFG=$(printf 'g%.0s' $(seq 1 $NL)) \
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_DUMP_LOGITS=/tmp/q2z_student.bin \
        ./ds4quant_run "$IDS" 1716 2>&1 | grep -E 'ops=|VERDICT' | tail -5
    cd "$ROOT"
    python3 "$SC/anchor_metrics.py" --ref "$ANCHOR" --ids "$IDS" \
        --student /tmp/q2z_student.bin --fit 1287 || true
}

case "${1:-all}" in
    anchor) stage_anchor ;;
    quant)  stage_quant ;;
    zside)  stage_zside ;;
    merge)  stage_merge ;;
    metrics) stage_metrics ;;
    all)    stage_anchor && stage_quant && stage_zside ;;
    *) echo "用法: $0 [anchor|quant|zside|merge|all]" >&2; exit 1 ;;
esac
