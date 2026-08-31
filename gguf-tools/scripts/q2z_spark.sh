#!/bin/bash
# q2z_spark.sh — q2z 战役的 Spark(GB10/aarch64) 驱动: M1 离线, 锚/层件在 spark 重造。
# 与 q2z_campaign.sh 同参数同语义, 差异只有: ①QBIN=spark 编译的 ds4quant_run(openblas)
# ②阶段可从 anchor 起。用法: bash q2z_spark.sh [anchor|quant|zside|all]
set -uo pipefail
ROOT="$HOME/ds4-main"
[ -x "$ROOT/gguf-tools/amp/zlayer" ] || make -C "$ROOT/gguf-tools" zlayer   # C 反修解算器(zlayer.py 已删)
SC="$ROOT/gguf-tools/scripts"
R30="$ROOT/gguf/go-onebit/r30"
OUTF="$R30/full"
LAYERS="$OUTF/layers"
QBIN="$ROOT/gguf-tools/amp/ds4quant_run"
ANCHOR="$R30/anchor_r30_s1716.bin"
IDS="$ROOT/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids"
export DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}"
LOG(){ echo "[q2z-spark $(date +%H:%M:%S)] $*" >&2; }
NL=43

stage_anchor(){
    [ -f "$ANCHOR" ] && { LOG "锚已在"; return 0; }
    cd "$ROOT/gguf-tools/amp"
    "$QBIN" "$IDS" 1716 --hf "$DS4_HF" --fp-only --anchor "$ANCHOR" \
        --nfit 933 --threads "${DS4_THREADS:-20}" || { LOG "★锚失败 rc=$?★"; exit 3; }
    [ -f "$ANCHOR" ] || { LOG "★锚没落盘★"; exit 3; }
    LOG "锚 ✓ $(ls -l "$ANCHOR" | awk '{printf "%.2f GiB",$5/1073741824}')"
}

quant_flags(){  # q2z_campaign.sh 原样配方(平权 VQ + FULLSET); 2026-08-31 env 大扫除改 flag 拼装
    # 外层专家循环已 pthread 并行; scipy-openblas 内层再开线程=20×20 锁争用(实测 CPU
    # 只吃 11.5/20 核) → BLAS 钉单线程。锚路由(原 DS4_ANCHOR_ROUTE)/体积闸(原
    # DS4_VOL_BUDGET_GIB=72)已分别写死进二进制/由产物 manifest 核账。
    export OPENBLAS_NUM_THREADS=1
    QF=(--hf "$DS4_HF" --anchor "$ANCHOR" --nfit 933 --threads "${DS4_THREADS:-20}" --calib-fullset)
    QF+=(--minvol --mv-baseline --tune --pure-vq)
    QF+=(--vq --tgt-alpha 1.0 --vq-rplan "$OUTF/rplan.txt")
    QF+=(--go2b-hot 1 --go2b-hot-table "$ROOT/gguf-tools/data/corpus/prog_active_top49.txt")
    QF+=(--bf-gain-gate 0.05)
    QF+=(--plan "$OUTF/plan.txt" --ckpt-dir "$OUTF/ckpt" --layer-dir "$LAYERS")
    QF+=(--zfile /tmp/zfile_junk.bin --zchain /tmp/zchain_junk.bin)
    QF+=(--bf-memgb "${DS4_BF_MEMGB:-60}")   # spark 121G: 放宽 fp16 缓存驱逐阈
}

stage_quant(){
    mkdir -p "$LAYERS" "$OUTF/ckpt"
    quant_flags
    # rplan: q2z 平权计划(原战役产物); 缺则从 q2_plan.json 生成或停
    [ -f "$OUTF/rplan.txt" ] || { LOG "★rplan 缺: $OUTF/rplan.txt — 看 r30/plan 或重生成★"; exit 2; }
    cd "$ROOT/gguf-tools/amp"
    "$QBIN" "$IDS" 1716 "${QF[@]}" || { LOG "★量化批量失败 rc=$?★"; exit 2; }
    LOG "量化批量完: $(ls "$LAYERS"/dql_vq_L*.bin 2>/dev/null | wc -l | tr -d ' ')/43 层"
}

stage_zside(){
    cd "$ROOT"
    for L in $(seq 0 $((NL-1))); do
        F="$LAYERS/$(printf 'dql_vq_L%02d.bin' $L)"
        [ -f "$F" ] || { LOG "★L$L 未量化, 停★"; exit 2; }
        "$ROOT/gguf-tools/amp/zlayer" "$DS4_HF" "$LAYERS" "$ANCHOR" "$L" 1024 1 \
            2>&1 | tee -a /tmp/q2z_zside.log | grep '★' || true
    done
    LOG "z 侧车段收官"
}

stage_merge(){   # 合并 GGUF(q2z_campaign stage_merge 的 spark 适配): skeleton+blob --no-down + zchain 外挂
    MDL="$ROOT/gguf/go-onebit/ds4-q2z.gguf"
    MAN="$LAYERS/manifest.txt"
    [ -f "$MAN" ] || { LOG "★manifest 缺★"; exit 4; }
    "$(dirname "$0")/../amp/dql_to_zchain" "$LAYERS" "$OUTF/zchain_q2z.bin" 43 \
        || { LOG "★zchain 抽取失败★"; exit 5; }
    FREE=$(df -BG --output=avail "$ROOT/gguf" | sed -n 2p | tr -dc 0-9)
    [ "${FREE:-0}" -ge 90 ] || { LOG "★盘不足 90G 停★"; exit 7; }
    LOG "起合并(全VQ形态, 非消费式)"
    "$(dirname "$0")/../quantize/vq_merge_v4" --merge \
        --skeleton "$R30/r30_skeleton.gguf" \
        --blob-sizes "$MAN" --no-down \
        --dql-host 127.0.0.1 --dql-dir "$LAYERS" \
        --out "$MDL" > /tmp/q2z_merge.log 2>&1 \
        || { LOG "★合并失败★"; tail -5 /tmp/q2z_merge.log >&2; exit 6; }
    LOG "合并完: $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}')  zchain=$OUTF/zchain_q2z.bin"
}

stage_metrics(){  # 五指标回放(量化+z侧车 vs FP 锚, held=位置1287..1716) — q2z_campaign 同口径
    cd "$ROOT/gguf-tools/amp"
    env OPENBLAS_NUM_THREADS=1 \
        ./ds4quant_run "$IDS" 1716 --hf "$DS4_HF" \
        --gsweep 0 --bf-only --coadapt 1 --calib-fullset \
        --export-bytes 0 --anchor "$ANCHOR" --nfit 1287 --threads 20 \
        --layer-dir "$LAYERS" --lcfg "$(printf 'g%.0s' $(seq 1 $NL))" \
        --vq --tgt-alpha 1.0 --dump-logits /tmp/q2z_student.bin \
        2>&1 | grep -E 'ops=|VERDICT' | tail -5
    cd "$ROOT"
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$ANCHOR" --ids "$IDS" \
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
