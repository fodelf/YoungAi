#!/bin/bash
# q2z_campaign.sh — 86G 平权 VQ + 每层 z 侧车战役(2026-08-08 用户设计定案)。
# 设计: ①量化=43层全平权 VQ(q2_plan 配方, 无q1/无冷热) ②反修=逐层闭式 z^L(rank1024,
#   ~17MB/层)+GE, 四损失+感知口径, 记录注入 dql(超冠混装形态) ③五指标(held=位置1287..1716)
#   ④合并 GGUF ⑤编码场景。解算只用位置 0..1287, held 段永不入解。
# 阶段: q2z_campaign.sh [quantL <L>|quant|zsideL <L>|zside|metrics|merge|bench]
#   quantL/zsideL = 单层计时探针(先跑一层看时间)。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SC="$ROOT/gguf-tools/scripts"
ZLB="$ROOT/gguf-tools/amp/zlayer"   # C 反修解算器(zlayer.py 已删)
[ -x "$ZLB" ] || make -C "$ROOT/gguf-tools" zlayer
R30="$ROOT/gguf/go-onebit/r30"
OUTF="$R30/full"
LAYERS="$OUTF/layers"
QBIN="$ROOT/gguf-tools/amp/ds4quant_run.dchunk"     # 量化(已验证 L00/L21 配方)
RBIN="$ROOT/gguf-tools/amp/ds4quant_run.zk1024"     # 回放/指标(zl k≤1024 + pv堆化)
ANCHOR="$R30/anchor_r30_s1716.bin"
IDS="$ROOT/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids"
export DS4_HF="${DS4_HF:-$HOME/ds4-main/hf/DeepSeek-V4-Flash-0731}"
LOG(){ echo "[q2z $(date +%H:%M:%S)] $*" >&2; }
NL=43

quant_env(){  # 已验证配方(L00/L21 同款): 平权 VQ, 锚路由, FULLSET
    export DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-8}" DS4_CALIB_FULLSET=1
    export DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_TUNE=1 DS4_PURE_VQ=1
    export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$OUTF/rplan.txt" DS4_VOL_BUDGET_GIB=72
    export DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/data/corpus/prog_active_top49.txt"
    export DS4_ANCHOR_ROUTE=1 DS4_BF_GAIN_GATE=0.05
    export DS4_PLAN="$OUTF/plan.txt" DS4_CKPT_DIR="$OUTF/ckpt" DS4_LAYER_DIR="$LAYERS"
    export DS4_ZFILE=/tmp/zfile_junk.bin DS4_ZCHAIN=/tmp/zchain_junk.bin
    unset DS4_COADAPT DS4_MV_COAD_BASE DS4_BF_ONLY DS4_GSWEEP 2>/dev/null || true
}

stage_quantL(){   # 单层计时: quantL <L>
    local L=$1; quant_env
    cd "$ROOT/gguf-tools/amp"
    local T0=$(date +%s)
    env DS4_MV_PROBE_L=$L DS4_MINVOL_MAXL=$((L+1)) "$QBIN" "$IDS" 1716
    LOG "量化 L$L 用时 $(( $(date +%s)-T0 ))s"
    ls -la "$LAYERS/$(printf 'dql_vq_L%02d.bin' $L)" || { LOG "★产物缺★"; exit 2; }
}

stage_quant(){    # 43 层批量顺序(单进程, 误差前向吸收=部署口径, 里程碑每5层)
    quant_env
    cd "$ROOT/gguf-tools/amp"
    local T0=$(date +%s)
    "$QBIN" "$IDS" 1716 || { LOG "★量化批量失败 rc=$?★"; exit 2; }
    LOG "量化批量完 $(( $(date +%s)-T0 ))s: $(ls "$LAYERS"/dql_vq_L*.bin | wc -l | tr -d ' ')/43 层"
}

stage_zsideL(){   # 单层计时: zsideL <L>(反修=闭式 z^L+GE+注入)
    local L=$1
    cd "$ROOT"
    "$ZLB" "$DS4_HF" "$LAYERS" "$ANCHOR" "$L" 1024 1
}

stage_zside(){    # 逐层反修: 第一层到最后一层
    cd "$ROOT"
    for L in $(seq 0 $((NL-1))); do
        [ -f "$LAYERS/$(printf 'dql_vq_L%02d.bin' $L)" ] || { LOG "★L$L 未量化, 停★"; exit 2; }
        "$ZLB" "$DS4_HF" "$LAYERS" "$ANCHOR" "$L" 1024 1 \
            2>&1 | tee -a /tmp/q2z_zside.log | grep '★' || { LOG "★L$L z侧车失败, 停★"; exit 3; }
    done
    LOG "反修段收官: 账本 $(wc -l < "$LAYERS/zinject_manifest.txt" | tr -d ' ')/43 层"
}

stage_metrics(){  # 五指标: 全43层回放(量化+z侧车) vs FP 锚, held=位置1287..1716
    cd "$ROOT/gguf-tools/amp"
    local T0=$(date +%s)
    env -u DS4_TUNE -u DS4_MINVOL -u DS4_VQ_RPLAN -u DS4_ZCHAIN \
        DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
        DS4_EXPORT_BYTES=0 DS4_ANCHOR="$ANCHOR" DS4_NFIT=1287 DS4_THREADS=8 \
        DS4_LAYER_DIR="$LAYERS" DS4_LCFG=$(printf 'g%.0s' $(seq 1 $NL)) \
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_DUMP_LOGITS=/tmp/q2z_student.bin \
        "$RBIN" "$IDS" 1716 2>&1 | grep -E 'ops=|VERDICT' | tail -5
    LOG "回放遍用时 $(( $(date +%s)-T0 ))s"
    cd "$ROOT"
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$ANCHOR" --ids "$IDS" \
        --student /tmp/q2z_student.bin --fit 1287
}

stage_merge(){    # 合并 GGUF: skeleton + vq blob(--no-down: w2 在 blob which=2 槽) + zchain 外挂
    grep -q 'zk <= 16\|zk<=16' "$ROOT/ds4_zchain.c" && {
        LOG "★前置未过: 引擎 ds4_zchain.c 仍是 k≤16★"; exit 4; }
    MDL="$ROOT/gguf/go-onebit/ds4-q2z.gguf"
    MAN="$LAYERS/manifest.txt"
    [ "$(wc -l < "$MAN" | tr -d ' ')" = 43 ] || { LOG "manifest 不齐"; exit 4; }
    "$(dirname "$0")/../amp/dql_to_zchain" "$LAYERS" "$OUTF/zchain_q2z.bin" 43 || { LOG "★zchain 抽取失败★"; exit 5; }
    FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
    [ "$FREE" -ge 82 ] || { LOG "★free ${FREE}G <82G(非消费合并需全额) — 停★"; exit 7; }
    LOG "起合并(全VQ形态: skeleton+blob --no-down, 非消费式)"
    "$(dirname "$0")/../quantize/vq_merge_v4" --merge \
        --skeleton "$R30/r30_skeleton.gguf" \
        --blob-sizes "$MAN" --no-down \
        --dql-host 127.0.0.1 --dql-dir "$LAYERS" \
        --out "$MDL" > /tmp/q2z_merge.log 2>&1 \
        || { LOG "★合并失败 rc=$?★"; tail -5 /tmp/q2z_merge.log >&2; exit 6; }
    LOG "合并完: $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}')  zchain=$OUTF/zchain_q2z.bin"
}

stage_bench(){    # 编码场景 — 依赖 merge 产物
    LOG "TODO: 合并产物就位后接编码基准(逐题隔离口径)"; exit 4
}

case "${1:-}" in
    quantL) stage_quantL "${2:?层号}" ;;
    quant)  stage_quant ;;
    zsideL) stage_zsideL "${2:?层号}" ;;
    zside)  stage_zside ;;
    metrics) stage_metrics ;;
    merge)  stage_merge ;;
    bench)  stage_bench ;;
    *) echo "用法: q2z_campaign.sh [quantL <L>|quant|zsideL <L>|zside|metrics|merge|bench]" >&2; exit 1 ;;
esac
