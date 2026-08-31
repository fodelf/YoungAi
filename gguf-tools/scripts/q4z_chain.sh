#!/bin/bash
# q4z_chain.sh — q4 基准全链(2026-08-09 用户令: 量化→zchain集成→反修→评分→合并→测代码)。
# 配方: vq4×1024 全平权(rplan_q4.txt, 2.5117bpw) + 逐层质量优先 z^L(K≤2048 argmax)+GE
# 注入 dql(评分用混装口径) + 外挂 zchain 抽取(部署口径) + --consume 合并(盘紧)。
# 在 M1 上运行。阶段: quant | zside | metrics | zchain | merge | all
set -uo pipefail
ROOT=/Users/fodelf/ds4-main
Q=$ROOT/gguf-tools/amp
ZLB="$ROOT/gguf-tools/amp/zlayer"   # C 反修解算器(zlayer.py 已删)
[ -x "$ZLB" ] || make -C "$ROOT/gguf-tools" zlayer
SC=$ROOT/gguf-tools/scripts
R30=$ROOT/gguf/go-onebit/r30
LAYERS=$R30/q4/layers
ANCHOR=$R30/anchor_r30_s1716.bin
IDS=$ROOT/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids
MDL=$ROOT/gguf/go-onebit/ds4-q4z.gguf
LOG(){ echo "[q4z $(date +%H:%M:%S)] $*" >&2; }

# 2026-08-31 env 大扫除: QENV(env 拼串)改 QFLAGS(flag 拼串); .dchunk/.zk1024 冻结件已删,
# 同走现役 ds4quant_run。锚路由(原 DS4_ANCHOR_ROUTE)写死进二进制; 体积闸(原
# DS4_VOL_BUDGET_GIB=92)随 env 拔死, 判停看 manifest 实账。
QFLAGS=(--hf "$ROOT/hf/DeepSeek-V4-Flash-0731" --anchor "$ANCHOR" --nfit 933
  --threads 8 --calib-fullset --minvol --mv-baseline --tune --pure-vq
  --vq --tgt-alpha 1.0 --vq-rplan "$ROOT/gguf/go-onebit/rplan_q4.txt"
  --go2b-hot 1 --go2b-hot-table "$ROOT/gguf-tools/data/corpus/prog_active_top49.txt"
  --bf-gain-gate 0.05 --plan "$R30/q4/plan.txt" --ckpt-dir "$R30/q4/ckpt"
  --layer-dir "$LAYERS" --zfile /tmp/zf_junk.bin --zchain /tmp/zc_junk.bin)

stage_quant(){   # 批量顺序(误差前向吸收), 已有层自动跳过; dql=稀疏洞文件(评分要用, 不删)
    cd "$Q"; local T0=$(date +%s)
    ./ds4quant_run "$IDS" 1716 "${QFLAGS[@]}" || { LOG "★量化失败 rc=$?★"; exit 2; }
    LOG "量化完 $((($(date +%s)-T0)/60))min: $(ls $LAYERS/dql_vq_L*.bin | wc -l | tr -d ' ')/43"
}
stage_zside(){   # 反修=逐层质量优先 z^L+GE, 注入 dql; zcache 滚动删(盘紧)
    cd "$ROOT"
    for L in $(seq 0 42); do
        "$ZLB" $ROOT/hf/DeepSeek-V4-Flash-0731 $LAYERS $ANCHOR $L 2048 1 \
            2>&1 | grep '★' || { LOG "★L$L zside失败★"; exit 3; }
        rm -f $LAYERS/zcache_L$(printf %02d $L).npz
    done
    LOG "反修段完: 账本 $(wc -l < $LAYERS/zinject_manifest.txt | tr -d ' ')/43"
}
stage_metrics(){ # 评分=全43层回放(量化+注入z) vs FP锚, held=位置1287..1716
    cd "$Q"
    ./ds4quant_run "$IDS" 1716 --hf "$ROOT/hf/DeepSeek-V4-Flash-0731" \
        --gsweep 0 --bf-only \
        --coadapt 1 --calib-fullset --export-bytes 0 --anchor "$ANCHOR" --nfit 1287 \
        --threads 8 --layer-dir "$LAYERS" --lcfg "$(printf 'g%.0s' $(seq 1 43))" \
        --vq --tgt-alpha 1.0 --dump-logits /tmp/q4z_student.bin \
        2>&1 | grep -E 'VERDICT' | tail -2
    cd "$ROOT" && "$(dirname "$0")/../bench/anchor_metrics" --ref $ANCHOR --ids $IDS \
        --student /tmp/q4z_student.bin --fit 1287
}
stage_zchain(){  # 部署外挂 zchain 抽取(dql 混装记录 → DQZ2)
    "$(dirname "$0")/../amp/dql_to_zchain" $LAYERS $ROOT/gguf/go-onebit/ds4-q4z.gguf.zchain.bin 43
}
stage_merge(){   # skeleton 重造 + --consume 合并(层文件边并边删, 盘峰值≈骨架+输出增量)
    [ -f $R30/r30_skeleton.gguf ] || { LOG "骨架重造(skel_from_hf ~30min)";
        SKEL_HF=$ROOT/hf/DeepSeek-V4-Flash-0731 SKEL_TMPL=$R30/template_head.gguf \
        bash $SC/skel_from_hf.sh $R30/r30_skeleton.gguf 4 || { LOG "★骨架失败★"; exit 5; } }
    "$(dirname "$0")/../quantize/vq_merge_v4" --merge --skeleton $R30/r30_skeleton.gguf \
        --blob-sizes $LAYERS/manifest.txt --no-down --consume \
        --dql-host 127.0.0.1 --dql-dir $LAYERS --out $MDL || { LOG "★合并失败★"; exit 6; }
    LOG "合并完: $(ls -l $MDL | awk '{printf "%.2f GB", $5/1e9}')"
}
case "${1:-all}" in
    quant) stage_quant ;; zside) stage_zside ;; metrics) stage_metrics ;;
    zchain) stage_zchain ;; merge) stage_merge ;;
    all) stage_quant && stage_zside && stage_metrics && stage_zchain && stage_merge
         LOG "全链收官(测代码阶段=引擎冒烟+pubbench, 由 M4 侧驱动)" ;;
    *) echo "用法: q4z_chain.sh [quant|zside|metrics|zchain|merge|all]"; exit 1 ;;
esac
