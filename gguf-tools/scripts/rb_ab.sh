#!/bin/bash
# rb_ab.sh — 路由反修(路由偏置侧车 Δb)拟合 + α 扫描。
# 2026-08-27 按当前布局重写(原版写死旧 Mac 路径/Base 模型/rr_code 语料, 已不可用)。
#
# 【这是什么】量化后专家【选择】会漂: 同一个 token, FP 原模型选专家 A, 量化模型选了 B。
# 权重修得再准也白搭 —— 选错专家等于用错了函数。路由偏置就修这一件事:
#   逐层逐专家学一个标量 Δb, 只加在【选择分】上, 不动专家算出来的值。
#   漏选(FP 选了学生没选) Δb += thr−v; 多选(学生选了 FP 没选) Δb −= v−thr; thr=学生第6名分。
# 产物 22KB(43×256 f32 + 计数)。部署=合并时把 α·Δb 烘进 blk.L.exp_probs_b.bias, 引擎零改动。
#
# 【α 有阈值, 别用小 α 下结论】2026-07-28 实测曲线(agree):
#   α≤1.0 → 77.6(阈下, 和没加一样) | 1.5 → 80.3 | 2.0 → 81.6 | ★2.5 → 84.2★ | 3.0 → 81.6(过冲)
#   看到 α=1 没动就说"路由偏置无效"是错的 —— 它要过翻转阈值才开始改变 argmax。
#
# 【两个坑】
#  ① 偏置必须在本底座重新拟合。直接搬别的模型的 Δb 实测净负(fable5: "别人的漂移药方")。
#  ② 哈希路由层(如 L0)选择零漂移, 这层拿不到收益, 日志会打"RB 不适用", 属正常。
#
# 【口径】拟合走校准语料(放大器半), 判决走 wt2 —— 在判决语料上拟合=作弊。
# 拟合与判决同一个二进制(现行 ds4quant_run — 判官归一 2026-08-31, .old 冻结件已删), 零口径缝。
#
# 用法: rb_ab.sh fit   <层件目录> <输出.bin>
#       rb_ab.sh sweep <层件目录> <Δb.bin> [α...]   默认 α=1.5 2.0 2.5 3.0
set -uo pipefail
ROOT="$HOME/ds4-main"; GT="$ROOT/gguf-tools"
D2="$ROOT/gguf/go-onebit/vqhalf"
ANC="$D2/anchor_a_clean_s8192.bin"          # 放大器半校准锚(判决锚零混入)
IDS="$D2/vqhalf_a.ids"                       # 放大器半语料
LOG(){ echo "[rb $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }

stage_fit(){
    local LAY="$1" OUT="$2"
    [ "$(ls "$LAY"/dql_vq_L*.bin 2>/dev/null | wc -l)" = 43 ] || DIE "层件不齐 $LAY"
    [ -s "$ANC" ] || DIE "校准锚缺 $ANC"
    [ -s "$IDS" ] || DIE "校准语料缺 $IDS"
    LOG "拟合: $LAY × 放大器半 8192 → $OUT"
    export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
    local LCx; LCx=$(printf 'g%.0s' $(seq 1 43))
    ( cd "$GT/amp" && env OPENBLAS_NUM_THREADS=1 \
        ./ds4quant_run "$IDS" 8192 --hf "$ROOT/hf/DeepSeek-V4-Flash-0731" \
        --bf-memgb 8 --bf-only --coadapt 1 \
        --calib-fullset --export-bytes 0 --anchor "$ANC" --nfit 1 --threads 20 \
        --layer-dir "$LAY" --lcfg "$LCx" --vq --tgt-alpha 1.0 \
        --route-bias-fit --route-bias-out "$OUT" \
        2>&1 | grep -aE "路由|Δb|rb_save" | tail -6 ) || DIE "拟合失败"
    [ -s "$OUT" ] || DIE "Δb 没落盘(看是否全是哈希路由层, 或 --route-bias-out 未生效)"
    LOG "Δb 落盘 $(ls -l --block-size=1 "$OUT" | awk '{printf "%.1f KB", $5/1024}')"
    awk '{s+=$0}END{}' /dev/null
}

stage_sweep(){
    local LAY="$1" RB="$2"; shift 2
    local AS=("$@"); [ ${#AS[@]} -gt 0 ] || AS=(1.5 2.0 2.5 3.0)
    [ -s "$RB" ] || DIE "Δb 缺 $RB"
    echo "══ α 扫描(判决尺 wt2); 对表 无偏置基线 ══"
    for A in "${AS[@]}"; do
        LOG "α=$A 判决中"
        bash "$GT/scripts/caliper_ref.sh" "$LAY" "/tmp/qc_rb_a$A.bin" 20 "$RB" "$A" \
            > "/tmp/caliper_rb_a$A.log" 2>&1 || { LOG "α=$A 判决失败"; continue; }
        echo "── α=$A ──"
        grep -aE "PPL\(student\)|Σmin|Mean KLD|Same top" "/tmp/caliper_rb_a$A.log" | head -4
    done
}

case "${1:-}" in
  fit)   shift; stage_fit "$@";;
  sweep) shift; stage_sweep "$@";;
  *) echo "用法: rb_ab.sh fit <层件目录> <输出.bin> | sweep <层件目录> <Δb.bin> [α...]"; exit 2;;
esac
