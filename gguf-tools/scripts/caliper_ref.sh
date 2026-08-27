#!/bin/bash
# caliper_ref.sh — 校尺: 量化器参考前向对任意层件出 wt2 判决(2026-08-24 尺子事故)。
# 口径=base86p_spark.sh 裸判段逐字照抄; 二进制=ds4quant_run.old(08-22, 08-23 改动把 lfile 加载改坏待修)(历史对表 M10裸底0.4956/官方q2 0.4207 同尺)。
# 用法: caliper_ref.sh <层件目录> <输出logits> [线程=20]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LAYERS="$(realpath "${1:?层件目录}")"; OUT="${2:?输出logits}"; THR="${3:-20}"
# 可选路由偏置(2026-08-27): $4=Δb 文件 $5=α。不给则完全走原路, 字节与历史判决逐位同。
# 尺子只此一份 —— 想量"带偏置的分数"就从这里量, 不许另抄一份判决脚本。
RB="${4:-}"; RBA="${5:-2.5}"
RB_ENV=(); [ -n "$RB" ] && RB_ENV=(DS4_ROUTE_BIAS="$RB" DS4_ROUTE_BIAS_ALPHA="$RBA")
R30="$ROOT/gguf/go-onebit/r30"; G7="$ROOT/gguf/go-onebit/g7"
N=$(ls "$LAYERS"/dql_vq_L*.bin 2>/dev/null | wc -l)
[ "$N" = 43 ] || { echo "层件不齐 $N/43" >&2; exit 2; }
# ★内存预算按机器给, 不写死★(2026-08-27): 原为 DS4_BF_MEMGB=8(16GiB Mac 时代), 在 121GiB
# 机器上逼着判决尺反复驱逐 fp16 层缓存再从 HF 盘重载(~0.6s/访)。
# ★注意: 判决尺跑的是冻结的 ds4quant_run.old, 源码改不进去 —— 必须走 env 覆盖★
# (老二进制内置默认也才 9.5, 去掉 =8 等于没改)。
# ★纯速度开关, 不动数值★: 驱逐后重载回来的权重逐位相同; 已用同一层件前后两跑
# logits md5 对拍验证(见 fable5)。
BFMEM=$(awk '/MemTotal/{printf "%.0f", $2/1048576/2}' /proc/meminfo 2>/dev/null || echo 8)
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
LCx=$(printf "g%.0s" $(seq 1 43))
cd "$ROOT/gguf-tools/amp"
env DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731" OPENBLAS_NUM_THREADS=1 DS4_BF_MEMGB="$BFMEM" \
    DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
    DS4_EXPORT_BYTES=0 DS4_ANCHOR="$R30/anchor_wt2_s2653.bin" DS4_NFIT=1 DS4_THREADS="$THR" \
    DS4_LAYER_DIR="$LAYERS" DS4_LCFG="$LCx" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
    DS4_DUMP_LOGITS="$OUT" "${RB_ENV[@]}" ./ds4quant_run.old "$G7/wt2.ids" 8000 2>&1 | tail -2
cd "$ROOT"
# 五指标判决器=C 版(2026-08-25 Python→C 迁移 Wave A; 金标对拍 amp2 verdict 全五指标
# 与 anchor_metrics.py 逐字符一致, C 版另多 Σmin 主尺; 金标记录 migrate/golden.txt)
AM="$ROOT/gguf-tools/bench/anchor_metrics"
[ -x "$AM" ] || make -C "$ROOT/gguf-tools" anchor_metrics
"$AM" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" --student "$OUT" --tail 3
