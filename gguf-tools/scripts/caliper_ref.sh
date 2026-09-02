#!/bin/bash
# caliper_ref.sh — 校尺: 量化器参考前向对任意层件出 wt2 判决(2026-08-24 尺子事故)。
# 口径=base86p_spark.sh 裸判段逐字照抄。二进制=现行 ds4quant_run(2026-08-31 用户令"同一功能只许一份实现",
# .old 冻结判官清除; n_fit=1 ⇒ 纯回放不 sweep)。
# 用法: caliper_ref.sh <层件目录> <输出logits> [线程=20]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LAYERS="$(realpath "${1:?层件目录}")"
# ★OUT 必须 realpath(2026-08-29 实撞)★: 本脚本中途 cd 到 gguf-tools/amp, 传相对路径会
# 跑偏到那里 → 前向报"打不开" → 后面的 anchor_metrics 拿不到 student 文件【静默跳过】,
# 只剩前向内部 VERDICT, 五指标看着像没跑。-m 允许目标尚不存在。
OUT="$(realpath -m "${2:?输出logits}")"; THR="${3:-20}"
# 可选路由偏置(2026-08-27): $4=Δb 文件 $5=α。不给则完全走原路, 字节与历史判决逐位同。
# 尺子只此一份 —— 想量"带偏置的分数"就从这里量, 不许另抄一份判决脚本。
# ★α 必须显式给(2026-08-31 魔数扫除)★: 旧默认 2.5 是 v4bf 冠军值, 但 α 在平权 86G
# 底座上已判"单调有害"(fable5 十一) —— 只传 Δb 不传 α 会静默吃到有害值。
RB="${4:-}"; RBA="${5:-}"
RB_FLAGS=()
if [ -n "$RB" ]; then
    [ -n "$RBA" ] || { echo "给了路由偏置 Δb($RB)必须显式给 α(\$5): 无默认值" >&2; exit 2; }
    RB_FLAGS=(--route-bias "$RB" --route-bias-alpha "$RBA")
fi
R30="$ROOT/gguf/go-onebit/r30"; G7="$ROOT/gguf/go-onebit/g7"
# ★$6=ids $7=锚: 只给【诊断针】用, 不是判决★(2026-08-28)
# 判决口径永远是 wt2(默认值), 铁律"判决只认参考前向尺"不因这两个可选参数松动。
# 加它们是为了回答一个具体问题: 反修在【与校准语料同源但不相交】的片上是改善还是退化 ——
# 改善=分布错配(换语料有用), 退化=对那 8192 token 过拟合(换语料白搭)。
# 换了语料就必须自己配对应的 FP 锚, 两者错配会出 PPL 2.4e7 这种一眼假的数。
IDS="${6:-$G7/wt2.ids}"; ANC="${7:-$R30/anchor_wt2_s2653.bin}"
SN=8000; [ "$IDS" != "$G7/wt2.ids" ] && { SN=$(wc -l < "$IDS"); echo "★★诊断针口径(非判决): ids=$(basename "$IDS") S=$SN 锚=$(basename "$ANC")★★" >&2; }
N=$(ls "$LAYERS"/dql_vq_L*.bin 2>/dev/null | wc -l)
[ "$N" = 43 ] || { echo "层件不齐 $N/43" >&2; exit 2; }
# ★内存预算按机器给, 不写死★(2026-08-27): 16GiB Mac 时代的 8GiB 缓存预算在 121GiB
# 机器上逼着判决尺反复驱逐 fp16 层缓存再从 HF 盘重载(~0.6s/访)。取物理内存一半。
# ★纯速度开关, 不动数值★: 驱逐后重载回来的权重逐位相同; 已用同一层件前后两跑
# logits md5 对拍验证(见 fable5)。macOS 无 /proc ⇒ 落 0 = 二进制默认(同样是物理内存一半)。
BFMEM=$(awk '/MemTotal/{printf "%.0f", $2/1048576/2}' /proc/meminfo 2>/dev/null || echo 0)
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
LCx=$(printf "g%.0s" $(seq 1 43))
cd "$ROOT/gguf-tools/amp"
# 2026-08-31 env 大扫除: 判决尺全参数走 CLI flag(发车命令一眼可见), OPENBLAS 线程是
# 外部库自己的 env 不在禁令内。
env OPENBLAS_NUM_THREADS=1 ./ds4quant_run "$IDS" "$SN" \
    --hf "$ROOT/hf/DeepSeek-V4-Flash-Vision-Exp" --bf-memgb "$BFMEM" \
    --bf-only --coadapt 1 --calib-fullset \
    --export-bytes 0 --anchor "$ANC" --nfit 1 --threads "$THR" \
    --layer-dir "$LAYERS" --lcfg "$LCx" --vq --tgt-alpha 1.0 \
    --dump-logits "$OUT" "${RB_FLAGS[@]}" 2>&1 | tail -2
cd "$ROOT"
# 五指标判决器=C 版(2026-08-25 Python→C 迁移 Wave A; 金标对拍 amp2 verdict 全五指标
# 与 anchor_metrics.py 逐字符一致, C 版另多 Σmin 主尺; 金标记录 migrate/golden.txt)
AM="$ROOT/gguf-tools/bench/anchor_metrics"
[ -x "$AM" ] || make -C "$ROOT/gguf-tools" anchor_metrics
"$AM" --ref "$ANC" --ids "$IDS" --student "$OUT" --tail 3
