#!/bin/bash
# v41_rate_probe_spark.sh — 113 GB 方案的"索引 / 共享"率侧探针(2026-09-21): 一层多档, 只量数字不出模型。
#
# 【量什么】(全部零语料, 码本 Lloyd 解在权重自身; 每档只重做一层, 分片写完读完统计就删)
#   base12  现役配方 nc4096 —— 索引经验熵 H(熵编码能省 12−H bit/索引)、分块定宽 w16/w32、专家残差(须与 09-21 stats_L20 逐专家同 = nocal 路没动)
#   fix13/fix14  nc8192/16384 定长 —— (H, 残差) 两个点, 画"码本变大后熵涨多少、失真降多少"
#   ecvq13-λ / ecvq14-λ  ECVQ 率惩罚 —— 同一本大码本上把平均码长压回 ~12 bit 时失真剩多少 = "索引"这条路在等体积下的真肉
#   sh12/sh13  每层一本共享码本 —— 亏多少残差 = "共享"这条路的代价(省 3.02 GB)
#   fp8-13  nc8192 码本舍 E4M3 —— 8192 词码本塞进 64 KB shared 的代价
# 【读数】每档一行进 summary.txt: 档 nc λ 残差(sse/en) 索引熵H 用到码字 半数码字占比 w16 w32 秒数。
# 用法(spark 本机, nohup): v41_rate_probe_spark.sh [--layer 20] [--only base12,fix13,...] [--keep]
# 停车规则: 任一档量化非 0 停; 尾行 PROBE_EXIT <rc>。看门狗由 v41_quantize_spark.sh 自带(available < 8 GB 杀)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
LAYER=20; ONLY=""; KEEP=0
while [ $# -gt 0 ]; do case "$1" in
  --layer) LAYER="${2:?}"; shift 2;;
  --only) ONLY="${2:?}"; shift 2;;
  --keep) KEEP=1; shift;;
  *) echo "★不认识的参数 $1★"; exit 2;;
esac; done
Q="$ROOT/gguf-tools/scripts/v41_quantize_spark.sh"
PD="$ROOT/gguf/v41/probe-rate"; mkdir -p "$PD"
SUM="$PD/summary.txt"
M(){ echo "[rate $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "PROBE_EXIT 1"; exit 1; }
if pgrep -f "^\./ds4 -m |v41_teacher[.]py|quantize/v41_quantiz[e]" >/dev/null; then die "有模型/量化进程在跑, 不发"; fi
exec > >(tee -a "$PD/probe_log.txt") 2>&1
[ -s "$SUM" ] || echo "# 档 nc λ 残差(sse/en) H(bit/索引) 用到码字 半数码字占比 w16 w32 秒 层" >"$SUM"
# 档表: 名字 | 量化器参数
ARMS=(
  "base12|--vq-nc 4096"
  "fix13|--vq-nc 8192"
  "ecvq13-0.2|--vq-nc 8192 --vq-ecvq 0.2"
  "sh12|--vq-nc 4096 --vq-shared-cb"
  "fix14|--vq-nc 16384"
  "ecvq14-0.4|--vq-nc 16384 --vq-ecvq 0.4"
  "fp8-13|--vq-nc 8192 --vq-cb-fp8"
  "ecvq13-0.1|--vq-nc 8192 --vq-ecvq 0.1"
  "ecvq13-0.4|--vq-nc 8192 --vq-ecvq 0.4"
  "ecvq14-0.2|--vq-nc 16384 --vq-ecvq 0.2"
  "ecvq14-0.8|--vq-nc 16384 --vq-ecvq 0.8"
  "sh13|--vq-nc 8192 --vq-shared-cb"
  "ecvq12-0.2|--vq-nc 4096 --vq-ecvq 0.2"
  "sh13s4|--vq-nc 8192 --vq-shared-cb --vq-stride 4"     # 13 位共享码本池加倍(1/384 取样 ≈ 4.4M 向量): sh13 对 fix13 亏 0.96%, 看是不是池太薄
  "sh12s4|--vq-nc 4096 --vq-shared-cb --vq-stride 4"     # 同上 12 位对照
)
LL=$(printf "%02d" "$LAYER")
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; args="${arm#*|}"
  if [ -n "$ONLY" ] && ! grep -q "\(^\|,\)$name\(,\|$\)" <<<"$ONLY"; then continue; fi
  OUT="$PD/$name-L$LL"; ST="$PD/$name-L$LL.stats"
  lam=$(grep -o -- '--vq-ecvq [0-9.]*' <<<"$args" | awk '{print $2}'); [ -n "$lam" ] || lam=0
  nc=$(grep -o -- '--vq-nc [0-9]*' <<<"$args" | awk '{print $2}')
  M "$name: $args (L$LL)"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  bash "$Q" --out "$OUT" --layers "$LAYER:$((LAYER+1))" --no-common --no-judge --force --skel q4k $args --stats-out "$ST" || die "$name 量化失败"
  dt=$(( $(date +%s) - t0 ))
  [ -s "$ST" ] || die "$name 没有统计文件"
  # 汇总: 残差 = Σsse/Σen(整层 384 专家), 其余按专家平均
  awk -v n="$name" -v nc="$nc" -v lam="$lam" -v dt="$dt" -v L="$LAYER" '
    !/^#/ { s+=$4; e+=$5; H+=$6; u+=$7; m+=$8; w16+=$9; w32+=$10; k++ }
    END { if (k) printf "%s %s %s %.5f %.4f %.0f %.4f %.3f %.3f %d L%d\n", n, nc, lam, s/e, H/k, u/k, m/k, w16/k, w32/k, dt, L }' "$ST" | tee -a "$SUM"
  [ "$KEEP" = 1 ] || rm -f "$OUT"/model-layer*.safetensors    # 分片是探针副产品, 统计已落 .stats; 目录留日志
done
M "收工: $SUM"; cat "$SUM"; echo "PROBE_EXIT 0"
