#!/bin/bash
# v41_mix_probe_spark.sh — 混位宽分配探针(2026-09-20, spark 本机跑): 哪 10 层降到 11 位最不伤。
#
# 【干什么】两份现成产物按层软链拼成 5 个目录(一个字节不量化), 两把尺各判一趟 5 个学生:
#   ref    = 40 层 12 位(现役 vq8x4096-fp4 的层分片) + 100 GB 档的 common(q4_K embed/head + 三塔 VQ)
#   b0..b3 = ref 里把 L00-09 / L10-19 / L20-29 / L30-39 一块换成 100 GB 档(vq8x2048-q4k)的 11 位分片
#   逐块 ΔΣmin(对 ref) = 这 10 层降位的代价。110 GB 配方要降 15 层: 最不敏感的块整块 + 次不敏感块的深 5 层。
# 【为什么零量化】两份产物每层都是同一个量化器、同一套零语料配方量出来的, 拼出来的目录与"重新量成这样"
#   逐字节同; 判决只跑学生(两把尺的教师锚都在 gguf/v41judge/ 缓存), 一把尺 5 学生。
# 【顺带收账】ref 对 09-19 现役目录(fp4 common)的差 = common 单变量(q4_K embed/head + 三塔 VQ)。
#   注意层分片里 attention 骨架跟着那一层的产物走(12 位层 = FP4 骨架, 11 位层 = q4_K 骨架), 这个 ~0.001/块
#   的偏差四块同向, 不影响排序, 但 ref 不是"全 q4_K 骨架"。
# 用法: v41_mix_probe_spark.sh [--hi DIR] [--lo DIR] [--tag NAME] [--no-judge]
#   hi 默认现役 vq8x4096-fp4 目录(12 位), lo 默认 vq8x2048-q4k(11 位, common 也从它取)
# 产物: gguf/v41/<tag>-{ref,b0,b1,b2,b3}/(软链 + index.json), 总日志 gguf/v41/<tag>-log.txt,
#   汇总 gguf/v41/<tag>-summary.txt; 判决逐学生日志在 gguf/v41judge/log_stu_*_file_<tag>-*.txt
# 出错会怎样: 拼装缺分片直接停(基底目录不完整); 判决失败的学生在汇总里没有那一行 —— 别拿"没数"当"零差"。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
HI="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4"
LO="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x2048-q4k"
TAG="probe-mix"; JUDGE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --hi) HI="$2"; shift 2;;
    --lo) LO="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --no-judge) JUDGE=0; shift;;
    *) echo "★不认识的参数 $1★"; exit 2;;
  esac
done
Q="$ROOT/gguf-tools/scripts/v41_quantize_spark.sh"
J="$ROOT/gguf-tools/scripts/v41_judge.sh"
MAIN="$ROOT/gguf/v41/$TAG-log.txt"; SUM="$ROOT/gguf/v41/$TAG-summary.txt"
LOG(){ echo "[mixprobe $(date '+%m-%d %H:%M:%S')] $*"; }
exec > >(tee -a "$MAIN") 2>&1
[ -d "$HI" ] && [ -d "$LO" ] || { echo "★hi=$HI / lo=$LO 有一个不是目录★"; exit 2; }
# 实例锁: 学生 mmap 100 GB, 对面有模型/量化进程就不发(模式用 [x] 防 pgrep 自匹配)
if pgrep -f "^\./ds4 -m |v41_teacher[.]py|quantize/v41_quantiz[e]" >/dev/null; then echo "★有模型/量化进程在跑, 不发★"; exit 1; fi

# 拼一个目录: base = lo 全量(40 层 + common), 再把 hi 的若干层段盖上; 然后数软链指向核对
assemble() {   # $1 = 目录名  $2 = 期望指向 hi 的层数  $3.. = 用 hi 的层段 a:b
    local name="$1" want_hi="$2"; shift 2
    local out="$ROOT/gguf/v41/$name" args=() seg
    for seg in "$@"; do args+=(--base-layers "$HI:$seg"); done
    LOG "拼 $name: base=$(basename "$LO"), 12 位段 [$*] ← $(basename "$HI")"
    bash "$Q" --out "$out" --base "$LO" "${args[@]+"${args[@]}"}" --layers 0:0 --no-common --no-judge \
        || { LOG "★拼装 $name 失败★"; exit 3; }
    local n_hi n_lo
    n_hi=$(find "$out" -maxdepth 1 -name 'model-layer??.safetensors' -lname "$HI/*" | wc -l)
    n_lo=$(find "$out" -maxdepth 1 -name 'model-layer??.safetensors' -lname "$LO/*" | wc -l)
    [ "$n_hi" = "$want_hi" ] && [ $((n_hi + n_lo)) = 40 ] && [ -e "$out/model-common.safetensors" ] \
        && [ -s "$out/model.safetensors.index.json" ] \
        || { LOG "★$name 指向核对失败: hi=$n_hi(要 $want_hi) lo=$n_lo common=$([ -e "$out/model-common.safetensors" ] && echo 有 || echo 无)★"; exit 3; }
    LOG "  $name ✓ 12 位 $n_hi 层 / 11 位 $n_lo 层 / common ← $(readlink "$out/model-common.safetensors" | xargs dirname | xargs basename)"
}
assemble "$TAG-ref" 40 0:40
assemble "$TAG-b0"  30 10:40
assemble "$TAG-b1"  30 0:10 20:40
assemble "$TAG-b2"  30 0:20 30:40
assemble "$TAG-b3"  30 0:30

[ "$JUDGE" = 1 ] || { LOG "收工(只拼装, 未判决)"; exit 0; }
SPECS=(); for k in ref b0 b1 b2 b3; do SPECS+=("file:$ROOT/gguf/v41/$TAG-$k"); done
# 金融 j 8192 是主尺(部署域), wt2 512 守门(英文, 快); 八域 8192 不跑 —— 09-19 三尺同向同幅, 排序用两把够
for R in "gguf/go-onebit/vqfin41/vqhalf_j.ids 8192" "gguf/go-onebit/g7/wt2.ids 512"; do
    set -- $R
    LOG "判决 $(basename "$(dirname "$1")")/$(basename "$1") 前 $2 token: 5 学生(教师锚缓存)"
    bash "$J" "$ROOT/$1" "$2" "${SPECS[@]}" || LOG "★判决 $1 有失败项, 看 gguf/v41judge/log_stu_*_file_$TAG-*.txt★"
done

# 汇总: 从本日志抠 anchor_metrics 的四行, 按 尺 × 学生 列表, 并算对 ref 的 ΔΣmin / ΔSame top
awk -v tag="$TAG" '
  /^\[mixprobe .*判决 / { ruler = $0; sub(/.*判决 /, "", ruler); sub(/ token.*/, "", ruler) }
  /五指标 落盘模型/ { stu = $NF; sub(tag "-", "", stu) }
  /Σmin = / { for (i = 1; i <= NF; i++) if ($i == "Σmin") { smin[ruler, stu] = $(i + 2) } }
  /Same top token/ { v = $NF; sub(/%/, "", v); same[ruler, stu] = v; seen[ruler] = 1; order[ruler] = order[ruler] " " stu }
  END {
    for (r in seen) {
      printf "\n== %s ==\n%-6s %8s %9s %10s %9s\n", r, "学生", "Σmin", "ΔΣmin", "Same top", "Δpp";
      n = split(order[r], st, " ");
      for (i = 1; i <= n; i++) { s = st[i]; if (s == "") continue;
        ds = ((r, "ref") in smin) ? smin[r, s] - smin[r, "ref"] : 0;
        dp = ((r, "ref") in same) ? same[r, s] - same[r, "ref"] : 0;
        printf "%-6s %8.4f %+9.4f %9.2f%% %+8.2f\n", s, smin[r, s], ds, same[r, s], dp }
    }
  }' "$MAIN" | tee "$SUM"
LOG "收工: 汇总 $SUM"
