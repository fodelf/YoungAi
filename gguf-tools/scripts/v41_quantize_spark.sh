#!/bin/bash
# v41_quantize_spark.sh — V4.1 量化发车(spark 本机跑, 2026-09-12)。
#
# 【干什么】①编 v41_quantize + libv41vq.so ②建输出目录: 软链 config/tokenizer/inference/
# encoding 与 engram 所在的第 47/48 分片(engram 203 GB 不量化也不复制) ③跑量化(routed 专家
# VQ dim8×nc4096, 骨架 FP4, 无语料校准) ④判决: 教师端从【量化目录】加载, 对现成教师锚打五指标。
#
# 【为什么判决必须从目录加载】此前的五指标都是"量化-反量化后原地跑"的读数; 这次的产物是
# 文件, 文件读回来的数才算数(铁律: 捕获与部署同路)。解码核与量化器内部算残差的是同一个。
#
# 用法: v41_quantize_spark.sh [--out DIR] [--layers a:b] [--no-common] [--no-judge] [--force]
#                             [--vq-nc N] [--vq-iters N] [--vq-stride N] [--skel q4k|fp4]
#                             [--base DIR] [--base-layers DIR:a:b]... [--judge-set fin|eight|wt2|all]
#                             [--calib <取料目录>] [--calib-ab]   (金融域校准量化, 2026-09-20; 料由 v41_amp_run --dump-calib 落)
#   冒烟: v41_quantize_spark.sh --layers 0:1 --no-common --no-judge   (只出第 0 层一个分片)
#   探针: --base <现役目录> --layers 0:10 --no-common --vq-nc 2048   (只重做前 10 层, 其余软链现役)
#   拼装: --base <11位目录> --base-layers <12位目录>:10:40 --layers 0:0 --no-common --no-judge
#         (零量化: 只软链 + 重生成 index.json; 混位宽探针目录就是这么拼的, 见 v41_mix_probe_spark.sh)
#
# 【--base 在做什么】把一份已有量化目录的分片软链过来当基底, 量化器只重做 --layers 指定的层
# (它本来就"续跑按层跳过"), 于是能用 10 层的代价量出"只换这几层"的单变量差值。
# ★写盘器是 .part 完成才 rename★, rename 替换的是符号链接本身, 不会写穿到基底目录。
# 【--base-layers DIR:a:b】在 --base 之上再盖一段: 把 DIR 的 model-layer[a,b) 软链过来(可重复给多段)。
# 两份现成产物(40 层 12 位 / 40 层 11 位)按层拼成一个目录, 一个字节不量化就能判"哪几层降位最不伤"。
# 为什么要 --layers 0:0: 量化器层循环空转、common 跳过, 只剩 write_index 按盘上实际文件重生成 index.json。
#
# 【--judge-set】判决尺: fin=金融 vqfin41/vqhalf_j 8192 / eight=八域 vqhalf/vqhalf_j 8192 /
# wt2=英文 g7/wt2 512 / all=三尺全跑。三份教师锚都已在 gguf/v41judge/ 缓存, 只跑学生趟。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
HF="$ROOT/hf/DeepSeek-V4.1-Flash"
OUT="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4"
QARGS=(); JUDGE=1; BASE=""; BASEL=(); JSET="eight"; VS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --no-judge) JUDGE=0; shift;;
    --base) BASE="$2"; shift 2;;
    --base-layers) BASEL+=("$2"); shift 2;;
    --judge-set) JSET="$2"; shift 2;;
    --judge-vs) VS="$2"; shift 2;;
    --layers|--vq-nc|--vq-iters|--vq-stride|--skel|--calib|--calib-ef|--calib-alpha|--vq-nc-table|--vq-nc-layers|--vq-ecvq|--stats-out) QARGS+=("$1" "$2"); shift 2;;
    --no-common|--force|--mtp-vq|--calib-ab|--vq-shared-cb|--vq-cb-fp8) QARGS+=("$1"); shift;;
    *) echo "★不认识的参数 $1★"; exit 2;;
  esac
done
case "$JSET" in
  fin)   JUDGES=("gguf/go-onebit/vqfin41/vqhalf_j.ids 8192");;
  eight) JUDGES=("gguf/go-onebit/vqhalf/vqhalf_j.ids 8192");;
  wt2)   JUDGES=("gguf/go-onebit/g7/wt2.ids 512");;
  all)   JUDGES=("gguf/go-onebit/vqfin41/vqhalf_j.ids 8192"
                 "gguf/go-onebit/vqhalf/vqhalf_j.ids 8192"
                 "gguf/go-onebit/g7/wt2.ids 512");;
  *) echo "★--judge-set 只认 fin/eight/wt2/all, 收到 $JSET★"; exit 2;;
esac
LOG(){ echo "[v41q $(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/quantize_log.txt") 2>&1

LOG "① 编译 v41_quantize / libv41vq.so"
make -C "$ROOT/gguf-tools" v41_quantize || { LOG "★编译失败★"; exit 1; }
file "$ROOT/gguf-tools/quantize/v41_quantize" | grep -q "ELF.*aarch64" || { LOG "★不是 ELF aarch64 二进制★"; exit 1; }

LOG "② 输出目录 $OUT: 软链非权重文件与 engram 分片"
for f in config.json tokenizer.json tokenizer_config.json encoding inference README.md; do
    [ -e "$HF/$f" ] && ln -sfn "$HF/$f" "$OUT/$f"
done
# engram 只住这两片(09-12 查过分片头: 47/48 里只有 layers.{1,14}.engram.* 六种张量, 不混别的)
for s in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    ln -sfn "$HF/$s" "$OUT/$s"
done

if [ -n "$BASE" ]; then
    [ -d "$BASE" ] || { echo "★--base $BASE 不是目录★"; exit 2; }
    # 只软链本程序写出的分片(layer/common); index.json 由量化器按盘上实际文件重生成, 不抄基底的。
    n=0
    for f in "$BASE"/model-layer*.safetensors "$BASE"/model-common.safetensors; do
        [ -e "$f" ] || continue
        b=$(basename "$f")
        # 已经是本目录真产物(不是软链)就不动 —— 防止把自己量出来的分片换回基底的
        [ -f "$OUT/$b" ] && [ ! -L "$OUT/$b" ] && continue
        ln -sfn "$(cd "$(dirname "$f")" && pwd)/$b" "$OUT/$b"; n=$((n+1))
    done
    echo "[v41q] 基底 $BASE: 软链 $n 个分片(量化器会按 --layers 覆盖指定层)"
fi
for spec in "${BASEL[@]+"${BASEL[@]}"}"; do
    IFS=: read -r d a b <<<"$spec"
    [ -d "$d" ] && [ -n "$a" ] && [ -n "$b" ] || { echo "★--base-layers 要 DIR:a:b, 收到 $spec★"; exit 2; }
    n=0
    for ((L=a; L<b; L++)); do
        bn=$(printf "model-layer%02d.safetensors" "$L")
        [ -e "$d/$bn" ] || { echo "★$d/$bn 不存在★"; exit 2; }
        [ -f "$OUT/$bn" ] && [ ! -L "$OUT/$bn" ] && continue    # 本目录真产物不动(同 --base)
        ln -sfn "$(cd "$d" && pwd)/$bn" "$OUT/$bn"; n=$((n+1))
    done
    echo "[v41q] 基底层 $d [$a,$b): 软链 $n 个分片"
done

# ★看门狗★: 按 free 的 available 算, 不看 RSS(RSS 对 mmap 虚高, fable5 09-12)。
# 量化器本身只该吃几 GB(逐专家流式), available 掉到 8 GB 以下就是出事了, 直接杀。
watchdog() {
    while sleep 20; do
        local av; av=$(free -g | awk '/^内存|^Mem/{print $7}')
        echo "$(date '+%H:%M:%S') avail=${av}G" >>"$OUT/mem_log.txt"
        if [ "${av:-99}" -lt 8 ]; then
            echo "★看门狗: available ${av}G < 8G, 停车★" | tee -a "$OUT/mem_log.txt"
            pkill -f quantize/v41_quantize; pkill -f v41_teacher.py
            return 1
        fi
    done
}
watchdog & WD_PID=$!
trap 'kill $WD_PID 2>/dev/null' EXIT

LOG "③ 量化 → $OUT  (${QARGS[*]:-全层+common})"
df -h "$OUT" | tail -1
"$ROOT/gguf-tools/quantize/v41_quantize" "$HF" "$OUT" "${QARGS[@]}" || { LOG "★量化失败★"; exit 3; }
du -sh --apparent-size "$OUT" | awk '{print "目录落地(含软链 engram): "$1}'
find "$OUT" -maxdepth 1 -type f -name "model-*.safetensors" -printf "%s\n" | awk '{s+=$1} END{printf "本程序写出的分片合计 %.3f GB (十进制)\n", s/1e9}'

[ "$JUDGE" = 1 ] || { LOG "收工(未判决)"; exit 0; }
# 用 bash 显式调用: scp 过来的脚本没有 x 位(09-12 实撞: 量化跑完 58 分钟, 判决一步因"权限不够"没起)
rc=0
# --judge-vs: 同一把尺上多跑一个对照目录(教师锚是缓存的, 只多一趟学生)。
# 为什么要它: 现役那份的历史读数是【引擎路】的, 与这里的【文件态 Python 路】不同口径,
# 直接比就是拿两把尺量两个东西 —— 对照必须在同一趟同一把尺上产生。
SPECS=("file:$OUT"); [ -n "$VS" ] && SPECS+=("file:$VS")
for J in "${JUDGES[@]}"; do
    set -- $J
    LOG "④ 判决($JSET): 教师端从量化目录加载, $1 前 $2 token, 对现成教师锚; 档: ${SPECS[*]}"
    bash "$ROOT/gguf-tools/scripts/v41_judge.sh" "$ROOT/$1" "$2" "${SPECS[@]}" || { LOG "★判决失败 $1★"; rc=4; }
done
[ "$rc" = 0 ] || exit "$rc"
LOG "收工"
