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
#   冒烟: v41_quantize_spark.sh --layers 0:1 --no-common --no-judge   (只出第 0 层一个分片)
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
HF="$ROOT/hf/DeepSeek-V4.1-Flash"
OUT="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-fp4"
QARGS=(); JUDGE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --no-judge) JUDGE=0; shift;;
    --layers) QARGS+=("$1" "$2"); shift 2;;
    --no-common|--force) QARGS+=("$1"); shift;;
    *) echo "★不认识的参数 $1★"; exit 2;;
  esac
done
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
LOG "④ 判决: 教师端从量化目录加载, 判决份 vqhalf_j 8192 token, 对现成教师锚"
# 用 bash 显式调用: scp 过来的脚本没有 x 位(09-12 实撞: 量化跑完 58 分钟, 判决一步因"权限不够"没起)
bash "$ROOT/gguf-tools/scripts/v41_judge.sh" "$ROOT/gguf/go-onebit/vqhalf/vqhalf_j.ids" 8192 "file:$OUT" || { LOG "★判决失败★"; exit 4; }
LOG "收工"
