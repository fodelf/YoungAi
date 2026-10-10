#!/bin/bash
# release_pack.sh — 打 GitHub Release 的可执行包(2026-10-10, 用户: "release 直接下载到比如 spark, 然后直接执行就启动这个页面,
# 然后自己到模型页面下载模型")。
#
# 用法: 在编好的机器上(spark: make cuda-spark 之后), 仓库根里跑
#   bash gguf-tools/scripts/release_pack.sh [输出目录, 默认 <仓库>/dist]
# 产物: youngai-<系统>-<架构>-<日期>.tar.gz 和同名 .sha256。拿到包的人:
#   tar xzf youngai-*.tar.gz && ./youngai-*/ds4-train      # 浏览器开 http://主机:8000/, 到"模型"页下载、加载
#
# 包里放两条流程要的东西, 不放模型(三百多 GB, 页面里下):
#   跑模型: ds4-server + ds4-train(主进程: 页面、起服、训练编排全在它的 C 里) + web/ 页面 + hf_install.sh(模型页下载, hf CLI 的包装)
#   跑训练: ds4(训练器与判决学生都是它) + anchor_metrics(wt2 门的五指标)
#           + 保持料 hold.jsonl + 判决料 wt2.ids + 它的 FP 教师锚 teacher_g7_wt2_n512.bin(265 MB: 要 HF 出厂权重才能出, 拿包的人出不了)
# 二进制是在本机编的, 只能发给同平台机器: spark 上打的(linux-aarch64, CUDA 架构 = GB10)给 DGX Spark 用; Mac 上打的(darwin-arm64, Metal)给
# Apple Silicon 用 —— Mac 包多带 metal/*.metal(着色器源, 引擎运行时按相对路径拼接加载), 且★Mac 上从没跑过真模型★(要 ≥128 GB 统一内存的机器, 手头只有 16 GB)。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
OUT="${1:-$ROOT/dist}"
NAME="youngai-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)-$(date +%Y%m%d)"
# web/ 只收页面与图: spark 的工作树里 web/ 下还躺着老构建留下的 .o, 整目录拷会一起进包
FILES=(ds4-server ds4-train ds4 gguf-tools/bench/anchor_metrics web/*.html web/*.jpg LICENSE LICENSE-DeepSeek README.md README.zh-CN.md
       gguf-tools/scripts/hf_install.sh
       gguf-tools/data/posttrain/hold.jsonl gguf/go-onebit/g7/wt2.ids gguf/v41judge/teacher_g7_wt2_n512.bin)
# 只带一个脚本: 下载(hf CLI 的包装)。起服/训练/门/选轮全在 ds4-train 的 C 里(src/train/train_model.c train_job.c train_gate.c), 10-10 起不再随包发脚本。
[ "$(uname -s)" = Darwin ] && FILES+=(metal/*.metal)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "缺 $f(二进制要先编: make cuda-spark(Mac: make) 与 make -C gguf-tools anchor_metrics)"; exit 1; }; done
# 动态库在本机都得找得到: 找不到的话别人一执行就是 "error while loading shared libraries", 还以为包坏了
if command -v ldd >/dev/null; then
    miss=$(ldd ds4-server ds4-train ds4 gguf-tools/bench/anchor_metrics | grep "not found" || true)
    [ -z "$miss" ] || { echo "缺动态库:"; echo "$miss"; exit 1; }
fi
STAGE="$OUT/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE"   # 只清本脚本自己的暂存目录
for f in "${FILES[@]}"; do mkdir -p "$STAGE/$(dirname "$f")"; cp -a "$f" "$STAGE/$f"; done
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
if command -v sha256sum >/dev/null; then (cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
else (cd "$OUT" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256"); fi   # Mac 没有 sha256sum
rm -rf "$STAGE"
echo "包: $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
echo "校验: $(cat "$OUT/$NAME.tar.gz.sha256")"
