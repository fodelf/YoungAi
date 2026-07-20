#!/bin/sh
# v3_pipeline.sh — 新模型 v3 全链 (跑在 M1, 后台 nohup):
#   [0] 模板头就位 (published GGUF 前 200MiB, 只用元数据/张量结构, 不读权重字节)
#   [1] gen 1-bit base: HF fp8 -> go1b 45.6G (gen_go1b.sh 自带 RSS 看门狗@11.5G)
#   [2] 逐层 GPTQ sign 重写 L6..42 (v3_gptq1.sh all, 每层 SUMMARY 一行 = 逐层规律表)
# 进度: /tmp/v3_pipeline.log (阶段) + /tmp/gen_go1b.log (逐张量) + /tmp/v3_gptq1.log (逐层)
set -u
ROOT=/Users/fodelf/ds4-main
HF=$ROOT/hf/DeepSeek-V4-Flash-Base
TMPL=$ROOT/gguf/tmpl_hdr.gguf
OUT=$ROOT/gguf/ds4-go1b-v3.gguf
SCRIPTS=$ROOT/gguf-tools/go-onebit/scripts

echo "=== v3 pipeline start $(date) ==="
if [ ! -f "$TMPL" ]; then
  echo "[0] fetch template head (200MiB) $(date +%H:%M:%S)"
  curl -sL -r 0-209715199 \
    "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" \
    -o "$TMPL" || { echo "TEMPLATE-FETCH-FAILED"; exit 1; }
fi
ls -la "$TMPL"

if [ ! -f "$OUT" ]; then
  echo "[1] gen 1-bit base $(date +%H:%M:%S) (逐张量进度: /tmp/gen_go1b.log)"
  sh "$SCRIPTS/gen_go1b.sh" "$HF" "$TMPL" "$OUT" 6 || { echo "GEN-FAILED"; exit 1; }
fi
[ -f "$OUT" ] || { echo "GEN-NO-OUTPUT"; exit 1; }
ls -la "$OUT"

echo "[2] GPTQ sign rewrite L6..42 $(date +%H:%M:%S) (逐层: /tmp/v3_gptq1.log)"
sh "$SCRIPTS/v3_gptq1.sh" all
echo "=== v3 pipeline DONE $(date) ==="
