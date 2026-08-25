#!/bin/bash
# 官方 q2 GGUF 下载 + 同尺对拍一体接力 (在 M1 上跑, repo=/Users/fodelf/ds4-main)
# 单流断点续传: 上游封顶 ~4MB/s, 并发/双机实测不叠加 (2026-08-14 探针);
# --speed-limit/--speed-time = 僵死熔断 (上一版 curl 无熔断, TCP 僵连接挂 6 天 0 进度)。
set -u
REPO="${REPO:-/Users/fodelf/ds4-main}"
G="$REPO/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
TOT=86720111488
U="https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
R30="$REPO/gguf/go-onebit/r30"
IDS="$REPO/gguf/go-onebit/g7/rr_hard.ids"

while :; do
  SZ=$(stat -f %z "$G" 2>/dev/null || echo 0)
  [ "$SZ" = "$TOT" ] && break
  if [ "$SZ" -gt "$TOT" ]; then echo "Q2DL_OVERSIZE $SZ"; exit 1; fi
  echo "$(date +%F_%T) RESUME at ${SZ}B ($(echo "scale=1;$SZ/1000000000" | bc)GB)"
  curl -sL -C - --speed-limit 102400 --speed-time 45 -o "$G" "$U"
  echo "$(date +%F_%T) curl exit=$? size=$(stat -f %z "$G" 2>/dev/null || echo 0)"
  sleep 3
done
echo "Q2DL_DONE $(stat -f %z "$G")B"

cd "$REPO"
./ds4 -m "$G" --metal --score-ids "$IDS" --score-out /tmp/openq2_rrhard.bin 2>&1 | tail -3
echo Q2SCORE_DONE
python3 gguf-tools/go-onebit/scripts/anchor_metrics.py \
  --ref "$R30/anchor_rr_hard_s1716.bin" --ids "$IDS" \
  --student /tmp/openq2_rrhard.bin --tail 10
echo SHOWDOWN_DONE
