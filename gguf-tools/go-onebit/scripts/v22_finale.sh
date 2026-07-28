#!/bin/bash
# v22_finale.sh — 今夜收官段(2026-07-26): merge → DS4_VQ_DIR 起服 → 冒烟 → 编码基准+速度。
# 前置: 反修+rr 终判完成(夜链); 在 M4 驱动, 服务跑 M1。
set -uo pipefail
M1=192.168.1.2; M1ROOT=/Users/fodelf/ds4-main
LDIR=$M1ROOT/gguf/go-onebit/layers
OUT=$M1ROOT/gguf/go-onebit/ds4-code1b.gguf
PORT=8013
log(){ echo "[finale $(date +%H:%M:%S)] $*"; }

case "${1:-all}" in
merge)
  ssh $M1 "pgrep -x ds4quant_run" >/dev/null && { log "量化进程仍在, 拒 merge"; exit 3; }
  FREE=$(ssh $M1 "df -g /System/Volumes/Data | awk 'NR==2{print \$4}'")
  [ "$FREE" -ge 14 ] || { log "★盘闸: M1 ${FREE}G <14G(合一底座~12G)★"; exit 6; }
  log "merge: quant_layer merge 入口(骨架+偏移+注入+zchain 全套编排; 产物=合一 GGUF)"
  ssh $M1 "cd $M1ROOT && env DS4_VQ=1 DS4_GO2B_HOT=1 \
    DS4_GO2B_HOT_TABLE=$M1ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt \
    DS4_CORPUS=/tmp/rr_calib_prog_v1.ids DS4_NTOK=530 \
    ./quant_layer.sh merge" > /tmp/v22_merge.out 2>&1 || { log "merge rc=$? M1 日志尾:"; ssh $M1 "tail -8 /tmp/quant_all.log"; exit 2; }
  ssh $M1 "ls -la $M1ROOT/gguf/go-onebit/*.gguf | tail -2"
  ;;
serve)
  log "M1 起服(A3 流式 + VQ 直读 + 12G 看门狗)"
  ssh $M1 "pkill -f ds4-server; sleep 1; cd $M1ROOT && nohup env DS4_VQ_DIR=$LDIR \
    DS4_MEM_BUDGET_MB=11264 DS4_METAL_PREFILL_CHUNK=4 \
    ./ds4-server -m $OUT --metal --ctx 32768 --port $PORT > /tmp/v22_serve.log 2>&1 < /dev/null & sleep 3; tail -3 /tmp/v22_serve.log"
  ;;
smoke)
  log "冒烟: 1+1=? (24 token 短探针)"
  ssh $M1 "curl -s --noproxy '*' -m 300 -X POST http://127.0.0.1:$PORT/v1/messages \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"ds4\",\"max_tokens\":24,\"mode\":\"code\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=?\"}]}'" | head -c 400
  echo
  ;;
bench)
  log "编码基准: gen_coding_probe 三真实任务 + 计时"
  T0=$(date +%s)
  ssh $M1 "cd $M1ROOT && PORT=$PORT MAXTOK=500 TAG=v22 bash gguf-tools/go-onebit/scripts/gen_coding_probe.sh" 2>&1 | tail -3
  log "总耗时 $(( $(date +%s) - T0 ))s"
  ssh $M1 "ls $M1ROOT/gguf-tools/go-onebit/reports/gen_coding_v22_*.report | tail -1"
  ;;
all) "$0" merge && "$0" serve && sleep 20 && "$0" smoke && "$0" bench ;;
esac
