#!/bin/bash
# v21_campaign.sh — v2.2-VQ 全量 43 层战役(2026-07-25): VQ 码本代(DS4_VQ=1 + DS4_TGT_ALPHA=1.0)。
# 冒烟2全绿: VQ_GATE PASS(hot 0.9576/cold 0.8109), held −6.4%, 7min/层, parity 过。
# ★全 M1 单机(盘账反转: M1 66G vs M4 5G; 前代对在 M4 受交接铁律保护)★
# 配方=07-24 战役同款(emit 全层 BACKFIT_INCR=0 → BF_ONLY 反修 → verdict), 差异仅 +DS4_TGT_ALPHA=1.0。
# 产物: M1 gguf/go-onebit/layers/(规范位置) + go2b 侧车; merge 延后(DS4_SKIP_MERGE=1)。
# 用法(在 M1): v21_campaign.sh emit|backfit|status
set -uo pipefail
M1ROOT=/Users/fodelf/ds4-main
CORPUS=/tmp/rr_calib_prog_v1.ids
HOTTAB=$M1ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt
log(){ echo "[v21 $(date +%H:%M:%S)] $*" >&2; }

case "${1:-emit}" in
emit)
  FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  [ "$FREE" -ge 58 ] || { log "★盘闸: M1 free ${FREE}G <58G(层51.5G+余量), 停★"; exit 6; }
  pgrep -f ds4quant_run >/dev/null && { log "★已有量化进程, 拒绝并发★"; exit 3; }
  [ -f "$CORPUS" ] || { log "语料缺"; exit 2; }
  [ -f "$HOTTAB" ] || { log "热表缺"; exit 2; }
  find "$M1ROOT/gguf/go-onebit/layers" -name 'dql_*' -delete 2>/dev/null || true
  log "起飞 emit: 43层 S=530 α=1.0 (无blk); 产物 gguf/go-onebit/layers/"
  cd "$M1ROOT"
  export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$HOTTAB" \
         DS4_CORPUS="$CORPUS" DS4_NTOK=530 DS4_SKIP_MERGE=1 \
         DS4_BACKFIT_INCR=0 DS4_GSWEEP=0
  exec ./quant_layer.sh
  ;;
backfit)
  # ★满档合并(用户令 07-25 夜, 覆盖 07-22 反修族默认关): 终局收敛 sweep + 终端反调 BWD
  #   + 终端 TERM_MAXP + 回扫 GSWEEP 全部并入本段一次执行, 榨到无落地为止★
  # 前置: bfsmoke 段先过(GSWEEP 全缓存路径有已知 12G OOM 隐患, 3 层实测 footprint 后才许全量)
  FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  [ "$FREE" -ge 12 ] || { log "★盘闸 <12G 停★"; exit 6; }
  N=$(ls "$M1ROOT"/gguf/go-onebit/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
  [ "$N" = 43 ] || { log "层文件 $N/43 不齐, 拒 backfit"; exit 2; }
  [ -f /tmp/v22_bfsmoke.pass ] || { log "★bfsmoke 未过, 先跑 $0 bfsmoke★"; exit 7; }
  log "起飞满档反修: BF_ONLY+终局sweep+BWD+TERM_MAXP=1+GSWEEP=3 (每层反修日志原生, 外挂足迹看门狗10.5G)"
  cd "$M1ROOT"
  export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$HOTTAB" \
         DS4_CORPUS="$CORPUS" DS4_NTOK=530 DS4_SKIP_MERGE=1 DS4_BF_MEMGB=8 \
         DS4_BF_TERM_MAXP=1 DS4_BWD=1 DS4_GSWEEP=3 DS4_BF_JUSTIFIED=1
  ./quant_layer.sh backfit &
  QLP=$!
  ( while kill -0 $QLP 2>/dev/null; do
      BP=$(pgrep -nf 'ds4quant_run' 2>/dev/null)
      if [ -n "$BP" ]; then
        FP=$(ps -o rss= -p "$BP" 2>/dev/null | awk '{print $1/1048576}')
        [ -n "$FP" ] && awk -v f="$FP" 'BEGIN{exit !(f>10.5)}' && { echo "[v21] ★反修足迹 ${FP}G>10.5G 杀(GSWEEP全缓存OOM保护)★" >&2; kill -9 "$BP" $QLP; exit 9; }
      fi
      sleep 5
    done ) &
  wait $QLP
  ;;
bfsmoke)
  # 3 层满档反修内存冒烟: 拷完成层到隔离目录, DS4_NL=3 直调二进制, 外挂 footprint 看门狗(10.5G 杀)
  SM=$M1ROOT/gguf/go-onebit/bfsmoke; rm -rf "$SM"; mkdir -p "$SM"
  for L in 00 01 02; do
    cp "$M1ROOT/gguf/go-onebit/layers/dql_L$L.bin" "$SM/" || { log "L$L dql 缺"; exit 2; }
    cp "$M1ROOT/gguf/go-onebit/layers/dql_vq_L$L.bin" "$SM/" 2>/dev/null || true
    cp "$M1ROOT/gguf/go-onebit/layers/opt_L$L.bin" "$SM/" 2>/dev/null || true
  done
  log "bfsmoke 起跑(3层满档 sweep, footprint 看门狗 10.5G)"
  cd "$M1ROOT/gguf-tools/go-onebit/quant"
  ( export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$HOTTAB" \
           DS4_ANCHOR=/tmp/g2_anchor_L3.bin DS4_NL=3 DS4_LCFG=g DS4_SIGNREF_MU=10 DS4_COADAPT=1 \
           DS4_BF_ONLY=1 DS4_BF_MEMGB=8 DS4_BF_TERM_MAXP=1 DS4_BWD_FINAL=1 DS4_GSWEEP=3 \
           DS4_LAYER_DIR="$SM" DS4_ZFILE="$SM/zfile.bin" DS4_ZCHAIN="$SM/zchain.bin" DS4_THREADS=6
    "${BFBIN:-./ds4quant_run}" "$CORPUS" 530 ) >"$SM/all.out" 2>"$SM/run.log" &
  QP=$!
  ( while kill -0 $QP 2>/dev/null; do
      FP=$(ps -o rss= -p $QP 2>/dev/null | awk '{print $1/1048576}')
      [ -n "$FP" ] && awk -v f="$FP" 'BEGIN{exit !(f>10.5)}' && { log "★足迹 ${FP}G>10.5G 杀★"; kill -9 $QP; exit 9; }
      sleep 5
    done ) &
  wait $QP; RC=$?
  if [ $RC -eq 0 ]; then touch /tmp/v22_bfsmoke.pass; log "★bfsmoke PASS → 可跑满档 backfit★"
  else log "bfsmoke rc=$RC 见 $SM/run.log"; fi
  ;;
status)
  N=$(ls "$M1ROOT"/gguf/go-onebit/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
  echo "layers=$N/43 free=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')G"
  tail -2 /tmp/quant_all.log 2>/dev/null | head -c 200
  ;;
esac
