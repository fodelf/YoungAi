#!/bin/bash
# r86_champ_backfit.sh — 超冠(573b7f5)反修原样, 应用于 r86 量化产物(2026-08-08 用户令
# "原样还原设计和代码")。env 链逐行抄自 573b7f5 campaign_v4.sh backfit 段:
#   锚路由(ANCHOR_ROUTE=1) + BF_ONLY=1(超冠 quant_layer L197 原样: SEARCH 跳过,
#   op 链回放推进→终局收敛 sweep) + 全量判据
#   + TERM_MAXP=1 + BWD=1 + JUSTIFIED=1 + GSWEEP=0 + env -i 白名单净化 + 10.5G 足迹看门狗。
# 现场项(路径非设计): layers=r30/full/layers, 语料=v5mini(冻结令), HF=0731, 热表=top149。
# 量化器=ds4quant_run.champ(超冠反修语义还原版, diff 可审)。
set -uo pipefail
ROOT=/Users/fodelf/ds4-main
cd "$ROOT"
OUTF=$ROOT/gguf/go-onebit/r30/full
QBIN=$ROOT/gguf-tools/go-onebit/quant/ds4quant_run.champ
FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
[ "$FREE" -ge 12 ] || { echo "★盘闸: free ${FREE}G <12G 停★" >&2; exit 6; }
pgrep -f ds4quant_run >/dev/null && { echo "★已有量化进程, 拒并发★" >&2; exit 3; }
N=$(ls "$OUTF"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
[ "$N" = 43 ] || { echo "★层文件 $N/43 不齐★" >&2; exit 2; }

# 白名单 env(campaign_v4 同款全净化; 反修族按超冠 backfit 相配置)
ENVS=(
  HOME="$HOME" PATH="$PATH" USER="$USER"
  DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
  DS4_VQ=1 DS4_TGT_ALPHA=1.0
  DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top149.txt"
  DS4_ANCHOR="$ROOT/gguf/go-onebit/r30/anchor_r30_s1716.bin"
  DS4_CALIB_FULLSET=1 DS4_THREADS=8
  DS4_LAYER_DIR="$OUTF/layers"
  DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin"
  DS4_COADAPT=1 DS4_LZ="${DS4_LZ:-32}" DS4_LZ_LAMBDA="${DS4_LZ_LAMBDA:-1.0}"
  DS4_SKIP_MERGE=1 DS4_BACKFIT_INCR=0
)
echo "[champ-bf] 反修起跑(锚路由+全量判据+★z^L 隐变量 rank<=${DS4_LZ:-32}★, 10.5G 看门狗)" >&2
env -i "${ENVS[@]}" DS4_BF_ONLY=1 DS4_BF_JUSTIFIED=1 DS4_BF_TERM_MAXP=1 DS4_BWD=1 DS4_GSWEEP=0 \
    DS4_ANCHOR_ROUTE=1 DS4_BF_MEMGB=8 \
    "$QBIN" "$ROOT/gguf/go-onebit/g7/rr_calib_prog_v5mini.ids" 305 &   # 超冠满档 NTOK=305 原样(quant_layer 当年值)
QLP=$!
( while kill -0 $QLP 2>/dev/null; do
    BP=$(pgrep -nf 'ds4quant_run' 2>/dev/null)
    if [ -n "$BP" ]; then
      FP=$(ps -o rss= -p "$BP" 2>/dev/null | awk '{print $1/1048576}')
      [ -n "$FP" ] && awk -v f="$FP" 'BEGIN{exit !(f>10.5)}' && \
        { echo "[champ-bf] ★足迹 ${FP}G>10.5G 杀★" >&2; kill -9 "$BP" $QLP; exit 9; }
    fi
    sleep 5
  done ) &
wait $QLP
echo "[champ-bf] 反修 rc=$?" >&2
