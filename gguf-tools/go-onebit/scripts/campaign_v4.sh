#!/bin/bash
# campaign_v4.sh — v4 量化战役点火(2026-07-27, 在 M1 上跑)。
# 五修合一(fable5 审计全表): ①v4 语料(fit/held 显式分区, DS4_NFIT 钉边界) ②NTOK=1340
# ③可变 H_L 热表(hot_v4.txt, 损伤谱水填充+L42 保底 128, 等槽等体积) ④DS4_CALIB_FULLSET=1
# (g_r/GPTQ-H 全集喂入, 修每专家~9行饿死) ⑤emit-only+白名单 env(反修族全关, 防 07-26
# env 泄漏/existence-gate 事故链)。
# 用法(M1): campaign_v4.sh fast|emit
set -uo pipefail
ROOT=/Users/fodelf/ds4-main
MODE="${1:?fast|emit|backfit}"
cd "$ROOT"

FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
# 盘闸按模式分档: emit 要写 51.5G 层文件→58G; backfit/fast 原地改+日志→12G(v21 同款)
NEED=58; { [ "$MODE" = backfit ] || [ "$MODE" = fast ]; } && NEED=12
[ "$FREE" -ge "$NEED" ] || { echo "★盘闸: free ${FREE}G <${NEED}G(mode=$MODE), 停★" >&2; exit 6; }
pgrep -f ds4quant_run >/dev/null && { echo "★已有量化进程, 拒并发★" >&2; exit 3; }
[ -f /tmp/rr_calib_prog_v4.ids ] || { echo "语料 ids 缺" >&2; exit 2; }
[ -f gguf-tools/go-onebit/corpus/hot_v4.txt ] || { echo "热表缺" >&2; exit 2; }
NTOK=$(grep NTOK /tmp/rr_calib_prog_v4.meta | cut -d= -f2)
NFIT=$(grep NFIT /tmp/rr_calib_prog_v4.meta | cut -d= -f2)
echo "[v4] mode=$MODE NTOK=$NTOK NFIT=$NFIT free=${FREE}G" >&2

# 量化器强制重编(DS4_CALIB_FULLSET 改动)
( cd gguf-tools/go-onebit/quant && cc -O3 -Wall -Wextra -Wno-unused-parameter -lm \
    -framework Accelerate -I"$ROOT" -o ./ds4quant_run ds4quant_run.c -lpthread ) || exit 4
cp gguf-tools/go-onebit/quant/ds4quant_run ./ds4quant_run 2>/dev/null || true
echo "[v4] 量化器已重编(fullset 版)" >&2

[ "$MODE" = emit ] && find "$ROOT/gguf/go-onebit/layers" -name 'dql_*' -delete 2>/dev/null || true

# 白名单 env(env -i 全净化, 只带必需项; 反修族显式关死)
ENVS=(
  HOME="$HOME" PATH="$PATH" USER="$USER"
  DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-Base"
  DS4_VQ=1 DS4_TGT_ALPHA=1.0
  DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/hot_v4.txt"
  DS4_CORPUS=/tmp/rr_calib_prog_v4.ids DS4_NTOK="$NTOK" DS4_NFIT="$NFIT"
  DS4_CALIB_FULLSET=1 DS4_THREADS=8
  DS4_SKIP_MERGE=1 DS4_BACKFIT_INCR=0 DS4_GSWEEP=0 DS4_BWD=0
)
if [ "$MODE" = fast ]; then
  exec env -i "${ENVS[@]}" ./quant_layer.sh fast
elif [ "$MODE" = backfit ]; then
  # ★反修唯一一相(2026-07-27 深夜, 用户令: 杀重跑带路由修正)★: sweep+BWD+TERM_MAXP
  # + DS4_ANCHOR_ROUTE=1(锚路由反修: FP 专家选择+权重贯穿前向/校准/判据, 禁翻转噪声;
  # 粗筛自动关=判据回全量; rr 终判在 quant_layer.sh 调用点 env -u 剥离, 保学生路由诚实口径)。
  # GSWEEP=0(12G OOM 隐患在 GSWEEP 全缓存路径; 第二相已取消)。v21 同款外挂足迹看门狗 10.5G。
  env -i "${ENVS[@]}" DS4_BF_JUSTIFIED=1 DS4_BF_TERM_MAXP=1 DS4_BWD=1 DS4_GSWEEP=0       DS4_ANCHOR_ROUTE=1 DS4_BF_MEMGB=8 ./quant_layer.sh backfit &
  QLP=$!
  ( while kill -0 $QLP 2>/dev/null; do
      BP=$(pgrep -nf 'ds4quant_run' 2>/dev/null)
      if [ -n "$BP" ]; then
        FP=$(ps -o rss= -p "$BP" 2>/dev/null | awk '{print $1/1048576}')
        [ -n "$FP" ] && awk -v f="$FP" 'BEGIN{exit !(f>10.5)}' &&           { echo "[v4bf] ★足迹 ${FP}G>10.5G 杀★" >&2; kill -9 "$BP" $QLP; exit 9; }
      fi
      sleep 5
    done ) &
  wait $QLP
else
  exec env -i "${ENVS[@]}" ./quant_layer.sh
fi
