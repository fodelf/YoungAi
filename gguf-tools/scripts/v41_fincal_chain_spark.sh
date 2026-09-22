#!/bin/bash
# v41_fincal_chain_spark.sh — 金融域校准量化 → gr-only 反修 全链(2026-09-20 深夜, 用户令"按金融域量化, 再反修, 明早给五指标")。
#
# 只串既有入口, 数值全在各自的程序里:
#   S0 取料   v41_amp_run --dump-calib: 裸 q4k-12 上跑金融【量化份】vqhalf_q 8192 行, 40 层同趟落 calib_Lnn.bin(3.4 GB)
#   S1 冒烟   v41_quantize --layers 0:1 --calib --calib-ab: 一层 384 专家各做两遍(校准指派 / 平权指派), 同一把加权尺上
#             校准必须赢平权 —— 机制审计(列权有没有真进指派与训练); 不赢就停, 不许带着假机制跑 66 分钟
#   B1 全量   v41_quantize_spark.sh --calib … --judge-set all: 40 层 + common, 文件态三尺(金融 j 8192 / 八域 j 8192 / wt2 512),
#             对照裸 q4k-12 的文件态读数(金融 69.86% / wt2 76.95%, fable5 09-20 晚)
#   G  转 GGUF  v41_to_gguf(靶 113.68 ± 0.05 GB; 配方 KV 会带 cal=…)
#   C  引擎对拍 v41_engine_parity_spark.sh --gguf <新> wt2 512: 引擎 vs 文件态 Python 学生, KL ≤ 0.07 才算接线对
#   ②  v41_gronly_gate.sh <新>: 解 gr-only(金融拟合份 vqhalf_a) + 金融五指标 + wt2 守门 + 打转尺
# 停车规则: 任一段非 0 就停在那一段(半成品往下传只出假数); 每段开头打 [chain] 时间戳, 尾行 CHAIN_EXIT <rc>。
# 看门狗: S0 段本脚本自带(available < 8 GB 杀 v41_amp_run); 其余段各脚本自带。
# 用法(spark 本机, nohup): v41_fincal_chain_spark.sh [--from S0|S1|B1|G|C|GATE] [--ef β1,β2]
#   --ef: 级 2 误差反馈(v41_quantize --calib-ef), 产物名带 fincal2 与级 1 并存; 取料(S0)与级 1 共用同一份
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM="S0"; EFB=""; ALPHA=""
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM="${2:?--from 要段名}"; shift 2;;
  --ef) EFB="${2:?--ef 要 β 表}"; shift 2;;
  --alpha) ALPHA="${2:?--alpha 要 [0,1] 的数}"; shift 2;;   # 列权指数(01:41 实撞: 全额列权 wt2 −2.15 pp), 产物名带 a<α>
  *) echo "★不认识的参数 $1★"; exit 2;;
esac; done
HF="hf/DeepSeek-V4.1-Flash"
BASE="gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-q4k-mtpnative.gguf"           # 取料用的裸 ①(现役 q4k-12)
QIDS="gguf/go-onebit/vqfin41/vqhalf_q.ids"                                 # 金融【量化份】(a 拟合 / j 判决不动)
CAL="gguf/v41/calib-fin-q8192"
TAG="fincal"; QEXTRA=()
[ -n "$EFB" ] && { TAG="fincal2"; QEXTRA=(--calib-ef "$EFB"); }
[ -n "$ALPHA" ] && { TAG="$TAG-a${ALPHA//./}"; QEXTRA+=(--calib-alpha "$ALPHA"); }
SMOKE="gguf/v41/smoke-$TAG-L0"
OUTDIR="gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-q4k-$TAG"
GG="gguf/v41/DeepSeek-V4.1-Flash-vq8x4096-q4k-$TAG-mtpnative.gguf"
WT2="gguf/go-onebit/g7/wt2.ids"
M(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "CHAIN_EXIT 1"; exit 1; }
# 段序比较: 段 $1 在 --from 指定的段之后(含)才跑
stage_ge(){ local order="S0 S1 B1 G C GATE"; local a=${order%%$1*}; local b=${order%%$FROM*}; [ ${#a} -ge ${#b} ]; }
for f in "$BASE" "$QIDS" "$QIDS.layout" "$WT2" "$HF/config.json"; do [ -e "$f" ] || die "缺 $f"; done

if stage_ge S0; then
  M "S0 取料: 裸 $(basename "$BASE") × $QIDS 8192 → $CAL"
  make -C gguf-tools v41_amp_run >/tmp/fincal_make_amp.log 2>&1 || die "v41_amp_run 编译失败(见 /tmp/fincal_make_amp.log)"
  mkdir -p "$CAL"
  ( while sleep 20; do av=$(free -g | awk '/^内存|^Mem/{print $7}'); if [ "${av:-99}" -lt 8 ]; then echo "★看门狗: available ${av}G < 8G, 停车★"; pkill -f "amp/v41_amp_ru[n]"; fi; done ) & WD=$!
  gguf-tools/amp/v41_amp_run "$BASE" "$HF" "$QIDS" 8192 "$CAL" --dump-calib --mem-budget-mb 40000 --weight-cache-mb 88000 2>&1 \
      | tee /tmp/fincal_s0.log | grep --line-buffered -E "校准|引擎|★|error|Error|失败"
  rc=${PIPESTATUS[0]}; kill $WD 2>/dev/null
  [ "$rc" = 0 ] && [ -s "$CAL/calib.txt" ] && [ -s "$CAL/calib_L39.bin" ] || die "S0 取料失败 rc=$rc"
fi

if stage_ge S1; then
  M "S1 冒烟: L0 校准指派 vs 平权指派(同一把加权尺)"
  make -C gguf-tools v41_quantize >/tmp/fincal_make_q.log 2>&1 || die "v41_quantize 编译失败(见 /tmp/fincal_make_q.log)"
  rm -rf "$SMOKE"; mkdir -p "$SMOKE"
  gguf-tools/quantize/v41_quantize "$HF" "$SMOKE" --layers 0:1 --no-common --vq-nc 4096 --skel q4k --calib "$CAL" --calib-ab "${QEXTRA[@]}" --force 2>&1 \
      | tee /tmp/fincal_smoke.log | grep --line-buffered -E "校准|分片|索引|EF|★|失败"
  [ "${PIPESTATUS[0]}" = 0 ] || die "S1 量化器失败"
  grep -q "加权尺上校准赢平权" /tmp/fincal_smoke.log || die "S1 机制审计没过: 加权尺上校准没赢平权(列权没进指派/训练)"
  # 平权 cos 的先验: 09-20 nocal 那趟 L00 是 0.9105; 等于 0.9105 = 列权没生效。校准指派在平权尺上【会明显掉】——
  # 09-21 00:10 实测 0.8861(加权 cos 0.9126 → 0.9322): 把误差挪到激活小的列就是这样, 不是坏; 低于 0.85 才当坏了
  pc=$(grep "^\[分片\] model-layer00" /tmp/fincal_smoke.log | sed -E 's/.*参数 cos ([0-9.]+).*/\1/')
  # 级 2(误差反馈按 H 而不是按平权 MSE 选索引)平权 cos 还会再掉, 下限放到 0.80 —— 真坏的形状是 cos 掉到 0.7 以下(09-11 x 错位那种)
  awk -v c="$pc" 'BEGIN{exit !(c>=0.80 && c<0.9105)}' || die "S1 平权 cos $pc 不在 [0.80, 0.9105)"
  rm -rf "$SMOKE"
fi

if stage_ge B1; then
  M "B1 全量: 40 层 + common, --calib $CAL, 文件态三尺"
  bash gguf-tools/scripts/v41_quantize_spark.sh --out "$OUTDIR" --vq-nc 4096 --skel q4k --mtp-vq --calib "$CAL" "${QEXTRA[@]}" --judge-set all || die "B1 失败"
fi

if stage_ge G; then
  M "G 转 GGUF → $GG"
  make -C gguf-tools v41_to_gguf >/tmp/fincal_make_g.log 2>&1 || die "v41_to_gguf 编译失败(见 /tmp/fincal_make_g.log)"
  gguf-tools/quantize/v41_to_gguf "$OUTDIR" gguf/v41/consts/engram_consts.bin gguf/v41/consts/tokenizer_consts.bin "$GG" "$HF" 2>&1 | tail -5
  [ "${PIPESTATUS[0]}" = 0 ] && [ -s "$GG" ] || die "G 转换失败"
  sz=$(stat -c %s "$GG"); M "GGUF $sz B = $(awk -v s="$sz" 'BEGIN{printf "%.3f", s/1e9}') GB(靶 113.68 ± 0.05)"
  ./ds4 -m "$GG" --cuda --inspect 2>&1 | grep -E "张量|GiB|校验|失败|★" | head -8
fi

if stage_ge C; then
  M "C 引擎对拍 wt2 512: 引擎 vs 文件态 Python 学生"
  STU=$(ls -t gguf/v41judge/stu_g7_wt2_n512_file_"$(basename "$OUTDIR")"*.bin 2>/dev/null | head -1)
  [ -s "$STU" ] || die "C 缺文件态 wt2 学生 logits(B1 的 --judge-set all 该落它)"
  # --engram: 文件态学生带 engram, 引擎也要带, 否则比的是两个不同的模型(09-20 C 段同口径)
  bash gguf-tools/scripts/v41_engine_parity_spark.sh "$WT2" 512 "$STU" --engram --gguf "$GG" 2>&1 | grep -E "KLD|Same|PPL|★" | head -12
fi

if stage_ge GATE; then
  M "② gr-only 三道门(解 + 金融五指标 + wt2 守门 + 打转尺)"
  bash gguf-tools/scripts/v41_gronly_gate.sh "$GG" || die "三道门失败"
fi
M "全链收工"; echo "CHAIN_EXIT 0"
