#!/bin/bash
# skel_variant_spark.sh — 骨架精度变体全链(spark 本机跑, 2026-09-05 夜, 用户令"全部 q2k 骨架 + 冠军 VQ,
# 明早给五指标和速度"):
#   ① 从 HF(Vision-Exp, 与冠军专家/锚同底座)抽指定精度的自产骨架(skel_from_hf.sh, $3 起透传量化旗标)
#   ② 与冠军 VQ 层件(champ86amp/layers, sweep3 交付态)合并成可部署 GGUF(merge_base86p.sh, 无路由偏置)
#   ③ 引擎路五指标: --score-ids wt2(2653 token) 对 FP 锚 anchor_wt2_s2653(Vision-Exp) 出 PPL/Σmin/KLD/RMS/top1
#   ④ 速度曲线(speed_champ_spark.sh: 探针 + ds4-bench 2048..8192)
# 为什么反修/sweep 不重跑: 放大器(zrec/zchain)按铁律以 FP 锚口径解(参考前向的骨架是 FP), 产物与
# 部署骨架精度无关 —— 重跑只会复现同一份文件(champ3 全链 2h11m + sweep3 10h), 复用冠军 zchain.bin。
# 为什么骨架从 Vision-Exp 抽: 09-05 夜实测 0731 与 Vision-Exp 的全部非专家权重逐张不同(embed/head/
# attn/compressor 18/18 DIFF), 而 r30_skeleton.gguf(08-17)是 0731 抽的 —— 现役冠军 ds4-champ86amp.gguf
# 是 0731 骨架 + Vision-Exp 专家的错位合并, 引擎路对锚 KLD 1.53 vs 参考尺 0.32 的鸿沟至少有这一份。
# 用法: skel_variant_spark.sh <标签> [量化旗标...]
#   例: skel_variant_spark.sh q2k --attention-proj q2_k --attention q2_k --shared q2_k --output q2_k --dense q2_k --embedding q2_k
#       skel_variant_spark.sh q8ve            # 量化器默认 = Q8 骨架(r30 同款精度, 但底座对上了)
# 产物: gguf/go-onebit/skel/<标签>_skeleton.gguf, gguf/ds4-champ86<标签>.gguf, vqhalf/champ86<标签>/{zchain.bin→冠军, metrics_wt2.txt, speed/}
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?标签}"; shift
SC="$ROOT/gguf-tools/scripts"; VQH="$ROOT/gguf/go-onebit/vqhalf"; R30="$ROOT/gguf/go-onebit/r30"; G7="$ROOT/gguf/go-onebit/g7"
HF="$ROOT/hf/DeepSeek-V4-Flash-Vision-Exp"
SKD="$ROOT/gguf/go-onebit/skel"; SKEL="$SKD/${TAG}_skeleton.gguf"
WS="champ86$TAG"; D="$VQH/$WS"; MDL="$ROOT/gguf/ds4-$WS.gguf"
CH="$VQH/champ86amp"
LOG(){ echo "[skelvar $TAG $(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$SKD" "$D"
[ -d "$HF" ] || { LOG "★HF 缺 $HF★"; exit 2; }
[ -s "$CH/zchain.bin" ] || { LOG "★冠军 zchain 缺★"; exit 2; }
[ "$(ls "$CH/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)" = 43 ] || { LOG "★冠军层件不齐★"; exit 2; }
[ -s "$R30/template_head.gguf" ] || { LOG "★模版缺(从 ds4-allq2.gguf 头 64 MiB 重建: head -c 67108864)★"; exit 2; }
BUSY=$(for p in ds4 ds4-bench ds4-server ds4quant_run zlayer vq_merge_v4 deepseek4-quantize; do pgrep -x "$p"; done)
[ -z "$BUSY" ] || { LOG "★机器非空: $BUSY★"; exit 3; }
FREE=$(df -BG --output=avail "$ROOT/gguf" | sed -n 2p | tr -dc 0-9)
[ "${FREE:-0}" -ge 120 ] || { LOG "★盘 ${FREE}G < 120G(骨架+合并 ~90G)★"; exit 4; }

if [ ! -s "$SKEL" ]; then
    LOG "① 骨架 ← $HF ($*)"
    # 阈值按 121G 机器给: 进程 RSS 上限 60G, 盘 free 地板 10G(脚本默认), 起跑盘闸 20G
    SKEL_HF="$HF" SKEL_TMPL="$R30/template_head.gguf" SKEL_WDOG_MB=60000 \
        bash "$SC/skel_from_hf.sh" "$SKEL" 20 "$@" || { LOG "★骨架失败★"; exit 5; }
else
    LOG "① 骨架已在 $(ls -l "$SKEL" | awk '{printf "%.2f GB", $5/1e9}'), 跳过"
fi
LOG "骨架 $(ls -l "$SKEL" | awk '{printf "%.2f GB", $5/1e9}')"

LOG "② 合并 骨架 + 冠军 43 层 VQ → $MDL"
M86_LAYERS="$CH/layers" M86_MDL="$MDL" M86_SKEL="$SKEL" M86_ZCH= M86_RB=/dev/null \
    bash "$SC/merge_base86p.sh" || { LOG "★合并失败★"; exit 6; }
LOG "模型 $(ls -l "$MDL" | awk '{printf "%.2f GB", $5/1e9}')"
ln -sfn "$CH/zchain.bin" "$D/zchain.bin"

LOG "③ 引擎路五指标 wt2 × FP 锚(Vision-Exp)"
( while true; do A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "${A:-0}" -lt 6000 ] && { echo "[wdog] avail ${A}MB ★杀★"; pkill -9 -x ds4; pkill -9 -x ds4-bench; break; }; sleep 5; done ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
timeout --foreground 3600 ./ds4 --cuda -m "$MDL" --zchain "$D/zchain.bin" --mem-budget-mb 110000 \
    --score-ids "$G7/wt2.ids" --score-out "$D/score_wt2.bin" > "$D/score_wt2.log" 2>&1 </dev/null \
    || { LOG "★score 失败(见 $D/score_wt2.log)★"; tail -3 "$D/score_wt2.log"; exit 7; }
gguf-tools/bench/anchor_metrics --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
    --student "$D/score_wt2.bin" 2>&1 | grep -v "^$" | tail -8 | tee "$D/metrics_wt2.txt"
kill $WD 2>/dev/null

LOG "④ 速度曲线"
bash "$SC/speed_champ_spark.sh" "$WS" "$WS" 8192 || LOG "★速度段失败★"
LOG "收官: 模型 $MDL, 指标 $D/metrics_wt2.txt, 速度 $D/speed/${WS}_zchain.csv"
LOG "SKELVAR_${TAG}_DONE"
