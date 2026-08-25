#!/bin/bash
# r29_chain.sh — R29 无人值守串联(2026-08-01 用户令"明天早上量化好模型, 我看代码测试报告")。
#
# 量化已在跑(单独起的进程)。本脚本等它结束, 然后自动串: 换脚本 → 反修 ‖ 骨架自产 → 合并。
# 双机基准留给人工触发(要占两台机器的 GPU, 且 M4 得先摆渡 28 GiB 模型)。
#
# 为什么反修和骨架并行: 两者互不依赖(反修吃层文件+锚, 骨架吃 HF 原始), M1 Pro 有 8 性能核。
# 线程分配 反修 6 / 骨架 2: 反修是关键路径(43 层, 数小时), 骨架只要 30-60 分钟就退场,
# 之后反修独占。反之(反修 4/骨架 3)会让关键路径全程慢 33%。
#
# 每一步都写 /tmp/r29_chain.log, 失败即停(不吞错继续跑下一步 —— 那只会浪费几小时)。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
LOG(){ echo "[chain $(date +%H:%M:%S)] $*" | tee -a /tmp/r29_chain.log >&2; }

LOG "等量化结束…"
while pgrep -x ds4quant_run >/dev/null; do sleep 60; done
LOG "量化进程已退出"

# 量化是否真跑完 43 层
N=$(grep -ac "★体积★" /tmp/r29_quant.log 2>/dev/null || echo 0)
TOT=$(awk '{s+=$2} END{printf "%.3f",s/1073741824}' "$ROOT/gguf/go-onebit/r29/full/layers/manifest.txt" 2>/dev/null || echo 0)
LOG "落盘层数 $N/43, manifest 合计 ${TOT} GiB"
[ "$N" -eq 43 ] || { LOG "★量化未跑满 43 层 — 停, 需人工看 /tmp/r29_quant.log★"; exit 2; }

# ★换用带"锚路由×粗筛行映射"的新量化器二进制(2026-08-02)★
# 量化跑的是旧二进制(锚路由会硬关反修粗筛 ⇒ 反修要 7 天)。新版加了 g_anc_rowmap 行映射,
# 让锚路由与粗筛共存。量化已退出, 此刻替换是安全的。
NEWQ="$ROOT/gguf-tools/go-onebit/quant/ds4quant_run.new"
if [ -x "$NEWQ" ]; then
    mv -f "$NEWQ" "$ROOT/gguf-tools/go-onebit/quant/ds4quant_run"
    LOG "已换用新量化器(锚路由 × 粗筛行映射)"
else
    LOG "★新量化器不在 — 反修会因锚路由关掉粗筛而极慢, 停★"; exit 6
fi

# ★禁覆盖运行中脚本(铁律)★ 量化已退出, 现在换成带自产骨架合并逻辑的新版
if [ -f /tmp/r29_campaign_new.sh ]; then
    cp /tmp/r29_campaign_new.sh "$SC/r29_campaign.sh" && chmod +x "$SC/r29_campaign.sh"
    LOG "已换用新版 campaign(自产骨架合并)"
fi

# 骨架自产: 后台并行(它只吃 HF 原始, 与反修无依赖)
LOG "起骨架自产(后台, 2 线程)"
nohup bash "$SC/skel_from_hf.sh" "$ROOT/gguf/go-onebit/r29/r29_skeleton.gguf" 2 \
    > /tmp/r29_skel.log 2>&1 &
SKPID=$!

LOG "起反修(前台, 6 线程)"
DS4_THREADS=6 bash "$SC/r29_campaign.sh" backfit > /tmp/r29_backfit.log 2>&1
BRC=$?
if grep -aq "BACKFIT_SCREEN 关" /tmp/r29_backfit.log 2>/dev/null; then
    LOG "★粗筛仍被关掉 — 行映射没生效, 反修会极慢★"
fi
LOG "反修 rc=$BRC"
[ $BRC -eq 0 ] || { LOG "★反修失败 — 停★"; exit 3; }

LOG "等骨架自产收尾…"
wait $SKPID 2>/dev/null
SKEL="$ROOT/gguf/go-onebit/r29/r29_skeleton.gguf"
[ -f "$SKEL" ] || { LOG "★自产骨架不在 — 合并会拒跑(绝不回退冠军骨架)★"; exit 4; }
LOG "自产骨架 $(ls -l "$SKEL" | awk '{printf "%.2f GiB",$5/1073741824}')"

# ★合并前腾盘(2026-08-02 算账)★ 合并要同时存在 层文件 19.8 + 骨架 8.2 + 输出 28 = 56 GiB,
# 而 M1 可用约 56 GiB —— 零余量。反修已完成 ⇒ ckpt(43×112 MB ≈ 4.8 GiB)不再需要(它只用于
# 断点续跑), 旧 g7 产物(out/out_base ≈ 1.4 GiB)也是历史轮次的。删掉换 ~6 GiB 余量。
BEFORE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
rm -rf "$ROOT/gguf/go-onebit/r29/full/ckpt"
rm -rf "$ROOT/gguf/go-onebit/g7/out" "$ROOT/gguf/go-onebit/g7/out_base"
# anchor_rrh(1.1 GiB)是 rr 判决用的备用锚, 可从 ids 重建; 反修必需的 anchor_v5mini(6.5 GiB)保留。
# 实测账: 不删这个, 合并峰值会差约 0.5 GiB 写不下。
rm -f "$ROOT/gguf/go-onebit/g7/ds4quant_anchor_rrh.bin"
AFTER=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
LOG "合并前腾盘: free ${BEFORE}G → ${AFTER}G"
[ "$AFTER" -ge 30 ] || { LOG "★free ${AFTER}G < 30G, 合并会写不下 28 GiB — 停★"; exit 7; }

LOG "起合并"
bash "$SC/r29_campaign.sh" merge > /tmp/r29_merge_stage.log 2>&1
MRC=$?
LOG "合并 rc=$MRC"
[ $MRC -eq 0 ] || { LOG "★合并失败 — 停★"; tail -5 /tmp/r29_merge_stage.log >&2; exit 5; }

MDL="$ROOT/gguf/go-onebit/ds4-r29.gguf"
LOG "★R29 模型就绪 $(ls -l "$MDL" | awk '{printf "%.2f GiB",$5/1073741824}')★"
LOG "下一步(人工): r29_bench.sh all  — 摆渡 M4 + 双机 lane + 40 题基准"
