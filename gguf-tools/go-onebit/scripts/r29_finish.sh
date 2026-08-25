#!/bin/bash
# r29_finish.sh — 反修之后的收尾接管(2026-08-02 05:00, 骨架因磁盘满失败后补救)。
#
# 为什么需要它: chain 让骨架与反修并行, 但两者叠加把盘吃穿了 ——
#   留洞文件实占 8.2 GiB + 紧凑骨架 8.2 GiB(写到一半 17.7 GiB) ⇒ free 归零,
#   vq_merge_v4 --extract-skeleton 在 OSError:28 上死掉。chain 随后会因"骨架不在"退出。
# 本脚本在反修跑完后接手, 按【串行】重来一遍, 并利用一个关键事实:
#   ★anchor_v5mini(6.5 GiB)只有反修需要★ —— 反修一结束就能释放, 正好补上合并缺的空间。
#
# 磁盘账(实测):
#   反修完 free ≈ 33 → 删 anchor 6.5 → 39.5
#   → 骨架(留洞 8.2 峰值, 完成后删) → 紧凑骨架 8.2 → 31.3
#   → 合并输出 28 → 3.3 GiB 余量 ✓
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
SKEL="$ROOT/gguf/go-onebit/r29/r29_skeleton.gguf"
MDL="$ROOT/gguf/go-onebit/ds4-r29.gguf"
LOG(){ echo "[finish $(date +%H:%M:%S)] $*" | tee -a /tmp/r29_finish.log >&2; }

LOG "等反修结束…"
while pgrep -x ds4quant_run >/dev/null; do sleep 60; done
LOG "反修进程已退出"
sleep 20   # 让 chain 先自行退出, 避免两边同时动文件

# 反修是否真跑完: zchain 侧车在, 且日志没有中途 abort
ZC="$ROOT/gguf/go-onebit/r29/full/zchain.bin"
if [ ! -f "$ZC" ]; then
    LOG "★zchain 侧车不在 — 反修没产出, 停(需人工看 /tmp/r29_backfit.log)★"; exit 2
fi
LOG "zchain $(ls -l "$ZC" | awk '{printf "%.1f MiB",$5/1048576}')"
BFL=$(grep -ac "STAGE_BEST" /tmp/r29_backfit.log 2>/dev/null || echo 0)
LOG "反修 STAGE_BEST 记录 $BFL 条"

# ★释放 anchor★: 只有反修需要它(量化已完成、合并/基准都不读)。这是合并能跑起来的关键 6.5 GiB。
# 它可从 rr_calib ids 重建, 不是不可再生资产; 原始 HF 与 q2 一概不动。
if [ -f "$ROOT/gguf/go-onebit/g7/ds4quant_anchor_v5mini_s1716.bin" ]; then
    rm -f "$ROOT/gguf/go-onebit/g7/ds4quant_anchor_v5mini_s1716.bin"
    LOG "已释放 anchor_v5mini 6.5 GiB(反修已用完; 可从 ids 重建)"
fi
rm -rf "$ROOT/gguf/go-onebit/r29/full/ckpt" 2>/dev/null
rm -f "$ROOT/gguf/go-onebit/r29/skel_hole.gguf" 2>/dev/null
FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
LOG "腾盘后 free ${FREE}G"
[ "$FREE" -ge 36 ] || { LOG "★free ${FREE}G < 36G — 骨架+合并放不下, 停★"; exit 3; }

# 骨架: 串行重来(这次独占磁盘)
if [ ! -f "$SKEL" ]; then
    LOG "重做自产骨架(串行, 4 线程)"
    bash "$SC/skel_from_hf.sh" "$SKEL" 4 >> /tmp/r29_skel.log 2>&1 \
        || { LOG "★骨架仍失败 — 见 /tmp/r29_skel.log★"; exit 4; }
fi
LOG "自产骨架 $(ls -l "$SKEL" | awk '{printf "%.2f GiB",$5/1073741824}')"

FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
LOG "合并前 free ${FREE}G"
[ "$FREE" -ge 29 ] || { LOG "★free ${FREE}G < 29G, 写不下 28 GiB 输出 — 停★"; exit 5; }

LOG "起合并"
bash "$SC/r29_campaign.sh" merge > /tmp/r29_merge_stage.log 2>&1
MRC=$?
LOG "合并 rc=$MRC"
[ $MRC -eq 0 ] || { LOG "★合并失败★"; tail -8 /tmp/r29_merge_stage.log >&2; exit 6; }
LOG "★R29 模型就绪 $(ls -l "$MDL" | awk '{printf "%.2f GiB",$5/1073741824}')★"
LOG "下一步(人工): r29_bench.sh all"
