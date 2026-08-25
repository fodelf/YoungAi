#!/bin/bash
# r30_chain.sh — R30 无人值守串联(2026-08-02, 用户令: 0731 源 + 36G + 全流程 + 五指标)。
#
# 锚构建已单独在跑(r30_campaign.sh anchor)。本脚本等它退出, 校验锚 + PPL 自检,
# 然后起 campaign all —— anchor 阶段见锚在场会短路, 接着 plan → quant → backfit
# → student(学生回放 dump logits)→ merge(裁锚→骨架→消费式合并)→ metrics(五指标)。
#
# 盘账定序(M1 free 50G 起, 铁律: hf/hf-base 不动):
#   锚 6.5 → layers 27.8 + ckpt 5(量化)→ 学生回放(0 增量)→ 裁锚 −6.5+0.9 → 删 ckpt −5
#   → 骨架峰值 +16.4(稀疏 hole, 完成后 8.2 常驻)→ 消费合并净增 ~9.2 → 终态余量 ~4G
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
LOG(){ echo "[chain30 $(date +%H:%M:%S)] $*" | tee -a /tmp/r30_chain.log >&2; }

LOG "等锚构建退出…"
while pgrep -f "ds4quant_run.r30" >/dev/null; do sleep 60; done
LOG "锚进程已退出"

A="$ROOT/gguf/go-onebit/r30/anchor_r30_s1716.bin"
[ -f "$A" ] || { LOG "★锚不在 — 停(看 /tmp/r30_anchor.log)★"; exit 2; }
LOG "锚 $(ls -l "$A" | awk '{printf "%.2f GiB",$5/1073741824}')"
grep -aq "冒烟判决: FAIL" /tmp/r30_anchor.log && { LOG "★锚 PPL 自检 FAIL — 停★"; exit 3; }
grep -aq "冒烟判决: PASS" /tmp/r30_anchor.log || LOG "★锚自检没见 PASS 行 — 继续但需人工复核★"

LOG "起 campaign all(anchor 短路 → plan → quant → backfit → student → merge → metrics)"
bash "$SC/r30_campaign.sh" all >> /tmp/r30_all.log 2>&1
RC=$?
LOG "campaign all rc=$RC"
MDL="$ROOT/gguf/go-onebit/ds4-r30.gguf"
[ -f "$MDL" ] && LOG "★R30 模型就绪 $(ls -l "$MDL" | awk '{printf "%.2f GiB",$5/1073741824}')★"
exit $RC
