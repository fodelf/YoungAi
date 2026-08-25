#!/bin/bash
# r30_disk_janitor.sh — 学生回放完成后删 dql_LXX.bin 反修工作文件(2026-08-02 用户问
# "凭什么中间产物这么多"暴露的盘账漏项: 合并期清理清单漏了它, ~25G 死重)。
#
# 为什么独立进程: campaign all 正在运行, 铁律禁覆盖运行中脚本(bash 按字节偏移续读,
# 覆盖=错位执行)。本脚本从外部等窗口, 与 campaign 零交集。
#
# 删除窗口判据(三重, 全满足才动手):
#   ① student_logits.bin 已落盘(学生回放完成 — dql 的最后一个读者)
#   ② 无 ds4quant_run 进程(量化/反修/回放都不在跑)
#   ③ lsof 确认没有任何进程打开 dql 文件
# 删除物只有 dql_L[0-9][0-9].bin(精确模式, 绝不碰 dql_vq_L*.bin 合并载荷)。
set -uo pipefail
OUTF="$HOME/ds4-main/gguf/go-onebit/r30/full"
LOG(){ echo "[janitor $(date +%H:%M:%S)] $*" | tee -a /tmp/r30_janitor.log >&2; }

LOG "等学生回放完成(student_logits.bin + 无量化进程)…"
while true; do
    if [ -f "$OUTF/student_logits.bin" ] && ! pgrep -x ds4quant_run.r30 >/dev/null; then
        sleep 60   # 稳定窗: 让 campaign 写完收尾
        [ -f "$OUTF/student_logits.bin" ] || continue
        if lsof +D "$OUTF/layers" 2>/dev/null | grep -q "dql_L[0-9]"; then
            LOG "仍有进程打开 dql — 继续等"; sleep 60; continue
        fi
        break
    fi
    sleep 120
done
BEFORE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
N=$(ls "$OUTF"/layers/dql_L[0-9][0-9].bin 2>/dev/null | wc -l | tr -d ' ')
rm -f "$OUTF"/layers/dql_L[0-9][0-9].bin
AFTER=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
LOG "已删 $N 个 dql 反修工作文件: free ${BEFORE}G → ${AFTER}G(vq 载荷/zchain/ref_logits 全保留)"
