#!/bin/bash
# r60_deploy_backfit_fire.sh — 部署态重反修链 (2026-08-06 用户令"那就重新反修")。
# 前提已完成: ①0.46 缺口定案=教师路由拟合轨迹失配 ②campaign 反修/评分段已切部署态
# (unset DS4_ANCHOR_ROUTE) ③口径验证遍 KL 0.682→1.070 与引擎 1.139 对齐 ④教师版侧车
# ops_teacher_backup/ 留档 ⑤M4 现役留档 ds4-r30-tbf.gguf。
# 链: M1 反修(.4loss 同二进制, 唯一变量=路由口径) → 评分 → 合并 → 回传 M4。
# 判决(回传后由驾驶员跑): 引擎 32 针链版 KL(目标 <1.082) + 行为门双针。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
M1=192.168.1.2; M1DIR=/Users/fodelf/ds4-main
SC_M1=$M1DIR/gguf-tools/go-onebit/scripts
LOG(){ echo "[dbf $(date +%H:%M:%S)] $*" >&2; }

# 12G 红线看门狗(M1 侧独立进程, 链尾清)
ssh $M1 'nohup bash -c "while true; do P=\$(pgrep -nf \"[d]s4quant_run\" || true); [ -n \"\$P\" ] && { MB=\$(footprint -p \$P 2>/dev/null | grep -Eo \"Footprint: *[0-9.]+ *[KMG]B\" | head -1 | awk \"{v=\\\$2;u=\\\$3; if(u==\\\"GB\\\")v*=1024; else if(u==\\\"KB\\\")v/=1024; printf \\\"%d\\\",v}\"); [ -n \"\$MB\" ] && [ \"\$MB\" -gt 11900 ] && kill -9 \$P; }; sleep 5; done" > /tmp/dbf_wdog.log 2>&1 & echo wdog-pid=$!' || true

LOG "反修起跑(部署态自路由 + RB α2.5, .4loss 二进制)"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.dchunk \
         HOT_TABLE=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top72.txt \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=0 DS4_BF_CHUNK=7 DS4_BF_SCREEN_DIV=24 DS4_THREADS=8 \
         bash $SC_M1/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
LOG "部署态反修 ✓"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.dchunk \
         HOT_TABLE=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top72.txt \
         bash $SC_M1/r30_campaign.sh student" || { LOG "★评分失败★"; exit 3; }
LOG "部署态评分 ✓"
ssh $M1 "pkill -f 'dbf_wdog\|while true; do P=' " 2>/dev/null || true
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh merge" || { LOG "★合并失败★"; exit 3; }
LOG "合并+烘焙 ✓"
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf $ROOT/gguf/go-onebit/ds4-r30.gguf || { LOG "★回传失败★"; exit 4; }
LOG "回传 ✓ $(ls -l $ROOT/gguf/go-onebit/ds4-r30.gguf | awk '{printf "%.2f GB", $5/1e9}') — 接: 引擎 32 针 + 行为门"
