#!/bin/bash
# r60_deploy_campaign.sh — 部署态反修战役【总驱动, 全流程自包含可复制】(2026-08-07)。
# 背景定案(fable5 2026-08-06/07): 0.46 KL 缺口=教师路由拟合轨迹失配; ONEPASS 全窗
# 终验在部署态误判劣化→静默全回滚; 修复=部署态口径(campaign 反修/评分段 unset
# DS4_ANCHOR_ROUTE)+分块顺序 ONEPASS(DS4_BF_CHUNK)+zfile_commit type7 分支。
# 本脚本=从干净现场到引擎判决的完整一条链, 每步机器化, 无人肉环节:
#   ①量化器编译自检(源比二进制新→重编 .dchunk) ②侧车复位教师干净态+nrec 校验
#   ③反修(分块)→评分→合并→回传 ④引擎 32 针判决(chain+bare 对照)
# 用法: r60_deploy_campaign.sh   (零参数=唯一行为; 分块7=量化器代码默认,
#        粗筛/线程=campaign 定版值 — 调优变更走 campaign/代码, 不走本脚本传参)
set -uo pipefail
cd /Users/fodelf/git/ds4-main
M1=192.168.1.2; M1DIR=/Users/fodelf/ds4-main
SC_M1=$M1DIR/gguf-tools/go-onebit/scripts
QSRC=gguf-tools/go-onebit/quant/ds4quant_run.c
QBIN_M1=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.dchunk
LOG(){ echo "[dcamp $(date +%H:%M:%S)] $*" >&2; }

# ── ①量化器编译自检(M4 源为准 → M1 编译; 源新于二进制则重编) ──
scp -q "$QSRC" $M1:$M1DIR/gguf-tools/go-onebit/quant/
ssh $M1 "cd $M1DIR/gguf-tools/go-onebit/quant && \
    if [ ! -x ds4quant_run.dchunk ] || [ ds4quant_run.c -nt ds4quant_run.dchunk ]; then \
        echo '[dcamp] 重编 ds4quant_run.dchunk' >&2; \
        cc -O3 -Wall -Wextra -Wno-unused-parameter -lm -framework Accelerate \
           -I$M1DIR -o ./ds4quant_run.dchunk ds4quant_run.c -lpthread || exit 1; \
    fi" || { LOG "★量化器编译失败★"; exit 2; }
scp -q gguf-tools/go-onebit/scripts/r30_campaign.sh $M1:$SC_M1/
LOG "①编译+脚本同步 ✓"

# ── ②侧车复位教师干净态 + nrec 一致性校验(2026-08-07 nrec 污染教训) ──
ssh $M1 "cd $M1DIR/gguf/go-onebit/r30/full && \
    [ -d ops_teacher_backup ] || { echo '★教师侧车备份缺★' >&2; exit 1; } && \
    cp -c ops_teacher_backup/dql_ops_L*.bin ops_teacher_backup/opt_L*.bin layers/ && \
    python3 -c \"
import struct,sys
bad=0
for L in range(43):
    raw=open(f'layers/dql_ops_L{L:02d}.bin','rb').read()
    nrec,=struct.unpack_from('<I',raw,8)
    off=12; n=0
    while off+116<=len(raw):
        psz,=struct.unpack_from('<Q',raw,off+88); off+=116+psz; n+=1
    if nrec!=n: bad+=1
print(f'nrec 校验 {43-bad}/43')
sys.exit(1 if bad else 0)\"" || { LOG "★侧车复位/校验失败★"; exit 2; }
LOG "②侧车复位+校验 ✓"

# ── ③反修链(12G 看门狗伴随; 分块部署态; 评分; 合并; 回传) ──
ssh $M1 'nohup bash -c "while true; do P=\$(pgrep -nf \"[d]s4quant_run\" || true); [ -n \"\$P\" ] && { MB=\$(footprint -p \$P 2>/dev/null | grep -Eo \"Footprint: *[0-9.]+ *[KMG]B\" | head -1 | awk \"{v=\\\$2;u=\\\$3; if(u==\\\"GB\\\")v*=1024; else if(u==\\\"KB\\\")v/=1024; printf \\\"%d\\\",v}\"); [ -n \"\$MB\" ] && [ \"\$MB\" -gt 11900 ] && kill -9 \$P; }; sleep 5; done" > /tmp/dbf_wdog.log 2>&1 & echo wdog-armed' || true

LOG "③反修起跑(部署态自路由+RB α2.5, 分块=代码默认7)"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$QBIN_M1 \
         HOT_TABLE=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top72.txt \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=0 \
         bash $SC_M1/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
LOG "反修 ✓"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$QBIN_M1 \
         HOT_TABLE=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top72.txt \
         bash $SC_M1/r30_campaign.sh student" || { LOG "★评分失败★"; exit 3; }
LOG "评分 ✓"
ssh $M1 "pkill -f 'while true; do P=' " 2>/dev/null || true
ssh $M1 "cd $M1DIR && bash $SC_M1/r30_campaign.sh merge" || { LOG "★合并失败★"; exit 3; }
LOG "合并+烘焙 ✓"
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf gguf/go-onebit/ds4-r30.gguf || { LOG "★回传失败★"; exit 4; }
LOG "回传 ✓ $(ls -l gguf/go-onebit/ds4-r30.gguf | awk '{printf "%.2f GB", $5/1e9}')"

# ── ④引擎判决(chain=新op链; bare=空链对照) ──
bash gguf-tools/go-onebit/scripts/r60_engine_verdict.sh gguf/go-onebit/ds4-r30.gguf chain
LOG "★战役收官 — 行为门(双针+败题)由驾驶员按判决决定★"
