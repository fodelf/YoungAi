#!/bin/bash
# r64_rb_refit.sh — 2026-08-05 用户批: 冠军配方路由工序补齐(RB FIT+烘焙), 在回扫流程内实现。
# 链: 保全 → 回扫遍反修(BACKFIT_INCR=0+GSWEEP=1 单轮, FIT 寄生零额外前向) → 学生回放
#     → 合并(含 RB 烘焙 α2.5) → 五指标 → 回传 M4。
# 速度账: 回扫遍 ~40-60min + student ~40min + merge ~1.5h + metrics ~30min ≈ 3.5h。
set -uo pipefail
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
OUTF=$M1DIR/gguf/go-onebit/r30/full
SC=$M1DIR/gguf-tools/go-onebit/scripts
LOG(){ echo "[rb_refit $(date +%H:%M:%S)] $*"; }

# ① 前置: 清场 + 破坏前保全(cp -c 零成本: op 侧车/zfile/zchain — 回扫会增量更新它们)
ssh $M1 "pgrep -f 'ds4quant_run|ds4-server' >/dev/null" && { LOG "★M1 有进程在跑 — 停"; exit 2; }
pgrep -f ds4-server >/dev/null && { LOG "★M4 有 server — 停"; exit 2; }
ssh $M1 "rm -rf $OUTF/rb_refit_backup && mkdir -p $OUTF/rb_refit_backup && \
         cp -c $OUTF/layers/dql_ops_L*.bin $OUTF/zfile.bin $OUTF/zchain.bin $OUTF/rb_refit_backup/ 2>/dev/null; \
         ls $OUTF/rb_refit_backup | wc -l" | { read N; LOG "保全 $N 文件 → rb_refit_backup/"; }

# ② 完整反修(4loss 二进制 + 联合损失 ONEPASS + GSWEEP 单轮回扫 + FIT 寄生统计)
LOG "完整反修起跑(ONEPASS 逐层 + GSWEEP 单轮 + FIT)"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=1 DS4_GS_CONV_PCT=0 \
         bash $SC/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
ssh $M1 "ls -la $OUTF/route_bias_r30.bin | awk '{print \$5}'" | { read SZ; LOG "反修 ✓ Δb 落盘 ${SZ}B"; }

# ③ 学生回放 + 合并(含 RB 烘焙) + 五指标
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         bash $SC/r30_campaign.sh student" || { LOG "★学生回放失败★"; exit 3; }
LOG "学生回放 ✓"
ssh $M1 "cd $M1DIR && bash $SC/r30_campaign.sh merge"   || { LOG "★合并/烘焙失败★"; exit 3; }
LOG "合并+RB 烘焙 ✓"
ssh $M1 "cd $M1DIR && bash $SC/r30_campaign.sh metrics" || { LOG "★指标失败★"; exit 3; }

# ④ 回传 M4(覆盖同名; scp O_TRUNC 不双份)
LOG "回传 M4…"
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf $ROOT/gguf/go-onebit/ds4-r30.gguf \
    || { LOG "★回传失败★"; exit 4; }
LOG "★收官: RB 版模型两机在位 — 下一步 Go/0+Go/1 双针判决★"
