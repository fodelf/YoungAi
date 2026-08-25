#!/bin/bash
# r64_4loss_refit.sh — 2026-08-05 用户令: Go 跑完→删两机 gguf→四损失+感知完整反修(4loss 版
# ONEPASS: KGRID 13 组进 E/C/F 三形态+行权布线)→重新合并→五指标评分→学生回放验证。
# 平行架构红利: 量化 dql/vq 侧车不动, 只删 op 侧车重长。
set -uo pipefail
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
OUTF=$M1DIR/gguf/go-onebit/r30/full
SC=$M1DIR/gguf-tools/go-onebit/scripts
LOG(){ echo "[4loss $(date +%H:%M:%S)] $*"; }

# ① 等 Go 批(错题针+补题)清场
while pgrep -f pubbench_serial >/dev/null; do sleep 30; done
LOG "Go 批已清场"

# ② 删两机旧 gguf(用户令 2026-08-05)
rm -f /Users/fodelf/git/ds4-main/gguf/go-onebit/ds4-r30.gguf && LOG "M4 gguf 已删"
ssh $M1 "rm -f $M1DIR/gguf/go-onebit/ds4-r30.gguf" && LOG "M1 gguf 已删"

# ③ 删反修产物(op 侧车/zfile/zchain/opt/学生logits), 量化 dql+vq 侧车+锚保留
ssh $M1 "rm -f $OUTF/layers/dql_ops_L*.bin $OUTF/layers/opt_L*.bin $OUTF/layers/opbak_L*.bin \
         $OUTF/zfile.bin $OUTF/zchain.bin $OUTF/student_logits.bin"
LOG "反修产物已清(dql/vq/锚在)"

# ④ 四损失+感知完整反修 + 学生回放(4loss 二进制)
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=1 DS4_GS_CONV_PCT=0 bash $SC/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
LOG "反修(四损失+感知)完成"
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss \
         bash $SC/r30_campaign.sh student" || { LOG "★学生回放失败★"; exit 3; }
LOG "学生回放完成"

# ⑤ 合并 + 五指标
ssh $M1 "cd $M1DIR && bash $SC/r30_campaign.sh merge" || { LOG "★合并失败★"; exit 3; }
ssh $M1 "cd $M1DIR && bash $SC/r30_campaign.sh metrics" || { LOG "★指标失败★"; exit 3; }
LOG "★4loss 链收官: 合并+五指标完成★"
