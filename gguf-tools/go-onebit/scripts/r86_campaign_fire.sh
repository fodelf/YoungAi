#!/bin/bash
# r86_campaign_fire.sh — 86G 战役一条龙(2026-08-07 用户令"86G 开跑")。
# 配方(r86_plan.json, 85.96GB 落地): 热149@vq4x512 + 冷107 w1/w3@vq4x256(2bpw 补偿区)
#   + w2 signref(档位=实验参数非定理, 由 plan 生成器 PLAN_W2_* 可改) + 骨架 8.202。
# 链: 热表149 → plan 同步 → 量化43层 → 分块部署态反修(修复版) → 评分 → 合并 → 回传 → 32针。
# 盘账: 量化+反修 ~99G ≤ M1 可用; 合并需再 86G — 到站前由用户定盘方案(清数据/外置/特批消费式)。
set -uo pipefail
cd /Users/fodelf/git/ds4-main
M1=192.168.1.2; M1DIR=/Users/fodelf/ds4-main
SC_M1=$M1DIR/gguf-tools/go-onebit/scripts
QBIN_M1=$M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.dchunk
HOT149=$M1DIR/gguf-tools/go-onebit/corpus/prog_active_top149.txt
LOG(){ echo "[r86 $(date +%H:%M:%S)] $*" >&2; }

# ── ①同步(源/脚本/计划) + 量化器编译自检 ──
scp -q gguf-tools/go-onebit/quant/ds4quant_run.c $M1:$M1DIR/gguf-tools/go-onebit/quant/
scp -q gguf-tools/go-onebit/scripts/r30_campaign.sh gguf-tools/go-onebit/scripts/anchor_top_experts.py $M1:$SC_M1/
scp -q gguf/go-onebit/r30/r86_plan.json $M1:$M1DIR/gguf/go-onebit/r30/
ssh $M1 "cd $M1DIR/gguf-tools/go-onebit/quant && \
    if [ ! -x ds4quant_run.dchunk ] || [ ds4quant_run.c -nt ds4quant_run.dchunk ]; then \
        cc -O3 -Wall -Wextra -Wno-unused-parameter -lm -framework Accelerate \
           -I$M1DIR -o ./ds4quant_run.dchunk ds4quant_run.c -lpthread || exit 1; fi" \
    || { LOG "★量化器编译失败★"; exit 2; }
LOG "①同步+编译 ✓"

# ── ①b 锚(被清理则自动重建; campaign stage_anchor 自幂等) ──
ssh $M1 "cd $M1DIR && QBIN_OVERRIDE=$QBIN_M1 bash $SC_M1/r30_campaign.sh anchor" \
    || { LOG "★锚重建失败★"; exit 2; }
LOG "①b 锚 ✓"

# ── ②top149 热表(从全量锚生成; 已在且 43 行则跳过) ──
ssh $M1 "[ -f $HOT149 ] && [ \$(wc -l < $HOT149) -eq 43 ] || \
    python3 $SC_M1/anchor_top_experts.py $M1DIR/gguf/go-onebit/r30/anchor_r30_s1716.bin 149 $HOT149" \
    || { LOG "★热表生成失败★"; exit 2; }
LOG "②top149 热表 ✓"

# ── ③量化 43 层(计划=r86_plan; 量化段自清旧 layers; 盘闸/载荷闸在 campaign 内) ──
ssh $M1 'nohup bash -c "while true; do P=\$(pgrep -nf \"[d]s4quant_run\" || true); [ -n \"\$P\" ] && { MB=\$(footprint -p \$P 2>/dev/null | grep -Eo \"Footprint: *[0-9.]+ *[KMG]B\" | head -1 | awk \"{v=\\\$2;u=\\\$3; if(u==\\\"GB\\\")v*=1024; else if(u==\\\"KB\\\")v/=1024; printf \\\"%d\\\",v}\"); [ -n \"\$MB\" ] && [ \"\$MB\" -gt 11900 ] && kill -9 \$P; }; sleep 5; done" > /tmp/r86_wdog.log 2>&1 & echo wdog-armed' || true
LOG "③量化起跑(43 层)"
ssh $M1 "cd $M1DIR && PLAN_JSON=$M1DIR/gguf/go-onebit/r30/r86_plan.json \
         HOT_TABLE=$HOT149 QBIN_OVERRIDE=$QBIN_M1 \
         bash $SC_M1/r30_campaign.sh quant" || { LOG "★量化失败★"; exit 3; }
LOG "量化 ✓"

# ── ④分块部署态反修(修复版: 绝对语义+毒值拒收+分块终验=代码默认) ──
ssh $M1 "cd $M1DIR && PLAN_JSON=$M1DIR/gguf/go-onebit/r30/r86_plan.json \
         HOT_TABLE=$HOT149 QBIN_OVERRIDE=$QBIN_M1 \
         DS4_NO_MILESTONE=1 DS4_GSWEEP=0 \
         bash $SC_M1/r30_campaign.sh backfit" || { LOG "★反修失败★"; exit 3; }
LOG "反修 ✓"
ssh $M1 "cd $M1DIR && PLAN_JSON=$M1DIR/gguf/go-onebit/r30/r86_plan.json \
         HOT_TABLE=$HOT149 QBIN_OVERRIDE=$QBIN_M1 \
         bash $SC_M1/r30_campaign.sh student" || { LOG "★评分失败★"; exit 3; }
LOG "评分 ✓"
ssh $M1 "pkill -f 'while true; do P=' " 2>/dev/null || true

# ── ⑤合并(前置盘检: free<模型账 → 停在这里等盘方案, 量化/反修成果无损) ──
NEED_G=$(python3 -c "import json;print(int(json.load(open('gguf/go-onebit/r30/r86_plan.json'))['model_GB'])+2)")
FREE_G=$(ssh $M1 "df -g /System/Volumes/Data | awk 'NR==2{print \$4}'")
[ "$FREE_G" -ge "$NEED_G" ] || { LOG "★合并盘闸: M1 free ${FREE_G}G < ${NEED_G}G — 停在合并前, 量化/反修产物完好, 等盘方案★"; exit 7; }
ssh $M1 "cd $M1DIR && PLAN_JSON=$M1DIR/gguf/go-onebit/r30/r86_plan.json bash $SC_M1/r30_campaign.sh merge" \
    || { LOG "★合并失败★"; exit 3; }
LOG "合并+烘焙 ✓"
scp -q $M1:$M1DIR/gguf/go-onebit/ds4-r30.gguf gguf/go-onebit/ds4-r86.gguf || { LOG "★回传失败★"; exit 4; }
LOG "回传 ✓ $(ls -l gguf/go-onebit/ds4-r86.gguf | awk '{printf "%.2f GB",$5/1e9}')"
bash gguf-tools/go-onebit/scripts/r60_engine_verdict.sh gguf/go-onebit/ds4-r86.gguf chain
LOG "★r86 战役收官 — 行为门由驾驶员接★"
