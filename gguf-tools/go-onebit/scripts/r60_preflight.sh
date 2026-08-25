#!/bin/bash
# r60_preflight.sh — 发车前机器自检(2026-08-06 用户铁律"所有流程进脚本, 不靠临时发现"):
# 任何不一致 = 拒发车 + 打印修法。被 r60_fire.sh 头部强制调用。
set -uo pipefail
M1=192.168.1.2
M1DIR=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
HOT="${PLAN_HOT:-72}"
CAP_GB="${CAP_GB:-60.0}"
FAIL=0
say(){ echo "[preflight] $*"; }
bad(){ echo "[preflight] ★FAIL: $*"; FAIL=1; }

# ① plan 账 vs 闸(单点真相)
PLAN_HOT=$HOT python3 $ROOT/gguf-tools/go-onebit/scripts/r30_plan_champ.py /tmp/pf_plan.json >/dev/null 2>&1
read -r GB GIB PAY <<<"$(python3 -c "
import json; d=json.load(open('/tmp/pf_plan.json'))
lay=d['layers'][0]['bytes']; hot=d['layers'][0]['hot']
emb=(lay+(256-hot)*2*1.008*2**20)*43/2**30   # 内嵌载荷(blob+冷副本)=量化器 VOL 闸口径... 实为 blob 载荷闸=lay×43
print(d['model_GB'], d['model_GiB'], round(emb,2))")"
awk -v g="$GB" -v c="$CAP_GB" 'BEGIN{exit !(g>c)}' && bad "账面落地 ${GB}GB > 上限 ${CAP_GB}GB — 降 PLAN_HOT" || say "账面 ${GB}GB ≤ ${CAP_GB}GB ✓ (载荷 ${PAY}GiB)"

# ② 热表: 在位 + 43 行(两机)
N4=$(wc -l < $ROOT/gguf-tools/go-onebit/corpus/prog_active_top${HOT}.txt 2>/dev/null | tr -d ' ' || echo 0)
N1=$(ssh $M1 "wc -l < $M1DIR/gguf-tools/go-onebit/corpus/prog_active_top${HOT}.txt 2>/dev/null" | tr -d ' ' || echo 0)
[ "$N1" = "43" ] || bad "M1 热表 top${HOT} 行数 $N1≠43"
[ "$N4" = "43" ] || say "M4 热表缺(仅 M1 消费, 非致命)"
[ "$N1" = "43" ] && say "热表 top${HOT} 43 行 ✓"

# ③ 两机源一致 + 二进制新鲜(源 md5 同 & 二进制 mtime ≥ 源 mtime)
for f in gguf-tools/go-onebit/quant/ds4quant_run.c gguf-tools/go-onebit/quant/vq_merge_v4.c gguf-tools/go-onebit/scripts/r30_campaign.sh; do
  A=$(md5 -q $ROOT/$f 2>/dev/null); B=$(ssh $M1 "md5 -q $M1DIR/$f" 2>/dev/null)
  [ "$A" = "$B" ] || bad "两机源不一致: $f — scp 同步后重试"
done
say "两机源 md5 ✓"
ssh $M1 "[ $M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.4loss -nt $M1DIR/gguf-tools/go-onebit/quant/ds4quant_run.c ]" \
  || bad "M1 .4loss 旧于源 — 重编: cc ... -o ds4quant_run.4loss"
say "M1 .4loss 新鲜 ✓"

# ④ 锚在位 + 盘空间(plan 派生 NEED)
ssh $M1 "[ -f $M1DIR/gguf/go-onebit/r30/anchor_r30_s1716.bin ]" || bad "M1 锚缺 — campaign anchor 段会重建(~5min), 非致命"
NLAY=$(ssh $M1 "ls $M1DIR/gguf/go-onebit/r30/full/layers/dql_L*.bin 2>/dev/null | wc -l" | tr -d ' ')
FREE=$(ssh $M1 "df -g /System/Volumes/Data | awk 'NR==2{print \$4}'")
if [ "$NLAY" = "43" ]; then
  ANC_G=$(ssh $M1 "ls -l $M1DIR/gguf/go-onebit/r30/anchor_r30_s1716.bin 2>/dev/null | awk '{printf \"%d\", \$5/1073741824}'" || echo 0)
  EFF=$((FREE+${ANC_G:-0}-1))   # merge 自带裁锚: 有效空间=free+锚-ref(0.9≈1)
  NEED=$(python3 -c "print(int(float('$GB')+1))")
  if [ "$EFF" -ge "$NEED" ]; then say "M1 有效 ${EFF}G(free ${FREE}+裁锚 ${ANC_G:-0}-1) ≥ 合并需 ${NEED}G ✓"; else bad "M1 有效 ${EFF}G < 合并需 ${NEED}G"; fi
else
  NEED=$(python3 -c "print(int(38.6+float('$PAY')*0.63+2))")
  if [ "$FREE" -ge "$NEED" ]; then say "M1 free ${FREE}G ≥ ${NEED}G ✓ (fresh)"; else bad "M1 free ${FREE}G < 需 ${NEED}G (fresh 起跑口径)"; fi
fi
FREE4=$(df -g / | awk 'NR==2{print $4}')
awk -v f="$FREE4" -v g="$GB" 'BEGIN{exit !(f<g+0.5)}' && bad "M4 free ${FREE4}G < 回传需 ~${GB}+0.5G" || say "M4 free ${FREE4}G ✓"

[ "$FAIL" = 0 ] && { say "★全绿 — 放行★"; exit 0; } || { say "★存在 FAIL — 拒发车★"; exit 9; }
