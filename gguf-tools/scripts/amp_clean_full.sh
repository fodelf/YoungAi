#!/bin/bash
# amp_clean_full.sh — champ86 反修一条龙: 工作区重建(保护原件) → zlayer 全家 43 层 → caliper 五指标
#
# ★零参数配方(2026-08-31 用户令"硬编码全部删除")★: 冠军配方全部死写在下面常量区,
# 本脚本不接受任何 env / 位置参数配方 —— 行为不随发车姿势变(原 WS/XC/K/XA/ANC/SRCBASE
# 六个传参位就是"配方靠发车姿势拼"的病根: K 漏传静默 1024、SRCBASE 喂错底座整跑作废,
# 都实撞过)。要换配方就改这里并记录进 fable5。
# 唯一参数: [L_lo L_hi] 只缩层范围做 10 分钟级单层验证(验证铁律), 不改配方。
set -u
cd ~/ds4-main
# ══ 冠军配方常量区(r64c 定型 0.42510, fable5 6685/6688; 每个值给出处) ══
D2=gguf/go-onebit/vqhalf
WS=champ86amp                        # 反修工作区(相对 $D2); 重建时整目录清掉
SRCBASE=$D2/champ86/layers_quant     # 底座=本轮量化态备份(唯一还原点; 拿旧语料底座当新底座=整跑作废, 08-28 实撞)
ANC=$D2/anchor_a_clean_s8192.bin     # 干净 FP 锚(2026-08-24 毒锚事故后口径; 尺寸下面实检, 错锚 PPL 飙 2.4e7 一眼假)
ZL_K=64                              # 冠军秩: K64 直解 0.42510 > k1024 解算再截断 0.43410 > amp2 0.42792(fable5 6688)
ZL_NTOK=8192                         # 拟合行数=放大器半语料全量(与锚同源 S=8192)
# 跨语料闸料(2026-08-31 落地): 量化半 FP 锚 —— 与反修半零重叠、又不是判决锚(判决锚进闸
# =对判决做模型选择, 铁律禁止)。同语料 held 正/量化半负 = 过拟合层, zlayer 闸内自动拒注。
QANC=$D2/anchor_vqhalf_q_s8192.bin
HF=$HOME/ds4-main/hf/DeepSeek-V4-Flash-0731
# XCAP/XANCHOR 不在冠军配方(FP-x 口径; "学生=引擎真值"支柱未兑现 —— 实测不接引擎捕获时
# x=锚fin 与部署分布错位, 行cos L3=0.961/L20=0.875/L40=0.797, z 层内收益部署不兑现,
# 2026-08-24 定位的主 bug, 修法挂账)。zlayer 的 XCAP 代码路保留, 兑现之日改这里接上。
LRANGE=""
if [ $# -gt 0 ]; then LRANGE=$(seq "${1:?}" "${2:?给了 L_lo 必须给 L_hi}"); fi
LOG(){ echo "[$WS $(date +%H:%M:%S)] $*"; }
# ①锚实检(硬停不等待: 锚必须先于本链存在; 旧"轮询等锚写完"是 r64c 并行造锚时代的拐棍)
# ★镜像公式★: DQA2 锚布局的单一权威是 ds4quant_anchor.inc.c 的写者, 这里按同一布局
# 反推总字节数当完整性判据 —— 改锚格式必须同步改这行, 否则判据静默失效。
NL=43; S=8192; DIM=4096; NSEL=6; VOCAB=129280   # 层数/校准行数/隐维/每行选中专家数/词表
EXP=$(( 40 + NL*S*DIM*4 + 2*NL*S*NSEL*4 + NL*S*4*DIM*4 + S*VOCAB*4 ))
sz=$(stat -c %s "$ANC" 2>/dev/null || echo 0)
[ "$sz" -eq "$EXP" ] || { LOG "★锚缺/尺寸不符($sz≠$EXP): $ANC★"; exit 1; }
[ -s "$QANC" ] || { LOG "★跨语料闸料缺(量化半锚): $QANC★"; exit 1; }
LOG "①干净锚实检 ✓ $(du -h $ANC | cut -f1) + 闸料 ✓"
# ②反修工作区(全量重跑铁律: 从量化态备份干净起; dql_vq 硬链只读, dql_L 真拷贝可注入)
rm -rf $D2/$WS
mkdir -p $D2/$WS/layers
cd "$SRCBASE" || { LOG "★底座层件目录缺: $SRCBASE★"; exit 1; }
for f in dql_vq_L*.bin; do ln -f "$f" "$HOME/ds4-main/$D2/$WS/layers/$f" 2>/dev/null || cp "$f" "$HOME/ds4-main/$D2/$WS/layers/"; done
cp dql_ops_L*.bin opt_L*.bin manifest.txt "$HOME/ds4-main/$D2/$WS/layers/" 2>/dev/null
cp dql_L*.bin "$HOME/ds4-main/$D2/$WS/layers/"
cd ~/ds4-main
LOG "②工作区就绪(底座=$SRCBASE)"
# 行掩码不在这里算(2026-08-29): 已下沉进 zlayer/ds4quant_run —— 它们从【自己拿到的
# 锚路径】读 <锚>.layout 推导行域。掩码写在脚本里 = 语料一换就静默错位 = z 全拒 378/378 的根因。
# ③zlayer 全家 43 层(干净锚)
# zlayer=C 版(2026-08-25 迁移 Wave B 一期, 金标 migrate/golden.txt: z支线与py精确一致/
# GE=py-CPU路真解口径/注入产物 rec_fidelity 复评一致)。XANCHOR/GGUF/ADDON 等二期模式 C 版硬拒。
ZLB="$HOME/ds4-main/gguf-tools/amp/zlayer"
[ -x "$ZLB" ] || make -C "$HOME/ds4-main/gguf-tools" zlayer
for L in ${LRANGE:-$(seq 0 42)}; do
  "$ZLB" "$HF" $D2/$WS/layers "$ANC" $L "$ZL_K" 1 \
    --ntok "$ZL_NTOK" --gate-anchor "$QANC" \
    2>&1 | grep -aE "XCAP|跨语料|Error|assert|★" || { LOG "★L$L 失败★"; exit 1; }
  # 进度可观测铁律: 每层收官打一行(tail -f 就能看到 43 层推进)
  # INJ=1 走 dql 注入不写 zrec ⇒ 旧的 zrec 计数恒 0/43(观测 bug); 改数注入账本行
  LOG "L$L ✓ $(grep -c '' "$D2/$WS/layers/zinject_manifest.txt" 2>/dev/null || echo 0)/43 K=$ZL_K"
done
rm -f $D2/$WS/layers/zcache_L*.npz
LOG "③反修收官"
# ④官方语料五指标终判(单层验证 LRANGE 在场时跳过 — 10 分钟级铁律)
[ -n "$LRANGE" ] && { LOG "④跳过(单层验证)"; exit 0; }
bash gguf-tools/scripts/caliper_ref.sh $D2/$WS/layers /tmp/qc_${WS}_wt2.bin > /tmp/caliper_${WS}.log 2>&1
grep -E "PPL\(stu|Mean KLD|RMS|Same top|Δp" /tmp/caliper_${WS}.log
LOG "④五指标终判完成"
