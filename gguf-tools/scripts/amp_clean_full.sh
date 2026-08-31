#!/bin/bash
# amp_clean_full.sh — 干净锚反修一条龙(2026-08-24 毒锚事故修复后: 锚=FP遍重造干净口径)
# 流程: 等干净锚就绪 → 反修工作区(保护原件) → zlayer 全家 43 层 → caliper 五指标终判
# 对表: 毒锚版 amp2 KLD 0.42792/top1 79.53; 裸底 0.47055/78.36; 官方 q2 0.4207/77.92
#
# 用法: amp_clean_full.sh [工作区名=amp_clean] [XCAP捕获目录名=空]
#   $1 工作区名(相对 gguf/go-onebit/vqhalf), 默认 amp_clean = 原行为
#   $2 引擎捕获目录名(同上相对), 传了就给 zlayer 第7参 XCAP —— 学生 x/被乘量=引擎真值。
#      不传时 zlayer 回落 x=锚fin(FP链态), 与部署分布错位(实测行cos L3=0.961/L20=0.875/
#      L40=0.797), z 的层内收益在部署 x 上不兑现 —— 2026-08-24 定位的主 bug。
#      cap_a = 解码路取料 + 两遍逐位复现闸已过(2026-08-23 vqpipe2)。
set -u
cd ~/ds4-main
export DS4_HF=$HOME/ds4-main/hf/DeepSeek-V4-Flash-0731
D2=gguf/go-onebit/vqhalf
WS=${1:-amp_clean}
XC=${2:-}
K=${3:?秩 K 必传(冠军=64)。★不给默认★: 旧版 ${3:-1024} 让漏传时静默变成 1024, 与 LZRANK 静默兜底 16 同款事故}
XA=${4:-}
ANCOV=${5:-}
LRANGE=${6:-}          # ★$6=层范围 "a b"(默认全 43 层): 10 分钟级单层验证用, 不新造脚本
ANC=${ANCOV:?锚路径必传($5)。★不给默认★: 喂错锚 PPL 会飙到 2.4e7, 但喂【旧】锚只会静默出错数}
LOG(){ echo "[$WS $(date +%H:%M:%S)] $*"; }
# ①等锚(FP遍写完的判据=文件尺寸到位)  [python3 内联清零(2026-08-25 迁移 Wave C): bash 算术同式]
# ★镜像公式★: DQA2 锚布局的单一权威是 ds4quant_anchor.inc.c 的写者, 这里只是按同一布局
# 反推总字节数当等待判据 —— 改锚格式必须同步改这行, 否则判据静默失效(永等或早发车)。
NL=43; S=8192; DIM=4096; NSEL=6; VOCAB=129280   # 层数/校准行数/隐维/每行选中专家数/词表
EXP=$(( 40 + NL*S*DIM*4 + 2*NL*S*NSEL*4 + NL*S*4*DIM*4 + S*VOCAB*4 ))
while true; do
  sz=$(stat -c %s "$ANC" 2>/dev/null || echo 0)
  [ "$sz" -ge "$EXP" ] && break
  sleep 60
done
LOG "①干净锚就绪 $(du -h $ANC | cut -f1)"
if [ -n "$XC" ]; then
  [ -s "$D2/$XC/raw_ffn_in_L0" ] || { LOG "★XCAP $D2/$XC 缺 raw_ffn_in_L0★"; exit 1; }
  LOG "①XCAP=$D2/$XC (学生=引擎真值)"
fi
# ②反修工作区(全量重跑铁律: 从 noz 干净态起)
rm -rf $D2/$WS
mkdir -p $D2/$WS/layers
cd $D2/${SRCBASE:?底座层件目录必传(env SRCBASE=, 如 champ86/layers_quant)。★不给默认★: 默认到 vq86h_noz 会拿旧语料量化的底座当新底座}
for f in dql_vq_L*.bin; do ln -f "$f" "$HOME/ds4-main/$D2/$WS/layers/$f" 2>/dev/null || cp "$f" "$HOME/ds4-main/$D2/$WS/layers/"; done
cp dql_ops_L*.bin opt_L*.bin manifest.txt "$HOME/ds4-main/$D2/$WS/layers/" 2>/dev/null
cp dql_L*.bin "$HOME/ds4-main/$D2/$WS/layers/"
cd ~/ds4-main
LOG "②工作区就绪"
# 行掩码不再在这里算(2026-08-29): 已下沉进 zlayer/ds4quant_run —— 它们从【自己拿到的
# 锚路径】读 <锚>.layout 推导行域。脚本现算现传(原 DS4_ZL_FIT_RANGES/EV_RANGE, 死名已删)
# 那条路已删: 掩码写在脚本里 = 语料一换就静默错位 = 今天 z 全拒 378/378 的根因。
# ③zlayer 全家 43 层(干净锚; XC 非空=第7参引擎捕获)
# zlayer=C 版(2026-08-25 迁移 Wave B 一期, 金标 migrate/golden.txt: z支线与py精确一致/
# GE=py-CPU路真解口径(py-GPU路 XCAP 下 GE 恒死是 py 自身分裂, 见 zlayer_transcription_notes #1)/
# 注入产物 rec_fidelity 复评一致)。XANCHOR/GGUF/ADDON 等二期模式 C 版硬拒(响亮失败, 无静默兜底)。
# ⚠二期欠账: C 版纯 CPU ~570s/层(py-GPU 54s), 43层≈6.8h — 按"spark重计算必须GPU化"铁律须补 CUDA 路。
ZLB="$HOME/ds4-main/gguf-tools/amp/zlayer"
# 构建收进 gguf-tools/Makefile(批1): 平台特判(Accelerate/scipy_openblas/CUDA)都在那边。
[ -x "$ZLB" ] || make -C "$HOME/ds4-main/gguf-tools" zlayer
for L in ${LRANGE:-$(seq 0 42)}; do
  # 原 DS4_ZL_NTOK/XANCHOR env → 位置参数后的 --flag(env 大扫除); DS4_ZL_NFIT 死名已删
  "$ZLB" "$DS4_HF" $D2/$WS/layers "$ANC" $L "$K" 1 ${XC:+$D2/$XC} \
    --ntok 8192 ${XA:+--xanchor "$XA"} \
    2>&1 | grep -aE "XCAP|Error|assert|★" || { LOG "★L$L 失败★"; exit 1; }
  # 进度可观测铁律: 每层收官打一行(tail -f 就能看到 43 层推进)
  # INJ=1 走 dql 注入不写 zrec ⇒ 旧的 zrec 计数恒 0/43(观测 bug); 改数注入账本行
  LOG "L$L ✓ $(grep -c '' "$D2/$WS/layers/zinject_manifest.txt" 2>/dev/null || echo 0)/43 K=$K"
done
rm -f $D2/$WS/layers/zcache_L*.npz
LOG "③反修收官"
# ④官方语料五指标终判(单层验证 LRANGE 在场时跳过 — 10 分钟级铁律)
[ -n "$LRANGE" ] && { LOG "④跳过(单层验证)"; exit 0; }
bash gguf-tools/scripts/caliper_ref.sh $D2/$WS/layers /tmp/qc_${WS}_wt2.bin > /tmp/caliper_${WS}.log 2>&1
grep -E "PPL\(stu|Mean KLD|RMS|Same top|Δp" /tmp/caliper_${WS}.log
LOG "④五指标终判完成"
