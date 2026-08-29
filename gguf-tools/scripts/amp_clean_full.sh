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
K=${3:-1024}
XA=${4:-}
ANCOV=${5:-}
IDSP=${6:-$D2/vqhalf_a.ids}   # ★$6=ids 路径(必须与 $5 锚配对)★ 行布局从 $IDSP.layout 读
LRANGE=${7:-}          # ★$7=层范围 "a b"(默认全 43 层): 10 分钟级单层验证用, 不新造脚本
ANC=${ANCOV:-$D2/anchor_a_clean_s8192.bin}
LOG(){ echo "[$WS $(date +%H:%M:%S)] $*"; }
# ①等锚(FP遍写完的判据=文件尺寸到位)  [python3 内联清零(2026-08-25 迁移 Wave C): bash 算术同式]
EXP=$(( 40 + 43*8192*4096*4 + 2*43*8192*6*4 + 43*8192*4*4096*4 + 8192*129280*4 ))
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
cd $D2/${SRCBASE:-vq86h_noz/layers}   # SRCBASE=相对 D2 的【层件目录】(如 champ86/layers_quant)
for f in dql_vq_L*.bin; do ln -f "$f" "$HOME/ds4-main/$D2/$WS/layers/$f" 2>/dev/null || cp "$f" "$HOME/ds4-main/$D2/$WS/layers/"; done
cp dql_ops_L*.bin opt_L*.bin manifest.txt "$HOME/ds4-main/$D2/$WS/layers/" 2>/dev/null
cp dql_L*.bin "$HOME/ds4-main/$D2/$WS/layers/"
cd ~/ds4-main
LOG "②工作区就绪"
# ★行掩码: 一个数字都不写死, 全从布局文件读(2026-08-29 用户令)★
# 为什么不能写死: 旧版把 "32块×256" 抄死在这里。语料后来换成 8域×1024行/窗128, 掩码
# 静默错位 —— 没人记得回来改。后果实测: fit 拉丁91% / val 西里尔26% / held 阿拉伯59%+
# 中日韩33%, 三段几乎零重叠 ⇒ z 落地 378/378 全拒。改成读布局后同一层 z 立刻复活
# (L0 组合 held 11.3%)。布局由 amp_campaign.sh stage_idshalf 在抽样时落盘, 生产方写、
# 消费方读, 换语料自动跟随。★读不到就硬停, 绝不回退到猜★。
LAY="$IDSP.layout"
[ -s "$LAY" ] || { LOG "★行布局缺: $LAY"; LOG "  先跑: bash gguf-tools/scripts/amp_campaign.sh idshalf (ids 已在时它只补布局, 逐字节校验, 不动 ids)"; exit 1; }
W=$(awk '$1=="win"{print $2}' "$LAY")
[ -n "$W" ] && [ "$W" -gt 0 ] 2>/dev/null || { LOG "★布局里没有 win"; exit 1; }
# 两个比例沿用冠军配方(唯一的"配方常数", 与语料无关, 就地注明出处):
#   剔窗首 25%(冠军 64/256) = 拼接毒: 窗首是源语料里的跳跃, 模型在那儿没上下文
#   末 25% 窗给 eval(冠军 8/32) = fit/eval 同分布: 【每个域块各出一份】而不是按行号切
FR=""; ER=""
while read -r d off cnt; do
  case "$d" in \#*|win|"") continue;; esac
  nw=$(( cnt / W )); [ "$nw" -lt 2 ] && nw=2
  skip=$(( W / 4 ))
  nev=$(( nw / 4 )); [ "$nev" -lt 1 ] && nev=1
  for w in $(seq 0 $((nw-1))); do
    x=$(( off + w*W + skip )); y=$(( off + (w+1)*W ))
    [ "$y" -gt $(( off + cnt )) ] && y=$(( off + cnt ))
    [ "$x" -ge "$y" ] && continue
    if [ "$w" -lt $(( nw - nev )) ]; then FR="${FR:+$FR,}$x:$y"; else ER="${ER:+$ER,}$x:$y"; fi
  done
done < "$LAY"
[ -n "$FR" ] && [ -n "$ER" ] || { LOG "★行掩码算空(布局有问题)"; exit 1; }
LOG "行掩码 ← $LAY: win=$W, fit $(echo "$FR"|tr , '\n'|wc -l) 段 / eval $(echo "$ER"|tr , '\n'|wc -l) 段"
# ③zlayer 全家 43 层(干净锚; XC 非空=第7参引擎捕获)
# zlayer=C 版(2026-08-25 迁移 Wave B 一期, 金标 migrate/golden.txt: z支线与py精确一致/
# GE=py-CPU路真解口径(py-GPU路 XCAP 下 GE 恒死是 py 自身分裂, 见 zlayer_transcription_notes #1)/
# 注入产物 rec_fidelity 复评一致)。XANCHOR/GGUF/ADDON 等二期模式 C 版硬拒(响亮失败, 无静默兜底)。
# ⚠二期欠账: C 版纯 CPU ~570s/层(py-GPU 54s), 43层≈6.8h — 按"spark重计算必须GPU化"铁律须补 CUDA 路。
ZLB="$HOME/ds4-main/gguf-tools/amp/zlayer"
# 构建收进 gguf-tools/Makefile(批1): 平台特判(Accelerate/scipy_openblas/CUDA)都在那边。
[ -x "$ZLB" ] || make -C "$HOME/ds4-main/gguf-tools" zlayer
for L in ${LRANGE:-$(seq 0 42)}; do
  env DS4_ZL_NTOK=8192 DS4_ZL_NFIT=6144 ${XA:+DS4_ZL_XANCHOR=$XA} DS4_ZL_FIT_RANGES="$FR" DS4_ZL_EV_RANGE="$ER" \
  "$ZLB" "$DS4_HF" $D2/$WS/layers "$ANC" $L ${K:-1024} 1 ${XC:+$D2/$XC} \
    2>&1 | grep -aE "XCAP|Error|assert|★" || { LOG "★L$L 失败★"; exit 1; }
  # 进度可观测铁律: 每层收官打一行(tail -f 就能看到 43 层推进)
  LOG "L$L ✓ $(ls "$D2/$WS/layers"/zrec_L*.bin 2>/dev/null | wc -l | tr -d ' ')/43 K=$K"
done
rm -f $D2/$WS/layers/zcache_L*.npz
LOG "③反修收官"
# ④官方语料五指标终判(单层验证 LRANGE 在场时跳过 — 10 分钟级铁律)
[ -n "$LRANGE" ] && { LOG "④跳过(单层验证)"; exit 0; }
bash gguf-tools/scripts/caliper_ref.sh $D2/$WS/layers /tmp/qc_${WS}_wt2.bin > /tmp/caliper_${WS}.log 2>&1
grep -E "PPL\(stu|Mean KLD|RMS|Same top|Δp" /tmp/caliper_${WS}.log
LOG "④五指标终判完成"
