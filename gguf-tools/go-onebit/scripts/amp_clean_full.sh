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
cd $D2/vq86h_noz/layers
for f in dql_vq_L*.bin; do ln -f "$f" "$HOME/ds4-main/$D2/$WS/layers/$f" 2>/dev/null || cp "$f" "$HOME/ds4-main/$D2/$WS/layers/"; done
cp dql_ops_L*.bin opt_L*.bin manifest.txt "$HOME/ds4-main/$D2/$WS/layers/" 2>/dev/null
cp dql_L*.bin "$HOME/ds4-main/$D2/$WS/layers/"
cd ~/ds4-main
LOG "②工作区就绪"
# 行掩码(2026-08-24 拼接毒定罪: 256块互织语料每块前64行=异域上下文污染行, z被毒死;
# 剔污染行后 z 复活 L3 7.2%/组合11.7%=连续锚同档) — 复用 08-09 拼接修正既有开关
FR=""; ER=""
for b in $(seq 0 31); do
  seg="$((b*256+64)):$(( (b+1)*256 ))"
  if [ "$b" -lt 24 ]; then FR="${FR:+$FR,}$seg"; else ER="${ER:+$ER,}$seg"; fi
done
# ③zlayer 全家 43 层(干净锚; XC 非空=第7参引擎捕获)
for L in $(seq 0 42); do
  env DS4_ZL_NTOK=8192 DS4_ZL_NFIT=6144 ${XA:+DS4_ZL_XANCHOR=$XA} DS4_ZL_FIT_RANGES="$FR" DS4_ZL_EV_RANGE="$ER" \
  python3 -u gguf-tools/go-onebit/zlever/zlayer.py "$DS4_HF" $D2/$WS/layers "$ANC" $L ${K:-1024} 1 ${XC:+$D2/$XC} \
    2>&1 | grep -aE "XCAP|★" || { LOG "★L$L 失败★"; exit 1; }
done
rm -f $D2/$WS/layers/zcache_L*.npz
LOG "③反修收官"
# ④官方语料五指标终判
bash gguf-tools/go-onebit/scripts/caliper_ref.sh $D2/$WS/layers /tmp/qc_${WS}_wt2.bin > /tmp/caliper_${WS}.log 2>&1
grep -E "PPL\(stu|Mean KLD|RMS|Same top|Δp" /tmp/caliper_${WS}.log
LOG "④五指标终判完成"
