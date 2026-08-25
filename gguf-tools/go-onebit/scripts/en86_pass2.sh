#!/bin/bash
# en86_pass2.sh — α第二段: 干净EN锚 ADDON 叠加(2026-08-14)。
# 背景: EN殿后摆位污染案(因果注意力: EN行看见4683异域前缀→侧车学污染英文, 判决干净英文误开火)。
# 本段: 纯wikitext-train语料(EN在位置0零前缀) S=2383 → FP锚+链锚 → stage_addon Δ解算合并注入
#       (单调门Δ≤0.2%保留原记录=编程侧默认安全) → 双盲判。
# 门: 自等 α1 完成且判决过闸(wt2 KL<0.60 且 rrh Σmin>0.83), 不过即停(元凶另有其人, 等人工)。
set -u
R=/Users/fodelf/ds4-main; R30=$R/gguf/go-onebit/r30; G7=$R/gguf/go-onebit/g7
Q=$R/gguf-tools/go-onebit/quant/ds4quant_run.dchunk
AL=/tmp/en86_alpha1.log
ENIDS=$G7/wt2train_en.ids; ENS=2383; ENFIT=2083
ENA=$R30/anchor_entrain_s2383.bin; ENC=$R30/anchor_chain_entrain_s2383.bin

until grep -aq ALPHA1_ALL_DONE $AL 2>/dev/null; do sleep 60; done
# 门=官方基准单尺(2026-08-14 用户令"只跑官方量化基准"): wt2 KL 回中性即放行, rr_hard 只作参考不设门
L1=$(grep -a "分布还原率" $AL | head -1)
WKL=$(echo "$L1" | sed -E 's/.*KL\(fp‖q\)=([0-9.]+).*/\1/')
python3 -c "exit(0 if float('$WKL')<0.60 else 1)" \
  || { echo "PASS2_GATE_REJECT wt2KL=$WKL"; exit 1; }
echo "PASS2_GATE_OK wt2KL=$WKL $(date +%T)"

cd $R/gguf-tools/go-onebit/quant
if ! ( cd $R && "$(dirname "$0")/../calib/anchor_metrics" --ref $ENA --ids $ENIDS >/dev/null 2>&1 ); then
  rm -f $ENA; echo "捕干净EN FP锚 S=$ENS $(date +%T)"
  env DS4_HF=$R/hf/DeepSeek-V4-Flash-0731 DS4_FP_ONLY=1 DS4_ANCHOR=$ENA DS4_THREADS=8 "$Q" $ENIDS 9999 2>&1 | tail -2
  ( cd $R && "$(dirname "$0")/../calib/anchor_metrics" --ref $ENA --ids $ENIDS >/dev/null 2>&1 ) \
    || { echo "PASS2_EN_ANCHOR_BAD"; exit 2; }
fi
echo "EN锚 ✓ $(date +%T)"

cd $R
rm -f $ENC
env WDOG_MB=27648 ZL_SWLIM=60 QBIN_OVERRIDE=$Q \
    MD_ANCHOR=$ENA MD_CHAIN=$ENC MD_LAYERS=$R30/en86/layers \
    MD_IDS=$ENIDS MD_S=$ENS MD_NFIT=$ENFIT \
    MD_FR=0:$ENFIT MD_EV=$ENFIT:$ENS \
    bash gguf-tools/go-onebit/scripts/r30_campaign.sh addon
echo "ADDON段完 $(date +%T)"
env Q86_OUT=$R30/en86 QBIN_OVERRIDE=$Q bash gguf-tools/go-onebit/scripts/r30_campaign.sh en86judge
echo PASS2_ALL_DONE
