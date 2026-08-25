#!/bin/bash
# c86_spark.sh — c86 战役薄启动器(2026-08-20 用户令: ①86G 设计=只压专家, backbone/attn/embd
# 不动 ②反修只要纯z=z变量+四损失+感知(无 AMP/RTE/GE) ③语料=开源全场景小体积(datav3=cal12 族)
# ④速度要求不变 ⑤量化/反修 ~20s/层)。
# 底座=cal12 层(datav3 量化 43/43 已在盘 → 量化段幂等秒过; 重跑复现值 cal10 24s/层·cal11
# 36s/层, GPU 常驻量化器) → 合并 ds4-cal12.gguf(~90G, 幂等) → 引擎裸捕获链态锚 → 纯z 4-lane
# (吞吐 ~19s/层) → 五指标判决(cal12z+wt2, 裸/+z/旧离线口径z对照) → 速度 + 328 题 4 并发。
# 产物全落 gguf/go-onebit/r30/c86/; dql 权重经 symlink 原地读, 不污染 cal12。
set -uo pipefail
ROOT="$HOME/ds4-main"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
OUT="$R30/c86"
mkdir -p "$OUT/layers"
# 量化权重零拷贝入战役目录: zlayer 原生模式读 dql_vq blob(=合并 GGUF 同源字节, 口径零差),
# dql/ops/opt 只为 ADDON 等旁路; zrec/zcache 落本目录(战役隔离, 不污染 cal12)
for f in "$R30"/cal12/layers/dql_vq_L*.bin "$R30"/cal12/layers/dql_L*.bin \
         "$R30"/cal12/layers/dql_ops_L*.bin "$R30"/cal12/layers/opt_L*.bin; do
    [ -e "$f" ] && ln -sf "$f" "$OUT/layers/$(basename "$f")"
done

# 量化段(幂等: cal12 43/43 在则秒过; 配方=datav3 cal12q 2048 tok, 平权 experts-only)
export Q86_IDS="$G7/cal12q.ids" Q86_S=2048 Q86_NFIT=1638
export Q86_ANCHOR="$R30/anchor_cal12_s2048.bin" Q86_OUT="$R30/cal12"
export RPLAN86="$R30/rplan_base86p.txt" VOLB86=95
# 合并段(dql 内嵌链不抽取: INJ=2 外挂时代)
export M86_LAYERS="$R30/cal12/layers" M86_ZCH=""
# 战役主参: 反修=乘性放大器设计(2026-08-20 用户令; 加性纯z首版存档 layers/rrr_v1/),
# RTE 空, 全场景 datav3 锚, held 对照=cal12 旧离线口径侧车
export F86_OUT="$OUT" F86_MDL="$ROOT/gguf/ds4-cal12.gguf" F86_RTE="" F86_AMP=1
export F86_FPA="$R30/anchor_cal12z_s2048.bin" F86_IDS="$G7/cal12z.ids"
export F86_XZC="$R30/cal12/zchain_cal12_pz.bin"
# ★每次重头跑(2026-08-20 用户铁律"每次脚本都要重头跑, 不然怎么知道你哪里又改错了"):
# all 一律先清全部反修产物, 断点续跑只在单段重入时用
if [ "${1:-all}" = all ]; then
    echo "[c86 $(date +%H:%M:%S)] 反修产物全清(重头跑)"
    rm -f "$OUT"/layers/zrec_L*.bin "$OUT"/layers/zrec_amp_L*.bin \
          "$OUT"/layers/zcache_L*.npz "$OUT"/layers/amp_L*.out "$OUT"/layers/zl_L*.out \
          "$OUT"/zchain_*.bin "$OUT"/anchor_chain.bin
    rm -rf "$OUT/mixrec" "$OUT/layers/rrr_v1"
fi
exec bash "$ROOT/gguf-tools/go-onebit/scripts/amp86z_spark.sh" "${1:-all}"
