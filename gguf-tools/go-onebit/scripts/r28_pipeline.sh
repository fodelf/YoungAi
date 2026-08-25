#!/bin/bash
# r28_pipeline.sh — R28 v2 收官后自动衔接(2026-07-31): 量化完成 → 反修(内存架构已改)
#   → 合并(验证后删源) → 生成冒烟 → 判决。全程日志 /tmp/r28_pipe.log,每阶段可单独重入。
# 铁律遵守: 每阶段前置检查(层齐/盘/无并发), 看门狗常驻, 失败即停不静默降级。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
R28="$ROOT/gguf/go-onebit/r28"
OUTF="$R28/full"
LOG(){ echo "[pipe $(date +%H:%M:%S)] $*" >&2; }

# ---- 阶段1: 等量化收官 ----
LOG "等量化收官(43 层)…"
until [ "$(ls "$OUTF"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')" = 43 ] && ! pgrep -f "ds4quant_run .*rr_calib" >/dev/null; do
    sleep 30
done
LOG "量化收官 ✓ 43 层齐"
[ -f "$OUTF/route_bias_r28.bin" ] && LOG "路由 Δb 落盘 ✓ ($(ls -l "$OUTF/route_bias_r28.bin" | awk '{print $5}') B)" \
    || LOG "★警告: route_bias_r28.bin 未落盘"

# ---- 阶段2: 反修(终局收敛 sweep 是 KL 主修器) ----
# ★已有反修在跑就等它, 不要自己再起一个: 否则会在反修没跑完时
if pgrep -f "ds4quant_run .*rr_calib" >/dev/null; then
    LOG "检出已在跑的反修 — 等它收官(不并发起第二个)"
    while pgrep -f "ds4quant_run .*rr_calib" >/dev/null; do sleep 60; done
    LOG "已有反修结束 ✓"
    BFRC=0
else
    LOG "★反修驱动脚本已随 mvq 战役族移除 — 用 r30_campaign.sh stage_backfit 同款 env 手动起(历史脚本, 不再直接可跑)★"
    BFRC=9
fi
LOG "反修 rc=$BFRC(非0=看门狗/异常, 层文件仍完好, 继续合并)"

# ---- 阶段3: 合并(VQ 路线用 vq_merge_v4: R28 产物是 VQ 侧车) ----
pgrep -f "ds4quant_run" >/dev/null && { LOG "仍有量化进程, 停"; exit 3; }
rm -f "$ROOT/gguf/go-onebit/ds4-r28.gguf"
LOG "预扫 blob 真尺寸"
"$(dirname "$0")/../calib/vq_blob_truesize" "$OUTF/layers" \
    > "$OUTF/blob_sizes.txt" 2>/tmp/r28_truesize.log || { LOG "预扫失败"; exit 4; }
# ★--no-down(2026-07-31): R28 计划表 43/43 层 w2dim=16 ⇒ 冷 w2 的 VQ 载荷在 blob 的
#   which=2 槽里, base ffn_down_exps(43×0.2656=11.42 GiB)是纯死重。带上它 = 38.74 GiB,
#   省掉 = 27.32 GiB。引擎侧由 ds4.c routed_down_shadow() 合成影子张量顶上。
# ★不传 --route-bias: 骨架抄自冠军 v4bf, exp_probs_b 里已烘焙 2.5·Δb_champ
#   (skel_bias_probe 实测 k=+2.27 r=+0.94)。在它上面再加 R28 的 Δb = 两份修正叠加,
#   而冠军那份是针对冠军量化误差的、对 R28 是噪声。故合并阶段不碰 bias, 交给下面
#   rebake: 先减冠军 2.5·Δb_v4fix, 再按实测最优 α 加 R28 自己的 Δb。
LOG "起合并(VQ; 骨架=冠军抽取, --no-down, bias 留给 rebake)"
"$(dirname "$0")/../quant/vq_merge_v4" --merge --no-down \
    --skeleton "$ROOT/gguf/go-onebit/r28_skeleton.gguf" \
    --blob-sizes "$OUTF/blob_sizes.txt" \
    --dql-host 127.0.0.1 --dql-dir "$OUTF/layers" \
    --out "$ROOT/gguf/go-onebit/ds4-r28.gguf" >/tmp/r28_merge3.log 2>&1
MRC=$?
[ $MRC -eq 0 ] || { LOG "合并失败 rc=$MRC — 停(层文件保留)"; tail -5 /tmp/r28_merge3.log >&2; exit $MRC; }
GIB=$(ls -l "$ROOT/gguf/go-onebit/ds4-r28.gguf" | awk '{printf "%.2f", $5/1073741824}')
LOG "合并 ✓ ${GIB} GiB"
# 体积闸: 27.32 期望值 ±1; 超 30 = down 混进来了(--no-down 失效), 停下别浪费后续时间
awk -v g="$GIB" 'BEGIN{exit !(g>30.0)}' && { LOG "★体积 ${GIB} GiB >30 — base down 混入, 停★"; exit 7; }

# ---- 阶段3.5: 路由三步(减冠军 → 快照裸态 → 按 α 绝对写) ----
# 拆三步而不是一次 rebake, 是为了让 α 可反复扫: route_bias_rebake 是增量的(-=/+=),
# 连调两次会叠加; route_alpha_set 从裸态快照绝对重写, 幂等。
MDL="$ROOT/gguf/go-onebit/ds4-r28.gguf"
CHRB="$ROOT/gguf/go-onebit/g7/route_bias_v4fix.bin"
RB28="$OUTF/route_bias_r28.bin"
rm -f "$MDL.bias0.bin"                         # 新模型 ⇒ 旧快照作废
LOG "路由① 减冠军 2.5·Δb_v4fix(骨架抄自 v4bf, 实测 k=+2.27 r=+0.94)"
"$(dirname "$0")/../calib/route_bias_rebake" \
    "$MDL" "$CHRB" 2.5 "$RB28" 0.0 >&2 \
    || { LOG "★减冠军失败 — 路由仍是冠军的, 停★"; exit 8; }
LOG "路由② 快照裸态"
"$(dirname "$0")/../calib/route_alpha_set" \
    "$MDL" "$RB28" 0.0 --snapshot-only >&2 || exit 9
LOG "路由③ 写 α=${RB_ALPHA:-2.5}(冠军定标起点; 最终值由合并后实扫定)"
"$(dirname "$0")/../calib/route_alpha_set" \
    "$MDL" "$RB28" "${RB_ALPHA:-2.5}" >&2 || exit 9

# ---- 阶段4: 生成冒烟(判决: 输出是否恢复正常) ----
LOG "生成冒烟"
env DS4_ZCHAIN="$OUTF/zchain.bin" DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 \
    "$ROOT/ds4" -m "$ROOT/gguf/go-onebit/ds4-r28.gguf" --ctx 8192 \
    -p "写一个Go函数,计算两个整数之和" -n 64 --temp 0 > /tmp/r28_smoke_final.out 2>/tmp/r28_smoke_final.log
LOG "★冒烟原始输出:"
head -c 500 /tmp/r28_smoke_final.out >&2
echo >&2

# ---- 阶段5: α 扫(SWEEP_ALPHA=1 开; 按真实生成质量定 α, 不靠 relL2) ----
# 为什么在这里扫而不是量化器里: teacher-forced 的 relL2/Σmin 判不出自由生成退化
# (fable5 07-31 09:30 exposure-bias 实锤), 而 α 修的正是自回归路由漂移 —— 只有真实
# 生成看得见。route_alpha_set 幂等, 每个 α 都从裸态重写, 秒级, 不碰层文件。
if [ "${SWEEP_ALPHA:-0}" = 1 ]; then
    for A in 0 1 2.5 4; do
        "$(dirname "$0")/../calib/route_alpha_set" "$MDL" "$RB28" "$A" >&2 || continue
        env DS4_ZCHAIN="$OUTF/zchain.bin" DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 \
            "$ROOT/ds4" -m "$MDL" --ctx 8192 \
            -p "写一个Go函数,计算两个整数之和" -n 64 --temp 0 \
            > "/tmp/r28_alpha_$A.out" 2>"/tmp/r28_alpha_$A.log"
        LOG "★α=$A 原始输出:"
        head -c 400 "/tmp/r28_alpha_$A.out" >&2
        echo >&2
    done
    LOG "α 扫完毕 — 各档原始输出在 /tmp/r28_alpha_*.out, 由主控判决后写回最终 α"
fi
LOG "流水线收官(基准由主控按可用性判决后手动放行)"
