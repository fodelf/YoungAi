#!/bin/bash
# r28v2_campaign.sh — R28 v2 全自动战役(2026-08-01 用户令"从头重新量化/反修/合并/后训练/
#   双机流水线/40题基准, 28g 目标, 不要打断, 明早要报告")。
#
# 与 v1 的三处根本差异(v1 实测 31.28 GiB 超标 11.7% 且无人可见):
#   ① 计划表由 rplan_solve.py 按【真实布局公式】反解 → 每层字节事前确定, 合计 27.948 GiB
#   ② 量化器逐层报体积 + 落 manifest.txt + DS4_VOL_BUDGET_GIB 硬闸(超预算当场 exit 9)
#   ③ 反修不再重写 VQ 载荷(权重未变) ⇒ 体积不漂移 + 反修提速一倍
# 合并端读 manifest, 不再用 vq_blob_truesize.py 反解段结构(它认不全新档位, 曾低估 833MiB)。
#
# 阶段可单跑: r28v2_campaign.sh [quant|backfit|merge|smoke|bench|all]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SC="$ROOT/gguf-tools/go-onebit/scripts"
G7="$ROOT/gguf/go-onebit/g7"
R28="$ROOT/gguf/go-onebit/r28v2"
OUTF="$R28/full"
# ★配置 JSON 是唯一真源(2026-08-01 用户令"量化脚本接入配置 json, 根据 json 配置量化")★
# 计划表 .txt 由 JSON 每次派生, 不手改、不复用旧的 —— 避免"JSON 改了但 txt 是老的"这类
# 双份真相事故。派生时用 vq_qc.h 同一套公式重算每层字节并与 JSON 对账, 手改过 hot/档位
# 却没更新 bytes 的 JSON 会当场被抓, 而不是跑完几小时才发现。
PLAN_JSON="${PLAN_JSON:-$ROOT/gguf/go-onebit/plan/r28_rplan_v4.json}"
RPLAN="$OUTF/rplan.txt"          # 每次从 JSON 派生到本轮产物目录, 与产物同代
MDL="$ROOT/gguf/go-onebit/ds4-r28v2.gguf"
BUDGET="${VOL_BUDGET:-19.80}"        # 载荷预算 GiB; + backbone 8.202 = 27.95
LOG(){ echo "[r28v2 $(date +%H:%M:%S)] $*" >&2; }

WDOG(){ while true; do
    P=$(pgrep -nf "ds4quant_run .*rr_calib" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(footprint -p "$P" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
         | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
    [ -n "${MB:-}" ] && [ "$MB" -gt 11900 ] && { echo "[r28v2][wdog] ${MB}MB >11.9G 杀" >&2; kill -9 "$P" 2>/dev/null || true; }
    sleep 5; done }

stage_quant(){
    [ -f "$PLAN_JSON" ] || { LOG "配置 JSON $PLAN_JSON 缺"; exit 2; }
    FREE=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
    [ "$FREE" -ge 26 ] || { LOG "★盘闸 free ${FREE}G <26G 停★"; exit 6; }
    pgrep -x ds4quant_run >/dev/null && { LOG "已有量化进程"; exit 3; }
    rm -rf "$OUTF"; mkdir -p "$OUTF/layers" "$OUTF/ckpt"
    # JSON → 计划表(含校验+对账); 预算闸同时卡在这里, 超了直接拒跑
    LOG "从配置 JSON 派生计划表: $(basename "$PLAN_JSON")"
    python3 "$SC/plan_json2rplan.py" "$PLAN_JSON" "$RPLAN" --budget-gib "$BUDGET" >&2 \
        || { LOG "★配置 JSON 校验不通过, 拒跑★"; exit 7; }
    cp "$PLAN_JSON" "$OUTF/plan_config.json"     # 产物自带当代配置, 事后可追溯
    cd "$ROOT/gguf-tools/go-onebit/quant"
    export DS4_ANCHOR="$G7/ds4quant_anchor_v5mini_s1716.bin" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-6}"
    export DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_MV_COAD_BASE=1 DS4_TUNE=1 DS4_COADAPT=1
    export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$RPLAN"
    export DS4_VOL_BUDGET_GIB="$BUDGET"          # ★体积硬闸: 超预算当场停, 不再跑完才发现
    export DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt"
    # 量化阶段每层 vq_rplan(L)+hot_from_anchor(L,S,g_vq_hot) 会用计划表 hot 覆盖上面这张表, 故体积由计划表定。
    export DS4_ROUTE_BIAS_FIT=1 DS4_ROUTE_BIAS_OUT="$OUTF/route_bias_r28.bin" DS4_ROUTE_BIAS_ALPHA=1.0
    export DS4_PLAN="$OUTF/plan.txt" DS4_CKPT_DIR="$OUTF/ckpt"
    export DS4_LAYER_DIR="$OUTF/layers" DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin"
    unset DS4_MV_PROBE_L DS4_MINVOL_MAXL \
          DS4_MINVOL_HIST DS4_MINVOL_FLOOR DS4_VQ_COLD_DIM DS4_VQ_COLD_NC \
          DS4_MV_FLOOR_LINE DS4_RR_IDS DS4_ANCHOR2 DS4_MINVOL_TARGET DS4_BWD 2>/dev/null || true
    LOG "量化起跑 43 层(计划表 v2, 预算 ${BUDGET} GiB 载荷)"
    ./ds4quant_run "$G7/rr_calib_prog_v5mini.ids" 1716
    LOG "量化 rc=$?"
}

stage_backfit(){
    N=$(ls "$OUTF"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    [ "$N" = 43 ] || { LOG "层文件 $N/43 不齐, 拒反修"; exit 2; }
    cd "$ROOT/gguf-tools/go-onebit/quant"
    export DS4_ANCHOR="$G7/ds4quant_anchor_v5mini_s1716.bin" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-4}"
    export DS4_LAYER_DIR="$OUTF/layers" DS4_LCFG=$(printf 'g%.0s' $(seq 1 43)) DS4_COADAPT=1
    export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$RPLAN"
    export DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt"
    export DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin"
    export DS4_BWD=1 DS4_GSWEEP=3 DS4_BF_JUSTIFIED=1 DS4_BF_TERM_MAXP=1 DS4_BF_MEMGB="${BF_MEMGB:-1}"
    # ★路由反修同环(用户令"反修路由要跟其他反修一起, 不然是错误叠加"): 每层定稿前 FIT Δb
    #   + α{1,2.5,4} 三点扫, 选层优即时生效于下游层。与 ANCHOR_ROUTE 互斥, 必须关。
    # ★路由 FIT 必须走【非序贯】口(2026-08-01 实锤)★ 门控在 ds4quant_run.c:2603
    #   if((RB_SEQ||getenv("DS4_ROUTE_BIAS_FIT")) && … && !(RB_SEQ && g_rb_fit_L<0))
    # 反修路径下 g_rb_fit_L 恒为 -1(序贯 FIT 只在 minvol 贪心段调用), 若设 DS4_ROUTE_SEQ
    # 则 RB_SEQ=1 ⇒ 最后一项为假 ⇒ ★Δb 统计被整个跳过★(实测 4 层 路由FIT 0 次, 而 L03
    # 路由一致已掉到 89.8% = 确实在漂移)。α 扫代码也只存在于 minvol 贪心段, 反修够不到。
    # 正解(代码注释原文): DS4_ROUTE_BIAS_FIT 走非序贯统计(既有前向里顺带累计, 零额外前向),
    # 落盘同格式, α 由烘焙侧在合并时给(冠军同款 2.5)。
    # ★反修必须在【修正后的路由】上进行(2026-08-01 用户令"反修只走一次, 后面再合并路由
    #   不是又偏移了")★ 顺序错误会让整轮反修作废:
    #   错: 量化 → 反修(在漂移路由 82% 上修权重)→ 合并烘 Δb(路由被拉回)⇒ 反修的 z/侧车
    #       是对着旧路由解的, 输入分布一变全部失配 = 白跑
    #   对: 量化(收 Δb) → 反修【带 Δb 应用】(路由=部署态)→ 合并只做烘焙(路由不再变)
    # 入口 ds4quant_run.c:2580 DS4_ROUTE_BIAS=<文件> 懒加载, 每层 gate 打分用 gbias+α·Δb,
    # 与合并后 exp_probs_b 烘焙的效果逐位等价 ⇒ 反修看到的就是最终部署路由。
    # α 用冠军定标 2.5(fable5 实测峰位: 1.5→80.3 / 2.0→81.6 / 2.5→84.2 / 3.0→81.6 回落)。
    export DS4_ROUTE_BIAS="$OUTF/route_bias_r28.bin" DS4_ROUTE_BIAS_ALPHA="${RB_ALPHA:-2.5}" DS4_ROUTE_BIAS_MINCNT=8
    # ★量化 env 全清扫(2026-08-01 事故修)★ all 模式下 stage_quant 与 stage_backfit 在
    # 【同一进程】顺序执行, 量化的 export 会全部继承进反修。首犯是 DS4_TUNE:
    #   ds4quant_run.c:5470  if(getenv("DS4_TUNE")){ 渐进调优…; return 0; }
    #                 5479  fprintf("量化遍 (LCFG=%s)")  ← 反修真正要走的分支
    # DS4_TUNE 在 LCFG 分支【之前】截胡并 return, 反修代码一行没跑, 日志表现为
    # "[渐进调优v2·目标导向] / L00 ★选 g10", 且没有 "[反修] HQE 快照"。
    # 上一轮反修是独立进程启动, 无继承, 所以正常 —— 这不是 VQ 不支持反修。
    # 教训同 fable5 rr 污染事故: 语义反转型 env 必须整族清扫, 逐个 unset 永远漏。
    unset DS4_TUNE DS4_MINVOL DS4_MV_BASELINE DS4_MV_COAD_BASE \
          DS4_MV_PROBE_L DS4_MINVOL_MAXL \
          DS4_MINVOL_HIST DS4_MINVOL_FLOOR DS4_MINVOL_TARGET DS4_MINVOL_ALPHA \
          DS4_VQ_RPLAN DS4_VOL_BUDGET_GIB DS4_PLAN DS4_CKPT_DIR \
          DS4_ANCHOR_ROUTE DS4_ROUTE_SEQ DS4_ROUTE_BIAS_FIT DS4_ROUTE_BIAS_OUT \
          DS4_RR_IDS DS4_ANCHOR2 DS4_EXPORT_GGUF DS4_REPAIR_COLD DS4_FP_ONLY 2>/dev/null || true
    # 起跑前自检: 这些一旦在场, 反修会走错分支
    for v in DS4_TUNE DS4_MINVOL DS4_MV_BASELINE DS4_VQ_RPLAN; do
        [ -z "$(eval echo \$$v)" ] || { LOG "★$v 仍在场, 反修会走错分支 — 停★"; exit 8; }
    done
    LOG "反修起跑(序贯路由入环, VQ 载荷保留不重编码)"
    ./ds4quant_run "$G7/rr_calib_prog_v5mini.ids" 1716
    LOG "反修 rc=$?"
}

stage_merge(){
    pgrep -x ds4quant_run >/dev/null && { LOG "仍有量化进程, 停"; exit 3; }
    MAN="$OUTF/layers/manifest.txt"
    [ -f "$MAN" ] || { LOG "manifest 缺 — 量化未用新版量化器?"; exit 4; }
    TOT=$(awk '{s+=$2} END{printf "%.3f",s/1073741824}' "$MAN")
    LOG "manifest ${TOT} GiB / $(wc -l < "$MAN") 层 → 预期模型 $(awk -v t="$TOT" 'BEGIN{printf "%.2f",t+8.202}') GiB"
    rm -f "$MDL" "$MDL.bias0.bin"
    python3 "$ROOT/gguf-tools/go-onebit/quant/vq_merge_v4.py" --merge --no-down \
        --skeleton "$ROOT/gguf/go-onebit/r28_skeleton.gguf" \
        --blob-sizes "$MAN" \
        --dql-host 127.0.0.1 --dql-dir "$OUTF/layers" \
        --out "$MDL" >/tmp/r28v2_merge.log 2>&1
    MRC=$?
    [ $MRC -eq 0 ] || { LOG "合并失败 rc=$MRC"; tail -5 /tmp/r28v2_merge.log >&2; exit $MRC; }
    GIB=$(ls -l "$MDL" | awk '{printf "%.2f",$5/1073741824}')
    LOG "合并 ✓ ${GIB} GiB"
    awk -v g="$GIB" 'BEGIN{exit !(g>29.0)}' && { LOG "★体积 ${GIB} >29 GiB — 停★"; exit 7; }
    # 路由: 减冠军 2.5·Δb(骨架抄自 v4bf, 实测 k=+2.27 r=+0.94) → 快照裸态 → 写 α
    CHRB="$G7/route_bias_v4fix.bin"; RB="$OUTF/route_bias_r28.bin"
    if [ -f "$CHRB" ] && [ -f "$RB" ]; then
        python3 "$ROOT/gguf-tools/go-onebit/scripts/route_bias_rebake.py" "$MDL" "$CHRB" 2.5 "$RB" 0.0 >&2 || LOG "减冠军偏置失败"
        python3 "$ROOT/gguf-tools/go-onebit/scripts/route_alpha_set.py" "$MDL" "$RB" 0.0 --snapshot-only >&2 || true
        python3 "$ROOT/gguf-tools/go-onebit/scripts/route_alpha_set.py" "$MDL" "$RB" "${RB_ALPHA:-2.5}" >&2 || true
        LOG "路由偏置 α=${RB_ALPHA:-2.5} 已烘焙"
    fi
}

stage_smoke(){
    LOG "生成冒烟(单机 GPU 路)"
    env DS4_ZCHAIN="$OUTF/zchain.bin" DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_VQ_GPU=1 \
        "$ROOT/ds4" -m "$MDL" --ctx 8192 -p "写一个Go函数,计算两个整数之和" -n 96 --temp 0 \
        > /tmp/r28v2_smoke.out 2>/tmp/r28v2_smoke.log
    LOG "★冒烟原始输出:"; cat /tmp/r28v2_smoke.out >&2; echo >&2
}

case "${1:-all}" in
    quant)   WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_quant ;;
    backfit) WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_backfit ;;
    merge)   stage_merge ;;
    smoke)   stage_smoke ;;
    all)     WDOG & trap 'kill %1 2>/dev/null||true' EXIT
             stage_quant && stage_backfit; kill %1 2>/dev/null||true
             stage_merge && stage_smoke
             LOG "战役收官 — 双机/基准由主控接管" ;;
    *) echo "用法: r28v2_campaign.sh [quant|backfit|merge|smoke|all]" >&2; exit 1 ;;
esac
