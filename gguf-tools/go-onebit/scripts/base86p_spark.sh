#!/bin/bash
# base86p_spark.sh — 平权基座对照战役: 检验"九宫格冷专家1bit是基座输官方q2的主因"。
# 单变量对照: M10裸底(cal9+九宫格)=0.4956 vs base86p(cal9+全员2.25bit) vs 官方 0.4207。
# 链: 等cal9锚 → 让位(杀rb86) → stage_quant86(平权rplan) → wt2裸判。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"
CAL9_ANCHOR="$R30/anchor_cal9_s2906.bin"
LOG(){ echo "[base86p $(date +%H:%M:%S)] $*"; }

# ★2026-08-19 参数化(cal10 开源语料战役复用): Q86_*/RPLAN86/VOLB86/JUDGE_DUMP 全部可 env
# 覆盖, 默认=cal9 原语义; 锚缺失时 FP 前向现造(rb86 外部造锚时代结束)。
export Q86_IDS="${Q86_IDS:-$G7/wt2train_cal9.ids}" Q86_S="${Q86_S:-2906}" Q86_NFIT="${Q86_NFIT:-2906}"
export Q86_ANCHOR="${Q86_ANCHOR:-$CAL9_ANCHOR}" Q86_OUT="${Q86_OUT:-$R30/base86p}"
export RPLAN86="${RPLAN86:-$R30/rplan_base86p.txt}" VOLB86="${VOLB86:-95}" DS4_THREADS=6   # GPU 常驻版: 6 流大 kernel(20 流小 kernel 队列争用在案)
JUDGE_DUMP="${JUDGE_DUMP:-/tmp/base86p_wt2.bin}"

if ! { [ -f "$Q86_ANCHOR" ] && python3 "$SC/anchor_metrics.py" --ref "$Q86_ANCHOR" --ids "$Q86_IDS" >/dev/null 2>&1; }; then
    LOG "锚缺, FP 前向现造 S=$Q86_S → $Q86_ANCHOR"
    cd "$ROOT/gguf-tools/go-onebit/quant"
    DS4_HF="${DS4_HF:-$ROOT/hf/DeepSeek-V4-Flash-0731}" \
    DS4_FP_ONLY=1 DS4_ANCHOR="$Q86_ANCHOR" DS4_THREADS=20 OPENBLAS_NUM_THREADS=1 \
        ./ds4quant_run "$Q86_IDS" "$Q86_S" || { LOG "★锚捕获失败★"; exit 3; }
    cd "$ROOT"
    [ -f "$Q86_ANCHOR" ] || { LOG "★锚没落盘★"; exit 3; }
fi
LOG "锚 ✓"
tmux kill-session -t rb86 2>/dev/null; sleep 2
P=quant; pkill -9 -f "ds4${P}_run" 2>/dev/null; sleep 2
export DS4_BF_MEMGB=80 DS4_VQ_TIMING=1
export DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512   # 每专家校准行帽(08-18): H/g_r ∝行数, 512=4×过采样质量安全
# mmap 锁争用根修(2026-08-18 wchan 实锤 vm_mmap_pgoff/__vm_munmap): 专家循环反复 malloc/free
# 大缓冲走 mmap 路径, 20 线程抢进程 mmap 写锁 → CPU 只吃一半。强制大分配走堆复用。
export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824   # spark 121G: fp16 权重缓存放宽(默认小阈=每层重读 HF, 线程全 D 态 IO 等待, 9分/层实锤)
export OPENBLAS_NUM_THREADS=1
LOG "量化发车(平权 2.25bpw × 43 层)"
bash "$SC/r30_campaign.sh" quant86 || { LOG "★量化失败★"; exit 2; }
LOG "量化收官(43/43)"
[ "${1:-all}" = quant ] && { LOG "quant 子命令: 到此暂停(裸判待令)"; exit 0; }

LOG "wt2 裸判"
LCx=$(printf 'g%.0s' $(seq 1 43))
cd "$ROOT/gguf-tools/go-onebit/quant"
env DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731" OPENBLAS_NUM_THREADS=1 DS4_BF_MEMGB=8 \
    DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 DS4_CALIB_FULLSET=1 \
    DS4_EXPORT_BYTES=0 DS4_ANCHOR="$R30/anchor_wt2_s2653.bin" DS4_NFIT=1 DS4_THREADS=20 \
    DS4_LAYER_DIR="$Q86_OUT/layers" DS4_LCFG="$LCx" DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
    DS4_DUMP_LOGITS="$JUDGE_DUMP" ./ds4quant_run "$G7/wt2.ids" 8000 2>&1 | tail -2
cd "$ROOT"
echo "══ $(basename "$Q86_OUT") 裸判 wt2 五指标(对表: M10裸底 0.4956 | 官方q2 0.4207) ══"
python3 "$SC/anchor_metrics.py" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
    --student "$JUDGE_DUMP" --tail 5
LOG "战役收官"
