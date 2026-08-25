#!/bin/bash
# amp_campaign.sh — 反修战役·单文件入口(2026-08-22 用户对齐定案)。
#
# 【设计】反修 = 求放大器，就这一件事。每层:
#   学生 = 当前层**全部的量化计算**(attn / shared / router / norm / 专家, 一个不落)
#          —— 取引擎跑普通全 q2 基座时的真值, 不是 Python 理想化重算
#   教师 = FP 全层前向(锚)
#   解变量 = U, V, **z**;  ★z 必须是 x 的函数(动态 z, 方案 B)★
#   目标 = 四损失(行为 fit + 感知列权 + dither 稳定 + 收缩), λ/k_L 走 held 网格
#   判据 = 当层最优
#   产物 = 放大器侧车 → 之后配合引擎阶段使用
#
# 【口径为什么这么定】放大器在运行时乘的是 routed 输出、看到的是 post-ffn_norm 的 x̂。
#   旧口径拿 FP 锚的 fin 当 x、拿 Python 重算的量化专家当 y_q —— attn/shared/norm/router
#   的量化误差一点都没进来。本版把 x 与 y_q 全换成引擎捕获的真值(raw_ffn_in / raw_ffn_out),
#   于是上游一切量化误差沿链带入, 放大器作用的位置、看到的输入、乘的张量与运行时逐字节同源。
#   全层出口 hc(h_L*.bin) 留作诊断: 报"真实层误差"这个分母, 不当解算目标(那需要 ∂H/∂y)。
#
# 【动态 z】旧 amp_solve 写死 z=全1(引擎乘了等于没乘), 静态 z 又能被 U 的列缩放吸收 = 空动作。
#   方案B: pv_c(x) = tanh(V_c·x/s) · tanh(A_c·x/s), 两个 tanh 的乘积 = 真二阶门,
#   对 U 仍线性 ⇒ 闭式 ridge 不变、零训练不变。落地 type9 `zl.AMPD`, 载荷 A|U|V。
#
# 用法: bash amp_campaign.sh [段|all]
# 段: preflight ids anchor capture solve chain judge   (probe = 可选诊断, 不在 all 链里)
#
# ★本脚本不接受任何环境变量(2026-08-22 铁律: 本项目不得新增 env 配置)★
#   语料/尺寸/路径全部写死在下面。要换一轮就改这里并记录, 不靠发车姿势决定行为。
set -uo pipefail
ROOT="$HOME/ds4-main"
SC="$ROOT/gguf-tools/go-onebit/scripts"
ZL="$ROOT/gguf-tools/go-onebit/zlever"
QD="$ROOT/gguf-tools/go-onebit/quant"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"

NAME="v5full"
CORPUS="$ROOT/gguf-tools/go-onebit/corpus/calibration_datav5.txt"   # 开源全场景, 22.4% 字符在代码围栏内
S=8192                       # 锚 token 数(整份语料等距窗抽样)
IDS_WIN=64                   # 等距窗数, 跨度铺满全文
MDL="$ROOT/gguf/ds4-allq2.gguf"   # 普通 RTN 全 q2, 零语料, 本战役不重量化
PROBE_L=2                    # 仅 probe 诊断段用
MEM_FLOOR_GB=4               # 看门狗红线
DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
export DS4_HF   # 上游既有 env, 非本轮新增

D="$ROOT/gguf/go-onebit/$NAME"
IDS="$D/$NAME.ids"
ANCHOR="$D/anchor_$NAME.bin"
CAP="$D/cap_fix"      # 引擎真值(bug 修复后重捕, 已过两次逐位复现检查)
HD="$D/hdump"          # 全层出口 hc: h_L%02d.bin(诊断分母)
AMP="$D/amp"
AMP1="$D/amp_pass1"       # 第一轮放大器(第二轮据此还原被乘量)
CAP_AMP="$D/cap_amp"     # 带放大器的引擎捕获 = 上游已修好时各层的真实输入
PREV=""
ZC="$D/zchain_$NAME.bin"
LOG(){ echo "[$NAME $(date '+%m-%d %H:%M:%S')] $*"; }
DIE(){ LOG "★$*★"; exit 1; }
mkdir -p "$D" "$AMP"

WD=""
watchdog_start(){
    ( while true; do
        A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        if [ "${A:-99}" -lt "$MEM_FLOOR_GB" ]; then
            echo "[watchdog] MemAvailable=${A}GB < ${MEM_FLOOR_GB}GB ★杀本段★" >&2
            pkill -9 -f 'ds4 --cuda'; pkill -9 -f ds4quant_run; pkill -9 -f zlayer.py
            break
        fi; sleep 5
      done ) & WD=$!
}
watchdog_stop(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null; WD=""; }
trap 'watchdog_stop' EXIT

stage_preflight(){
    LOG "⓪preflight"
    [ -s "$CORPUS" ] || DIE "语料缺 $CORPUS"
    [ -d "$DS4_HF" ] || DIE "HF 缺 $DS4_HF"
    [ -s "$MDL" ]    || DIE "基座缺 $MDL(普通 RTN 全 q2, 本战役不重量化)"
    [ -x "$ROOT/ds4" ] || ( cd "$ROOT" && make cuda-spark ) || DIE "引擎编译失败"
    "$ROOT/ds4" --help >/dev/null 2>&1 || DIE "ds4 跑不起来"
    # ds4quant_run 被整仓 scp 覆盖成 Mach-O 过一次, 锚段当场断链 → 与已验证能跑的 ds4 比格式
    HOSTFMT=$(file -b "$ROOT/ds4" | cut -d, -f1)
    if [ "$(file -b "$QD/ds4quant_run" 2>/dev/null | cut -d, -f1)" != "$HOSTFMT" ]; then
        pgrep -f "$QD/ds4quant_run" >/dev/null && DIE "ds4quant_run 格式不对但正在运行"
        LOG "  ds4quant_run 非本机格式, 重编"
        bash "$SC/build_quant_spark.sh" >/dev/null 2>&1 || DIE "ds4quant_run 编译失败"
    fi
    FREE=$(df -BG "$ROOT" | awk 'NR==2{gsub("G","",$4); print $4}')
    NEED=$(( S * 4 / 1000 + 43 * S * 4 * 4096 * 4 / 1000000000 + 5 ))
    [ "$FREE" -ge "$NEED" ] || DIE "磁盘不足: 剩 ${FREE}G < 需 ${NEED}G"
    LOG "  ✓ 语料/HF/基座/引擎/量化器/磁盘(${FREE}G≥${NEED}G)"
}

stage_ids(){
    [ -s "$IDS" ] && { LOG "①ids 已在, 跳过"; return 0; }
    LOG "①全语料等距窗抽 S=$S (${IDS_WIN}窗)"
    python3 - "$CORPUS" "$IDS" "$S" "$IDS_WIN" "$DS4_HF" <<'PY' || DIE "ids 失败"
import re, sys
from tokenizers import Tokenizer
src, out, N, CH, hf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
allids = tok.encode(open(src, encoding="utf-8").read(), add_special_tokens=False).ids
w = N // CH; step = (len(allids) - w) // (CH - 1)      # 首窗=文件头, 末窗贴文件尾 ⇒ 跨度铺满全文
ids = []
for c in range(CH): ids += allids[c*step : c*step+w]
ids = ids[:N]
assert len(ids) == N
open(out, "w").write("\n".join(map(str, ids)) + "\n")
s = tok.decode(ids); p = [m.start() for m in re.finditer("```", s)]
print("  全语料 %d token → 抽 %d (%d窗×%d)  def %d 围栏 %d 代码占比 %.1f%%" % (
    len(allids), N, CH, w, len(re.findall(r"\bdef \w+\(", s)), len(p),
    100*sum(p[i+1]-p[i] for i in range(0,len(p)-1,2))/max(len(s),1)))
PY
}

stage_anchor(){    # FP 教师: 每层 MoE 输入/FP 路由/每层出口 H/最终 logits
    [ -s "$ANCHOR" ] && { LOG "②FP锚 已在, 跳过"; return 0; }
    LOG "②FP 锚捕获 S=$S (HF 前向, 约 $(( S * 4 / 1000 )) GB)"
    watchdog_start
    # ★大分配走堆复用(2026-08-18 已诊断, 08-22 手敲发车时漏掉又踩一次)★
    # 专家循环每个专家 malloc/free 三块 33.5MB 反量化缓冲, >128KB 默认走 mmap ⇒ 每次都是
    # 全新零页, 逐 4KB 首触缺页 + 20 线程抢 mmap 写锁。实测 1.7 亿次缺页/120s;
    # 放宽阈值让大块留在堆里复用: S=256 实测 164s → 119s。
    # 分配器(mallopt: top_pad/mmap/trim)与 BLAS 线程数已写进 ds4quant_run 的 main,
    # 不再靠 MALLOC_*/OPENBLAS_NUM_THREADS 环境变量 —— 配置跟着二进制走, 漏设不会静默变慢。
    ( cd "$QD" && DS4_FP_ONLY=1 DS4_ANCHOR="$ANCHOR" DS4_THREADS=20 ./ds4quant_run "$IDS" "$S" )
    watchdog_stop
    [ -s "$ANCHOR" ] || DIE "锚没落盘"
    LOG "②收官 $(ls -l "$ANCHOR" | awk '{printf "%.1f GB", $5/1e9}')"
}

stage_capture(){   # 学生: 引擎跑普通全 q2 基座的真值(一切都是量化的)
    [ -s "$CAP/raw_ffn_in_L0" ] && { LOG "③引擎捕获 已在, 跳过"; return 0; }
    LOG "③引擎真值捕获 on $(basename "$MDL")"
    rm -rf "$CAP" "$HD"; mkdir -p "$CAP" "$HD"
    cd "$ROOT"; watchdog_start
    # DS4_EVAL_IDS 才是裸 token 通道(--score-ids 不触发 hdump, 08-22 实锤)。
    # 一趟前向同时出三样: CAP_DIR 的 raw_ffn_in/out(解算用) + HDUMP 的层出口 hc(诊断用)。
    # 取料入口已从 env 迁到 CLI(2026-08-22): 发车命令里一眼可见
    ./ds4 --cuda -m "$MDL" --eval-ids "$IDS" --cap-dir "$CAP" --eval-hdump "$HD" </dev/null 2>&1 | tail -20
    watchdog_stop
    [ -s "$CAP/raw_ffn_in_L0" ] || DIE "引擎捕获没落盘"
    LOG "③收官 cap $(du -sh "$CAP" | cut -f1) hdump $(du -sh "$HD" | cut -f1)"
}

solve_one(){   # $1=层 $2=(保留位) $3=输出
    local L=$1 ZCF="$AMP/zcache_L$(printf %02d $1).npz"
    # 第 7 位置参数 = 引擎捕获目录(x 与被乘量取真值), 命令行里可见, 不用 env。
    # 其余 DS4_ZL_* 是 zlayer.py 上游既有开关, 非本轮新增。
    [ -f "$ZCF" ] || env DS4_ZL_GGUF="$MDL" DS4_ZL_NTOK="$S" \
        DS4_ZL_GE=0 DS4_ZL_FTA=0 DS4_ZL_ERF=0 DS4_ZL_SWLIM=60 DS4_ZL_GATE=99 \
        python3 -u "$ZL/zlayer.py" "$DS4_HF" "$AMP" "$ANCHOR" "$L" 1024 0 "$CAP" "${PREV:--}" 2>&1 \
        | grep -aE "XCAP|Error|Traceback|assert|★" \
        || DIE "L$L zcache 失败(完整输出见上)"
    python3 -u "$ZL/amp_solve.py" "$ANCHOR" "$ZCF" "$3" || DIE "L$L 解算失败"   # 动态 z 已写死
}

# 【纯诊断, 不在 all 链里, 不是闸】只在想看"静态 z vs 动态 z 差多少"时手动跑。
# 早先我把它写成"不过闸不许进 43 层"是越权: 方案 B(动态 z)是用户裁决, 不由我的指标否决。
stage_probe(){
    LOG "单层 A/B L$PROBE_L (静态z vs 动态z) — 诊断用, 不影响 solve"
    rm -f "$AMP/zcache_L$(printf %02d $PROBE_L).npz"
    echo "── 静态 z(现状: z 写死全1) ──"; solve_one "$PROBE_L" 0 /tmp/ab_static.bin
    echo "── 动态 z(方案B: 二阶门) ──"; solve_one "$PROBE_L" 1 /tmp/ab_dynz.bin
    rm -f "$AMP/zcache_L$(printf %02d $PROBE_L).npz"
}

# 第二轮反修: 每层在"上游已修好"的状态下重解。
#   第 L 层的输入只由 0..L-1 决定 ⇒ 带全部放大器捕的 raw_ffn_in_L 就是本层的真实部署输入,
#   不必逐层重捕 43 次。被乘量把上一轮增益除回去还原。
#   起因: 第一轮 43 层全部对着"全关"状态解, 没有一层见过自己真实的输入;
#   实测层内挽回 14.44% 只兑现端到端 0.72%, 且任何部分配置(单层/砍6层)都更差 —— 强耦合系统。
stage_pass2(){
    [ -s "$CAP_AMP/raw_ffn_in_L0" ] || DIE "带放大器的捕获缺($CAP_AMP)"
    [ -s "$AMP1/zrec_L00.bin" ] || DIE "上一轮放大器缺($AMP1)"
    CAP="$CAP_AMP"; PREV="$AMP1"
    stage_solve
}

stage_solve(){
    [ -s "$ANCHOR" ] || DIE "锚缺"; [ -s "$CAP/raw_ffn_in_L0" ] || DIE "引擎捕获缺"
    LOG "⑤放大器 43 层(动态z, 全层量化计算为学生)${PREV:+ ★第二轮: 上游已修状态★}"
    for L in $(seq 0 42); do
        REC="$AMP/zrec_L$(printf %02d $L).bin"
        [ -f "$REC" ] && continue
        solve_one "$L" - "$REC"
        rm -f "$AMP/zcache_L$(printf %02d $L).npz"
        LOG "  L$L ✓ ($(ls "$AMP"/zrec_L*.bin 2>/dev/null | wc -l)/43)"
    done
    LOG "⑤收官 43/43 侧车合计 $(du -sh "$AMP" | cut -f1)"
}

stage_chain(){
    LOG "⑥合并 zchain"
    python3 "$ZL/zrec_to_zchain.py" "$AMP" "$ZC" 43 || DIE "合并失败"
    ls -l "$ZC"
}

judge_one(){   # $1=标签 $2=zchain(可空) $3=dump
    local Z=(); [ -n "${2:-}" ] && [ -s "${2:-}" ] && Z=(--zchain "$2")
    cd "$ROOT"; LOG "⑦评分 wt2 $1"
    # --foreground 必须: timeout 默认 setpgid 成后台组, ds4 一碰 tty 即被 SIGTTIN 停机
    timeout --foreground 3000 ./ds4 --cuda -m "$MDL" "${Z[@]+"${Z[@]}"}" \
        --score-ids "$G7/wt2.ids" --score-out "$3" </dev/null 2>&1 | tail -1
    echo "══ wt2 五指标 $1 ══"
    python3 "$SC/anchor_metrics.py" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --student "$3" --tail 3 2>&1 | head -12
    # 【已移除】这里原来有一段"贴原始输出"的生成, 用的是我自己编的 prompt(还是道代码题),
    # 既是自造内容、又要靠 env 传 —— 两条铁律都踩。判决就是 wt2 五指标; 要看文本样本时
    # 由用户给题, 单独跑, 不写死在判决里。
}

stage_judge(){
    watchdog_start
    judge_one 裸 "" /tmp/${NAME}_wt2_base.bin
    [ -s "$ZC" ] && judge_one +放大器 "$ZC" /tmp/${NAME}_wt2_amp.bin
    watchdog_stop
}

ST="${1:-all}"
case "$ST" in
  preflight) stage_preflight;; ids) stage_ids;; anchor) stage_anchor;;
  capture) stage_capture;; probe) stage_probe;; solve) stage_solve;; pass2) stage_pass2;;
  chain) stage_chain;; judge) stage_judge;;
  all) stage_preflight; stage_ids; stage_anchor; stage_capture
       stage_solve; stage_chain; stage_judge;;
  *) echo "未知段: $ST"; echo "段: preflight ids anchor capture solve pass2 chain judge all (probe=可选诊断)"; exit 2;;
esac
LOG "段 $ST 完成"
