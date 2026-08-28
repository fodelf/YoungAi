#!/bin/bash
# amp_campaign.sh — 反修战役·单文件入口(2026-08-22 用户对齐定案)。
#
# 【设计】反修 = 求放大器，就这一件事。每层:
#   学生 = 当前层**全部的量化计算**(attn / shared / router / norm / 专家, 一个不落)
#          —— 取引擎跑普通全 q2 基座时的真值, 不是 Python 理想化重算
#   教师 = FP 全层前向(锚)
#   解变量 = U, V, **z**;  ★z 必须是 x 的函数(动态 z, 方案 B)★
#   目标 = ridge 拟合(方差列权 + dither 增广 + 收缩), λ/k_L 走 held 网格
#   ★正名(2026-08-26 用户纠正): 这不是"四损失"。四损失=ds4_loss.{c,h}(ALGORITHM.md §4:
#   L_align/L_classify/L_smooth/L_fixed), 落地在量化器 ds4quant_run 调优链;
#   本反修解算(zlayer)未接 ds4_loss —— 移植欠账在案, 别再把这里叫四损失。
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
SC="$ROOT/gguf-tools/scripts"
ZLB="$ROOT/gguf-tools/amp/zlayer"   # C 反修解算器(zlayer.py 已删, 接线照 amp_clean_full.sh 已验证参数位)
[ -x "$ZLB" ] || make -C "$ROOT/gguf-tools" zlayer
QD="$ROOT/gguf-tools/amp"
R30="$ROOT/gguf/go-onebit/r30"
G7="$ROOT/gguf/go-onebit/g7"

NAME="v5full"
CORPUS="$ROOT/gguf-tools/data/corpus/calibration_datav5.txt"   # 开源全场景, 22.4% 字符在代码围栏内
S=8192                       # 锚 token 数(整份语料等距窗抽样)
IDS_WIN=64                   # 等距窗数, 跨度铺满全文
MDL="$ROOT/gguf/ds4-allq2.gguf"   # 普通 RTN 全 q2, 零语料, 本战役不重量化
PROBE_L=2                    # 仅 probe 诊断段用
# 看门狗红线 = 总内存 1/8(下限 8G)。★不要写死小数字★(2026-08-27 用户纠正): 原值 4 是
# 小机器时代的遗留, 在 121G 的 spark 上等于"跌到只剩 3% 才停车" —— 而实测恶化(逐层耗时
# 5s→38s)从 MemAvailable 还很充裕时就开始了, 等 4G 才响已经白磨了五六层。
# 按比例取阈值 ⇒ 换机器不用改脚本, 也不会再留下一个过时的魔法数。
MEM_FLOOR_GB=4   # 系统 MemAvailable 地板 —— ★4 是实测值, 不是小机器遗留, 别再往上调★
# 2026-08-27 我在这个数上连错三次: 原值 4 全天正常; 改 15(总内存 1/8) 误杀正在出正收益的
# 反修; 改 8(绝对值) 误杀量化于 L39。实测: 本工作负载【正常工况】就把 MemAvailable 压到
# 5-7GB —— 121GB 机器 + 33GB mmap 锚 + page cache, 吃满是设计意图不是失控。
# ★两个阈值语义不同★
#   · 真限制 = 进程 RSS(r30_campaign 内部 wdog, 按机器内存 3/4) —— 那个该随机器缩放
#   · 这里 = 系统 MemAvailable 最后一道网, 只为抢在内核 OOM killer 前面留个可控停车点。
#     它必须【低于】负载的正常低点, 否则每次都误杀。调高 = 把正常工况判成失控。
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
            pkill -9 -f 'ds4 --cuda'; pkill -9 -f ds4quant_run; pkill -9 -f 'amp/zlayer'
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
    # 其余 DS4_ZL_* 是 zlayer(C 版)既有开关, 非本轮新增。
    [ -f "$ZCF" ] || env DS4_ZL_GGUF="$MDL" DS4_ZL_NTOK="$S" \
        DS4_ZL_GE=0 DS4_ZL_FTA=0 DS4_ZL_ERF=0 DS4_ZL_SWLIM=60 DS4_ZL_GATE=99 \
        "$ZLB" "$DS4_HF" "$AMP" "$ANCHOR" "$L" 1024 0 "$CAP" "${PREV:--}" 2>&1 \
        | grep -aE "XCAP|Error|assert|★" \
        || DIE "L$L zcache 失败(完整输出见上)"
    "$(dirname "$0")/../legacy/amp_solve_zc" "$ANCHOR" "$ZCF" "$3" || DIE "L$L 解算失败"   # 动态 z 已写死
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
    "$(dirname "$0")/../amp/zrec_to_zchain" "$AMP" "$ZC" 43 || DIE "合并失败"
    ls -l "$ZC"
}

judge_one(){   # $1=标签 $2=zchain(可空) $3=dump
    local Z=(); [ -n "${2:-}" ] && [ -s "${2:-}" ] && Z=(--zchain "$2")
    cd "$ROOT"; LOG "⑦评分 wt2 $1"
    # --foreground 必须: timeout 默认 setpgid 成后台组, ds4 一碰 tty 即被 SIGTTIN 停机
    timeout --foreground 3000 ./ds4 --cuda -m "$MDL" "${Z[@]+"${Z[@]}"}" \
        --score-ids "$G7/wt2.ids" --score-out "$3" </dev/null 2>&1 | tail -1
    echo "══ wt2 五指标 $1 ══"
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --student "$3" --tail 3 2>&1 | head -12
    # 【已移除】这里原来有一段"贴原始输出"的生成, 用的是我自己编的 prompt(还是道代码题),
    # 既是自造内容、又要靠 env 传 —— 两条铁律都踩。判决就是 wt2 五指标; 要看文本样本时
    # 由用户给题, 单独跑, 不写死在判决里。
}

# 链态稀释判决: 同一放大器, 只 arm 一层 vs 全 43 层 arm, 看端到端增益是不是线性叠加。
# 单层 held-out 挽回 28%(引擎实测), 全层端到端却 0.x% —— 若"只 arm L32"端到端也几乎为 0,
# 则根因是【单层 routed_out 改进被下游稀释】; 若只 arm L32 明显 > 全 43 层的 1/43,
# 则根因是【链态失配: 逐层独立拟合的 x 在全 arm 后全变了】。二者解法完全不同。
stage_dilute(){
    watchdog_start
    judge_one 裸基座        ""                    /tmp/${NAME}_dil_bare.bin
    judge_one 只armL32      /tmp/zc_phi32.bin     /tmp/${NAME}_dil_l32.bin
    judge_one 全43层arm     "$D/zchain_v2.bin"    /tmp/${NAME}_dil_all.bin
    watchdog_stop
}

# ★VQ86 战役(2026-08-23 用户裁决): 一半语料量化, 一半语料放大器。
# 切法 = 按 1024-token 块【交错】分半, 不是前后对切 —— 前后切会让两半的领域组成不同
# (语料是按主题拼接的), 交错切则两半的全场景组成/代码占比几乎逐块相同, 且零重叠。
# 产出: vqhalf_q.ids(量化半) / vqhalf_a.ids(放大器半)。
stage_idshalf(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"; mkdir -p "$D2"
    local QI="$D2/vqhalf_q.ids" AI="$D2/vqhalf_a.ids"
    [ -s "$QI" ] && [ -s "$AI" ] && { LOG "①ids 两半已在, 跳过"; return 0; }
    LOG "①语料交错切半 → 两半各 S=8192(对称; 每专家 192 校准行, base86p 的 2906 只有 68)"
    python3 - "$CORPUS" "$QI" "$AI" 8192 8192 64 "$DS4_HF" <<'PY' || DIE "切半失败"
import re, sys
from tokenizers import Tokenizer
src, oq, oa, NQ, NA, CH, hf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
text = open(src, encoding="utf-8").read()
enc = tok.encode(text, add_special_tokens=False)
allids, offs = enc.ids, enc.offsets

# 代码区 = ``` 围栏之间(在【原文】上定界, 切半后不再重算 —— 交错切会把围栏拆散,
# 在半文本上数 ``` 必然错配, 这正是先前算出"两半都 40% 代码"的原因)。
fences = [m.start() for m in re.finditer("```", text)]
spans = [(fences[i], fences[i+1]) for i in range(0, len(fences)-1, 2)]
def in_code(pos):
    for a, b in spans:
        if a <= pos < b: return True
        if pos < a: return False
    return False
mark = [in_code(o[0]) for o in offs]          # 每 token 是否落在代码围栏内
print("  全语料 %d token, 代码 token 占比 %.1f%% (围栏 %d 段)" % (
    len(allids), 100.0*sum(mark)/len(mark), len(spans)))

B = 256   # 块越细, 交错后两半的主题组成越接近
idx = list(range(len(allids)))
blocks = [idx[i:i+B] for i in range(0, len(idx), B)]
halfQ = [t for i, b in enumerate(blocks) if i % 2 == 0 for t in b]
halfA = [t for i, b in enumerate(blocks) if i % 2 == 1 for t in b]
def code_pct(sel): return 100.0*sum(mark[t] for t in sel)/max(len(sel), 1)
pq, pa = code_pct(halfQ), code_pct(halfA)
print("  交错切半(块=%d, 零重叠): 量化半 %d token 代码 %.1f%% | 放大器半 %d token 代码 %.1f%% (差 %.2fpp)" % (
    B, len(halfQ), pq, len(halfA), pa, abs(pq-pa)))
if abs(pq - pa) > 2.0: sys.exit("★两半池子组成失衡 —— 切法不合格★")

def sample_win(pool, N, CH, path):
    w = N // CH; step = (len(pool) - w - N % CH) // (CH - 1)
    sel = []
    for c in range(CH):
        ww = w + (N % CH if c == CH - 1 else 0)
        sel += pool[c*step : c*step+ww]
    sel = sel[:N]
    assert len(sel) == N, (len(sel), N)
    open(path, "w").write("\n".join(str(allids[t]) for t in sel) + "\n")
    print("  %-16s %6d token  代码 %.1f%%" % (path.split("/")[-1], N, code_pct(sel)))
    return code_pct(sel)
cq = sample_win(halfQ, NQ, CH, oq)
ca = sample_win(halfA, NA, CH, oa)
if abs(cq - ca) > 6.0:
    print("  注: 两半抽样代码占比差 %.1fpp (64 窗仅覆盖池子 4%%, 抽样方差); 判据以池子为准。" % abs(cq-ca))
PY
}

# ★放大器半扩样(2026-08-26 zloss90 战役, 用户令: 体积≤2GB 换 Σmin→0.90)★
# 依据: 大秩 held 单调恶化(过拟合)+段漂移 δcos 塌+GEc 段覆盖墙 三指纹同指
# "锚 8192tok/32段 数据饿"; 2GB 预算对应 rank ~1400-2800/层, fit 4608 行喂不动。
# 池=放大器半(奇数 256 块, idshalf 同式), 同一冻结语料内多采=不违语料冻结铁律;
# 量化半 ids 一字不动(零重叠纪律)。窗宽 128 与原 vqhalf_a.ids 同(行掩码几何不变)。
# 用法: amp_campaign.sh idshalf_ext [N=32768] [CH=256] [出名=vqhalf_a32k.ids]
stage_idshalf_ext(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local N="${1:-32768}" CH="${2:-256}" OUT="$D2/${3:-vqhalf_a32k.ids}"
    [ -s "$OUT" ] && { LOG "①扩样 ids 已在 $OUT, 跳过"; return 0; }
    LOG "①放大器半扩样 → $OUT (N=$N CH=$CH; 量化半不动)"
    python3 - "$CORPUS" "$OUT" "$N" "$CH" "$DS4_HF" <<'PY' || DIE "扩样失败"
import sys
from tokenizers import Tokenizer
src, oa, N, CH, hf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
allids = tok.encode(open(src, encoding="utf-8").read(), add_special_tokens=False).ids
B = 256
nblk = (len(allids) + B - 1) // B
pool = [t for i in range(nblk) if i % 2 == 1 for t in range(i*B, min((i+1)*B, len(allids)))]
w = N // CH; step = (len(pool) - w - N % CH) // (CH - 1)
assert step > w, ("窗重叠", step, w)                   # 步距>窗宽 = 窗间零重叠
sel = []
for c in range(CH):
    ww = w + (N % CH if c == CH - 1 else 0)
    sel += pool[c*step : c*step+ww]
sel = sel[:N]
assert len(sel) == N, (len(sel), N)
open(oa, "w").write("\n".join(str(allids[t]) for t in sel) + "\n")
print("  %s: %d token / 池 %d / 窗 %d×%d / 步距 %d" % (oa.split("/")[-1], N, len(pool), CH, w, step))
PY
}

# ★VQ86 半语料量化(2026-08-23): 与 base86p 单变量对照 —— 配方(平权 vq4x512 = 2.25bpw × 43 层)
# 完全不动, 只换两样: ① 语料 wt2train_cal9 → 开源全场景 v5 的【量化半】; ② 校准规模
# S 2906 → 8192(每专家 192 校准行 vs 68, DS4_CALIB_CAP 帽是 512, 原来欠采样 7.5 倍)。
# 放大器只许看【放大器半】, 与量化半零重叠 —— 这是本战役的核心实验纪律。
stage_vqquant(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    [ -s "$D2/vqhalf_q.ids" ] || DIE "量化半 ids 缺, 先跑 idshalf"
    LOG "②VQ86 量化发车: 平权 vq4x512 2.25bpw × 43 层, 语料=量化半 S=8192"
    Q86_IDS="$D2/vqhalf_q.ids" Q86_S=8192 Q86_NFIT=8192 \
    Q86_ANCHOR="$D2/anchor_vqhalf_q_s8192.bin" Q86_OUT="$D2/vq86h" \
    RPLAN86="$ROOT/gguf/go-onebit/r30/rplan_base86p.txt" \
        bash "$SC/base86p_spark.sh" quant
}

# VQ86 合并 + 裸判: 复用参数化的 merge_base86p.sh(幂等), 判决=wt2 五指标, 对表 base86p 4.1222。
stage_vqmerge(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local N=$(ls "$D2/vq86h/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)
    [ "$N" = 43 ] || DIE "vq86h 层不齐($N/43), 量化未收官"
    M86_LAYERS="$D2/vq86h/layers" M86_MDL="$ROOT/gguf/ds4-vq86h.gguf" \
    M86_SKEL="$ROOT/gguf/go-onebit/r30/r30_skeleton.gguf" M86_ZCH= \
        bash "$SC/merge_base86p.sh" || DIE "合并失败"
    watchdog_start
    cd "$ROOT"; LOG "③vq86h 裸判 wt2 (对表 base86p 4.1222 / allq2 12.1494)"
    timeout --foreground 3000 ./ds4 --cuda -m "$ROOT/gguf/ds4-vq86h.gguf" \
        --score-ids "$G7/wt2.ids" --score-out /tmp/vq86h_wt2.bin </dev/null 2>&1 | tail -1
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --student /tmp/vq86h_wt2.bin --tail 3 2>&1 | head -12
    watchdog_stop
}

# VQ86 放大器取料: 学生=vq86h(合并后整模型), 语料=【放大器半】vqhalf_a.ids(8192, 与量化半零重叠)。
# 取料走解码路(--score-ids+--cap-dir)=部署同路+确定(铁律); 批量路(--eval-ids)只出 hdump 杠杆诊断。
# 捕获即验: 同命令两次, L0/L16/L32 三档逐位比对(浅=哈希层不算数, 必须含中深)。
stage_vqcap(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" M="$ROOT/gguf/ds4-vq86h.gguf"
    [ -s "$M" ] || DIE "vq86h.gguf 缺, 先跑 vqmerge"
    [ -s "$D2/vqhalf_a.ids" ] || DIE "放大器半 ids 缺"
    cd "$ROOT"; watchdog_start
    if [ ! -s "$D2/cap_a/raw_ffn_in_L0" ]; then
        LOG "④取料#1 解码路 on vq86h × 放大器半"
        rm -rf "$D2/cap_a" "$D2/cap_a2"; mkdir -p "$D2/cap_a" "$D2/cap_a2"
        timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$D2/vqhalf_a.ids" \
            --score-out /tmp/vq86h_capa.bin --cap-dir "$D2/cap_a" </dev/null 2>&1 | tail -1
        LOG "④取料#2 复现检查遍"
        timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$D2/vqhalf_a.ids" \
            --score-out /tmp/vq86h_capa2.bin --cap-dir "$D2/cap_a2" </dev/null 2>&1 | tail -1
        for L in 0 16 32; do
            cmp "$D2/cap_a/raw_ffn_in_L$L" "$D2/cap_a2/raw_ffn_in_L$L" || DIE "L$L 捕获不复现"
            cmp "$D2/cap_a/raw_route_L$L"  "$D2/cap_a2/raw_route_L$L"  || DIE "L$L 路由不复现"
        done
        LOG "④复现 ✓ (L0/16/32 ffn_in+route 逐位一致), 清检查遍"
        rm -rf "$D2/cap_a2"
    else LOG "④取料 已在, 跳过"; fi
    if [ ! -s "$D2/hdump_a/h_L00.bin" ]; then
        LOG "④hdump 杠杆诊断遍(批量路)"
        mkdir -p "$D2/hdump_a"
        timeout --foreground 7200 ./ds4 --cuda -m "$M" --eval-ids "$D2/vqhalf_a.ids" \
            --eval-hdump "$D2/hdump_a" </dev/null 2>&1 | tail -1
    fi
    LOG "④npy 转换 + 教师(C)"
    mkdir -p "$D2/capnpy_a"
    "$ROOT/gguf-tools/cap_raw2npy" --raw "$D2/cap_a" --out "$D2/capnpy_a" --ntok 8192 || DIE "npy 转换失败"
    "$ROOT/gguf-tools/teacher_routed" --hf "$DS4_HF" --cap "$D2/capnpy_a" \
        --layers 0-42 --ntok 8192 --threads 20 --swlim 60 || DIE "教师失败"
    watchdog_stop
    LOG "④收官: cap $(du -sh "$D2/cap_a" | cut -f1) capnpy $(du -sh "$D2/capnpy_a" | cut -f1)"
}

# VQ86 放大器解算(C, amp_solve.c) + 合链 + 端到端判决。
# 解算=铁律 C 实现(乘性动态z, 随机基, 四损失行为空间目标); 合并器只做字节编排(许可的 Python)。
stage_vqsolve(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    [ -s "$D2/capnpy_a/routed_L42.npy" ] || DIE "教师 npy 缺, 先跑 vqcap"
    mkdir -p "$D2/amp_c"
    LOG "⑤C 解算 43 层(乘性动态z, ~1.2h)"
    "$ROOT/gguf-tools/amp_solve" --cap "$D2/capnpy_a" --out "$D2/amp_c" \
        --layers 0-42 --threads 20 || DIE "解算失败"
    "$(dirname "$0")/../amp/zrec_to_zchain" \
        "$D2/amp_c" "$D2/zchain_vq86h.bin" 43 || DIE "合链失败"
    watchdog_start
    cd "$ROOT"; LOG "⑥判决 wt2: vq86h裸 vs vq86h+放大器"
    timeout --foreground 3000 ./ds4 --cuda -m "$ROOT/gguf/ds4-vq86h.gguf" \
        --zchain "$D2/zchain_vq86h.bin" \
        --score-ids "$G7/wt2.ids" --score-out /tmp/vq86h_amp_wt2.bin </dev/null 2>&1 | tail -1
    echo "══ wt2 五指标 vq86h+放大器 (对表: 裸 vq86h / base86p 4.1222) ══"
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$R30/anchor_wt2_s2653.bin" --ids "$G7/wt2.ids" \
        --student /tmp/vq86h_amp_wt2.bin --tail 3 2>&1 | head -12
    watchdog_stop
}

stage_judge(){
    watchdog_start
    judge_one 裸 "" /tmp/${NAME}_wt2_base.bin
    [ -s "$ZC" ] && judge_one +放大器 "$ZC" /tmp/${NAME}_wt2_amp.bin
    watchdog_stop
}

# ═══ dyn86 ①档梯标定(2026-08-27, 用户令"按 86G 生成动态配置 json")═══
# 【全流程】① 档梯标定(本段) → ② rplan_solve 按预算出配置 json+计划表 → ③ 量化(量化半语料)
#           → ④ 反修(放大器半语料) → ⑤ caliper 五指标终判
# 【本段干什么】量出"每个位宽档的重建余弦 cos"。分配器完全靠这张表决定"多给这层
#   0.25bpw 值不值"——表是估计值, 优化的就是假问题。所以标定必须先做, 不能跳。
# 【为什么档梯要向上延伸】原分配器 rplan_solve_v4.py 的档梯只有 vq4x512 往【下】的档,
#   于是 86G 预算下它的最优解必然是"每层拉满 vq4x512"= 平权。这是当年平权在 86G 赢的
#   机制原因: 动态无处可去, 不是动态更差。本版加 vq4x1024(2.51bpw) 这一级往上的档。
# 【做法】L0-5 六层 × 三档 vq4x{256,512,1024}, 读量化器导出路打的 VQ_GATE 冷 cos 均值。
#   dim 固定 4 只动 nc ⇒ 位宽 = ceil(log2 nc)/4 bpw, 单变量。nc=512 也重跑一遍(不复用
#   vq86h 旧日志)是为了三档同批同语料同口径, 跨批比 cos 会把批次差当成档位差。
stage_dynladder(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" P="$ROOT/gguf/go-onebit/vqhalf/dyn86"
    [ -s "$D2/vqhalf_q.ids" ] || DIE "量化半 ids 缺"
    [ -s "$D2/anchor_vqhalf_q_s8192.bin" ] || DIE "量化半锚缺"
    mkdir -p "$P"
    # base86p_spark.sh 同款运行时设置: 缺了会从 24s/层 劣化到 9min/层(mmap 写锁争用实锤)
    export DS4_BF_MEMGB=80 DS4_VQ_TIMING=1 DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512
    export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
    export OPENBLAS_NUM_THREADS=1 DS4_THREADS=20 DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
    local NC L
    for NC in 256 512 1024; do
        local RP="$P/rplan_nc$NC.txt"
        : > "$RP"; for L in $(seq 0 42); do echo "L=$L dim=4 nc=$NC hot=0 w2dim=4 w2nc=$NC" >> "$RP"; done
        if grep -q "VQ_GATE" "$P/nc$NC.log" 2>/dev/null; then LOG "  档 vq4x$NC 已标定, 跳过"; continue; fi
        LOG "  档 vq4x$NC 标定发车(L0-5)"
        # QBIN_OVERRIDE: r30_campaign 默认找 ds4quant_run.r30(重构后不存在), 指到现役量化器。
        # ★别用管道包 grep★: 管道退出码是 grep 的, 量化器崩了也报"完成"(08-27 首跑即中招)。
        if ! env DS4_MINVOL_MAXL=6 QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" Q86_IDS="$D2/vqhalf_q.ids" Q86_S=8192 Q86_NFIT=8192 Q86_ANCHOR="$D2/anchor_vqhalf_q_s8192.bin" Q86_OUT="$P/nc$NC" RPLAN86="$RP" bash "$SC/r30_campaign.sh" quant86 > "$P/nc$NC.log" 2>&1; then
            tail -6 "$P/nc$NC.log"; DIE "档 vq4x$NC 标定失败, 见 $P/nc$NC.log"
        fi
        grep -o "VQ_GATE L=[0-9]* .*cold=[0-9.]*" "$P/nc$NC.log" | tail -6
    done
    echo "══ dyn86 档梯标定结果(cos 越高越准) ══"
    : > "$P/ladder_measured.txt"
    echo "# ladder_measured.txt — dyn86 档梯标定(L0-5 均值, 量化半语料 S=8192, 同批同口径)" >> "$P/ladder_measured.txt"
    echo "# 格式: dim nc cos   # 出处" >> "$P/ladder_measured.txt"
    for NC in 256 512 1024; do
        local C
        C=$(grep -o "cold=[0-9.]*" "$P/nc$NC.log" 2>/dev/null | sed 's/cold=//' \
            | awk '{s+=$1;n++} END{if(n)printf "%.4f",s/n; else printf "0"}')
        local N; N=$(grep -c "VQ_GATE" "$P/nc$NC.log" 2>/dev/null || echo 0)
        printf "  vq4 x%-5s cos=%s  (%s 条 VQ_GATE)\n" "$NC" "$C" "$N"
        echo "4 $NC $C   # dyn86 标定 $(date +%m-%d), L0-5 均值, $N 条 VQ_GATE" >> "$P/ladder_measured.txt"
    done
    echo "档梯 → $P/ladder_measured.txt  (下一步: rplan_solve --ladder 它 --budget-gib 72.857)"
}

# ═══ dyn86 ②量化(动态配置)═══
# 配置来源 = rplan_solve 按 86G 预算解出的 dyn86/rplan_dyn86.txt(等体积对打平权 vq86h)。
# 与 vq86h 的唯一变量 = 位宽分配方式; 语料/锚/S/工序全部相同, 所以判决可直接对表。
# $1 = 计划表变体: dyn86(动态专家+动态层) | lyr86(纯动态层, 全 256 专家层内同档)
stage_dynquant(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" P="$ROOT/gguf/go-onebit/vqhalf/dyn86"
    local V="${1:-dyn86}" RP MODEL
    RP="$P/rplan_$V.txt"; MODEL="$P/model_$V"
    [ "$V" = dyn86 ] && { [ -s "$P/rplan_dyn86.txt" ] && RP="$P/rplan_dyn86.txt"; MODEL="$P/model"; }
    [ -s "$RP" ] || DIE "计划表缺 $RP"
    [ -s "$D2/anchor_vqhalf_q_s8192.bin" ] || DIE "量化半锚缺"
    export DS4_BF_MEMGB=80 DS4_VQ_TIMING=1 DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512
    export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
    export OPENBLAS_NUM_THREADS=1 DS4_THREADS=20 DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
    LOG "②量化发车 变体=$V 计划表=$RP → $MODEL"
    # 断点续跑是自动的: plan_lookup(L) 命中且 ckpt 装得上就复用该层, 重跑同命令即续。
    # 看门狗必挂(2026-08-27 教训): 首跑 L36 被系统 OOM killer 干掉(rc=137), 用户态先杀才可控。
    watchdog_start
    if ! env QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" Q86_IDS="$D2/vqhalf_q.ids" \
        Q86_S=8192 Q86_NFIT=8192 Q86_ANCHOR="$D2/anchor_vqhalf_q_s8192.bin" \
        Q86_OUT="$MODEL" RPLAN86="$RP" VOLB86=76 \
        bash "$SC/r30_campaign.sh" quant86 >> "$P/quant_$V.log" 2>&1; then
        watchdog_stop; tail -8 "$P/quant_$V.log"; DIE "$V 量化失败, 见 $P/quant.log"
    fi
    watchdog_stop
    LOG "$V 量化收官 43/43"
    grep -E "★贪心选|档位|冷档" "$P/quant_$V.log" | tail -5
}

# ═══ dyn86 ③合并 + 判决 ═══
# 判决尺只认参考前向(caliper_ref.sh, 铁律 08-24: 引擎 CUDA score/eval 路 4× 分歧未修前不当判官)。
# 对表 = 同语料同锚同工序的平权 vq86h_noz, 唯一变量 = 位宽分配方式。
stage_dynjudge(){
    local P="$ROOT/gguf/go-onebit/vqhalf/dyn86"
    local N; N=$(ls "$P/model/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)
    [ "$N" = 43 ] || DIE "dyn86 层不齐($N/43)"
    # caliper 直接吃层件目录, 判决不需要先合 GGUF(合并留给引擎部署/真代码基准那一步)。
    # ★口径风险在案★ 判决尺是冻结的 ds4quant_run.old(08-22), 早于本次热档可配改动, 不认
    # hotdim/hotnc。理论上无碍: VQ blob 自描述(每载荷头带 dim/nc, 偏移走 blob 内 256×3 表),
    # 老二进制照样解得开。但这是理论 —— 读数落在 0.40-0.50 合理带才可信, 若崩成天文数字
    # 就是老尺读不了新格式, 那时必须先解决尺子问题再谈质量, 不许拿坏尺的数当结论。
    LOG "③dyn86 裸判(参考前向尺); 对表平权 vq86h_noz: KLD 0.47055 / Σmin 0.7799 / top1 78.36%"
    watchdog_start
    bash "$SC/caliper_ref.sh" "$P/model/layers" /tmp/qc_dyn86_wt2.bin 2>&1 | tail -14
    watchdog_stop
}

# ═══ 冠军配方反修(2026-08-27 用户令"按 67g 冠军版设计重新反修和路由反修")═══
# 【这不是新写的东西】直接调 r30_campaign.sh backfit —— 那就是超冠当时跑的那一段, 原封不动。
# 它一段里同时做完【反修 + 路由反修】, 不是两件事:
#   DS4_BF_ONLY=1     跳过 ALT 逐层(层内判据, 保险门实锤端到端负贡献)
#   DS4_BF_ONEPASS=1  冻结基线一遍选型 + 统一终验 + 劣化全回滚(用户 08-04 裁决"一遍就够")
#   DS4_GSWEEP=0      回扫不跑(侧车架构下需要时删侧车补跑即可)
#   DS4_EXPORT_BYTES=0 平行架构: dql 只读不可变, 只重建 op 侧车(用户"反修不许动量化模型")
#   DS4_ROUTE_BIAS_FIT=1 + ALPHA=2.5  ★路由偏置寄生在反修自身前向, 零额外前向★
#   DS4_ANCHOR_ROUTE=1 锚路由反修(超冠原样)
# 段内自带"架构组件在场自检": z变量[E/C/F] 四损失[KGRID la/lf/ls+lc] 感知[pc行权]
# 向后[TREF-B] 路由[GE-D投影] —— 四损失本来就在这一段, 不在后来另起的 zloss_solve 里。
#
# $1 = 层件目录名(model_lyr86 | model | vq86h_noz); $2 = probe 则只跑 L00 验机制
stage_champbf(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" V="${1:?层件目录}" MODE="${2:-full}"
    local OUT
    case "$V" in
      vq86h_noz) OUT="$D2/vq86h_noz";;
      champ86)   OUT="$D2/champ86";;    # 平权底座工作副本(原始 vq86h_noz 只读保全)
      *)         OUT="$D2/dyn86/$V";;
    esac
    [ "$(ls "$OUT/layers"/dql_L*.bin 2>/dev/null | wc -l)" = 43 ] || DIE "层件不齐 $OUT/layers"
    [ -s "$D2/anchor_a_clean_s8192.bin" ] || DIE "校准锚缺"
    # ★三个内存闸要拉开距离★(2026-08-27 实撞): 原设 BF_MEMGB=80(驱逐线) + RSS 杀线 93G,
    # 只留 13G 缓冲 —— 驱逐刚在 L14 起步(footprint 89.73G), 进程就撞 93.5G 被杀。
    # 驱逐是渐进的, 触发点必须【远低于】杀线才来得及。上一跑 footprint 稳在 59-62G,
    # 说明 60 左右是本负载的自然工作点, 取 55 让驱逐早介入, 给杀线留 38G 缓冲。
    export DS4_BF_MEMGB=55 MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
    export OPENBLAS_NUM_THREADS=1 DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
    # 这四个必须不在场, 否则 backfit 走错分支(段内自带硬闸会停)
    unset DS4_TUNE DS4_MINVOL DS4_MV_BASELINE DS4_VQ_RPLAN
    local PB=""; [ "$MODE" = probe ] && PB=1
    # 纯 VQ 底座无 go2b 热专家 ⇒ 关 GO2B_HOT(冠军底座是 go2b 热, 这是底座差异不是配方改动)
    local G2H=0
    LOG "冠军配方反修发车: $OUT ${PB:+(单层探针 L00)}"
    watchdog_start
    ( cd "$ROOT" && env ${PB:+PROBE1=1} OUTF_OVERRIDE="$OUT" ANCHOR_OVERRIDE="$D2/anchor_a_clean_s8192.bin" \
        IDS_OVERRIDE="$D2/vqhalf_a.ids" BF_S=8192 BF_NFIT=6144 DS4_THREADS=20 \
        QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" DS4_GSWEEP=0 DS4_GO2B_HOT=$G2H \
        bash "$SC/r30_campaign.sh" backfit ) > "$OUT/backfit.log" 2>&1
    # ★别用管道包 tail★(2026-08-27 实撞): 管道把 stderr 全缓冲, 跑一小时看不到任何逐层进度,
    # 违反"长任务必须逐单元可观测"铁律。改为直接落盘 —— 跑中随时 tail -f 看真进度。
    tail -40 "$OUT/backfit.log"
    watchdog_stop
}

# ═══ 量化→反修 串跑(2026-08-27 用户令"重头跑量化, 再跑反修")═══
# 两段必须同一进程串起来: 中间人工接力是今天多次事故的来源(续跑链失稳/清理时改坏层文件)。
# $1 = 计划表变体(lyr86 | dyn86)
stage_full(){
    local V="${1:?变体}"
    stage_dynquant "$V"
    stage_champbf "model_$V"
}

# ═══ champ86: 冠军 r64 原设计全链(2026-08-28 用户令"一切还原冠军设计")═══
# 与 r64 的差异仅三项(用户已确认): 语料(calibration_datav5 切半) / 体积(87.04GB) /
# 底座类型(平权 VQ 2.25bpw, 非热108 go2b+冷1bit ⇒ DS4_GO2B_HOT=0)。
# 其余一律 r64 原样: ONEPASS 不分块(CHUNK 是 08-07 才加的, r64 没有)、粗筛只定排序
# (全闸复核维持 !onep 跳过)、不加 ERF(它在 zside2 链不在冠军段)。
#
# 链: ①平权量化(量化半语料+量化半锚) → ②反修+路由反修(放大器半语料+放大器半锚, 冠军
#     stage_backfit 一段做完) → ③五指标(判决尺吃层件) → ④合并+烘 α·Δb
# ★锚口径★ 两个锚各配各的语料, 已用 anchor_metrics 实测校验:
#   量化半 anchor_vqhalf_q_s8192 + vqhalf_q.ids → PPL 13.55/top1 52.5% ✓
#   放大器半 anchor_a_clean_s8192 + vqhalf_a.ids → PPL 11.95/top1 55.7% ✓
#   (交叉喂错语料 PPL 会飙到 2.4e7/top1 0.8%, 即错锚一眼可辨)
# 冠军 stage_backfit 默认也是"反修锚=反修语料的 FP 锚"(anchor_r30_s1716 配
# rr_calib_prog_v5mini.ids), 口径一致。
stage_champ86(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" W="$ROOT/gguf/go-onebit/vqhalf/champ86"
    [ -s "$D2/vqhalf_q.ids" ] && [ -s "$D2/vqhalf_a.ids" ] || DIE "两半语料 ids 缺"
    [ -s "$D2/anchor_vqhalf_q_s8192.bin" ] || DIE "量化半锚缺"
    [ -s "$D2/anchor_a_clean_s8192.bin" ] || DIE "放大器半锚缺"
    # ①平权量化 — 从头, 不复用任何既有层件
    if [ "$(ls "$W/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)" != 43 ]; then
        LOG "①平权量化发车(vq4x512 ×43 不动态, 量化半语料 S=8192)"
        rm -rf "$W"; mkdir -p "$W"
        export DS4_BF_MEMGB=55 DS4_VQ_TIMING=1 DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512
        export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
        export OPENBLAS_NUM_THREADS=1 DS4_THREADS=20 DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
        watchdog_start
        if ! env QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" Q86_IDS="$D2/vqhalf_q.ids" \
            Q86_S=8192 Q86_NFIT=8192 Q86_ANCHOR="$D2/anchor_vqhalf_q_s8192.bin" Q86_OUT="$W" \
            RPLAN86="$ROOT/gguf/go-onebit/r30/rplan_base86p.txt" VOLB86=76 \
            bash "$SC/r30_campaign.sh" quant86 >> "$W/quant.log" 2>&1; then
            watchdog_stop; tail -8 "$W/quant.log"; DIE "平权量化失败"
        fi
        watchdog_stop
        LOG "①量化收官 $(ls "$W/layers"/dql_vq_L*.bin | wc -l)/43"
    else LOG "①平权量化已在 43/43, 跳过"; fi
    # ②冠军反修(含路由反修, 一段做完)
    stage_champbf champ86
    # ③五指标(判决尺吃层件, 08-24 铁律: 只认参考前向)
    LOG "③五指标; 对表平权裸 KLD 0.47055 / Σmin 0.7799 / top1 78.36%"
    bash "$SC/caliper_ref.sh" "$W/layers" /tmp/qc_champ86.bin 2>&1 | tail -10
}

# ═══ champ86 ⓪: 把层件复位成"刚量化完"的状态(2026-08-28)═══
# 用途: 反修被打断/中途改过代码后重跑。用户令是"重头跑从量化开始", 而量化段 08-28 09:26
# 已 rc=0 跑满 43/43, 且 champ86/layers/dql_L00.bin 与只读原件 vq86h_noz 的同名文件
# ★md5 逐字节相同★ —— 说明这套量化是确定性的, 重跑只会产出同样的字节。
# 于是"从量化开始"在实质上 = 把层件恢复到量化刚结束的字节状态, 再让反修从零起跑。
# (铁律 feedback_staged_rerun_scope: 下游段失败不许无脑销毁完好的上游产物。)
# 反修会往层件里追加记录、并落 zrec_LXX.bin, 所以复位 = 逐文件比对只读原件, 不等就覆盖,
# 原件没有的(zrec 等反修产物)一律删。全程只动 champ86, 绝不碰 vq86h_noz。
stage_champ_reset(){
    # ★别写成一句 local A=.. B="$A/.."★: bash 会先把这一行的所有名字建成(未赋值的)局部变量,
    # 再逐个赋值, 于是同句里引用前一个名字在 set -u 下直接报"未绑定的变量"。分三句写。
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local W="$D2/champ86"
    local SRC="$D2/vq86h_noz"
    [ -d "$SRC/layers" ] || DIE "只读原件不在: $SRC/layers"
    [ -d "$W/layers" ]   || DIE "工作副本不在: $W/layers"
    local n_rm=0 n_cp=0 n_ok=0
    # ① 原件没有的文件 = 反修产物, 删
    for f in "$W/layers"/*; do
        local b; b=$(basename "$f")
        [ -e "$SRC/layers/$b" ] || { rm -f "$f"; n_rm=$((n_rm+1)); }
    done
    # ② 原件有的: 逐字节比, 不等就从原件覆盖
    for f in "$SRC/layers"/*; do
        local b; b=$(basename "$f")
        if cmp -s "$f" "$W/layers/$b"; then n_ok=$((n_ok+1))
        else cp -f "$f" "$W/layers/$b"; n_cp=$((n_cp+1)); LOG "  复位 $b"; fi
    done
    rm -f "$W/backfit.log" "$W/route_bias_r30.bin" "$W/route_bias_r30.bin.alpha.txt" "$W/rb_alpha.txt"
    LOG "层件复位完成: 原样 $n_ok / 覆盖 $n_cp / 删反修产物 $n_rm"
    # ③ 复位后逐字节全量复核(不许只信上面的循环)
    if diff -rq "$SRC/layers" "$W/layers" > /tmp/champ_reset_diff.txt 2>&1; then
        LOG "★复核通过: champ86/layers 与只读原件逐字节一致★"
    else DIE "复核失败, 差异见 /tmp/champ_reset_diff.txt: $(head -3 /tmp/champ_reset_diff.txt)"; fi
}

# ═══ 诊断针: 第三片(同语料·不相交)★不是判决★(2026-08-28)═══
# 要回答的问题只有一个: 反修在【与校准语料同源、但一个 token 都不重叠】的片上, 是改善还是退化?
#   改善 ⇒ 问题是分布错配(校准语料混合场景+22.4%代码, 判决尺是 WikiText-2) ⇒ 换/扩语料有用
#   退化 ⇒ 问题是对那 8192 个 token 过拟合(每专家才 ~192 行) ⇒ 换语料白搭, 要加量或改设计
# 切得出第三片的依据: calibration_datav5.txt 约 40 万 token, 两半各只取 8192(64 个 128-token
# 窗口), 95% 从没用过。第三片取【同一个奇数块池(放大器半的池子)、窗口整体后移一个窗宽】——
# 分布同源、位置零重叠。判决口径不变: 终判永远只认 wt2。
stage_champ3rd(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local CI="$D2/vqhalf_c.ids"
    local CA="$D2/anchor_c_s8192.bin"
    if [ ! -s "$CI" ]; then
        LOG "①切第三片(同池·后移一窗·零重叠)"
        python3 - "$CORPUS" "$CI" 8192 64 "$DS4_HF" <<'PY2' || DIE "第三片切失败"
import sys
from tokenizers import Tokenizer
src, oc, N, CH, hf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
allids = tok.encode(open(src, encoding="utf-8").read(), add_special_tokens=False).ids
B = 256
blocks = [list(range(i, min(i+B, len(allids)))) for i in range(0, len(allids), B)]
poolA = [t for i, b in enumerate(blocks) if i % 2 == 1 for t in b]
w = N // CH
step = (len(poolA) - w - N % CH) // (CH - 1)
used, sel = set(), []
for c in range(CH):
    ww = w + (N % CH if c == CH-1 else 0)
    used.update(poolA[c*step : c*step + ww])
    sel += poolA[c*step + w : c*step + w + ww]
sel = sel[:N]
assert len(sel) == N, (len(sel), N)
ov = len(set(sel) & used)
assert ov == 0, "★与放大器半重叠 %d 个位置★" % ov
open(oc, "w").write("\n".join(str(allids[t]) for t in sel) + "\n")
print("  第三片 %d token, 与放大器半位置重叠=0 ✓ (池 %d token)" % (N, len(poolA)))
PY2
    fi
    if [ ! -s "$CA" ]; then
        LOG "②给第三片跑 FP 锚(锚与 ids 错配会出 PPL 2.4e7 这种一眼假的数, 必须自己配)"
        watchdog_start
        ( cd "$ROOT/gguf-tools/amp" && env DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731" \
            OPENBLAS_NUM_THREADS=1 DS4_THREADS=20 DS4_BF_MEMGB=55 \
            DS4_FP_ONLY=1 DS4_ANCHOR="$CA" ./ds4quant_run.old "$CI" 8192 ) > /tmp/anc_c.log 2>&1
        watchdog_stop
        [ -s "$CA" ] || { tail -5 /tmp/anc_c.log; DIE "第三片锚没落盘"; }
        LOG "②锚 ✓ $(ls -l "$CA" | awk '{printf "%.1f GiB", $5/1073741824}')"
    fi
    echo "══ 诊断针(★不是判决★): 同语料不相交第三片 ══"
    for V in vq86h_noz champ86; do
        LOG "针: $V"
        bash "$SC/caliper_ref.sh" "$D2/$V/layers" "/tmp/c3_$V.bin" 20 "" 2.5 "$CI" "$CA" \
            > "/tmp/c3_$V.log" 2>&1 || { LOG "$V 针失败"; tail -3 "/tmp/c3_$V.log"; continue; }
        printf -- "── %s ──\n" "$V"
        grep -aE "PPL\(student\)|分布还原率|Mean KLD|Same top" "/tmp/c3_$V.log"
    done
    echo "★读法: champ86 相对 vq86h_noz 若【改善】=分布错配(换语料有用); 若【退化】=过拟合(换语料白搭)★"
}

# ═══ champ86 ④: 路由偏置 α 在【本底座本语料】上实扫(2026-08-28 用户纠)═══
# ★不许直接用冠军的 2.5★ 那是 r64 在【它自己的底座(热108 go2b+冷1bit)+它自己的语料
# (rr code S=305)】上扫出来的峰(曲线: ≤1.0 阈下无效 / 1.5→80.3 / 2.0→81.6 / ★2.5→84.2★
# / 3.0 过冲回落)。我们换了底座(平权 VQ 2.25bpw)和语料(calibration_datav5 半), 峰位没理由
# 还在原处 —— fable5 3300 行有前车之鉴: 迁移版 Δb 直接搬 = 净负("别人的漂移药方")。
# α 有翻转阈值(小 α 完全不动 argmax), 所以必须扫到 2 以上才看得出东西, 别用 α=1 下结论。
# 判决尺应用 DS4_ROUTE_BIAS/ALPHA 与合并时烘进 exp_probs_b.bias 语义同构, 故层件上扫即可
# 预测合并后行为, 不必每个 α 合一次 87GB 模型。
stage_champ_rbsweep(){
    local W="$ROOT/gguf/go-onebit/vqhalf/champ86" RB="$ROOT/gguf/go-onebit/vqhalf/champ86/route_bias_r30.bin"
    [ -s "$RB" ] || DIE "Δb 不在($RB) — 反修未跑到 rb_save 或哈希路由无对象"
    local AS=(1.5 2.0 2.5 3.0)
    echo "══ α 扫描(本底座本语料); α=0 基线见 ③ 的五指标 ══"
    for A in "${AS[@]}"; do
        LOG "α=$A 判决中"
        bash "$SC/caliper_ref.sh" "$W/layers" "/tmp/qc_champ86_a$A.bin" 20 "$RB" "$A" \
            > "/tmp/cal_champ86_a$A.log" 2>&1 || { LOG "α=$A 判决失败"; continue; }
        printf "── α=%s ──\n" "$A"
        grep -aE "PPL\(student\)|分布还原率|Mean KLD|Same top" "/tmp/cal_champ86_a$A.log" | head -4
    done
    echo "★选峰依据: 主尺 Σmin 优先; Same-top 是路由偏置的直接靶(冠军按它选), 两者背离时报给用户定★"
    echo "选定后写 $W/rb_alpha.txt, 合并段自动读取"
}

ST="${1:-all}"
case "$ST" in
  preflight) stage_preflight;; ids) stage_ids;; anchor) stage_anchor;;
  capture) stage_capture;; probe) stage_probe;; solve) stage_solve;; pass2) stage_pass2;;
  chain) stage_chain;; judge) stage_judge;; dilute) stage_dilute;; idshalf) stage_idshalf;; idshalf_ext) shift; stage_idshalf_ext "$@";; vqquant) stage_vqquant;; vqmerge) stage_vqmerge;; vqcap) stage_vqcap;; vqsolve) stage_vqsolve;; dynladder) stage_dynladder;; dynquant) shift; stage_dynquant "$@";; dynjudge) stage_dynjudge;; champbf) shift; stage_champbf "$@";; champ86) stage_champ86;; champreset) stage_champ_reset;; champ3rd) stage_champ3rd;; champrb) stage_champ_rbsweep;; full) shift; stage_full "$@";;
  all) stage_preflight; stage_ids; stage_anchor; stage_capture
       stage_solve; stage_chain; stage_judge;;
  *) echo "未知段: $ST"; echo "段: preflight ids anchor capture solve pass2 chain judge dilute all (probe/dilute=诊断)"; exit 2;;
esac
LOG "段 $ST 完成"
