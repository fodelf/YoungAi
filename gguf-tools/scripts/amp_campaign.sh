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
export DS4_HF   # 脚本间接口(r30_campaign 等子脚本读); 二进制一律走 --hf 显式传(2026-08-31 env 大扫除)

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
    # fail-closed(2026-08-31 魔数扫除): 旧版 ${A:-99} 在 meminfo 读不出时按"99GB 可用"放行,
    # 等于看门狗静默缴械(macOS 无 /proc = 结构性瞎; Linux 读取抖动同样瞎)。内存护栏是
    # 最高铁律, 读不到内存数一律当危险处理: 启动即验一次, 循环内读空同样杀。
    awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo >/dev/null 2>&1 \
        || DIE "看门狗读不到 /proc/meminfo(此机不支持), 拒绝无护栏发车"
    ( while true; do
        A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null)
        if [ -z "$A" ] || [ "$A" -lt "$MEM_FLOOR_GB" ]; then
            echo "[watchdog] MemAvailable=${A:-读取失败}GB (地板 ${MEM_FLOOR_GB}GB) ★杀本段★" >&2
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
    ( cd "$QD" && ./ds4quant_run "$IDS" "$S" --hf "$DS4_HF" --fp-only --anchor "$ANCHOR" --threads 20 )
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
    # 第 7 位置参数 = 引擎捕获目录(x 与被乘量取真值)。原 DS4_ZL_* 开关已随 env 大扫除
    # 改为位置参数后的 --flag(值语义未动)。
    [ -f "$ZCF" ] || "$ZLB" "$DS4_HF" "$AMP" "$ANCHOR" "$L" 1024 0 "$CAP" "${PREV:--}" \
        --gguf "$MDL" --ntok "$S" --ge 0 --fta 0 --erf 0 --swlim 60 --gate 99 2>&1 \
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
    local QI="$D2/vqhalf_q.ids" AI="$D2/vqhalf_a.ids" JI="$D2/vqhalf_j.ids"
    # ★布局是产物的一部分(2026-08-29)★: ids 在但 .layout 缺 = 补布局机制之前切的。
    # 切分是确定性的(无随机源), 所以同一段代码重跑到 tmp、逐字节比对 ids 一致后, 只把
    # .layout 装回去 —— ids 一个字节不动 ⇒ 三个锚(各 30.8G/57 分钟)不作废。
    # 比对不过就硬停: 宁可没布局, 也不许拿一份【算出来的】布局去配一份【对不上的】ids。
    local HAVE=0 T=""
    [ -s "$QI" ] && [ -s "$AI" ] && [ -s "$JI" ] && HAVE=1
    if [ "$HAVE" = 1 ] && [ -s "$QI.layout" ] && [ -s "$AI.layout" ] && [ -s "$JI.layout" ]; then
        LOG "①ids 三份 + 行布局已在, 跳过(要重切先 mv 走)"; return 0; fi
    local OQ="$QI" OA="$AI" OJ="$JI"
    if [ "$HAVE" = 1 ]; then
        T="$(mktemp -d)"; OQ="$T/q.ids"; OA="$T/a.ids"; OJ="$T/j.ids"
        LOG "①ids 已在但行布局缺 → 确定性重算 + 逐字节校验, 只补布局(不动 ids)"
    fi
    # ★域分层整簇切半(2026-08-28 用户令"不要相似的, 全域都要有")★
    # 旧切法(B=256 token 定长块偶奇交替)的病: 同一篇文档的【相邻段落】被分进两半 —— 实测
    # 两半各自取到的是同一篇小鼠肠道菌群论文的相邻段, 两半几乎是复制品。校准半见过的东西
    # 反修半又见一遍, 等于没有独立的拟合料。
    # 新切法三条:
    #   ① 一行=一个文档(这份语料的代码是用字面 \n 转义嵌在行内的, 所以最长行 13183 字符,
    #      按行切绝不会把一段代码劈开)
    #   ② 按内容分 8 域(prose/code/math/euro/cyrillic/arabic/cjk/academic), 连续同域行合成
    #      "文档簇", ★整簇只进一半★ —— 相邻段落再不可能跨半
    #   ③ 每个域【各自】做贪心平衡分配(大簇优先给当前较轻的一半) ⇒ 两半都拿到全部 8 个域,
    #      且每域 token 量接近
    # 抽样也按域分层: 每半每域按其全局占比取配额, 域内等距铺窗, 保证抽出来的 8192 token
    # 仍然全域齐全(旧切法 64 窗盲抽, 小域可能一个 token 都抽不到)。
    LOG "①域分层整簇三切 → 量化/反修/判决 各 S=8192(全域齐全, 整簇不拆, 三份零重叠)"
    python3 - "$CORPUS" "$OQ" "$OA" "$OJ" 8192 "$DS4_HF" "$ROOT/gguf/go-onebit/g7/wt2.ids" <<'PY' || DIE "三切失败"
import re, sys, collections
from tokenizers import Tokenizer
src, oq, oa, oj, N, hf, wt2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7]
OUTS = [("量化份", oq), ("反修份", oa), ("判决份", oj)]
tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
lines = [ln for ln in open(src, encoding="utf-8").read().split("\n") if ln.strip()]

CODEKW = re.compile(r"\b(import |from \w+ import|def |class |function |const |let |var |public |private |return |print\(|console\.log|#include|package |func |fn |=>|\bfor\s*\(|\bif\s*\()")
def dom(s):
    """★域判定, 顺序即优先级(2026-08-28 收紧)★
    首版把 'Fix this code taken from an OCR result ...' 判进了 math(触发词误命中), 抽样时
    math 段抽出来的其实是代码。规则改成: code 一律最优先(围栏 / 行内转义换行里的代码 /
    高符号密度+代码关键词三选一), math 只留真数学(LaTeX 或算术应用题且不带代码特征)。"""
    n = [c for c in s if not c.isspace()]
    if "```" in s: return "code"
    # ★LaTeX 判定必须在代码关键词之前(2026-08-28 二次收紧)★: 数学题常把 Asymptote 绘图码
    # 嵌在题面里(如 "The function $f(x)=|x+2|+1$ is graphed below. [asy] import graph;"),
    # 里面的 import/size() 会让它落进 code —— 它本质是数学题。带 LaTeX 的一律先归 math,
    # 真代码里出现 $ 的只有 shell/PHP 变量, 不会同时命中 \frac|\cos|\[ 这类。
    if re.search(r"\\frac|\\sqrt|\\cos|\\sin|\\sum|\\int|\\alpha|\\beta|\\pi\b|\\\[", s): return "math"
    if re.search(r"\$[^$\n]{2,}\$", s) and not re.search(r"\$\w+\s*=|\$\{", s): return "math"
    if "\\n" in s and len(CODEKW.findall(s)) >= 2: return "code"
    sym = sum(1 for c in s if c in "{}[]()=;<>+*/_|&")
    if sym/max(len(s), 1) > 0.08 and CODEKW.search(s): return "code"
    if re.search(r"\b(How many|How much|What is the (value|number|smallest|largest|sum|product)|"
                 r"Find the (value|number|sum|area)|Calculate the|Simplify|Solve for|Evaluate the)\b", s) \
       and not CODEKW.search(s): return "math"
    cjk = sum(1 for c in n if 0x3040<=ord(c)<=0x30ff or 0x4e00<=ord(c)<=0x9fff or 0xac00<=ord(c)<=0xd7af)
    cyr = sum(1 for c in n if 0x400<=ord(c)<=0x4ff)
    ara = sum(1 for c in n if 0x600<=ord(c)<=0x6ff)
    if (cjk+cyr+ara)/max(len(n), 1) > 0.10:
        return "cjk" if cjk>=max(cyr,ara) else ("cyrillic" if cyr>=ara else "arabic")
    if sum(1 for c in n if 128<=ord(c)<0x250)/max(len(n), 1) > 0.015: return "euro"
    if re.search(r"\{#sec|\[@ref|\{ref-type=|\^\[@", s): return "academic"
    return "prose"

# ② 连续同域行 → 文档簇(整簇不拆)
clusters, cur = [], None
for ln in lines:
    d = dom(ln)
    if cur and cur[0] == d: cur[1].append(ln)
    else:
        if cur: clusters.append(cur)
        cur = [d, [ln]]
if cur: clusters.append(cur)
for c in clusters:
    c.append(tok.encode("\n".join(c[1]), add_special_tokens=False).ids)

# ③ 每域内贪心平衡: 大簇优先给当前 token 较少的一半
byd = collections.defaultdict(list)
for c in clusters: byd[c[0]].append(c)
# ★三路贪心(2026-08-28 用户令"切三份: 量化/反修/判决")★ 每域各自把文档簇按 token 降序
# 依次投给当前最轻的一份 ⇒ 三份都拿到全部域, 且每域 token 量近乎相等; 整簇不拆 ⇒ 同源
# 文档绝不跨份(这正是两份时代"两半是同篇论文相邻段"的根治法)。
K = 3
pools = [collections.defaultdict(list) for _ in range(K)]
for d, cs in byd.items():
    cs.sort(key=lambda c: -len(c[2]))
    tk = [0]*K
    for c in cs:
        i = tk.index(min(tk)); pools[i][d] += c[2]; tk[i] += len(c[2])

alld = sorted(byd, key=lambda d: -sum(len(c[2]) for c in byd[d]))
tot = sum(len(c[2]) for cs in byd.values() for c in cs)
print("  语料 %d token / %d 行 / %d 文档簇, %d 个域 → 三份" % (tot, len(lines), len(clusters), len(byd)))
print("  %-10s %8s | %8s %8s %8s" % ("域", "全语料", "量化池", "反修池", "判决池"))
for d in alld:
    print("  %-10s %8d | %8d %8d %8d" % (d, sum(len(c[2]) for c in byd[d]),
          len(pools[0][d]), len(pools[1][d]), len(pools[2][d])))

def sample(pool, N, path):
    """按域配额分层抽, 域内等距铺窗(窗宽 128)。
    ★配额=域等权(2026-08-28 用户令"领域都不对等")★: 8 域各 N/8。
    原来按语料占比分配, 直接继承了语料本身 prose:academic = 64:1 的失衡 ——
    cjk/arabic/cyrillic/academic 各只分到 128 token, 去校准 43 层×256 专家等于没有信号,
    而北极星是【全能力还原】不是【还原语料里最多的那一类】。
    等权的上限由最小域的池子定: academic 每份 ~1198 token ⇒ 严格等权 S 最大约 8×1198。
    池子不够的域, 拿满它全部池子, 缺口按池子大小回补给其余域(不留空域)。"""
    W = 128
    D = [d for d in byd if pool[d]]
    per = max(W, (N//len(D))//W*W)
    quota = {d: min(per, len(pool[d])//W*W or len(pool[d])) for d in D}
    short = N - sum(quota.values())
    while short > 0:                                     # 缺口按剩余池子大小回补
        cand = [d for d in D if len(pool[d]) - quota[d] >= W]
        if not cand: break
        d = max(cand, key=lambda x: len(pool[x]) - quota[x]); quota[d] += W; short -= W
    while sum(quota.values()) > N:                       # 超额从最大配额回收
        d = max(quota, key=lambda x: quota[x]); quota[d] -= W
    sel, got = [], {}
    for d in sorted(quota, key=lambda x: -quota[x]):
        p, q = pool[d], min(quota[d], len(pool[d])//W*W or len(pool[d]))
        nw = max(1, q//W)
        step = max(1, (len(p)-W)//max(1, nw-1)) if nw > 1 else 0
        s2 = []
        for k in range(nw): s2 += p[k*step : k*step+W]
        s2 = s2[:q]; got[d] = len(s2); sel += s2
    order = [d for d in sorted(quota, key=lambda x: -quota[x]) if got.get(d)]
    if len(sel) < N:                                     # 补足: 从最大域续窗
        d = max(pool, key=lambda x: len(pool[x])); need = N-len(sel)
        sel += pool[d][:need]; got[d] = got.get(d,0)+need; order.append(d)
    sel = sel[:N]
    open(path, "w").write("\n".join(str(t) for t in sel) + "\n")
    # ★布局落盘(2026-08-29 用户令"不要写死任何参数")★
    # 抽样把各域【连续】铺进 8192 行, 每域内部又是 W 宽的等距窗 —— 这个布局只有本函数
    # 知道。下游(行掩码/fit-val 切分)以前是把 "32块×256" 抄死在脚本里, 语料一换就静默
    # 错位, 而且没人记得回来改 —— 2026-08-29 的 z 全拒(378/378)就是这么来的。
    # 从此布局由生产方落盘, 消费方一律读这个文件; 文件缺失=硬失败, 不许猜。
    with open(path + ".layout", "w") as lf:
        lf.write("# %s 的行布局 — stage_idshalf 自动产出, 勿手改\n" % path.split("/")[-1])
        lf.write("# win=每域内的等距窗宽(窗与窗之间是源语料里的跳跃=上下文断点)\n")
        lf.write("# 每行: <域名> <起始行> <行数>\n")
        lf.write("win %d\n" % W)
        off = 0
        for d in order:
            lf.write("%s %d %d\n" % (d, off, got[d])); off += got[d]
    return sel, got

sels, gots = [], []
for i, (nm, path) in enumerate(OUTS):
    sel, got = sample(pools[i], N, path); sels.append(sel); gots.append(got)
print()
print("  抽样后每域 token(★全域都要有★):")
print("  %-10s %8s %8s %8s" % ("域", "量化份", "反修份", "判决份"))
for d in alld: print("  %-10s %8d %8d %8d" % (d, *[g.get(d,0) for g in gots]))
miss = [d for d in byd if any(g.get(d,0)==0 for g in gots)]
assert not miss, "★有域没被抽到: %s★" % miss

def ov(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b); na, nb = len(a), len(b)
    return sum(min(ca[t]/na, cb[t]/nb) for t in set(a)|set(b))
w = [int(x) for x in open(wt2) if x.strip()]
print()
print("  三份两两 token 分布重合(越低越不像):")
for i in range(K):
    for j in range(i+1, K):
        print("    %s ↔ %s  %.3f" % (OUTS[i][0], OUTS[j][0], ov(sels[i], sels[j])))
print("  三份对 wt2 词表覆盖: " + " / ".join(
    "%s %.1f%%" % (OUTS[i][0], 100*len(set(w)&set(sels[i]))/len(set(w))) for i in range(K)))
PY
    if [ -n "$T" ]; then
        cmp -s "$OQ" "$QI" && cmp -s "$OA" "$AI" && cmp -s "$OJ" "$JI" \
            || DIE "★重算 ids 与盘上不一致(切分不确定 或 语料变过) — 停, 不许拿算出来的布局配对不上的 ids★"
        cp "$OQ.layout" "$QI.layout"; cp "$OA.layout" "$AI.layout"; cp "$OJ.layout" "$JI.layout"
        rm -rf "$T"; LOG "①行布局补齐 ✓ (三份 ids 逐字节复现, 锚全部继续有效)"
    fi
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
    export DS4_BF_MEMGB=80 DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512
    export MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824
    export OPENBLAS_NUM_THREADS=1 DS4_THREADS=20 DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
    local NC L
    for NC in 256 512 1024; do
        local RP="$P/rplan_nc$NC.txt"
        : > "$RP"; for L in $(seq 0 42); do echo "L=$L dim=4 nc=$NC hot=0 w2dim=4 w2nc=$NC" >> "$RP"; done
        if grep -q "VQ_GATE" "$P/nc$NC.log" 2>/dev/null; then LOG "  档 vq4x$NC 已标定, 跳过"; continue; fi
        LOG "  档 vq4x$NC 标定发车(L0-5)"
        # QBIN_OVERRIDE: 显式钉现役量化器(2026-08-31 起 r30_campaign 默认已是它, 留着只为发车命令自明)。
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
    export DS4_BF_MEMGB=80 DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512
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
    # (历史口径风险已解除: 判决尺曾是冻结 ds4quant_run.old, 2026-08-31 判官归一后
    # caliper_ref.sh 走现行 ds4quant_run, .old 已删。读数仍应落 0.40-0.50 合理带才可信。)
    LOG "③dyn86 裸判(参考前向尺); 对表平权 vq86h_noz: KLD 0.47055 / Σmin 0.7799 / top1 78.36%"
    watchdog_start
    bash "$SC/caliper_ref.sh" "$P/model/layers" /tmp/qc_dyn86_wt2.bin 2>&1 | tail -14
    watchdog_stop
}

# ═══ 冠军配方反修(2026-08-27 用户令"按 67g 冠军版设计重新反修和路由反修")═══
# 【这不是新写的东西】直接调 r30_campaign.sh backfit —— 那就是超冠当时跑的那一段, 原封不动。
# 它一段里同时做完【反修 + 路由反修】, 不是两件事(2026-08-31 env 大扫除后为 flag/写死):
#   --bf-only         跳过 ALT 逐层(层内判据, 保险门实锤端到端负贡献)
#   ONEPASS(写死)     冻结基线一遍选型 + 统一终验 + 劣化全回滚(用户 08-04 裁决"一遍就够")
#   --gsweep 0        回扫不跑(侧车架构下需要时删侧车补跑即可)
#   --export-bytes 0  平行架构: dql 只读不可变, 只重建 op 侧车(用户"反修不许动量化模型")
#   --route-bias-fit + α2.5  ★路由偏置寄生在反修自身前向, 零额外前向★
#   锚路由(写死)      锚路由反修(超冠原样)
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
      champ86amp) OUT="$D2/champ86amp";;  # zlayer 反修完成态 ⇒ sweep 在它上面做
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
    # (2026-08-31 env 大扫除: 原"这四个必须不在场"的 unset DS4_TUNE/MINVOL/MV_BASELINE/VQ_RPLAN
    #  已无必要 — backfit 段现按 flag 拼装, 不传这些开关结构上就不会走错分支)
    # ★bug#5 修(2026-08-29)★ 原写法 `shift 2 2>/dev/null || true`: 只传 1 个参数时 shift 2
    # 失败, || true 把错吞了, "$@" 里还剩【工作区名】—— 于是 ds4quant_run <ids> 8192 champ86,
    # 那个位置本该是冠军的秩(r64c 传的是 64)。argv[3] 无人解析 ⇒ 静默丢弃, 秩落到 p14:289
    # 的 COADAPT 兜底 16。改成按实际参数个数 shift, 少于 2 个就清空透传。
    if [ "$#" -ge 2 ]; then shift 2; else shift "$#"; fi
    local PB=""; [ "$MODE" = probe ] && PB=1
    # 纯 VQ 底座无 go2b 热专家 ⇒ 关 GO2B_HOT(冠军底座是 go2b 热, 这是底座差异不是配方改动)
    local G2H=0
    LOG "冠军配方反修发车: $OUT ${PB:+(单层探针 L00)}"
    watchdog_start
    ( cd "$ROOT" && env ${PB:+PROBE1=1} OUTF_OVERRIDE="$OUT" ANCHOR_OVERRIDE="$D2/anchor_a_clean_s8192.bin" \
        IDS_OVERRIDE="$D2/vqhalf_a.ids" BF_S=8192 BF_NFIT=6144 DS4_THREADS=20 \
        QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" DS4_GSWEEP=0 DS4_GO2B_HOT=$G2H \
        bash "$SC/r30_campaign.sh" backfit "$@" ) > "$OUT/backfit.log" 2>&1
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
        export DS4_BF_MEMGB=55 DS4_CALIB_CAP=512 DS4_CALIB_EXPORT_CAP=512
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
    # ★量化态立刻备份(2026-08-28 用户令, 破坏前先保全铁律)★
    # 反修/sweep 是【往 dql 层文件里追加记录 + 原地改写】—— 改的就是量化产物本身。
    # 今天 champreset 能救回来靠的是只读原件 vq86h_noz, 但那是【旧语料】量化出来的;
    # 换语料后新底座没有任何后备, 反修一跑量化态就永久没了, 想重来只能整轮重量化(25 min)。
    # 这里存一份纯净量化态, 后续任意次反修实验都从它还原, 不用重量化。
    # ext4 不支持 reflink(实测), 只能真拷贝: 37GB / 约 30s / 盘上余 2.1T。
    if [ ! -d "$W/layers_quant" ]; then
        LOG "备份量化态层件 → layers_quant(反修唯一还原点)"
        cp -a "$W/layers" "$W/layers_quant.part" || DIE "量化态备份失败"
        mv "$W/layers_quant.part" "$W/layers_quant"
        if diff -rq "$W/layers" "$W/layers_quant" >/dev/null 2>&1; then
            LOG "备份 ✓ 逐字节一致 $(du -sh "$W/layers_quant" | cut -f1)"
        else DIE "★备份逐字节复核失败★"; fi
    else LOG "量化态备份已在, 跳过"; fi
    # ①b 裸态 wt2 五指标(2026-08-31 全链重跑补位): 量化产物先出自己的官方尺读数,
    # 反修的增益才有同一轮的裸对照 —— 不再拿旧语料 vq86h_noz 的 0.47055 隔轮对表。
    LOG "①b 裸态 wt2 官方尺(layers_quant)"
    bash "$SC/caliper_ref.sh" "$W/layers_quant" /tmp/qc_champ86_bare_wt2.bin \
        > /tmp/caliper_champ86_bare.log 2>&1 || { tail -3 /tmp/caliper_champ86_bare.log; DIE "裸判失败"; }
    grep -aE "PPL\(stu|分布还原率|Mean KLD|Same top" /tmp/caliper_champ86_bare.log
    # ★②反修+sweep: 改走冠军 zlayer 路(2026-08-29)★
    # 原来这里是 stage_champbf → r30_campaign backfit → ds4quant_run 内建反修。三处与
    # 冠军 r64c(0.42510 = −9.7%)不符, 合起来让 z 落地 378/378 全拒:
    #   ①ds4quant_run 反修路【没有行掩码】(DS4_ZL_FIT_RANGES 只有 zlayer 认, zlayer_p4:285)
    #     ⇒ fit/val 按行号切; 而新语料 8 个域各占一个连续 1024 行块 ⇒ 拉丁上拟合、
    #     西里尔上判落地、阿拉伯+中日韩当 held(解码实测三段几乎零重叠)
    #   ②秩顶格静默兜底 16(p14:289 COADAPT 分支), 冠军是 K=64
    #   ③冠军压根不走这条: r64c = amp_clean_full.sh → zlayer 二进制 ×43(fable5 6685)
    # sweep 在 zlayer 内部就有(p5:50-54 秩网格逐秩算 held 取最大 / p5:48 落地闸 /
    # p7:141 不过闸写空 zrec), 不需要 ds4quant_run 那个"终局收敛 sweep" —— 后者正是
    # 08-29 产出 42 层过拟合标量增益(判决份四项全负)的来源。
    LOG "②反修+sweep(冠军 zlayer 路 K=64; 行掩码从 vqhalf_a.ids.layout 读, 缺则硬停)"
    SRCBASE=champ86/layers_quant bash "$SC/amp_clean_full.sh" \
        champ86amp "" 64 "" "$D2/anchor_a_clean_s8192.bin" \
        || DIE "冠军路反修失败"
    # ★③sweep 已下链(2026-08-31 chain9 定谳, fable5)★: 部署真尺(g_bkl_live 抑制钉路)下
    # 逐单元真降不可组合(全 8192 行终验 0.74096→0.74271 劣化), GL/GE/z重解 op 族在
    # 跨语料闸落地前无净肉 —— 生产链只到 ②反修完成态(=③态交付物)。
    # 单独复扫走 `amp_campaign.sh champbf champ86amp full`, 不进本段。
    # ④五指标: amp_clean_full 的 ④ 已出 wt2 官方尺; 这里补【判决份同域全能力尺】(裸+反修后两跑)
    LOG "④判决份同域尺; wt2 官方尺已由 ② 内部跑完(裸对照见 ①b 本轮读数)"
    stage_judge3 champ86amp
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
    # ★还原点优先级★: ①本战役自己的量化态备份 layers_quant(同语料同配方, 唯一正确的还原点)
    # ②只读原件 vq86h_noz(旧语料, 只在没备份时兜底 —— 换语料后它已不是同一个底座)
    local SRC="$D2/vq86h_noz"
    [ -d "$W/layers_quant" ] && SRC="$W"   # layers_quant 就在 W 底下, 下面统一按 $SRC/layers 取
    local SUB="layers"
    [ "$SRC" = "$W" ] && SUB="layers_quant"
    [ -d "$SRC/$SUB" ] || DIE "还原点不在: $SRC/$SUB"
    LOG "还原点 = $SRC/$SUB"
    [ -d "$W/layers" ]   || DIE "工作副本不在: $W/layers"
    local n_rm=0 n_cp=0 n_ok=0
    # ① 原件没有的文件 = 反修产物, 删
    for f in "$W/layers"/*; do
        local b; b=$(basename "$f")
        [ -e "$SRC/$SUB/$b" ] || { rm -f "$f"; n_rm=$((n_rm+1)); }
    done
    # ② 原件有的: 逐字节比, 不等就从原件覆盖
    for f in "$SRC/$SUB"/*; do
        local b; b=$(basename "$f")
        if cmp -s "$f" "$W/layers/$b"; then n_ok=$((n_ok+1))
        else cp -f "$f" "$W/layers/$b"; n_cp=$((n_cp+1)); LOG "  复位 $b"; fi
    done
    rm -f "$W/backfit.log" "$W/route_bias_r30.bin" "$W/route_bias_r30.bin.alpha.txt" "$W/rb_alpha.txt"
    LOG "层件复位完成: 原样 $n_ok / 覆盖 $n_cp / 删反修产物 $n_rm"
    # ③ 复位后逐字节全量复核(不许只信上面的循环)
    if diff -rq "$SRC/$SUB" "$W/layers" > /tmp/champ_reset_diff.txt 2>&1; then
        LOG "★复核通过: champ86/layers 与还原点逐字节一致★"
    else DIE "复核失败, 差异见 /tmp/champ_reset_diff.txt: $(head -3 /tmp/champ_reset_diff.txt)"; fi
}

# ═══ 三份锚捕获 + 判决份同域尺(2026-08-28 用户令"切三份: 量化/反修/判决")═══
# ids 一变锚就废(锚与 ids 错配会出 PPL 2.4e7 这种一眼假的数), 所以三份各配一个 FP 锚。
# ★两把尺并存★: wt2 仍是官方判决(铁律"判决只认参考前向尺"+所有历史数字都在它上面:
#   官方 q2 KLD 0.4207 / 平权裸 Σmin 0.7799 / 冠军对表); 判决份是【同域全能力尺】——
#   8 个域齐全, 覆盖 wt2 完全不测的代码/数学/多语。两个数一起报, 不互相取代。
stage_anchors3(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local NAMES=(量化 反修 判决)
    local IDSF=("$D2/vqhalf_q.ids" "$D2/vqhalf_a.ids" "$D2/vqhalf_j.ids")
    local ANCF=("$D2/anchor_vqhalf_q_s8192.bin" "$D2/anchor_a_clean_s8192.bin" "$D2/anchor_j_s8192.bin")
    local i
    for i in 0 1 2; do
        [ -s "${IDSF[$i]}" ] || DIE "${NAMES[$i]}份 ids 缺: ${IDSF[$i]}"
        # 锚比 ids 旧 = 上一版 ids 的锚, 必须重捕
        if [ -s "${ANCF[$i]}" ] && [ "${ANCF[$i]}" -nt "${IDSF[$i]}" ]; then
            LOG "${NAMES[$i]}锚已是最新, 跳过"; continue; fi
        LOG "捕${NAMES[$i]}锚 S=8192 (约 30 分钟, 23GB)"
        rm -f "${ANCF[$i]}"
        watchdog_start
        # 判官归一(2026-08-31): .old 冻结件已删, 建锚同走现行 ds4quant_run + flag
        ( cd "$ROOT/gguf-tools/amp" && env OPENBLAS_NUM_THREADS=1 \
            ./ds4quant_run "${IDSF[$i]}" 8192 --hf "$ROOT/hf/DeepSeek-V4-Flash-0731" \
            --threads 20 --bf-memgb 55 --fp-only --anchor "${ANCF[$i]}" ) \
            > "/tmp/anc3_$i.log" 2>&1
        watchdog_stop
        [ -s "${ANCF[$i]}" ] || { tail -5 "/tmp/anc3_$i.log"; DIE "${NAMES[$i]}锚没落盘"; }
        # ★布局随锚走(2026-08-29)★: 消费方(zlayer / ds4quant_run 的 sweep)手里只有【锚路径】,
        # 没有 ids 路径。把 <ids>.layout 复制成 <锚>.layout, 它们就能从自己已有的路径推导出
        # 行域, 不需要任何 env 开关、不需要额外参数。缺布局 = 硬停, 不许回退到"按行号切"。
        [ -s "${IDSF[$i]}.layout" ] || DIE "${NAMES[$i]}份行布局缺(先跑 idshalf 补)"
        cp -f "${IDSF[$i]}.layout" "${ANCF[$i]}.layout"
        LOG "${NAMES[$i]}锚 ✓ $(ls -l "${ANCF[$i]}" | awk '{printf "%.1f GiB", $5/1073741824}')"
    done
}
# 判决份同域尺: 对任意层件目录出五指标(与 wt2 尺同一把 caliper, 只换 ids/锚)
stage_judge3(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" V="${1:-champ86}"
    local JI="$D2/vqhalf_j.ids" JA="$D2/anchor_j_s8192.bin"
    [ -s "$JA" ] && [ "$JA" -nt "$JI" ] || DIE "判决份锚缺或过期, 先跑 anchors3"
    echo "══ 同域全能力尺(判决份 8 域齐全) ══"
    for B in vq86h_noz "$V"; do
        [ -d "$D2/$B/layers" ] || continue
        LOG "尺: $B"
        bash "$SC/caliper_ref.sh" "$D2/$B/layers" "/tmp/j3_$B.bin" 20 "" 2.5 "$JI" "$JA" \
            > "/tmp/j3_$B.log" 2>&1 || { LOG "$B 失败"; tail -3 "/tmp/j3_$B.log"; continue; }
        printf -- "── %s ──\n" "$B"
        grep -aE "PPL\(student\)|分布还原率|Mean KLD|Same top" "/tmp/j3_$B.log"
    done
    echo "★注: 这是同域尺, 不取代 wt2 官方判决 —— 两个数一起看★"
}

# ═══ 三段单层针(2026-08-28 用户令"量化/反修/sweep 各一层, 速度质量都要")═══
# 用【旧锚+配套旧 ids】跑, 不等新锚(新锚要 90 分钟, 而这针要的是速度和质量的当前读数)。
#   锚 anchor_a_clean_s8192.bin(8月24, 30.8G) ↔ old_split/vqhalf_a.ids —— 这一对是配套的;
#   量化锚已被 anchors3 的 rm -f 删掉(重捕被我停在半路), 所以量化段也借这一对, 反正针只量
#   速度和质量, 不产交付物。
# 三段各跑一层, 每段单独计墙钟 + 打质量:
#   ①量化一层: 时间 + VQ 逐层 cos/bpw + 收官 VERDICT(Σmin/KL/top1)
#   ②反修一层: 时间 + 分布还原率(Σmin/KL/top1) + 逐段计时 [LT]
#   ③sweep 一层: 时间 + BFUNIT Δ + 出口分
# 目标 20s/段(用户令)。跑完把三个数并排打出来, 达不到就报实测不粉饰。
stage_probe3(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local W="$D2/champ86"
    local AN="$D2/anchor_a_clean_s8192.bin"
    local ID="$D2/old_split/vqhalf_a.ids"
    local PW="$D2/p3"
    [ -s "$AN" ] || DIE "旧反修锚不在: $AN"
    [ -s "$ID" ] || DIE "旧 ids 不在: $ID"
    local T0 T1 T2 T3
    # ── ①量化一层(写进独立 scratch, 不碰任何交付层件) ──
    LOG "①量化一层 → $PW"
    rm -rf "$PW"; mkdir -p "$PW"
    T0=$(date +%s)
    watchdog_start
    # ★闸要发 DS4_MINVOL_MAXL 不是 PROBE1★: quant86 本来就带 DS4_MINVOL=1, C 侧早退闸
    # (p12) 认的是 DS4_MINVOL_MAXL; PROBE1 是【旧量化段】的开关, 对 quant86 不起作用 ——
    # 实撞: 发 PROBE1 跑到第 2 层还没停。两个都是既有变量, 不算新增 env。
    ( cd "$ROOT" && env DS4_MINVOL_MAXL=1 QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" \
        Q86_IDS="$ID" Q86_S=8192 Q86_NFIT=6144 Q86_ANCHOR="$AN" Q86_OUT="$PW" \
        RPLAN86="$ROOT/gguf/go-onebit/r30/rplan_base86p.txt" VOLB86=76 \
        DS4_BF_MEMGB=55 DS4_THREADS=20 OPENBLAS_NUM_THREADS=1 \
        DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731" \
        bash "$SC/r30_campaign.sh" quant86 ) > /tmp/p3_quant.log 2>&1
    watchdog_stop
    T1=$(date +%s)
    # ── ②反修一层(层件先复位到纯净量化态) ──
    LOG "②反修一层"
    stage_champ_reset >/dev/null 2>&1 || true
    T1=$(date +%s)
    stage_champbf champ86 probe >/dev/null 2>&1 || true
    cp -f "$W/backfit.log" /tmp/p3_bf.log 2>/dev/null || true
    T2=$(date +%s)
    # ── ③sweep 一层(要 ≥2 层才有 BFUNIT 单元: 前沿 L1 修前层 L0) ──
    LOG "③sweep 一层"
    stage_champ_reset >/dev/null 2>&1 || true
    T2=$(date +%s)
    ( cd "$ROOT" && env PROBE_NL2=1 OUTF_OVERRIDE="$W" ANCHOR_OVERRIDE="$AN" IDS_OVERRIDE="$ID" \
        BF_S=8192 BF_NFIT=6144 DS4_THREADS=20 DS4_NL=2 DS4_GSWEEP=0 DS4_GO2B_HOT=0 \
        DS4_BF_MEMGB=55 OPENBLAS_NUM_THREADS=1 DS4_HF="$ROOT/hf/DeepSeek-V4-Flash-0731" \
        MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824 \
        QBIN_OVERRIDE="$ROOT/gguf-tools/amp/ds4quant_run" \
        bash "$SC/r30_campaign.sh" backfit ) > /tmp/p3_sweep.log 2>&1
    T3=$(date +%s)
    echo
    echo "════════ 三段单层针 ════════"
    printf "①量化一层   墙钟 %3ds\n" $((T1-T0))
    grep -aE "cos=|bpw|VERDICT" /tmp/p3_quant.log | tail -4 | sed 's/^/    /'
    printf "②反修一层   墙钟 %3ds\n" $((T2-T1))
    grep -aE "\[LT\] L00|分布还原率" /tmp/p3_bf.log 2>/dev/null | tail -3 | sed 's/^/    /'
    printf "③sweep 一层 墙钟 %3ds\n" $((T3-T2))
    grep -aE "^BFUNIT|出口L|BF_ONEPASS 终验" /tmp/p3_sweep.log | tail -4 | sed 's/^/    /'
    echo "★目标 20s/段。日志: /tmp/p3_{quant,bf,sweep}.log★"
}

# ═══ ELM 针(2026-08-28 用户令"1、2 打一针再决策")═══
# 要回答两件事, 都靠同一次跑的读数, 不靠推理:
#   ①ELM 闭式乘性非线性 z 在【本底座本语料】上 held 行为挽回能不能上 10%
#     (历史 L35=+12.02% / L40 全场景=+14.1%, 但那是单层探针, 本底座没验过)
#   ②同口径【乘性线性】对照打多少(判例 +0.6%) —— 若两者接近, 说明 tanh 在这个底座上没肉,
#     ELM 整条路要重估; 若拉开数量级, 八要件里的"非线性"就坐实了。
# 针只读不写: 不落侧车、不改层件、不动模型。层选 L2/L20/L40(与历史三层针同位置, 可直接对表)。
# 前置: ①反修锚必须与 vqhalf_a.ids 配套(anchors3) ②层件必须是纯净量化态(champreset)。
stage_elmprobe(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local W="$D2/champ86"
    local AN="$D2/anchor_a_clean_s8192.bin"
    local ID="$D2/vqhalf_a.ids"
    [ -s "$AN" ] && [ "$AN" -nt "$ID" ] || DIE "反修锚缺或比 ids 旧, 先跑 anchors3"
    [ "$(ls "$W/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)" = 43 ] || DIE "层件不齐, 先量化"
    LOG "层件复位到纯净量化态(针不能吃带反修 op 的层件)"
    stage_champ_reset
    LOG "ELM 针发车: L2/L20/L40, 历史对表 编程 7.5/5.1/12.0 · 全场景 6.8/4.7/12.3"
    stage_champbf champ86 full --elm-probe 2,20,40
    echo "══ ELM 针结果 ══"
    grep -aE "^★ELM " "$W/backfit.log" || echo "(没有 ELM 行 —— 钩子没触发, 查 --elm-probe 是否透传到 QBIN)"
    echo "★读法: held ≥10% ⇒ 八要件成立可上全量; 若与'乘性线性对照'接近 ⇒ tanh 无肉, 整条路重估★"
}

# ═══ champ3: 三份语料全链一条龙(2026-08-28 用户令"三份 8192 每个域都有")═══
# ①三锚 → ②平权 86G 量化(量化份) → ③冠军反修+路由 Δb(反修份) → ④双尺判决
# 中间不留人工接力(接力是今天多次事故的来源)。语料换了 ⇒ 层件必须重量化, 所以先清 champ86;
# 只读原件 vq86h_noz 与 95G 备份都不动。
stage_champ3(){
    local D2="$ROOT/gguf/go-onebit/vqhalf"
    local W="$D2/champ86"   # 见 champreset 注释: local 同句不能引用前一个名字
    [ -s "$D2/vqhalf_q.ids" ] && [ -s "$D2/vqhalf_a.ids" ] && [ -s "$D2/vqhalf_j.ids" ] \
        || DIE "三份 ids 不齐, 先跑 idshalf"
    # ★行布局必须在场(2026-08-29)★: ②反修的行掩码从 <ids>.layout 读, 缺了会硬停。
    # idshalf 在 ids 已存在时不会重切, 只做"确定性重算 + 逐字节校验 + 装布局", 不动 ids
    # ⇒ 三个锚(各 30.8G/57 分钟)不作废。
    stage_idshalf
    stage_anchors3
    if [ -d "$W" ]; then
        cp -f "$W/backfit.log" /tmp/champ86_prev_backfit.log 2>/dev/null || true
        LOG "语料已换 ⇒ 清 champ86 重量化(只读原件 vq86h_noz 不动; 上轮日志留 /tmp)"
        rm -rf "$W"
    fi
    stage_champ86          # ①量化 ②反修 ③wt2 官方五指标
    stage_judge3 champ86   # ④判决份同域全能力尺
}

# ═══ 语料对拍: 判决尺(wt2) vs 切半的开源语料(datav5 两半)(2026-08-28 用户令)═══
# 把三份 ids 解回文本逐项量: 字符构成 / 代码占比 / 词表重合 / token 分布重合。
# 目的是给"校准料和判决尺到底差多远"一个硬数字, 不再靠"感觉像"。
stage_corpdiff(){
    local D2="$ROOT/gguf/go-onebit/vqhalf" G7="$ROOT/gguf/go-onebit/g7"
    python3 - "$DS4_HF" "$G7/wt2.ids" "$D2/vqhalf_q.ids" "$D2/vqhalf_a.ids" <<'PY2'
import sys, re, collections, math
from tokenizers import Tokenizer
hf = sys.argv[1]
tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
names = ["判决尺 wt2", "量化半 q", "反修半 a"]
paths = sys.argv[2:5]
docs = []
for nm, p in zip(names, paths):
    ids = [int(x) for x in open(p) if x.strip()]
    txt = tok.decode(ids)
    docs.append((nm, ids, txt))

def compo(t):
    f = [m.start() for m in re.finditer("```", t)]
    code = set()
    for i in range(0, len(f)-1, 2):
        code.update(range(f[i], f[i+1]))
    c = collections.Counter()
    for i, ch in enumerate(t):
        if ch.isspace(): continue
        o = ord(ch)
        if i in code: k = "代码围栏"
        elif 0x3040 <= o <= 0x30ff or 0x4e00 <= o <= 0x9fff: k = "CJK"
        elif 0x400 <= o <= 0x4ff: k = "西里尔"
        elif 0x600 <= o <= 0x6ff: k = "阿拉伯"
        elif o < 128: k = "ASCII"
        else: k = "其他非ASCII"
        c[k] += 1
    return c

print("═══ ① 体量与字符构成 ═══")
print("%-12s %8s %9s | %s" % ("语料", "token", "字符", "构成(非空白)"))
for nm, ids, txt in docs:
    c = compo(txt); tot = sum(c.values())
    top = "  ".join("%s %.1f%%" % (k, 100*v/tot) for k, v in c.most_common(4))
    print("%-12s %8d %9d | %s" % (nm, len(ids), len(txt), top))

print()
print("═══ ② 内容指纹(每千字符出现次数) ═══")
pats = [("LaTeX $..$", r"\$[^$\n]{2,}\$"), ("代码围栏 ```", r"```"),
        ("数学题触发词", r"\b(How many|what is the|Find the|Calculate)\b"),
        ("维基式括注 (born|born in|is a)", r"\b(is a|was a|born)\b"),
        ("URL/markup", r"https?://|\{ref-type|\[@ref")]
print("%-32s %s" % ("指纹", "  ".join("%-12s" % n for n, _, in [(d[0], 0) for d in docs])))
for nm, pat in pats:
    row = []
    for _, _, txt in docs:
        row.append("%-12.2f" % (1000.0*len(re.findall(pat, txt, re.I))/max(len(txt), 1)))
    print("%-32s %s" % (nm, "  ".join(row)))

print()
print("═══ ③ 词表重合 / token 分布重合(以判决尺 wt2 为基准) ═══")
base_ids = docs[0][1]
bs, bc = set(base_ids), collections.Counter(base_ids)
bn = sum(bc.values())
for nm, ids, _ in docs[1:]:
    s2, c2 = set(ids), collections.Counter(ids)
    n2 = sum(c2.values())
    jac = len(bs & s2) / len(bs | s2)
    cov = len(bs & s2) / len(bs)          # wt2 的词有多少被校准料见过
    over = sum(min(bc[t]/bn, c2[t]/n2) for t in bs | s2)   # 分布重合(Σmin, 与主尺同式)
    print("  %-10s 唯一token %5d | Jaccard %.3f | 覆盖wt2词表 %.1f%% | ★token分布重合 %.3f★"
          % (nm, len(s2), jac, 100*cov, over))
print("  %-10s 唯一token %5d (基准)" % (docs[0][0], len(bs)))

print()
print("═══ ④ 实际长相(各取 200 字符) ═══")
for nm, _, txt in docs:
    print("[%s] %s" % (nm, txt[300:500].replace("\n", " ⏎ ")))
PY2
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
        # 判官归一(2026-08-31): .old 冻结件已删, 建锚同走现行 ds4quant_run + flag
        ( cd "$ROOT/gguf-tools/amp" && env OPENBLAS_NUM_THREADS=1 \
            ./ds4quant_run "$CI" 8192 --hf "$ROOT/hf/DeepSeek-V4-Flash-0731" \
            --threads 20 --bf-memgb 55 --fp-only --anchor "$CA" ) > /tmp/anc_c.log 2>&1
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
  chain) stage_chain;; judge) stage_judge;; dilute) stage_dilute;; idshalf) stage_idshalf;; idshalf_ext) shift; stage_idshalf_ext "$@";; vqquant) stage_vqquant;; vqmerge) stage_vqmerge;; vqcap) stage_vqcap;; vqsolve) stage_vqsolve;; dynladder) stage_dynladder;; dynquant) shift; stage_dynquant "$@";; dynjudge) stage_dynjudge;; champbf) shift; stage_champbf "$@";; champ86) stage_champ86;; champreset) stage_champ_reset;; champ3rd) stage_champ3rd;; corpdiff) stage_corpdiff;; anchors3) stage_anchors3;; champ3) stage_champ3;; elmprobe) stage_elmprobe;; probe3) stage_probe3;; judge3) shift; stage_judge3 "$@";; champrb) stage_champ_rbsweep;; full) shift; stage_full "$@";;
  all) stage_preflight; stage_ids; stage_anchor; stage_capture
       stage_solve; stage_chain; stage_judge;;
  *) echo "未知段: $ST"; echo "段: preflight ids anchor capture solve pass2 chain judge dilute all (probe/dilute=诊断)"; exit 2;;
esac
LOG "段 $ST 完成"
