#!/bin/bash
# quant_layer.sh [分钟|fast] —— 43 层全模型量化方案(表格=第一原则)。
#
# 参数语义: 分钟 = 结束当前脚本的时间(硬停量化; 层文件全齐→原地自动merge收尾, 未齐不合并);
#           fast = 快速走流程(全 43 层·S=16·轮数封顶·不向前修复·跳 42G 骨架), 只验流程通不出质量;
#           不填 = 执行到全部完成(43 层收敛 + 合并), 满档校准。
# 层规则: 收敛绝对优先 — 每层必须(本层+累积)整轮零接管才进下一层, 优化永不截断(fast 除外);
#         每层收敛后当场留存该层量化文件(可单层重跑微调)。
# 反修规则(裁决2026-07-12): fast=不向前修复(无逐层反修/无末层全量/GSWEEP默认关);
#         非fast=每一层完成即全量反修所有前层("末层最后一次全量"特例已删)。
# 截停收尾(裁决2026-07-12): 看门狗/到时/异常退出时若层文件全齐 → 原地自动merge(补侧车+表+GGUF),
#         不再需要第二遍 ./quant_layer.sh merge; Ctrl+C 中断仍立即停(事后手动恢复才用 merge 入口)。
# 校准换档: 不填/≥120分→S=305 | ≥30→128 | <30→64; 不填/≥60 开终端反调。
# 产物(全部本地可查):
#   layer-tables/L00.md … L42.md        每层明细(元素表+探索日志+逐元素日志)
#   layer-tables/ALL.md                 总方案: 每层最优一览 + 全模型 VERDICT(vs 原始大模型)
#   layer-tables/raw/all.out            原始输出(每个数字可回查)
#   gguf/go-onebit/layers/dql_L<NN>.bin 每层【量化】文件(1bit 权重字节+全部元素记录)
#   gguf/go-onebit/layers/opt_L<NN>.bin 每层【优化】文件(最优 z 链/向后/GE, 单层 DQZ2; 四损失+感知已重解进系数)
#   gguf/go-onebit/zchain_all.bin       优化链汇总(DQZ2 43 层终值 = 43 份 opt 的合订)
#   gguf/go-onebit/zfile_all.bin        SEARCH 胜者摘要侧车(台账)
#   gguf/go-onebit/ds4-code1b.gguf      ★最终合一 GGUF: 43 量化+43 优化(86 份)合并 —
#                                       1bit 专家字节 + blk.L.opt_* 优化张量 + ds4.zchain.present,
#                                       引擎单文件自动装载(无需外挂)
# 模块: M-1 清除上次遗留 → M0 环境闸 → M1 构建 → M2 执行 → M3 完整性 → M4 出表+合并 → M5 汇报
set -euo pipefail

ARG="${1:-}"; FAST=0; TMIN=""; MODE=""
case "$ARG" in
    fast|FAST|f) FAST=1 ;;
    merge|MERGE) MODE=merge ;;   # merge-only: 复用既有层文件+OUT, 只补 M3表+M4.6 GGUF。裁决2026-07-12: 看门狗/到时/异常截停且层齐时脚本已原地自动merge, 本入口只留作手动恢复(如 Ctrl+C 中断后)/幂等重并
    backfit|BACKFIT|bf) MODE=backfit ;;   # 只跑反修(裁决2026-07-13): 复用完整43层dql(含已落地修正), SEARCH跳过按op链回放推进 → 终局收敛sweep+反调+回扫+合并; 产物不清理, OUT/LOG 追加
    "") : ;;
    *[!0-9.]*) echo "参数非法: '$ARG' — 用【分钟数】或【fast】或【merge】或【空=跑到完成】" >&2; exit 1 ;;
    *) TMIN="$ARG" ;;
esac
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ---------- 进程纪律: 启动默认杀旧实例; Ctrl+C/TERM 全链带走(本机, 远端在 M0 转发段处理) ----------
SELF_PID=$$
kill_stale(){
    pkill -9 -f 'ds4quant_run' 2>/dev/null || true
    for p in $(pgrep -f 'quant_layer\.sh' 2>/dev/null); do
        [ "$p" = "$SELF_PID" ] || [ "$p" = "$PPID" ] || kill -9 "$p" 2>/dev/null || true
    done
}
kill_stale
echo "【全模型】脚本 算法=进程纪律 进度=启动 体积=- 还原度=- 研判=旧实例已默认清杀" >&2
trap 'echo "[中断] 杀本机量化进程" >&2; pkill -9 -f ds4quant_run 2>/dev/null || true; exit 130' INT TERM

# ---------- 换档 ----------
if [ "$FAST" = 1 ]; then
    NLAY="${DS4_FAST_LAYERS:-43}"; NTOK="${DS4_FAST_NTOK:-16}"; BWD=1; GSW="${DS4_GSWEEP:-0}"   # fast=不向前修复: 逐层反修 C 侧按 DS4_FAST 硬跳, GSWEEP 回扫默认也关(DS4_GSWEEP=1 可开; 全缓存路径有 12G OOM bug 待修)
    echo "【全模型】脚本 算法=快速模式 进度=开始 体积=- 还原度=- 研判=全${NLAY}层快速档: S=${NTOK}·DS4_FAST(QC1+2轮封顶)·含末层终局反修(裁决2026-07-14)·GSWEEP=${GSW}·产物完整(dql/opt/zchain/合一GGUF)" >&2
elif [ -z "$TMIN" ]; then NLAY=43; NTOK="${DS4_NTOK:-305}"
    # ★反修前提铁律(用户裁决2026-07-22)★: 反修族(终端反调BWD/终端反修TERM/回扫GSWEEP/复检)
    # 必须先有"指标(rr)+真实场景(行为门)"双判决依据才许开——满档默认全关, 出基线模型先判,
    # 有问题且反修对症再走 ./quant_layer.sh backfit(带 DS4_BF_JUSTIFIED=1)。
    # 实证依据: v2 重型终端反修 rr 零贡献+行为面回退(假工具帧, fable5 2026-07-22 三腿合判)。
    BWD="${DS4_BWD:-0}"; GSW="${DS4_GSWEEP:-0}"
    export DS4_BF_TERM_MAXP="${DS4_BF_TERM_MAXP:-0}"
    export DS4_BF_NO_RECHECK="${DS4_BF_NO_RECHECK:-1}"   # 判决前置(2026-07-21): 复检遍默认关
else
    NLAY=43; GSW="${DS4_GSWEEP:-3}"
    TI=${TMIN%.*}; [ -n "$TI" ] || TI=0
    if   [ "$TI" -ge 120 ]; then NTOK=305
    elif [ "$TI" -ge 30 ];  then NTOK=128
    else                         NTOK=64; fi
    BWD=0; [ "$TI" -ge 60 ] && BWD=1
    echo "【全模型】脚本 算法=换档 进度=开始 体积=- 还原度=- 研判=${TMIN}分后硬停, 校准S=$NTOK, 终端反调=$BWD, 每层全量反修前层; 层收敛不截断" >&2
fi
LMAX=$((NLAY-1))
[ "$MODE" = merge ] || [ "$MODE" = backfit ] || [ "$FAST" = 1 ] || [ -n "$TMIN" ] || echo "【全模型】脚本 算法=满档 进度=开始 体积=- 还原度=- 研判=跑到完成(${NLAY}层收敛+合并), 校准S=$NTOK, 终端反调=$BWD, 每层全量反修前层; 层收敛不截断" >&2

# ---------- 表格元素全集(每层核对) ----------
REQ_EXACT="1bit loss.align loss.cls loss.fix loss.smooth percept"
REQ_PREFIX="z."

# ---------- M0 环境闸 ----------
HF="${DS4_HF:-/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base}"
if [ ! -d "$HF" ]; then
    REMOTE="${DS4_REMOTE:-192.168.1.2}"; RPATH="${DS4_REMOTE_ROOT:-/Users/fodelf/ds4-main}"
    echo "【全模型】脚本 算法=ssh转发 进度=开始 体积=- 还原度=- 研判=本机无HF→转 $REMOTE" >&2
    EFWD=""   # ★转发本机 DS4_* 调优 env 到 M1(否则 DS4_FAST_LAYERS 等在 ssh 后丢失)★
    for v in DS4_FAST_LAYERS DS4_FAST_NTOK DS4_NTOK DS4_GSWEEP DS4_BWD DS4_BF_TERM_MAXP DS4_BACKFIT_INCR DS4_CORPUS DS4_THREADS DS4_SKELETON DS4_SIGNREF_MU; do
        eval "val=\${$v:-}"; [ -n "$val" ] && EFWD="$EFWD $v='$val'"
    done
    echo "【全模型】脚本 算法=ssh转发 进度=- 体积=- 还原度=- 研判=转发env:${EFWD:-无}; Ctrl+C/kill 将同时清远端" >&2
    # ★ssh 放后台 + wait(可被信号打断)★: 前台 ssh 会把 trap 挂起到 ssh 结束(实测), 且无 TTY 时
    # 远端收不到任何信号 = "停了还在跑"根因。trap 里显式清远端 + 杀本地 ssh。
    ssh -o ServerAliveInterval=15 "$REMOTE" "$EFWD $RPATH/quant_layer.sh ${ARG:-}" &
    SSHPID=$!
    trap 'echo "[中断] 杀本地ssh+清远端进程..." >&2; kill -9 "$SSHPID" 2>/dev/null || true;
          ssh -o ConnectTimeout=5 "$REMOTE" "pkill -9 -f ds4quant_run 2>/dev/null; pkill -9 -f quant_layer 2>/dev/null" 2>/dev/null || true;
          echo "[中断] 远端已清" >&2; exit 130' INT TERM
    RC=0; wait "$SSHPID" || RC=$?   # ||捕获: set -e 下裸 wait 收远端非零码会当场杀本地脚本, 明细表拉不回
    trap 'echo "[中断] 杀本机量化进程" >&2; pkill -9 -f ds4quant_run 2>/dev/null || true; exit 130' INT TERM
    rsync -a "$REMOTE:$RPATH/gguf-tools/go-onebit/layer-tables/" "$ROOT/gguf-tools/go-onebit/layer-tables/" 2>/dev/null || true
    [ "$RC" = 0 ] && echo "【全模型】脚本 算法=ssh转发 进度=完成 体积=- 还原度=- 研判=明细已拉回 layer-tables/; 侧车在 $REMOTE:$RPATH/gguf/go-onebit/zfile_all.bin"
    exit "$RC"
fi
QDIR="$ROOT/gguf-tools/go-onebit/quant"
TBL="$ROOT/gguf-tools/go-onebit/layer-tables"
LDIR="$ROOT/gguf/go-onebit/layers"
# ★对齐目标 = 编程域★ (rr_code.ids 编程语料; 覆盖用 DS4_CORPUS)
CORPUS="${DS4_CORPUS:-/tmp/rr_code.ids}"
DOMAIN="code"
zchain_fix(){   # 截停恢复: 优化链侧车缺失/过期(量化没跑到 zchain_write 点)→ 从 dql 秒级重建 43 份 opt + zchain; $1=横幅算法名
    ZCB0="$ROOT/gguf/go-onebit/zchain_all.bin"
    NEWEST_DQL=$(ls -t "$LDIR"/dql_L*.bin 2>/dev/null | head -1)
    if [ -n "$NEWEST_DQL" ] && { [ ! -s "$ZCB0" ] || [ "$ZCB0" -ot "$NEWEST_DQL" ]; }; then
        echo "【全模型】脚本 算法=$1 进度=补优化侧车 体积=- 还原度=- 研判=zchain/opt 缺失或旧于层文件 → DS4_ZCHAIN_ONLY 从 dql 重建" >&2
        DS4_ZCHAIN_ONLY=1 DS4_LAYER_DIR="$LDIR" DS4_ZCHAIN="$ZCB0" DS4_NL="$NLAY" ./ds4quant_run 2>&1 | grep -a ZCHAIN || true
    fi
}
RESUME=""   # merge-续并态(部分 dql 已消费): M4.6 禁重建骨架 + 注入传 DS4_MERGE_RESUME
if [ "$MODE" = merge ]; then
    # ---------- merge-only 入口: 量化 M2 完成后被看门狗/中断截停时, 层文件+OUT 已齐 → 跳过清理/锚/量化, 直补 M3 表 + M4.6 GGUF ----------
    OUT=/tmp/quant_all.out; LOG=/tmp/quant_all.log; ZFILE="$ROOT/gguf/go-onebit/zfile_all.bin"
    mkdir -p "$TBL/raw" "$LDIR"; cd "$QDIR"
    [ -x ./ds4quant_run ] || ../scripts/quant_verify.sh build
    N_HAVE=$(ls "$LDIR"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ' || true)   # ||true: dql 已 consume 释放(零文件)时 pipefail 会杀脚本, 走不到下方"已完成"分支
    GOUT="$ROOT/gguf/go-onebit/ds4-code1b.gguf"
    if [ "$N_HAVE" = 0 ] && [ -s "$GOUT" ] && \
       python3 "$ROOT/gguf-tools/go-onebit/scripts/gguf_offsets.py" "$GOUT" 2>/dev/null | grep -q "opt_chain"; then
        NOPT=$(python3 "$ROOT/gguf-tools/go-onebit/scripts/gguf_offsets.py" "$GOUT" 2>/dev/null | grep -c "opt_")
        echo "【全模型】脚本 算法=merge-only 进度=已完成 体积=$(stat -f%z "$GOUT")B 还原度=- 研判=✓ 合一GGUF已在且含 ${NOPT} 个优化张量; dql 层文件已按 consume 设计释放 — 无需也无法再合并(重出请跑 fast)" >&2
        exit 0
    fi
    if [ "$N_HAVE" = "$NLAY" ] && [ -s "$OUT" ]; then
        zchain_fix merge-only
        echo "【全模型】脚本 算法=merge-only 进度=开始 体积=- 还原度=- 研判=复用既有${NLAY}层文件+OUT 补表+GGUF(跳过量化)" >&2
    elif [ "$N_HAVE" -gt 0 ] && [ -s "$OUT" ] && [ -s "$GOUT" ] && \
         python3 "$ROOT/gguf-tools/go-onebit/scripts/gguf_offsets.py" "$GOUT" 2>/dev/null | grep -q "opt_chain"; then
        # ★续并★: 上次合并中途截停 — 已注入层的 dql 被 consume 删除(其唯一字节已在 GGUF) → 只补剩余层。
        # 不跑 zchain_fix: 用部分 dql 重建会把已消费层的 opt/zchain 清空(侧车已是终值, 续并只注专家字节)。
        RESUME=1
        echo "【全模型】脚本 算法=merge-续并 进度=开始 体积=- 还原度=- 研判=层文件 $N_HAVE/$NLAY + 合一GGUF在位 → 续注剩余层(缺失=已消费跳过; 侧车不重建)" >&2
    else
        echo "[merge] 层文件 $N_HAVE/$NLAY 或 $OUT 缺失 — 拒并(防半成品GGUF); 需完整量化产物" >&2; exit 3
    fi
else
# ---------- M-1 清除上次遗留(只清产物; 锚/HF/q2 永不动; backfit=复用产物不清理) ----------
if [ "$MODE" = backfit ]; then
    # ★依据闸(2026-07-22)★: 不盲目反修 — 必须已有指标(rr VERDICT)+真实场景(行为门)双判决
    # 且反修对症, 由操作者以 DS4_BF_JUSTIFIED=1 确认(确认即声明依据已在 fable5 入档)。
    [ "${DS4_BF_JUSTIFIED:-0}" = 1 ] || { echo "[backfit] 依据闸: 需 DS4_BF_JUSTIFIED=1(指标+真实场景双判决在案且反修对症) — 拒跑" >&2; exit 3; }
    N_HAVE=$(ls "$LDIR"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ' || true)
    [ "$N_HAVE" = "$NLAY" ] || { echo "[backfit] 层文件 $N_HAVE/$NLAY 不齐 — 只跑反修需完整推进段产物, 拒跑(先满档或 fast)" >&2; exit 3; }
    echo "【全模型】脚本 算法=只跑反修 进度=开始 体积=- 还原度=- 研判=复用${NLAY}层dql(含已落地修正)按op链回放, SEARCH跳过 → 终局收敛sweep+反调+回扫+合并; OUT/LOG 追加" >&2
else
echo "【全模型】脚本 算法=清理遗留 进度=开始 体积=- 还原度=- 研判=清 layer-tables/*.md raw/ layers/ zfile dql_full" >&2
rm -f "$TBL"/L*.md "$TBL"/ALL.md "$TBL"/raw/*.out 2>/dev/null || true
rm -f "$LDIR"/dql_L*.bin "$LDIR"/opt_L*.bin "$ROOT/gguf/go-onebit/dql_full.bin" "$ROOT/gguf/go-onebit/zfile_all.bin" "$ROOT/gguf/go-onebit/zchain_all.bin" 2>/dev/null || true
rm -f /tmp/quant_all.out /tmp/quant_all.log 2>/dev/null || true
fi
mkdir -p "$TBL/raw" "$LDIR"
cd "$QDIR"
IDS="$CORPUS"
[ -f "$IDS" ] || { echo "[M0] 编程语料 $IDS 缺失 — 拒跑" >&2; exit 2; }
CTOK=$(grep -c . "$IDS")
echo "【全模型】脚本 算法=对齐域 进度=- 体积=- 还原度=- 研判=编程域($IDS, ${CTOK}tok)" >&2

# ---------- M1 构建 ----------
../scripts/quant_verify.sh build

# ---------- M0b 锚闸(按 S+层数 分档; 缺失全新自动建, 尺寸不符硬拒) ----------
[ "$NTOK" -gt "$CTOK" ] && NTOK=$CTOK   # 语料不足则用满语料
# 锚大小随层数线性: 40 + NTOK×(NLAY×81968 + VOCAB×4); NLAY=43 → 每token 4041744(与旧一致)
if [ "$NLAY" = 43 ]; then ANCH="/tmp/ds4quant_anchor_${DOMAIN}_s${NTOK}.bin"     # 满档: 名字不变(兼容旧锚)
else                      ANCH="/tmp/ds4quant_anchor_${DOMAIN}_s${NTOK}_L${NLAY}.bin"; fi
EXP_SZ=$(( NTOK*(NLAY*81968 + 517120) + 40 ))
if [ -f "$ANCH" ]; then
    ASZ=$(stat -f%z "$ANCH")
    [ "$ASZ" = "$EXP_SZ" ] || { echo "[M0] 锚 $ANCH 尺寸 $ASZ ≠ $EXP_SZ — 硬拒(防覆盖)" >&2; exit 2; }
else
    echo "【全模型】脚本 算法=FP建锚(S=$NTOK,${NLAY}层) 进度=开始(一次性) 体积=$EXP_SZ 还原度=- 研判=锚缺失全新生成" >&2
    DS4_ANCHOR="$ANCH" DS4_NL="$NLAY" DS4_FP_ONLY=1 ./ds4quant_run "$IDS" "$NTOK" >/tmp/anchor_build.out 2>/tmp/anchor_build.log \
        || { echo "[M0] 建锚失败, 见 /tmp/anchor_build.log" >&2; exit 2; }
    ASZ=$(stat -f%z "$ANCH" 2>/dev/null || echo 0)
    [ "$ASZ" = "$EXP_SZ" ] || { echo "[M0] 建锚后尺寸 $ASZ ≠ $EXP_SZ — 停" >&2; exit 2; }
fi

# ---------- M2 执行: 渐进 43 层(每层在累积状态上调优, C 内按每层死线收敛) ----------
OUT=/tmp/quant_all.out LOG=/tmp/quant_all.log
ZFILE="$ROOT/gguf/go-onebit/zfile_all.bin"
echo "【全模型】脚本 算法=渐进${NLAY}层 进度=启动 体积=- 还原度=- 研判=每层收敛制(不截断), 产物 $TBL/ + $ZFILE" >&2
BIN_PAT="ds4quant_run $IDS"
if [ "$MODE" != backfit ]; then : >"$OUT"; : >"$LOG"; fi   # backfit=追加(保留推进段 TABREC/日志供 M4 出表)
( export DS4_ANCHOR="$ANCH" DS4_NL="$NLAY" DS4_LCFG=g DS4_SIGNREF_MU=10 DS4_COADAPT=1 \
         DS4_ZFILE="$ZFILE" DS4_ZCHAIN="$ROOT/gguf/go-onebit/zchain_all.bin" \
         DS4_TUNE_MIN="${TMIN:-0}" DS4_THREADS="${DS4_THREADS:-6}" \
         DS4_LAYER_DIR="$LDIR"
  if [ "$GSW" != 0 ]; then export DS4_GSWEEP="$GSW"; fi   # C 门只查存在性: "0" 也会触发 → 0 时不导出
  if [ "$BWD" = 1 ]; then export DS4_BWD_FINAL=1; fi
  if [ "$FAST" = 1 ]; then export DS4_FAST=1; fi
  if [ "$MODE" = backfit ]; then export DS4_BF_ONLY=1; fi
  exec ./ds4quant_run "$IDS" "$NTOK"
) >>"$OUT" 2> >(tee -a "$LOG" >&2) &
SPID=$!   # exec 后 SPID 即量化进程本体
trap 'kill -9 $(pgrep -f "$BIN_PAT") "$SPID" 2>/dev/null || true; echo "[中断] 已杀量化进程" >&2; exit 130' INT TERM
T0=$(date +%s); TIMEDOUT=0; WDOG=0
sleep 2
BPID=$(pgrep -nf "$BIN_PAT" || true)
while kill -0 "$SPID" 2>/dev/null; do
    if [ -n "$TMIN" ]; then
        EL=$(( $(date +%s) - T0 ))
        DL=$(awk -v t="$TMIN" 'BEGIN{printf "%d", t*60}')
        if [ "$EL" -ge "$DL" ]; then
            echo "【全模型】脚本 算法=到时硬停 进度=${EL}s 体积=- 还原度=- 研判=已完成层文件保留, 未完成层丢弃" >&2
            kill -9 "$BPID" "$SPID" 2>/dev/null || true; TIMEDOUT=1; break
        fi
    fi
    [ -n "${BPID:-}" ] || BPID=$(pgrep -nf "$BIN_PAT" || true)
    if [ -n "${BPID:-}" ]; then
        MB=$(footprint -p "$BPID" 2>/dev/null | grep -Eo 'Footprint: *[0-9.]+ *[KMG]B' | head -1 \
             | awk '{v=$2;u=$3; if(u=="GB")v*=1024; else if(u=="KB")v/=1024; printf "%d",v}' || true)
        if [ -n "${MB:-}" ] && [ "$MB" -gt 11776 ]; then
            echo "[watchdog] 量化进程 ${MB}MB >11.5G → kill; 层文件若已齐(常见于死在GSWEEP抛光段)将原地自动merge收尾" >&2
            kill -9 "$BPID" "$SPID" 2>/dev/null || true; WDOG=1; break; fi
    fi
    sleep 5
done
RC=0; wait "$SPID" 2>/dev/null || RC=$?   # ||捕获: set -e 下裸 wait 收非零码(如 kill -9 的137)会当场杀脚本, 截停收尾全走不到
trap 'echo "[中断] 杀本机量化进程" >&2; pkill -9 -f ds4quant_run 2>/dev/null || true; exit 130' INT TERM   # 恢复全局(M4.6 merge 阶段仍受保护)
if [ "$TIMEDOUT" = 1 ] || [ "$WDOG" = 1 ] || [ "$RC" != 0 ]; then
    cp "$OUT" "$TBL/raw/all.out" 2>/dev/null || true
    DONE=$(ls "$LDIR"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ' || true)   # ||true: 零文件时 pipefail 会杀脚本
    if [ "$DONE" = "$NLAY" ]; then
        # ★自动merge(裁决2026-07-12: 截停后不再要求第二遍 ./quant_layer.sh merge)★:
        # 层文件全齐(常见于死在GSWEEP抛光段) → 原地续尾: 补侧车+表+GGUF; MODE=merge 让 M3 对 VERDICT/ZFILE 降级为警告
        RSN="异常退出rc=$RC"; [ "$WDOG" = 1 ] && RSN="看门狗截停"; [ "$TIMEDOUT" = 1 ] && RSN="到时硬停"
        echo "【全模型】脚本 算法=自动merge 进度=接管 体积=- 还原度=- 研判=${RSN}但${NLAY}层文件已齐 → 原地自动合并收尾(无需再跑 merge)" >&2
        # ★固定裁判先于消费(2026-07-15 教训: v3p 截停自动合并吃掉 dql 后 rr 双判决永久不可补)★:
        # rr_verdict 只读 dql 回放; merge 消费 dql 是不可逆点。DS4_SKIP_RRVERDICT=1 跳过(赶时间)。
        if [ "${DS4_SKIP_RRVERDICT:-0}" != "1" ]; then
            for RRIDS in /tmp/rr_hard.ids:64 /tmp/rr_code.ids:305; do
                RRF="${RRIDS%%:*}"; RRN="${RRIDS##*:}"
                [ -f "$RRF" ] && env -u DS4_ANCHOR_ROUTE bash "$ROOT/gguf-tools/go-onebit/scripts/rr_verdict.sh" "$RRF" "$RRN" 2>&1 \
                    | grep -E 'VERDICT|watchdog' || echo "[自动merge] rr_verdict $RRF 未出分(不阻塞合并)" >&2
            done
        fi
        MODE=merge
        zchain_fix 自动merge
    elif [ "$TIMEDOUT" = 1 ]; then
        echo "【全模型】脚本 算法=到时终止 进度=完成层=$DONE/$NLAY 体积=- 还原度=- 研判=层文件在 $LDIR, 明细表按已完成层生成; 未齐不合并(半成品防护), 重跑将从头开始"
        python3 "$ROOT/gguf-tools/go-onebit/scripts/gen_tables.py" "$OUT" "$TBL" || true
        exit 0
    elif [ "$WDOG" = 1 ]; then
        echo "[watchdog] 层文件 $DONE/$NLAY 未齐 → 不合并(半成品防护)" >&2
        exit 9
    else
        echo "[M2] 运行失败 rc=$RC (层文件 $DONE/$NLAY 未齐), 尾部日志:" >&2; tail -5 "$LOG" >&2; exit "$RC"
    fi
fi
# ★裁决2026-07-14★ fast 反修判据升档: S=64 反修实测过拟合(val 仅16行×27连环落地=选择偏差
# 复利, code S=305 smin 0.51→0.29 崩) → fast 推进段(S=64)落盘后自动接 backfit 段: BF_ONLY
# 回放重建态 + 反修全闸 @S=305 判据, 之后走 backfit 的表/合并。本段默认关终端反调与回扫
# (时间账 ≈ 推进1h+诚实反修3h; DS4_BWD=1/DS4_GSWEEP=N 可开)。
if [ "$FAST" = 1 ]; then
    DONE=$(ls "$LDIR"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ' || true)
    [ "$DONE" = "$NLAY" ] || { echo "[fast] 层文件 $DONE/$NLAY 未齐 — 不接诚实反修段" >&2; exit 3; }
    echo "【全模型】脚本 算法=fast→诚实反修 进度=接段 体积=- 还原度=- 研判=推进段(S=$NTOK)完成 → exec backfit(S=305判据全闸反修+表+合并)" >&2
    export DS4_BWD="${DS4_BWD:-0}" DS4_GSWEEP="${DS4_GSWEEP:-0}"
    # $0 是相对路径且 M2 段已 cd 进 quant/ — 用启动时钉死的 ROOT 绝对路径(门3 实证坑)
    exec "$ROOT/quant_layer.sh" backfit
fi
fi

# ---------- M3 完整性: 43 层 × 元素全集(TABREC), 缺=失败 ----------
MISS=""
for LL in $(seq 0 "$LMAX"); do
    for e in $REQ_EXACT; do
        grep -q "^TABREC L=$LL $e " "$OUT" || MISS="$MISS L$LL:$e"
    done
    for pfx in $REQ_PREFIX; do
        grep -q "^TABREC L=$LL $pfx" "$OUT" || MISS="$MISS L$LL:${pfx}*"
    done
done
if [ "$MODE" = merge ]; then
    # 截停恢复态: VERDICT/ZFILE 是运行报告不是产物, 缺失只警告(质量判决以下次完整跑为准)
    grep -q "^VERDICT " "$OUT" || echo "[M3] 提示: OUT 无 VERDICT(量化被截停) — GGUF 照出, 质量数字缺席" >&2
    grep -q "^ZFILE "   "$OUT" || true
else
    grep -q "^VERDICT " "$OUT" || MISS="$MISS VERDICT"
    grep -q "^ZFILE "   "$OUT" || MISS="$MISS ZFILE"
fi
for LL in $(seq 0 "$LMAX"); do
    F=$(printf 'dql_L%02d.bin' "$LL")   # C 侧 %02d 命名; seq -w 在 <10 层时不补零会误报缺失
    [ -s "$LDIR/$F" ] || [ "$MODE" = merge ] || MISS="$MISS 层文件L${LL}"   # merge态缺dql=已消费(consume后合法)
    FO=$(printf 'opt_L%02d.bin' "$LL")  # 一层两份: 量化(dql)+优化(opt); 86 份=最终合并 GGUF 的全部输入
    [ -s "$LDIR/$FO" ] || MISS="$MISS 优化文件L${LL}"
done
if [ -n "$MISS" ]; then
    echo "[M3] 完整性失败 — 缺:$(echo "$MISS" | tr ' ' '\n' | head -20 | tr '\n' ' ')…(共$(echo "$MISS"|wc -w|tr -d ' ')项)" >&2
    exit 3
fi
echo "【全模型】脚本 算法=完整性校验 进度=完成 体积=$(stat -f%z "$ZFILE" 2>/dev/null || echo 0)B 还原度=- 研判=✓ ${NLAY}层元素全集齐" >&2

# ---------- M4 出表: 每层 L<NN>.md + 总表 ALL.md ----------
cp "$OUT" "$TBL/raw/all.out"
python3 "$ROOT/gguf-tools/go-onebit/scripts/gen_tables.py" "$OUT" "$TBL"
# ---------- M4.4 判决停点(2026-07-20 域放大): DS4_SKIP_MERGE=1 → 裁判后停在 dql 态 ----------
# 依据: 跑机盘装不下合一 GGUF 时, merge(consume 消费 dql)是不可逆点(07-15 教训同族);
# 停点=出表后先跑 rr 固定裁判(只读回放), dql/opt 全留存, 裁决赢了再手动 ./quant_layer.sh merge。
if [ "${DS4_SKIP_MERGE:-0}" = 1 ]; then
    if [ "${DS4_SKIP_RRVERDICT:-0}" != "1" ]; then
        for RRIDS in /tmp/rr_hard.ids:64 /tmp/rr_code.ids:305; do
            RRF="${RRIDS%%:*}"; RRN="${RRIDS##*:}"
            [ -f "$RRF" ] && env -u DS4_ANCHOR_ROUTE bash "$ROOT/gguf-tools/go-onebit/scripts/rr_verdict.sh" "$RRF" "$RRN" 2>&1 \
                | grep -E 'VERDICT|watchdog' || echo "[skip-merge] rr_verdict $RRF 未出分(不阻塞停点)" >&2
        done
    fi
    # ★dql 快照分叉(2026-07-22 用户三修正之一)★: DS4_DQL_SNAP=1 且盘余>载荷×1.5 时
    # 硬链克隆 layers → layers_snap(同卷零拷贝); merge consume 后快照仍持字节 →
    # 终端反修类实验可从快照分叉, 永不再全重跑。盘紧自动跳过(consume 空间账依赖 unlink)。
    if [ "${DS4_DQL_SNAP:-0}" = 1 ]; then
        SNKB=$(du -sk "$LDIR" | awk '{print $1}'); SFREE=$(df -k / | tail -1 | awk '{print $4}')
        if [ "$SFREE" -gt $(( SNKB + SNKB/2 )) ]; then
            rm -rf "$LDIR/../layers_snap"; cp -al "$LDIR" "$LDIR/../layers_snap" \
                && echo "[M4.4] dql 快照 → layers_snap (硬链, 反修分叉用)" >&2
        else
            echo "[M4.4] 盘紧(余$((SFREE/1048576))G) → 跳过 dql 快照(consume 需 unlink 释放)" >&2
        fi
    fi
    echo "【全模型】脚本 算法=判决停点 进度=完成(不合并) 体积=- 还原度=见上VERDICT 研判=DS4_SKIP_MERGE=1: dql/opt 全留存, 裁决后手动 merge(交接铁律: 新模型行为门过前不删前代)" >&2
    exit 0
fi
# ---------- M4.5 合并整文件(可选; 盘不够自动跳过 — 真正的合并产物是 M4.6 的 GGUF) ----------
FULL="$ROOT/gguf/go-onebit/dql_full.bin"
NEED_KB=$(du -sk "$LDIR" | awk '{print $1}'); FREE_KB=$(df -k / | tail -1 | awk '{print $4}')
NDQL=$(ls "$LDIR"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ' || true)   # 续并态部分层已消费 → 拼不了整文件
if [ "$NDQL" = "$NLAY" ] && [ "$FREE_KB" -gt $(( NEED_KB + NEED_KB/10 )) ] && [ ! -L "$FULL" ]; then
: > "$FULL"
{
  echo "## 合并清单(dql_full.bin 内各层偏移)"
  OFF=0
  for LL in $(seq 0 "$LMAX"); do
      F="$LDIR/$(printf 'dql_L%02d.bin' "$LL")"; SZ=$(stat -f%z "$F")
      echo "| L${LL} | offset=$OFF | bytes=$SZ |"
      cat "$F" >> "$FULL"
      OFF=$((OFF+SZ))
  done
  echo "| 合计 | $OFF bytes ($(awk -v b=$OFF 'BEGIN{printf "%.2f GiB", b/1073741824}')) |"
} >> "$TBL/ALL.md"
echo "【全模型】脚本 算法=合并 进度=完成 体积=$(stat -f%z "$FULL")B 还原度=- 研判=✓ dql_full.bin" >&2
else
    echo "【全模型】脚本 算法=合并 进度=跳过 体积=- 还原度=- 研判=dql_full 不拼(层文件 $NDQL/$NLAY$([ "$NDQL" = "$NLAY" ] && echo "; 盘不够: 需$((NEED_KB/1048576))G 余$((FREE_KB/1048576))G" || echo "=已消费")); 真合并产物=GGUF" >&2
fi

# ---------- M4.6 GGUF(★铁律: 不管什么模式最后都要生成 GGUF★) ----------
# 稀疏骨架: --experts-hole 专家槽只留洞(不算不写, 实占≈骨干几GB), merge 按偏移 pwrite 层文件字节回填;
# 盘紧(<载荷×1.2)自动 DS4_MERGE_CONSUME=1 边并边释放层文件(峰值盘占恒定)。
GGUF_OUT="$ROOT/gguf/go-onebit/ds4-code1b.gguf"
SKEL="${DS4_SKELETON:-$ROOT/gguf/go-onebit/skeleton_go1b.gguf}"
OFF="$ROOT/gguf/go-onebit/skeleton_off.txt"
TMPL="${DS4_TMPL:-$ROOT/gguf/tmpl_hdr.gguf}"
# ★量化+优化合一(用户产品形态)★: zchain_all.bin(43 份 opt 文件的终值汇总)经
# deepseek4-quantize --zchain 写成 blk.L.opt_* 原生张量 + KV ds4.zchain.present,
# 与 1bit 专家字节同在一个 GGUF; 引擎装载时自动识别, 无需外挂文件。
ZCB="$ROOT/gguf/go-onebit/zchain_all.bin"
ZARG=""; [ -s "$ZCB" ] && ZARG="--zchain $ZCB"
if [ -s "$GGUF_OUT" ] && [ -s "$ZCB" ]; then
    # 优化链变了 → opt 张量尺寸/内容随之变 → 不能就地复用, 必须重建合一骨架
    REB=0
    python3 "$ROOT/gguf-tools/go-onebit/scripts/gguf_offsets.py" "$GGUF_OUT" > "$OFF" 2>/dev/null || REB=1
    grep -q "opt_chain" "$OFF" 2>/dev/null || REB=1
    [ "$GGUF_OUT" -ot "$ZCB" ] && REB=1
    if [ "$REB" = 1 ]; then
        # ★续并护栏★: 已消费层的 1bit 字节只存在于这个 GGUF 里, 删=永久丢(重出须整轮重量化) → 硬拒
        [ "$RESUME" = 1 ] && { echo "[M4.6] 拒重建: 续并态 GGUF 含已消费层的唯一字节; 若确要重建先跑完整量化" >&2; exit 4; }
        echo "【全模型】脚本 算法=GGUF骨架 进度=重建 体积=- 还原度=- 研判=优化链未并入/已过期 → 删旧GGUF重建合一骨架(量化+优化单文件)" >&2
        rm -f "$GGUF_OUT"
    fi
fi
if [ -s "$GGUF_OUT" ]; then
    # 已有成品/半成品(如上次 merge 中途失败): 偏移不变 → 就地幂等 re-merge(重写同字节+补缺层), 免 6min 骨架重建
    echo "【全模型】脚本 算法=GGUF骨架 进度=复用 体积=$(stat -f%z "$GGUF_OUT")B 还原度=- 研判=$GGUF_OUT 已在(优化张量在位), 就地 re-merge" >&2
elif [ ! -s "$SKEL" ]; then
    # 骨干类型必须=引擎硬契约(token_embd F16/attn+shexp+output Q8_0, ds4.c weights_validate_layout);
    # 模板头取自已发布可加载模型 → 不传骨干 flags = template-copy 全对。q4_k 骨干装载即拒(实测)。
    echo "【全模型】脚本 算法=GGUF骨架 进度=构建(稀疏, 一次性) 体积=实占~10G 还原度=- 研判=--experts-hole 专家留洞, 骨干=模板类型(Q8/F16 引擎契约)" >&2
    QZ="$ROOT/gguf-tools/deepseek4-quantize"
    if [ -x "$QZ" ] && [ -s "$TMPL" ]; then
        "$QZ" --hf "$HF" --template "$TMPL" --out "$SKEL" --overwrite \
              --experts go1b --experts-hole $ZARG \
              >/tmp/skel_build.log 2>&1 \
              || { echo "[M4.6] 骨架构建失败(见 /tmp/skel_build.log 尾部):" >&2; tail -3 /tmp/skel_build.log >&2; rm -f "$SKEL"; }
    else
        echo "[M4.6] 缺 deepseek4-quantize 或模板 $TMPL — 无法建骨架" >&2
    fi
fi
if [ ! -s "$GGUF_OUT" ] && [ -s "$SKEL" ]; then mv "$SKEL" "$GGUF_OUT"; fi   # 稀疏骨架直接就地回填(cp 会实体化洞+双倍盘); 骨架可随时重建(分钟级)
if [ -s "$GGUF_OUT" ]; then
    python3 "$ROOT/gguf-tools/go-onebit/scripts/gguf_offsets.py" "$GGUF_OUT" > "$OFF" 2>/dev/null
    PAY_KB=$(du -sk "$LDIR" | awk '{print $1}'); FREE_KB=$(df -k / | tail -1 | awk '{print $4}')
    CONSUME=""; [ "$FREE_KB" -lt $(( PAY_KB + PAY_KB/5 )) ] && CONSUME=1 \
        && echo "[M4.6] 盘紧(余$((FREE_KB/1048576))G<载荷$((PAY_KB/1048576))G×1.2) → 边并边释放层文件(DS4_MERGE_CONSUME)" >&2
    ( cd "$QDIR" && export DS4_MERGE_GGUF="$GGUF_OUT" DS4_MERGE_OFF="$OFF" DS4_LAYER_DIR="$LDIR" DS4_NL="$NLAY"
      [ -n "$CONSUME" ] && export DS4_MERGE_CONSUME=1   # C 门只查存在性: 空值也触发 → 必须条件导出(此前 ""=恒consume 是bug)
      [ -n "$RESUME" ]  && export DS4_MERGE_RESUME=1
      ./ds4quant_run 2>&1 | tee -a "$LOG" ) || true
    if grep -q "MERGE_GGUF.*完成" "$LOG" 2>/dev/null; then
        MSMK=""; [ "$FAST" = 1 ] && MSMK="(fast冒烟档S=$NTOK)"
        echo "【全模型】脚本 算法=合并GGUF 进度=完成 体积=$(stat -f%z "$GGUF_OUT" 2>/dev/null || echo 0)B 还原度=- 研判=✓ $GGUF_OUT(可 ds4 -m 加载)$MSMK" >&2
    else
        echo "[M4.6] 合并未完成 — 见 $LOG 尾部" >&2; tail -3 "$LOG" >&2
    fi
fi

# ---------- M5 汇报 ----------
echo ""
echo "=== 全模型判决 ==="
grep -aE "^(MILESTONE|VERDICT|ZFILE|MERGE_GGUF)" "$OUT" "$LOG" 2>/dev/null | tail -6
[ -s "$GGUF_OUT" ] && echo "=== 合并模型: $GGUF_OUT ($(stat -f%z "$GGUF_OUT")B) ==="
echo "=== 明细文件 ==="
ls "$TBL"/L*.md "$TBL"/ALL.md
exit 0
