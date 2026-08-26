#!/bin/bash
# zloss_campaign.sh — 反修战役 v2(2026-08-26 重设计: FP 自洽口径 + 动态 z)。
#
# 设计对齐(用户两点理论): ①存在低维动态 z 映射高维推理行为 —— 模式条件化落地
# (方案B, 修正图随 token 行为模式切换); ②不存在空层 —— 判空废除, 输裸=停车审计。
# v1 混合口径(教师 FP 锚 × 学生引擎链捕获)36 层判空已定罪删除, 记录在 fable5.md。
#
# 链: needle(L20/30/41 针: zcache 重建 + (M,λ,k) 网格 + ER 容量曲线)
#     → [针过闸后] solve 43 层(档位按针定, 未定档拒跑)
#     → judge/engine(动态 z 注入格式 + 判决尺升级闸, 针后设计)
# 目标: Σmin ≥ 0.90(裸 0.7799/冠军 r64c 0.7903), 体积 ≤ 2GB(用户 08-26 拍板)。
# 用法: bash zloss_campaign.sh [needle|solve|judge|engine]
# ★无环境变量铁律: 本脚本参数全部写死; DS4_ZL_* 是 zlayer(冻结转录件)的既有
# 面板, 只用不新增。zlayer 只当 zcache 数据生产器(CACHE_ONLY), 解算线唯一
# = zloss_solve(反修只留一份解算器, 用户 08-26 铁律)。★
set -uo pipefail
ROOT="$HOME/ds4-main"; GT="$ROOT/gguf-tools"
D2="$ROOT/gguf/go-onebit/vqhalf"
HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
ANC="$D2/anchor_a_clean_s8192.bin"
WS="$D2/zloss"
NTOK=8192; NFIT=6144
# 夹持轮参数(2026-08-26 用户纠正跑法浪费后收窄): EM 模式网格已出结论(x/路由门
# 都读不出标签, 三层同款)不再重跑; 只跑有判决价值的臂 = 裸/GE/x静态/ftA/GE+ftA。
# 全网格版本的读数在 needle.log.v4unclamped 与 fable5.md。
MODES="1"; RANKS="16,64,128,256"; LAMBDAS="3e-3,3e-2,3e-1"
LOG(){ echo "[zloss $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }

# 行掩码(08-24 拼接毒定罪沿用): 256-token 块互织语料, 每块前 64 行=异域上下文
# 污染行剔除; 前 24 块=拟合, 后 8 块=held。
FR=""; ER=""
for b in $(seq 0 31); do
    seg="$((b*256+64)):$(( (b+1)*256 ))"
    if [ "$b" -lt 24 ]; then FR="${FR:+$FR,}$seg"; else ER="${ER:+$ER,}$seg"; fi
done

WD=""
wd_start(){ ( while true; do
    A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "${A:-99}" -lt 4 ] && { echo "[watchdog] MemAvailable=${A}GB <4GB ★杀★" >&2
        pkill -9 -f 'amp/zlayer'; pkill -9 -f zloss_solve; break; }
    sleep 5; done ) & WD=$!; }
wd_stop(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null; WD=""; }
trap 'wd_stop' EXIT

stage_needle(){
    [ -s "$ANC" ] || DIE "锚缺 $ANC"
    [ -d "$WS/layers" ] || DIE "工作区缺 $WS/layers(dql_vq 硬链自 vq86h_noz)"
    [ -x "$GT/amp/zlayer" ] || DIE "zlayer 二进制缺, 先 make -C gguf-tools zlayer"
    [ -x "$GT/amp/zloss_solve" ] || DIE "zloss_solve 二进制缺"
    rm -f "$WS"/z_L*.ds4z "$WS"/fourloss_L*.txt      # v1 废弃产物清场(用户令)
    mkdir -p "$WS/needle"
    wd_start
    NEEDLE_RC=0
    for L in 20 30 41; do
        Lz=$(printf '%02d' "$L")
        if [ ! -s "$WS/layers/zcache_L$Lz.npz" ]; then
            LOG "①L$L zcache 重建(zlayer 模式①: 锚 x + 本进程重算量化学生, 只建缓存)"
            env DS4_ZL_NTOK=$NTOK DS4_ZL_NFIT=$NFIT DS4_ZL_FIT_RANGES="$FR" DS4_ZL_EV_RANGE="$ER" \
                DS4_ZL_CACHE_ONLY=1 "$GT/amp/zlayer" "$HF" "$WS/layers" "$ANC" "$L" 1024 0 \
                2>&1 | grep -aE "zcache|缓存|Error|assert|★" || DIE "L$L zcache 失败"
        else
            LOG "①L$L zcache 已在, 复用"
        fi
        LOG "②L$L 臂解算(M=$MODES × λ=$LAMBDAS × k=$RANKS + GE/ftA/GE+ftA + ER, 信任域0.5)"
        "$GT/amp/zloss_solve" --anchor "$ANC" --zcache "$WS/layers" --out "$WS/needle" \
            --layers "$L-$L" --ntok $NTOK --threads 18 \
            --modes "$MODES" --ranks "$RANKS" --lambdas "$LAMBDAS" \
            --fit-ranges "$FR" --ev-ranges "$ER"
        rc=$?
        if [ "$rc" -eq 3 ]; then LOG "★L$L 全网格输裸 — 停车审计信号★"; NEEDLE_RC=3
        elif [ "$rc" -ne 0 ]; then DIE "L$L 解算异常 rc=$rc"; fi
    done
    wd_stop
    LOG "③针收官(rc=$NEEDLE_RC), 表在 $WS/needle/fourloss_L{20,30,41}.txt"
    exit "$NEEDLE_RC"
}

# GE-only 43 层端到端斜率标定(2026-08-26 用户令"跑"): 针终判饱和 ~1%/层, 但
# held层内≠端到端(冠军 43 层小赢复利 KL−9.7%)——本段把唯一稳定赢家 GE 全层
# 注入(bf.GE 冻结判决尺原生认, 零判决尺升级债)跑 caliper, 标定 held↔端到端斜率。
# 对表: 裸 0.47055 / r64c 冠军 0.42510 / 官方 q2 0.4207。
stage_solve(){
    [ -s "$ANC" ] || DIE "锚缺 $ANC"
    [ -x "$GT/amp/zlayer" ] || DIE "zlayer 缺"
    [ -x "$GT/amp/zloss_solve" ] || DIE "zloss_solve 缺(带 --emit-ge 的版本)"
    GE="$D2/zge"
    # ①注入工作区从 vq86h_noz 干净态重建(amp_clean_full.sh ②同式)。zloss/layers
    # 的 dql_L 有 7 层 v1 作废 zl.RRR 残留(L02-06/08/09 尺寸对不上 noz 实锤),
    # 只当 zcache 数据面用, 不做注入床。
    rm -rf "$GE"; mkdir -p "$GE/layers"
    ( cd "$D2/vq86h_noz/layers" || exit 1
      for f in dql_vq_L*.bin; do
          ln -f "$f" "$GE/layers/$f" 2>/dev/null || cp "$f" "$GE/layers/"; done
      cp dql_ops_L*.bin opt_L*.bin manifest.txt "$GE/layers/" 2>/dev/null
      cp dql_L*.bin "$GE/layers/" ) || DIE "工作区重建失败"
    N=$(ls "$GE/layers"/dql_vq_L*.bin 2>/dev/null | wc -l)
    [ "$N" -eq 43 ] || DIE "工作区层件不齐 $N/43"
    mkdir -p "$WS/ge"
    LOG "①注入工作区就绪 $GE/layers(43 层件, dql_L 全新拷贝)"
    wd_start
    # ②逐层: zcache 缺则建(zlayer 模式①只写缓存不动 dql) → GE 解算+四损失闸+注入
    for L in $(seq 0 42); do
        Lz=$(printf '%02d' "$L")
        if [ ! -s "$WS/layers/zcache_L$Lz.npz" ]; then
            env DS4_ZL_NTOK=$NTOK DS4_ZL_NFIT=$NFIT DS4_ZL_FIT_RANGES="$FR" DS4_ZL_EV_RANGE="$ER" \
                DS4_ZL_CACHE_ONLY=1 "$GT/amp/zlayer" "$HF" "$WS/layers" "$ANC" "$L" 1024 0 \
                2>&1 | grep -aE "zcache|就绪|Error|assert|★" || DIE "L$L zcache 失败"
        fi
        "$GT/amp/zloss_solve" --anchor "$ANC" --zcache "$WS/layers" --out "$WS/ge" \
            --layers "$L-$L" --ntok $NTOK --threads 18 \
            --fit-ranges "$FR" --ev-ranges "$ER" --emit-ge "$GE/layers" \
            || DIE "L$L GE 解算/注入异常"
        NI=$(wc -l < "$GE/layers/zinject_manifest.txt" 2>/dev/null || echo 0)
        LOG "L$L ✓ 已注入 $NI 层"
    done
    wd_stop
    LOG "②43 层收官(注入账 $GE/layers/zinject_manifest.txt) → caliper 五指标"
    bash "$GT/scripts/caliper_ref.sh" "$GE/layers" /tmp/qc_zge_wt2.bin \
        > /tmp/caliper_zge.log 2>&1 || DIE "caliper 失败, 看 /tmp/caliper_zge.log"
    grep -aE "PPL|KLD|RMS|Same|Δp|min" /tmp/caliper_zge.log | tail -10
    LOG "③端到端斜率标定完成: 对表 裸 0.47055 / r64c 0.42510, 全表 /tmp/caliper_zge.log"
}
stage_judge(){  DIE "动态 z 注入格式 + 判决尺升级闸(新尺须逐字节复刻裸 0.47055/r64c 0.42510)未定, 针后设计"; }
stage_engine(){ DIE "引擎多模式 z 加载路未实现(ds4_z 模块扩容器), 针后设计"; }

case "${1:-needle}" in
    needle)  stage_needle;;
    solve)   stage_solve;;
    judge)   stage_judge;;
    engine)  stage_engine;;
    *) DIE "用法: bash zloss_campaign.sh [needle|solve|judge|engine]";;
esac
