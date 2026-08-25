#!/bin/bash
# r30_campaign.sh — R30 全自动战役(2026-08-02 用户令: 源模型换 DeepSeek-V4-Flash-0731,
#   预算 36 GiB, 流程 = 配置 JSON → 量化 → 反修 → 合并 → 基准; 本轮必须加入标准指标
#   PPL / Mean KLD / RMS Δp / Same top token / Bit-exact weights)。
#
# 与 R29 的四处根本差异:
#   ① 【源=0731】新 checkpoint 是 MXFP4 routed 专家(I8 nibble + E8M0 1×32 scale)+ fp8
#      backbone(E8M0 块 scale)。st_read.c 已加两种解码(2026-08-02); tokenizer/超参与
#      template 逐项一致(preflight_r30.py PASS), 元数据零迁移风险。
#   ② 【预算 36 GiB】载荷 27.798 + backbone 8.202。R29(28G)败因不是配方是密度:
#      0.30 bpw 层自由生成误差复利塌缩。
#   ③ 【档梯地板】rplan_solve_r30.py 删掉 cos<0.6 的最稀两档, 层地板 0.623 bpw,
#      富余 7.7 GiB 按 R29 实跑 d_energy 证据(evidence_r29run.json)KKT 竞价。
#   ④ 【指标闸】合并后 stage_metrics 出五项标准指标, 参考分布 = 0731 FP 锚 logits。
#
# 锚/hotcurve 全部从 0731 重建(锚是权重的激活, 换源必换锚)。
# 阶段: r30_campaign.sh [anchor|plan|quant|backfit|merge|metrics|all]
set -uo pipefail
export LC_ALL=en_US.UTF-8   # nohup/C locale 下 bash 会把 "$L失败" 解析成多字节变量名, set -u 误杀 lane(08-14 事故)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SC="$ROOT/gguf-tools/scripts"
. "$SC/_portable.sh"   # 盘闸/看门狗/stat 的跨平台形态(Linux 上 BSD 形态会静默失效)
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
OUTF="${OUTF_OVERRIDE:-$R30/full}"          # ★nl86 复用(2026-08-12 用户令"反修全部用C"): 战役目录可覆盖
QBIN="${QBIN_OVERRIDE:-$ROOT/gguf-tools/amp/ds4quant_run.r30}"   # QBIN_OVERRIDE: 探针 A/B(旧二进制/lldb 包装)
export DS4_HF="${DS4_HF:-$HOME/ds4-main/hf/DeepSeek-V4-Flash-0731}"
ANCHOR="${ANCHOR_OVERRIDE:-$R30/anchor_r30_s1716.bin}"
IDS="${IDS_OVERRIDE:-$G7/rr_calib_prog_v5mini.ids}"
PLAN_JSON="${PLAN_JSON:-$R30/r30_rplan.json}"
RPLAN="$OUTF/rplan.txt"
MDL="$ROOT/gguf/go-onebit/ds4-r30.gguf"
# ★体积单点真相(2026-08-06 用户问责"漏闸"后修根)★: 载荷预算从 plan json 派生
# (budget_GiB=blob 载荷)+0.5 垫, 不再各处写死; VOL_BUDGET 仍可显式覆盖。
BUDGET="${VOL_BUDGET:-$(python3 -c "import json,sys;print(round(json.load(open('$PLAN_JSON'))['budget_GiB']+0.5,1))" 2>/dev/null || echo 37.0)}"       # ★v4b★ blob 实测 592M/层(热113.6+冷w1w3 1bit副本478.6, 引擎热侧车布局需要)×43=24.9G; GGUF 仍=骨架+GUD 42.468(blob 外挂)
# v2 与 v1 的差异(首跑 25 层实测审计后): ①证据换混合(L00-24 R30 实测 / L25-42 R29 抬地板 0.015,
# 上轮深层自愈是背上游债的表象已实证失效) ②"撑着层免底线"例外删除 ③hot 硬地板 24。
# 失配层修正: L23 hot7→104 / L21 16→80 / L20 29→84 / L10 28→68; bpw 0.776-1.556 均值 1.048。
LOG(){ echo "[r30 $(date +%H:%M:%S)] $*" >&2; }

[ -x "$QBIN" ] || { LOG "量化器 $QBIN 缺(需带 MXFP4/E8M0 读支持的新编译)"; exit 2; }
[ -d "$DS4_HF" ] || { LOG "源模型缺: $DS4_HF"; exit 2; }

WDOG(){ while true; do
    P=$(pgrep -nf "${WDOG_PAT:-[d]s4quant_run.* $IDS}" || true); [ -n "$P" ] || { sleep 5; continue; }
    MB=$(proc_mem_mb "$P" || true)
    [ -n "${MB:-}" ] && [ "$MB" -gt "${WDOG_MB:-11900}" ] && { echo "[r30][wdog] ${MB}MB >${WDOG_MB:-11900}MB 杀" >&2; kill -9 "$P" 2>/dev/null || true; }
    sleep 5; done }

stage_anchor(){
    mkdir -p "$OUTF"
    if [ -f "$ANCHOR" ]; then LOG "锚已在 $(ls -l "$ANCHOR" | awk '{printf "%.2f GiB",$5/1073741824}')"; return 0; fi
    FREE=$(disk_free_gb "$R30")
    [ "$FREE" -ge 10 ] || { LOG "★盘闸 free ${FREE}G <10G 停★"; exit 6; }
    LOG "0731 全 FP 锚定遍 S=1716(同时是五指标的参考分布)"
    cd "$ROOT/gguf-tools/amp"
    DS4_FP_ONLY=1 DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-8}" \
        "$QBIN" "$IDS" 1716
    [ -f "$ANCHOR" ] || { LOG "★锚没落盘★"; exit 3; }
    LOG "锚 ✓ $(ls -l "$ANCHOR" | awk '{printf "%.2f GiB",$5/1073741824}')"
    # 冒烟判决同款自检: FP 参考 PPL 必须在正常范围(权重读错这里当场炸)
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$ANCHOR" --ids "$IDS" --fit 933 >&2 \
        || { LOG "★锚 PPL 自检 FAIL — 权重/前向有错, 停★"; exit 4; }
}

stage_plan(){
    # ★v3(2026-08-03 用户令"看冠军 git 配置修复")★: 计划 = 冠军 vq4bf 固定配方
    # (git 573b7f5 九宫格判决: 热64 vq4×512+GPTQ / 冷 w1w3 vq8×256+GPTQ / 冷 w2 signref
    # "码本无效"), 无竞价。账: blob 34.47 + down 11.42 + 骨架 8.202 = 54.09 GiB(=vq4bf 定版)。
    python3 "$SC/r30_plan_champ.py" "$PLAN_JSON" >&2 || { LOG "★冠军计划生成失败★"; exit 4; }
}

stage_quant(){
    [ -f "$PLAN_JSON" ] || { LOG "配置 JSON $PLAN_JSON 缺, 先 plan"; exit 2; }
    pgrep -f "[d]s4quant_run.* $IDS" >/dev/null && { LOG "已有量化进程"; exit 3; }
    [ -n "${RESUME_QUANT:-}" ] || rm -rf "$OUTF/layers" "$OUTF/ckpt"
    mkdir -p "$OUTF/layers" "$OUTF/ckpt"
    # ★盘闸必须在清旧产物之后(2026-08-03 rc=6 事故: 旧层文件占 25.7G 把自己闸死)★
    # code2b 真账: dql 962.6M×43=40.4 + blob 621M×43=26.1 + ckpt(温和janitor后)~1 ≈ 67G
    FREE=$(disk_free_gb "$R30")
    NEED=64   # r60: dql 38.6+vq侧车(热72) ~23+ckpt ~1
    [ -n "${PROBE1:-}" ] && NEED=5   # 探针只产 L00(~1.6G)
    [ "$FREE" -ge "$NEED" ] || { LOG "★盘闸 free ${FREE}G <${NEED}G 停★"; exit 6; }
    LOG "从配置 JSON 派生计划表: $(basename "$PLAN_JSON")"
    python3 "$SC/plan_json2rplan.py" "$PLAN_JSON" "$RPLAN" --budget-gib "$BUDGET" >&2 \
        || { LOG "★配置 JSON 校验不通过, 拒跑★"; exit 7; }
    cp "$PLAN_JSON" "$OUTF/plan_config.json"
    cd "$ROOT/gguf-tools/amp"
    export DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-6}"
    export DS4_CALIB_FULLSET=1   # ★冠军工序(campaign_v4 同款): g_r/GPTQ-H 全集喂入, 修每专家~9行饿死
    # ★纯VQ量化(2026-08-03 用户令"只要vq量化"): 量化段不跑 coadapt 定稿(菜单/z/四损失/感知/向后D3
    #   全撤到反修段), 每层=计划表档位量化+前向+导出。DS4_MV_COAD_BASE/DS4_COADAPT 即定稿遍火源。
    export DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_TUNE=1
    export DS4_PURE_VQ=1   # ★导出先行+B回放: 每层=量化落盘一遍+盘上字节回放出链态(裸评撤, 省一遍VQ编码)
    unset DS4_COADAPT DS4_MV_COAD_BASE 2>/dev/null || true
    export DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$RPLAN"
    export DS4_VOL_BUDGET_GIB="$BUDGET"
    export DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="${HOT_TABLE:-$ROOT/gguf-tools/data/corpus/prog_active_top49.txt}"
    export DS4_ROUTE_BIAS_FIT=1 DS4_ROUTE_BIAS_OUT="$OUTF/route_bias_r30.bin" DS4_ROUTE_BIAS_ALPHA=1.0
    export DS4_ANCHOR_ROUTE=1                       # 100% 锚路由(R29 起标配)
    export DS4_BF_GAIN_GATE="${GAIN_GATE:-0.05}"    # 落地增益门
    export DS4_PLAN="$OUTF/plan.txt" DS4_CKPT_DIR="$OUTF/ckpt"
    export DS4_LAYER_DIR="$OUTF/layers" DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin"
    unset DS4_MV_PROBE_L DS4_MINVOL_MAXL \
          DS4_MINVOL_HIST DS4_MINVOL_FLOOR DS4_VQ_COLD_DIM DS4_VQ_COLD_NC \
          DS4_MV_FLOOR_LINE DS4_RR_IDS DS4_ANCHOR2 DS4_MINVOL_TARGET DS4_BWD DS4_FP_ONLY 2>/dev/null || true
    [ -n "${PROBE1:-}" ] && { export DS4_MINVOL_MAXL="${PROBE_MAXL:-1}"; LOG "★探针模式: 只量化前 ${DS4_MINVOL_MAXL} 层早退★"; }
    LOG "量化起跑 $([ -n "${PROBE1:-}" ] && echo 'L00 探针' || echo '43 层')(0731 源, 预算 ${BUDGET} GiB 载荷)"
    "$QBIN" "$IDS" 1716
    LOG "量化 rc=$?"
}

stage_backfit(){
    N=$(ls "$OUTF"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    EXPN=43; [ -n "${PROBE1:-}" ] && EXPN=1
    [ "$N" = "$EXPN" ] || { LOG "层文件 $N/$EXPN 不齐, 拒反修"; exit 2; }
    cd "$ROOT/gguf-tools/amp"
    export DS4_ANCHOR="$ANCHOR" DS4_NFIT="${BF_NFIT:-933}" DS4_THREADS="${DS4_THREADS:-6}"
    export DS4_CALIB_FULLSET=1
    export DS4_LAYER_DIR="$OUTF/layers" DS4_LCFG=$(printf 'g%.0s' $(seq 1 43)) DS4_COADAPT=1
    export DS4_VQ=1 DS4_TGT_ALPHA=1.0
    export DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="${HOT_TABLE:-$ROOT/gguf-tools/data/corpus/prog_active_top49.txt}"
    export DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin"
    # ★一次从头到尾的反修(2026-08-04 用户终裁)★: 不要 ALT逐层+sweep+回扫三段叠罗汉 —
    #   ①BF_ONLY: 跳过 ALT 逐层反修(层内判据, 保险门实锤端到端负贡献 1.9045>1.6474)
    #   ②ONEPASS sweep: 每层在端到端判据下选型+解z, 冻结基线一次遍历+统一终验+劣化全回滚
    #   ③GSWEEP=0: 回扫不跑 — 侧车架构下需要时删侧车一条 env 即可补跑
    LOG "★架构组件在场自检(2026-08-05 铁律)★ z变量[E/C/F] 四损失[KGRID la/lf/ls+lc] 感知[pc行权] 向后[TREF-B] 路由[GE-D投影; RB对哈希路由无对象(终审)]"
    export DS4_BF_ONLY=1
    export DS4_BWD=1 DS4_GSWEEP="${DS4_GSWEEP:-0}" DS4_BF_JUSTIFIED=1   # GSWEEP 可外覆盖(2026-08-05 一遍全局回扫) DS4_BF_TERM_MAXP=1 DS4_BF_MEMGB="${BF_MEMGB:-1}"
    export DS4_BF_FROM_BYTES=1   # ★字节起步(2026-08-03): 冷基座=量化段盘上字节 dequant, 免重编码
    export DS4_EXPORT_BYTES=0    # ★平行架构(2026-08-04): 反修段 dql 不可变 — export 只重建 op 侧车
    export DS4_BF_SCREEN_DIV=12 DS4_BF_SCREEN_K=4   # sweep 快刀(抽1/12行+top4候选)
    export DS4_BF_ONEPASS=1
    export DS4_GS_CONV_PCT=0.1   # (回扫若手动补跑: 轮间判停)
    # 二分 sweep(DS4_BF_SWEEP_ORDER=bisect + DS4_BF_BISECT_SKIP/_MAXSEG)已实现但默认不武装:
    #   单测实账(2026-08-04, mock=R30 首遍真实Δbest): 密集分布(35/42层有增益)只省2单元还漏
    #   L08(+0.53%)/L24(+0.29%); 稀疏分布端点孤岛(L41+3.0%)会被相邻平坦mid连坐剪掉 —
    #   剪枝与零漏检不可兼得, 质量门铁律优先。增益稀疏的增量复跑场景可手动开。
    export DS4_ROUTE_BIAS="$OUTF/route_bias_r30.bin" DS4_ROUTE_BIAS_ALPHA="${RB_ALPHA:-2.5}" DS4_ROUTE_BIAS_MINCNT=8
    unset DS4_TUNE DS4_MINVOL DS4_MV_BASELINE DS4_MV_COAD_BASE \
          DS4_MV_PROBE_L DS4_MINVOL_MAXL \
          DS4_MINVOL_HIST DS4_MINVOL_FLOOR DS4_MINVOL_TARGET DS4_MINVOL_ALPHA \
          DS4_VQ_RPLAN DS4_VOL_BUDGET_GIB DS4_PLAN DS4_CKPT_DIR \
          DS4_ROUTE_SEQ \
          DS4_RR_IDS DS4_ANCHOR2 DS4_EXPORT_GGUF DS4_REPAIR_COLD DS4_FP_ONLY 2>/dev/null || true
    # ★RB FIT 复活(2026-08-05 用户批): 0731 只有 L0-L2 哈希路由(hash_layer_count=3 实锤),
    #   L3-L42 部署态活分数 top-k — 8-03"全程哈希→RB 无对象"终审只对 3 层成立。Δb 统计
    #   寄生在反修自身前向(零额外前向, 统计段在锚 override 之前读学生 top-k), rb_save 收官
    #   落盘 → merge 段烘进 exp_probs_b(α 冠军档)。多次同层前向重复累计=均值归一无偏。
    export DS4_ROUTE_BIAS_FIT=1 DS4_ROUTE_BIAS_OUT="$OUTF/route_bias_r30.bin"
    export DS4_ANCHOR_ROUTE=1   # 超冠原样(2026-08-08 用户令还原): 锚路由反修
    export DS4_BF_GAIN_GATE="${GAIN_GATE:-0.05}"
    for v in DS4_TUNE DS4_MINVOL DS4_MV_BASELINE DS4_VQ_RPLAN; do
        [ -z "$(eval echo \"\${$v:-}\")" ] || { LOG "★$v 仍在场, 反修会走错分支 — 停★"; exit 8; }
    done
    [ -n "${DS4_ANCHOR_ROUTE:-}" ] || { LOG "★DS4_ANCHOR_ROUTE 不在场 — 停★"; exit 8; }
    [ -n "${PROBE1:-}" ] && { export DS4_NL=1; LOG "★探针模式: 只反修 L00(DS4_NL=1)★"; }
    LOG "反修起跑"
    "$QBIN" "$IDS" "${BF_S:-1716}"
    LOG "反修 rc=$?"
}

# (2026-08-04 平行架构后备份不再串接: dql 物理不可变有 md5 判决作证; 函数留作手动工具)
# 任何后段(反修/sweep/回扫)改写层文件前, 纯量化态永远有一份秒级可恢复的副本。
# 恢复: rm layers/*.bin && cp -c layers_quant_backup/* layers/
stage_qbackup(){
    B="$OUTF/layers_quant_backup"
    [ -d "$B" ] && { LOG "量化态备份已在 $B"; return 0; }
    mkdir -p "$B"
    cp -c "$OUTF"/layers/dql_L*.bin "$OUTF"/layers/dql_vq_L*.bin "$OUTF"/layers/opt_L*.bin "$B"/ 2>/dev/null
    N=$(ls "$B"/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    LOG "量化态 APFS 克隆备份 ✓ $N/43 层 → $B(零空间, 秒级)"
    [ "$N" = 43 ] || { LOG "★备份不齐 $N/43 — 停, 别让后段改写★"; exit 3; }
}
stage_student(){
    # ★学生回放遍【必须在合并/骨架之前】(2026-08-02 盘账定序)★
    # 五指标的学生分布要"层文件回放 + 完整锚(锚路由)"— 两者只在这个窗口同时在场。
    # 回放完把锚裁成 logits 段(0.887G), 释放 6.5G 给骨架抽取(峰值 16.4G)和消费合并。
    # (2026-08-07 假✓两连事故: "已在即跳过"在重反修场景让旧 logits 挡新评分 — 删, 评分永远重跑)
    [ -f "$ANCHOR" ] || { LOG "锚缺 — 学生回放没有锚路由"; exit 2; }
    N=$(ls "$OUTF"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    [ "$N" = 43 ] || { LOG "层文件 $N/43 不齐"; exit 2; }
    cd "$ROOT/gguf-tools/amp"
    STU_T0=$(date +%s)
    LOG "学生回放遍(层文件 + 100%锚路由 + Δb α 部署态, dump [S,VOCAB])"
    # ★真·纯回放(2026-08-03 实锤): -u DS4_GSWEEP 是暗雷(代码默认3=回扫照跑); COADAPT=1 会点燃
    #   逐层sweep(incr 门)。显式 GSWEEP=0 + BACKFIT_INCR=0 + BF_ONLY=1 ⇒ 字节回放+VERDICT+logits dump。
    env -u DS4_TUNE -u DS4_MINVOL -u DS4_MV_BASELINE -u DS4_VQ_RPLAN -u DS4_BWD \
        DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 \
        DS4_CALIB_FULLSET=1 \
        DS4_ANCHOR="$ANCHOR" DS4_NFIT=933 DS4_THREADS="${DS4_THREADS:-8}" \
        DS4_LAYER_DIR="$OUTF/layers" DS4_LCFG=$(printf 'g%.0s' $(seq 1 43)) \
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_COADAPT=1 \
        DS4_GO2B_HOT=1 DS4_GO2B_HOT_TABLE="${HOT_TABLE:-$ROOT/gguf-tools/data/corpus/prog_active_top49.txt}" \
        DS4_ZFILE="$OUTF/zfile.bin" DS4_ZCHAIN="$OUTF/zchain.bin" \
        DS4_ROUTE_BIAS="$OUTF/route_bias_r30.bin" DS4_ROUTE_BIAS_ALPHA="${RB_ALPHA:-2.5}" DS4_ROUTE_BIAS_MINCNT=8 \
        DS4_ANCHOR_ROUTE=1 DS4_DUMP_LOGITS="$OUTF/student_logits.bin" \
        "$QBIN" "$IDS" 1716 || { LOG "★学生回放失败★"; exit 3; }
    [ -f "$OUTF/student_logits.bin" ] || { LOG "★学生 logits 没落盘★"; exit 3; }
    # ★假✓自曝闸(2026-08-07: 两次静默秒过事故): logits 必须比本段起跑新
    [ "$(file_mtime "$OUTF/student_logits.bin")" -ge "$STU_T0" ] \
        || { LOG "★评分假完成 — logits 未更新(mtime 旧于起跑), 停★"; exit 3; }
}

stage_merge(){
    pgrep -f "[d]s4quant_run.* $IDS" >/dev/null && { LOG "仍有量化进程, 停"; exit 3; }
    MAN="$OUTF/layers/manifest.txt"
    [ -f "$MAN" ] || { LOG "manifest 缺"; exit 4; }
    TOT=$(awk '{s+=$2} END{printf "%.3f",s/1073741824}' "$MAN")
    # ★真账公式(2026-08-03 用户纠"预告 33.07 跑小了"): VQ 形态 = blob(manifest) + down 11.42 + 骨架 8.202
    LOG "manifest(blob) ${TOT} GiB / $(wc -l < "$MAN") 层 → 预期模型 $(awk -v t="$TOT" 'BEGIN{printf "%.2f",t+11.42+8.202}') GiB(blob+down+骨架)"
    rm -f "$MDL" "$MDL.bias0.bin"
    # ★裁锚腾盘(2026-08-02 盘账)★: 学生回放已完成 ⇒ 完整锚(6.5G)只剩 logits 段还有用
    # (五指标参考分布)。抽成 ref_logits.bin(0.887G, 学生 dump 同格式)后删锚。
    if [ -f "$ANCHOR" ]; then
        [ -f "$OUTF/student_logits.bin" ] || { LOG "★学生回放还没跑(student 阶段)— 删锚前必须先回放, 停★"; exit 8; }
        if [ ! -f "$R30/ref_logits.bin" ]; then
            python3 - "$ANCHOR" "$R30/ref_logits.bin" <<'PYEOF' || { LOG "★裁锚失败★"; exit 8; }
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
f = open(src, 'rb')
hd = struct.unpack('<8I', f.read(32)); assert hd[0] == 0x32415144
_, S, HCM, DIM, NL, VOCAB, NACT, _ = hd
f.read(8)
f.seek(NL*S*DIM*4 + NL*S*NACT*8 + NL*S*HCM*DIM*4, 1)
g = open(dst, 'wb'); g.write(struct.pack('<2i', S, VOCAB))
left = S*VOCAB*4
while left:
    b = f.read(min(1 << 24, left))
    assert b, '锚 logits 截断'
    g.write(b); left -= len(b)
print(f'ref_logits {S}x{VOCAB} ✓')
PYEOF
        fi
        LOG "ref_logits.bin 已抽取; 锚保留不删(2026-08-03 用户令: 合并不删除任何输入)"
    fi
    rm -rf "$OUTF/ckpt"
    SKEL="$R30/r30_skeleton.gguf"
    if [ ! -f "$SKEL" ]; then
        LOG "自产骨架不在, 现抽(0731 → --experts-hole → 紧凑骨架)"
        SKEL_HF="$DS4_HF" SKEL_TMPL="$R30/template_head.gguf" \
            bash "$SC/skel_from_hf.sh" "$SKEL" "${SKEL_TH:-4}" || { LOG "★骨架自产失败 — 停★"; exit 5; }
    fi
    bash "$SC/preflight_skeleton.sh" "$SKEL" 0.30 >&2 || LOG "★骨架干净度自检未过 — 继续但需人工看★"
    # ★盘账(非消费式, 2026-08-03 用户令"合并不删除"): 输出 ~45G 全新增, 源全保留 ⇒ free ≥ 46G。
    #   (2026-08-03 事故教训: 消费式把源删了而产物有损, 无法重合并 — 先验后删都不要, 直接不删。)
    FREE=$(disk_free_gb "$R30")
    [ "$FREE" -ge 56 ] || { LOG "★free ${FREE}G <56G(r60 非消费合并需全额) — 停★"; exit 7; }
    # ★v3: 冠军合并形态(vq_merge v4.1 同款) — 带 down(冷 w2 signref D 段), 偏移表权威解析
    python3 "$SC/dql_down_offset.py" "$OUTF/layers" 43 > "$OUTF/down_offsets.txt" \
        || { LOG "★down 偏移表生成失败★"; exit 6; }
    # ★VQ 路线(2026-08-03 用户令"vq必须支持"): 冠军 vq4bf 同构 = blob(冷w1w3 vq8x256+热16
    #   vq4x512 三矩阵)进 GGUF + down(dql D 段冷 w2 signref, 热槽=洞由 blob o2 覆盖) + 自产骨架。
    #   --gud(code2b 全 signref 搬运)与 VQ 产物不兼容(G/U=稀疏洞), 已弃用。
    LOG "起合并(VQ 冠军形态: blob+down+骨架, 非消费式=源全保留)"
    "$(dirname "$0")/../quantize/vq_merge_v4" --merge \
        --skeleton "$SKEL" \
        --blob-sizes "$MAN" \
        --down-offsets "$OUTF/down_offsets.txt" \
        --dql-host 127.0.0.1 --dql-dir "$OUTF/layers" \
        --out "$MDL" >/tmp/r30_merge.log 2>&1
    MRC=$?
    [ $MRC -eq 0 ] || { LOG "合并失败 rc=$MRC"; tail -5 /tmp/r30_merge.log >&2; exit $MRC; }
    GB=$(ls -l "$MDL" | awk '{printf "%.2f",$5/1e9}')
    GIB=$(ls -l "$MDL" | awk '{printf "%.2f",$5/1073741824}')
    LOG "合并 ✓ 落地 ${GB} GB (${GIB} GiB) — 口径=Finder十进制GB(2026-08-05 用户终裁)"
    CAPGB=$(python3 -c "import json;print(json.load(open('$PLAN_JSON'))['model_GB']+1)" 2>/dev/null || echo 999)
    awk -v g="$GB" -v c="$CAPGB" 'BEGIN{exit !(g>c)}' && { LOG "★落地 ${GB} GB > 计划 ${CAPGB} GB — 停★"; exit 7; }
    # ★RB 烘焙(2026-08-05 用户批, 8-03 终审纠偏): 哈希路由只有 L0-L2, L3-L42 部署态
    #   活分数 top-k 有量化漂移 — 反修段 FIT 落盘的 α·Δb 烘进 exp_probs_b.bias(原位,
    #   体积不变)。骨架自产无旧偏置 ⇒ champ 槽传零 riba。L0-L2 无该张量, rebake 自动跳过。
    RBF="$OUTF/route_bias_r30.bin"
    if [ -n "${RB_REBAKE:-}" ] && [ -f "$RBF" ]; then   # 超冠形态默认不烘焙(2026-08-08 用户令还原): Δb=侧车挂载
        python3 - /tmp/rb_zero.bin <<'PYEOF' || { LOG "★零 riba 生成失败★"; exit 8; }
import struct, sys
NL, NEXP = 43, 256
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<4I', 0x41494252, NL, NEXP, 0))
    f.write(b'\0' * (NL * NEXP * 4 * 2))
PYEOF
        "$(dirname "$0")/../amp/route_bias_rebake" "$MDL" /tmp/rb_zero.bin 0 "$RBF" "${RB_ALPHA:-2.5}" \
            || { LOG "★RB 烘焙失败 — 停(裸路由模型不交付)★"; exit 8; }
        LOG "RB 烘焙 ✓ (α=${RB_ALPHA:-2.5}, L3+ 分数路由漂移补偿)"
    else
        LOG "★route_bias_r30.bin 缺 — 反修段未跑 FIT? 裸路由合并, 需人工确认★"
    fi
}

stage_metrics(){
    # ★五项标准指标(2026-08-02 用户令)★ 学生分布在 student 阶段已 dump;
    # 参考分布 = 完整锚(若还在)或裁剪后的 ref_logits.bin(merge 阶段产)。
    [ -f "$OUTF/student_logits.bin" ] || { LOG "学生 logits 缺 — 先跑 student 阶段"; exit 2; }
    REF_ARGS=(--ref "$ANCHOR")
    [ -f "$ANCHOR" ] || { [ -f "$R30/ref_logits.bin" ] && REF_ARGS=(--ref-raw "$R30/ref_logits.bin") \
        || { LOG "参考分布缺(锚和 ref_logits 都不在)"; exit 2; }; }
    LOG "===== 五项标准指标(量化域, v5mini held=783) ====="
    "$(dirname "$0")/../bench/anchor_metrics" "${REF_ARGS[@]}" --ids "$IDS" \
        --student "$OUTF/student_logits.bin" --fit 933 | tee "$R30/metrics_v5mini.txt" >&2
    if [ -f "$MDL" ] && [ -f "$R30/r30_skeleton.gguf" ]; then
        LOG "===== Bit-exact weights(合并 GGUF vs 骨架) ====="
        python3 "$SC/bitexact_check.py" "$R30/r30_skeleton.gguf" "$MDL" | tee "$R30/metrics_bitexact.txt" >&2
    fi
}

# ==== md86 两段式反修(2026-08-10 归并; 用户设计: ①每层贪心最优 ②收尾链态一遍) ====
# 消费 zlayer.py 全套新旋钮(组件门/ftA/链模式); 锚与层目录经 env 注入:
#   MD_ANCHOR=FP锚  MD_CHAIN=链态锚  MD_LAYERS=层目录  MD_IDS=ids  MD_S=总token  MD_FR=fit区间  MD_EV=ev区间
stage_zside2(){   # ①统一反修43层(2026-08-13 定版): 每层 z+GE → 组合<ERF_BAR 时叠加 ERF(权重空间
    #   ΔW_w2 SVD r 方向+α残差重加权+token能量门, 记录 zl.ERF, C 回放 type8)。双路并行;
    #   MD_L0/MD_L1 层范围(单层修复同用)。默认口径=md86v2 布局(可 env 覆盖)。
    MD_S="${MD_S:-4683}"
    MD_FR="${MD_FR:-0:1287,1716:3003,3383:4683}"
    MD_EV="${MD_EV:-1287:1716}"
    MD_LAYERS="${MD_LAYERS:-$OUTF/layers}"
    MD_ANCHOR="${MD_ANCHOR:?需 MD_ANCHOR=解算FP锚(与量化校准语料不相交)}"
    cd "$ROOT"
    _glane(){ for L in "$@"; do
        rm -f "$MD_LAYERS/zcache_L$(printf %02d $L).npz"   # 层前必删: 陈旧缓存(异尺锚)复用=IndexError 崩 lane(08-14 事故)
        env VECLIB_MAXIMUM_THREADS=4 DS4_ZL_NTOK=$MD_S DS4_ZL_FIT_RANGES=$MD_FR DS4_ZL_EV_RANGE=$MD_EV \
            DS4_ZL_ERF_R="${ERF_R:-16}" DS4_ZL_ERF_BAR="${ERF_BAR:-0.01}" DS4_ZL_SWLIM="${ZL_SWLIM:-10}" DS4_ZL_GE_LAM="${GE_LAM:-1e-3}" \
            python3 -u gguf-tools/go-onebit/zlever/zlayer.py "$DS4_HF" "$MD_LAYERS" "$MD_ANCHOR" $L 1024 1 \
            || LOG "★贪心L${L}失败★"
        rm -f "$MD_LAYERS/zcache_L$(printf %02d $L).npz"
    done; }
    local L0=${MD_L0:-0} L1=${MD_L1:-42} MID
    if [ $((L1-L0)) -lt 3 ]; then _glane $(seq $L0 $L1)
    else MID=$(( (L0+L1)/2 ))
        _glane $(seq $L0 $MID) & local P1=$!
        _glane $(seq $((MID+1)) $L1) & local P2=$!
        wait $P1 $P2
    fi
    LOG "①贪心完($L0-$L1, 双路)"
}
stage_sweep(){    # ②收尾链态一遍: 回放建链锚(带全部贪心记录)→双路并行逐层整替
    cd "$ROOT/gguf-tools/amp"
    env DS4_HF=$DS4_HF DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 \
        DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 DS4_ANCHOR=$MD_ANCHOR DS4_NFIT=${MD_NFIT:-3874} \
        DS4_THREADS=8 DS4_LAYER_DIR=$MD_LAYERS DS4_LCFG=$(printf "g%.0s" $(seq 1 43)) \
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_CHAIN_ANCHOR=$MD_CHAIN \
        "$QBIN" "$MD_IDS" "$MD_S" >/dev/null || { LOG "★链锚回放失败★"; exit 2; }
    cd "$ROOT"
    _lane(){ for L in "$@"; do
        local LL=$(printf %02d $L)
        cp -c "$MD_LAYERS/dql_L$LL.bin" "$MD_LAYERS/dql_L$LL.bak" 2>/dev/null \
            || cp "$MD_LAYERS/dql_L$LL.bin" "$MD_LAYERS/dql_L$LL.bak"   # APFS clone 零成本快照(破坏前先保全)
        local MANBAK=$(grep "^$L " "$MD_LAYERS/zinject_manifest.txt" 2>/dev/null || true)
        python3 gguf-tools/go-onebit/zlever/pop_layer.py $L "$MD_LAYERS"
        rm -f "$MD_LAYERS/zcache_L$LL.npz"
        env VECLIB_MAXIMUM_THREADS=4 DS4_ZL_XANCHOR=$MD_CHAIN DS4_ZL_NTOK=$MD_S \
            DS4_ZL_FIT_RANGES=$MD_FR DS4_ZL_EV_RANGE=$MD_EV \
            python3 -u gguf-tools/go-onebit/zlever/zlayer.py "$DS4_HF" "$MD_LAYERS" "$MD_ANCHOR" $L 1024 1 \
            || LOG "★收尾L$L失败★"
        rm -f "$MD_LAYERS/zcache_L$LL.npz"
        if ! grep -q "^$L " "$MD_LAYERS/zinject_manifest.txt" 2>/dev/null; then
            # ★谁好留谁(2026-08-10 用户令): 链态解被闸拒 → 恢复原记录, 不许层裸奔
            mv "$MD_LAYERS/dql_L$LL.bak" "$MD_LAYERS/dql_L$LL.bin"
            [ -n "$MANBAK" ] && echo "$MANBAK" >> "$MD_LAYERS/zinject_manifest.txt"
            LOG "L$L 链态拒 → 原记录已恢复"
        else
            rm -f "$MD_LAYERS/dql_L$LL.bak"
        fi
    done; }
    _lane $(seq 0 21) & local P1=$!
    _lane $(seq 22 42) & local P2=$!
    wait $P1 $P2
    LOG "②收尾链态一遍完(双路)"
}

stage_addon(){   # ②叠加式链修正遍(2026-08-10 用户设计): 链回放(带贪心记录)→双路Δ解算+合并注入
    cd "$ROOT/gguf-tools/amp"
    env DS4_HF=$DS4_HF DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 \
        DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 DS4_ANCHOR=$MD_ANCHOR DS4_NFIT=${MD_NFIT:-3874} \
        DS4_THREADS=8 DS4_LAYER_DIR=$MD_LAYERS DS4_LCFG=$(printf "g%.0s" $(seq 1 43)) \
        DS4_VQ=1 DS4_TGT_ALPHA=1.0 DS4_CHAIN_ANCHOR=$MD_CHAIN \
        "$QBIN" "$MD_IDS" "$MD_S" >/dev/null || { LOG "★链锚回放失败★"; exit 2; }
    cd "$ROOT"
    _alane(){ for L in "$@"; do
        rm -f "$MD_LAYERS/zcache_L$(printf %02d $L).npz"
        env VECLIB_MAXIMUM_THREADS=4 DS4_ZL_ADDON=1 DS4_ZL_XANCHOR=$MD_CHAIN DS4_ZL_NTOK=$MD_S \
            DS4_ZL_FIT_RANGES=$MD_FR DS4_ZL_EV_RANGE=$MD_EV \
            python3 -u gguf-tools/go-onebit/zlever/zlayer.py "$DS4_HF" "$MD_LAYERS" "$MD_ANCHOR" $L 1024 1 \
            || LOG "★叠加L${L}失败★"
        rm -f "$MD_LAYERS/zcache_L$(printf %02d $L).npz"
    done; }
    _alane $(seq 0 21) & local P1=$!
    _alane $(seq 22 42) & local P2=$!
    wait $P1 $P2
    LOG "②叠加式链修正完(双路)"
}

stage_score2(){   # 五指标双尺: A=老编程锚(历史对表) B=当前战役锚(MD_ANCHOR 在则跑)
    cd "$ROOT/gguf-tools/amp"
    local LCx=$(printf "g%.0s" $(seq 1 43))
    env DS4_HF=$DS4_HF DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 \
        DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 DS4_ANCHOR=$R30/anchor_r30_s1716.bin DS4_NFIT=1287 \
        DS4_THREADS=8 DS4_LAYER_DIR=$MD_LAYERS DS4_LCFG=$LCx DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
        DS4_DUMP_LOGITS=/tmp/score_prog_student.bin \
        "$QBIN" "$IDS" 1716 2>&1 | tail -3
    cd "$ROOT"
    "$(dirname "$0")/../bench/anchor_metrics" --ref $R30/anchor_r30_s1716.bin \
        --ids "$IDS" --student /tmp/score_prog_student.bin --fit 1287
    if [ -n "${MD_ANCHOR:-}" ] && [ -n "${MD_IDS:-}" ]; then
        cd "$ROOT/gguf-tools/amp"
        env DS4_HF=$DS4_HF DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 \
            DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 DS4_ANCHOR=$MD_ANCHOR DS4_NFIT=${MD_NFIT:-3874} \
            DS4_THREADS=8 DS4_LAYER_DIR=$MD_LAYERS DS4_LCFG=$LCx DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
            DS4_DUMP_LOGITS=/tmp/score_md_student.bin \
            "$QBIN" "$MD_IDS" "$MD_S" 2>&1 | tail -3
        cd "$ROOT"
        "$(dirname "$0")/../bench/anchor_metrics" --ref $MD_ANCHOR \
            --ids "$MD_IDS" --student /tmp/score_md_student.bin --fit ${MD_NFIT:-3874}
    fi
}

stage_dynvol(){   # 动态体积单层探针(2026-08-11 用户令): PL=层 PT=档nc — 该层@PT 隔离量化 + z重解只报数
    local PL=${PL:?需 PL=层号} PT=${PT:?需 PT=档位nc}
    local DV="$R30/dynvol"; mkdir -p "$DV"
    local LL=$(printf %02d $PL)
    [ -f "$DV/dql_vq_L$LL.orig" ] || cp -c "$R30/nl86/layers/dql_vq_L$LL.bin" "$DV/dql_vq_L$LL.orig" 2>/dev/null \
        || cp "$R30/nl86/layers/dql_vq_L$LL.bin" "$DV/dql_vq_L$LL.orig"
    if [ -f "$R30/nl86/ckpt/L$LL.bin" ] && [ ! -f "$DV/L$LL.ckpt.orig" ]; then mv "$R30/nl86/ckpt/L$LL.bin" "$DV/L$LL.ckpt.orig"; fi
    rm -f "$R30/nl86/ckpt/L$LL.bin"
    sed "s/^L=$PL dim=4 nc=[0-9]*/L=$PL dim=4 nc=$PT/" "$R30/rplan_q2_86g.txt" > "$DV/rplan_L${LL}_$PT.txt"
    LOG "dynvol L$LL@$PT 量化发车(隔离探针, v2锚)"
    cd "$ROOT/gguf-tools/amp"
    env DS4_HF=$DS4_HF DS4_ANCHOR=$R30/anchor_md86v2_s4683.bin DS4_NFIT=3874 DS4_THREADS=8 \
        DS4_CALIB_FULLSET=1 DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_TUNE=1 DS4_PURE_VQ=1 DS4_VQ=1 \
        DS4_TGT_ALPHA=1.0 DS4_VQ_RPLAN="$DV/rplan_L${LL}_$PT.txt" DS4_VOL_BUDGET_GIB=92 \
        DS4_MV_PROBE_L=$PL DS4_ANCHOR_ROUTE=1 DS4_BF_GAIN_GATE=0.05 DS4_PLAN=$R30/nl86/plan.txt \
        DS4_CKPT_DIR=$R30/nl86/ckpt DS4_LAYER_DIR=$R30/nl86/layers DS4_ZFILE=$R30/nl86/zfile.bin \
        DS4_ZCHAIN=$R30/nl86/zchain.bin \
        "$QBIN" "$G7/rr_md86v2.ids" 4683 2>&1 | grep -E "贪心选|体积账|held|VERDICT" | tail -8
    cd "$ROOT"
    rm -f "$R30/nl86/layers/zcache_L$LL.npz"
    env VECLIB_MAXIMUM_THREADS=4 DS4_ZL_NTOK=4683 DS4_ZL_FIT_RANGES=0:1287,1716:3003,3383:4683 DS4_ZL_EV_RANGE=1287:1716 \
        python3 -u gguf-tools/go-onebit/zlever/zlayer.py "$DS4_HF" "$R30/nl86/layers" "$R30/anchor_md86v2_s4683.bin" $PL 1024 1 2>&1 | tail -6
    rm -f "$R30/nl86/layers/zcache_L$LL.npz"
    LOG "dynvol L$LL@$PT 完"
}

stage_zstrip(){   # 按账本剥离全部侧车记录回裸态(重跑反修的前置; 账本清零)
    python3 - <<'PYEOF'
import os,struct
ld=os.environ.get("MD_LAYERS") or os.path.join(os.environ["OUTF_DIR"],"layers")
man=os.path.join(ld,"zinject_manifest.txt")
n=0
if os.path.exists(man):
    for ln in open(man):
        p=ln.split()
        L,osz,n0=int(p[0]),int(p[1]),int(p[2])
        if osz<=0: continue
        f=open(os.path.join(ld,f"dql_L{L:02d}.bin"),"r+b")
        f.truncate(osz); f.seek(8); f.write(struct.pack("<I",n0)); f.close(); n+=1
    os.remove(man)
print(f"剥离{n}层, 回裸态")
PYEOF
}

stage_quant86(){   # 86G 底座量化(冠军 08-13 配方原样 env 化; en86 等战役复用)
    local Q_IDS="${Q86_IDS:?}" Q_S="${Q86_S:?}" Q_NFIT="${Q86_NFIT:?}" Q_ANCHOR="${Q86_ANCHOR:?}" Q_OUT="${Q86_OUT:?}"
    local NDONE=$(ls "$Q_OUT"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    [ "$NDONE" = 43 ] && { LOG "quant86 已有 43/43 层, 跳过(EV 换装重启复用)"; return 0; }
    [ -f "$Q_ANCHOR" ] || { LOG "★校准锚缺 $Q_ANCHOR★"; exit 2; }
    mkdir -p "$Q_OUT/layers" "$Q_OUT/ckpt"
    FREE=$(disk_free_gb "$R30")
    [ "$FREE" -ge 75 ] || { LOG "★盘闸 free ${FREE}G <75G 停★"; exit 6; }
    cd "$ROOT/gguf-tools/amp"
    env DS4_HF=$DS4_HF DS4_ANCHOR="$Q_ANCHOR" DS4_NFIT=$Q_NFIT DS4_THREADS="${DS4_THREADS:-8}" DS4_CALIB_FULLSET=1 \
        DS4_MINVOL=1 DS4_MV_BASELINE=1 DS4_TUNE=1 DS4_PURE_VQ=1 DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
        DS4_VQ_RPLAN="${RPLAN86:-$R30/rplan_q2_86g.txt}" DS4_VOL_BUDGET_GIB="${VOLB86:-92}" \
        DS4_ANCHOR_ROUTE=1 DS4_BF_GAIN_GATE=0.05 DS4_PLAN=$Q_OUT/plan.txt DS4_CKPT_DIR=$Q_OUT/ckpt \
        DS4_LAYER_DIR=$Q_OUT/layers DS4_ZFILE=$Q_OUT/zfile.bin DS4_ZCHAIN=$Q_OUT/zchain.bin \
        "$QBIN" "$Q_IDS" "$Q_S"
    LOG "quant86 rc=$?"
    N=$(ls "$Q_OUT"/layers/dql_L*.bin 2>/dev/null | wc -l | tr -d ' ')
    [ "$N" = 43 ] || { LOG "★quant86 层不齐 $N/43★"; exit 3; }
}

_anchor_ok(){   # 锚完整性实读验证(08-14 事故: wdog 杀在写出峰, 部分锚骗过 -f 弱闸 → 双实例互杀链)
    [ -f "$1" ] || return 1
    ( cd "$ROOT" && "$(dirname "$0")/../bench/anchor_metrics" --ref "$1" --ids "$2" >/dev/null 2>&1 )
}
stage_en86_anchors(){   # en86 双锚捕获(缺/废哪捕哪): 底座校准锚 + 反修解算锚
    cd "$ROOT/gguf-tools/amp"
    if ! _anchor_ok "$Q86_ANCHOR" "$Q86_IDS"; then
        rm -f "$Q86_ANCHOR"; LOG "捕底座校准锚 S=$Q86_S"
        DS4_FP_ONLY=1 DS4_ANCHOR="$Q86_ANCHOR" DS4_THREADS="${DS4_THREADS:-8}" "$QBIN" "$Q86_IDS" "$Q86_S"
        _anchor_ok "$Q86_ANCHOR" "$Q86_IDS" || { LOG "★校准锚缺/截断★"; exit 3; }
    fi
    if ! _anchor_ok "$EN_MD_ANCHOR" "$EN_MD_IDS"; then
        rm -f "$EN_MD_ANCHOR"; LOG "捕反修解算锚 S=$EN_MD_S"
        DS4_FP_ONLY=1 DS4_ANCHOR="$EN_MD_ANCHOR" DS4_THREADS="${DS4_THREADS:-8}" "$QBIN" "$EN_MD_IDS" "$EN_MD_S"
        _anchor_ok "$EN_MD_ANCHOR" "$EN_MD_IDS" || { LOG "★解算锚缺/截断★"; exit 3; }
    fi
    LOG "双锚 ✓ $(ls -l "$Q86_ANCHOR" "$EN_MD_ANCHOR" | awk '{printf "%.1fG ",$5/1073741824}')"
}

stage_en86_judge(){   # 双盲判(零泄漏协议=08-13 盲判同款: NFIT=1 无锚路由无RB): wt2 test + rr_hard
    local LCx=$(printf "g%.0s" $(seq 1 43)) J TAG AN ID ST
    for J in "wt2:$R30/anchor_wt2_s2653.bin:$G7/wt2.ids:/tmp/en86_wt2_student.bin" \
             "rrh:$R30/anchor_rr_hard_s1716.bin:$G7/rr_hard.ids:/tmp/en86_rrhard_student.bin"; do
        IFS=: read -r TAG AN ID ST <<<"$J"
        [ -f "$AN" ] || { LOG "★判决锚缺 $AN★"; continue; }
        cd "$ROOT/gguf-tools/amp"
        env DS4_HF=$DS4_HF DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 \
            DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 DS4_ANCHOR="$AN" DS4_NFIT=1 DS4_THREADS=8 \
            DS4_LAYER_DIR=$R30/en86/layers DS4_LCFG=$LCx DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
            DS4_DUMP_LOGITS="$ST" "$QBIN" "$ID" 8000 2>&1 | tail -3
        cd "$ROOT"
        echo "== 盲判[$TAG] =="
        "$(dirname "$0")/../bench/anchor_metrics" --ref "$AN" --ids "$ID" --student "$ST" --tail 5
    done
}

stage_progz_judge(){   # 真域held盲判(2026-08-16 用户设计): prog语料回放, --fit 1287 → held 1287:1716 解算全程未见
    local LCx=$(printf "g%.0s" $(seq 1 43))
    cd "$ROOT/gguf-tools/amp"
    env DS4_HF=$DS4_HF DS4_GSWEEP=0 DS4_BF_TERMINAL=0 DS4_BF_ONLY=1 DS4_COADAPT=1 \
        DS4_CALIB_FULLSET=1 DS4_EXPORT_BYTES=0 DS4_ANCHOR="$R30/anchor_prog_s1716.bin" DS4_NFIT=1 DS4_THREADS=8 \
        DS4_LAYER_DIR=$R30/en86/layers DS4_LCFG=$LCx DS4_VQ=1 DS4_TGT_ALPHA=1.0 \
        DS4_DUMP_LOGITS="${PROG_ST:-/tmp/progz_student.bin}" "$QBIN" "$G7/rr_calib_prog_v5mini.ids" 8000 2>&1 | tail -3
    cd "$ROOT"
    echo "== 真域盲判[prog held=1287:1716] =="
    "$(dirname "$0")/../bench/anchor_metrics" --ref "$R30/anchor_prog_s1716.bin" --ids "$G7/rr_calib_prog_v5mini.ids" \
        --student "${PROG_ST:-/tmp/progz_student.bin}" --fit 1287 --tail 5
}

case "${1:-all}" in
    anchor)  WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_anchor ;;
    en86)  # ★英文补域战役一键(2026-08-14 用户令"跑", 语料解冻: 校准+wikitext-2 train, 判决 test 零相交)★
           # 链: 双锚→86G量化(冠军配方+EN1287入fit)→zstrip→统一反修(SWLIM=60,z/GE/ERF)→双盲判。
           # 语料: rr_md86en.ids(S=6019 fit5161=3874+EN1287|held858殿后) / rr_md86v2en.ids(S=5970=v2+EN尾接)
           export Q86_IDS="$G7/rr_md86en.ids" Q86_S=6019 Q86_NFIT=5161
           export Q86_ANCHOR="$R30/anchor_md86en_s6019.bin" Q86_OUT="$R30/en86"
           export EN_MD_ANCHOR="$R30/anchor_md86v2en_s5970.bin" EN_MD_IDS="$G7/rr_md86v2en.ids" EN_MD_S=5970
           export WDOG_PAT="[d]s4quant_run"   # 锚捕获/量化/判决全程盯(锚合法峰~19G, 发车须 WDOG_MB=22528)
           WDOG & trap 'kill %1 2>/dev/null||true' EXIT
           stage_en86_anchors
           stage_quant86
           export OUTF_DIR="$Q86_OUT" MD_LAYERS="$Q86_OUT/layers"
           stage_zstrip
           # ★组件门评审区=prog+EN 混合(2026-08-14 用户令"全能力不跑偏"): EN 尾 250tok 出解算池入 EV★
           MD_ANCHOR="$EN_MD_ANCHOR" MD_IDS="$EN_MD_IDS" MD_S="$EN_MD_S" MD_NFIT=3874 \
               MD_FR="0:1287,1716:3003,3383:5720" MD_EV="1287:1716,5720:5970" stage_zside2
           stage_en86_judge ;;
    en86judge) stage_en86_judge ;;
    progzjudge) stage_progz_judge ;;
    progz86)  # ★真域链上判决(2026-08-16 用户设计"专业语料反修, 通用+无泄漏专业分别跑分")★
              # 形态修正(首跑两事故: ①判决回放合法峰12.3G被11900误杀→22528; ②INJ=1对已注入层
              # "跳过"=ADDON空转→标准战役流): M10快照→M10基线prog held判→zstrip回裸底→prog反修
              # (裸底全新注入, fit0:1150/组件门1150:1287, held1287:1716解算全程不见=零泄漏)
              # →三尺终判(prog held + wt2 + rr_hard)。对照=M10快照(wt2 0.4522/rrh 0.1979在案)。
              # 回滚=rm -rf layers && mv layers_m10snap layers。
              SNAP="$R30/en86/layers_m10snap"
              if [ ! -d "$SNAP" ]; then
                  cp -c -R "$R30/en86/layers" "$SNAP" || { LOG "★快照失败★"; exit 7; }
                  LOG "M10 快照 ✓ layers_m10snap (cp -c 零空间)"
              else LOG "快照已在(前跑遗留), 复用"; fi
              export WDOG_PAT="[d]s4quant_run" WDOG_MB=22528
              WDOG & trap 'kill %1 2>/dev/null||true' EXIT
              if [ ! -f /tmp/progz_student_pre.bin ]; then
                  LOG "== 基线: M10 态 prog held 盲判 =="
                  PROG_ST=/tmp/progz_student_pre.bin stage_progz_judge
              fi
              export OUTF_DIR="$R30/en86" MD_LAYERS="$R30/en86/layers" ZL_SWLIM=60
              stage_zstrip
              MD_ANCHOR="$R30/anchor_prog_s1716.bin" MD_S=1716 MD_FR="0:1150" MD_EV="1150:1287" stage_zside2
              LOG "== 终判: 裸底+progz 态 三尺 =="
              PROG_ST=/tmp/progz_student_post.bin stage_progz_judge
              stage_en86_judge
              LOG "progz86 全链完" ;;
    quant86)   WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_quant86 ;;
    dynvol)  stage_dynvol ;;
    zstrip)  export OUTF_DIR="$OUTF"; stage_zstrip ;;
    zscore)  # ★终版一键流水(2026-08-13 用户令"流水线脚本必须对, 别停留在上下文")★:
             # 剥记录→统一反修(z+GE+叠加ERF)→双尺评分。env: MD_ANCHOR 必填, MD_IDS/MD_S 评分段用。
             export OUTF_DIR="$OUTF"; stage_zstrip
             stage_zside2
             stage_score2 ;;
    plan)    stage_plan ;;
    zside2)  stage_zside2 ;;
    score2)  stage_score2 ;;
    addon)   stage_addon ;;
    sweep)   stage_sweep ;;
    quant)   WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_quant ;;
    qbackup) stage_qbackup ;;
    backfit) WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_backfit ;;
    probe1bf) # ★反修段单层复跑(SIGSEGV 归因探针): 只跑 stage_backfit(PROBE1), 产物需已在场★
             WDOG & trap 'kill %1 2>/dev/null||true' EXIT
             export PROBE1=1
             stage_backfit ;;
    probe1)  # ★单层探针(2026-08-03 用户令: L00 只量化→L00 只反修, 日志对预期)★
             # 兼作: Jacobi w2 求解改造判决(held 对拍 v4c L00=0.1510±0.5%)+ mvq 删除后冒烟
             # + 固定模型论断验证(反修前后 VQ 侧车 md5 必须一致)。
             WDOG & trap 'kill %1 2>/dev/null||true' EXIT
             export PROBE1=1
             stage_quant || exit $?
             M5A=$(md5 -q "$OUTF/layers/dql_vq_L00.bin" 2>/dev/null || echo MISSING)
             D5A=$(md5 -q "$OUTF/layers/dql_L00.bin" 2>/dev/null || echo MISSING)
             LOG "探针·量化段完 VQ侧车md5=$M5A dql本体md5=$D5A $(ls -l "$OUTF/layers/" 2>/dev/null | awk 'NR>1{printf "%s(%.1fM) ",$NF,$5/1048576}')"
             stage_backfit || exit $?
             M5B=$(md5 -q "$OUTF/layers/dql_vq_L00.bin" 2>/dev/null || echo MISSING)
             D5B=$(md5 -q "$OUTF/layers/dql_L00.bin" 2>/dev/null || echo MISSING)
             if [ "$M5A" = "$M5B" ]; then LOG "探针·反修段完 VQ侧车md5一致✓(固定模型: 反修未动VQ载荷一个bit)";
             else LOG "★探针·VQ侧车md5变了($M5A→$M5B) — 预期外, 反修改写了固定模型★"; fi
             # parallel-arch verdict (2026-08-04): backfit must never touch dql bytes
             if [ "${D5A}" = "${D5B}" ]; then LOG "探针·dql本体md5一致✓(平行架构: 反修零触碰量化模型)";
             else LOG "★探针·dql本体md5变了(${D5A} -> ${D5B}) — 平行架构被破坏★"; exit 8; fi
             LOG "探针收官: 日志自检点=档位vq0x0/SEARCH base/VQ_GATE/★贪心选held/vq_keep/ALT行" ;;
    student) WDOG & trap 'kill %1 2>/dev/null||true' EXIT; stage_student ;;
    merge)   stage_merge ;;
    metrics) stage_metrics ;;
    all)     WDOG & trap 'kill %1 2>/dev/null||true' EXIT
             stage_anchor && stage_plan && stage_quant && stage_backfit && stage_student \
                 || { LOG "★推进段失败 — 停★"; exit 9; }
             kill %1 2>/dev/null||true
             stage_merge && stage_metrics
             LOG "战役收官" ;;
    *) echo "用法: r30_campaign.sh [anchor|plan|quant|backfit|student|merge|metrics|probe1|all]" >&2; exit 1 ;;
esac
