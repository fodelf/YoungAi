#!/bin/bash
# zloss_campaign.sh — 单层贪心四损失反修战役(2026-08-26 用户设计重实现)。
#
# 链: capture(解码路×2+逐位复现闸) → npy → teacher(FP 锚口径, ±10 截断)
#     → probe(L20 十分钟针) → solve(43 层, zloss_solve: ds4_z+ds4_loss 模块复用)
#     → judge(caliper 五指标, 对表 裸 0.47055 / r64c 冠军 0.42510)
#     → engine(dql_to_zchain + --zchain smoke)
# 用法: bash zloss_campaign.sh [capture|npy|teacher|probe|solve|judge|engine|all]
# ★无环境变量(铁律): 参数全部写死在此, 换轮改这里并记录。★
set -uo pipefail
ROOT="$HOME/ds4-main"
GT="$ROOT/gguf-tools"
D2="$ROOT/gguf/go-onebit/vqhalf"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
HF="$ROOT/hf/DeepSeek-V4-Flash-0731"
M="$ROOT/gguf/ds4-vq86h.gguf"
IDS="$D2/vqhalf_a.ids"          # 放大器半(与量化半零重叠, 语料纪律不变)
ANC="$D2/anchor_a_clean_s8192.bin"
CAP="$D2/zloss_cap"; NPY="$D2/zloss_npy"; WS="$D2/zloss"
NTOK=8192; NFIT=6144
RANKS="16,64"; LAMBDAS="3e-1,1,3,10"
WA=1; WC=0.5; WSM=0.1; WF=1e-3   # 四损失权(ds4_loss_total; CLI 面板, 换轮改此处)
LOG(){ echo "[zloss $(date +%H:%M:%S)] $*"; }
DIE(){ LOG "★$*★"; exit 1; }

# 行掩码(08-24 拼接毒定罪沿用): 256-token 块互织语料, 每块前 64 行=异域上下文污染行剔除;
# 前 24 块=拟合, 后 8 块=held。
FR=""; ER=""
for b in $(seq 0 31); do
    seg="$((b*256+64)):$(( (b+1)*256 ))"
    if [ "$b" -lt 24 ]; then FR="${FR:+$FR,}$seg"; else ER="${ER:+$ER,}$seg"; fi
done

WD=""
wd_start(){ ( while true; do
    A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "${A:-99}" -lt 4 ] && { echo "[watchdog] MemAvailable=${A}GB <4GB ★杀★" >&2
        pkill -9 -f 'ds4 --cuda'; pkill -9 -f zloss_solve; pkill -9 -f teacher_routed; break; }
    sleep 5; done ) & WD=$!; }
wd_stop(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null; WD=""; }
trap 'wd_stop' EXIT

stage_capture(){
    [ -s "$CAP/raw_ffn_in_L0" ] && { LOG "①捕获已在, 跳过"; return 0; }
    [ -s "$M" ] || DIE "基座缺 $M"; [ -s "$IDS" ] || DIE "ids 缺 $IDS"
    cd "$ROOT"; wd_start
    LOG "①取料#1 解码路 on vq86h × 放大器半(含 raw_ffn_out=学生全量化计算输出)"
    rm -rf "$CAP" "${CAP}2"; mkdir -p "$CAP" "${CAP}2"
    timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$IDS" \
        --score-out /tmp/zloss_cap1.bin --cap-dir "$CAP" </dev/null 2>&1 | tail -1
    [ -s "$CAP/raw_ffn_in_L42" ] || DIE "取料#1 没落盘"
    LOG "①取料#2 复现检查遍"
    timeout --foreground 7200 ./ds4 --cuda -m "$M" --score-ids "$IDS" \
        --score-out /tmp/zloss_cap2.bin --cap-dir "${CAP}2" </dev/null 2>&1 | tail -1
    for L in 0 16 32 42; do
        cmp "$CAP/raw_ffn_in_L$L"  "${CAP}2/raw_ffn_in_L$L"  || DIE "L$L ffn_in 不复现, 产物作废"
        cmp "$CAP/raw_ffn_out_L$L" "${CAP}2/raw_ffn_out_L$L" || DIE "L$L ffn_out 不复现, 产物作废"
        cmp "$CAP/raw_route_L$L"   "${CAP}2/raw_route_L$L"   || DIE "L$L route 不复现, 产物作废"
    done
    LOG "①复现 ✓ (L0/16/32/42 ffn_in+ffn_out+route 逐位一致), 清检查遍"
    rm -rf "${CAP}2"; wd_stop
}

stage_npy(){
    [ -s "$NPY/obase_v3_L42.npy" ] && { LOG "②npy 已在, 跳过"; return 0; }
    mkdir -p "$NPY"
    "$GT/cap_raw2npy" --raw "$CAP" --out "$NPY" --ntok $NTOK || DIE "npy 失败"
    LOG "②npy ✓ $(du -sh "$NPY" | cut -f1)"
}

stage_teacher(){
    [ -s "$NPY/routed_L42.npy" ] && { LOG "③教师已在, 跳过"; return 0; }
    [ -s "$ANC" ] || DIE "干净锚缺 $ANC"
    # ★流对齐闸(2026-08-26 用户质疑后固化)★: 教师(锚行)与学生(捕获行)必须同一条
    # token 流逐行对齐 —— 锚 fin L0 行 vs 捕获 ffn_in L0 行 cos 同行应 ~0.99+,
    # 错位应 <0.5。错位=R 全是垃圾, 解算全废, 必须停车。
    python3 - "$ANC" "$CAP/raw_ffn_in_L0" <<'PY' || DIE "教师/学生流不对齐(锚≠捕获同一 ids), 停车"
import numpy as np, sys
A, C = sys.argv[1], sys.argv[2]
D, S = 4096, 8192
cap = np.fromfile(C, dtype=np.float16)
n = cap.size // D
cap = cap[: n * D].reshape(n, D).astype(np.float32)
af = open(A, "rb")
bad = 0
for i in [0, 100, 1000, 4000, n - 1]:
    af.seek(40 + i * D * 4)
    a = np.frombuffer(af.read(D * 4), dtype=np.float32)
    cos = float(a @ cap[i] / (np.linalg.norm(a) * np.linalg.norm(cap[i]) + 1e-30))
    print("  对齐闸 row %d cos=%.4f" % (i, cos))
    if cos < 0.9: bad = 1
sys.exit(bad)
PY
    wd_start
    LOG "③教师 FP 锚口径(±10 截断产线对齐) 43 层"
    "$GT/teacher_routed" --hf "$HF" --cap "$NPY" --layers 0-42 \
        --ntok $NTOK --threads 20 --swlim 10 --anchor "$ANC" || DIE "教师失败"
    wd_stop
    LOG "③教师 ✓"
}

prep_ws(){
    rm -rf "$WS"; mkdir -p "$WS/layers"
    cd "$D2/vq86h_noz/layers" || DIE "noz 层件缺"
    for f in dql_vq_L*.bin; do ln -f "$f" "$WS/layers/$f" 2>/dev/null || cp "$f" "$WS/layers/"; done
    cp dql_ops_L*.bin opt_L*.bin manifest.txt "$WS/layers/" 2>/dev/null
    cp dql_L*.bin "$WS/layers/"
    cd "$ROOT"
    LOG "工作区就绪(全清重头跑, dql_vq 硬链只读 / dql_L 实拷贝承接注入)"
}

stage_probe(){   # L20 十分钟针: 四损失数字+耗时, 全量前的机制审计
    prep_ws
    LOG "④L20 针发车"
    time "$GT/amp/zloss_solve" --cap "$NPY" --out "$WS" --dql "$WS/layers" \
        --layers 20-20 --ntok $NTOK --nfit $NFIT --ranks "$RANKS" --lambdas "$LAMBDAS" \
        --wa $WA --wc $WC --ws $WSM --wf $WF --threads 20 \
        --fit-ranges "$FR" --ev-ranges "$ER" || DIE "L20 针失败"
    cat "$WS/fourloss_L20.txt"
}

stage_solve(){
    [ -d "$WS/layers" ] || prep_ws
    LOG "⑤全量 43 层解算(单层贪心, 四损失择优)"
    "$GT/amp/zloss_solve" --cap "$NPY" --out "$WS" --dql "$WS/layers" \
        --layers 0-42 --ntok $NTOK --nfit $NFIT --ranks "$RANKS" --lambdas "$LAMBDAS" \
        --wa $WA --wc $WC --ws $WSM --wf $WF --threads 20 \
        --fit-ranges "$FR" --ev-ranges "$ER" || DIE "解算失败"
    LOG "⑤收官: z 文件 $(ls "$WS"/z_L*.ds4z 2>/dev/null | wc -l) 层, 合计 $(du -ch "$WS"/z_L*.ds4z 2>/dev/null | tail -1 | cut -f1)"
}

stage_judge(){
    LOG "⑥五指标终判(caliper, 对表 裸 0.47055 / r64c 冠军 0.42510)"
    bash "$GT/scripts/caliper_ref.sh" "$WS/layers" /tmp/qc_zloss_wt2.bin > /tmp/caliper_zloss.log 2>&1
    grep -E "PPL\(stu|Σmin|Mean KLD|RMS|Same top|Δp" /tmp/caliper_zloss.log
}

stage_engine(){
    LOG "⑦引擎加载 smoke(dql_to_zchain → --zchain)"
    "$GT/amp/dql_to_zchain" "$WS/layers" "$D2/zchain_zloss.bin" 43 || DIE "提取失败"
    cd "$ROOT"; wd_start
    timeout --foreground 900 ./ds4 --cuda -m "$M" --zchain "$D2/zchain_zloss.bin" \
        --temp 0 -n 24 -p "The three most important properties of a distributed cache are" \
        </dev/null 2>&1 | tail -4
    wd_stop
}

ST="${1:-all}"
case "$ST" in
  capture) stage_capture;; npy) stage_npy;; teacher) stage_teacher;;
  probe) stage_probe;; solve) stage_solve;; judge) stage_judge;; engine) stage_engine;;
  all) stage_capture; stage_npy; stage_teacher; stage_probe; stage_solve; stage_judge; stage_engine;;
  *) echo "段: capture npy teacher probe solve judge engine all"; exit 2;;
esac
LOG "段 $ST 完成"
