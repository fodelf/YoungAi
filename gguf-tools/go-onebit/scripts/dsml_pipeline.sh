#!/usr/bin/env bash
# dsml_pipeline.sh — P2 首个域侧车(DSML 工具域)全流程总控。六步幂等, 每步独立可跑。
#
# 目标能力 (Gate v2 判决): 长上下文里检索用户真实值填入工具参数。
# 产物: gguf/sidecars/dsml.gguf → 常驻服务 --corr 挂载 → tool-call 探针复验。
#
# 用法: dsml_pipeline.sh corpus|capture|ref|solve|mount|verify|status
#   corpus  秒级   生成校准语料 (dsml_corpus_gen.py, 已可跑)
#   capture 耗时★  mono 前向语料 prompts, 捕获每层 X=ffn_in + O_BASE=ffn_out
#   ref     耗时★  HF 原始模型参考前向 (pyfwd 双机 shard 铁律), 产 O_REF=ffn_out
#   solve   分钟级 R=O_REF−O_BASE → ./zsolve → gguf/sidecars/dsml.gguf
#   mount   秒级   常驻服务加 --corr 重启 (tools/svc.sh 环境)
#   verify  分钟级 tool-call 探针 (长上下文变体) + twoSum 回归判据
#
# ★ 前置检查 (capture 首次执行时必须确认):
#   1. 引擎 DS4_CAP_DIR 机制目前只抓 raw_ffn_in_L* (X)。R 需要 ffn_out ——
#      若 DS4_CAP 无 ffn_out 键, 先在 ds4.c capture 点补一行 (搜 DS4_CAP_DIR)。
#   2. capture 跑 mono 需独占实例锁: 先 tools/svc.sh down。
#   3. 内存: 单机 mono 前向 + 捕获缓冲, DS4_MEM_BUDGET_MB=12000 + 看门狗
#      (memory 铁律: capture OOM 前科, 关 prefetch + 低预算 + 先小跑测有界)。
#   4. ref 步 HF shards: 双机各跑本机 shard (dual_host_hf_original_calibration 铁律);
#      hf 目录若不在, 经 NFS (memory: hf 经 NFS 可读)。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CORPUS=$HERE/../corpus/dsml/gen_v1
CAPDIR=${CAPDIR:-$ROOT/cap_dsml}
REFDIR=${REFDIR:-$ROOT/ref_dsml}
LAYERS=${LAYERS:-20-30}              # v0 先做中段 10 层 (mapnet: 中层数据饥饿收益最大)
RANK=${RANK:-32}
OUT=${OUT:-$ROOT/gguf/sidecars/dsml.gguf}
log(){ echo "[dsml-pipeline] $*"; }

case "${1:-status}" in
corpus)
  python3 "$HERE/dsml_corpus_gen.py" "${2:-200}"
  python3 "$HERE/dsml_corpus_gen.py" "${3:-50}" --ctx-pad=8000 || true   # 长上下文变体
  log "语料就绪: $CORPUS"
  ;;
capture)
  # 双机批捕获 (~30min)。单机 teacher-forced 逐 token 路线实测 0.3 t/s (每 token×层×4
  # 次 GPU 同步读) = 55h 不可行; 且单机 mono 持续跑有内核 panic 前科 (memory 铁律:
  # 安全做法=双机层切片)。改为: 语料切 4 段, 每段 BOS 裸 prefill 走双机 (batch 路径
  # capture 每 chunk 才读一次 GPU), L20-30 全在 worker → CAP env 只配 M1 侧。
  # M1 /tmp 只剩 ~7G, 捕获全量 ~11G → 段间收割: 杀 worker(句柄缓存 exit 才 flush) →
  # rsync 回本机 → 清 M1 段文件(仅本脚本刚生成的 /tmp/cap_dsml, 已 rsync 校验后删)。
  log "★双机批捕获 (~30min, 4 段循环: worker 起→prefill→杀→收割)。前置: svc.sh down。"
  [ -d "$CORPUS" ] || { log "先跑 corpus"; exit 1; }
  M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}; DPORT=${DPORT:-5599}
  MODEL=$ROOT/gguf/ds4-mono-mixed.gguf
  ENVSTR="DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=2048 DS4_DIST_PREFILL_CAP=2048 DS4_METAL_EXPERT_GATHER_THREADS=8 DS4_METAL_NO_MODEL_WARMUP=1 DS4_MEM_BUDGET_MB=12000 DS4_METAL_EXPERT_PREFETCH_AHEAD=0"
  mkdir -p "$CAPDIR"
  ALL=$CAPDIR/dsml_all.txt
  : > "$ALL"
  while IFS= read -r f; do
    sed -n '/=== PROMPT ===/,$p' "$f" | sed '1d' | sed 's/^=== GOLD ===$//' >> "$ALL"
    printf '\n\n' >> "$ALL"
  done < "$CORPUS/prompts.txt"
  # 切 4 段 + BOS 前缀 (CLI is_rendered_chat_prompt: BOS 开头=裸渲染不包 chat 模板;
  # ref 步 HF 前向对同一段文件同法 tokenize → token 序对齐)
  python3 - "$ALL" "$CAPDIR" <<'PY'
import sys
text = open(sys.argv[1]).read()
seg = len(text) // 4 + 1
for i in range(4):
    open(f"{sys.argv[2]}/seg{i}.txt", "w").write("<｜begin▁of▁sentence｜>" + text[i*seg:(i+1)*seg])
print("[dsml-pipeline] 4 段就绪:", [len(open(f"{sys.argv[2]}/seg{i}.txt","rb").read()) for i in range(4)], "字节")
PY
  # 双机共享 CORE_OBJS: raw_ffn_out 捕获是新代码, worker 二进制必须同步 (铁律)
  if [ "$(shasum "$ROOT/ds4" | cut -d' ' -f1)" != "$(ssh "$M1" "shasum $M1DIR/ds4" | cut -d' ' -f1)" ]; then
    ssh "$M1" "pkill -f 'role worker'" 2>/dev/null; sleep 2
    scp "$ROOT/ds4" "$M1:$M1DIR/ds4" || { log "二进制同步失败"; exit 1; }
    log "worker 二进制已同步 (raw_ffn_out 捕获代码)"
  fi
  for i in 0 1 2 3; do
    if [ -f "$CAPDIR/seg$i/raw_ffn_in_L${LAYERS%-*}" ]; then log "seg$i 已捕获, 跳过"; continue; fi
    log "── seg$i: 起 worker(CAP env) → prefill → 收割 ──"
    ssh "$M1" "pkill -f 'role worker'; mkdir -p /tmp/cap_dsml && rm -f /tmp/cap_dsml/raw_* /tmp/ds4_worker_cap.log" 2>/dev/null; sleep 2
    # 整个后台列表包 ( ) 并重定向全部 fd: '&' 优先级低于 '&&', 不包的话中间子壳
    # (ds4 的父进程) 持有 sshd 的 stdout/stderr 管道 → ssh 到 ds4 退出才返回 (实测坑)
    ssh "$M1" "( cd $M1DIR && $ENVSTR DS4_CAP_DIR=/tmp/cap_dsml DS4_CAP_LAYERS=$LAYERS nohup ./ds4 -m gguf/ds4-mono-mixed.gguf --role worker --listen $M1 $DPORT --layers 20:output -c 18432 --temp 0 --nothink ) > /tmp/ds4_worker_cap.log 2>&1 < /dev/null & echo ok"
    until ssh "$M1" "grep -q 'waiting for coordinator' /tmp/ds4_worker_cap.log" 2>/dev/null; do sleep 3; done
    # 双机 12G 看门狗 (coordinator 运行期间; 越线两边同杀)
    ( sleep 15   # 宽限: coordinator 在本子壳之后才起, 首查太早会误判已退出
      while true; do
        cp=$(pgrep -f "role coordinator" | head -1); [ -z "$cp" ] && exit 0
        kb=$(ps -o rss= -p "$cp" 2>/dev/null | tr -d ' '); g=$(( ${kb:-0} / 1048576 ))
        rkb=$(ssh "$M1" "pgrep -f 'role worker' | head -1 | xargs -I{} ps -o rss= -p {}" 2>/dev/null | tr -d ' '); rg=$(( ${rkb:-0} / 1048576 ))
        if [ "$g" -gt 12 ] || [ "$rg" -gt 12 ]; then echo "[dsml-pipeline] MEM-BREACH L=${g}G R=${rg}G 同杀"; kill "$cp"; ssh "$M1" "pkill -f 'role worker'"; exit 1; fi
        sleep 20
      done ) & WD=$!
    ( cd "$ROOT" && env $ENVSTR ./ds4 -m "$MODEL" --role coordinator --coordinator "$M1" "$DPORT" \
        --layers 0:19 -c 18432 -n 1 --temp 0 --nothink \
        -p "$(cat "$CAPDIR/seg$i.txt")" > /tmp/dsml_cap_seg$i.out 2> /tmp/dsml_cap_seg$i.log )
    kill "$WD" 2>/dev/null
    ssh "$M1" "pkill -f 'role worker'" 2>/dev/null; sleep 3   # exit flush 句柄
    mkdir -p "$CAPDIR/seg$i"
    rsync -a --remove-source-files "$M1:/tmp/cap_dsml/raw_*" "$CAPDIR/seg$i/" || { log "seg$i rsync 失败, M1 数据保留在 /tmp/cap_dsml"; exit 1; }
    ntok=$(( $(stat -f%z "$CAPDIR/seg$i/raw_ffn_in_L${LAYERS%-*}" 2>/dev/null || echo 0) / 8192 ))
    log "seg$i 收割: $(du -sh "$CAPDIR/seg$i" | cut -f1), ${ntok} tokens/层"
    [ "$ntok" -gt 1000 ] || { log "seg$i 捕获异常 (tokens=$ntok), 停"; exit 1; }
  done
  log "capture 完成: $CAPDIR/seg{0..3}"
  ;;
ref)
  # O_REF = 引擎捕获 x̂ + 引擎路由 (ids/gate w) + HF 原始 fp8 专家权重重算 FFN
  # (dsml_oref.py 头注: error-feedback 语义, 不需要完整 HF 前向)。
  # HF 46 shard 全在 M1 (/Users/fodelf/ds4-main/hf), L20-30≈shard22-32 → 计算跟数据
  # 走, ref 整段在 M1 跑 (dual_host_hf_original_calibration 铁律: 各跑本机 shard;
  # M4 本地 hf 已损坏清空)。数据流按层: 拼 4 段 x̂/route → scp M1 → 重算 → 拉回,
  # M1 峰值 ~1.5GB << 7GB 剩余盘。预计 ~2min/层 × 11 层。
  M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
  M1HF=${M1HF:-$M1DIR/hf/DeepSeek-V4-Flash-Base}
  PYFWD=$ROOT/gguf-tools/go-onebit/calib/pyfwd
  mkdir -p "$REFDIR"
  ssh "$M1" "mkdir -p /tmp/oref /tmp/oref/pyfwd"
  scp -q "$HERE/dsml_oref.py" "$M1:/tmp/oref/"
  scp -q "$PYFWD/dsv4_fwd.py" "$PYFWD/ds4reader.py" "$M1:/tmp/oref/pyfwd/"
  for L in $(seq "${LAYERS%-*}" "${LAYERS#*-}"); do
    [ -f "$REFDIR/ffn_out_L$L" ] && { log "L$L 已有 O_REF, 跳过"; continue; }
    for k in raw_ffn_in raw_route raw_route_w; do
      cat "$CAPDIR"/seg0/${k}_L$L "$CAPDIR"/seg1/${k}_L$L "$CAPDIR"/seg2/${k}_L$L "$CAPDIR"/seg3/${k}_L$L > "/tmp/oref_${k}_L$L" || { log "L$L 捕获不全"; exit 1; }
    done
    scp -q /tmp/oref_raw_ffn_in_L$L "$M1:/tmp/oref/raw_ffn_in_L$L"
    scp -q /tmp/oref_raw_route_L$L "$M1:/tmp/oref/raw_route_L$L"
    scp -q /tmp/oref_raw_route_w_L$L "$M1:/tmp/oref/raw_route_w_L$L"
    rm -f /tmp/oref_raw_*_L$L
    ssh "$M1" "cd /tmp/oref && DS4_HF=$M1HF python3 dsml_oref.py --cap /tmp/oref --layer $L --out /tmp/oref/o_ref_L$L" || { log "L$L oref 失败"; exit 1; }
    scp -q "$M1:/tmp/oref/o_ref_L$L" "$REFDIR/ffn_out_L$L"
    ssh "$M1" "rm -f /tmp/oref/raw_*_L$L /tmp/oref/o_ref_L$L"   # 只清本步生成的临时件
    log "L$L O_REF 就绪 ($(du -sh "$REFDIR/ffn_out_L$L" | cut -f1))"
  done
  log "ref 完成: $REFDIR"
  ;;
solve)
  [ -x "$ROOT/zsolve" ] || make -C "$ROOT" zsolve
  mkdir -p "$(dirname "$OUT")"
  # R = O_REF − O_BASE 逐层拼 npy (输入是引擎 raw f16: capture=seg{0..3}/ 分段,
  # ref=按层单文件; 三者按行天然对齐 —— O_REF 就是对同一批 x̂ 行重算的)。
  python3 - "$CAPDIR" "$REFDIR" "$LAYERS" <<'PY'
import sys, os
import numpy as np
cap, ref, layers = sys.argv[1], sys.argv[2], sys.argv[3]
lo, hi = map(int, layers.split('-'))
D = 4096
def segcat(name, L):
    ps = [f"{cap}/seg{i}/{name}_L{L}" for i in range(4)]
    if not all(os.path.isfile(p) for p in ps): return None
    return np.concatenate([np.fromfile(p, dtype='<f2').reshape(-1, D) for p in ps])
for L in range(lo, hi + 1):
    X = segcat("raw_ffn_in", L)
    OB = segcat("raw_ffn_out", L)
    orf_p = f"{ref}/ffn_out_L{L}"
    if X is None or OB is None or not os.path.isfile(orf_p):
        print(f"L{L}: 缺数据 (X:{X is not None} O_BASE:{OB is not None} O_REF:{os.path.isfile(orf_p)}) — 跳过"); continue
    OR = np.fromfile(orf_p, dtype='<f2').reshape(-1, D)
    assert len(X) == len(OB) == len(OR), f"L{L} 行数不齐 {len(X)}/{len(OB)}/{len(OR)}"
    # 丢 O_BASE 全零行: 首 chunk 在入口层 (L20) 读到未计算缓冲 (实测每段恰好
    # 前 2048 行), R=O_REF-0=整份 FFN 输出会教坏解 (运行时输出翻倍→乱码)。
    ok = np.abs(OB).max(axis=1) > 0
    if (~ok).any(): print(f"L{L}: 丢弃 {int((~ok).sum())} 行 O_BASE 零行")
    X, OB, OR = X[ok], OB[ok], OR[ok]
    np.save(f"/tmp/dsml_X_L{L}.npy", X.astype('<f4'))
    np.save(f"/tmp/dsml_R_L{L}.npy", (OR.astype(np.float32) - OB.astype(np.float32)).astype('<f4'))
    print(f"L{L}: n={len(X)} 就绪 (R rms={np.sqrt(((OR.astype(np.float32)-OB.astype(np.float32))**2).mean()):.4g})")
PY
  ARGS=""
  for L in $(seq "${LAYERS%-*}" "${LAYERS#*-}"); do
    [ -f "/tmp/dsml_X_L$L.npy" ] && ARGS="$ARGS --layer $L /tmp/dsml_X_L$L.npy /tmp/dsml_R_L$L.npy"
  done
  [ -n "$ARGS" ] || { log "无可解层"; exit 1; }
  "$ROOT/zsolve" --out "$OUT" --rank "$RANK" --lambda 1e-3 $ARGS
  log "侧车写出: $OUT"
  ;;
mount)
  # corr 校正的 L20-30 跑在 worker → 侧车两端都要 (svc.sh 已接 CORR env, 双端传)。
  [ -f "$OUT" ] || { log "侧车不存在: $OUT (先跑 solve)"; exit 1; }
  M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
  scp -q "$OUT" "$M1:$M1DIR/$(basename "$OUT")" || { log "侧车同步 M1 失败"; exit 1; }
  "$ROOT/tools/svc.sh" down; sleep 2
  CORR="$OUT" "$ROOT/tools/svc.sh" up
  log "服务已带侧车重启 (两端 --corr)。基线回退: svc.sh down && svc.sh up (不带 CORR)"
  ;;
verify)
  # A/B: `verify baseline` (服务不带 CORR 时) → mount → `verify sidecar`。
  # 探针 held-out (dsml_verify.py 头注); 两份齐后自动打印 A/B 判决表。
  # 回归门: A/B 后另跑 CORR=$OUT tools/mtp_pipe_q2_speed.sh (twoSum 裸续写不得退化)。
  LABEL=${2:-baseline}
  python3 "$HERE/dsml_verify.py" --label "$LABEL" --n "${N:-20}" 2>&1 | tee /tmp/dsml_verify_$LABEL.log
  ;;
status)
  log "语料: $(ls "$CORPUS" 2>/dev/null | wc -l | tr -d ' ') 条 (gen_v1)"
  log "capture: $(ls "$CAPDIR" 2>/dev/null | wc -l | tr -d ' ') 项 → $CAPDIR"
  log "ref: $(ls "$REFDIR" 2>/dev/null | wc -l | tr -d ' ') 项 → $REFDIR"
  log "侧车: $([ -f "$OUT" ] && echo 已生成 || echo 未生成) → $OUT"
  ;;
*) echo "用法: dsml_pipeline.sh corpus|capture|ref|solve|mount|verify|status"; exit 2 ;;
esac
