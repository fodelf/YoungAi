#!/usr/bin/env bash
# algo_pipeline.sh — P2 算法域侧车全流程总控(2026-07-22, 克隆 dsml_pipeline 六步幂等前例)。
# 目标: 治算法域缺口(12针仅1/3真体, Go 三针全败)。产物: gguf/sidecars/algo.gguf → --corr 挂载。
# 用法: algo_pipeline.sh all|capture|ref|solve|mount|verify|status  (all=链式总入口, 唯一合法组链方式)   (语料已就绪: corpus/algo_calib.txt)
# 域差异 vs dsml: 模型=v3(ds4-code1b, 部署同款) / 语料单段(~1450tok) / LAYERS=20-42(深半可映射) /
#                verify=algo_probes 12针 + behavior_gate(侧车不许伤工具帧, runbook 铁律)。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CAPDIR=${CAPDIR:-$ROOT/cap_algo}
REFDIR=${REFDIR:-$ROOT/ref_algo}
LAYERS=${LAYERS:-20-42}
RANK=${RANK:-32}
OUT=${OUT:-$ROOT/gguf/sidecars/algo.gguf}
M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}; DPORT=${DPORT:-5599}
MODEL=gguf/go-onebit/ds4-code1b.gguf
# (env 大扫除 2026-08-31: EXPERT_OFFLOAD=1 归 AUTO 按 --mem-budget-mb 判定; GATHER_THREADS
#  写死 8; NO_MODEL_WARMUP 已删; PREFETCH_AHEAD=0 不再需要 — 捕获仪器武装时引擎自动关预取)
FLAGSTR="--residual gguf/sidecars/code-hot-res-v3p.gguf --reverse-connect --prefill-chunk 2048 --mem-budget-mb 12000"
COORD_FLAGS="--dist-prefill-cap 2048"   # 只 coordinator 侧认
log(){ echo "[algo-pipeline] $*"; }

case "${1:-status}" in
capture)
  log "双机批捕获(单段 ~1450tok, v3 模型, L$LAYERS 全在 worker)。前置: svc.sh down"
  pgrep -f '^\./ds4-server' >/dev/null && { log "server 在跑 — 先 tools/svc.sh down"; exit 1; }
  mkdir -p "$CAPDIR"
  python3 - "$ROOT/gguf-tools/data/corpus/algo_calib.txt" "$CAPDIR" <<'PY'
import sys
text = open(sys.argv[1]).read()
open(f"{sys.argv[2]}/seg0.txt", "w").write("<｜begin▁of▁sentence｜>" + text)
print("[algo-pipeline] seg0:", len(text), "字节")
PY
  ssh "$M1" "pkill -f 'role worker'; mkdir -p /tmp/cap_algo2 && rm -f /tmp/cap_algo2/raw_* /tmp/ds4_worker_cap.log" 2>/dev/null; sleep 2
  # 层过滤 env(原 DS4_CAP_LAYERS)已死: worker 只算 20:output 那片, 捕获天然只落本片层
  ssh "$M1" "( cd $M1DIR && nohup ./ds4 -m $MODEL $FLAGSTR --cap-dir /tmp/cap_algo2 --role worker --listen $M1 $DPORT --layers 20:output -c 4096 --temp 0 --nothink ) > /tmp/ds4_worker_cap.log 2>&1 < /dev/null & echo ok"
  until ssh "$M1" "grep -q 'waiting for coordinator' /tmp/ds4_worker_cap.log" 2>/dev/null; do sleep 3; done
  ( sleep 15
    while true; do
      cp=$(pgrep -f "role coordinator" | head -1); [ -z "$cp" ] && exit 0
      kb=$(ps -o rss= -p "$cp" 2>/dev/null | tr -d ' '); g=$(( ${kb:-0} / 1048576 ))
      rkb=$(ssh "$M1" "pgrep -f 'role worker' | head -1 | xargs -I{} ps -o rss= -p {}" 2>/dev/null | tr -d ' '); rg=$(( ${rkb:-0} / 1048576 ))
      if [ "$g" -gt 12 ] || [ "$rg" -gt 12 ]; then echo "[algo-pipeline] MEM-BREACH L=${g}G R=${rg}G 同杀"; kill "$cp"; ssh "$M1" "pkill -f 'role worker'"; exit 1; fi
      sleep 20
    done ) & WD=$!
  ( cd "$ROOT" && ./ds4 -m "$MODEL" $FLAGSTR $COORD_FLAGS --role coordinator --coordinator "$M1" "$DPORT" \
      --layers 0:19 -c 4096 -n 1 --temp 0 --nothink \
      -p "$(cat "$CAPDIR/seg0.txt")" > /tmp/algo_cap_seg0.out 2> /tmp/algo_cap_seg0.log )
  kill "$WD" 2>/dev/null
  ssh "$M1" "pkill -f 'role worker'" 2>/dev/null; sleep 3
  mkdir -p "$CAPDIR/seg0"
  rsync -a --remove-source-files "$M1:/tmp/cap_algo2/raw_*" "$CAPDIR/seg0/" || { log "rsync 失败, 数据在 M1:/tmp/cap_algo2"; exit 1; }
  ntok=$(( $(stat -f%z "$CAPDIR/seg0/raw_ffn_in_L${LAYERS%-*}" 2>/dev/null || echo 0) / 8192 ))
  log "捕获收割: $(du -sh "$CAPDIR/seg0" | cut -f1), ${ntok} tokens/层"
  [ "$ntok" -gt 800 ] || { log "捕获异常(tokens=$ntok), 停"; exit 1; }
  ;;
ref)
  # O_REF 段原依赖已删 Python 前向(dsml_oref.py + calib/pyfwd 的 dsv4_fwd/ds4reader,
  # 全仓 Python 清零删除, 见 git 历史), 无 C 承接 — 响亮失败不静默。
  log "★ref 段依赖已删 Python 前向(见 git 历史), 待 C 承接★"; exit 1
  ;;
solve)
  [ -x "$ROOT/zsolve" ] || make -C "$ROOT" zsolve
  mkdir -p "$(dirname "$OUT")"
  python3 - "$CAPDIR" "$REFDIR" "$LAYERS" "$RANK" "$OUT" "$ROOT" <<'PY'
import sys, os, subprocess
import numpy as np
cap, ref, layers, rank, out, root = sys.argv[1:7]
lo, hi = map(int, layers.split('-'))
D = 4096
os.makedirs("/tmp/algo_solve", exist_ok=True)
lays = []
for L in range(lo, hi + 1):
    fx = f"{cap}/seg0/raw_ffn_in_L{L}"; fb = f"{cap}/seg0/raw_ffn_out_L{L}"; fr = f"{ref}/ffn_out_L{L}"
    if not (os.path.isfile(fx) and os.path.isfile(fb) and os.path.isfile(fr)):
        print(f"[solve] L{L} 数据不全, 跳过"); continue
    X = np.fromfile(fx, dtype='<f2').reshape(-1, D).astype(np.float32)
    OB = np.fromfile(fb, dtype='<f2').reshape(-1, D).astype(np.float32)
    OR = np.fromfile(fr, dtype='<f2').reshape(-1, D).astype(np.float32)
    n = min(len(X), len(OB), len(OR))
    R = OR[:n] - OB[:n]
    # zsolve 要求 float32 C-order .npy(tools/zsolve.c 头注), 非裸 f16 字节
    np.save(f"/tmp/algo_solve/x_L{L}.npy", X[:n].astype('<f4'))
    np.save(f"/tmp/algo_solve/r_L{L}.npy", R.astype('<f4'))
    lays.append(L)
    print(f"[solve] L{L} n={n} |R|={np.abs(R).mean():.4f}")
print("[solve] 层就绪:", lays)
# zsolve 真实 CLI(tools/zsolve.c): --out P --rank N + 每层一组 --layer L x文件 r文件
cmd = [f"{root}/zsolve", "--out", out, "--rank", rank]
for L in lays:
    cmd += ["--layer", str(L), f"/tmp/algo_solve/x_L{L}.npy", f"/tmp/algo_solve/r_L{L}.npy"]
subprocess.check_call(cmd)
PY
  [ -s "$OUT" ] || { log "solve 未产出侧车 — 停"; exit 1; }
  log "solve 完成: $OUT"
  ;;
mount)
  [ -s "$OUT" ] || { log "侧车 $OUT 不存在 — 先 solve"; exit 1; }
  log "挂载: CORR=$OUT 双机重启(侧车先 scp M1 同名相对路径)"
  scp -q "$OUT" "$M1:$M1DIR/$(basename "$OUT")" || { log "scp 失败"; exit 1; }
  "$ROOT/tools/svc.sh" down >/dev/null 2>&1; sleep 3
  ( CORR="$OUT" "$ROOT/tools/svc.sh" up > /tmp/svc_up_algo.log 2>&1 & )
  until grep -qa listening /tmp/ds4-svc.log 2>/dev/null && pgrep -f '^\./ds4-server' >/dev/null; do sleep 5; done
  grep -a "corr" /tmp/ds4-svc.log | head -2
  ;;
verify)
  log "verify: algo 12针 A/B + 行为门(侧车不许伤工具帧铁律)"
  cd "$HERE"
  CORPUS=corpus/algo_probes.txt NPRED=28 PORT=8013 ./pillar_probe_srv.sh
  cp /tmp/algo_probes_srv.report "$ROOT/gguf-tools/reports/algo_probes_sidecar_$(date +%F).report"
  rm -f /tmp/behavior_gate_at.report
  MAXTOK=192 ./behavior_gate_at.sh
  cp /tmp/behavior_gate_at.report "$ROOT/gguf-tools/reports/behavior_gate_sidecar_$(date +%F).report"
  log "对照腿: reports/algo_probes_v3_stack_2026-07-22.report + behavior_gate_v3_stack_mt192_2026-07-22.report"
  ;;
all)
  # ★链式总入口(2026-07-22 用户令: 链条进脚本, 不许在会话上下文里内联组链)★
  # 幂等: 各步自查已完成产物; FORCE=1 清 capture/侧车重来。守卫: 任一步失败即停。
  # ⚠ mount 前置铁律: 侧车判决为毒(2026-07-22 首版)后, 重挂载必须人工确认
  #   DS4_CORR_VERDICT_OK=1(=声明毒根因已修并有依据), 否则 all 停在 solve 后。
  [ "${FORCE:-0}" = 1 ] && { rm -rf "$CAPDIR/seg0"; rm -f "$OUT"; log "FORCE: capture/侧车已清"; }
  [ -f "$CAPDIR/seg0/raw_ffn_in_L${LAYERS%-*}" ] || "$0" capture || exit 1
  "$0" ref || exit 1
  [ -s "$OUT" ] || "$0" solve || exit 1
  if [ "${DS4_CORR_VERDICT_OK:-0}" = 1 ]; then
    "$0" mount || exit 1
    "$0" verify || exit 1
  else
    log "all 停在 solve 后(毒判在案): 挂载需 DS4_CORR_VERDICT_OK=1(根因修复+依据入档后)"
  fi
  ;;
status)
  log "capture: $(ls "$CAPDIR/seg0" 2>/dev/null | wc -l | tr -d ' ') 文件; ref: $(ls "$REFDIR" 2>/dev/null | wc -l | tr -d ' ')/23 层; 侧车: $(ls -la "$OUT" 2>/dev/null | awk '{print $5}' || echo 无)"
  ;;
*) log "用法: all|capture|ref|solve|mount|verify|status"; exit 2 ;;
esac
