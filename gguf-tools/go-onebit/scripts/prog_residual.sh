#!/usr/bin/env bash
# prog_residual.sh — 全编程场景残差侧车重造(2026-07-22 用户令: 删旧时代产物, 按新域重生成)。
# 链: corpus(prog宽语料拼接) → capture(43层路由, drain修复后干净数据) → hotlist(prog路由
#     top-K, 替代 Go-hot) → emit(M1 全层单机 lane, M4 HF 已损坏不参与) → mount → verify(43针)。
# 产物: gguf/sidecars/code-hot-res-prog.gguf (两机)。
# 用法: prog_residual.sh all|corpus|capture|hotlist|emit|mount|verify|status
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
M1=${M1:-192.168.1.2}; M1DIR=${M1DIR:-/Users/fodelf/ds4-main}
K=${K:-64}
CAP=${CAP:-capprog}
OUT=${OUT:-$ROOT/gguf/sidecars/code-hot-res-prog.gguf}
ACTIVE=${ACTIVE:-$ROOT/gguf-tools/go-onebit/corpus/prog_active_top$K.txt}
log(){ echo "[prog-residual] $*"; }

case "${1:-status}" in
corpus)
  # v2(2026-07-23): 补回落域(web/文件IO/shell/SQL)语料 — 热表覆盖面与目标负载对齐铁律
  cat "$ROOT/gguf-tools/go-onebit/corpus/calib_prog_v1.txt" \
      "$ROOT/gguf-tools/go-onebit/corpus/algo_calib.txt" \
      "$ROOT/gguf-tools/go-onebit/corpus/hot_extra_v2.txt" \
      "$ROOT/gguf-tools/go-onebit/corpus/hot_extra_v3.txt" \
      > "$ROOT/gguf-tools/go-onebit/corpus/prog_hot_calib.txt"
  wc -c "$ROOT/gguf-tools/go-onebit/corpus/prog_hot_calib.txt"
  ;;
capture)
  pgrep -f '^\./ds4-server' >/dev/null && { log "server 在跑 — 先 tools/svc.sh down"; exit 1; }
  PROMPT_FILE="$ROOT/gguf-tools/go-onebit/corpus/prog_hot_calib.txt" \
    MODEL=gguf/go-onebit/ds4-code1b.gguf CAP="$CAP" \
    "$HERE/capture_alllayers.sh" || exit 1
  ;;
hotlist)
  # raw_route(i16 n×6) → npy → 点火计数 top-K。capture_alllayers 产物在 M1:/tmp/$CAP。
  ssh "$M1" "cd /tmp/$CAP && python3 - <<'PY'
import numpy as np, glob
for f in glob.glob('raw_route_L*'):
    L = f.split('_L')[1]
    ids = np.fromfile(f, dtype='<i2').astype(np.int64)
    np.save(f'route_L{L}.npy', ids)
print('npy done')
PY" || exit 1
  rsync -a "$M1:/tmp/$CAP/route_L*.npy" /tmp/${CAP}_routes/ || exit 1
  python3 "$HERE/gen_active_topk.py" /tmp/${CAP}_routes "$K" "$ACTIVE" || exit 1
  head -2 "$ACTIVE"
  ;;
emit)
  [ -s "$ACTIVE" ] || { log "热表 $ACTIVE 缺 — 先 hotlist"; exit 1; }
  # M1 全层单机 lane(46 shard 全在 M1; M4 hf 损坏铁律不参与)。~数小时, nohup 自持。
  scp -q "$ROOT/gguf-tools/go-onebit/emit_residual_stream" "$M1:$M1DIR/gguf-tools/go-onebit/emit_residual" 2>/dev/null || true   # 2026-07-23 流式版(峰值盘受控)
  scp -q "$ACTIVE" "$M1:/tmp/prog_active.txt" || exit 1
  ssh "$M1" "cd $M1DIR/gguf-tools/go-onebit && rm -f /tmp/prog_residual_emit.log && ( nohup ./emit_residual --hf $M1DIR/hf/DeepSeek-V4-Flash-Base --out /tmp/code-hot-res-prog.gguf --layers \$(seq -s, 0 42) --active-experts /tmp/prog_active.txt > /tmp/prog_residual_emit.log 2>&1 & ) && echo 'emit_started'"
  log "进度: ssh $M1 tail -f /tmp/prog_residual_emit.log; 完成后跑 collect"
  ;;
collect)
  scp "$M1:/tmp/code-hot-res-prog.gguf" "$OUT" || exit 1
  scp -q "$OUT" "$M1:$M1DIR/gguf/sidecars/$(basename "$OUT")" || exit 1
  ssh "$M1" "rm -f /tmp/code-hot-res-prog.gguf"
  ls -la "$OUT"
  ;;
mount)
  [ -s "$OUT" ] || { log "侧车缺 — 先 collect"; exit 1; }
  "$ROOT/tools/svc.sh" down >/dev/null 2>&1; sleep 3
  pkill -9 -f "^\./ds4-server" 2>/dev/null; sleep 2   # 幂等up陷阱: down后确保server真死才起(2026-07-23事故)
  ( RESID="gguf/sidecars/$(basename "$OUT")" "$ROOT/tools/svc.sh" up > /tmp/svc_up_progres.log 2>&1 & )
  until grep -qa listening /tmp/ds4-svc.log 2>/dev/null && pgrep -f '^\./ds4-server' >/dev/null; do sleep 5; done
  grep -a "residual loaded" /tmp/ds4-svc.log | head -1
  ;;
verify)
  TAG="${TAG:-progres}" "$HERE/prog_sweep.sh"   # TAG 透传(2026-07-23 覆盖事故修复)
  ;;
all)
  "$0" corpus && "$0" capture && "$0" hotlist && "$0" emit
  log "emit 在 M1 后台(数小时); 完成后: $0 collect && $0 mount && $0 verify"
  ;;
finish)
  # emit 完成接力(2026-07-23 入脚本): 等 M1 emit 收官 → collect→mount→verify 全自动
  until ! ssh "$M1" "pgrep -f emit_residual >/dev/null" 2>/dev/null \
        && ssh "$M1" "[ -s /tmp/code-hot-res-prog.gguf ]" 2>/dev/null; do sleep 300; done
  "$0" collect && "$0" mount && "$0" verify
  log "★finish 链完成★"
  ;;
status)
  ssh "$M1" "tail -2 /tmp/prog_residual_emit.log 2>/dev/null; ls -la /tmp/code-hot-res-prog.gguf 2>/dev/null | awk '{print \$5}'"
  ;;
*) log "用法: all|corpus|capture|hotlist|emit|collect|mount|verify|status"; exit 2 ;;
esac
