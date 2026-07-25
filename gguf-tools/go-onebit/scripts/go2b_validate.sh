#!/usr/bin/env bash
# go2b_validate.sh — go2b 漂移方案生成质量验证链(2026-07-24, 用户令: 脚本化不私改)。
# 链: build(激活最优 go2b 热overlay) → 挂载 → 算法针 → 对比现役 go1b残差。
# 全阶段脚本化, 会话不再私下拼命令。用法:
#   go2b_validate.sh build [L起-L止]     # 建 go2b overlay(默认 20-42, 需 cap_algo 激活)
#   go2b_validate.sh probe               # 挂 go2b_deep 跑算法12针
#   go2b_validate.sh baseline            # 挂现役 go1b残差 跑算法12针(对照)
#   go2b_validate.sh compare             # 两报告并排 md5+逐针
#   go2b_validate.sh all [L起-L止]       # build→probe→baseline→compare 全自动
# 铁律: 激活来自 cap_algo/seg0(仅 L20-42); 挂载走干净 down+pkill-9(svc幂等陷阱); 输出到数据卷。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
M1=${M1:-192.168.1.2}
GO2B=$ROOT/gguf/sidecars/go2b_deep.gguf
RESID_CUR=gguf/sidecars/code-hot-res-prog.gguf
ACTIVE=/tmp/prog_active.txt
CAP=$ROOT/cap_algo43/seg0   # 全43层激活
RPT=$ROOT/gguf-tools/go-onebit/reports
KN=gguf-tools/go-onebit/corpus/knowledge.txt
log(){ echo "[go2b-val] $*" >&2; }

remount() {  # $1=RESID相对路径 (干净 down+pkill 防 svc 幂等陷阱)
  "$ROOT/tools/svc.sh" down >/dev/null 2>&1; sleep 3
  pkill -9 -f "^\./ds4-server" 2>/dev/null; sleep 2
  ( DS4_KNOWLEDGE_FILE="$KN" RESID="$1" "$ROOT/tools/svc.sh" up >/tmp/svc_go2bval.log 2>&1 & )
  until grep -qa listening /tmp/ds4-svc.log 2>/dev/null && pgrep -f '^\./ds4-server' >/dev/null; do sleep 5; done
  grep -aE "residual loaded|go2b|correction" /tmp/ds4-svc.log | tail -2 >&2
}

case "${1:-all}" in
build)
  R="${2:-0-42}"; LO=${R%-*}; HI=${R#*-}
  cp "$ROOT/gguf-tools/go-onebit/corpus/prog_active_top64.txt" "$ACTIVE"
  LAYERS=$(python3 -c "print(','.join(str(x) for x in range($LO,$HI+1)))")
  mkdir -p "$CAP"; rsync -a "$M1:/tmp/cap_algo43/raw_ffn_in_L*" "$CAP/" 2>/dev/null   # 全43层激活从M1
  [ -f "$CAP/raw_ffn_in_L$LO" ] || { log "激活 $CAP/raw_ffn_in_L$LO 缺 — 拒建"; exit 2; }
  log "build go2b L$LO-$HI (激活最优 DS4_GO2B_ACT_SCALE=1) → $GO2B"
  rm -f "$GO2B" /tmp/sc_L*
  DS4_GO2B_ACT_SCALE=1 python3 "$ROOT/gguf-tools/go-onebit/quant/build_go2b_hot.py" "$GO2B" "$LAYERS" "$ACTIVE" "$CAP"
  ls -la "$GO2B" | awk '{print "[go2b-val] 产物:", $5}' >&2
  ;;
probe)
  [ -s "$GO2B" ] || { log "go2b_deep 缺 — 先 build"; exit 2; }
  scp -q "$GO2B" "$M1:/Users/fodelf/ds4-main/gguf/sidecars/$(basename "$GO2B")" 2>/dev/null || true
  remount "gguf/sidecars/$(basename "$GO2B")"
  cd "$HERE"; for C in prog_probes algo_probes; do CORPUS=corpus/$C.txt NPRED=28 PORT=8013 ./pillar_probe_srv.sh; cp /tmp/${C}_srv.report "$RPT/${C}_go2b_$(date +%F).report"; done
  log "go2b prog+algo针 → reports/*_go2b_$(date +%F).report"
  ;;
baseline)
  remount "$RESID_CUR"
  cd "$HERE"; for C in prog_probes algo_probes; do CORPUS=corpus/$C.txt NPRED=28 PORT=8013 ./pillar_probe_srv.sh; cp /tmp/${C}_srv.report "$RPT/${C}_baseline_$(date +%F).report"; done
  log "现役 prog+algo针 → reports/*_baseline_$(date +%F).report"
  ;;
compare)
  for pan in prog_probes algo_probes; do
    A="$RPT/${pan}_go2b_$(date +%F).report"; B="$RPT/${pan}_baseline_$(date +%F).report"
    [ -s "$A" ] && [ -s "$B" ] || { log "缺 $pan 报告"; continue; }
    echo "═══$pan: go2b vs baseline md5 (不同=有变化)═══"
    echo "go2b=$(md5 -q "$A")  baseline=$(md5 -q "$B")"
    echo "──go2b $pan──"; grep -v "^\[finish\|^── 片段\|^$" "$A" | sed 's/^── 续写(原始):/→/'
  done
  ;;
all)
  "$0" build "${2:-20-42}" && "$0" baseline && "$0" probe && "$0" compare
  ;;
*) log "用法: build|probe|baseline|compare|all"; exit 2 ;;
esac
