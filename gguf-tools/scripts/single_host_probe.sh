#!/bin/sh
# single_host_probe.sh — 单机 mono 短跑生成 + 外部内存看门狗 (panic 防护)。
# 铁律: 单机 mono offload 持续生成→wired 暴涨→内核 panic; 只短跑 + 看门狗守 free%。
# 在本机(设计跑 M1=有魂那台)运行。用法: single_host_probe.sh "PROMPT" [N] [FREE_KILL_PCT]
set -u
ROOT=${ROOT:-/Users/fodelf/ds4-main}
MODEL="${MODEL:-gguf/ds4-mono-mixed.gguf}"
PROMPT=${1:?prompt}
N=${2:-48}
KILL_PCT=${3:-12}          # free% 低于此 → 杀 ds4 防 panic
cd "$ROOT"
pkill -9 -f 'ds4 ' 2>/dev/null; sleep 1

echo "[单机] mono $MODEL, n=$N, offload+GO数值, 看门狗 free<${KILL_PCT}%"
DS4_MEM_BUDGET_MB=11000 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_EXPERT_OFFLOAD_DIRECT=1 \
  DS4_METAL_PREFILL_CHUNK=256 DS4_REPEAT_FREQ=1 \
  DS4_METAL_MATH_SAFE=1 DS4_METAL_KV_RAW_F32=1 DS4_METAL_ROPE_EXP2_LOG2=1 \
  nohup ./ds4 -m "$MODEL" --metal --ctx 4096 --temp 0 -n "$N" -p "$PROMPT" \
  > /tmp/single_probe.out 2> /tmp/single_probe.log &
DSPID=$!
echo "ds4 PID=$DSPID"

# 外部看门狗: free% 低于阈值即杀 (mono offload panic 防线)
( while kill -0 $DSPID 2>/dev/null; do
    fp=$(memory_pressure 2>/dev/null | awk '/free percentage/{print $NF+0}')
    if [ -n "$fp" ] && [ "$fp" -lt "$KILL_PCT" ]; then
      echo "[看门狗] free ${fp}% < ${KILL_PCT}% → 杀 ds4 防 panic" >> /tmp/single_probe.log
      kill -9 $DSPID 2>/dev/null; break
    fi
    sleep 2
  done ) &
WDPID=$!

wait $DSPID 2>/dev/null
kill $WDPID 2>/dev/null
echo "--- 输出 ---"; cat /tmp/single_probe.out
echo "--- t/s + 看门狗 ---"; grep -aE 't/s|看门狗' /tmp/single_probe.log | tail -3
echo "--- 收尾 free% ---"; memory_pressure 2>/dev/null | grep 'free percentage'
