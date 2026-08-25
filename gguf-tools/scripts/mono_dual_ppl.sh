#!/bin/sh
# mono_dual_ppl.sh — 双机 mono teacher-forced perplexity (部署模型还原率, 免疫跨GPU脆性)。
# teacher-forced NLL 不采样→不翻argmax→跨GPU epsilon无影响; 双机装得下55G; 有界不panic。
# 用法: mono_dual_ppl.sh <text_file>
set -u
M1=192.168.1.2; M4IP=192.168.1.3
ROOT=/Users/fodelf/git/ds4-main; M1ROOT=/Users/fodelf/ds4-main
PORT=51730; MODEL="${MODEL:-gguf/ds4-mono-mixed.gguf}"
TXT=${1:?text file}
GOENV="DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_REPEAT_FREQ=1 DS4_METAL_MATH_SAFE=1 DS4_METAL_KV_RAW_F32=1 DS4_METAL_ROPE_EXP2_LOG2=1"

echo "[1/2] M1 worker 启动 (层 25:output)"
cat > /tmp/mono_ppl_worker.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -9 -f 'ds4 --role worker' 2>/dev/null; sleep 1
$GOENV nohup ./ds4 --role worker --listen 0.0.0.0 $PORT --coordinator $M4IP $PORT \
  --layers 25:output --ctx 4096 -m $MODEL --metal > /tmp/mono_ppl_worker.log 2>&1 &
echo WORKER-PID \$!
EOF
scp -o BatchMode=yes -q /tmp/mono_ppl_worker.sh $M1:/tmp/
ssh -o BatchMode=yes $M1 'sh /tmp/mono_ppl_worker.sh < /dev/null'
sleep 4

echo "[2/2] M4 coordinator perplexity (层 0:24, teacher-forced)"
cd "$ROOT"
env $GOENV ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 --ctx 4096 \
  -m "$MODEL" --perplexity-file "$TXT" --metal 2>/tmp/mono_ppl_coord.log | tee /tmp/mono_ppl.txt
echo "--- NLL/PPL ---"; grep -aiE 'perplex|nll|ppl|token' /tmp/mono_ppl_coord.log | tail -8
ssh -o BatchMode=yes $M1 "pkill -f 'ds4 --role worker' 2>/dev/null" 2>/dev/null
