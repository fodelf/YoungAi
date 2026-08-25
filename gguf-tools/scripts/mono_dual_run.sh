#!/bin/sh
# mono_dual_run.sh — go2b-NF 单块模型 双机分层流水线运行 (distributed layered)。
# M4=coordinator 层 0:24 (拥 tokenizer/采样), M1=worker 层 25:output; 反连(M1 出站 EHOSTUNREACH 绕过)。
# 两机各有 gguf/ds4-mono-mixed.gguf, --layers 只触碰本机层 → 专家 IO 双机并行分摊。
# 用法: mono_dual_run.sh "PROMPT" [N_TOKENS]
set -u
M1=192.168.1.2; M4IP=192.168.1.3
ROOT=/Users/fodelf/git/ds4-main; M1ROOT=/Users/fodelf/ds4-main
PORT=51730
MODEL="${MODEL:-gguf/ds4-mono-mixed.gguf}"
PROMPT=${1:?prompt}
N=${2:-256}

echo "[1/3] M1 worker 启动 (层 25:output, 反连监听)"
cat > /tmp/mono_worker.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -f 'ds4 --role worker' 2>/dev/null; sleep 1
DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_REPEAT_FREQ=1 DS4_METAL_MATH_SAFE=${DS4_METAL_MATH_SAFE:-1} DS4_METAL_KV_RAW_F32=${DS4_METAL_KV_RAW_F32:-1} DS4_METAL_ROPE_EXP2_LOG2=${DS4_METAL_ROPE_EXP2_LOG2:-1} \
  nohup ./ds4 --role worker --listen 0.0.0.0 $PORT --coordinator $M4IP $PORT \
  --layers 25:output --ctx 4096 -m $MODEL --metal > /tmp/mono_worker.log 2>&1 &
echo WORKER-PID \$!
EOF
scp -o BatchMode=yes -q /tmp/mono_worker.sh $M1:/tmp/
ssh -o BatchMode=yes $M1 'sh /tmp/mono_worker.sh < /dev/null'
sleep 4

echo "[2/3] M4 coordinator 生成 (层 0:24, temp0, n=$N)"
cd "$ROOT"
DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_REPEAT_FREQ=1 DS4_METAL_MATH_SAFE=${DS4_METAL_MATH_SAFE:-1} DS4_METAL_KV_RAW_F32=${DS4_METAL_KV_RAW_F32:-1} DS4_METAL_ROPE_EXP2_LOG2=${DS4_METAL_ROPE_EXP2_LOG2:-1} \
  ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 --ctx 4096 \
  -m "$MODEL" --temp 0 -n "$N" -p "$PROMPT" --metal 2>/tmp/mono_coord.log | tee /tmp/mono_out.txt
echo "--- t/s ---"; grep -a 't/s' /tmp/mono_coord.log | tail -1
echo "[3/3] worker 留驻 (换配置/收工才杀)"
