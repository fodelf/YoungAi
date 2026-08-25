#!/bin/sh
# mono_server_dual.sh — 双机 ds4-server (Claude Code /v1/messages 后端, mono-mixed)。
# M4=coordinator 层 0:19 (拥 HTTP :8000 + tokenizer/采样 + primer_copy), M1=worker 层 20:output。
# 反连(M1 出站 EHOSTUNREACH 绕过) + 专家流式(resident ≤12G/机, mmap 不驻留) + GO 数值默认。
# 内存安全: DS4_METAL_EXPERT_OFFLOAD=1 让 55G 模型走 SSD 流式, 两机 RSS 实测 <1G。
# 用法: mono_server_dual.sh   (前台不阻塞, server 留驻; curl :8000/v1/messages 测)
set -u
M1=192.168.1.2; M4IP=192.168.1.3
ROOT=/Users/fodelf/git/ds4-main; M1ROOT=/Users/fodelf/ds4-main
DPORT=5599; HTTP=8000
MODEL="${MODEL:-gguf/ds4-mono-mixed.gguf}"
KVDIR=/tmp/ds4-kv-cc
GOENV="DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_REPEAT_FREQ=1 DS4_METAL_MATH_SAFE=1 DS4_METAL_KV_RAW_F32=1 DS4_METAL_ROPE_EXP2_LOG2=1"

echo "[1/2] M1 worker 启动 (层 20:output, 反连监听 $DPORT)"
cat > /tmp/mono_srv_worker.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -f 'ds4-server --role worker' 2>/dev/null; sleep 1
$GOENV nohup ./ds4-server --role worker --listen 0.0.0.0 $DPORT --coordinator $M4IP $DPORT \
  --layers 20:output -c 65536 -m $MODEL --metal > /tmp/mono_srv_worker.log 2>&1 &
echo WORKER-PID \$!
EOF
scp -o BatchMode=yes -q /tmp/mono_srv_worker.sh $M1:/tmp/
ssh -o BatchMode=yes $M1 'sh /tmp/mono_srv_worker.sh < /dev/null'
sleep 4

echo "[2/2] M4 coordinator 启动 (层 0:19, HTTP :$HTTP)"
cd "$ROOT"
mkdir -p "$KVDIR"
pkill -f 'ds4-server --role coordinator' 2>/dev/null; sleep 1
env $GOENV nohup ./ds4-server --role coordinator --coordinator $M1 $DPORT \
  --layers 0:19 -c 65536 --max-output-tokens 512 --kv-cache-continued-interval-tokens 0 \
  --kv-disk-dir "$KVDIR" --kv-disk-space-mb 8192 --port $HTTP \
  -m "$MODEL" --metal > /tmp/mono_srv_coord.log 2>&1 &
echo "COORD-PID $!"
echo "日志: /tmp/mono_srv_coord.log (M4) + /tmp/mono_srv_worker.log (M1)"
echo "测: curl http://localhost:$HTTP/v1/messages ..."
