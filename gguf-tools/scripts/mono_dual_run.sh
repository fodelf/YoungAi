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
# (env 大扫除 2026-08-31: EXPERT_OFFLOAD=1 归 AUTO 判定; REPEAT_FREQ 引擎已无此路;
#  原 MATH_SAFE/KV_RAW_F32/ROPE_EXP2_LOG2 三个可覆盖 env 成组升格为 --strict-fp,
#  跨 GPU 漂移诊断要关就 STRICT_FP=0)
SFP=""; [ "${STRICT_FP:-1}" = 1 ] && SFP="--strict-fp"
GOFLAGS="--reverse-connect --prefill-chunk 512 $SFP"

echo "[1/3] M1 worker 启动 (层 25:output, 反连监听)"
cat > /tmp/mono_worker.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -f 'ds4 --role worker' 2>/dev/null; sleep 1
nohup ./ds4 $GOFLAGS --role worker --listen 0.0.0.0 $PORT --coordinator $M4IP $PORT \
  --layers 25:output --ctx 4096 -m $MODEL --metal > /tmp/mono_worker.log 2>&1 &
echo WORKER-PID \$!
EOF
scp -o BatchMode=yes -q /tmp/mono_worker.sh $M1:/tmp/
ssh -o BatchMode=yes $M1 'sh /tmp/mono_worker.sh < /dev/null'
sleep 4

echo "[2/3] M4 coordinator 生成 (层 0:24, temp0, n=$N)"
cd "$ROOT"
./ds4 $GOFLAGS --role coordinator --coordinator $M1 $PORT --layers 0:24 --ctx 4096 \
  -m "$MODEL" --temp 0 -n "$N" -p "$PROMPT" --metal 2>/tmp/mono_coord.log | tee /tmp/mono_out.txt
echo "--- t/s ---"; grep -a 't/s' /tmp/mono_coord.log | tail -1
echo "[3/3] worker 留驻 (换配置/收工才杀)"
