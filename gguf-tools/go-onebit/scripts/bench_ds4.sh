#!/bin/sh
# bench_ds4.sh — ★统一准确基准★ 用【生成时的同一个 ds4 引擎】测【实际 GGUF】在【固定语料】上的质量。
# 构造上保证: 测试质量 == 部署质量 (同引擎/同格式/同前向), 不再有 numpy-测量 vs 引擎-部署 的割裂。
#
# 两个指标 (都跑 ds4 引擎, 实际 GGUF):
#   ① teacher-forced PPL  (定量, 快):   ds4 --perplexity-file  在 bench/coding_eval.txt
#   ② 自由生成样本 (定性, 对齐 Claude Code): ds4 -p <固定代码prompt> -n 64
# 大模型 (>16G) 走双机 offload (内存安全); 小模型可单机 (设 SINGLE=1)。
#
# 判读锚 (ds4引擎 PPL, 同语料): 旧 mono-mixed=6.47 可用 / 纯1bit ds4-code-dyn≈16M 崩。
# PPL <10 可用, >100 退化, >1e4 崩坏。
#
# 用法: bench_ds4.sh MODEL.gguf            (双机, 默认)
#       SINGLE=1 bench_ds4.sh MODEL.gguf   (单机, 仅小模型)
set -u
M1=192.168.1.2; M4IP=192.168.1.3
ROOT=/Users/fodelf/git/ds4-main; M1ROOT=/Users/fodelf/ds4-main
DPORT=5599; PORT=51730
MODEL=${1:?MODEL.gguf}
CORPUS="$ROOT/gguf-tools/go-onebit/bench/coding_eval.txt"
PROMPT='func twoSum(nums []int, target int) []int {
	for i := 0; i < len(nums); i++ {'
GOENV="DS4_METAL_EXPERT_OFFLOAD=1 DS4_METAL_PREFILL_CHUNK=512 DS4_REPEAT_FREQ=1 DS4_METAL_MATH_SAFE=1 DS4_METAL_KV_RAW_F32=1 DS4_METAL_ROPE_EXP2_LOG2=1"
cd "$ROOT"

echo "=== ★统一基准★ ds4引擎 × $MODEL × 固定语料 $(basename $CORPUS) ==="
pkill -9 -f 'ds4 ' 2>/dev/null; ssh -o BatchMode=yes $M1 "pkill -9 -f 'ds4 ' 2>/dev/null" 2>/dev/null; sleep 2

if [ "${SINGLE:-0}" = 1 ]; then
  echo "[单机] "
  env $GOENV DS4_MEM_BUDGET_MB=11000 ./ds4 -m "$MODEL" --metal --ctx 4096 --perplexity-file "$CORPUS" 2>/tmp/bench_ppl.log | tee /tmp/bench_ppl.txt
  echo "--- 自由生成 ---"
  env $GOENV DS4_MEM_BUDGET_MB=11000 ./ds4 -m "$MODEL" --metal --ctx 4096 --temp 0 -n 64 -p "$PROMPT" 2>/dev/null | tee /tmp/bench_gen.txt
else
  # 双机: M1 worker 25:output (反连) + M4 coordinator 0:24
  cat > /tmp/bench_worker.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -9 -f 'ds4 --role worker' 2>/dev/null; sleep 1
DS4_DIST_REVERSE_CONNECT=1 $GOENV nohup ./ds4 --role worker --listen 0.0.0.0 $PORT --coordinator $M4IP $PORT \
  --layers 25:output --ctx 4096 -m $MODEL --metal > /tmp/bench_worker.log 2>&1 &
echo W-\$!
EOF
  scp -o BatchMode=yes -q /tmp/bench_worker.sh $M1:/tmp/; ssh -o BatchMode=yes $M1 'sh /tmp/bench_worker.sh </dev/null'; sleep 4
  echo "--- ① teacher-forced PPL (双机) ---"
  DS4_DIST_REVERSE_CONNECT=1 env $GOENV ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 --ctx 4096 \
    -m "$MODEL" --perplexity-file "$CORPUS" --metal 2>/tmp/bench_ppl.log | tee /tmp/bench_ppl.txt
  echo "--- ② 自由生成 (双机, 代码强制prompt) ---"
  DS4_DIST_REVERSE_CONNECT=1 env $GOENV ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 --ctx 4096 \
    -m "$MODEL" --temp 0 -n 64 -p "$PROMPT" --metal 2>/dev/null | tee /tmp/bench_gen.txt
  ssh -o BatchMode=yes $M1 "pkill -f 'ds4 --role worker' 2>/dev/null" 2>/dev/null
fi

echo ""; echo "=== ★基准结论★ ==="
PPL=$(grep -aoE 'ppl=[0-9.]+' /tmp/bench_ppl.txt | tail -1 | cut -d= -f2)
echo "ds4引擎 teacher-forced PPL = ${PPL:-?} (锚: 可用<10 / 退化>100 / 崩坏>1e4; 旧mono=6.47)"
echo "自由生成质量: 见上 ② (对齐 Claude Code 实际输出)"
