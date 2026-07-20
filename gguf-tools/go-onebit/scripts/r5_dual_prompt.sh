#!/bin/sh
# r5_dual_prompt.sh — 最优栈（z+hot-res）双机流水单提示词跑分。
# 用法: r5_dual_prompt.sh PROMPT_FILE [N_TOKENS]
# 输出: /tmp/r5_dual_prompt_out.txt（正文+t/s）; worker 日志 M1:/tmp/r5_worker.log
set -u
ROOT=/Users/fodelf/git/ds4-main
M1=192.168.1.2
M4IP=192.168.1.3
M1ROOT=/Users/fodelf/ds4-main
PORT=51715
PF=${1:?usage: r5_dual_prompt.sh PROMPT_FILE [N]}
N=${2:-24}
cd "$ROOT" || exit 1

echo "[1/3] worker 探活/启动 (反连)"
if ssh -o BatchMode=yes $M1 "pgrep -f 'ds4 --role worker' >/dev/null && netstat -an | grep -q '\.$PORT .*LISTEN'"; then
  echo "worker 已常驻, 复用"
else
cat > /tmp/r5_worker_launch.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -f 'ds4 --role worker' 2>/dev/null; sleep 1
DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_EVENT_DRAIN=1 nohup ./ds4 --role worker --listen 0.0.0.0 $PORT \
  --coordinator $M4IP $PORT --layers 25:output \
  -m gguf/ds4-go1b-v2.gguf \
  --corr gguf/sidecars/go.gguf --residual gguf/sidecars/go-hot-res.gguf \
  --metal > /tmp/r5_worker.log 2>&1 &
echo WORKER-LAUNCHED \$!
EOF
scp -o BatchMode=yes -q /tmp/r5_worker_launch.sh $M1:/tmp/
ssh -o BatchMode=yes $M1 'sh /tmp/r5_worker_launch.sh < /dev/null'
sleep 3
fi

echo "[2/3] coordinator 生成"
: > /tmp/r5_dual_prompt_out.txt
DS4_DIST_REVERSE_CONNECT=1 DS4_METAL_EXPERT_EVENT_DRAIN=1 ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 \
  -m gguf/ds4-go1b-v2.gguf \
  --corr gguf/sidecars/go.gguf --residual gguf/sidecars/go-hot-res.gguf \
  --temp 0 -n "$N" -p "$(cat "$PF")" --metal >> /tmp/r5_dual_prompt_out.txt 2>/tmp/r5_coord_p.log
grep -a 't/s' /tmp/r5_coord_p.log | tail -1 >> /tmp/r5_dual_prompt_out.txt

echo "[3/3] worker 常驻留用 (换配置/收工才杀)"
cat /tmp/r5_dual_prompt_out.txt
echo R5-DUAL-PROMPT-DONE
