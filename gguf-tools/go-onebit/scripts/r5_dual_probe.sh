#!/bin/sh
# r5_dual_probe.sh — 最优栈（z 冠军 + hot-res 残差）双机流水算法探针。
# 拓扑: M4 coordinator L0:24 (42.5G 全量) ←TCP→ M1 worker L25:output (18G 切片)
# 两侧都挂 --corr go.gguf + --residual go-hot-res.gguf（各自应用本机层段）。
# 产物: /tmp/r5_dual_probe.txt（生成正文 + t/s），worker 日志 M1:/tmp/r5_worker.log
# 用法: r5_dual_probe.sh   （幂等：先杀旧 worker，跑完杀 worker——清理=杀进程）
set -u
ROOT=/Users/fodelf/git/ds4-main
M1=192.168.1.2
M4IP=192.168.1.3
M1ROOT=/Users/fodelf/ds4-main
PORT=51715
cd "$ROOT" || exit 1

echo "[1/4] 启动 M1 worker (两步启动器)"
cat > /tmp/r5_worker_launch.sh <<EOF
#!/bin/sh
cd $M1ROOT
pkill -f 'ds4 --role worker' 2>/dev/null; sleep 1
# M1 出站 connect 有 EHOSTUNREACH 怪癖(macOS 本地网络隐私拒 ds4 走 TB 桥):
# 反连模式 worker 只 accept, coordinator 主动拨入。
DS4_DIST_REVERSE_CONNECT=1 nohup ./ds4 --role worker --listen 0.0.0.0 $PORT \
  --coordinator $M4IP $PORT --layers 25:output \
  -m gguf/ds4-go1b-v2.gguf \
  --corr gguf/sidecars/go.gguf --residual gguf/sidecars/go-hot-res.gguf \
  --metal > /tmp/r5_worker.log 2>&1 &
echo WORKER-LAUNCHED \$!
EOF
scp -o BatchMode=yes -q /tmp/r5_worker_launch.sh $M1:/tmp/
ssh -o BatchMode=yes $M1 'sh /tmp/r5_worker_launch.sh < /dev/null'

echo "[2/4] 双机探针 BinarySearch"
B='// BinarySearch returns the index of target in the sorted slice xs, or -1 if absent.
func BinarySearch(xs []int, target int) int {'
M='// Max returns the largest value in xs. It panics if xs is empty.
func Max(xs []int) int {'
printf '=== dual z+res BinarySearch ===\n' > /tmp/r5_dual_probe.txt
DS4_DIST_REVERSE_CONNECT=1 ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 \
  -m gguf/ds4-go1b-v2.gguf \
  --corr gguf/sidecars/go.gguf --residual gguf/sidecars/go-hot-res.gguf \
  --temp 0 -n 120 -p "$B" --metal >> /tmp/r5_dual_probe.txt 2>/tmp/r5_coord_b.log
grep -a 't/s' /tmp/r5_coord_b.log | tail -1 >> /tmp/r5_dual_probe.txt

echo "[3/4] 双机探针 Max"
printf '\n=== dual z+res Max ===\n' >> /tmp/r5_dual_probe.txt
DS4_DIST_REVERSE_CONNECT=1 ./ds4 --role coordinator --coordinator $M1 $PORT --layers 0:24 \
  -m gguf/ds4-go1b-v2.gguf \
  --corr gguf/sidecars/go.gguf --residual gguf/sidecars/go-hot-res.gguf \
  --temp 0 -n 120 -p "$M" --metal >> /tmp/r5_dual_probe.txt 2>/tmp/r5_coord_m.log
grep -a 't/s' /tmp/r5_coord_m.log | tail -1 >> /tmp/r5_dual_probe.txt

echo "[4/4] 收尾: 杀 worker (不删任何文件)"
ssh -o BatchMode=yes $M1 "pkill -f 'ds4 --role worker' 2>/dev/null; true"
cat /tmp/r5_dual_probe.txt
echo R5-DUAL-PROBE-DONE
