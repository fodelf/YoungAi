#!/bin/bash
# 把编程模型(gguf/reactgo-prog.gguf)按均衡切点 L 分双机, 每片 ≤ GPU上限-scratch 才能全驻留:
#   M4(coordinator) = 全模型 + --layers 0:L (含3个大hash层; 不切分片, m4分片会 segfault) ≤ M4_LIMIT
#   M1(worker)      = worker 分片 层L+1:末 + output (流式传 M1, 不落地中转)              ≤ M1_LIMIT
# 切点 L 自动按每层实测尺寸算均衡。只产生并上传 M1 worker 分片(M4 不切, 省盘)。
# 纯文件操作(不加载模型/不推理), 不要 sudo。
# 用法: bash split_prog.sh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
M1=192.168.1.2
MODEL="${MODEL:-$ROOT/gguf/reactgo-prog.gguf}"
M4_LIMIT="${M4_LIMIT:-9.8}"     # M4 GPU上限 11.84 - scratch 2.07
M1_LIMIT="${M1_LIMIT:-8.6}"     # M1 GPU上限 10.67 - scratch 2.07
M1_REMOTE="/Users/fodelf/ds4-main/gguf/reactgo-prog-worker.gguf"
SPLIT="$ROOT/gguf-tools/split_gguf_layers.py"
BAL="$ROOT/gguf-tools/balanced_split.py"
[ -f "$MODEL" ] || { echo "✗ 找不到模型 $MODEL"; exit 1; }
[ -f "$SPLIT" ] || { echo "✗ 找不到切分工具 $SPLIT"; exit 1; }

echo "==== [1/4] 算均衡切点 L (M4≤${M4_LIMIT}G含hash+embed, M1≤${M1_LIMIT}G含output) ===="
read -r L NLAYER < <(python3 "$BAL" "$MODEL" "$M4_LIMIT" "$M1_LIMIT")
{ [ -z "${L:-}" ] || [ "$L" = "-1" ]; } && { echo "✗ 算不出可行切点(模型太大, 降 KEEP_TOP_K 重量化)"; exit 1; }
WLO=$((L+1))
echo "  → 切点 L=$L  (M4=层0:$L, M1=层$WLO:$NLAYER)"

echo "==== [2/3] M4 用全模型 + --layers 0:$L (不切 m4 分片: 分片会 segfault coordinator) ===="
echo "  ✓ M4 直接用全模型 $MODEL (resident=层0:$L); 不产生 m4 分片, 省 ~9.5G 盘"

echo "==== [3/3] 切 M1 worker 分片 (层$WLO:$NLAYER + output) → 流式传 M1 ===="
python3 "$SPLIT" "$MODEL" --keep-layers "$WLO:$NLAYER" --head output | ssh "$M1" "cat > '$M1_REMOTE'" || { echo "✗ M1 切片/传输失败"; exit 1; }
ssh "$M1" "ls -la '$M1_REMOTE'" | awk '{printf "  ✓ M1 片(已传): %.2f GiB -> %s\n",$5/1073741824,$NF}'

echo "==== [完成] 跑双机验证(用驱动脚本, 自带内存看门狗/超时/只杀不删) ===="
echo "  COORD_LAYERS=0:$L WORKER_LAYERS=$WLO:output TEMP=0 PREFILL_CHUNK=128 bash tools/reactgo_prog_dual.sh"
echo "    (M4=全模型 --layers 0:$L; M1=worker 分片 --layers $WLO:output; budget 各自 GPU 上限内全驻留)"
