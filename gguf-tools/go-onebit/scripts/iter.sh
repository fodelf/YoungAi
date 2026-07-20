#!/bin/sh
# iter.sh — 秒级内环: 调 1-bit / z隐变量 / 四损失 后立刻看质量, 不跑全模型(那才是几小时).
#
# 改这些 C 后跑本脚本 (会自动 make calib_run):
#   onebit_quant.c                      -> 1-bit 量化方案 (sign/scale/分组)
#   hiddenvar_solve.c / solve_*.c       -> z 隐变量 + 四损失闭式解
#   linalg_small.c                      -> Cholesky / power-iteration
#
# 两档(瓶颈是 4096² solve ~100s, 跟专家数无关; base_cos 不需要 solve):
#   默认 快档(--no-solve): 只算 base_cos(1-bit 专家方向保真) — ~10s, 迭代 1-bit 方案用这个.
#   --solve  慢档: 全四损失解出 corr_cos(+z 后) — ~2-3min, 迭代 z/四损失 时才用(需≥128专家防过拟合).
# 越高越好的是 cos; relL2 是幅值误差(低=好). 文本输出用 test_go1b.sh(分钟级).
#
# 用法:
#   ./iter.sh                # 快档: 层8 nx32 32专家 base_cos ~10s
#   ./iter.sh -L 0,8,12      # 几层一起(M4 只有 0-12)
#   ./iter.sh --solve        # 慢档: 全解 corr_cos(自动 256 专家) ~2-3min
#   ./iter.sh -x 64          # 改 nx
#   ./iter.sh --no-build     # 没改 C 时跳过编译
set -e
ROOT="${ROOT:-/Users/fodelf/git/ds4-main}"; GT="$ROOT/gguf-tools"
HF="${HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"; CAP="${CAP:-$ROOT/cap_m4}"
LAYER=8; NX=32; NE=32; BUILD=1; SOLVE=0; RANK=16; WA=0
while [ $# -gt 0 ]; do case "$1" in
  -L|--layer)  LAYER="$2"; shift 2;;
  -x|--nx)     NX="$2"; shift 2;;
  -E|--nexp)   NE="$2"; shift 2;;
  -R|--maxrank) RANK="$2"; shift 2;;
  -A|--w-align) WA="$2"; shift 2;;   # 方向损失(loss4): 0=关 1=开(方向感知U, 提 corr_cos)
  --solve)     SOLVE=1; shift;;
  --no-build)  BUILD=0; shift;;
  -h|--help)   sed -n '2,28p' "$0"; exit 0;;
  *) echo "unknown: $1 (--help)"; exit 1;; esac; done
if [ "$BUILD" = 1 ]; then
  printf 'build calib_run ... '
  ( cd "$GT" && make calib_run ) > /tmp/iter_build.log 2>&1 \
    || { echo "FAILED:"; tail -20 /tmp/iter_build.log; exit 1; }
  echo ok
fi
SOLVE_FLAG="--no-solve"; MODE="快档(base_cos, 跳过 solve)"
if [ "$SOLVE" = 1 ]; then
  SOLVE_FLAG=""; MODE="慢档(全四损失解 corr_cos)"
  [ "$NE" -lt 128 ] && { NE=256; echo "note: --solve 需样本数 n_exp*nx > 4096 防过拟合 -> n_exp 自动设 256"; }
fi
mkdir -p /tmp/iter_z; rm -f /tmp/iter_z/z_L*.bin
echo "=== calib_run L=$LAYER nx=$NX n_exp=$NE  $MODE ==="
cd "$GT"
T0=$(date +%s 2>/dev/null || echo 0)
DS4_Z_DUMP_DIR=/tmp/iter_z ./calib_run --hf "$HF" --cap "$CAP" --layers "$LAYER" \
    --solver hv --nx "$NX" --n-experts "$NE" --maxrank "$RANK" --w-align "$WA" $SOLVE_FLAG --threads 8 2>&1 \
  | grep -E 'L +\| tot_energy|^ *[0-9]+ *\|| d_l=|cannot open|worker.*fail' || true
T1=$(date +%s 2>/dev/null || echo 0)
echo ""
echo "读数: base_cos=1-bit专家方向保真(快档主信号) | corr_cos=+z后(慢档) | relL2=幅值误差(低=好)"
echo "      cos 越高越好. 改 1-bit 看 base_cos↑; 改 z/四损失 加 --solve 看 corr_cos↑.  耗时 $((T1-T0))s"
