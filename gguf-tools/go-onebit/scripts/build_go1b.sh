#!/bin/sh
# build_go1b.sh — 一次生成定制模型: 1-bit 路由专家 + z 隐变量 + 四损失, 从原始 HF.
# 基于调优结论(PROGRESS.md): 四损失的方向损失 w_align=1 → stage-2 残差单位归一化 → 方向感知共享基 U,
# 配合 rank 把 corr_cos 单调推高(0.524→0.547@rank64). 体积 base 45.6G + sidecar(rank≤256≈≤374MB) ≤46G.
#
# 步骤: 0.编译工具  1.1-bit base(没有才 gen)  2.z 全层(--w-align dir-aware U)+看门狗  3.emit sidecar  4.size≤46G
# 跑在有对应 HF shard + cap 的机器. 全 43 层: M1(全 46 shard)+ full cap; 或 dual 分两机各跑半 LAYERS 再 merge z.
# 内存安全: calib nx=256 峰 ~7G(<12G), 自带看门狗@12G.
#
# 用法/env:
#   ./build_go1b.sh                          # 全 43 层 rank64 w_align1 nx256 (≈80min 单机)
#   RANK=128 ./build_go1b.sh                 # 更高 rank=更高质量, sidecar 188MB, 仍<46G(rank≤256)
#   LAYERS="$(seq -s, 0 21)" ./build_go1b.sh # 只算 0-21(dual 时 M4 段); M1 跑 22-42 再 merge zdump
#   WALIGN=0 ./build_go1b.sh                 # 关方向损失(对照基线)
#   NX=128 ./build_go1b.sh                   # 更少样本=更快(质量略降)
set -e
ROOT="${ROOT:-/Users/fodelf/git/ds4-main}"; GT="$ROOT/gguf-tools"
HF="${HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}"; CAP="${CAP:-$ROOT/cap_m4}"
BASE="${BASE:-$ROOT/gguf/ds4-go1b.gguf}"; CORR="${CORR:-$ROOT/gguf/ds4-go1b-corr.gguf}"; ZDIR="${ZDIR:-$ROOT/zdump}"
RANK="${RANK:-64}"; WALIGN="${WALIGN:-1}"; NX="${NX:-256}"; LAYERS="${LAYERS:-$(seq -s, 0 42)}"
TEMPLATE="${TEMPLATE:-}"; BUILD="${BUILD:-1}"
NL=$(printf '%s' "$LAYERS" | tr ',' '\n' | grep -c .)

# 0. 编译最新工具
if [ "$BUILD" = 1 ]; then
  printf 'build calib_run + emit_z ... '
  ( cd "$GT" && make calib_run emit_z ) > /tmp/build_tools.log 2>&1 || { echo "FAILED"; tail -15 /tmp/build_tools.log; exit 1; }
  echo ok
fi

# 1. 1-bit base (存在则跳过)
if [ -f "$BASE" ]; then
  echo "[1/4] 1-bit base 已存在, 跳过 gen: $(du -h "$BASE" | cut -f1)"
else
  [ -n "$TEMPLATE" ] || { echo "[1/4] base 不存在且无 TEMPLATE — set TEMPLATE=<published gguf header>(见 gen_go1b.sh)"; exit 1; }
  echo "[1/4] gen 1-bit base from HF ..."
  sh "$GT/go-onebit/scripts/gen_go1b.sh" "$HF" "$TEMPLATE" "$BASE" || exit 1
fi

# 2. z 全层 (tuned 四损失: w_align 方向感知 U) + RSS 看门狗@12G
echo "[2/4] z 计算: $NL 层 rank=$RANK w_align=$WALIGN nx=$NX (看门狗@12G) ..."
mkdir -p "$ZDIR"; cd "$GT"
DS4_Z_DUMP_DIR="$ZDIR" ./calib_run --hf "$HF" --cap "$CAP" --layers "$LAYERS" \
    --solver hv --nx "$NX" --maxrank "$RANK" --w-align "$WALIGN" --threads 8 > /tmp/build_z.log 2>&1 &
QPID=$!; PEAK=0
while kill -0 "$QPID" 2>/dev/null; do
  RSS=$(ps -o rss= -p "$QPID" 2>/dev/null | tr -d ' ')
  [ -n "$RSS" ] && { [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS; [ "$RSS" -gt 12000000 ] && { echo "  WDKILL RSS=${RSS}KB"; kill -9 "$QPID"; break; }; }
  sleep 15
done
wait "$QPID" 2>/dev/null
echo "    z dumped: $(grep -c 'z dumped' /tmp/build_z.log 2>/dev/null) 层, peak_rss=$((PEAK/1024))MiB"
grep -E 'cannot open|worker.*fail' /tmp/build_z.log | head -3 || true

# 3. emit corr sidecar (~base 不动)
echo "[3/4] emit corr sidecar ..."
./emit_z --out "$CORR" --zdir "$ZDIR" --layers 43 || exit 1

# 4. size 检查 ≤46G
BSZ=$(stat -f%z "$BASE" 2>/dev/null || stat -c%s "$BASE")
CSZ=$(stat -f%z "$CORR" 2>/dev/null || stat -c%s "$CORR")
TOT=$(( (BSZ + CSZ) / 1000000 ))
echo "[4/4] base $((BSZ/1000000))MB + sidecar $((CSZ/1000000))MB = ${TOT}MB"
if [ "$TOT" -le 46000 ]; then echo "    ✓ ≤46G"; else echo "    ✗ 超 46G — 降 RANK(≤256) 或缩 base"; fi
echo "done. 测: $GT/go-onebit/scripts/test_go1b.sh   质量: iter.sh --solve -A $WALIGN -R $RANK"
