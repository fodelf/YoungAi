#!/bin/sh
# z_v3_solve.sh — v3 z 四损失逐层解, ★双机认领制★ (第一原则: 双机均衡, 认领即注册发现).
# 用法: sh z_v3_solve.sh m1|m4 [LANE_TAG]
#   m4 lane: 本机解 (obase/caps/routed 全在 /tmp/cap_v3prep, calib_run 本机二进制),
#            z bin 回传 M1 zdump_v3/ (emit_z 汇集点)
#   m1 lane: ssh 驱动 M1 解 (obase 按层从 M4 送料)
# 认领: M1 上 zdump_v3/claim_L$L (mkdir 原子); 完成凭据 = zdump_v3/z_L$L.bin。
# 跳过 L24 与 L0-5 直到教师目标 ready (routed_L 文件存在即自动纳入)。
set -u
M1=192.168.1.2
M1ROOT=/Users/fodelf/ds4-main
ROOT=/Users/fodelf/git/ds4-main
SSH="ssh -o BatchMode=yes $M1"
OB=/tmp/cap_v3prep
WHO=${1:?m1|m4}
TAG=${2:-$WHO}
RANK=${RANK:-64}
THREADS=${THREADS:-6}   # 多 lane 时调小 (每 lane RSS ~2.5G, CPU 弹性超订可接受)
ZD=$M1ROOT/zdump_v3
LOG=/tmp/z_v3.log

$SSH "mkdir -p /tmp/obase_v3 $ZD"

solve_common_args() { # $1=cap_dir $2=obase_dir $3=L
  echo "--cap $1 --layers $3 --solver rrr --obase-dir $2 --rank $RANK --nx 20000 --ntest 2048 --heldout-frac 0.17 --chunk 512 --w-align 1 --threads $THREADS"
}

while :; do
  L=""
  for c in $(seq 6 42) 0 1 2 3 4 5; do
    $SSH "[ -f $ZD/z_L$c.bin ] && exit 1
          mkdir $ZD/claim_L$c 2>/dev/null || exit 1" || continue
    # 教师目标就绪检查 (L24/L0-5 在 inject 完成前自动跳过并释放认领)
    if [ "$WHO" = m4 ]; then
      CAPD=$OB
      [ -f "$OB/routed_L$c.npy" ] || { $SSH "rm -rf $ZD/claim_L$c"; continue; }
    else
      CAPD=$($SSH "for d in $M1ROOT/cap_v2r2 $M1ROOT/cap_v2r1_deep $M1ROOT/cap_l05; do [ -f \$d/routed_L$c.npy ] && { echo \$d; break; }; done")
      [ -n "$CAPD" ] || { $SSH "rm -rf $ZD/claim_L$c"; continue; }
    fi
    [ -f "$OB/obase_v3_L$c.npy" ] || { $SSH "rm -rf $ZD/claim_L$c"; continue; }
    L=$c; break
  done
  [ -z "$L" ] && {
    N=$($SSH "ls $ZD 2>/dev/null | grep -c '^z_L.*bin'")
    echo "$TAG: no claimable layer (done=$N/43) $(date +%H:%M:%S)" | tee -a "$LOG"
    [ "$N" -ge 43 ] && break
    sleep 120; continue   # 等 inject 产出新教师目标
  }
  echo "=== z L$L $TAG $(date +%H:%M:%S) ===" | tee -a "$LOG"
  if [ "$WHO" = m4 ]; then
    mkdir -p /tmp/zdump_v3_m4
    ( cd "$ROOT/gguf-tools" && DS4_Z_DUMP_DIR=/tmp/zdump_v3_m4 ./calib_run --hf /dev/null \
      $(solve_common_args "$OB" "$OB" "$L") > "/tmp/z_L$L.m4.log" 2>&1 )
    grep -aE "ŷ from obase|corr_cos|scalar-gain|z dumped|failed|缺失" "/tmp/z_L$L.m4.log" | tail -4 | tee -a "$LOG"
    if [ -f "/tmp/zdump_v3_m4/z_L$L.bin" ]; then
      scp -o BatchMode=yes -q "/tmp/zdump_v3_m4/z_L$L.bin" "$M1:$ZD/" && rm -f "/tmp/zdump_v3_m4/z_L$L.bin"
      scp -o BatchMode=yes -q "/tmp/z_L$L.m4.log" "$M1:/tmp/z_L$L.log" 2>/dev/null || true
    else
      echo "$TAG L$L solve FAILED" | tee -a "$LOG"
    fi
  else
    scp -o BatchMode=yes -q "$OB/obase_v3_L$L.npy" "$M1:/tmp/obase_v3/" || { $SSH "rm -rf $ZD/claim_L$L"; continue; }
    $SSH "cd $M1ROOT/gguf-tools && DS4_Z_DUMP_DIR=$ZD ./calib_run --hf $M1ROOT/hf/DeepSeek-V4-Flash-Base \
        $(solve_common_args "$CAPD" /tmp/obase_v3 "$L") > /tmp/z_L$L.log 2>&1; \
        grep -aE 'ŷ from obase|corr_cos|scalar-gain|z dumped|failed|缺失' /tmp/z_L$L.log | tail -4; \
        rm -f /tmp/obase_v3/obase_v3_L$L.npy" | tee -a "$LOG"
  fi
  $SSH "rm -rf $ZD/claim_L$L"
done
echo "$TAG Z-LANE-EXIT $(date +%H:%M:%S)" | tee -a "$LOG"
