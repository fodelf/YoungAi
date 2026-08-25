#!/bin/bash
# g4d_lanes.sh — 每层最小 bit 扫描三路并发驱动 + 合并汇总(2026-07-28)。
# 前置: g4c_capture_v5x.sh 已产 /tmp/g4c_x_L00..L42.npy + /tmp/g4c_cells.tsv。
# 产物: /tmp/g4d_lane{0,1,2}.rpt + stdout 全层表 + 最小体积账。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f /tmp/g4c_x_L00.npy ] && [ -f /tmp/g4c_x_L42.npy ] || { echo "[g4d] 全层 X 缺失, 先跑捕获" >&2; exit 2; }
for k in 0 1 2; do rm -f "/tmp/g4d_lane$k.rpt"; done
python3 "$HERE/g4d_layer_minbit.py" --l0 0  --l1 14 --out /tmp/g4d_lane0.rpt > /tmp/g4d_lane0.log 2>&1 &
P0=$!
python3 "$HERE/g4d_layer_minbit.py" --l0 15 --l1 28 --out /tmp/g4d_lane1.rpt > /tmp/g4d_lane1.log 2>&1 &
P1=$!
python3 "$HERE/g4d_layer_minbit.py" --l0 29 --l1 42 --out /tmp/g4d_lane2.rpt > /tmp/g4d_lane2.log 2>&1 &
P2=$!
echo "[g4d] 三路已发: 0-14($P0) 15-28($P1) 29-42($P2)" >&2
FAIL=0
wait $P0 || { echo "[g4d] lane0 失败:" >&2; tail -3 /tmp/g4d_lane0.log >&2; FAIL=1; }
wait $P1 || { echo "[g4d] lane1 失败:" >&2; tail -3 /tmp/g4d_lane1.log >&2; FAIL=1; }
wait $P2 || { echo "[g4d] lane2 失败:" >&2; tail -3 /tmp/g4d_lane2.log >&2; FAIL=1; }
[ "$FAIL" = 0 ] || exit 1
python3 - <<'PEOF'
rows = []
for k in range(3):
    with open(f"/tmp/g4d_lane{k}.rpt") as f:
        hdr = f.readline().split()
        rows += [ln.split() for ln in f if ln.strip()]
rows.sort(key=lambda r: r[0])
print(" ".join(hdr))
for r in rows: print(" ".join(r))
bpws = [float(r[-1]) for r in rows]
mean_bpw = sum(bpws)/len(bpws)
# 专家总量 72.6 GiB fp8(=8bpw) → 扫描口径体积 = 72.6 × mean_bpw/8 (w1 口径外推 w1/w3/w2)
print(f"\n== 最小体积账 (w1 口径, 层内锚=vq8 生产 cos) ==")
print(f"层数={len(bpws)} 平均 min_bpw={mean_bpw:.4f} 分布: " +
      " ".join(f"{v}:{bpws.count(v)}" for v in sorted(set(bpws))))
print(f"专家体积: 72.6 GiB(fp8) × {mean_bpw:.4f}/8 = {72.6*mean_bpw/8:.2f} GiB")
print("[口径] w1 短针外推全专家(w2 历史抗码本未验); 端到端须全栈 rr 判决确认")
PEOF
