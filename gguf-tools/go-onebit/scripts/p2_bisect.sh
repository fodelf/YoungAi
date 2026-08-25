#!/bin/bash
# p2_bisect.sh — 第2针: 同模型"离线回放0.48 vs 引擎在线1.23"2.6×洞的二分探针(2026-08-20)。
# 关键更正: "官方q2在线0.4207"实为unsloth公开表数字(ctx512协议), 从未过我们引擎 ⇒
# 引擎图通病未被排除。二分设计: ds4-iq2.gguf=量化器直产(无vq_merge骨架+blob合并路),
# 离线回放0.495≈cal12的0.48 ⇒ 引擎在线跑它:
#   ≈1.2 → 合并链路洗清, 嫌疑=引擎图数值/骨架配方(共享)
#   ≈0.5-0.6 → 钉死 vq_merge 合并链路
# 段: score_iq2 | dtypes(骨架dtype地图) 。产物/日志: /tmp/p2_*.{bin,log}
set -uo pipefail
ROOT="$HOME/ds4-main"
G7="$ROOT/gguf/go-onebit/g7"
R30="$ROOT/gguf/go-onebit/r30"
SC="$ROOT/gguf-tools/go-onebit/scripts"
LOG(){ echo "[p2 $(date +%H:%M:%S)] $*"; }
ST="${1:-score_iq2}"

case "$ST" in
score_iq2)
    cd "$ROOT"
    LOG "引擎在线 score-ids: ds4-iq2.gguf(量化器直产) × wt2.ids 2653tok"
    env DS4_CUDA_NO_TOKEN_GRAPH=1 timeout --foreground 3000 ./ds4 --cuda \
        -m "$ROOT/gguf/ds4-iq2.gguf" --score-ids "$G7/wt2.ids" \
        --score-out /tmp/p2_iq2_wt2.bin </dev/null 2>&1 | grep -aE "score|完成|error|fail" | tail -3
    [ -s /tmp/p2_iq2_wt2.bin ] || { LOG "★score 没落盘★"; exit 2; }
    LOG "五指标 vs FP 锚"
    python3 "$SC/anchor_metrics.py" --ref "$R30/anchor_wt2_s2653.bin" \
        --ids "$G7/wt2.ids" --student /tmp/p2_iq2_wt2.bin --tail 0 2>&1 | tail -7
    ;;
dtypes)
    python3 - <<'PY'
from gguf import GGUFReader
import collections, sys
for f in ["gguf/go-onebit/r30/r30_skeleton.gguf", "gguf/ds4-cal12.gguf", "gguf/ds4-iq2.gguf"]:
    import os
    p = os.path.expanduser("~/ds4-main/"+f)
    r = GGUFReader(p)
    fam = collections.Counter()
    for t in r.tensors:
        n = t.name
        # 家族归并: blk.N.xxx → xxx
        parts = n.split(".")
        key = ".".join(parts[2:]) if n.startswith("blk.") else n
        fam[(key, str(t.tensor_type).split(".")[-1])] += 1
    print("=====", f, f"n_tensors={len(r.tensors)}")
    for (k, ty), c in sorted(fam.items()):
        print(f"  {k:44s} {ty:10s} x{c}")
PY
    ;;
*) echo "用法: $0 [score_iq2|dtypes]"; exit 1 ;;
esac
LOG "p2 $ST 收官"
# ===== 追加段(2026-08-20): perpos — 逐位置 KL 分布, 定位爆炸位置形态 =====
# 用法: p2_bisect.sh perpos <student.bin> [student2.bin ...]
# 判读: 头部集中=BOS/协议; 均匀=量化底噪; 尾段集中=长上下文/indexer; mod-128 周期=score 分块缝
