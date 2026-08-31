#!/bin/bash
# p2_truncprobe.sh — 未来泄漏判决探针(2026-08-20 第2针)。
# 原理: 截断序列使"真值下一token"不在输入里。若引擎 p(true)从1.000塌回正常
# → score批路存在未来泄漏(mask/indexer/压缩KV); 若仍1.000 → 引擎真能预测, 锚有病。
# 检查点: ids截1528 → pos1526(未来在输入)应仍1.0, pos1527(未来不在)见分晓; ids截217 → pos216。
set -uo pipefail
ROOT="$HOME/ds4-main"
G7="$ROOT/gguf/go-onebit/g7"
cd "$ROOT"
python3 - <<'PY'
ids = open("gguf/go-onebit/g7/wt2.ids").read().split()
open("/tmp/p2_trunc1528.ids","w").write(" ".join(ids[:1528]))
open("/tmp/p2_trunc217.ids","w").write(" ".join(ids[:217]))
print("truncated ids written:", len(ids[:1528]), len(ids[:217]))
PY
for N in 1528 217; do
    # (env 大扫除 2026-08-31: CUDA_NO_TOKEN_GRAPH 已无读取者, 删)
    timeout --foreground 1200 ./ds4 --cuda \
        -m "$ROOT/gguf/ds4-iq2.gguf" --score-ids "/tmp/p2_trunc$N.ids" \
        --score-out "/tmp/p2_trunc$N.bin" </dev/null 2>&1 | grep -aE "完成" | tail -1
done
# 原逐位置 lnp 细察(anchor_metrics.py)随全仓 Python 清零删除(见 git 历史)。
# C 版 bench/anchor_metrics 以全长跑 dump 作参考、截断跑 dump 作学生, 五指标聚合判
# "截断是否改了截断点之前的分布"(逐位置打印见 git 历史 py 版)。
AM="$(dirname "$0")/../bench/anchor_metrics"
[ -x "$AM" ] || make -C "$(dirname "$0")/.." anchor_metrics
for N in 1528 217; do
  echo "== 截断$N vs 全长 =="
  "$AM" --ref-raw /tmp/p2_iq2_wt2.bin --ids gguf/go-onebit/g7/wt2.ids \
        --stu-raw "/tmp/p2_trunc$N.bin"
done
