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
    env DS4_CUDA_NO_TOKEN_GRAPH=1 timeout --foreground 1200 ./ds4 --cuda \
        -m "$ROOT/gguf/ds4-iq2.gguf" --score-ids "/tmp/p2_trunc$N.ids" \
        --score-out "/tmp/p2_trunc$N.bin" </dev/null 2>&1 | grep -aE "完成" | tail -1
done
python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, "gguf-tools/scripts")
from anchor_metrics import read_student_logits, log_softmax
ids = [int(t) for t in open("gguf/go-onebit/g7/wt2.ids").read().split()]
full = read_student_logits("/tmp/p2_iq2_wt2.bin")
lfull = log_softmax(full)
for N, poss in [(1528, [1526, 1527, 216]), (217, [216])]:
    t = read_student_logits(f"/tmp/p2_trunc{N}.bin")
    lt = log_softmax(t)
    for p in poss:
        true = ids[p+1]
        infut = "未来在输入" if p+1 < N else "★未来不在输入★"
        print(f"截断{N} pos={p} true={true} [{infut}]  "
              f"截断跑 lnp(true)={lt[p,true]:+.3f} top1={int(np.argmax(lt[p]))}  |  "
              f"全长跑 lnp(true)={lfull[p,true]:+.3f}")
PY
