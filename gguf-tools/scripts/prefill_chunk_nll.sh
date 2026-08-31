#!/usr/bin/env bash
# prefill 分块一致性/正确性裁判:
#   同一 id 流在不同 --prefill-chunk 下走批前向, 逐位置导出 logits, 算 next-token NLL。
#   批路径若数值正确, NLL 应与分块无关; 谁的 NLL 最低谁离真模型最近。
#   (env 大扫除 2026-08-31: 原 DS4_EVAL_CHUNK 已死; eval-ids 仪器固定 512 分块并被
#    session prefill_cap 封顶, 故 ≤512 的扫点经 --prefill-chunk 依旧成立, >512 扫不动)
set -euo pipefail
MODEL=${MODEL:-gguf/ds4-allq2.gguf}
ZCHAIN=${ZCHAIN:-gguf/go-onebit/r30/full86/zchain_noge.bin}
IDS=${IDS:-/tmp/chunk_nll.ids}
CHUNKS=${CHUNKS:-"512 16 4 1"}
OUT=${OUT:-/tmp/chunk_nll}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
for C in $CHUNKS; do
  echo "=== chunk=$C ===" >&2
  ./ds4 --cuda -m "$MODEL" --eval-ids "$IDS" --eval-logits "$OUT/lg_$C.bin" --prefill-chunk "$C" \
      ${ZCHAIN:+--zchain "$ZCHAIN"} ${EXTRA:-} -n 1 -p x </dev/null \
      >"$OUT/run_$C.log" 2>&1 || { tail -5 "$OUT/run_$C.log"; exit 1; }
done
python3 - "$IDS" "$OUT" $CHUNKS <<'PY'
import sys, os, re, numpy as np
ids_path, out = sys.argv[1], sys.argv[2]
chunks = sys.argv[3:]
ids = [int(x) for x in open(ids_path).read().split()]
ref = None
for c in chunks:
    p = os.path.join(out, f"lg_{c}.bin")
    sz = os.path.getsize(p)
    # 引擎按 [EVAL_IDS] S= 决定实际序列长(可能首插 BOS), 从 run 日志取真长度
    log = open(os.path.join(out, f"run_{c}.log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"S=(\d+)", log)
    n = int(m.group(1)) if m else len(ids)
    seq = [int(x) for x in re.search(r"首8: ([\d ]+)", log).group(1).split()] if False else None
    vocab = sz // 4 // n
    assert sz == n * vocab * 4, (sz, n, vocab)
    tgt_ids = ([1] if n == len(ids) + 1 else []) + ids  # BOS 占位, 只用于对齐 next-token
    lg = np.fromfile(p, dtype=np.float32).reshape(n, vocab)
    x = lg[:-1].astype(np.float64)
    x -= x.max(1, keepdims=True)
    lse = np.log(np.exp(x).sum(1))
    tgt = np.array(tgt_ids[1:])
    nll = float((lse - x[np.arange(len(tgt)), tgt]).mean())
    if ref is None: ref = lg
    d = float(np.abs(lg - ref).max())
    print(f"chunk={c:>4} NLL={nll:.4f} PPL={np.exp(nll):8.3f} max|Δlogit vs 首个|={d:.4f}")
PY
