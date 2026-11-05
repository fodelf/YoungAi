# Request memory and lifetime reproduction

The memory changes apply to Metal and CUDA. Workspace rows follow the uncached
suffix; decode releases prefill rows while preserving KV and history. Failed
forwards close their command batch and request cleanup drains outstanding work.
Metal tensor APIs scope autoreleased references; Engram raw buffers retain their
owners. Upload ordering and Q4_K/attention arithmetic are a separate change.

## Model-free checks

```sh
make -j6 ds4 ds4-server ds4_test
make linecount
./ds4_test --metal-tensor-lifetime --metal-v41-memory --server --engine-units --rax --tp-allreduce
```

`--metal-v41-memory` covers prefix matching before allocation, failed-copy cold
fallback, KV/history preservation during shrink, and failed-forward recovery.
The large synthetic prefixes test capacity contracts, not full-model quality.
To reproduce the autorelease leak, install test-only probes in a clean baseline:

```sh
python3 tests/repro/metal_baseline_probes.py /absolute/path/to/baseline
```

## Full-model checks

Use one model process at a time. Run the same model, sidecar, optional posttrain,
Engram data, and request settings on baseline and candidate. On 128 GiB hosts use
`--mem-budget-mb 110000`; retain the external memory watchdog. Start from the
checkout root, supplying absolute model/data paths:

```sh
cc -O2 tests/repro/metal_memory_probe.c -o /tmp/youngai-memory-probe
python3 tests/repro/metal_guard.py --probe /tmp/youngai-memory-probe \
  --out /tmp/youngai-memory-run --process-gib 120 --wired-gib 120 \
  --swap-mib 8192 --compressed-gib 24 -- \
  ./ds4-server --metal -m "$MODEL_PATH" --zchain "$ZCHAIN_PATH" \
  --engram-dir "$ENGRAM_PATH" --mem-budget-mb 110000 \
  --host 127.0.0.1 --port 19010 --batch 0 --vision-gguf off
python3 tests/repro/metal_api.py --url http://127.0.0.1:19010 --out /tmp/quick --suite quick
python3 tests/repro/metal_api.py --url http://127.0.0.1:19010 --out /tmp/questions --suite questions
python3 tests/repro/metal_api.py --url http://127.0.0.1:19010 --out /tmp/lifetime --suite lifetime
```

For CUDA, use the maintainer's `speed-bench/pr6_verify_spark.sh` from current
`restructure`; the watchdog above is macOS-only. Compare completed output hashes,
workspace rows, cleanup allocations, cache reuse, and disconnect recovery. Record skips and failures
separately. Keep raw logs and timing reports outside the source tree; report the
exact configuration and results in the PR. Full-model Metal numeric parity is a
separate gate and is not implied by the memory checks.

`fable5.md` is maintained by the upstream maintainer; this contribution leaves it unchanged.
