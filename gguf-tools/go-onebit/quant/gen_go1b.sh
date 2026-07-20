#!/bin/sh
# Generate the go1b (strict-1-bit routed experts) GGUF from a source HF model, with RSS watchdog.
# Args: $1=HF_dir $2=TEMPLATE_GGUF $3=OUT.gguf [$4=THREADS(6)]
# TEMPLATE provides GGUF metadata/tensor structure ONLY (weights re-quantized from HF; template
# weight data is never read). A published GGUF's header (first ~200 MiB) suffices as template:
#   curl -L -r 0-209715199 \
#     "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" \
#     -o /tmp/tmpl_hdr.gguf
# Output ~45.6 GiB (experts go1b ~1.06 bit + backbone at the template's precision). RSS peak ~1 GiB.
HF="$1"; TMPL="$2"; OUT="$3"; THREADS="${4:-6}"
Q="$(cd "$(dirname "$0")/../.." && pwd)/deepseek4-quantize"
[ -x "$Q" ] || { echo "build first: make -C $(dirname "$Q") deepseek4-quantize"; exit 1; }
# IMATRIX (optional env): llama.cpp-style .dat with per-expert segmented E[x²]
# stats -> L_fix input-aware go1b scale s* = Σ E[x²]|W| / Σ E[x²]; plain
# mean|w| rows when unset.
IMAT_ARGS=""
[ -n "${IMATRIX:-}" ] && IMAT_ARGS="--imatrix $IMATRIX"
echo "free before: $(df -h "$(dirname "$OUT")" | tail -1 | awk '{print $4}')"
"$Q" --hf "$HF" --template "$TMPL" --experts go1b $IMAT_ARGS --out "$OUT" --threads "$THREADS" > /tmp/gen_go1b.log 2>&1 &
QPID=$!; echo "quantizer pid=$QPID watchdog@11.5GiB"; PEAK=0
while kill -0 "$QPID" 2>/dev/null; do
  RSS=$(ps -o rss= -p "$QPID" 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ]; then
    [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
    if [ "$RSS" -gt 11500000 ]; then echo "WDKILL RSS=${RSS}KB"; kill -9 "$QPID"; break; fi
  fi
  sleep 15
done
wait "$QPID" 2>/dev/null
echo "=== quantizer exit=$? peak_rss=$((PEAK/1024))MiB ==="
tail -5 /tmp/gen_go1b.log
ls -la "$OUT" 2>/dev/null | awk '{print "OUTPUT:",$5,$NF}'
