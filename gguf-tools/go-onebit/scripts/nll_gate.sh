#!/bin/sh
# nll_gate.sh — teacher-forced NLL gate on the Go heldout anchor slice.
# Usage: nll_gate.sh [corr.gguf|-] [expected_avg_nll]
#   corr '-' = bare Θ_fix. With expected value, exits 1 on mismatch (bit-exact
#   string compare of the printed avg_nll — the corr-delta/knob paths must not
#   move it by even one ulp).
# Free knobs default ON (token-byte-exact, 2.2x); override via env.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || exit 1
CORR=${1:--}
EXPECT=${2:-}
HELDOUT=${HELDOUT:-gguf-tools/go-onebit/go_heldout_300.txt}
MODEL=${MODEL:-gguf/ds4-go1b.gguf}
LOG=/tmp/nll_gate_$$.log
if [ "$CORR" = "-" ]; then
    ./ds4 -m "$MODEL" --perplexity-file "$HELDOUT" --metal > "$LOG" 2>&1
else
    ./ds4 -m "$MODEL" --corr "$CORR" --perplexity-file "$HELDOUT" --metal > "$LOG" 2>&1
fi
LINE=$(grep -a 'avg_nll' "$LOG" | tail -1)
echo "$LINE"
NLL=$(printf '%s\n' "$LINE" | sed -n 's/.*avg_nll[= ]*\([0-9.]*\).*/\1/p')
if [ -n "$EXPECT" ]; then
    if [ "$NLL" = "$EXPECT" ]; then echo "NLL-GATE PASS ($NLL)"; else
        echo "NLL-GATE FAIL got=$NLL expect=$EXPECT (full log: $LOG)"; exit 1; fi
fi
