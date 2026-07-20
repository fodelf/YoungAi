#!/bin/sh
# e7_quad.sh — E7 four-arm end-to-end verdict: bare Θ_fix vs +corr sidecar on
# the same held-out slice (teacher-forced NLL) and the same greedy Go prompt
# (generation + decode t/s). Arms run sequentially (ds4 instance lock).
#   usage: e7_quad.sh [CORR=gguf/ds4-go1b-corr-rrr.gguf] [EVAL=go_heldout_600.txt]
set -u
ROOT=${ROOT:-/Users/fodelf/git/ds4-main}
MODEL=${MODEL:-$ROOT/gguf/ds4-go1b.gguf}
CORR=${1:-$ROOT/gguf/ds4-go1b-corr-rrr.gguf}
EVAL=${2:-$ROOT/gguf-tools/go-onebit/go_heldout_600.txt}
NTOK=${NTOK:-24}
PROMPT='<｜begin▁of▁sentence｜>package main

import "fmt"

func Add(a, b int) int {'

cd "$ROOT" || exit 1
echo "=== ARM1 bare NLL ($EVAL) ==="
./ds4 -m "$MODEL" --perplexity-file "$EVAL" --metal 2>&1 | grep -E 'avg_nll|tokens='
echo "=== ARM2 bare gen ==="
./ds4 -m "$MODEL" -p "$PROMPT" -n "$NTOK" --temp 0 --metal 2>&1 | grep -vE '^ds4: (Metal|registered|metal|context|routed-expert|q2)' | tail -6
echo "=== ARM3 corr NLL ($CORR) ==="
./ds4 -m "$MODEL" --corr "$CORR" --perplexity-file "$EVAL" --metal 2>&1 | grep -E 'avg_nll|tokens='
echo "=== ARM4 corr gen ==="
./ds4 -m "$MODEL" --corr "$CORR" -p "$PROMPT" -n "$NTOK" --temp 0 --metal 2>&1 | grep -vE '^ds4: (Metal|registered|metal|context|routed-expert|q2)' | tail -6
echo "=== E7-QUAD-DONE ==="
