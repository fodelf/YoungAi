#!/bin/sh
# e7_inject_follow.sh — pipeline stage 2 tailing stage 1: as student_traj lands
# each deep layer's ffn_in_L{n}.npy, run teacher_inject on THAT layer (teacher
# routed targets on student inputs). Single extra process, nice'd, ~2G RSS —
# fits beside the running student_traj under the 9G line.
#   usage (on M1): e7_inject_follow.sh STUDENT_CAP "20 21 22 25 26 ... 42"
set -u
ROOT=${ROOT:-/Users/fodelf/ds4-main}
CAP=${1:?student cap dir}
LAYERS=${2:?layer list}
HF=${HF:-$ROOT/hf/DeepSeek-V4-Flash-Base}
PY=$ROOT/cap_work/venv/bin/python
for L in $LAYERS; do
    # stage-1 writes ffn_in_L$L.npy only after finishing layer $L for ALL
    # chunks (layer-major run) — poll for it, then inject this layer.
    until [ -f "$CAP/ffn_in_L$L.npy" ]; do sleep 60; done
    sleep 10   # np.save is not atomic; the layer-major loop is well past by now
    env DS4_HF="$HF" nice -n 10 "$PY" \
        "$ROOT/gguf-tools/go-onebit/pyfwd/teacher_inject.py" --cap "$CAP" --layers "$L" \
        >> /tmp/inject_follow.log 2>&1 || echo "INJECT-FAIL L$L" >> /tmp/inject_follow.log
    echo "L$L injected" >> /tmp/inject_follow.log
done
echo "INJECT-FOLLOW DONE" >> /tmp/inject_follow.log
