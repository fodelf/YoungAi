#!/bin/sh
# Run calib_run (per-layer four-loss z solve) for a layer range, with RSS watchdog @13GiB.
# Foreground/blocking variant (use calib_detached.sh under an unstable harness that kills launchers).
# Args: $1=gguf-tools-dir $2=LAYERS(comma e.g. "0,1,2") $3=CAP_dir $4=HF_dir $5=ZDIR [$6=NX(default 256)]
cd "$1" || exit 1
NX="${6:-256}"
echo "host calib: layers=$2 cap=$3 zdir=$5 nx=$NX"
./calib_run --hf "$4" --cap "$3" --layers "$2" --z-dump-dir "$5" --solver "${SOLVER:-hv}" ${EXTRA_ARGS:-} --nx "$NX" --threads 8 > /tmp/calib_z.log 2>&1 &
QPID=$!
echo "calib_run pid=$QPID watchdog@13GiB"
PEAK=0
while kill -0 "$QPID" 2>/dev/null; do
  RSS=$(ps -o rss= -p "$QPID" 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ]; then
    [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
    if [ "$RSS" -gt 13000000 ]; then echo "WDKILL RSS=${RSS}KB"; kill -9 "$QPID"; break; fi
  fi
  sleep 20
done
wait "$QPID" 2>/dev/null
echo "=== calib exit=$? peak_rss=$((PEAK/1024))MiB ==="
echo "total z files in zdir: $(ls "$5"/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')"
tail -3 /tmp/calib_z.log
