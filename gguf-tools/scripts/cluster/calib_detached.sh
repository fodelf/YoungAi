#!/bin/sh
# Launch calib_run DETACHED (subshell + nohup -> reparented to init; survives the harness
# killing the launching shell). No in-process watchdog (the wrapper would be killed); monitor
# RSS via poll_z.sh and kill manually if it exceeds budget. nx=256 peaks ~2-6 GiB (safe).
# Args: $1=gguf-tools-dir $2=LAYERS(comma) $3=CAP_dir $4=HF_dir $5=ZDIR [$6=NX(256)] [$7=LOG(/tmp/calib_z.log)]
cd "$1" || exit 1
NX="${6:-256}"; LOG="${7:-/tmp/calib_z.log}"
rm -f "$LOG"
( nohup ./calib_run --hf "$4" --cap "$3" --layers "$2" --z-dump-dir "$5" --solver "${SOLVER:-hv}" ${EXTRA_ARGS:-} --nx "$NX" --threads 8 > "$LOG" 2>&1 & )
sleep 3
P=$(pgrep -f "calib_run" | head -1)
[ -n "$P" ] && echo "calib_run DETACHED pid=$P RSS=$(ps -o rss= -p "$P"|awk '{print int($1/1024)"MiB"}') layers=$2" \
            || { echo "launch FAILED:"; head -3 "$LOG"; }
