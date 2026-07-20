#!/bin/sh
# Poll dual-host z compute progress + RSS (manual watchdog: kill if RSS exceeds budget).
# Run on M4 (coordinator). Args: $1=M1_host(192.168.1.2) $2=ROOT(/Users/fodelf/git/ds4-main)
M1="${1:-192.168.1.2}"; ROOT="${2:-/Users/fodelf/git/ds4-main}"
echo "=== M4 ==="
P=$(pgrep -f "calib_run.*cap_m4" | head -1)
if [ -n "$P" ]; then
  R=$(ps -o rss= -p "$P" | awk '{print int($1/1024)}')
  echo "RUNNING pid=$P RSS=${R}MiB"; [ "$R" -gt 13000 ] && { echo "OVER BUDGET -> kill"; kill -9 "$P"; }
else echo "done/dead"; fi
echo "M4 z dumped: $(grep -c 'z dumped' /tmp/calib_z_m4.log 2>/dev/null)"
echo "=== M1 ($M1) ==="
ssh "$M1" 'P=$(pgrep -f calib_run|head -1); if [ -n "$P" ]; then R=$(ps -o rss= -p "$P"|awk "{print int(\$1/1024)}"); echo "RUNNING pid=$P RSS=${R}MiB"; [ "$R" -gt 13000 ] && { echo "OVER BUDGET -> kill"; kill -9 "$P"; }; else echo "done/dead"; fi; echo "M1 z dumped: $(grep -c "z dumped" /tmp/calib_z.log 2>/dev/null)"'
echo "=== total z files in M4 zdump: $(ls "$ROOT"/zdump/z_L*.bin 2>/dev/null | wc -l | tr -d ' ')/43 ==="
