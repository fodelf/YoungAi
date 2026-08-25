#!/bin/sh
# lib.sh — single-source runtime for every go-onebit pipeline script.
# Kills the accident classes measured on 2026-07-02/03:
#   env-default drift (5), bare ssh/scp hangs (2), silent black boxes (3).
# Usage (top of every script):   . "$(dirname "$0")/lib.sh"
#
# Provides:
#   $GO_ROOT $REPO_ROOT      resolved from the script location, never guessed
#   conf [FILE]              load pipeline.conf (default $GO_ROOT/pipeline.conf)
#   rsh HOST CMD... / rcp SRC DST
#                            ssh/scp with BatchMode+ConnectTimeout, fail-fast
#   progress MSG             one unbuffered stderr line with timestamp+elapsed
#   die MSG                  progress + exit 1
#   require VAR...           die unless every named variable is non-empty
#   atomic_out PATH          echo a tmp path; call atomic_land PATH after write
#   atomic_land PATH         rename tmp -> final (crash-safe artifact landing)

# 批4 重组: go-onebit 目录消亡, GO_ROOT 现指 gguf-tools(cluster 脚本在 scripts/cluster/ 下两层)。
GO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REPO_ROOT=$(cd "$GO_ROOT/.." && pwd)
_LIB_T0=$(date +%s)

conf() {
    _c=${1:-$GO_ROOT/legacy/pipeline.conf}
    # shellcheck disable=SC1090
    [ -f "$_c" ] && . "$_c"
}

rsh() {
    _h=$1; shift
    ssh -o BatchMode=yes -o ConnectTimeout=12 "$_h" "$@"
}

rcp() {
    scp -o BatchMode=yes -o ConnectTimeout=12 -q "$@"
}

progress() {
    printf '[%s +%ss] %s\n' "$(basename "$0")" "$(( $(date +%s) - _LIB_T0 ))" "$*" >&2
}

die() { progress "FATAL: $*"; exit 1; }

require() {
    for _v in "$@"; do
        eval "_x=\${$_v:-}"
        [ -n "$_x" ] || die "required variable $_v is empty (no silent defaults)"
    done
}

atomic_out() { printf '%s.tmp.%s' "$1" "$$"; }
atomic_land() { mv -f "$(atomic_out "$1")" "$1"; }
