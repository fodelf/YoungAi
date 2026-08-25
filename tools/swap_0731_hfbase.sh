#!/bin/sh
# Two-way, volume-matched swap between the Macs over the Thunderbolt bridge:
#
#   M4  hf/DeepSeek-V4-Flash-0731/*   ->  M1  (so M1 holds the whole 0731)
#   M1  hf-base/DeepSeek-V4-Flash-Base/*.safetensors  ->  M4
#
# Both directions are interleaved so each machine's disk usage stays roughly
# flat: after every file the script picks whichever direction has moved fewer
# bytes so far. Neither disk has room to take one side wholesale first (M4 has
# ~15 GiB free, M1 ~47), so the pairing is what makes this possible at all.
#
# MOVE semantics, done safely: copy -> verify byte size on the destination ->
# only then delete the source. A failed or short transfer leaves the source
# untouched and the script stops. Re-running resumes (rsync --partial, and
# already-delivered files are skipped by the size check).
#
#   ./tools/swap_0731_hfbase.sh --plan    # show the pairing, move nothing
#   ./tools/swap_0731_hfbase.sh           # do it
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NAME=${NAME:-DeepSeek-V4-Flash-0731}
BASE_NAME=${BASE_NAME:-DeepSeek-V4-Flash-Base}
REMOTE_SSH=${REMOTE_SSH:-fodelf@192.168.1.2}
REMOTE_ROOT=${REMOTE_ROOT:-/Users/fodelf/ds4-main}

L_0731="$ROOT/hf/$NAME"                       # source A (M4)
R_0731="$REMOTE_ROOT/hf/$NAME"                # dest   A (M1)
R_BASE="$REMOTE_ROOT/hf-base/$BASE_NAME"      # source B (M1)
L_BASE="$ROOT/hf-base/$BASE_NAME"             # dest   B (M4)

# Keep this much free on each disk; a transfer that would break it waits/stops.
MIN_FREE_GIB=${MIN_FREE_GIB:-6}

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
PLAN_ONLY=0
[ "$1" = "--plan" ] && PLAN_ONLY=1

human() { awk -v b="$1" 'BEGIN { printf "%.2f GiB", b/1073741824 }'; }
lfree()  { df -Pg "$1" | awk 'NR==2{print $4}'; }
rfree()  { $SSH "$REMOTE_SSH" "df -Pg '$1'" </dev/null 2>/dev/null | awk 'NR==2{print $4}'; }
lsize()  { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo 0; }
rsize()  { $SSH "$REMOTE_SSH" "wc -c < '$1' 2>/dev/null" </dev/null 2>/dev/null | tr -d ' ' || echo 0; }

# --- refuse to run while the download is still writing these dirs ------------
parts=$(find "$L_0731" -name '*.part' -o -name '*.rr' 2>/dev/null | wc -l | tr -d ' ')
rparts=$($SSH "$REMOTE_SSH" "ls '$R_0731'/*.part 2>/dev/null | wc -l" </dev/null 2>/dev/null | tr -d ' ')
if [ "${parts:-0}" != 0 ] || [ "${rparts:-0}" != 0 ]; then
    echo "REFUSING: download still in flight (M4 .part/.rr=$parts, M1 .part=$rparts)." >&2
    echo "Let it finish first — moving a file being written would corrupt it." >&2
    exit 1
fi
if pgrep -f download_base_model.sh >/dev/null 2>&1; then
    echo "REFUSING: download_base_model.sh is still running." >&2
    exit 1
fi

$SSH "$REMOTE_SSH" "mkdir -p '$R_0731'" </dev/null
mkdir -p "$L_BASE"

# --- build both lists --------------------------------------------------------
ALIST=$(mktemp); BLIST=$(mktemp)
find "$L_0731" -type f ! -name '.DS_Store' -exec stat -f '%z %N' {} + 2>/dev/null \
    | awk -v p="$L_0731/" '{ sz=$1; nm=substr($0, index($0,$2)); sub(p,"",nm); print sz "\t" nm }' | sort -k2 > "$ALIST"
$SSH "$REMOTE_SSH" "find '$R_BASE' -type f -name '*.safetensors' -exec stat -f '%z %N' {} + 2>/dev/null" </dev/null \
    | awk -v p="$R_BASE/" '{ sz=$1; nm=substr($0, index($0,$2)); sub(p,"",nm); print sz "\t" nm }' | sort -k2 > "$BLIST"

atot=$(awk -F'\t' '{s+=$1} END{printf "%.0f", s+0}' "$ALIST")
btot=$(awk -F'\t' '{s+=$1} END{printf "%.0f", s+0}' "$BLIST")
na=$(wc -l < "$ALIST" | tr -d ' '); nb=$(wc -l < "$BLIST" | tr -d ' ')

echo "A: M4 $L_0731  ->  M1"
echo "   $na files, $(human "$atot")"
echo "B: M1 $R_BASE  ->  M4"
echo "   $nb files, $(human "$btot")  (only ~$(human "$atot") of it will move, to stay volume-matched)"
echo "free now: M4 $(lfree "$ROOT") GiB / M1 $(rfree "$REMOTE_ROOT") GiB   keep >= $MIN_FREE_GIB GiB"
echo

if [ "$PLAN_ONLY" = 1 ]; then
    echo "-- A (to M1) --"; awk -F'\t' '{ printf "  %8.2f GiB  %s\n", $1/1073741824, $2 }' "$ALIST"
    echo "-- B (to M4), taken in order until volume matches A --"
    awk -F'\t' '{ printf "  %8.2f GiB  %s\n", $1/1073741824, $2 }' "$BLIST" | head -20
    rm -f "$ALIST" "$BLIST"; exit 0
fi

# --- interleaved move --------------------------------------------------------
moved_a=0; moved_b=0; ai=1; bi=1
fail() { echo "STOP: $*" >&2; rm -f "$ALIST" "$BLIST"; exit 1; }

move_a() { # M4 -> M1, then delete local
    line=$(sed -n "${ai}p" "$ALIST"); [ -n "$line" ] || return 1
    size=${line%%	*}; name=${line#*	}
    ai=$((ai + 1))
    src="$L_0731/$name"; dst="$R_0731/$name"
    if [ "$(rsize "$dst")" = "$size" ]; then
        rm -f "$src"; moved_a=$((moved_a + size)); echo "A ok(already) $name"; return 0
    fi
    need=$(awk -v s="$size" -v m="$MIN_FREE_GIB" 'BEGIN{ printf "%.0f", s/1073741824+m }')
    # Not enough room yet is not an error: the other direction frees space, so
    # step back and let the caller run it. Only a real transfer fault stops us.
    if [ "$(rfree "$REMOTE_ROOT")" -lt "$need" ]; then ai=$((ai - 1)); return 2; fi
    echo "A ->M1 $name ($(human "$size"))"
    $SSH "$REMOTE_SSH" "mkdir -p \"\$(dirname '$dst')\"" </dev/null
    rsync -e "$SSH" --partial --inplace "$src" "$REMOTE_SSH:$dst" </dev/null || fail "rsync failed: $name"
    [ "$(rsize "$dst")" = "$size" ] || fail "size mismatch after copy: $name (source kept)"
    rm -f "$src"
    moved_a=$((moved_a + size))
}

move_b() { # M1 -> M4, then delete remote
    line=$(sed -n "${bi}p" "$BLIST"); [ -n "$line" ] || return 1
    size=${line%%	*}; name=${line#*	}
    bi=$((bi + 1))
    src="$R_BASE/$name"; dst="$L_BASE/$name"
    if [ "$(lsize "$dst")" = "$size" ]; then
        $SSH "$REMOTE_SSH" "rm -f '$src'" </dev/null; moved_b=$((moved_b + size)); echo "B ok(already) $name"; return 0
    fi
    need=$(awk -v s="$size" -v m="$MIN_FREE_GIB" 'BEGIN{ printf "%.0f", s/1073741824+m }')
    if [ "$(lfree "$ROOT")" -lt "$need" ]; then bi=$((bi - 1)); return 2; fi
    echo "B ->M4 $name ($(human "$size"))"
    rsync -e "$SSH" --partial --inplace "$REMOTE_SSH:$src" "$dst" </dev/null || fail "rsync failed: $name"
    [ "$(lsize "$dst")" = "$size" ] || fail "size mismatch after copy: $name (source kept)"
    $SSH "$REMOTE_SSH" "rm -f '$src'" </dev/null
    moved_b=$((moved_b + size))
}

while [ "$ai" -le "$na" ]; do
    # Whichever direction is behind on bytes goes next, so neither disk drifts.
    # A direction that has no room yet returns 2; we fall back to the other one,
    # which is exactly what frees the space it was waiting for.
    if [ "$moved_a" -le "$moved_b" ]; then
        move_a || { [ "$?" = 2 ] && { move_b || true; }; }
    else
        move_b || { move_a || true; }
    fi
    echo "   moved A $(human "$moved_a") | B $(human "$moved_b") | free M4 $(lfree "$ROOT")G M1 $(rfree "$REMOTE_ROOT")G"
done
# Top B up so the two directions end volume-matched.
while [ "$moved_b" -lt "$moved_a" ] && [ "$bi" -le "$nb" ]; do
    move_b || break
    echo "   moved A $(human "$moved_a") | B $(human "$moved_b") | free M4 $(lfree "$ROOT")G M1 $(rfree "$REMOTE_ROOT")G"
done

rm -f "$ALIST" "$BLIST"
echo
echo "done. A moved $(human "$moved_a"), B moved $(human "$moved_b")"
echo "M4 free $(lfree "$ROOT") GiB / M1 free $(rfree "$REMOTE_ROOT") GiB"
