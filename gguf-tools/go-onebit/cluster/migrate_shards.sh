#!/bin/sh
# migrate_shards.sh — copy this host's HF safetensors shards for a LAYER RANGE
# to another mac. Generalizes the hand-run done 2026-07-04 (push layers 0-3 =
# model-0000{2,3,4,5} to M1:~/ds4-main/hf4).
#
# COPY-ONLY BY DESIGN: it never deletes the source. Dropping a local layer
# shard would break this host's own forward/quantize half (M4 holds L0-11),
# and the project's guardrail forbids destructive "move". So "migrate" here
# means replicate; delete the source yourself only if you are certain.
#
#   usage:  migrate_shards.sh LAYERS [DEST_HOST] [DEST_DIR]
#     LAYERS     layer spec: "0-3" | "0,1,2" | "0-3,10-12"
#     DEST_HOST  ssh host   (default $M1 from pipeline.conf = 192.168.1.2)
#     DEST_DIR   remote dir (default $ROOT_M1/hf4)
#   env:
#     HF_SRC          source HF dir on THIS host
#                     (default: <repo>/hf/DeepSeek-V4-Flash-Base — NOT conf's
#                      HF_DIR, which points at the M1 path)
#     INCLUDE_EMBED=1  also send the embed_tokens shard (model-00001)
#     INCLUDE_OUTPUT=1 also send the tail shards (final norm / lm_head / mtp)
#     FREE_MARGIN_GB   refuse if dest free < bytes + margin (default 5)
#     DRY_RUN=1        print the plan (files, sizes, dest free) and stop
#
#   background it yourself:  nohup sh migrate_shards.sh 0-3 >/tmp/mig.log 2>&1 &
#
# Resumable: rsync --partial --inplace; a killed run continues the partial
# file. Verifies every shard's byte size on the far side before declaring done.
. "$(dirname "$0")/lib.sh"
conf                      # pull M1 / ROOT_M1 defaults (ignore conf's HF_DIR)

set -u
LAYERS=${1:?layer spec like "0-3" or "0-3,10-12"}
DEST_HOST=${2:-${M1:-192.168.1.2}}
DEST_DIR=${3:-${ROOT_M1:-/Users/fodelf/ds4-main}/hf4}
SRC=${HF_SRC:-$REPO_ROOT/hf/DeepSeek-V4-Flash-Base}
INDEX="$SRC/model.safetensors.index.json"
FREE_MARGIN_GB=${FREE_MARGIN_GB:-5}

[ -d "$SRC" ]   || die "source HF dir not found on this host: $SRC"
[ -f "$INDEX" ] || die "shard index not found: $INDEX"

gib() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1073741824}'; }

# Resolve the layer spec -> shard filenames straight from the weight_map, so we
# never hardcode a layer->shard offset (robust to any repacking).
FILES=$(INDEX="$INDEX" LAYERS="$LAYERS" \
        INCLUDE_EMBED="${INCLUDE_EMBED:-0}" INCLUDE_OUTPUT="${INCLUDE_OUTPUT:-0}" \
        python3 - <<'PY'
import json, os, re, sys
wm = json.load(open(os.environ["INDEX"]))["weight_map"]
want = set()
for part in os.environ["LAYERS"].split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        a, b = part.split("-"); want.update(range(int(a), int(b) + 1))
    else:
        want.add(int(part))
files = set()
for name, shard in wm.items():                      # layer shards
    m = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
    if m and int(m.group(1)) in want:
        files.add(shard)
if os.environ.get("INCLUDE_EMBED") == "1":
    for name, shard in wm.items():
        if "embed" in name:
            files.add(shard)
if os.environ.get("INCLUDE_OUTPUT") == "1":         # non-layer, non-embed tail
    for name, shard in wm.items():
        if re.search(r'layers\.\d+\.', name) or "embed" in name:
            continue
        files.add(shard)
if not files:
    sys.stderr.write("no shards matched layer spec\n"); sys.exit(2)
for f in sorted(files):
    print(f)
PY
) || die "layer->shard resolution failed for spec '$LAYERS'"

require FILES
NF=$(printf '%s\n' "$FILES" | wc -l | tr -d ' ')
progress "src   = $SRC"
progress "dest  = $DEST_HOST:$DEST_DIR"
progress "layers=[$LAYERS] -> $NF shard(s)"

# Create dest dir, then snapshot the sizes already there so the space guard can
# count only the NET new bytes (dest keeps a same-size file => 0 add; --inplace
# resume of a partial => only the remainder). One round trip, not one per file.
rsh "$DEST_HOST" "mkdir -p '$DEST_DIR'" || die "cannot mkdir $DEST_HOST:$DEST_DIR"
# $FILES is newline-separated; collapse to one line so the remote for-loop
# doesn't get its command split at the newlines.
FILES_1L=$(printf '%s ' $FILES)
DEST_SIZES=$(rsh "$DEST_HOST" "cd '$DEST_DIR' && for f in $FILES_1L; do printf '%s %s\n' \"\$f\" \"\$(stat -f%z \"\$f\" 2>/dev/null || echo 0)\"; done" 2>/dev/null || true)
dest_size() { printf '%s\n' "$DEST_SIZES" | awk -v f="$1" '$1==f{print $2; g=1} END{if(!g) print 0}'; }

TOTAL=0        # full byte size of the selection (for display)
NEED_NET=0     # bytes actually landing on dest given what it already holds
for f in $FILES; do
    p="$SRC/$f"
    [ -f "$p" ] || die "source shard missing (this host may not hold it): $f"
    s=$(stat -f%z "$p")
    d=$(dest_size "$f")
    TOTAL=$((TOTAL + s))
    add=$((s - d)); [ "$add" -lt 0 ] && add=0
    NEED_NET=$((NEED_NET + add))
    if [ "$d" = "$s" ]; then note=" (already on dest, 0 add)"
    elif [ "$d" -gt 0 ] 2>/dev/null; then note=" (resume: +$(gib "$add") GiB)"
    else note=""; fi
    progress "    $f  $(gib "$s") GiB$note"
done
progress "selection $(gib "$TOTAL") GiB; net to transfer $(gib "$NEED_NET") GiB"

# Dest disk-wall guard: refuse if the NET add won't fit with margin.
FREE_B=$(rsh "$DEST_HOST" "df -k '$DEST_DIR' | tail -1 | awk '{print \$4*1024}'") \
    || die "cannot read dest free space"
NEED=$((NEED_NET + FREE_MARGIN_GB * 1073741824))
progress "dest free $(gib "$FREE_B") GiB, need $(gib "$NEED") GiB (incl ${FREE_MARGIN_GB}G margin)"
[ "$FREE_B" -ge "$NEED" ] || die "insufficient space on $DEST_HOST:$DEST_DIR (disk-wall guard)"

if [ "${DRY_RUN:-0}" = "1" ]; then
    progress "DRY_RUN: plan only, nothing transferred"
    exit 0
fi

# Transfer (resumable, own-checksum delta). --inplace so a kill resumes cleanly.
cd "$SRC" || die "cd $SRC"
# shellcheck disable=SC2086
rsync -a --partial --inplace --progress $FILES \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=12' \
    "$DEST_HOST:$DEST_DIR/" || die "rsync failed"

# Verify every shard's size on the far side against the local source.
progress "verifying sizes on $DEST_HOST ..."
FAIL=0
for f in $FILES; do
    s=$(stat -f%z "$SRC/$f")
    d=$(rsh "$DEST_HOST" "stat -f%z '$DEST_DIR/$f' 2>/dev/null" || echo -1)
    if [ "$s" = "$d" ]; then
        progress "OK  $f ($s)"
    else
        progress "MISMATCH $f: src=$s dst=$d"; FAIL=1
    fi
done
[ "$FAIL" = "0" ] || die "size verification failed"
progress "MIGRATE-DONE: $NF shard(s) -> $DEST_HOST:$DEST_DIR"
