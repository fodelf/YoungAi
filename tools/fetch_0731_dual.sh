#!/bin/sh
# Dual-host fetch of deepseek-ai/DeepSeek-V4-Flash-0731 (155.4 GiB, 48 shards)
# into ./hf/DeepSeek-V4-Flash-0731 on BOTH Macs, via download_base_model.sh
# --repo. Both hosts pull from HF in parallel through their own clash proxy
# (--remote-mode direct), so aggregate bandwidth is doubled.
#
# Space reality (2026-08-02): the model does NOT fit even across both disks.
# M4 ~39.6 GiB free, M1 ~34.7 GiB free; clearing M1's r29 campaign artifacts
# (~36 GiB) + the hf-base .CORRUPT/.part leftovers (~6.2 GiB) gets the pair to
# ~117 GiB — still ~46 GiB short. That is expected and accepted: the downloader
# fills what fits, and re-running it after more space is freed resumes exactly
# where it stopped (existing files keep their host in the placement plan).
#
# The old Base HF tree (M1 ~/ds4-main/hf-base, 279 GiB) is NEVER touched here —
# original HF models are off-limits to automated cleanup.
#
#   ./tools/fetch_0731_dual.sh --clean    # free M1 campaign artifacts first
#   ./tools/fetch_0731_dual.sh --plan     # print the split, download nothing
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO=${REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}
# Measured 2026-08-02: through the clash proxy huggingface.co gave 0.23 MB/s on
# M4 and TLS-handshake failures for M1's (x86_64) aria2. hf-mirror.com direct,
# no proxy, is 10-100x better (M1 25-35 MB/s). It redirects to a pre-signed CDN
# URL that pins one ByteRange, so aria2's -x/-s segments 403 -> --no-aria2.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
NAME=${REPO##*/}
REMOTE_SSH=${REMOTE_SSH:-fodelf@192.168.1.2}
REMOTE_ROOT=${REMOTE_ROOT:-/Users/fodelf/ds4-main}
REMOTE_DIR="$REMOTE_ROOT/hf/$NAME"
# System volumes on both Macs: keep more headroom than the downloader's default
# 4 GiB so a full disk never destabilizes the OS.
LOCAL_RESERVE_GIB=${LOCAL_RESERVE_GIB:-8}
REMOTE_RESERVE_GIB=${REMOTE_RESERVE_GIB:-6}
LOG=${LOG:-/tmp/dl_0731.log}

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
CLEAN=0; PASS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --clean) CLEAN=1 ;;
        *) PASS="$PASS $1" ;;
    esac
    shift
done

if [ "$CLEAN" = 1 ]; then
    echo "== cleaning M1 campaign artifacts (r29) + hf-base download leftovers =="
    $SSH "$REMOTE_SSH" '
        set -e
        cd ~/ds4-main
        du -sh gguf/go-onebit/ds4-r29.gguf gguf/go-onebit/r29 2>/dev/null || true
        rm -rf gguf/go-onebit/ds4-r29.gguf gguf/go-onebit/ds4-r29.gguf.bias0.bin gguf/go-onebit/r29
        # Leftovers from the Base download: a byte-identical .CORRUPT twin and a
        # 23 MB .part, both superseded by complete shards. The shards stay.
        rm -f hf-base/DeepSeek-V4-Flash-Base/*.CORRUPT hf-base/DeepSeek-V4-Flash-Base/*.part
        df -h ~ | tail -1
    '
fi

echo "== dual-host download: $REPO =="
echo "   local : $ROOT/hf/$NAME"
echo "   remote: $REMOTE_SSH:$REMOTE_DIR"
echo "   log   : $LOG"

# caffeinate: this runs for hours; the Mac must not sleep mid-transfer.
# direct: each Mac downloads its OWN slice of the shards, concurrently.
exec caffeinate -i "$ROOT/download_base_model.sh" \
    --repo "$REPO" \
    --remote "$REMOTE_SSH:$REMOTE_DIR" \
    --remote-mode direct \
    --no-proxy --no-remote-proxy --no-aria2 \
    --local-share "${LOCAL_SHARE:-0.55}" \
    --reverse-relay \
    --local-reserve "$LOCAL_RESERVE_GIB" \
    --remote-reserve "$REMOTE_RESERVE_GIB" \
    $PASS
