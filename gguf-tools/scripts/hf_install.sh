#!/usr/bin/env bash
# install.sh — download, install and start YoungAi-DeepSeek-V4.1-Flash on an NVIDIA DGX Spark, in one command.
# (Source of the install.sh published at huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash.)
#
#   bash install.sh                  check the machine, download ~317 GB (resumable), start the server on
#                                    127.0.0.1:8000 and run a one-line smoke test
#   bash install.sh --domain code    the same, serving with the coding sidecar instead of the finance one
#                                    (domains: finance, code, law, medicine, science)
#   bash install.sh --no-start       everything except starting the server
#   bash install.sh start | stop | status
#
# Options:
#   --domain NAME      which domain sidecar the server loads (README §3): finance (default), code, law, medicine,
#                      science, or none (bare base). Every sidecar is downloaded (~40 MB each), so switching is just: stop, start --domain X
#   --dir DIR          install directory (default: $HOME/youngai); needs ~325 GB free, ~122 GB with --engram-dir
#   --engram-dir DIR   you already have the two official n-gram shards (a folder holding
#                      model-00047-of-00048.safetensors and model-00048-of-00048.safetensors): use them where they
#                      are instead of downloading 203 GB. They are only read, never moved or modified
#   --host ADDR        listen address (default 127.0.0.1; 0.0.0.0 serves your LAN)
#   --port N           listen port (default 8000)
#   --endpoint URL     Hugging Face endpoint for the downloads, e.g. https://hf-mirror.com
#   --pip-index URL    package index, used only if the `hf` CLI has to be installed first
#   --posttrain        also load the experimental post-training file (see README, §7; finance only)
#   --no-xet           download over the plain LFS channel instead of Xet (use it if downloads keep failing with
#                      "peer closed connection": some proxies cut Xet transfers, see README §8)
#
# Running it again is safe: finished files are skipped, partial downloads resume.
# Everything is written under --dir; no sudo, nothing outside it.
set -uo pipefail

REPO="wenzhouwu/YoungAi-DeepSeek-V4.1-Flash"
BASE_REPO="deepseek-ai/DeepSeek-V4.1-Flash"
GGUF="DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
GGUF_BYTES=113556639424
# The base is published as 40 parts (…gguf.part01-of-40 … part40-of-40) because a single 113.6 GB upload could not
# survive our own uplink. `split -n 40` gives the first PART_EXTRA parts one byte more than the rest:
# parts 1–24 are 2,838,915,986 bytes, parts 25–40 are 2,838,915,985. SHA256SUMS holds every hash.
NPART=40
PART_BASE=2838915985
PART_EXTRA=24
# Domain → sidecar directory in this repository. A new domain = one more line here (and in README §7).
sidecar_of() {
    case "$1" in
        finance) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine";;
        code)    echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine";;
        law)     echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine";;
        medicine) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine";;
        science) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine";;
        none)    echo "";;
        *)       return 1;;
    esac
}
DOMAINS="finance | code | law | medicine | science | none"
# ③ was solved on top of the finance sidecar (its base.fnv is that sidecar's fingerprint); the engine refuses any
# other pairing, but only after a 2-minute load — so the script refuses first.
POSTTRAIN="posttrain-experimental-20260924"; POSTTRAIN_DOMAIN="finance"
SHARDS=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors)
SHARD_BYTES=(101535150936 101537926640)
# The GGUF records these two shards by the absolute path they had on the machine that built it
# (deepseek4.engram.N.table_path), which does not exist anywhere else. The server is started with
# --engram-dir "$ENGRAM" and takes only the file names from the GGUF — so $ENGRAM is the one path that matters.
BUDGET_MB=110000      # engine refuses to start if the model does not fit this budget
START_FLOOR_MB=100000  # MemAvailable needed before loading (an idle DGX Spark shows ~118 GB)
RUN_FLOOR_MB=10000     # MemAvailable needed after loading; below it the machine starts swapping and stalls
KILL_MB=2500           # watchdog: two samples in a row below this and the server is stopped

CMD="install"; DIR="$HOME/youngai"; HOST="127.0.0.1"; PORT=8000; ENDPOINT=""; PIP_INDEX=""; WITH_PT=0; START=1; NO_XET=0
ENGRAM_SRC=""; DOMAIN="finance"
while [ $# -gt 0 ]; do
    case "$1" in
        install|start|stop|status) CMD="$1"; shift;;
        --domain) DOMAIN="$2"; shift 2;;
        --dir) DIR="$2"; shift 2;;
        --engram-dir) ENGRAM_SRC="$2"; shift 2;;
        --host) HOST="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --endpoint) ENDPOINT="$2"; shift 2;;
        --pip-index) PIP_INDEX="$2"; shift 2;;
        --posttrain) WITH_PT=1; shift;;
        --no-xet) NO_XET=1; shift;;
        --no-start) START=0; shift;;
        -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0;;   # = the header above
        *) echo "unknown argument: $1 (see --help)"; exit 2;;
    esac
done
MODEL="$DIR/model"; ENGRAM="$DIR/deepseek-engram"; LOGS="$DIR/logs"; PIDF="$DIR/server.pid"; WDPIDF="$DIR/watchdog.pid"
# What the running server was started with ("domain code on 127.0.0.1:8011"). status/start read it from here, not
# from their own command line: `status` without --port would otherwise report the default port, not the real one.
INFOF="$DIR/server.info"

say() { echo "[youngai $(date +%H:%M:%S)] $*"; }
die() { echo "[youngai] ERROR: $*" >&2; exit 1; }
SIDECAR=$(sidecar_of "$DOMAIN") || die "unknown --domain '$DOMAIN' (choose $DOMAINS)"
[ "$WITH_PT" = 0 ] || [ "$DOMAIN" = "$POSTTRAIN_DOMAIN" ] \
    || die "--posttrain was solved on the $POSTTRAIN_DOMAIN sidecar and cannot be stacked on --domain $DOMAIN"
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
fsize() { stat -L -c %s "$1" 2>/dev/null || echo 0; }
part_name() { printf '%s.part%02d-of-%d' "$GGUF" "$1" "$NPART"; }
parts_len() { local k="$1"; echo $(( k * PART_BASE + (k < PART_EXTRA ? k : PART_EXTRA) )); }   # bytes in parts 1..k
alive() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

preflight() {
    [ "$(uname -s)/$(uname -m)" = "Linux/aarch64" ] || die "this package ships aarch64 Linux binaries for DGX Spark; this machine is $(uname -s)/$(uname -m)"
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver"
    local gpu; gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$gpu" in *GB10*) ;; *) die "GPU is '$gpu'; the binaries are compiled for GB10 (sm_121) only";; esac
    local ldc; ldc=$(command -v ldconfig || echo /sbin/ldconfig)
    for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
        "$ldc" -p | grep -q "$lib" || die "$lib not found: the CUDA 13 runtime is required (DGX OS ships it)"
    done
    local total; total=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
    [ "$total" -ge 115000 ] || die "only $total MB of memory; the model needs a 128 GB DGX Spark"
    say "machine ok: $gpu, CUDA 13 runtime, $total MB memory"
}

ensure_hf() {
    if command -v hf >/dev/null; then HF=hf; return; fi
    HF="$DIR/.hfenv/bin/hf"
    [ -x "$HF" ] && return
    say "installing the Hugging Face CLI into $DIR/.hfenv (the system Python is left alone)"
    mkdir -p "$DIR"
    python3 -m venv "$DIR/.hfenv" || die "python3 venv failed; install it with: sudo apt install python3-venv"
    "$DIR/.hfenv/bin/pip" install -q ${PIP_INDEX:+-i "$PIP_INDEX"} -U "huggingface_hub[hf_xet]" \
        || die "pip install huggingface_hub failed (behind a firewall? try --pip-index https://pypi.tuna.tsinghua.edu.cn/simple)"
}

engram_ok() {   # both shards in $ENGRAM, with the right sizes
    [ "$(fsize "$ENGRAM/${SHARDS[0]}")" = "${SHARD_BYTES[0]}" ] && [ "$(fsize "$ENGRAM/${SHARDS[1]}")" = "${SHARD_BYTES[1]}" ]
}

# --engram-dir: make $ENGRAM a link to the user's own copy. Checked before check_disk so its 203 GB are not
# counted as still to download. Only a link we made ourselves is ever replaced; a real folder is left alone.
use_engram_src() {
    [ -n "$ENGRAM_SRC" ] || return 0
    local src i
    src=$(cd "$ENGRAM_SRC" 2>/dev/null && pwd -P) || die "--engram-dir $ENGRAM_SRC is not a directory"
    for i in 0 1; do
        [ "$(fsize "$src/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] || die "$src/${SHARDS[$i]} is missing or not ${SHARD_BYTES[$i]} bytes"
    done
    mkdir -p "$DIR"
    if [ -L "$ENGRAM" ]; then ln -sfn "$src" "$ENGRAM"
    elif [ -e "$ENGRAM" ]; then die "$ENGRAM is a folder from an earlier download; to use --engram-dir instead, remove it first (rm -r $ENGRAM)"
    else ln -s "$src" "$ENGRAM"; fi
    say "n-gram tables: using the official shards in $src (linked as $ENGRAM)"
}

check_disk() {
    local need=0 free
    [ "$(fsize "$MODEL/$GGUF")" = "$GGUF_BYTES" ] || need=$((need + GGUF_BYTES + PART_BASE + 1 - $(fsize "$MODEL/$GGUF.merging")))
    if ! engram_ok; then
        for i in 0 1; do [ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] || need=$((need + SHARD_BYTES[i])); done
    fi
    mkdir -p "$DIR"
    free=$(df -B1 --output=avail "$DIR" | tail -1)
    [ "$free" -gt $((need + 5000000000)) ] || die "need $((need / 1000000000)) GB more on $(df --output=target "$DIR" | tail -1), only $((free / 1000000000)) GB free (use --dir elsewhere)"
    say "disk ok: $((need / 1000000000)) GB still to download, $((free / 1000000000)) GB free"
}

download() {
    [ -n "$ENDPOINT" ] && export HF_ENDPOINT="$ENDPOINT"
    [ "$NO_XET" = 1 ] && export HF_HUB_DISABLE_XET=1
    mkdir -p "$MODEL"
    say "① $REPO: everything except the base-model parts → $MODEL"
    "$HF" download "$REPO" --local-dir "$MODEL" --exclude "*.part*-of-*" || die "download of $REPO failed; run the same command again to resume"
    [ -s "$MODEL/SHA256SUMS" ] || die "SHA256SUMS missing from the download"
    chmod +x "$MODEL/bin/ds4" "$MODEL/bin/ds4-server"   # downloads do not keep the executable bit
    merge_parts
    if engram_ok; then say "③ n-gram tables already in $ENGRAM, skipping their download"; return; fi
    say "③ n-gram tables: two shards of $BASE_REPO → $ENGRAM (203 GB)"
    "$HF" download "$BASE_REPO" "${SHARDS[@]}" --local-dir "$ENGRAM" || die "download of the official shards failed; run again to resume"
    for i in 0 1; do
        [ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] || die "${SHARDS[$i]} has the wrong size; run again to resume"
    done
}

# ② Download the 40 parts one at a time, check each against SHA256SUMS, append it, delete it. Peak extra disk = one part.
# Resumable: .merge-progress holds how many parts are already in the .merging file; on restart the file is cut back
# to exactly that many parts (an append interrupted halfway leaves junk at the end — this removes it).
merge_parts() {
    local out="$MODEL/$GGUF.merging" prog="$MODEL/.merge-progress" k=0 i p want got
    if [ "$(fsize "$MODEL/$GGUF")" = "$GGUF_BYTES" ]; then say "② base model already assembled"; return; fi
    [ -s "$prog" ] && k=$(cat "$prog")
    [ "$(fsize "$out")" -ge "$(parts_len "$k")" ] || die "$out is shorter than the $k parts recorded in $prog; delete both and run again"
    truncate -s "$(parts_len "$k")" "$out"
    for ((i = k + 1; i <= NPART; i++)); do
        p=$(part_name "$i")
        say "② part $i/$NPART"
        "$HF" download "$REPO" "$p" --local-dir "$MODEL" >/dev/null || die "download of $p failed; run again to resume"
        want=$(awk -v f="$p" '$2 == f {print $1}' "$MODEL/SHA256SUMS")
        got=$(sha256sum "$MODEL/$p" | cut -d' ' -f1)
        if [ "$got" != "$want" ]; then rm -f "$MODEL/$p"; die "$p is corrupted (sha256 mismatch) and was deleted; run again to fetch it anew"; fi
        cat "$MODEL/$p" >>"$out" || die "cannot append $p (disk full?)"
        echo "$i" >"$prog"
        rm -f "$MODEL/$p"
    done
    [ "$(fsize "$out")" = "$GGUF_BYTES" ] || die "assembled file is $(fsize "$out") bytes, expected $GGUF_BYTES"
    mv "$out" "$MODEL/$GGUF" && rm -f "$prog"
    say "② base model assembled: $GGUF ($GGUF_BYTES bytes, every part sha256-checked)"
}

watchdog() {   # lives and dies with the server; stopping it beats a frozen machine
    local pid="$1" bad=0 a
    while kill -0 "$pid" 2>/dev/null; do
        a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$a" -lt "$2" ]; then bad=$((bad + 1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then echo "$(date) MemAvailable ${a} MB < $2 MB twice, stopping the server" >>"$3"; kill "$pid"; return; fi
        sleep 5
    done
}

start() {
    alive && { say "already running (pid $(cat "$PIDF"), $(cat "$INFOF" 2>/dev/null)); stop it first to switch"; return; }
    [ -s "$MODEL/$GGUF" ] && [ -x "$MODEL/bin/ds4-server" ] || die "not installed yet: run 'bash install.sh --no-start' first"
    engram_ok || die "the n-gram tables are not in $ENGRAM: run 'bash install.sh --no-start' first"
    # An install made before a domain was published has no directory for it; hf download fetches it on the next run.
    [ -z "$SIDECAR" ] || [ -s "$MODEL/$SIDECAR/manifest.txt" ] \
        || die "the $DOMAIN sidecar is not in $MODEL: run 'bash install.sh --no-start' to download it"
    # install.sh is fetched with curl, bin/ by hf download — a new script over old binaries would start a server
    # that rejects --engram-dir and exits while "loading". Catch it here with a clear fix instead.
    # (Captured first, not piped into grep -q: grep quits at the match, the server can die of SIGPIPE, and
    # pipefail would then report a perfectly good binary as old.)
    local help; help=$("$MODEL/bin/ds4-server" --help 2>&1)
    case "$help" in *--engram-dir*) ;; *) die "bin/ds4-server is older than this install.sh: run 'bash install.sh --no-start' to update it";; esac
    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB of memory available, need $START_FLOOR_MB: close other programs first"
    mkdir -p "$LOGS"
    local zc=() pt=()
    [ -n "$SIDECAR" ] && zc=(--zchain "$MODEL/$SIDECAR")
    [ "$WITH_PT" = 1 ] && pt=(--posttrain "$MODEL/$POSTTRAIN")
    say "starting ds4-server on $HOST:$PORT, domain $DOMAIN (loading takes ~2 minutes; log: $LOGS/server.log)"
    nohup "$MODEL/bin/ds4-server" --cuda -m "$MODEL/$GGUF" "${zc[@]}" --engram-dir "$ENGRAM" "${pt[@]}" \
        --mem-budget-mb "$BUDGET_MB" --host "$HOST" --port "$PORT" >"$LOGS/server.log" 2>&1 </dev/null &
    echo $! >"$PIDF"; echo "domain $DOMAIN on $HOST:$PORT" >"$INFOF"
    export -f watchdog
    nohup bash -c "watchdog $(cat "$PIDF") $KILL_MB '$LOGS/watchdog.log'" >/dev/null 2>&1 </dev/null &
    echo $! >"$WDPIDF"
    local i
    for i in $(seq 1 180); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
        alive || { tail -8 "$LOGS/server.log"; die "the server exited while loading (full log: $LOGS/server.log)"; }
        sleep 5
    done
    curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { stop; die "the server did not come up in 15 minutes (log: $LOGS/server.log)"; }
    a=$(avail_mb)
    [ "$a" -ge "$RUN_FLOOR_MB" ] || { stop; die "only $a MB left after loading (need $RUN_FLOOR_MB); the machine would stall"; }
    say "loaded; $a MB of memory left. smoke test:"
    local out q="In one sentence, what is a price-to-earnings ratio?"
    case "$DOMAIN" in          # one question from the served domain, so the smoke test exercises that sidecar
        code)     q="In one sentence, what does git rebase do?";;
        law)      q="In one sentence, what is a statute of limitations?";;
        medicine) q="In one sentence, what is hypertension?";;
        science)  q="In one sentence, what is entropy in thermodynamics?";;
    esac
    out=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"deepseek-chat\",\"temperature\":0,\"max_tokens\":48,\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}]}")
    printf '%s' "$out" | grep -q '"choices"' || { echo "$out" | head -c 400; stop; die "the smoke request failed"; }
    printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]*\)".*/  → \1/p' | head -1
    say "ready: http://$HOST:$PORT/v1, domain $DOMAIN  (OpenAI: /v1/chat/completions /v1/completions /v1/responses · Anthropic: /v1/messages)"
    say "stop with: bash install.sh stop --dir $DIR   ·   switch domain: stop, then start --domain <$DOMAINS>"
}

stop() {
    [ -s "$WDPIDF" ] && kill "$(cat "$WDPIDF")" 2>/dev/null
    if alive; then
        kill "$(cat "$PIDF")"; local i
        for i in $(seq 1 20); do alive || break; sleep 1; done
        alive && kill -9 "$(cat "$PIDF")"
        say "stopped"
    else
        say "not running"
    fi
    rm -f "$PIDF" "$WDPIDF" "$INFOF"
}

status() {
    if alive; then say "running (pid $(cat "$PIDF")): $(cat "$INFOF" 2>/dev/null)"; else say "not running"; fi
    say "memory available: $(avail_mb) MB"
}

case "$CMD" in
    install) preflight; ensure_hf; use_engram_src; check_disk; download
             say "installed in $DIR"
             [ "$START" = 1 ] && start;;
    start)   preflight; use_engram_src; start;;
    stop)    stop;;
    status)  status;;
esac
