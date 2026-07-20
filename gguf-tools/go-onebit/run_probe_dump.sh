#!/usr/bin/env bash
#
# run_probe_dump.sh — drive the Q8 REFERENCE model through `ds4` to dump
# per-layer Go-trajectory activations (SPEC.md §6 "Go 轨迹探测").
#
# It exports the Metal graph-dump env vars that ds4.c honours and runs ds4 over
# SHORT Go prompts so the deep-half FFN tensors needed by the 1-bit fitter get
# written to disk as raw little-endian float32 (.bin) / int32 (.i32) files.
#
# Dump env vars (confirmed in ds4.c: metal_graph_debug_wants / *_dump_tensor):
#   DS4_METAL_GRAPH_DUMP_PREFIX   output path prefix (REQUIRED to enable dumping)
#   DS4_METAL_GRAPH_DUMP_NAME     substring filter; a stage S is dumped iff S is
#                                 a substring of this value. Unset = dump every
#                                 stage. We target three stages:
#                                     ffn_norm        (n_embd f32  — the input x)
#                                     ffn_moe_out     (n_embd f32  — routed y_ref)
#                                     ffn_moe_logits  (256    f32  — router logits)
#   DS4_METAL_GRAPH_DUMP_LAYER    a single layer index, or the literal "all"
#   DS4_METAL_GRAPH_DUMP_POS      a single token position; unset = every position
#
# Output filename pattern (from ds4.c snprintf):
#   <PREFIX>_<stage>-<layer>_pos<pos>.bin   (f32 / f16 tensors)
#   <PREFIX>_<stage>-<layer>_pos<pos>.i32   (i32 tensors; not used by us)
# e.g.  probe-dump/p0_ffn_moe_out-30_pos7.bin
#
# These hooks live on the METAL path only, so we force --metal.
#
# !! This script LOADS A MODEL when run for real. It does NOT run it under
# !! DRYRUN=1 (default-off): set DRYRUN=1 to print the exact commands/env and
# !! skip launching ds4 entirely — use that to inspect logic without a model.
#
set -euo pipefail

# --------------------------------------------------------------------------
# Parameters (all overridable via environment).
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# REF_MODEL must point at the Q8_0 *reference* gguf (the ground truth we fit to),
# NOT the q2 release model. Override to your Q8 file.
REF_MODEL="${REF_MODEL:-${REPO_ROOT}/ds4flash.gguf}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/go_prompts.txt}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/probe-dump}"
# LAYER_RANGE: "all" (one forward dumps every layer — cheapest), or "A-B", or a
# space/comma list like "24 30 36". Anything but "all" forces one model load per
# layer (the LAYER env var only takes a single index), so prefer "all".
LAYER_RANGE="${LAYER_RANGE:-all}"
DS4_BIN="${DS4_BIN:-${REPO_ROOT}/ds4}"
CTX="${CTX:-4096}"                 # short Go prompts; keep ctx tiny
NPREDICT="${NPREDICT:-1}"          # we only need prefill activations
# POS unset => dump every prompt position (collect the per-token activation pool
# {x_i} of SPEC §6). Set DUMP_POS=<n> to pin one position.
DUMP_POS="${DUMP_POS:-}"
# PER_NAME_PASS=1 reloads the model once per stage name (explicit, expensive).
# Default 0 = single pass capturing all three stages at once (one model load).
PER_NAME_PASS="${PER_NAME_PASS:-0}"
DRYRUN="${DRYRUN:-0}"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-2}"   # seconds between RSS polls

# The three stages we want (also drives the optional per-name loop).
NAMES=(ffn_norm ffn_moe_out ffn_moe_logits)

# --------------------------------------------------------------------------
# MEMORY-SAFETY PREAMBLE (hard guardrail — see CLAUDE.md "Guardrails").
#   (a) DS4_MEM_BUDGET_MB clamped to the 12 GiB red line and exported so the
#       engine's own L1 resident gate refuses startup over budget.
#   (b) a background watchdog kills the ds4 PID if its RSS exceeds the budget.
#   (c) dual-host (M4 + M1) note: NEVER double-load the base on one host.
# --------------------------------------------------------------------------
: "${DS4_MEM_BUDGET_MB:=12000}"
if [ "${DS4_MEM_BUDGET_MB}" -gt 12000 ]; then
    echo "FATAL: DS4_MEM_BUDGET_MB=${DS4_MEM_BUDGET_MB} exceeds the hard 12000 MB red line." >&2
    echo "       Both hosts must stay <=12 GiB. Refusing to start." >&2
    exit 1
fi
export DS4_MEM_BUDGET_MB
BUDGET_KB=$(( DS4_MEM_BUDGET_MB * 1024 ))

cat >&2 <<EOF
========================================================================
 run_probe_dump.sh  (Go-domain 1-bit probe, SPEC §6)
 ds4 binary     : ${DS4_BIN}
 reference model: ${REF_MODEL}   (MUST be the Q8_0 reference, not q2)
 prompts        : ${PROMPT_FILE}
 out dir        : ${OUT_DIR}
 layers         : ${LAYER_RANGE}    pos: ${DUMP_POS:-<all>}
 mem budget     : ${DS4_MEM_BUDGET_MB} MB  (watchdog kill @ ${BUDGET_KB} KB RSS)
 mode           : $( [ "${PER_NAME_PASS}" = 1 ] && echo "per-name (reloads model per stage)" || echo "single-pass (all 3 stages, 1 load)" )
 dry run        : ${DRYRUN}

 DUAL-HOST NOTE (M4 mini coordinator + MacBook M1 worker, Thunderbolt/NFS/SSH):
   The 81 GiB base model is NEVER loaded twice on a single host. Split this
   probe across the two machines EITHER layer-sliced (each host owns a layer
   range via LAYER_RANGE) OR by-prompt (each host runs a disjoint slice of
   ${PROMPT_FILE}); copy each host's probe-dump/ back over NFS/SSH and merge.
   Each host stays <=12 GiB. This invocation is ONE host's share only.

 NOTE on the watchdog: ps RSS undercounts clean mmap'd weight pages, so the
 engine's own DS4_MEM_BUDGET_MB + L1 resident gate is the primary guard; this
 watchdog is a backstop against runaway dirty/wired growth. For a true
 phys_footprint use: vmmap --summary <pid> | grep 'Physical footprint'.
========================================================================
EOF

# --------------------------------------------------------------------------
# Watchdog: poll RSS (KB on macOS via `ps -o rss=`) and kill if over budget.
# Runs in the background against a specific ds4 PID.
# --------------------------------------------------------------------------
watchdog() {
    local pid="$1" budget_kb="$2" interval="$3"
    while kill -0 "${pid}" 2>/dev/null; do
        local rss
        rss="$(ps -o rss= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        if [ -n "${rss:-}" ] && [ "${rss}" -gt "${budget_kb}" ] 2>/dev/null; then
            echo "WATCHDOG: ds4 pid=${pid} RSS ${rss} KB > budget ${budget_kb} KB — KILLING" >&2
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 1
            kill -KILL "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep "${interval}"
    done
}

# Launch ds4 (background) under the watchdog and wait for it. Args are ds4 argv.
run_ds4_guarded() {
    if [ "${DRYRUN}" = 1 ]; then
        echo "DRYRUN: DS4_METAL_GRAPH_DUMP_PREFIX=${DS4_METAL_GRAPH_DUMP_PREFIX:-} \
DS4_METAL_GRAPH_DUMP_NAME=${DS4_METAL_GRAPH_DUMP_NAME:-<unset>} \
DS4_METAL_GRAPH_DUMP_LAYER=${DS4_METAL_GRAPH_DUMP_LAYER:-<unset>} \
DS4_METAL_GRAPH_DUMP_POS=${DS4_METAL_GRAPH_DUMP_POS:-<unset>} \
${DS4_BIN} $*" >&2
        return 0
    fi
    # Run from REPO_ROOT so metal/*.metal resolves (the `ds4` CLI has no
    # --chdir; that flag only exists on ds4-agent/ds4-server). All paths in
    # "$@" are absolute, so the cd is safe. `exec` makes $! the ds4 PID.
    ( cd "${REPO_ROOT}" && exec "${DS4_BIN}" "$@" ) &
    local pid=$!
    watchdog "${pid}" "${BUDGET_KB}" "${WATCHDOG_INTERVAL}" &
    local wd=$!
    local rc=0
    wait "${pid}" || rc=$?
    kill "${wd}" 2>/dev/null || true
    wait "${wd}" 2>/dev/null || true
    return "${rc}"
}

# Expand LAYER_RANGE into the list of DS4_METAL_GRAPH_DUMP_LAYER values to use.
expand_layers() {
    local spec="$1"
    if [ "${spec}" = "all" ]; then
        printf 'all\n'
        return
    fi
    if [[ "${spec}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return
    fi
    # whitespace/comma separated list
    printf '%s\n' "${spec}" | tr ',' '\n' | tr -s ' ' '\n' | sed '/^$/d'
}

# --------------------------------------------------------------------------
# Preflight checks (skipped naturally for the model load itself under DRYRUN).
# --------------------------------------------------------------------------
if [ "${DRYRUN}" != 1 ]; then
    [ -x "${DS4_BIN}" ]    || { echo "FATAL: ds4 binary not found/executable: ${DS4_BIN}" >&2; exit 1; }
    [ -e "${REF_MODEL}" ]  || { echo "FATAL: reference model not found: ${REF_MODEL}" >&2; exit 1; }
fi
[ -f "${PROMPT_FILE}" ] || { echo "FATAL: prompt file not found: ${PROMPT_FILE}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

# Canonicalize to absolute paths: run_ds4_guarded cd's into REPO_ROOT, so any
# relative model/prompt/output paths would otherwise break.
abspath() { local p="$1"; if [ -d "$p" ]; then (cd "$p" && pwd); else echo "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"; fi; }
OUT_DIR="$(abspath "${OUT_DIR}")"
PROMPT_FILE="$(abspath "${PROMPT_FILE}")"
[ -e "${REF_MODEL}" ] && REF_MODEL="$(abspath "${REF_MODEL}")"
[ -e "${DS4_BIN}" ]   && DS4_BIN="$(abspath "${DS4_BIN}")"

# Read layers into an array (avoid `mapfile`: macOS ships bash 3.2 without it).
LAYERS=()
while IFS= read -r _layer; do
    [ -n "${_layer}" ] && LAYERS+=("${_layer}")
done < <(expand_layers "${LAYER_RANGE}")
[ "${#LAYERS[@]}" -gt 0 ] || { echo "FATAL: empty layer set from LAYER_RANGE='${LAYER_RANGE}'" >&2; exit 1; }

# --------------------------------------------------------------------------
# Main loops: prompt  ->  layer  ->  (single pass | per-name passes).
# Each ds4 invocation loads the model exactly once.
# --------------------------------------------------------------------------
NAMES_JOINED="$(IFS=,; echo "${NAMES[*]}")"   # "ffn_norm,ffn_moe_out,ffn_moe_logits"

idx=0
while IFS= read -r prompt || [ -n "${prompt}" ]; do
    # skip blank lines / comments
    case "${prompt}" in ""|\#*) continue;; esac

    for layer in "${LAYERS[@]}"; do
        export DS4_METAL_GRAPH_DUMP_LAYER="${layer}"
        if [ -n "${DUMP_POS}" ]; then
            export DS4_METAL_GRAPH_DUMP_POS="${DUMP_POS}"
        else
            unset DS4_METAL_GRAPH_DUMP_POS || true
        fi

        # One prefix per (prompt,layer) keeps files from different prompts apart.
        export DS4_METAL_GRAPH_DUMP_PREFIX="${OUT_DIR}/p${idx}_L${layer}"

        # Build the ds4 argv (greedy, deterministic, Metal). cwd is set to
        # REPO_ROOT by run_ds4_guarded so the Metal shaders resolve.
        DS4_ARGS=(-m "${REF_MODEL}" --metal
                  --ctx "${CTX}" --temp 0 -n "${NPREDICT}" -p "${prompt}")

        if [ "${PER_NAME_PASS}" = 1 ]; then
            # Explicit per-stage passes (loops over the three NAME values).
            # WARNING: reloads the model once per name — expensive; use only when
            # you must isolate a single tensor stream.
            for name in "${NAMES[@]}"; do
                export DS4_METAL_GRAPH_DUMP_NAME="${name}"
                echo ">> prompt#${idx} layer=${layer} stage=${name}" >&2
                run_ds4_guarded "${DS4_ARGS[@]}"
            done
        else
            # Single pass: the comma-joined value matches all three stages at
            # once (strstr(name_env, stage) — each stage is a substring of it),
            # so one forward captures ffn_norm + ffn_moe_out + ffn_moe_logits.
            export DS4_METAL_GRAPH_DUMP_NAME="${NAMES_JOINED}"
            echo ">> prompt#${idx} layer=${layer} stages=${NAMES_JOINED}" >&2
            run_ds4_guarded "${DS4_ARGS[@]}"
        fi
    done

    idx=$(( idx + 1 ))
done < "${PROMPT_FILE}"

echo "DONE: dumped ${idx} prompt(s) into ${OUT_DIR}" >&2
if [ "${DRYRUN}" != 1 ]; then
    # Report what landed (helps confirm all three stages were captured).
    for name in "${NAMES[@]}"; do
        cnt="$(find "${OUT_DIR}" -name "*_${name}-*_pos*.bin" 2>/dev/null | wc -l | tr -d ' ')"
        echo "  stage ${name}: ${cnt} file(s)" >&2
    done
fi
