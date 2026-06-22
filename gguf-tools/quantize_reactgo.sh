#!/usr/bin/env bash
# =============================================================================
# quantize_reactgo.sh — precision-inversion / hot-resident quantization driver
#
# Builds the react/go GGUF described in react-go-opus46-design.md §3.5/§3.6/§3.8:
# only the react/go HOT experts are quantized into a small GGUF (IQ2_XXS,
# resident), with ds4.expert_keep_map.* so the runtime maps router id -> compact
# slot; COLD experts are dropped here and streamed from the HF safetensors
# (hybrid layout).  This is the "Mode P" base for dynamic mode switching (§3.7).
#
# Pipeline (each step is its own tool; this driver just sequences them):
#   [model] 1. ds4 --imatrix-out         react/go activation imatrix .dat
#   [safe ] 2. router_norms_from_imatrix  .dat -> per-layer per-expert rank JSON
#   [safe ] 3. make_expert_mask           rank JSON + per-layer K -> DSXM hot mask
#   [model] 4. deepseek4-quantize         HF + hot mask -> small hot-only GGUF
#
# SAFETY (CLAUDE.md / memory 铁律):
#   * Steps 2 and 3 are pure offline python, 64 MB-streaming, never load the
#     model — they run by default.
#   * Steps 1 and 4 read the model (a model run / a 240 GB HF sweep).  They are
#     GATED: they only execute when RUN_MODEL_STEPS=1 is set in the environment,
#     AND they print their exact command first.  Do not set that flag without
#     (a) the HF download being complete, (b) an RSS/watchdog memory-safety
#     check, (c) explicit per-run authorization.  Default = print, don't run.
#   * Never double-load the base on one host; run heavy steps on the big-disk /
#     big-RAM machine.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# ---- knobs (override via env) ----------------------------------------------
HF_DIR="${HF_DIR:-$ROOT/hf/DeepSeek-V4-Flash-Base}"     # original safetensors
TEMPLATE="${TEMPLATE:-$ROOT/gguf/ds4flash.gguf}"        # existing DS4 GGUF (metadata/shape donor)
CALIB="${CALIB:-/tmp/reactgo_calib.txt}"                # react/go calibration corpus
IMATRIX="${IMATRIX:-/tmp/reactgo_router.dat}"           # step 1 output
GENERAL_IMATRIX="${GENERAL_IMATRIX:-}"                  # optional general-corpus .dat;
                                                        # set it to score programming-SPECIFIC
                                                        # experts (specialty) vs generalists.
NORMS="${NORMS:-/tmp/router_norms_reactgo.json}"        # step 2 output
MASK="${MASK:-/tmp/mask-reactgo-hot.bin}"               # step 3 output
OUT="${OUT:-$ROOT/gguf/ds4-reactgo-hot-iq2xxs.gguf}"    # step 4 output (small hot-only GGUF)

N_EXPERT="${N_EXPERT:-256}"                             # Flash 256, Pro 384
KEEP_TOP_K="${KEEP_TOP_K:-32}"                           # hot experts/layer (uniform); see --keep-list for per-layer
EXPERTS_TYPE="${EXPERTS_TYPE:-iq2_xxs}"                 # gate/up hot type
ROUTED_W2="${ROUTED_W2:-q2_k}"                          # down hot type
RANK_PART="${RANK_PART:-sum}"                           # gate|up|down|sum
RUN_MODEL_STEPS="${RUN_MODEL_STEPS:-0}"                 # 1 = actually run steps 1 & 4 (authorized only)

QUANT="$HERE/deepseek4-quantize"
PY="python3"

banner() { printf '\n=== %s ===\n' "$*"; }
gated()  { # print a command; run it only if RUN_MODEL_STEPS=1
  printf '\n[MODEL STEP] %s\n' "$*"
  if [[ "$RUN_MODEL_STEPS" == "1" ]]; then
    printf '[MODEL STEP] RUN_MODEL_STEPS=1 -> executing (ensure memory-safety + authorization)\n'
    eval "$*"
  else
    printf '[MODEL STEP] RUN_MODEL_STEPS!=1 -> NOT executed. Review, prove memory-safe, authorize, then re-run with RUN_MODEL_STEPS=1.\n'
  fi
}

# ---- step 1: react/go imatrix (MODEL) --------------------------------------
banner "step 1: react/go activation imatrix (gated)"
if [[ -f "$IMATRIX" ]]; then
  echo "imatrix exists, skipping: $IMATRIX"
else
  if [[ ! -f "$CALIB" ]]; then
    echo "NOTE: calibration corpus missing: $CALIB"
    echo "      build it from real react/go sources first, e.g.:"
    echo "      cat calib_react/*.tsx calib_react/*.jsx calib_go/*.go calib_reactgo_tasks/*.txt > $CALIB"
  fi
  gated "'$ROOT/ds4' -m '$TEMPLATE' --imatrix-dataset '$CALIB' --imatrix-out '$IMATRIX' --ctx 32768"
fi

# ---- step 2: derive per-layer per-expert rank (SAFE) -----------------------
banner "step 2: router_norms_from_imatrix (offline, memory-safe)"
# The HOT MASK ranks by --rank-by energy (react/go absolute activation = the
# residency/coverage set; generalist experts MUST be kept or react/go cold-misses).
# When GENERAL_IMATRIX is given, the same JSON also carries per-expert
# specialty_log2 = log2(react/go share / general share): that is the
# "which experts are programming-SPECIFIC" answer (specialty >= 1 ~ used >=2x more
# on react/go than on general text), for REAP dead-expert pruning / training targets.
if [[ -f "$IMATRIX" ]]; then
  BASELINE_ARGS=()
  if [[ -n "$GENERAL_IMATRIX" && -f "$GENERAL_IMATRIX" ]]; then
    BASELINE_ARGS=(--baseline "$GENERAL_IMATRIX")
    echo "specialty scoring ON (baseline: $GENERAL_IMATRIX)"
  elif [[ -n "$GENERAL_IMATRIX" ]]; then
    echo "NOTE: GENERAL_IMATRIX set but missing ($GENERAL_IMATRIX); specialty scoring OFF."
    echo "      produce it once with the repo's general corpus, e.g.:"
    echo "      ds4 -m $TEMPLATE --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt --imatrix-out $GENERAL_IMATRIX --ctx 32768"
  else
    echo "specialty scoring OFF (set GENERAL_IMATRIX=<general .dat> to identify programming-specific experts)."
  fi
  "$PY" "$HERE/router_norms_from_imatrix.py" "$IMATRIX" \
      --n-expert "$N_EXPERT" --part "$RANK_PART" --rank-by energy \
      "${BASELINE_ARGS[@]}" --out "$NORMS"
else
  echo "imatrix not present yet ($IMATRIX); skipping step 2 (needs step 1)."
fi

# ---- step 3: hot mask (SAFE) -----------------------------------------------
banner "step 3: make_expert_mask hot mask (offline, memory-safe)"
if [[ -f "$NORMS" ]]; then
  # --hash-layers 3: the first DS4_N_HASH_LAYER layers route by a frozen
  # token->expert hash that bypasses the -inf keep-mask; shrinking them silently
  # misroutes (route_translate clamps dropped picks to slot 0).  Force-keep them
  # whole for routing correctness (costs ~3 layers of full experts).
  "$PY" "$HERE/make_expert_mask.py" "$NORMS" --keep-top-k "$KEEP_TOP_K" \
      --hash-layers "${HASH_LAYERS:-3}" --out "$MASK"
  echo "hot mask: $MASK  (keep-top-$KEEP_TOP_K/layer + hash layers 0..2 full; use --keep-list for per-layer K)"
else
  echo "router_norms not present yet ($NORMS); skipping step 3 (needs step 2)."
fi

# ---- step 4: emit hot-only quantized GGUF (MODEL) --------------------------
banner "step 4: deepseek4-quantize hot-only IQ2 build (gated)"
if [[ ! -x "$QUANT" ]]; then
  echo "building deepseek4-quantize ..."
  ( cd "$HERE" && make >/dev/null )
fi
if [[ -f "$MASK" ]]; then
  gated "'$QUANT' --hf '$HF_DIR' --template '$TEMPLATE' \
      --experts-hot-mask '$MASK' \
      --experts '$EXPERTS_TYPE' --routed-w2 '$ROUTED_W2' \
      --imatrix '$IMATRIX' \
      --out '$OUT' --overwrite"
  echo
  echo "Tip: add --dry-run to step 4 to preview the plan (tensor sizes + keep_map)"
  echo "     without reading any HF data — safe to run anytime."
else
  echo "hot mask not present yet ($MASK); skipping step 4 (needs step 3)."
fi

banner "done"
echo "Offline-safe steps (2,3) ran where inputs existed; model steps (1,4) printed."
echo "When the HF download is complete and you are authorized, set RUN_MODEL_STEPS=1."
