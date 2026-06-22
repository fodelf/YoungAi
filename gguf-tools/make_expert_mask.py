#!/usr/bin/env python3
"""Build a DS4 expert-mask file from router_norms.py output.

A mask file tells `ds4` which routed experts to keep at runtime.  Disabled
experts get their router logit forced to -inf before top-K selection, so
they are never picked and their tensor pages are never paged into RAM.

The mask format is small and explicit:

    magic       u8[4]  "DSXM"
    version     u32    1
    n_layers    u32
    n_expert    u32    (per layer; must match the GGUF)
    reserved    u32    0
    bits        ceil(n_layers * n_expert / 8) bytes
                bit (layer * n_expert + expert) == 1 means KEEP that expert,
                0 means DISABLE.

Usage:
    python3 gguf-tools/make_expert_mask.py /tmp/router_norms.json \\
        --keep-top-k 16 --out /tmp/mask-k16.bin

    # Or keep different K per layer with --keep-list:
    python3 gguf-tools/make_expert_mask.py /tmp/router_norms.json \\
        --keep-list 8,16,16,32,... --out /tmp/mask-mixed.bin
"""

from __future__ import annotations

import argparse
import json
import struct
import sys


MAGIC = b"DSXM"
VERSION = 1


def build_mask_from_topk(by_layer: list[dict], n_expert: int, keep_k: int) -> list[set[int]]:
    """Per-layer: keep the top-K experts by L2 norm."""
    if keep_k < 1 or keep_k > n_expert:
        raise ValueError(f"keep_k={keep_k} out of range [1, {n_expert}]")
    out = []
    for entry in by_layer:
        rank = entry["rank"]  # descending
        out.append(set(rank[:keep_k]))
    return out


def build_mask_from_keep_list(by_layer: list[dict], n_expert: int, keep_list: list[int]) -> list[set[int]]:
    """Per-layer K from a CSV list, one K per layer."""
    if len(keep_list) != len(by_layer):
        raise ValueError(
            f"--keep-list has {len(keep_list)} entries but router has {len(by_layer)} layers"
        )
    out = []
    for k, entry in zip(keep_list, by_layer):
        if k < 1 or k > n_expert:
            raise ValueError(f"layer {entry['layer']}: keep_k={k} out of [1,{n_expert}]")
        rank = entry["rank"]
        out.append(set(rank[:k]))
    return out


def force_keep_hash_layers(keep_sets: list[set[int]], n_expert: int, n_hash: int) -> int:
    """Force the first `n_hash` layers to keep ALL experts (correctness, not tuning).

    DeepSeek-V4's first DS4_N_HASH_LAYER layers route by a frozen tid2eid hash
    table keyed on token id, NOT by router logits.  The runtime's -inf keep-mask
    only gates the learned-router path (`has_bias && !hash_mode`); the hash path
    bypasses it and any selected-but-dropped expert is silently clamped to slot 0
    in route_translate -> wrong expert, full route weight, no renorm, no error.
    Activation-rank top-K cannot safely shrink these layers because the hash can
    deterministically select any of its experts for some token.  So we keep them
    whole.  (A tighter future option is to keep only the tid2eid-reachable union
    read from the GGUF; keep-all is the conservative, always-correct invariant.)

    Returns the number of experts newly added across the forced layers.
    """
    if n_hash <= 0:
        return 0
    n_hash = min(n_hash, len(keep_sets))
    full = set(range(n_expert))
    added = 0
    for layer in range(n_hash):
        added += n_expert - len(keep_sets[layer])
        keep_sets[layer] = set(full)
    return added


def serialize_mask(keep_sets: list[set[int]], n_expert: int) -> bytes:
    n_layers = len(keep_sets)
    n_bits = n_layers * n_expert
    n_bytes = (n_bits + 7) // 8
    bits = bytearray(n_bytes)
    for layer, kept in enumerate(keep_sets):
        for e in kept:
            idx = layer * n_expert + e
            bits[idx >> 3] |= 1 << (idx & 7)

    header = MAGIC + struct.pack("<IIII", VERSION, n_layers, n_expert, 0)
    return bytes(header) + bytes(bits)


def _csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("router_norms_json", help="output of router_norms.py")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep-top-k", type=int, help="per-layer: keep this many top experts")
    g.add_argument("--keep-list", type=_csv_ints, help="per-layer K values, comma separated")
    ap.add_argument("--hash-layers", type=int, default=3,
                    help="force the first N layers (DeepSeek hash-routed layers, "
                         "DS4_N_HASH_LAYER=3) to keep ALL experts. These route by a "
                         "frozen token->expert hash and bypass the -inf keep-mask, so "
                         "shrinking them silently misroutes. Use 0 to disable (e.g. for "
                         "a non-hash model).")
    ap.add_argument("--out", required=True, help="output mask path")
    args = ap.parse_args()

    with open(args.router_norms_json) as f:
        result = json.load(f)

    n_expert = result["n_expert"]
    by_layer = result["by_layer"]
    # Ensure layers are in increasing order.
    by_layer = sorted(by_layer, key=lambda x: x["layer"])
    # Sanity: contiguous 0..n-1.
    for i, entry in enumerate(by_layer):
        if entry["layer"] != i:
            raise SystemExit(
                f"router_norms layers not contiguous from 0: entry {i} has layer={entry['layer']}"
            )

    if args.keep_top_k is not None:
        kept = build_mask_from_topk(by_layer, n_expert, args.keep_top_k)
        desc = f"top-{args.keep_top_k}"
    else:
        kept = build_mask_from_keep_list(by_layer, n_expert, args.keep_list)
        desc = f"per-layer K (median={sorted(args.keep_list)[len(args.keep_list)//2]})"

    hash_added = force_keep_hash_layers(kept, n_expert, args.hash_layers)
    if args.hash_layers > 0:
        desc += f" +hash-keep-all[0:{min(args.hash_layers, len(kept))}]"

    blob = serialize_mask(kept, n_expert)
    with open(args.out, "wb") as f:
        f.write(blob)

    total_kept = sum(len(s) for s in kept)
    total_slots = len(kept) * n_expert
    print(
        f"wrote {args.out}  ({desc})\n"
        f"  layers       = {len(kept)}\n"
        f"  n_expert     = {n_expert}\n"
        f"  kept slots   = {total_kept} / {total_slots} "
        f"({100.0 * total_kept / total_slots:.1f}%)\n"
        f"  disabled     = {total_slots - total_kept}\n"
        f"  hash-layers  = {min(args.hash_layers, len(kept)) if args.hash_layers > 0 else 0} "
        f"(forced full; +{hash_added} experts kept for routing correctness)\n"
        f"  mask bytes   = {len(blob)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
