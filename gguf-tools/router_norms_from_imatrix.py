#!/usr/bin/env python3
"""Derive per-layer per-expert importance ranking from a DS4 imatrix .dat.

This is the first link of the react/go precision-inversion pipeline
(react-go-opus46-design.md §3.5.3 / §3.6.6).  `make_expert_mask.py` needs a
`router_norms.json` that ranks, per layer, the routed experts by how heavily
the *react/go* calibration corpus activates them.  No such producer existed in
the repo; this script fills that gap **offline** — it reads only the small
imatrix `.dat` (never the model), so it is memory-safe and needs no model run.

How importance is computed
--------------------------
DS4's collector packs one vector per routed expert tensor:

    entry name   = blk.N.ffn_{gate,up,down}_exps.weight
    entry values = n_expert * n_columns   (expert e -> values[e*ncols:(e+1)*ncols])

Each value is an accumulated squared activation for one input column of one
expert (sum over tokens, optionally with a per-tensor call count).  An expert
that the corpus rarely routes to accumulates near-zero energy.  So the
per-expert importance is the sum of that expert's column segment; ranking the
experts of a layer by this sum descending gives "which experts react/go uses
most" — exactly the order `make_expert_mask.py --keep-list/--keep-top-k` wants.

By default the importances of all available expert parts in a layer
(gate+up+down) are summed for robustness; `--part` picks a single one.

Usage
-----
    python3 gguf-tools/router_norms_from_imatrix.py /tmp/reactgo_router.dat \
        --n-expert 256 --out /tmp/router_norms_reactgo.json

    # sanity check without a real .dat:
    python3 gguf-tools/router_norms_from_imatrix.py --self-test

Output is the same schema make_expert_mask.py / make_hotset_manifest.py read:

    { "n_expert": 256,
      "source": "...", "part": "sum",
      "by_layer": [ {"layer": 0, "rank": [e...], "importance": [f...]}, ... ] }
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import struct
import sys
import tempfile

_EXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def _read_exact(fp, n: int, what: str) -> bytes:
    b = fp.read(n)
    if len(b) != n:
        raise SystemExit(f"imatrix: short read for {what} (wanted {n}, got {len(b)})")
    return b


def _read_i32(fp, what: str) -> int:
    return struct.unpack("<i", _read_exact(fp, 4, what))[0]


def per_expert_importance(values_le_floats, n_expert: int, ncols: int) -> list[float]:
    """Sum each expert's ncols-segment. values is a sequence of n_expert*ncols floats."""
    out = [0.0] * n_expert
    for e in range(n_expert):
        base = e * ncols
        s = 0.0
        # local accumulation; values supports fast slicing for array('f')
        seg = values_le_floats[base:base + ncols]
        for v in seg:
            s += v
        out[e] = s
    return out


def parse_imatrix(path: str, n_expert: int, parts: set[str]) -> tuple[dict[int, list[float]], int]:
    """Stream the .dat; return {layer: per-expert importance list} and detected ncols.

    Memory-safe: each entry's float block (<= ~2 MB for Flash) is read, reduced
    to n_expert scalars, then discarded.  The whole model is never touched.
    """
    by_layer: dict[int, list[float]] = {}
    ncols_seen = 0
    with open(path, "rb") as fp:
        n_entries = _read_i32(fp, "entry count")
        if n_entries < 1:
            raise SystemExit("imatrix has no entries")
        for _ in range(n_entries):
            name_len = _read_i32(fp, "name length")
            if name_len <= 0 or name_len > 4096:
                raise SystemExit(f"imatrix: bad name length {name_len}")
            name = _read_exact(fp, name_len, "name").decode("utf-8", "replace")
            ncall = _read_i32(fp, "calls")
            nval = _read_i32(fp, "values")
            if nval < 1:
                raise SystemExit("imatrix: bad value count")
            raw = _read_exact(fp, nval * 4, f"values for {name}")
            m = _EXP_RE.match(name)
            if not m:
                continue  # non-expert tensor; ignored for routing rank
            part = m.group(2)
            if part not in parts:
                continue
            layer = int(m.group(1))
            if nval % n_expert != 0:
                raise SystemExit(
                    f"imatrix: {name} has {nval} values, not divisible by n_expert={n_expert}"
                )
            ncols = nval // n_expert
            if ncols_seen and ncols != ncols_seen:
                # gate/up share an input dim; down differs — both fine, just note.
                pass
            ncols_seen = ncols
            vals = array.array("f")
            vals.frombytes(raw)
            # dividing by ncall is a per-tensor constant -> does not change ranking,
            # but keeps importances comparable across tensors/parts when summed.
            scale = 1.0 / ncall if ncall > 0 else 1.0
            imp = per_expert_importance(vals, n_expert, ncols)
            if scale != 1.0:
                imp = [v * scale for v in imp]
            if layer not in by_layer:
                by_layer[layer] = [0.0] * n_expert
            acc = by_layer[layer]
            for e in range(n_expert):
                acc[e] += imp[e]
    if not by_layer:
        raise SystemExit("imatrix: no routed-expert entries matched blk.N.ffn_*_exps.weight")
    return by_layer, ncols_seen


def _check_contiguous(by_layer: dict[int, list[float]], what: str) -> list[int]:
    layers = sorted(by_layer)
    # make_expert_mask.py requires contiguous layers from 0.
    for i, lyr in enumerate(layers):
        if lyr != i:
            raise SystemExit(
                f"{what} layers not contiguous from 0: position {i} is layer {lyr}; got {layers}"
            )
    return layers


def build_router_norms(
    by_layer: dict[int, list[float]],
    n_expert: int,
    baseline_by_layer: dict[int, list[float]] | None = None,
    rank_by: str = "energy",
    eps: float = 1e-9,
) -> list[dict]:
    """Rank experts per layer.

    rank_by='energy'    -> react/go ABSOLUTE activation. This is the residency /
                           coverage set (what the router actually picks on react/go,
                           generalist experts included). Use this for the hot mask.
    rank_by='specialty' -> react/go-vs-general PREFERENCE (needs --baseline). This is
                           the "which experts are programming-SPECIFIC" answer:
                           specialty_log2[e] = log2(p_code[e] / p_gen[e]) per layer,
                           where p_* is the in-layer normalized activation share.
                           >0 means react/go leans on it harder than general text.
    Either way `importance` carries the react/go absolute energy, and when a
    baseline is given every entry also carries `specialty_log2` (aligned to rank).
    """
    layers = _check_contiguous(by_layer, "imatrix")
    if baseline_by_layer is not None:
        bl = _check_contiguous(baseline_by_layer, "baseline imatrix")
        if len(bl) != len(layers):
            raise SystemExit(
                f"baseline has {len(bl)} layers but react/go imatrix has {len(layers)}"
            )
    if rank_by == "specialty" and baseline_by_layer is None:
        raise SystemExit("--rank-by specialty requires --baseline (general-corpus imatrix)")

    out = []
    for lyr in layers:
        imp = by_layer[lyr]
        tot_c = sum(imp) or 1.0
        s_code = [imp[e] / tot_c for e in range(n_expert)]  # react/go in-layer share
        specialty = None
        if baseline_by_layer is not None:
            gen = baseline_by_layer[lyr]
            tot_g = sum(gen) or 1.0
            s_gen = [gen[e] / tot_g for e in range(n_expert)]  # general in-layer share
            specialty = [
                math.log2((s_code[e] + eps) / (s_gen[e] + eps)) for e in range(n_expert)
            ]
        if rank_by == "specialty":
            order = sorted(range(n_expert), key=lambda e: (-specialty[e], e))
        else:  # energy: descending react/go activation, ties by ascending id.
            order = sorted(range(n_expert), key=lambda e: (-imp[e], e))
        entry = {
            "layer": lyr,
            "rank": order,
            "importance": [round(imp[e], 8) for e in order],
        }
        if specialty is not None:
            entry["specialty_log2"] = [round(specialty[e], 4) for e in order]
        out.append(entry)
    return out


def write_synthetic_dat(path: str, n_layers: int, n_expert: int, ncols: int) -> list[list[int]]:
    """Write a tiny but format-faithful .dat; return the expected per-layer rank.

    Expert e in layer L gets a flat energy = (e+1) + L (so larger id -> larger
    energy -> rank should be strictly descending by id), summed over gate+up+down.
    """
    expected = []
    with open(path, "wb") as fp:
        entries = []
        for L in range(n_layers):
            for part in ("gate", "up", "down"):
                name = f"blk.{L}.ffn_{part}_exps.weight".encode("utf-8")
                vals = array.array("f")
                for e in range(n_expert):
                    energy = float((e + 1) + L)
                    vals.extend([energy / ncols] * ncols)  # column-mean so sum == energy
                entries.append((name, vals))
            # descending by id -> n_expert-1 .. 0
            expected.append(list(range(n_expert - 1, -1, -1)))
        fp.write(struct.pack("<i", len(entries)))
        for name, vals in entries:
            fp.write(struct.pack("<i", len(name)))
            fp.write(name)
            fp.write(struct.pack("<i", 1))            # ncall
            fp.write(struct.pack("<i", len(vals)))    # nval
            fp.write(vals.tobytes())
    return expected


def self_test() -> int:
    n_layers, n_expert, ncols = 4, 8, 16
    tmp = tempfile.mkdtemp(prefix="rn_selftest_")
    dat = os.path.join(tmp, "synthetic.dat")
    expected = write_synthetic_dat(dat, n_layers, n_expert, ncols)
    by_layer, detected_ncols = parse_imatrix(dat, n_expert, {"gate", "up", "down"})
    norms = build_router_norms(by_layer, n_expert)
    ok = True
    if detected_ncols != ncols:
        print(f"FAIL ncols: {detected_ncols} != {ncols}", file=sys.stderr); ok = False
    if len(norms) != n_layers:
        print(f"FAIL layer count: {len(norms)} != {n_layers}", file=sys.stderr); ok = False
    for entry, exp in zip(norms, expected):
        if entry["rank"] != exp:
            print(f"FAIL layer {entry['layer']} rank {entry['rank']} != {exp}", file=sys.stderr)
            ok = False
    # importance must be strictly descending within each layer.
    for entry in norms:
        imp = entry["importance"]
        if any(imp[i] < imp[i + 1] for i in range(len(imp) - 1)):
            print(f"FAIL layer {entry['layer']} importance not descending", file=sys.stderr)
            ok = False
    os.remove(dat)
    os.rmdir(tmp)
    print("self-test: " + ("OK" if ok else "FAILED"), file=sys.stderr)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("imatrix_dat", nargs="?", help="DS4 imatrix .dat (from ds4 --imatrix-out)")
    ap.add_argument("--n-expert", type=int, default=256,
                    help="routed experts per layer (Flash 256, Pro 384)")
    ap.add_argument("--part", choices=["gate", "up", "down", "sum"], default="sum",
                    help="which expert part(s) to rank by; 'sum' = gate+up+down (default)")
    ap.add_argument("--baseline",
                    help="general-corpus imatrix .dat; enables react/go-vs-general "
                         "specialty scoring (so 'programming-specific' experts are visible)")
    ap.add_argument("--rank-by", choices=["energy", "specialty"], default="energy",
                    help="energy = react/go absolute activation = residency/coverage set "
                         "(default, use for the hot mask, generalists included); "
                         "specialty = react/go-vs-general preference = programming-SPECIFIC "
                         "experts (requires --baseline)")
    ap.add_argument("--out", help="output router_norms.json (default: stdout)")
    ap.add_argument("--self-test", action="store_true",
                    help="run an in-memory format+ranking self-check and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.imatrix_dat:
        ap.error("imatrix_dat is required (or use --self-test)")
    if args.n_expert < 1:
        ap.error("--n-expert must be >= 1")

    parts = {"gate", "up", "down"} if args.part == "sum" else {args.part}
    by_layer, ncols = parse_imatrix(args.imatrix_dat, args.n_expert, parts)
    baseline_by_layer = None
    if args.baseline:
        baseline_by_layer, _ = parse_imatrix(args.baseline, args.n_expert, parts)
    norms = build_router_norms(by_layer, args.n_expert, baseline_by_layer, args.rank_by)
    result = {
        "n_expert": args.n_expert,
        "source": f"imatrix:{os.path.abspath(args.imatrix_dat)}",
        "baseline": f"imatrix:{os.path.abspath(args.baseline)}" if args.baseline else None,
        "part": args.part,
        "rank_by": args.rank_by,
        "n_columns_last": ncols,
        "by_layer": norms,
    }
    blob = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob)
        dest = args.out
    else:
        sys.stdout.write(blob + "\n")
        dest = "(stdout)"

    nonzero = sum(1 for L in norms for v in L["importance"] if v > 0.0)
    total = len(norms) * args.n_expert
    msg = (
        f"wrote {dest}\n"
        f"  layers          = {len(norms)}\n"
        f"  n_expert        = {args.n_expert}\n"
        f"  part            = {args.part}\n"
        f"  rank_by         = {args.rank_by}\n"
        f"  active experts  = {nonzero}/{total} ({100.0 * nonzero / total:.1f}% ever routed)"
    )
    if baseline_by_layer is not None:
        specialist = sum(1 for L in norms for v in L["specialty_log2"] if v >= 1.0)  # >=2x vs general
        generalist = sum(1 for L in norms for v in L["specialty_log2"] if -1.0 < v < 1.0)
        msg += (
            f"\n  programming-SPECIFIC (specialty >= 2x general) = {specialist}/{total}"
            f"\n  generalist (within 2x)                         = {generalist}/{total}"
        )
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
