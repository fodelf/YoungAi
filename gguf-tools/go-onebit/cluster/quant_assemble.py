#!/usr/bin/env python3
# quant_assemble.py — cluster split-quantize glue (no shared FS).
#   pack:   extract the ranges listed in MANIFEST from a (sparse) partial GGUF
#           into a compact .segs stream (ranges back-to-back, manifest is the
#           index) for a single scp hop.
#   unpack: splice a .segs stream into the coordinator's GGUF at the manifest
#           offsets. After unpack the file is byte-identical to a single-host
#           run (quantization is deterministic).
# usage:
#   quant_assemble.py pack   MANIFEST PARTIAL.gguf OUT.segs
#   quant_assemble.py unpack MANIFEST IN.segs      TARGET.gguf
import sys

def read_manifest(p):
    out = []
    with open(p) as f:
        for ln in f:
            name, off, ln2 = ln.rstrip("\n").split("\t")
            out.append((name, int(off), int(ln2)))
    return out

mode, man_p, a_p, b_p = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
man = read_manifest(man_p)
total = sum(n for _, _, n in man)
CH = 64 << 20

if mode == "pack":
    with open(a_p, "rb") as src, open(b_p, "wb") as dst:
        done = 0
        for name, off, n in man:
            src.seek(off)
            left = n
            while left:
                buf = src.read(min(CH, left))
                if not buf: sys.exit(f"pack: short read at {name}")
                dst.write(buf); left -= len(buf)
            done += n
            print(f"pack {name} {n}B ({done*100//total}%)", file=sys.stderr, flush=True)
elif mode == "unpack":
    src_f = sys.stdin.buffer if a_p == "-" else open(a_p, "rb")
    with src_f as src, open(b_p, "r+b") as dst:
        done = 0
        for name, off, n in man:
            dst.seek(off)
            left = n
            while left:
                buf = src.read(min(CH, left))
                if not buf: sys.exit(f"unpack: short segs at {name}")
                dst.write(buf); left -= len(buf)
            done += n
            print(f"unpack {name} -> {off} ({done*100//total}%)", file=sys.stderr, flush=True)
else:
    sys.exit("mode must be pack|unpack")
print(f"{mode.upper()}-DONE {len(man)} tensors {total} bytes", flush=True)
