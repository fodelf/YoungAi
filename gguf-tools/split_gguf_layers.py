#!/usr/bin/env python3
"""Split a DS4 GGUF into a per-machine layer shard (for dual-host sharding).

Usage:
    split_gguf_layers.py MODEL.gguf --keep-layers 8:42 --head output  > shard.gguf
    split_gguf_layers.py MODEL.gguf --keep-layers 0:7  --head embed   > shard.gguf

Keeps blk.{L}.* for L in [lo,hi], plus the chosen head half, plus all non-blk
global tensors, plus ALL metadata KV verbatim (incl. ds4.expert_keep_map.* — the
loader only consults the entries for layers it holds). Writes a self-contained
GGUF to stdout so it can be piped straight to the worker:
    python split_gguf_layers.py m.gguf --keep-layers 8:42 --head output | ssh M1 'cat > shard.gguf'

The matching ds4.c weights_bind skips absent layers / head halves, so each shard
loads with --layers matching its range. Whole-model GGUFs are untouched.
"""
import sys, struct, argparse

MAGIC = 0x46554747
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
SCALAR = {T_U8:1,T_I8:1,T_U16:2,T_I16:2,T_U32:4,T_I32:4,T_F32:4,T_BOOL:1,T_U64:8,T_I64:8,T_F64:8}

def ru32(fp): return struct.unpack("<I", fp.read(4))[0]
def ru64(fp): return struct.unpack("<Q", fp.read(8))[0]

def skip_value(fp, t, aln_box):
    if t == T_STR:
        n = ru64(fp); fp.read(n)
    elif t == T_ARR:
        et = ru32(fp); n = ru64(fp)
        for _ in range(n): skip_value(fp, et, aln_box)
    else:
        fp.read(SCALAR[t])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--keep-layers", required=True, help="lo:hi inclusive")
    ap.add_argument("--head", choices=["embed","output","both","none"], default="both")
    ap.add_argument("--report", action="store_true", help="print shard size to stderr and exit")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.keep_layers.split(":"))

    fp = open(a.src, "rb")
    assert ru32(fp) == MAGIC, "not a GGUF"
    version = ru32(fp); n_tensors = ru64(fp); n_kv = ru64(fp)

    # --- KV: parse to delimit byte range (copied verbatim) + extract alignment ---
    kv_start = fp.tell(); alignment = 32
    for _ in range(n_kv):
        klen = ru64(fp); key = fp.read(klen)
        vt = ru32(fp)
        if key == b"general.alignment" and vt == T_U32:
            alignment = ru32(fp)
        else:
            skip_value(fp, vt, None)
    kv_end = fp.tell()
    fp.seek(kv_start); kv_bytes = fp.read(kv_end - kv_start)

    # --- tensor infos ---
    infos = []  # [name(bytes), dims, type, orig_off]
    for _ in range(n_tensors):
        nlen = ru64(fp); name = fp.read(nlen)
        nd = ru32(fp); dims = [ru64(fp) for _ in range(nd)]
        typ = ru32(fp); off = ru64(fp)
        infos.append([name, dims, typ, off])
    info_end = fp.tell()
    data_start = (info_end + alignment - 1)//alignment*alignment
    fp.seek(0, 2); file_size = fp.tell()

    # original size of each tensor = gap to next tensor in offset order
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    size = [0]*len(infos)
    for k, i in enumerate(order):
        nxt = data_start + infos[order[k+1]][3] if k+1 < len(order) else file_size
        size[i] = nxt - (data_start + infos[i][3])

    def keep(name):
        s = name.decode("utf-8","replace")
        if s.startswith("blk."):
            L = int(s.split(".")[1]); return lo <= L <= hi
        if s == "token_embd.weight": return a.head in ("embed","both")
        if s.startswith("output"):   return a.head in ("output","both")  # output, output_norm, output_hc_*
        return True  # other globals (rope, etc.)
    kept = [i for i in range(len(infos)) if keep(infos[i][0])]
    total = sum(size[i] for i in kept)
    print(f"shard: keep blk[{lo}:{hi}] head={a.head}  {len(kept)}/{len(infos)} tensors  {total/1073741824:.2f} GiB", file=sys.stderr)
    if a.report:
        return

    # --- new offsets (contiguous, aligned), in kept order ---
    new_off = {}; cur = 0
    for i in kept:
        cur = (cur + alignment - 1)//alignment*alignment
        new_off[i] = cur; cur += size[i]

    out = sys.stdout.buffer
    out.write(struct.pack("<IIQQ", MAGIC, version, len(kept), n_kv))
    out.write(kv_bytes)
    for i in kept:
        nm, dims, typ, _ = infos[i]
        out.write(struct.pack("<Q", len(nm))); out.write(nm)
        out.write(struct.pack("<I", len(dims)))
        for d in dims: out.write(struct.pack("<Q", d))
        out.write(struct.pack("<IQ", typ, new_off[i]))
    pos = out.tell() if out.seekable() else (24 + len(kv_bytes) + sum(8+len(infos[i][0])+4+8*len(infos[i][1])+12 for i in kept))
    new_data_start = (pos + alignment - 1)//alignment*alignment
    out.write(b"\0" * (new_data_start - pos))
    # stream data in new-offset order (== kept order)
    written = 0
    for i in kept:
        pad = new_off[i] - written
        if pad: out.write(b"\0"*pad); written += pad
        fp.seek(data_start + infos[i][3])
        remaining = size[i]
        while remaining:
            chunk = fp.read(min(8*1024*1024, remaining))
            if not chunk: break
            out.write(chunk); remaining -= len(chunk); written += len(chunk)
    out.flush()

if __name__ == "__main__":
    main()
