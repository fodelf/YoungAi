#!/usr/bin/env python3
"""go2b_backbone_extract.py — 从完整 HF 抽"骨干子集"safetensors(剔除 routed 专家)。
用途: BF_ONLY 反修/rr_verdict 回放只读骨干(attention/norm/router/shared/embed/head),
专家从 dql/go2b 侧车字节回放 → M4(无完整HF)只需 ~≤15G 骨干子集即可跑反修。
纯字节拷贝(不解码不重编码), 输出 dir: backbone-0000N.safetensors + model.safetensors.index.json。
用法: DS4_HF=<完整HF目录> go2b_backbone_extract.py <输出目录>
"""
import os, sys, json, re, struct

EXPERT = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
CHUNK = 8 << 30   # 单输出 shard 上限 8GB

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def main():
    src = os.environ.get("DS4_HF") or sys.exit("需 DS4_HF")
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(f"{src}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    keep = {}   # shard -> [names]
    for name, shard in wm.items():
        if EXPERT.match(name): continue
        keep.setdefault(shard, []).append(name)
    total = sum(1 for ns in keep.values() for _ in ns)
    print(f"[backbone] 保留张量 {total} 个, 来源 shard {len(keep)} 个", flush=True)
    out_i = 1; out_f = None; out_hdr = {}; out_off = 0; new_map = {}; written = 0
    def flush_shard():
        nonlocal out_f, out_hdr, out_off, out_i
        if not out_f: return
        # ★紧凑 JSON 必须★: st_read.c st_find 按 "dtype":" 无空格匹配, 默认 json.dumps 的
        # ": " 会静默解析失败 → NULL 权重 → 段错误(2026-07-24 反修首跑实证)
        hdr = json.dumps(out_hdr, separators=(",", ":")).encode()
        pad = (-len(hdr)) % 8; hdr += b" " * pad
        final = f"{out}/backbone-{out_i:05d}.safetensors"
        with open(final, "wb") as g:
            g.write(struct.pack("<Q", len(hdr))); g.write(hdr)
            with open(out_f, "rb") as t:
                while True:
                    b = t.read(64 << 20)
                    if not b: break
                    g.write(b)
        os.remove(out_f)
        print(f"[backbone] {final} 张量={len(out_hdr)} 数据={out_off>>20}MiB", flush=True)
        out_f = None; out_hdr = {}; out_off = 0; out_i += 1
    for shard, names in sorted(keep.items()):
        hdr, base = read_header(f"{src}/{shard}")
        with open(f"{src}/{shard}", "rb") as f:
            for name in names:
                info = hdr[name]; b, e = info["data_offsets"]; sz = e - b
                if out_f and out_off + sz > CHUNK: flush_shard()
                if not out_f:
                    out_f = f"{out}/.tmp_shard"; open(out_f, "wb").close()
                f.seek(base + b)
                with open(out_f, "ab") as t:
                    left = sz
                    while left > 0:
                        blk = f.read(min(left, 64 << 20)); t.write(blk); left -= len(blk)
                out_hdr[name] = {"dtype": info["dtype"], "shape": info["shape"],
                                 "data_offsets": [out_off, out_off + sz]}
                new_map[name] = f"backbone-{out_i:05d}.safetensors"
                out_off += sz; written += sz
    flush_shard()
    json.dump({"metadata": {"total_size": written}, "weight_map": new_map},
              open(f"{out}/model.safetensors.index.json", "w"))
    for extra in ("tokenizer.json", "config.json"):
        p = f"{src}/{extra}"
        if os.path.isfile(p):
            import shutil; shutil.copy(p, f"{out}/{extra}")
    print(f"BACKBONE-OK {out} 总数据={written>>30}GiB 张量={len(new_map)}", flush=True)

if __name__ == "__main__":
    main()
