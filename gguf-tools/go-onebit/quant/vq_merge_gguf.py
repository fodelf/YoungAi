#!/usr/bin/env python3
"""vq_merge_gguf.py — 合并 base GGUF + DS4_VQ_DIR 侧车 → 单一 VQ 模型 GGUF (2026-07-27)。

产物 = base 全部张量, 除去被 VQ 覆盖的死重 blk.*.ffn_{gate,up}_exps.weight(go1b, 22.85 GiB),
      + 每层一个 blk.{L}.ffn_exps_vq.blob 张量(type=42 "vqblob", 1B/元素, 字节=DQVL 文件原样)。
消费端已在引擎: ds4.c residual 通道 type==42 → rs.vq=1; ffn_down_exps 整张保留(冷 w2 源, 索引零改动)。

磁盘纪律(用户授权"边合并边删除产物"):
  - 先写 base 保留张量(净增 ~19.6G), 后逐层写 blob;
  - 每层 blob: 写入 → F_FULLFSYNC → 从输出文件读回与侧车逐字节比对 → 比对过才删侧车文件;
  - base GGUF 永不在本脚本删(行为门后人工删) → 任一步失败时信息零丢失
    (已删侧车的字节已验证存活于输出文件, 可由 extract 逆向恢复)。

用法: vq_merge_gguf.py BASE.gguf VQ_DIR OUT.gguf [--dry-run] [--keep-sidecars]
"""
import os, struct, sys, fcntl

ALIGN = 32
DROP_PAT = ("ffn_gate_exps.weight", "ffn_up_exps.weight")
VQBLOB_TYPE = 42
N_LAYER = 43


def rd(f, n):
    b = f.read(n)
    assert len(b) == n, "short read"
    return b


def parse_header(f):
    magic, ver, n_t, n_kv = struct.unpack("<IIQQ", rd(f, 24))
    assert magic == 0x46554747 and ver == 3, f"bad gguf magic/ver {magic:#x}/{ver}"
    kv_start = f.tell()

    def rstr():
        n, = struct.unpack("<Q", rd(f, 8))
        return rd(f, n).decode()

    def skip_val(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rstr()
        elif t == 9:
            et, = struct.unpack("<I", rd(f, 4))
            n, = struct.unpack("<Q", rd(f, 8))
            for _ in range(n):
                skip_val(et)
        else:
            rd(f, sz[t])

    for _ in range(n_kv):
        rstr()
        t, = struct.unpack("<I", rd(f, 4))
        skip_val(t)
    kv_end = f.tell()
    f.seek(kv_start)
    kv_raw = rd(f, kv_end - kv_start)

    tens = []
    for _ in range(n_t):
        nm = rstr()
        nd, = struct.unpack("<I", rd(f, 4))
        ne = list(struct.unpack("<%dQ" % nd, rd(f, 8 * nd)))
        ty, = struct.unpack("<I", rd(f, 4))
        off, = struct.unpack("<Q", rd(f, 8))
        tens.append({"name": nm, "ne": ne, "type": ty, "off": off})
    data0 = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
    return n_t, n_kv, kv_raw, tens, data0


def tensor_sizes_by_offset(tens, file_size, data0):
    # 真实字节 = 相邻 offset 差(与 bpw_audit 同法, 不依赖 type 表, 自定义类型亦准)
    order = sorted(tens, key=lambda t: t["off"])
    for i, t in enumerate(order):
        end = order[i + 1]["off"] if i + 1 < len(order) else file_size - data0
        t["bytes"] = end - t["off"]
    return tens


def ser_tensor_info(t):
    b = struct.pack("<Q", len(t["name"])) + t["name"].encode()
    b += struct.pack("<I", len(t["ne"]))
    b += struct.pack("<%dQ" % len(t["ne"]), *t["ne"])
    b += struct.pack("<IQ", t["type"], t["off"])
    return b


def full_fsync(fobj):
    fobj.flush()
    try:
        fcntl.fcntl(fobj.fileno(), fcntl.F_FULLFSYNC)
    except OSError:
        os.fsync(fobj.fileno())


def copy_range(src, src_off, dst, nbytes, chunk=1 << 26):
    src.seek(src_off)
    left = nbytes
    while left:
        b = src.read(min(chunk, left))
        assert b, "short read in copy"
        dst.write(b)
        left -= len(b)


def compare_out_vs_file(out_path, out_off, ref_path, nbytes, chunk=1 << 26):
    with open(out_path, "rb") as a, open(ref_path, "rb") as b:
        a.seek(out_off)
        left = nbytes
        while left:
            n = min(chunk, left)
            if a.read(n) != b.read(n):
                return False
            left -= n
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    keep = "--keep-sidecars" in sys.argv
    base_p, vq_dir, out_p = args

    fsz = os.path.getsize(base_p)
    with open(base_p, "rb") as f:
        n_t, n_kv, kv_raw, tens, data0 = parse_header(f)
    tensor_sizes_by_offset(tens, fsz, data0)

    keep_t = [t for t in tens if not t["name"].endswith(DROP_PAT)]
    drop_t = [t for t in tens if t["name"].endswith(DROP_PAT)]
    # 断点续跑(2026-07-27 ENOSPC 事故): 已删侧车的层 → 其字节已验证存活于输出文件
    # (删除前逐字节比对过), 尺寸从上次的输出文件张量表回读; 在场侧车照常写+验+删。
    prev_blob_sz = {}
    if os.path.exists(out_p):
        with open(out_p, "rb") as pf:
            try:
                _, _, _, ptens, _ = parse_header(pf)
                for t in ptens:
                    if t["name"].endswith("ffn_exps_vq.blob"):
                        prev_blob_sz[t["name"]] = t["ne"][0]
            except Exception as e:  # noqa: BLE001
                print(f"[merge] 旧输出头不可读({e}) — 全量重写")
    blobs = []
    for il in range(N_LAYER):
        p = os.path.join(vq_dir, f"dql_vq_L{il:02d}.bin")
        nm = f"blk.{il}.ffn_exps_vq.blob"
        if os.path.exists(p):
            with open(p, "rb") as f:
                assert f.read(4) == b"DQVL", f"bad magic in {p}"
            blobs.append({"name": nm, "path": p, "bytes": os.path.getsize(p)})
        else:
            assert nm in prev_blob_sz, f"侧车 {p} 缺席且旧输出无该 blob — 无源可恢复"
            blobs.append({"name": nm, "path": None, "bytes": prev_blob_sz[nm]})

    kept_bytes = sum(t["bytes"] for t in keep_t)
    drop_bytes = sum(t["bytes"] for t in drop_t)
    blob_bytes = sum(b["bytes"] for b in blobs)

    # 新张量表: base 保留张量(原顺序) + 43 blob; 重排 offset(对齐 ALIGN)
    new_t = []
    off = 0
    for t in keep_t:
        e = dict(t)
        e["off"] = off
        off += (t["bytes"] + ALIGN - 1) // ALIGN * ALIGN
        new_t.append(e)
    for b in blobs:
        new_t.append({"name": b["name"], "ne": [b["bytes"]], "type": VQBLOB_TYPE,
                      "off": off, "bytes": b["bytes"], "path": b["path"]})
        off += (b["bytes"] + ALIGN - 1) // ALIGN * ALIGN

    hdr = struct.pack("<IIQQ", 0x46554747, 3, len(new_t), n_kv) + kv_raw
    info = b"".join(ser_tensor_info(t) for t in new_t)
    ndata0 = (len(hdr) + len(info) + ALIGN - 1) // ALIGN * ALIGN
    total = ndata0 + off

    print(f"[merge] base 张量 {n_t} → 保留 {len(keep_t)} + blob {len(blobs)} (丢 {len(drop_t)})")
    print(f"[merge] 保留 {kept_bytes/2**30:.2f} GiB | 死重丢弃 {drop_bytes/2**30:.2f} GiB | blob {blob_bytes/2**30:.2f} GiB")
    print(f"[merge] 产物总大小 {total/2**30:.2f} GiB → {out_p}")
    if dry:
        return

    mode = "r+b" if os.path.exists(out_p) else "wb"
    with open(out_p, mode) as out, open(base_p, "rb") as src:
        out.seek(0)
        out.write(hdr)
        out.write(info)
        out.write(b"\x00" * (ndata0 - len(hdr) - len(info)))
        for i, t in enumerate(new_t[:len(keep_t)]):
            base_t = keep_t[i]           # 同序
            assert base_t["name"] == t["name"]
            out.seek(ndata0 + t["off"])
            copy_range(src, data0 + base_t["off"], out, base_t["bytes"])
            done_gib = (t["off"] + t["bytes"]) / 2**30
            if i % 100 == 0 or i == len(keep_t) - 1:
                print(f"[merge] base 张量 {i+1}/{len(keep_t)} 已写 {done_gib:.1f} GiB", flush=True)
        # 2) blob 逐层: 写 → FULLFSYNC → 读回比对 → 删侧车; 无侧车(续跑)= 验 DQVL 魔数即过
        for bi, t in enumerate(new_t[len(keep_t):]):
            if t["path"] is None:
                out.seek(ndata0 + t["off"])
                mg = out.read(4)
                assert mg == b"DQVL", f"续跑校验失败 {t['name']}: 输出文件该偏移非 DQVL ({mg!r})"
                print(f"[merge] blob {bi+1}/{len(blobs)} {t['name']} 续跑跳过(已验证在档)", flush=True)
                continue
            out.seek(ndata0 + t["off"])
            with open(t["path"], "rb") as bf:
                copy_range(bf, 0, out, t["bytes"])
            full_fsync(out)
            ok = compare_out_vs_file(out_p, ndata0 + t["off"], t["path"], t["bytes"])
            if not ok:
                print(f"[merge] ★比对失败★ {t['name']} — 侧车保留, 中止", flush=True)
                sys.exit(1)
            if not keep:
                os.remove(t["path"])
            print(f"[merge] blob {bi+1}/{len(blobs)} {t['name']} "
                  f"{t['bytes']/2**20:.0f} MiB 已验证{'(' + '侧车已删)' if not keep else ''}", flush=True)
        full_fsync(out)
    print(f"[merge] 完成: {out_p} {os.path.getsize(out_p)/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
