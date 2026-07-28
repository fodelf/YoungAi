#!/usr/bin/env python3
"""dql_down_offset.py — 每层 dql_L 的 D 段权威偏移(2026-07-28)。
DQL2 = 'DQL2' u32 | L u32 | nrec u32, 然后 nrec 条记录:
  name[16] algo[64] vol u64 paysz u64 m1..m4 f32 verdict i32 (=116B) + payload[paysz]
[G 256×szG][U 256×szG][D 256×szD] 是 '1bit' 记录的载荷 → 记录链前缀逐层可变,
固定 35104 偏移是错误假设(L01 差 292B 实证)。D 偏移 = 1bit载荷起点 + 2*256*szG。
输出每层 "L d_offset" 到 stdout(供 vq_merge_v4 --down-offsets)。
用法(M1): python3 dql_down_offset.py /Users/fodelf/ds4-main/gguf/go-onebit/layers [NL=43]
"""
import os, struct, sys

NEXP, MOEI, DIM = 256, 2048, 4096
SZ_G = MOEI * (DIM // 256) * 34      # 1,114,112
SZ_D = DIM * (MOEI // 256) * 34      # 1,114,112
PAY_1BIT = NEXP * (2 * SZ_G + SZ_D)


def d_offset(path):
    fsz = os.path.getsize(path)
    with open(path, "rb") as f:
        mg, L, nrec = struct.unpack("<III", f.read(12))
        assert mg == 0x324C5144, f"{path}: 非 DQL2"
        pos = 12
        for _ in range(nrec):
            f.seek(pos)
            hdr = f.read(116)
            assert len(hdr) == 116, f"{path}: 记录头截断 @{pos}"
            name = hdr[:16].split(b"\0")[0].decode()
            paysz, = struct.unpack("<Q", hdr[88:96])
            if name == "1bit":
                assert paysz == PAY_1BIT, f"{path}: 1bit 载荷 {paysz} ≠ {PAY_1BIT}"
                off = pos + 116 + 2 * NEXP * SZ_G
                assert off + NEXP * SZ_D <= fsz, f"{path}: D 段越界"
                return pos + 116, off
            pos += 116 + paysz
    raise AssertionError(f"{path}: 无 1bit 记录")


def main():
    d = sys.argv[1]
    nl = int(sys.argv[2]) if len(sys.argv) > 2 else 43
    for L in range(nl):
        p = f"{d}/dql_L{L:02d}.bin"
        pay0, off = d_offset(p)
        print(f"{L} {off}")
        print(f"L{L:02d} 1bit载荷@{pay0} D段@{off}", file=sys.stderr)


if __name__ == "__main__":
    main()
