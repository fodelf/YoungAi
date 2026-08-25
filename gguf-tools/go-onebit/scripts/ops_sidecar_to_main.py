#!/usr/bin/env python3
"""ops_sidecar_to_main.py — op 侧车 → dql 主文件混装(还原超冠架构, 2026-08-08 用户令
"侧车不是我要求加入的")。

背景: 平行架构(2026-08-04, 非用户要求)把反修 op 从 dql 主文件挪进 dql_ops_L%02d.bin
侧车。超冠(573b7f5)架构 = op 记录直接混装在主文件里(lfile_load 见 vd==1 即 parse_op_rec)。
两者记录格式逐字节相同(REC_HDR=116), 所以还原=纯数据迁移, 不需要重新量化。

做法: 侧车里每条 REC 原样 append 到主文件末尾, 主文件 nrec += 实际条数, 侧车改名备份。
用法: ops_sidecar_to_main.py <layers_dir> [nl=43] [--dry]
"""
import os, struct, sys

REC_HDR = 116

def recs(raw):
    """按实际内容遍历(不信头部 nrec — 历史 nrec 污染案)。返回 [(off, total_len)]"""
    out, off = [], 12
    while off + REC_HDR <= len(raw):
        psz, = struct.unpack_from("<Q", raw, off + 88)
        if psz < 0 or off + REC_HDR + psz > len(raw):
            break
        out.append((off, REC_HDR + psz))
        off += REC_HDR + psz
    return out

def main():
    ld = sys.argv[1]
    nl = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 43
    dry = "--dry" in sys.argv
    tot_moved = 0
    for L in range(nl):
        mp = os.path.join(ld, f"dql_L{L:02d}.bin")
        sp = os.path.join(ld, f"dql_ops_L{L:02d}.bin")
        if not os.path.exists(mp):
            print(f"L{L:02d}: 主文件缺, 跳过"); continue
        if not os.path.exists(sp):
            print(f"L{L:02d}: 无侧车"); continue
        sraw = open(sp, "rb").read()
        if sraw[:4] != b"DQO2":
            print(f"L{L:02d}: 侧车魔数错 {sraw[:4]!r}, 跳过"); continue
        rs = recs(sraw)
        with open(mp, "rb") as f:
            head = f.read(12)
        assert head[:4] == b"DQL2", f"L{L}: 主文件魔数 {head[:4]!r}"
        mnrec, = struct.unpack_from("<I", head, 8)
        if not rs:
            print(f"L{L:02d}: 侧车 0 条 op(主 nrec={mnrec} 不变)")
            if not dry:
                os.rename(sp, sp + ".migrated")
            continue
        if dry:
            print(f"L{L:02d}: [dry] 将迁移 {len(rs)} 条 → 主 nrec {mnrec}→{mnrec+len(rs)}")
            continue
        with open(mp, "r+b") as f:
            f.seek(0, os.SEEK_END)
            for off, ln in rs:
                f.write(sraw[off:off + ln])
            f.seek(8)
            f.write(struct.pack("<I", mnrec + len(rs)))
        os.rename(sp, sp + ".migrated")
        tot_moved += len(rs)
        print(f"L{L:02d}: 迁移 {len(rs)} 条 ✓ 主 nrec {mnrec}→{mnrec+len(rs)}")
    print(f"合计迁移 {tot_moved} 条 op → 主文件混装(超冠架构); 侧车已改名 .migrated")

if __name__ == "__main__":
    main()
