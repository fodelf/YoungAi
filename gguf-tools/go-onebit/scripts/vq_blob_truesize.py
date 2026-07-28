#!/usr/bin/env python3
"""vq_blob_truesize.py — dql_vq 侧车真载荷尺寸预扫(2026-07-28, 合并修剪用)。
侧车文件是预分配的(920.8MiB/层预留), 真载荷 = 最后一个 DQVQ 段的末尾。
扫描口径与 gr_refit_layer.parse_segments 同源: 段头 [DQVQ][dim u16][nc u16][rows u32][cols u32],
索引流 dim4=9bit 打包 / dim8=1B。输出每层 "L 真尺寸" 到 stdout(供 vq_merge_v4 --blob-sizes)。
用法(M1): python3 vq_blob_truesize.py /Users/fodelf/ds4-main/gguf/go-onebit/layers [NL=43]
"""
import mmap, os, struct, sys

MOEI, D = 2048, 4096


def true_size(path):
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        assert mm[:4] == b"DQVL", f"{path}: 非 DQVL"
        end, i = 16 + 256 * 3 * 8, 0
        while True:
            j = mm.find(b"DQVQ", i)
            if j < 0:
                break
            dim, nc, rows, cols = struct.unpack_from("<HHII", mm, j + 4)
            if dim in (4, 8) and nc in (256, 512) and rows in (MOEI, D) and cols in (MOEI, D):
                nidx = rows * cols // dim
                nb = nidx if (dim == 8 and nc == 256) else (nidx * 9 + 7) // 8 + 1
                seg_end = j + 16 + nc * dim * 2 + rows * 2 + nb
                if seg_end > end:
                    end = seg_end
                i = seg_end
            else:
                i = j + 4
        mm.close()
    assert end <= sz, f"{path}: 段尾 {end} 超文件 {sz}"
    return end


def main():
    d = sys.argv[1]
    nl = int(sys.argv[2]) if len(sys.argv) > 2 else 43
    tot_t = tot_f = 0
    for L in range(nl):
        p = f"{d}/dql_vq_L{L:02d}.bin"
        t, fsz = true_size(p), os.path.getsize(p)
        tot_t += t; tot_f += fsz
        print(f"{L} {t}")
        print(f"L{L:02d} true={t/2**20:.1f}MiB file={fsz/2**20:.1f}MiB slack={(fsz-t)/2**20:.1f}MiB",
              file=sys.stderr)
    print(f"Σ true={tot_t/2**30:.2f}GiB file={tot_f/2**30:.2f}GiB 省={(tot_f-tot_t)/2**30:.2f}GiB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
