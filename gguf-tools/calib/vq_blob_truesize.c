/* vq_blob_truesize.c — dql_vq 侧车真载荷尺寸预扫(合并修剪用)。
 * (C, 2026-08-25 Python→C 迁移; 取代 scripts/vq_blob_truesize.py, stdout 逐字节同)
 * 侧车文件是预分配的(920.8MiB/层预留), 真载荷 = 最后一个 DQVQ 段的末尾。
 * 扫描口径与 gr_refit_layer.parse_segments 同源: 段头 [DQVQ][dim u16][nc u16][rows u32][cols u32],
 * 索引流 dim4=9bit 打包 / dim8=1B。输出每层 "L 真尺寸" 到 stdout(供 vq_merge_v4 --blob-sizes)。
 * 用法: vq_blob_truesize <layers_dir> [NL=43] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define MOEI 2048
#define D 4096

static long long true_size(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    struct stat st; fstat(fd, &st);
    long long sz = st.st_size;
    uint8_t *mm = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mm == MAP_FAILED) { fprintf(stderr, "%s mmap 失败\n", path); exit(2); }
    if (memcmp(mm, "DQVL", 4)) { fprintf(stderr, "AssertionError: %s: 非 DQVL\n", path); exit(1); }
    long long end = 16 + 256 * 3 * 8, i = 0;
    for (;;) {
        /* mm.find(b"DQVQ", i) */
        long long j = -1;
        for (long long k = i; k + 4 <= sz; k++)
            if (mm[k] == 'D' && !memcmp(mm + k, "DQVQ", 4)) { j = k; break; }
        if (j < 0) break;
        if (j + 16 > sz) break;
        uint16_t dim, nc; uint32_t rows, cols;
        memcpy(&dim, mm + j + 4, 2); memcpy(&nc, mm + j + 6, 2);
        memcpy(&rows, mm + j + 8, 4); memcpy(&cols, mm + j + 12, 4);
        if ((dim == 4 || dim == 8) && (nc == 256 || nc == 512) &&
            (rows == MOEI || rows == D) && (cols == MOEI || cols == D)) {
            long long nidx = (long long)rows * cols / dim;
            long long nb = (dim == 8 && nc == 256) ? nidx : (nidx * 9 + 7) / 8 + 1;
            long long seg_end = j + 16 + (long long)nc * dim * 2 + (long long)rows * 2 + nb;
            if (seg_end > end) end = seg_end;
            i = seg_end;
        } else i = j + 4;
    }
    munmap(mm, sz); close(fd);
    if (end > sz) { fprintf(stderr, "AssertionError: %s: 段尾 %lld 超文件 %lld\n", path, end, sz); exit(1); }
    return end;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: vq_blob_truesize <layers_dir> [NL=43]\n"); return 1; }
    int nl = argc > 2 ? atoi(argv[2]) : 43;
    long long tot_t = 0, tot_f = 0;
    for (int L = 0; L < nl; L++) {
        char p[1024]; snprintf(p, sizeof p, "%s/dql_vq_L%02d.bin", argv[1], L);
        long long t = true_size(p);
        struct stat st; stat(p, &st);
        long long fsz = st.st_size;
        tot_t += t; tot_f += fsz;
        printf("%d %lld\n", L, t);
        fprintf(stderr, "L%02d true=%.1fMiB file=%.1fMiB slack=%.1fMiB\n",
                L, t / 1048576.0, fsz / 1048576.0, (fsz - t) / 1048576.0);
    }
    fprintf(stderr, "Σ true=%.2fGiB file=%.2fGiB 省=%.2fGiB\n",
            tot_t / 1073741824.0, tot_f / 1073741824.0, (tot_f - tot_t) / 1073741824.0);
    return 0;
}
