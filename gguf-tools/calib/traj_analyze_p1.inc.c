/* traj_analyze.c — Go-domain "trajectory" (go轨迹) structure analyzer.
 *
 * Purpose: find the LOW-DIMENSIONAL regularity in a DeepSeek-V4 MoE model's
 * expert ROUTING / COMBINATION and its activation manifold, *purely* from
 * pre-captured calibration data — NO model is loaded. The point is to expose
 * structure that a Go-specialized quantizer can exploit (which experts are
 * Go-dead and droppable, how few distinct top-6 combinations Go actually uses,
 * how correlated the routing path is across depth, and how low the effective
 * rank of the FFN activation manifold is).
 *
 * Input: a capture directory (default /private/tmp/m1_ds4/cap_m1) holding, for
 * every transformer layer L in 0..42:
 *   route_L<L>.npy   (12288, 6)    int16  — top-6 routed expert IDs per token
 *   ffn_in_L<L>.npy  (12288, 4096) float16 — FFN input activations
 *   ffn_out_L<L>.npy (12288, 4096) float16 — MoE output activations
 * 256 routed experts, top-6 per token, 12288 Go tokens, 43 layers.
 *
 * Four analyses are printed as tables, then a summary:
 *   PART 1  per-layer expert-usage concentration + global dead/always-on counts
 *   PART 2  per-layer distinct top-6 combination count (combination diversity)
 *   PART 3  cross-layer routing path structure (Jaccard / primary agreement)
 *   PART 4  activation manifold effective rank via PCA (Gram-trick eigenspectrum)
 *
 * Memory policy: all 43 tiny route files are loaded once into a 3.2 MB uint8
 * buffer (expert IDs fit a byte) and reused by parts 1-3. The big ffn files
 * (part 4) are read one at a time (~200 MB as float32), immediately subsampled
 * into a 2048-row double matrix, then freed — peak RSS stays well under 1 GB.
 *
 * Pure C99/C11. Links only npy.c (reader) and linalg_small.c (sym_eig_topk).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "npy.h"
#include "linalg_small.h"

#define NLAYERS     43
#define NTOK        12288
#define TOPK        6
#define NEXPERT     256
#define FFN_DIM     4096
#define FIRINGS     (NTOK * TOPK)        /* 73728 expert firings per layer */

/* PART 4 PCA knobs. Subsample 2048 of 12288 tokens evenly (stride 6). The
 * Gram matrix is then 2048x2048; we ask for the top PCA_K eigenpairs. */
#define PCA_SAMPLES 2048
#define PCA_STRIDE  (NTOK / PCA_SAMPLES) /* = 6, exact */
#define PCA_K       256
#define PCA_ITERS   100                  /* power-iteration budget per eigenvector */
#define PCA_SEED    0xD5A4C0DEULL

/* ------------------------------------------------------------------ */
/* small utilities                                                     */
/* ------------------------------------------------------------------ */

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* insertion-sort 6 bytes ascending, in place */
static void sort6(uint8_t *v) {
    for (int i = 1; i < TOPK; i++) {
        uint8_t key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
        v[j + 1] = key;
    }
}

/* intersection size of two ascending 6-element id sets */
static int inter6(const uint8_t *a, const uint8_t *b) {
    int i = 0, j = 0, n = 0;
    while (i < TOPK && j < TOPK) {
        if (a[i] == b[j]) { n++; i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    return n;
}

static double wall_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* route loading                                                       */
/* ------------------------------------------------------------------ */

/* routes_orig: 43*12288*6 uint8, original top-k order (col 0 = primary expert)
 * routes_sort: same, each token's 6 ids sorted ascending (for set ops)        */
static uint8_t *routes_orig = NULL;
static uint8_t *routes_sort = NULL;

static inline const uint8_t *row_orig(int L, int t) {
    return routes_orig + ((size_t)L * NTOK + t) * TOPK;
}
static inline const uint8_t *row_sort(int L, int t) {
    return routes_sort + ((size_t)L * NTOK + t) * TOPK;
}

static int load_all_routes(const char *cap) {
    size_t bytes = (size_t)NLAYERS * NTOK * TOPK;
    routes_orig = (uint8_t *)malloc(bytes);
    routes_sort = (uint8_t *)malloc(bytes);
    if (!routes_orig || !routes_sort) { fprintf(stderr, "OOM routes\n"); return -1; }

    for (int L = 0; L < NLAYERS; L++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/route_L%d.npy", cap, L);
        npy_meta m;
        float *r = npy_read_f32(path, &m);
        if (!r) { fprintf(stderr, "cannot read %s\n", path); return -1; }
        if (m.ndim != 2 || m.shape[0] != NTOK || m.shape[1] != TOPK) {
            fprintf(stderr, "%s: unexpected shape\n", path);
            free(r); return -1;
        }
        for (int t = 0; t < NTOK; t++) {
            uint8_t *o = routes_orig + ((size_t)L * NTOK + t) * TOPK;
            uint8_t *s = routes_sort + ((size_t)L * NTOK + t) * TOPK;
            for (int j = 0; j < TOPK; j++) {
                long id = lroundf(r[(size_t)t * TOPK + j]);
                if (id < 0) id = 0;
                if (id > NEXPERT - 1) id = NEXPERT - 1;
                o[j] = (uint8_t)id;
                s[j] = (uint8_t)id;
            }
            sort6(s);
        }
        free(r);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* PART 1 — expert usage concentration                                 */
/* ------------------------------------------------------------------ */

/* global per-id tallies, accumulated across layers */
static long long g_count[NEXPERT];   /* total firings of id e over whole net */
static int       g_layers_fired[NEXPERT]; /* # layers in which id e ever fires */

/* running sums for the summary */
static double sum_n50 = 0, sum_n90 = 0, sum_n95 = 0, sum_n99 = 0;
static double sum_dead = 0, sum_Hnorm = 0;

static void part1(void) {
    printf("================================================================================\n");
    printf("PART 1  Expert usage concentration  (per layer, over %d firings = %d tok x top-%d)\n",
           FIRINGS, NTOK, TOPK);
    printf("  nXX = # experts (sorted desc) whose cumulative firings first reach XX%% of total\n");
    printf("  dead = experts that NEVER fire in this layer (Go-unused -> droppable)\n");
    printf("  H/8  = usage entropy in bits / 8  (1.0 = perfectly uniform over 256 experts)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("   L |  n50  n90  n95  n99 | dead |  H(bits)  H/8\n");
    printf("-----+---------------------+------+-----------------\n");

    for (int e = 0; e < NEXPERT; e++) { g_count[e] = 0; g_layers_fired[e] = 0; }

    for (int L = 0; L < NLAYERS; L++) {
        long cnt[NEXPERT];
        for (int e = 0; e < NEXPERT; e++) cnt[e] = 0;

        for (int t = 0; t < NTOK; t++) {
            const uint8_t *o = row_orig(L, t);
            for (int j = 0; j < TOPK; j++) cnt[o[j]]++;
        }

        long total = 0;
        for (int e = 0; e < NEXPERT; e++) {
            total += cnt[e];
            g_count[e] += cnt[e];
            if (cnt[e] > 0) g_layers_fired[e]++;
        }

        /* sorted-desc cumulative thresholds */
        long sc[NEXPERT];
        memcpy(sc, cnt, sizeof(sc));
        /* simple desc sort (256 elems) */
        for (int a = 0; a < NEXPERT; a++) {
            int best = a;
            for (int b = a + 1; b < NEXPERT; b++) if (sc[b] > sc[best]) best = b;
            if (best != a) { long tmp = sc[a]; sc[a] = sc[best]; sc[best] = tmp; }
        }
        int n50 = 0, n90 = 0, n95 = 0, n99 = 0;
        long cum = 0;
        for (int a = 0; a < NEXPERT; a++) {
            cum += sc[a];
            double f = (double)cum / (double)total;
            if (!n50 && f >= 0.50) n50 = a + 1;
            if (!n90 && f >= 0.90) n90 = a + 1;
            if (!n95 && f >= 0.95) n95 = a + 1;
            if (!n99 && f >= 0.99) n99 = a + 1;
        }

        int dead = 0;
        double H = 0.0;
        for (int e = 0; e < NEXPERT; e++) {
            if (cnt[e] == 0) { dead++; continue; }
            double p = (double)cnt[e] / (double)total;
            H -= p * log2(p);
        }
        double Hnorm = H / 8.0;

        printf("  %2d | %4d %4d %4d %4d | %4d | %8.4f  %.4f\n",
               L, n50, n90, n95, n99, dead, H, Hnorm);

        sum_n50 += n50; sum_n90 += n90; sum_n95 += n95; sum_n99 += n99;
        sum_dead += dead; sum_Hnorm += Hnorm;
    }

    printf("-----+---------------------+------+-----------------\n");
    printf(" avg | %4.0f %4.0f %4.0f %4.0f | %4.0f | %8s  %.4f\n",
           sum_n50 / NLAYERS, sum_n90 / NLAYERS, sum_n95 / NLAYERS, sum_n99 / NLAYERS,
           sum_dead / NLAYERS, "-", sum_Hnorm / NLAYERS);

    /* global union/intersection over layers */
    int global_dead = 0, used_all = 0, used_some = 0;
    for (int e = 0; e < NEXPERT; e++) {
        if (g_layers_fired[e] == 0)        global_dead++;
        else                               used_some++;
        if (g_layers_fired[e] == NLAYERS)  used_all++;
    }
    printf("\n  GLOBAL (expert ID treated as an index across all %d layers):\n", NLAYERS);
    printf("    experts that fire in at least one layer ......... %3d / 256\n", used_some);
    printf("    experts NEVER firing anywhere (network-dead) .... %3d / 256\n", global_dead);
    printf("    experts firing in ALL %d layers ................. %3d / 256\n", NLAYERS, used_all);
    printf("    (note: each layer has its own physical experts; this index view shows how\n");
    printf("     uniformly the same ID slot is exercised across depth.)\n\n");
}

/* ------------------------------------------------------------------ */
/* PART 2 — combination diversity                                      */
/* ------------------------------------------------------------------ */

static double sum_frac_unique = 0;
static long   min_distinct = 1L << 60, max_distinct = 0;
static long   distinct_L[NLAYERS];      /* distinct top-6 sets per layer */

/* key figures captured for the data-driven summary */
static double g_rnd_jac = 0;            /* analytic random-pair Jaccard */
static double g_dL1_jac = 0;            /* avg top-6 Jaccard at separation 1 */
static double g_l0l1_agree = 0, g_l0l1_chance = 0;  /* L0<->L1 primary agreement */
/* PART 4 results in load order: in0,out0,in21,out21,in42,out42 */
static char   g_pca_label[6][32];
static double g_pca_cum128[6], g_pca_lam1[6];
static int    g_pca_rank95[6];          /* -1 => not reached within PCA_K */

static void part2(void) {
    printf("================================================================================\n");
    printf("PART 2  Combination diversity  (distinct sorted top-%d sets among %d tokens)\n", TOPK, NTOK);
    printf("  low distinct count / low frac => Go reuses few expert combinations (low-dim path)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("   L | distinct top-6 sets | frac unique\n");
    printf("-----+---------------------+------------\n");

    uint64_t *keys = (uint64_t *)malloc((size_t)NTOK * sizeof(uint64_t));
    if (!keys) { fprintf(stderr, "OOM keys\n"); return; }

    for (int L = 0; L < NLAYERS; L++) {
        for (int t = 0; t < NTOK; t++) {
            const uint8_t *s = row_sort(L, t);   /* already ascending */
            uint64_t k = 0;
            for (int j = 0; j < TOPK; j++) k = (k << 8) | (uint64_t)s[j];
            keys[t] = k;
        }
        qsort(keys, NTOK, sizeof(uint64_t), cmp_u64);
        long distinct = 1;
        for (int t = 1; t < NTOK; t++) if (keys[t] != keys[t - 1]) distinct++;

        double frac = (double)distinct / (double)NTOK;
        printf("  %2d | %19ld | %.4f\n", L, distinct, frac);

        distinct_L[L] = distinct;
        sum_frac_unique += frac;
        if (distinct < min_distinct) min_distinct = distinct;
        if (distinct > max_distinct) max_distinct = distinct;
    }
    printf("-----+---------------------+------------\n");
    printf(" avg | %19s | %.4f\n", "-", sum_frac_unique / NLAYERS);
    printf("  (theoretical max distinct = %d; combinatorial space C(256,6) ~ 3.7e11)\n\n", NTOK);
    free(keys);
}

