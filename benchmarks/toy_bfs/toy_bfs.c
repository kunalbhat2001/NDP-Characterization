/*
 * toy_bfs.c  –  Direction-Optimizing BFS Microbenchmark
 *
 * Supports:
 *   1. Memory-mapped (.bin) pre-generated graph files (instant startup):
 *        ./toy_bfs -f graph_s16_d16.bin [-t num_threads]
 *        ./toy_bfs graph_s16_d16.bin [-t num_threads]
 *
 *   2. In-memory graph generation (for quick standalone runs):
 *        ./toy_bfs [scale] [avg_degree] [seed] [threads]
 *        ./toy_bfs 16 16 42 -t 4
 *
 * Multi-threading:
 *   Optionally multi-threaded using OpenMP (set with -t/--threads or OMP_NUM_THREADS).
 *   When run with 1 thread or compiled without OpenMP, runs single-threaded.
 *
 * ROI markers (xchg rbx,rbx) are compatible with the StarNUMA Pin tracer.
 * Tracing starts after mmap and warmup are complete.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "toy_graph.h"

#ifdef _OPENMP
#include <omp.h>
#else
static inline int  omp_get_max_threads(void) { return 1; }
static inline int  omp_get_thread_num(void)  { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

/* ============================================================
 *  Direction-switching thresholds (GAPBS defaults)
 * ============================================================ */
#define ALPHA 15
#define BETA  18

/* ============================================================
 *  ROI magic instructions for Intel Pin tracer
 * ============================================================ */
static inline void roi_start(void)
{
    volatile uint64_t d = 0;
    /* debug: print current instruction count */
    __asm__ __volatile__("xchg %%rbx, %%rbx;"
        ::"b"((uint64_t)(&d)), "c"((uint64_t)0x5));
    /* signal master ROI start */
    __asm__ __volatile__("xchg %%rbx, %%rbx;"
        ::"b"((uint64_t)(&d)), "c"((uint64_t)0x2));
}

static inline void roi_end(void)
{
    volatile uint64_t d = 0;
    __asm__ __volatile__("xchg %%rbx, %%rbx;"
        ::"b"((uint64_t)(&d)), "c"((uint64_t)0x4));
}

/* ============================================================
 *  Graph storage (CSR forward and backward)
 * ============================================================ */
static uint32_t  g_n;           /* number of nodes */
static uint64_t  g_e;           /* total directed edges in each CSR */

static const uint64_t *out_idx = NULL; /* [g_n + 1]  CSR row offsets  (out) */
static const uint32_t *out_nbr = NULL; /* [g_e]      CSR column values (out) */
static const uint64_t *in_idx  = NULL; /* [g_n + 1]  CSR row offsets  (in)  */
static const uint32_t *in_nbr  = NULL; /* [g_e]      CSR column values (in)  */

/* mmap tracking */
static void   *g_mmap_ptr = NULL;
static size_t  g_mmap_size = 0;
static int     g_mmap_fd = -1;

/* Dynamic 64-bit upgraded / fallback buffers */
static uint64_t *g_alloc_out_idx = NULL;
static uint32_t *g_alloc_out_nbr = NULL;
static uint64_t *g_alloc_in_idx  = NULL;
static uint32_t *g_alloc_in_nbr  = NULL;

/* ============================================================
 *  BFS state buffers
 * ============================================================ */
static int32_t   *g_parent = NULL;      /* [g_n]  -out_degree(v) if unvisited, else parent id */
static uint32_t  *g_qbuf = NULL;        /* [2*g_n]  two halves: current and next frontier queue */
static uint64_t  *g_bmap[2] = {NULL, NULL}; /* two bitmaps for BU direction (front / next) */

/* Thread-local queue buffers */
static int        g_num_threads = 1;
static uint32_t **g_thread_qbufs = NULL;

static void alloc_thread_buffers(int num_threads, uint32_t n)
{
    g_num_threads = (num_threads > 0) ? num_threads : 1;
    g_thread_qbufs = (uint32_t **)malloc((size_t)g_num_threads * sizeof(uint32_t *));
    if (!g_thread_qbufs) {
        perror("malloc thread qbufs table");
        exit(1);
    }
    for (int t = 0; t < g_num_threads; t++) {
        g_thread_qbufs[t] = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
        if (!g_thread_qbufs[t]) {
            perror("malloc per-thread qbuf");
            exit(1);
        }
    }
}

static void free_thread_buffers(void)
{
    if (g_thread_qbufs) {
        for (int t = 0; t < g_num_threads; t++) {
            if (g_thread_qbufs[t]) free(g_thread_qbufs[t]);
        }
        free(g_thread_qbufs);
        g_thread_qbufs = NULL;
    }
}

/* ============================================================
 *  Bitmap helpers
 * ============================================================ */
/* [COMPUTE-BOUND]: Pure ALU bit-shift and bit-mask instructions */
static inline int bm_get(const uint64_t *bm, uint32_t i)
{
    return (int)((bm[i >> 6] >> (i & 63)) & 1u);
}

/* [COMPUTE-BOUND]: Pure ALU bitwise OR and shift */
static inline void bm_set(uint64_t *bm, uint32_t i)
{
    bm[i >> 6] |= (uint64_t)1u << (i & 63);
}

/* [COMPUTE/PIPELINE & MEMORY-COHERENCY]: Atomic fetch-and-or introduces pipeline lock / coherency traffic */
static inline void bm_set_atomic(uint64_t *bm, uint32_t i)
{
    __sync_fetch_and_or(&bm[i >> 6], (uint64_t)1u << (i & 63));
}

/* [MEMORY-BANDWIDTH-BOUND]: Sequential streaming memory zeroing (saturates L1/L2/DRAM fill bandwidth) */
static inline void bm_clear_all(uint64_t *bm)
{
    memset(bm, 0, (((size_t)g_n + 63) >> 6) * sizeof(uint64_t));
}

/* ============================================================
 *  Memory-mapped Graph Loader
 * ============================================================ */
static int load_graph_mmap(const char *filename)
{
    g_mmap_fd = open(filename, O_RDONLY);
    if (g_mmap_fd < 0) {
        perror("open graph file");
        return -1;
    }

    struct stat st;
    if (fstat(g_mmap_fd, &st) < 0) {
        perror("fstat graph file");
        close(g_mmap_fd);
        return -1;
    }

    if ((size_t)st.st_size < sizeof(ToyGraphHeader)) {
        fprintf(stderr, "Error: '%s' is too small to be a valid ToyGraph file.\n", filename);
        close(g_mmap_fd);
        return -1;
    }

    g_mmap_size = st.st_size;
    g_mmap_ptr = mmap(NULL, g_mmap_size, PROT_READ, MAP_SHARED, g_mmap_fd, 0);
    if (g_mmap_ptr == MAP_FAILED) {
        perror("mmap graph file");
        close(g_mmap_fd);
        return -1;
    }

    const ToyGraphHeader *hdr = (const ToyGraphHeader *)g_mmap_ptr;
    if (memcmp(hdr->magic, TOYGRAPH_MAGIC, 8) != 0) {
        fprintf(stderr, "Error: '%s' has invalid header magic (expected '%s').\n",
                filename, TOYGRAPH_MAGIC);
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
        return -1;
    }

    if (hdr->version != TOYGRAPH_VERSION_32 && hdr->version != TOYGRAPH_VERSION_64) {
        fprintf(stderr, "Error: unsupported ToyGraph version %u (supported: %u or %u)\n",
                hdr->version, TOYGRAPH_VERSION_32, TOYGRAPH_VERSION_64);
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
        return -1;
    }

    g_n = hdr->num_nodes;
    g_e = hdr->num_edges;

    if (hdr->version == TOYGRAPH_VERSION_64) {
        out_idx = (const uint64_t *)((const char *)g_mmap_ptr + hdr->out_idx_offset);
        in_idx  = (const uint64_t *)((const char *)g_mmap_ptr + hdr->in_idx_offset);
    } else {
        /* Upgrade 32-bit offsets to 64-bit, correcting 2^32 overflow wraps for >4B edges */
        const uint32_t *raw_out = (const uint32_t *)((const char *)g_mmap_ptr + hdr->out_idx_offset);
        g_alloc_out_idx = (uint64_t *)malloc(((size_t)g_n + 1) * sizeof(uint64_t));
        uint64_t wrap_out = 0;
        uint32_t prev_out = 0;
        for (uint64_t i = 0; i <= g_n; i++) {
            uint32_t val = raw_out[i];
            if (val < prev_out) {
                wrap_out += (1ULL << 32);
            }
            prev_out = val;
            g_alloc_out_idx[i] = wrap_out + (uint64_t)val;
        }
        out_idx = g_alloc_out_idx;

        const uint32_t *raw_in = (const uint32_t *)((const char *)g_mmap_ptr + hdr->in_idx_offset);
        g_alloc_in_idx = (uint64_t *)malloc(((size_t)g_n + 1) * sizeof(uint64_t));
        uint64_t wrap_in = 0;
        uint32_t prev_in = 0;
        for (uint64_t i = 0; i <= g_n; i++) {
            uint32_t val = raw_in[i];
            if (val < prev_in) {
                wrap_in += (1ULL << 32);
            }
            prev_in = val;
            g_alloc_in_idx[i] = wrap_in + (uint64_t)val;
        }
        in_idx = g_alloc_in_idx;
    }

    out_nbr = (const uint32_t *)((const char *)g_mmap_ptr + hdr->out_nbr_offset);
    in_nbr  = (const uint32_t *)((const char *)g_mmap_ptr + hdr->in_nbr_offset);

    /* Allocate BFS dynamic state buffers */
    g_parent  = (int32_t  *)malloc(g_n * sizeof(int32_t));
    g_qbuf    = (uint32_t *)malloc(2 * (size_t)g_n * sizeof(uint32_t));
    size_t bm_words = ((size_t)g_n + 63) >> 6;
    g_bmap[0] = (uint64_t *)calloc(bm_words, sizeof(uint64_t));
    g_bmap[1] = (uint64_t *)calloc(bm_words, sizeof(uint64_t));

    if (!g_parent || !g_qbuf || !g_bmap[0] || !g_bmap[1]) {
        perror("malloc BFS state");
        exit(1);
    }

    printf("[mmap] Successfully mapped binary graph '%s':\n", filename);
    printf("  Nodes           : %u (scale %u)\n", g_n, hdr->scale);
    printf("  Directed Edges  : %llu\n", (unsigned long long)g_e);
    printf("  Avg Degree      : %u\n", hdr->avg_deg);
    printf("  File Size       : %llu bytes (%.2f MB)\n\n",
           (unsigned long long)g_mmap_size,
           (double)g_mmap_size / (1024.0 * 1024.0));

    return 0;
}

/* ============================================================
 *  In-Memory Graph Construction (Fallback)
 * ============================================================ */
typedef struct { uint32_t u, v; } Edge;

static int cmp_by_u(const void *a, const void *b)
{
    const Edge *ea = (const Edge *)a, *eb = (const Edge *)b;
    if (ea->u != eb->u) return (ea->u < eb->u) ? -1 : 1;
    return (ea->v < eb->v) ? -1 : (ea->v > eb->v) ? 1 : 0;
}
static int cmp_by_v(const void *a, const void *b)
{
    const Edge *ea = (const Edge *)a, *eb = (const Edge *)b;
    if (ea->v != eb->v) return (ea->v < eb->v) ? -1 : 1;
    return (ea->u < eb->u) ? -1 : (ea->u > eb->u) ? 1 : 0;
}

static void build_graph_in_memory(uint32_t n, int avg_deg, unsigned seed)
{
    g_n = n;
    uint64_t n_raw = (uint64_t)n * (uint64_t)avg_deg / 2;
    uint64_t n_sym = n_raw * 2;
    g_e = n_sym;

    printf("[graph] Generating %llu undirected edges in memory (%u nodes, avg_deg~%d)...\n",
           (unsigned long long)n_raw, n, avg_deg);

    Edge *edges = (Edge *)malloc(n_sym * sizeof(Edge));
    if (!edges) { perror("malloc edges"); exit(1); }

    /* [COMPUTE-BOUND]: PRNG calculation and branch check */
    srand(seed);
    for (uint64_t i = 0; i < n_raw; i++) {
        uint32_t u = (uint32_t)((unsigned)rand() % n);
        uint32_t v = (uint32_t)((unsigned)rand() % n);
        if (u == v) v = (v + 1) % n;
        edges[2 * i]     = (Edge){u, v};
        edges[2 * i + 1] = (Edge){v, u};
    }

    g_alloc_out_idx = (uint64_t *)calloc((size_t)n + 1, sizeof(uint64_t));
    g_alloc_out_nbr = (uint32_t *)malloc(n_sym * sizeof(uint32_t));
    if (!g_alloc_out_idx || !g_alloc_out_nbr) { perror("malloc out-CSR"); exit(1); }

    /* [COMPUTE-BOUND]: O(E log E) comparisons and quicksort branches */
    qsort(edges, n_sym, sizeof(Edge), cmp_by_u);
    /* [MEMORY-BANDWIDTH-BOUND]: Sequential scan and prefix sum */
    for (uint64_t i = 0; i < n_sym; i++) g_alloc_out_idx[edges[i].u + 1]++;
    for (uint32_t i = 1; i <= n; i++)    g_alloc_out_idx[i] += g_alloc_out_idx[i - 1];
    for (uint64_t i = 0; i < n_sym; i++) g_alloc_out_nbr[i] = edges[i].v;

    g_alloc_in_idx = (uint64_t *)calloc((size_t)n + 1, sizeof(uint64_t));
    g_alloc_in_nbr = (uint32_t *)malloc(n_sym * sizeof(uint32_t));
    if (!g_alloc_in_idx || !g_alloc_in_nbr) { perror("malloc in-CSR"); exit(1); }

    /* [COMPUTE-BOUND]: O(E log E) comparisons and quicksort branches */
    qsort(edges, n_sym, sizeof(Edge), cmp_by_v);
    /* [MEMORY-BANDWIDTH-BOUND]: Sequential scan and prefix sum */
    for (uint64_t i = 0; i < n_sym; i++) g_alloc_in_idx[edges[i].v + 1]++;
    for (uint32_t i = 1; i <= n; i++)    g_alloc_in_idx[i] += g_alloc_in_idx[i - 1];
    for (uint64_t i = 0; i < n_sym; i++) g_alloc_in_nbr[i] = edges[i].u;

    free(edges);

    out_idx = g_alloc_out_idx;
    out_nbr = g_alloc_out_nbr;
    in_idx  = g_alloc_in_idx;
    in_nbr  = g_alloc_in_nbr;

    g_parent  = (int32_t  *)malloc(n * sizeof(int32_t));
    g_qbuf    = (uint32_t *)malloc(2 * (size_t)n * sizeof(uint32_t));
    size_t bm_words = ((size_t)n + 63) >> 6;
    g_bmap[0] = (uint64_t *)calloc(bm_words, sizeof(uint64_t));
    g_bmap[1] = (uint64_t *)calloc(bm_words, sizeof(uint64_t));
    if (!g_parent || !g_qbuf || !g_bmap[0] || !g_bmap[1]) {
        perror("malloc BFS state"); exit(1);
    }
}

/* ============================================================
 *  BFS Kernels
 * ============================================================ */

/* Top-Down step
 * ------------------------------------------------------------
 * Overall: DOMINATED BY MEMORY LATENCY (RANDOM ACCESS)
 * - cur[qi] lookup: Sequential / buffer read
 * - out_idx[u] & out_nbr[ei]: Sequential streaming read per node
 * - g_parent[v] load: IRREGULAR / RANDOM MEMORY LATENCY BOUND (causes cache misses)
 * - g_parent[v] write / CAS: RANDOM STORE & ATOMIC COHERENCY OVERHEAD
 * - scout count & queue arithmetic: COMPUTE / ALU (minority)
 */
__attribute__((noinline))
int64_t td_step(int64_t cur_sz, const uint32_t *cur,
                uint32_t *nxt, int64_t *nxt_sz_out)
{
    int64_t scout = 0;
    int64_t nq    = 0;

    #pragma omp parallel reduction(+ : scout)
    {
        int tid = omp_get_thread_num();
        uint32_t *l_nxt = (g_thread_qbufs && tid < g_num_threads) ? g_thread_qbufs[tid] : NULL;
        int64_t l_nq = 0;

        #pragma omp for schedule(dynamic, 64) nowait
        for (int64_t qi = 0; qi < cur_sz; qi++) {
            /* [MEMORY-BANDWIDTH-BOUND]: Reading active node from frontier queue */
            uint32_t u  = cur[qi];
            uint64_t eo = out_idx[u];
            uint64_t ee = out_idx[u + 1];

            /* [MEMORY-BANDWIDTH-BOUND]: Reading contiguous slice of outgoing neighbors */
            for (uint64_t ei = eo; ei < ee; ei++) {
                uint32_t v  = out_nbr[ei];

                /* [MEMORY-LATENCY-BOUND]: Highly random lookup into parent array for arbitrary neighbor v */
                int32_t  pv = g_parent[v];
                if (pv < 0) {
                    /* [MEMORY-LATENCY & COHERENCY / PIPELINE]: Atomic CAS random update */
                    if (__sync_bool_compare_and_swap(&g_parent[v], pv, (int32_t)u)) {
                        if (l_nxt) {
                            l_nxt[l_nq++] = v;
                        } else {
                            /* [COMPUTE/PIPELINE]: Atomic thread-safe queue index bump */
                            int64_t idx = __sync_fetch_and_add(&nq, 1);
                            nxt[idx] = v;
                        }
                        /* [COMPUTE-BOUND]: Degree accumulation arithmetic */
                        scout += (int64_t)(-pv);
                    }
                }
            }
        }

        /* [MEMORY-BANDWIDTH-BOUND]: Local thread buffer copy to global next frontier */
        if (l_nxt && l_nq > 0) {
            int64_t base = __sync_fetch_and_add(&nq, l_nq);
            memcpy(&nxt[base], l_nxt, (size_t)l_nq * sizeof(uint32_t));
        }
    }

    *nxt_sz_out = nq;
    return scout;
}

/* Bottom-Up step
 * ------------------------------------------------------------
 * Overall: HYBRID MEMORY-BANDWIDTH & MEMORY-LATENCY BOUND
 * - Sweeping u = 0..g_n-1 & in_idx[u]: STREAMING / MEMORY-BANDWIDTH BOUND
 * - bm_get(front, v): RANDOM MEMORY-LATENCY BOUND (random bit lookup across bitmap)
 * - bm_set_atomic & parent store: LOCAL WRITE & ATOMIC COHERENCY
 */
__attribute__((noinline))
int64_t bu_step(int64_t cur_sz, const uint64_t *front, uint64_t *next)
{
    (void)cur_sz;
    int64_t awake = 0;
    /* [MEMORY-BANDWIDTH-BOUND]: Zeroing next bitmap */
    bm_clear_all(next);

    #pragma omp parallel for reduction(+ : awake) schedule(dynamic, 1024)
    for (uint32_t u = 0; u < g_n; u++) {
        /* [MEMORY-BANDWIDTH-BOUND]: Sequential linear scan over parent array */
        if (g_parent[u] < 0) {
            uint64_t ib = in_idx[u];
            uint64_t ie = in_idx[u + 1];

            /* [MEMORY-BANDWIDTH-BOUND]: Sequential read of incoming CSR neighbors */
            for (uint64_t ei = ib; ei < ie; ei++) {
                uint32_t v = in_nbr[ei];

                /* [MEMORY-LATENCY-BOUND (with ALU bit-op)]: Random word read from bitmap front[v>>6] + shift/mask */
                if (bm_get(front, v)) {
                    /* [MEMORY STORE]: Write discovered parent */
                    g_parent[u] = (int32_t)v;
                    /* [COMPUTE/PIPELINE & MEMORY-COHERENCY]: Atomic bit update */
                    bm_set_atomic(next, u);
                    awake++;
                    break;
                }
            }
        }
    }
    return awake;
}

/* [MEMORY-BANDWIDTH-BOUND & COMPUTE BIT-OPS]: Linear scan to populate bitmap from queue */
__attribute__((noinline))
void queue_to_bitmap(int64_t sz, const uint32_t *q, uint64_t *bm)
{
    bm_clear_all(bm);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < sz; i++) {
        bm_set_atomic(bm, q[i]);
    }
}

/* [MEMORY-BANDWIDTH-BOUND & COMPUTE BIT-OPS]: Sequential scan of bitmap to rebuild queue */
__attribute__((noinline))
int64_t bitmap_to_queue(const uint64_t *bm, uint32_t *q)
{
    int64_t n = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t *l_q = (g_thread_qbufs && tid < g_num_threads) ? g_thread_qbufs[tid] : NULL;
        int64_t l_n = 0;

        #pragma omp for schedule(static) nowait
        for (uint32_t u = 0; u < g_n; u++) {
            if (bm_get(bm, u)) {
                if (l_q) {
                    l_q[l_n++] = u;
                } else {
                    int64_t idx = __sync_fetch_and_add(&n, 1);
                    q[idx] = u;
                }
            }
        }

        if (l_q && l_n > 0) {
            int64_t base = __sync_fetch_and_add(&n, l_n);
            memcpy(&q[base], l_q, (size_t)l_n * sizeof(uint32_t));
        }
    }

    return n;
}

/* Host CPU Candidate: Direction-switching heuristic comparison */
__attribute__((noinline))
int bfs_should_switch_bu(int64_t scout_count, int64_t edges_to_check, int alpha)
{
    return (scout_count > edges_to_check / alpha);
}

/* Direction-optimizing BFS */
__attribute__((noinline))
void bfs(uint32_t source, int max_levels, int alpha, int beta)
{
    /* [MEMORY-BANDWIDTH-BOUND]: Linear streaming initialization of parent array with node degrees */
    #pragma omp parallel for schedule(static)
    for (uint32_t n = 0; n < g_n; n++) {
        uint64_t deg = out_idx[n+1] - out_idx[n];
        g_parent[n]  = (deg > 0) ? -(int32_t)deg : -1;
    }
    g_parent[source] = (int32_t)source;

    uint32_t *cur = g_qbuf;
    uint32_t *nxt = g_qbuf + g_n;
    cur[0] = source;
    int64_t cur_sz = 1;

    int64_t edges_to_check = (int64_t)g_e;
    int64_t scout_count    = (int64_t)(out_idx[source+1] - out_idx[source]);

    int level   = 0;
    int bm_cur  = 0;
    int in_bu   = 0;

    /* [COMPUTE-BOUND (Control)]: Level progression and heuristic thresholds */
    while (cur_sz > 0) {
        if (max_levels > 0 && level >= max_levels) {
            printf("  [bfs] Level limit %d reached; breaking.\n", max_levels);
            break;
        }

        /* [COMPUTE-BOUND]: Direction selection comparison (scout threshold vs edges remaining) */
        if (bfs_should_switch_bu(scout_count, edges_to_check, alpha)) {
            if (!in_bu) {
                queue_to_bitmap(cur_sz, cur, g_bmap[bm_cur]);
                in_bu = 1;
            }

            int64_t awake = cur_sz, old_awake;
            int bm_nxt = 1 - bm_cur;
            do {
                if (max_levels > 0 && level >= max_levels) break;
                old_awake = awake;
                awake = bu_step(old_awake, g_bmap[bm_cur], g_bmap[bm_nxt]);
                int tmp = bm_cur; bm_cur = bm_nxt; bm_nxt = tmp;
                printf("  BU  lvl %-3d : awake = %lld\n",
                       level++, (long long)awake);
            } while (awake >= old_awake || awake > (int64_t)(g_n / beta));

            cur_sz = bitmap_to_queue(g_bmap[bm_cur], cur);
            scout_count = 1;
            in_bu = 0;
        } else {
            int64_t nxt_sz;
            scout_count    = td_step(cur_sz, cur, nxt, &nxt_sz);
            edges_to_check -= scout_count;

            uint32_t *tmp = cur; cur = nxt; nxt = tmp;
            cur_sz = nxt_sz;

            printf("  TD  lvl %-3d : frontier = %-8lld  scout = %lld\n",
                       level++, (long long)cur_sz, (long long)scout_count);
        }
    }
}

/* ============================================================
 *  Main
 * ============================================================ */
int main(int argc, char *argv[])
{
    const char *graph_file = NULL;
    int scale   = 16;
    int avg_deg = 16;
    unsigned seed = 42;
    int requested_threads = 0;
    int max_levels = 0;
    int alpha = ALPHA;
    int beta = BETA;
    int skip_warmup = 0;

    /* Parse CLI options:
     *   ./toy_bfs -f <graph.bin> [-t <threads>] [-k <levels>] [-W]
     *   ./toy_bfs <graph.bin> [-t <threads>]
     *   ./toy_bfs [scale] [avg_degree] [seed] [threads]
     */
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "-p") == 0 ||
             strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            requested_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            max_levels = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            alpha = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            beta = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-W") == 0 || strcmp(argv[i], "--no-warmup") == 0) {
            skip_warmup = 1;
        } else if (argv[i][0] != '-' && strstr(argv[i], ".bin") != NULL) {
            graph_file = argv[i];
        }
    }

    if (!graph_file && argc > 1 && argv[1][0] != '-') {
        /* Check if argv[1] is a number (scale) */
        char *endptr;
        long val = strtol(argv[1], &endptr, 10);
        if (*endptr == '\0') {
            scale = (int)val;
            if (argc > 2 && argv[2][0] != '-') avg_deg = atoi(argv[2]);
            if (argc > 3 && argv[3][0] != '-') seed    = (unsigned)atoi(argv[3]);
            if (argc > 4 && argv[4][0] != '-' && requested_threads <= 0) {
                requested_threads = atoi(argv[4]);
            }
        } else {
            graph_file = argv[1];
        }
    }

    /* Configure thread count */
    if (requested_threads <= 0) {
        const char *omp_env = getenv("OMP_NUM_THREADS");
        if (omp_env) {
            requested_threads = atoi(omp_env);
        }
    }
    if (requested_threads <= 0) {
#ifdef _OPENMP
        requested_threads = 1; /* Default to 1 thread unless explicitly specified */
#else
        requested_threads = 1;
#endif
    }
    if (requested_threads < 1) requested_threads = 1;

    omp_set_num_threads(requested_threads);

    printf("=== Toy BFS Benchmark ===\n");
    printf("  Threads    : %d\n", requested_threads);
    if (max_levels > 0) printf("  Max Levels : %d\n", max_levels);

    if (graph_file) {
        if (load_graph_mmap(graph_file) != 0) {
            fprintf(stderr, "Failed to load graph from '%s'\n", graph_file);
            return 1;
        }
    } else {
        if (scale < 1 || scale > 28) {
            fprintf(stderr, "Scale must be 1-28 (got %d)\n", scale);
            return 1;
        }
        uint32_t n = (uint32_t)1u << scale;
        printf("  Scale      : %d  (%u nodes)\n", scale, n);
        printf("  Avg degree : %d\n", avg_deg);
        printf("  Seed       : %u\n", seed);
        printf("  Mode       : In-Memory Generation (use -f <file.bin> for mmap)\n");
        build_graph_in_memory(n, avg_deg, seed);
    }

    /* Allocate thread-local queue buffers */
    alloc_thread_buffers(requested_threads, g_n);

    /* Find non-isolated source vertex */
    uint32_t source = 0;
    while (source < g_n && (out_idx[source+1] - out_idx[source]) == 0)
        source++;
    if (source == g_n) { fprintf(stderr, "All nodes isolated!\n"); return 1; }

    printf("\n[setup] Source node: %u (out-degree = %llu)\n\n",
           source, (unsigned long long)(out_idx[source+1] - out_idx[source]));

    /* Warm-up run outside ROI (if enabled) */
    if (!skip_warmup) {
        printf("[warmup] Running BFS (outside ROI, not traced)...\n");
        bfs(source, max_levels, alpha, beta);

        int64_t reachable = 0;
        #pragma omp parallel for reduction(+ : reachable) schedule(static)
        for (uint32_t i = 0; i < g_n; i++)
            if (g_parent[i] >= 0) reachable++;
        printf("[warmup] Reachable: %lld / %u nodes\n\n",
               (long long)reachable, g_n);
    } else {
        printf("[warmup] Skipped per -W/--no-warmup option.\n\n");
    }

    /* ---- ROI START: Pin tracer begins capturing instructions here ---- */
    roi_start();

    printf("[bfs] Running BFS from node %u (TRACED)...\n", source);
    bfs(source, max_levels, alpha, beta);

    /* ---- ROI END ---- */
    roi_end();

    int64_t reachable = 0;
    #pragma omp parallel for reduction(+ : reachable) schedule(static)
    for (uint32_t i = 0; i < g_n; i++)
        if (g_parent[i] >= 0) reachable++;
    printf("\n[result] Reachable: %lld / %u nodes\n",
           (long long)reachable, g_n);

    /* Cleanup */
    free_thread_buffers();
    free(g_parent);
    free(g_qbuf);
    free(g_bmap[0]);
    free(g_bmap[1]);

    if (g_mmap_ptr) {
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
    }
    if (g_alloc_out_idx) free(g_alloc_out_idx);
    if (g_alloc_out_nbr) free(g_alloc_out_nbr);
    if (g_alloc_in_idx)  free(g_alloc_in_idx);
    if (g_alloc_in_nbr)  free(g_alloc_in_nbr);

    return 0;
}
