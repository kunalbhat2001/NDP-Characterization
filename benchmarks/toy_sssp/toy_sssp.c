/*
 * toy_sssp.c – Single-Source Shortest Paths (GAPBS Delta-Stepping)
 *
 * Designed for StarNUMA + CXL Near-Data Processing (NDP) Offload evaluation.
 * Supports 64-bit CSR offsets (Version 2) for >100GB graph datasets.
 *
 * Function Partitioning:
 *   [NDP CPU (Memory-Heavy)]:
 *     - sssp_relax_edges: Relaxes outgoing edges from frontier nodes.
 *       Characterized by irregular random loads/stores into dist[v], atomic CAS,
 *       and edge weight lookups.
 *   [Host CPU (Compute-Heavy)]:
 *     - sssp_bucket_manage: Manages priority bucket queues, resolves bucket indices,
 *       and re-inserts relaxed vertices into appropriate buckets.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../toy_bfs/toy_graph.h"

#ifdef _OPENMP
#include <omp.h>
#else
static inline int  omp_get_max_threads(void) { return 1; }
static inline int  omp_get_thread_num(void)  { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

#define INF_DIST        0x3f3f3f3f
#define DEFAULT_DELTA   2
#define NUM_BUCKETS     64

/* ============================================================
 *  ROI magic instructions for Intel Pin tracer
 * ============================================================ */
static inline void roi_start(void)
{
    volatile uint64_t d = 0;
    __asm__ __volatile__("xchg %%rbx, %%rbx;"
        ::"b"((uint64_t)(&d)), "c"((uint64_t)0x5));
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
 *  Graph & SSSP Data Structures (64-bit CSR)
 * ============================================================ */
static uint32_t  g_n;
static uint64_t  g_e;
static const uint64_t *out_idx = NULL;
static const uint32_t *out_nbr = NULL;
static const uint32_t *out_weights = NULL;

static void   *g_mmap_ptr = NULL;
static size_t  g_mmap_size = 0;
static int     g_mmap_fd = -1;

static uint64_t *g_alloc_out_idx = NULL;
static uint32_t *g_alloc_out_nbr = NULL;
static uint32_t *g_alloc_weights = NULL;

/* SSSP State */
static uint32_t *g_dist = NULL;

/* Bucket structure */
typedef struct {
    uint32_t *nodes;
    int64_t   size;
    int64_t   capacity;
} Bucket;

static Bucket g_buckets[NUM_BUCKETS];

static void init_bucket(Bucket *b, int64_t cap) {
    b->nodes = (uint32_t *)malloc(cap * sizeof(uint32_t));
    b->size = 0;
    b->capacity = cap;
}
static void free_bucket(Bucket *b) {
    if (b->nodes) free(b->nodes);
    b->nodes = NULL;
    b->size = b->capacity = 0;
}

/* ============================================================
 *  Kernel 1: NDP CPU Candidate (Memory-Heavy Edge Relaxation)
 * ============================================================
 * Relaxes outgoing edges of nodes in the current bucket.
 * Signature exposes (frontier_size, current_bucket) in arg0, arg1
 * so func_cache_sim captures how frontier scale impacts memory.
 */
__attribute__((noinline))
int64_t sssp_relax_edges(uint64_t frontier_size,
                         uint32_t current_bucket,
                         const uint32_t *curr_nodes,
                         const uint64_t *idx,
                         const uint32_t *nbr,
                         const uint32_t *weights,
                         uint32_t *dist,
                         uint32_t *relaxed_nodes,
                         uint32_t *relaxed_dists)
{
    int64_t count = 0;

    #pragma omp parallel for reduction(+ : count) schedule(dynamic, 64)
    for (uint64_t i = 0; i < frontier_size; i++) {
        uint32_t u = curr_nodes[i];
        uint32_t du = dist[u];

        uint64_t start = idx[u];
        uint64_t end   = idx[u + 1];

        for (uint64_t e = start; e < end; e++) {
            uint32_t v = nbr[e];
            uint32_t w = weights[e];
            uint32_t cand_dist = du + w;

            uint32_t old_dist = dist[v];
            while (cand_dist < old_dist) {
                if (__sync_bool_compare_and_swap(&dist[v], old_dist, cand_dist)) {
                    int64_t pos = __sync_fetch_and_add(&count, 1);
                    if (pos < (int64_t)g_n) {
                        relaxed_nodes[pos] = v;
                        relaxed_dists[pos] = cand_dist;
                    }
                    break;
                }
                old_dist = dist[v];
            }
        }
    }

    return count;
}

/* ============================================================
 *  Kernel 2: Host CPU Candidate (Compute-Heavy Bucket Management)
 * ============================================================
 */
__attribute__((noinline))
void sssp_bucket_manage(int64_t relaxed_count,
                        const uint32_t *relaxed_nodes,
                        const uint32_t *relaxed_dists,
                        uint32_t delta,
                        uint32_t *cur_bucket_idx)
{
    for (int64_t i = 0; i < relaxed_count; i++) {
        uint32_t node = relaxed_nodes[i];
        uint32_t d    = relaxed_dists[i];
        uint32_t dest_b = (d / delta) % NUM_BUCKETS;

        Bucket *b = &g_buckets[dest_b];
        if (b->size >= b->capacity) {
            b->capacity = b->capacity ? (b->capacity * 2) : 1024;
            b->nodes = (uint32_t *)realloc(b->nodes, b->capacity * sizeof(uint32_t));
        }
        b->nodes[b->size++] = node;
    }

    while (*cur_bucket_idx < NUM_BUCKETS && g_buckets[*cur_bucket_idx].size == 0) {
        (*cur_bucket_idx)++;
    }
}

/* ============================================================
 *  Overall SSSP Orchestration (ROI)
 * ============================================================ */
__attribute__((noinline))
void run_sssp(uint32_t source, uint32_t delta, uint32_t max_buckets)
{
    for (uint32_t i = 0; i < g_n; i++) {
        g_dist[i] = INF_DIST;
    }
    g_dist[source] = 0;

    for (int i = 0; i < NUM_BUCKETS; i++) {
        init_bucket(&g_buckets[i], 1024);
    }

    g_buckets[0].nodes[g_buckets[0].size++] = source;

    uint32_t *relaxed_nodes = (uint32_t *)malloc(g_n * sizeof(uint32_t));
    uint32_t *relaxed_dists = (uint32_t *)malloc(g_n * sizeof(uint32_t));

    printf("[SSSP] Starting Delta-Stepping (Source: %u, Delta: %u, %u nodes, max_buckets: %u)...\n",
           source, delta, g_n, max_buckets ? max_buckets : NUM_BUCKETS);

    roi_start();

    uint32_t cur_bucket = 0;
    int64_t total_relaxations = 0;

    while (cur_bucket < NUM_BUCKETS) {
        if (max_buckets > 0 && cur_bucket >= max_buckets) break;
        Bucket *b = &g_buckets[cur_bucket];
        if (b->size == 0) {
            cur_bucket++;
            continue;
        }

        uint64_t frontier_sz = (uint64_t)b->size;
        uint32_t *curr_frontier = b->nodes;
        b->nodes = (uint32_t *)malloc(1024 * sizeof(uint32_t));
        b->size = 0;
        b->capacity = 1024;

        // Step 1: Memory-heavy edge relaxation [NDP CPU]
        int64_t relaxed = sssp_relax_edges(frontier_sz, cur_bucket, curr_frontier,
                                           out_idx, out_nbr, out_weights, g_dist,
                                           relaxed_nodes, relaxed_dists);
        total_relaxations += relaxed;
        free(curr_frontier);

        // Step 2: Compute-heavy bucket placement [Host CPU]
        sssp_bucket_manage(relaxed, relaxed_nodes, relaxed_dists, delta, &cur_bucket);
    }

    roi_end();

    printf("[SSSP] Complete. Total edge relaxations performed: %lld\n", (long long)total_relaxations);

    free(relaxed_nodes);
    free(relaxed_dists);
    for (int i = 0; i < NUM_BUCKETS; i++) {
        free_bucket(&g_buckets[i]);
    }
}

/* ============================================================
 *  Graph Loading (64-bit and 32-bit Compatibility)
 * ============================================================ */
static int load_mmap_graph(const char *filename)
{
    struct stat st;
    g_mmap_fd = open(filename, O_RDONLY);
    if (g_mmap_fd < 0 || fstat(g_mmap_fd, &st) < 0) return -1;
    if ((size_t)st.st_size < sizeof(ToyGraphHeader)) { close(g_mmap_fd); return -1; }

    g_mmap_size = st.st_size;
    g_mmap_ptr = mmap(NULL, g_mmap_size, PROT_READ, MAP_SHARED, g_mmap_fd, 0);
    if (g_mmap_ptr == MAP_FAILED) { close(g_mmap_fd); return -1; }

    const ToyGraphHeader *hdr = (const ToyGraphHeader *)g_mmap_ptr;
    if (memcmp(hdr->magic, TOYGRAPH_MAGIC, 8) != 0) {
        munmap(g_mmap_ptr, g_mmap_size); close(g_mmap_fd); return -1;
    }

    g_n = hdr->num_nodes;
    g_e = hdr->num_edges;

    if (hdr->version == TOYGRAPH_VERSION_64) {
        out_idx = (const uint64_t *)((const char *)g_mmap_ptr + hdr->out_idx_offset);
    } else {
        const uint32_t *raw_out_idx = (const uint32_t *)((const char *)g_mmap_ptr + hdr->out_idx_offset);
        g_alloc_out_idx = (uint64_t *)malloc((g_n + 1) * sizeof(uint64_t));
        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i <= g_n; i++) {
            g_alloc_out_idx[i] = (uint64_t)raw_out_idx[i];
        }
        out_idx = g_alloc_out_idx;
    }

    out_nbr = (const uint32_t *)((const char *)g_mmap_ptr + hdr->out_nbr_offset);

    // Synthetic weights
    g_alloc_weights = (uint32_t *)malloc(g_e * sizeof(uint32_t));
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < g_e; i++) {
        g_alloc_weights[i] = (uint32_t)((i % 4) + 1);
    }
    out_weights = g_alloc_weights;

    printf("[mmap] Loaded graph '%s': %u nodes, %llu edges (Version %u, %.2f GB).\n",
           filename, g_n, (unsigned long long)g_e, hdr->version,
           (double)st.st_size / (1024.0 * 1024.0 * 1024.0));
    return 0;
}

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *state = x;
}

static void build_synthetic_graph(uint32_t n, int avg_deg, unsigned seed)
{
    g_n = n;
    g_e = (uint64_t)n * (uint64_t)avg_deg;

    printf("[synthetic] Generating %llu edges (%u nodes, avg_deg %d)...\n",
           (unsigned long long)g_e, n, avg_deg);

    g_alloc_out_idx = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
    g_alloc_out_nbr = (uint32_t *)malloc(g_e * sizeof(uint32_t));
    g_alloc_weights = (uint32_t *)malloc(g_e * sizeof(uint32_t));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t thr_seed = (uint64_t)seed * 1103515245ULL + (uint64_t)tid * 6364136223846793005ULL;

        #pragma omp for schedule(static)
        for (uint32_t u = 0; u < n; u++) {
            g_alloc_out_idx[u] = (uint64_t)u * (uint64_t)avg_deg;
            uint64_t base = g_alloc_out_idx[u];
            for (int d = 0; d < avg_deg; d++) {
                uint32_t v = (uint32_t)(xorshift64(&thr_seed) % n);
                if (v == u) v = (v + 1) % n;
                g_alloc_out_nbr[base + d] = v;
                g_alloc_weights[base + d] = (uint32_t)((xorshift64(&thr_seed) % 4) + 1);
            }
        }
    }
    g_alloc_out_idx[n] = g_e;

    out_idx = g_alloc_out_idx;
    out_nbr = g_alloc_out_nbr;
    out_weights = g_alloc_weights;
}

int main(int argc, char *argv[])
{
    const char *graph_file = NULL;
    uint32_t scale = 14;
    int avg_deg = 16;
    unsigned seed = 42;
    int num_threads = 1;
    uint32_t delta = DEFAULT_DELTA;
    uint32_t source = 0;
    uint32_t max_buckets = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i + 1 < argc) graph_file = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) delta = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) source = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) max_buckets = (uint32_t)atoi(argv[++i]);
        else if (i == 1 && argv[i][0] != '-') scale = (uint32_t)atoi(argv[i]);
        else if (i == 2 && argv[i][0] != '-') avg_deg = atoi(argv[i]);
        else if (i == 3 && argv[i][0] != '-') seed = (unsigned)atoi(argv[i]);
        else if (i == 4 && argv[i][0] != '-') num_threads = atoi(argv[i]);
        else if (i == 5 && argv[i][0] != '-') delta = (uint32_t)atoi(argv[i]);
    }

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    if (graph_file) {
        if (load_mmap_graph(graph_file) != 0) return 1;
    } else {
        uint32_t n = 1u << scale;
        build_synthetic_graph(n, avg_deg, seed);
    }

    g_dist = (uint32_t *)malloc(g_n * sizeof(uint32_t));

    run_sssp(source, delta, max_buckets);

    uint32_t reachable = 0;
    uint32_t max_d = 0;
    for (uint32_t i = 0; i < g_n; i++) {
        if (g_dist[i] != INF_DIST) {
            reachable++;
            if (g_dist[i] > max_d) max_d = g_dist[i];
        }
    }
    printf("[Result] Reachable nodes: %u / %u, Max distance: %u\n", reachable, g_n, max_d);

    free(g_dist);
    if (g_alloc_weights) free(g_alloc_weights);
    if (g_alloc_out_idx) free(g_alloc_out_idx);
    if (g_alloc_out_nbr) free(g_alloc_out_nbr);
    if (g_mmap_ptr) munmap(g_mmap_ptr, g_mmap_size);
    if (g_mmap_fd >= 0) close(g_mmap_fd);

    return 0;
}
