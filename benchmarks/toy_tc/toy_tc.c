/*
 * toy_tc.c – Triangle Counting Microbenchmark (GAPBS OrderedCount)
 *
 * Designed for StarNUMA + CXL Near-Data Processing (NDP) Offload evaluation.
 * Supports 64-bit CSR offsets (Version 2) for >100GB graph datasets.
 *
 * Function Partitioning:
 *   [NDP CPU (Memory-Heavy - Streaming Intersection)]:
 *     - tc_intersect_neighbors / tc_intersect_chunk: Two-finger merge intersection
 *       of sorted adjacency lists between neighboring vertices. Bound by DRAM
 *       streaming memory bandwidth and cache capacity.
 *   [Host CPU (Compute-Heavy)]:
 *     - tc_order_by_degree: Computes vertex degrees, re-labels/filters edges into
 *       a DAG, and sorts adjacency lists.
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
 *  Graph & TC Data Structures (64-bit CSR)
 * ============================================================ */
static uint32_t  g_n;
static uint64_t  g_e;
static const uint64_t *raw_out_idx = NULL;
static const uint32_t *raw_out_nbr = NULL;

static void   *g_mmap_ptr = NULL;
static size_t  g_mmap_size = 0;
static int     g_mmap_fd = -1;

static uint64_t *g_alloc_out_idx = NULL;
static uint32_t *g_alloc_out_nbr = NULL;

static uint64_t *dag_idx = NULL;
static uint32_t *dag_nbr = NULL;

static int cmp_u32(const void *a, const void *b)
{
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    return (ua < ub) ? -1 : (ua > ub ? 1 : 0);
}

/* ============================================================
 *  Kernel 1: Host CPU Candidate (Compute-Heavy - Graph DAG Prep)
 * ============================================================
 */
__attribute__((noinline))
void tc_order_by_degree(uint32_t num_nodes,
                        const uint64_t *in_idx,
                        const uint32_t *in_nbr,
                        uint64_t **out_dag_idx,
                        uint32_t **out_dag_nbr)
{
    printf("[TC] Building degree-ordered DAG for %u nodes (Host CPU)...\n", num_nodes);

    uint64_t *new_idx = (uint64_t *)calloc(num_nodes + 1, sizeof(uint64_t));

    #pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u < num_nodes; u++) {
        uint32_t deg_u = (uint32_t)(in_idx[u + 1] - in_idx[u]);
        uint64_t count = 0;
        uint64_t start = in_idx[u];
        uint64_t end   = in_idx[u + 1];

        for (uint64_t e = start; e < end; e++) {
            uint32_t v = in_nbr[e];
            if (v >= num_nodes) continue;
            uint32_t deg_v = (uint32_t)(in_idx[v + 1] - in_idx[v]);
            if (deg_u < deg_v || (deg_u == deg_v && u < v)) {
                count++;
            }
        }
        new_idx[u + 1] = count;
    }

    for (uint32_t u = 1; u <= num_nodes; u++) {
        new_idx[u] += new_idx[u - 1];
    }

    uint64_t total_dag_edges = new_idx[num_nodes];
    uint32_t *new_nbr = (uint32_t *)malloc((total_dag_edges + 1) * sizeof(uint32_t));

    #pragma omp parallel for schedule(dynamic, 512)
    for (uint32_t u = 0; u < num_nodes; u++) {
        uint32_t deg_u = (uint32_t)(in_idx[u + 1] - in_idx[u]);
        uint64_t start = in_idx[u];
        uint64_t end   = in_idx[u + 1];
        uint64_t pos   = new_idx[u];

        for (uint64_t e = start; e < end; e++) {
            uint32_t v = in_nbr[e];
            if (v >= num_nodes) continue;
            uint32_t deg_v = (uint32_t)(in_idx[v + 1] - in_idx[v]);
            if (deg_u < deg_v || (deg_u == deg_v && u < v)) {
                new_nbr[pos++] = v;
            }
        }

        uint64_t list_len = pos - new_idx[u];
        if (list_len > 1) {
            qsort(&new_nbr[new_idx[u]], list_len, sizeof(uint32_t), cmp_u32);
        }
    }

    *out_dag_idx = new_idx;
    *out_dag_nbr = new_nbr;
    printf("[TC] DAG constructed with %llu directed edges.\n", (unsigned long long)total_dag_edges);
}

/* ============================================================
 *  Kernel 2: NDP CPU Candidate (Memory-Heavy - Streaming Merge)
 * ============================================================
 * Exposes (start_u, end_u, chunk_edges) to func_cache_sim
 * to capture how argument chunk size determines DRAM access density.
 */
__attribute__((noinline))
int64_t tc_intersect_chunk(uint32_t start_u,
                           uint32_t end_u,
                           uint64_t chunk_edges,
                           const uint64_t *idx,
                           const uint32_t *nbr)
{
    int64_t triangles = 0;

    #pragma omp parallel for reduction(+ : triangles) schedule(dynamic, 64)
    for (uint32_t u = start_u; u < end_u; u++) {
        uint64_t u_start = idx[u];
        uint64_t u_end   = idx[u + 1];
        if (u_start == u_end) continue;

        for (uint64_t e = u_start; e < u_end; e++) {
            uint32_t v = nbr[e];
            uint64_t v_start = idx[v];
            uint64_t v_end   = idx[v + 1];

            uint64_t it_u = u_start;
            uint64_t it_v = v_start;

            while (it_u < u_end && it_v < v_end) {
                uint32_t nu = nbr[it_u];
                uint32_t nv = nbr[it_v];

                if (nu == nv) {
                    triangles++;
                    it_u++;
                    it_v++;
                } else if (nu < nv) {
                    it_u++;
                } else {
                    it_v++;
                }
            }
        }
    }

    return triangles;
}

__attribute__((noinline))
int64_t tc_intersect_neighbors(uint32_t num_nodes,
                               const uint64_t *idx,
                               const uint32_t *nbr)
{
    int64_t total = 0;
    uint32_t u = 0;

    // Chunk 1: Small (500 nodes) - low footprint, fits in cache
    if (u + 500 <= num_nodes) {
        uint64_t edges = idx[u + 500] - idx[u];
        total += tc_intersect_chunk(u, u + 500, edges, idx, nbr);
        u += 500;
    }
    // Chunk 2: Small (1,000 nodes)
    if (u + 1000 <= num_nodes) {
        uint64_t edges = idx[u + 1000] - idx[u];
        total += tc_intersect_chunk(u, u + 1000, edges, idx, nbr);
        u += 1000;
    }
    // Chunk 3: Medium (5,000 nodes)
    if (u + 5000 <= num_nodes) {
        uint64_t edges = idx[u + 5000] - idx[u];
        total += tc_intersect_chunk(u, u + 5000, edges, idx, nbr);
        u += 5000;
    }
    // Large chunks (20,000 nodes) - memory-bound, spills LLC
    while (u < num_nodes) {
        uint32_t chunk = (num_nodes - u > 20000) ? 20000 : (num_nodes - u);
        uint64_t edges = idx[u + chunk] - idx[u];
        total += tc_intersect_chunk(u, u + chunk, edges, idx, nbr);
        u += chunk;
    }
    return total;
}

/* ============================================================
 *  Triangle Counting Orchestration (ROI)
 * ============================================================ */
__attribute__((noinline))
int64_t run_tc(uint32_t num_nodes, const uint64_t *idx, const uint32_t *nbr)
{
    tc_order_by_degree(num_nodes, idx, nbr, &dag_idx, &dag_nbr);

    printf("[TC] Starting Triangle Counting (ROI)...\n");

    roi_start();

    int64_t count = tc_intersect_neighbors(num_nodes, dag_idx, dag_nbr);

    roi_end();

    return count;
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
        raw_out_idx = (const uint64_t *)((const char *)g_mmap_ptr + hdr->out_idx_offset);
    } else {
        const uint32_t *raw_out_idx_32 = (const uint32_t *)((const char *)g_mmap_ptr + hdr->out_idx_offset);
        g_alloc_out_idx = (uint64_t *)malloc((g_n + 1) * sizeof(uint64_t));
        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i <= g_n; i++) {
            g_alloc_out_idx[i] = (uint64_t)raw_out_idx_32[i];
        }
        raw_out_idx = g_alloc_out_idx;
    }

    raw_out_nbr = (const uint32_t *)((const char *)g_mmap_ptr + hdr->out_nbr_offset);

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
            }
        }
    }
    g_alloc_out_idx[n] = g_e;

    raw_out_idx = g_alloc_out_idx;
    raw_out_nbr = g_alloc_out_nbr;
}

int main(int argc, char *argv[])
{
    const char *graph_file = NULL;
    uint32_t scale = 14;
    int avg_deg = 16;
    unsigned seed = 42;
    int num_threads = 1;
    uint32_t eval_nodes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i + 1 < argc) graph_file = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) eval_nodes = (uint32_t)atoi(argv[++i]);
        else if (i == 1 && argv[i][0] != '-') scale = (uint32_t)atoi(argv[i]);
        else if (i == 2 && argv[i][0] != '-') avg_deg = atoi(argv[i]);
        else if (i == 3 && argv[i][0] != '-') seed = (unsigned)atoi(argv[i]);
        else if (i == 4 && argv[i][0] != '-') num_threads = atoi(argv[i]);
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

    uint32_t target_nodes = (eval_nodes > 0 && eval_nodes < g_n) ? eval_nodes : g_n;
    int64_t triangles = run_tc(target_nodes, raw_out_idx, raw_out_nbr);

    printf("[Result] Total Triangles: %lld\n", (long long)triangles);

    if (dag_idx) free(dag_idx);
    if (dag_nbr) free(dag_nbr);
    if (g_alloc_out_idx) free(g_alloc_out_idx);
    if (g_alloc_out_nbr) free(g_alloc_out_nbr);
    if (g_mmap_ptr) munmap(g_mmap_ptr, g_mmap_size);
    if (g_mmap_fd >= 0) close(g_mmap_fd);

    return 0;
}
