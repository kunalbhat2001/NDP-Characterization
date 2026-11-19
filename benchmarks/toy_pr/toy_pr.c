/*
 * toy_pr.c – PageRank Microbenchmark (GAPBS-derived)
 *
 * Designed for StarNUMA + CXL Near-Data Processing (NDP) Offload evaluation.
 * Supports 64-bit CSR offsets (Version 2) for >100GB graph datasets.
 *
 * Function Partitioning:
 *   [NDP CPU (Memory-Heavy)]:
 *     - pr_pull_step / pr_pull_range: Gathers neighbor rank contributions across
 *       incoming CSR edges. Characterized by irregular indirect reads (scores[v])
 *       and memory latency bound execution.
 *   [Host CPU (Compute-Heavy)]:
 *     - pr_dampen_and_norm: Applies damping factor, calculates convergence error
 *       residual, and normalizes contributions. High arithmetic density, contiguous.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
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

#define DAMPING_FACTOR  0.85
#define EPSILON         1e-4
#define DEFAULT_ITERS   5

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
 *  Graph & PageRank Data Structures (64-bit CSR)
 * ============================================================ */
static uint32_t  g_n;
static uint64_t  g_e;
static const uint64_t *in_idx = NULL;
static const uint32_t *in_nbr = NULL;

static void   *g_mmap_ptr = NULL;
static size_t  g_mmap_size = 0;
static int     g_mmap_fd = -1;

static uint64_t *g_alloc_in_idx = NULL;
static uint32_t *g_alloc_in_nbr = NULL;
static uint32_t *g_out_degree   = NULL;

/* PageRank State Arrays */
static double *g_scores = NULL;
static double *g_outgoing_contrib = NULL;
static double *g_incoming_sums = NULL;

/* ============================================================
 *  Kernel 1: NDP CPU Candidate (Memory-Heavy Pull Step)
 * ============================================================
 * Gathers neighbor rank contributions across incoming CSR edges.
 * Parameterized with (start_node, end_node, chunk_edges) so that
 * func_cache_sim captures how argument bounds affect DRAM accesses.
 */
__attribute__((noinline))
void pr_pull_range(uint32_t start_u,
                   uint32_t end_u,
                   uint64_t chunk_edges,
                   const uint64_t *idx,
                   const uint32_t *nbr,
                   const double *contrib,
                   double *sums)
{
    #pragma omp parallel for schedule(dynamic, 1024)
    for (uint32_t u = start_u; u < end_u; u++) {
        uint64_t start = idx[u];
        uint64_t end   = idx[u + 1];
        double incoming_total = 0.0;

        for (uint64_t e = start; e < end; e++) {
            uint32_t v = nbr[e];
            incoming_total += contrib[v];
        }
        sums[u] = incoming_total;
    }
}

__attribute__((noinline))
void pr_pull_step(uint32_t num_nodes,
                  const uint64_t *idx,
                  const uint32_t *nbr,
                  const double *contrib,
                  double *sums)
{
    uint32_t u = 0;

    // Chunk 1: Small (500 nodes) - low footprint, fits in cache
    if (u + 500 <= num_nodes) {
        uint64_t edges = idx[u + 500] - idx[u];
        pr_pull_range(u, u + 500, edges, idx, nbr, contrib, sums);
        u += 500;
    }
    // Chunk 2: Small (1,000 nodes)
    if (u + 1000 <= num_nodes) {
        uint64_t edges = idx[u + 1000] - idx[u];
        pr_pull_range(u, u + 1000, edges, idx, nbr, contrib, sums);
        u += 1000;
    }
    // Chunk 3: Medium (10,000 nodes) - transition boundary
    if (u + 10000 <= num_nodes) {
        uint64_t edges = idx[u + 10000] - idx[u];
        pr_pull_range(u, u + 10000, edges, idx, nbr, contrib, sums);
        u += 10000;
    }
    // Chunk 4: Medium-Large (50,000 nodes)
    if (u + 50000 <= num_nodes) {
        uint64_t edges = idx[u + 50000] - idx[u];
        pr_pull_range(u, u + 50000, edges, idx, nbr, contrib, sums);
        u += 50000;
    }
    // Large chunks (200,000 nodes) - memory-bound, spills LLC
    while (u < num_nodes) {
        uint32_t chunk = (num_nodes - u > 200000) ? 200000 : (num_nodes - u);
        uint64_t edges = idx[u + chunk] - idx[u];
        pr_pull_range(u, u + chunk, edges, idx, nbr, contrib, sums);
        u += chunk;
    }
}

/* ============================================================
 *  Kernel 2: Host CPU Candidate (Compute-Heavy Math)
 * ============================================================
 */
__attribute__((noinline))
double pr_dampen_and_norm(uint32_t num_nodes,
                          const double *sums,
                          double *scores,
                          double *contrib,
                          const uint32_t *out_deg,
                          double base_score,
                          double damping)
{
    double total_error = 0.0;

    #pragma omp parallel for reduction(+ : total_error) schedule(static)
    for (uint32_t u = 0; u < num_nodes; u++) {
        double old_score = scores[u];
        double new_score = base_score + damping * sums[u];
        scores[u] = new_score;

        total_error += fabs(new_score - old_score);

        uint32_t deg = out_deg[u];
        contrib[u] = (deg > 0) ? (new_score / (double)deg) : 0.0;
    }

    return total_error;
}

/* ============================================================
 *  Overall PageRank Orchestration (ROI)
 * ============================================================ */
__attribute__((noinline))
int run_pagerank(int max_iters, uint32_t eval_nodes)
{
    if (eval_nodes == 0 || eval_nodes > g_n) eval_nodes = g_n;
    double init_score = 1.0 / (double)g_n;
    double base_score = (1.0 - DAMPING_FACTOR) / (double)g_n;

    for (uint32_t i = 0; i < g_n; i++) {
        g_scores[i] = init_score;
        uint32_t deg = g_out_degree[i];
        g_outgoing_contrib[i] = (deg > 0) ? (init_score / (double)deg) : 0.0;
    }

    printf("[PageRank] Starting computation (%u of %u nodes, %llu graph edges, max %d iters)...\n",
           eval_nodes, g_n, (unsigned long long)g_e, max_iters);

    roi_start();

    int iter;
    for (iter = 0; iter < max_iters; iter++) {
        // Step 1: Memory-heavy pull step [NDP CPU Candidate]
        pr_pull_step(eval_nodes, in_idx, in_nbr, g_outgoing_contrib, g_incoming_sums);

        // Step 2: Compute-heavy damping and error reduction [Host CPU Candidate]
        double err = pr_dampen_and_norm(eval_nodes, g_incoming_sums, g_scores,
                                        g_outgoing_contrib, g_out_degree,
                                        base_score, DAMPING_FACTOR);

        printf("  Iter %2d: L1-norm residual error = %e\n", iter + 1, err);
        if (err < EPSILON) {
            iter++;
            break;
        }
    }

    roi_end();
    return iter;
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
        in_idx = (const uint64_t *)((const char *)g_mmap_ptr + hdr->in_idx_offset);
    } else {
        // Upgrade 32-bit offsets to 64-bit
        const uint32_t *raw_in_idx = (const uint32_t *)((const char *)g_mmap_ptr + hdr->in_idx_offset);
        g_alloc_in_idx = (uint64_t *)malloc((g_n + 1) * sizeof(uint64_t));
        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i <= g_n; i++) {
            g_alloc_in_idx[i] = (uint64_t)raw_in_idx[i];
        }
        in_idx = g_alloc_in_idx;
    }

    in_nbr = (const uint32_t *)((const char *)g_mmap_ptr + hdr->in_nbr_offset);

    // Populate out_degree from average or in_idx
    g_out_degree = (uint32_t *)malloc(g_n * sizeof(uint32_t));
    for (uint32_t i = 0; i < g_n; i++) {
        g_out_degree[i] = (uint32_t)(in_idx[i + 1] - in_idx[i]);
        if (g_out_degree[i] == 0) g_out_degree[i] = 1;
    }

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

    printf("[synthetic] Generating %llu directed edges (%u nodes, avg_deg %d)...\n",
           (unsigned long long)g_e, n, avg_deg);

    g_alloc_in_idx = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
    g_alloc_in_nbr = (uint32_t *)malloc(g_e * sizeof(uint32_t));
    g_out_degree   = (uint32_t *)malloc(n * sizeof(uint32_t));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t thr_seed = (uint64_t)seed * 1103515245ULL + (uint64_t)tid * 6364136223846793005ULL;

        #pragma omp for schedule(static)
        for (uint32_t u = 0; u < n; u++) {
            g_alloc_in_idx[u] = (uint64_t)u * (uint64_t)avg_deg;
            // Introduce degree variance for realistic power-law-like distribution
            uint32_t deg = avg_deg / 2 + (uint32_t)(xorshift64(&thr_seed) % (avg_deg + 1));
            g_out_degree[u] = deg > 0 ? deg : 1;
            uint64_t base = g_alloc_in_idx[u];
            for (int d = 0; d < avg_deg; d++) {
                uint32_t v = (uint32_t)(xorshift64(&thr_seed) % n);
                if (v == u) v = (v + 1) % n;
                g_alloc_in_nbr[base + d] = v;
            }
        }
    }
    g_alloc_in_idx[n] = g_e;

    in_idx = g_alloc_in_idx;
    in_nbr = g_alloc_in_nbr;
}

int main(int argc, char *argv[])
{
    const char *graph_file = NULL;
    uint32_t scale = 14;
    int avg_deg = 16;
    unsigned seed = 42;
    int num_threads = 1;
    int max_iters = DEFAULT_ITERS;
    uint32_t max_eval_nodes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i + 1 < argc) graph_file = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) max_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_eval_nodes = (uint32_t)atoi(argv[++i]);
        else if (i == 1 && argv[i][0] != '-') scale = (uint32_t)atoi(argv[i]);
        else if (i == 2 && argv[i][0] != '-') avg_deg = atoi(argv[i]);
        else if (i == 3 && argv[i][0] != '-') seed = (unsigned)atoi(argv[i]);
        else if (i == 4 && argv[i][0] != '-') num_threads = atoi(argv[i]);
        else if (i == 5 && argv[i][0] != '-') max_iters = atoi(argv[i]);
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

    g_scores           = (double *)malloc(g_n * sizeof(double));
    g_outgoing_contrib = (double *)malloc(g_n * sizeof(double));
    g_incoming_sums    = (double *)malloc(g_n * sizeof(double));

    run_pagerank(max_iters, max_eval_nodes);

    uint32_t best_v = 0;
    double best_score = -1.0;
    for (uint32_t i = 0; i < g_n; i++) {
        if (g_scores[i] > best_score) {
            best_score = g_scores[i];
            best_v = i;
        }
    }
    printf("[Result] Highest rank vertex: %u with score %e\n", best_v, best_score);

    free(g_scores);
    free(g_outgoing_contrib);
    free(g_incoming_sums);
    free(g_out_degree);
    if (g_alloc_in_idx) free(g_alloc_in_idx);
    if (g_alloc_in_nbr) free(g_alloc_in_nbr);
    if (g_mmap_ptr) munmap(g_mmap_ptr, g_mmap_size);
    if (g_mmap_fd >= 0) close(g_mmap_fd);

    return 0;
}
