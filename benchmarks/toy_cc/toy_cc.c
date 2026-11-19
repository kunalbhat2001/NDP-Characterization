/*
 * toy_cc.c – Connected Components (GAPBS-derived Afforest)
 *
 * Designed for StarNUMA + CXL Near-Data Processing (NDP) Offload evaluation.
 * Supports 64-bit CSR offsets (Version 2) for >100GB graph datasets.
 *
 * Function Partitioning:
 *   [NDP CPU (Memory-Heavy)]:
 *     - cc_link_step: Hooking neighbor components across CSR edges via atomic CAS.
 *     - cc_compress_step: Multi-hop pointer jumping (comp[u] = comp[comp[u]]).
 *   [Host CPU (Compute-Heavy)]:
 *     - cc_sample_frequent: PRNG sampling and frequency histogram of component IDs.
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

#define DEFAULT_NEIGHBOR_ROUNDS 2

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
 *  Graph & Connected Components Data (64-bit CSR)
 * ============================================================ */
static uint32_t  g_n;
static uint64_t  g_e;
static const uint64_t *out_idx = NULL;
static const uint32_t *out_nbr = NULL;

static void   *g_mmap_ptr = NULL;
static size_t  g_mmap_size = 0;
static int     g_mmap_fd = -1;

static uint64_t *g_alloc_out_idx = NULL;
static uint32_t *g_alloc_out_nbr = NULL;

static uint32_t *g_comp = NULL;

/* ============================================================
 *  Kernel 1: NDP CPU Candidate (Memory-Heavy Hooking)
 * ============================================================
 * Exposes (r, skip_component) in arguments to observe how sampling
 * and giant component filtering affect memory impact.
 */
__attribute__((noinline))
void cc_link_step(uint32_t num_nodes,
                  uint32_t r,
                  uint32_t skip_component,
                  const uint64_t *idx,
                  const uint32_t *nbr,
                  uint32_t *comp)
{
    #pragma omp parallel for schedule(dynamic, 1024)
    for (uint32_t u = 0; u < num_nodes; u++) {
        uint32_t cu = comp[u];
        if (cu == skip_component) continue;

        uint64_t start = idx[u];
        uint64_t end   = idx[u + 1];
        if (start + r < end) end = start + r;

        for (uint64_t e = start; e < end; e++) {
            uint32_t v = nbr[e];
            uint32_t cv = comp[v];
            if (cv == skip_component) continue;

            while (cu != cv) {
                uint32_t high = (cu > cv) ? cu : cv;
                uint32_t low  = (cu < cv) ? cu : cv;

                uint32_t p_high = comp[high];
                if (p_high == low || (p_high == high && __sync_bool_compare_and_swap(&comp[high], high, low))) {
                    break;
                }
                cu = comp[comp[high]];
                cv = comp[low];
            }
        }
    }
}

/* ============================================================
 *  Kernel 2: NDP CPU Candidate (Memory-Heavy Pointer Jumping)
 * ============================================================
 * Exposes (round_id, num_nodes) in arguments to capture how
 * successive path compression passes shift from DRAM stalls to L1/L2 hits.
 */
__attribute__((noinline))
int64_t cc_compress_step(uint32_t round_id, uint32_t num_nodes, uint32_t *comp)
{
    int64_t changes = 0;

    #pragma omp parallel for reduction(+ : changes) schedule(static)
    for (uint32_t n = 0; n < num_nodes; n++) {
        while (comp[n] != comp[comp[n]]) {
            comp[n] = comp[comp[n]];
            changes++;
        }
    }
    return changes;
}

/* ============================================================
 *  Kernel 3: Host CPU Candidate (Compute-Heavy Sampling)
 * ============================================================
 */
__attribute__((noinline))
uint32_t cc_sample_frequent(const uint32_t *comp, uint32_t num_nodes, int num_samples)
{
    if (num_nodes == 0) return 0;
    #define HASH_CAP 1024
    uint32_t keys[HASH_CAP] = {0};
    int counts[HASH_CAP] = {0};

    unsigned seed = 12345;
    for (int i = 0; i < num_samples; i++) {
        seed = seed * 1103515245 + 12345;
        uint32_t n = (seed >> 16) % num_nodes;
        uint32_t c = comp[n];

        uint32_t h = c % HASH_CAP;
        int probes = 0;
        while (counts[h] > 0 && keys[h] != c && ++probes < HASH_CAP) {
            h = (h + 1) % HASH_CAP;
        }
        if (probes < HASH_CAP) {
            keys[h] = c;
            counts[h]++;
        }
    }

    uint32_t best_c = 0;
    int max_cnt = -1;
    for (int i = 0; i < HASH_CAP; i++) {
        if (counts[i] > max_cnt) {
            max_cnt = counts[i];
            best_c = keys[i];
        }
    }
    return best_c;
}

/* ============================================================
 *  Overall Connected Components Orchestration (ROI)
 * ============================================================ */
__attribute__((noinline))
void run_cc(int neighbor_rounds, uint32_t eval_nodes)
{
    if (eval_nodes == 0 || eval_nodes > g_n) eval_nodes = g_n;

    for (uint32_t i = 0; i < g_n; i++) {
        g_comp[i] = i;
    }

    printf("[CC] Starting Afforest Connected Components (Rounds: %d, %u of %u nodes, %llu edges)...\n",
           neighbor_rounds, eval_nodes, g_n, (unsigned long long)g_e);

    roi_start();

    // Round 1: Initial linking [NDP CPU]
    cc_link_step(eval_nodes, neighbor_rounds, (uint32_t)-1, out_idx, out_nbr, g_comp);

    // Round 2: Multi-hop path compression [NDP CPU]
    cc_compress_step(1, eval_nodes, g_comp);

    // Round 3: Host identifies dominant component [Host CPU]
    uint32_t giant_comp = cc_sample_frequent(g_comp, eval_nodes, 1024);

    // Round 4: Final linking skipping giant component [NDP CPU]
    cc_link_step(eval_nodes, (uint32_t)-1, giant_comp, out_idx, out_nbr, g_comp);

    // Round 5: Final path compression [NDP CPU]
    cc_compress_step(2, eval_nodes, g_comp);

    roi_end();
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

    out_idx = g_alloc_out_idx;
    out_nbr = g_alloc_out_nbr;
}

int main(int argc, char *argv[])
{
    const char *graph_file = NULL;
    uint32_t scale = 14;
    int avg_deg = 16;
    unsigned seed = 42;
    int num_threads = 1;
    int neighbor_rounds = DEFAULT_NEIGHBOR_ROUNDS;
    uint32_t eval_nodes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i + 1 < argc) graph_file = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) neighbor_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) eval_nodes = (uint32_t)atoi(argv[++i]);
        else if (i == 1 && argv[i][0] != '-') scale = (uint32_t)atoi(argv[i]);
        else if (i == 2 && argv[i][0] != '-') avg_deg = atoi(argv[i]);
        else if (i == 3 && argv[i][0] != '-') seed = (unsigned)atoi(argv[i]);
        else if (i == 4 && argv[i][0] != '-') num_threads = atoi(argv[i]);
        else if (i == 5 && argv[i][0] != '-') neighbor_rounds = atoi(argv[i]);
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

    g_comp = (uint32_t *)malloc(g_n * sizeof(uint32_t));

    run_cc(neighbor_rounds, eval_nodes);

    uint64_t num_components = 0;
    for (uint32_t i = 0; i < g_n; i++) {
        if (g_comp[i] == i) num_components++;
    }
    printf("[Result] Total connected components found: %llu\n", (unsigned long long)num_components);

    free(g_comp);
    if (g_alloc_out_idx) free(g_alloc_out_idx);
    if (g_alloc_out_nbr) free(g_alloc_out_nbr);
    if (g_mmap_ptr) munmap(g_mmap_ptr, g_mmap_size);
    if (g_mmap_fd >= 0) close(g_mmap_fd);

    return 0;
}
