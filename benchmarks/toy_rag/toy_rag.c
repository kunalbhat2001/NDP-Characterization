/*
 * toy_rag.c – Retrieval-Augmented Generation (RAG) Microbenchmark
 *
 * Models a realistic end-to-end RAG query pipeline partitioned into:
 *   Phase 1: rag_encode_query     – Dense GEMV query projection (Host CPU, compute-heavy)
 *   Phase 2: rag_graph_retrieval  – HNSW-style proximity graph traversal (NDP CPU, memory-heavy)
 *   Phase 3: rag_llm_generation   – Multi-head attention & decoder GEMM (Host CPU, compute-heavy)
 *
 * Supports:
 *   1. Memory-mapped (.bin) pre-generated graph & vector databases (instant startup for Pin):
 *        ./toy_rag -f rag_db_s16_d32_v64.bin [-q num_queries] [-k top_k] [--ef ef_search]
 *   2. In-memory generation fallback (for quick standalone runs):
 *        ./toy_rag [scale] [avg_degree] [vec_dim] [num_queries]
 *
 * StarNUMA / Pin ROI markers (xchg rbx,rbx) encompass the query execution loop.
 * Memory warmup is performed before roi_start() to eliminate page faults during tracing.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>
#include "toy_rag.h"

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
 *  RAG Database Storage (Graph + Vectors)
 * ============================================================ */
static RagGraph g_graph;

/* mmap tracking */
static void   *g_mmap_ptr  = NULL;
static size_t  g_mmap_size = 0;
static int     g_mmap_fd   = -1;

/* In-memory fallback buffers */
static float    *g_alloc_vectors   = NULL;
static uint32_t *g_alloc_graph_idx = NULL;
static uint32_t *g_alloc_graph_nbr = NULL;

/* Context & Model weights for Phase 1 & Phase 3 */
static float *g_proj_weights    = NULL; /* [vec_dim * vec_dim] */
static float *g_context_weights = NULL; /* [vec_dim * vec_dim] */
static float *g_ffn_weights     = NULL; /* [vec_dim * vec_dim] */

/* Dynamic query buffers */
static float    *g_prompt_tokens = NULL; /* [vec_dim] */
static float    *g_query_vec     = NULL; /* [vec_dim] */
static uint32_t *g_retrieved_ids = NULL; /* [top_k] */
static float    *g_output_tokens = NULL; /* [num_gen_tokens * vec_dim] */

/* Visited bitmap for graph traversal */
static uint64_t *g_visited_bmap  = NULL; /* [num_nodes / 64] */

/* Visited bitmap helpers */
static inline int bm_get(const uint64_t *bm, uint32_t i)
{
    return (int)((bm[i >> 6] >> (i & 63)) & 1u);
}

static inline void bm_set(uint64_t *bm, uint32_t i)
{
    bm[i >> 6] |= (uint64_t)1u << (i & 63);
}

/* ============================================================
 *  Distance Metric (Squared Euclidean Distance)
 * ============================================================ */
static inline float compute_l2_dist_sq(const float *a, const float *b, uint32_t dim)
{
    float dist = 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}

/* ============================================================
 *  PHASE 1: Query Encoding / Embedding Projection (Host CPU)
 *
 *  [COMPUTE-BOUND]: High arithmetic intensity GEMV projection
 *  q = Activation(W_proj * x_in + b)
 * ============================================================ */
__attribute__((noinline))
void rag_encode_query(const float *prompt_tokens, const float *proj_weights,
                      float *query_vec, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++) {
        float sum = 0.0f;
        const float *row = &proj_weights[i * dim];
        for (uint32_t j = 0; j < dim; j++) {
            sum += row[j] * prompt_tokens[j];
        }
        /* GELU-like polynomial activation: x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))) / 2 */
        float x3 = sum * sum * sum;
        float act = 0.5f * sum * (1.0f + 0.79788f * (sum + 0.044715f * x3));
        query_vec[i] = act;
    }

    /* Normalize query vector */
    float norm_sq = 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
        norm_sq += query_vec[d] * query_vec[d];
    }
    float inv_norm = 1.0f / sqrtf(norm_sq > 1e-6f ? norm_sq : 1.0f);
    for (uint32_t d = 0; d < dim; d++) {
        query_vec[d] *= inv_norm;
    }
}

/* ============================================================
 *  PHASE 2: Memory-Heavy Graph Traversal (NDP CPU Offload Target)
 *
 *  [MEMORY-BOUND & POINTER CHASING]:
 *  HNSW / Vamana Greedy Beam Search over proximity graph.
 *  Traverses node neighbor lists and loads high-dimensional
 *  node vectors scattered across memory, causing high LLC miss rates.
 * ============================================================ */
__attribute__((noinline))
uint32_t rag_graph_retrieval(const float *query_vec, const RagGraph *graph,
                             uint32_t ef_search, uint32_t top_k,
                             uint32_t *retrieved_ids)
{
    uint32_t num_nodes = graph->num_nodes;
    uint32_t dim = graph->vec_dim;
    const float *vectors = graph->vectors;
    const uint32_t *graph_idx = graph->graph_idx;
    const uint32_t *graph_nbr = graph->graph_nbr;

    /* Reset visited bitmap */
    size_t bm_words = ((size_t)num_nodes + 63) >> 6;
    memset(g_visited_bmap, 0, bm_words * sizeof(uint64_t));

    /* Bounded candidate pool (sorted by distance ascending) */
    RagCandidate candidates[64];
    uint32_t cand_count = 0;

    /* Start at graph entry node */
    uint32_t entry = graph->entry_node;
    bm_set(g_visited_bmap, entry);

    const float *entry_vec = &vectors[(uint64_t)entry * dim];
    float entry_dist = compute_l2_dist_sq(query_vec, entry_vec, dim);

    candidates[0].id = entry;
    candidates[0].dist = entry_dist;
    cand_count = 1;

    /* Beam Search Loop */
    uint32_t curr_idx = 0;
    while (curr_idx < cand_count && curr_idx < ef_search) {
        RagCandidate curr = candidates[curr_idx++];

        /* Retrieve neighbor range (Pointer Chasing in CSR) */
        uint32_t start_edge = graph_idx[curr.id];
        uint32_t end_edge   = graph_idx[curr.id + 1];

        for (uint32_t e = start_edge; e < end_edge; e++) {
            uint32_t nbr_id = graph_nbr[e];

            if (bm_get(g_visited_bmap, nbr_id)) {
                continue;
            }
            bm_set(g_visited_bmap, nbr_id);

            /* Random DRAM / CXL Load: Fetch neighbor's D-dimensional vector */
            const float *nbr_vec = &vectors[(uint64_t)nbr_id * dim];
            float nbr_dist = compute_l2_dist_sq(query_vec, nbr_vec, dim);

            /* Insert into sorted candidate list if within ef_search */
            if (cand_count < ef_search || nbr_dist < candidates[cand_count - 1].dist) {
                uint32_t pos = cand_count < ef_search ? cand_count : (cand_count - 1);
                while (pos > 0 && nbr_dist < candidates[pos - 1].dist) {
                    if (pos < 64) {
                        candidates[pos] = candidates[pos - 1];
                    }
                    pos--;
                }
                candidates[pos].id = nbr_id;
                candidates[pos].dist = nbr_dist;
                if (cand_count < ef_search && cand_count < 64) {
                    cand_count++;
                }
            }
        }
    }

    /* Output top-K nearest node IDs */
    uint32_t result_count = (cand_count < top_k) ? cand_count : top_k;
    for (uint32_t k = 0; k < result_count; k++) {
        retrieved_ids[k] = candidates[k].id;
    }

    return result_count;
}

/* ============================================================
 *  PHASE 3: Context Assembly & LLM Generation (Host CPU)
 *
 *  [COMPUTE-BOUND]: Dense Matrix Operations simulating
 *  Multi-Head Attention and Transformer Decoder FFN projection.
 * ============================================================ */
__attribute__((noinline))
void rag_llm_generation(const uint32_t *retrieved_ids, uint32_t top_k,
                        const float *context_matrix, float *output_tokens,
                        uint32_t num_tokens, uint32_t dim)
{
    /* Local context accumulator buffer */
    float ctx_accum[128];
    uint32_t d_limit = (dim < 128) ? dim : 128;
    for (uint32_t d = 0; d < d_limit; d++) {
        ctx_accum[d] = 0.0f;
    }

    /* Multi-head attention simulation: Aggregate retrieved context embeddings */
    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t doc_id = retrieved_ids[k];
        /* Weight scaling based on rank */
        float attn_weight = 1.0f / (float)(k + 1);
        const float *doc_vec = &g_graph.vectors[(uint64_t)doc_id * dim];

        for (uint32_t d = 0; d < d_limit; d++) {
            ctx_accum[d] += attn_weight * doc_vec[d];
        }
    }

    /* Autoregressive token generation simulation */
    for (uint32_t t = 0; t < num_tokens; t++) {
        float *token_out = &output_tokens[t * dim];

        /* Transformer Decoder FFN GEMM */
        for (uint32_t i = 0; i < dim; i++) {
            float val = 0.0f;
            const float *w_row = &context_matrix[i * dim];
            for (uint32_t j = 0; j < d_limit; j++) {
                val += w_row[j] * ctx_accum[j];
            }
            /* Non-linear projection */
            token_out[i] = tanhf(val);
        }

        /* Update context autoregressively for next token */
        for (uint32_t d = 0; d < d_limit; d++) {
            ctx_accum[d] = 0.9f * ctx_accum[d] + 0.1f * token_out[d];
        }
    }
}

/* ============================================================
 *  Binary Database mmap Loader
 * ============================================================ */
static int load_mmap_db(const char *filename)
{
    g_mmap_fd = open(filename, O_RDONLY);
    if (g_mmap_fd < 0) {
        perror("open binary rag db file");
        return -1;
    }

    struct stat st;
    if (fstat(g_mmap_fd, &st) < 0) {
        perror("fstat");
        close(g_mmap_fd);
        return -1;
    }
    g_mmap_size = (size_t)st.st_size;

    g_mmap_ptr = mmap(NULL, g_mmap_size, PROT_READ, MAP_PRIVATE, g_mmap_fd, 0);
    if (g_mmap_ptr == MAP_FAILED) {
        perror("mmap");
        close(g_mmap_fd);
        return -1;
    }

    if (g_mmap_size < sizeof(ToyRagHeader)) {
        fprintf(stderr, "Error: file '%s' too small for ToyRagHeader\n", filename);
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
        return -1;
    }

    const ToyRagHeader *hdr = (const ToyRagHeader *)g_mmap_ptr;
    if (memcmp(hdr->magic, TOYRAG_MAGIC, 8) != 0) {
        fprintf(stderr, "Error: invalid magic in '%s' (expected '%.8s')\n",
                filename, TOYRAG_MAGIC);
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
        return -1;
    }

    if (hdr->version != TOYRAG_VERSION) {
        fprintf(stderr, "Error: unsupported ToyRag version %u (supported: %u)\n",
                hdr->version, TOYRAG_VERSION);
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
        return -1;
    }

    g_graph.num_nodes  = hdr->num_nodes;
    g_graph.vec_dim    = hdr->vec_dim;
    g_graph.avg_deg    = hdr->avg_deg;
    g_graph.num_edges  = hdr->num_edges;
    g_graph.entry_node = hdr->entry_node;

    g_graph.vectors   = (const float *)((const char *)g_mmap_ptr + hdr->vec_offset);
    g_graph.graph_idx = (const uint32_t *)((const char *)g_mmap_ptr + hdr->graph_idx_offset);
    g_graph.graph_nbr = (const uint32_t *)((const char *)g_mmap_ptr + hdr->graph_nbr_offset);

    printf("[mmap] Successfully mapped binary RAG DB '%s':\n", filename);
    printf("  Nodes       : %u (scale %u)\n", g_graph.num_nodes, hdr->scale);
    printf("  Avg Degree  : %u edges/node\n", g_graph.avg_deg);
    printf("  Vector Dim  : %u floats (%zu bytes/vector)\n", g_graph.vec_dim, g_graph.vec_dim * sizeof(float));
    printf("  Total Edges : %llu\n", (unsigned long long)g_graph.num_edges);
    printf("  File Size   : %llu bytes (%.2f MB)\n\n",
           (unsigned long long)g_mmap_size, (double)g_mmap_size / (1024.0 * 1024.0));

    return 0;
}

/* ============================================================
 *  In-Memory Database Construction (Fallback)
 * ============================================================ */
static void build_db_in_memory(uint32_t scale, uint32_t avg_deg, uint32_t vec_dim, uint32_t seed)
{
    uint32_t num_nodes = 1u << scale;
    uint64_t total_vectors = (uint64_t)num_nodes * vec_dim;
    uint64_t total_edges = (uint64_t)num_nodes * avg_deg;

    printf("[fallback] Generating in-memory RAG database (%u nodes, %u dim, %u deg)...\n",
           num_nodes, vec_dim, avg_deg);

    g_alloc_vectors   = (float *)malloc(total_vectors * sizeof(float));
    g_alloc_graph_idx = (uint32_t *)malloc(((size_t)num_nodes + 1) * sizeof(uint32_t));
    g_alloc_graph_nbr = (uint32_t *)malloc((size_t)total_edges * sizeof(uint32_t));

    if (!g_alloc_vectors || !g_alloc_graph_idx || !g_alloc_graph_nbr) {
        perror("malloc in-memory DB");
        exit(1);
    }

    uint32_t rng = seed;
    for (uint32_t i = 0; i < num_nodes; i++) {
        float *vec = &g_alloc_vectors[(uint64_t)i * vec_dim];
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < vec_dim; d++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            float val = ((float)(rng & 0xFFFF) / 32768.0f) - 1.0f;
            vec[d] = val;
            norm_sq += val * val;
        }
        float inv_norm = 1.0f / sqrtf(norm_sq > 1e-6f ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < vec_dim; d++) {
            vec[d] *= inv_norm;
        }
    }

    uint64_t edge_pos = 0;
    for (uint32_t i = 0; i < num_nodes; i++) {
        g_alloc_graph_idx[i] = (uint32_t)edge_pos;
        for (uint32_t deg = 0; deg < avg_deg; deg++) {
            if (deg == 0) {
                g_alloc_graph_nbr[edge_pos + deg] = (i + 1) % num_nodes;
            } else if (deg == 1) {
                g_alloc_graph_nbr[edge_pos + deg] = (i + num_nodes - 1) % num_nodes;
            } else {
                uint32_t step = 1u << ((deg % 14) + 1);
                g_alloc_graph_nbr[edge_pos + deg] = (i + step) % num_nodes;
            }
        }
        edge_pos += avg_deg;
    }
    g_alloc_graph_idx[num_nodes] = (uint32_t)edge_pos;

    g_graph.num_nodes  = num_nodes;
    g_graph.vec_dim    = vec_dim;
    g_graph.avg_deg    = avg_deg;
    g_graph.num_edges  = total_edges;
    g_graph.entry_node = num_nodes / 2;
    g_graph.vectors    = g_alloc_vectors;
    g_graph.graph_idx  = g_alloc_graph_idx;
    g_graph.graph_nbr  = g_alloc_graph_nbr;
}

/* ============================================================
 *  Pre-ROI Memory Warmup Pass
 *
 *  Touches every page in mapped arrays and heap buffers to
 *  ensure physical page allocation and prevent page fault stalls.
 * ============================================================ */
static void memory_warmup(void)
{
    printf("[warmup] Touching all memory pages to prevent page fault stalls during tracing...\n");
    volatile uint32_t sink = 0;

    /* Touch vectors */
    const float *v = g_graph.vectors;
    size_t vec_floats = (size_t)g_graph.num_nodes * g_graph.vec_dim;
    for (size_t i = 0; i < vec_floats; i += 1024) {
        sink += (uint32_t)v[i];
    }

    /* Touch graph index */
    const uint32_t *idx = g_graph.graph_idx;
    for (size_t i = 0; i <= g_graph.num_nodes; i += 1024) {
        sink += idx[i];
    }

    /* Touch graph neighbors */
    const uint32_t *nbr = g_graph.graph_nbr;
    for (size_t i = 0; i < g_graph.num_edges; i += 1024) {
        sink += nbr[i];
    }

    /* Touch model weight matrices */
    size_t dim = g_graph.vec_dim;
    for (size_t i = 0; i < dim * dim; i += 1024) {
        sink += (uint32_t)g_proj_weights[i];
        sink += (uint32_t)g_context_weights[i];
    }

    (void)sink;
    printf("[warmup] Memory warmup complete.\n\n");
}

/* ============================================================
 *  Main
 * ============================================================ */
int main(int argc, char *argv[])
{
    const char *graph_file = NULL;
    uint32_t scale       = 14;  /* 16K nodes (standalone default) */
    uint32_t avg_deg     = 32;  /* 32 edges/node */
    uint32_t vec_dim     = 64;  /* 64 floats */
    uint32_t num_queries = 20;  /* Number of RAG query loops */
    uint32_t top_k       = 8;   /* Retrieve top-8 nearest chunks */
    uint32_t ef_search   = 32;  /* Beam search candidate window */
    uint32_t num_tokens  = 16;  /* Generate 16 tokens */
    uint32_t seed        = 42;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) && i + 1 < argc) {
            graph_file = argv[++i];
        } else if ((strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--queries") == 0) && i + 1 < argc) {
            num_queries = (uint32_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--topk") == 0) && i + 1 < argc) {
            top_k = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ef") == 0 && i + 1 < argc) {
            ef_search = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            num_tokens = (uint32_t)atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            /* Positional arguments: [scale] [avg_deg] [vec_dim] [queries] */
            static int pos = 0;
            if (pos == 0) scale = (uint32_t)atoi(argv[i]);
            else if (pos == 1) avg_deg = (uint32_t)atoi(argv[i]);
            else if (pos == 2) vec_dim = (uint32_t)atoi(argv[i]);
            else if (pos == 3) num_queries = (uint32_t)atoi(argv[i]);
            pos++;
        }
    }

    printf("============================================================\n");
    printf(" Toy RAG Benchmark (Host GEMV -> NDP Graph Retrieval -> Host LLM)\n");
    printf("============================================================\n");

    /* Initialize database (mmap or in-memory) */
    if (graph_file) {
        if (load_mmap_db(graph_file) != 0) {
            fprintf(stderr, "Failed to load binary graph '%s'\n", graph_file);
            return 1;
        }
    } else {
        build_db_in_memory(scale, avg_deg, vec_dim, seed);
    }

    uint32_t dim = g_graph.vec_dim;
    if (ef_search > 64) ef_search = 64;
    if (top_k > ef_search) top_k = ef_search;

    /* Allocate state buffers */
    size_t bm_words = ((size_t)g_graph.num_nodes + 63) >> 6;
    g_visited_bmap = (uint64_t *)calloc(bm_words, sizeof(uint64_t));

    g_prompt_tokens = (float *)malloc(dim * sizeof(float));
    g_query_vec     = (float *)malloc(dim * sizeof(float));
    g_retrieved_ids = (uint32_t *)malloc(top_k * sizeof(uint32_t));
    g_output_tokens = (float *)malloc((size_t)num_tokens * dim * sizeof(float));

    g_proj_weights    = (float *)malloc((size_t)dim * dim * sizeof(float));
    g_context_weights = (float *)malloc((size_t)dim * dim * sizeof(float));
    g_ffn_weights     = (float *)malloc((size_t)dim * dim * sizeof(float));

    if (!g_visited_bmap || !g_prompt_tokens || !g_query_vec || !g_retrieved_ids ||
        !g_output_tokens || !g_proj_weights || !g_context_weights || !g_ffn_weights) {
        perror("malloc state buffers");
        return 1;
    }

    /* Initialize projection and context weights */
    uint32_t w_rng = 12345;
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        w_rng ^= w_rng << 13; w_rng ^= w_rng >> 17; w_rng ^= w_rng << 5;
        g_proj_weights[i]    = (((float)(w_rng & 0xFFFF) / 32768.0f) - 1.0f) / sqrtf((float)dim);
        w_rng ^= w_rng << 13; w_rng ^= w_rng >> 17; w_rng ^= w_rng << 5;
        g_context_weights[i] = (((float)(w_rng & 0xFFFF) / 32768.0f) - 1.0f) / sqrtf((float)dim);
        w_rng ^= w_rng << 13; w_rng ^= w_rng >> 17; w_rng ^= w_rng << 5;
        g_ffn_weights[i]     = (((float)(w_rng & 0xFFFF) / 32768.0f) - 1.0f) / sqrtf((float)dim);
    }

    /* Warm up memory */
    memory_warmup();

    printf("Executing RAG Pipeline:\n");
    printf("  Queries       : %u\n", num_queries);
    printf("  Top-K Chunks  : %u\n", top_k);
    printf("  Beam efSearch : %u\n", ef_search);
    printf("  Tokens/Query  : %u\n", num_tokens);
    printf("------------------------------------------------------------\n");

    /* ============================================================
     *  REGION OF INTEREST (ROI)
     * ============================================================ */
    roi_start();

    for (uint32_t q = 0; q < num_queries; q++) {
        /* Generate synthetic query prompt tokens */
        uint32_t q_rng = seed + q * 10007;
        for (uint32_t d = 0; d < dim; d++) {
            q_rng ^= q_rng << 13; q_rng ^= q_rng >> 17; q_rng ^= q_rng << 5;
            g_prompt_tokens[d] = ((float)(q_rng & 0xFFFF) / 32768.0f) - 1.0f;
        }

        /* ----------------------------------------------------
         * Phase 1: Query Encoding (Host CPU, Compute-Heavy)
         * ---------------------------------------------------- */
        rag_encode_query(g_prompt_tokens, g_proj_weights, g_query_vec, dim);

        /* ----------------------------------------------------
         * Phase 2: Graph-Based Vector Retrieval (NDP CPU Offload)
         * ---------------------------------------------------- */
        uint32_t n_ret = rag_graph_retrieval(g_query_vec, &g_graph, ef_search, top_k, g_retrieved_ids);

        /* ----------------------------------------------------
         * Phase 3: Context Assembly & Generation (Host CPU)
         * ---------------------------------------------------- */
        rag_llm_generation(g_retrieved_ids, n_ret, g_context_weights, g_output_tokens, num_tokens, dim);
    }

    roi_end();
    /* ============================================================
     *  END OF REGION OF INTEREST
     * ============================================================ */

    printf("RAG benchmark completed successfully!\n");
    printf("Sample query output token[0]: %.4f\n", g_output_tokens[0]);
    printf("Sample retrieved top-1 chunk ID: %u\n", g_retrieved_ids[0]);
    printf("============================================================\n");

    /* Cleanup */
    if (g_mmap_ptr) {
        munmap(g_mmap_ptr, g_mmap_size);
        close(g_mmap_fd);
    }
    if (g_alloc_vectors)   free(g_alloc_vectors);
    if (g_alloc_graph_idx) free(g_alloc_graph_idx);
    if (g_alloc_graph_nbr) free(g_alloc_graph_nbr);

    free(g_visited_bmap);
    free(g_prompt_tokens);
    free(g_query_vec);
    free(g_retrieved_ids);
    free(g_output_tokens);
    free(g_proj_weights);
    free(g_context_weights);
    free(g_ffn_weights);

    return 0;
}

