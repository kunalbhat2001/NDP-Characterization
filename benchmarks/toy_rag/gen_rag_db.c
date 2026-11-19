/*
 * gen_rag_db.c – Generator for Toy RAG Pre-Generated Vector & Graph Database (.bin)
 *
 * Generates:
 *   1. High-dimensional vector embeddings for N document chunks (N x D floats).
 *   2. Navigable proximity graph adjacency in CSR format (short-range + long-range small-world edges).
 *   3. Binary file with ToyRagHeader for zero-latency mmap loading during tracing.
 *
 * Usage:
 *   ./gen_rag_db <output.bin> [scale] [avg_degree] [vec_dim] [seed]
 *   ./gen_rag_db rag_db_s16_d32_v64.bin 16 32 64 42
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "toy_rag.h"

static inline uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand_float(uint32_t *state)
{
    return ((float)(xorshift32(state) & 0xFFFF) / 32768.0f) - 1.0f;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    return (ua < ub) ? -1 : (ua > ub);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output.bin> [scale] [avg_degree] [vec_dim] [seed]\n", argv[0]);
        fprintf(stderr, "Default: scale=16 (64K nodes), avg_degree=32, vec_dim=64, seed=42\n");
        return 1;
    }

    const char *out_filename = argv[1];
    uint32_t scale = (argc > 2) ? (uint32_t)atoi(argv[2]) : 16;
    uint32_t avg_deg = (argc > 3) ? (uint32_t)atoi(argv[3]) : 32;
    uint32_t vec_dim = (argc > 4) ? (uint32_t)atoi(argv[4]) : 64;
    uint32_t seed = (argc > 5) ? (uint32_t)atoi(argv[5]) : 42;

    if (scale < 8 || scale > 24) {
        fprintf(stderr, "Error: scale must be between 8 and 24\n");
        return 1;
    }
    if (avg_deg < 4 || avg_deg > 256) {
        fprintf(stderr, "Error: avg_deg must be between 4 and 256\n");
        return 1;
    }
    if (vec_dim < 8 || vec_dim > 1024) {
        fprintf(stderr, "Error: vec_dim must be between 8 and 1024\n");
        return 1;
    }

    uint32_t num_nodes = 1u << scale;
    uint64_t total_vectors = (uint64_t)num_nodes * vec_dim;
    uint64_t total_edges = (uint64_t)num_nodes * avg_deg;

    printf("============================================================\n");
    printf(" Generating Toy RAG Database Binary File\n");
    printf("   Output file : %s\n", out_filename);
    printf("   Scale       : %u (2^%u = %u nodes)\n", scale, scale, num_nodes);
    printf("   Avg Degree  : %u edges/node\n", avg_deg);
    printf("   Vector Dim  : %u floats (%zu bytes/vector)\n", vec_dim, vec_dim * sizeof(float));
    printf("   Total Edges : %llu\n", (unsigned long long)total_edges);
    printf("   Seed        : %u\n", seed);
    printf("============================================================\n");

    /* Allocate vector embeddings */
    printf("[1/3] Generating normalized vector embeddings (%.2f MB)...\n",
           (double)(total_vectors * sizeof(float)) / (1024.0 * 1024.0));
    float *vectors = (float *)malloc(total_vectors * sizeof(float));
    if (!vectors) {
        perror("malloc vectors");
        return 1;
    }

    uint32_t rng = seed;
    for (uint32_t i = 0; i < num_nodes; i++) {
        float *vec = &vectors[(uint64_t)i * vec_dim];
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < vec_dim; d++) {
            float val = rand_float(&rng);
            vec[d] = val;
            norm_sq += val * val;
        }
        float inv_norm = 1.0f / sqrtf(norm_sq > 1e-6f ? norm_sq : 1.0f);
        for (uint32_t d = 0; d < vec_dim; d++) {
            vec[d] *= inv_norm;
        }
    }

    /* Allocate CSR graph structures */
    printf("[2/3] Constructing Navigable Small-World Proximity Graph (%.2f MB)...\n",
           (double)(total_edges * sizeof(uint32_t)) / (1024.0 * 1024.0));
    uint32_t *graph_idx = (uint32_t *)malloc(((size_t)num_nodes + 1) * sizeof(uint32_t));
    uint32_t *graph_nbr = (uint32_t *)malloc((size_t)total_edges * sizeof(uint32_t));
    if (!graph_idx || !graph_nbr) {
        perror("malloc CSR arrays");
        return 1;
    }

    /* Generate small-world topology (HNSW / Kleinberg style):
     * Mix of local clustered ring neighbors + logarithmic power-of-2 skip links + random short shortcuts.
     */
    uint32_t *temp_nbrs = (uint32_t *)malloc(avg_deg * 2 * sizeof(uint32_t));
    uint64_t edge_pos = 0;

    for (uint32_t i = 0; i < num_nodes; i++) {
        graph_idx[i] = (uint32_t)edge_pos;
        uint32_t count = 0;

        /* Ring neighbors (local clusters) */
        uint32_t ring_span = avg_deg / 4;
        if (ring_span < 1) ring_span = 1;
        for (uint32_t r = 1; r <= ring_span && count < avg_deg; r++) {
            temp_nbrs[count++] = (i + r) % num_nodes;
            temp_nbrs[count++] = (i + num_nodes - r) % num_nodes;
        }

        /* Navigable Small-World logarithmic skip hops (power-of-2) */
        for (uint32_t s = 1; (1u << s) < num_nodes && count < avg_deg; s++) {
            temp_nbrs[count++] = (i + (1u << s)) % num_nodes;
            temp_nbrs[count++] = (i + num_nodes - (1u << s)) % num_nodes;
        }

        /* Fill remainder with pseudo-random shortcuts */
        while (count < avg_deg) {
            uint32_t target = xorshift32(&rng) % num_nodes;
            if (target != i) {
                temp_nbrs[count++] = target;
            }
        }

        /* Sort neighbors for cache-friendly scan and remove duplicates */
        qsort(temp_nbrs, count, sizeof(uint32_t), cmp_u32);
        uint32_t unique_count = 0;
        for (uint32_t c = 0; c < count && unique_count < avg_deg; c++) {
            if (c == 0 || temp_nbrs[c] != temp_nbrs[c - 1]) {
                graph_nbr[edge_pos + unique_count] = temp_nbrs[c];
                unique_count++;
            }
        }
        /* Pad to avg_deg if duplicates were removed */
        while (unique_count < avg_deg) {
            graph_nbr[edge_pos + unique_count] = (i + unique_count + 1) % num_nodes;
            unique_count++;
        }
        edge_pos += avg_deg;
    }
    graph_idx[num_nodes] = (uint32_t)edge_pos;
    free(temp_nbrs);

    /* Construct header */
    ToyRagHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, TOYRAG_MAGIC, 8);
    hdr.version = TOYRAG_VERSION;
    hdr.scale = scale;
    hdr.num_nodes = num_nodes;
    hdr.avg_deg = avg_deg;
    hdr.vec_dim = vec_dim;
    hdr.entry_node = num_nodes / 2; /* Centered entry point */
    hdr.seed = seed;
    hdr.num_edges = total_edges;

    /* Offsets aligned to 64 bytes (cache line friendly) */
    uint64_t cur_off = sizeof(ToyRagHeader);
    cur_off = (cur_off + 63) & ~63ULL;
    hdr.vec_offset = cur_off;

    cur_off += total_vectors * sizeof(float);
    cur_off = (cur_off + 63) & ~63ULL;
    hdr.graph_idx_offset = cur_off;

    cur_off += ((uint64_t)num_nodes + 1) * sizeof(uint32_t);
    cur_off = (cur_off + 63) & ~63ULL;
    hdr.graph_nbr_offset = cur_off;

    cur_off += total_edges * sizeof(uint32_t);
    hdr.file_size = cur_off;

    printf("[3/3] Writing binary database to '%s' (%.2f MB)...\n",
           out_filename, (double)hdr.file_size / (1024.0 * 1024.0));

    FILE *fp = fopen(out_filename, "wb");
    if (!fp) {
        perror("fopen output file");
        return 1;
    }

    /* Write header */
    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        perror("fwrite header");
        fclose(fp);
        return 1;
    }

    /* Seek and write vectors */
    fseek(fp, (long)hdr.vec_offset, SEEK_SET);
    if (fwrite(vectors, sizeof(float), total_vectors, fp) != total_vectors) {
        perror("fwrite vectors");
        fclose(fp);
        return 1;
    }

    /* Seek and write graph_idx */
    fseek(fp, (long)hdr.graph_idx_offset, SEEK_SET);
    if (fwrite(graph_idx, sizeof(uint32_t), (size_t)num_nodes + 1, fp) != (size_t)num_nodes + 1) {
        perror("fwrite graph_idx");
        fclose(fp);
        return 1;
    }

    /* Seek and write graph_nbr */
    fseek(fp, (long)hdr.graph_nbr_offset, SEEK_SET);
    if (fwrite(graph_nbr, sizeof(uint32_t), (size_t)total_edges, fp) != (size_t)total_edges) {
        perror("fwrite graph_nbr");
        fclose(fp);
        return 1;
    }

    fclose(fp);
    free(vectors);
    free(graph_idx);
    free(graph_nbr);

    printf("Database generation complete! File: %s (%llu bytes)\n\n",
           out_filename, (unsigned long long)hdr.file_size);
    return 0;
}

