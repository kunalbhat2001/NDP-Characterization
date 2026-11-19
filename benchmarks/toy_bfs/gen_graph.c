/*
 * gen_graph.c  –  Independent Graph Generator for ToyBFS
 *
 * Generates an undirected random graph, constructs forward and reverse CSR,
 * and saves everything into a memory-mappable binary file (.bin).
 *
 * Usage:
 *   ./gen_graph <output_file.bin> [scale] [avg_degree] [seed]
 *   ./gen_graph graph_s16_d16.bin 16 16 42
 *   ./gen_graph graph_s18_d16.bin 18 16 42
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "toy_graph.h"

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

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output_file.bin> [scale] [avg_degree] [seed]\n", argv[0]);
        fprintf(stderr, "Example: %s graph_s16.bin 16 16 42\n", argv[0]);
        return 1;
    }

    const char *out_filename = argv[1];
    int      scale    = (argc > 2) ? atoi(argv[2]) : 16;
    int      avg_deg  = (argc > 3) ? atoi(argv[3]) : 16;
    unsigned seed     = (argc > 4) ? (unsigned)atoi(argv[4]) : 42;

    if (scale < 1 || scale > 28) {
        fprintf(stderr, "scale must be between 1 and 28 (got %d)\n", scale);
        return 1;
    }

    uint32_t n = (uint32_t)1u << scale;
    uint64_t n_raw = (uint64_t)n * (uint64_t)avg_deg / 2;
    uint64_t n_sym = n_raw * 2;

    printf("=== Toy Graph Generator ===\n");
    printf("  Output File : %s\n", out_filename);
    printf("  Scale       : %d (%u nodes)\n", scale, n);
    printf("  Avg Degree  : %d\n", avg_deg);
    printf("  Seed        : %u\n", seed);
    printf("  Total Edges : %llu directed edges\n", (unsigned long long)n_sym);

    clock_t t_start = clock();

    /* 1. Generate edge pairs */
    printf("[1/4] Generating raw edges...\n");
    Edge *edges = (Edge *)malloc(n_sym * sizeof(Edge));
    if (!edges) { perror("malloc edges"); return 1; }

    srand(seed);
    for (uint64_t i = 0; i < n_raw; i++) {
        uint32_t u = (uint32_t)((unsigned)rand() % n);
        uint32_t v = (uint32_t)((unsigned)rand() % n);
        if (u == v) v = (v + 1) % n;
        edges[2 * i]     = (Edge){u, v};
        edges[2 * i + 1] = (Edge){v, u};
    }

    /* 2. Build Out-CSR */
    printf("[2/4] Sorting and building Out-CSR...\n");
    uint32_t *out_idx = (uint32_t *)calloc(n + 1, sizeof(uint32_t));
    uint32_t *out_nbr = (uint32_t *)malloc(n_sym * sizeof(uint32_t));
    if (!out_idx || !out_nbr) { perror("malloc out-CSR"); return 1; }

    qsort(edges, n_sym, sizeof(Edge), cmp_by_u);
    for (uint64_t i = 0; i < n_sym; i++) out_idx[edges[i].u + 1]++;
    for (uint32_t i = 1; i <= n; i++)    out_idx[i] += out_idx[i - 1];
    for (uint64_t i = 0; i < n_sym; i++) out_nbr[i] = edges[i].v;

    /* 3. Build In-CSR */
    printf("[3/4] Sorting and building In-CSR...\n");
    uint32_t *in_idx = (uint32_t *)calloc(n + 1, sizeof(uint32_t));
    uint32_t *in_nbr = (uint32_t *)malloc(n_sym * sizeof(uint32_t));
    if (!in_idx || !in_nbr) { perror("malloc in-CSR"); return 1; }

    qsort(edges, n_sym, sizeof(Edge), cmp_by_v);
    for (uint64_t i = 0; i < n_sym; i++) in_idx[edges[i].v + 1]++;
    for (uint32_t i = 1; i <= n; i++)    in_idx[i] += in_idx[i - 1];
    for (uint64_t i = 0; i < n_sym; i++) in_nbr[i] = edges[i].u;

    free(edges);

    /* 4. Write binary file */
    printf("[4/4] Writing binary graph to '%s'...\n", out_filename);
    FILE *fp = fopen(out_filename, "wb");
    if (!fp) { perror("fopen output file"); return 1; }

    uint64_t header_sz  = sizeof(ToyGraphHeader);
    uint64_t out_idx_sz = (uint64_t)(n + 1) * sizeof(uint32_t);
    uint64_t out_nbr_sz = (uint64_t)n_sym * sizeof(uint32_t);
    uint64_t in_idx_sz  = (uint64_t)(n + 1) * sizeof(uint32_t);
    uint64_t in_nbr_sz  = (uint64_t)n_sym * sizeof(uint32_t);

    ToyGraphHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, TOYGRAPH_MAGIC, 8);
    hdr.version = TOYGRAPH_VERSION;
    hdr.scale = scale;
    hdr.avg_deg = avg_deg;
    hdr.seed = seed;
    hdr.num_nodes = n;
    hdr.num_edges = n_sym;

    hdr.out_idx_offset = header_sz;
    hdr.out_nbr_offset = hdr.out_idx_offset + out_idx_sz;
    hdr.in_idx_offset  = hdr.out_nbr_offset + out_nbr_sz;
    hdr.in_nbr_offset  = hdr.in_idx_offset + in_idx_sz;
    hdr.file_size      = hdr.in_nbr_offset + in_nbr_sz;

    if (fwrite(&hdr, sizeof(ToyGraphHeader), 1, fp) != 1 ||
        fwrite(out_idx, sizeof(uint32_t), n + 1, fp) != (size_t)(n + 1) ||
        fwrite(out_nbr, sizeof(uint32_t), n_sym, fp) != (size_t)n_sym ||
        fwrite(in_idx, sizeof(uint32_t), n + 1, fp) != (size_t)(n + 1) ||
        fwrite(in_nbr, sizeof(uint32_t), n_sym, fp) != (size_t)n_sym) {
        perror("fwrite binary graph");
        fclose(fp);
        return 1;
    }

    fclose(fp);
    free(out_idx);
    free(out_nbr);
    free(in_idx);
    free(in_nbr);

    double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    printf("[+] Graph generated successfully in %.2fs!\n", elapsed);
    printf("    File Size : %llu bytes (%.2f MB)\n\n",
           (unsigned long long)hdr.file_size,
           (double)hdr.file_size / (1024.0 * 1024.0));

    return 0;
}
