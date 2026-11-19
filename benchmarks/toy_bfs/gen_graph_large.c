/*
 * gen_graph_large.c – Fast Parallel Graph Generator for >100GB Datasets
 *
 * Designed for StarNUMA + CXL NDP large-scale evaluation.
 * Uses OpenMP parallel generation and 64-bit CSR offsets (Version 2)
 * to support graphs with tens of billions of edges without integer overflow.
 *
 * Usage:
 *   ./gen_graph_large <output_file.bin> [scale] [avg_degree] [seed] [threads]
 * Example for >100GB graph:
 *   ./gen_graph_large graph_105GB.bin 28 48 42 64
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <omp.h>
#include "toy_graph.h"

// Fast xorshift64 pseudo-random generator
static inline uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    return (ua < ub) ? -1 : (ua > ub ? 1 : 0);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output_file.bin> [scale] [avg_degree] [seed] [threads]\n", argv[0]);
        fprintf(stderr, "Example: %s graph_105GB.bin 28 48 42 64\n", argv[0]);
        return 1;
    }

    const char *out_filename = argv[1];
    int scale    = (argc > 2) ? atoi(argv[2]) : 28;
    int avg_deg  = (argc > 3) ? atoi(argv[3]) : 48;
    unsigned seed = (argc > 4) ? (unsigned)atoi(argv[4]) : 42;
    int num_threads = (argc > 5) ? atoi(argv[5]) : omp_get_max_threads();

    omp_set_num_threads(num_threads);

    uint32_t n = 1u << scale;
    uint64_t e = (uint64_t)n * (uint64_t)avg_deg;

    // 64-bit CSR sizing
    uint64_t header_sz  = sizeof(ToyGraphHeader);
    uint64_t out_idx_sz = (uint64_t)(n + 1) * sizeof(uint64_t);
    uint64_t out_nbr_sz = (uint64_t)e * sizeof(uint32_t);
    uint64_t in_idx_sz  = (uint64_t)(n + 1) * sizeof(uint64_t);
    uint64_t in_nbr_sz  = (uint64_t)e * sizeof(uint32_t);
    uint64_t total_file_size = header_sz + out_idx_sz + out_nbr_sz + in_idx_sz + in_nbr_sz;

    printf("================================================================\n");
    printf("  Large Graph Generator (StarNUMA 64-bit CSR)\n");
    printf("  Output File      : %s\n", out_filename);
    printf("  Scale            : %d (Nodes: %u)\n", scale, n);
    printf("  Avg Degree       : %d\n", avg_deg);
    printf("  Directed Edges   : %llu\n", (unsigned long long)e);
    printf("  File Size        : %.2f GB (%llu bytes)\n",
           (double)total_file_size / (1024.0 * 1024.0 * 1024.0), (unsigned long long)total_file_size);
    printf("  Parallel Threads : %d\n", num_threads);
    printf("================================================================\n");

    double t_start = omp_get_wtime();

    FILE *fp = fopen(out_filename, "wb+");
    if (!fp) {
        perror("Error opening output file");
        return 1;
    }

    // Pre-allocate / truncate file to target size
    int fd = fileno(fp);
    if (ftruncate(fd, total_file_size) != 0) {
        perror("ftruncate failed");
    }

    // 1. Write Header
    ToyGraphHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, TOYGRAPH_MAGIC, 8);
    hdr.version = TOYGRAPH_VERSION_64;
    hdr.scale = scale;
    hdr.avg_deg = avg_deg;
    hdr.seed = seed;
    hdr.num_nodes = n;
    hdr.num_edges = e;

    hdr.out_idx_offset = header_sz;
    hdr.out_nbr_offset = hdr.out_idx_offset + out_idx_sz;
    hdr.in_idx_offset  = hdr.out_nbr_offset + out_nbr_sz;
    hdr.in_nbr_offset  = hdr.in_idx_offset + in_idx_sz;
    hdr.file_size      = total_file_size;

    if (fwrite(&hdr, sizeof(ToyGraphHeader), 1, fp) != 1) {
        perror("fwrite header failed");
        fclose(fp);
        return 1;
    }

    // 2. Build & Write out_idx (64-bit)
    printf("[1/4] Writing out_idx (64-bit offsets)...\n");
    uint64_t *out_idx = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
    if (!out_idx) { perror("malloc out_idx"); fclose(fp); return 1; }

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i <= n; i++) {
        out_idx[i] = (uint64_t)i * (uint64_t)avg_deg;
    }

    if (fwrite(out_idx, sizeof(uint64_t), n + 1, fp) != (size_t)(n + 1)) {
        perror("fwrite out_idx failed");
        free(out_idx);
        fclose(fp);
        return 1;
    }

    // 3. Generate & Write out_nbr in streaming chunks
    printf("[2/4] Generating & writing out_nbr (%llu edges)...\n", (unsigned long long)e);
    uint32_t chunk_nodes = 1048576; // 1M nodes per chunk
    if (chunk_nodes > n) chunk_nodes = n;
    uint64_t chunk_edges = (uint64_t)chunk_nodes * (uint64_t)avg_deg;
    uint32_t *nbr_buf = (uint32_t *)malloc(chunk_edges * sizeof(uint32_t));
    if (!nbr_buf) { perror("malloc nbr_buf"); free(out_idx); fclose(fp); return 1; }

    for (uint32_t chunk_start = 0; chunk_start < n; chunk_start += chunk_nodes) {
        uint32_t current_chunk_nodes = chunk_nodes;
        if (chunk_start + current_chunk_nodes > n) current_chunk_nodes = n - chunk_start;
        uint64_t current_chunk_edges = (uint64_t)current_chunk_nodes * (uint64_t)avg_deg;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint64_t thr_seed = (uint64_t)seed * 1103515245ULL + (uint64_t)(chunk_start + tid) * 6364136223846793005ULL;

            #pragma omp for schedule(static)
            for (uint32_t u_rel = 0; u_rel < current_chunk_nodes; u_rel++) {
                uint32_t u_abs = chunk_start + u_rel;
                uint64_t buf_pos = (uint64_t)u_rel * (uint64_t)avg_deg;

                for (int d = 0; d < avg_deg; d++) {
                    uint32_t v = (uint32_t)(xorshift64(&thr_seed) % n);
                    if (v == u_abs) v = (v + 1) % n;
                    nbr_buf[buf_pos + d] = v;
                }

                // Sort local neighbor list for binary search / merge intersection
                if (avg_deg > 1) {
                    qsort(&nbr_buf[buf_pos], avg_deg, sizeof(uint32_t), cmp_u32);
                }
            }
        }

        if (fwrite(nbr_buf, sizeof(uint32_t), current_chunk_edges, fp) != (size_t)current_chunk_edges) {
            perror("fwrite out_nbr chunk failed");
            free(nbr_buf);
            free(out_idx);
            fclose(fp);
            return 1;
        }
    }
    free(nbr_buf);

    // 4. Write in_idx (identical uniform offset structure for symmetric / in-edges)
    printf("[3/4] Writing in_idx (64-bit offsets)...\n");
    if (fwrite(out_idx, sizeof(uint64_t), n + 1, fp) != (size_t)(n + 1)) {
        perror("fwrite in_idx failed");
        free(out_idx);
        fclose(fp);
        return 1;
    }
    free(out_idx);

    // 5. Generate in_nbr (reverse CSR edges for PageRank pull operations)
    printf("[4/4] Generating in_nbr...\n");
    nbr_buf = (uint32_t *)malloc(chunk_edges * sizeof(uint32_t));
    for (uint32_t chunk_start = 0; chunk_start < n; chunk_start += chunk_nodes) {
        uint32_t current_chunk_nodes = chunk_nodes;
        if (chunk_start + current_chunk_nodes > n) current_chunk_nodes = n - chunk_start;
        uint64_t current_chunk_edges = (uint64_t)current_chunk_nodes * (uint64_t)avg_deg;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint64_t thr_seed = (uint64_t)(seed + 999) * 1103515245ULL + (uint64_t)(chunk_start + tid) * 6364136223846793005ULL;

            #pragma omp for schedule(static)
            for (uint32_t u_rel = 0; u_rel < current_chunk_nodes; u_rel++) {
                uint32_t u_abs = chunk_start + u_rel;
                uint64_t buf_pos = (uint64_t)u_rel * (uint64_t)avg_deg;

                for (int d = 0; d < avg_deg; d++) {
                    uint32_t v = (uint32_t)(xorshift64(&thr_seed) % n);
                    if (v == u_abs) v = (v + 1) % n;
                    nbr_buf[buf_pos + d] = v;
                }

                if (avg_deg > 1) {
                    qsort(&nbr_buf[buf_pos], avg_deg, sizeof(uint32_t), cmp_u32);
                }
            }
        }

        if (fwrite(nbr_buf, sizeof(uint32_t), current_chunk_edges, fp) != (size_t)current_chunk_edges) {
            perror("fwrite in_nbr chunk failed");
            free(nbr_buf);
            fclose(fp);
            return 1;
        }
    }
    free(nbr_buf);

    fclose(fp);
    double elapsed = omp_get_wtime() - t_start;

    printf("\n[SUCCESS] Generated %.2f GB binary graph in %.2f seconds (%.2f MB/s)!\n",
           (double)total_file_size / (1024.0 * 1024.0 * 1024.0),
           elapsed,
           ((double)total_file_size / (1024.0 * 1024.0)) / elapsed);
    return 0;
}
