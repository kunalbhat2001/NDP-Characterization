/*
 * toy_cxl_stream.c – High-MLP Streaming Microbenchmark to Saturate CXL Link (64 GB/s)
 *
 * Designed for StarNUMA / ChampSim simulation.
 * Features:
 *   - Configurable large working set (default 256MB) to ensure 100% LLC miss rate.
 *   - High Memory-Level Parallelism (MLP): unrolled 64-byte strided accesses.
 *   - Multiple streaming access kernels: read, write, copy, triad.
 *   - OpenMP multi-threading support for multi-core evaluations.
 *   - StarNUMA Pin tracer compatible ROI markers (roi_start / roi_end).
 *   - Pre-ROI memory touch/warmup to eliminate page fault interference in traces.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int  omp_get_max_threads(void) { return 1; }
static inline int  omp_get_thread_num(void)  { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

#define CACHELINE_BYTES 64
#define WORDS_PER_LINE  (CACHELINE_BYTES / sizeof(uint64_t))  /* 8 uint64_t per line */

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
 *  Streaming Kernels
 * ============================================================ */

/* Kernel 1: High-MLP 8-way unrolled read using AVX2 registers */
__attribute__((noinline))
uint64_t kernel_stream_read_8(const uint64_t *arr, size_t num_lines)
{
    __m256i a0 = _mm256_setzero_si256();
    __m256i a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256();
    __m256i a3 = _mm256_setzero_si256();
    __m256i a4 = _mm256_setzero_si256();
    __m256i a5 = _mm256_setzero_si256();
    __m256i a6 = _mm256_setzero_si256();
    __m256i a7 = _mm256_setzero_si256();

    size_t i = 0;
    size_t limit = (num_lines >= 8) ? (num_lines - 7) : 0;

    for (; i < limit; i += 8) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)&arr[(i + 0) * WORDS_PER_LINE]);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)&arr[(i + 1) * WORDS_PER_LINE]);
        __m256i v2 = _mm256_loadu_si256((const __m256i*)&arr[(i + 2) * WORDS_PER_LINE]);
        __m256i v3 = _mm256_loadu_si256((const __m256i*)&arr[(i + 3) * WORDS_PER_LINE]);
        __m256i v4 = _mm256_loadu_si256((const __m256i*)&arr[(i + 4) * WORDS_PER_LINE]);
        __m256i v5 = _mm256_loadu_si256((const __m256i*)&arr[(i + 5) * WORDS_PER_LINE]);
        __m256i v6 = _mm256_loadu_si256((const __m256i*)&arr[(i + 6) * WORDS_PER_LINE]);
        __m256i v7 = _mm256_loadu_si256((const __m256i*)&arr[(i + 7) * WORDS_PER_LINE]);

        a0 = _mm256_add_epi64(a0, v0);
        a1 = _mm256_add_epi64(a1, v1);
        a2 = _mm256_add_epi64(a2, v2);
        a3 = _mm256_add_epi64(a3, v3);
        a4 = _mm256_add_epi64(a4, v4);
        a5 = _mm256_add_epi64(a5, v5);
        a6 = _mm256_add_epi64(a6, v6);
        a7 = _mm256_add_epi64(a7, v7);
    }

    __m256i sum0 = _mm256_add_epi64(a0, a1);
    __m256i sum1 = _mm256_add_epi64(a2, a3);
    __m256i sum2 = _mm256_add_epi64(a4, a5);
    __m256i sum3 = _mm256_add_epi64(a6, a7);
    __m256i total = _mm256_add_epi64(_mm256_add_epi64(sum0, sum1),
                                     _mm256_add_epi64(sum2, sum3));

    uint64_t res[4];
    _mm256_storeu_si256((__m256i*)res, total);
    uint64_t tail_sum = res[0] + res[1] + res[2] + res[3];
    for (; i < num_lines; i++) {
        tail_sum += arr[i * WORDS_PER_LINE];
    }
    return tail_sum;
}

/* Kernel 2: High-MLP 8-way unrolled write / stream */
__attribute__((noinline))
void kernel_stream_write_8(uint64_t *arr, size_t num_lines, uint64_t val)
{
    __m256i v0 = _mm256_set1_epi64x((int64_t)val);
    __m256i v1 = _mm256_set1_epi64x((int64_t)(val + 1));

    size_t i = 0;
    size_t limit = (num_lines >= 8) ? (num_lines - 7) : 0;

    for (; i < limit; i += 8) {
        _mm256_stream_si256((__m256i*)&arr[(i + 0) * WORDS_PER_LINE], v0);
        _mm256_stream_si256((__m256i*)&arr[(i + 1) * WORDS_PER_LINE], v1);
        _mm256_stream_si256((__m256i*)&arr[(i + 2) * WORDS_PER_LINE], v0);
        _mm256_stream_si256((__m256i*)&arr[(i + 3) * WORDS_PER_LINE], v1);
        _mm256_stream_si256((__m256i*)&arr[(i + 4) * WORDS_PER_LINE], v0);
        _mm256_stream_si256((__m256i*)&arr[(i + 5) * WORDS_PER_LINE], v1);
        _mm256_stream_si256((__m256i*)&arr[(i + 6) * WORDS_PER_LINE], v0);
        _mm256_stream_si256((__m256i*)&arr[(i + 7) * WORDS_PER_LINE], v1);
    }
    for (; i < num_lines; i++) {
        arr[i * WORDS_PER_LINE] = val + i;
    }
}

/* Kernel 3: High-MLP 8-way unrolled copy (dst = src) */
__attribute__((noinline))
void kernel_stream_copy_8(uint64_t *dst, const uint64_t *src, size_t num_lines)
{
    size_t i = 0;
    size_t limit = (num_lines >= 8) ? (num_lines - 7) : 0;

    for (; i < limit; i += 8) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)&src[(i + 0) * WORDS_PER_LINE]);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)&src[(i + 1) * WORDS_PER_LINE]);
        __m256i v2 = _mm256_loadu_si256((const __m256i*)&src[(i + 2) * WORDS_PER_LINE]);
        __m256i v3 = _mm256_loadu_si256((const __m256i*)&src[(i + 3) * WORDS_PER_LINE]);
        __m256i v4 = _mm256_loadu_si256((const __m256i*)&src[(i + 4) * WORDS_PER_LINE]);
        __m256i v5 = _mm256_loadu_si256((const __m256i*)&src[(i + 5) * WORDS_PER_LINE]);
        __m256i v6 = _mm256_loadu_si256((const __m256i*)&src[(i + 6) * WORDS_PER_LINE]);
        __m256i v7 = _mm256_loadu_si256((const __m256i*)&src[(i + 7) * WORDS_PER_LINE]);

        _mm256_stream_si256((__m256i*)&dst[(i + 0) * WORDS_PER_LINE], v0);
        _mm256_stream_si256((__m256i*)&dst[(i + 1) * WORDS_PER_LINE], v1);
        _mm256_stream_si256((__m256i*)&dst[(i + 2) * WORDS_PER_LINE], v2);
        _mm256_stream_si256((__m256i*)&dst[(i + 3) * WORDS_PER_LINE], v3);
        _mm256_stream_si256((__m256i*)&dst[(i + 4) * WORDS_PER_LINE], v4);
        _mm256_stream_si256((__m256i*)&dst[(i + 5) * WORDS_PER_LINE], v5);
        _mm256_stream_si256((__m256i*)&dst[(i + 6) * WORDS_PER_LINE], v6);
        _mm256_stream_si256((__m256i*)&dst[(i + 7) * WORDS_PER_LINE], v7);
    }
    for (; i < num_lines; i++) {
        dst[i * WORDS_PER_LINE] = src[i * WORDS_PER_LINE];
    }
}

/* Kernel 4: High-MLP 8-way unrolled triad (C = A + scalar * B) */
__attribute__((noinline))
void kernel_stream_triad_8(uint64_t *c, const uint64_t *a, const uint64_t *b,
                           uint64_t scalar, size_t num_lines)
{
    __m256i s = _mm256_set1_epi64x((int64_t)scalar);
    size_t i = 0;
    size_t limit = (num_lines >= 8) ? (num_lines - 7) : 0;

    for (; i < limit; i += 8) {
        __m256i va0 = _mm256_loadu_si256((const __m256i*)&a[(i + 0) * WORDS_PER_LINE]);
        __m256i vb0 = _mm256_loadu_si256((const __m256i*)&b[(i + 0) * WORDS_PER_LINE]);
        __m256i va1 = _mm256_loadu_si256((const __m256i*)&a[(i + 1) * WORDS_PER_LINE]);
        __m256i vb1 = _mm256_loadu_si256((const __m256i*)&b[(i + 1) * WORDS_PER_LINE]);
        __m256i va2 = _mm256_loadu_si256((const __m256i*)&a[(i + 2) * WORDS_PER_LINE]);
        __m256i vb2 = _mm256_loadu_si256((const __m256i*)&b[(i + 2) * WORDS_PER_LINE]);
        __m256i va3 = _mm256_loadu_si256((const __m256i*)&a[(i + 3) * WORDS_PER_LINE]);
        __m256i vb3 = _mm256_loadu_si256((const __m256i*)&b[(i + 3) * WORDS_PER_LINE]);

        _mm256_stream_si256((__m256i*)&c[(i + 0) * WORDS_PER_LINE], _mm256_add_epi64(va0, _mm256_mul_epu32(vb0, s)));
        _mm256_stream_si256((__m256i*)&c[(i + 1) * WORDS_PER_LINE], _mm256_add_epi64(va1, _mm256_mul_epu32(vb1, s)));
        _mm256_stream_si256((__m256i*)&c[(i + 2) * WORDS_PER_LINE], _mm256_add_epi64(va2, _mm256_mul_epu32(vb2, s)));
        _mm256_stream_si256((__m256i*)&c[(i + 3) * WORDS_PER_LINE], _mm256_add_epi64(va3, _mm256_mul_epu32(vb3, s)));
    }
    for (; i < num_lines; i++) {
        c[i * WORDS_PER_LINE] = a[i * WORDS_PER_LINE] + scalar * b[i * WORDS_PER_LINE];
    }
}

/* ============================================================
 *  Helper Functions
 * ============================================================ */
static void *allocate_buffer(size_t bytes)
{
    void *ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void warmup_buffer(uint64_t *arr, size_t num_lines)
{
    /* First touch to fault in physical pages before tracing ROI */
    for (size_t i = 0; i < num_lines; i++) {
        arr[i * WORDS_PER_LINE] = (uint64_t)i;
    }
}

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --size-mb <MB>     Buffer size in megabytes (default: 256)\n");
    printf("  --kernel <name>    Kernel type: 'read', 'write', 'copy', 'triad' (default: read)\n");
    printf("  --threads <N>      Number of OpenMP threads (default: 1)\n");
    printf("  --iters <N>        Number of iterations inside ROI (default: 1)\n");
    printf("  --help, -h         Show this help message\n");
}

/* ============================================================
 *  Main
 * ============================================================ */
int main(int argc, char *argv[])
{
    size_t size_mb = 256;
    char kernel_type[32] = "read";
    int num_threads = 1;
    int iters = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size-mb") == 0 && i + 1 < argc) {
            size_mb = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            strncpy(kernel_type, argv[++i], sizeof(kernel_type) - 1);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    omp_set_num_threads(num_threads);

    size_t total_bytes = size_mb * 1024ULL * 1024ULL;
    size_t num_lines = total_bytes / CACHELINE_BYTES;

    printf("============================================================\n");
    printf("  toy_cxl_stream: CXL Link Saturation Microbenchmark\n");
    printf("============================================================\n");
    printf("  Target Bandwidth : 64 GB/s\n");
    printf("  Buffer Size      : %zu MB (%zu cache lines)\n", size_mb, num_lines);
    printf("  Kernel           : %s\n", kernel_type);
    printf("  Threads          : %d\n", num_threads);
    printf("  Iterations       : %d\n", iters);
    printf("============================================================\n");

    /* Allocate buffers */
    uint64_t *buf_a = (uint64_t *)allocate_buffer(total_bytes);
    uint64_t *buf_b = NULL;
    uint64_t *buf_c = NULL;

    if (strcmp(kernel_type, "copy") == 0) {
        buf_b = (uint64_t *)allocate_buffer(total_bytes);
    } else if (strcmp(kernel_type, "triad") == 0) {
        buf_b = (uint64_t *)allocate_buffer(total_bytes);
        buf_c = (uint64_t *)allocate_buffer(total_bytes);
    }

    /* Warmup / first-touch pages before ROI */
    printf("[*] Warming up buffers (pre-ROI first touch)...\n");
    warmup_buffer(buf_a, num_lines);
    if (buf_b) warmup_buffer(buf_b, num_lines);
    if (buf_c) warmup_buffer(buf_c, num_lines);

    printf("[*] Entering ROI...\n");
    fflush(stdout);

    volatile uint64_t sink_val = 0;
    double t_start = get_time_sec();

    /* Begin StarNUMA Pin ROI */
    roi_start();

    for (int it = 0; it < iters; it++) {
        if (strcmp(kernel_type, "read") == 0) {
            uint64_t local_sum = 0;
            #pragma omp parallel reduction(+:local_sum)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_max_threads();
                size_t chunk = num_lines / nth;
                size_t start = tid * chunk;
                size_t end = (tid == nth - 1) ? num_lines : start + chunk;

                local_sum += kernel_stream_read_8(&buf_a[start * WORDS_PER_LINE], end - start);
            }
            sink_val += local_sum;
        } else if (strcmp(kernel_type, "write") == 0) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_max_threads();
                size_t chunk = num_lines / nth;
                size_t start = tid * chunk;
                size_t end = (tid == nth - 1) ? num_lines : start + chunk;

                kernel_stream_write_8(&buf_a[start * WORDS_PER_LINE], end - start, (uint64_t)it);
            }
        } else if (strcmp(kernel_type, "copy") == 0) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_max_threads();
                size_t chunk = num_lines / nth;
                size_t start = tid * chunk;
                size_t end = (tid == nth - 1) ? num_lines : start + chunk;

                kernel_stream_copy_8(&buf_b[start * WORDS_PER_LINE],
                                     &buf_a[start * WORDS_PER_LINE],
                                     end - start);
            }
        } else if (strcmp(kernel_type, "triad") == 0) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_max_threads();
                size_t chunk = num_lines / nth;
                size_t start = tid * chunk;
                size_t end = (tid == nth - 1) ? num_lines : start + chunk;

                kernel_stream_triad_8(&buf_c[start * WORDS_PER_LINE],
                                      &buf_a[start * WORDS_PER_LINE],
                                      &buf_b[start * WORDS_PER_LINE],
                                      3ULL,
                                      end - start);
            }
        }
    }

    /* End StarNUMA Pin ROI */
    roi_end();

    double t_end = get_time_sec();
    printf("[*] Exited ROI.\n");

    /* Prevent optimization / dead code elimination */
    if (sink_val == 0xdeadbeef) {
        printf("Sink: %lu\n", sink_val);
    }

    double elapsed = t_end - t_start;
    double gb_transferred = 0.0;
    if (strcmp(kernel_type, "read") == 0 || strcmp(kernel_type, "write") == 0) {
        gb_transferred = (double)(total_bytes * iters) / (1024.0 * 1024.0 * 1024.0);
    } else if (strcmp(kernel_type, "copy") == 0) {
        gb_transferred = (double)(2 * total_bytes * iters) / (1024.0 * 1024.0 * 1024.0);
    } else if (strcmp(kernel_type, "triad") == 0) {
        gb_transferred = (double)(3 * total_bytes * iters) / (1024.0 * 1024.0 * 1024.0);
    }
    double bw_gbps = (elapsed > 0) ? (gb_transferred / elapsed) : 0.0;

    printf("============================================================\n");
    printf("  Execution Time   : %.4f s\n", elapsed);
    printf("  Native Bandwidth : %.2f GB/s\n", bw_gbps);
    printf("============================================================\n");

    munmap(buf_a, total_bytes);
    if (buf_b) munmap(buf_b, total_bytes);
    if (buf_c) munmap(buf_c, total_bytes);

    return 0;
}
