/*
 * toy_masstree.c – Masstree Key-Value Trie-of-B+Trees Index Microbenchmark
 *
 * Designed for StarNUMA + CXL Near-Data Processing (NDP) Offload evaluation.
 * Supports configurable memory footprint up to >100GB via memory arena (-m <GB>).
 *
 * Function Partitioning:
 *   [NDP CPU (Memory-Heavy)]:
 *     - masstree_trie_traverse: Multi-hop pointer chasing through trie layers.
 *     - masstree_leaf_search: Linear scan of 15-key B+ tree leaf cachelines.
 *     - masstree_range_scan: Traversal of sibling leaf pointers across DRAM.
 *   [Host CPU (Compute-Heavy)]:
 *     - masstree_encode_hash_key: Hash mixing, CRC-style bitwise rotations,
 *       and byte-swapping (bswap_64) to format keys.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include "toy_masstree.h"

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
 *  Memory Arena for >100GB Scale
 * ============================================================ */
static char    *g_arena_base = NULL;
static size_t   g_arena_size = 0;
static uint64_t g_arena_offset = 0;

static void *arena_alloc(size_t sz)
{
    // 64-byte alignment
    sz = (sz + 63) & ~63ULL;
    if (g_arena_base && g_arena_offset + sz <= g_arena_size) {
        uint64_t off = __sync_fetch_and_add(&g_arena_offset, sz);
        if (off + sz <= g_arena_size) {
            return (void *)(g_arena_base + off);
        }
    }
    return calloc(1, sz);
}

static masstree_leafnode_t *create_leaf(uint8_t layer)
{
    masstree_leafnode_t *l = (masstree_leafnode_t *)arena_alloc(sizeof(masstree_leafnode_t));
    memset(l, 0, sizeof(masstree_leafnode_t));
    l->hdr.type = NODE_LEAF;
    l->hdr.layer = layer;
    return l;
}

static masstree_internode_t *create_internode(uint8_t layer)
{
    masstree_internode_t *in = (masstree_internode_t *)arena_alloc(sizeof(masstree_internode_t));
    memset(in, 0, sizeof(masstree_internode_t));
    in->hdr.type = NODE_INTERNAL;
    in->hdr.layer = layer;
    return in;
}

/* ============================================================
 *  Kernel 1: Host CPU Candidate (Compute-Heavy Key Encoding)
 * ============================================================
 */
__attribute__((noinline))
masstree_key_t masstree_encode_hash_key(uint64_t raw_id)
{
    uint64_t h = raw_id ^ 0x517cc1b727220a95ULL;
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= (h >> 33);
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= (h >> 33);

    uint64_t prefix = raw_id / 100;
    uint64_t suffix = h;

    masstree_key_t k;
    k.slice[0] = __builtin_bswap64(prefix);
    k.slice[1] = __builtin_bswap64(suffix);
    return k;
}

/* ============================================================
 *  Kernel 2: NDP CPU Candidate (Memory-Heavy Leaf Scan)
 * ============================================================
 */
__attribute__((noinline))
int masstree_leaf_search(const masstree_leafnode_t *leaf, uint64_t slice)
{
    for (int i = 0; i < leaf->hdr.num_keys; i++) {
        if (leaf->keys[i] == slice) return i;
        if (leaf->keys[i] > slice) break;
    }
    return -1;
}

static masstree_leafnode_t *find_leaf(void *root, uint64_t slice)
{
    void *curr = root;
    while (curr) {
        masstree_node_header_t *hdr = (masstree_node_header_t *)curr;
        if (hdr->type == NODE_LEAF) {
            return (masstree_leafnode_t *)curr;
        }
        masstree_internode_t *in = (masstree_internode_t *)curr;
        int child_idx = 0;
        while (child_idx < in->hdr.num_keys && slice >= in->keys[child_idx]) {
            child_idx++;
        }
        curr = in->children[child_idx];
    }
    return NULL;
}

/* ============================================================
 *  Kernel 3: NDP CPU Candidate (Memory-Heavy Trie Traversal)
 * ============================================================
 */
__attribute__((noinline))
uint64_t masstree_trie_traverse(masstree_t *tree, masstree_key_t key)
{
    masstree_leafnode_t *leaf0 = find_leaf(tree->top_layer.root, key.slice[0]);
    if (!leaf0) return (uint64_t)-1;

    int idx0 = masstree_leaf_search(leaf0, key.slice[0]);
    if (idx0 < 0 || !leaf0->is_layer[idx0]) return (uint64_t)-1;

    masstree_layer_t *sub_layer = (masstree_layer_t *)leaf0->values[idx0];
    masstree_leafnode_t *leaf1 = find_leaf(sub_layer->root, key.slice[1]);
    if (!leaf1) return (uint64_t)-1;

    int idx1 = masstree_leaf_search(leaf1, key.slice[1]);
    if (idx1 < 0) return (uint64_t)-1;

    return leaf1->values[idx1];
}

/* ============================================================
 *  Kernel 4: NDP CPU Candidate (Memory-Heavy Sibling Range Scan)
 * ============================================================
 * Exposes (scan_len) to func_cache_sim to capture how scan length
 * transitions from cache hits to streaming DRAM bandwidth.
 */
__attribute__((noinline))
int masstree_range_scan(masstree_t *tree, masstree_key_t start_key, int scan_len, uint64_t *results_out)
{
    masstree_leafnode_t *leaf0 = find_leaf(tree->top_layer.root, start_key.slice[0]);
    if (!leaf0) return 0;

    int idx0 = masstree_leaf_search(leaf0, start_key.slice[0]);
    if (idx0 < 0 || !leaf0->is_layer[idx0]) return 0;

    masstree_layer_t *sub_layer = (masstree_layer_t *)leaf0->values[idx0];
    masstree_leafnode_t *leaf1 = find_leaf(sub_layer->root, start_key.slice[1]);
    if (!leaf1) return 0;

    int retrieved = 0;
    masstree_leafnode_t *curr = leaf1;
    while (curr && retrieved < scan_len) {
        for (int i = 0; i < curr->hdr.num_keys && retrieved < scan_len; i++) {
            results_out[retrieved++] = curr->values[i];
        }
        curr = curr->next_leaf;
    }
    return retrieved;
}

/* ============================================================
 *  Tree Mutation (Warmup / Setup)
 * ============================================================ */
typedef struct {
    uint64_t key;
    void    *right_node;
} split_result_t;

static split_result_t insert_recursive(void *node, uint8_t layer, uint64_t key, uint64_t value, uint8_t is_layer)
{
    masstree_node_header_t *hdr = (masstree_node_header_t *)node;
    split_result_t res = {0, NULL};

    if (hdr->type == NODE_LEAF) {
        masstree_leafnode_t *leaf = (masstree_leafnode_t *)node;
        int pos = 0;
        while (pos < leaf->hdr.num_keys && leaf->keys[pos] < key) pos++;

        if (pos < leaf->hdr.num_keys && leaf->keys[pos] == key) {
            leaf->values[pos] = value;
            leaf->is_layer[pos] = is_layer;
            return res;
        }

        if (leaf->hdr.num_keys < MASSTREE_FANOUT) {
            for (int i = leaf->hdr.num_keys; i > pos; i--) {
                leaf->keys[i] = leaf->keys[i - 1];
                leaf->values[i] = leaf->values[i - 1];
                leaf->is_layer[i] = leaf->is_layer[i - 1];
            }
            leaf->keys[pos] = key;
            leaf->values[pos] = value;
            leaf->is_layer[pos] = is_layer;
            leaf->hdr.num_keys++;
            return res;
        }

        masstree_leafnode_t *right_leaf = create_leaf(layer);
        uint64_t temp_keys[MASSTREE_FANOUT + 1];
        uint64_t temp_vals[MASSTREE_FANOUT + 1];
        uint8_t  temp_is_layer[MASSTREE_FANOUT + 1];

        for (int i = 0; i < pos; i++) {
            temp_keys[i] = leaf->keys[i];
            temp_vals[i] = leaf->values[i];
            temp_is_layer[i] = leaf->is_layer[i];
        }
        temp_keys[pos] = key;
        temp_vals[pos] = value;
        temp_is_layer[pos] = is_layer;
        for (int i = pos; i < MASSTREE_FANOUT; i++) {
            temp_keys[i + 1] = leaf->keys[i];
            temp_vals[i + 1] = leaf->values[i];
            temp_is_layer[i + 1] = leaf->is_layer[i];
        }

        int mid = (MASSTREE_FANOUT + 1) / 2;
        leaf->hdr.num_keys = mid;
        for (int i = 0; i < mid; i++) {
            leaf->keys[i] = temp_keys[i];
            leaf->values[i] = temp_vals[i];
            leaf->is_layer[i] = temp_is_layer[i];
        }

        right_leaf->hdr.num_keys = (MASSTREE_FANOUT + 1) - mid;
        for (int i = mid; i < MASSTREE_FANOUT + 1; i++) {
            right_leaf->keys[i - mid] = temp_keys[i];
            right_leaf->values[i - mid] = temp_vals[i];
            right_leaf->is_layer[i - mid] = temp_is_layer[i];
        }

        right_leaf->next_leaf = leaf->next_leaf;
        right_leaf->prev_leaf = leaf;
        if (leaf->next_leaf) leaf->next_leaf->prev_leaf = right_leaf;
        leaf->next_leaf = right_leaf;

        res.key = right_leaf->keys[0];
        res.right_node = right_leaf;
        return res;
    } else {
        masstree_internode_t *in = (masstree_internode_t *)node;
        int child_idx = 0;
        while (child_idx < in->hdr.num_keys && key >= in->keys[child_idx]) child_idx++;

        split_result_t child_split = insert_recursive(in->children[child_idx], layer, key, value, is_layer);
        if (!child_split.right_node) return res;

        uint64_t insert_key = child_split.key;
        void *insert_child = child_split.right_node;
        int pos = child_idx;

        if (in->hdr.num_keys < MASSTREE_FANOUT) {
            for (int i = in->hdr.num_keys; i > pos; i--) {
                in->keys[i] = in->keys[i - 1];
                in->children[i + 1] = in->children[i];
            }
            in->keys[pos] = insert_key;
            in->children[pos + 1] = insert_child;
            in->hdr.num_keys++;
            return res;
        }

        masstree_internode_t *right_in = create_internode(layer);
        uint64_t temp_keys[MASSTREE_FANOUT + 1];
        void *temp_children[MASSTREE_FANOUT + 2];

        for (int i = 0; i < pos; i++) temp_keys[i] = in->keys[i];
        temp_keys[pos] = insert_key;
        for (int i = pos; i < MASSTREE_FANOUT; i++) temp_keys[i + 1] = in->keys[i];

        for (int i = 0; i <= pos; i++) temp_children[i] = in->children[i];
        temp_children[pos + 1] = insert_child;
        for (int i = pos + 1; i <= MASSTREE_FANOUT; i++) temp_children[i + 1] = in->children[i];

        int mid = (MASSTREE_FANOUT + 1) / 2;
        in->hdr.num_keys = mid;
        for (int i = 0; i < mid; i++) {
            in->keys[i] = temp_keys[i];
            in->children[i] = temp_children[i];
        }
        in->children[mid] = temp_children[mid];

        res.key = temp_keys[mid];
        right_in->hdr.num_keys = MASSTREE_FANOUT - mid;
        for (int i = mid + 1; i <= MASSTREE_FANOUT; i++) {
            right_in->keys[i - (mid + 1)] = temp_keys[i];
            right_in->children[i - (mid + 1)] = temp_children[i];
        }
        right_in->children[right_in->hdr.num_keys] = temp_children[MASSTREE_FANOUT + 1];

        res.right_node = right_in;
        return res;
    }
}

static void layer_insert(masstree_layer_t *layer, uint64_t key, uint64_t value, uint8_t is_layer)
{
    if (!layer->root) {
        layer->root = create_leaf(layer->layer);
    }
    split_result_t split = insert_recursive(layer->root, layer->layer, key, value, is_layer);
    if (split.right_node) {
        masstree_internode_t *new_root = create_internode(layer->layer);
        new_root->hdr.num_keys = 1;
        new_root->keys[0] = split.key;
        new_root->children[0] = layer->root;
        new_root->children[1] = split.right_node;
        layer->root = new_root;
    }
}

static void masstree_insert_key(masstree_t *tree, masstree_key_t key, uint64_t val)
{
    masstree_leafnode_t *leaf0 = find_leaf(tree->top_layer.root, key.slice[0]);
    int idx0 = -1;
    if (leaf0) idx0 = masstree_leaf_search(leaf0, key.slice[0]);

    masstree_layer_t *sub_layer = NULL;
    if (idx0 >= 0 && leaf0->is_layer[idx0]) {
        sub_layer = (masstree_layer_t *)leaf0->values[idx0];
    } else {
        sub_layer = (masstree_layer_t *)arena_alloc(sizeof(masstree_layer_t));
        memset(sub_layer, 0, sizeof(masstree_layer_t));
        sub_layer->layer = 1;
        layer_insert(&tree->top_layer, key.slice[0], (uint64_t)sub_layer, 1);
    }

    layer_insert(sub_layer, key.slice[1], val, 0);
    tree->num_records++;
}

/* ============================================================
 *  Masstree Query Workload Orchestration (ROI)
 * ============================================================ */
__attribute__((noinline))
void run_masstree(masstree_t *tree, int num_queries, double range_scan_ratio, int default_scan_len)
{
    printf("[Masstree] Executing %d queries (ROI)...\n", num_queries);

    int num_range = (int)(num_queries * range_scan_ratio);
    int num_point = num_queries - num_range;

    int scan_lengths[] = {4, 16, 64, 256, 1024};
    int num_scan_options = sizeof(scan_lengths) / sizeof(scan_lengths[0]);

    uint64_t *results = (uint64_t *)malloc(2048 * sizeof(uint64_t));
    int total_found = 0;
    int total_scanned = 0;

    roi_start();

    // 1. Point Queries
    #pragma omp parallel for reduction(+ : total_found) schedule(dynamic, 128)
    for (int q = 0; q < num_point; q++) {
        uint64_t raw_id = ((uint64_t)q * 1103515245ULL + 12345ULL) % tree->num_records;
        masstree_key_t key = masstree_encode_hash_key(raw_id);
        uint64_t val = masstree_trie_traverse(tree, key);
        if (val != (uint64_t)-1) {
            total_found++;
        }
    }

    // 2. Range Scans with varying scan lengths
    for (int q = 0; q < num_range; q++) {
        uint64_t raw_id = ((uint64_t)(q + num_point) * 1103515245ULL + 12345ULL) % tree->num_records;
        masstree_key_t key = masstree_encode_hash_key(raw_id);

        int cur_len = scan_lengths[q % num_scan_options];
        if (default_scan_len > 0) cur_len = default_scan_len;

        int count = masstree_range_scan(tree, key, cur_len, results);
        total_scanned += count;
    }

    roi_end();

    free(results);
    printf("[Result] Point lookups matched: %d / %d, Range scan items: %d\n",
           total_found, num_point, total_scanned);
}

int main(int argc, char *argv[])
{
    int num_records = 50000;
    int num_queries = 20000;
    double range_ratio = 0.20;
    int scan_len = 0; // 0 = sweep [4, 16, 64, 256, 1024]
    int num_threads = 1;
    unsigned seed = 42;
    int mem_gb = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) num_records = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q") && i + 1 < argc) num_queries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) range_ratio = atof(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) scan_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) mem_gb = atoi(argv[++i]);
    }

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    srand(seed);

    if (mem_gb > 0) {
        g_arena_size = (size_t)mem_gb * 1024ULL * 1024ULL * 1024ULL;
        g_arena_base = (char *)mmap(NULL, g_arena_size, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (g_arena_base == MAP_FAILED) {
            perror("mmap large memory arena failed");
            g_arena_base = NULL;
            g_arena_size = 0;
        } else {
            printf("[Arena] Pre-allocated %d GB memory arena at %p\n", mem_gb, (void *)g_arena_base);
        }
    }

    printf("================================================================\n");
    printf("  Toy Masstree Microbenchmark (StarNUMA + CXL NDP)\n");
    printf("  Initial Records : %d\n", num_records);
    printf("  Queries (ROI)   : %d (%.0f%% range scans)\n", num_queries, range_ratio * 100.0);
    printf("  Threads         : %d\n", num_threads);
    if (g_arena_size) printf("  Memory Footprint: %.2f GB\n", (double)g_arena_size / (1024.0 * 1024.0 * 1024.0));
    printf("================================================================\n");

    masstree_t tree;
    memset(&tree, 0, sizeof(tree));
    tree.top_layer.layer = 0;

    printf("[Setup] Populating Masstree index...\n");
    for (int i = 0; i < num_records; i++) {
        masstree_key_t k = masstree_encode_hash_key((uint64_t)i);
        masstree_insert_key(&tree, k, (uint64_t)(i + 1000));
    }
    printf("[Setup] Successfully populated %llu records across Trie-of-B+Trees.\n",
           (unsigned long long)tree.num_records);

    run_masstree(&tree, num_queries, range_ratio, scan_len);

    if (g_arena_base) {
        munmap(g_arena_base, g_arena_size);
    }
    return 0;
}
