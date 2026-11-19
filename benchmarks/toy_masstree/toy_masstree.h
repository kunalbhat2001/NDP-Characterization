/*
 * toy_masstree.h – Data Structures for Masstree Microbenchmark
 *
 * Implements a Trie-of-B+Trees in-memory key-value index.
 * Layers:
 *   Layer 0: B+ tree indexed by 8-byte slice 0.
 *   Layer 1: B+ tree indexed by 8-byte slice 1, storing the payload values.
 *
 * Designed for StarNUMA + CXL Near-Data Processing (NDP) Offload evaluation.
 */

#ifndef TOY_MASSTREE_H
#define TOY_MASSTREE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define MASSTREE_FANOUT 15  // Maximum keys per node (fits cleanly in cache lines)

typedef enum {
    NODE_INTERNAL = 0,
    NODE_LEAF     = 1
} node_type_t;

typedef struct masstree_node_header {
    node_type_t type;
    uint16_t    num_keys;
    uint8_t     layer;
} masstree_node_header_t;

typedef struct masstree_internode {
    masstree_node_header_t hdr;
    uint64_t keys[MASSTREE_FANOUT];
    void    *children[MASSTREE_FANOUT + 1];
} masstree_internode_t;

typedef struct masstree_leafnode {
    masstree_node_header_t    hdr;
    uint64_t                  keys[MASSTREE_FANOUT];
    uint64_t                  values[MASSTREE_FANOUT];
    uint8_t                   is_layer[MASSTREE_FANOUT]; // 1 if value is ptr to sub-tree
    struct masstree_leafnode *prev_leaf;
    struct masstree_leafnode *next_leaf;
} masstree_leafnode_t;

// A 16-byte key consisting of two 64-bit slices
typedef struct {
    uint64_t slice[2];
} masstree_key_t;

typedef struct {
    void    *root;
    uint8_t  layer;
} masstree_layer_t;

typedef struct {
    masstree_layer_t top_layer;
    uint64_t         num_records;
} masstree_t;

#endif /* TOY_MASSTREE_H */
