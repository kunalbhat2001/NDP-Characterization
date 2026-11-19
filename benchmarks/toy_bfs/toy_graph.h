#ifndef TOY_GRAPH_H
#define TOY_GRAPH_H

#include <stdint.h>
#include <stddef.h>

#define TOYGRAPH_MAGIC      "TOYGRAPH"
#define TOYGRAPH_VERSION    1
#define TOYGRAPH_VERSION_32 1  /* 32-bit offsets in out_idx/in_idx */
#define TOYGRAPH_VERSION_64 2  /* 64-bit offsets in out_idx/in_idx for >100GB / >4B edges */

#pragma pack(push, 1)
typedef struct {
    char     magic[8];       /* "TOYGRAPH" */
    uint32_t version;        /* Format version (1) */
    uint32_t scale;          /* Graph scale (2^scale nodes) */
    uint32_t avg_deg;        /* Average degree */
    uint32_t seed;           /* Seed used to generate */
    uint32_t num_nodes;      /* g_n */
    uint64_t num_edges;      /* g_e (total directed edges) */
    uint64_t out_idx_offset; /* Byte offset to out_idx array */
    uint64_t out_nbr_offset; /* Byte offset to out_nbr array */
    uint64_t in_idx_offset;  /* Byte offset to in_idx array */
    uint64_t in_nbr_offset;  /* Byte offset to in_nbr array */
    uint64_t file_size;      /* Total file size in bytes */
    char     reserved[48];   /* Reserved for padding / future use */
} ToyGraphHeader;
#pragma pack(pop)

#endif /* TOY_GRAPH_H */
