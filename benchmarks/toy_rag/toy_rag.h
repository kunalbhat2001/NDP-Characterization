#ifndef TOY_RAG_H
#define TOY_RAG_H

#include <stdint.h>
#include <stddef.h>

#define TOYRAG_MAGIC   "TOYRAGDB"
#define TOYRAG_VERSION 1

#pragma pack(push, 1)
typedef struct {
    char     magic[8];          /* "TOYRAGDB" */
    uint32_t version;           /* Format version (1) */
    uint32_t scale;             /* Graph scale (2^scale nodes) */
    uint32_t num_nodes;         /* Total document/vector nodes */
    uint32_t avg_deg;           /* Outgoing edges per node */
    uint32_t vec_dim;           /* Dimension of vector embedding (e.g. 64) */
    uint32_t entry_node;        /* Graph entry node for beam search */
    uint32_t seed;              /* Random seed */
    uint64_t num_edges;         /* Total edges */
    uint64_t vec_offset;        /* Byte offset to float vectors array [num_nodes * vec_dim] */
    uint64_t graph_idx_offset;  /* Byte offset to CSR row offset array [num_nodes + 1] */
    uint64_t graph_nbr_offset;  /* Byte offset to CSR neighbor array [num_edges] */
    uint64_t file_size;         /* Total file size in bytes */
    char     reserved[48];      /* Reserved for padding / future use */
} ToyRagHeader;
#pragma pack(pop)

typedef struct {
    uint32_t num_nodes;
    uint32_t vec_dim;
    uint32_t avg_deg;
    uint64_t num_edges;
    uint32_t entry_node;
    const float    *vectors;   /* [num_nodes * vec_dim] */
    const uint32_t *graph_idx; /* [num_nodes + 1] */
    const uint32_t *graph_nbr; /* [num_edges] */
} RagGraph;

typedef struct {
    uint32_t id;
    float    dist;
} RagCandidate;

#endif /* TOY_RAG_H */

