# Toy RAG Microbenchmark (`toy_rag`)

A representative microbenchmark modeling a **Retrieval-Augmented Generation (RAG)** pipeline with memory-heavy graph traversal designed for **Near-Data Processing (NDP) / CXL Offloading** on the StarNUMA / ChampSim simulation platform.

---

## 1. Workload Architecture

The benchmark partitions a real-world RAG query pipeline into three distinct phases:

```
[ User Query Tokens ]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: rag_encode_query (Host CPU)                  │
│ Dense GEMV query projection: q = Act(W_proj * x + b)   │
│ Compute-heavy, high arithmetic intensity, L1/L2 hits   │
└────────────────────────────────────────────────────────┘
         │  Query vector q (D-dim)
         ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: rag_graph_retrieval (NDP CPU, CXL Offloaded) │
│ HNSW / Vamana Greedy Beam Search over Proximity Graph  │
│ Memory-heavy: Pointer chasing + Scattered Vector Loads │
│ Spills LLC into CXL / DRAM memory channels             │
└────────────────────────────────────────────────────────┘
         │  Top-K Retrieved Document IDs
         ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: rag_llm_generation (Host CPU)                │
│ Multi-Head Attention & Transformer Decoder GEMM       │
│ Compute-heavy, matrix multiplications & token decoding │
└────────────────────────────────────────────────────────┘
         │
         ▼
[ Generated Response Tokens ]
```

### Real-Life RAG Implementations & Sources Modeled
1. **HNSW Proximity Graph Retrieval**:
   - Malkov, Y. A., & Yashunin, D. A. (2018). *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."* IEEE TPAMI.
   - Production systems: **Meta FAISS** (`IndexHNSWFlat`), Milvus, Qdrant, ChromaDB, pgvector.
2. **DiskANN / Vamana Graph**:
   - Subramanya, S. J., et al. (2019). *"DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node."* NeurIPS 2019.
3. **Knowledge Graph RAG**:
   - Microsoft **GraphRAG**: Edge et al. (2024), *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization."*
   - **HippoRAG**: Gutiérrez et al. (NeurIPS 2024), *"HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models."*
4. **Near-Data Processing (NDP) Acceleration**:
   - Samsung CXL-PNM (Processing-Near-Memory for In-Memory Vector Search, Hot Chips 34).
   - Ahn et al. (ISCA 2015), *"A Scalable Processing-in-Memory Accelerator for Parallel Graph Processing (Tesseract)."*

---

## 2. Directory Structure

```
benchmarks/toy_rag/
├── toy_rag.h              # Binary format header (ToyRagHeader) and data structures
├── toy_rag.c              # Core benchmark: Phase 1, Phase 2, Phase 3 with Pin ROI markers
├── gen_rag_db.c           # Offline database generator (produces .bin pre-generated files)
├── Makefile               # Build script with debug symbols and ChampSim symbol extractor
├── toy_rag_trace_run.py   # Pin tracer runner script (supports 200M insts per phase)
├── rag_db_s16_d32_v64.bin # Pre-generated Scale-16 database (65K nodes, 32 deg, 64 dim)
└── README.md              # Documentation
```

---

## 3. Building

To build the benchmark, generator, and extract ChampSim symbols:

```bash
cd benchmarks/toy_rag
make
```

To generate a new pre-generated binary database of custom scale:
```bash
make db SCALE=16 DEG=32 DIM=64
```

---

## 4. Running Standalone

### Using Pre-Generated Binary File (Instant mmap startup):
```bash
./toy_rag -f rag_db_s16_d32_v64.bin -q 50 -k 8 --ef 32
```

### Using In-Memory Fallback:
```bash
./toy_rag 14 32 64 20
```

---

## 5. Tracing with Intel Pin

Set environment variables:
```bash
export SNUMA_PINPATH=/path/to/pin
export SNUMA_TRACER_PATH=/path/to/combined_tracer.so
export STARNUMA_ARTIFACT_PATH=/nethome/kbhat43/Development/StarNUMA+NDP
```

Run trace generation (defaults to 200M instructions per phase and pre-generated DB):
```bash
python3 toy_rag_trace_run.py -f rag_db_s16_d32_v64.bin --trace-insts-M 200 -q 200
```

Compress the trace:
```bash
xz -k ../../tracer/TRACES/TOY_RAG/champsim_0_0.trace
```

---

## 6. Running with ChampSim Simulator (NDP Offload)

The memory-heavy graph traversal phase (`rag_graph_retrieval`) is registered as an offload function:

```bash
# NDP CPU Offload (Asymmetric Memory-Heavy CXL Core)
../../NUMA_Csim/bin/1C_1CXL_ndp_memheavy \
  --warmup_instructions 10000000 \
  --simulation_instructions 50000000 \
  --page_owner_file page_owner_pre.txt \
  --offload_funcs rag_graph_retrieval:cxl0 \
  ../../tracer/TRACES/TOY_RAG/champsim_0_0.trace.xz
```

### Baseline (Host-Only Execution):
```bash
../../NUMA_Csim/bin/1C_1S \
  --warmup_instructions 10000000 \
  --simulation_instructions 50000000 \
  --page_owner_file page_owner_pre.txt \
  ../../tracer/TRACES/TOY_RAG/champsim_0_0.trace.xz
```

