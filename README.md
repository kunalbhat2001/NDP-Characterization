# NDP Characterization

Workload characterization for host-CPU vs. CXL near-data-processing (NDP) offload.

## Layout

- `benchmarks/toy_*` -- the microbenchmarks (BFS, CC, PR, SSSP, TC, Masstree, RAG, CXL stream). Build each with `make -C benchmarks/<name>`.
- `pintools/func_cache_sim/` -- custom per-function cache-hierarchy simulator Pin tool.
- `pin-3.30-98830-g1d7b601b3-gcc-linux/` -- Intel Pin 3.30 kit. **Not tracked**; fetch it with `scripts/get_pin.sh`.
- `scripts/` -- setup and build scripts.

## Setup

```bash
scripts/get_pin.sh          # download + checksum-verify Pin 3.30 into the repo root (--force to reinstall)
scripts/build_pintools.sh   # build every tool in pintools/ -> pintools/<name>/obj-intel64/<name>.so
```

`build_pintools.sh` accepts tool names (`scripts/build_pintools.sh func_cache_sim`), `--clean`,
and the env vars `PIN_ROOT` (use a different Pin kit) and `JOBS`.

To add a tool, create `pintools/<name>/` containing `<name>.cpp`, Pin's standard `makefile`, and a
`makefile.rules` with `TEST_TOOL_ROOTS := <name>` (copy from `pintools/func_cache_sim/`).

## Datasets

The large graph inputs (`benchmarks/toy_bfs/graph*`) and RAG database (`*.bin`) are git-ignored. Regenerate them with `gen_graph`, `gen_graph_large` and `gen_rag_db`.
