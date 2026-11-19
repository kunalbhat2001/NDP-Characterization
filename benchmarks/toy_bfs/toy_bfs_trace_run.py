#!/usr/bin/env python3
"""
toy_bfs_trace_run.py  –  Run toy_bfs under the StarNUMA Pin tracer.

Generates:
  champsim_0_<phase>.trace   – instruction trace (for timing simulation)
  memtrace_t0.out            – cache-filtered memory accesses (for profiling)
  mtrace_t0_<phase>.out      – per-phase raw memory accesses (for TB-CPU)

Environment variables required:
  SNUMA_PINPATH      – path to the 'pin' binary
  SNUMA_TRACER_PATH  – path to compiled combined_tracer.so
  STARNUMA_ARTIFACT_PATH – root of the StarNUMA artifact

Usage:
  export SNUMA_PINPATH=/path/to/pin
  export SNUMA_TRACER_PATH=/path/to/combined_tracer.so
  export STARNUMA_ARTIFACT_PATH=/nethome/kbhat43/Development/StarNUMA+NDP
  python3 toy_bfs_trace_run.py [--scale 16] [--degree 16] [--seed 42]
"""

import os
import sys
import argparse

def main():
    ap = argparse.ArgumentParser(description="Trace toy_bfs with StarNUMA pin tracer")
    ap.add_argument("--scale",  type=int, default=16,
                    help="Graph scale: 2^scale nodes (default 16 = 64K nodes)")
    ap.add_argument("--degree", type=int, default=16,
                    help="Average node degree (default 16)")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed for graph generation (default 42)")
    ap.add_argument("--graph-file", "-f", type=str, default=None,
                    help="Path to pre-generated binary graph file (.bin)")
    ap.add_argument("--trace-insts-M", type=int, default=150,
                    help="Instructions to trace per phase in millions (default 150)")
    ap.add_argument("--threads", "-t", type=int, default=None,
                    help="Number of threads (default: 1 or OMP_NUM_THREADS)")
    args = ap.parse_args()

    # ---- Resolve paths ----
    pinpath     = os.getenv("SNUMA_PINPATH")
    tracerpath  = os.getenv("SNUMA_TRACER_PATH")
    artifact    = os.getenv("STARNUMA_ARTIFACT_PATH")

    if not pinpath or pinpath == "YOU_NEED_TO_SET_IT_WITH_EXPORT":
        sys.exit("ERROR: SNUMA_PINPATH not set")
    if not tracerpath or tracerpath == "YOU_NEED_TO_SET_IT_WITH_EXPORT":
        sys.exit("ERROR: SNUMA_TRACER_PATH not set")
    if not artifact:
        artifact = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    toy_bfs_bin = os.path.join(artifact, "benchmarks", "toy_bfs", "toy_bfs")
    if not os.path.isfile(toy_bfs_bin):
        sys.exit(f"ERROR: toy_bfs binary not found at {toy_bfs_bin}\n"
                 f"       Run:  make -C {os.path.dirname(toy_bfs_bin)}")

    trace_dir = os.path.join(artifact, "tracer", "TRACES", "TOY_BFS")
    os.makedirs(trace_dir, exist_ok=True)
    os.chdir(trace_dir)

    trace_insts = args.trace_insts_M * 1_000_000

    thread_opt = f" -t {args.threads}" if args.threads else ""
    if args.graph_file:
        prog_cmd = f"{toy_bfs_bin} -f {os.path.abspath(args.graph_file)}{thread_opt}"
    else:
        prog_cmd = f"{toy_bfs_bin} {args.scale} {args.degree} {args.seed}{thread_opt}"

    # -startFF  → only trace between roi_start() / roi_end() magic instructions
    # -t N      → trace up to N instructions per phase
    cmd = (f"{pinpath} -t {tracerpath} -startFF -t {trace_insts} -- {prog_cmd}")

    print("=" * 60)
    print(f"  Scale   : {args.scale}  ({1 << args.scale} nodes)")
    print(f"  Degree  : {args.degree}")
    print(f"  Seed    : {args.seed}")
    if args.threads:
        print(f"  Threads : {args.threads}")
    print(f"  Trace   : {args.trace_insts_M}M insts/phase")
    print(f"  Outdir  : {trace_dir}")
    print("=" * 60)
    print(f"\nCMD:\n  {cmd}\n")

    with open(f"trace_insts{args.trace_insts_M}M.txt", "w") as f:
        f.write(f"trace {args.trace_insts_M}M insts per phase\n")
        f.write(f"scale={args.scale} degree={args.degree} seed={args.seed}\n")

    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"Pin tracer exited with code {ret}")

    print("\nTracing complete.")
    print(f"Traces written to: {trace_dir}")

    # ---- Auto-generate function symbols for ChampSim ----
    sym_script = os.path.join(artifact, "tracer", "generate_symbols.py")
    if os.path.isfile(sym_script):
        numa_csim_dir = os.path.join(artifact, "NUMA_Csim")
        exp_dir = os.path.join(artifact, "experiments", "toy_bfs_1C1S")
        sym_cmd = f"python3 {sym_script} {toy_bfs_bin} -f bfs,bu_step,td_step -c {numa_csim_dir} {exp_dir} {trace_dir}"
        print(f"\n[*] Generating function symbols for ChampSim:")
        os.system(sym_cmd)

    # ---- Auto-generate page maps (page_owner_pre.txt) ----
    pagemap_script = os.path.join(artifact, "tracer", "gen_pagemap.py")
    memtrace_file = os.path.join(trace_dir, "memtrace_t0.out")
    if os.path.isfile(pagemap_script) and os.path.isfile(memtrace_file):
        numa_csim_dir = os.path.join(artifact, "NUMA_Csim")
        pagemap_cmd = f"python3 {pagemap_script} -m {memtrace_file} -o {os.path.join(trace_dir, 'page_owner_pre.txt')} -c {numa_csim_dir}"
        print(f"\n[*] Generating virtual page mappings for ChampSim:")
        os.system(pagemap_cmd)

    print("\nNext step: xz-compress instruction traces before running the simulator:")
    print("  xz -k champsim_0_*.trace")

if __name__ == "__main__":
    main()

