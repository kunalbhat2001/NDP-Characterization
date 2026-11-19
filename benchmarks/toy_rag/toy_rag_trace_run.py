#!/usr/bin/env python3
"""
toy_rag_trace_run.py  –  Run toy_rag under the StarNUMA Pin tracer.

Generates:
  champsim_0_<phase>.trace   – instruction trace (for ChampSim timing simulation)
  memtrace_t0.out            – cache-filtered memory accesses (for profiling & page mapping)
  mtrace_t0_<phase>.out      – per-phase raw memory accesses (for TB-CPU contention)

Environment variables required:
  SNUMA_PINPATH          – path to the 'pin' binary
  SNUMA_TRACER_PATH      – path to compiled combined_tracer.so
  STARNUMA_ARTIFACT_PATH – root of the StarNUMA artifact

Usage:
  export SNUMA_PINPATH=/path/to/pin
  export SNUMA_TRACER_PATH=/path/to/combined_tracer.so
  export STARNUMA_ARTIFACT_PATH=/nethome/kbhat43/Development/StarNUMA+NDP
  python3 toy_rag_trace_run.py [-f rag_db_s16_d32_v64.bin] [--trace-insts-M 200]
"""

import os
import sys
import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser(description="Trace toy_rag with StarNUMA Pin tracer")
    ap.add_argument("--graph-file", "-f", type=str, default=None,
                    help="Path to pre-generated binary RAG database file (.bin)")
    ap.add_argument("--scale",  type=int, default=16,
                    help="Graph scale: 2^scale nodes (default 16 = 64K nodes)")
    ap.add_argument("--degree", type=int, default=32,
                    help="Average node degree (default 32)")
    ap.add_argument("--dim",    type=int, default=64,
                    help="Vector embedding dimension (default 64)")
    ap.add_argument("--queries", "-q", type=int, default=100,
                    help="Number of RAG query iterations (default 100)")
    ap.add_argument("--topk",    "-k", type=int, default=8,
                    help="Number of retrieved nearest chunks (default 8)")
    ap.add_argument("--ef",      type=int, default=32,
                    help="Beam search window size (default 32)")
    ap.add_argument("--trace-insts-M", type=int, default=200,
                    help="Instructions to trace per phase in millions (default 200)")
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

    toy_rag_bin = os.path.join(artifact, "benchmarks", "toy_rag", "toy_rag")
    if not os.path.isfile(toy_rag_bin):
        sys.exit(f"ERROR: toy_rag binary not found at {toy_rag_bin}\n"
                 f"       Run:  make -C {os.path.dirname(toy_rag_bin)}")

    trace_dir = os.path.join(artifact, "tracer", "TRACES", "TOY_RAG")
    os.makedirs(trace_dir, exist_ok=True)
    os.chdir(trace_dir)

    trace_insts = args.trace_insts_M * 1_000_000

    # Ensure pre-generated DB file is present if specified
    db_arg = ""
    if args.graph_file:
        db_path = os.path.abspath(args.graph_file)
        if not os.path.isfile(db_path):
            sys.exit(f"ERROR: Database file not found at: {db_path}")
        db_arg = f"-f {db_path}"
    else:
        # Check if default pre-generated binary file exists in benchmarks/toy_rag
        default_db = os.path.join(artifact, "benchmarks", "toy_rag", f"rag_db_s{args.scale}_d{args.degree}_v{args.dim}.bin")
        if os.path.isfile(default_db):
            db_arg = f"-f {default_db}"
        else:
            print(f"[*] No pre-generated DB provided; auto-generating {default_db} before tracing...")
            gen_bin = os.path.join(artifact, "benchmarks", "toy_rag", "gen_rag_db")
            if not os.path.isfile(gen_bin):
                subprocess.run(["make", "-C", os.path.dirname(gen_bin), "gen_rag_db"], check=True)
            gen_cmd = [gen_bin, default_db, str(args.scale), str(args.degree), str(args.dim), "42"]
            subprocess.run(gen_cmd, check=True)
            db_arg = f"-f {default_db}"

    prog_cmd = f"{toy_rag_bin} {db_arg} -q {args.queries} -k {args.topk} --ef {args.ef}"

    # -startFF  → only trace between roi_start() / roi_end() magic instructions
    # -t N      → trace up to N instructions per phase
    cmd = f"{pinpath} -t {tracerpath} -startFF -t {trace_insts} -- {prog_cmd}"

    print("=" * 60)
    print(" StarNUMA Pin Tracer: Toy RAG Workload")
    print("=" * 60)
    print(f"  DB File : {db_arg}")
    print(f"  Queries : {args.queries}")
    print(f"  Top-K   : {args.topk}")
    print(f"  Beam EF : {args.ef}")
    print(f"  Trace   : {args.trace_insts_M}M instructions")
    print(f"  Outdir  : {trace_dir}")
    print("=" * 60)
    print(f"\nCMD:\n  {cmd}\n")

    with open(f"trace_insts{args.trace_insts_M}M.txt", "w") as f:
        f.write(f"trace {args.trace_insts_M}M insts per phase\n")
        f.write(f"scale={args.scale} degree={args.degree} dim={args.dim} queries={args.queries}\n")

    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"Pin tracer exited with code {ret}")

    print("\nTracing complete.")
    print(f"Traces written to: {trace_dir}")

    # ---- Auto-generate function symbols for ChampSim ----
    sym_script = os.path.join(artifact, "tracer", "generate_symbols.py")
    if os.path.isfile(sym_script):
        numa_csim_dir = os.path.join(artifact, "NUMA_Csim")
        exp_dir = os.path.join(artifact, "experiments", "toy_rag_1C1S")
        os.makedirs(exp_dir, exist_ok=True)
        sym_cmd = (f"python3 {sym_script} {toy_rag_bin} "
                   f"-f rag_encode_query,rag_graph_retrieval,rag_llm_generation "
                   f"-c {numa_csim_dir} {exp_dir} {trace_dir}")
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

