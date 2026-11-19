#!/usr/bin/env python3
"""
toy_cxl_stream_trace_run.py – Run toy_cxl_stream under Intel Pin tracer and prepare StarNUMA simulation inputs.

Generates:
  champsim_0_0.trace.xz      – instruction trace (for ChampSim timing simulation)
  memtrace_t0.out            – binary memory access log (virtual addresses)
  page_owner_pre.txt         – page allocation map (mapped to CXL pool = 100)
  page_owner_post.txt        – post-warmup page allocation map (mapped to CXL pool = 100)

Usage:
  python3 toy_cxl_stream_trace_run.py [--size-mb 256] [--kernel read] [--trace-insts-M 20]
"""

import os
import sys
import argparse
import subprocess
import shutil

def main():
    parser = argparse.ArgumentParser(description="Trace toy_cxl_stream with StarNUMA Pin tracer")
    parser.add_argument("--size-mb", type=int, default=256,
                        help="Working set size in MB (default: 256 MB)")
    parser.add_argument("--kernel", type=str, default="read",
                        choices=["read", "write", "copy", "triad"],
                        help="Streaming kernel to execute (default: read)")
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="Number of OpenMP threads (default: 1)")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of streaming iterations inside ROI (default: 1)")
    parser.add_argument("--trace-insts-M", type=int, default=20,
                        help="Instructions to trace in millions (default: 20)")
    parser.add_argument("--dest-node", type=int, default=100,
                        help="Destination node for pages (100 = CXL pool, 0 = Local DRAM, default: 100)")
    args = parser.parse_args()

    # Resolve paths
    artifact = os.getenv("STARNUMA_ARTIFACT_PATH")
    if not artifact:
        artifact = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    pinpath = os.getenv("SNUMA_PINPATH")
    if not pinpath or pinpath == "YOU_NEED_TO_SET_IT_WITH_EXPORT":
        candidate = os.path.join(artifact, "pin-3.30-98830-g1d7b601b3-gcc-linux", "pin")
        if os.path.isfile(candidate):
            pinpath = candidate
        else:
            sys.exit(f"ERROR: SNUMA_PINPATH not set and pin binary not found at {candidate}")

    tracerpath = os.getenv("SNUMA_TRACER_PATH")
    if not tracerpath or tracerpath == "YOU_NEED_TO_SET_IT_WITH_EXPORT":
        candidate = os.path.join(artifact, "tracer", "pin_tracer", "obj-intel64", "combined_tracer.so")
        if os.path.isfile(candidate):
            tracerpath = candidate
        else:
            sys.exit(f"ERROR: SNUMA_TRACER_PATH not set and tracer not found at {candidate}")

    # Build toy_cxl_stream if needed
    bench_dir = os.path.join(artifact, "benchmarks", "toy_cxl_stream")
    bench_bin = os.path.join(bench_dir, "toy_cxl_stream")
    if not os.path.isfile(bench_bin):
        print(f"[*] Building {bench_bin}...")
        subprocess.check_call(["make", "-C", bench_dir])

    trace_dir = os.path.join(artifact, "tracer", "TRACES", "TOY_CXL_STREAM")
    os.makedirs(trace_dir, exist_ok=True)
    os.chdir(trace_dir)

    trace_insts = args.trace_insts_M * 1_000_000
    prog_cmd = f"{bench_bin} --size-mb {args.size_mb} --kernel {args.kernel} --threads {args.threads} --iters {args.iters}"

    # Intel Pin command:
    # -startFF : wait for roi_start() magic instruction before tracing
    # -t N     : trace up to N instructions
    pin_cmd = f"{pinpath} -t {tracerpath} -startFF -t {trace_insts} -- {prog_cmd}"

    print("=" * 60)
    print("  StarNUMA Pin Tracing: toy_cxl_stream")
    print("=" * 60)
    print(f"  Working Set : {args.size_mb} MB")
    print(f"  Kernel      : {args.kernel}")
    print(f"  Threads     : {args.threads}")
    print(f"  Trace Limit : {args.trace_insts_M}M instructions")
    print(f"  Output Dir  : {trace_dir}")
    print(f"  Page Dest   : {args.dest_node} ({'CXL Pool' if args.dest_node == 100 else 'Local DRAM'})")
    print("=" * 60)
    print(f"\n[Command]\n  {pin_cmd}\n")

    ret = os.system(pin_cmd)
    if ret != 0:
        sys.exit(f"ERROR: Pin tracer failed with exit code {ret}")

    print("\n[*] Pin tracing completed successfully.")

    # Compress trace file for ChampSim
    raw_trace = os.path.join(trace_dir, "champsim_0_0.trace")
    if os.path.isfile(raw_trace):
        print(f"[*] Compressing {raw_trace} with xz...")
        xz_cmd = f"xz -f -k {raw_trace}"
        ret = os.system(xz_cmd)
        if ret != 0:
            sys.exit(f"ERROR: xz compression failed with exit code {ret}")
    else:
        print(f"WARNING: {raw_trace} not found.")

    # Generate page mappings (assigning pages to CXL pool = 100)
    pagemap_script = os.path.join(artifact, "tracer", "gen_pagemap.py")
    memtrace_file = os.path.join(trace_dir, "memtrace_t0.out")
    if os.path.isfile(pagemap_script) and os.path.isfile(memtrace_file):
        numa_csim_dir = os.path.join(artifact, "NUMA_Csim")
        exp_dir = os.path.join(artifact, "experiments", "toy_cxl_stream")
        os.makedirs(exp_dir, exist_ok=True)
        pm_cmd = (f"python3 {pagemap_script} -m {memtrace_file} -d {args.dest_node} "
                  f"-o {trace_dir} -c {exp_dir} {numa_csim_dir}")
        print(f"\n[*] Generating virtual page map (dest={args.dest_node}) via gen_pagemap.py:")
        ret = os.system(pm_cmd)
        if ret != 0:
            sys.exit(f"ERROR: gen_pagemap failed with exit code {ret}")
    else:
        print(f"WARNING: Missing {pagemap_script} or {memtrace_file}.")

    print("\n" + "=" * 60)
    print("  Artifacts Ready for ChampSim / NUMA_Csim Simulation:")
    print(f"    Trace   : {os.path.join(trace_dir, 'champsim_0_0.trace.xz')}")
    print(f"    Pagemap : {os.path.join(trace_dir, 'page_owner_pre.txt')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
