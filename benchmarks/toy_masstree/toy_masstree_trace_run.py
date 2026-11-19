#!/usr/bin/env python3
"""
toy_masstree_trace_run.py – Run Pin tracer with NDP function offload on toy_masstree.
"""

import os
import sys
import argparse

def main():
    ap = argparse.ArgumentParser(description="Trace toy_masstree with StarNUMA pin tracer")
    ap.add_argument("--records", "-n", type=int, default=50000, help="Initial records inserted (default 50000)")
    ap.add_argument("--queries", "-q", type=int, default=20000, help="Total queries in ROI (default 20000)")
    ap.add_argument("--range-ratio", "-r", type=float, default=0.20, help="Fraction of range queries (default 0.20)")
    ap.add_argument("--scan-len", "-k", type=int, default=32, help="Range scan length (default 32)")
    ap.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--trace-insts-M", type=int, default=100, help="Million instructions to trace (default 100)")
    ap.add_argument("--threads", "-t", type=int, default=1, help="Thread count (default 1)")
    args = ap.parse_args()

    starnuma_path = os.getenv('STARNUMA_ARTIFACT_PATH', os.path.abspath('../..'))
    default_pin = os.path.join(starnuma_path, 'pin-3.30-98830-g1d7b601b3-gcc-linux', 'pin')
    if not os.path.exists(default_pin):
        default_pin = os.path.expanduser('~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin')
    pinpath = os.getenv('SNUMA_PINPATH', default_pin)

    tracerpath = os.getenv('SNUMA_TRACER_PATH', os.path.join(starnuma_path, 'tracer', 'pin_tracer', 'obj-intel64', 'combined_tracer.so'))

    if not os.path.exists(pinpath):
        sys.exit(f"ERROR: pin binary not found at {pinpath}")
    if not os.path.exists(tracerpath):
        sys.exit(f"ERROR: tracer not found at {tracerpath}")

    target_dir = os.path.join(starnuma_path, 'tracer', 'TRACES', 'TOY_MASSTREE')
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)

    app_path = os.path.abspath(os.path.join(starnuma_path, 'benchmarks', 'toy_masstree', 'toy_masstree'))
    if not os.path.exists(app_path):
        sys.exit(f"ERROR: toy_masstree binary not found at {app_path}. Run make -C benchmarks/toy_masstree first.")

    prog_cmd = (f"{app_path} -n {args.records} -q {args.queries} -r {args.range_ratio} "
                f"-k {args.scan_len} -t {args.threads} -s {args.seed}")

    trace_insts = args.trace_insts_M * 1000000

    # Offload pointer-chasing and memory-heavy kernels to NDP CPU
    offload_list = "masstree_trie_traverse,masstree_leaf_search,masstree_range_scan"
    cmd = f"{pinpath} -t {tracerpath} -startFF -offload_funcs {offload_list} -t {trace_insts} -- {prog_cmd}"
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"Pin tracer exited with code {ret}")

    print("\nTracing complete.")
    print(f"Traces written to: {target_dir}")

if __name__ == '__main__':
    main()
