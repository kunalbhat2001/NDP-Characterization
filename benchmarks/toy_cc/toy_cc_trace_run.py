#!/usr/bin/env python3
"""
toy_cc_trace_run.py – Run Pin tracer with NDP function offload on toy_cc (Connected Components).
"""

import os
import sys
import argparse

def main():
    ap = argparse.ArgumentParser(description="Trace toy_cc with StarNUMA pin tracer")
    ap.add_argument("--scale", type=int, default=14, help="Graph scale: 2^scale nodes (default 14)")
    ap.add_argument("--degree", type=int, default=16, help="Average node degree (default 16)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--graph-file", "-f", type=str, default=None, help="Pre-generated graph .bin")
    ap.add_argument("--rounds", "-r", type=int, default=2, help="Neighbor rounds (default 2)")
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

    target_dir = os.path.join(starnuma_path, 'tracer', 'TRACES', 'TOY_CC')
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)

    app_path = os.path.abspath(os.path.join(starnuma_path, 'benchmarks', 'toy_cc', 'toy_cc'))
    if not os.path.exists(app_path):
        sys.exit(f"ERROR: toy_cc binary not found at {app_path}. Run make -C benchmarks/toy_cc first.")

    if args.graph_file:
        prog_cmd = f"{app_path} -f {os.path.abspath(args.graph_file)} -t {args.threads} -r {args.rounds}"
    else:
        prog_cmd = f"{app_path} {args.scale} {args.degree} {args.seed} {args.threads} {args.rounds}"

    trace_insts = args.trace_insts_M * 1000000

    # Offload cc_link_step and cc_compress_step to NDP CPU
    cmd = f"{pinpath} -t {tracerpath} -startFF -offload_funcs cc_link_step,cc_compress_step -t {trace_insts} -- {prog_cmd}"
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"Pin tracer exited with code {ret}")

    print("\nTracing complete.")
    print(f"Traces written to: {target_dir}")

if __name__ == '__main__':
    main()
