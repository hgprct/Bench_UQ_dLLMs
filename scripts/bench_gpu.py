#!/usr/bin/env python3
"""GPU benchmark for node performance comparison.

Subcommands:
  bench   -- run benchmarks on this node (must run inside container with a GPU)
  report  -- read all benchmark JSONs and print a ranked table with recommendations
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# bench subcommand
# ---------------------------------------------------------------------------

def _cuda_timed(fn, n_warmup=5, n_iters=20):
    import torch
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0, n_iters  # seconds, count


def _bench_matmul_bf16(device):
    import torch
    M = K = N = 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    elapsed_s, n = _cuda_timed(lambda: torch.mm(a, b))
    tflops = 2 * M * K * N * n / elapsed_s / 1e12
    return round(tflops, 2)


def _bench_hbm_bandwidth(device):
    """Clone a 2 GB float32 tensor: 1 read + 1 write = 2 passes over HBM."""
    import torch
    n_bytes = 2 * 1024 ** 3
    x = torch.empty(n_bytes // 4, dtype=torch.float32, device=device)
    elapsed_s, n = _cuda_timed(lambda: x.clone(), n_warmup=3, n_iters=10)
    gbs = 2 * n_bytes * n / elapsed_s / 1e9
    return round(gbs, 1)


def _bench_h2d(device):
    """Transfer 1 GB from pinned CPU memory to GPU."""
    import torch
    import time
    n_bytes = 1024 ** 3
    cpu = torch.empty(n_bytes // 4, dtype=torch.float32).pin_memory()
    gpu = torch.empty(n_bytes // 4, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    # warmup
    for _ in range(2):
        gpu.copy_(cpu)
        torch.cuda.synchronize()
    n_iters = 5
    t0 = time.perf_counter()
    for _ in range(n_iters):
        gpu.copy_(cpu)
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    return round(n_bytes * n_iters / elapsed_s / 1e9, 1)


def _nvidia_smi(gpu_index):
    fields = "temperature.gpu,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw,ecc.errors.uncorrected.volatile.total"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
        keys = ["temp_c", "sm_clock_mhz", "mem_clock_mhz", "pcie_gen", "pcie_width", "power_w", "ecc_errors"]
        vals = [v.strip() for v in out.split(",")]
        return dict(zip(keys, vals))
    except Exception:
        return {}


def cmd_bench(args):
    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    node = socket.gethostname()

    print(f"Benchmarking GPU {args.gpu} on {node}")
    print(f"  {props.name}  {round(props.total_memory / 1e9, 1)} GB")

    smi = _nvidia_smi(args.gpu)
    if smi:
        print(f"  Temp: {smi.get('temp_c','?')} C  SM: {smi.get('sm_clock_mhz','?')} MHz  "
              f"PCIe gen{smi.get('pcie_gen','?')} x{smi.get('pcie_width','?')}")

    print("  Running BF16 matmul ...", end=" ", flush=True)
    matmul = _bench_matmul_bf16(device)
    print(f"{matmul:.1f} TFLOPS")

    print("  Running HBM bandwidth ...", end=" ", flush=True)
    hbm_bw = _bench_hbm_bandwidth(device)
    print(f"{hbm_bw:.0f} GB/s")

    print("  Running H2D bandwidth ...", end=" ", flush=True)
    h2d_bw = _bench_h2d(device)
    print(f"{h2d_bw:.0f} GB/s")

    result = {
        "node": node,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu_index": args.gpu,
        "gpu_name": props.name,
        "gpu_memory_gb": round(props.total_memory / 1e9, 1),
        **smi,
        "matmul_bf16_tflops": matmul,
        "hbm_bandwidth_gbs": hbm_bw,
        "h2d_bandwidth_gbs": h2d_bw,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {args.output}")


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------

def cmd_report(args):
    import glob

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if not paths:
        print(f"No JSON files found in {args.results_dir}")
        sys.exit(1)

    rows = []
    for p in paths:
        with open(p) as f:
            rows.append(json.load(f))

    best_matmul = max(r["matmul_bf16_tflops"] for r in rows)
    best_hbm = max(r["hbm_bandwidth_gbs"] for r in rows)
    best_h2d = max(r["h2d_bandwidth_gbs"] for r in rows)

    for r in rows:
        r["_score"] = round(100 * (
            0.50 * r["matmul_bf16_tflops"] / best_matmul +
            0.35 * r["hbm_bandwidth_gbs"] / best_hbm +
            0.15 * r["h2d_bandwidth_gbs"] / best_h2d
        ), 1)

    rows.sort(key=lambda r: r["_score"], reverse=True)

    W = [5, 12, 14, 11, 11, 7]
    header = ["Rank", "Node", "MatMul (TFLOPS)", "HBM (GB/s)", "H2D (GB/s)", "Score"]
    sep = "-" * (sum(W) + 3 * (len(W) - 1) + 4)

    print()
    print("=" * len(sep))
    print("  GPU Node Benchmark Report")
    print("=" * len(sep))
    print()
    fmt = "  {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}}"
    print(fmt.format(*[v for pair in zip(header, W) for v in pair]))
    print("  " + sep)
    for i, r in enumerate(rows, 1):
        tier = ""
        if r["_score"] >= 95:
            tier = " ✓"
        elif r["_score"] < 80:
            tier = " ✗"
        print(fmt.format(
            i, W[0],
            r["node"] + tier, W[1],
            f"{r['matmul_bf16_tflops']:.1f}", W[2],
            f"{r['hbm_bandwidth_gbs']:.0f}", W[3],
            f"{r['h2d_bandwidth_gbs']:.0f}", W[4],
            f"{r['_score']:.1f}", W[5],
        ))

    prefer = [r["node"] for r in rows if r["_score"] >= 95]
    avoid = [r["node"] for r in rows if r["_score"] < 80]

    print()
    print("RECOMMENDATIONS")
    print("-" * 40)
    if prefer:
        print(f"  Prefer  (≥95%):  {', '.join(prefer)}")
    if avoid:
        print(f"  Avoid   (<80%):  {', '.join(avoid)}")

    print()
    print("SLURM HINTS  (add to your #SBATCH headers)")
    print("-" * 40)
    if prefer:
        print(f"  Best nodes only:    #SBATCH --nodelist={','.join(prefer)}")
    if avoid:
        print(f"  Exclude slow nodes: #SBATCH --exclude={','.join(avoid)}")

    print()
    print("DIAGNOSTICS (from nvidia-smi at benchmark time)")
    print("-" * 40)
    diag_header = ["Node", "Temp(C)", "SM(MHz)", "Mem(MHz)", "PCIe", "Power(W)", "ECC errs"]
    diag_W = [12, 8, 8, 9, 7, 9, 9]
    diag_fmt = "  {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}}"
    print(diag_fmt.format(*[v for pair in zip(diag_header, diag_W) for v in pair]))
    print("  " + "-" * (sum(diag_W) + 3 * (len(diag_W) - 1) + 4))
    for r in rows:
        pcie = f"g{r.get('pcie_gen','?')}x{r.get('pcie_width','?')}"
        print(diag_fmt.format(
            r["node"], diag_W[0],
            r.get("temp_c", "?"), diag_W[1],
            r.get("sm_clock_mhz", "?"), diag_W[2],
            r.get("mem_clock_mhz", "?"), diag_W[3],
            pcie, diag_W[4],
            r.get("power_w", "?"), diag_W[5],
            r.get("ecc_errors", "?"), diag_W[6],
        ))
    print()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("bench", help="run benchmarks on this GPU node")
    p_bench.add_argument("--output", required=True, help="path to write JSON result")
    p_bench.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")

    p_report = sub.add_parser("report", help="print ranked table from saved JSON results")
    p_report.add_argument("--results_dir", required=True, help="directory containing *.json benchmark results")

    args = parser.parse_args()
    if args.cmd == "bench":
        cmd_bench(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
