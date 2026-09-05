#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GH200 research benchmark; numerical screens are NOT interval certificates.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. The shared library is built
by CMake's CMK_ENABLE_CUDA target. All reported fallback and setup costs are
measured; a CPU decision synchronizes the current stream on every invocation.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmk.reference import bf16_cells, evaluate, summarize

NAMES = ("count", "mu", "nu", "cov", "cross", "diagonal", "eta", "key_radius", "value_radius")


class SummaryView(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("B", "d", "r", "h")] + [(n, C.c_void_p) for n in NAMES]


def timing(fn, repeats, inner=10, warmup=3):
    """CUDA events include launches; host samples include Python and sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        b.synchronize()
        wall.append((time.perf_counter() - start) * 1000 / inner)
        gpu.append(a.elapsed_time(b) / inner)
    return dict(
        gpu_ms=float(np.median(gpu)),
        wall_ms=float(np.median(wall)),
        gpu_samples_ms=gpu,
        wall_samples_ms=wall,
        inner=inner,
        repeats=repeats,
    )


def quantize(a):
    return torch.from_numpy(np.asarray(a)).to(torch.bfloat16).to(torch.float64).numpy()


def graph_timing(fn, repeats, inner):
    """Fixed-input replay microbenchmark; no adaptive scheduler is captured."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    start = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - start) * 1000
    measured = timing(graph.replay, repeats, inner=1)
    for key in ("gpu_ms", "wall_ms"):
        measured[key] /= inner
    for key in ("gpu_samples_ms", "wall_samples_ms"):
        measured[key] = [x / inner for x in measured[key]]
    measured.update(
        inner=inner,
        capture_ms=capture_ms,
        scope="fixed input graph; setup and fallback excluded from replay",
    )
    return measured


def synthetic(n, d, h, Q, profile, seed):
    rng = np.random.default_rng(seed)
    groups = np.array_split(np.arange(n), max(1, n // 256))
    centers = quantize(rng.normal(size=(len(groups), d)) * 0.3)
    active = min(4, d)
    spread, discarded = {
        "tight": (0.003, 0.000001),
        "moderate": (0.03, 0.0001),
        "broad": (0.3, 0.3),
        "centered": (0.003, 0.000001),
        "midpoint": (0.003, 0.000001),
    }[profile]
    radius = np.r_[np.full(active, spread), np.full(d - active, discarded)]
    k = quantize(
        np.concatenate(
            [centers[b] + rng.normal(size=(len(g), d)) * radius for b, g in enumerate(groups)]
        )
    )
    v = quantize(rng.normal(size=(n, h)) * 0.3 + (0 if profile == "centered" else 1))
    q = quantize(rng.normal(size=(Q, d)) / np.sqrt(d))
    if profile == "midpoint":
        # Every exact output equals the BF16 midpoint 257/256. Both inputs are BF16.
        q[:] = 0
        v[::2] = 1
        v[1::2] = 1 + 1 / 128
    return (
        k,
        v,
        q,
        groups,
        dict(
            active_spread=spread,
            discarded_spread=discarded,
            input_values="BF16 represented exactly in binary64",
        ),
    )


def make_lib(path):
    lib = C.CDLL(str(path))
    for name in ("cmk_run", "cmk_phase"):
        fn = getattr(lib, name)
        fn.argtypes = [
            C.POINTER(SummaryView),
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
        ]
        fn.restype = C.c_int
    return lib


def run_case(lib, spec, args):
    seed = args.seed + spec.get("seed_offset", 0)
    if "trace" in spec:
        with np.load(spec["trace"]) as z:
            k, v, q = [
                np.asarray(z[name], dtype=np.float64) for name in ("keys", "values", "queries")
            ]
        if q.ndim == 1:
            q = q[None]
        groups = np.array_split(np.arange(len(k)), max(1, math.ceil(len(k) / 256)))
        metadata = {
            "input_values": "supplied post-transform trace values; BF16 baseline quantizes if needed"
        }
    else:
        k, v, q, groups, metadata = synthetic(
            spec["N"], spec["d"], spec["h"], spec["Q"], spec["profile"], seed
        )
    n, d = k.shape
    h, Q, rank = v.shape[1], len(q), spec["rank"]
    if not all(np.isfinite(x).all() for x in (k, v, q)):
        raise ValueError("Nonfinite trace/input")
    start = time.perf_counter()
    summary = summarize(k, v, groups, rank=rank)
    summary_cpu_ms = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()
    start = time.perf_counter()
    dev = {
        name: torch.from_numpy(np.array(getattr(summary, name), copy=True, order="C")).cuda()
        for name in NAMES
    }
    torch.cuda.synchronize()
    summary_upload_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    kd, vd, qd = [torch.from_numpy(x.copy()).cuda() for x in (k, v, q)]
    # The exact same BF16 input values are used by the synthetic baseline.
    kb, vb, qb = [x.to(torch.bfloat16)[None, None] for x in (kd, vd, qd)]
    torch.cuda.synchronize()
    inputs_upload_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    B = len(groups)
    shifts = torch.empty(Q, dtype=torch.float64, device="cuda")
    boxes = torch.empty((Q, B, h, 6), dtype=torch.float64, device="cuda")
    out = torch.empty((Q, h, 7), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    workspace_allocation_ms = (time.perf_counter() - start) * 1000
    view = SummaryView(B, d, rank, h, *(dev[name].data_ptr() for name in NAMES))
    common = (C.byref(view), qd.data_ptr(), Q, shifts.data_ptr(), boxes.data_ptr(), out.data_ptr())

    def launch(shared=2):
        code = lib.cmk_run(*common, torch.cuda.current_stream().cuda_stream, shared)
        if code:
            raise RuntimeError(f"CUDA cmk_run failed with error {code}")

    def phase(p):
        code = lib.cmk_phase(*common, torch.cuda.current_stream().cuda_stream, p)
        if code:
            raise RuntimeError(f"CUDA cmk_phase failed with error {code}")

    def dense64():
        return torch.softmax(qd @ kd.T, dim=-1) @ vd

    def flash():
        return F.scaled_dot_product_attention(qb, kb, vb, dropout_p=0.0, is_causal=False, scale=1.0)

    launch(0)
    original = out.cpu().numpy().copy()
    launch(1)
    shared = out.cpu().numpy().copy()
    if not np.array_equal(original[..., 6], shared[..., 6]):
        raise AssertionError("Shared scalar ablation changed a screening decision")
    scalar_difference = float(np.nanmax(np.abs(original[..., :6] - shared[..., :6])))
    if scalar_difference > 1e-10:
        raise AssertionError(f"Shared scalar mismatch: {scalar_difference}")
    launch(2)
    parallel = out.cpu().numpy().copy()
    if not np.array_equal(shared[..., 6], parallel[..., 6]):
        raise AssertionError("Parallel reduction changed a screening decision")
    parallel_difference = float(np.nanmax(np.abs(parallel[..., :6] - shared[..., :6])))
    parallel_scaled_difference = float(
        np.nanmax(np.abs(parallel[..., :6] - shared[..., :6]) / (1 + np.abs(shared[..., :6])))
    )
    if parallel_scaled_difference > 1e-12:
        raise AssertionError(f"Parallel reduction scaled mismatch: {parallel_scaled_difference}")
    shared = parallel
    exact_approx = dense64().cpu().numpy()
    # Oracle is ordinary binary64 attention, not an exact-real interval oracle.
    accepted = shared[..., 6] > 0.5
    inside = (shared[..., 2] < exact_approx) & (exact_approx < shared[..., 3])
    false_numeric = int(np.count_nonzero(accepted & ~inside))
    oracle_bf16 = np.array([bf16_cells(row)[2] for row in exact_approx])
    chosen = shared[..., 1]
    accepted_mismatch = int(np.count_nonzero(accepted & (chosen != oracle_bf16)))
    if false_numeric or accepted_mismatch:
        raise AssertionError(
            f"Numerical false screen vs binary64 oracle: {false_numeric}, {accepted_mismatch}"
        )
    # Compare the independent CPU evaluator once; this is untimed validation.
    cpu_error = 0.0
    for qi in range(min(Q, 3)):
        e = evaluate(q[qi], summary)
        cpu_error = max(cpu_error, float(np.max(np.abs(e.candidate() - shared[qi, :, 0]))))
    if cpu_error > 1e-10:
        raise AssertionError(f"CPU/GPU candidate mismatch: {cpu_error}")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        flash_output = flash().double()[0, 0].cpu().numpy()

        # The full batch falls back if any output coordinate fails. This simple
        # scheduler deliberately includes its CPU synchronization cost.
        def with_fallback():
            launch(2)
            if bool((out[..., 6] > 0.5).all().item()):
                return out[..., 1].to(torch.bfloat16).reshape(1, 1, Q, h)
            return flash()

        timings = {
            "dense_bf16_flash": timing(flash, args.repeats, args.inner),
            "dense_binary64": timing(dense64, args.repeats, args.inner),
            "screen_original": timing(lambda: launch(0), args.repeats, args.inner),
            "screen_shared": timing(lambda: launch(1), args.repeats, args.inner),
            "screen_parallel": timing(lambda: launch(2), args.repeats, args.inner),
            "screen_and_batch_fallback": timing(with_fallback, args.repeats, args.inner),
        }
        for name, p in [
            ("shift", 0),
            ("shared_evaluate", 1),
            ("reduce_screen", 2),
            ("original_evaluate", 3),
            ("parallel_reduce_screen", 4),
        ]:
            timings[name] = timing(lambda p=p: phase(p), args.repeats, args.inner)
        graph_timings = dict(
            dense_bf16_flash=graph_timing(flash, args.repeats, args.inner),
            screen_shared=graph_timing(lambda: launch(1), args.repeats, args.inner),
            screen_parallel=graph_timing(lambda: launch(2), args.repeats, args.inner),
            screen_original=graph_timing(lambda: launch(0), args.repeats, args.inner),
        )
    candidate_error = shared[..., 0] - exact_approx
    setup_ms = summary_cpu_ms + summary_upload_ms + workspace_allocation_ms
    total_screen = timings["screen_and_batch_fallback"]["wall_ms"]
    flash_ms = timings["dense_bf16_flash"]["wall_ms"]
    saved = flash_ms - total_screen
    accepted_error = np.abs(chosen - exact_approx)[accepted]
    pass_rows = accepted.all(axis=1)
    per_query = [
        dict(
            numerical_pass=bool(pass_rows[i]),
            accepted_coordinates=int(accepted[i].sum()),
            candidate_max_abs_error=float(np.max(np.abs(candidate_error[i]))),
            min_lower_residual=float(np.min(shared[i, :, 4])),
            max_upper_residual=float(np.max(shared[i, :, 5])),
        )
        for i in range(Q)
    ]
    return spec | dict(
        seed=seed,
        N=n,
        d=d,
        h=h,
        Q=Q,
        B=B,
        metadata=metadata,
        status="ok",
        arithmetic="binary64 numerical; not outward certified",
        mask="all and only supplied keys visible; queries already scaled",
        baseline="PyTorch forced FLASH_ATTENTION, BF16, scale=1, noncausal, no dropout",
        query_layout="single decode query" if Q == 1 else "shared-KV noncausal query batch",
        summary_bytes=summary.nbytes,
        gpu_summary_bytes=sum(t.numel() * t.element_size() for t in dev.values()),
        kv_binary64_bytes=k.nbytes + v.nbytes,
        kv_bf16_bytes=(k.size + v.size) * 2,
        workspace_bytes=(shifts.numel() + boxes.numel() + out.numel()) * 8,
        setup=dict(
            summary_cpu_ms=summary_cpu_ms,
            summary_upload_ms=summary_upload_ms,
            input_upload_and_cast_ms=inputs_upload_ms,
            workspace_allocation_ms=workspace_allocation_ms,
            reusable_summary_and_workspace_ms=setup_ms,
        ),
        numerical_whole_output_passes=int(pass_rows.sum()),
        numerical_coordinate_passes=int(accepted.sum()),
        coordinates=Q * h,
        whole_batch_fallback=not bool(accepted.all()),
        numerical_false_screens_vs_binary64=false_numeric,
        accepted_bf16_mismatches_vs_binary64=accepted_mismatch,
        accepted_bf16_mismatches_vs_flash=int(
            np.count_nonzero(accepted & (chosen != flash_output))
        ),
        candidate_max_abs_error=float(np.max(np.abs(candidate_error))),
        candidate_rmse=float(np.sqrt(np.mean(candidate_error**2))),
        accepted_rounded_max_abs_error=float(accepted_error.max()) if accepted_error.size else None,
        flash_max_abs_error_vs_binary64=float(np.max(np.abs(flash_output - exact_approx))),
        shared_scalar_max_abs_difference=scalar_difference,
        parallel_reduce_max_abs_difference=parallel_difference,
        cpu_gpu_candidate_max_abs_difference=cpu_error,
        parallel_reduce_max_scaled_difference=parallel_scaled_difference,
        timings=timings,
        graph_timings=graph_timings,
        per_query=per_query,
        query_speed_ratio_wall=flash_ms / total_screen,
        query_speed_ratio_device=timings["dense_bf16_flash"]["gpu_ms"]
        / timings["screen_parallel"]["gpu_ms"],
        break_even_reuses_vs_flash=math.ceil(setup_ms / saved) if saved > 0 else None,
        total_ms_by_reuse={
            str(reuse): setup_ms + reuse * total_screen for reuse in (1, 32, 1024, 32768)
        },
        dense_flash_ms_by_reuse={str(reuse): reuse * flash_ms for reuse in (1, 32, 1024, 32768)},
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, default=ROOT / "build/gh200/libcmk_benchmark.so")
    ap.add_argument("--output", type=Path, default=ROOT / "results/gh200/benchmark.json")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--seed", type=int, default=902105)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trace", type=Path, action="append", default=[])
    ap.add_argument("--trace-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lib = make_lib(args.library)
    specs = []
    if not args.trace_only:
        for n in [1024] if args.quick else [1024, 8192, 32768]:
            for Q in [8] if args.quick else [1, 32]:
                for rank in [4] if args.quick else [0, 4, 16]:
                    for profile in (
                        ["tight", "broad"] if args.quick else ["tight", "moderate", "broad"]
                    ):
                        specs.append(dict(N=n, d=64, h=64, Q=Q, rank=rank, profile=profile))
        if not args.quick:
            for profile in ["centered", "midpoint"]:
                specs.append(dict(N=8192, d=64, h=64, Q=32, rank=4, profile=profile))
            specs.append(dict(N=1024, d=64, h=64, Q=32, rank=64, profile="tight"))
    for path in args.trace:
        for rank in [0, 4, 16]:
            specs.append(dict(trace=str(path), profile="trained_trace", rank=rank))
    env = dict(
        python=sys.version,
        numpy=np.__version__,
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        platform=platform.platform(),
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        omp_threads=os.environ.get("OMP_NUM_THREADS"),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True),
        nvidia_smi=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            text=True,
        ),
    )
    result = dict(
        status="actual GPU measurements; numerical gates are not interval certificates",
        command=sys.argv,
        environment=env,
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(specs):
        print(f"[{i + 1}/{len(specs)}] {spec}", flush=True)
        try:
            row = run_case(lib, spec, args)
        except Exception as exc:
            row = dict(**spec, status="failed", error=str(exc), traceback=traceback.format_exc())
            print(row["traceback"], flush=True)
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        torch.cuda.empty_cache()
    print(args.output)
    return int(any(c["status"] != "ok" for c in result["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
