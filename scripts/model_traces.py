#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture actual post-RoPE Q/K/V at the attention call, then diagnose bounds.

The last prefill position attends to the entire supplied prefix. Its query
heads share K/V through the model's actual grouped-query mapping. Diagnostic
probes are original generated text, not an accuracy benchmark. No remote model
code is executed. Large arrays are regenerated locally rather than committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmk.reference import bf16_cells, dense_attention, evaluate, summarize, tail_coefficient

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def probe_text(domain: str) -> str:
    """Deterministic original text with changing entities and arithmetic."""
    rows = []
    for i in range(800):
        if domain == "prose":
            rows.append(
                f"Field note {i}: the survey team reached station {i % 37} before noon. "
                f"They counted {11 + (i * 17) % 113} birds near the river and recorded "
                f"a water level of {20 + i % 19} centimeters. The next team must compare "
                "the measurement with the previous visit, explain any difference, "
                "and keep the original observation in the archive.\n"
            )
        elif domain == "code":
            rows.append(
                f"def transform_{i}(records):\n"
                f"    threshold = {i % 23 + 1}\n"
                "    total = 0\n"
                "    for key, value in records:\n"
                f"        if key % {i % 7 + 2} == 0 and value > threshold:\n"
                f"            total += value * {i % 11 + 1}\n"
                "    return total\n\n"
            )
        elif domain == "arithmetic":
            a, b = 7 + i % 29, 3 + (i * 13) % 31
            rows.append(
                f"Problem {i}. A store has {a} boxes with {b} pieces in each box. "
                f"It sells {i % a} boxes. Initially it has {a} times {b} = {a * b} pieces. "
                f"After the sale, {a - i % a} boxes remain, containing {(a - i % a) * b} pieces. "
                "Check the result by subtracting the sold pieces from the initial total.\n"
            )
        else:
            raise ValueError(domain)
    return "".join(rows)


def capture(args):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    torch.manual_seed(20260905)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = (
        AutoModel.from_pretrained(
            MODEL,
            revision=REVISION,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    config = model.config
    if config.model_type != "qwen2" or config.use_sliding_window:
        raise ValueError("This extractor requires Qwen2 full-prefix attention.")
    base = ALL_ATTENTION_FUNCTIONS["sdpa"]
    captured = {}

    def intercept(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is not None:
            last_mask = attention_mask[..., -1, : key.shape[-2]]
            visible = last_mask if last_mask.dtype == torch.bool else last_mask == 0
            if last_mask.shape[-1] != key.shape[-2] or not bool(visible.all().item()):
                raise ValueError(
                    "The final query must see every captured key; masked/windowed prefixes are unsupported"
                )
        result = base(module, query, key, value, attention_mask, **kwargs)
        captured[module.layer_idx] = {
            "queries": query[0, :, -1].detach().float().cpu().numpy(),
            "keys": key[0].detach().float().cpu().numpy(),
            "values": value[0].detach().float().cpu().numpy(),
            "kernel_output": result[0][0, -1].detach().float().cpu().numpy(),
            "scale": float(kwargs["scaling"]),
        }
        return result

    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "seed": 20260905,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "dtype": "bfloat16",
        "capture": "actual Q/K/V arguments after RoPE at Transformers SDPA call",
        "semantics": "last prefill position; all N keys visible; queries already scaled",
        "mask_check": "reject any non-visible key in a supplied effective final-row mask",
        "scope": "three deterministic original probe texts; no model accuracy benchmark",
        "source_sha256": {
            "scripts/model_traces.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
        "num_layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "traces": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = intercept
    try:
        for domain in args.domains:
            text = probe_text(domain)
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            for length in args.lengths:
                if len(tokens) < length:
                    raise ValueError("Probe text is shorter than the requested prefix")
                ids = torch.tensor([tokens[:length]], device="cuda")
                captured.clear()
                start = time.perf_counter()
                with torch.inference_mode():
                    model(input_ids=ids, use_cache=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                if set(captured) != set(range(config.num_hidden_layers)):
                    raise RuntimeError("Incomplete layer capture")
                keys = np.stack([captured[i]["keys"] for i in range(config.num_hidden_layers)])
                values = np.stack([captured[i]["values"] for i in range(config.num_hidden_layers)])
                queries = np.stack(
                    [
                        captured[i]["queries"] * captured[i]["scale"]
                        for i in range(config.num_hidden_layers)
                    ]
                )
                kernel_output = np.stack(
                    [captured[i]["kernel_output"] for i in range(config.num_hidden_layers)]
                )
                path = args.output / f"{domain}_{length}.npz"
                np.savez_compressed(
                    path,
                    keys=keys,
                    values=values,
                    queries=queries,
                    kernel_output=kernel_output,
                    token_ids=np.array(tokens[:length]),
                )
                name = f"{domain}_{length}"
                record = {
                    "name": name,
                    "file": path.name,
                    "domain": domain,
                    "N": length,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "token_sha256": hashlib.sha256(
                        np.array(tokens[:length], dtype="<i8").tobytes()
                    ).hexdigest(),
                    "array_sha256": {
                        k: hashlib.sha256(v.tobytes()).hexdigest()
                        for k, v in {"keys": keys, "values": values, "queries": queries}.items()
                    },
                    "capture_wall_seconds": seconds,
                }
                manifest["traces"].append(record)
                # Representative shared-KV query-head batch for the CUDA harness.
                group = config.num_attention_heads // config.num_key_value_heads
                for layer in [0, config.num_hidden_layers // 2, config.num_hidden_layers - 1]:
                    np.savez_compressed(
                        args.output / f"gpu_{name}_layer{layer}.npz",
                        keys=keys[layer, 0],
                        values=values[layer, 0],
                        queries=queries[layer, :group],
                    )
                print(json.dumps(record), flush=True)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = base
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def finite(x):
    x = float(x)
    return x if math.isfinite(x) else None


def diagnose(args):
    """CPU diagnostics on actual GPU activations; these are not GPU timings."""
    manifest = json.loads((args.output / "manifest.json").read_text())
    rows = []
    for trace in manifest["traces"]:
        with np.load(args.output / trace["file"], allow_pickle=False) as data:
            keys, values, queries = (
                data[k].astype(np.float64) for k in ["keys", "values", "queries"]
            )
            kernel_output = data["kernel_output"]
        layers, kv_heads, n, d = keys.shape
        heads = queries.shape[1]
        group_size = heads // kv_heads
        groups = tuple(
            np.arange(start, min(start + args.block_size, n))
            for start in range(0, n, args.block_size)
        )
        for layer in range(layers):
            # Full rank isolates radius looseness on representative layers.
            ranks = sorted(
                set([args.rank] + ([d] if layer in [0, layers // 2, layers - 1] else []))
            )
            for kv_head in range(kv_heads):
                k, v = keys[layer, kv_head], values[layer, kv_head]
                for rank in ranks:
                    start = time.perf_counter()
                    s = summarize(k, v, groups, rank=rank)
                    setup_ms = 1000 * (time.perf_counter() - start)
                    for head in range(kv_head * group_size, (kv_head + 1) * group_size):
                        q = queries[layer, head]
                        dense = dense_attention(q, k, v)
                        true_lower, true_upper, rounded = bf16_cells(dense)
                        margin = np.minimum(dense - true_lower, true_upper - dense)
                        u = q[:rank]
                        rho = s.key_radius[:, :rank] @ np.abs(u)
                        eps = s.key_radius[:, rank:] @ np.abs(q[rank:])
                        variance = np.maximum(0, np.einsum("i,bij,j->b", u, s.cov, u))
                        true_rho = np.array(
                            [
                                np.max(np.abs((k[g, :rank] - s.mu[b, :rank]) @ u))
                                for b, g in enumerate(groups)
                            ]
                        )
                        true_eps = np.array(
                            [
                                np.max(np.abs((k[g, rank:] - s.mu[b, rank:]) @ q[rank:]))
                                for b, g in enumerate(groups)
                            ]
                        )
                        row = {
                            "trace": trace["name"],
                            "domain": trace["domain"],
                            "N": n,
                            "layer": layer,
                            "head": head,
                            "kv_head": kv_head,
                            "d": d,
                            "h": v.shape[1],
                            "rank": rank,
                            "block_size": args.block_size,
                            "summary_bytes": s.nbytes,
                            "kv_bytes_float64": k.nbytes + v.nbytes,
                            "kv_bytes_bf16": (k.size + v.size) * 2,
                            "setup_cpu_ms": setup_ms,
                            "rho_max": finite(rho.max()),
                            "rho_median": finite(np.median(rho)),
                            "epsilon_max": finite(eps.max()),
                            "epsilon_median": finite(np.median(eps)),
                            "actual_projected_radius_max": finite(true_rho.max()),
                            "actual_discarded_radius_max": finite(true_eps.max()),
                            "variance_max": finite(variance.max()),
                            "boundary_margin_min": finite(margin.min()),
                            "kernel_vs_real_numerical_max_abs": finite(
                                np.max(np.abs(kernel_output[layer, head] - dense))
                            ),
                            "kernel_vs_real_bf16_agree_coordinates": int(
                                np.sum(kernel_output[layer, head] == rounded)
                            ),
                        }
                        try:
                            e = evaluate(q, s)
                            lower, upper, _ = bf16_cells(e.candidate())
                            accepted = e.contains_cell(lower, upper)
                            false = accepted & ~((dense > lower) & (dense < upper))
                            if false.any():
                                raise AssertionError(
                                    "Numerical screen contradicted dense diagnostic"
                                )
                            w = s.count * np.exp(s.mu @ q - e.shift)
                            tau = np.array([tail_coefficient(float(r)) for r in rho]) * variance
                            projected_hi = w * (1 + variance / 2 + tau)
                            quadratic = np.sum(w[:, None] * s.eta * float(u @ u) / 2, axis=0)
                            taylor = np.sum(w[:, None] * tau[:, None] * s.value_radius, axis=0)
                            discarded = np.sum(
                                np.expm1(eps)[:, None] * projected_hi[:, None] * s.value_radius,
                                axis=0,
                            )
                            true_mass = np.exp(k @ q - e.shift).sum()
                            actual_lower, actual_upper = e.residual(dense)
                            lower_res, _ = e.residual(true_lower)
                            _, upper_res = e.residual(true_upper)
                            row.update(
                                status="evaluated",
                                accepted_coordinates=int(accepted.sum()),
                                full_output_pass=bool(accepted.all()),
                                false_numerical_passes=int(false.sum()),
                                candidate_max_abs_error=finite(
                                    np.max(np.abs(e.candidate() - dense))
                                ),
                                candidate_bf16_agree_coordinates=int(
                                    np.sum(bf16_cells(e.candidate())[2] == rounded)
                                ),
                                omitted_quadratic_max=finite(quadratic.max()),
                                taylor_remainder_max=finite(taylor.max()),
                                discarded_score_max=finite(discarded.max()),
                                omitted_quadratic_over_true_mass=finite(
                                    quadratic.max() / true_mass
                                ),
                                taylor_remainder_over_true_mass=finite(taylor.max() / true_mass),
                                discarded_score_over_true_mass=finite(discarded.max() / true_mass),
                                residual_width_over_true_mass=finite(
                                    np.max(actual_upper - actual_lower) / true_mass
                                ),
                                minimum_margin_over_residual_width=finite(
                                    margin.min() * true_mass / np.max(actual_upper - actual_lower)
                                ),
                                true_cell_lower_residual_min=finite(lower_res.min()),
                                true_cell_upper_residual_max=finite(upper_res.max()),
                            )
                        except (ValueError, OverflowError, FloatingPointError) as error:
                            row.update(
                                status="rejected_domain",
                                reason=str(error),
                                accepted_coordinates=0,
                                full_output_pass=False,
                                false_numerical_passes=0,
                            )
                        rows.append(row)
            print(f"diagnosed {trace['name']} layer {layer + 1}/{layers}", flush=True)
        # Save incrementally to retain failures and completed cases after interruption.
        report = {
            "status": "numerical diagnostics on actual model activations; no roundoff certificate",
            "source_manifest": "manifest.json",
            "rows": rows,
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__), ROOT / "cmk" / "reference.py"]
            },
        }
        (args.output / "diagnostics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "diagnose", "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model_traces")
    parser.add_argument("--domains", nargs="+", default=["prose", "code", "arithmetic"])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    if args.mode in ["capture", "all"]:
        capture(args)
    if args.mode in ["diagnose", "all"]:
        diagnose(args)


if __name__ == "__main__":
    main()
