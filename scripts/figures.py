#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate manuscript figures from recorded data and labeled analytic examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmk.reference import tail_coefficient

OUT = ROOT / "paper" / "figures"
COLORS = {
    "blue": "#2369a1",
    "orange": "#cc6534",
    "green": "#208578",
    "gray": "#71808d",
    "red": "#b33b54",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.35,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
MANIFEST = []
LAYOUT = []


def panels(height=2.8):
    """Draw at the manuscript's physical width so text is not scaled down."""
    fig, axes = plt.subplots(1, 2, figsize=(6.75, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.1)
    return fig, axes


def legend_below(axis, columns=1):
    return axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncols=columns,
        borderaxespad=0,
        handlelength=2,
    )


def shared_legend(fig, axis, columns=3):
    handles, labels = axis.get_legend_handles_labels()
    return fig.legend(handles, labels, loc="outside lower center", ncols=columns)


def context_ticks(axis, values):
    """Label the measured context lengths instead of unrelated log ticks."""
    axis.xaxis.set_major_locator(FixedLocator(values))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name, sources, explanation):
    OUT.mkdir(parents=True, exist_ok=True)
    plot_axes = [axis for axis in fig.axes if axis.get_label() != "<colorbar>"]
    for letter, axis in zip("ab", plot_axes):
        title = axis.get_title(loc="left")
        axis.set_title(f"({letter}) {title}", loc="left")
    # Outside legends depend on axes height. Let constrained layout settle,
    # then keep identical positions in all three export backends.
    for _ in range(3):
        fig.canvas.draw()
    fig.set_layout_engine(None)
    renderer = fig.canvas.get_renderer()
    legends = [*fig.legends, *(axis.get_legend() for axis in plot_axes if axis.get_legend())]
    overlaps = [
        {"legend": index, "panel": panel}
        for index, legend in enumerate(legends)
        for panel, axis in enumerate(plot_axes)
        if legend.get_window_extent(renderer).overlaps(axis.get_window_extent(renderer))
    ]
    if overlaps:
        raise ValueError(f"{name}: legend covers a data panel: {overlaps}")
    figure_bounds = fig.get_window_extent(renderer)
    for axis in fig.axes:
        bounds = axis.get_tightbbox(renderer)
        if bounds is not None and (
            bounds.x0 < figure_bounds.x0 - 1
            or bounds.y0 < figure_bounds.y0 - 1
            or bounds.x1 > figure_bounds.x1 + 1
            or bounds.y1 > figure_bounds.y1 + 1
        ):
            raise ValueError(
                f"{name}: axis decorations {bounds.bounds} extend beyond canvas {figure_bounds.bounds}"
            )
    LAYOUT.append(
        {
            "figure": name,
            "width_inches": float(fig.get_figwidth()),
            "height_inches": float(fig.get_figheight()),
            "legend_count": len(legends),
            "legend_data_panel_overlaps": overlaps,
            "axis_decorations_inside_canvas": True,
            "minimum_legend_font_points": min(
                (text.get_fontsize() for legend in legends for text in legend.get_texts()),
                default=None,
            ),
        }
    )
    for extension in ["svg", "pdf", "png"]:
        output = OUT / f"{name}.{extension}"
        fig.savefig(
            output,
            bbox_inches=None,
            metadata={"Creator": "scripts/figures.py"} if extension in ["svg", "pdf"] else None,
        )
        if extension == "svg":
            output.write_text(
                "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
            )
    LAYOUT[-1]["export_sha256"] = {
        f"paper/figures/{name}.{extension}": hashlib.sha256(
            (OUT / f"{name}.{extension}").read_bytes()
        ).hexdigest()
        for extension in ("svg", "pdf", "png")
    }
    MANIFEST.append(
        {
            "figure": name,
            "description": explanation,
            "sources": [
                {"path": s, "sha256": hashlib.sha256((ROOT / s).read_bytes()).hexdigest()}
                for s in sources
            ],
        }
    )
    plt.close(fig)


def analytic_figures():
    fig, ax = panels()
    c = np.linspace(0.25, 1.75, 250)
    lower = np.minimum(-c, -2 * c) + np.minimum(2 - c, 2 * (2 - c))
    upper = np.maximum(-c, -2 * c) + np.maximum(2 - c, 2 * (2 - c))
    ax[0].plot(c, lower, label=r"Lower residual $\mathcal{L}(c)$", color=COLORS["blue"])
    ax[0].plot(c, upper, label=r"Upper residual $\mathcal{U}(c)$", color=COLORS["orange"])
    ax[0].fill_between(c, lower, upper, alpha=0.09, color=COLORS["blue"])
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].scatter([0.6, 1.4], [0.2, -0.2], zorder=4, color=COLORS["green"])
    ax[0].axvline(0.6, color=COLORS["green"], ls=":")
    ax[0].axvline(1.4, color=COLORS["green"], ls=":")
    ax[0].set(
        xlabel="Boundary c",
        ylabel="Attainable residual N − cZ",
        title="Boundary residuals",
    )
    legend_below(ax[0])
    for y, (lo, hi), color in zip(
        [2, 1, 0],
        [(0.6, 1.4), (2 / 3, 4 / 3), (0.5, 1.5)],
        [COLORS["green"], COLORS["blue"], COLORS["orange"]],
    ):
        ax[1].plot([lo, hi], [y, y], lw=5, solid_capstyle="butt", color=color)
        ax[1].scatter(
            [lo, hi],
            [y, y],
            facecolors="white" if y == 2 else color,
            edgecolors=color,
            s=45,
            zorder=4,
        )
    ax[1].set(
        yticks=[0, 1, 2],
        yticklabels=["Symmetric residual bound", "Exact box output range", "Required open cell"],
        xlabel="Normalized output",
        ylim=(-0.7, 2.7),
        xlim=(0.35, 1.65),
        title="Output intervals",
    )
    save(
        fig,
        "boundary_geometry",
        [],
        "Analytic two-block example: centers 0 and 2, masses in [1,2], centered numerators zero. No measured data.",
    )

    fig, ax = panels(3.0)
    polygon = np.array([[1, -0.25], [3, -0.75], [3, 1.5], [1, 0.5]])
    ax[0].add_patch(
        Rectangle((1, -3), 2, 6, facecolor=COLORS["orange"], alpha=0.12, edgecolor=COLORS["orange"])
    )
    ax[0].add_patch(
        Polygon(polygon, facecolor=COLORS["blue"], alpha=0.25, edgecolor=COLORS["blue"])
    )
    z = np.linspace(0.5, 3.5, 80)
    ax[0].plot(z, -0.25 * z, color=COLORS["blue"], label="M = −Z/4")
    ax[0].plot(z, 0.5 * z, color=COLORS["green"], label="M = Z/2")
    ax[0].set(
        xlim=(0.5, 3.5),
        ylim=(-3.4, 3.4),
        xlabel="Block mass Z",
        ylabel="Centered numerator M",
        title="Coupled feasible set",
    )
    legend_below(ax[0])
    bounds = [(-2.7, 3.9), (0.05, 2.4), (-4.8, 2.4), (-2.55, -0.1)]
    for y, (lo, hi), color in zip(
        range(4), bounds, [COLORS["orange"], COLORS["blue"], COLORS["orange"], COLORS["blue"]]
    ):
        ax[1].plot([lo, hi], [y, y], lw=4, color=color)
        ax[1].scatter([lo, hi], [y, y], s=25, color=color)
    ax[1].axvline(0, color="black", lw=0.9)
    ax[1].set(
        yticks=range(4),
        yticklabels=[
            "Box at lower boundary",
            "Coupled at lower boundary",
            "Box at upper boundary",
            "Coupled at upper boundary",
        ],
        xlabel="Residual interval",
        title="Boundary residual intervals",
        ylim=(-0.6, 3.6),
    )
    ax[1].invert_yaxis()
    save(
        fig,
        "coupled_geometry",
        [],
        "Analytic block: Z in [1,3], M in [-3,3], centered values in [-1/4,1/2]. Polygon support establishes both strict signs.",
    )

    fig, ax = panels()
    rho = np.geomspace(0.001, 8, 300)
    sharp = np.array([tail_coefficient(x) for x in rho])
    old = np.array([tail_coefficient(x, False) for x in rho])
    ax[0].loglog(rho, old, color=COLORS["orange"], label=r"Lagrange $e^\rho\rho/6$")
    ax[0].loglog(rho, sharp, color=COLORS["blue"], label=r"Sharp $\kappa(\rho)$")
    ax[0].set(
        xlabel="Retained score radius ρ",
        ylabel="Quadratic remainder coefficient",
        title="Exponential remainder growth",
    )
    legend_below(ax[0])
    epsilon = np.linspace(0, 8, 200)
    ax[1].semilogy(epsilon, np.expm1(epsilon) + 1e-7, color=COLORS["red"], label=r"$e^\epsilon-1$")
    ax[1].axhline(1 / 256, color=COLORS["green"], ls="--", label="Upper BF16 half-cell at 1")
    ax[1].set(
        xlabel="Discarded score radius ε",
        ylabel="Projection inflation coefficient",
        title="Discarded-score inflation",
    )
    legend_below(ax[1])
    save(
        fig,
        "remainder_mechanism",
        [],
        "Analytic coefficients, evaluated numerically for visualization. The BF16 guide is a unit-scale reference, not a universal acceptance threshold.",
    )


def cpu_refinement():
    path = ROOT / "results" / "experiments.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = [
        r
        for r in data["ablations"]
        if r["rank"] == 4 and r.get("adaptive_scanned_tokens") is not None
    ]
    fig, ax = panels(3.1)
    x = np.arange(len(rows))
    labels = ["Tight blocks", "One broad block", "All blocks broad"]
    ax[0].bar(
        x - 0.17,
        [r["full_gate_passes"] / r["queries"] for r in rows],
        0.34,
        color=COLORS["blue"],
        label="Initial whole-output pass",
    )
    ax[0].bar(
        x + 0.17,
        [r["adaptive_mean_scanned_fraction"] for r in rows],
        0.34,
        color=COLORS["orange"],
        label="Tokens in correction blocks",
    )
    ax[0].set(
        xticks=x,
        xticklabels=["Tight\nblocks", "One broad\nblock", "All blocks\nbroad"],
        ylabel="Fraction",
        ylim=(0, 1.14),
        title="Acceptance and correction work",
    )
    legend_below(ax[0])
    for i, row in enumerate(rows):
        fractions = np.asarray(row["adaptive_scanned_tokens"]) / data["dimensions"]["N"]
        ax[1].scatter(np.arange(len(fractions)), fractions, s=20, label=labels[i], alpha=0.8)
    ax[1].set(
        xlabel="Query index",
        ylabel="Correction-token fraction",
        title="Correction work for every query",
        ylim=(-0.05, 1.08),
    )
    legend_below(ax[1])
    save(
        fig,
        "selective_refinement",
        ["results/experiments.json"],
        "Numerical CPU mechanism experiment. Correction tokens exclude full-source validation/fingerprinting reads. CPU timings include those reads; the counter is not memory traffic or speedup.",
    )


def coupling_figure():
    path = ROOT / "results" / "certification" / "coupling.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    summary = data["summary"]
    coords = [c for case in data["cases"] for c in case["stages"][0]["coordinates"]]
    ratios = sorted(c["width_coupled"] / c["width_box"] for c in coords if c["width_box"] > 0)
    fig, ax = panels(2.4)
    counts = [
        summary[k] for k in ["global_value_hull_accepted", "box_accepted", "coupled_accepted"]
    ]
    ax[0].bar(range(3), counts, color=[COLORS["gray"], COLORS["orange"], COLORS["blue"]])
    for i, count in enumerate(counts):
        ax[0].text(i, count + 1.5, f"{count}/149", ha="center")
    ax[0].set(
        xticks=range(3),
        xticklabels=["Global value\nhull", "Block\nboxes", "Coupled\nblocks"],
        ylabel="Initially certified coordinates",
        ylim=(0, 78),
        title="Initial certificates",
    )
    ax[1].step(
        ratios, np.arange(1, len(ratios) + 1) / len(ratios), where="post", color=COLORS["blue"]
    )
    ax[1].set(
        xlabel="Coupled / box residual width",
        ylabel="Cumulative fraction",
        xlim=(-0.02, 1.02),
        ylim=(0, 1.05),
        title="Residual-width distribution",
    )
    ax[1].text(
        0.04,
        0.76,
        "30 strict improvements\n119 unchanged\n1 added accept beyond global hull",
        transform=ax[1].transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=8.5,
    )
    save(
        fig,
        "coupling_ablation",
        ["results/certification/coupling.json"],
        "All 149 initial exact-rational coordinates. Zero box-width cases are included in counts but omitted from the width ratio CDF. Four added accepts include three explained by a global hull and one prescribed mass-weighted outlier witness.",
    )


def gpu_figures():
    path = ROOT / "results" / "gh200" / "benchmark.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = [r for r in data["cases"] if r.get("status") == "ok"]
    fig, axes = panels(3.1)
    for ax, Q in zip(axes, [1, 32]):
        selected = sorted(
            [r for r in rows if r.get("profile") == "tight" and r["rank"] == 4 and r["Q"] == Q],
            key=lambda r: r["N"],
        )
        for name, label, color in [
            ("dense_bf16_flash", "Fused BF16 attention", COLORS["green"]),
            ("screen_parallel", "Summary screen", COLORS["blue"]),
            ("screen_and_batch_fallback", "Screen + decision + fallback", COLORS["orange"]),
        ]:
            values = np.array([r["timings"][name]["wall_ms"] * 1000 for r in selected])
            low = np.array(
                [np.quantile(r["timings"][name]["wall_samples_ms"], 0.1) * 1000 for r in selected]
            )
            high = np.array(
                [np.quantile(r["timings"][name]["wall_samples_ms"], 0.9) * 1000 for r in selected]
            )
            ax.plot([r["N"] for r in selected], values, "o-", label=label, color=color, ms=4)
            ax.fill_between([r["N"] for r in selected], low, high, color=color, alpha=0.1)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Visible keys N",
            ylabel="Measured batch latency (µs)",
            title="One decode query" if Q == 1 else "32 queries, shared K/V",
        )
        context_ticks(ax, [r["N"] for r in selected])
    shared_legend(fig, axes[0])
    save(
        fig,
        "gh200_latency",
        ["results/gh200/benchmark.json"],
        "Actual GH200 wall-clock medians and 10th–90th sample quantiles, tight synthetic rank 4 inputs. Shared-KV batches are not independent request batches. Setup plotted separately.",
    )

    chosen = [
        r
        for r in rows
        if r.get("profile") in ["tight", "moderate", "broad", "centered", "midpoint"]
        and r["rank"] == 4
        and r["Q"] == 32
        and r["N"] == 8192
    ]
    if chosen:
        fig, ax = panels()
        labels = [r["profile"] for r in chosen]
        x = np.arange(len(chosen))
        ax[0].bar(
            x, [r["numerical_whole_output_passes"] / r["Q"] for r in chosen], color=COLORS["blue"]
        )
        ax[0].set(
            xticks=x,
            xticklabels=labels,
            ylim=(0, 1.08),
            ylabel="Whole-output pass fraction",
            title="Complete-output acceptance",
        )
        plt.setp(ax[0].get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        for name, label, color in [
            ("dense_bf16_flash", "Fused BF16", COLORS["green"]),
            ("screen_and_batch_fallback", "Complete screened path", COLORS["orange"]),
        ]:
            ax[1].plot(
                x,
                [r["timings"][name]["wall_ms"] * 1000 for r in chosen],
                "o-",
                label=label,
                color=color,
            )
        ax[1].set(
            xticks=x,
            xticklabels=labels,
            ylabel="Batch wall time (µs)",
            title="Complete-path latency",
        )
        plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        legend_below(ax[1])
        save(
            fig,
            "gh200_coverage_cost",
            ["results/gh200/benchmark.json"],
            "N=8192, Q=32, r=4 profile controls. A single failed coordinate sends this simple scheduler's complete batch to dense attention.",
        )

    selected = sorted(
        [r for r in rows if r.get("profile") == "tight" and r["rank"] == 4 and r["Q"] == 32],
        key=lambda r: r["N"],
    )
    if selected:
        fig, ax = panels(2.55)
        x = np.arange(len(selected))
        bottom = np.zeros(len(selected))
        for name, label, color in [
            ("shift", "Query shift", COLORS["gray"]),
            ("shared_evaluate", "Summary evaluation", COLORS["blue"]),
            ("parallel_reduce_screen", "Reduction and screen", COLORS["orange"]),
        ]:
            values = np.array([r["timings"][name]["gpu_ms"] * 1000 for r in selected])
            ax[0].bar(x, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax[0].set(
            xticks=x,
            xticklabels=[f"{r['N']:,}" for r in selected],
            xlabel="Visible keys N",
            ylabel="Sum of phase medians (µs)",
            title="Isolated device phases",
        )
        legend_below(ax[0])
        for name, label, color in [
            ("summary_cpu_ms", "CPU summary construction", COLORS["blue"]),
            ("summary_upload_ms", "Summary upload", COLORS["orange"]),
            ("workspace_allocation_ms", "Workspace allocation", COLORS["green"]),
        ]:
            ax[1].plot(
                [r["N"] for r in selected],
                [r["setup"][name] for r in selected],
                "o-",
                label=label,
                color=color,
            )
        ax[1].set(
            xscale="log",
            yscale="log",
            xlabel="Visible keys N",
            ylabel="One-time measured cost (ms)",
            title="Construction and setup",
        )
        context_ticks(ax[1], [r["N"] for r in selected])
        legend_below(ax[1])
        save(
            fig,
            "gh200_cost_breakdown",
            ["results/gh200/benchmark.json"],
            "Isolated phase times diagnose bottlenecks; their sum is not substituted for measured end-to-end latency. Setup includes construction, upload, and allocation.",
        )

        fig, ax = panels(2.55)
        for r in [selected[0], selected[-1]]:
            reuse = np.geomspace(1, 1e6, 160)
            setup = r["setup"]["reusable_summary_and_workspace_ms"]
            total = setup + reuse * r["timings"]["screen_and_batch_fallback"]["wall_ms"]
            dense = reuse * r["timings"]["dense_bf16_flash"]["wall_ms"]
            ax[0].loglog(reuse, total / dense, label=f"N={r['N']:,}, Q=32")
        ax[0].axhline(1, color=COLORS["green"], ls="--", label="Break-even")
        ax[0].set(
            xlabel="Reuses of one immutable summary",
            ylabel="Total time / fused dense time",
            title="Setup amortization",
        )
        legend_below(ax[0])
        for name, label, color in [
            ("screen_parallel", "Summary screen", COLORS["blue"]),
            ("dense_bf16_flash", "Fused BF16", COLORS["green"]),
        ]:
            ax[1].loglog(
                [r["N"] for r in selected],
                [r["graph_timings"][name]["gpu_ms"] * 1000 for r in selected],
                "o-",
                label=label,
                color=color,
            )
        ax[1].set(
            xlabel="Visible keys N",
            ylabel="Graph latency (µs / batch)",
            title="Device graph replay",
        )
        context_ticks(ax[1], [r["N"] for r in selected])
        legend_below(ax[1])
        save(
            fig,
            "gh200_amortization",
            ["results/gh200/benchmark.json"],
            "Left: derived amortization from measured setup and steady-state costs, no modeled speedup. Right: matched fixed-input CUDA graphs, excluding construction and fallback; not serving throughput.",
        )

        fig, ax = panels(2.5)
        for axis, Q in zip(ax, [1, 32]):
            sweep = sorted(
                [r for r in rows if r.get("profile") == "tight" and r["rank"] == 4 and r["Q"] == Q],
                key=lambda r: r["N"],
            )
            for key, label, color in [
                ("screen_original", "Original", COLORS["gray"]),
                ("screen_shared", "Shared block scalars", COLORS["orange"]),
                ("screen_parallel", "Parallel block reduction", COLORS["blue"]),
                ("dense_bf16_flash", "Fused BF16", COLORS["green"]),
            ]:
                axis.loglog(
                    [r["N"] for r in sweep],
                    [r["graph_timings"][key]["gpu_ms"] * 1000 for r in sweep],
                    "o-",
                    color=color,
                    label=label,
                )
            axis.set(
                xlabel="Visible keys N",
                ylabel="Graph latency (µs / batch)",
                title="One decode query" if Q == 1 else "32 queries, shared K/V",
            )
            context_ticks(axis, [r["N"] for r in sweep])
        shared_legend(fig, ax[0], columns=2)
        save(
            fig,
            "gh200_kernel_ablation",
            ["results/gh200/benchmark.json"],
            "Matched three-variant kernel ablation. Sharing scalar work barely changes the serial-reduction bottleneck; parallel block reduction changes it. Every small-N regression is retained.",
        )


def trace_figures():
    path = ROOT / "results" / "model_traces" / "diagnostics.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    low = [r for r in rows if r["rank"] == 8]
    full = [r for r in rows if r["rank"] == r["d"]]
    fig, ax = panels(3.1)
    heat = np.array(
        [
            [
                np.median(
                    [r["epsilon_max"] for r in low if r["layer"] == layer and r["head"] == head]
                )
                for head in range(14)
            ]
            for layer in range(24)
        ]
    )
    im = ax[0].imshow(heat, aspect="auto", cmap="magma", origin="upper")
    ax[0].set(xlabel="Query head", ylabel="Layer", title="Discarded radius at rank 8")
    ax[0].set(xticks=range(0, 14, 2), yticks=[0, 4, 8, 12, 16, 20, 23])
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], label="Median max-block ε", pad=0.035, fraction=0.055)
    a = np.array([r["actual_projected_radius_max"] for r in full])
    b = np.array([r["rho_max"] for r in full])
    ax[1].scatter(a, b, alpha=0.45, s=15, color=COLORS["blue"])
    t = np.geomspace(min(a.min(), b.min()) * 0.8, max(a.max(), b.max()) * 1.1, 100)
    ax[1].plot(t, t, color=COLORS["green"], ls="--", label="Equality line")
    ax[1].set(
        xscale="log",
        yscale="log",
        xlabel=r"Scanned radius $\max |q^T\delta|$",
        ylabel="Coordinate-radius bound ρ",
        title="Full-rank score-radius bounds",
    )
    legend_below(ax[1])
    save(
        fig,
        "model_score_radii",
        ["results/model_traces/diagnostics.json", "results/model_traces/manifest.json"],
        "Qwen2.5-0.5B actual post-RoPE arrays. Left: all 24 layers and 14 heads at rank 8. Right: full rank in layers 0, 12, 23; scan-derived radii are diagnostic and cost token reads.",
    )

    fig, ax = panels(2.4)
    for axis, selected, title in zip(
        ax, [low, full], ["Rank 8: projection term", "Rank 64: Taylor term"]
    ):
        fields = [
            "omitted_quadratic_over_true_mass",
            "taylor_remainder_over_true_mass",
            "discarded_score_over_true_mass",
            "boundary_margin_min",
        ]
        values = [
            [np.log10(r[f]) for r in selected if r.get(f) is not None and r[f] > 0] for f in fields
        ]
        active = [(i, v) for i, v in enumerate(values) if v]
        plot = axis.boxplot(
            [v for _, v in active],
            positions=[i + 1 for i, _ in active],
            tick_labels=["Quadratic", "Taylor", "Projection", "Cell\nmargin"]
            if len(active) == 4
            else None,
            showfliers=False,
            widths=0.55,
            patch_artist=True,
        )
        for patch, (i, _) in zip(plot["boxes"], active):
            patch.set_facecolor(
                [COLORS["gray"], COLORS["orange"], COLORS["red"], COLORS["green"]][i]
            )
            patch.set_alpha(0.65)
        axis.set(
            xticks=[1, 2, 3, 4],
            xticklabels=["Quadratic", "Taylor", "Projection", "Cell\nmargin"],
            ylabel="log₁₀(value in output units)",
            title=title,
            xlim=(0.5, 4.5),
        )
        if not values[2]:
            axis.text(3, axis.get_ylim()[0] + 1, "exactly zero", ha="center", fontsize=8.5)
    save(
        fig,
        "model_error_budget",
        ["results/model_traces/diagnostics.json"],
        "Maximum-channel centered-numerator terms divided by true shifted mass, versus minimum-channel BF16 margin. Box plots show quartiles and 1.5 IQR whiskers; extreme values remain in JSON. Terms diagnose scale and are not added as an exact output error decomposition.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-only", action="store_true")
    args = parser.parse_args()
    analytic_figures()
    if not args.analytic_only:
        cpu_refinement()
        coupling_figure()
        gpu_figures()
        trace_figures()
    (OUT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    (ROOT / "results/figure-layout.json").write_text(json.dumps(LAYOUT, indent=2) + "\n")
    print(f"Generated {len(MANIFEST)} figures as SVG, PDF, and PNG.")


if __name__ == "__main__":
    main()
