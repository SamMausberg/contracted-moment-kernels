# Figure index

The native [LaTeX manuscript](../PAPER.tex) contains twelve figures: two TikZ
geometric drawings compiled from `paper/diagrams/` and ten imported vector PDF
plots. The table follows the [built PDF](../PAPER.pdf); numbering in the expanded
[Markdown companion](../PAPER.md) differs.

`make figures` runs `scripts/figures.py` to export SVG/PDF figures and PNG
previews from recorded data and analytic examples. It also retains standalone
geometric exports for the companion; the native paper uses TikZ for those two
drawings. Data sources and hashes are recorded in [manifest.json](manifest.json).
`make paper` builds the manuscript using committed plot PDFs, checks the PDF,
and packages the editable drawings and plots in
[latex-source.zip](../latex-source.zip), without rerunning experiments.

The official ICML layout uses two-column main text and permitted single-column
appendices. Imported plots are generated at 6.75 inches wide with 8.5pt axis and
legend text; legends sit outside the data panels. The generator rejects legend
overlaps with data panels and axis decorations outside the canvas, recording
geometry and export hashes in [figure-layout.json](../../results/figure-layout.json).
The [Research checks workflow](../../.github/workflows/verify.yml) regenerates
plots and checks this placement from saved data using the existing Matplotlib
environment, without rerunning benchmarks. The numbered figure mapping remains:

| PDF figure | Source | Question |
| --- | --- | --- |
| 1: Coupled geometry | [TikZ](../diagrams/coupled_geometry.tex) | Which impossible box corners do value bounds exclude? |
| 2: Coupling ablation | [Vector PDF](coupling_ablation.pdf) | How much does coupling add beyond a simple value hull? |
| 3: Model error budget | [Vector PDF](model_error_budget.pdf) | Which uncertainty term prevents the model traces from passing? |
| 4: GH200 kernel ablation | [Vector PDF](gh200_kernel_ablation.pdf) | Does scalar sharing or block parallelism remove the bottleneck? |
| 5: Boundary geometry | [TikZ](../diagrams/boundary_geometry.tex) | What information do the two residual signs retain? |
| 6: Remainder mechanism | [Vector PDF](remainder_mechanism.pdf) | Why do retained and discarded score radii have different costs? |
| 7: Selective refinement | [Vector PDF](selective_refinement.pdf) | Is uncertainty localized or spread across all blocks? |
| 8: GH200 coverage and cost | [Vector PDF](gh200_coverage_cost.pdf) | Why does coordinate coverage fail to predict full-output coverage? |
| 9: Model score radii | [Vector PDF](model_score_radii.pdf) | Which layers and heads have broad score bounds? |
| 10: GH200 latency | [Vector PDF](gh200_latency.pdf) | How does the complete GPU path compare with fused dense attention? |
| 11: GH200 cost breakdown | [Vector PDF](gh200_cost_breakdown.pdf) | What do execution phases and one-time setup actually cost? |
| 12: GH200 amortization | [Vector PDF](gh200_amortization.pdf) | Can reuse repay setup at the measured query costs? |
