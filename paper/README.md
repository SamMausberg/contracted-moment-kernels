# Native LaTeX manuscript

[Read the PDF](PAPER.pdf) · [Edit PAPER.tex](PAPER.tex) ·
[Download the source archive](latex-source.zip) · [Mathematical companion](PAPER.md)

The paper uses the official ICML 2026 two-column style in `preprint` mode.
`PAPER.tex` controls the title, abstract, section order, appendix, and bibliography.
Edit prose and proofs in `sections/`, geometric drawings in `diagrams/`, and
references in `references.bib`. The twelve figures include two native TikZ
drawings and ten plots based on the existing measurements or analytic bounds.
The Markdown companion retains the expanded mathematical exposition.

From the repository root, run `make paper`. This builds the PDF, checks layout,
and creates `latex-source.zip`. It uses the committed plots without rerunning
experiments. Use `make figures` separately when plot data or presentation changes.

For the source archive, extract it and run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error PAPER.tex
```

Required Ubuntu packages are `latexmk`, `texlive-latex-extra`,
`texlive-science`, `texlive-fonts-recommended`, and `poppler-utils`.
The build uses pdfLaTeX, BibTeX, `amsmath`, `amsthm`, `mathtools`, `microtype`,
`booktabs`, `tabularx`, `siunitx`, `natbib`, `hyperref`, `cleveref`, and TikZ.
Auxiliary files go in `build/paper/` during a repository build.

`python3 scripts/check_paper.py --output results/paper-layout.json` checks
references, captions, font embedding, page dimensions, and text bounds.
The result includes hashes of the native sources and imported figure PDFs;
visual review remains necessary for graphical overlap and reading order.

The upstream style files are unchanged; [their provenance](vendor/README.md)
records the official download and hashes. The preprint author footnote links
the repository without inferring an affiliation or conference acceptance.
