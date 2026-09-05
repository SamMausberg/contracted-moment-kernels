#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the manuscript PDF's mechanical publication/layout properties.

Requires Python 3 and Poppler's pdftotext, pdfinfo, and pdffonts commands
(Debian/Ubuntu: apt install poppler-utils). Build with LaTeX's recorder enabled;
the final .log and sibling .fls must accompany the PDF. No Python packages are
required. Example: python scripts/check_paper.py --output build/paper/check.json

This checks extracted text bounds, not graphical ink, overlapping content,
scientific claims, numerical correctness, or formal proofs. Reference checks
use the final LaTeX log; remote hyperlink availability is not checked.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
EXPECTED_FIGURES = set(range(1, 13))


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    if result.returncode:
        raise ValueError(f"{args[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def recorder_paths(path):
    """Resolve only paths actually recorded by this LaTeX invocation."""
    directory = path.parent
    inputs, outputs = set(), set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, _, name = line.partition(" ")
        if kind == "PWD":
            directory = Path(name)
        elif kind in {"INPUT", "OUTPUT"}:
            item = Path(name)
            if not item.is_absolute():
                item = directory / item
            (inputs if kind == "INPUT" else outputs).add(item.resolve())
    return inputs, outputs


def validate(pdf, log):
    failures = []
    fls = log.with_suffix(".fls")
    for path in (pdf, log, fls):
        if not path.is_file():
            raise ValueError(f"Required file is missing: {display_path(path)}")
    for executable in ("pdftotext", "pdfinfo", "pdffonts"):
        if shutil.which(executable) is None:
            raise ValueError(f"Missing command {executable}; install poppler-utils")

    log_text = log.read_text(encoding="utf-8", errors="replace")
    # Wrapped LaTeX diagnostics are matched after collapsing whitespace.
    compact_log = " ".join(log_text.split())
    error_patterns = (
        r"^!.*$",
        r"^.*(?:LaTeX|Package \S+) Error:.*$",
        r"^.*(?:Emergency stop|Fatal error occurred|Undefined control sequence).*$",
    )
    errors = sorted(
        {match for pattern in error_patterns for match in re.findall(pattern, log_text, re.M)}
    )
    unresolved_patterns = (
        r"(?:Reference|Citation)\s+.{0,250}?\bundefined",
        r"There were undefined (?:references|citations)",
        r"Rerun to get cross-references right",
        r"Label\(s\) may have changed",
        r"Label\s+.{0,250}?\bmultiply defined",
        r"There were multiply-defined labels",
    )
    unresolved = sorted(
        {
            match
            for pattern in unresolved_patterns
            for match in re.findall(pattern, compact_log, re.I)
        }
    )
    overfull = re.findall(r"Overfull \\[hv]box[^\n]*", log_text)
    underfull = re.findall(r"Underfull \\[hv]box[^\n]*", log_text)
    warnings = re.findall(r"^.*Warning:.*$", log_text, re.M)
    if errors:
        failures.append(f"LaTeX errors: {len(errors)}")
    if unresolved:
        failures.append(
            "Unresolved or multiply defined references/citations; rebuild or fix the source"
        )
    if overfull:
        failures.append(f"Overfull boxes: {len(overfull)}")
    if "Output written on " not in log_text:
        failures.append("LaTeX log does not report a completed PDF")

    info = command("pdfinfo", str(pdf))
    pages_match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)
    if pages_match is None:
        raise ValueError("pdfinfo did not report a page count")
    page_count = int(pages_match.group(1))
    # bbox-layout includes word rectangles and line grouping for caption labels.
    bbox = ET.fromstring(command("pdftotext", "-bbox-layout", str(pdf), "-"))
    pages = bbox.findall(".//{*}page")
    if len(pages) != page_count:
        failures.append("pdfinfo and pdftotext disagree on the page count")
    dimensions, outside, captions, page_lines = [], [], [], []
    word_count = 0
    tolerance = 0.1  # PDF points: absorb extraction/serialization rounding only.
    for number, page in enumerate(pages, 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        dimensions.append({"page": number, "width_pt": width, "height_pt": height})
        if not all(math.isfinite(value) for value in (width, height)) or (
            abs(width - 612) > tolerance or abs(height - 792) > tolerance
        ):
            failures.append(f"Page {number} is not portrait US Letter (612 x 792 pt)")
        words = page.findall(".//{*}word")
        word_count += len(words)
        for word in words:
            bounds = [float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax")]
            left, top, right, bottom = bounds
            if (
                not all(math.isfinite(value) for value in bounds)
                or left < -tolerance
                or top < -tolerance
                or right > width + tolerance
                or bottom > height + tolerance
                or left > right
                or top > bottom
            ):
                outside.append({"page": number, "text": word.text, "bounds_pt": bounds})
        lines = [
            " ".join("".join(word.itertext()) for word in line.findall("{*}word"))
            for line in page.findall(".//{*}line")
        ]
        page_lines.append(lines)
        for line in lines:
            match = re.match(r"^Figure\s+(\d+)\.(?:\s|$)", line)
            if match:
                captions.append({"number": int(match.group(1)), "page": number})
    if not word_count:
        failures.append("PDF contains no extractable text")
    if outside:
        failures.append(f"Text rectangles outside page bounds: {len(outside)}")
    counts = Counter(caption["number"] for caption in captions)
    if set(counts) != EXPECTED_FIGURES or any(count != 1 for count in counts.values()):
        failures.append(
            f"Expected figure captions 1 through 12 exactly once; found {dict(sorted(counts.items()))}"
        )

    font_output = command("pdffonts", str(pdf))
    fonts = []
    for line in font_output.splitlines()[2:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 8 or fields[-5] not in {"yes", "no"}:
            raise ValueError(f"Unrecognized pdffonts row: {line}")
        fonts.append(
            {
                "name": fields[0],
                "type": " ".join(fields[1:-6]),
                "encoding": fields[-6],
                "embedded": fields[-5] == "yes",
                "subset": fields[-4] == "yes",
                "unicode_map": fields[-3] == "yes",
            }
        )
    if not fonts or any(not font["embedded"] for font in fonts):
        failures.append("Missing font inventory or unembedded PDF fonts")

    inputs, outputs = recorder_paths(fls)
    figure_inputs = sorted(
        path
        for path in inputs
        if path.suffix.lower() == ".pdf" and path.is_relative_to(PAPER / "figures")
    )
    if not figure_inputs:
        failures.append("Recorder contains no imported vector figure PDFs")
    source_inputs = set(figure_inputs)
    for suffix in ("tex", "bib", "sty", "bst"):
        source_inputs.update(PAPER.glob(f"*.{suffix}"))
    for directory in ("sections", "diagrams"):
        source_inputs.update((PAPER / directory).glob("*.tex"))
    missing = sorted(path for path in source_inputs if not path.is_file())
    if missing:
        failures.append("Recorded sources are missing: " + ", ".join(map(display_path, missing)))
    recorded_pdfs = sorted(path for path in outputs if path.suffix.lower() == ".pdf")
    matching_build = any(path.is_file() and sha256(path) == sha256(pdf) for path in recorded_pdfs)
    if not matching_build:
        failures.append("Selected PDF does not match a PDF output recorded by this build")

    # These are explicit, numbered manuscript headings, not incidental prose.
    appendix_pages = [
        number
        for number, lines in enumerate(page_lines, 1)
        if any(
            re.match(r"^A\.\s+Supporting Certificate and Moment Proofs\s*$", line) for line in lines
        )
    ]
    reference_pages = [
        number
        for number, lines in enumerate(page_lines, 1)
        if any(line.strip() == "References" for line in lines)
    ]
    appendix_start = appendix_pages[0] if len(appendix_pages) == 1 else None
    references_start = reference_pages[0] if len(reference_pages) == 1 else None
    if appendix_start is None:
        failures.append("Expected exactly one numbered appendix opening")
    elif appendix_start > 9:
        failures.append("Main manuscript exceeds the conference's eight-page limit")
    if references_start is None:
        failures.append("Expected exactly one References heading")
    elif appendix_start is not None and references_start <= appendix_start:
        failures.append("References must follow the appendices")
    pagination = {"appendix_start_page": appendix_start, "references_start_page": references_start}
    if appendix_start is not None:
        pagination["main_pages_before_appendix"] = appendix_start - 1
        pagination["appendix_pages_including_references"] = page_count - appendix_start + 1
        if references_start is not None and references_start >= appendix_start:
            pagination["appendix_pages_before_references"] = references_start - appendix_start
    inspected = {pdf, log, fls, *source_inputs}
    manifest = {
        "schema_version": 1,
        "passed": not failures,
        "scope": "Mechanical PDF/layout checks only; no scientific or formal verification claims.",
        "pdf": display_path(pdf),
        "pdf_sha256": sha256(pdf),
        "log": display_path(log),
        "log_sha256": sha256(log),
        "recorder": display_path(fls),
        "recorder_sha256": sha256(fls),
        "page_count": page_count,
        "pagination": pagination,
        "page_dimensions": dimensions,
        "font_checks": {
            "all_embedded": bool(fonts) and all(font["embedded"] for font in fonts),
            "fonts": fonts,
        },
        "text_bounds": {
            "all_inside_page": not outside,
            "word_count": word_count,
            "tolerance_pt": tolerance,
            "violations": outside,
        },
        "figure_captions": captions,
        "latex": {
            "errors": errors,
            "unresolved_references": unresolved,
            "overfull_count": len(overfull),
            "overfull": overfull,
            "underfull_count": len(underfull),
            "warnings": warnings,
        },
        "reference_checks": "Final LaTeX log checked for undefined/multiply-defined labels and citations and rerun requests; remote URLs not fetched.",
        "source_sha256": {
            display_path(path): sha256(path) for path in sorted(source_inputs) if path.is_file()
        },
        "vector_figure_inputs": list(map(display_path, figure_inputs)),
        "recorded_pdf_matches": matching_build,
        "inspected_files": [display_path(path) for path in sorted(inspected) if path.is_file()],
        "failures": failures,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PAPER / "PAPER.pdf")
    parser.add_argument("--log", type=Path, default=ROOT / "build/paper/PAPER.log")
    parser.add_argument("--output", type=Path, help="Write the JSON validation manifest")
    args = parser.parse_args()
    try:
        manifest = validate(args.pdf.resolve(), args.log.resolve())
    except (OSError, ValueError, ET.ParseError) as error:
        manifest = {"schema_version": 1, "passed": False, "failures": [str(error)]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not manifest["passed"]:
        for failure in manifest["failures"]:
            print(f"Paper check failed: {failure}", file=sys.stderr)
        return 1
    print(
        f"Paper checks passed: {manifest['page_count']} US Letter pages, "
        f"12 unique figure captions, all {len(manifest['font_checks']['fonts'])} fonts embedded, "
        f"no overfull boxes or extracted text outside pages. "
        f"Underfull boxes (informational): {manifest['latex']['underfull_count']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
