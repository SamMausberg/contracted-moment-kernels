#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build checks for local Markdown links, display-math fences and figures."""

from pathlib import Path
from urllib.parse import unquote, urlparse

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = MarkdownIt("commonmark").enable("table")
    paths = [ROOT / "README.md", ROOT / "paper" / "PAPER.md"]
    paths += sorted((ROOT / "docs").glob("*.md"))
    paths += sorted((ROOT / "results").rglob("*.md"))
    checked = 0
    for path in paths:
        text = path.read_text()
        assert text.endswith("\n"), f"Missing final newline: {path}"
        assert all(line == line.rstrip() for line in text.splitlines()), (
            f"Trailing whitespace: {path}"
        )
        assert sum(line.strip() == "$$" for line in text.splitlines()) % 2 == 0, path
        for token in parser.parse(text):
            for child in token.children or []:
                target = child.attrGet("src") if child.type == "image" else child.attrGet("href")
                if not target or urlparse(target).scheme or target.startswith("#"):
                    continue
                relative = unquote(urlparse(target).path)
                assert (path.parent / relative).exists(), f"Broken local link: {path}: {target}"
                checked += 1
    print(f"Markdown checks passed: {len(paths)} documents, {checked} local links.")


if __name__ == "__main__":
    main()
