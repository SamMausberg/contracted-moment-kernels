#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check local theorem/audit coverage; optionally validate an actual axiom log."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-log", type=Path)
    parser.add_argument("--lean-only", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    files = sorted((ROOT / "lean" / "CMK").glob("*.lean"))
    names = []
    for path in [*files, ROOT / "lean" / "CMK.lean"]:
        source = path.read_text()
        assert not re.search(r"\b(sorry|admit|axiom|unsafe|native_decide)\b", source), path
        names.extend(re.findall(r"^(?:theorem|lemma)\s+(\w+)", source, re.M))
    audit = (ROOT / "lean" / "CMK" / "Audit.lean").read_text()
    audited = re.findall(r"^#print axioms CMK\.(\w+)\s*$", audit, re.M)
    assert names and len(set(names)) == len(names), "Missing or duplicate theorem declarations"
    assert len(audited) == len(set(audited)), "Duplicate axiom audit entries"
    assert set(names) == set(audited), (
        f"Unaudited declarations: {set(names) - set(audited)}; "
        f"unknown audit entries: {set(audited) - set(names)}"
    )
    if not args.lean_only:
        for path in [ROOT / "README.md", ROOT / "paper" / "PAPER.md", *sorted((ROOT / "docs").glob("*.md"))]:
            assert "\u2014" not in path.read_text(), f"Unexpected em dash in {path}"
    if args.audit_log:
        seen = {}
        for line in args.audit_log.read_text().splitlines():
            match = re.fullmatch(r"'CMK\.(\w+)' (?:depends on axioms: \[(.*)\]|does not depend on any axioms)", line)
            assert match, f"Unrecognized or incomplete axiom audit output: {line!r}"
            name, dependencies = match.groups()
            assert name not in seen, f"Duplicate audit result: {name}"
            axioms = set(dependencies.split(", ")) if dependencies else set()
            assert axioms <= ALLOWED_AXIOMS, f"Unapproved axioms for {name}: {axioms - ALLOWED_AXIOMS}"
            seen[name] = axioms
        assert set(seen) == set(names), "Axiom log does not cover exactly the local declarations"
        if args.write_manifest:
            inputs = [*files, ROOT / "lean" / "CMK.lean", ROOT / "lean" / "lean-toolchain",
                      ROOT / "lean" / "lakefile.toml", ROOT / "lean" / "lake-manifest.json"]
            manifest = {
                "theorem_count": len(names),
                "allowed_axioms": sorted(ALLOWED_AXIOMS),
                "theorem_axioms": {name: sorted(seen[name]) for name in sorted(seen)},
                "source_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in inputs
                },
                "audit_log_sha256": hashlib.sha256(args.audit_log.read_bytes()).hexdigest(),
            }
            args.write_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Axiom audit passed: {len(names)} declarations; dependencies limited to {sorted(ALLOWED_AXIOMS)}.")
    else:
        print(f"Static source checks passed: {len(names)} theorem declarations and matching audit entries. Lean not executed by this check.")


if __name__ == "__main__":
    main()
