# Development constraints

Keep the README short. Put mathematical details in `paper/PAPER.md`, execution
semantics in `docs/GH200.md`, and verification gaps in `docs/FORMALIZATION.md`.
Keep research citations at the bottom of the README and paper.

Do not call proof source machine-checked without a successful Lean build and
axiom audit. Do not insert proof holes, custom axioms, unsafe proof machinery,
or native proof-evaluation shortcuts. A conditional lifting lemma does not
prove its premises. Match every public verification claim to its actual theorem
and implementation scope.

The numerical summary path is not an outward interval implementation. Never
reinterpret it as sound metadata for the directed checker. Keep exact-real
attention distinct from bitwise equality to a particular GPU kernel. Reject
invalid summaries, masks, epochs, dimensions, or unsupported numerical domains.

Retain failure cases and setup/fallback costs in performance reports. Do not
claim GPU performance from NumPy timings, scalar host checks, or an operation
count. A novelty claim needs a current primary-literature comparison, not just
a new method name. Do not edit unrelated repositories to run these experiments.
