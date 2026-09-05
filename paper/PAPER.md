# Boundary certificates for contracted moment reductions

**Research manuscript, version 0.1.0. September 4, 2026.**

## Abstract

A positive normalized reduction can be evaluated indirectly by determining the
sign of its residual at the boundaries of a required observation cell. We derive
an exact certificate for independently enclosed block masses and centered
numerators. Its bounds are attained at box vertices, and intersecting sound
block enclosures improves both boundary tests monotonically. For exponential
attention, we construct the enclosures from projected moments, an explicitly
retained signed diagonal, a row-sum witness for the omitted quadratic form, and
a sharpened exponential remainder. Discarded score coordinates contribute an
explicit error term. This produces a conditional route from a token scan to a
small-summary query with selective block refinement. The results concern real
arithmetic; they are not a general GPU speedup theorem.

**Verification boundary.** The paper gives ordinary mathematical proofs. The
repository contains 25 Lean theorem declarations with proof scripts for the
algebraic certificate and supporting lemmas, but Lean could not be executed in
the authoring environment. Analytic and implementation bridges are not fully
formalized. None of the development is advertised as machine-checked. See
[formalization coverage](../docs/FORMALIZATION.md). The CPU reference, rational
checks, and host C++ checks were executed. CUDA was not compiled or run.

## 1. Definitions and the observation contract

Let the visible input be a finite, nonempty set partitioned into nonempty blocks
$b=1,\ldots,B$. Write a positive normalized reduction as

$$
y_j=\frac{N_j}{Z},\qquad Z=\sum_b z_b>0,\qquad N_j=\sum_b n_{bj}.
$$

For a fixed block center $\nu_{bj}$, define the *centered numerator*

$$m_{bj}=n_{bj}-\nu_{bj}z_b.$$

Suppose metadata establishes sound intervals

$$0\leq l_b\leq z_b\leq u_b,
\qquad m^-_{bj}\leq m_{bj}\leq m^+_{bj}.$$

All intervals refer to the same input, mask, query, scaling, and version of the
underlying data. Positivity of $Z$ is a separate premise. In executable checks,
$\sum_b l_b>0$ is a sufficient way to discharge it.

An observation cell $(a_j,b_j)$ is an open interval on which the consumer is
constant. For finite numerical BF16 rounding, one may choose the midpoints to
the preceding and following representable values. Strict inequalities avoid
tie-breaking ambiguity. This formulation models numerical values, not the
bitwise distinction between positive and negative zero.

**Lemma 1 (constant consumer).** If $y\in S$ and $C(x)=c$ for every $x\in S$,
then $C(y)=c$.

*Proof.* Substitute $y$ in the premise. For monotone rounding, an alternative
closed-interval test is $Q(l)=Q(u)$ with $l\leq y\leq u$; monotonicity gives
$Q(l)\leq Q(y)\leq Q(u)$. $\square$

Lean source: `constant_consumer`, `monotone_rounding`. A concrete IEEE-754 model
and its monotonicity theorem are not supplied.

## 2. Exact residual certificates over block boxes

At a boundary $c$, define

$$
\begin{aligned}
\mathcal L_j(c)&=\sum_b\left[m^-_{bj}+
\min\{(\nu_{bj}-c)l_b,(\nu_{bj}-c)u_b\}\right],\\
\mathcal U_j(c)&=\sum_b\left[m^+_{bj}+
\max\{(\nu_{bj}-c)l_b,(\nu_{bj}-c)u_b\}\right].
\end{aligned}
$$

**Theorem 2 (boundary certificate).** Under the interval and positivity premises,

$$\mathcal L_j(c)\leq N_j-cZ\leq\mathcal U_j(c).$$

Consequently,

$$\boxed{\mathcal L_j(a_j)>0\quad\text{and}\quad
\mathcal U_j(b_j)<0\quad\Longrightarrow\quad a_j<y_j<b_j.}$$

*Proof.* The exact residual identity is

$$N_j-cZ=\sum_b\{m_{bj}+(\nu_{bj}-c)z_b\}.$$

For a fixed scalar $\alpha$, the extrema of $\alpha z$ on $[l,u]$ are
$\min(\alpha l,\alpha u)$ and $\max(\alpha l,\alpha u)$. Apply this identity
with $\alpha=\nu_{bj}-c$, add the centered-numerator bounds, and sum. Dividing
the first strict sign inequality by the positive $Z$ gives $a_j<y_j$; the other
gives $y_j<b_j$. $\square$

Lean source: `mul_enclosure`, `residual_identity`, `residual_enclosure`,
`observation_cell`.

This test does not divide to evaluate the certificate. A numerical division may
still be used to choose a candidate cell. A poor candidate cannot invalidate a
sound certificate: it only makes certification less likely.

**Theorem 3 (optimality for the stated metadata).** Treat the block intervals as
an independent Cartesian product, with no extra coupling constraints. For every
fixed $c$, $\mathcal L_j(c)$ and $\mathcal U_j(c)$ are the attained minimum and
maximum of the residual over that product.

*Proof.* For the minimum, choose $m_{bj}=m^-_{bj}$ and choose $z_b=l_b$ when
$\nu_{bj}-c\geq0$, otherwise $z_b=u_b$. These choices are simultaneous,
admissible vertices of the product box and make every summand its minimum. The
opposite choices and $m_{bj}=m^+_{bj}$ attain the maximum. Theorem 2 supplies the
matching inequalities. $\square$

Lean source: `lowerChoice_mem`, `upperChoice_mem`, `lowerChoice_attains`,
`upperChoice_attains`, `lower_attained`, `upper_attained`.

This is not a universal optimality claim about attention algorithms. Actual
block masses and centered numerators can be correlated; the product box forgets
that information. Richer metadata may give tighter bounds. The two extrema need
not be attained by the same input. If the extremizers are interpreted as
normalized outputs rather than residuals, require $\sum_b l_b>0$ so that
all vertices have positive denominator. Residual-box optimality itself does
not need that additional condition.

**Example.** Take two blocks with centers $0,2$, masses in $[1,2]$, and centered
numerators identically zero. The true output range is $[2/3,4/3]$. The boundary
certificate proves membership in $(0.6,1.4)$. A symmetric residual bound around
$1$, using midpoint masses, yields radius $0.5$ and does not prove this cell.
This establishes a strict improvement for this example, not domination of
every possible earlier correlated certificate.

## 3. Monotone local refinement

**Theorem 4 (refinement).** Replace each mass interval by a nonempty subinterval
and each centered-numerator interval by a nonempty subinterval. For every fixed
boundary $c$, $\mathcal L_j(c)$ cannot decrease and $\mathcal U_j(c)$ cannot
increase. If both the old and new enclosures contain the truth, their
intersection does too.

*Proof.* A minimum over a smaller set cannot be smaller; a maximum cannot be
larger. Equivalently, apply the endpoint multiplication bounds to both new
endpoints, then sum. Intersection contains any point common to its two
operands. $\square$

Lean source: `term_refinement`, `residual_refinement`, `interval_intersection`.

A scheduler may therefore refine selected blocks rather than discard all
summary work when one test fails. The invariant is sound enclosure, not the
particular choice of priority. The implementation's uncertainty priority is a
heuristic, not an optimal scheduling theorem. Monotonicity concerns a fixed
cell; changing the candidate cell changes the question being tested.

No unconditional termination with an open-cell certificate is asserted. A true
output at a rounding midpoint may remain ambiguous forever under interval
refinement. A complete executor needs a tie-aware path or a declared reference
fallback, and a cap on refinement work.

## 4. Projected signed-moment attention enclosures

### 4.1 Stored quantities

Attention is

$$y_j(q)=\frac{\sum_i e^{q^Tk_i}v_{ij}}{\sum_i e^{q^Tk_i}}.$$

The query already includes attention scaling. Keys are those actually attended
to after positional transformations. Let $\mu_b,\nu_b$ be exact block means,
$\delta_i=k_i-\mu_b$, and $x_i=v_i-\nu_b$.

Select $r\leq d$ coordinates. The implementation uses a coordinate prefix;
permutation can express another fixed subset. It does not silently substitute
an approximate PCA basis. Split

$$q^T\delta_i=t_i+e_i,
\quad t_i=u^T\delta_i^{(r)},
\quad |t_i|\leq\rho_b,
\quad |e_i|\leq\epsilon_b.$$

Store coordinate radii $a_{bk}\geq\max_{i\in b}|\delta_{ik}|$ and value radii
$R_{bj}\geq\max_{i\in b}|x_{ij}|$. Valid query bounds are

$$\rho_b=\sum_{k<r}|q_k|a_{bk},
\qquad\epsilon_b=\sum_{k\geq r}|q_k|a_{bk}.$$

Store

$$\Sigma_b=\operatorname{mean}(\delta^{(r)}\delta^{(r)T}),
\quad C_b=\operatorname{mean}(\delta^{(r)}x^T),
\quad H_{bj}=\operatorname{mean}(\delta^{(r)}\delta^{(r)T}x_j).$$

The full $H$ tensor is temporary construction data, not query metadata. Retain
its diagonal $D_{bj}$ and a scalar witness

$$\eta_{bj}\geq\max_a\sum_k |(H_{bj}-D_{bj})_{ak}|.$$

Here $D_{bj}$ denotes the diagonal matrix or its stored diagonal as appropriate.
The choice $D=0$ is also valid. Keeping a diagonal need not improve every bound:
the norm of the residual matrix is not universally smaller than the norm of the
original matrix. The implementation exposes both representations.

**Lemma 5 (quadratic witness).** If $F$ is symmetric and
$\eta\geq\max_a\sum_k|F_{ak}|$, then

$$|u^TFu|\leq\eta\|u\|_2^2.$$

*Proof.* Use $|u_a u_k|\leq(u_a^2+u_k^2)/2$. Symmetry makes the row and column
contributions equal, hence

$$|u^TFu|\leq\tfrac12\sum_{a,k}|F_{ak}|(u_a^2+u_k^2)
=\sum_a u_a^2\sum_k|F_{ak}|\leq\eta\sum_a u_a^2.\quad\square$$

This proof avoids numerical eigenvalue computations. Its finite-dimensional
matrix instantiation is not yet in the Lean source.

### 4.2 A sharper Taylor majorant

Define

$$\kappa(\rho)=
\begin{cases}
(e^\rho-1-\rho-\rho^2/2)/\rho^2,&\rho>0,\\
0,&\rho=0.
\end{cases}$$

**Lemma 6 (second-order exponential remainder).** For $\rho\geq0$ and
$|t|\leq\rho$,

$$|e^t-1-t-t^2/2|\leq\kappa(\rho)t^2,
\qquad\kappa(\rho)\leq e^\rho\rho/6.$$

The latter inequality is strict for $\rho>0$. The first coefficient is the
smallest uniform quadratic majorant on this interval: $t=\rho$ attains equality.

*Proof.* At $\rho=0$, $t=0$. Otherwise absolute convergence of the exponential
series gives

$$\left|\sum_{n\geq3}\frac{t^n}{n!}\right|
\leq t^2\sum_{n\geq3}\frac{\rho^{n-2}}{n!}
=\kappa(\rho)t^2.$$

Writing $n=m+3$ and using $(m+3)!\geq6m!$ yields the second inequality by
termwise comparison with $(\rho/6)e^\rho$. For $m\geq1$ the factorial inequality
is strict and $\rho^{m+1}>0$. $\square$

The exact-real inequality is sharper than the earlier Lagrange bound. Near zero,
the implementation evaluates a series rather than subtracting nearly equal
floating values. That implementation is numerical unless an outward remainder
is included. The rational path uses an upper enclosure of $e^\rho$.

Lean source includes only the exact decomposition (`exp_decomposition`) and
conditional lifting (`centered_remainder_bound`), not the analytic proof above.

### 4.3 The enclosure theorem

Set $a_b=q^T\mu_b$, $w_b=n_b e^{a_b-s}$ for any common finite shift $s$, and

$$\sigma_b^2=u^T\Sigma_bu,\qquad
A_b=1+\sigma_b^2/2,\qquad\tau_b=\kappa(\rho_b)\sigma_b^2.$$

For the projected scores, define

$$
\begin{aligned}
\widetilde l_b&=w_b\max\{1,A_b-\tau_b\},&
\widetilde u_b&=w_b(A_b+\tau_b),\\
\widehat m_{bj}&=w_b\left[(C_b^Tu)_j+\tfrac12 u^TD_{bj}u\right],&
\beta_{bj}&=w_b\left[\tfrac12\eta_{bj}\|u\|_2^2+\tau_bR_{bj}\right].
\end{aligned}
$$

**Theorem 7 (full-score block enclosures).** The following are sound enclosures
for the actual, unprojected attention block:

$$
\boxed{l_b=e^{-\epsilon_b}\widetilde l_b,
\qquad u_b=e^{\epsilon_b}\widetilde u_b,}
$$

$$
\boxed{m^\pm_{bj}=\widehat m_{bj}\ \pm\
\left[\beta_{bj}+(e^{\epsilon_b}-1)\widetilde u_bR_{bj}\right].}
$$

*Proof.* Centering gives $\operatorname{mean}(t_i)=0$,
$\operatorname{mean}(t_i^2)=\sigma_b^2$ and $\operatorname{mean}(x_i)=0$.
By Lemma 6, the mean absolute remainder is at most $\tau_b$. Consequently the
projected mass is in $w_b[A_b-\tau_b,A_b+\tau_b]$. Jensen's inequality also
makes it at least $w_b$, proving the asserted projected mass interval.

The exact projected centered numerator is

$$w_b\left[(C_b^Tu)_j+\tfrac12u^TH_{bj}u+
\operatorname{mean}(r_i x_{ij})\right],$$

where $r_i=e^{t_i}-1-t_i-t_i^2/2$. Lemma 5 bounds the omitted quadratic form;
$|\operatorname{mean}(r_i x_{ij})|\leq\tau_bR_{bj}$ bounds the other omission.
This proves the projected centered interval.

Restoring discarded coordinates multiplies each positive projected weight by
$e^{e_i}\in[e^{-\epsilon_b},e^{\epsilon_b}]$, giving the mass enclosure.
Moreover $|e^{e_i}-1|\leq e^{\epsilon_b}-1$, so the absolute change in the
centered numerator is at most

$$(e^{\epsilon_b}-1)R_{bj}\sum_{i\in b}e^{a_b-s+t_i}
\leq(e^{\epsilon_b}-1)\widetilde u_bR_{bj}.$$

Add this perturbation to the projected interval. $\square$

Lean source covers the scalar expansion algebra (`mass_expansion`,
`central_expansion`) and conditional weight perturbations
(`positive_weight_perturbation`, `central_weight_perturbation`). The end-to-end
instantiation of Theorem 7 is not yet formalized. A theorem taking its desired
error bound as a hypothesis would not close that gap.

The cases $r=0$ and $r=d$ are included. At $r=0$, the retained centered moment
vanishes and all score variation is bounded as discarded variation. At $r=d$,
$\epsilon_b=0$ and no projection inflation is paid. Neither projection nor a
fixed rank is assumed to be useful on arbitrary trained-model activations.

## 5. Rational checking and outward export

`cmk/rational.py` evaluates the summaries with exact Python fractions and
bounded exponential series. For $0\leq x\leq1/2$, after summing through degree
$n$, the first omitted term is $a_{n+1}=x^{n+1}/(n+1)!$. Every subsequent term
ratio is at most $q=x/(n+2)<1$, so the tail is at most $a_{n+1}/(1-q)$.
The partial sum and partial sum plus this tail enclose $e^x$. Positive interval
squaring handles range reduction; reciprocal intervals handle negative inputs.
All arithmetic in this procedure is rational. The implementation restricts its
input domain rather than returning an unbounded or silently approximate result.

A direct rational exponential-sum oracle does not use moment formulas. Block
refinement intersects its interval result with the existing enclosure. Exact
fractions are expensive; these routines are correctness artifacts, not the
proposed fast GPU implementation.

`cmk/export.py` converts each rational endpoint outward to binary64 and compares
the converted float's exact rational value against the original. This also
exports intervals for the centers, whose rational means may not be exactly
representable. The host checker conservatively expands basic operations by one
ULP. The CUDA draft uses directed-rounding arithmetic. Their validity still
requires the stated floating-point semantics and faithful compilation. Neither
program was extracted from Lean; the CUDA program was not executed.

It is invalid to pass the ordinary NumPy/CUDA moment estimates to this checker
and assume that directed rounding at the final sum repairs missing uncertainty
in the input metadata. Sound inputs are a necessary premise.

## 6. Generalization to consumers and smooth gates

Theorems 2 through 4 require only positive normalized weights, not exponentials,
attention heads, or a particular network. A new weight function must supply its
own sound block envelopes. Signed, unnormalized contractions can instead sum
intervals directly.

For a smooth gate $\phi$ and scalar deviations $\alpha,\gamma$, set
$p=\phi(t+\alpha)$, $p_0=\phi(t)$, and $p_1=\phi'(t)$. Then

$$p(s+\gamma)-(p_0s+p_1\alpha s+p_0\gamma)
=p_1\alpha\gamma+(p-p_0-p_1\alpha)(s+\gamma).$$

If $|\alpha|\leq\rho_g$, $|\gamma|\leq\rho_u$, and a Taylor witness supplies
$|p-p_0-p_1\alpha|\leq M\rho_g^2/2$, the triangle inequality gives

$$|\mathrm{error}|\leq |p_1|\rho_g\rho_u+
\tfrac12 M\rho_g^2(|s|+\rho_u).$$

Lean source: `gated_residual_identity`, `gated_remainder_bound`. The analytic
curvature bound for a concrete gate is an additional obligation.

For SwiGLU, this identity applies to each neuron before the down-projection.
Summing downstream weights times these local errors gives a componentwise
output bound. Contracted coefficients can avoid a hidden intermediate, but may
cost more storage than the original weights. No useful general feed-forward
compression or dense-GEMM acceleration is established here.

For an argmax consumer, $l_w>u_j$ for every $j\ne w$ proves the winner $w$.
For a deterministic stateful network, equality of each complete state transition
implies equality of repeated execution. These statements do not turn local
approximation bounds into whole-model equality.

Lean source: `strict_argmax`, `transition_composition`.

## 7. Cost model and implementation architecture

For full coordinate radii, projected covariance, cross moments, retained
diagonals, and scalar witnesses, the numerical representation holds

$$B(2d+r^2+2rh+3h+1)$$

floating scalars. This count excludes original K/V, fallback index maps, scratch,
and allocator overhead. In an implementation that shares the block scalars
across output coordinates, query arithmetic is

$$O\bigl(B(d+r^2+rh+h)\bigr).$$

A direct single-query scan of the explicit original arrays costs
$\Theta(N(d+h))$. Thus successful summary queries admit an arithmetic ratio
scaling as $N(d+h)/[B(d+r^2+rh+h)]$, conditional on the same $B,r$, valid metadata,
and sufficiently separated observation boundaries. This is a word/arithmetic
cost analysis, not a rational bit-complexity theorem or a hardware timing proof.

The initial CUDA draft repeats some scalar work per output channel. Its current
operation count is worse than the ideal shared-work expression. Redesigning it
into a block-scalar pass and a coalesced channel pass is an optimization task,
not an optimization already measured.

The direct construction currently forms $H$ and costs roughly $O(Nr^2h)$.
Clustering is not included in the experiments; blocks are supplied. Prefix
reuse, immutable pages, and a small exact recent-token tail are plausible
amortization strategies, not implemented performance results. A general
streaming compiler must additionally track mutations, masks, precision, and
summary identity.

The method has at least three logical stages: construction/maintenance,
query-and-certificate evaluation, and selective refinement or fallback. Some
stages may be fused; a single stateless replacement kernel cannot create free
reusable summaries. Worst-case fallback reads all keys and values and pays
additional screening overhead. A policy should bypass the method when it is
not profitable.

## 8. Executed evidence and counterexamples

The [measurement artifact](../results/experiments.json) uses synthetic arrays
with $N=8192,d=16,h=8,B=16,Q=24$ and one BLAS thread. The key distribution is
intentionally favorable to a four-coordinate representation except in the
negative control. Full-rank and rank-four summary arrays occupy 72,832 and
17,536 bytes respectively, a 4.15-fold reduction in those arrays alone.

In the tight case all 24 queries pass the numerical full-output screen without
scanning original tokens. With one broad block, the initial screen passes none;
selective refinement scans exactly one of 16 equal-size blocks for every query.
The broad negative control requires all blocks. These numerical experiments
show fewer queried tokens in a constructed regime, not calibrated real-model
certification rates.

Python screening with full fallback is slower than the dense NumPy baseline in
every recorded configuration. It includes exact-fraction candidate rounding,
which dominates these small CPU workloads. Setup-inclusive batch ratios are
also recorded. No CPU, GPU, or end-to-end speedup is inferred from smaller
metadata or fewer scanned tokens.

Ten pytest groups include 500 randomized numerical envelope cases, 100
vertex-enumeration cases, 30 rational attention cases, 1,000 outward-conversion
cases, tail inequalities, invalid inputs, and BF16 midpoint tests. These are
finite tests, not substitutes for universal proofs. A separate host experiment
compares 160 rationally generated output rows: both rational and conservative
host checkers certify 74, with no false host certificates. The shared numerical
C++ core matches its NumPy fixture within $2.83\cdot10^{-16}$ maximum scaled
error on 48 coordinates. CUDA remains untested.

## 9. Relation to prior work and claim boundary

IO-aware attention reorganizes exact attention execution [1]. Clustered and
multipole attention already combines compact approximations with selected
exact work [2,8]. Polynomial attention representations and symmetry-aware
features already change sequence-length scaling [3,4]. Adaptive exact
predicates and Taylor models supply a long history of numerical filters and
bounded remainders [5,9]. Runtime certificates for attention memory, including
Lean artifacts and local fallback policies, are explicitly addressed in
WitCert and subsequent work [6,7]. Compact page-local key summaries are also
recent prior art [10]. COBS retains compressed second-order block statistics
for selection [11], and SPLA combines second-order selection with a linear
representation of unselected blocks [12].

The candidate contribution here is the particular combination of a
boundary-residual interface, projected signed-diagonal moment envelopes, and
selective refinement under a strict observation contract. The residual box
extrema are elementary interval/linear optimization, not a newly discovered
mathematical principle. A literature search cannot establish absence of all
prior equivalent constructions. Novelty, real-model usefulness, full
formalization, and GPU profitability are unresolved in this release.

## References

[1] T. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness*. 2022. https://arxiv.org/abs/2205.14135

[2] C. Hooper et al. *Multipole Attention for Efficient Long Context Reasoning*.
2025. https://arxiv.org/abs/2506.13059

[3] T. C. Nauen, S. Palacio, and A. Dengel. *TaylorShift: Shifting the Complexity
of Self-Attention from Squared to Linear (and Back) using Taylor-Softmax*.
2024. https://arxiv.org/abs/2403.02920

[4] F. A. Heinsen and L. Kozachkov. *Self-Attention at Constant Cost per Token via
Symmetry-Aware Taylor Approximation*. 2026. https://arxiv.org/abs/2602.00294

[5] J. R. Shewchuk. *Adaptive Precision Floating-Point Arithmetic and Fast Robust
Geometric Predicates*. Discrete & Computational Geometry 18, 305-363, 1997.
https://www.cs.cmu.edu/~quake/robust.html

[6] F. Wei et al. *WitCert: Sound Runtime Risk Observability and Gating for
KV-Cache Quantization*. 2026. https://arxiv.org/abs/2607.28699

[7] F. Wei et al. *Runtime Observability for Heterogeneous Attention Memory*.
2026. https://arxiv.org/abs/2608.05863

[8] Y. Kang, G. Tran, and H. De Sterck. *Fast Multipole Attention: A Scalable
Multilevel Attention Mechanism for Text and Images*.
2023. https://arxiv.org/abs/2310.11960

[9] M. Berz and G. Hoffstätter. *Computation and Application of Taylor Polynomials
with Interval Remainder Bounds*. Reliable Computing 4, 83-97, 1998.
https://doi.org/10.1023/A:1009958918582

[10] J. Hwang. *Locks: Page-Local Compact Key Summaries for Efficient Long-Context
Decoding*. 2026. https://arxiv.org/abs/2607.24555

[11] A. Tian et al. *COBS: Cumulant Order Block Sparse Attention*.
2026. https://arxiv.org/abs/2607.09052

[12] B. Wang et al. *SPLA: Block Sparse Plus Linear Attention for Long Context
Modeling*. 2026. https://arxiv.org/abs/2601.22379
