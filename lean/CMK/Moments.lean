/- SPDX-License-Identifier: Apache-2.0
   The analytic Taylor inequality is an explicit input contract here.
   This file does not claim to derive it from Real.exp's power series.
-/
import Mathlib

namespace CMK
variable {ι : Type*}

noncomputable def expRemainder2 (t : ℝ) : ℝ :=
  Real.exp t - (1 + t + t^2/2)

theorem exp_decomposition (t : ℝ) :
    Real.exp t = 1 + t + t^2/2 + expRemainder2 t := by
  unfold expRemainder2
  ring

theorem mass_expansion (s : Finset ι) (t : ι → ℝ)
    (ht : s.sum t = 0) :
    s.sum (fun i => Real.exp (t i)) = (s.card : ℝ) +
      s.sum (fun i => (t i)^2)/2 + s.sum (fun i => expRemainder2 (t i)) := by
  calc
    s.sum (fun i => Real.exp (t i)) =
        s.sum (fun i => 1 + t i + (t i)^2/2 + expRemainder2 (t i)) := by
      apply Finset.sum_congr rfl
      intro i hi
      exact exp_decomposition (t i)
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_div, Finset.sum_const,
        nsmul_eq_mul, mul_one, ht]
      ring

/-- Signed second moments are retained before absolute-value bounds are taken. -/
theorem central_expansion (s : Finset ι) (t v : ι → ℝ) (c : ℝ)
    (hv : s.sum (fun i => v i-c) = 0) :
    s.sum (fun i => Real.exp (t i) * (v i-c)) =
      s.sum (fun i => t i * (v i-c)) +
      s.sum (fun i => (t i)^2 * (v i-c))/2 +
      s.sum (fun i => expRemainder2 (t i) * (v i-c)) := by
  calc
    s.sum (fun i => Real.exp (t i) * (v i-c)) =
        s.sum (fun i => (v i-c) + t i * (v i-c) +
          (t i)^2 * (v i-c)/2 + expRemainder2 (t i) * (v i-c)) := by
      apply Finset.sum_congr rfl
      intro i hi
      unfold expRemainder2
      ring
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_div, hv]
      ring

/-- Contract lifting: a scalar remainder witness bounds a centered weighted term. -/
theorem centered_remainder_bound (r t v c K R : ℝ)
    (hK : 0 ≤ K) (hR : 0 ≤ R)
    (hr : |r| ≤ K*t^2) (hv : |v-c| ≤ R) :
    |r*(v-c)| ≤ K*t^2*R := by
  rw [abs_mul]
  exact mul_le_mul hr hv (abs_nonneg _) (mul_nonneg hK (sq_nonneg t))

/-- The generic smooth-gate algebra also covers SwiGLU after analytic witnesses. -/
theorem gated_residual_identity (p p0 p1 a s b : ℝ) :
    p*(s+b) - (p0*s+p1*a*s+p0*b) =
      p1*a*b + (p-p0-p1*a)*(s+b) := by ring

theorem gated_remainder_bound (p p0 p1 a s b rg ru E : ℝ)
    (hga : |a| ≤ rg) (hub : |b| ≤ ru) (hE : 0 ≤ E)
    (hr : |p-p0-p1*a| ≤ E) :
    |p*(s+b)-(p0*s+p1*a*s+p0*b)| ≤
      |p1|*rg*ru + E*(|s|+ru) := by
  have hrg : 0 ≤ rg := (abs_nonneg a).trans hga
  have hru : 0 ≤ ru := (abs_nonneg b).trans hub
  rw [gated_residual_identity]
  calc
    |p1*a*b+(p-p0-p1*a)*(s+b)| ≤
        |p1*a*b|+|(p-p0-p1*a)*(s+b)| := abs_add _ _
    _ = |p1|*|a|*|b| + |p-p0-p1*a|*|s+b| := by simp only [abs_mul]
    _ ≤ |p1|*rg*ru + E*(|s|+ru) := by
      apply add_le_add
      · exact mul_le_mul (mul_le_mul_of_nonneg_left hga (abs_nonneg p1)) hub
          (abs_nonneg b) (mul_nonneg (abs_nonneg p1) hrg)
      · have hs : |s+b| ≤ |s|+ru := (abs_add s b).trans (add_le_add_left hub _)
        exact mul_le_mul hr hs (abs_nonneg _) hE

end CMK
