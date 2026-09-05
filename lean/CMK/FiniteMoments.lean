/- SPDX-License-Identifier: Apache-2.0 -/
import CMK.Quadratic

namespace CMK
variable {ι κ : Type*}

noncomputable def finiteMean (s : Finset ι) (v : ι → ℝ) : ℝ := (∑ i ∈ s, v i) / s.card

/-- Exact means center a finite block, including the empty-sum convention. -/
theorem finite_mean_centering (s : Finset ι) (v : ι → ℝ) :
    ∑ i ∈ s, (v i - finiteMean s v) = 0 := by
  classical
  by_cases hs : s = ∅
  · subst s; simp
  · have hn : (s.card : ℝ) ≠ 0 := by exact_mod_cast Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr hs)
    simp only [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, finiteMean]
    field_simp
    ring

noncomputable def projectedScore (c : Finset κ) (u : κ → ℝ) (δ : ι → κ → ℝ) (i : ι) : ℝ :=
  ∑ k ∈ c, u k * δ i k

noncomputable def secondMoment (s : Finset ι) (δ : ι → κ → ℝ) (a k : κ) : ℝ :=
  ∑ i ∈ s, δ i a * δ i k

noncomputable def signedSecondMoment (s : Finset ι) (δ : ι → κ → ℝ) (x : ι → ℝ)
    (a k : κ) : ℝ := ∑ i ∈ s, δ i a * δ i k * x i

theorem projected_score_centering (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (hδ : ∀ k ∈ c, ∑ i ∈ s, δ i k = 0) :
    ∑ i ∈ s, projectedScore c u δ i = 0 := by
  unfold projectedScore
  rw [Finset.sum_comm]
  apply Finset.sum_eq_zero
  intro k hk
  rw [← Finset.mul_sum, hδ k hk, mul_zero]

theorem second_moment_symmetric (s : Finset ι) (δ : ι → κ → ℝ) (a k : κ) :
    secondMoment s δ a k = secondMoment s δ k a := by
  unfold secondMoment
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem signed_second_moment_symmetric (s : Finset ι) (δ : ι → κ → ℝ)
    (x : ι → ℝ) (a k : κ) :
    signedSecondMoment s δ x a k = signedSecondMoment s δ x k a := by
  unfold signedSecondMoment
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- Contracting the actual finite signed tensor gives the quadratic score moment. -/
theorem signed_moment_contraction (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (x : ι → ℝ) :
    ∑ i ∈ s, (projectedScore c u δ i) ^ 2 * x i =
      quadraticForm c (signedSecondMoment s δ x) u := by
  unfold projectedScore quadraticForm signedSecondMoment
  simp only [pow_two, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a ha
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem second_moment_contraction (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) :
    ∑ i ∈ s, (projectedScore c u δ i) ^ 2 = quadraticForm c (secondMoment s δ) u := by
  simpa [quadraticForm, signedSecondMoment, secondMoment] using signed_moment_contraction s c u δ (fun _ => 1)

theorem linear_moment_contraction (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (x : ι → ℝ) :
    ∑ i ∈ s, projectedScore c u δ i * x i =
      ∑ k ∈ c, u k * ∑ i ∈ s, δ i k * x i := by
  unfold projectedScore
  simp only [Finset.sum_mul, Finset.mul_sum, mul_assoc]
  exact Finset.sum_comm

theorem quadratic_sub (c : Finset κ) (H D : κ → κ → ℝ) (u : κ → ℝ) :
    quadraticForm c H u - quadraticForm c D u =
      quadraticForm c (fun a k => H a k - D a k) u := by
  simp only [quadraticForm, mul_sub, sub_mul, Finset.sum_sub_distrib]

/-- A symmetric retained tensor and an explicit row-sum witness bound omitted moments. -/
theorem signed_moment_omission_bound (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (x : ι → ℝ) (D : κ → κ → ℝ) (η : ℝ)
    (hD : ∀ a ∈ c, ∀ k ∈ c, D a k = D k a)
    (hη : ∀ a ∈ c, ∑ k ∈ c, |signedSecondMoment s δ x a k - D a k| ≤ η) :
    |(∑ i ∈ s, (projectedScore c u δ i) ^ 2 * x i) - quadraticForm c D u| ≤
      η * ∑ k ∈ c, (u k) ^ 2 := by
  rw [signed_moment_contraction, quadratic_sub]
  apply quadratic_rowsum_bound c _ u η _ hη
  intro a ha k hk
  rw [signed_second_moment_symmetric s δ x a k, hD a ha k hk]

end CMK
