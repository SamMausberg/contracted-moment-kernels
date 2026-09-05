/- SPDX-License-Identifier: Apache-2.0 -/
import Mathlib

namespace CMK
variable {ι : Type*}

noncomputable def quadraticForm (s : Finset ι) (F : ι → ι → ℝ) (u : ι → ℝ) : ℝ :=
  ∑ a ∈ s, ∑ k ∈ s, u a * F a k * u k

/-- A finite symmetric row-sum witness controls the whole quadratic form. -/
theorem quadratic_rowsum_bound (s : Finset ι) (F : ι → ι → ℝ) (u : ι → ℝ) (η : ℝ)
    (hF : ∀ a ∈ s, ∀ k ∈ s, F a k = F k a)
    (hη : ∀ a ∈ s, ∑ k ∈ s, |F a k| ≤ η) :
    |quadraticForm s F u| ≤ η * ∑ a ∈ s, (u a) ^ 2 := by
  have hab (a k : ι) : |u a| * |u k| ≤ ((u a) ^ 2 + (u k) ^ 2) / 2 := by
    nlinarith [sq_nonneg (|u a| - |u k|), sq_abs (u a), sq_abs (u k)]
  have hr : (∑ a ∈ s, ∑ k ∈ s, |F a k| * (u a) ^ 2) ≤
      η * ∑ a ∈ s, (u a) ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro a ha
    rw [← Finset.sum_mul]
    simpa [mul_comm] using mul_le_mul_of_nonneg_right (hη a ha) (sq_nonneg (u a))
  have hc : (∑ a ∈ s, ∑ k ∈ s, |F a k| * (u k) ^ 2) =
      ∑ a ∈ s, ∑ k ∈ s, |F a k| * (u a) ^ 2 := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro a ha
    apply Finset.sum_congr rfl
    intro k hk
    rw [hF k hk a ha]
  calc
    |quadraticForm s F u| ≤ ∑ a ∈ s, ∑ k ∈ s, |u a * F a k * u k| := by
      unfold quadraticForm
      exact (Finset.abs_sum_le_sum_abs _ _).trans
        (Finset.sum_le_sum (fun a _ => Finset.abs_sum_le_sum_abs _ _))
    _ ≤ ∑ a ∈ s, ∑ k ∈ s, |F a k| * ((u a) ^ 2 + (u k) ^ 2) / 2 := by
      apply Finset.sum_le_sum
      intro a ha
      apply Finset.sum_le_sum
      intro k hk
      simpa only [abs_mul, mul_comm, mul_left_comm, mul_assoc, mul_div_assoc] using
        mul_le_mul_of_nonneg_left (hab a k) (abs_nonneg (F a k))
    _ ≤ η * ∑ a ∈ s, (u a) ^ 2 := by
      simp only [mul_add, add_div, Finset.sum_add_distrib, ← Finset.sum_div]
      rw [hc]
      linarith

/-- Coordinate radii give a query-dependent score radius. -/
theorem coordinate_radius_bound (s : Finset ι) (u δ r : ι → ℝ)
    (hr : ∀ k ∈ s, |δ k| ≤ r k) :
    |∑ k ∈ s, u k * δ k| ≤ ∑ k ∈ s, |u k| * r k := by
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro k hk
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left (hr k hk) (abs_nonneg (u k))

end CMK
