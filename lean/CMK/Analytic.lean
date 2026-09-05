/- SPDX-License-Identifier: Apache-2.0 -/
import CMK.Moments

namespace CMK

/-- Sharp uniform quadratic coefficient, including its continuous zero value. -/
noncomputable def expQuadraticCoefficient (ρ : ℝ) : ℝ := expRemainder2 ρ / ρ ^ 2

theorem exp_tail_hasSum (t : ℝ) :
    HasSum (fun n : ℕ => t ^ (n + 3) / (n + 3).factorial) (expRemainder2 t) := by
  have h := (hasSum_nat_add_iff' 3).2 (NormedSpace.expSeries_div_hasSum_exp ℝ t)
  simpa [← Real.exp_eq_exp_ℝ, Finset.sum_range_succ, expRemainder2, Nat.factorial] using h

theorem exp_coefficient_hasSum (t : ℝ) :
    HasSum (fun n : ℕ => t ^ (n + 1) / (n + 3).factorial)
      (expQuadraticCoefficient t) := by
  by_cases ht : t = 0
  · subst t
    simpa [expQuadraticCoefficient, expRemainder2] using (hasSum_zero : HasSum (fun _ : ℕ => (0 : ℝ)) 0)
  · have h := (exp_tail_hasSum t).div_const (t ^ 2)
    apply h.congr_fun
    intro n
    dsimp
    have hf : ((n + 3).factorial : ℝ) ≠ 0 := by positivity
    field_simp
    ring

theorem exp_remainder_factor (t : ℝ) :
    expRemainder2 t = t ^ 2 * expQuadraticCoefficient t := by
  by_cases ht : t = 0
  · subst t; simp [expRemainder2, expQuadraticCoefficient]
  · unfold expQuadraticCoefficient
    field_simp

theorem exp_coefficient_nonneg (ρ : ℝ) (hρ : 0 ≤ ρ) :
    0 ≤ expQuadraticCoefficient ρ := by
  rw [← (exp_coefficient_hasSum ρ).tsum_eq]
  exact tsum_nonneg (fun n => by positivity)

theorem exp_coefficient_abs_le (t ρ : ℝ) (h : |t| ≤ ρ) :
    |expQuadraticCoefficient t| ≤ expQuadraticCoefficient ρ := by
  have hs := (exp_coefficient_hasSum t).summable.norm
  rw [← (exp_coefficient_hasSum t).tsum_eq, ← (exp_coefficient_hasSum ρ).tsum_eq]
  calc
    |∑' n : ℕ, t ^ (n + 1) / (n + 3).factorial| ≤
        ∑' n : ℕ, ‖t ^ (n + 1) / (n + 3).factorial‖ := norm_tsum_le_tsum_norm hs
    _ ≤ ∑' n : ℕ, ρ ^ (n + 1) / (n + 3).factorial := by
      apply Summable.tsum_le_tsum _ hs (exp_coefficient_hasSum ρ).summable
      intro n
      simp only [norm_div, Real.norm_eq_abs, abs_pow, Nat.abs_cast]
      exact div_le_div_of_nonneg_right (pow_le_pow_left₀ (abs_nonneg t) h _) (by positivity)

/-- The scalar sharp Taylor inequality follows from the actual exponential series. -/
theorem exp_remainder_sharp (t ρ : ℝ) (h : |t| ≤ ρ) :
    |expRemainder2 t| ≤ expQuadraticCoefficient ρ * t ^ 2 := by
  rw [exp_remainder_factor t, abs_mul, abs_of_nonneg (sq_nonneg t)]
  simpa [mul_comm] using mul_le_mul_of_nonneg_left (exp_coefficient_abs_le t ρ h) (sq_nonneg t)

/-- The endpoint attains the sharp bound. -/
theorem exp_remainder_endpoint (ρ : ℝ) (hρ : 0 ≤ ρ) :
    |expRemainder2 ρ| = expQuadraticCoefficient ρ * ρ ^ 2 := by
  rw [exp_remainder_factor, abs_mul, abs_of_nonneg (sq_nonneg ρ),
    abs_of_nonneg (exp_coefficient_nonneg ρ hρ), mul_comm]

/-- No smaller coefficient works uniformly on a nondegenerate symmetric interval. -/
theorem exp_coefficient_optimal (ρ K : ℝ) (hρ : 0 < ρ)
    (hK : ∀ t : ℝ, |t| ≤ ρ → |expRemainder2 t| ≤ K * t ^ 2) :
    expQuadraticCoefficient ρ ≤ K := by
  have h := hK ρ (by simp [abs_of_pos hρ])
  rw [exp_remainder_endpoint ρ hρ.le] at h
  exact (mul_le_mul_iff_left₀ (sq_pos_of_pos hρ)).1 h

/-- Elementary factorial comparison for the exponential majorant. -/
theorem factorial_shift_three (n : ℕ) :
    (6 : ℝ) * n.factorial ≤ (n + 3).factorial := by
  have hn : (0 : ℝ) ≤ n := by positivity
  have hf : (0 : ℝ) < n.factorial := by positivity
  simp only [Nat.factorial_succ, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
  norm_num
  calc
    (6 : ℝ) * n.factorial = 3 * (2 * (1 * n.factorial)) := by ring
    _ ≤ (↑n + 2 + 1) * ((↑n + 1 + 1) * ((↑n + 1) * ↑n.factorial)) := by gcongr <;> linarith

theorem exp_coefficient_term_le (ρ : ℝ) (hρ : 0 ≤ ρ) (n : ℕ) :
    ρ ^ (n + 1) / (n + 3).factorial ≤ (ρ / 6) * (ρ ^ n / n.factorial) := by
  have h := div_le_div_of_nonneg_left (pow_nonneg hρ (n + 1))
    (by positivity : (0 : ℝ) < 6 * n.factorial) (factorial_shift_three n)
  calc
    ρ ^ (n + 1) / (n + 3).factorial ≤ ρ ^ (n + 1) / (6 * n.factorial) := h
    _ = _ := by rw [pow_succ]; ring

/-- The sharp coefficient is bounded by the usual Lagrange coefficient. -/
theorem exp_coefficient_le_lagrange (ρ : ℝ) (hρ : 0 ≤ ρ) :
    expQuadraticCoefficient ρ ≤ Real.exp ρ * ρ / 6 := by
  have hs := (NormedSpace.expSeries_div_hasSum_exp ℝ ρ).mul_left (ρ / 6)
  rw [← Real.exp_eq_exp_ℝ] at hs
  have h := Summable.tsum_le_tsum (exp_coefficient_term_le ρ hρ)
    (exp_coefficient_hasSum ρ).summable hs.summable
  rw [(exp_coefficient_hasSum ρ).tsum_eq, hs.tsum_eq] at h
  nlinarith

/-- The improvement over the Lagrange coefficient is strict away from zero. -/
theorem exp_coefficient_lt_lagrange (ρ : ℝ) (hρ : 0 < ρ) :
    expQuadraticCoefficient ρ < Real.exp ρ * ρ / 6 := by
  have hs := (NormedSpace.expSeries_div_hasSum_exp ℝ ρ).mul_left (ρ / 6)
  rw [← Real.exp_eq_exp_ℝ] at hs
  have hi : ρ ^ (1 + 1) / (1 + 3).factorial < (ρ / 6) * (ρ ^ 1 / Nat.factorial 1) := by
    norm_num
    nlinarith [sq_pos_of_pos hρ]
  have h := Summable.tsum_lt_tsum (exp_coefficient_term_le ρ hρ.le) hi
    (exp_coefficient_hasSum ρ).summable hs.summable
  rw [(exp_coefficient_hasSum ρ).tsum_eq, hs.tsum_eq] at h
  nlinarith

/-- Restoring a bounded discarded score multiplies positive weights within this interval. -/
theorem exp_discarded_bounds (e ε : ℝ) (he : |e| ≤ ε) :
    Real.exp (-ε) ≤ Real.exp e ∧ Real.exp e ≤ Real.exp ε := by
  exact ⟨Real.exp_le_exp.mpr (abs_le.mp he).1, Real.exp_le_exp.mpr (abs_le.mp he).2⟩

/-- The same score bound controls distance from the unit multiplier. -/
theorem exp_discarded_distance (e ε : ℝ) (he : |e| ≤ ε) :
    |Real.exp e - 1| ≤ Real.exp ε - 1 := by
  have hε : 0 ≤ ε := (abs_nonneg e).trans he
  have hp : 0 < Real.exp ε := Real.exp_pos ε
  have hproduct : Real.exp ε * Real.exp (-ε) = 1 := by rw [← Real.exp_add]; simp
  have hsum : 2 ≤ Real.exp ε + Real.exp (-ε) := by
    nlinarith [sq_nonneg (Real.exp ε - 1), Real.exp_pos (-ε)]
  have hb := exp_discarded_bounds e ε he
  exact abs_le.mpr ⟨by linarith, by linarith⟩

end CMK
