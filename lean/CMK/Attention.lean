/- SPDX-License-Identifier: Apache-2.0 -/
import CMK.Analytic
import CMK.FiniteMoments
import CMK.Projection
import CMK.Envelopes

namespace CMK
variable {ι κ : Type*}

noncomputable def projectedMass (s : Finset ι) (t : ι → ℝ) : ℝ := ∑ i ∈ s, Real.exp (t i)
noncomputable def massApprox (s : Finset ι) (t : ι → ℝ) : ℝ :=
  s.card + (∑ i ∈ s, (t i) ^ 2) / 2
noncomputable def massError (s : Finset ι) (t : ι → ℝ) (ρ : ℝ) : ℝ :=
  expQuadraticCoefficient ρ * ∑ i ∈ s, (t i) ^ 2
noncomputable def projectedLower (s : Finset ι) (t : ι → ℝ) (ρ : ℝ) : ℝ :=
  max s.card (massApprox s t - massError s t ρ)
noncomputable def projectedUpper (s : Finset ι) (t : ι → ℝ) (ρ : ℝ) : ℝ :=
  massApprox s t + massError s t ρ

/-- The exponential tangent inequality supplies the centered mass floor. -/
theorem centered_mass_lower (s : Finset ι) (t : ι → ℝ) (ht : ∑ i ∈ s, t i = 0) :
    (s.card : ℝ) ≤ projectedMass s t := by
  have h := Finset.sum_le_sum (fun i (_ : i ∈ s) => Real.add_one_le_exp (t i))
  simpa [projectedMass, Finset.sum_add_distrib, ht] using h

theorem summed_exp_remainder (s : Finset ι) (t : ι → ℝ) (ρ : ℝ)
    (ht : ∀ i ∈ s, |t i| ≤ ρ) :
    |∑ i ∈ s, expRemainder2 (t i)| ≤ massError s t ρ := by
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  unfold massError
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum (fun i hi => exp_remainder_sharp (t i) ρ (ht i hi))

theorem summed_centered_remainder (s : Finset ι) (t x : ι → ℝ) (ρ R : ℝ)
    (hρ : 0 ≤ ρ) (hR : 0 ≤ R)
    (ht : ∀ i ∈ s, |t i| ≤ ρ) (hx : ∀ i ∈ s, |x i| ≤ R) :
    |∑ i ∈ s, expRemainder2 (t i) * x i| ≤ massError s t ρ * R := by
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  unfold massError
  rw [Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_le_sum
  intro i hi
  simpa using centered_remainder_bound (expRemainder2 (t i)) (t i) (x i) 0
    (expQuadraticCoefficient ρ) R (exp_coefficient_nonneg ρ hρ) hR
    (exp_remainder_sharp (t i) ρ (ht i hi)) (by simpa using hx i hi)

/-- A complete scalar exponential-to-mass enclosure, with the centered floor. -/
theorem projected_mass_enclosure (s : Finset ι) (t : ι → ℝ) (ρ : ℝ)
    (hcenter : ∑ i ∈ s, t i = 0) (ht : ∀ i ∈ s, |t i| ≤ ρ) :
    projectedLower s t ρ ≤ projectedMass s t ∧ projectedMass s t ≤ projectedUpper s t ρ := by
  have hr := abs_le.mp (summed_exp_remainder s t ρ ht)
  have he := mass_expansion s t hcenter
  have hj := centered_mass_lower s t hcenter
  unfold projectedLower projectedUpper massApprox projectedMass at *
  exact ⟨max_le hj (by linarith), by linarith⟩

noncomputable def centerApprox (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (x : ι → ℝ) (D : κ → κ → ℝ) : ℝ :=
  (∑ i ∈ s, projectedScore c u δ i * x i) + quadraticForm c D u / 2
noncomputable def centerError (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (ρ R η : ℝ) : ℝ :=
  (η * ∑ k ∈ c, (u k) ^ 2) / 2 + massError s (projectedScore c u δ) ρ * R

/-- Signed tensor omission and the proved exponential tail give the centered enclosure. -/
theorem projected_center_enclosure (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (x : ι → ℝ) (D : κ → κ → ℝ) (ρ R η : ℝ)
    (hρ : 0 ≤ ρ) (hR : 0 ≤ R)
    (hcenter : ∑ i ∈ s, x i = 0)
    (ht : ∀ i ∈ s, |projectedScore c u δ i| ≤ ρ) (hx : ∀ i ∈ s, |x i| ≤ R)
    (hD : ∀ a ∈ c, ∀ k ∈ c, D a k = D k a)
    (hη : ∀ a ∈ c, ∑ k ∈ c, |signedSecondMoment s δ x a k - D a k| ≤ η) :
    |(∑ i ∈ s, Real.exp (projectedScore c u δ i) * x i) - centerApprox s c u δ x D| ≤
      centerError s c u δ ρ R η := by
  have hq := abs_le.mp (signed_moment_omission_bound s c u δ x D η hD hη)
  have hr := abs_le.mp (summed_centered_remainder s (projectedScore c u δ) x ρ R hρ hR ht hx)
  have he := central_expansion s (projectedScore c u δ) x 0 (by simpa using hcenter)
  simp only [sub_zero] at he
  unfold centerApprox centerError
  apply abs_le.mpr
  constructor <;> linarith

/-- Discarded coordinates are restored by monotonicity of their positive multipliers. -/
theorem full_mass_enclosure (s : Finset ι) (t e : ι → ℝ) (ρ ε : ℝ)
    (hcenter : ∑ i ∈ s, t i = 0)
    (ht : ∀ i ∈ s, |t i| ≤ ρ) (he : ∀ i ∈ s, |e i| ≤ ε) :
    Real.exp (-ε) * projectedLower s t ρ ≤ (∑ i ∈ s, Real.exp (t i + e i)) ∧
      (∑ i ∈ s, Real.exp (t i + e i)) ≤ Real.exp ε * projectedUpper s t ρ := by
  have hp := projected_mass_enclosure s t ρ hcenter ht
  have hw := positive_weight_perturbation s (fun i => Real.exp (t i))
    (fun i => Real.exp (e i)) (Real.exp (-ε)) (Real.exp ε)
    (fun i _ => Real.exp_nonneg _) (fun i hi => exp_discarded_bounds (e i) ε (he i hi))
  simp only [← Real.exp_add] at hw
  exact ⟨(mul_le_mul_of_nonneg_left hp.1 (Real.exp_nonneg _)).trans hw.1,
    hw.2.trans (mul_le_mul_of_nonneg_left hp.2 (Real.exp_nonneg _))⟩

/-- The full centered numerator differs from its projected value by a controlled sum. -/
theorem discarded_center_perturbation (s : Finset ι) (t e x : ι → ℝ) (ε R U : ℝ)
    (hε : 0 ≤ ε) (hR : 0 ≤ R)
    (he : ∀ i ∈ s, |e i| ≤ ε) (hx : ∀ i ∈ s, |x i| ≤ R)
    (hU : projectedMass s t ≤ U) :
    |(∑ i ∈ s, Real.exp (t i + e i) * x i) - (∑ i ∈ s, Real.exp (t i) * x i)| ≤
      (Real.exp ε - 1) * U * R := by
  have hE : 0 ≤ Real.exp ε - 1 := sub_nonneg.mpr (Real.one_le_exp hε)
  rw [← Finset.sum_sub_distrib]
  calc
    |∑ i ∈ s, (Real.exp (t i + e i) * x i - Real.exp (t i) * x i)| ≤
        ∑ i ∈ s, |Real.exp (t i + e i) * x i - Real.exp (t i) * x i| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ s, Real.exp (t i) * (Real.exp ε - 1) * R := by
      apply Finset.sum_le_sum
      intro i hi
      simpa [Real.exp_add] using central_weight_perturbation (Real.exp (t i))
        (Real.exp (e i)) (x i) 0 (Real.exp ε - 1) R (Real.exp_nonneg _) hE
        (exp_discarded_distance (e i) ε (he i hi)) (by simpa using hx i hi)
    _ ≤ (Real.exp ε - 1) * U * R := by
      rw [← Finset.sum_mul, ← Finset.sum_mul]
      have hu : (∑ i ∈ s, Real.exp (t i)) * (Real.exp ε - 1) ≤ (Real.exp ε - 1) * U := by
        simpa [projectedMass, mul_comm] using mul_le_mul_of_nonneg_right hU hE
      exact mul_le_mul_of_nonneg_right hu hR

/-- Integrated full-score block theorem: every analytic and finite-tensor error is derived. -/
theorem full_score_enclosure (s : Finset ι) (c : Finset κ) (u : κ → ℝ)
    (δ : ι → κ → ℝ) (e x : ι → ℝ) (D : κ → κ → ℝ) (ρ ε R η : ℝ)
    (hρ : 0 ≤ ρ) (hε : 0 ≤ ε) (hR : 0 ≤ R)
    (hδ : ∀ k ∈ c, ∑ i ∈ s, δ i k = 0) (hcenter : ∑ i ∈ s, x i = 0)
    (ht : ∀ i ∈ s, |projectedScore c u δ i| ≤ ρ)
    (he : ∀ i ∈ s, |e i| ≤ ε) (hx : ∀ i ∈ s, |x i| ≤ R)
    (hD : ∀ a ∈ c, ∀ k ∈ c, D a k = D k a)
    (hη : ∀ a ∈ c, ∑ k ∈ c, |signedSecondMoment s δ x a k - D a k| ≤ η) :
    let t := projectedScore c u δ
    let Z := ∑ i ∈ s, Real.exp (t i + e i)
    let M := ∑ i ∈ s, Real.exp (t i + e i) * x i
    let B := centerError s c u δ ρ R η + (Real.exp ε - 1) * projectedUpper s t ρ * R
    Real.exp (-ε) * projectedLower s t ρ ≤ Z ∧
      Z ≤ Real.exp ε * projectedUpper s t ρ ∧
      centerApprox s c u δ x D - B ≤ M ∧ M ≤ centerApprox s c u δ x D + B := by
  dsimp only
  have htc := projected_score_centering s c u δ hδ
  have hm := full_mass_enclosure s (projectedScore c u δ) e ρ ε htc ht he
  have hc := projected_center_enclosure s c u δ x D ρ R η hρ hR hcenter ht hx hD hη
  have hp := discarded_center_perturbation s (projectedScore c u δ) e x ε R
    (projectedUpper s (projectedScore c u δ) ρ) hε hR he hx
    (projected_mass_enclosure s (projectedScore c u δ) ρ htc ht).2
  have hca := abs_le.mp hc
  have hpa := abs_le.mp hp
  exact ⟨hm.1, hm.2, by linarith, by linarith⟩

end CMK
