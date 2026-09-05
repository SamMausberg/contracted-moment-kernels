/- SPDX-License-Identifier: Apache-2.0 -/
import CMK.Attention

namespace CMK

/-- Exact real block data and its query-time metadata. Coordinates may be empty. -/
structure MomentBlock (ι κ : Type*) where
  tokens : Finset ι
  coordinates : Finset κ
  query : κ → ℝ
  centeredKeys : ι → κ → ℝ
  discardedScores : ι → ℝ
  centeredValues : ι → ℝ
  retainedTensor : κ → κ → ℝ
  scoreRadius : ℝ
  discardedRadius : ℝ
  valueRadius : ℝ
  tensorRadius : ℝ
  offset : ℝ
  valueMean : ℝ

namespace MomentBlock
variable {ι κ : Type*}

noncomputable def score (B : MomentBlock ι κ) (i : ι) : ℝ :=
  projectedScore B.coordinates B.query B.centeredKeys i
noncomputable def fullScore (B : MomentBlock ι κ) (i : ι) : ℝ :=
  B.offset + (B.score i + B.discardedScores i)
noncomputable def exactMass (B : MomentBlock ι κ) : ℝ :=
  ∑ i ∈ B.tokens, Real.exp (B.fullScore i)
noncomputable def exactCenter (B : MomentBlock ι κ) : ℝ :=
  ∑ i ∈ B.tokens, Real.exp (B.fullScore i) * B.centeredValues i
noncomputable def lowerMass (B : MomentBlock ι κ) : ℝ :=
  Real.exp B.offset * (Real.exp (-B.discardedRadius) * projectedLower B.tokens B.score B.scoreRadius)
noncomputable def upperMass (B : MomentBlock ι κ) : ℝ :=
  Real.exp B.offset * (Real.exp B.discardedRadius * projectedUpper B.tokens B.score B.scoreRadius)
noncomputable def centerEstimate (B : MomentBlock ι κ) : ℝ :=
  Real.exp B.offset * centerApprox B.tokens B.coordinates B.query B.centeredKeys
    B.centeredValues B.retainedTensor
noncomputable def centerRadius (B : MomentBlock ι κ) : ℝ :=
  Real.exp B.offset * (centerError B.tokens B.coordinates B.query B.centeredKeys
    B.scoreRadius B.valueRadius B.tensorRadius +
    (Real.exp B.discardedRadius - 1) * projectedUpper B.tokens B.score B.scoreRadius * B.valueRadius)
noncomputable def lowerCenter (B : MomentBlock ι κ) : ℝ := B.centerEstimate - B.centerRadius
noncomputable def upperCenter (B : MomentBlock ι κ) : ℝ := B.centerEstimate + B.centerRadius

/-- These are concrete centering, coordinate, value, symmetry, and row-sum conditions. -/
structure Witness (B : MomentBlock ι κ) : Prop where
  nonempty : B.tokens.Nonempty
  scoreRadius_nonneg : 0 ≤ B.scoreRadius
  discardedRadius_nonneg : 0 ≤ B.discardedRadius
  valueRadius_nonneg : 0 ≤ B.valueRadius
  keys_centered : ∀ k ∈ B.coordinates, ∑ i ∈ B.tokens, B.centeredKeys i k = 0
  values_centered : ∑ i ∈ B.tokens, B.centeredValues i = 0
  score_bound : ∀ i ∈ B.tokens, |B.score i| ≤ B.scoreRadius
  discarded_bound : ∀ i ∈ B.tokens, |B.discardedScores i| ≤ B.discardedRadius
  value_bound : ∀ i ∈ B.tokens, |B.centeredValues i| ≤ B.valueRadius
  retained_symmetric : ∀ a ∈ B.coordinates, ∀ k ∈ B.coordinates,
    B.retainedTensor a k = B.retainedTensor k a
  row_sum_bound : ∀ a ∈ B.coordinates, ∑ k ∈ B.coordinates,
    |signedSecondMoment B.tokens B.centeredKeys B.centeredValues a k - B.retainedTensor a k| ≤
      B.tensorRadius

end MomentBlock

variable {ι κ β : Type*}

/-- The common score offset scales the fully derived block certificate. -/
theorem moment_block_enclosure (B : MomentBlock ι κ) (hB : B.Witness) :
    B.lowerMass ≤ B.exactMass ∧ B.exactMass ≤ B.upperMass ∧
      B.lowerCenter ≤ B.exactCenter ∧ B.exactCenter ≤ B.upperCenter := by
  have h := full_score_enclosure B.tokens B.coordinates B.query B.centeredKeys
    B.discardedScores B.centeredValues B.retainedTensor B.scoreRadius B.discardedRadius
    B.valueRadius B.tensorRadius hB.scoreRadius_nonneg hB.discardedRadius_nonneg
    hB.valueRadius_nonneg hB.keys_centered hB.values_centered hB.score_bound
    hB.discarded_bound hB.value_bound hB.retained_symmetric hB.row_sum_bound
  dsimp only at h
  have hz : B.exactMass = Real.exp B.offset *
      (∑ i ∈ B.tokens, Real.exp (B.score i + B.discardedScores i)) := by
    simp [MomentBlock.exactMass, MomentBlock.fullScore, Real.exp_add, Finset.mul_sum]
  have hm : B.exactCenter = Real.exp B.offset *
      (∑ i ∈ B.tokens, Real.exp (B.score i + B.discardedScores i) * B.centeredValues i) := by
    simp [MomentBlock.exactCenter, MomentBlock.fullScore, Real.exp_add, Finset.mul_sum, mul_assoc]
  have hw := Real.exp_nonneg B.offset
  rw [hz, hm]
  refine ⟨mul_le_mul_of_nonneg_left h.1 hw, mul_le_mul_of_nonneg_left h.2.1 hw, ?_, ?_⟩
  · simpa [MomentBlock.lowerCenter, MomentBlock.centerEstimate, MomentBlock.centerRadius,
      MomentBlock.score, mul_sub] using mul_le_mul_of_nonneg_left h.2.2.1 hw
  · simpa [MomentBlock.upperCenter, MomentBlock.centerEstimate, MomentBlock.centerRadius,
      MomentBlock.score, mul_add] using mul_le_mul_of_nonneg_left h.2.2.2 hw

theorem moment_block_mass_positive (B : MomentBlock ι κ) (hB : B.tokens.Nonempty) :
    0 < B.exactMass := by
  obtain ⟨i, hi⟩ := hB
  exact Finset.sum_pos' (fun j _ => Real.exp_nonneg _) ⟨i, hi, Real.exp_pos _⟩

/-- The algebraic block numerator equals the original per-token weighted value sum. -/
theorem moment_block_numerator (B : MomentBlock ι κ) :
    B.valueMean * B.exactMass + B.exactCenter =
      ∑ i ∈ B.tokens, Real.exp (B.fullScore i) * (B.valueMean + B.centeredValues i) := by
  simp only [MomentBlock.exactMass, MomentBlock.exactCenter, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- The full real-attention quotient has a strict observation certificate from concrete block witnesses. -/
theorem full_attention_observation (blocks : Finset β) (B : β → MomentBlock ι κ)
    (hne : blocks.Nonempty) (hw : ∀ b ∈ blocks, (B b).Witness) (a z : ℝ)
    (ha : 0 < lowerResidual blocks (fun b => (B b).lowerMass) (fun b => (B b).upperMass)
      (fun b => (B b).lowerCenter) (fun b => (B b).valueMean) a)
    (hz : upperResidual blocks (fun b => (B b).lowerMass) (fun b => (B b).upperMass)
      (fun b => (B b).upperCenter) (fun b => (B b).valueMean) z < 0) :
    let y := (∑ b ∈ blocks, ∑ i ∈ (B b).tokens,
      Real.exp ((B b).fullScore i) * ((B b).valueMean + (B b).centeredValues i)) /
      (∑ b ∈ blocks, ∑ i ∈ (B b).tokens, Real.exp ((B b).fullScore i))
    a < y ∧ y < z := by
  have henc := fun b hb => moment_block_enclosure (B b) (hw b hb)
  have hpos : 0 < mass blocks (fun b => (B b).exactMass) := by
    obtain ⟨b, hb⟩ := hne
    exact Finset.sum_pos' (fun j hj => (moment_block_mass_positive (B j) (hw j hj).nonempty).le)
      ⟨b, hb, moment_block_mass_positive (B b) (hw b hb).nonempty⟩
  have h := observation_cell blocks (fun b => (B b).exactMass) (fun b => (B b).exactCenter)
    (fun b => (B b).valueMean) (fun b => (B b).lowerMass) (fun b => (B b).upperMass)
    (fun b => (B b).lowerCenter) (fun b => (B b).upperCenter)
    (fun b hb => ⟨(henc b hb).1, (henc b hb).2.1⟩)
    (fun b hb => ⟨(henc b hb).2.2.1, (henc b hb).2.2.2⟩) hpos a z ha hz
  simp only [numerator, mass] at h
  simp_rw [moment_block_numerator] at h
  simpa only [MomentBlock.exactMass] using h

end CMK
