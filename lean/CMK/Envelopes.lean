/- SPDX-License-Identifier: Apache-2.0
   All witnesses below are explicit hypotheses, not new axioms.
-/
import Mathlib

namespace CMK

variable {ι : Type*}

noncomputable def lowerTerm (a l u : ℝ) : ℝ := min (a * l) (a * u)
noncomputable def upperTerm (a l u : ℝ) : ℝ := max (a * l) (a * u)

theorem mul_enclosure (a l u z : ℝ) (hl : l ≤ z) (hu : z ≤ u) :
    lowerTerm a l u ≤ a * z ∧ a * z ≤ upperTerm a l u := by
  by_cases ha : 0 ≤ a
  · exact ⟨(min_le_left _ _).trans (mul_le_mul_of_nonneg_left hl ha),
      (mul_le_mul_of_nonneg_left hu ha).trans (le_max_right _ _)⟩
  · have hn : a ≤ 0 := le_of_lt (lt_of_not_ge ha)
    exact ⟨(min_le_right _ _).trans (mul_le_mul_of_nonpos_left hu hn),
      (mul_le_mul_of_nonpos_left hl hn).trans (le_max_left _ _)⟩

noncomputable def mass (s : Finset ι) (z : ι → ℝ) : ℝ := s.sum z
noncomputable def numerator (s : Finset ι) (z m v : ι → ℝ) : ℝ :=
  s.sum (fun i => v i * z i + m i)
noncomputable def lowerResidual (s : Finset ι) (l u ml v : ι → ℝ) (a : ℝ) : ℝ :=
  s.sum (fun i => ml i + lowerTerm (v i - a) (l i) (u i))
noncomputable def upperResidual (s : Finset ι) (l u mu v : ι → ℝ) (a : ℝ) : ℝ :=
  s.sum (fun i => mu i + upperTerm (v i - a) (l i) (u i))

theorem residual_identity (s : Finset ι) (z m v : ι → ℝ) (a : ℝ) :
    numerator s z m v - a * mass s z =
      s.sum (fun i => m i + (v i - a) * z i) := by
  unfold numerator mass
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

theorem residual_enclosure (s : Finset ι) (z m v l u ml mu : ι → ℝ)
    (hz : ∀ i ∈ s, l i ≤ z i ∧ z i ≤ u i)
    (hm : ∀ i ∈ s, ml i ≤ m i ∧ m i ≤ mu i) (a : ℝ) :
    lowerResidual s l u ml v a ≤ numerator s z m v - a * mass s z ∧
    numerator s z m v - a * mass s z ≤ upperResidual s l u mu v a := by
  rw [residual_identity]
  constructor
  · apply Finset.sum_le_sum
    intro i hi
    exact add_le_add (hm i hi).1
      (mul_enclosure (v i - a) (l i) (u i) (z i) (hz i hi).1 (hz i hi).2).1
  · apply Finset.sum_le_sum
    intro i hi
    exact add_le_add (hm i hi).2
      (mul_enclosure (v i - a) (l i) (u i) (z i) (hz i hi).1 (hz i hi).2).2

/-- Division-free strict observation-cell certificate. -/
theorem observation_cell (s : Finset ι) (z m v l u ml mu : ι → ℝ)
    (hz : ∀ i ∈ s, l i ≤ z i ∧ z i ≤ u i)
    (hm : ∀ i ∈ s, ml i ≤ m i ∧ m i ≤ mu i)
    (hpos : 0 < mass s z) (a b : ℝ)
    (ha : 0 < lowerResidual s l u ml v a)
    (hb : upperResidual s l u mu v b < 0) :
    a < numerator s z m v / mass s z ∧
      numerator s z m v / mass s z < b := by
  have hla := (residual_enclosure s z m v l u ml mu hz hm a).1
  have hub := (residual_enclosure s z m v l u ml mu hz hm b).2
  constructor
  · apply (lt_div_iff₀ hpos).2
    linarith
  · apply (div_lt_iff₀ hpos).2
    linarith

noncomputable def lowerChoice (a l u : ℝ) : ℝ := if 0 ≤ a then l else u
noncomputable def upperChoice (a l u : ℝ) : ℝ := if 0 ≤ a then u else l

theorem lowerChoice_mem (a l u : ℝ) (h : l ≤ u) :
    l ≤ lowerChoice a l u ∧ lowerChoice a l u ≤ u := by
  unfold lowerChoice
  split_ifs <;> exact ⟨by linarith, by linarith⟩

theorem upperChoice_mem (a l u : ℝ) (h : l ≤ u) :
    l ≤ upperChoice a l u ∧ upperChoice a l u ≤ u := by
  unfold upperChoice
  split_ifs <;> exact ⟨by linarith, by linarith⟩

theorem lowerChoice_attains (a l u : ℝ) (h : l ≤ u) :
    a * lowerChoice a l u = lowerTerm a l u := by
  unfold lowerChoice lowerTerm
  by_cases ha : 0 ≤ a
  · rw [if_pos ha, min_eq_left (mul_le_mul_of_nonneg_left h ha)]
  · have hn : a ≤ 0 := le_of_lt (lt_of_not_ge ha)
    rw [if_neg ha, min_eq_right (mul_le_mul_of_nonpos_left h hn)]

theorem upperChoice_attains (a l u : ℝ) (h : l ≤ u) :
    a * upperChoice a l u = upperTerm a l u := by
  unfold upperChoice upperTerm
  by_cases ha : 0 ≤ a
  · rw [if_pos ha, max_eq_right (mul_le_mul_of_nonneg_left h ha)]
  · have hn : a ≤ 0 := le_of_lt (lt_of_not_ge ha)
    rw [if_neg ha, max_eq_left (mul_le_mul_of_nonpos_left h hn)]

/-- The lower residual bound is attained at an admissible box vertex. -/
theorem lower_attained (s : Finset ι) (l u ml v : ι → ℝ)
    (h : ∀ i ∈ s, l i ≤ u i) (a : ℝ) :
    numerator s (fun i => lowerChoice (v i-a) (l i) (u i)) ml v -
      a * mass s (fun i => lowerChoice (v i-a) (l i) (u i)) =
      lowerResidual s l u ml v a := by
  rw [residual_identity]
  unfold lowerResidual
  apply Finset.sum_congr rfl
  intro i hi
  rw [lowerChoice_attains (v i-a) (l i) (u i) (h i hi)]

/-- The upper residual bound is attained at an admissible box vertex. -/
theorem upper_attained (s : Finset ι) (l u mu v : ι → ℝ)
    (h : ∀ i ∈ s, l i ≤ u i) (a : ℝ) :
    numerator s (fun i => upperChoice (v i-a) (l i) (u i)) mu v -
      a * mass s (fun i => upperChoice (v i-a) (l i) (u i)) =
      upperResidual s l u mu v a := by
  rw [residual_identity]
  unfold upperResidual
  apply Finset.sum_congr rfl
  intro i hi
  rw [upperChoice_attains (v i-a) (l i) (u i) (h i hi)]

theorem term_refinement (a l u lp up : ℝ)
    (h1 : l ≤ lp) (h2 : lp ≤ up) (h3 : up ≤ u) :
    lowerTerm a l u ≤ lowerTerm a lp up ∧
      upperTerm a lp up ≤ upperTerm a l u := by
  have hp := mul_enclosure a l u lp h1 (h2.trans h3)
  have hq := mul_enclosure a l u up (h1.trans h2) h3
  exact ⟨le_min hp.1 hq.1, max_le hp.2 hq.2⟩

/-- Nested envelopes improve both residual tests at every fixed boundary. -/
theorem residual_refinement (s : Finset ι) (l u ml mu lp up mlp mup v : ι → ℝ)
    (h : ∀ i ∈ s, l i ≤ lp i ∧ lp i ≤ up i ∧ up i ≤ u i)
    (hm : ∀ i ∈ s, ml i ≤ mlp i ∧ mup i ≤ mu i) (a : ℝ) :
    lowerResidual s l u ml v a ≤ lowerResidual s lp up mlp v a ∧
      upperResidual s lp up mup v a ≤ upperResidual s l u mu v a := by
  constructor
  · apply Finset.sum_le_sum
    intro i hi
    exact add_le_add (hm i hi).1
      (term_refinement (v i-a) (l i) (u i) (lp i) (up i)
        (h i hi).1 (h i hi).2.1 (h i hi).2.2).1
  · apply Finset.sum_le_sum
    intro i hi
    exact add_le_add (hm i hi).2
      (term_refinement (v i-a) (l i) (u i) (lp i) (up i)
        (h i hi).1 (h i hi).2.1 (h i hi).2.2).2

/-- With a positive mass floor, the strict residual tests exactly characterize a box-wide cell. -/
theorem observation_cell_iff (s : Finset ι) (l u ml mu v : ι → ℝ) (a b : ℝ)
    (hlu : ∀ i ∈ s, l i ≤ u i) (hmu : ∀ i ∈ s, ml i ≤ mu i)
    (hfloor : 0 < mass s l) :
    (0 < lowerResidual s l u ml v a ∧ upperResidual s l u mu v b < 0) ↔
      (∀ z m : ι → ℝ,
        (∀ i ∈ s, l i ≤ z i ∧ z i ≤ u i) →
        (∀ i ∈ s, ml i ≤ m i ∧ m i ≤ mu i) →
        a < numerator s z m v / mass s z ∧ numerator s z m v / mass s z < b) := by
  have hpositive (z : ι → ℝ) (hz : ∀ i ∈ s, l i ≤ z i ∧ z i ≤ u i) :
      0 < mass s z :=
    hfloor.trans_le (Finset.sum_le_sum (fun i hi => (hz i hi).1))
  constructor
  · intro h z m hz hm
    exact observation_cell s z m v l u ml mu hz hm (hpositive z hz) a b h.1 h.2
  · intro h
    have hlz : ∀ i ∈ s, l i ≤ lowerChoice (v i - a) (l i) (u i) ∧
        lowerChoice (v i - a) (l i) (u i) ≤ u i :=
      fun i hi => lowerChoice_mem _ _ _ (hlu i hi)
    have huz : ∀ i ∈ s, l i ≤ upperChoice (v i - b) (l i) (u i) ∧
        upperChoice (v i - b) (l i) (u i) ≤ u i :=
      fun i hi => upperChoice_mem _ _ _ (hlu i hi)
    have hl := (h (fun i => lowerChoice (v i - a) (l i) (u i)) ml hlz
      (fun i hi => ⟨le_refl _, hmu i hi⟩)).1
    have hu := (h (fun i => upperChoice (v i - b) (l i) (u i)) mu huz
      (fun i hi => ⟨hmu i hi, le_refl _⟩)).2
    have hl' := (lt_div_iff₀ (hpositive _ hlz)).1 hl
    have hu' := (div_lt_iff₀ (hpositive _ huz)).1 hu
    have hla := lower_attained s l u ml v hlu a
    have hua := upper_attained s l u mu v hlu b
    exact ⟨by linarith, by linarith⟩

end CMK
