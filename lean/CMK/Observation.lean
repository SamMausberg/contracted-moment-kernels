/- SPDX-License-Identifier: Apache-2.0 -/
import Mathlib

namespace CMK
variable {ι : Type*} {α : Type*} {β : Type*}

/-- A consumer constant on a sound enclosure requires no further producer work. -/
theorem constant_consumer (S : Set α) (C : α → β) (y : α) (c : β)
    (hy : y ∈ S) (hC : ∀ x ∈ S, C x = c) : C y = c := hC y hy

/-- Applies to monotone numerical rounding; it does not model signed-zero bits. -/
theorem monotone_rounding (Q : ℝ → ℝ) (hQ : Monotone Q) (l y u : ℝ)
    (hl : l ≤ y) (hu : y ≤ u) (heq : Q l = Q u) : Q y = Q l := by
  apply le_antisymm
  · calc Q y ≤ Q u := hQ hu
         _ = Q l := heq.symm
  · exact hQ hl

/-- Separated component intervals prove a unique winning index. -/
theorem strict_argmax (y l u : ι → ℝ) (winner : ι)
    (hl : ∀ i, l i ≤ y i) (hu : ∀ i, y i ≤ u i)
    (hgap : ∀ i, i ≠ winner → u i < l winner) :
    ∀ i, i ≠ winner → y i < y winner := by
  intro i hi
  exact (hu i).trans_lt ((hgap i hi).trans_le (hl winner))

/-- Each sound refinement may be intersected without losing the true value. -/
theorem interval_intersection (y l u lp up : ℝ)
    (h : l ≤ y ∧ y ≤ u) (hp : lp ≤ y ∧ y ≤ up) :
    max l lp ≤ y ∧ y ≤ min u up := by
  exact ⟨max_le h.1 hp.1, le_min h.2 hp.2⟩

/-- Local exact state transitions compose; approximate equality is not enough. -/
theorem transition_composition (f g : α → α) (h : ∀ x, g x = f x)
    (n : ℕ) (x : α) : (g^[n]) x = (f^[n]) x := by
  have hfg : g = f := funext h
  rw [hfg]

end CMK
