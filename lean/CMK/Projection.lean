/- SPDX-License-Identifier: Apache-2.0 -/
import Mathlib

namespace CMK
variable {ι : Type*}

/-- Perturb positive weights multiplicatively; no attention-specific assumption. -/
theorem positive_weight_perturbation (s : Finset ι) (w f : ι → ℝ) (l u : ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i)
    (hf : ∀ i ∈ s, l ≤ f i ∧ f i ≤ u) :
    l*s.sum w ≤ s.sum (fun i => w i*f i) ∧
      s.sum (fun i => w i*f i) ≤ u*s.sum w := by
  constructor
  · rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    nlinarith [mul_le_mul_of_nonneg_left (hf i hi).1 (hw i hi)]
  · rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    nlinarith [mul_le_mul_of_nonneg_left (hf i hi).2 (hw i hi)]

/-- Per-term central perturbation. Sum this inequality for block certificates. -/
theorem central_weight_perturbation (w f v c E R : ℝ)
    (hw : 0 ≤ w) (hE : 0 ≤ E) (hf : |f-1| ≤ E) (hv : |v-c| ≤ R) :
    |w*f*(v-c)-w*(v-c)| ≤ w*E*R := by
  have hR : 0 ≤ R := (abs_nonneg _).trans hv
  have hid : w*f*(v-c)-w*(v-c) = w*((f-1)*(v-c)) := by ring
  rw [hid, abs_mul, abs_of_nonneg hw, abs_mul]
  calc
    w*(|f-1|*|v-c|) ≤ w*(E*R) :=
      mul_le_mul_of_nonneg_left
        (mul_le_mul hf hv (abs_nonneg _) hE) hw
    _ = w*E*R := by ring

end CMK
