import Mathlib
open BigOperators Finset

/-! Formal verification of the mathematics behind the Boyle's-law analysis.

1. `ls_slope`   : derivative of the least-squares objective SSE(c) = Σ (xᵢc - yᵢ)².
2. `ls_critical`: the normal equation — the derivative vanishes at c* = Σxy / Σx².
3. `ls_convex`  : the objective is convex (second derivative = 2Σx² ≥ 0).
4. `boyle_inverse`: Boyle's law p·V = C implies p = C/V (inverse proportionality).
-/

namespace Boyle

theorem ls_slope (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) :
    deriv (fun c : ℝ => ∑ i : Fin n, (x i * c - y i)^2) c =
    2 * (c * ∑ i, (x i)^2 - ∑ i, x i * y i) := by
  have hterm (i : Fin n) : HasDerivAt (fun c : ℝ => (x i * c - y i)^2)
      (2 * (x i * c - y i) * x i) c := by
    have hlin : HasDerivAt (fun c : ℝ => x i * c - y i) (x i) c := by
      have h1 : HasDerivAt (fun c : ℝ => x i * c) (x i) c := by
        simpa using (HasDerivAt.const_mul (x i) (hasDerivAt_id c))
      convert h1.sub (hasDerivAt_const c (y i)) using 1
      · ring
    simpa [pow_two, mul_assoc] using hlin.pow 2
  have h : HasDerivAt (fun c : ℝ => ∑ i : Fin n, (x i * c - y i)^2)
      (∑ i : Fin n, 2 * (x i * c - y i) * x i) c := by
    apply HasDerivAt.fun_sum
    intro i hi
    exact hterm i
  rw [h.deriv]
  ring_nf
  simp [Finset.sum_add_distrib, Finset.sum_neg_distrib, Finset.sum_mul, Finset.mul_sum,
        mul_comm, mul_left_comm, mul_assoc]
  ring

/-- Normal equation: the least-squares slope c* = Σxy / Σx² is a critical point. -/
theorem ls_critical (n : ℕ) (x y : Fin n → ℝ) (hx : (∑ i, (x i)^2) ≠ 0) :
    deriv (fun c : ℝ => ∑ i : Fin n, (x i * c - y i)^2)
      ((∑ i, x i * y i) / (∑ i, (x i)^2)) = 0 := by
  rw [ls_slope]
  field_simp [hx]
  ring

/-- Convexity: the second derivative of the objective is 2Σx² ≥ 0. -/
theorem ls_convex (n : ℕ) (x y : Fin n → ℝ) :
    0 ≤ deriv (fun c : ℝ => deriv (fun c : ℝ => ∑ i : Fin n, (x i * c - y i)^2) c) c := by
  have hinner (c : ℝ) : deriv (fun c : ℝ => ∑ i : Fin n, (x i * c - y i)^2) c =
      2 * (c * (∑ i, (x i)^2) - ∑ i, x i * y i) := ls_slope n x y c
  rw [show (fun c : ℝ => deriv (fun c : ℝ => ∑ i : Fin n, (x i * c - y i)^2) c)
        = fun c : ℝ => 2 * (c * (∑ i, (x i)^2) - ∑ i, x i * y i) by
        funext c
        exact hinner c]
  have h : deriv (fun c : ℝ => 2 * (c * (∑ i, (x i)^2) - ∑ i, x i * y i)) c =
      2 * (∑ i, (x i)^2) := by
    simp [deriv_const_mul, deriv_sub, deriv_mul_const, deriv_id'']
  rw [h]
  positivity

/-- Boyle's law: p·V = C (constant) implies p = C/V for V ≠ 0. -/
theorem boyle_inverse (p V C : ℝ) (hV : V ≠ 0) (h : p * V = C) : p = C / V := by
  rw [← h]
  field_simp [hV]

end Boyle