% filepath: RAO_BALLARD_1999_EXACT_DIFFERENCES.md

# Rao & Ballard (1999) Exact Replication - Differences Document

## KEY METHODOLOGICAL CHANGES

### Change 1: Explicit Error Neurons (CRITICAL)
**Rao & Ballard Architecture:**
- Two neural populations per layer
  - R-neurons (representation): code for features
  - E-neurons (error): code for prediction errors
  - E_i^(L) = R_i^(L) - Σ_j W_{ij}^(L) * R_j^(L+1)

**Your Previous Model:**
- Implicit error computation
- No separate error neuron populations

**New Implementation:**
```matlab
E_L1 = R_L1 - pred_L1;  % Explicit error neurons
E_L2 = R_L2 - pred_L2;
E_L3 = R_L3 - pred_L3;
```

---

### Change 2: Weight Matrices Instead of Scalars (CRITICAL)
**Rao & Ballard:**
- W^(L) is n_L × n_{L+1} weight matrix
- Predictions: pred^(L)_i = Σ_j W^(L)_{ij} * R^(L+1)_j
- Learned via Hebbian rule

**Your Previous Model:**
- Scalar predictions: pred.v = rep.a

**New Implementation:**
```matlab
pred_L2 = R_L3 * W_L2_from_L3';  % Matrix multiplication
pred_L1 = R_L2 * W_L1_from_L2';
```

---

### Change 3: Hebbian Learning Rule (CRITICAL)
**Rao & Ballard:**
- ΔW_{ij}^(L) = -η * π * E_i^(L) * R_j^(L+1)
- Outer product: ΔW = -η * π * E * R^T
- Error signal directly trains weights

**Your Previous Model:**
- Simple error subtraction: ΔW ∝ -error

**New Implementation:**
```matlab
dW = -(eta_W * pi) * (E' * R);  % Outer product (correct Hebbian)
W = W + dW;
```

---

### Change 4: Representation Updates from Free Energy Gradient (IMPORTANT)
**Rao & Ballard:**
- ∂R_i^(L)/∂t = -∂F/∂R_i^(L)
- F = Σ_L Σ_i (E_i^(L))^2 / (2σ_L^2)
- ∂F/∂R_i^(L) = E_i^(L) - Σ_k W_{ki}^(L-1) * E_k^(L-1)

**Your Previous Model:**
- Ad-hoc coupling terms: delta_v = error - coupling*error_below

**New Implementation:**
```matlab
% Gradient descent on free energy
coupling_below = E_below * W_below';
delta_R = E - coupling_below;
R = R - eta_rep * delta_R;
```

---

### Change 5: Separate Error from Precision Weighting (IMPORTANT)
**Rao & Ballard:**
- Errors computed: E_i = R_i - pred_i (pure, no π)
- Precision applied ONLY in learning: ΔW = -η * π * E * R^T
- Representation updates use pure errors: ΔR ∝ E

**Your Previous Model:**
- err.x = π_x * (obs - pred)  [mixing error and precision]

**New Implementation:**
```matlab
E = R - pred;  % Pure error, no π
delta_R = E;   % Update with pure error
dW = -eta_W * pi * (E' * R_above);  % π only in learning
```

---

## Summary Table

| Aspect | Rao & Ballard 1999 | Your Previous | New Implementation |
|--------|------------------|---|---|
| **Error Representation** | Explicit E-neurons | Implicit errors | Explicit E_L1, E_L2, E_L3 |
| **Predictions** | W^(L) * R^(L+1) | Scalar functions | W_L * R_L^T (matrix) |
| **Learning Rule** | ΔW = -η*π*E*R^T | ΔW ∝ -error | Hebbian outer product |
| **Rep Updates** | ΔR = -∂F/∂R | Ad-hoc coupling | Gradient descent on F |
| **Precision Role** | Learning only | Mixed in errors | Separated (π in ΔW) |
| **Free Energy** | F = Σ E^2/(2σ^2) | Implicit | Explicit minimization |
| **Architecture** | Feedforward + Feedback | Scalar hierarchy | Full matrix hierarchy |

---

## Biological Interpretation

**Rao & Ballard (1999) shows:**
1. Feedback connections ONLY carry predictions (not errors)
2. Feedforward connections carry ERROR signals
3. This explains extra-classical receptive fields (suppressive surrounds)
4. V1 learns oriented filters naturally
5. Higher areas learn coarse features

**Your Motion Model Now:**
1. Feedback carries velocity/acceleration predictions
2. Feedforward carries position/velocity errors
3. Error neurons show "surprise" about motion
4. Network learns motion dynamics through experience
5. Scales to 2D/3D motion naturally

---

## References
- Rao, R. P., & Ballard, D. H. (1999). Nature Neuroscience, 2(1), 79-87.
- Sections: Methods (pp. 81-83), especially "Feedback and feedforward connections"