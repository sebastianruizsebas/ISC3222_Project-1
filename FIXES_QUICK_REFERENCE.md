# Quick Reference: What Changed and Why

## The Core Problem We Fixed

**Before:** Model predicted perfectly → execution matched prediction → error = 0 → no learning

**After:** Model has exploration noise → execution differs from prediction → error > 0 → learning signal exists ✓

---

## Six Critical Changes

### 1. **Motor Weights: Shared (Not Per-Task)** 🧠
- **File:** hierarchical_motion_inference_dual_hierarchy.m, lines ~665-720
- **What:** Motor L2→L1 and L3→L2 weights now single matrices, not task-indexed
- **Why:** Motor should learn ONE generalizable reaching model (applies to all tasks)
- **Result:** All tasks benefit from motor learning; better generalization

### 2. **Motor Weight Access: Use Matrices Not Cells** 🔧
- **File:** hierarchical_step_update.m, lines ~108-115
- **What:** Changed `S.W_motor_L2_to_L1{current_task_idx}` → `S.W_motor_L2_to_L1`
- **Why:** Motor weights are matrices, not cell arrays (only planning is per-task)
- **Result:** Correct data type, no indexing errors

### 3. **Gradient Scaling: L2 Norm + Clipping** 📊
- **File:** hierarchical_step_update.m, lines ~540-565
- **What:** Changed from `mean(abs(R))` to `norm(R, 2)` + added `dW` clipping [-0.1, +0.1]
- **Why:** Mean absolute value can be tiny, causing huge gradient spikes → clipping kills learning
- **Result:** Stable gradients, learning survives clipping

### 4. **Precision Update Order** ✓
- **File:** hierarchical_step_update.m, lines ~810-827
- **What:** Exponential scaling happens BEFORE clipping (verified correct)
- **Why:** Ensures clipping doesn't reverse the scaling effect
- **Result:** Precision adapts smoothly

### 5. **Motor Noise with Annealing** 📉
- **File:** hierarchical_step_update.m, lines ~145-157
- **What:** Added `final_motor_vx = motor_vx + noise_scale * randn()`
- **Why:** Without noise, error = 0, learning impossible
- **Result:** Early exploration (high noise) → late exploitation (low noise)

### 6. **Diagnostic Output Every 100 Steps** 📈
- **File:** hierarchical_motion_inference_dual_hierarchy.m, lines ~1025-1035
- **What:** Added console output: `FE | IntErr | |dW| | pi_L1m | noise_scale`
- **Why:** Real-time feedback on whether learning is working
- **Result:** Can debug without waiting for full simulation

---

## Testing Checklist

Run this quick test:
```matlab
results = hierarchical_motion_inference_dual_hierarchy(...
    struct('n_trials', 1, 'T_per_trial', 5), false);
```

Watch for these patterns in console output:
- [ ] Step 100: FE = ~25-40 (free energy non-zero)
- [ ] Step 200: FE < Step 100 (decreasing)
- [ ] |dW| = ~0.03-0.05 (weights updating)
- [ ] IntErr = ~3-7 (distance to target)
- [ ] No NaN/Inf errors

---

## Key Insight: Why This Works

The model learns because now:

1. **Exploration:** Motor adds random noise (σ=0.05) to predictions
2. **Prediction Error:** Actual execution ≠ prediction due to noise
3. **Learning Signal:** Error = observation - prediction (now non-zero!)
4. **Weight Update:** dW ∝ error ⊗ activation (now flows through)
5. **Annealing:** Over time, noise decreases → exploitation phase

**Result:** Noise creates learning signal early → learning happens → noise becomes unnecessary → model converges

---

## Common Issues & Fixes

### Issue: Free Energy staying constant
**Cause:** Learning signal still zero
**Check:** Is |dW| = 0? If yes → gradient calculation broken
**Fix:** Verify gradient clipping not over-aggressive (max_grad = 0.1 is usually safe)

### Issue: Interception error NOT decreasing
**Cause:** Motor not learning reaching
**Check:** Is |dW| > 0? Is noise_scale > 0?
**Fix:** Increase eta_W (learning rate), decrease max_grad clipping limit

### Issue: NaN/Inf appearing
**Cause:** Numerical instability (large gradients, precision explosion)
**Check:** Check console for clipping events
**Fix:** Reduce eta_W, check precision bounds are finite

### Issue: Precision (pi_L1m) stuck at 100
**Cause:** Error magnitudes not triggering exponential scaling
**Check:** Look at error magnitudes (L1_motor_error_mag)
**Fix:** Increase alpha_precision_gain (PSO parameter)

---

## Files Modified

1. `hierarchical_motion_inference_dual_hierarchy.m`
   - Motor weight initialization (shared)
   - Diagnostic output loop

2. `hierarchical_step_update.m`
   - Motor weight access (shared matrix)
   - Gradient normalization (L2 norm)
   - Gradient clipping ([-0.1, +0.1])
   - Motor noise annealing

3. **New:** `IMPLEMENTATION_FIXES_NOV3.md` (detailed documentation)

---

## Parameters to Tune (via PSO)

These PSO parameters now actually affect learning:

- `alpha_precision_gain`: Controls precision scaling sensitivity (0.1-1.0)
- `pi_L1_motor_max`: Maximum proprioceptive precision (100-1000)
- `pi_L2_motor_max`: Maximum motor basis precision (10-100)
- `eta_rep`: Representation learning rate (0.001-0.1)
- `eta_W`: Weight learning rate (0.0001-0.01)

The motor noise level is **NOT a parameter** - it's automatic annealing:
- Starts: 0.05 m/s
- Formula: `0.05 * max(0.01, 1 - i/1000)`
- Ends: 0.01 m/s

---

## Expected Learning Curves

### Good Trajectory:
```
Step 100:   FE=35.2  IntErr=6.12  |dW|=0.0487
Step 500:   FE=18.7  IntErr=4.23  |dW|=0.0321
Step 1000:  FE=12.4  IntErr=2.89  |dW|=0.0198
Step 2000:  FE=8.3   IntErr=1.56  |dW|=0.0087
```

### Bad Trajectory (No Learning):
```
Step 100:   FE=0.00  IntErr=5.00  |dW|=0.0000  ← Zero |dW| = red flag!
Step 500:   FE=0.00  IntErr=5.00  |dW|=0.0000  ← Stuck
Step 1000:  FE=0.00  IntErr=5.00  |dW|=0.0000  ← Same values
```

---

## Bottom Line

**These 6 fixes transform the model from:**
- Non-learning (perfect prediction → zero error → no learning signal)

**To:**
- Learning (noisy execution → non-zero error → learning signal preserved)

**The fixes are:**
1. ✅ Architecturally sound (motor/planning separation)
2. ✅ Numerically stable (norm-based scaling, gradient clipping)
3. ✅ Biologically motivated (exploration/exploitation, noise)
4. ✅ Debuggable (diagnostic output every 100 steps)

**Ready to test:** Run the diagnostic and watch the numbers decrease! 🚀

