# CODE REVIEW FINDINGS - VISUAL SUMMARY

## 🔴 CRITICAL BUGS (PSO Optimization Blocked)

### Bug #1: Hardcoded alpha_precision_gain
```
hierarchical_step_update.m ~ line 500
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  alpha_precision_gain = 0.5;  ❌ HARDCODED
  
  ↓ PSO sends [0.1, 2.0] range
  ↓ Code ignores it, always uses 0.5
  ↓ Optimization effect: ZERO
```

### Bug #2: Hardcoded Precision Bounds
```
hierarchical_step_update.m ~ lines 525-530
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  S.pi_L1_motor = max(10, min(500, ...))  ❌ HARDCODED
  S.pi_L2_motor = max(0.5, min(50, ...))  ❌ HARDCODED
  S.pi_L1_plan = max(10, min(500, ...))  ❌ HARDCODED
  S.pi_L2_plan = max(0.5, min(50, ...))  ❌ HARDCODED
  
  ↓ PSO sends 8 different bound parameters
  ↓ Code ignores ALL of them, uses fixed values
  ↓ Optimization effect: ZERO
  
  Parameters blocked:
  • pi_L1_motor_min [5-50]
  • pi_L1_motor_max [200-1000]
  • pi_L2_motor_min [0.1-5]
  • pi_L2_motor_max [20-200]
  • pi_L1_plan_min [5-50]
  • pi_L1_plan_max [100-500]
  • pi_L2_plan_min [0.1-5]
  • pi_L2_plan_max [20-100]
```

### Bug #3: Triple Precision Update Mechanisms
```
EXECUTION TIMELINE IN hierarchical_step_update.m:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step i Processing:

1. Line ~515-520: update_pi() called
   └─ Updates from 100-step error history
   └─ Conservative (α=0.999, only 0.1% change)
   └─ Uses variance normalization ✓

2. Line ~525-530: Hardcoded exponential scaling
   └─ Updates immediately from current error
   └─ CONFLICTS with mechanism #1 ❌
   └─ Uses hardcoded alpha=0.5 ❌
   └─ Uses hardcoded bounds ❌

3. Line ~600: update_pi() called AGAIN (for diagnostics)
   └─ Updates AGAIN from history
   └─ Double update! ❌

4. Line ~650-700: Error-driven scaling (CORRECT VERSION)
   └─ Updates immediately from current error
   └─ Uses P.alpha_precision_gain ✓
   └─ Uses P.pi_bounds ✓
   └─ CONFLICTS with mechanism #2 ❌

5. Line ~730: Bounds clipping (after all updates)
   └─ Final clipping from hardcoded bounds ❌

RESULT: Three competing adaptation strategies all modifying
        precision on same step. Behavior is chaotic.
```

---

## 🟠 HIGH-PRIORITY BUGS

### Bug #4: Missing Parameter in P Struct
```
hierarchical_motion_inference_dual_hierarchy.m ~ line 715
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Missing:
  P.interference_penalty_weight = params.interference_penalty_weight;
  
  Impact:
  • Parameter read from params (line ~50) ✓
  • But NOT passed to helper function ❌
  • Helper falls back to hardcoded 0.01 ❌
  • PSO optimization of this parameter: PARTIALLY BLOCKED
```

---

## 🟡 MEDIUM-PRIORITY ISSUES

### Issue #5: Inconsistent Velocity Blending
```
hierarchical_step_update.m ~ lines 180-210
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motor region blending:
  motor_cmd = 0.6 * DESIRED + 0.4 * LEARNED

Planning region blending:
  plan_cmd = 0.4 * DESIRED + 0.6 * LEARNED

Question: Why different ratios?
• Undocumented
• No parameters to control
• Might be intentional (motor more constrained, planning more free)
• But currently hardcoded with no justification
```

### Issue #6: Inconsistent NaN/Inf Thresholds
```
Two different clipping thresholds used:
  • Debug section (line ~365): max_finite_value = 1e8
  • Main FE section (line ~625): 1e7 (lower)

Which is authoritative? Should be unified.
```

---

## ✅ WHAT'S WORKING CORRECTLY

### Task-Selective Weight Updates
```
hierarchical_step_update.m ~ lines 350-380
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Only current_task_idx weights updated
✓ Off-task weights remain frozen
✓ Correctly indexed into cell arrays
✓ Proper task gating applied
✓ Implementation is clean and correct
```

### Multiplicative Task Gating
```
hierarchical_step_update.m ~ lines 140-160
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ L0 task context correctly extracted
✓ Task-specific weights loaded from cells
✓ Multiplicative gating applied correctly
✓ task_gate values in valid range [0,1]
✓ Predictions suppressed for off-task appropriately
```

### Cross-Task Error Monitoring
```
hierarchical_step_update.m ~ lines 240-260
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Computes prediction errors for ALL tasks
✓ Correctly loops over all task-specific weights
✓ Stores in task_errors_motor/plan arrays
✓ Used for interference penalty (when enabled)
```

---

## 📊 IMPACT ANALYSIS

### PSO Parameter Effectiveness
```
19-D PSO Parameter Space:

Learning Rates (2):
  η_rep, η_W                                    ✓ WORKING

Task-Conditional (1):
  interference_penalty_weight                   ⚠️ PARTIAL (needs P fix)

Motor Dynamics (3):
  motor_gain, damping, reaching_speed_scale     ✓ WORKING

Weight Initialization (2):
  W_motor_gain, W_plan_gain                     ✓ WORKING

Phase Decay (2):
  decay_motor, decay_plan                       ✓ WORKING

Momentum/Regularization (2):
  momentum, weight_decay                        ✓ WORKING

Adaptive Precision (4):
  alpha_precision_gain                          ❌ BLOCKED
  (+ 8 precision bounds)                        ❌ BLOCKED

SUMMARY:
  12 parameters working correctly
  1 parameter partially blocked
  6 parameters completely blocked
  
  Optimization effectiveness: ~65% (should be 100%)
```

---

## 🔧 FIX CHECKLIST

- [ ] Remove lines ~515-530 (first hardcoded exp scaling)
- [ ] Update lines ~680-700 to use P.alpha_precision_gain
- [ ] Update lines ~700-710 to use P.pi_bounds instead of hardcoded
- [ ] Add P.interference_penalty_weight assignment (line ~715)
- [ ] Test PSO with alpha_precision_gain sweep
- [ ] Test PSO with precision bounds sweep
- [ ] Verify precision traces show expected exponential shape
- [ ] Compare results before/after fix

---

## 📋 DOCUMENTATION CREATED

1. **CRITICAL_CODE_REVIEW_NOV_2.md** — Detailed findings with examples (8 findings)
2. **QUICK_FIX_SUMMARY.md** — One-page action items
3. **CODE_REVIEW_VISUAL_SUMMARY.md** — This file (diagrams and overview)

---

**Review Date:** November 2, 2025  
**Reviewer:** Code Analysis Assistant  
**Status:** ⚠️ CRITICAL - PSO optimization currently ineffective due to hardcoded parameters
