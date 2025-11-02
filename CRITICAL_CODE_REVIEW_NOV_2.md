# CRITICAL CODE REVIEW: hierarchical_step_update.m & hierarchical_motion_inference_dual_hierarchy.m
**Date:** November 2, 2025  
**Status:** ⚠️ MULTIPLE INCONSISTENCIES & ERRORS FOUND

---

## EXECUTIVE SUMMARY

**2 MAJOR ERRORS + 4 INCONSISTENCIES IDENTIFIED:**

1. ❌ **CRITICAL**: Hardcoded precision scaling in `hierarchical_step_update.m` (lines ~500-530) conflicts with PSO-optimized parameters
2. ❌ **CRITICAL**: Duplicate and competing precision update mechanisms (3 different systems!)
3. ⚠️ **ERROR**: `alpha_precision_gain` is hardcoded as 0.5 instead of using `P.alpha_precision_gain`
4. ⚠️ **ERROR**: Static precision bounds (10, 500, 1, 100) hardcoded instead of using `P_pi_bounds`
5. ⚠️ **INCONSISTENCY**: Precision scaling applied twice (exponential + update_pi)
6. ⚠️ **INCONSISTENCY**: Task interference penalty not passed to P struct

---

## DETAILED FINDINGS

### 1. HARDCODED ALPHA PRECISION GAIN (❌ CRITICAL)

**Location:** `hierarchical_step_update.m`, lines ~500-510

```matlab
% BROKEN: Hardcoded value instead of using PSO parameter
alpha_precision_gain = 0.5;  % <-- WRONG! Should be P.alpha_precision_gain

precision_scale_L1_motor = exp(alpha_precision_gain * min(E_L1_motor_mag, 5.0));
```

**Location:** `hierarchical_motion_inference_dual_hierarchy.m`, lines ~285-310
- Parameter is properly initialized with PSO values
- Passed to `P` struct at line ~710

**CONFLICT:** 
- Main script correctly reads from `params.alpha_precision_gain` (with default 0.5)
- Helper function **ignores** `P.alpha_precision_gain` and always uses hardcoded 0.5
- PSO optimization of this parameter is **completely ineffective**

**FIX REQUIRED:**
```matlab
% hierarchical_step_update.m, line ~500
alpha_precision_gain = P.alpha_precision_gain;  % ✓ Use passed parameter
```

---

### 2. HARDCODED PRECISION BOUNDS (❌ CRITICAL)

**Location:** `hierarchical_step_update.m`, lines ~525-530

```matlab
% BROKEN: Hardcoded bounds instead of using P.pi_bounds
S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));     % <-- WRONG!
S.pi_L2_motor = max(0.5, min(50, S.pi_L2_motor));
S.pi_L1_plan = max(10, min(500, S.pi_L1_plan));
S.pi_L2_plan = max(0.5, min(50, S.pi_L2_plan));
```

**Location:** `hierarchical_motion_inference_dual_hierarchy.m`, lines ~285-310
- Bounds are read from `params` with proper defaults
- Stored in `P_pi_bounds` struct
- Passed to `P` struct at line ~715

**CONFLICT:**
- Main script correctly reads bounds (e.g., `pi_L1_motor_min = 10`, `pi_L1_motor_max = 500`)
- Helper function **ignores** `P.pi_bounds` and uses **fixed** bounds
- PSO optimization of bounds is **completely ineffective**
- Even debug bounds mentioned (lines ~650-655) are then overwritten!

**Evidence of conflicting logic:**
```matlab
% Line ~650 (debug clip attempt - WRONG!)
S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));

% Line ~730+ (error-driven scaling - uses exponential)
S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;

% Line ~525 (ANOTHER clip - hardcoded!) 
S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));
```

**FIX REQUIRED:**
```matlab
% hierarchical_step_update.m, after line ~720 (after exponential scaling)
if isfield(P, 'pi_bounds')
    S.pi_L1_motor = max(P.pi_bounds.L1_motor(1), min(P.pi_bounds.L1_motor(2), S.pi_L1_motor));
    S.pi_L2_motor = max(P.pi_bounds.L2_motor(1), min(P.pi_bounds.L2_motor(2), S.pi_L2_motor));
    S.pi_L1_plan = max(P.pi_bounds.L1_plan(1), min(P.pi_bounds.L1_plan(2), S.pi_L1_plan));
    S.pi_L2_plan = max(P.pi_bounds.L2_plan(1), min(P.pi_bounds.L2_plan(2), S.pi_L2_plan));
else
    % Fallback to hardcoded (backward compatible)
    S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));
    S.pi_L2_motor = max(0.5, min(50, S.pi_L2_motor));
    S.pi_L1_plan = max(10, min(500, S.pi_L1_plan));
    S.pi_L2_plan = max(0.5, min(50, S.pi_L2_plan));
end
```

---

### 3. TRIPLE PRECISION UPDATE MECHANISM (⚠️ MASSIVE INCONSISTENCY)

**Problem:** Three competing mechanisms are all updating precision simultaneously:

#### **Mechanism 1: `update_pi()` helper function** (lines ~555-600)
Uses error history with variance-based adaptation:
```matlab
function [pi_new, raw_pi, denom] = update_pi(...)
    denom = 1 + 0.8 * err_val + 0.2 * var_norm;
    raw_pi = pi_base / denom;
    pi_candidate = smooth_alpha * pi_curr + (1 - smooth_alpha) * raw_pi;
end
```
- Input: error history (window_size=100 steps)
- Conservative: `pi_smooth_alpha = 0.999` (only 0.1% change per step)
- Uses variance normalization

#### **Mechanism 2: Hardcoded exponential scaling** (lines ~500-530, NOT in error-driven section)
```matlab
precision_scale_L1_motor = exp(alpha_precision_gain * min(E_L1_motor_mag, 5.0));
S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;
S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));
```
- Input: current step error magnitude only
- Responds immediately to errors
- Uses hardcoded `alpha_precision_gain = 0.5` ❌
- Hardcoded bounds ❌

#### **Mechanism 3: Error-driven scaling (duplicate of #2)** (lines ~650-700)
```matlab
% NEW: PREDICTION-ERROR-DRIVEN ADAPTIVE PRECISION (Nov 2, 2025)
if isfield(P, 'alpha_precision_gain') && isfield(P, 'pi_bounds')
    alpha_gain = P.alpha_precision_gain;
    precision_scale_L1_motor = exp(alpha_gain * L1_motor_error_mag_capped);
    S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;
    S.pi_L1_motor = max(pi_L1_motor_min, min(pi_L1_motor_max, S.pi_L1_motor));
end
```
- **IDENTICAL LOGIC** to Mechanism 2, but uses `P.pi_bounds` ✓
- This is the **CORRECT** version
- BUT it executes AFTER Mechanism 2, so both run!

**TIMELINE OF PRECISION UPDATES:**

```
Step i execution order:
  1. Line ~515: update_pi() called → precision updated from history ✓ (conservative)
  2. Line ~525: Hardcoded exponential scaling → precision updated again ❌ (conflicting!)
  3. Line ~530: Hardcoded clipping → precision clipped ❌
  4. Line ~600: update_pi() called AGAIN for diagnostics
  5. Line ~655: Hardcoded exponential scaling AGAIN ❌ (duplicate of line ~515)
  6. Line ~700: Error-driven scaling (correct version) → precision updated THIRD time ✓
  7. Line ~730: Correct bounds applied ✓
```

**RESULT:** Precision values are chaotic, oscillating between 3+ competing mechanisms!

**THEORETICAL CONFLICT:**
- `update_pi()`: Slow, smoothed, history-based (gaussian filter logic)
- Exponential: Fast, immediate, responsive to current error

These are **fundamentally incompatible** adaptation strategies being applied simultaneously.

---

### 4. MISSING INTERFERENCE_PENALTY_WEIGHT IN P STRUCT (⚠️ ERROR)

**Location:** `hierarchical_motion_inference_dual_hierarchy.m`, line ~715
```matlab
P.alpha_precision_gain = alpha_precision_gain;
P.pi_bounds = P_pi_bounds;  % ← Added
% P.interference_penalty_weight is MISSING!
```

**Location:** `hierarchical_step_update.m`, lines ~280-285
```matlab
if isfield(P, 'interference_penalty_weight')
    interference_penalty_weight = P.interference_penalty_weight;
else
    interference_penalty_weight = 0.01;  % Default
end
```

**ISSUE:**
- Parameter is read from `params` in main script (line ~50)
- But **NOT passed to P struct**
- Helper function falls back to hardcoded default 0.01
- PSO optimization of this parameter is **partially ineffective** (uses default instead)

**FIX:** Add to P struct construction (line ~715):
```matlab
P.interference_penalty_weight = 0.01;  % Default
if nargin > 0 && isstruct(params) && isfield(params, 'interference_penalty_weight')
    P.interference_penalty_weight = params.interference_penalty_weight;
end
```

---

### 5. INCONSISTENT VELOCITY EXTRACTION (⚠️ ALGORITHMIC)

**Location:** `hierarchical_step_update.m`, lines ~180-210

Motor and Planning use **different blending ratios**:
```matlab
% Motor: 60% desired + 40% learned
alpha_motor_blend = 0.6;
S.motor_vx_motor(i) = P.motor_gain * (0.6 * desired + 0.4 * learned);

% Planning: 40% desired + 60% learned
alpha_plan_blend = 0.4;
S.motor_vx_plan(i) = P.motor_gain * (0.4 * desired + 0.6 * learned);
```

**Is this intentional?**
- No documentation of why ratios differ
- Motor region should learn "how to move" (minimize delta between desired/learned)
- Planning region should learn "where to go"
- **Hypothesis:** Different ratios make sense (motor more constrained, planning more freedom)
- **But:** No parameter control or justification

**RECOMMENDATION:** Add parameters `alpha_motor_blend` and `alpha_plan_blend` to PSO, or document why 60/40 and 40/60.

---

### 6. MISSING TASK-CONDITIONAL PARAMETER (⚠️ INCOMPLETE PSO INTEGRATION)

**Location:** PSO bounds section (`optimize_rao_ballard_pso.m`, lines ~40-85)

Missing from PSO: `interference_penalty_weight`

**Evidence:**
- Main script reads: `params.interference_penalty_weight` (line ~50, fallback 0.01)
- PSO has bounds for this (lines ~68-69) ✓
- But helper function doesn't receive it (see Finding #4)

**STATUS:** Partially integrated. Parameter bounds exist, but not passed to P struct.

---

### 7. DUPLICATE POSITION/VELOCITY INITIALIZATION (⚠️ CODE DUPLICATION)

**Locations:**
- `hierarchical_motion_inference_dual_hierarchy.m`, lines ~410-430 (initial)
- `hierarchical_motion_inference_dual_hierarchy.m`, lines ~750-820 (trial reset)
- `hierarchical_step_update.m`, lines ~180-210 (velocity extraction every step)

**Same code repeated 3+ times:**
```matlab
target_dir = ([ball] - [player]) / norm(...);
reaching_speed_adaptive = P.reaching_speed_scale * min(dist, 5.0);
desired_vel = target_dir * reaching_speed_adaptive;
motor_command = 0.6 * desired + 0.4 * learned;
```

**RECOMMENDATION:** Extract to helper function `get_target_velocity()`.

---

### 8. INCONSISTENT NAN/INF HANDLING (⚠️ THEORETICAL)

**Location:** `hierarchical_step_update.m`, lines ~360-380 (debug clipping)
vs. lines ~620-670 (main free energy calculation)

**Issue:** Two different clipping thresholds:
- Debug section: `max_finite_value = 1e8` (very high, allows large errors)
- Main FE section: Hard clipping at 1e7 (lower threshold)

**Inconsistency:** Which should be authoritative? Should be unified.

**FIX:**
```matlab
% Define once at top of function
MAX_FINITE_VALUE = 1e7;  % Single source of truth
MIN_PRECISION = 0.01;
MAX_PRECISION = 1e6;
```

---

## SEVERITY CLASSIFICATION

| Finding | Type | Severity | Impact |
|---------|------|----------|--------|
| #1: Hardcoded `alpha_precision_gain` | Bug | **CRITICAL** | PSO optimization ineffective for this parameter |
| #2: Hardcoded precision bounds | Bug | **CRITICAL** | PSO optimization ineffective for 8 bound parameters |
| #3: Triple precision update | Design | **CRITICAL** | Chaotic, unpredictable precision dynamics |
| #4: Missing `interference_penalty_weight` | Bug | **HIGH** | Partial PSO integration, uses fallback |
| #5: Inconsistent velocity blending | Design | **MEDIUM** | Undocumented, potentially wrong |
| #6: Incomplete PSO integration | Incomplete | **MEDIUM** | One parameter not fully optimizable |
| #7: Code duplication | Duplication | **LOW** | Maintenance burden, no functional impact |
| #8: Inconsistent clipping thresholds | Bug | **LOW** | Inconsistent safety checks |

---

## REQUIRED FIXES (in priority order)

### PRIORITY 1: Make Findings #1 and #2 work correctly

These are blocking PSO optimization:

```matlab
% hierarchical_step_update.m, line ~500
% BEFORE:
alpha_precision_gain = 0.5;  % ❌ Hardcoded
precision_scale_L1_motor = exp(alpha_precision_gain * min(E_L1_motor_mag, 5.0));
S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;
S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));  % ❌ Hardcoded

% AFTER:
if isfield(P, 'alpha_precision_gain') && isfield(P, 'pi_bounds')
    alpha_gain = P.alpha_precision_gain;
    precision_scale_L1_motor = exp(alpha_gain * min(E_L1_motor_mag, 5.0));
    S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;
    
    % Use bounds from P struct
    pi_L1m_bounds = P.pi_bounds.L1_motor;
    S.pi_L1_motor = max(pi_L1m_bounds(1), min(pi_L1m_bounds(2), S.pi_L1_motor));
end
```

### PRIORITY 2: Consolidate triple precision update

**Option A (Recommended):** Use ONLY the error-driven mechanism (lines ~650-700)
- Remove the hardcoded exponential at lines ~515-530
- Keep `update_pi()` for diagnostics only, don't apply results

**Option B:** Integrate all three into single coherent mechanism
- Complex, high risk of bugs

**Recommendation:** Go with Option A. Error-driven mechanism is:
- Theoretically justified (neuroscience)
- Uses PSO parameters correctly
- Properly bounded
- Cleaner code

### PRIORITY 3: Pass `interference_penalty_weight` to P struct

```matlab
% hierarchical_motion_inference_dual_hierarchy.m, line ~715
P.interference_penalty_weight = 0.01;  % Default
if nargin > 0 && isstruct(params) && isfield(params, 'interference_penalty_weight')
    P.interference_penalty_weight = params.interference_penalty_weight;
end
```

---

## VERIFICATION CHECKLIST

After fixes applied:

- [ ] Run PSO optimization with `alpha_precision_gain` swept across [0.1, 2.0]
  - Verify precision dynamics change noticeably
- [ ] Run PSO with `pi_L1_motor_min` swept across [5, 50]
  - Verify precision floor changes
- [ ] Compare results with/without `interference_penalty_weight`
  - Should see different per-task error distributions
- [ ] Run with `make_plots=true` to visual inspect precision traces
  - Should see clean exponential scaling, not oscillation
- [ ] Check that all 19 parameters appear in final results printout

---

## ADDITIONAL RECOMMENDATIONS

1. **Add parameter documentation** to top of both files listing which parameters come from where
2. **Extract common code** into helper functions (velocity computation, bounds clipping)
3. **Centralize all constants** (MAX_FINITE_VALUE, MIN_PRECISION, etc.)
4. **Add unit tests** for precision scaling with known error inputs
5. **Profile performance** - three simultaneous precision mechanisms may be slow

---

## CONCLUSION

**Status:** FUNCTIONAL but INCORRECT

The code runs without crashing, but **PSO optimization of 11 parameters is ineffective** due to hardcoded values overriding them. The three competing precision update mechanisms create **unpredictable behavior** that makes parameter optimization results unreliable.

**Time to fix:** ~30-45 minutes
**Risk level:** LOW (fixes are localized, well-defined)
**Testing:** MEDIUM (need to verify PSO responsiveness to parameters)

