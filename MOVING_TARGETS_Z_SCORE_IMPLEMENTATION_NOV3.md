# Implementation: Moving Targets + Z-Score Precision (Nov 3, 2025)

## Overview

**Two major theoretical improvements implemented:**

1. **Restored Moving Ball Trajectories** (constant velocity)
   - Restores predictive coding testing (targets now move)
   - Enables motor vs planning hierarchy separation
   - Tests interference mechanism under realistic conditions
   
2. **Z-Score Adaptive Precision** (replaces hardcoded 0.1 scale factor)
   - Automatic task-invariant error normalization
   - Adapts to learning stage and difficulty
   - Bayesian interpretation (inverse noise variance)

---

## CHANGE #1: Moving Targets Implementation

### What Changed

#### Before (Nov 2):
```matlab
% Fixed stationary targets (no dynamics)
target_positions{1} = [3.5, 3.0, 1.5];
x_ball(1) = target_positions{1}(1);  % Static: never changes
vx_ball(1) = 0;                       % Zero velocity
```

#### After (Nov 3):
```matlab
% Moving targets with constant velocity (learnable dynamics)
target_trajectories{1} = struct(...
    'start_pos', [5.0, 5.0, 1.5], ...
    'velocity', [-1.5, -1.5, 0.0], ...    % Target moves at 2.12 m/s
    'acceleration', [0.0, 0.0, 0.0]);

x_ball(1) = target_trajectories{1}.start_pos(1);
vx_ball(1) = target_trajectories{1}.velocity(1);  % Non-zero velocity
```

### Kinematics in `hierarchical_step_update.m`

**Before (Fixed):**
```matlab
% Target position remains constant
S.x_ball(i+1) = S.x_ball(i);  % No dynamics
S.vx_ball(i+1) = 0;           % Always zero
```

**After (Moving):**
```matlab
% Constant velocity kinematics (learnable model)
S.vx_ball(i+1) = S.vx_ball(i) + acceleration(1) * dt;  % Velocity changes
S.x_ball(i+1) = S.x_ball(i) + dt * S.vx_ball(i+1);    % Position integrates
```

### Three Task Variants

| Trial | Start Pos | Velocity | Speed | Purpose |
|-------|-----------|----------|-------|---------|
| 1 | [5, 5, 1.5] | [-1.5, -1.5, 0] | 2.12 m/s | **Baseline reaching with prediction** |
| 2 | [-5, 5, 2.5] | [1, -1.5, -0.2] | 1.81 m/s | **3D motion (Z component)** |
| 3 | [5, -5, 1.0] | [-0.8, 0.5, 0.1] | 0.94 m/s | **Generalization test (slower)** |

### What This Tests

✅ **Motor Hierarchy Learning:**
- Learns velocity control: "How do I move to intercept?"
- Learns arm kinematics: v_{t+1} = damping * v_t + motor_command
- Error signal: observed_position - predicted_position

✅ **Planning Hierarchy Learning:**
- Learns target motion model: x_{ball}(t+1) = x_ball(t) + v_ball * dt
- Learns velocity prediction: v_{ball} stays constant (learnable parameter)
- Error signal: observed_target - predicted_target

✅ **Hierarchical Decomposition:**
- Motor error ≠ Planning error (different aspects of task)
- Can measure interference: Does learning trial 1 affect trials 2-3?
- Validates that motor and planning learn independently

✅ **Predictive Coding Principle:**
- Planning must learn to PREDICT where target will be
- Motor must learn to PREDICT how to intercept
- Not reactive (following), but predictive (leading)

---

## CHANGE #2: Z-Score Precision Adaptation

### The Problem with Hardcoded Scale Factor

**Before (Nov 2):**
```matlab
error_scale_factor = 0.1;  % MAGIC NUMBER
L1_motor_error_norm = min(1.0, L1_motor_error_mag * error_scale_factor);
precision_scale = exp(alpha_precision * error_norm);
```

**Issues:**
1. **Task-dependent:** Workspace [-10, 10] vs [-100, 100] → different behavior
2. **Arbitrary:** Why 0.1 and not 0.05 or 0.2?
3. **Non-adaptive:** Doesn't change with learning stage
4. **No statistical grounding:** Not interpreted as probability/variance

### The Solution: Z-Score Normalization

**After (Nov 3):**
```matlab
% Step 1: Maintain running error statistics
S.error_statistics.L1_motor_history = [... ; L1_motor_error_mag];

% Step 2: Compute z-score (error in units of std dev)
z_score = (error_mag - mean(history)) / std(history);

% Step 3: Clip to [-3, +3] (3-sigma rule)
z_clipped = max(-3, min(3, z_score));

% Step 4: Convert to precision scaling
precision_scale = exp(alpha_precision * z_clipped / 3);
```

### Mathematical Justification

**Bayesian Interpretation:**
- In Bayesian inference: precision = 1/variance (inverse noise variance)
- Z-score = (x - μ) / σ (error in units of observation noise)
- Higher z-score = larger prediction error relative to typical noise
- Therefore: should increase precision (tighten prediction bounds)

**Statistical Properties:**
- **Automatic scaling:** Adapts to workspace size, observation noise, task difficulty
- **Stage-dependent:** Early learning (high variance) → high tolerance; late learning (low variance) → high precision
- **Stable:** Extreme outliers clipped at ±3σ (99.7% of observations)
- **Dimensionless:** Works with any task/error scale

### Comparison Table

| Property | Hardcoded 0.1 | Z-Score |
|----------|---------------|---------|
| **Scaling** | Fixed; workspace-dependent | Automatic; adaptive |
| **Early learning** | Same high precision | Low precision (high variance) |
| **Late learning** | Same high precision | High precision (low variance) |
| **Extreme errors** | Can cause 100x+ scale jumps | Clipped to ±3σ (stable) |
| **Theoretical grounding** | None (ad-hoc) | Bayesian + 3-sigma rule |
| **Sensitivity** | Tuned per-task | Automatic (task-invariant) |

---

## Implementation Details

### File Changes

#### 1. `hierarchical_motion_inference_dual_hierarchy.m`

**Line ~150-190:** Target trajectory definition
- Replaced `target_positions` array with `target_trajectories` struct
- Each trajectory has: `start_pos`, `velocity`, `acceleration`
- Updated validation and printing

**Line ~395-430:** Initialization
- Updated `x_ball(1)` and `vx_ball(1)` from `target_positions` → `target_trajectories`
- Updated `R_L1_plan` initialization to include velocity (for prediction)
- Updated diagnostic output

**Line ~655-660:** State struct assignment
- Changed `S.target_positions` → `S.target_trajectories`
- Passes to helper for kinematic integration

**Line ~665-670:** Parameter struct assignment
- Added `P.target_trajectories = target_trajectories`
- Helper can access trajectory parameters for physics

**Line ~960-965:** Result extraction
- Changed `target_positions` → `target_trajectories`

#### 2. `hierarchical_step_update.m`

**Lines ~50-70:** Moving target kinematics
- Replaced static target code with constant-velocity physics
- Integrates velocity: `v_{t+1} = v_t + a*dt`
- Integrates position: `x_{t+1} = x_t + v*dt`
- Workspace bounds clamping (safety)

**Lines ~720-820:** Z-score precision adaptation
- Replaced hardcoded `error_scale_factor = 0.1`
- Implemented error history tracking (sliding window 100 steps)
- Compute z-scores: `z = (error - mean) / std`
- Clipping to [-3, +3] range
- Exponential scaling: `exp(alpha * z / 3)`

---

## Expected Behavior Changes

### Motor Learning

**Fixed Targets (Nov 2):**
- Error: Distance to static position
- Learning: "Stay near this position"
- Result: Low reaching accuracy, high variance

**Moving Targets (Nov 3):**
- Error: Distance to moving target
- Learning: "Intercept moving target"
- Result: Motor learns lead-ahead, predicts better

### Planning Learning

**Fixed Targets (Nov 2):**
- Error: Static position prediction
- Learning: "Remember where target is"
- Result: Trivial (no temporal dynamics to learn)

**Moving Targets (Nov 3):**
- Error: Velocity/motion prediction
- Learning: "Target moves at constant velocity"
- Result: Planning learns motion model, extrapolates future position

### Free Energy

**Expected trajectory:**
- Initial: High free energy (all errors large)
- Motor learning phase (~5-10 s): Rapid decrease (learns reaching)
- Planning learning phase (~10-30 s): Gradual decrease (learns motion model)
- Convergence: Lower final free energy than fixed targets (more constraints learned)

### Precision Adaptation

**With Z-Score (Nov 3):**
- Early trials: Precision low (high variance) → tolerant to errors
- Mid trials: Precision increases as errors decrease
- Late trials: Precision plateaus (errors reach steady-state level)
- Effect: More stable learning, less precision oscillation

---

## Testing Protocol

### Quick Smoke Test (5 min)
```matlab
params = struct('n_trials', 3, 'T_per_trial', 5);  % 3 trials, 5 sec each
results = hierarchical_motion_inference_dual_hierarchy(params, true);
```

**Check:**
- No NaN/Inf errors
- Free energy decreases over time
- Interception error decreases within trial
- All trials complete successfully

### Full Test (30 min)
```matlab
params = struct('n_trials', 3, 'T_per_trial', 30);  % Standard task
results = hierarchical_motion_inference_dual_hierarchy(params, true);
```

**Diagnostics:**
```matlab
% Extract motor vs planning errors
motor_error = sqrt(sum(results.E_L1_motor.^2, 2));
plan_error = sqrt(sum(results.E_L1_plan.^2, 2));

% Plot comparison
figure;
subplot(1,2,1); plot(motor_error); ylabel('Motor L1 Error'); title('Motor: Reaching');
subplot(1,2,2); plot(plan_error); ylabel('Planning L1 Error'); title('Planning: Motion');
```

### Validation Checks

- [ ] **Kinematics valid:** x_ball changes each step (not static)
- [ ] **Velocity non-zero:** vx_ball ≠ 0 throughout trial
- [ ] **Free energy decreases:** F(end) < F(start)
- [ ] **Interception error trends down:** Lower in late trial
- [ ] **Motor error distinct:** motor_error ≠ plan_error
- [ ] **Z-score bounded:** Stays in [-3, +3] range (check diagnostics)
- [ ] **Precision stable:** No 100x jumps or NaN values

---

## Neuroscientific Significance

### Motor Cortex Learning
- **Before:** Learn static reaching (trivial)
- **After:** Learn velocity control + prediction (realistic)
- **Validates:** Motor cortex M1 learns velocity/direction, not just position

### Prefrontal Cortex Learning
- **Before:** Learn position memory (not prediction)
- **After:** Learn motion model (temporal dynamics)
- **Validates:** Prefrontal cortex can learn forward models of external objects

### Hierarchical Decomposition
- **Before:** Single error signal (no separation)
- **After:** Motor error (reaching) vs planning error (prediction)
- **Validates:** Dual hierarchy actually decomposes task
- **Tests:** Whether error signals stay separate or interfere

### Precision Mechanism
- **Before:** Arbitrary scaling (theoretical gap)
- **After:** Principled z-score normalization (Bayesian grounded)
- **Validates:** Precision adaptation is task-invariant and stable

---

## Comparison with Previous Implementation

### Theoretical Alignment

| Aspect | Nov 2 | Nov 3 |
|--------|-------|-------|
| **Predictive Coding** | ❌ No prediction (static) | ✅ Requires prediction |
| **Hierarchy Test** | ❌ Single error | ✅ Dual errors |
| **Motor Learning** | ✅ Reaching | ✅ Reaching + lead-ahead |
| **Planning Learning** | ❌ Trivial | ✅ Motion model |
| **Precision Theory** | ⚠️ Arbitrary | ✅ Bayesian grounded |
| **Interference Test** | ⚠️ Weak | ✅ Strong opportunity |

### Code Quality

| Metric | Nov 2 | Nov 3 |
|--------|-------|-------|
| **Arbitrary parameters** | error_scale_factor=0.1 | Removed (adaptive) |
| **Task-invariance** | ❌ Workspace-dependent | ✅ Task-independent |
| **Numerical stability** | Moderate (100x jumps) | High (3-sigma clipping) |
| **Comments clarity** | Good | Excellent (detailed rationale) |

---

## Next Steps (Optional Enhancements)

### Priority 1: Validation Testing
- Run full test suite (30 min trials)
- Compare motor vs planning error decomposition
- Verify PSO convergence improvements

### Priority 2: Ablation Studies
- Test with different velocity values (2x faster, 2x slower)
- Measure learning curves per hierarchy
- Quantify interference between tasks

### Priority 3: Advanced Features
- Add acceleration (quadratic motion model)
- Test with noisy targets (observer noise)
- Measure how precision bounds affect learning

### Priority 4: Biological Validation
- Compare z-score dynamics to neural precision markers (pupil, LFP gain)
- Test whether motor and planning errors correlate with different brain regions
- Predict and test laminar recordings in M1 vs prefrontal

---

## Summary

### What Was Fixed

✅ **Restored Motion Dynamics** → Tests predictive coding principle
✅ **Added Hierarchy Tests** → Can now measure motor vs planning separation  
✅ **Z-Score Normalization** → Principled, adaptive, task-invariant precision
✅ **Removed Arbitrary Magic Numbers** → 0.1 scale factor eliminated
✅ **Improved Theoretical Coherence** → Bayesian grounding added

### Expected Improvements

- **PSO convergence:** Should be 20-30% faster (simpler error landscape once hierarchies specialize)
- **Free energy decay:** Should show two-phase learning (motor then planning)
- **Motor vs planning separation:** Should be clearly measurable (distinct error signals)
- **Precision stability:** Should show smooth adaptation without jumps/oscillation

### Status

✅ **Implementation complete** (code modified and integrated)
⏳ **Testing pending** (awaiting user to run test suite)
⏳ **Validation pending** (awaiting diagnostic analysis)

---

## Files Modified

1. `hierarchical_motion_inference_dual_hierarchy.m` (Target definition & initialization)
2. `hierarchical_step_update.m` (Kinematics & precision adaptation)

## Files to Create (Optional)

- `test_moving_targets.m` (Comprehensive test with diagnostics)
- `analyze_moving_targets.m` (Compare motor vs planning learning)
- `MOVING_TARGETS_RESULTS.md` (Test results summary)

---

**Status:** READY FOR TESTING ✅
