# Implementation Fixes - November 3, 2025
## Critical Fixes to Enable Learning

This document summarizes the comprehensive fixes applied to enable learning in the dual-hierarchy model.

---

## Summary of Changes

### ✅ FIX #1: Motor Weights → Shared (Not Per-Task)
**File:** `hierarchical_motion_inference_dual_hierarchy.m` (lines ~665-720)

**Problem:** Motor weights were being redefined in a loop, creating inconsistent initialization.

**Change:**
```matlab
% BEFORE: Motor weights initialized multiple times per task (incorrect)
for task_idx = 1:n_trials
    W_motor_L2_to_L1(idx_vel, 1:map_vel) = eye(map_vel);  % Redefined each iteration!
end

% AFTER: Single shared initialization (correct - motor learns generic velocity control)
W_motor_L2_to_L1 = zeros(n_L1_motor, n_L2_motor);  % NOT a cell array
W_motor_L3_to_L2 = zeros(n_L2_motor, n_L3_motor);

% Initialize once (shared across all tasks)
map_vel = min(n_vel, n_L2_motor);
W_motor_L2_to_L1(idx_vel(1:map_vel), 1:map_vel) = eye(map_vel);
% ... rest of motor init ...

% Planning stays task-indexed (learns task-specific ball dynamics)
W_plan_L2_to_L1 = cell(n_trials, 1);   % Cell array - one per task
W_plan_L3_to_L2 = cell(n_trials, 1);
for task_idx = 1:n_trials
    W_plan_L2_to_L1{task_idx} = zeros(n_L1_plan, n_L2_plan);
    % ... planning init ...
end
```

**Rationale:**
- Motor region: learns **stable forward model** (velocity control generalizes across tasks)
- Planning region: learns **task-specific strategies** (each ball trajectory is different)
- This separation ensures motor develops generalizable skills while planning specializes

**Impact:** ⭐⭐⭐ CRITICAL
- Fixes incorrect weight access patterns
- Enables motor generalization across tasks
- Prevents per-task motor overfitting

---

### ✅ FIX #2: Step Update Motor Weight Indexing
**File:** `hierarchical_step_update.m` (lines ~108-115)

**Problem:** Code tried to index motor weights as cells `S.W_motor_L2_to_L1{current_task_idx}` but they are matrices, not cells.

**Change:**
```matlab
% BEFORE (crashes or silent failure):
W_motor_L2_to_L1_active = S.W_motor_L2_to_L1{current_task_idx};  % ERROR: not a cell!

% AFTER (correct - motor is shared matrix):
W_motor_L2_to_L1_active = S.W_motor_L2_to_L1;  % Always use shared motor weights
W_motor_L3_to_L2_active = S.W_motor_L3_to_L2;

% Planning still task-indexed (cell array):
W_plan_L2_to_L1_active = S.W_plan_L2_to_L1{current_task_idx};  % Correct: planning is per-task
W_plan_L3_to_L2_active = S.W_plan_L3_to_L2{current_task_idx};
```

**Impact:** ⭐⭐ CRITICAL
- Fixes data type mismatch
- Ensures correct weight matrices are used

---

### ✅ FIX #3: Gradient Normalization (Layer Scale → Layer Norm)
**File:** `hierarchical_step_update.m` (lines ~540-565)

**Problem:** Gradient scaling used `mean(abs(R))` which could be very small, causing huge gradient spikes that clipping then kills.

**Change:**
```matlab
% BEFORE (unstable - mean can be tiny):
layer_scale_motor_1 = max(0.1, mean(abs(S.R_L2_motor(i,:))));
dW_motor_1 = -(P.eta_W * S.pi_L1_motor / layer_scale_motor_1) * (E_L1_motor' * R_L2_motor);
% If layer_scale_motor_1 = 0.1, division gives huge dW → then clipped to near-zero

% AFTER (stable - L2 norm with adaptive floor):
layer_norm_motor_2 = max(0.1, norm(S.R_L2_motor(i,:), 2));  % L2 norm
layer_norm_motor_3 = max(0.1, norm(S.R_L3_motor(i,:), 2));

dW_motor_L2_to_L1 = -(P.eta_W * S.pi_L1_motor / layer_norm_motor_2) * (S.E_L1_motor(i,:)' * S.R_L2_motor(i,:));
dW_motor_L3_to_L2 = -(P.eta_W * S.pi_L2_motor / layer_norm_motor_3) * (S.E_L2_motor(i,:)' * S.R_L3_motor(i,:));

% FIX: Add gradient clipping to prevent explosion
max_motor_grad = 0.1;  % Clip to [-0.1, +0.1]
dW_motor_L2_to_L1 = max(-max_motor_grad, min(max_motor_grad, dW_motor_L2_to_L1));
dW_motor_L3_to_L2 = max(-max_motor_grad, min(max_motor_grad, dW_motor_L3_to_L2));

% Apply updates with clipped gradients
S.W_motor_L2_to_L1 = S.W_motor_L2_to_L1 + dW_motor_L2_to_L1;
S.W_motor_L3_to_L2 = S.W_motor_L3_to_L2 + dW_motor_L3_to_L2;
```

**Rationale:**
- L2 norm (Euclidean distance) is more stable than mean absolute value
- Gradient clipping prevents explosion while allowing learning to proceed
- Adaptive floor prevents division by near-zero

**Impact:** ⭐⭐⭐ CRITICAL
- Fixes gradient explosion problem
- Enables stable learning
- Prevents clipping-induced learning signal death

---

### ✅ FIX #4: Precision Clipping Order (Already Correct)
**File:** `hierarchical_step_update.m` (lines ~810-827)

**Status:** ✅ Already implemented correctly!

**Current (correct) code:**
```matlab
% Step 1: Update precision with exponential scaling
S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;
S.pi_L2_motor = S.pi_L2_motor * precision_scale_L2_motor;
% ... etc ...

% Step 2: THEN clip to bounds
S.pi_L1_motor = max(bounds_L1_motor(1), min(bounds_L1_motor(2), S.pi_L1_motor));
S.pi_L2_motor = max(bounds_L2_motor(1), min(bounds_L2_motor(2), S.pi_L2_motor));
```

**Why this is correct:**
- Exponential scaling happens first (learning signal preserved)
- Clipping applied AFTER (bounds enforced, but scaling not reversed)
- Previous concern about clipping reversing scaling is unfounded - clipping happens AFTER scaling completes

**Impact:** ✅ No change needed

---

### ✅ FIX #5: Add Motor Noise/Exploration (Annealing)
**File:** `hierarchical_step_update.m` (lines ~145-157)

**Problem:** Without noise, motor commands are deterministic → execution = prediction → error = 0 → no learning signal.

**Change:**
```matlab
% BEFORE (deterministic - no learning signal):
final_motor_vx = S.motor_vx_motor(i);
final_motor_vy = S.motor_vy_motor(i);
final_motor_vz = S.motor_vz_motor(i);

% AFTER (stochastic with annealing):
% Noise scale: start high, decay to zero over training
noise_annealing_factor = max(0.01, 1.0 - (i / 1000));  % Decreases over ~1000 steps
noise_scale = 0.05 * noise_annealing_factor;  % Max: 0.05 m/s

% Add exploration noise (Gaussian)
final_motor_vx = S.motor_vx_motor(i) + noise_scale * randn();
final_motor_vy = S.motor_vy_motor(i) + noise_scale * randn();
final_motor_vz = S.motor_vz_motor(i) + noise_scale * randn();
```

**Rationale:**
- **Early training:** High noise (exploration) → large prediction errors → strong learning signal
- **Late training:** Low noise (exploitation) → small errors → fine-tuning
- **Biologically plausible:** Motor system has inherent variability (muscle noise, proprioceptive uncertainty)

**Impact:** ⭐⭐⭐ CRITICAL
- Creates non-zero prediction errors
- Enables learning from beginning of simulation
- Annealing provides curriculum learning naturally

---

### ✅ FIX #6: Add Learning Diagnostics to Main Loop
**File:** `hierarchical_motion_inference_dual_hierarchy.m` (lines ~1021-1035)

**Change:**
```matlab
% BEFORE:
for i = 1:N-1
    if mod(i, 100) == 0, fprintf('.'); end
    % ...
end

% AFTER:
for i = 1:N-1
    print_diagnostics = (mod(i, 100) == 0);
    if print_diagnostics, fprintf('.'); end
    % ...
    S = hierarchical_step_update(i, S, P);
    
    % FIX: Print learning diagnostics every 100 steps
    if print_diagnostics
        fprintf('\n  Step %d: FE=%.2e | IntErr=%.4f | |dW|=%.2e | pi_L1m=%.1f | noise_scale=%.4f\n', ...
            i, ...
            S.free_energy_all(i), ...
            S.interception_error_all(i), ...
            S.learning_trace_W(i), ...
            S.pi_L1_motor, ...
            max(0.01, 0.05 * (1.0 - i/1000)));  % Current noise level
    end
end
```

**Diagnostics Explained:**
- **FE (Free Energy):** Total prediction error weighted by precision. Should decrease over time if learning works.
- **IntErr (Interception Error):** Distance between player and ball. Should decrease as motor learns.
- **|dW| (Weight Update Magnitude):** Norm of weight changes. If ~0, weights aren't updating (learning blocked).
- **pi_L1m (L1 Motor Precision):** Scales importance of proprioceptive errors. Should increase with surprise.
- **noise_scale:** Current exploration noise. Starts at ~0.05, decays to ~0.01.

**Usage:**
```
Run simulation and watch the console output every 100 steps.
Example healthy run:
  Step 100: FE=2.35e+01 | IntErr=5.123 | |dW|=0.0342 | pi_L1m=125.3 | noise_scale=0.0498
  Step 200: FE=1.87e+01 | IntErr=4.891 | |dW|=0.0298 | pi_L1m=143.7 | noise_scale=0.0495
  Step 300: FE=1.42e+01 | IntErr=4.456 | |dW|=0.0251 | pi_L1m=152.1 | noise_scale=0.0492
  
  Good signs:
  - FE decreases
  - IntErr decreases (player getting closer)
  - |dW| is consistently non-zero (weights learning)
  - pi_L1m increases (model gaining confidence)
```

**Impact:** ⭐⭐ HELPFUL
- Enables real-time debugging
- Shows whether learning is happening
- Identifies failure modes early

---

## Unified Impact Summary

| Fix | Type | Severity | Impact |
|-----|------|----------|--------|
| #1: Motor weights shared | Architecture | ⭐⭐⭐ | Enables motor generalization |
| #2: Motor indexing | Bug | ⭐⭐ | Fixes data type error |
| #3: Gradient norm + clipping | Stability | ⭐⭐⭐ | Prevents learning collapse |
| #4: Precision clipping order | Verification | ✅ | Already correct |
| #5: Motor noise annealing | Learning | ⭐⭐⭐ | Creates learning signal |
| #6: Diagnostics | Debug | ⭐⭐ | Enables troubleshooting |

---

## Expected Behavior After Fixes

### Console Output Pattern
```
Running dual-hierarchy learning with player chasing moving ball...
Total iterations: 3000 (dt=0.01s per step, ~30.0 seconds estimated)
.
  Step 100: FE=3.42e+01 | IntErr=6.234 | |dW|=0.0487 | pi_L1m=102.3 | noise_scale=0.0499
.
  Step 200: FE=2.91e+01 | IntErr=5.891 | |dW|=0.0432 | pi_L1m=118.7 | noise_scale=0.0495
.
  Step 300: FE=2.45e+01 | IntErr=5.123 | |dW|=0.0378 | pi_L1m=135.2 | noise_scale=0.0492
  
  [Trial 2 started at step 1000, Task Context: R_L0(i,2)=1]
  Player reset to: [1.23, 1.45, 0.00]
  Ball reset to: [7.82, 6.91, 0.32]
  Weight decay (Motor: 0.95→95%, Planning: 0.70→70%)

.
  Step 1100: FE=1.87e+01 | IntErr=3.456 | |dW|=0.0291 | pi_L1m=156.8 | noise_scale=0.0445
```

### Success Indicators
✅ **FE decreases** (Free energy dropping = learning happening)
✅ **IntErr decreases** (Player getting closer to ball)
✅ **|dW| consistently non-zero** (Weights updating each step)
✅ **pi_L1m increases** (Model gaining confidence in proprioceptive predictions)
✅ **No NaN/Inf errors** (Numerical stability maintained)

### Failure Indicators  
❌ **FE constant or increasing** → No learning signal created
❌ **IntErr constant** → Motor not improving
❌ **|dW| near zero** → Weights frozen (learning blocked)
❌ **NaN/Inf in console** → Numerical instability remains

---

## Next Steps to Validate

### 1. Run a Quick Test
```matlab
% minimal test - 1 trial, 5 seconds, small layer sizes
results = hierarchical_motion_inference_dual_hierarchy(...
    struct('n_trials', 1, 'T_per_trial', 5, 'eta_rep', 0.01, 'eta_W', 0.001), ...
    false);  % make_plots = false for speed
```

### 2. Check Diagnostic Output
- Look for decreasing FE and IntErr
- Verify |dW| is non-zero (not ~1e-10 or zero)
- Check pi_L1m increasing gradually

### 3. Verify Motor Noise is Working
- At early steps (i=1-100), noise_scale should be ~0.05
- At late steps (i=2500+), noise_scale should be ~0.01
- This controls exploration vs exploitation

### 4. Inspect Weight Updates
```matlab
% After simulation, check final weights
figure;
subplot(2,2,1); imagesc(results.W_motor_L2_to_L1); colorbar; title('Motor L2→L1 (shared)');
subplot(2,2,2); imagesc(results.W_motor_L3_to_L2); colorbar; title('Motor L3→L2 (shared)');
subplot(2,2,3); imagesc(results.W_plan_L2_to_L1{1}); colorbar; title('Planning L2→L1 (task 1)');
subplot(2,2,4); imagesc(results.W_plan_L3_to_L2{1}); colorbar; title('Planning L3→L2 (task 1)');

% Check for learning: weights should NOT be all identical after training
% Motor weights should show structured patterns
% Planning weights should differ across tasks
```

---

## References to Documentation

- **MODEL ALGORITHM:** `docs/MODEL_ALGORITHM_EXPLANATION.md`
- **CORTICAL MAPPING:** `docs/MODEL_TO_CORTICAL_MAPPING.md`
- **DUAL HIERARCHY:** `DUAL_HIERARCHY_ARCHITECTURE.md`

---

## Author Notes

These fixes address the fundamental issue: **the model was learning nothing because execution ≠ prediction**.

The key insight: **Predictive coding requires a learning signal**, which only exists when there's mismatch between prediction and execution. The motor noise annealing fix (FIX #5) is the most critical—it ensures this mismatch exists, especially early in training when predictions are wild guesses.

The gradient normalization fix (FIX #3) ensures this learning signal isn't immediately destroyed by numerical overflow and clipping.

Together, these fixes should enable the model to learn meaningful forward models and planning policies across the three interception tasks.

---

**Date:** November 3, 2025  
**Status:** All fixes implemented and documented  
**Ready for testing:** ✅ Yes
