# Fixed Targets Implementation (Nov 2, 2025)

## Task Transformation: Ball Tracking → Fixed Target Reaching

Successfully converted the experiment from **dynamic ball tracking** to **static fixed target reaching**. This tests whether the hierarchical predictive coding model can learn pure reaching dynamics without predictive ball motion modeling.

---

## Changes Made

### 1. **Target Position Generation** (`hierarchical_motion_inference_dual_hierarchy.m`, lines ~155-205)

**Before (Ball Trajectories with Dynamics):**
```matlab
ball_trajectories{1} = struct(...
    'start_pos', [2.0, 2.0, 1.0], ...
    'velocity', [2.5, 1.5, 1.0], ...
    'acceleration', [0.1, 0.0, 0.0]);
```

**After (Fixed Target Positions):**
```matlab
target_positions{1} = [3.5, 3.0, 1.5];  % Trial 1: static position
target_positions{2} = [5.0, 2.0, 1.2];  % Trial 2: static position
target_positions{3} = [-2.5, 4.0, 2.0]; % Trial 3: static position
```

**Impact:**
- Removed all velocity/acceleration fields
- Target position never changes during trial
- Simplified initialization (only 3 fields instead of 9)
- Removed trajectory generation complexity

---

### 2. **Initial State Setup** (`hierarchical_motion_inference_dual_hierarchy.m`, lines ~380-410)

**Before:**
```matlab
x_ball(1) = ball_trajectories{1}.start_pos(1);
vx_ball(1) = ball_trajectories{1}.velocity(1);
```

**After:**
```matlab
x_ball(1) = target_positions{1}(1);
vx_ball(1) = 0;  % Target has zero velocity
```

**Impact:**
- Target initialized with zero velocity
- No velocity dynamics to predict
- Planning layer only needs to remember fixed position per task

---

### 3. **Target Physics** (`hierarchical_step_update.m`, lines ~48-80)

**Before (Ball Physics with Dynamics):**
```matlab
% Integrate accelerations → velocities → positions
S.vx_ball(i+1) = S.vx_ball(i) + ax * dt;
S.x_ball(i+1) = S.x_ball(i) + dt * S.vx_ball(i+1);

% Apply gravity, air drag, bouncing, etc.
S.vz_ball(i+1) = -P.restitution * S.vz_ball(i+1);
```

**After (Static Target):**
```matlab
% Target position never changes
S.x_ball(i+1) = S.x_ball(i);
S.y_ball(i+1) = S.y_ball(i);
S.z_ball(i+1) = S.z_ball(i);

% Target has zero velocity (always)
S.vx_ball(i+1) = 0;
S.vy_ball(i+1) = 0;
S.vz_ball(i+1) = 0;
```

**Impact:**
- Removed all physics computations (gravity, drag, bouncing, collisions)
- Physics loop now O(1) instead of O(n) operations
- No numerical instabilities from physics integration

---

### 4. **Trial Reset** (`hierarchical_motion_inference_dual_hierarchy.m`, lines ~760-770)

**Before:**
```matlab
S.x_ball(i) = ball_trajectories{trial}.start_pos(1);
S.vx_ball(i) = ball_trajectories{trial}.velocity(1);
```

**After:**
```matlab
S.x_ball(i) = target_positions{trial}(1);
S.vx_ball(i) = 0;  % Target is stationary
```

**Impact:**
- Trial resets now reference fixed target positions
- Each trial has a distinct reaching target
- No randomization in target trajectories

---

## Expected Behavioral Changes

### Learning Task Simplification

| Aspect | Ball Tracking | Fixed Targets |
|--------|---------------|---------------|
| **Motor Learning** | Learn interception + prediction | Learn pure reaching |
| **Planning Learning** | Predict ball motion (complex) | Remember target position (simple) |
| **Task Difficulty** | Hard (moving target) | Easy (static target) |
| **Error Signal** | High variance (ball dynamics) | Low variance (fixed position) |
| **PSO Convergence** | ~80-100 iterations | ~40-60 iterations (faster) |
| **Interception RMSE** | 0.5-1.0 m | 0.1-0.3 m |
| **Free Energy Decay** | Slow initial → exponential | Rapid → saturation |

### Expected Learning Curves

**Motor Region:**
- **Trial 1:** Large reaching errors, rapid learning
- **Trial 2-3:** Faster error reduction (motor generalizes from trial 1)
- **By trial 3:** Near-perfect reaching to each target

**Planning Region:**
- **Trial 1:** Learns target 1 position
- **Trial 2:** Learns target 2 position (replaces target 1)
- **Trial 3:** Learns target 3 position (replaces target 2)
- Note: Planning weights decay (~70%) between trials, so previous targets are partially forgotten

---

## Neuroscientific Interpretation

### Computational Roles

**Motor Hierarchy** → Learning **reaching dynamics**
- Maps desired target direction → motor commands
- Should converge quickly (dynamics are simple: velocity → acceleration)
- Generalizes across target positions

**Planning Hierarchy** → **Target memory** (per-trial context)
- Encodes which target is active this trial
- Should be task-selective (weights decay between trials)
- Each trial = new reaching target

### Expected Neural Properties

**Motor L1 (Proprioception):**
- Should show **clear modulation** when player moves
- Minimal task-dependence (reaching dynamics same across targets)

**Motor L3 (Output):**
- Should encode **reaching velocity** to target
- Pattern rotates based on target position (but dynamics invariant)

**Planning L1 (Goals):**
- Should encode **target position** during each trial
- Should show **discrete jumps** at trial boundaries (new target)
- Should be **task-selective** (different across trials)

**Predictions to Test:**
1. **Generalization:** Can motor reach a novel target (test trial) better than random?
2. **Task Selectivity:** Do planning L1 neurons encode different goals per task?
3. **Faster Learning:** Does fixed-target learning converge faster than ball tracking?

---

## Files Modified

✅ `hierarchical_motion_inference_dual_hierarchy.m`
- Replaced `ball_trajectories` → `target_positions` (lines 155-205)
- Updated target position validation (lines 180-206)
- Updated initialization with fixed target (lines 380-410)
- Updated trial reset logic (lines 760-770)

✅ `hierarchical_step_update.m`
- Replaced ball physics → static target (lines 48-80)

---

## Testing Recommendations

### Quick Validation (5 min)

```matlab
params.n_trials = 3;
params.T_per_trial = 10;  % 10 seconds per trial
params.scale_factor = 2;   % Small layers for speed
results = hierarchical_motion_inference_dual_hierarchy(params, false);

% Check that interception errors decrease each trial
fprintf('Mean reaching error per trial:\n');
for t = 1:3
    trial_indices = phases_indices{t};
    fprintf('  Trial %d: %.3f m\n', t, mean(results.interception_error_all(trial_indices)));
end
```

### Full Experiment (30-60 min)

```matlab
params.n_trials = 3;
params.T_per_trial = 30;
params.scale_factor = 2;
params.n_param_samples = 100;  % PSO iterations
[best_params, best_score] = optimize_rao_ballard_pso(params, @hierarchical_motion_inference_dual_hierarchy);

% Should see faster PSO convergence (~40-60 iterations) than ball tracking
```

---

## Performance Expectations

### Single Run (No Optimization)
- Motor should reach target by end of trial 1 (error < 0.3 m)
- Planning should stabilize target representation (minimal error)
- Free energy should decay rapidly (target = fixed, easy prediction)

### PSO Optimization
- Fewer iterations needed (40-60 vs 80-100 for ball tracking)
- Parameters should converge more cleanly (less noisy landscape)
- Best scores should be higher (easier task overall)

---

## Next Steps

1. **Run smoke test** (3 trials, 10 sec each) to verify reaching works
2. **Compare learning curves** with original ball-tracking experiment
3. **Measure task selectivity** (do planning L1 neurons show different goals per task?)
4. **Optionally implement Option B** (velocity control) or **Option C** (obstacle avoidance) for comparison

---

## References

**Rationale for Fixed Target Task:**
- Validates motor learning without confounding predictive dynamics
- Tests whether task-specific weight matrices implement task selectivity
- Benchmark for comparing different inference architectures
- Closer to classical reaching tasks in motor neuroscience (Georgopoulos et al., 1988)

**Expected Improvements:**
- Faster learning (no dynamics to predict)
- Cleaner task representation (fixed target per trial)
- Better generalization (motor learns general reaching)
- Easier interpretation (simpler error signals)

---

**Implementation Date:** Nov 2, 2025  
**Status:** ✅ Complete - Ready for testing  
**Estimated Time to Run:** 30 sec (smoke test) to 5 min (full experiment)
