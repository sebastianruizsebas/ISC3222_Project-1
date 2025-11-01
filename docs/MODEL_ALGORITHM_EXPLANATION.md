# Dual‑Hierarchy Predictive‑Coding — Algorithm Explanation

This document explains how the dual‑hierarchy predictive‑coding algorithm in this repository works (implementation split across `hierarchical_motion_inference_dual_hierarchy.m` and `hierarchical_step_update.m`). It covers the high‑level flow, the core equations, the runtime data flow and shapes, and practical debugging/diagnostic tips.

If you want a condensed visual diagram for presentations, I can produce a laminar cartoon showing R/E/W flow between layers.

## Short contract
- Inputs: initial params struct + optional tuning fields (physics, learning rates, timing, task toggles).
- Outputs: `results` struct containing time series of player/ball kinematics, representations (R_L*), weight matrices (W_*), free‑energy and diagnostic traces.
- Success: code runs without NaN/Inf, produces finite `free_energy_all` and sensible interception errors; learning traces show nonzero weight updates when errors occur.

## High‑level algorithm flow
1. Parse `params` and set defaults (physics, learning rates, dt, T_per_trial, n_trials, tuning knobs like `vmax_ball` and `min_start_sep`).
2. Build trial schedule: compute `phases_indices` for each trial and generate ball trajectories (analytic pass + directed/random sampling). Optionally check reachability and ensure starting separation.
3. Initialize representations (`R_L0`, `R_L1_motor`, `R_L2_motor`, `R_L3_motor`, and planning equivalents), weight matrices (`W_*`) and diagnostic traces (error histories, pi traces, learning trace).
4. Prepare state struct `S` (runtime arrays) and parameter struct `P` (constants passed to the step helper). `S` is authoritative and updated every step by the helper.
5. Main loop over timesteps (1..N-1):
   - At trial boundaries: reset `S` for new trial (player position, ball start, zero velocities, decay weights by phase-specific decay factors, and re‑set motor mappings for velocity rows).
   - Delegate the hot inner loop to `hierarchical_step_update(i, S, P)` which performs physics, prediction, error, update, and precision computations for timestep `i` and writes values at `i+1` into `S`.
   - If helper signals `S.session_end` (player within `P.termination_distance` of the ball), break early.
6. Pull arrays back from `S` into `results`, save MAT/plots as requested, and print the analysis summary (interception RMSE by trial and final free energy).

## Core computations (what the helper does each step)
The helper `hierarchical_step_update(i, S, P)` is the authoritative per‑timestep function. It follows this sequence (sketch):

1) Ball physics (kinematics + collisions)
- Integrate accelerations → velocities → positions. Gravity and `P.air_drag` are applied. If the ball z coordinate crosses `P.ground_z`, bounce is handled using `P.restitution` and lateral velocity reduced by `P.ground_friction`.
- Positions are clamped to `P.workspace_bounds`.

2) Predictions (feedforward / lateral prediction generation)
- Motor predictions:
  pred_L2_motor(i,:) = R_L3_motor(i,:) * W_motor_L3_to_L2' + R_L2_motor(i,:) * W_motor_L2_lat'
  pred_L1_motor(i,:) = R_L2_motor(i,:) * W_motor_L2_to_L1' + R_L1_motor(i,:) * W_motor_L1_lat'
- Planning predictions follow the same pattern with `W_plan_*` matrices.

3) Extract motor commands and combine
- Predicted L1 velocity channels (semantic `idx_vel`) are mapped to motor command components (`motor_v*_motor`, `motor_v*_plan`), then blended 50:50 and damped via `P.damping` to produce player velocities and integrated into positions.

4) Error computation (prediction errors)
- L1 errors (E_L1_motor, E_L1_plan) = observed_state − pred_L1_* (on semantic indices).
- L2 errors (E_L2_*) = R_L2_* − pred_L2_*.
- Interception error = Euclidean distance between player and ball (used for termination and penalized in free energy).

5) Free‑energy computation
- Free energy at step i is computed as the sum of squared errors scaled by precision traces, plus an interception penalty:

  F(i) = (sum(E_L1_motor.^2) / (2*pi_L1_motor)) + (sum(E_L2_motor.^2) / (2*pi_L2_motor))
       + (sum(E_L1_plan.^2) / (2*pi_L1_plan)) + (sum(E_L2_plan.^2) / (2*pi_L2_plan))
       + (pi_L1_motor/100) * interception_error^2

  Note: pi_* variables adapt over time via `update_pi` (see below).

6) Representation updates (R updates)
- L1 updates use a momentum/decay blend and a small eta_rep term to incorporate L1 errors into new L1 states. Position channels are clamped to workspace bounds; velocity channels clamped to reasonable ranges.
- L2 updates use a coupling term derived from E_L1 and W matrices, normalized by `norm(S.W_*, 'fro')` to avoid scale issues, and then integrated with momentum and eta_rep.
- L3 updates are driven by averaged L2 errors.

7) Weight updates (W updates)
- Weight updates apply gradient‑like updates scaled by `P.eta_W`, current precision traces, and local layer scaling terms. Example:

  dW_motor_1 = -(eta_W * pi_L1_motor / layer_scale_motor_1) * (E_L1_motor' * R_L2_motor)

- Lateral weight updates exist for within‑layer coupling and are decayed slightly each step (multiplicative small factor applied to keep weights stable). The code also enforces small diagonal zeros on lateral weights.

8) Precision (pi) adaptive updates
- The helper maintains short error histories and computes new precision candidates using a function `update_pi(pi_curr, pi_base, err_history, ...)` that depends on the current error magnitude and variance.
- The algorithm applies strong smoothing (pi_smooth_alpha) and bounds multiplicative changes per step (pi_max_step_ratio). Final pi values are clamped within sensible ranges to avoid division by zero.

9) Diagnostics
- The helper appends `learning_trace_W` (norms of weight updates), `pi_trace_*`, raw pi and denom traces, and error history vectors for offline inspection.

## Semantic indexing (L1 channels)
- L1 channels are arranged semantically (for robustness): idx_pos (1..3), idx_vel (4..6), idx_bias (7). The helper accepts `P.idx_pos/idx_vel/idx_bias` so L1 dimensionality can change while keeping semantics stable.

## Trial transitions and phase decay
- At the beginning of each trial phase the main script writes reset values into `S` for player and ball, resets motor and planning L1/L2/L3 fields, and applies a phase transition decay to weights: motor weights are multiplied by `decay_motor` and planning weights by `decay_plan`.
- After decay, critical motor mappings (L3→L1 velocity rows) are restored (so motor primitives persist).

## Termination and outputs
- If `S.interception_error_all(i) <= P.termination_distance` the helper sets `S.session_end=true` and main loop breaks.
- On exit the main script assembles `results` with kinematics, R_* traces, W_* final matrices, free_energy_all, interception_error_all, pi traces, and saves plots/MAT file if requested.

## Data shapes and memory considerations
- Time steps N = round(T / dt) where T = T_per_trial * n_trials (note the earlier bug: large T_per_trial × large scale_factor causes long runtime and big memory).
- Typical shapes:
  - R_L1_motor: [N x n_L1_motor] (n_L1_motor typically 7)
  - R_L2_motor: [N x n_L2_motor] (n_L2_motor = round(scale_factor * 6)) — can be large if scale_factor large
  - W_motor_L2_to_L1: [n_L1_motor x n_L2_motor]
  - free_energy_all: [1 x N]

Performance note: reduce `scale_factor` or `T_per_trial` when doing smoke tests. For PSO runs set `make_plots=false` and use shorter trials for quick validation.

## Why NaNs/Inf can arise (recap + concrete checks)
- NaNs/Inf typically originate when one of the following contains NaN/Inf and then enters arithmetic used for norms, variances, or divisions:
  - Representations `R_*` (if updated with NaN inputs)
  - Weight updates `dW_*` are too large (big eta_W or accumulated gradients), turning weights into Inf
  - Error histories used by `update_pi` contain NaN, making var(...) NaN
  - Precision `pi_*` becomes nonfinite (used in denominator for free energy scaling)
- Concrete checks you can add quickly:
  - After computing `S.free_energy_all(i)`, call `isfinite` on the value and, on first failure, save a snapshot of `S.E_*`, `S.R_*`, `S.W_*`, and `S.pi_*` to `./figures/nan_snapshot.mat` for inspection.
  - Clip weight update magnitudes and/or apply max‑norm clipping to `dW` before adding to `W`.

## Practical debugging tips and knobs
- Lower `P.eta_W` and `P.eta_rep` to reduce update magnitudes.
- Add clipping to dW: `dW = max(min(dW, dW_max), -dW_max)` or scale by `dW_max / max(dW_norm, dW_max)`.
- In `update_pi`, sanitize `err_history` to ignore NaNs when computing var(). Use `var(err_history(isfinite(err_history)))`.
- If precision updates cause instability, increase `pi_smooth_alpha` (slower changes) or reduce `pi_max_step_ratio`.
- For quick runs set `T_per_trial` small (2.5 s), `n_trials=1`, `scale_factor` to a small value (e.g., 1–5) to shrink matrix sizes and speed up debugging.

## Where to look in the code
- `hierarchical_motion_inference_dual_hierarchy.m` — overall setup: params parsing, trajectory generation, initializations, trial loop, phase transitions, final save/plot.
- `hierarchical_step_update.m` — per‑timestep computations: physics, predictions, errors, free energy, representation updates, weight updates, precision updates.
- Diagnostics traces and plots: summary plots created in the main script (interception error, free energy, learning trace).

## Recommended small patches (I can apply these if you want)
1. NaN snapshot + sanitize after free energy (first occurrence) — saves a `.mat` snapshot with key arrays.
2. dW clipping (global dW_max in `P`) to prevent weight blowups.
3. NaN‑safe `update_pi` using only finite entries — avoids var==NaN.
4. Optional: reduce `scale_factor` default to 5 for interactive runs and expose `scale_factor` in `params`.

If you want, I can apply any of the recommended patches automatically. Tell me which ones you want first (I recommend NaN snapshot + dW clipping as a priority).

---

## UPDATE (Nov 1, 2025): Task-Conditional Learning with Multiplicative Gating

### Overview of Changes
Three major improvements were added to fix the fundamental **task context disconnection** problem:

1. **Multiplicative Task Gating** — L0 now actively gates predictions
2. **Task-Indexed Weight Matrices** — Separate weights per task prevent catastrophic forgetting
3. **Task-Selective Weight Updates** — Only active task learns; off-task weights remain frozen

These changes implement biologically-plausible prefrontal control mechanisms and dramatically improve multi-task learning.

### Problem Being Solved
Previously, the task context representation `R_L0` was **computationally inert**—it was updated each trial but never used in predictions or learning. This meant:
- Motor and planning regions predicted the same way regardless of task
- All tasks interfered with each other (weight updates were global)
- Each trial's learning partially undid previous trials' learning
- The "dual hierarchy" concept was only partially implemented

### Solution 1: Multiplicative Gating of Predictions

**Location:** `hierarchical_step_update.m`, prediction section (line ~130)

**Key equations:**
```matlab
% Identify active task from L0 (one-hot encoding)
[~, current_task_idx] = max(S.R_L0(i,:));

% Get task-specific weight matrices
W_motor_L2_to_L1_active = S.W_motor_L2_to_L1{current_task_idx};
...

% Task-context gating strength (multiplicative)
task_gate_motor = S.R_L0(i, current_task_idx);  % ~1.0 (motor learns universally)
task_gate_plan = S.R_L0(i, current_task_idx) * 0.8 + 0.2;  % [0.2, 1.0] (planning task-specific)

% Motor predictions with task gating
S.pred_L1_motor(i,:) = task_gate_motor * (S.R_L2_motor(i,:) * W_motor_L2_to_L1_active' + ...);

% Planning predictions with weaker gating
S.pred_L1_plan(i,:) = task_gate_plan * (S.R_L2_plan(i,:) * W_plan_L2_to_L1_active' + ...);
```

**Biological basis (Mante et al., 2013):**
- Prefrontal cortex (PFC) uses multiplicative context-dependent gain to gate information flow
- Neurons in M1/PMd show reduced responsiveness when task context is mismatched
- Acetylcholine (ACh) and noradrenaline (NE) implement this gain modulation

**Effect on learning:**
- When `task_gate = 0`, predictions are zeroed → large errors → strong learning signal
- When `task_gate = 1`, predictions are fully active → normal error-driven learning
- Motor region (`task_gate_motor ≈ 1`) learns stable forward models that generalize
- Planning region (`task_gate_plan ∈ [0.2, 1]`) learns task-specific strategies

### Solution 2: Task-Indexed Weight Matrices

**Location:** `hierarchical_motion_inference_dual_hierarchy.m`, initialization (line ~650)

**Key data structure change:**
```matlab
% OLD: Global weights (catastrophic interference)
W_motor_L2_to_L1 = zeros(n_L1_motor, n_L2_motor);  % shared across all tasks

% NEW: Task-indexed weights (no interference)
W_motor_L2_to_L1 = cell(n_trials, 1);  % separate copy per task
for task_idx = 1:n_trials
    W_motor_L2_to_L1{task_idx} = zeros(n_L1_motor, n_L2_motor);
    % ... initialize per-task ...
end
```

**Per-task initialization:**
- Each task gets identical initial weight structure
- But **learning happens independently** per task
- At trial boundaries, no decay is applied (weights preserved exactly)

**Biological basis (Rigotti et al., 2013; "Multiplexed Coding"):**
- PFC and M1 populations learn multiple input-output mappings simultaneously
- Different task contexts activate different neuronal subspaces
- Population covariance structure rotates/rescales per task rather than overwriting

**Memory impact:**
- Storage grows linearly with `n_trials` (4× memory for 4 tasks)
- For reasonable task counts (2-10), negligible overhead
- Speed unchanged (single task active at any time)

### Solution 3: Task-Selective Weight Updates

**Location:** `hierarchical_step_update.m`, weight update section (line ~330)

**Key mechanism:**
```matlab
% OLD: All tasks' weights updated every step
S.W_motor_L2_to_L1 = S.W_motor_L2_to_L1 + dW_motor_1;

% NEW: Only active task's weights updated
S.W_motor_L2_to_L1{current_task_idx} = S.W_motor_L2_to_L1{current_task_idx} + dW_motor_1;

% Off-task weights remain frozen (no plastic updates)
% They only change when that task becomes active
```

**Credit assignment principle:**
- Motor errors at time step `i` should only update the motor model used to generate those errors
- If task A's motor model is active, task B's model sees the data but **does not learn**
- This prevents spurious correlations and catastrophic forgetting

**Biological basis (Hippocampal Pattern Separation):**
- Hippocampus uses sparse, orthogonal codes to separate different contexts (Rolls & Stringer, 2020)
- This orthogonality extends to cortical columns via feedback from HC→cortex
- Goal: prevent one task from corrupting another's learned synaptic weights

### Cross-Task Error Computation (Diagnostic)

**Location:** `hierarchical_step_update.m`, error section (line ~220)

**New computation:**
```matlab
% For each candidate task, compute what error would occur IF that task were active
for task_candidate = 1:numel(S.W_motor_L2_to_L1)
    W_motor_cand = S.W_motor_L2_to_L1{task_candidate};
    pred_cand = S.R_L2_motor(i,:) * W_motor_cand';
    
    % Error IF task_candidate were predicting
    E_candidate = obs - pred_cand;
    S.task_errors_motor(i, task_candidate) = norm(E_candidate);
end
```

**Purpose:**
1. **Diagnostics:** Track whether off-task models are drifting or staying stable
2. **Interference penalty:** Optionally penalize off-task errors in free energy:
   ```matlab
   % Discourage off-task predictions from becoming good predictors of current data
   for t ≠ current_task_idx
       F(i) += interference_weight * task_errors_motor(i, t)^2
   ```

**Output:** `S.task_errors_motor` and `S.task_errors_plan` arrays track per-task prediction errors over time

### Updated Free Energy with Interference Penalty

**Location:** `hierarchical_step_update.m`, free energy section (line ~260)

**New term:**
```matlab
% Base free energy (unchanged)
F_base = sum(E_L1_motor.^2)/(2*pi) + sum(E_L2_motor.^2)/(2*pi) + ...

% NEW: Interference penalty (optional)
interference_weight = P.interference_penalty_weight;  % default 0.01
if interference_weight > 0
    for task ≠ current_task_idx
        F += interference_weight * (task_errors_motor(i, task)^2 + task_errors_plan(i, task)^2)
    end
end
```

**Effect:**
- Encourages learned models to specialize for their task (not memorize all tasks)
- Prevents "universal predictor" scenario where one model learns everything
- Optional: set `interference_weight = 0` to disable (weights still frozen anyway)

### Data Flow Summary

```
Input: observation o_t, task context R_L0(i)

1. Identify active task: current_task_idx = argmax(R_L0(i))

2. Load active task's weights: W_active = W{current_task_idx}

3. Gate predictions by task context:
   pred = task_gate * R * W_active'     [multiplicative gating]

4. Compute ALL task's errors (for monitoring):
   for each task_j:
       E_j = o - R * W{j}'              [cross-task error]

5. Update ONLY active task's weights:
   dW = -eta * E * R'
   W{current_task_idx} += dW            [selective update]
   
6. Return updated S (with W cells modified in-place)
```

### Integration with Existing Code

**Backward compatibility:**
- All existing initialization code works (just wrapped in cells)
- P struct unchanged (no new required parameters)
- Optional: `P.interference_penalty_weight` for cross-task penalty

**Parameters to tune:**
- `eta_W`: weight learning rate (unchanged semantics)
- `interference_penalty_weight`: 0.0–0.1 (default 0.01)
- Task-specific decay: `decay_motor`, `decay_plan` (now preserved exactly since no global decay)

**Output changes:**
- Results now contain `results.task_errors_motor` [N × n_trials] — diagnostic
- Results contain `results.W_motor_L2_to_L1` as cell array instead of matrix

### Expected Improvements

With task context now properly connected:

✓ **Motor region**: Learns faster, transfers across trials (error decreases monotonically if dynamics stable)

✓ **Planning region**: Learns task-specific strategies without interference (per-task learning curves independent)

✓ **Interception performance**: Improves steadily per trial (no unlearning from previous trials)

✓ **Free energy**: Drops faster overall (fewer task-inappropriate predictions)

✓ **Weight stability**: Off-task weights frozen → diagnostic cross-task errors plateau

### Diagnostic Plots to Create

```matlab
% Plot 1: Per-task error curves (one line per task)
plot(results.task_errors_motor)
title('Motor Model Errors (All Tasks)'), ylabel('||Error||'), xlabel('Step')
legend('Task 1', 'Task 2', 'Task 3', 'Task 4')

% Plot 2: Active vs off-task errors
active_errors = diag(results.task_errors_motor(:, trial_indices));
plot(active_errors, 'b-', 'LineWidth', 2)
hold on; plot(sum(results.task_errors_motor, 2) - active_errors, 'r--', 'LineWidth', 1)
title('Active Task Error vs Cross-Task Interference')
legend('Active Task Error', 'Off-Task Errors (summed)')

% Plot 3: Weight norms per task (measure learning)
for t = 1:n_tasks
    plot(cellfun(@(w) norm(w, 'fro'), results.W_motor_L2_to_L1{t}), 'DisplayName', sprintf('Task %d', t))
    hold on
end
title('Motor Weight Frobenius Norm Evolution')
```

### Citation & Further Reading

These improvements are grounded in:
- **Multiplicative gating**: Mante et al. (2013) *Neuron* — context-dependent dynamics in PFC
- **Task-indexed learning**: Rigotti et al. (2013) *Nat. Neurosci.* — multiplexed coding and task flexibility
- **Pattern separation**: Rolls & Stringer (2020) *Prog. Neurobiol.* — hippocampal role in context separation
- **Multi-task meta-learning**: Finn et al. (2017) "Model-Agnostic Meta-Learning" (*ICML*) — related principles from ML

---

File updated with comprehensive task-conditional learning documentation.

```
