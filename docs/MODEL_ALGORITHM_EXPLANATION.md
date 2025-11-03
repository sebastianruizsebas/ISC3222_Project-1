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

3) Extract motor commands (Pure Predictive Coding — Nov 2, 2025 FIX #2)
- Predicted L1 velocity channels (semantic `idx_vel`) are mapped to motor command components (`motor_v*_motor`, `motor_v*_plan`)
- **Motor executes ONLY motor region predictions** (pure, unblended):
  ```matlab
  final_motor_vx = S.motor_vx_motor(i);  % No blending with planning
  S.vx_player(i+1) = P.damping * S.vx_player(i) + final_motor_vx;
  ```
- Planning predictions (`motor_v*_plan`) are computed (for diagnostics) but NOT used in execution
- This ensures **execution = prediction** (core predictive coding axiom)
- Planning influences motor **through weight learning across trials**, not moment-by-moment blending

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

## UPDATE (Nov 2, 2025): Critical Implementation Fixes (v4) — Five Theoretical Corrections

### Overview: Five Fixes Correcting Nov 1 Documentation

On November 2, 2025, comprehensive code audit revealed **five critical misalignments** between the Nov 1 documentation and actual implementation. All fixes have been applied to the codebase. This section documents the *correct* implementation.

| # | Issue | Nov 1 (Incorrect) | Nov 2 (Correct) |
|---|-------|------|---------|
| FIX #1 | Prediction gating | `pred *= task_gate` | Pure: `pred` (no gating) |
| FIX #2 | Execution formula | Gated predictions | Pure predictions executed |
| FIX #3 | Representation gating | `R += eta * E * task_gate` | Pure: `R += eta * E` |
| FIX #4 | Off-task weights | Frozen (no learning) | Learn with penalty opposition |
| FIX #5 | PSO parameters | `alpha_precision_gain` unused | All 17 parameters now used ✓ |

**Key insight:** Task context enters via **weight indexing + interference penalty**, not multiplicative gating. This aligns with single-dopamine-signal control theory (synaptic tagging via D1/D2 pathways).

---

### FIX #1: Remove Multiplicative Task Gating from Predictions

**What Nov 1 said (incorrect):**
```matlab
task_gate_motor = S.R_L0(i, current_task_idx);  % ~1.0
S.pred_L1_motor(i,:) = task_gate_motor * (R_L2 * W');
```

**Actual Nov 2 implementation** (`hierarchical_step_update.m`, lines ~210-240):
```matlab
% Motor predictions are PURE (no multiplicative gating)
S.pred_L1_motor(i,:) = S.R_L2_motor(i,:) * W_motor_L2_to_L1_active' + S.R_L1_motor(i,:) * W_motor_L1_lat_active';

% Planning predictions are also PURE (no multiplicative gating)
S.pred_L1_plan(i,:) = S.R_L2_plan(i,:) * W_plan_L2_to_L1_active' + S.R_L1_plan(i,:) * W_plan_L1_lat_active';
```

**Why this is correct:**
- **Predictive coding axiom:** execution = prediction. Multiplicative gating violates this (they become inequal).
- **Theoretical coherence:** Task context selects WHICH weights are active, not the magnitude of prediction.
- **Biological mapping:** M1→muscle is unmodulated; task gating happens at PMd→M1 (planning input gating, not motor output).

---

### FIX #2: Pure Predictive Coding Execution

**Correct Nov 2 implementation** (`hierarchical_step_update.m`, lines ~200-215):
```matlab
% Extract velocity predictions from motor L1 (semantic indices idx_vel)
pred_vel_motor = S.pred_L1_motor(i, idx_vel);
pred_vel_plan = S.pred_L1_plan(i, idx_vel);

% PURE MOTOR EXECUTION (NO BLENDING - Strategy A, Nov 2, 2025)
% Motor executes learned motor predictions exclusively
final_motor_vx = S.motor_vx_motor(i);
final_motor_vy = S.motor_vy_motor(i);
final_motor_vz = S.motor_vz_motor(i);

% Kinematics: execution = prediction (predictive coding axiom)
S.vx_player(i+1) = P.damping * S.vx_player(i) + final_motor_vx;
S.vy_player(i+1) = P.damping * S.vy_player(i) + final_motor_vy;
S.vz_player(i+1) = P.damping * S.vz_player(i) + final_motor_vz;
```

**Key points:**
- ✅ **Execution = Prediction** (purely motor, 100% learned, no blending)
- ✅ **Planning is computed separately** (`motor_v*_plan` computed for diagnostics but NOT used in execution)
- ✅ **Planning influences motor through weight learning**, not moment-by-moment blending
- ✅ **Temporal timescales match neurobiology**: 
  - Motor output: immediate (10-50ms via M1 layer 5B)
  - Planning influence: slow (multi-trial via weight decay and context switching)

**Why removal of blending improves learning:**
- Motor learns clean reaching dynamics (not corrupted by planning signals)
- Planning learns ball dynamics (not confounded with motor execution)
- Error signals are causally valid (execution matches prediction for motor hierarchy)
- PSO convergence faster (fewer confounded parameters)

---

### FIX #3: Remove Task Gating from Representation Updates

**What Nov 1 said (incorrect):**
```matlab
S.R_L1_plan(i+1,:) = task_gate_plan * (S.R_L1_plan(i,:) + eta_rep * dR_L1_plan);
```

**Actual Nov 2 implementation** (`hierarchical_step_update.m`, lines ~490-550):
```matlab
% L1 updates: direct error integration (NO multiplicative task gate)
S.R_L1_motor(i+1,:) = P.momentum * S.R_L1_motor(i,:) + P.eta_rep * dR_L1_motor;
S.R_L1_plan(i+1,:) = P.momentum * S.R_L1_plan(i,:) + P.eta_rep * dR_L1_plan;

% L2 updates: coupled to L1 errors via learned weights (NO multiplicative task gate)
S.R_L2_motor(i+1,:) = P.momentum * S.R_L2_motor(i,:) + decay * P.eta_rep * delta_R_L2_motor;
S.R_L2_plan(i+1,:) = P.momentum * S.R_L2_plan(i,:) + decay * P.eta_rep * delta_R_L2_plan;

```% L3 updates: driven by averaged L2 errors (NO multiplicative task gate)
S.R_L3_motor(i+1,:) = S.R_L3_motor(i,:) + P.eta_rep * E_L3_motor_proj;
S.R_L3_plan(i+1,:) = S.R_L3_plan(i,:) + P.eta_rep * E_L3_plan_proj;
```

**Rationale:**
- Representations are hierarchical inference variables; they infer best estimate of latent state given observations.
- No reason to suppress inference based on task (all observations are valid for inference in all hierarchies).
- Task selectivity emerges naturally: different tasks produce different errors → different R trajectories.

---

### FIX #4: Interference Penalty Now Drives Weight Learning

**What Nov 1 said (incorrect/incomplete):**
> "Off-task weights remain frozen (no plastic updates)... penalty only affects free energy"

**Actual Nov 2 implementation** (`hierarchical_step_update.m`, lines ~560-600):
```matlab
% Compute normal weight update (for active task)
dW_motor_1 = -(P.eta_W * pi_L1_motor / layer_scale) * (E_L1_motor' * R_L2_motor);

% For EACH task, apply appropriate update rule
for task_idx = 1:numel(S.W_motor_L2_to_L1)
    if task_idx == current_task_idx
        % Active task: normal error-driven update
        S.W_motor_L2_to_L1{task_idx} = S.W_motor_L2_to_L1{task_idx} + dW_motor_1;
    else
        % Off-task: interference penalty opposes learning on current data
        if P.interference_penalty_weight > 0
            % Compute what error this off-task weight would produce on current data
            W_off = S.W_motor_L2_to_L1{task_idx};
            pred_off = S.R_L2_motor(i,:) * W_off';
            E_off = S.E_L1_motor(i,:) - pred_off;
            
            % Penalty gradient: pushes weights AWAY from current task's manifold
            penalty_gradient = P.interference_penalty_weight * (E_off' * R_L2_motor);
            
            % Update: normal learning MINUS penalty opposition
            S.W_motor_L2_to_L1{task_idx} = S.W_motor_L2_to_L1{task_idx} + dW_motor_1 - penalty_gradient;
        end
    end
end
```

**Effect on learning:**
- **Active task weights**: Learn normally (error gradient when large errors occur)
- **Off-task weights**: Learn slowly (error gradient opposed by penalty term)
- **Result**: Natural task specialization (weights separate without explicit freezing)

**Updated free energy:**
```matlab
F(i) = sum(E_L1_motor.^2)/(2*pi_L1_motor) + ... [base errors]
     + interference_penalty_weight * sum([E_off_task.^2 for all off-tasks])
```

---

### FIX #5: All 17 PSO Parameters Now Active

**Problem identified:** PSO computed `alpha_precision_gain` and `pi_L*_max` bounds but code used hard-coded values.

**Correct implementation** (`hierarchical_step_update.m`, lines ~730-810):
```matlab
% Compute error-driven exponential scaling
L1_motor_error_mag = sqrt(sum(S.E_L1_motor(i,:).^2));
error_scale_factor = 0.1;  % Normalize error to [0,1] range
L1_motor_error_norm = min(1.0, L1_motor_error_mag * error_scale_factor);

% Exponential scaling uses PSO-optimized parameter (NOW ACTIVE)
alpha_precision = P.alpha_precision_gain;
precision_multiplier = exp(alpha_precision * L1_motor_error_norm);

% Apply PSO-provided bounds (NOW ACTIVE)
pi_L1_motor_bound_min = P.pi_bounds.L1_motor(1);
pi_L1_motor_bound_max = P.pi_bounds.L1_motor(2);

% Update precision with clamping to bounds
pi_L1_motor_new = precision_multiplier * pi_L1_motor_curr;
pi_L1_motor_new = max(pi_L1_motor_bound_min, min(pi_L1_motor_bound_max, pi_L1_motor_new));
```

**All 17 PSO parameters now used:**
1. `eta_rep` — Representation learning rate
2. `eta_W` — Weight matrix learning rate
3. `momentum` — Representation momentum
4. `weight_decay` — Lateral weight decay
5. `interference_penalty_weight` — Off-task opposition strength
6. `alpha_precision_gain` — ✓ NOW USED (precision scaling rate)
7. `pi_L1_motor_max` — ✓ NOW USED (motor L1 ceiling)
8. `pi_L2_motor_max` — ✓ NOW USED (motor L2 ceiling)
9. `pi_L1_plan_max` — ✓ NOW USED (planning L1 ceiling)
10. `pi_L2_plan_max` — ✓ NOW USED (planning L2 ceiling)
11. `gravity` — Physics: ball z-acceleration
12. `restitution` — Physics: bounce energy
13. `ground_friction` — Physics: lateral bounce friction
14. `air_drag` — Physics: velocity decay
15. `vmax_ball` — Physics: max ball speed
16. `min_start_sep` — Physics: min player-ball separation
17. `motor_gain` — Execution mapping: velocity scale

---

### Corrected Data Flow (Nov 2 — Strategy A: Pure Motor Execution)

```
INPUT:
  Observation o_t
  Task context: current_task_idx = argmax(R_L0(i))
  
STEP 1: LOAD ACTIVE TASK WEIGHTS
  W_motor_active = W_motor_L2_to_L1{current_task_idx}
  W_plan_active = W_plan_L2_to_L1{current_task_idx}
  
STEP 2: PURE PREDICTIONS (NO MULTIPLICATIVE GATING)
  pred_L1_motor = R_L2_motor * W_motor_active'
  pred_L1_plan = R_L2_plan * W_plan_active'
  
STEP 3: PURE MOTOR EXECUTION (NO BLENDING - Strategy A)
  final_motor_command = motor_vx_motor(i), motor_vy_motor(i), motor_vz_motor(i)
  execute = final_motor_command  [100% motor, no planning blending]
  
  NOTE: Planning predictions computed but NOT executed
        Planning influences motor ONLY through:
        - Weight decay at trial boundaries (multi-trial timescale)
        - Shared error signal from ball observations
  
STEP 4: COMPUTE ERRORS (MOTOR LEARNS FROM MOTOR-DRIVEN EXECUTION)
  E_motor = observation - pred_motor  [error validates motor's prediction]
  E_plan = observation - pred_plan    [error validates planning's prediction]
  
STEP 5: UPDATE REPRESENTATIONS (DIRECT, NO GATING)
  R_L1 += eta_rep * E_motor  [no multiplicative modulation]
  R_L2 += eta_rep * (W' * E_L1)
  R_L3 += eta_rep * avg(E_L2)
  
STEP 6: UPDATE ACTIVE TASK'S WEIGHTS (NORMAL ERROR-DRIVEN)
  dW_active = -eta_W * pi * E' * R
  W{current_task} += dW_active
  
STEP 7: UPDATE OFF-TASK WEIGHTS (INTERFERENCE PENALTY OPPOSITION)
  For each task_j ≠ current_task_idx:
    E_off = observation - R * W{task_j}'
    penalty_grad = interference_weight * E_off' * R
    W{task_j} += dW_active - penalty_grad
    
STEP 8: UPDATE PRECISION (ERROR-DRIVEN, PSO-CONTROLLED)
  error_norm = min(1.0, ||E|| * 0.1)
  pi_new = exp(alpha_precision_gain * error_norm) * pi_curr
  pi_new = clamp(pi_new, min_bound, max_bound)
  
OUTPUT: Updated S with R, W, pi, free_energy
```

**Key Difference from Nov 1:**
- STEP 3: Changed from "50:50 blending" to "100% pure motor execution"
- Result: Execution = Prediction (predictive coding axiom satisfied)
- Motor and planning learn independently without mutual interference

---

### Summary of Corrections

| Component | Nov 1 (Incorrect) | Nov 2 (Correct - Strategy A) |
|-----------|------|---------|
| **Predictions** | `pred *= task_gate` | Pure: no gating ✓ |
| **Execution** | Blended: 50% motor + 50% planning | **Pure motor (100% motor): execution = prediction** ✓ |
| **Representations** | `R += eta * E * task_gate` | Pure: `R += eta * E` ✓ |
| **Off-task learning** | Frozen (no updates) | Active with penalty opposition ✓ |
| **PSO parameters** | 10/17 used | 17/17 used ✓ |
| **Theory alignment** | Inconsistent (pred ≠ exec) | **Consistent (pred = exec) — Predictive Coding Axiom Satisfied ✓** |

---

## Strategy A Implementation Summary (Nov 2, 2025)

**Problem Solved:** Removed 50:50 blending that violated predictive coding principle (execution ≠ prediction)

**Key Changes:**
1. Motor now executes pure learned predictions (no planning mixture)
2. Planning influences motor through weight learning, not execution blending
3. Error signals are now causally valid (execution exactly matches what motor hierarchy predicts)
4. Motor and planning learn independently (no mutual interference on same step)

**Expected Improvements:**
- ✅ Faster PSO convergence (fewer confounded parameters)
- ✅ Higher final interception accuracy (cleaner motor control)
- ✅ Better generalization (motor learns general reaching, not task-specific tricks)
- ✅ Neuroscientific coherence (aligns with laminar motor output architecture)

All corrections have been applied to the codebase as of Nov 2, 2025.

```
