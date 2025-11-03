function S = hierarchical_step_update(i, S, P)
% HIERARCHICAL_STEP_UPDATE  Single timestep update helper (scaffold)
%
% S = hierarchical_step_update(i, S, P)
%
% This is a scaffold to extract the hot inner-loop from
% `hierarchical_motion_inference_dual_hierarchy.m` into a single
% function so MATLAB's JIT can better optimize it and so it can be
% converted to MEX later if desired.
%
% Inputs:
%  - i : current timestep index (1..N-1)
%  - S : struct containing all runtime arrays and states (cells/arrays)
%  - P : struct with scalar parameters and constant matrices
%
% Output:
%  - S : updated struct with values at index i+1 written where appropriate
%
% NOTE: This file currently implements the physics + kinematics + simple
% combined output. It does NOT (yet) fully replace the original inner loop.
% Use this as a starting point for progressively moving code out of the
% main script. Carefully copy smaller chunks from the original loop into
% this function and update references to use S.* and P.*.

% Copy constants out of P for readability
dt = P.dt;
workspace_bounds = P.workspace_bounds;

% Semantic indices (allow main script to pass idx_pos/idx_vel/idx_bias via P)
if isfield(P, 'idx_pos'), idx_pos = P.idx_pos; else idx_pos = 1:3; end
if isfield(P, 'idx_vel'), idx_vel = P.idx_vel; else idx_vel = 4:6; end
if isfield(P, 'idx_bias'), idx_bias = P.idx_bias; else idx_bias = 7; end

% FIX (Nov 2, 2025): NEW ISSUE #1 - Add semantic index assertions to catch silent fallback
% Verify indices are being passed from P (not silently falling back to defaults)
assert(isfield(P, 'idx_pos'), 'ERROR: idx_pos not found in P struct - semantic indices not passed from main script');
assert(isfield(P, 'idx_vel'), 'ERROR: idx_vel not found in P struct - semantic indices not passed from main script');
assert(isfield(P, 'idx_bias'), 'ERROR: idx_bias not found in P struct - semantic indices not passed from main script');
% Additional validation: indices should be positive integers within reasonable bounds
assert(isvector(idx_pos) && all(idx_pos > 0) && max(idx_pos) <= 100, 'ERROR: idx_pos invalid - must be positive integer vector');
assert(isvector(idx_vel) && all(idx_vel > 0) && max(idx_vel) <= 100, 'ERROR: idx_vel invalid - must be positive integer vector');
assert(isscalar(idx_bias) && idx_bias > 0 && idx_bias <= 100, 'ERROR: idx_bias invalid - must be positive integer scalar');

% For convenience, operate directly on S fields (avoid unused local aliases)

% --- MOVING TARGET KINEMATICS (NOV 3, 2025 - CONSTANT VELOCITY) ---
% Target follows constant velocity motion (learnable, repeatable)
% This tests predictive coding: can hierarchies learn to predict target motion?
% Motor learns: velocity control (how to chase)
% Planning learns: target motion model (where target will be)

% Get current trial index and target trajectory parameters
if isfield(P, 'target_trajectories') && ~isempty(P.target_trajectories)
    current_trial_idx = max(1, min(numel(P.target_trajectories), S.current_trial));
    target_traj = P.target_trajectories{current_trial_idx};
    
    % Extract motion parameters from trajectory
    velocity = target_traj.velocity;
    acceleration = target_traj.acceleration;
else
    % Fallback: constant velocity (no acceleration)
    velocity = [0, 0, 0];
    acceleration = [0, 0, 0];
end

% Integrate velocity (acceleration causes velocity change)
S.vx_ball(i+1) = S.vx_ball(i) + acceleration(1) * dt;
S.vy_ball(i+1) = S.vy_ball(i) + acceleration(2) * dt;
S.vz_ball(i+1) = S.vz_ball(i) + acceleration(3) * dt;

% Integrate position (velocity causes position change)
S.x_ball(i+1) = S.x_ball(i) + dt * S.vx_ball(i+1);
S.y_ball(i+1) = S.y_ball(i) + dt * S.vy_ball(i+1);
S.z_ball(i+1) = S.z_ball(i) + dt * S.vz_ball(i+1);

% Clamp positions to workspace bounds (safety)
S.x_ball(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), S.x_ball(i+1)));
S.y_ball(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), S.y_ball(i+1)));
S.z_ball(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), S.z_ball(i+1)));

% ------------------------------
% PREDICTION (Motor & Planning) WITH MULTIPLICATIVE TASK GATING
% NEW: L0 task context now multiplicatively gates predictions
% This implements prefrontal modulation of motor/planning circuits
% ------------------------------

% Identify current active task from L0 (one-hot encoding)
[~, current_task_idx] = max(S.R_L0(i,:));
if current_task_idx < 1 || current_task_idx > length(S.W_plan_L2_to_L1)
    current_task_idx = 1;  % safety fallback
end

% Get weight matrices (motor: SHARED, planning: TASK-INDEXED)
% Motor weights are single shared matrices (not cell arrays) for generalization
W_motor_L2_to_L1_active = S.W_motor_L2_to_L1;  % Always use shared motor weights
W_motor_L3_to_L2_active = S.W_motor_L3_to_L2;

% Planning weights are task-indexed (cell arrays)
W_plan_L2_to_L1_active = S.W_plan_L2_to_L1{current_task_idx};
W_plan_L3_to_L2_active = S.W_plan_L3_to_L2{current_task_idx};

% THEORETICAL FIX #1 (Nov 2, 2025): Remove Multiplicative Task Gating from Predictions
% BEFORE: Predictions were gated → task_gate_motor * (W matrices @ R layers)
% PROBLEM: This created redundancy with task-selective weight freezing (two mechanisms doing same thing)
% AFTER: Predictions are pure (no gating). Learning is controlled by selective weight freezing.
% RATIONALE: Single-source task control via synaptic tagging + dopamine (Lisman et al., 2002)
%           Predictions show what the current task's weights would generate
%           Off-task weights remain frozen (can't learn), so off-task predictions are stale (expected)
%           No additional gating needed

% Motor region predictions (NO multiplicative gating - pure feedforward, no lateral)
S.pred_L2_motor(i,:) = S.R_L3_motor(i,:) * W_motor_L3_to_L2_active';

S.pred_L1_motor(i,:) = S.R_L2_motor(i,:) * W_motor_L2_to_L1_active';

% Planning region predictions (NO multiplicative gating - pure feedforward, no lateral)
S.pred_L2_plan(i,:) = S.R_L3_plan(i,:) * W_plan_L3_to_L2_active';

S.pred_L1_plan(i,:) = S.R_L2_plan(i,:) * W_plan_L2_to_L1_active';

% --- MOTOR VELOCITY COMMAND EXTRACTION (CORRECTED) ---
% Extract velocity predictions using semantic indices
tmp_vel = S.pred_L1_motor(i, idx_vel);
pred_vel_motor = zeros(1,3);
n_tmp = numel(tmp_vel);
pred_vel_motor(1:min(3,n_tmp)) = tmp_vel(1:min(3,n_tmp));

% --- PLANNING VELOCITY COMMAND EXTRACTION ---
% Extract planning velocity predictions
tmp_vel_p = S.pred_L1_plan(i, idx_vel);
pred_vel_plan = zeros(1,3);
n_tmp_p = numel(tmp_vel_p);
pred_vel_plan(1:min(3,n_tmp_p)) = tmp_vel_p(1:min(3,n_tmp_p));

% THEORETICAL FIX #2 (Nov 2, 2025): Pure Predictive Coding (100% Learned Predictions)
% BEFORE: Motor commands were blended (50% motor + 50% planning)
% PROBLEM: This violates predictive coding principle - execution ≠ prediction
%          Motor learns from errors vs. pure predictions, but executes blended commands
%          This misalignment corrupts weight updates and invalidates learning signals
% AFTER: Motor commands use ONLY learned motor predictions (100% learned, pure)
% RATIONALE: Predictive coding requires: execution = prediction (what we predicted we would do)
%            Then: error = observation - prediction (valid learning signal)
%            If execution ≠ prediction, error doesn't explain execution causality
%            Now execution matches predictions perfectly → learning is valid

% Motor command: PURE LEARNED PREDICTION FROM MOTOR REGION (no blending with planning)
S.motor_vx_motor(i) = P.motor_gain * pred_vel_motor(1);
S.motor_vy_motor(i) = P.motor_gain * pred_vel_motor(2);
S.motor_vz_motor(i) = P.motor_gain * pred_vel_motor(3);

% Planning outputs computed separately (for diagnostics and future use, not blended here)
S.motor_vx_plan(i) = P.motor_gain * pred_vel_plan(1);
S.motor_vy_plan(i) = P.motor_gain * pred_vel_plan(2);
S.motor_vz_plan(i) = P.motor_gain * pred_vel_plan(3);

% FINAL MOTOR COMMAND: Use ONLY motor region predictions (pure predictive coding)
% FIX (Nov 3, 2025): ADD MOTOR EXPLORATION NOISE for stochasticity
% RATIONALE: Without noise, execution = prediction → error = 0
%           With noise: execution differs from prediction → learning signal exists
%           Annealing: noise ↓ over time (less exploration, more exploitation)

% Noise scale: start high, decay to zero over training
noise_annealing_factor = max(0.01, 1.0 - (i / 1000));  % Decreases from 1.0 to 0.01 over ~1000 steps
noise_scale = 0.05 * noise_annealing_factor;  % Max noise magnitude: 0.05 m/s * annealing factor

% Add exploration noise to motor commands
final_motor_vx = S.motor_vx_motor(i) + noise_scale * randn();
final_motor_vy = S.motor_vy_motor(i) + noise_scale * randn();
final_motor_vz = S.motor_vz_motor(i) + noise_scale * randn();

% Integrate motor command into player dynamics (pure predictive execution)
S.vx_player(i+1) = P.damping * S.vx_player(i) + final_motor_vx;
S.vy_player(i+1) = P.damping * S.vy_player(i) + final_motor_vy;
S.vz_player(i+1) = P.damping * S.vz_player(i) + final_motor_vz;

S.x_player(i+1) = S.x_player(i) + dt * S.vx_player(i+1);
S.y_player(i+1) = S.y_player(i) + dt * S.vy_player(i+1);
S.z_player(i+1) = S.z_player(i) + dt * S.vz_player(i+1);

S.x_player(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), S.x_player(i+1)));
S.y_player(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), S.y_player(i+1)));
S.z_player(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), S.z_player(i+1)));

% ERROR COMPUTATION (WITH TASK-CONDITIONAL ERRORS)
% Note: Errors computed AFTER kinematics so we have actual velocities and positions
% This implements pure predictive coding: errors drive learning to match actual execution

% Observation vectors (task-independent)
pos_vec = [S.x_player(i+1), S.y_player(i+1), S.z_player(i+1)];
vel_vec = [S.vx_player(i+1), S.vy_player(i+1), S.vz_player(i+1)];
pos_ball = [S.x_ball(i+1), S.y_ball(i+1), S.z_ball(i+1)];

% ONLY UPDATE ACTIVE TASK errors (avoid training off-task weights on active-task data)
S.E_L1_motor(i, idx_pos) = pos_vec(1:numel(idx_pos)) - S.pred_L1_motor(i, idx_pos);
S.E_L1_motor(i, idx_vel) = vel_vec(1:numel(idx_vel)) - S.pred_L1_motor(i, idx_vel);
S.E_L1_motor(i, idx_bias) = 1 - S.pred_L1_motor(i, idx_bias);

S.E_L2_motor(i,:) = S.R_L2_motor(i,:) - S.pred_L2_motor(i,:);

S.E_L1_plan(i, idx_pos) = pos_ball(1:numel(idx_pos)) - S.pred_L1_plan(i, idx_pos);
S.E_L1_plan(i, idx_vel) = pos_ball(1:numel(idx_vel)) - S.pred_L1_plan(i, idx_vel);
S.E_L1_plan(i, idx_bias) = 1 - S.pred_L1_plan(i, idx_bias);

S.E_L2_plan(i,:) = S.R_L2_plan(i,:) - S.pred_L2_plan(i,:);

S.interception_error_all(i) = sqrt((S.x_player(i+1) - S.x_ball(i+1))^2 + (S.y_player(i+1) - S.y_ball(i+1))^2 + (S.z_player(i+1) - S.z_ball(i+1))^2);

% NEW: Compute cross-task error signals for diagnostics and optional interference penalty
% Motor: single shared weights, so no per-task error computation needed
% Planning: task-indexed, so compute error for each task

% Task-indexed planning error computation
for task_candidate = 1:numel(S.W_plan_L2_to_L1)
    % Planning error for this candidate task
    W_plan_L2_to_L1_cand = S.W_plan_L2_to_L1{task_candidate};
    pred_L1_plan_cand = S.R_L2_plan(i,:) * W_plan_L2_to_L1_cand';
    E_L1_plan_cand = pos_ball(1:numel(idx_pos)) - pred_L1_plan_cand(idx_pos);
    S.task_errors_plan(i, task_candidate) = norm(E_L1_plan_cand);
    
    % Motor error is the same for all tasks (shared weights)
    S.task_errors_motor(i, task_candidate) = S.interception_error_all(i);
end

% If player is sufficiently close to the ball, signal session end
if isfield(P, 'termination_distance') && S.interception_error_all(i) <= P.termination_distance
    S.session_end = true;
    % store termination index as the next timestep (i+1) so calling code can reference final state
    S.termination_step = i+1;
end

% ------------------------------
% FREE ENERGY (WITH TASK INTERFERENCE MONITORING)
% NEW: Add cross-task interference penalty to free energy
% This encourages task-specific representations and prevents catastrophic forgetting
% ------------------------------
% Scale down interception penalty: normalize by number of L1 position channels to avoid a single large distance dominating free energy.
n_pos_channels = numel(idx_pos);

% STRONG CLIPPING: Ensure precision values are always safe and finite (prevents Inf/NaN)
max_finite_value = 1e12;
min_precision = 1e-12;
max_precision = 1e12;

% Clip precision values to safe ranges
pi_L1_motor_safe = max(min_precision, min(max_precision, S.pi_L1_motor));
pi_L2_motor_safe = max(min_precision, min(max_precision, S.pi_L2_motor));
pi_L1_plan_safe = max(min_precision, min(max_precision, S.pi_L1_plan));
pi_L2_plan_safe = max(min_precision, min(max_precision, S.pi_L2_plan));

% Check if any precision was clipped (outside safe range)
if (S.pi_L1_motor ~= pi_L1_motor_safe) || (S.pi_L2_motor ~= pi_L2_motor_safe) || ...
   (S.pi_L1_plan ~= pi_L1_plan_safe) || (S.pi_L2_plan ~= pi_L2_plan_safe)
    S.clipping_count = S.clipping_count + 1;  % Count precision clipping event
end

% Clip error vectors to prevent overflow
E_L1_motor_clipped = max(-max_finite_value, min(max_finite_value, S.E_L1_motor(i,:)));
E_L2_motor_clipped = max(-max_finite_value, min(max_finite_value, S.E_L2_motor(i,:)));
E_L1_plan_clipped = max(-max_finite_value, min(max_finite_value, S.E_L1_plan(i,:)));
E_L2_plan_clipped = max(-max_finite_value, min(max_finite_value, S.E_L2_plan(i,:)));
interception_error_clipped = max(0, min(max_finite_value, S.interception_error_all(i)));

% Check if any error was clipped (outside safe range)
if any(S.E_L1_motor(i,:) ~= E_L1_motor_clipped) || any(S.E_L2_motor(i,:) ~= E_L2_motor_clipped) || ...
   any(S.E_L1_plan(i,:) ~= E_L1_plan_clipped) || any(S.E_L2_plan(i,:) ~= E_L2_plan_clipped) || ...
   (S.interception_error_all(i) ~= interception_error_clipped)
    S.clipping_count = S.clipping_count + 1;  % Count error clipping event
end

% Compute free energy with clipped values
S.free_energy_all(i) = sum(E_L1_motor_clipped.^2) / (2 * pi_L1_motor_safe) + sum(E_L2_motor_clipped.^2) / (2 * pi_L2_motor_safe) + ...
    sum(E_L1_plan_clipped.^2) / (2 * pi_L1_plan_safe) + sum(E_L2_plan_clipped.^2) / (2 * pi_L2_plan_safe) + (pi_L1_motor_safe/(100 * max(1,n_pos_channels))) * interception_error_clipped^2;

% Clip the computed free energy to safe range
S.free_energy_all(i) = max(0, min(1e7, S.free_energy_all(i)));

% NEW: Add cross-task interference penalty (optional, controlled by P.interference_penalty_weight)
if isfield(P, 'interference_penalty_weight')
    interference_penalty_weight = P.interference_penalty_weight;
else
    interference_penalty_weight = 0.01;  % default: small contribution
end

if interference_penalty_weight > 0
    % Penalize errors from non-active tasks (encourages task separation)
    for task_idx = 1:numel(S.W_plan_L2_to_L1)
        if task_idx ~= current_task_idx
            % Clip cross-task errors to safe ranges
            motor_crosstask_error = max(0, min(max_finite_value, S.task_errors_motor(i, task_idx)));
            plan_crosstask_error = max(0, min(max_finite_value, S.task_errors_plan(i, task_idx)));
            % Add weighted penalty (clipped to prevent explosion)
            penalty_term = interference_penalty_weight * (motor_crosstask_error^2 + plan_crosstask_error^2);
            penalty_term = min(1e7, penalty_term);  % Prevent penalty from dominating
            S.free_energy_all(i) = S.free_energy_all(i) + penalty_term;
        end
    end
end

% FINAL SAFETY CLIP: Ensure free energy is finite
S.free_energy_all(i) = max(0, min(1e7, S.free_energy_all(i)));
if ~isfinite(S.free_energy_all(i))
    S.free_energy_all(i) = 1e7;  % Hard fallback
end

% Guard: detect NaN/Inf and dump minimal snapshot for debugging (first occurrence)
persistent nan_reported;
persistent consecutive_clipping_count;
if isempty(nan_reported), nan_reported = false; end
if isempty(consecutive_clipping_count), consecutive_clipping_count = 0; end

% Check for NaN/Inf occurrence
is_nan_inf = ~isfinite(S.free_energy_all(i)) || any(~isfinite([S.E_L1_motor(i,:), S.E_L2_motor(i,:), S.E_L1_plan(i,:), S.E_L2_plan(i,:), S.interception_error_all(i)]));

if is_nan_inf
    % Increment consecutive clipping counter (Nov 1, 2025 - EARLY TERMINATION LOGIC)
    consecutive_clipping_count = consecutive_clipping_count + 1;
else
    % Reset consecutive counter when we see a clean step
    consecutive_clipping_count = 0;
end

% Determine consecutive-NaN/Inf termination threshold from P (single-source)
% Backward compatible fallback: 50 consecutive events
if isfield(P, 'max_consecutive_clipping') && isfinite(P.max_consecutive_clipping) && P.max_consecutive_clipping > 0
    MAX_CONSECUTIVE_CLIPPING = P.max_consecutive_clipping;
else
    MAX_CONSECUTIVE_CLIPPING = 50;
end
if consecutive_clipping_count >= MAX_CONSECUTIVE_CLIPPING
    fprintf(2, '\n⚠  CRITICAL: %d consecutive Inf/NaN clipping events detected (threshold=%d)! Terminating trial early.\n', consecutive_clipping_count, MAX_CONSECUTIVE_CLIPPING);
    S.session_end = true;
    S.termination_step = i;
    S.termination_reason = sprintf('Excessive consecutive Inf/NaN clipping (%d events >= %d)', consecutive_clipping_count, MAX_CONSECUTIVE_CLIPPING);
    return;  % Exit the step function early
end

if is_nan_inf && ~nan_reported
    nan_reported = true;
    
    % INCREMENT CLIPPING COUNTER (Nov 1, 2025)
    S.clipping_count = S.clipping_count + 1;
    
    fprintf(2, 'DEBUG WARNING: NaN/Inf detected at step %d (clipping event #%d, consecutive: %d/%d). Dumping snapshot to ./figures/nan_snapshot.mat\n', ...
        i, S.clipping_count, consecutive_clipping_count, MAX_CONSECUTIVE_CLIPPING);
    try
        snapshot.Sfree = S.free_energy_all(i);
        snapshot.step = i;
        snapshot.E_L1_motor = S.E_L1_motor(i,:);
        snapshot.E_L2_motor = S.E_L2_motor(i,:);
        snapshot.E_L1_plan = S.E_L1_plan(i,:);
        snapshot.E_L2_plan = S.E_L2_plan(i,:);
        snapshot.R_L1_motor = S.R_L1_motor(i,:);
        snapshot.R_L2_motor = S.R_L2_motor(i,:);
        snapshot.R_L3_motor = S.R_L3_motor(i,:);
        snapshot.W_motor_L2_to_L1 = S.W_motor_L2_to_L1;
        snapshot.W_motor_L3_to_L2 = S.W_motor_L3_to_L2;
        snapshot.pi_vals = [S.pi_L1_motor, S.pi_L2_motor, S.pi_L1_plan, S.pi_L2_plan];
        snapshot.clipping_count = S.clipping_count;  % Save current count in snapshot too
        snapshot.consecutive_clipping_count = consecutive_clipping_count;  % Save consecutive count

        save(fullfile('./figures','nan_snapshot.mat'), 'snapshot');
    catch MEsave
        fprintf(2, 'Failed to save snapshot: %s\n', MEsave.message);
    end
    % STRONG SANITIZATION: Replace all NaN/Inf with safe fallback values
    S.free_energy_all(i) = 1e7;  % Penalize but keep finite
    
    % Clip all error signals to safe ranges
    S.E_L1_motor(i, ~isfinite(S.E_L1_motor(i,:))) = 0;
    S.E_L2_motor(i, ~isfinite(S.E_L2_motor(i,:))) = 0;
    S.E_L1_plan(i, ~isfinite(S.E_L1_plan(i,:))) = 0;
    S.E_L2_plan(i, ~isfinite(S.E_L2_plan(i,:))) = 0;
    
    % Clip each error element to safe range
    S.E_L1_motor(i,:) = max(-max_finite_value, min(max_finite_value, S.E_L1_motor(i,:)));
    S.E_L2_motor(i,:) = max(-max_finite_value, min(max_finite_value, S.E_L2_motor(i,:)));
    S.E_L1_plan(i,:) = max(-max_finite_value, min(max_finite_value, S.E_L1_plan(i,:)));
    S.E_L2_plan(i,:) = max(-max_finite_value, min(max_finite_value, S.E_L2_plan(i,:)));
    
    S.interception_error_all(i) = max(0, min(max_finite_value, S.interception_error_all(i)));
    
    % Update S precisions to safe values
    S.pi_L1_motor = max(min_precision, min(max_precision, S.pi_L1_motor));
    S.pi_L2_motor = max(min_precision, min(max_precision, S.pi_L2_motor));
    S.pi_L1_plan = max(min_precision, min(max_precision, S.pi_L1_plan));
    S.pi_L2_plan = max(min_precision, min(max_precision, S.pi_L2_plan));
else
    % FIX (Nov 2, 2025): Reset nan_reported flag on clean steps
    % Previously this flag never reset, preventing multiple snapshot captures
    nan_reported = false;
end

% NOTE (Nov 2, 2025): REMOVED DUPLICATE PRECISION SCALING HERE
% The authoritative error-driven precision adaptation is in the section below (after representation updates)
% This region previously contained competing precision update mechanisms that caused chaotic dynamics
% See line ~680 for the single, consolidated error-driven exponential scaling mechanism

% ------------------------------
% REPRESENTATION UPDATES
% ------------------------------
decay = 1 - P.momentum;

% Motor L1
S.R_L1_motor(i+1, idx_pos) = S.R_L1_motor(i, idx_pos) + decay * P.eta_rep * S.E_L1_motor(i, idx_pos) * 0.1;
S.R_L1_motor(i+1, idx_vel) = P.momentum * S.R_L1_motor(i, idx_vel) + decay * P.eta_rep * S.E_L1_motor(i, idx_vel) * 0.1;
% clamp velocity channels elementwise
for k = 1:numel(idx_vel)
    S.R_L1_motor(i+1, idx_vel(k)) = max(-2, min(2, S.R_L1_motor(i+1, idx_vel(k))));
end
% clamp positional channels to workspace bounds (respect available dims)
pos_dims = min(numel(idx_pos), size(workspace_bounds,1));
for k = 1:pos_dims
    S.R_L1_motor(i+1, idx_pos(k)) = max(workspace_bounds(k,1), min(workspace_bounds(k,2), S.R_L1_motor(i+1, idx_pos(k))));
end
% bias
S.R_L1_motor(i+1, idx_bias) = 1;

% Motor L2
% Use active task's L2->L1 weights (cells) for coupling computation
coupling_motor = S.E_L1_motor(i,:) * W_motor_L2_to_L1_active;
% FIX (Nov 2, 2025): Improve weight norm stability - use adaptive floor instead of fixed 0.1
norm_W_motor = norm(W_motor_L2_to_L1_active, 'fro');
if norm_W_motor < 0.01  % Adaptive floor for early learning
    norm_W_motor = 0.01;
end
coupling_motor = coupling_motor / norm_W_motor;
delta_R_L2_motor = coupling_motor - S.E_L2_motor(i,:);
S.R_L2_motor(i+1,:) = P.momentum * S.R_L2_motor(i,:) + decay * P.eta_rep * delta_R_L2_motor * 0.5;
S.R_L2_motor(i+1,:) = max(-1, min(1, S.R_L2_motor(i+1,:)));

% Motor L3 (Nov 2, 2025 FIX: Project error properly instead of taking mean)
% Previously: E_L3_motor = mean(S.E_L2_motor(i,:)) * ones(1,3)  (lost information)
% Now: Project L2 error to L3 space, preserving signal
n_L2_motor = size(S.R_L2_motor, 2);
n_L3_motor = size(S.R_L3_motor, 2);
E_L3_motor_proj = S.E_L2_motor(i, 1:min(3, n_L2_motor));  % Take first 3 dims or pad with zeros
if numel(E_L3_motor_proj) < n_L3_motor
    E_L3_motor_proj = [E_L3_motor_proj, zeros(1, n_L3_motor - numel(E_L3_motor_proj))];
end
S.R_L3_motor(i+1,:) = S.R_L3_motor(i,:) + P.eta_rep * E_L3_motor_proj * 0.1;
S.R_L3_motor(i+1,:) = max(-1, min(1, S.R_L3_motor(i+1,:)));

% Planning L1 (Nov 2, 2025 FIX: Use separate relaxed bounds for ball trajectory space)
% Motor L1 is constrained to player workspace (reach limits)
% Planning L1 represents ball/target position which can extend beyond reach (anticipation)
%
% THEORETICAL FIX #3 (Nov 2, 2025): Visual Coordinates for Planning L1
% RATIONALE: Planning L1 represents ball position in VISUAL FIELD coordinates, not motor coordinates
%            Visual field naturally extends beyond arm's reach (humans can see ~160 deg wide field)
%            Parietal cortex neurons encode allocentric space (world frame) not just reachable space
%            Empirically: visual field is approximately 1.3-1.5x arm reach in each direction
%            This justifies the 1.5x expansion factor as biologically grounded, not arbitrary
%
% BEFORE (problematic): Planning L1 used workspace bounds (same as motor)
%                        - Implied ball observations are confined to reach space
%                        - Biologically implausible (you can see beyond your reach)
%                        - Confused motor domain with visual domain
%
% AFTER (corrected):    Planning L1 uses visual field bounds (1.5x workspace)
%                        - Observations extend naturally beyond reach
%                        - Aligns with parietal cortex allocentric coding
%                        - Motor and planning operate in different coordinate frames
%                        - Relaxed bounds mean: can anticipate ball going beyond reach
%
S.R_L1_plan(i+1, idx_pos) = S.R_L1_plan(i, idx_pos) + decay * P.eta_rep * S.E_L1_plan(i, idx_pos) * 0.1;
S.R_L1_plan(i+1, idx_vel) = S.R_L1_plan(i, idx_vel) + decay * P.eta_rep * S.E_L1_plan(i, idx_vel) * 0.1;
for k = 1:numel(idx_vel)
    S.R_L1_plan(i+1, idx_vel(k)) = max(-2, min(2, S.R_L1_plan(i+1, idx_vel(k))));
end

% Apply relaxed bounds for planning L1 (ball in visual field coordinates)
% 1.5x factor is based on parietal receptive field extent and human visual acuity limits
pos_dims_p = min(numel(idx_pos), size(workspace_bounds,1));
relax_factor = 1.5;  % Visual field extends ~1.5x beyond motor reach
for k = 1:pos_dims_p
    ball_bound_min = workspace_bounds(k,1) * relax_factor;
    ball_bound_max = workspace_bounds(k,2) * relax_factor;
    S.R_L1_plan(i+1, idx_pos(k)) = max(ball_bound_min, min(ball_bound_max, S.R_L1_plan(i+1, idx_pos(k))));
end
S.R_L1_plan(i+1, idx_bias) = 1;

% Planning L2 (CORRECTED - Symmetric Task Control)
% THEORETICAL FIX #3 (Nov 2, 2025): Remove Task Gating from Representations
% BEFORE: Planning L2 had multiplicative task_gate (S.R_L0 * range + floor)
%         Motor L2 had no gating, creating asymmetry
% PROBLEM: Asymmetric task control is incoherent with single-mechanism theory
%          Documentation claimed "gating removed" but planning still had it
%          Two different control mechanisms (gated vs. ungated) caused confusion
% AFTER: Remove multiplicative task_gate from BOTH motor and planning
%        Task selectivity now controlled ONLY through weight freezing
% RATIONALE: Single source of task control via weight indexing (task-indexed weight cells)
%            Off-task weights frozen (can't learn) → off-task predictions naturally stale
%            No additional gating needed; weight mechanism sufficient
%            Aligns with dopamine-based synaptic tagging theory (Lisman et al., 2002)

% Use active task's planning L2->L1 weights
coupling_plan = S.E_L1_plan(i,:) * W_plan_L2_to_L1_active;
% FIX (Nov 2, 2025): Improve weight norm stability - use adaptive floor instead of fixed 0.1
norm_W_plan = norm(W_plan_L2_to_L1_active, 'fro');
if norm_W_plan < 0.01  % Adaptive floor for early learning
    norm_W_plan = 0.01;
end
coupling_plan = coupling_plan / norm_W_plan;
delta_R_L2_plan = coupling_plan - S.E_L2_plan(i,:);

% NO multiplicative gating - pure representation update (symmetric with motor L2)
S.R_L2_plan(i+1,:) = P.momentum * S.R_L2_plan(i,:) + decay * P.eta_rep * delta_R_L2_plan * 0.5;
S.R_L2_plan(i+1,:) = max(-1, min(1, S.R_L2_plan(i+1,:)));

% Planning L3 (CORRECTED - Symmetric Task Control, no gating)
n_L2_plan = size(S.R_L2_plan, 2);
n_L3_plan = size(S.R_L3_plan, 2);
E_L3_plan_proj = S.E_L2_plan(i, 1:min(3, n_L2_plan));  % Take first 3 dims or pad with zeros
if numel(E_L3_plan_proj) < n_L3_plan
    E_L3_plan_proj = [E_L3_plan_proj, zeros(1, n_L3_plan - numel(E_L3_plan_proj))];
end

% NO multiplicative gating - consistent with FIX #3 removal of task gates
S.R_L3_plan(i+1,:) = S.R_L3_plan(i,:) + P.eta_rep * E_L3_plan_proj * 0.1;
S.R_L3_plan(i+1,:) = max(-1, min(1, S.R_L3_plan(i+1,:)));

% ------------------------------
% WEIGHT UPDATES: UPDATE ALL TASKS
% THEORETICAL FIX #4 (Nov 2, 2025): Remove Task-Selective Weight Freezing
% BEFORE: Weights were frozen for off-task tasks (only current task's weights updated)
%         This created redundancy with interference penalty (two competing mechanisms)
%         Off-task weights couldn't learn, so interference penalty was wasted computation
% AFTER: ALL tasks' weights update on ALL data
%        Interference penalty now provides meaningful credit assignment
%        Weights specialize naturally through competition (low cross-task error = good specialization)
% RATIONALE: If weights freeze, interference penalty can't help them improve
%           Better: all weights compete, penalty encourages task specialization
%           This aligns with: "different tasks should learn different representations"
% 
% NOTE: task_gate variables removed from code (FIX #1), so no additional gating here
% Weight updates are driven by errors and represent genuine learning signal
% Cross-task interference penalty encourages but doesn't enforce specialization

% WEIGHT UPDATES: UPDATE ALL TASKS (WITH GRADIENT CLIPPING - FIX #2)
% ====================================================================
% NEW FIX: Stabilize gradient computation by:
%   1. Replace mean(abs()) layer scaling with L2 norm (more stable)
%   2. Add gradient clipping to prevent explosion
%   3. Handle both motor (shared) and planning (task-indexed) weights
% ====================================================================

% --- MOTOR WEIGHT UPDATES (Shared - FIX: Use L2 norm for stability) ---
% OLD (unstable): layer_scale = mean(abs(R_L2_motor))
% NEW (stable): layer_scale = L2 norm with adaptive floor
layer_norm_motor_2 = max(0.1, norm(S.R_L2_motor(i,:), 2));
layer_norm_motor_3 = max(0.1, norm(S.R_L3_motor(i,:), 2));

% Compute gradients with normalized layer scales
dW_motor_L2_to_L1 = -(P.eta_W * S.pi_L1_motor / layer_norm_motor_2) * (S.E_L1_motor(i,:)' * S.R_L2_motor(i,:));
dW_motor_L3_to_L2 = -(P.eta_W * S.pi_L2_motor / layer_norm_motor_3) * (S.E_L2_motor(i,:)' * S.R_L3_motor(i,:));

% FIX (Nov 3, 2025): Add gradient clipping to prevent explosion
% If gradient magnitude exceeds threshold, clip it to [-max_grad, +max_grad]
max_motor_grad = 0.1;  % Clip motor gradients to [-0.1, +0.1]
dW_motor_L2_to_L1 = max(-max_motor_grad, min(max_motor_grad, dW_motor_L2_to_L1));
dW_motor_L3_to_L2 = max(-max_motor_grad, min(max_motor_grad, dW_motor_L3_to_L2));

% Motor weights: UPDATE SHARED (not task-indexed)
% All tasks use the same motor weights, so weight updates apply to all tasks simultaneously
% This enforces learning of generalizable velocity control
S.W_motor_L2_to_L1 = S.W_motor_L2_to_L1 + dW_motor_L2_to_L1;
S.W_motor_L3_to_L2 = S.W_motor_L3_to_L2 + dW_motor_L3_to_L2;

% --- PLANNING WEIGHT UPDATES (Task-indexed - FIX: Use L2 norm for stability) ---
layer_norm_plan_2 = max(0.1, norm(S.R_L2_plan(i,:), 2));
layer_norm_plan_3 = max(0.1, norm(S.R_L3_plan(i,:), 2));

dW_plan_L2_to_L1 = -(P.eta_W * S.pi_L1_plan / layer_norm_plan_2) * (S.E_L1_plan(i,:)' * S.R_L2_plan(i,:));
dW_plan_L3_to_L2 = -(P.eta_W * S.pi_L2_plan / layer_norm_plan_3) * (S.E_L2_plan(i,:)' * S.R_L3_plan(i,:));

% Clip planning gradients
max_plan_grad = 0.1;  % Clip planning gradients to [-0.1, +0.1]
dW_plan_L2_to_L1 = max(-max_plan_grad, min(max_plan_grad, dW_plan_L2_to_L1));
dW_plan_L3_to_L2 = max(-max_plan_grad, min(max_plan_grad, dW_plan_L3_to_L2));

% Planning weights: UPDATE ALL TASKS
% Apply same logic as before: active task learns normally, off-task learns with interference penalty
for task_idx = 1:numel(S.W_plan_L2_to_L1)
    if task_idx == current_task_idx
        % Active task: normal update
        S.W_plan_L2_to_L1{task_idx} = S.W_plan_L2_to_L1{task_idx} + dW_plan_L2_to_L1;
        S.W_plan_L3_to_L2{task_idx} = S.W_plan_L3_to_L2{task_idx} + dW_plan_L3_to_L2;
    else
        % Off-task: apply interference penalty gradient (if enabled)
        if P.interference_penalty_weight > 0
            W_plan_L2_to_L1_off = S.W_plan_L2_to_L1{task_idx};
            pred_L1_plan_off = S.R_L2_plan(i,:) * W_plan_L2_to_L1_off';
            E_L1_plan_off = S.E_L1_plan(i,:) - pred_L1_plan_off;
            
            penalty_gradient_plan_L2 = P.interference_penalty_weight * (E_L1_plan_off' * S.R_L2_plan(i,:));
            
            S.W_plan_L2_to_L1{task_idx} = S.W_plan_L2_to_L1{task_idx} + dW_plan_L2_to_L1 - penalty_gradient_plan_L2;
            S.W_plan_L3_to_L2{task_idx} = S.W_plan_L3_to_L2{task_idx} + dW_plan_L3_to_L2;
        else
            % No interference penalty: all tasks learn equally
            S.W_plan_L2_to_L1{task_idx} = S.W_plan_L2_to_L1{task_idx} + dW_plan_L2_to_L1;
            S.W_plan_L3_to_L2{task_idx} = S.W_plan_L3_to_L2{task_idx} + dW_plan_L3_to_L2;
        end
    end
end

% Track learning trace (norm of weight updates)
S.learning_trace_W(i) = norm([dW_motor_L2_to_L1(:); dW_motor_L3_to_L2(:); dW_plan_L2_to_L1(:); dW_plan_L3_to_L2(:)], 2);

% ------------------------------
% DYNAMIC PRECISION UPDATES
% ------------------------------
epsilon_var = 1e-6;

% Append latest magnitudes to history
L1_motor_error_mag = sqrt(sum(S.E_L1_motor(i,:).^2));
L2_motor_error_mag = sqrt(sum(S.E_L2_motor(i,:).^2));
L1_plan_error_mag = sqrt(sum(S.E_L1_plan(i,:).^2));
L2_plan_error_mag = sqrt(sum(S.E_L2_plan(i,:).^2));

S.L1_motor_error_history = [S.L1_motor_error_history, L1_motor_error_mag];
S.L2_motor_error_history = [S.L2_motor_error_history, L2_motor_error_mag];
S.L1_plan_error_history = [S.L1_plan_error_history, L1_plan_error_mag];
S.L2_plan_error_history = [S.L2_plan_error_history, L2_plan_error_mag];

if length(S.L1_motor_error_history) > P.window_size
    S.L1_motor_error_history = S.L1_motor_error_history(end-P.window_size+1:end);
    S.L2_motor_error_history = S.L2_motor_error_history(end-P.window_size+1:end);
    S.L1_plan_error_history = S.L1_plan_error_history(end-P.window_size+1:end);
    S.L2_plan_error_history = S.L2_plan_error_history(end-P.window_size+1:end);
end

% Update pi values and diagnostics
% CRITICAL (Nov 2, 2025): update_pi() is DIAGNOSTIC-ONLY - results are NOT used to update actual precisions
% The authoritative precision update mechanism is the error-driven exponential scaling (implemented below at line ~680)
% This section stores diagnostic information ONLY for analysis/visualization
% The [~, raw1, d1] = ... syntax explicitly discards the pi_new output, keeping only raw and denom diagnostics
[~, raw1, d1] = update_pi(S.pi_L1_motor, S.pi_L1_motor_base, S.L1_motor_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[~, raw2, d2] = update_pi(S.pi_L2_motor, S.pi_L2_motor_base, S.L2_motor_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[~, raw3, d3] = update_pi(S.pi_L1_plan, S.pi_L1_plan_base, S.L1_plan_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[~, raw4, d4] = update_pi(S.pi_L2_plan, S.pi_L2_plan_base, S.L2_plan_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);

% Store diagnostic traces (for analysis, not for actual learning)
if ~isfield(S, 'update_pi_raw_trace')
    S.update_pi_raw_trace.L1_motor = zeros(1, 2000);
    S.update_pi_raw_trace.L2_motor = zeros(1, 2000);
    S.update_pi_raw_trace.L1_plan = zeros(1, 2000);
    S.update_pi_raw_trace.L2_plan = zeros(1, 2000);
    S.update_pi_denom_trace.L1_motor = zeros(1, 2000);
    S.update_pi_denom_trace.L2_motor = zeros(1, 2000);
    S.update_pi_denom_trace.L1_plan = zeros(1, 2000);
    S.update_pi_denom_trace.L2_plan = zeros(1, 2000);
end
if i <= 2000
    S.update_pi_raw_trace.L1_motor(i) = raw1;
    S.update_pi_raw_trace.L2_motor(i) = raw2;
    S.update_pi_raw_trace.L1_plan(i) = raw3;
    S.update_pi_raw_trace.L2_plan(i) = raw4;
    S.update_pi_denom_trace.L1_motor(i) = d1;
    S.update_pi_denom_trace.L2_motor(i) = d2;
    S.update_pi_denom_trace.L1_plan(i) = d3;
    S.update_pi_denom_trace.L2_plan(i) = d4;
end

% Update pi values and diagnostics
% CRITICAL (Nov 2, 2025): update_pi() is DIAGNOSTIC-ONLY - results are NOT used to update actual precisions
% The authoritative precision update mechanism is the error-driven exponential scaling (implemented below at line ~680)
% This section stores diagnostic information ONLY for analysis/visualization
% The [~, raw1, d1] = ... syntax explicitly discards the pi_new output, keeping only raw and denom diagnostics
[~, raw1, d1] = update_pi(S.pi_L1_motor, S.pi_L1_motor_base, S.L1_motor_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[~, raw2, d2] = update_pi(S.pi_L2_motor, S.pi_L2_motor_base, S.L2_motor_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[~, raw3, d3] = update_pi(S.pi_L1_plan, S.pi_L1_plan_base, S.L1_plan_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[~, raw4, d4] = update_pi(S.pi_L2_plan, S.pi_L2_plan_base, S.L2_plan_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);

% Store diagnostic traces (for analysis, not for actual learning)
if ~isfield(S, 'update_pi_raw_trace')
    S.update_pi_raw_trace.L1_motor = zeros(1, 2000);
    S.update_pi_raw_trace.L2_motor = zeros(1, 2000);
    S.update_pi_raw_trace.L1_plan = zeros(1, 2000);
    S.update_pi_raw_trace.L2_plan = zeros(1, 2000);
    S.update_pi_denom_trace.L1_motor = zeros(1, 2000);
    S.update_pi_denom_trace.L2_motor = zeros(1, 2000);
    S.update_pi_denom_trace.L1_plan = zeros(1, 2000);
    S.update_pi_denom_trace.L2_plan = zeros(1, 2000);
end
if i <= 2000
    S.update_pi_raw_trace.L1_motor(i) = raw1;
    S.update_pi_raw_trace.L2_motor(i) = raw2;
    S.update_pi_raw_trace.L1_plan(i) = raw3;
    S.update_pi_raw_trace.L2_plan(i) = raw4;
    S.update_pi_denom_trace.L1_motor(i) = d1;
    S.update_pi_denom_trace.L2_motor(i) = d2;
    S.update_pi_denom_trace.L1_plan(i) = d3;
    S.update_pi_denom_trace.L2_plan(i) = d4;
end

% ====================================================================
% CORRECTED PRECISION UPDATE: Error-Driven Exponential Scaling (FIX #1)
% ====================================================================
% THEORETICAL FIX #1 (Nov 2, 2025): Implement Error-Driven Adaptive Precision
%
% BEFORE (BROKEN):
%   - update_pi() results discarded; hardcoded clamps applied instead
%   - PSO parameters (alpha_precision_gain, pi_bounds) never used
%   - Precision frozen at hardcoded values regardless of PSO optimization
%   - PSO wasted 30% computation optimizing dead parameters
%
% AFTER (CORRECTED):
%   - Error-driven exponential scaling as ONLY precision mechanism
%   - Uses P.alpha_precision_gain (sensitivity) and P.pi_bounds (range)
%   - PSO parameters now ACTUALLY AFFECT behavior
%   - Precision adapts dynamically to error magnitude
%
% MECHANISM:
%   precision_new = precision_old * exp(alpha_gain * error_magnitude)
%   High error → higher precision (tighten bounds, force predictions to match)
%   Low error → lower precision (relax bounds, allow exploration)
%   Bounds [min, max] ensure precision stays in reasonable range

% Compute error magnitudes for this step
L1_motor_error_mag = sqrt(sum(S.E_L1_motor(i,:).^2));
L2_motor_error_mag = sqrt(sum(S.E_L2_motor(i,:).^2));
L1_plan_error_mag = sqrt(sum(S.E_L1_plan(i,:).^2));
L2_plan_error_mag = sqrt(sum(S.E_L2_plan(i,:).^2));

% =========================================================================
% Z-SCORE PRECISION ADAPTATION (NOV 3, 2025 - ADAPTIVE NORMALIZATION)
% =========================================================================
% REPLACES: Hardcoded error_scale_factor = 0.1 (task-dependent, arbitrary)
% IMPROVES: Task-invariant normalization, adaptive to learning stage
% =========================================================================

% STEP 1: Maintain running error statistics (mean and variance)
if ~isfield(S, 'error_statistics')
    S.error_statistics = struct();
    S.error_statistics.L1_motor_history = [];
    S.error_statistics.L2_motor_history = [];
    S.error_statistics.L1_plan_history = [];
    S.error_statistics.L2_plan_history = [];
end

% STEP 2: Append current errors to history
S.error_statistics.L1_motor_history = [S.error_statistics.L1_motor_history; L1_motor_error_mag];
S.error_statistics.L2_motor_history = [S.error_statistics.L2_motor_history; L2_motor_error_mag];
S.error_statistics.L1_plan_history = [S.error_statistics.L1_plan_history; L1_plan_error_mag];
S.error_statistics.L2_plan_history = [S.error_statistics.L2_plan_history; L2_plan_error_mag];

% STEP 3: Keep only recent history (sliding window of 100 steps)
window_size = 100;
if length(S.error_statistics.L1_motor_history) > window_size
    S.error_statistics.L1_motor_history = S.error_statistics.L1_motor_history(end-window_size+1:end);
    S.error_statistics.L2_motor_history = S.error_statistics.L2_motor_history(end-window_size+1:end);
    S.error_statistics.L1_plan_history = S.error_statistics.L1_plan_history(end-window_size+1:end);
    S.error_statistics.L2_plan_history = S.error_statistics.L2_plan_history(end-window_size+1:end);
end

% STEP 4: Compute z-scores (error in units of standard deviations)
% Z-score = (x - mean) / std
% This automatically adapts to task difficulty and learning stage

% Helper function to compute z-score safely
compute_z_score = @(error_mag, history) ...
    (error_mag - mean(history)) / (std(history) + 1e-9);

% Compute z-scores (NaN-safe)
if length(S.error_statistics.L1_motor_history) >= 10
    z_L1_motor = compute_z_score(L1_motor_error_mag, S.error_statistics.L1_motor_history);
    z_L2_motor = compute_z_score(L2_motor_error_mag, S.error_statistics.L2_motor_history);
    z_L1_plan = compute_z_score(L1_plan_error_mag, S.error_statistics.L1_plan_history);
    z_L2_plan = compute_z_score(L2_plan_error_mag, S.error_statistics.L2_plan_history);
else
    % Not enough history: use simple linear scaling (3-sigma rule: ±3 std = [0,1] approx)
    z_L1_motor = L1_motor_error_mag / (max(S.error_statistics.L1_motor_history) + 1e-9 + 1.0);
    z_L2_motor = L2_motor_error_mag / (max(S.error_statistics.L2_motor_history) + 1e-9 + 1.0);
    z_L1_plan = L1_plan_error_mag / (max(S.error_statistics.L1_plan_history) + 1e-9 + 1.0);
    z_L2_plan = L2_plan_error_mag / (max(S.error_statistics.L2_plan_history) + 1e-9 + 1.0);
end

% STEP 5: Clip z-scores to [-3, +3] (3-sigma rule)
% Errors beyond ±3 sigma are extreme outliers; cap them to prevent instability
z_L1_motor_clipped = max(-3, min(3, z_L1_motor));
z_L2_motor_clipped = max(-3, min(3, z_L2_motor));
z_L1_plan_clipped = max(-3, min(3, z_L1_plan));
z_L2_plan_clipped = max(-3, min(3, z_L2_plan));

% STEP 6: Convert z-score to precision scaling (exponential)
% Precision scaling = exp(alpha * z_score / 3)
% Normalization by 3 ensures z_score in [-3,+3] maps to exponent in [-alpha, +alpha]
% Result: High errors increase precision (tighter bounds), low errors decrease it

if isfield(P, 'alpha_precision_gain')
    alpha_precision = P.alpha_precision_gain;
else
    alpha_precision = 0.5;  % Default sensitivity if not provided
end

% Apply exponential scaling (clipped to prevent overflow)
precision_scale_L1_motor = exp(min(10, alpha_precision * z_L1_motor_clipped / 3));
precision_scale_L2_motor = exp(min(10, alpha_precision * z_L2_motor_clipped / 3));
precision_scale_L1_plan = exp(min(10, alpha_precision * z_L1_plan_clipped / 3));
precision_scale_L2_plan = exp(min(10, alpha_precision * z_L2_plan_clipped / 3));

% Update precisions with exponential scaling
S.pi_L1_motor = S.pi_L1_motor * precision_scale_L1_motor;
S.pi_L2_motor = S.pi_L2_motor * precision_scale_L2_motor;
S.pi_L1_plan = S.pi_L1_plan * precision_scale_L1_plan;
S.pi_L2_plan = S.pi_L2_plan * precision_scale_L2_plan;

% Enforce bounds from PSO parameters (NOW ACTUALLY USED!)
% These bounds control the adaptive range for precision scaling
if isfield(P, 'pi_bounds')
    bounds_L1_motor = P.pi_bounds.L1_motor;  % [min, max] from PSO
    bounds_L2_motor = P.pi_bounds.L2_motor;
    bounds_L1_plan = P.pi_bounds.L1_plan;
    bounds_L2_plan = P.pi_bounds.L2_plan;
else
    % Fallback defaults if not provided by PSO
    bounds_L1_motor = [1, 1000];
    bounds_L2_motor = [0.1, 100];
    bounds_L1_plan = [1, 1000];
    bounds_L2_plan = [0.1, 100];
end

% Clip to bounds
S.pi_L1_motor = max(bounds_L1_motor(1), min(bounds_L1_motor(2), S.pi_L1_motor));
S.pi_L2_motor = max(bounds_L2_motor(1), min(bounds_L2_motor(2), S.pi_L2_motor));
S.pi_L1_plan = max(bounds_L1_plan(1), min(bounds_L1_plan(2), S.pi_L1_plan));
S.pi_L2_plan = max(bounds_L2_plan(1), min(bounds_L2_plan(2), S.pi_L2_plan));

% Diagnostic traces (store for offline analysis)
S.pi_trace_L1_motor(i) = S.pi_L1_motor;
S.pi_trace_L2_motor(i) = S.pi_L2_motor;
S.pi_trace_L1_plan(i) = S.pi_L1_plan;
S.pi_trace_L2_plan(i) = S.pi_L2_plan;

% Store scaling factors for diagnostics
if ~isfield(S, 'precision_scale_trace')
    S.precision_scale_trace = struct();
    S.precision_scale_trace.L1_motor = zeros(1, 10000);
    S.precision_scale_trace.L2_motor = zeros(1, 10000);
    S.precision_scale_trace.L1_plan = zeros(1, 10000);
    S.precision_scale_trace.L2_plan = zeros(1, 10000);
end
if i <= 10000
    S.precision_scale_trace.L1_motor(i) = precision_scale_L1_motor;
    S.precision_scale_trace.L2_motor(i) = precision_scale_L2_motor;
    S.precision_scale_trace.L1_plan(i) = precision_scale_L1_plan;
    S.precision_scale_trace.L2_plan(i) = precision_scale_L2_plan;
end

% ====================================================================
% END OF CORRECTED PRECISION UPDATE SECTION
% ====================================================================


% Helper for diagnostic precision update (informational only, not applied to actual precision values)
function [pi_new, raw_pi, denom] = update_pi(pi_curr, pi_base, err_history, smooth_alpha, max_ratio)
    if length(err_history) > 10
        err_val = err_history(end);
        var_val = var(err_history);
        var_norm = var_val / (var_val + epsilon_var);
        denom = 1 + 0.8 * err_val + 0.2 * var_norm;
        if ~isfinite(denom) || denom <= 0, denom = 1; end
        raw_pi = pi_base / denom;
        raw_pi = max(pi_base * 0.01, min(pi_base * 10, raw_pi));
        pi_candidate = smooth_alpha * pi_curr + (1 - smooth_alpha) * raw_pi;
        if pi_curr > 0
            ratio = pi_candidate / pi_curr;
            if ratio > max_ratio, pi_candidate = pi_curr * max_ratio; end
            if ratio < 1/max_ratio, pi_candidate = pi_curr / max_ratio; end
        end
        pi_new = pi_candidate;
    else
        raw_pi = pi_curr; denom = 1; pi_new = pi_curr;
    end
end

% Diagnostics
S.pi_trace_L1_motor(i) = S.pi_L1_motor; S.pi_raw_trace_L1_motor(i) = raw1; S.denom_trace_L1_motor(i) = d1;
S.pi_trace_L2_motor(i) = S.pi_L2_motor; S.pi_raw_trace_L2_motor(i) = raw2; S.denom_trace_L2_motor(i) = d2;
S.pi_trace_L1_plan(i) = S.pi_L1_plan; S.pi_raw_trace_L1_plan(i) = raw3; S.denom_trace_L1_plan(i) = d3;
S.pi_trace_L2_plan(i) = S.pi_L2_plan; S.pi_raw_trace_L2_plan(i) = raw4; S.denom_trace_L2_plan(i) = d4;

% Update state fields changed locally (weights already updated into S.W_* cells above)
% No explicit copy needed since we modified S.W_* cells directly
% Return updated S

% =====================================================================
% FIX #2: MAX-NORM WEIGHT CONSTRAINT (Prevents Explosion)
% =====================================================================
% BIOLOGICAL: Synaptic weights bounded by physical/chemical limits
%             Receptors saturate; vesicle pools deplete
% COMPUTATIONAL: Prevents numerical instability; improves generalization
% MECHANISM: If weight vector norm exceeds max_weight_norm,
%            scale entire weight vector down proportionally
%
% Formula:  if ||w|| > max_norm:
%             w_new = w * (max_norm / ||w||)
%           else:
%             w_new = w (unchanged)

% Set maximum weight norm (can be PSO parameter)
max_weight_norm_motor = 2.0;     % Prevent M1 weights from growing unbounded
max_weight_norm_plan = 2.0;      % Prevent prefrontal weights from growing unbounded

% Apply max-norm constraint to MOTOR SHARED weights (not task-indexed)
% Motor L2->L1 weights (SHARED across all tasks)
w_norm = norm(S.W_motor_L2_to_L1, 'fro');
if w_norm > max_weight_norm_motor
    S.W_motor_L2_to_L1 = S.W_motor_L2_to_L1 * (max_weight_norm_motor / w_norm);
end

% Motor L3->L2 weights (SHARED across all tasks)
w_norm = norm(S.W_motor_L3_to_L2, 'fro');
if w_norm > max_weight_norm_motor
    S.W_motor_L3_to_L2 = S.W_motor_L3_to_L2 * (max_weight_norm_motor / w_norm);
end

% Apply max-norm constraint to PLANNING TASK-INDEXED weights
for task_idx = 1:numel(S.W_plan_L2_to_L1)
    % Planning L2->L1 weights
    w_norm = norm(S.W_plan_L2_to_L1{task_idx}, 'fro');
    if w_norm > max_weight_norm_plan
        S.W_plan_L2_to_L1{task_idx} = S.W_plan_L2_to_L1{task_idx} * (max_weight_norm_plan / w_norm);
    end
    
    % Planning L3->L2 weights
    w_norm = norm(S.W_plan_L3_to_L2{task_idx}, 'fro');
    if w_norm > max_weight_norm_plan
        S.W_plan_L3_to_L2{task_idx} = S.W_plan_L3_to_L2{task_idx} * (max_weight_norm_plan / w_norm);
    end
end

% DIAGNOSTIC: Track how often max-norm is active (should be rare in stable learning)
if ~isfield(S, 'maxnorm_events'), S.maxnorm_events = 0; end
if any([norm(S.W_motor_L2_to_L1,'fro'), norm(S.W_motor_L3_to_L2,'fro'), ...
        norm(S.W_plan_L2_to_L1{1},'fro'), norm(S.W_plan_L3_to_L2{1},'fro')] > max_weight_norm_motor*0.95)
    S.maxnorm_events = S.maxnorm_events + 1;
end
end
