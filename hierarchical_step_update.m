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

% For convenience, operate directly on S fields (avoid unused local aliases)

% ------------------------------
% BALL PHYSICS (unchanged)
% ------------------------------
time_in_trial = i - S.phases_indices{S.current_trial}(1);
acc_x = S.ball_trajectories{S.current_trial}.acceleration(1) * sin(time_in_trial * 0.001);
acc_y = S.ball_trajectories{S.current_trial}.acceleration(2) * sin(time_in_trial * 0.001 + 1);
acc_z = S.ball_trajectories{S.current_trial}.acceleration(3) * sin(time_in_trial * 0.001 + 2);

ax = acc_x; ay = acc_y; az = acc_z - P.gravity;

S.vx_ball(i+1) = S.vx_ball(i) + ax * dt;
S.vy_ball(i+1) = S.vy_ball(i) + ay * dt;
S.vz_ball(i+1) = S.vz_ball(i) + az * dt;

S.vx_ball(i+1) = S.vx_ball(i+1) * (1 - P.air_drag);
S.vy_ball(i+1) = S.vy_ball(i+1) * (1 - P.air_drag);
S.vz_ball(i+1) = S.vz_ball(i+1) * (1 - P.air_drag);

S.x_ball(i+1) = S.x_ball(i) + dt * S.vx_ball(i+1);
S.y_ball(i+1) = S.y_ball(i) + dt * S.vy_ball(i+1);
S.z_ball(i+1) = S.z_ball(i) + dt * S.vz_ball(i+1);

% Allow an explicit ground plane override (P.ground_z). Fall back to workspace lower bound.
if isfield(P, 'ground_z')
    ground_z = P.ground_z;
else
    ground_z = P.workspace_bounds(3,1);
end
if S.z_ball(i+1) <= ground_z
    S.z_ball(i+1) = ground_z;
    if S.vz_ball(i+1) < 0
        S.vz_ball(i+1) = -P.restitution * S.vz_ball(i+1);
    end
    S.vx_ball(i+1) = S.vx_ball(i+1) * P.ground_friction;
    S.vy_ball(i+1) = S.vy_ball(i+1) * P.ground_friction;
    if abs(S.vz_ball(i+1)) < 1e-3
        S.vz_ball(i+1) = 0;
    end
end

S.x_ball(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), S.x_ball(i+1)));
S.y_ball(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), S.y_ball(i+1)));
S.z_ball(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), S.z_ball(i+1)));

% --- check ball pre-clamp bounds (debug / safety) ---
% record the raw computed position (before clamp) if you want to inspect later
raw_x = S.x_ball(i+1);
raw_y = S.y_ball(i+1);
raw_z = S.z_ball(i+1);

% initialize log fields on first use (cheap)
if ~isfield(S, 'ball_out_of_bounds_log')
    S.ball_out_of_bounds_log = zeros(0,5); % columns: step, raw_x, raw_y, raw_z, reasonCode
    S.ball_out_of_bounds = false;
end

% small tolerance to avoid floating point jitter reporting
tol = 1e-9;
x_min = workspace_bounds(1,1) - tol; x_max = workspace_bounds(1,2) + tol;
y_min = workspace_bounds(2,1) - tol; y_max = workspace_bounds(2,2) + tol;
z_min = workspace_bounds(3,1) - tol; z_max = workspace_bounds(3,2) + tol;

out = (raw_x < x_min) || (raw_x > x_max) || (raw_y < y_min) || (raw_y > y_max) || (raw_z < z_min) || (raw_z > z_max);

if out
    S.ball_out_of_bounds = true;
    % reasonCode: 1=x,2=y,3=z, sum if multiple
    reasonCode = 0;
    if raw_x < x_min || raw_x > x_max, reasonCode = reasonCode + 1; end
    if raw_y < y_min || raw_y > y_max, reasonCode = reasonCode + 2; end
    if raw_z < z_min || raw_z > z_max, reasonCode = reasonCode + 4; end
    S.ball_out_of_bounds_log(end+1, :) = [i+1, raw_x, raw_y, raw_z, reasonCode];
    % optional: clamp to bounds (if you keep your existing clamp, this is redundant)
    S.x_ball(i+1) = min(max(raw_x, workspace_bounds(1,1)), workspace_bounds(1,2));
    S.y_ball(i+1) = min(max(raw_y, workspace_bounds(2,1)), workspace_bounds(2,2));
    S.z_ball(i+1) = min(max(raw_z, workspace_bounds(3,1)), workspace_bounds(3,2));
end

% ------------------------------
% PREDICTION (Motor & Planning) WITH MULTIPLICATIVE TASK GATING
% NEW: L0 task context now multiplicatively gates predictions
% This implements prefrontal modulation of motor/planning circuits
% ------------------------------

% Identify current active task from L0 (one-hot encoding)
[~, current_task_idx] = max(S.R_L0(i,:));
if current_task_idx < 1 || current_task_idx > length(S.W_motor_L2_to_L1)
    current_task_idx = 1;  % safety fallback
end

% Get task-specific weight matrices for current task
W_motor_L2_to_L1_active = S.W_motor_L2_to_L1{current_task_idx};
W_motor_L3_to_L2_active = S.W_motor_L3_to_L2{current_task_idx};
W_motor_L1_lat_active = S.W_motor_L1_lat{current_task_idx};
W_motor_L2_lat_active = S.W_motor_L2_lat{current_task_idx};
W_motor_L3_lat_active = S.W_motor_L3_lat{current_task_idx};

W_plan_L2_to_L1_active = S.W_plan_L2_to_L1{current_task_idx};
W_plan_L3_to_L2_active = S.W_plan_L3_to_L2{current_task_idx};
W_plan_L1_lat_active = S.W_plan_L1_lat{current_task_idx};
W_plan_L2_lat_active = S.W_plan_L2_lat{current_task_idx};
W_plan_L3_lat_active = S.W_plan_L3_lat{current_task_idx};

% Task context gating strength (multiplicative)
% task_gate ranges [0,1]: 0 = task completely inhibited, 1 = fully active
task_gate_motor = S.R_L0(i, current_task_idx);  % for motor: usually 1.0 (stable forward models learn regardless)
task_gate_plan = S.R_L0(i, current_task_idx) * 0.8 + 0.2;  % for planning: [0.2, 1.0] (some planning remains off-task)

% Motor region predictions with task gating
S.pred_L2_motor(i,:) = task_gate_motor * (S.R_L3_motor(i,:) * W_motor_L3_to_L2_active' + S.R_L2_motor(i,:) * W_motor_L2_lat_active');

S.pred_L1_motor(i,:) = task_gate_motor * (S.R_L2_motor(i,:) * W_motor_L2_to_L1_active' + S.R_L1_motor(i,:) * W_motor_L1_lat_active');

% extract velocity predictions using semantic indices (pad/truncate to 3 elements if needed)
tmp_vel = S.pred_L1_motor(i, idx_vel);
pred_vel_motor = zeros(1,3);
n_tmp = numel(tmp_vel);
pred_vel_motor(1:min(3,n_tmp)) = tmp_vel(1:min(3,n_tmp));
S.motor_vx_motor(i) = P.motor_gain * pred_vel_motor(1);
S.motor_vy_motor(i) = P.motor_gain * pred_vel_motor(2);
S.motor_vz_motor(i) = P.motor_gain * pred_vel_motor(3);

% Planning region predictions with task gating
S.pred_L2_plan(i,:) = task_gate_plan * (S.R_L3_plan(i,:) * W_plan_L3_to_L2_active' + S.R_L2_plan(i,:) * W_plan_L2_lat_active');

S.pred_L1_plan(i,:) = task_gate_plan * (S.R_L2_plan(i,:) * W_plan_L2_to_L1_active' + S.R_L1_plan(i,:) * W_plan_L1_lat_active');

tmp_vel_p = S.pred_L1_plan(i, idx_vel);
pred_vel_plan = zeros(1,3);
n_tmp_p = numel(tmp_vel_p);
pred_vel_plan(1:min(3,n_tmp_p)) = tmp_vel_p(1:min(3,n_tmp_p));
S.motor_vx_plan(i) = P.motor_gain * pred_vel_plan(1);
S.motor_vy_plan(i) = P.motor_gain * pred_vel_plan(2);
S.motor_vz_plan(i) = P.motor_gain * pred_vel_plan(3);

% ------------------------------
% COMBINED OUTPUT & KINEMATICS
% ------------------------------
combined_vx = 0.5 * S.motor_vx_motor(i) + 0.5 * S.motor_vx_plan(i);
combined_vy = 0.5 * S.motor_vy_motor(i) + 0.5 * S.motor_vy_plan(i);
combined_vz = 0.5 * S.motor_vz_motor(i) + 0.5 * S.motor_vz_plan(i);

S.vx_player(i+1) = P.damping * S.vx_player(i) + combined_vx;
S.vy_player(i+1) = P.damping * S.vy_player(i) + combined_vy;
S.vz_player(i+1) = P.damping * S.vz_player(i) + combined_vz;

S.x_player(i+1) = S.x_player(i) + dt * S.vx_player(i+1);
S.y_player(i+1) = S.y_player(i) + dt * S.vy_player(i+1);
S.z_player(i+1) = S.z_player(i) + dt * S.vz_player(i+1);

S.x_player(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), S.x_player(i+1)));
S.y_player(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), S.y_player(i+1)));
S.z_player(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), S.z_player(i+1)));

% ------------------------------
% ERROR COMPUTATION (WITH TASK-CONDITIONAL ERRORS)
% NEW: Compute error signals for ALL tasks, but only update current task
% This monitors task interference and enables interference penalties
% ------------------------------

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
% Loop over all tasks and compute what error WOULD be if that task were active
for task_candidate = 1:numel(S.W_motor_L2_to_L1)
    % Predictions if task_candidate were active
    W_motor_L2_to_L1_cand = S.W_motor_L2_to_L1{task_candidate};
    pred_L1_motor_cand = S.R_L2_motor(i,:) * W_motor_L2_to_L1_cand';
    
    % L1 error for this candidate task
    E_L1_motor_cand = [pos_vec - pred_L1_motor_cand(1:3)];
    S.task_errors_motor(i, task_candidate) = norm(E_L1_motor_cand);
    
    % Planning error candidate
    W_plan_L2_to_L1_cand = S.W_plan_L2_to_L1{task_candidate};
    pred_L1_plan_cand = S.R_L2_plan(i,:) * W_plan_L2_to_L1_cand';
    E_L1_plan_cand = [pos_ball - pred_L1_plan_cand(1:3)];
    S.task_errors_plan(i, task_candidate) = norm(E_L1_plan_cand);
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
max_finite_value = 1e8;
min_precision = 0.01;
max_precision = 1e6;

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
    for task_idx = 1:numel(S.W_motor_L2_to_L1)
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
if isempty(nan_reported), nan_reported = false; end
if (~isfinite(S.free_energy_all(i)) || any(~isfinite([S.E_L1_motor(i,:), S.E_L2_motor(i,:), S.E_L1_plan(i,:), S.E_L2_plan(i,:), S.interception_error_all(i)]))) && ~nan_reported
    nan_reported = true;
    
    % INCREMENT CLIPPING COUNTER (Nov 1, 2025)
    S.clipping_count = S.clipping_count + 1;
    
    fprintf(2, 'DEBUG WARNING: NaN/Inf detected at step %d (clipping event #%d). Dumping snapshot to ./figures/nan_snapshot.mat\n', i, S.clipping_count);
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
end

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
norm_W_motor = max(0.1, norm(W_motor_L2_to_L1_active, 'fro'));
coupling_motor = coupling_motor / norm_W_motor;
delta_R_L2_motor = coupling_motor - S.E_L2_motor(i,:);
S.R_L2_motor(i+1,:) = P.momentum * S.R_L2_motor(i,:) + decay * P.eta_rep * delta_R_L2_motor * 0.5;
S.R_L2_motor(i+1,:) = max(-1, min(1, S.R_L2_motor(i+1,:)));

% Motor L3
E_L3_motor = mean(S.E_L2_motor(i,:)) * ones(1,3);
S.R_L3_motor(i+1,1:3) = S.R_L3_motor(i,1:3) + P.eta_rep * E_L3_motor * 0.1;
S.R_L3_motor(i+1,1:3) = max(-1, min(1, S.R_L3_motor(i+1,1:3)));

% Planning L1
S.R_L1_plan(i+1, idx_pos) = S.R_L1_plan(i, idx_pos) + decay * P.eta_rep * S.E_L1_plan(i, idx_pos) * 0.1;
S.R_L1_plan(i+1, idx_vel) = S.R_L1_plan(i, idx_vel) + decay * P.eta_rep * S.E_L1_plan(i, idx_vel) * 0.1;
for k = 1:numel(idx_vel)
    S.R_L1_plan(i+1, idx_vel(k)) = max(-2, min(2, S.R_L1_plan(i+1, idx_vel(k))));
end
pos_dims_p = min(numel(idx_pos), size(workspace_bounds,1));
for k = 1:pos_dims_p
    S.R_L1_plan(i+1, idx_pos(k)) = max(workspace_bounds(k,1), min(workspace_bounds(k,2), S.R_L1_plan(i+1, idx_pos(k))));
end
S.R_L1_plan(i+1, idx_bias) = 1;

% Planning L2 (task gated)
task_gate = S.R_L0(i, S.current_trial) * 0.7 + 0.3;
% Use active task's planning L2->L1 weights
coupling_plan = S.E_L1_plan(i,:) * W_plan_L2_to_L1_active;
norm_W_plan = max(0.1, norm(W_plan_L2_to_L1_active, 'fro'));
coupling_plan = coupling_plan / norm_W_plan;
delta_R_L2_plan = coupling_plan - S.E_L2_plan(i,:);
S.R_L2_plan(i+1,:) = P.momentum * S.R_L2_plan(i,:) + decay * P.eta_rep * delta_R_L2_plan * 0.5 * task_gate;
S.R_L2_plan(i+1,:) = max(-1, min(1, S.R_L2_plan(i+1,:)));

% Planning L3
E_L3_plan = mean(S.E_L2_plan(i,:)) * ones(1,3);
S.R_L3_plan(i+1,1:3) = S.R_L3_plan(i,1:3) + P.eta_rep * E_L3_plan * 0.1 * task_gate;
S.R_L3_plan(i+1,1:3) = max(-1, min(1, S.R_L3_plan(i+1,1:3)));

% ------------------------------
% WEIGHT UPDATES (TASK-SELECTIVE): ONLY UPDATE ACTIVE TASK
% NEW: Weight updates are restricted to the currently active task
% This implements strong credit assignment and prevents interference learning
% Off-task weights remain frozen until that task becomes active
% ------------------------------
layer_scale_motor_1 = max(0.1, mean(abs(S.R_L2_motor(i,:))));
layer_scale_motor_3 = max(0.1, mean(abs(S.R_L3_motor(i,:))));

dW_motor_1 = -(P.eta_W * S.pi_L1_motor / layer_scale_motor_1) * (S.E_L1_motor(i,:)' * S.R_L2_motor(i,:));
S.W_motor_L2_to_L1{current_task_idx} = S.W_motor_L2_to_L1{current_task_idx} + dW_motor_1;

dW_motor_3 = -(P.eta_W * S.pi_L2_motor / layer_scale_motor_3) * (S.E_L2_motor(i,:)' * S.R_L3_motor(i,:));
S.W_motor_L3_to_L2{current_task_idx} = S.W_motor_L3_to_L2{current_task_idx} + dW_motor_3;

% lateral motor (task-selective)
dW_motor_L1_lat = -(P.eta_W * S.pi_L1_motor / max(0.1, mean(abs(S.R_L1_motor(i,:))))) * (S.E_L1_motor(i,:)' * S.R_L1_motor(i,:));
dW_motor_L2_lat = -(P.eta_W * S.pi_L2_motor / max(0.1, mean(abs(S.R_L2_motor(i,:))))) * (S.E_L2_motor(i,:)' * S.R_L2_motor(i,:));

S.W_motor_L1_lat{current_task_idx} = S.W_motor_L1_lat{current_task_idx} + dW_motor_L1_lat;
S.W_motor_L2_lat{current_task_idx} = S.W_motor_L2_lat{current_task_idx} + dW_motor_L2_lat;

S.W_motor_L1_lat{current_task_idx} = S.W_motor_L1_lat{current_task_idx} * 0.9999; S.W_motor_L1_lat{current_task_idx}(1:size(S.W_motor_L1_lat{current_task_idx},1)+1:end) = 0;
S.W_motor_L2_lat{current_task_idx} = S.W_motor_L2_lat{current_task_idx} * 0.9999; S.W_motor_L2_lat{current_task_idx}(1:size(S.W_motor_L2_lat{current_task_idx},1)+1:end) = 0;

% Planning weight updates (task gated AND task-selective)
layer_scale_plan_1 = max(0.1, mean(abs(S.R_L2_plan(i,:))));
layer_scale_plan_3 = max(0.1, mean(abs(S.R_L3_plan(i,:))));

dW_plan_1 = -(P.eta_W * S.pi_L1_plan / layer_scale_plan_1) * (S.E_L1_plan(i,:)' * S.R_L2_plan(i,:)) * task_gate_plan;
S.W_plan_L2_to_L1{current_task_idx} = S.W_plan_L2_to_L1{current_task_idx} + dW_plan_1;

dW_plan_3 = -(P.eta_W * S.pi_L2_plan / layer_scale_plan_3) * (S.E_L2_plan(i,:)' * S.R_L3_plan(i,:)) * task_gate_plan;
S.W_plan_L3_to_L2{current_task_idx} = S.W_plan_L3_to_L2{current_task_idx} + dW_plan_3;

dW_plan_L1_lat = -(P.eta_W * S.pi_L1_plan / max(0.1, mean(abs(S.R_L1_plan(i,:))))) * (S.E_L1_plan(i,:)' * S.R_L1_plan(i,:)) * task_gate_plan;
dW_plan_L2_lat = -(P.eta_W * S.pi_L2_plan / max(0.1, mean(abs(S.R_L2_plan(i,:))))) * (S.E_L2_plan(i,:)' * S.R_L2_plan(i,:)) * task_gate_plan;

S.W_plan_L1_lat{current_task_idx} = S.W_plan_L1_lat{current_task_idx} + dW_plan_L1_lat;
S.W_plan_L2_lat{current_task_idx} = S.W_plan_L2_lat{current_task_idx} + dW_plan_L2_lat;

S.W_plan_L1_lat{current_task_idx} = S.W_plan_L1_lat{current_task_idx} * 0.9999; S.W_plan_L1_lat{current_task_idx}(1:size(S.W_plan_L1_lat{current_task_idx},1)+1:end) = 0;
S.W_plan_L2_lat{current_task_idx} = S.W_plan_L2_lat{current_task_idx} * 0.9999; S.W_plan_L2_lat{current_task_idx}(1:size(S.W_plan_L2_lat{current_task_idx},1)+1:end) = 0;

S.learning_trace_W(i) = norm(dW_motor_1, 'fro') + norm(dW_motor_3, 'fro') + norm(dW_plan_1, 'fro') + norm(dW_plan_3, 'fro') + ...
    norm(dW_motor_L1_lat,'fro') + norm(dW_motor_L2_lat,'fro') + norm(dW_plan_L1_lat,'fro') + norm(dW_plan_L2_lat,'fro');

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

% Helper for generic precision update
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

% Update pi values and diagnostics
[S.pi_L1_motor, raw1, d1] = update_pi(S.pi_L1_motor, S.pi_L1_motor_base, S.L1_motor_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[S.pi_L2_motor, raw2, d2] = update_pi(S.pi_L2_motor, S.pi_L2_motor_base, S.L2_motor_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[S.pi_L1_plan, raw3, d3] = update_pi(S.pi_L1_plan, S.pi_L1_plan_base, S.L1_plan_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);
[S.pi_L2_plan, raw4, d4] = update_pi(S.pi_L2_plan, S.pi_L2_plan_base, S.L2_plan_error_history, P.pi_smooth_alpha, P.pi_max_step_ratio);

% Apply sensible clamps
S.pi_L1_motor = max(1, min(1000, S.pi_L1_motor));
S.pi_L2_motor = max(0.1, min(100, S.pi_L2_motor));
S.pi_L1_plan = max(1, min(1000, S.pi_L1_plan));
S.pi_L2_plan = max(0.1, min(100, S.pi_L2_plan));

% Diagnostics
S.pi_trace_L1_motor(i) = S.pi_L1_motor; S.pi_raw_trace_L1_motor(i) = raw1; S.denom_trace_L1_motor(i) = d1;
S.pi_trace_L2_motor(i) = S.pi_L2_motor; S.pi_raw_trace_L2_motor(i) = raw2; S.denom_trace_L2_motor(i) = d2;
S.pi_trace_L1_plan(i) = S.pi_L1_plan; S.pi_raw_trace_L1_plan(i) = raw3; S.denom_trace_L1_plan(i) = d3;
S.pi_trace_L2_plan(i) = S.pi_L2_plan; S.pi_raw_trace_L2_plan(i) = raw4; S.denom_trace_L2_plan(i) = d4;

% Update state fields changed locally (weights already updated into S.W_* cells above)
% No explicit copy needed since we modified S.W_* cells directly
% Return updated S
end
