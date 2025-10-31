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

ground_z = P.workspace_bounds(3,1);
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

% ------------------------------
% PREDICTION (Motor & Planning)
% ------------------------------

% Motor region predictions
S.pred_L2_motor(i,:) = S.R_L3_motor(i,:) * S.W_motor_L3_to_L2';
S.pred_L2_motor(i,:) = S.pred_L2_motor(i,:) + S.R_L2_motor(i,:) * S.W_motor_L2_lat';

S.pred_L1_motor(i,:) = S.R_L2_motor(i,:) * S.W_motor_L2_to_L1';
S.pred_L1_motor(i,:) = S.pred_L1_motor(i,:) + S.R_L1_motor(i,:) * S.W_motor_L1_lat';

pred_vx_motor = S.pred_L1_motor(i,4); pred_vy_motor = S.pred_L1_motor(i,5); pred_vz_motor = S.pred_L1_motor(i,6);
S.motor_vx_motor(i) = P.motor_gain * pred_vx_motor;
S.motor_vy_motor(i) = P.motor_gain * pred_vy_motor;
S.motor_vz_motor(i) = P.motor_gain * pred_vz_motor;

% Planning region predictions
S.pred_L2_plan(i,:) = S.R_L3_plan(i,:) * S.W_plan_L3_to_L2';
S.pred_L2_plan(i,:) = S.pred_L2_plan(i,:) + S.R_L2_plan(i,:) * S.W_plan_L2_lat';

S.pred_L1_plan(i,:) = S.R_L2_plan(i,:) * S.W_plan_L2_to_L1';
S.pred_L1_plan(i,:) = S.pred_L1_plan(i,:) + S.R_L1_plan(i,:) * S.W_plan_L1_lat';

pred_vx_plan = S.pred_L1_plan(i,4); pred_vy_plan = S.pred_L1_plan(i,5); pred_vz_plan = S.pred_L1_plan(i,6);
S.motor_vx_plan(i) = P.motor_gain * pred_vx_plan;
S.motor_vy_plan(i) = P.motor_gain * pred_vy_plan;
S.motor_vz_plan(i) = P.motor_gain * pred_vz_plan;

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
% ERROR COMPUTATION
% ------------------------------
S.E_L1_motor(i,1:3) = [S.x_player(i+1), S.y_player(i+1), S.z_player(i+1)] - S.pred_L1_motor(i,1:3);
S.E_L1_motor(i,4:6) = [S.vx_player(i+1), S.vy_player(i+1), S.vz_player(i+1)] - S.pred_L1_motor(i,4:6);
S.E_L1_motor(i,7) = 1 - S.pred_L1_motor(i,7);

S.E_L2_motor(i,:) = S.R_L2_motor(i,:) - S.pred_L2_motor(i,:);

S.E_L1_plan(i,1:3) = [S.x_ball(i+1), S.y_ball(i+1), S.z_ball(i+1)] - S.pred_L1_plan(i,1:3);
ball_pos_now = [S.x_ball(i+1), S.y_ball(i+1), S.z_ball(i+1)];
S.E_L1_plan(i,4:6) = ball_pos_now - S.pred_L1_plan(i,4:6);
S.E_L1_plan(i,7) = 1 - S.pred_L1_plan(i,7);

S.E_L2_plan(i,:) = S.R_L2_plan(i,:) - S.pred_L2_plan(i,:);

S.interception_error_all(i) = sqrt((S.x_player(i+1) - S.x_ball(i+1))^2 + (S.y_player(i+1) - S.y_ball(i+1))^2 + (S.z_player(i+1) - S.z_ball(i+1))^2);

% ------------------------------
% FREE ENERGY
% ------------------------------
S.free_energy_all(i) = sum(S.E_L1_motor(i,:).^2) / (2 * S.pi_L1_motor) + sum(S.E_L2_motor(i,:).^2) / (2 * S.pi_L2_motor) + ...
    sum(S.E_L1_plan(i,:).^2) / (2 * S.pi_L1_plan) + sum(S.E_L2_plan(i,:).^2) / (2 * S.pi_L2_plan) + (S.pi_L1_motor/100) * S.interception_error_all(i)^2;

% ------------------------------
% REPRESENTATION UPDATES
% ------------------------------
decay = 1 - P.momentum;

% Motor L1
S.R_L1_motor(i+1,1:3) = S.R_L1_motor(i,1:3) + decay * P.eta_rep * S.E_L1_motor(i,1:3) * 0.1;
S.R_L1_motor(i+1,4:6) = P.momentum * S.R_L1_motor(i,4:6) + decay * P.eta_rep * S.E_L1_motor(i,4:6) * 0.1;
S.R_L1_motor(i+1,4:6) = max(-2, min(2, S.R_L1_motor(i+1,4:6)));
S.R_L1_motor(i+1,1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), S.R_L1_motor(i+1,1)));
S.R_L1_motor(i+1,2) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), S.R_L1_motor(i+1,2)));
S.R_L1_motor(i+1,3) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), S.R_L1_motor(i+1,3)));
S.R_L1_motor(i+1,7) = 1;

% Motor L2
coupling_motor = S.E_L1_motor(i,:) * S.W_motor_L2_to_L1;
norm_W_motor = max(0.1, norm(S.W_motor_L2_to_L1, 'fro'));
coupling_motor = coupling_motor / norm_W_motor;
delta_R_L2_motor = coupling_motor - S.E_L2_motor(i,:);
S.R_L2_motor(i+1,:) = P.momentum * S.R_L2_motor(i,:) + decay * P.eta_rep * delta_R_L2_motor * 0.5;
S.R_L2_motor(i+1,:) = max(-1, min(1, S.R_L2_motor(i+1,:)));

% Motor L3
E_L3_motor = mean(S.E_L2_motor(i,:)) * ones(1,3);
S.R_L3_motor(i+1,1:3) = S.R_L3_motor(i,1:3) + P.eta_rep * E_L3_motor * 0.1;
S.R_L3_motor(i+1,1:3) = max(-1, min(1, S.R_L3_motor(i+1,1:3)));

% Planning L1
S.R_L1_plan(i+1,1:3) = S.R_L1_plan(i,1:3) + decay * P.eta_rep * S.E_L1_plan(i,1:3) * 0.1;
S.R_L1_plan(i+1,4:6) = S.R_L1_plan(i,4:6) + decay * P.eta_rep * S.E_L1_plan(i,4:6) * 0.1;
S.R_L1_plan(i+1,4:6) = max(-2, min(2, S.R_L1_plan(i+1,4:6)));
S.R_L1_plan(i+1,1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), S.R_L1_plan(i+1,1)));
S.R_L1_plan(i+1,2) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), S.R_L1_plan(i+1,2)));
S.R_L1_plan(i+1,3) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), S.R_L1_plan(i+1,3)));
S.R_L1_plan(i+1,7) = 1;

% Planning L2 (task gated)
task_gate = S.R_L0(i, S.current_trial) * 0.7 + 0.3;
coupling_plan = S.E_L1_plan(i,:) * S.W_plan_L2_to_L1;
norm_W_plan = max(0.1, norm(S.W_plan_L2_to_L1, 'fro'));
coupling_plan = coupling_plan / norm_W_plan;
delta_R_L2_plan = coupling_plan - S.E_L2_plan(i,:);
S.R_L2_plan(i+1,:) = P.momentum * S.R_L2_plan(i,:) + decay * P.eta_rep * delta_R_L2_plan * 0.5 * task_gate;
S.R_L2_plan(i+1,:) = max(-1, min(1, S.R_L2_plan(i+1,:)));

% Planning L3
E_L3_plan = mean(S.E_L2_plan(i,:)) * ones(1,3);
S.R_L3_plan(i+1,1:3) = S.R_L3_plan(i,1:3) + P.eta_rep * E_L3_plan * 0.1 * task_gate;
S.R_L3_plan(i+1,1:3) = max(-1, min(1, S.R_L3_plan(i+1,1:3)));

% ------------------------------
% WEIGHT UPDATES (motor & planning) with dynamic precision
% ------------------------------
layer_scale_motor_1 = max(0.1, mean(abs(S.R_L2_motor(i,:))));
layer_scale_motor_3 = max(0.1, mean(abs(S.R_L3_motor(i,:))));

dW_motor_1 = -(P.eta_W * S.pi_L1_motor / layer_scale_motor_1) * (S.E_L1_motor(i,:)' * S.R_L2_motor(i,:));
S.W_motor_L2_to_L1 = S.W_motor_L2_to_L1 + dW_motor_1;

dW_motor_3 = -(P.eta_W * S.pi_L2_motor / layer_scale_motor_3) * (S.E_L2_motor(i,:)' * S.R_L3_motor(i,:));
S.W_motor_L3_to_L2 = S.W_motor_L3_to_L2 + dW_motor_3;

% lateral motor
dW_motor_L1_lat = -(P.eta_W * S.pi_L1_motor / max(0.1, mean(abs(S.R_L1_motor(i,:))))) * (S.E_L1_motor(i,:)' * S.R_L1_motor(i,:));
dW_motor_L2_lat = -(P.eta_W * S.pi_L2_motor / max(0.1, mean(abs(S.R_L2_motor(i,:))))) * (S.E_L2_motor(i,:)' * S.R_L2_motor(i,:));
dW_motor_L3_lat = -(P.eta_W * S.pi_L3_motor / max(0.1, mean(abs(S.R_L3_motor(i,:))))) * (mean(S.E_L2_motor(i,:))' * S.R_L3_motor(i,:));

S.W_motor_L1_lat = S.W_motor_L1_lat + dW_motor_L1_lat;
S.W_motor_L2_lat = S.W_motor_L2_lat + dW_motor_L2_lat;
S.W_motor_L3_lat = S.W_motor_L3_lat + dW_motor_L3_lat;

S.W_motor_L1_lat = S.W_motor_L1_lat * 0.9999; S.W_motor_L1_lat(1:size(S.W_motor_L1_lat,1)+1:end) = 0;
S.W_motor_L2_lat = S.W_motor_L2_lat * 0.9999; S.W_motor_L2_lat(1:size(S.W_motor_L2_lat,1)+1:end) = 0;
S.W_motor_L3_lat = S.W_motor_L3_lat * 0.9999; S.W_motor_L3_lat(1:size(S.W_motor_L3_lat,1)+1:end) = 0;

% Planning weight updates (task gated)
layer_scale_plan_1 = max(0.1, mean(abs(S.R_L2_plan(i,:))));
layer_scale_plan_3 = max(0.1, mean(abs(S.R_L3_plan(i,:))));

dW_plan_1 = -(P.eta_W * S.pi_L1_plan / layer_scale_plan_1) * (S.E_L1_plan(i,:)' * S.R_L2_plan(i,:)) * task_gate;
S.W_plan_L2_to_L1 = S.W_plan_L2_to_L1 + dW_plan_1;

dW_plan_3 = -(P.eta_W * S.pi_L2_plan / layer_scale_plan_3) * (S.E_L2_plan(i,:)' * S.R_L3_plan(i,:)) * task_gate;
S.W_plan_L3_to_L2 = S.W_plan_L3_to_L2 + dW_plan_3;

dW_plan_L1_lat = -(P.eta_W * S.pi_L1_plan / max(0.1, mean(abs(S.R_L1_plan(i,:))))) * (S.E_L1_plan(i,:)' * S.R_L1_plan(i,:)) * task_gate;
dW_plan_L2_lat = -(P.eta_W * S.pi_L2_plan / max(0.1, mean(abs(S.R_L2_plan(i,:))))) * (S.E_L2_plan(i,:)' * S.R_L2_plan(i,:)) * task_gate;
dW_plan_L3_lat = -(P.eta_W * S.pi_L3_plan / max(0.1, mean(abs(S.R_L3_plan(i,:))))) * (mean(S.E_L2_plan(i,:))' * S.R_L3_plan(i,:)) * task_gate;

S.W_plan_L1_lat = S.W_plan_L1_lat + dW_plan_L1_lat;
S.W_plan_L2_lat = S.W_plan_L2_lat + dW_plan_L2_lat;
S.W_plan_L3_lat = S.W_plan_L3_lat + dW_plan_L3_lat;

S.W_plan_L1_lat = S.W_plan_L1_lat * 0.9999; S.W_plan_L1_lat(1:size(S.W_plan_L1_lat,1)+1:end) = 0;
S.W_plan_L2_lat = S.W_plan_L2_lat * 0.9999; S.W_plan_L2_lat(1:size(S.W_plan_L2_lat,1)+1:end) = 0;
S.W_plan_L3_lat = S.W_plan_L3_lat * 0.9999; S.W_plan_L3_lat(1:size(S.W_plan_L3_lat,1)+1:end) = 0;

S.learning_trace_W(i) = norm(dW_motor_1, 'fro') + norm(dW_motor_3, 'fro') + norm(dW_plan_1, 'fro') + norm(dW_plan_3, 'fro') + ...
    norm(dW_motor_L1_lat,'fro') + norm(dW_motor_L2_lat,'fro') + norm(dW_motor_L3_lat,'fro') + norm(dW_plan_L1_lat,'fro') + norm(dW_plan_L2_lat,'fro') + norm(dW_plan_L3_lat,'fro');

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

% Update state fields changed locally (weights already updated into S above)
S.W_motor_L3_to_L2 = S.W_motor_L3_to_L2; S.W_motor_L2_to_L1 = S.W_motor_L2_to_L1;
S.W_plan_L3_to_L2 = S.W_plan_L3_to_L2; S.W_plan_L2_to_L1 = S.W_plan_L2_to_L1;

% Keep current_trial unchanged for now (helper may later manage phase transitions)
% Return updated S
end
