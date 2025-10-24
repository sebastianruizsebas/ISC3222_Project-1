% STEP 4: HIGH-PRECISION ODE45 IMPLEMENTATION
% ============================================
% This script uses MATLAB's ode45 solver (adaptive Runge-Kutta method)
% instead of simple Euler integration. This provides:
%   - Higher accuracy
%   - Adaptive time-stepping
%   - Better for stiff systems
%
% This is the "production" version, similar to your Lab4 SIRV model.

clear; clc; close all;
fprintf('=== STEP 4: ODE45 HIGH-PRECISION SIMULATION ===\n\n');

%% Model Parameters
fprintf('Configuring model parameters...\n');

params = struct();
params.sigma_x = 0.1;     % Sensory precision
params.sigma_v = 1.0;     % Prior strength
params.mu_v = 0;          % Prior mean (stationary)
params.noise_std = 0.05;  % Sensory noise level

fprintf('  σ_x = %.2f (sensory precision)\n', params.sigma_x);
fprintf('  σ_v = %.2f (prior strength)\n', params.sigma_v);
fprintf('  μ_v = %.2f (expected velocity)\n\n', params.mu_v);

%% Define Sensory Input Function
% Create a function handle for time-varying sensory input
fprintf('Defining sensory input function...\n');

% Scenario: velocity changes at t=5s, with added noise
params.change_time = 5.0;
params.v_before = 2.0;
params.v_after = -1.0;

% Function to get true position at any time t
true_position_fn = @(t) (t < params.change_time) .* (params.v_before * t) + ...
                        (t >= params.change_time) .* (params.v_before * params.change_time + ...
                        params.v_after * (t - params.change_time));

% Store for later comparison
params.true_pos_fn = true_position_fn;

fprintf('  Velocity: %.1f → %.1f at t=%.1fs\n\n', ...
    params.v_before, params.v_after, params.change_time);

%% ODE System Definition
fprintf('Setting up ODE system...\n');

% State vector: y = [x; v]
%   x = estimated position
%   v = estimated velocity

% The dynamics function (nested at end of file)
ode_fn = @(t, y) predictive_coding_dynamics(t, y, params);

fprintf('  State variables: [x, v]^T\n');
fprintf('  Dynamics: from symbolic derivation (Step 1)\n\n');

%% Solve with ODE45
fprintf('Solving with ode45 (adaptive Runge-Kutta)...\n');

% Time span
t_span = [0 10];

% Initial conditions
y0 = [0; 0];  % Start at origin with zero velocity belief

% ODE45 options (similar to Lab4.m approach)
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 0.1);

% Solve!
tic;
[t, y] = ode45(ode_fn, t_span, y0, options);
solve_time = toc;

fprintf('  Solution complete in %.4f seconds\n', solve_time);
fprintf('  Time steps taken: %d (adaptive)\n', length(t));
fprintf('  Average dt: %.4f seconds\n\n', mean(diff(t)));

%% Extract Results
x_est = y(:, 1);  % Estimated position
v_est = y(:, 2);  % Estimated velocity

% Compute true values at solution times
true_pos = true_position_fn(t);
true_vel = (t < params.change_time) * params.v_before + ...
           (t >= params.change_time) * params.v_after;

% Compute errors
pos_error = abs(x_est - true_pos);
vel_error = abs(v_est - true_vel);

%% Visualization
fprintf('Generating plots...\n');

figure('Position', [100 100 1400 900], 'Name', 'ODE45 Predictive Coding');

% Subplot 1: Position trajectory
subplot(2,3,1);
plot(t, true_pos, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Position'); hold on;
plot(t, x_est, 'b-', 'LineWidth', 2, 'DisplayName', 'Estimated (ODE45)');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Position', 'FontSize', 11);
title('Position Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
grid on; box on;
xline(params.change_time, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 2: Velocity inference
subplot(2,3,2);
plot(t, true_vel, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Velocity'); hold on;
plot(t, v_est, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated (ODE45)');
yline(params.mu_v, 'g--', 'Prior', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
title('Velocity Inference', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on; box on;
xline(params.change_time, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 3: Phase portrait (x vs v)
subplot(2,3,3);
plot(x_est, v_est, 'b-', 'LineWidth', 2); hold on;
plot(x_est(1), v_est(1), 'go', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Start');
plot(x_est(end), v_est(end), 'rs', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'End');
xlabel('Position Estimate', 'FontSize', 11);
ylabel('Velocity Estimate', 'FontSize', 11);
title('Phase Portrait', 'FontSize', 12, 'FontWeight', 'bold');
legend('FontSize', 10);
grid on; box on;

% Subplot 4: Position error
subplot(2,3,4);
semilogy(t, pos_error, 'b-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Position Error (log scale)', 'FontSize', 11);
title('Tracking Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(params.change_time, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 5: Velocity error
subplot(2,3,5);
plot(t, vel_error, 'r-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity Error', 'FontSize', 11);
title('Inference Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(params.change_time, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 6: Adaptive time steps
subplot(2,3,6);
dt_adaptive = [diff(t); 0];
plot(t, dt_adaptive, 'k-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Time Step (s)', 'FontSize', 11);
title('ODE45 Adaptive Stepping', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(params.change_time, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

sgtitle('High-Precision Simulation with ODE45', 'FontSize', 14, 'FontWeight', 'bold');

%% Performance Metrics
fprintf('\nPerformance Metrics:\n');

% Mean errors
fprintf('  Mean position error: %.6f\n', mean(pos_error));
fprintf('  Mean velocity error: %.6f\n', mean(vel_error));

% Final estimates
fprintf('  Final position: %.4f (true: %.4f)\n', x_est(end), true_pos(end));
fprintf('  Final velocity: %.4f (true: %.4f)\n', v_est(end), true_vel(end));

% Adaptation time
change_idx = find(t >= params.change_time, 1);
post_change = change_idx:length(t);
threshold = 0.2 * abs(params.v_after - params.v_before);
adapted = find(vel_error(post_change) < threshold, 1);

if ~isempty(adapted)
    adapt_time = t(change_idx + adapted - 1) - params.change_time;
    fprintf('  Adaptation time: %.3f seconds\n', adapt_time);
end

%% Compare with Euler Method
fprintf('\nComparing with Euler method...\n');

% Run quick Euler for comparison
dt_euler = 0.01;
t_euler = 0:dt_euler:10;
N = length(t_euler);

x_euler = 0;
v_euler = 0;
x_hist_euler = zeros(1, N);
v_hist_euler = zeros(1, N);

for i = 1:N
    t_curr = t_euler(i);
    x_obs = true_position_fn(t_curr);
    
    epsilon_x = x_obs - v_euler;
    epsilon_v = v_euler - params.mu_v;
    
    dx_dt = epsilon_x / params.sigma_x^2;
    dv_dt = epsilon_x / params.sigma_x^2 - epsilon_v / params.sigma_v^2;
    
    x_euler = x_euler + dx_dt * dt_euler;
    v_euler = v_euler + dv_dt * dt_euler;
    
    x_hist_euler(i) = x_euler;
    v_hist_euler(i) = v_euler;
end

% Interpolate ODE45 solution to Euler times for comparison
x_ode45_interp = interp1(t, x_est, t_euler);
v_ode45_interp = interp1(t, v_est, t_euler);

% Compute difference
diff_x = abs(x_ode45_interp - x_hist_euler);
diff_v = abs(v_ode45_interp - v_hist_euler);

fprintf('  Max position difference: %.6f\n', max(diff_x));
fprintf('  Max velocity difference: %.6f\n', max(diff_v));
fprintf('  RMS position difference: %.6f\n', sqrt(mean(diff_x.^2)));
fprintf('  RMS velocity difference: %.6f\n', sqrt(mean(diff_v.^2)));

%% Save Results
fprintf('\nSaving ODE45 results...\n');
save('ode45_results.mat', 't', 'y', 'x_est', 'v_est', 'params', 'true_pos', 'true_vel');

fprintf('\n=== STEP 4 COMPLETE ===\n');
fprintf('Key advantages of ODE45:\n');
fprintf('  1. Adaptive time-stepping for efficiency\n');
fprintf('  2. Higher accuracy than fixed-step Euler\n');
fprintf('  3. Better handling of rapid changes\n');
fprintf('  4. Production-ready implementation\n\n');
fprintf('Next: Run run_all_experiments.m to execute all steps together.\n');

%% Nested Function: Dynamics
function dydt = predictive_coding_dynamics(t, y, params)
    % Unpack state
    x_est = y(1);  % Estimated position
    v_est = y(2);  % Estimated velocity
    
    % Get current sensory observation
    if t < params.change_time
        x_obs = params.v_before * t;
    else
        x_obs = params.v_before * params.change_time + ...
                params.v_after * (t - params.change_time);
    end
    
    % Add noise (deterministic for reproducibility in ODE)
    % Note: For stochastic ODE, would use different solver
    x_obs = x_obs + params.noise_std * sin(100*t);  % Pseudo-noise
    
    % Compute prediction errors (from Step 1 derivation)
    epsilon_x = x_obs - v_est;          % Sensory prediction error
    epsilon_v = v_est - params.mu_v;    % Prior prediction error
    
    % Update rules (gradient descent on free energy)
    dx_dt = epsilon_x / params.sigma_x^2;
    dv_dt = epsilon_x / params.sigma_x^2 - epsilon_v / params.sigma_v^2;
    
    % Return derivative
    dydt = [dx_dt; dv_dt];
end
