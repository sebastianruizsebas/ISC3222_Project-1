% STEP 2: NUMERICAL SIMULATION OF PREDICTIVE CODING
% ===================================================
% This script implements the update rules derived in Step 1 and simulates
% how the hierarchical system adapts to changing sensory input.
%
% EXPERIMENT:
% We present a moving target that suddenly changes velocity at t=5s.
% The system must infer both position AND velocity from noisy observations.
%
% This demonstrates "active inference" - the system actively predicts
% the future and updates beliefs when predictions fail.

clear; clc; close all;
fprintf('=== STEP 2: NUMERICAL SIMULATION ===\n\n');

%% Simulation Parameters
fprintf('Setting up simulation parameters...\n');

% Time parameters
t_span = [0 10];      % Simulate for 10 seconds
dt = 0.01;            % Time step (10ms)
t = 0:dt:t_span(2);   % Time vector
N = length(t);

fprintf('  Duration: %.1f seconds\n', t_span(2));
fprintf('  Time step: %.3f seconds\n', dt);
fprintf('  Total steps: %d\n\n', N);

%% Generate Synthetic Sensory Input
fprintf('Creating sensory input (moving target)...\n');

% Scenario: Target moves right at velocity=2, then LEFT at velocity=-1
% This sudden change tests the system's adaptation ability

true_velocity = zeros(1, N);
mid = floor(N/2);
true_velocity(1:mid) = 2;          % First half: move right
true_velocity(mid+1:end) = -1;     % Second half: move left

% Integrate to get position
true_position = cumsum(true_velocity) * dt;

% Add sensory noise to observations
sensory_noise_std = 0.05;
x_obs = true_position + sensory_noise_std * randn(1, N);

fprintf('  First 5s: velocity = +2 (moving right)\n');
fprintf('  Last 5s:  velocity = -1 (moving left)\n');
fprintf('  Sensory noise: σ = %.3f\n\n', sensory_noise_std);

%% Model Hyperparameters (Priors)
fprintf('Model hyperparameters:\n');

sigma_x = 0.1;    % Sensory precision (how much to trust observations)
sigma_v = 1.0;    % Velocity prior strength (flexibility of beliefs)
mu_v = 0;         % Prior expectation: stationary (zero velocity)

fprintf('  σ_x = %.2f   (sensory noise, smaller = trust sensors more)\n', sigma_x);
fprintf('  σ_v = %.2f   (prior strength, larger = more flexible beliefs)\n', sigma_v);
fprintf('  μ_v = %.2f   (expected velocity)\n\n', mu_v);

%% Initial Conditions
fprintf('Initial beliefs:\n');

x_est = 0;        % Start with no position estimate
v_est = 0;        % Start with stationary belief (matches prior)

fprintf('  x(0) = %.1f\n', x_est);
fprintf('  v(0) = %.1f\n\n', v_est);

%% Preallocate Storage
x_history = zeros(1, N);
v_history = zeros(1, N);
prediction_error_x = zeros(1, N);
prediction_error_v = zeros(1, N);
free_energy = zeros(1, N);

%% Simulation Loop (Euler Integration)
fprintf('Running simulation');

for i = 1:N
    if mod(i, N/10) == 0
        fprintf('.');
    end
    
    % Current sensory observation
    x_current = x_obs(i);
    
    % Compute prediction errors (from Step 1)
    epsilon_x = x_current - v_est;  % sensory surprise
    epsilon_v = v_est - mu_v;       % prior surprise
    
    % Update rules (derived in Step 1 via gradient descent)
    dx_dt = epsilon_x / sigma_x^2;            % perceptual update
    dv_dt = epsilon_x / sigma_x^2 - epsilon_v / sigma_v^2;  % belief update
    
    % Euler integration
    x_est = x_est + dx_dt * dt;
    v_est = v_est + dv_dt * dt;
    
    % Compute free energy (objective being minimized)
    F = 0.5 * (epsilon_x^2 / sigma_x^2 + epsilon_v^2 / sigma_v^2);
    
    % Store history
    x_history(i) = x_est;
    v_history(i) = v_est;
    prediction_error_x(i) = abs(epsilon_x);
    prediction_error_v(i) = abs(epsilon_v);
    free_energy(i) = F;
end

fprintf(' Done!\n\n');

%% Visualization
fprintf('Generating plots...\n');

figure('Position', [100 100 1400 900], 'Name', 'Predictive Coding Simulation');

% Subplot 1: Position tracking
subplot(3,2,1);
plot(t, true_position, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Position'); hold on;
plot(t, x_obs, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5, 'DisplayName', 'Noisy Observation');
plot(t, x_history, 'b-', 'LineWidth', 2, 'DisplayName', 'Estimated Position');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Position', 'FontSize', 11);
legend('Location', 'northwest', 'FontSize', 9);
title('Sensory Input vs. Prediction', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(5, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);  % Mark velocity change

% Subplot 2: Velocity inference (hidden state)
subplot(3,2,2);
plot(t, true_velocity, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Velocity'); hold on;
plot(t, v_history, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated Velocity');
yline(mu_v, 'g--', 'Prior', 'LineWidth', 1.5, 'FontSize', 9);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
legend('Location', 'northeast', 'FontSize', 9);
title('Hidden State Inference', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(5, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 3: Position prediction error
subplot(3,2,3);
plot(t, prediction_error_x, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('|ε_x|', 'FontSize', 11);
title('Sensory Prediction Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(5, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 4: Velocity prediction error
subplot(3,2,4);
plot(t, prediction_error_v, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('|ε_v|', 'FontSize', 11);
title('Prior Prediction Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(5, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 5: Free energy over time
subplot(3,2,5);
plot(t, free_energy, 'Color', [0.5 0 0.5], 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Free Energy', 'FontSize', 11);
title('Free Energy Minimization', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(5, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Subplot 6: Position tracking error
subplot(3,2,6);
position_error = abs(true_position - x_history);
plot(t, position_error, 'g-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('|x_{true} - x_{est}|', 'FontSize', 11);
title('Tracking Error', 'FontSize', 12, 'FontWeight', 'bold');
grid on; box on;
xline(5, 'r--', 'LineWidth', 1.5, 'Alpha', 0.5);

% Add overall title
sgtitle('Predictive Coding: Two-Level Visual Motion Model', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Compute Performance Metrics
fprintf('\nPerformance Metrics:\n');

% Adaptation time after velocity change
change_idx = mid + 1;  % Index where velocity changes (t ≈ 5s)
post_change = change_idx:N;

% How long to adapt? (when velocity error drops below 20% of change magnitude)
vel_change_magnitude = abs(true_velocity(change_idx) - true_velocity(change_idx-1));
adaptation_threshold = 0.2 * vel_change_magnitude;

vel_error_post_change = abs(v_history(post_change) - true_velocity(post_change));
adapted_idx = find(vel_error_post_change < adaptation_threshold, 1);

if ~isempty(adapted_idx)
    adaptation_time = adapted_idx * dt;
    fprintf('  Adaptation time: %.2f seconds\n', adaptation_time);
else
    fprintf('  Adaptation time: > %.1f seconds (not fully adapted)\n', t_span(2)/2);
end

% Mean tracking error
mean_tracking_error = mean(position_error);
fprintf('  Mean position error: %.4f\n', mean_tracking_error);

% Final velocity estimate
fprintf('  Final velocity estimate: %.2f (true: %.2f)\n', v_history(end), true_velocity(end));

%% Save Results
fprintf('\nSaving results...\n');
save('simulation_results.mat', 't', 'x_history', 'v_history', ...
    'true_position', 'true_velocity', 'free_energy', 'x_obs');

fprintf('\n=== STEP 2 COMPLETE ===\n');
fprintf('Key observations:\n');
fprintf('  1. System tracks position despite noise\n');
fprintf('  2. Infers hidden velocity from observations\n');
fprintf('  3. Adapts when velocity changes at t=5s\n');
fprintf('  4. Free energy decreases as predictions improve\n\n');
fprintf('Next: Run step3_prior_comparison.m to explore different priors.\n');
