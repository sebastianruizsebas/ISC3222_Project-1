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

%% Create and Run ODE45 Model
fprintf('Creating ODE45 predictive coding model...\n');

% Time parameters
dt = 0.01;            % For uniform output grid
T = 10;               % Simulate for 10 seconds

% Model hyperparameters
sigma_x = 0.1;        % Sensory precision
sigma_v = 1.0;        % Prior strength

fprintf('  Duration: %.1f seconds\n', T);
fprintf('  σ_x = %.2f (sensory precision)\n', sigma_x);
fprintf('  σ_v = %.2f (prior strength)\n\n', sigma_v);

% Create model instance
model = ODE45Model(dt, T, sigma_x, sigma_v);

% Generate sensory input
sensory_noise_std = 0.05;
model.generateSensoryInput(sensory_noise_std);

fprintf('  Velocity: %.1f → %.1f at t=%.1fs\n', ...
    model.params.v_before, model.params.v_after, model.params.change_time);
fprintf('  Sensory noise: σ = %.3f\n\n', sensory_noise_std);

% Run simulation
model.run();

% Extract results for compatibility
t = model.t';
x_est = model.x_history';
v_est = model.v_history';
true_pos = model.true_position';
true_vel = model.true_velocity';
pos_error = abs(x_est - true_pos);
vel_error = abs(v_est - true_vel);
params = model.params;

%% Visualization
fprintf('Generating plots...\n');
model.visualize();

%% Performance Metrics
model.printSummary();

%% Compare with Euler Method
fprintf('\nComparing with Euler method...\n');

% Create Euler model for comparison
model_euler = EulerModel(dt, T, sigma_x, sigma_v);
model_euler.generateSensoryInput(sensory_noise_std);
model_euler.run();

% Compute difference between ODE45 and Euler
diff_x = abs(model.x_history - model_euler.x_history);
diff_v = abs(model.v_history - model_euler.v_history);

fprintf('  Max position difference: %.6f\n', max(diff_x));
fprintf('  Max velocity difference: %.6f\n', max(diff_v));
fprintf('  RMS position difference: %.6f\n', sqrt(mean(diff_x.^2)));
fprintf('  RMS velocity difference: %.6f\n', sqrt(mean(diff_v.^2)));

%% Save Results
fprintf('\nSaving ODE45 results...\n');
save('ode45_results.mat', 't', 'x_est', 'v_est', 'params', 'true_pos', 'true_vel');

fprintf('\n=== STEP 4 COMPLETE ===\n');
fprintf('Key advantages of ODE45:\n');
fprintf('  1. Adaptive time-stepping for efficiency\n');
fprintf('  2. Higher accuracy than fixed-step Euler\n');
fprintf('  3. Better handling of rapid changes\n');
fprintf('  4. Production-ready implementation\n\n');
fprintf('Next: Run run_all_experiments.m to execute all steps together.\n');
