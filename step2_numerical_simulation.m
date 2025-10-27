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

%% Create and Run Euler Model
fprintf('Creating Euler predictive coding model...\n');

% Time parameters
dt = 0.01;            % Time step (10ms)
T = 10;               % Simulate for 10 seconds

% Model hyperparameters (priors)
sigma_x = 0.1;        % Sensory precision (how much to trust observations)
sigma_v = 1.0;        % Velocity prior strength (flexibility of beliefs)

fprintf('  Duration: %.1f seconds\n', T);
fprintf('  Time step: %.3f seconds\n', dt);
fprintf('  σ_x = %.2f   (sensory noise, smaller = trust sensors more)\n', sigma_x);
fprintf('  σ_v = %.2f   (prior strength, larger = more flexible beliefs)\n\n', sigma_v);

% Create model instance
model = EulerModel(dt, T, sigma_x, sigma_v);

% Generate sensory input
sensory_noise_std = 0.05;
model.generateSensoryInput(sensory_noise_std);

fprintf('  First 5s: velocity = +2 (moving right)\n');
fprintf('  Last 5s:  velocity = -1 (moving left)\n');
fprintf('  Sensory noise: σ = %.3f\n\n', sensory_noise_std);

% Run simulation
model.run();

% Extract results for compatibility with existing code
t = model.t;
x_history = model.x_history;
v_history = model.v_history;
prediction_error_x = model.prediction_error_x;
prediction_error_v = model.prediction_error_v;
free_energy = model.free_energy;
true_position = model.true_position;
true_velocity = model.true_velocity;
x_obs = model.x_obs;
N = model.N;
mu_v = model.mu_v;
mid = floor(N/2);

%% Visualization
fprintf('Generating plots...\n');
model.visualize();

%% Compute Performance Metrics
model.printSummary();

%% Save Results
model.save('simulation_results.mat');

fprintf('\n=== STEP 2 COMPLETE ===\n');
fprintf('Key observations:\n');
fprintf('  1. System tracks position despite noise\n');
fprintf('  2. Infers hidden velocity from observations\n');
fprintf('  3. Adapts when velocity changes at t=5s\n');
fprintf('  4. Free energy decreases as predictions improve\n\n');
fprintf('Next: Run step3_prior_comparison.m to explore different priors.\n');
