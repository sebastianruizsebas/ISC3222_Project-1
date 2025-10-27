% filepath: step5_rao_ballard_extension.m
% STEP 5: RAO & BALLARD PREDICTIVE CODING EXTENSION
% ====================================================
% This extends the continuous model to include explicit prediction
% and error units as in Rao & Ballard (1999).
%
% KEY ADDITIONS:
% 1. Separate representation vs. error units
% 2. Explicit top-down predictions
% 3. Bottom-up error propagation
% 4. Multi-level cascading inference

clear; clc; close all;
fprintf('=== STEP 5: RAO & BALLARD EXTENSION ===\n\n');

%% Create and Run Rao & Ballard Model
fprintf('Building three-level Rao & Ballard hierarchy...\n\n');

fprintf('Architecture:\n');
fprintf('  Level 3: Acceleration (a) → predicts velocity\n');
fprintf('  Level 2: Velocity (v)     → predicts position\n');
fprintf('  Level 1: Position (x)     → compared to sensors\n\n');

% Time parameters
dt = 0.01;
T = 10.0;

% Precision weights (inverse variances)
pi_x = 100;    % Level 1 sensory precision (1/σ_x²)
pi_v = 10;     % Level 2 prior precision (1/σ_v²)
pi_a = 1;      % Level 3 prior precision (1/σ_a²)

fprintf('Precision weights:\n');
fprintf('  π_x = %.0f (sensory precision)\n', pi_x);
fprintf('  π_v = %.0f (velocity prior precision)\n', pi_v);
fprintf('  π_a = %.0f (acceleration prior precision)\n\n', pi_a);

% Create model instance
model = RaoBallardModel(dt, T, pi_x, pi_v, pi_a);

% Generate sensory input
sensor_noise = 0.05;
a_before = 0;
a_after = -3;
change_time = 5.0;

fprintf('Creating sensory input with changing dynamics...\n');
model.generateSensoryInput(sensor_noise, a_before, a_after, change_time);

fprintf('  Acceleration changes at t=5s: %.0f → %.0f\n', a_before, a_after);
fprintf('  Sensory noise: σ = %.3f\n\n', sensor_noise);

% Run simulation
model.run();

% Extract results for compatibility
t = model.t;
x_rep = model.x_rep;
v_rep = model.v_rep;
a_rep = model.a_rep;
err_x = model.err_x;
err_v = model.err_v;
err_a = model.err_a;
pred_x = model.pred_x;
pred_v = model.pred_v;
true_x = model.true_x;
true_v = model.true_v;
true_a = model.true_a;
x_obs = model.x_obs;
free_energy = model.free_energy;

%% Visualization
model.visualize();

%% Performance Metrics
model.printSummary();

%% Save Results
model.save('rao_ballard_results.mat');

fprintf('\n=== STEP 5 COMPLETE ===\n');
fprintf('Rao & Ballard extension demonstrates:\n');
fprintf('  1. Explicit prediction and error units\n');
fprintf('  2. Three-level hierarchical inference\n');
fprintf('  3. Bidirectional information flow\n');
fprintf('  4. Cascading error correction\n\n');
fprintf('Compare with Step 2-4 to see architectural differences!\n');