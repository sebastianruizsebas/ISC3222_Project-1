% HIERARCHICAL MOTION INFERENCE TEMPLATE
% ======================================
% A clean, extensible implementation of hierarchical predictive coding
% for motion inference from noisy observations.
%
% THEORY:
% Infers hidden velocity and acceleration from noisy position observations
% by minimizing free energy (precision-weighted prediction errors) across
% a three-level hierarchy:
%
%   Level 3: Acceleration (a) → predicts Level 2
%   Level 2: Velocity (v)     → predicts Level 1  
%   Level 1: Position (x)     → compared to observations
%
% USAGE:
% 1. Set parameters in "Configuration" section
% 2. Run script to see hierarchical inference in action
% 3. Extend by modifying dynamics, adding levels, or changing priors
%
% AUTHOR: Sebastian Ruiz for FSU Symbolic & Numerical Computation with Dr. Alan Lemmon
% DATE: October 2025

clear; clc; close all;

%% ========================================================================
%  CONFIGURATION
%  ========================================================================

% Time parameters
dt = 0.01;              % Time step (seconds)
T = 10;                 % Total duration (seconds)
t = 0:dt:T;
N = length(t);

% Precision weights (inverse variances - higher = more precise)
pi_x = 100;             % Sensory precision (trust in observations)
pi_v = 10;              % Velocity prior precision
pi_a = 1;               % Acceleration prior precision

% Learning rates
eta_rep = 0.1;          % Representation update rate
eta_err = 0.5;          % Error propagation rate

% Prior expectations
mu_v = 0;               % Expected velocity (default: stationary)
mu_a = 0;               % Expected acceleration (default: constant velocity)

fprintf('=== HIERARCHICAL MOTION INFERENCE ===\n\n');
fprintf('Configuration:\n');
fprintf('  Time: %.1fs (dt=%.3fs)\n', T, dt);
fprintf('  Precision: π_x=%.0f, π_v=%.0f, π_a=%.0f\n', pi_x, pi_v, pi_a);
fprintf('  Learning rates: η_rep=%.2f, η_err=%.2f\n\n', eta_rep, eta_err);

%% ========================================================================
%  GENERATE SENSORY INPUT
%  ========================================================================

% True dynamics: acceleration changes at t=5s
a_true = zeros(1, N);
a_true(t < 5) = 0;      % Constant velocity initially
a_true(t >= 5) = -3;    % Deceleration after t=5s

% Integrate to get velocity and position
v_true = cumsum(a_true) * dt + 2;  % Initial velocity = 2
x_true = cumsum(v_true) * dt;

% Add observation noise
sensor_noise = 0.05;
x_obs = x_true + sensor_noise * randn(1, N);

fprintf('Sensory input:\n');
fprintf('  Initial velocity: %.1f m/s\n', v_true(1));
fprintf('  Acceleration change at t=5s: %.0f → %.0f m/s²\n', a_true(1), a_true(end));
fprintf('  Sensor noise: σ=%.3f\n\n', sensor_noise);

%% ========================================================================
%  INITIALIZE STATE VARIABLES
%  ========================================================================

% Representation units (beliefs about hidden states)
x_rep = zeros(1, N);    x_rep(1) = 0;
v_rep = zeros(1, N);    v_rep(1) = 0;
a_rep = zeros(1, N);    a_rep(1) = 0;

% Error units (precision-weighted prediction errors)
err_x = zeros(1, N);
err_v = zeros(1, N);
err_a = zeros(1, N);

% Predictions (top-down)
pred_x = zeros(1, N);
pred_v = zeros(1, N);

% Free energy (model evidence)
free_energy = zeros(1, N);

%% ========================================================================
%  HIERARCHICAL INFERENCE LOOP
%  ========================================================================

fprintf('Running hierarchical inference');

for i = 1:N-1
    if mod(i, 100) == 0, fprintf('.'); end
    
    % --- TOP-DOWN PREDICTIONS ---
    pred_v(i) = a_rep(i);           % Level 3 predicts Level 2
    pred_x(i) = v_rep(i);           % Level 2 predicts Level 1
    
    % --- COMPUTE PREDICTION ERRORS (bottom-up) ---
    % Level 1: Sensory error (observation vs. prediction)
    err_x(i) = pi_x * (x_obs(i) - x_rep(i));
    
    % Level 2: Velocity error (observation vs. prediction from above)
    observed_v_change = (x_obs(i) - (i>1)*x_rep(i-1)) / dt;
    err_v(i) = pi_v * (observed_v_change - pred_v(i));
    
    % Level 3: Acceleration error (prior mismatch)
    err_a(i) = pi_a * (a_rep(i) - mu_a);
    
    % --- FREE ENERGY ---
    free_energy(i) = 0.5 * (err_x(i)^2/pi_x + err_v(i)^2/pi_v + err_a(i)^2/pi_a);
    
    % --- UPDATE REPRESENTATIONS (minimize free energy) ---
    % Level 1: Position representation
    dx_rep = eta_rep * (err_x(i) / pi_x);
    x_rep(i+1) = x_rep(i) + dt * dx_rep;
    
    % Level 2: Velocity representation (receives error from above and below)
    dv_rep = eta_rep * (err_v(i)/pi_v - err_x(i)/pi_x);
    v_rep(i+1) = v_rep(i) + dt * dv_rep;
    
    % Level 3: Acceleration representation
    da_rep = eta_rep * (err_a(i)/pi_a - err_v(i)/pi_v);
    a_rep(i+1) = a_rep(i) + dt * da_rep;
end

fprintf(' Done!\n\n');

%% ========================================================================
%  COMPUTE PERFORMANCE METRICS
%  ========================================================================

% Compute inference errors
pos_error = abs(x_rep - x_true);
vel_error = abs(v_rep - v_true);
acc_error = abs(a_rep - a_true);

%% ========================================================================
%  PERFORMANCE SUMMARY
%  ========================================================================

fprintf('Performance Metrics:\n');
fprintf('  Mean position error:     %.4f\n', mean(pos_error));
fprintf('  Mean velocity error:     %.4f\n', mean(vel_error));
fprintf('  Mean acceleration error: %.4f\n', mean(acc_error));
fprintf('  Final free energy:       %.4f\n', free_energy(end));

% Adaptation time (how long to adapt after change at t=5s)
change_idx = find(t >= 5, 1);
post_change = change_idx:N;
threshold = 0.2 * abs(a_true(change_idx) - a_true(change_idx-1));
adapted_idx = find(acc_error(post_change) < threshold, 1);

if ~isempty(adapted_idx)
    adapt_time = adapted_idx * dt;
    fprintf('  Adaptation time:         %.3f s\n', adapt_time);
else
    fprintf('  Adaptation time:         > %.1f s\n', T-5);
end

fprintf('\n=== INFERENCE COMPLETE ===\n\n');

%% ========================================================================
%  EXTENSION GUIDE
%  ========================================================================

fprintf('Extension Guide:\n');
fprintf('  1. Add more levels: Define new representation/error units\n');
fprintf('  2. Change dynamics: Modify prediction equations (pred_v, pred_x)\n');
fprintf('  3. Adjust precision: Vary pi_x, pi_v, pi_a for different behaviors\n');
fprintf('  4. Add nonlinearity: Replace linear predictions with f(state)\n');
fprintf('  5. Implement learning: Make precision weights adaptive\n');
fprintf('  6. Active inference: Add action variables to minimize future errors\n');
fprintf('  7. Spatial extension: Generalize to 2D/3D state spaces\n\n');

fprintf('Key equations:\n');
fprintf('  Prediction: x_pred = f(v), v_pred = g(a)\n');
fprintf('  Error: ε = π(observation - prediction)\n');
fprintf('  Update: dx/dt = -∂F/∂x = function of errors\n');
fprintf('  Free energy: F = 0.5 * Σ(ε²/π)\n\n');

fprintf('Parameters to explore:\n');
fprintf('  • Precision weights (pi_x, pi_v, pi_a): Balance sensory vs. prior trust\n');
fprintf('  • Learning rates (eta_rep, eta_err): Speed of adaptation\n');
fprintf('  • Prior means (mu_v, mu_a): Expected dynamics\n');
fprintf('  • Sensor noise: Observation reliability\n\n');

%% ========================================================================
%  SAVE RESULTS (Optional)
%  ========================================================================

save_results = true;  % Set to true to save

if save_results
    save('hierarchical_inference_results.mat', 't', 'x_rep', 'v_rep', 'a_rep', ...
        'x_true', 'v_true', 'a_true', 'x_obs', 'err_x', 'err_v', 'err_a', ...
        'free_energy', 'pi_x', 'pi_v', 'pi_a', 'eta_rep', 'eta_err');
    fprintf('Results saved to: hierarchical_inference_results.mat\n');
end
