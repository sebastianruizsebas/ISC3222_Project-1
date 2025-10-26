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

%% Model Architecture
fprintf('Building three-level Rao & Ballard hierarchy...\n\n');

% Level structure:
% Level 3: Acceleration (a) → predicts → Level 2: Velocity (v)
% Level 2: Velocity (v)    → predicts → Level 1: Position (x)
% Level 1: Position (x)    → compared to → Sensory input (x_obs)

%% Parameters
% Time
dt = 0.01;
T = 10.0;
t = 0:dt:T;
N = length(t);

% Precision weights (inverse variances)
pi_x = 100;    % Level 1 sensory precision (1/σ_x²)
pi_v = 10;     % Level 2 prior precision (1/σ_v²)
pi_a = 1;      % Level 3 prior precision (1/σ_a²)

% Learning rates (Rao & Ballard use separate rates)
eta_rep = 0.1;   % Learning rate for representations
eta_err = 0.5;   % Learning rate for error units

% Prior expectations
mu_a = 0;        % Expected acceleration (stationary velocity)

fprintf('Architecture:\n');
fprintf('  Level 3: Acceleration (a) → predicts velocity\n');
fprintf('  Level 2: Velocity (v)     → predicts position\n');
fprintf('  Level 1: Position (x)     → compared to sensors\n\n');

fprintf('Precision weights:\n');
fprintf('  π_x = %.0f (sensory precision)\n', pi_x);
fprintf('  π_v = %.0f (velocity prior precision)\n', pi_v);
fprintf('  π_a = %.0f (acceleration prior precision)\n\n', pi_a);

%% Create Sensory Input
fprintf('Creating sensory input with changing dynamics...\n');
mid = floor(N/2);

% True motion: constant acceleration phases
true_a = [zeros(1, mid), -3*ones(1, N-mid)];  % Sudden deceleration at t=5s
true_v = zeros(1, N);
true_x = zeros(1, N);

for i = 2:N
    true_v(i) = true_v(i-1) + true_a(i-1) * dt;
    true_x(i) = true_x(i-1) + true_v(i-1) * dt;
end

% Add sensory noise
sensor_noise = 0.05;
x_obs = true_x + randn(1, N) * sensor_noise;

fprintf('  Acceleration changes at t=5s: 0 → -3\n');
fprintf('  Sensory noise: σ = %.3f\n\n', sensor_noise);

%% Initialize State Variables

% REPRESENTATIONS (beliefs about causes)
a_rep = zeros(1, N);  % Acceleration representation
v_rep = zeros(1, N);  % Velocity representation
x_rep = zeros(1, N);  % Position representation

% ERROR UNITS (prediction errors)
err_x = zeros(1, N);  % Level 1 error (sensory)
err_v = zeros(1, N);  % Level 2 error (velocity prior)
err_a = zeros(1, N);  % Level 3 error (acceleration prior)

% PREDICTIONS (top-down signals)
pred_x = zeros(1, N);  % Position predicted from velocity
pred_v = zeros(1, N);  % Velocity predicted from acceleration

% Initial conditions
a_rep(1) = 0;
v_rep(1) = 0;
x_rep(1) = x_obs(1);

%% Rao & Ballard Update Loop
fprintf('Running Rao & Ballard predictive coding simulation...\n');

for i = 1:N-1
    % ===== FORWARD PASS: Generate Predictions (Top-Down) =====
    % Level 3 → Level 2: Acceleration predicts velocity change
    pred_v(i) = a_rep(i);  % dv/dt = a
    
    % Level 2 → Level 1: Velocity predicts position change
    pred_x(i) = v_rep(i);  % dx/dt = v
    
    % ===== COMPUTE PREDICTION ERRORS (Bottom-Up) =====
    % Level 1: Sensory prediction error (weighted by precision)
    err_x(i) = pi_x * (x_obs(i) - x_rep(i));
    
    % Level 2: Velocity prediction error
    % Error = (observed change - predicted change)
    observed_v_change = (x_rep(i) - x_rep(max(1, i-1))) / dt;
    err_v(i) = pi_v * (observed_v_change - pred_x(i));
    
    % Level 3: Acceleration prior error
    observed_a_change = (v_rep(i) - v_rep(max(1, i-1))) / dt;
    err_a(i) = pi_a * (observed_a_change - pred_v(i));
    
    % ===== UPDATE REPRESENTATIONS (Error Correction) =====
    % Position update (driven by sensory error)
    x_rep(i+1) = x_rep(i) + dt * eta_rep * err_x(i);
    
    % Velocity update (driven by errors from above AND below)
    % Bottom-up: sensory prediction error pushes velocity up
    % Top-down: prior error pushes velocity toward prediction
    v_rep(i+1) = v_rep(i) + dt * eta_rep * (err_v(i) / pi_v + ...
                                             err_x(i) / pi_x);
    
    % Acceleration update (driven by velocity error and prior)
    a_rep(i+1) = a_rep(i) + dt * eta_rep * (err_a(i) / pi_a + ...
                                             err_v(i) / pi_v - ...
                                             (a_rep(i) - mu_a) * pi_a);
end

fprintf('Done!\n\n');

%% Compute Free Energy (Model Evidence)
free_energy = zeros(1, N);
for i = 1:N
    free_energy(i) = 0.5 * (err_x(i)^2 / pi_x + ...
                            err_v(i)^2 / pi_v + ...
                            err_a(i)^2 / pi_a);
end

%% Visualization
fprintf('Generating Rao & Ballard visualization...\n');

fig = figure('Position', [100 100 1600 1000]);
sgtitle('Rao & Ballard Predictive Coding: Three-Level Hierarchy', ...
        'FontSize', 16, 'FontWeight', 'bold');

% 1. Position inference
subplot(3, 3, 1);
plot(t, true_x, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
plot(t, x_obs, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5, ...
     'DisplayName', 'Noisy Obs');
plot(t, x_rep, 'b-', 'LineWidth', 2, 'DisplayName', 'Inferred');
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Position', 'FontSize', 11);
title('Level 1: Position Representation', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;

% 2. Velocity inference
subplot(3, 3, 2);
plot(t, true_v, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
plot(t, v_rep, 'g-', 'LineWidth', 2, 'DisplayName', 'Inferred');
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
title('Level 2: Velocity Representation', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;

% 3. Acceleration inference
subplot(3, 3, 3);
plot(t, true_a, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
plot(t, a_rep, 'r-', 'LineWidth', 2, 'DisplayName', 'Inferred');
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Acceleration', 'FontSize', 11);
title('Level 3: Acceleration Representation', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;

% 4. Sensory prediction error
subplot(3, 3, 4);
plot(t, err_x, 'b-', 'LineWidth', 2);
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Error Signal', 'FontSize', 11);
title('Level 1: Sensory Error (ε_x)', 'FontWeight', 'bold');
grid on;

% 5. Velocity prediction error
subplot(3, 3, 5);
plot(t, err_v, 'g-', 'LineWidth', 2);
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Error Signal', 'FontSize', 11);
title('Level 2: Velocity Error (ε_v)', 'FontWeight', 'bold');
grid on;

% 6. Acceleration prediction error
subplot(3, 3, 6);
plot(t, err_a, 'r-', 'LineWidth', 2);
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Error Signal', 'FontSize', 11);
title('Level 3: Acceleration Error (ε_a)', 'FontWeight', 'bold');
grid on;

% 7. Predictions vs. representations
subplot(3, 3, 7);
plot(t, v_rep, 'g-', 'LineWidth', 2, 'DisplayName', 'v (representation)'); hold on;
plot(t, pred_x, 'b--', 'LineWidth', 2, 'DisplayName', 'prediction from v');
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Value', 'FontSize', 11);
title('Top-Down Predictions', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;

% 8. Free energy
subplot(3, 3, 8);
plot(t, free_energy, 'Color', [0.6 0.2 0.8], 'LineWidth', 2);
xline(5, 'red', ':', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Free Energy', 'FontSize', 11);
title('Model Evidence (Lower = Better)', 'FontWeight', 'bold');
grid on;

% 9. Information flow diagram
subplot(3, 3, 9);
axis off;
text(0.5, 0.9, 'Rao & Ballard Architecture', 'FontSize', 13, ...
     'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Draw hierarchy
y_levels = [0.7, 0.5, 0.3, 0.1];
labels = {'Level 3: Acceleration (a)', 'Level 2: Velocity (v)', ...
          'Level 1: Position (x)', 'Sensory Input (x_{obs})'};
colors = {'red', 'green', 'blue', 'gray'};

for i = 1:4
    rectangle('Position', [0.2, y_levels(i)-0.05, 0.6, 0.08], ...
             'FaceColor', colors{i}, 'EdgeColor', 'black', ...
             'LineWidth', 2, 'Curvature', 0.2);
    text(0.5, y_levels(i), labels{i}, 'FontSize', 10, 'Color', 'white', ...
         'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    
    if i < 4
        % Prediction arrows (downward, green)
        annotation('arrow', [0.55, 0.55], [y_levels(i)-0.03, y_levels(i+1)+0.05], ...
                  'Color', 'green', 'LineWidth', 2);
        text(0.6, (y_levels(i) + y_levels(i+1))/2, 'pred', ...
            'FontSize', 8, 'Color', 'green');
        
        % Error arrows (upward, red)
        annotation('arrow', [0.45, 0.45], [y_levels(i+1)+0.03, y_levels(i)-0.05], ...
                  'Color', 'red', 'LineWidth', 2);
        text(0.35, (y_levels(i) + y_levels(i+1))/2, 'error', ...
            'FontSize', 8, 'Color', 'red', 'HorizontalAlignment', 'right');
    end
end

text(0.5, 0.02, '↓ Predictions (green) | ↑ Errors (red)', ...
    'FontSize', 9, 'HorizontalAlignment', 'center');

%% Performance Metrics
fprintf('\nPerformance Metrics:\n');
fprintf('  Final position error: %.4f\n', abs(x_rep(end) - true_x(end)));
fprintf('  Final velocity error: %.4f\n', abs(v_rep(end) - true_v(end)));
fprintf('  Final acceleration error: %.4f\n', abs(a_rep(end) - true_a(end)));
fprintf('  Final free energy: %.4f\n', free_energy(end));

% Adaptation analysis
change_idx = find(t >= 5, 1);
post_change_a_error = abs(a_rep(change_idx:end) - true_a(change_idx:end));
adapt_idx = find(post_change_a_error < 0.5, 1);
if ~isempty(adapt_idx)
    adapt_time = t(change_idx + adapt_idx) - 5;
    fprintf('  Adaptation time (acceleration): %.2f seconds\n', adapt_time);
else
    fprintf('  Adaptation time: > %.1f seconds\n', t(end) - 5);
end

%% Save Results
fprintf('\nSaving results...\n');
save('rao_ballard_results.mat', 't', ...
     'x_rep', 'v_rep', 'a_rep', ...
     'err_x', 'err_v', 'err_a', ...
     'pred_x', 'pred_v', ...
     'true_x', 'true_v', 'true_a', ...
     'x_obs', 'free_energy', ...
     'pi_x', 'pi_v', 'pi_a');

fprintf('\n=== STEP 5 COMPLETE ===\n');
fprintf('Rao & Ballard extension demonstrates:\n');
fprintf('  1. Explicit prediction and error units\n');
fprintf('  2. Three-level hierarchical inference\n');
fprintf('  3. Bidirectional information flow\n');
fprintf('  4. Cascading error correction\n\n');
fprintf('Compare with Step 2-4 to see architectural differences!\n');