% filepath: step6_compare_architectures.m
% STEP 6: COMPARE RAO & BALLARD VS. FREE ENERGY FORMULATION
% ===========================================================

clear; clc; close all;

% Load both models
load('simulation_results.mat');        % Your original (Step 2)
load('rao_ballard_results.mat');       % Rao & Ballard (Step 5)

fprintf('=== COMPARING ARCHITECTURES ===\n\n');

%% Side-by-Side Comparison
fig = figure('Position', [50 50 1800 800]);
sgtitle('Architecture Comparison: Free Energy vs. Rao & Ballard', ...
        'FontSize', 16, 'FontWeight', 'bold');

% Original model (left column)
subplot(2, 3, 1);
plot(t, v_history, 'b-', 'LineWidth', 2); hold on;
plot(t, true_velocity, 'k--', 'LineWidth', 2);
xline(5, 'r:', 'LineWidth', 1.5);
title('Free Energy: Velocity (2-level)', 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Velocity');
legend('Estimated', 'True', 'Location', 'best');
grid on;

% Rao & Ballard (right column)
subplot(2, 3, 2);
plot(t, v_rep, 'g-', 'LineWidth', 2); hold on;
plot(t, true_v, 'k--', 'LineWidth', 2);
xline(5, 'r:', 'LineWidth', 1.5);
title('Rao & Ballard: Velocity (3-level)', 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Velocity');
legend('Inferred', 'True', 'Location', 'best');
grid on;

subplot(2, 3, 3);
plot(t, a_rep, 'r-', 'LineWidth', 2); hold on;
plot(t, true_a, 'k--', 'LineWidth', 2);
xline(5, 'r:', 'LineWidth', 1.5);
title('Rao & Ballard: Acceleration', 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Acceleration');
legend('Inferred', 'True', 'Location', 'best');
grid on;

% Error comparison
subplot(2, 3, 4);
fe_err = abs(v_history - true_velocity);
rb_err = abs(v_rep - true_v);
plot(t, fe_err, 'b-', 'LineWidth', 2, 'DisplayName', 'Free Energy'); hold on;
plot(t, rb_err, 'g-', 'LineWidth', 2, 'DisplayName', 'Rao & Ballard');
xline(5, 'r:', 'LineWidth', 1.5);
title('Velocity Inference Error', 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Absolute Error');
legend('Location', 'best');
grid on;

% Free energy comparison
subplot(2, 3, 5);
plot(t, free_energy_original, 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Free Energy'); hold on;
plot(t, free_energy, 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Rao & Ballard');
xline(5, 'r:', 'LineWidth', 1.5);
title('Model Evidence', 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Free Energy');
legend('Location', 'best');
grid on;

% Architecture diagram
subplot(2, 3, 6);
axis off;
text(0.5, 0.95, 'Key Differences', 'FontSize', 14, ...
     'FontWeight', 'bold', 'HorizontalAlignment', 'center');

diff_text = sprintf([...
    'Free Energy (Steps 2-4):\n'...
    '• Implicit predictions\n'...
    '• Continuous gradient descent\n'...
    '• 2 levels (position, velocity)\n'...
    '• Single dynamics equation\n\n'...
    'Rao & Ballard (Step 5):\n'...
    '• Explicit prediction units\n'...
    '• Separate error neurons\n'...
    '• 3 levels (x, v, a)\n'...
    '• Cascading inference\n'...
    '• Closer to neural implementation']);

text(0.1, 0.75, diff_text, 'FontSize', 10, ...
    'VerticalAlignment', 'top', 'FontName', 'FixedWidth');

fprintf('Quantitative Comparison:\n');
fprintf('  Mean velocity error (Free Energy):  %.4f\n', mean(fe_err));
fprintf('  Mean velocity error (Rao & Ballard): %.4f\n', mean(rb_err));
fprintf('  Final free energy (original):        %.4f\n', free_energy_original(end));
fprintf('  Final free energy (Rao & Ballard):   %.4f\n', free_energy(end));