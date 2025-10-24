% RUN ALL EXPERIMENTS
% ====================
% Master script to run the complete predictive coding project.
% This executes all four steps in sequence:
%   1. Symbolic derivation
%   2. Numerical simulation (Euler)
%   3. Prior strength comparison
%   4. High-precision ODE45 simulation
%
% Author: Generated for Symbolic & Numeric Computation Course
% Date: October 2025

clear all; close all; clc;

fprintf('\n');
fprintf('========================================================\n');
fprintf('   PREDICTIVE CODING: ACTIVE INFERENCE SIMULATION\n');
fprintf('   Two-Level Hierarchical Visual Motion Model\n');
fprintf('========================================================\n');
fprintf('\n');

%% User Configuration
fprintf('Configuration:\n');
fprintf('  Press any key after each step to continue...\n');
fprintf('  (or modify this script to run non-interactively)\n\n');

pause_between_steps = true;  % Set to false to run all without pausing

%% Step 1: Symbolic Derivation
fprintf('\n');
fprintf('┌────────────────────────────────────────────────────────┐\n');
fprintf('│ STEP 1: Symbolic Derivation of Update Rules           │\n');
fprintf('└────────────────────────────────────────────────────────┘\n');
fprintf('\n');

try
    run('step1_symbolic_derivation.m');
    fprintf('\n✓ Step 1 completed successfully\n');
catch ME
    fprintf('\n✗ Step 1 failed: %s\n', ME.message);
    return;
end

if pause_between_steps
    fprintf('\nPress any key to continue to Step 2...\n');
    pause;
end

%% Step 2: Numerical Simulation
fprintf('\n');
fprintf('┌────────────────────────────────────────────────────────┐\n');
fprintf('│ STEP 2: Numerical Simulation (Euler Integration)      │\n');
fprintf('└────────────────────────────────────────────────────────┘\n');
fprintf('\n');

try
    run('step2_numerical_simulation.m');
    fprintf('\n✓ Step 2 completed successfully\n');
catch ME
    fprintf('\n✗ Step 2 failed: %s\n', ME.message);
    return;
end

if pause_between_steps
    fprintf('\nPress any key to continue to Step 3...\n');
    pause;
end

%% Step 3: Prior Comparison
fprintf('\n');
fprintf('┌────────────────────────────────────────────────────────┐\n');
fprintf('│ STEP 3: Prior Strength Comparison                     │\n');
fprintf('└────────────────────────────────────────────────────────┘\n');
fprintf('\n');

try
    run('step3_prior_comparison.m');
    fprintf('\n✓ Step 3 completed successfully\n');
catch ME
    fprintf('\n✗ Step 3 failed: %s\n', ME.message);
    return;
end

if pause_between_steps
    fprintf('\nPress any key to continue to Step 4...\n');
    pause;
end

%% Step 4: ODE45 Implementation
fprintf('\n');
fprintf('┌────────────────────────────────────────────────────────┐\n');
fprintf('│ STEP 4: High-Precision ODE45 Simulation               │\n');
fprintf('└────────────────────────────────────────────────────────┘\n');
fprintf('\n');

try
    run('step4_ode45_version.m');
    fprintf('\n✓ Step 4 completed successfully\n');
catch ME
    fprintf('\n✗ Step 4 failed: %s\n', ME.message);
    return;
end

%% Summary
fprintf('\n');
fprintf('========================================================\n');
fprintf('   ALL EXPERIMENTS COMPLETED SUCCESSFULLY!\n');
fprintf('========================================================\n');
fprintf('\n');

fprintf('Generated files:\n');
fprintf('  • symbolic_derivations.mat       (Step 1 output)\n');
fprintf('  • simulation_results.mat         (Step 2 output)\n');
fprintf('  • prior_comparison_results.mat   (Step 3 output)\n');
fprintf('  • ode45_results.mat              (Step 4 output)\n');
fprintf('\n');

fprintf('Figures generated:\n');
fprintf('  • Figure 1: Symbolic Derivation (equations)\n');
fprintf('  • Figure 2: Numerical Simulation (6 subplots)\n');
fprintf('  • Figure 3: Prior Comparison (6 subplots)\n');
fprintf('  • Figure 4: ODE45 High-Precision (6 subplots)\n');
fprintf('\n');

%% Optional: Summary Comparison Plot
fprintf('Creating summary comparison figure...\n');

% Load results from all steps
load('simulation_results.mat', 't', 'v_history');
v_euler = v_history;
t_euler = t;

load('ode45_results.mat', 't', 'v_est');
v_ode45 = v_est;
t_ode45 = t;

load('prior_comparison_results.mat', 'results');

% Create summary figure
figure('Position', [100 100 1400 500], 'Name', 'Project Summary');

% Subplot 1: Euler vs ODE45
subplot(1,3,1);
plot(t_euler, v_euler, 'b-', 'LineWidth', 2, 'DisplayName', 'Euler'); hold on;
plot(t_ode45, v_ode45, 'r--', 'LineWidth', 2, 'DisplayName', 'ODE45');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity Estimate', 'FontSize', 11);
title('Method Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on; box on;

% Subplot 2: Prior strength effects
subplot(1,3,2);
true_v = [ones(1, length(results(1).v_history)/2) * 2, ...
          ones(1, length(results(1).v_history)/2) * -1];
t_prior = linspace(0, 10, length(true_v));
plot(t_prior, true_v, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
for j = 1:length(results)
    plot(t_prior, results(j).v_history, 'LineWidth', 2, ...
        'DisplayName', sprintf('σ_v=%.1f', results(j).sigma_v));
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
title('Prior Strength Effects', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on; box on;

% Subplot 3: Key insight diagram
subplot(1,3,3);
axis off;
text(0.5, 0.9, 'KEY INSIGHTS', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

insights = {
    '1. Hierarchical inference works:'
    '   System infers hidden velocity'
    '   from noisy position observations'
    ''
    '2. Prior strength matters:'
    '   Strong → rigid (autism-like)'
    '   Weak → volatile (psychosis-like)'
    ''
    '3. Free energy minimization:'
    '   Balances prediction & flexibility'
    ''
    '4. Adaptive methods (ODE45):'
    '   More accurate than Euler'
};

y_pos = 0.75;
for i = 1:length(insights)
    text(0.1, y_pos, insights{i}, 'FontSize', 10, 'VerticalAlignment', 'top');
    y_pos = y_pos - 0.06;
end

sgtitle('Predictive Coding Project Summary', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n');
fprintf('========================================================\n');
fprintf('   Project complete! Review figures and .mat files.\n');
fprintf('   See README.md for detailed interpretation.\n');
fprintf('========================================================\n');
fprintf('\n');
