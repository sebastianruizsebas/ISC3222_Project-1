%% Quick Test: Fixed Target Reaching Task
% Verifies that the fixed-target version works correctly
% (Replaces ball tracking with static reaching targets)

clear all; close all;

fprintf('=====================================\n');
fprintf('FIXED TARGETS TEST (Nov 2, 2025)\n');
fprintf('=====================================\n\n');

%% Quick Smoke Test (3 trials, 5 seconds each)

fprintf('Running smoke test: 3 trials × 5 sec per trial\n');

params = struct();
params.n_trials = 3;
params.T_per_trial = 5;      % 5 seconds per trial
params.scale_factor = 1.0;   % Small layers for quick test
params.eta_rep = 0.01;
params.eta_W = 0.001;

% Run the experiment (no plots)
tic;
results = hierarchical_motion_inference_dual_hierarchy(params, false);
elapsed_time = toc;

fprintf('\n✓ Experiment completed in %.2f seconds\n\n', elapsed_time);

%% Analyze Results

fprintf('RESULTS SUMMARY:\n');
fprintf('=================\n\n');

% Extract trial phases
n_trials = params.n_trials;
dt = 0.01;
T_per_trial = params.T_per_trial;
trial_length = round(T_per_trial / dt);
N_total = length(results.interception_error_all);

fprintf('Data dimensions:\n');
fprintf('  Total timesteps: %d (≈ %.1f seconds)\n', N_total, N_total * dt);
fprintf('  Per-trial: %d timesteps (%.1f seconds)\n', trial_length, T_per_trial);
fprintf('  Trials: %d\n\n', n_trials);

% Compute per-trial statistics
fprintf('PER-TRIAL INTERCEPTION ERRORS (reaching error to fixed target):\n');
fprintf('─────────────────────────────────────────────────────────────\n');

for trial = 1:n_trials
    start_idx = (trial-1) * trial_length + 1;
    end_idx = min(trial * trial_length, N_total);
    
    trial_errors = results.interception_error_all(start_idx:end_idx);
    
    fprintf('  Trial %d (target: [%.1f, %.1f, %.1f]):\n', ...
        trial, results.target_positions{trial}(1), ...
        results.target_positions{trial}(2), results.target_positions{trial}(3));
    fprintf('    Initial error:  %.4f m\n', trial_errors(1));
    fprintf('    Final error:    %.4f m\n', trial_errors(end));
    fprintf('    Mean error:     %.4f m\n', mean(trial_errors));
    fprintf('    Min error:      %.4f m\n', min(trial_errors));
    fprintf('    Error reduction: %.1f%%\n\n', ...
        100 * (1 - trial_errors(end) / max(trial_errors(1), 1e-6)));
end

% Free energy summary
fprintf('FREE ENERGY (learning objective):\n');
fprintf('─────────────────────────────────\n');
fprintf('  Initial: %.4f\n', results.free_energy_all(1));
fprintf('  Final:   %.4f\n', results.free_energy_all(end));
fprintf('  Decay:   %.1f%%\n\n', 100 * (1 - results.free_energy_all(end) / max(results.free_energy_all(1), 1e-6)));

% Learning trace summary
fprintf('WEIGHT LEARNING (learning_trace_W):\n');
fprintf('─────────────────────────────────────\n');
fprintf('  Mean weight update magnitude: %.6f\n', mean(results.learning_trace_W));
fprintf('  Peak weight update:          %.6f\n', max(results.learning_trace_W));
fprintf('  Final weight update:         %.6f\n\n', results.learning_trace_W(end));

%% Validation Checks

fprintf('VALIDATION CHECKS:\n');
fprintf('──────────────────\n');

checks_passed = 0;
checks_total = 5;

% Check 1: No NaNs in key outputs
if ~any(isnan(results.interception_error_all)) && ~any(isnan(results.free_energy_all))
    fprintf('  ✓ Check 1: No NaN values in outputs\n');
    checks_passed = checks_passed + 1;
else
    fprintf('  ✗ Check 1: NaN detected in outputs\n');
end

% Check 2: Interception error decreases over trial
trial_errors_1 = results.interception_error_all(1:trial_length);
if trial_errors_1(end) < trial_errors_1(1)
    fprintf('  ✓ Check 2: Interception error decreases within trial\n');
    checks_passed = checks_passed + 1;
else
    fprintf('  ✗ Check 2: Interception error did not decrease\n');
end

% Check 3: Free energy decreases
if results.free_energy_all(end) < results.free_energy_all(1)
    fprintf('  ✓ Check 3: Free energy decreases over experiment\n');
    checks_passed = checks_passed + 1;
else
    fprintf('  ✗ Check 3: Free energy did not decrease\n');
end

% Check 4: Positive learning (weight updates occurring)
if mean(results.learning_trace_W) > 1e-8
    fprintf('  ✓ Check 4: Weights are being updated (learning occurring)\n');
    checks_passed = checks_passed + 1;
else
    fprintf('  ✗ Check 4: No weight updates detected\n');
end

% Check 5: Player moves (positions change)
player_motion = sqrt((results.x_player(end) - results.x_player(1))^2 + ...
                      (results.y_player(end) - results.y_player(1))^2 + ...
                      (results.z_player(end) - results.z_player(1))^2);
if player_motion > 0.01  % moved > 1cm
    fprintf('  ✓ Check 5: Player motion detected (%.3f m total displacement)\n', player_motion);
    checks_passed = checks_passed + 1;
else
    fprintf('  ✗ Check 5: Player did not move\n');
end

fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('TESTS PASSED: %d / %d\n', checks_passed, checks_total);
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

if checks_passed == checks_total
    fprintf('✅ ALL TESTS PASSED - Fixed targets implementation is working!\n\n');
else
    fprintf('⚠️  Some tests failed - review output above\n\n');
end

%% Visualization

fprintf('Generating plots...\n');

figure('Position', [100 100 1200 400]);

% Plot 1: Interception error over time
subplot(1, 3, 1);
hold on;
for trial = 1:n_trials
    start_idx = (trial-1) * trial_length + 1;
    end_idx = min(trial * trial_length, N_total);
    t_trial = (0:end_idx-start_idx) * dt;
    plot(t_trial, results.interception_error_all(start_idx:end_idx), ...
        'LineWidth', 2, 'DisplayName', sprintf('Trial %d', trial));
end
xlabel('Time within trial (s)');
ylabel('Interception error (m)');
title('Reaching Error to Fixed Targets');
legend;
grid on;
hold off;

% Plot 2: Free energy
subplot(1, 3, 2);
plot(results.free_energy_all, 'LineWidth', 2);
xlabel('Timestep');
ylabel('Free energy');
title('Free Energy Decay');
grid on;

% Plot 3: Learning trace
subplot(1, 3, 3);
plot(results.learning_trace_W, 'LineWidth', 2);
xlabel('Timestep');
ylabel('Weight update magnitude');
title('Weight Learning Trace');
grid on;

sgtitle('Fixed Target Reaching Task (Nov 2, 2025)');
savefig('./figures/fixed_targets_test_results.fig');
print('figures/fixed_targets_test_results.png', '-dpng', '-r150');

fprintf('✓ Plots saved to ./figures/\n\n');

fprintf('=====================================\n');
fprintf('TEST COMPLETE\n');
fprintf('=====================================\n');
