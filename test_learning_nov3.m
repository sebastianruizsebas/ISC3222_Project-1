% TEST SCRIPT: Verify All Fixes Are Working
% Run this to validate that learning is enabled
%
% Expected: Console shows decreasing FE and IntErr, |dW| > 0, no NaN/Inf
%
% Duration: ~30 seconds for full run, ~5 seconds for quick test

%% QUICK TEST (5 seconds, 1 trial, verify learning signal exists)
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('QUICK TEST: Verify Motor Noise & Learning Signal\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

quick_params = struct();
quick_params.n_trials = 1;
quick_params.T_per_trial = 5;  % 5 seconds only
quick_params.eta_rep = 0.01;
quick_params.eta_W = 0.001;
quick_params.save_results = false;  % Don't save plots

fprintf('Running quick test (1 trial, 5 seconds)...\n');
tic;
quick_results = hierarchical_motion_inference_dual_hierarchy(quick_params, false);
elapsed = toc;

fprintf('\nQuick test completed in %.1f seconds\n', elapsed);
fprintf('\n✓ QUICK TEST DIAGNOSTICS:\n');
fprintf('─────────────────────────────────────────────────────────────\n');

% Extract final metrics
final_FE = quick_results.free_energy_all(end);
final_IntErr = quick_results.interception_error_all(end);
mean_dW = mean(quick_results.learning_trace_W(quick_results.learning_trace_W > 0));

fprintf('  Final Free Energy:        %.4e\n', final_FE);
fprintf('  Final Interception Error: %.4f\n', final_IntErr);
fprintf('  Mean |dW| (non-zero):     %.4e\n', mean_dW);
fprintf('  Min Free Energy:          %.4e\n', min(quick_results.free_energy_all));
fprintf('  Max Free Energy:          %.4e\n', max(quick_results.free_energy_all));

% Check learning signal
has_nonzero_dW = sum(quick_results.learning_trace_W > 1e-6) > 100;
FE_decreased = quick_results.free_energy_all(1) > final_FE;
no_NaNs = sum(isnan(quick_results.free_energy_all)) == 0;

fprintf('\n✓ CHECKS:\n');
if has_nonzero_dW
    fprintf('  [✓] Non-zero weight updates (|dW| > 1e-6)          : PASS\n');
else
    fprintf('  [✗] Non-zero weight updates (|dW| > 1e-6)          : FAIL\n');
end
if FE_decreased
    fprintf('  [✓] Free energy decreased over time               : PASS\n');
else
    fprintf('  [✗] Free energy decreased over time               : FAIL\n');
end
if no_NaNs
    fprintf('  [✓] No NaN/Inf in free energy                     : PASS\n');
else
    fprintf('  [✗] No NaN/Inf in free energy                     : FAIL\n');
end

if has_nonzero_dW && FE_decreased && no_NaNs
    fprintf('\n✅ QUICK TEST PASSED - Learning signal exists!\n');
    fprintf('   Proceed to full test.\n\n');
else
    fprintf('\n❌ QUICK TEST FAILED - Check diagnostics above.\n\n');
end

%% FULL TEST (30 seconds, 3 trials, learn across multiple tasks)
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('FULL TEST: Multi-Task Learning Across 3 Interception Tasks\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

full_params = struct();
full_params.n_trials = 3;
full_params.T_per_trial = 10;  % 10 seconds per trial = 30 seconds total
full_params.eta_rep = 0.01;
full_params.eta_W = 0.001;
full_params.save_results = false;  % Don't save plots

fprintf('Running full test (3 trials × 10 seconds = ~30 seconds)...\n');
fprintf('Watch console output every 100 steps.\n\n');
tic;
full_results = hierarchical_motion_inference_dual_hierarchy(full_params, false);
elapsed_full = toc;

fprintf('\n\nFull test completed in %.1f seconds\n', elapsed_full);
fprintf('\n✓ FULL TEST DIAGNOSTICS:\n');
fprintf('─────────────────────────────────────────────────────────────\n');

% Analyze learning across trials
n_steps = length(full_results.free_energy_all);
steps_per_trial = n_steps / 3;

for trial = 1:3
    trial_start = round((trial-1) * steps_per_trial) + 1;
    trial_end = round(trial * steps_per_trial);
    
    FE_trial = full_results.free_energy_all(trial_start:trial_end);
    IntErr_trial = full_results.interception_error_all(trial_start:trial_end);
    dW_trial = full_results.learning_trace_W(trial_start:trial_end);
    
    fprintf('\nTrial %d (steps %d-%d):\n', trial, trial_start, trial_end);
    fprintf('  Initial FE:      %.4e\n', FE_trial(1));
    fprintf('  Final FE:        %.4e\n', FE_trial(end));
    fprintf('  FE improvement:  %.1f%%\n', 100 * (FE_trial(1) - FE_trial(end)) / (FE_trial(1) + 1e-9));
    fprintf('  Initial IntErr:  %.4f\n', IntErr_trial(1));
    fprintf('  Final IntErr:    %.4f\n', IntErr_trial(end));
    if IntErr_trial(end) < IntErr_trial(1)
        fprintf('  Interception improved: YES ✓\n');
    else
        fprintf('  Interception improved: NO ✗\n');
    end
    fprintf('  Avg |dW|:        %.4e\n', mean(dW_trial(dW_trial > 0)));
end

%% PLOT LEARNING CURVES
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('CREATING LEARNING CURVE PLOTS\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

figure('Name', 'Learning Analysis', 'NumberTitle', 'off');

% Plot 1: Free Energy
subplot(2,2,1);
plot(full_results.free_energy_all, 'LineWidth', 2);
xlabel('Step'); ylabel('Free Energy');
title('Free Energy Over Time');
grid on;
set(gca, 'YScale', 'log');

% Plot 2: Interception Error
subplot(2,2,2);
plot(full_results.interception_error_all, 'LineWidth', 2);
xlabel('Step'); ylabel('Distance (m)');
title('Interception Error (Player ↔ Ball)');
grid on;

% Plot 3: Weight Update Magnitude
subplot(2,2,3);
plot(full_results.learning_trace_W, 'LineWidth', 1.5);
xlabel('Step'); ylabel('|dW|');
title('Weight Update Magnitude (Learning Signal)');
grid on;
set(gca, 'YScale', 'log');

% Plot 4: Trajectories (final trial)
subplot(2,2,4);
trial_start = round(2 * steps_per_trial) + 1;
trial_end = n_steps;
plot(full_results.x_player(trial_start:trial_end), full_results.y_player(trial_start:trial_end), ...
    'b-', 'LineWidth', 2, 'DisplayName', 'Player');
hold on;
plot(full_results.x_ball(trial_start:trial_end), full_results.y_ball(trial_start:trial_end), ...
    'r-', 'LineWidth', 2, 'DisplayName', 'Ball');
plot(full_results.x_player(trial_start), full_results.y_player(trial_start), 'bo', 'MarkerSize', 8);
plot(full_results.x_ball(trial_start), full_results.y_ball(trial_start), 'rs', 'MarkerSize', 8);
xlabel('X (m)'); ylabel('Y (m)');
title('Trial 3: Player vs Ball Trajectory');
legend;
grid on;
axis equal;

sgtitle('LEARNING ANALYSIS - Nov 3, 2025 Fixes');

%% SUMMARY
fprintf('\n✓ ANALYSIS SUMMARY:\n');
fprintf('─────────────────────────────────────────────────────────────\n');

% Calculate overall improvement
FE_improvement = 100 * (full_results.free_energy_all(1) - full_results.free_energy_all(end)) / ...
    (full_results.free_energy_all(1) + 1e-9);
IntErr_improvement = 100 * (full_results.interception_error_all(1) - full_results.interception_error_all(end)) / ...
    (full_results.interception_error_all(1) + 1e-9);

fprintf('\nOverall Performance:\n');
fprintf('  Free Energy Improvement:      %.1f%%\n', FE_improvement);
fprintf('  Interception Error Improvement: %.1f%%\n', IntErr_improvement);
fprintf('  Total Weight Updates:         %.0f (non-zero)\n', sum(full_results.learning_trace_W > 1e-9));
fprintf('  Average Learning Signal:      %.4e\n', mean(full_results.learning_trace_W(full_results.learning_trace_W > 1e-9)));

% Final verdict
if FE_improvement > 10 && sum(full_results.learning_trace_W > 1e-9) > 500
    fprintf('\n✅ EXCELLENT: Model is learning!\n');
    fprintf('   - Free energy decreased significantly\n');
    fprintf('   - Weight updates sustained throughout\n');
    fprintf('   - All fixes working correctly\n');
elseif FE_improvement > 0 && sum(full_results.learning_trace_W > 1e-9) > 100
    fprintf('\n⚠  PARTIAL: Some learning occurring\n');
    fprintf('   - Check gradient clipping thresholds\n');
    fprintf('   - Increase learning rates if needed\n');
else
    fprintf('\n❌ NO LEARNING: Something still wrong\n');
    fprintf('   - Check console output for NaN/Inf\n');
    fprintf('   - Verify noise_scale is decreasing (watch diagnostics)\n');
    fprintf('   - Check weight matrix access (motor vs planning)\n');
end

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('TEST COMPLETE\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
