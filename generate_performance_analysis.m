%% COMPREHENSIVE PERFORMANCE ANALYSIS
% Analyzes the latest 3D reaching model run
% Generates detailed metrics and visualizations

clear all; close all; clc;

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  PERFORMANCE ANALYSIS - 3D SENSORIMOTOR REACHING            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Load results
results_file = './figures/3D_reaching_results.mat';
if ~isfile(results_file)
    error('Results file not found: %s', results_file);
end

load(results_file);
fprintf('✓ Loaded: %s\n\n', results_file);

n_trials = length(phases_indices);
dt = 0.01;

% ====================================================================
% 1. REACHING PERFORMANCE METRICS
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  REACHING PERFORMANCE                                       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('FINAL REACHING DISTANCE (Ground Truth):\n');
fprintf('─────────────────────────────────────────────────────────\n');

trial_reaching_improvement = [];
trial_learning_rates = [];
trial_learning_efficiency = [];

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Initial and final reaching distances
    initial_dist = reaching_error_all(trial_idx(1));
    final_dist = reaching_error_all(trial_idx(end));
    
    improvement = initial_dist - final_dist;
    improvement_pct = (improvement / initial_dist) * 100;
    
    % Learning rate (improvement per thousand steps)
    trial_steps = length(trial_idx);
    learning_rate = (initial_dist - final_dist) / (trial_steps / 1000);
    
    % Learning efficiency (improvement per unit free energy)
    initial_fe = free_energy_all(trial_idx(1));
    final_fe = free_energy_all(trial_idx(end));
    fe_reduction = initial_fe - final_fe;
    if fe_reduction > 0
        efficiency = improvement / fe_reduction;
    else
        efficiency = 0;
    end
    
    trial_reaching_improvement = [trial_reaching_improvement; improvement_pct];
    trial_learning_rates = [trial_learning_rates; learning_rate];
    trial_learning_efficiency = [trial_learning_efficiency; efficiency];
    
    fprintf('Trial %d: [%.2f m → %.2f m] = %.2f m improvement (%.1f%%)\n', ...
        trial, initial_dist, final_dist, improvement, improvement_pct);
    fprintf('  Learning Rate: %.4f m per 1k steps\n', learning_rate);
    fprintf('  Learning Efficiency: %.6f m/FE\n\n', efficiency);
end

overall_initial = reaching_error_all(1);
overall_final = reaching_error_all(end);
overall_improvement = overall_initial - overall_final;
overall_improvement_pct = (overall_improvement / overall_initial) * 100;

fprintf('OVERALL:\n');
fprintf('  Initial: %.2f m → Final: %.2f m\n', overall_initial, overall_final);
fprintf('  Total Improvement: %.2f m (%.1f%%)\n', overall_improvement, overall_improvement_pct);
fprintf('  Average Per Trial: %.1f%%\n\n', mean(trial_reaching_improvement));

% ====================================================================
% 2. PREDICTION ERROR ANALYSIS
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  PREDICTION ERROR ANALYSIS                                  ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Position errors
pos_error_all = sqrt((x_true - R_L1(:,1)').^2 + ...
                     (y_true - R_L1(:,2)').^2 + ...
                     (z_true - R_L1(:,3)').^2);

% Velocity errors
vel_error_all = sqrt((vx_true - R_L1(:,4)').^2 + ...
                     (vy_true - R_L1(:,5)').^2 + ...
                     (vz_true - R_L1(:,6)').^2);

fprintf('POSITION PREDICTION ERRORS:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('  Overall RMSE: %.4f m\n', sqrt(mean(pos_error_all.^2)));
fprintf('  Overall MAE:  %.4f m\n', mean(pos_error_all));
fprintf('  Max Error:    %.4f m\n\n', max(pos_error_all));

fprintf('VELOCITY PREDICTION ERRORS:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('  Overall RMSE: %.4f m/s\n', sqrt(mean(vel_error_all.^2)));
fprintf('  Overall MAE:  %.4f m/s\n', mean(vel_error_all));
fprintf('  Max Error:    %.4f m/s\n\n', max(vel_error_all));

% Per-trial analysis
fprintf('BY TRIAL:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    trial_pos_error = pos_error_all(trial_idx);
    trial_vel_error = vel_error_all(trial_idx);
    
    fprintf('Trial %d:\n', trial);
    fprintf('  Position - RMSE: %.4f m, MAE: %.4f m, Max: %.4f m\n', ...
        sqrt(mean(trial_pos_error.^2)), mean(trial_pos_error), max(trial_pos_error));
    fprintf('  Velocity - RMSE: %.4f m/s, MAE: %.4f m/s, Max: %.4f m/s\n\n', ...
        sqrt(mean(trial_vel_error.^2)), mean(trial_vel_error), max(trial_vel_error));
end

% ====================================================================
% 3. FREE ENERGY ANALYSIS
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  FREE ENERGY DYNAMICS                                       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('FREE ENERGY TRAJECTORY:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('  Initial FE: %.6e\n', free_energy_all(1));
fprintf('  Final FE:   %.6e\n', free_energy_all(end));
fprintf('  Reduction:  %.6e (%.1f%%)\n', ...
    free_energy_all(1) - free_energy_all(end), ...
    100 * (1 - free_energy_all(end) / free_energy_all(1)));

% FE reduction rate
fe_reduction_rate = (free_energy_all(end) - free_energy_all(1)) / length(free_energy_all);
fprintf('  Reduction Rate: %.6e per step\n\n', fe_reduction_rate);

% Per-trial FE
fprintf('BY TRIAL:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    trial_fe_start = free_energy_all(trial_idx(1));
    trial_fe_end = free_energy_all(trial_idx(end));
    trial_fe_reduction = trial_fe_start - trial_fe_end;
    
    if trial_fe_reduction > 0
        trial_fe_pct = 100 * trial_fe_reduction / trial_fe_start;
    else
        trial_fe_pct = 0;
    end
    
    fprintf('Trial %d: %.6e → %.6e (%.1f%% reduction)\n', ...
        trial, trial_fe_start, trial_fe_end, trial_fe_pct);
end

fprintf('\n');

% ====================================================================
% 4. MOTOR COMMAND ANALYSIS
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  MOTOR COMMAND ANALYSIS                                     ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Reconstruct motor commands from velocity integration
% motor command = (v_true(i+1) - damping*v_true(i)) / motor_gain
motor_gain = 0.3;  % From model (should match hierarchical_motion_inference_3D_EXACT.m)
damping = 0.9;     % From model

% Reconstruct actual executed velocity commands
motor_vx = (vx_true(2:end) - damping * vx_true(1:end-1)) / motor_gain;
motor_vy = (vy_true(2:end) - damping * vy_true(1:end-1)) / motor_gain;
motor_vz = (vz_true(2:end) - damping * vz_true(1:end-1)) / motor_gain;

motor_speed_all = sqrt(motor_vx.^2 + motor_vy.^2 + motor_vz.^2);

fprintf('MOTOR COMMAND STATISTICS (Reconstructed):\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('  Speed: Min=%.4f, Mean=%.4f, Max=%.4f m/s\n', ...
    min(motor_speed_all), mean(motor_speed_all), max(motor_speed_all));
fprintf('  X-axis: Min=%.4f, Mean=%.4f, Max=%.4f m/s\n', ...
    min(motor_vx), mean(motor_vx), max(motor_vx));
fprintf('  Y-axis: Min=%.4f, Mean=%.4f, Max=%.4f m/s\n', ...
    min(motor_vy), mean(motor_vy), max(motor_vy));
fprintf('  Z-axis: Min=%.4f, Mean=%.4f, Max=%.4f m/s\n\n', ...
    min(motor_vz), mean(motor_vz), max(motor_vz));

% ====================================================================
% 5. LEARNING DYNAMICS
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  LEARNING DYNAMICS                                          ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('NOTE: Weight matrices not saved in results file\n');
fprintf('(Would need to be saved from hierarchical_motion_inference_3D_EXACT.m)\n\n');

% Layer representations
fprintf('REPRESENTATION STATISTICS (Final State):\n');
fprintf('─────────────────────────────────────────────────────────\n');

if exist('R_L1', 'var')
    fprintf('  L1 (Proprioception) final state:\n');
    fprintf('    Position [x,y,z]: [%.4f, %.4f, %.4f]\n', R_L1(end,1), R_L1(end,2), R_L1(end,3));
    fprintf('    Velocity [vx,vy,vz]: [%.4f, %.4f, %.4f] m/s\n\n', R_L1(end,4), R_L1(end,5), R_L1(end,6));
end

if exist('R_L2', 'var')
    fprintf('  L2 (Motor Basis) final state:\n');
    if size(R_L2, 2) >= 6
        fprintf('    Motor commands [v1,v2,v3]: [%.4f, %.4f, %.4f]\n', R_L2(end,1), R_L2(end,2), R_L2(end,3));
        fprintf('    Auxiliary [a1,a2,a3]: [%.4f, %.4f, %.4f]\n\n', R_L2(end,4), R_L2(end,5), R_L2(end,6));
    else
        fprintf('    Motor basis: %s\n\n', mat2str(R_L2(end,:)));
    end
end

if exist('R_L3', 'var')
    fprintf('  L3 (Goal) final state:\n');
    if size(R_L3, 2) >= 4
        fprintf('    Target [tx,ty,tz]: [%.4f, %.4f, %.4f]\n', R_L3(end,1), R_L3(end,2), R_L3(end,3));
        fprintf('    Bias: %.4f\n\n', R_L3(end,4));
    else
        fprintf('    Goal representation: %s\n\n', mat2str(R_L3(end,:)));
    end
end

% ====================================================================
% 6. SUMMARY STATISTICS TABLE
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  SUMMARY TABLE                                              ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

summary_table = table(...
    (1:n_trials)', ...
    trial_reaching_improvement, ...
    trial_learning_rates, ...
    trial_learning_efficiency, ...
    'VariableNames', {'Trial', 'Improvement_%', 'LearningRate_m/1ksteps', 'Efficiency_m/FE'});

disp(summary_table);

fprintf('\n');

% ====================================================================
% 7. VISUALIZATION: MULTI-PANEL ANALYSIS
% ====================================================================

fprintf('Creating visualization...\n');

fig = figure('Name', 'Performance Analysis', 'Position', [100, 100, 1400, 900]);

% Panel 1: Reaching distance over time (by trial)
ax1 = subplot(2, 3, 1);
hold on;
colors = {'r', 'g', 'b', 'm'};
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_time = (trial_idx - 1) * dt;
    plot(trial_time, reaching_error_all(trial_idx), '-', 'Color', colors{trial}, 'LineWidth', 2);
end
xlabel('Time (s)');
ylabel('Reaching Distance (m)');
title('Reaching Distance Over Time');
grid on;
legend(arrayfun(@(t) sprintf('Trial %d', t), 1:n_trials, 'UniformOutput', false));

% Panel 2: Free energy dynamics
ax2 = subplot(2, 3, 2);
semilogy(free_energy_all, 'k-', 'LineWidth', 1.5);
xlabel('Step');
ylabel('Free Energy');
title('Free Energy Trajectory (log scale)');
grid on;

% Panel 3: Position prediction error
ax3 = subplot(2, 3, 3);
plot(pos_error_all, 'b-', 'LineWidth', 0.5);
hold on;
% Add trial boundaries
for trial = 2:n_trials
    xline(phases_indices{trial}(1), 'r--', 'Alpha', 0.5);
end
xlabel('Step');
ylabel('Position Error (m)');
title('Position Prediction Error');
grid on;

% Panel 4: Velocity prediction error
ax4 = subplot(2, 3, 4);
plot(vel_error_all, 'c-', 'LineWidth', 0.5);
hold on;
for trial = 2:n_trials
    xline(phases_indices{trial}(1), 'r--', 'Alpha', 0.5);
end
xlabel('Step');
ylabel('Velocity Error (m/s)');
title('Velocity Prediction Error');
grid on;

% Panel 5: Learning improvement per trial
ax5 = subplot(2, 3, 5);
bar(1:n_trials, trial_reaching_improvement, 'FaceColor', [0.2, 0.7, 0.9]);
ylabel('Improvement (%)');
xlabel('Trial');
title('Reaching Distance Improvement per Trial');
grid on;
ylim([0, max(trial_reaching_improvement) * 1.1]);

% Panel 6: Learning efficiency
ax6 = subplot(2, 3, 6);
bar(1:n_trials, trial_learning_efficiency, 'FaceColor', [0.9, 0.7, 0.2]);
ylabel('Efficiency (m/FE)');
xlabel('Trial');
title('Learning Efficiency per Trial');
grid on;

sgtitle('3D Sensorimotor Reaching - Performance Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
output_file = './figures/performance_analysis.png';
saveas(fig, output_file, 'png');
fprintf('✓ Saved: %s\n', output_file);

% ====================================================================
% 8. EXPORT SUMMARY TO TEXT FILE
% ====================================================================

summary_file = './figures/PERFORMANCE_ANALYSIS_SUMMARY.txt';
fid = fopen(summary_file, 'w');

fprintf(fid, '╔═══════════════════════════════════════════════════════════════╗\n');
fprintf(fid, '║  PERFORMANCE ANALYSIS SUMMARY - 3D SENSORIMOTOR REACHING    ║\n');
fprintf(fid, '╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf(fid, 'REACHING PERFORMANCE:\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
fprintf(fid, 'Overall improvement: %.2f m (%.1f%%)\n', overall_improvement, overall_improvement_pct);
fprintf(fid, 'Average per trial: %.1f%%\n\n', mean(trial_reaching_improvement));

fprintf(fid, 'PREDICTION ERRORS:\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
fprintf(fid, 'Position RMSE: %.4f m\n', sqrt(mean(pos_error_all.^2)));
fprintf(fid, 'Velocity RMSE: %.4f m/s\n\n', sqrt(mean(vel_error_all.^2)));

fprintf(fid, 'FREE ENERGY:\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
fprintf(fid, 'Initial: %.6e\n', free_energy_all(1));
fprintf(fid, 'Final:   %.6e\n', free_energy_all(end));
fprintf(fid, 'Reduction: %.1f%%\n\n', 100 * (1 - free_energy_all(end) / free_energy_all(1)));

fprintf(fid, 'LEARNING EFFICIENCY:\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
for trial = 1:n_trials
    fprintf(fid, 'Trial %d: %.1f%% improvement, efficiency %.6f m/FE\n', ...
        trial, trial_reaching_improvement(trial), trial_learning_efficiency(trial));
end

fclose(fid);
fprintf('✓ Saved: %s\n\n', summary_file);

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  ANALYSIS COMPLETE                                          ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');
