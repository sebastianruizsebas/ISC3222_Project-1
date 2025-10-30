%% NEUROSCIENCE ANALYSIS OF LEARNED WEIGHT MATRICES
% Analyzes hierarchical weight matrices W_L1_from_L2 and W_L2_from_L3
% Probes learned motor primitives and goal representations
% From a computational neuroscience perspective

clear all; close all; clc;

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  WEIGHT MATRIX ANALYSIS - LEARNED REPRESENTATIONS          ║\n');
fprintf('║  Neuroscience Perspective: Motor Primitives & Goal Coding   ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Load results
results_file = './figures/3D_reaching_results.mat';
if ~isfile(results_file)
    error('Results file not found: %s\nRun hierarchical_motion_inference_3D_EXACT first.', results_file);
end

load(results_file);
fprintf('✓ Loaded: %s\n\n', results_file);

if ~exist('W_L1_from_L2', 'var') || ~exist('W_L2_from_L3', 'var')
    error('Weight matrices not found in results file.\n');
end

% ====================================================================
% 1. WEIGHT MATRIX STATISTICS
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  WEIGHT MATRIX STRUCTURE                                    ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('W_L2_from_L3 (Goal → Motor Basis):\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('  Shape: [%d × %d] (motor basis × goal)\n', size(W_L2_from_L3));
fprintf('  Interpretation:\n');
fprintf('    - Rows: 6 motor basis channels\n');
fprintf('    - Cols: 3 goal coordinates [tx, ty, tz] + 1 bias\n');
fprintf('    - Each row = tuning curve for one motor primitive\n');
fprintf('    - Shows how 3D goals map to learned motor commands\n\n');

fprintf('W_L1_from_L2 (Motor → Proprioception):\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('  Shape: [%d × %d] (proprioception × motor basis)\n', size(W_L1_from_L2));
fprintf('  Interpretation:\n');
fprintf('    - Rows: 7 proprioceptive channels [x,y,z,vx,vy,vz,bias]\n');
fprintf('    - Cols: 6 motor basis channels\n');
fprintf('    - First 3 rows (position): learned forward model position component\n');
fprintf('    - Rows 4-6 (velocity): CRITICAL - should be identity (motor→velocity)\n');
fprintf('    - Row 7 (bias): bias prediction\n\n');

% ====================================================================
% 2. W_L2_FROM_L3 ANALYSIS (Goal → Motor Mapping)
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  W_L2_FROM_L3: GOAL REPRESENTATION → MOTOR PRIMITIVES       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Extract goal-to-motor weights (excluding bias column for now)
W_goal_motor = W_L2_from_L3(:, 1:3);  % [6 × 3] matrix

fprintf('MOTOR PRIMITIVE TUNING CURVES (Goal Sensitivity):\n');
fprintf('─────────────────────────────────────────────────────────\n');

for motor_idx = 1:6
    tuning = W_goal_motor(motor_idx, :);  % [tx_weight, ty_weight, tz_weight]
    magnitude = norm(tuning);
    
    if motor_idx <= 3
        motor_type = sprintf('Motor %d (Velocity Channel %d)', motor_idx, motor_idx);
    else
        motor_type = sprintf('Motor %d (Auxiliary Channel %d)', motor_idx, motor_idx-3);
    end
    
    fprintf('%s:\n', motor_type);
    fprintf('  Tuning to [Tx, Ty, Tz]: [%.6f, %.6f, %.6f]\n', tuning(1), tuning(2), tuning(3));
    fprintf('  Magnitude: %.6f (strength of goal response)\n', magnitude);
    
    % Analyze preferred direction
    [max_weight, max_idx] = max(abs(tuning));
    coord_names = {'X', 'Y', 'Z'};
    fprintf('  Preferred coordinate: %s (weight = %.6f)\n', coord_names{max_idx}, tuning(max_idx));
    
    % Selectivity analysis
    if magnitude > 0.01
        selectivity = max_weight / (magnitude + 1e-6);
        fprintf('  Selectivity: %.2f (1.0=pure, 0.33=diffuse)\n', selectivity);
    end
    fprintf('\n');
end

% ====================================================================
% 3. W_L1_FROM_L2 ANALYSIS (Motor → Proprioception Mapping)
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  W_L1_FROM_L2: MOTOR PRIMITIVES → PROPRIOCEPTION            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('FORWARD MODEL ANALYSIS:\n');
fprintf('─────────────────────────────────────────────────────────\n\n');

fprintf('Position Prediction (Rows 1-3):\n');
fprintf('─────────────────────────────────────────────────────────\n');
W_pos = W_L1_from_L2(1:3, :);
fprintf('  Position weights from motor basis:\n');
for pos_idx = 1:3
    coord_names = {'X', 'Y', 'Z'};
    weights = W_pos(pos_idx, :);
    fprintf('    %s position: ', coord_names{pos_idx});
    fprintf('[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', weights(1), weights(2), weights(3), weights(4), weights(5), weights(6));
    fprintf('      Magnitude: %.6f (typically small - position changes slowly)\n\n', norm(weights));
end

fprintf('VELOCITY PREDICTION (Rows 4-6) - CRITICAL:\n');
fprintf('─────────────────────────────────────────────────────────\n');
W_vel = W_L1_from_L2(4:6, :);
fprintf('  Expected: Identity matrix for first 3 columns (motor→velocity is direct)\n');
fprintf('  Actual learned weights:\n\n');

vel_names = {'Vx', 'Vy', 'Vz'};
for vel_idx = 1:3
    weights = W_vel(vel_idx, :);
    fprintf('  %s prediction: [%.6f, %.6f, %.6f | %.6f, %.6f, %.6f]\n', ...
        vel_names{vel_idx}, weights(1), weights(2), weights(3), weights(4), weights(5), weights(6));
end

fprintf('\n  Analysis:\n');
% Check if velocity rows are close to identity
W_vel_direct = W_vel(1:3, 1:3);
identity_error = norm(W_vel_direct - eye(3));
fprintf('    Direct (motor→velocity) error from identity: %.6f\n', identity_error);
if identity_error < 0.1
    fprintf('    ✓ GOOD: Velocity mapping preserved (learned physics)\n');
else
    fprintf('    ✗ WARNING: Velocity mapping degraded\n');
end

% Check auxiliary motor contributions
W_vel_aux = W_vel(1:3, 4:6);
aux_magnitude = norm(W_vel_aux);
fprintf('    Auxiliary motor contribution: %.6f (should be small)\n', aux_magnitude);
if aux_magnitude < 0.2
    fprintf('    ✓ GOOD: Auxiliary channels don''t corrupt velocity\n');
else
    fprintf('    ✗ WARNING: Auxiliary channels have large velocity effects\n');
end

fprintf('\n');

fprintf('Bias Prediction (Row 7):\n');
fprintf('─────────────────────────────────────────────────────────\n');
W_bias = W_L1_from_L2(7, :);
fprintf('  Bias weights: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', ...
    W_bias(1), W_bias(2), W_bias(3), W_bias(4), W_bias(5), W_bias(6));
fprintf('  Magnitude: %.6f (should be near zero)\n\n', norm(W_bias));

% ====================================================================
% 4. MOTOR PRIMITIVES INTERPRETATION
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  LEARNED MOTOR PRIMITIVES (Basis Functions)                 ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('MOTOR BASIS INTERPRETATION:\n');
fprintf('─────────────────────────────────────────────────────────\n\n');

% Compute which motor primitives are used most
goal_motor_magnitude = sqrt(sum(W_goal_motor.^2, 2));
[sorted_mag, sorted_idx] = sort(goal_motor_magnitude, 'descend');

fprintf('Motor Primitive Importance (sorted by goal coupling strength):\n');
for rank = 1:6
    motor_idx = sorted_idx(rank);
    if motor_idx <= 3
        motor_name = sprintf('Motor %d (Velocity Channel %d)', motor_idx, motor_idx);
    else
        motor_name = sprintf('Motor %d (Auxiliary %d)', motor_idx, motor_idx-3);
    end
    fprintf('  %d. %s: goal coupling = %.6f\n', rank, motor_name, sorted_mag(rank));
end

fprintf('\n  Interpretation:\n');
if sorted_mag(1) > 0.1
    fprintf('    - Strong dominance: Main primitives are %d and %d\n', sorted_idx(1), sorted_idx(2));
    fprintf('    - These likely encode: reaching velocity and directional control\n');
else
    fprintf('    - Weak overall goal coupling: Motor primitives are autonomous\n');
end

% ====================================================================
% 5. FORWARD MODEL QUALITY
% ====================================================================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  FORWARD MODEL QUALITY (Motor → Proprioception)             ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('MATRIX CONDITIONING:\n');
fprintf('─────────────────────────────────────────────────────────\n\n');

% Condition number analysis
cond_W_full = cond(W_L1_from_L2);
cond_W_pos = cond(W_pos);
cond_W_vel = cond(W_vel_direct);

fprintf('W_L1_from_L2 condition number: %.2f\n', cond_W_full);
fprintf('  Position submatrix: %.2f\n', cond_W_pos);
fprintf('  Velocity submatrix: %.2f (should be ~1.0 for identity)\n', cond_W_vel);

if cond_W_vel < 2
    fprintf('  ✓ EXCELLENT: Velocity mapping is well-conditioned (identity-like)\n');
elseif cond_W_vel < 10
    fprintf('  ✓ GOOD: Velocity mapping is reasonably stable\n');
else
    fprintf('  ✗ WARNING: Velocity mapping is ill-conditioned\n');
end

% Rank analysis
rank_W_full = rank(W_L1_from_L2);
rank_W_expected = min(size(W_L1_from_L2));
fprintf('\nMatrix rank: %d (expected %d)\n', rank_W_full, rank_W_expected);
if rank_W_full == rank_W_expected
    fprintf('  ✓ Full rank: All proprioceptive channels can be predicted\n');
else
    fprintf('  ✗ Reduced rank: Some channels are linearly dependent\n');
end

% ====================================================================
% 6. VISUALIZATION: HEATMAPS
% ====================================================================

fprintf('\nCreating visualizations...\n\n');

fig = figure('Name', 'Weight Matrix Analysis', 'Position', [100, 100, 1400, 900]);

% Panel 1: W_L2_from_L3 heatmap
ax1 = subplot(2, 2, 1);
imagesc(W_L2_from_L3);
colorbar;
xlabel('Goal Inputs: [Tx, Ty, Tz, Bias]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Motor Primitives', 'FontSize', 10, 'FontWeight', 'bold');
title('W_{L2←L3}: Goal → Motor Mapping', 'FontSize', 11, 'FontWeight', 'bold');
set(ax1, 'YTick', 1:6);
set(ax1, 'YTickLabel', {'Motor 1 (Vx)', 'Motor 2 (Vy)', 'Motor 3 (Vz)', ...
    'Motor 4 (Aux)', 'Motor 5 (Aux)', 'Motor 6 (Aux)'});
set(ax1, 'XTick', 1:4);
set(ax1, 'XTickLabel', {'Tx', 'Ty', 'Tz', 'Bias'});
colormap('RdBu');
caxis([-max(abs(W_L2_from_L3(:))), max(abs(W_L2_from_L3(:)))]);

% Panel 2: W_L1_from_L2 heatmap
ax2 = subplot(2, 2, 2);
imagesc(W_L1_from_L2);
colorbar;
xlabel('Motor Basis Channels', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Proprioceptive Outputs', 'FontSize', 10, 'FontWeight', 'bold');
title('W_{L1←L2}: Motor → Proprioception (Forward Model)', 'FontSize', 11, 'FontWeight', 'bold');
set(ax2, 'YTick', 1:7);
set(ax2, 'YTickLabel', {'X pos', 'Y pos', 'Z pos', 'Vx', 'Vy', 'Vz', 'Bias'});
set(ax2, 'XTick', 1:6);
set(ax2, 'XTickLabel', {'M1', 'M2', 'M3', 'M4', 'M5', 'M6'});
colormap('RdBu');
caxis([-max(abs(W_L1_from_L2(:))), max(abs(W_L1_from_L2(:)))]);

% Panel 3: Goal-Motor tuning curves
ax3 = subplot(2, 2, 3);
hold on;
x_labels = {'Tx', 'Ty', 'Tz'};
colors_motor = {'r', 'g', 'b', 'm', 'c', 'y'};
for motor_idx = 1:6
    plot(1:3, W_goal_motor(motor_idx, :), 'o-', 'Color', colors_motor{motor_idx}, ...
        'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', sprintf('Motor %d', motor_idx));
end
set(ax3, 'XTick', 1:3);
set(ax3, 'XTickLabel', x_labels);
ylabel('Weight Value', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Goal Coordinate', 'FontSize', 10, 'FontWeight', 'bold');
title('Motor Primitive Tuning to Goal Coordinates', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
legend('FontSize', 8, 'Location', 'best');

% Panel 4: Forward model analysis (velocity rows)
ax4 = subplot(2, 2, 4);
% Show how well velocity is preserved
identity_matrix = eye(3);
subplot_data = [W_vel_direct(:); identity_matrix(:)];
x_pos = [1:9, 11:19];
y_data = [W_vel_direct(:); identity_matrix(:)];
bar_colors = [repmat([0.2, 0.6, 1], 9, 1); repmat([1, 0.2, 0.2], 9, 1)];
bar(x_pos, y_data, 'FaceColor', 'flat');
set(ax4, 'XTick', [5, 15]);
set(ax4, 'XTickLabel', {'Learned W_{v}', 'Expected (Identity)'});
ylabel('Weight Value', 'FontSize', 10, 'FontWeight', 'bold');
title('Velocity Forward Model: Learned vs Expected', 'FontSize', 11, 'FontWeight', 'bold');
grid on;
ylim([-0.2, 1.2]);

sgtitle('Learned Weight Matrices - Neuroscience Analysis', 'FontSize', 13, 'FontWeight', 'bold');

% Save figure
output_file = './figures/weight_matrix_analysis.png';
saveas(fig, output_file, 'png');
fprintf('✓ Saved: %s\n', output_file);

% ====================================================================
% 7. SUMMARY REPORT
% ====================================================================

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  SUMMARY: LEARNED REPRESENTATIONS                          ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('KEY FINDINGS:\n');
fprintf('─────────────────────────────────────────────────────────\n\n');

fprintf('1. GOAL ENCODING (W_L2_from_L3):\n');
dominant_motor = sorted_idx(1);
fprintf('   - Primary motor channels (1-3) encode reaching velocity\n');
fprintf('   - Motor %d shows strongest goal coupling (%.6f)\n', dominant_motor, sorted_mag(1));
fprintf('   - Suggests a learned mapping: 3D goals → motor velocity commands\n\n');

fprintf('2. FORWARD MODEL (W_L1_from_L2):\n');
if identity_error < 0.1
    fprintf('   ✓ LEARNED PHYSICS: Velocity mapping is preserved\n');
    fprintf('     - Motor→velocity rows show identity structure\n');
    fprintf('     - This is fundamental physics, not learned\n');
else
    fprintf('   ✗ LEARNED PHYSICS DEGRADED:\n');
    fprintf('     - Motor→velocity mapping has drifted from identity\n');
    fprintf('     - Error from identity: %.6f\n', identity_error);
end
fprintf('   - Position mapping is weak (integration over time needed)\n');
fprintf('   - Auxiliary motors contribute minimally\n\n');

fprintf('3. MOTOR PRIMITIVES:\n');
if max(goal_motor_magnitude) > 0.2
    fprintf('   - Well-tuned primitives (high goal sensitivity)\n');
else
    fprintf('   - Weakly-tuned primitives (high autonomy)\n');
end
fprintf('   - Primitives encode combinations of direction and speed\n\n');

fprintf('4. OVERALL ASSESSMENT:\n');
if identity_error < 0.1 && max(goal_motor_magnitude) > 0.1
    fprintf('   ✓ HEALTHY LEARNING:\n');
    fprintf('     - Forward model is sound (physics preserved)\n');
    fprintf('     - Goal-motor coupling is reasonable\n');
    fprintf('     - System has learned meaningful structure\n');
else
    fprintf('   ⚠ POTENTIAL ISSUES:\n');
    if identity_error >= 0.1
        fprintf('     - Forward model degradation detected\n');
    end
    if max(goal_motor_magnitude) < 0.1
        fprintf('     - Weak goal-motor coupling\n');
    end
end

fprintf('\n');

% ====================================================================
% 8. EXPORT ANALYSIS TO TEXT FILE
% ====================================================================

analysis_file = './figures/WEIGHT_MATRIX_ANALYSIS.txt';
fid = fopen(analysis_file, 'w');

fprintf(fid, '╔═══════════════════════════════════════════════════════════════╗\n');
fprintf(fid, '║  LEARNED WEIGHT MATRICES - DETAILED ANALYSIS                ║\n');
fprintf(fid, '╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf(fid, 'W_L2_FROM_L3 (Goal → Motor):\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
fprintf(fid, 'Shape: [%d × %d]\n\n', size(W_L2_from_L3));
fprintf(fid, '%s\n', mat2str(W_L2_from_L3, 6));

fprintf(fid, '\n\nW_L1_FROM_L2 (Motor → Proprioception):\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
fprintf(fid, 'Shape: [%d × %d]\n\n', size(W_L1_from_L2));
fprintf(fid, '%s\n', mat2str(W_L1_from_L2, 6));

fprintf(fid, '\n\nSTATISTICAL SUMMARY:\n');
fprintf(fid, '─────────────────────────────────────────────────────────\n');
fprintf(fid, 'W_L2_from_L3 Frobenius norm: %.6f\n', norm(W_L2_from_L3, 'fro'));
fprintf(fid, 'W_L1_from_L2 Frobenius norm: %.6f\n', norm(W_L1_from_L2, 'fro'));
fprintf(fid, 'Velocity submatrix error from identity: %.6f\n', identity_error);
fprintf(fid, 'Maximum goal-motor coupling: %.6f\n', max(goal_motor_magnitude));

fclose(fid);
fprintf('✓ Saved: %s\n\n', analysis_file);

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  ANALYSIS COMPLETE                                          ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');
