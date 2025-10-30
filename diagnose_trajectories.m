%% DIAGNOSTIC: Check if trajectories are actually different

clear all; close all; clc;

% Load results
results_file = './figures/3D_reaching_results.mat';
if ~isfile(results_file)
    error('Results file not found: %s', results_file);
end

load(results_file);

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  TRAJECTORY DIAGNOSTIC                                      ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

n_trials = length(phases_indices);

% Check ground truth trajectories
fprintf('GROUND TRUTH TRAJECTORIES:\n');
fprintf('─────────────────────────────────────────────────────────\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    x_range = [min(x_true(trial_idx)), max(x_true(trial_idx))];
    y_range = [min(y_true(trial_idx)), max(y_true(trial_idx))];
    z_range = [min(z_true(trial_idx)), max(z_true(trial_idx))];
    
    x_motion = x_range(2) - x_range(1);
    y_motion = y_range(2) - y_range(1);
    z_motion = z_range(2) - z_range(1);
    
    fprintf('Trial %d:\n', trial);
    fprintf('  X range: [%.4f, %.4f] (motion: %.4f m)\n', x_range(1), x_range(2), x_motion);
    fprintf('  Y range: [%.4f, %.4f] (motion: %.4f m)\n', y_range(1), y_range(2), y_motion);
    fprintf('  Z range: [%.4f, %.4f] (motion: %.4f m)\n', z_range(1), z_range(2), z_motion);
    fprintf('  Total motion: %.4f m\n\n', sqrt(x_motion^2 + y_motion^2 + z_motion^2));
end

% Check learned trajectories
fprintf('LEARNED TRAJECTORIES (from R_L1):\n');
fprintf('─────────────────────────────────────────────────────────\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    x_pred = R_L1(trial_idx, 1);
    y_pred = R_L1(trial_idx, 2);
    z_pred = R_L1(trial_idx, 3);
    
    x_range = [min(x_pred), max(x_pred)];
    y_range = [min(y_pred), max(y_pred)];
    z_range = [min(z_pred), max(z_pred)];
    
    x_motion = x_range(2) - x_range(1);
    y_motion = y_range(2) - y_range(1);
    z_motion = z_range(2) - z_range(1);
    
    fprintf('Trial %d:\n', trial);
    fprintf('  X range: [%.4f, %.4f] (motion: %.4f m)\n', x_range(1), x_range(2), x_motion);
    fprintf('  Y range: [%.4f, %.4f] (motion: %.4f m)\n', y_range(1), y_range(2), y_motion);
    fprintf('  Z range: [%.4f, %.4f] (motion: %.4f m)\n', z_range(1), z_range(2), z_motion);
    fprintf('  Total motion: %.4f m\n\n', sqrt(x_motion^2 + y_motion^2 + z_motion^2));
    
    % Show first and last positions
    fprintf('  First pos:  [%.4f, %.4f, %.4f]\n', x_pred(1), y_pred(1), z_pred(1));
    fprintf('  Last pos:   [%.4f, %.4f, %.4f]\n\n', x_pred(end), y_pred(end), z_pred(end));
end

% Check position error
fprintf('POSITION ERROR STATISTICS:\n');
fprintf('─────────────────────────────────────────────────────────\n');
pos_error = sqrt((x_true - R_L1(:,1)').^2 + (y_true - R_L1(:,2)').^2 + (z_true - R_L1(:,3)').^2);

fprintf('Overall statistics:\n');
fprintf('  Mean error: %.6f m\n', mean(pos_error));
fprintf('  Max error:  %.6f m\n', max(pos_error));
fprintf('  Min error:  %.6f m\n\n', min(pos_error));

% Check if learned positions are clamped to initial position
fprintf('ARE LEARNED POSITIONS STUCK AT INITIAL POSITION?\n');
fprintf('─────────────────────────────────────────────────────────\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    x_pred = R_L1(trial_idx, 1);
    
    % Check how many points are different from first point
    unique_x = length(unique(round(x_pred, 6)));  % Round to avoid floating point issues
    total_pts = length(x_pred);
    
    fprintf('Trial %d: %d unique X positions out of %d points (%.1f%% variation)\n', ...
        trial, unique_x, total_pts, (unique_x/total_pts)*100);
    
    if unique_x < 10
        fprintf('  WARNING: Very little variation in learned trajectory!\n');
    end
end

fprintf('\n');

% Create simple plot to visualize
fig = figure('Name', 'Trajectory Diagnostic', 'Position', [100, 100, 1200, 400]);

colors = {'r', 'g', 'b', 'm'};

% 3D view
subplot(1, 2, 1);
hold on;
grid on;
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Trajectories (Ground Truth)');

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    plot3(x_true(trial_idx), y_true(trial_idx), z_true(trial_idx), ...
        'Color', colors{trial}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Trial %d', trial));
end
legend;
view(45, 30);

% 2D XY view
subplot(1, 2, 2);
hold on;
grid on;
axis equal;
xlabel('X'); ylabel('Y');
title('2D XY Trajectories Comparison');

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Ground truth
    plot(x_true(trial_idx), y_true(trial_idx), '-', ...
        'Color', colors{trial}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('T%d Truth', trial));
    
    % Learned
    plot(R_L1(trial_idx,1), R_L1(trial_idx,2), '--', ...
        'Color', colors{trial}, 'LineWidth', 2, ...
        'DisplayName', sprintf('T%d Learned', trial));
end
legend('FontSize', 8);

saveas(fig, './figures/trajectory_diagnostic.png');
fprintf('✓ Diagnostic plot saved to: ./figures/trajectory_diagnostic.png\n');

