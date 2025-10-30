%% 3D REACHING TRAJECTORY VISUALIZATION
% Load and visualize 3D sensorimotor reaching results
% Shows ground truth vs learned trajectories in 3D space with fading alpha

clear all; close all; clc;

% ====================================================================
% LOAD RESULTS
% ====================================================================

fprintf('Loading 3D reaching results...\n');

results_file = './figures/3D_reaching_results.mat';
if ~isfile(results_file)
    error('Results file not found: %s\nRun hierarchical_motion_inference_3D_EXACT first.', results_file);
end

load(results_file);
fprintf('✓ Loaded: %s\n\n', results_file);

% ====================================================================
% SETUP
% ====================================================================

n_trials = length(phases_indices);
colors = {'r', 'g', 'b', 'm'};
color_names = {'Red', 'Green', 'Blue', 'Magenta'};

fprintf('3D REACHING VISUALIZATION\n');
fprintf('Trials: %d | Total timesteps: %d\n\n', n_trials, length(x_true));

% ====================================================================
% CREATE FIGURE
% ====================================================================

fig = figure('Name', '3D Sensorimotor Reaching Trajectories', ...
    'Position', [100, 100, 1600, 900], 'Color', 'white');

% Main 3D plot
ax_main = subplot(1, 2, 1);
hold on;
axis equal;
grid on;
xlabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
zlabel('Z Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Ground Truth (Solid) vs Learned (Transparent Fading)', 'FontSize', 12, 'FontWeight', 'bold');
view(45, 30);
cameratoolbar('Show');

% Plot targets
for trial = 1:n_trials
    plot3(targets(trial,1), targets(trial,2), targets(trial,3), ...
        'o', 'Color', colors{trial}, 'MarkerSize', 15, 'MarkerFaceColor', colors{trial}, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', sprintf('T%d Target', trial));
end

% Plot start positions
for trial = 1:n_trials
    plot3(trial_start_positions(trial,1), trial_start_positions(trial,2), trial_start_positions(trial,3), ...
        's', 'Color', colors{trial}, 'MarkerSize', 12, 'MarkerFaceColor', colors{trial}, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', sprintf('T%d Start', trial));
end

% ====================================================================
% PLOT TRAJECTORIES WITH FADING ALPHA
% ====================================================================

fprintf('Plotting trajectories with fading transparency...\n\n');

% Ground truth trajectories (solid, opaque)
fprintf('Ground Truth (solid lines):\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    traj_x = x_true(trial_idx);
    traj_y = y_true(trial_idx);
    traj_z = z_true(trial_idx);
    
    n_pts = length(trial_idx);
    fprintf('  Trial %d: %d points\n', trial, n_pts);
    
    % Plot ground truth as solid lines with decreasing line width
    for i = 1:n_pts-1
        progress = i / n_pts;  % 0 to 1 over trajectory
        
        % Solid lines maintain full color but decrease in width
        lw = 3.5 * (1 - progress * 0.4);  % Width: 3.5 → 2.1
        
        line([traj_x(i), traj_x(i+1)], ...
             [traj_y(i), traj_y(i+1)], ...
             [traj_z(i), traj_z(i+1)], ...
             'Color', rgb(colors{trial}), 'LineWidth', lw, ...
             'LineStyle', '-', 'HandleVisibility', 'off');
    end
end

% Learned trajectories (dashed, fading alpha)
fprintf('\nLearned Predictions (dashed lines with fading):\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    traj_x = R_L1(trial_idx,1);
    traj_y = R_L1(trial_idx,2);
    traj_z = R_L1(trial_idx,3);
    
    n_pts = length(trial_idx);
    fprintf('  Trial %d: %d points\n', trial, n_pts);
    
    % Plot learned trajectories as dashed with fading alpha
    for i = 1:n_pts-1
        progress = i / n_pts;  % 0 to 1 over trajectory
        
        % Fading effect: opacity decreases from 1.0 to 0.2
        alpha_val = 1.0 - (progress * 0.8);  % Alpha: 1.0 → 0.2
        
        % Line width also decreases
        lw = 2.5 * (1 - progress * 0.4);  % Width: 2.5 → 1.5
        
        % Create dashed line with alpha via patch
        base_color = rgb(colors{trial});
        color_with_alpha = [base_color, alpha_val];  % [R, G, B, Alpha]
        
        line([traj_x(i), traj_x(i+1)], ...
             [traj_y(i), traj_y(i+1)], ...
             [traj_z(i), traj_z(i+1)], ...
             'Color', base_color, 'LineWidth', lw, ...
             'LineStyle', '--', 'Alpha', alpha_val, ...
             'HandleVisibility', 'off');
    end
end

legend('Location', 'best', 'FontSize', 10);

% ====================================================================
% SECONDARY VIEW: XY PLANE (Top-Down)
% ====================================================================

ax_xy = subplot(1, 2, 2);
hold on;
grid on;
xlabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Top-Down View (XY Plane)', 'FontSize', 12, 'FontWeight', 'bold');
axis equal;

% Plot targets and starts
for trial = 1:n_trials
    plot(targets(trial,1), targets(trial,2), 'o', 'Color', colors{trial}, ...
        'MarkerSize', 15, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    plot(trial_start_positions(trial,1), trial_start_positions(trial,2), 's', ...
        'Color', colors{trial}, 'MarkerSize', 12, 'MarkerFaceColor', colors{trial}, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
end

% Plot 2D trajectories
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    traj_x = x_true(trial_idx);
    traj_y = y_true(trial_idx);
    plot(traj_x, traj_y, '-', 'Color', rgb(colors{trial}), 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Trial %d (Truth)', trial));
    
    traj_x_pred = R_L1(trial_idx,1);
    traj_y_pred = R_L1(trial_idx,2);
    plot(traj_x_pred, traj_y_pred, '--', 'Color', rgb(colors{trial}), 'LineWidth', 2, ...
        'DisplayName', sprintf('Trial %d (Learned)', trial));
end

legend('Location', 'best', 'FontSize', 9);

sgtitle('3D Sensorimotor Reaching: Ground Truth vs Learned Predictions', ...
    'FontSize', 14, 'FontWeight', 'bold');

% ====================================================================
% ANALYSIS PANEL
% ====================================================================

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  3D TRAJECTORY ANALYSIS                                     ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Compute error statistics
pos_error = sqrt((x_true - R_L1(:,1)').^2 + (y_true - R_L1(:,2)').^2 + (z_true - R_L1(:,3)').^2);
vel_error = sqrt((vx_true - R_L1(:,4)').^2 + (vy_true - R_L1(:,5)').^2 + (vz_true - R_L1(:,6)').^2);

fprintf('POSITION ERROR STATISTICS:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('Overall Mean Position Error: %.4f m\n', mean(pos_error));
fprintf('Overall Max Position Error:  %.4f m\n', max(pos_error));
fprintf('Overall Min Position Error:  %.4f m\n\n', min(pos_error));

fprintf('By Trial:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_pos_error = pos_error(trial_idx);
    
    fprintf('  Trial %d (%s):\n', trial, color_names{trial});
    fprintf('    Start: [%.2f, %.2f, %.2f] → Target: [%.2f, %.2f, %.2f]\n', ...
        trial_start_positions(trial,1), trial_start_positions(trial,2), trial_start_positions(trial,3), ...
        targets(trial,1), targets(trial,2), targets(trial,3));
    fprintf('    Mean Error: %.4f m | Max Error: %.4f m | Min Error: %.4f m\n\n', ...
        mean(trial_pos_error), max(trial_pos_error), min(trial_pos_error));
end

fprintf('VELOCITY ERROR STATISTICS:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('Overall Mean Velocity Error: %.4f m/s\n', mean(vel_error));
fprintf('Overall Max Velocity Error:  %.4f m/s\n\n', max(vel_error));

fprintf('REACHING PERFORMANCE:\n');
fprintf('─────────────────────────────────────────────────────────\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    initial_dist = reaching_error_all(trial_idx(1));
    final_dist = reaching_error_all(trial_idx(end));
    improvement = initial_dist - final_dist;
    improvement_pct = (improvement / initial_dist) * 100;
    
    fprintf('  Trial %d: Initial distance = %.4f m, Final distance = %.4f m\n', ...
        trial, initial_dist, final_dist);
    fprintf('    Improvement: %.4f m (%.1f%%)\n\n', improvement, improvement_pct);
end

fprintf('FREE ENERGY:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('Initial Free Energy:  %.6e\n', free_energy_all(1));
fprintf('Final Free Energy:    %.6e\n', free_energy_all(end));
fprintf('Total Reduction:      %.6e (%.1f%%)\n\n', ...
    free_energy_all(1) - free_energy_all(end), ...
    ((free_energy_all(1) - free_energy_all(end)) / free_energy_all(1)) * 100);

% ====================================================================
% SAVE FIGURE
% ====================================================================

fprintf('Saving visualization...\n');

output_dir = './figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

try
    fig_filename = fullfile(output_dir, '3D_trajectories_interactive.png');
    saveas(fig, fig_filename, 'png');
    fprintf('✓ Figure saved: %s\n', fig_filename);
catch ME
    fprintf('Warning: Could not save figure: %s\n', ME.message);
end

try
    fig_filename_pdf = fullfile(output_dir, '3D_trajectories_interactive.pdf');
    saveas(fig, fig_filename_pdf, 'pdf');
    fprintf('✓ Figure saved: %s\n', fig_filename_pdf);
catch ME
    fprintf('Warning: Could not save PDF: %s\n', ME.message);
end

fprintf('\n✓ Visualization complete!\n');
fprintf('  Explore the 3D view using the mouse:\n');
fprintf('  - Left-click + drag: Rotate\n');
fprintf('  - Right-click + drag: Zoom\n');
fprintf('  - Middle-click + drag: Pan\n');

% ====================================================================
% HELPER FUNCTION: Color string to RGB
% ====================================================================

function rgb_val = rgb(color_str)
    % Convert color name to RGB vector
    switch color_str
        case 'r'
            rgb_val = [1, 0, 0];
        case 'g'
            rgb_val = [0, 0.7, 0];
        case 'b'
            rgb_val = [0, 0, 1];
        case 'm'
            rgb_val = [1, 0, 1];
        case 'c'
            rgb_val = [0, 1, 1];
        case 'y'
            rgb_val = [1, 1, 0];
        case 'k'
            rgb_val = [0, 0, 0];
        otherwise
            rgb_val = [0.5, 0.5, 0.5];
    end
end
