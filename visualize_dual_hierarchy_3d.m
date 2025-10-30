%% 3D DUAL-HIERARCHY VISUALIZATION: PLAYER CHASING MOVING BALL
% Load and visualize dual-hierarchy learning results
% Shows player vs ball trajectories in 3D space with trial-based color coding

clear all; close all; clc;

% Ensure graphics are enabled (override batch mode settings)
set(0, 'DefaultFigureVisible', 'on');
set(groot, 'defaultFigureCreateFcn', '');

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  DUAL-HIERARCHY VISUALIZATION: PLAYER CHASING BALL         ║\n');
fprintf('║  Motor Region (Stable) + Planning Region (Task-Specific)  ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

results_file = './figures/3D_dual_hierarchy_results.mat';
if ~isfile(results_file)
    error('Results file not found: %s\nRun hierarchical_motion_inference_dual_hierarchy() first.', results_file);
end

load(results_file);
fprintf('✓ Loaded: %s\n\n', results_file);

% ====================================================================
% SETUP
% ====================================================================

n_trials = length(phases_indices);
colors = {'r', 'g', 'b', 'm'};
color_names = {'Red', 'Green', 'Blue', 'Magenta'};
rgb_colors = {[1, 0, 0], [0, 0.7, 0], [0, 0, 1], [1, 0, 1]};

fprintf('DUAL-HIERARCHY VISUALIZATION\n');
fprintf('Trials: %d | Total timesteps: %d\n\n', n_trials, length(x_player));

% ====================================================================
% COMPUTE INTERCEPTION METRICS
% ====================================================================

fprintf('Computing interception metrics...\n');

% Interception error per trial
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_interception_error = interception_error_all(trial_idx);
    fprintf('  Trial %d: Mean interception error = %.4f m (std = %.4f m)\n', ...
        trial, mean(trial_interception_error), std(trial_interception_error));
end

fprintf('\n');

% ====================================================================
% CREATE MAIN 3D FIGURE
% ====================================================================

fig1 = figure('Name', 'Player Chasing Ball - 3D Trajectories', ...
    'Position', [100, 100, 1600, 900], 'Color', 'white', 'Visible', 'on');

% ====================================================================
% SUBPLOT 1: 3D VIEW (Main 3D Plot)
% ====================================================================

ax1 = subplot(2, 2, 1);
hold on;
axis equal;
grid on;
xlabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
zlabel('Z Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('3D Trajectories: Player vs Ball', 'FontSize', 12, 'FontWeight', 'bold');
view(ax1, 45, 30);
cameratoolbar(fig1, 'Show');

fprintf('Plotting 3D trajectories...\n');

% Plot trajectories for each trial
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Player trajectory (solid line)
    player_x = x_player(trial_idx);
    player_y = y_player(trial_idx);
    player_z = z_player(trial_idx);
    plot3(player_x, player_y, player_z, '-', ...
        'Color', rgb_colors{trial}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Trial %d - Player', trial));
    
    % Ball trajectory (dashed line)
    ball_x = x_ball(trial_idx);
    ball_y = y_ball(trial_idx);
    ball_z = z_ball(trial_idx);
    plot3(ball_x, ball_y, ball_z, '--', ...
        'Color', rgb_colors{trial}, 'LineWidth', 2, ...
        'DisplayName', sprintf('Trial %d - Ball', trial));
    
    % Plot start positions
    plot3(player_x(1), player_y(1), player_z(1), 'o', ...
        'Color', rgb_colors{trial}, 'MarkerSize', 12, 'MarkerFaceColor', rgb_colors{trial}, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2, 'HandleVisibility', 'off');
    
    % Plot end positions
    plot3(player_x(end), player_y(end), player_z(end), 's', ...
        'Color', rgb_colors{trial}, 'MarkerSize', 12, 'MarkerFaceColor', rgb_colors{trial}, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2, 'HandleVisibility', 'off');
end

legend('Location', 'best', 'FontSize', 9);

% ====================================================================
% SUBPLOT 2: XY PLANE (Top-Down View)
% ====================================================================

ax2 = subplot(2, 2, 2);
hold on;
grid on;
xlabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Top-Down View (XY Plane)', 'FontSize', 12, 'FontWeight', 'bold');
axis equal;

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Player trajectory
    player_x = x_player(trial_idx);
    player_y = y_player(trial_idx);
    plot(player_x, player_y, '-', 'Color', rgb_colors{trial}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Trial %d - Player', trial));
    
    % Ball trajectory
    ball_x = x_ball(trial_idx);
    ball_y = y_ball(trial_idx);
    plot(ball_x, ball_y, '--', 'Color', rgb_colors{trial}, 'LineWidth', 2, ...
        'DisplayName', sprintf('Trial %d - Ball', trial));
end

legend('Location', 'best', 'FontSize', 9);

% ====================================================================
% SUBPLOT 3: XZ PLANE (Side View)
% ====================================================================

ax3 = subplot(2, 2, 3);
hold on;
grid on;
xlabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Z Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Side View (XZ Plane)', 'FontSize', 12, 'FontWeight', 'bold');
axis equal;

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Player trajectory
    player_x = x_player(trial_idx);
    player_z = z_player(trial_idx);
    plot(player_x, player_z, '-', 'Color', rgb_colors{trial}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Trial %d - Player', trial));
    
    % Ball trajectory
    ball_x = x_ball(trial_idx);
    ball_z = z_ball(trial_idx);
    plot(ball_x, ball_z, '--', 'Color', rgb_colors{trial}, 'LineWidth', 2, ...
        'DisplayName', sprintf('Trial %d - Ball', trial));
end

legend('Location', 'best', 'FontSize', 9);

% ====================================================================
% SUBPLOT 4: INTERCEPTION ERROR OVER TIME
% ====================================================================

ax4 = subplot(2, 2, 4);
hold on;
grid on;
xlabel('Time (steps)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Distance to Ball (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Interception Error Over Time', 'FontSize', 12, 'FontWeight', 'bold');

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_interception = interception_error_all(trial_idx);
    plot(trial_idx, trial_interception, '-', 'Color', rgb_colors{trial}, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Trial %d', trial));
end

legend('Location', 'best', 'FontSize', 9);

sgtitle('Dual-Hierarchy: Player Chasing Moving Ball', ...
    'FontSize', 14, 'FontWeight', 'bold');

drawnow;

% ====================================================================
% CREATE COMPARISON FIGURE: PER-COORDINATE TRACKING
% ====================================================================

fig2 = figure('Name', 'Coordinate Tracking: Player vs Ball', ...
    'Position', [100, 1050, 1400, 800], 'Color', 'white', 'Visible', 'on');

fprintf('Plotting per-coordinate tracking...\n');

% X Coordinate
subplot(3, 1, 1);
hold on;
plot(x_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball X');
plot(x_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player X');
grid on;
ylabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('X Coordinate Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    yline(0, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% Y Coordinate
subplot(3, 1, 2);
hold on;
plot(y_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball Y');
plot(y_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player Y');
grid on;
ylabel('Y Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Y Coordinate Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% Z Coordinate
subplot(3, 1, 3);
hold on;
plot(z_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball Z');
plot(z_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player Z');
grid on;
xlabel('Time (steps)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Z Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Z Coordinate Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

sgtitle('Per-Coordinate Tracking Analysis', 'FontSize', 14, 'FontWeight', 'bold');

drawnow;

% ====================================================================
% CREATE VELOCITY COMPARISON FIGURE
% ====================================================================

fig3 = figure('Name', 'Velocity Analysis', ...
    'Position', [100, 2050, 1400, 800], 'Color', 'white', 'Visible', 'on');

fprintf('Plotting velocity comparisons...\n');

% VX
subplot(3, 1, 1);
hold on;
plot(vx_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball VX');
plot(vx_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player VX');
grid on;
ylabel('VX (m/s)', 'FontSize', 11, 'FontWeight', 'bold');
title('X Velocity Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% VY
subplot(3, 1, 2);
hold on;
plot(vy_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball VY');
plot(vy_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player VY');
grid on;
ylabel('VY (m/s)', 'FontSize', 11, 'FontWeight', 'bold');
title('Y Velocity Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% VZ
subplot(3, 1, 3);
hold on;
plot(vz_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball VZ');
plot(vz_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player VZ');
grid on;
xlabel('Time (steps)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('VZ (m/s)', 'FontSize', 11, 'FontWeight', 'bold');
title('Z Velocity Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

sgtitle('Velocity Analysis: Player Matching Ball Velocity', 'FontSize', 14, 'FontWeight', 'bold');

drawnow;

% ====================================================================
% CREATE FREE ENERGY & LEARNING FIGURE
% ====================================================================

fig4 = figure('Name', 'Learning Dynamics', ...
    'Position', [100, 2950, 1400, 600], 'Color', 'white', 'Visible', 'on');

fprintf('Plotting learning dynamics...\n');

% Free Energy
subplot(1, 2, 1);
semilogy(free_energy_all, 'k-', 'LineWidth', 2.5);
grid on;
xlabel('Time (steps)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Free Energy (log scale)', 'FontSize', 11, 'FontWeight', 'bold');
title('Free Energy Minimization', 'FontSize', 12, 'FontWeight', 'bold');

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% Learning Trace
subplot(1, 2, 2);
semilogy(learning_trace_W + 1e-10, 'g-', 'LineWidth', 2.5);
grid on;
xlabel('Time (steps)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Weight Change Magnitude (log scale)', 'FontSize', 11, 'FontWeight', 'bold');
title('Learning Trace: Weight Updates', 'FontSize', 12, 'FontWeight', 'bold');

% Mark trial boundaries
for trial = 2:n_trials
    boundary_idx = phases_indices{trial}(1);
    xline(boundary_idx, '--', 'Color', [0.5, 0.5, 0.5], 'Alpha', 0.5, 'HandleVisibility', 'off');
end

sgtitle('Learning Dynamics Over Time', 'FontSize', 14, 'FontWeight', 'bold');

drawnow;

% ====================================================================
% ANALYSIS SUMMARY
% ====================================================================

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  DUAL-HIERARCHY LEARNING ANALYSIS                           ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('INTERCEPTION PERFORMANCE:\n');
fprintf('─────────────────────────────────────────────────────────\n');

overall_rmse = sqrt(mean(interception_error_all.^2));
fprintf('Overall Interception RMSE: %.4f m\n\n', overall_rmse);

fprintf('Per-Trial Analysis:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_error = interception_error_all(trial_idx);
    
    initial_error = trial_error(1);
    final_error = trial_error(end);
    improvement = initial_error - final_error;
    improvement_pct = (improvement / initial_error) * 100;
    
    fprintf('  Trial %d (%s):\n', trial, color_names{trial});
    fprintf('    Initial interception error: %.4f m\n', initial_error);
    fprintf('    Final interception error:   %.4f m\n', final_error);
    fprintf('    Improvement: %.4f m (%.1f%%)\n', improvement, improvement_pct);
    fprintf('    Mean error: %.4f m | Std Dev: %.4f m\n', mean(trial_error), std(trial_error));
    fprintf('    Min error: %.4f m | Max error: %.4f m\n\n', min(trial_error), max(trial_error));
end

fprintf('POSITION TRACKING:\n');
fprintf('─────────────────────────────────────────────────────────\n');

% Compute per-coordinate RMS errors
x_error = abs(x_ball - x_player);
y_error = abs(y_ball - y_player);
z_error = abs(z_ball - z_player);

fprintf('X Coordinate:  Mean error = %.4f m | RMS = %.4f m\n', mean(x_error), sqrt(mean(x_error.^2)));
fprintf('Y Coordinate:  Mean error = %.4f m | RMS = %.4f m\n', mean(y_error), sqrt(mean(y_error.^2)));
fprintf('Z Coordinate:  Mean error = %.4f m | RMS = %.4f m\n\n', mean(z_error), sqrt(mean(z_error.^2)));

fprintf('VELOCITY TRACKING:\n');
fprintf('─────────────────────────────────────────────────────────\n');

% Compute per-coordinate velocity errors
vx_error = abs(vx_ball - vx_player);
vy_error = abs(vy_ball - vy_player);
vz_error = abs(vz_ball - vz_player);

fprintf('VX (X Velocity):  Mean error = %.4f m/s | RMS = %.4f m/s\n', mean(vx_error), sqrt(mean(vx_error.^2)));
fprintf('VY (Y Velocity):  Mean error = %.4f m/s | RMS = %.4f m/s\n', mean(vy_error), sqrt(mean(vy_error.^2)));
fprintf('VZ (Z Velocity):  Mean error = %.4f m/s | RMS = %.4f m/s\n\n', mean(vz_error), sqrt(mean(vz_error.^2)));

fprintf('LEARNING EFFICIENCY:\n');
fprintf('─────────────────────────────────────────────────────────\n');

fprintf('Initial Free Energy:  %.6e\n', free_energy_all(1));
fprintf('Final Free Energy:    %.6e\n', free_energy_all(end));
fprintf('Free Energy Reduction: %.6e (%.1f%%)\n', ...
    free_energy_all(1) - free_energy_all(end), ...
    ((free_energy_all(1) - free_energy_all(end)) / free_energy_all(1)) * 100);

fprintf('Total Learning Steps: %d\n', length(free_energy_all));
fprintf('Total Trials:         %d\n\n', n_trials);

fprintf('WEIGHT MATRIX STATISTICS:\n');
fprintf('─────────────────────────────────────────────────────────\n');

fprintf('Motor Region:\n');
fprintf('  W_motor_L2_to_L1 norm: %.4f (7×6 matrix)\n', norm(W_motor_L2_to_L1, 'fro'));
fprintf('  W_motor_L3_to_L2 norm: %.4f (3×6 matrix)\n', norm(W_motor_L3_to_L2, 'fro'));

fprintf('Planning Region:\n');
fprintf('  W_plan_L2_to_L1 norm:  %.4f (7×6 matrix)\n', norm(W_plan_L2_to_L1, 'fro'));
fprintf('  W_plan_L3_to_L2 norm:  %.4f (3×6 matrix)\n\n', norm(W_plan_L3_to_L2, 'fro'));

fprintf('TASK CONTEXT (L0):\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('One-hot encoding dimension: %d (one per trial)\n', size(R_L0, 2));
fprintf('Explicit task representation enables task-specific learning\n');
fprintf('Motor region: Always learning (goal-independent)\n');
fprintf('Planning region: Task-gated learning (gates 0.3-1.0)\n\n');

fprintf('✓ Visualization complete!\n');
fprintf('  Three figures created:\n');
fprintf('  1. Main 3D trajectories with top-down and side views\n');
fprintf('  2. Per-coordinate tracking (X, Y, Z)\n');
fprintf('  3. Velocity analysis (VX, VY, VZ)\n');
fprintf('  4. Learning dynamics (Free Energy & Weight Changes)\n');
fprintf('\n(Close figures to continue.)\n');

% Keep windows open
uiwait(msgbox('Visualizations complete! Close this dialog to exit.', 'Done'));
