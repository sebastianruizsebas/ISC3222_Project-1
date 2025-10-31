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

results_file = './figures/3D_dual_hierarchy_results_best.mat';
if ~isfile(results_file)
    error('Results file not found: %s\nRun hierarchical_motion_inference_dual_hierarchy() first.', results_file);
end

loaded = load(results_file);
fprintf('✓ Loaded: %s\n\n', results_file);

% Backwards-compat: PSO saves a single struct (`best_data`) into the MAT file.
% The visualizer expects variables like `x_player`, `x_ball`, `phases_indices`, etc.
% If we loaded a struct, unpack its fields into the script workspace.
if isfield(loaded, 'best_data') && isstruct(loaded.best_data)
    bd = loaded.best_data;
    fn = fieldnames(bd);
    for k = 1:numel(fn)
        try
            % assign into current workspace
            eval(sprintf('%s = bd.%s;', fn{k}, fn{k}));
        catch
            % ignore if assignment fails for any unusual field
        end
    end
    fprintf('  Unpacked `best_data` struct into variables for visualization.\n');
elseif isfield(loaded, 'results') && isstruct(loaded.results)
    bd = loaded.results;
    fn = fieldnames(bd);
    for k = 1:numel(fn)
        try
            eval(sprintf('%s = bd.%s;', fn{k}, fn{k}));
        catch
        end
    end
    fprintf('  Unpacked `results` struct into variables for visualization.\n');
else
    % If the MAT contained variables directly, copy them into locals
    vars = fieldnames(loaded);
    for k = 1:numel(vars)
        try
            eval(sprintf('%s = loaded.%s;', vars{k}, vars{k}));
        catch
        end
    end
end

% ----------------------------
% Sanity checks & diagnostics
% ----------------------------
needed = {'x_player','y_player','z_player','x_ball','y_ball','z_ball', 'vx_ball','vy_ball','vz_ball','vx_player','vy_player','vz_player','phases_indices','interception_error_all','free_energy_all','learning_trace_W'};
fprintf('\nSanity check: verifying required variables for plotting...\n');
missing = {};
for k = 1:numel(needed)
    if ~exist(needed{k}, 'var') || isempty(eval(needed{k}))
        missing{end+1} = needed{k}; %#ok<AGROW>
    else
        % report simple size info
        val = eval(needed{k});
        if iscell(val)
            fprintf('  %s : cell with %d entries\n', needed{k}, numel(val));
        else
            fprintf('  %s : size = %s\n', needed{k}, mat2str(size(val)));
        end
    end
end
if ~isempty(missing)
    warning('Missing or empty variables required for full plotting: %s\nFigures may be incomplete.', strjoin(missing, ', '));
end

% Helper to ensure vector orientation (row vectors are fine for MATLAB plotting)
ensure_vector = @(v) (isvector(v) && ~isempty(v));

% (Removed verbose per-variable diagnostics to avoid confusion.)


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

% Position using normalized units so figures appear on most screens
fig2 = figure('Name', 'Coordinate Tracking: Player vs Ball', ...
    'Units', 'normalized', 'Position', [0.05, 0.05, 0.45, 0.4], 'Color', 'white', 'Visible', 'on');

fprintf('Plotting per-coordinate tracking...\n');

% Clear figure and ensure it's active
figure(fig2); clf(fig2);

% X Coordinate
subplot(3, 1, 1);
hold on;
if exist('x_ball','var') && exist('x_player','var') && ~isempty(x_ball) && ~isempty(x_player)
    plot(x_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball X');
    plot(x_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player X');
else
    text(0.5,0.5,'No X data available','HorizontalAlignment','center','Units','normalized');
end
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
if exist('y_ball','var') && exist('y_player','var') && ~isempty(y_ball) && ~isempty(y_player)
    plot(y_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball Y');
    plot(y_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player Y');
else
    text(0.5,0.5,'No Y data available','HorizontalAlignment','center','Units','normalized');
end
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
if exist('z_ball','var') && exist('z_player','var') && ~isempty(z_ball) && ~isempty(z_player)
    plot(z_ball, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball Z');
    plot(z_player, 'r--', 'LineWidth', 2, 'DisplayName', 'Player Z');
else
    text(0.5,0.5,'No Z data available','HorizontalAlignment','center','Units','normalized');
end
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
    'Units', 'normalized', 'Position', [0.55, 0.05, 0.4, 0.4], 'Color', 'white', 'Visible', 'on');

fprintf('Plotting velocity comparisons...\n');

figure(fig3); clf(fig3);

% VX
subplot(3, 1, 1);
hold on;
if exist('vx_ball','var') && exist('vx_player','var') && ~isempty(vx_ball) && ~isempty(vx_player)
    vb = double(vx_ball(:)); vp = double(vx_player(:));
    vb(~isfinite(vb)) = NaN; vp(~isfinite(vp)) = NaN;
    if all(isnan(vb)) && all(isnan(vp))
        text(0.5,0.5,'VX arrays contain only NaN/Inf','HorizontalAlignment','center','Units','normalized');
    else
        plot(vb, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball VX');
        plot(vp, 'r--', 'LineWidth', 2, 'DisplayName', 'Player VX');
        axis tight;
    end
else
    text(0.5,0.5,'No VX data available','HorizontalAlignment','center','Units','normalized');
end
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
if exist('vy_ball','var') && exist('vy_player','var') && ~isempty(vy_ball) && ~isempty(vy_player)
    vb = double(vy_ball(:)); vp = double(vy_player(:));
    vb(~isfinite(vb)) = NaN; vp(~isfinite(vp)) = NaN;
    if all(isnan(vb)) && all(isnan(vp))
        text(0.5,0.5,'VY arrays contain only NaN/Inf','HorizontalAlignment','center','Units','normalized');
    else
        plot(vb, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball VY');
        plot(vp, 'r--', 'LineWidth', 2, 'DisplayName', 'Player VY');
        axis tight;
    end
else
    text(0.5,0.5,'No VY data available','HorizontalAlignment','center','Units','normalized');
end
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
if exist('vz_ball','var') && exist('vz_player','var') && ~isempty(vz_ball) && ~isempty(vz_player)
    vb = double(vz_ball(:)); vp = double(vz_player(:));
    vb(~isfinite(vb)) = NaN; vp(~isfinite(vp)) = NaN;
    if all(isnan(vb)) && all(isnan(vp))
        text(0.5,0.5,'VZ arrays contain only NaN/Inf','HorizontalAlignment','center','Units','normalized');
    else
        plot(vb, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Ball VZ');
        plot(vp, 'r--', 'LineWidth', 2, 'DisplayName', 'Player VZ');
        axis tight;
    end
else
    text(0.5,0.5,'No VZ data available','HorizontalAlignment','center','Units','normalized');
end
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
    'Units', 'normalized', 'Position', [0.05, 0.5, 0.9, 0.45], 'Color', 'white', 'Visible', 'on');

fprintf('Plotting learning dynamics...\n');

figure(fig4); clf(fig4);

% Free Energy
subplot(1, 2, 1);
if exist('free_energy_all','var') && ~isempty(free_energy_all)
    y = double(free_energy_all(:));
    % Replace non-finite or non-positive values with eps for log plotting
    y(~isfinite(y) | y <= 0) = eps;
    semilogy(y, 'k-', 'LineWidth', 2.5);
    axis tight;
else
    text(0.5,0.5,'No Free Energy data','HorizontalAlignment','center','Units','normalized');
end
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
if exist('learning_trace_W','var') && ~isempty(learning_trace_W)
    lw = double(learning_trace_W(:));
    lw(~isfinite(lw) | lw <= 0) = eps;
    semilogy(lw + 1e-10, 'g-', 'LineWidth', 2.5);
    axis tight;
else
    text(0.5,0.5,'No learning trace data','HorizontalAlignment','center','Units','normalized');
end
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
