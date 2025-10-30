fprintf('╔═════════════════════════════════════════════════════════════╗\n');
fprintf('║  SENSORIMOTOR LEARNING: 3D REACHING & GRASPING             ║\n');
fprintf('║  Learning to Reach Targets in 3D Space via Predictive Coding ║\n');
fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% HELPER FUNCTIONS: 3D TRAJECTORY VISUALIZATION
% ====================================================================
% Define these before main code so they're available when needed

    function plot_trajectory_3d(x, y, z, base_color, alpha)
        % Plot a 3D trajectory with fading effect (bright start, faint end)
        
        n_points = length(x);
        if n_points < 2, return; end
        
        % Create color gradient from bright to dim
        color_rgb = hex2rgb(base_color);
        
        % Plot segments with decreasing brightness
        for i = 1:n_points-1
            progress = i / n_points;  % 0 to 1 over trajectory
            
            % Fade effect: bright at start, dim at end
            brightness = 1 - (progress * 0.4);  % Goes from 1 to 0.6
            segment_color = color_rgb * brightness;
            
            % Line width decreases slightly over time
            lw = 2.5 * (1 - progress * 0.3);
            
            plot3([x(i), x(i+1)], [y(i), y(i+1)], [z(i), z(i+1)], ...
                'Color', segment_color, 'LineWidth', lw, 'HandleVisibility', 'off');
        end
    end

    function plot_trajectory_3d_dashed(x, y, z, base_color, alpha)
        % Plot a 3D trajectory with dashed line and fading effect
        
        n_points = length(x);
        if n_points < 2, return; end
        
        % Create color gradient from bright to dim
        color_rgb = hex2rgb(base_color);
        
        % Plot segments with decreasing brightness
        for i = 1:n_points-1
            progress = i / n_points;  % 0 to 1 over trajectory
            
            % Fade effect: bright at start, dim at end
            brightness = 1 - (progress * 0.4);  % Goes from 1 to 0.6
            segment_color = color_rgb * brightness;
            
            % Line width decreases slightly over time
            lw = 2.0 * (1 - progress * 0.3);
            
            plot3([x(i), x(i+1)], [y(i), y(i+1)], [z(i), z(i+1)], ...
                'Color', segment_color, 'LineWidth', lw, 'LineStyle', '--', 'HandleVisibility', 'off');
        end
    end

    function rgb = hex2rgb(color)
        % Convert color name to RGB
        switch color
            case 'r'
                rgb = [1, 0, 0];
            case 'g'
                rgb = [0, 1, 0];
            case 'b'
                rgb = [0, 0, 1];
            case 'm'
                rgb = [1, 0, 1];
            case 'c'
                rgb = [0, 1, 1];
            case 'y'
                rgb = [1, 1, 0];
            case 'k'
                rgb = [0, 0, 0];
            otherwise
                rgb = [0.5, 0.5, 0.5];
        end
    end

% ====================================================================
% 3D REACHING TASK CONFIGURATION
% ====================================================================

dt = 0.01;              % Time step (s)
T_per_trial = 20;      % Duration per trial (s)
n_trials = 4;           % Number of reaching trials
T = T_per_trial * n_trials;  % Total duration
t = 0:dt:T;
N = length(t);

% Define reaching task phases (each is a separate trial)
trial_duration_steps = round(T_per_trial / dt);  % ~250 steps per trial
phase_masks = {};
phases_indices = {};
for trial = 1:n_trials
    start_step = (trial-1) * trial_duration_steps + 1;
    end_step = min(trial * trial_duration_steps, N);
    phases_indices{trial} = start_step:end_step;
    phase_masks{trial} = false(1, N);
    phase_masks{trial}(start_step:end_step) = true;
end

% Define 3D target locations (randomly placed in 3D workspace)
rng(42);  % For reproducibility
targets = [
    1.5, 1.5, 1.0;      % Target 1
    -1.5, 1.5, 0.5;     % Target 2
    -1.5, -1.5, 1.5;    % Target 3
    1.5, -1.5, -0.5     % Target 4
];

% Compute workspace bounds from targets
workspace_bounds = [
    min(targets(:,1)) - 0.5, max(targets(:,1)) + 0.5;  % X bounds
    min(targets(:,2)) - 0.5, max(targets(:,2)) + 0.5;  % Y bounds
    min(targets(:,3)) - 0.5, max(targets(:,3)) + 0.5   % Z bounds
];

% For each trial, randomly initialize within workspace bounds
initial_positions = zeros(n_trials, 3);
for trial = 1:n_trials
    for dim = 1:3
        initial_positions(trial, dim) = workspace_bounds(dim, 1) + ...
            rand() * (workspace_bounds(dim, 2) - workspace_bounds(dim, 1));
    end
end

fprintf('3D REACHING TASK (Multi-Trial):\n');
fprintf('Workspace bounds: X∈[%.2f,%.2f], Y∈[%.2f,%.2f], Z∈[%.2f,%.2f]\n', ...
    workspace_bounds(1,1), workspace_bounds(1,2), ...
    workspace_bounds(2,1), workspace_bounds(2,2), ...
    workspace_bounds(3,1), workspace_bounds(3,2));
fprintf('\n');

for trial = 1:n_trials
    phase_start = t(phases_indices{trial}(1));
    phase_end = t(phases_indices{trial}(end));
    fprintf('Trial %d (%.2f-%.2fs): Start [%.2f, %.2f, %.2f] → Target [%.2f, %.2f, %.2f]\n', ...
        trial, phase_start, phase_end, ...
        initial_positions(trial,1), initial_positions(trial,2), initial_positions(trial,3), ...
        targets(trial,1), targets(trial,2), targets(trial,3));
end
fprintf('\n');

fprintf('3D REACHING TASK:\n');
fprintf('  Phase 1 (0-2.5s):  Reach to target [%.2f, %.2f, %.2f]\n', targets(1,1), targets(1,2), targets(1,3));
fprintf('  Phase 2 (2.5-5s):  Reach to target [%.2f, %.2f, %.2f]\n', targets(2,1), targets(2,2), targets(2,3));
fprintf('  Phase 3 (5-7.5s):  Reach to target [%.2f, %.2f, %.2f]\n', targets(3,1), targets(3,2), targets(3,3));
fprintf('  Phase 4 (7.5-10s): Reach to target [%.2f, %.2f, %.2f]\n\n', targets(4,1), targets(4,2), targets(4,3));

% ====================================================================
% LAYER DIMENSIONS - 3D VERSION
% ====================================================================

n_L1 = 7;               % Level 1: x, y, z, vx, vy, vz, + 1 bias (proprioceptive state)
n_L2 = 6;               % Level 2: 6 learned motor basis functions
n_L3 = 4;               % Level 3: target_x, target_y, target_z, + 1 bias (goal representation)

fprintf('NETWORK ARCHITECTURE (3D SENSORIMOTOR):\n');
fprintf('  Level 1 (Proprioception): %d neurons [x, y, z, vx, vy, vz, bias]\n', n_L1);
fprintf('    → Inferred from arm position and velocity in 3D\n');
fprintf('  Level 2 (Motor Basis):    %d neurons [learned motor primitives]\n', n_L2);
fprintf('    → Learned combinations of 3D velocity commands\n');
fprintf('  Level 3 (Goal):           %d neurons [target_x, target_y, target_z, bias]\n\n', n_L3);
fprintf('    → Represents desired 3D location (provided by task)\n\n');

% ====================================================================
% 3D KINEMATICS: INTEGRATING MOTOR COMMANDS INTO POSITION
% ====================================================================

% Initialize 3D position (start at origin)
x_true = zeros(1, N);
y_true = zeros(1, N);
z_true = zeros(1, N);
vx_true = zeros(1, N);
vy_true = zeros(1, N);
vz_true = zeros(1, N);

% Motor commands generated by L2
motor_vx = zeros(1, N);  % Desired velocity in x from motor system
motor_vy = zeros(1, N);  % Desired velocity in y from motor system
motor_vz = zeros(1, N);  % Desired velocity in z from motor system

% Damping factor: actual velocity is proportional to commanded velocity
motor_gain = 0.5;
damping = 0.95;

fprintf('3D MOTOR DYNAMICS:\n');
fprintf('  Motor gain: %.2f (scaling of commands to actual motion)\n', motor_gain);
fprintf('  Damping: %.2f (velocity decay per timestep)\n', damping);
fprintf('  Workspace: x,y,z ∈ [%.2f, %.2f] meters\n\n', workspace_bounds(1,1), workspace_bounds(1,2));

% ====================================================================
% LEARNING PARAMETERS
% ====================================================================

eta_rep = 0.001497;      % Representation learning rate
eta_W = 0.000155;        % Weight matrix learning rate
momentum = 0.9415;       % Momentum for representation updates
weight_decay = 0.9908;   % L2 regularization on weights
pi_L1 = 100;             % Precision (reliability) of L1 sensory input
pi_L2 = 10;              % Precision of L2 motor basis
pi_L3 = 1;               % Precision of L3 goal representation

fprintf('LEARNING PARAMETERS:\n');
fprintf('  η_rep = %.6f (representation learning rate)\n', eta_rep);
fprintf('  η_W   = %.6f (weight matrix learning rate)\n', eta_W);
fprintf('  Momentum = %.4f\n', momentum);
fprintf('  Weight Decay = %.4f\n', weight_decay);
fprintf('  π_L1  = %.0f, π_L2  = %.0f, π_L3  = %.0f\n\n', pi_L1, pi_L2, pi_L3);

% ====================================================================
% INITIALIZE REPRESENTATIONS (3D)
% ====================================================================

% Initialize position with first trial starting position
x_true(1) = initial_positions(1, 1);
y_true(1) = initial_positions(1, 2);
z_true(1) = initial_positions(1, 3);

% Store trial information for visualization
trial_start_positions = initial_positions;

% L1: Proprioceptive state (start at initial position, zero velocity)
R_L1(1,1:3) = [x_true(1), y_true(1), z_true(1)];  % Position
R_L1(1,4:6) = [vx_true(1), vy_true(1), vz_true(1)];  % Velocity
R_L1(1,7) = 1;  % Bias

% L2: Motor basis functions (initialize randomly)
R_L2(1,:) = 0.01 * randn(1, n_L2);

% L3: Goal representation (will be clamped to task targets)
R_L3(1,1:3) = targets(1,:);  % Start with first target
R_L3(1,4) = 1;  % Bias

fprintf('INITIAL CONDITIONS (Trial 1):\n');
fprintf('  Start position: [%.2f, %.2f, %.2f]\n', x_true(1), y_true(1), z_true(1));
fprintf('  Target position: [%.2f, %.2f, %.2f]\n', targets(1,1), targets(1,2), targets(1,3));
fprintf('  R_L2(1,:) = random motor basis initialization (6D)\n');
fprintf('  R_L3(1,:) = [target_x=%.2f, target_y=%.2f, target_z=%.2f, bias=1]\n\n', targets(1,1), targets(1,2), targets(1,3));

% Initialize weight matrices for 3D
W_L1_from_L2 = 0.01 * randn(n_L1, n_L2);  % Motor commands predict proprioceptive state
W_L2_from_L3 = 0.01 * randn(n_L2, n_L3);  % Goals predict motor commands

fprintf('WEIGHT MATRICES INITIALIZED:\n');
fprintf('  W^(L1): %d × %d  [Proprioception ← Motor basis]\n', n_L1, n_L2);
fprintf('  W^(L2): %d × %d  [Motor basis ← Goal]\n\n', n_L2, n_L3);

% Allocate storage for tracking - use single precision to save memory
E_L1 = zeros(N, n_L1, 'single');
E_L2 = zeros(N, n_L2, 'single');
pred_L1 = zeros(N, n_L1, 'single');
pred_L2 = zeros(N, n_L2, 'single');
free_energy_all = zeros(1, N, 'single');
reaching_error_all = zeros(1, N, 'single');

% ====================================================================
% MAIN LEARNING LOOP (3D)
% ====================================================================

fprintf('Running 3D sensorimotor reaching trials...\n');

current_trial = 1;

for i = 1:N-1
    if mod(i, 100) == 0, fprintf('.'); end
    
    % ==============================================================
    % CHECK FOR TRIAL TRANSITION
    % ==============================================================
    % At the start of each new trial, reinitialize position
    if i > 1
        for trial = 2:n_trials
            if i == phases_indices{trial}(1)
                % Reset to new trial initial position
                x_true(i) = initial_positions(trial, 1);
                y_true(i) = initial_positions(trial, 2);
                z_true(i) = initial_positions(trial, 3);
                vx_true(i) = 0;
                vy_true(i) = 0;
                vz_true(i) = 0;
                
                % Update L1 sensory representation
                R_L1(i,1:3) = [x_true(i), y_true(i), z_true(i)];
                R_L1(i,4:6) = [0, 0, 0];
                
                current_trial = trial;
                fprintf('\n[Trial %d started]\n', trial);
                break;
            end
        end
    end
    
    % ==============================================================
    % STEP 0: UPDATE GOAL (TASK INPUT)
    % ==============================================================
    % Goal is externally specified and clamped (like visual target in 3D)
    R_L3(i,1:3) = targets(current_trial,:);  % CLAMP: Task provides 3D target
    
    % ==============================================================
    % STEP 1: FORWARD PREDICTION (Top-Down)
    % ==============================================================
    
    % L2 predicts proprioceptive state via motor commands
    pred_motor = R_L2(i,:) * W_L1_from_L2';  % Predicted velocity (7D)
    pred_L1(i,:) = [pred_motor(1:3), pred_motor(1:3), 1];  % Position predicted from velocity
    
    % L3 (goal) predicts L2 (motor commands needed to reach goal)
    pred_L2(i,:) = R_L3(i,:) * W_L2_from_L3';
    
    % ==============================================================
    % STEP 2: MOTOR EXECUTION (3D)
    % ==============================================================
    
    % Motor commands are a weighted combination of basis functions
    motor_command = R_L2(i,1:6);  % All 6 basis functions
    
    % Apply motor gain and damping (simulates muscle dynamics in 3D)
    motor_vx(i) = motor_gain * motor_command(1);
    motor_vy(i) = motor_gain * motor_command(2);
    motor_vz(i) = motor_gain * motor_command(3);
    
    % Update velocity with damping
    vx_true(i+1) = damping * vx_true(i) + motor_vx(i);
    vy_true(i+1) = damping * vy_true(i) + motor_vy(i);
    vz_true(i+1) = damping * vz_true(i) + motor_vz(i);
    
    % Update position via integration (3D)
    x_true(i+1) = x_true(i) + dt * vx_true(i+1);
    y_true(i+1) = y_true(i) + dt * vy_true(i+1);
    z_true(i+1) = z_true(i) + dt * vz_true(i+1);
    
    % Clamp position to reachable workspace
    x_true(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), x_true(i+1)));
    y_true(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), y_true(i+1)));
    z_true(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), z_true(i+1)));
    
    % ==============================================================
    % STEP 3: PROPRIOCEPTIVE FEEDBACK (Sensory Input - 3D)
    % ==============================================================
    
    sensory_input = [x_true(i+1), y_true(i+1), z_true(i+1)];  % Direct 3D position feedback
    
    % ==============================================================
    % STEP 4: ERROR COMPUTATION (Bottom-Up - 3D)
    % ==============================================================
    
    E_L1_full = R_L1(i,:) - pred_L1(i,:);
    
    % Override position error with actual sensory input (3D)
    E_L1(i,1:3) = sensory_input - pred_L1(i,1:3);  % 3D position error
    E_L1(i,4:6) = E_L1_full(4:6);  % Velocity error from prediction
    E_L1(i,7) = E_L1_full(7);  % Bias error
    
    % L2 error
    E_L2(i,:) = R_L2(i,:) - pred_L2(i,:);
    
    % Calculate 3D reaching error
    reaching_error_all(i) = sqrt((x_true(i) - targets(current_trial,1))^2 + ...
                                 (y_true(i) - targets(current_trial,2))^2 + ...
                                 (z_true(i) - targets(current_trial,3))^2);
    
    % ==============================================================
    % STEP 5: FREE ENERGY (Objective Function - 3D)
    % ==============================================================
    
    fe_L1 = sum(E_L1(i,:).^2) / (2 * pi_L1);
    fe_L2 = sum(E_L2(i,:).^2) / (2 * pi_L2);
    
    % Add 3D reaching cost to free energy
    fe_reaching = (pi_L1 / 100) * reaching_error_all(i)^2;
    
    free_energy_all(i) = fe_L1 + fe_L2 + fe_reaching;
    
    % ==============================================================
    % STEP 6: REPRESENTATION UPDATES (3D)
    % ==============================================================
    
    % L1: Proprioceptive state updates (3D position clamped, velocity inferred)
    R_L1(i+1,1:3) = sensory_input;  % CLAMP: 3D Position from proprioception
    
    % Velocity inferred from prediction error
    delta_R_L1_inferred = -E_L1(i,4:6);
    decay = 1 - momentum;
    R_L1(i+1,4:6) = momentum * R_L1(i,4:6) + decay * eta_rep * delta_R_L1_inferred;
    R_L1(i+1,4:6) = max(-2, min(2, R_L1(i+1,4:6)));  % 3D Velocity bounds
    R_L1(i+1,7) = 1;  % Bias
    
    % L2: Motor basis updates driven by 3D goal-directed reaching
    coupling_from_L1 = E_L1(i,:) * W_L1_from_L2;
    norm_W1 = max(0.1, norm(W_L1_from_L2, 'fro'));
    coupling_from_L1 = coupling_from_L1 / norm_W1;
    
    delta_R_L2 = coupling_from_L1 - E_L2(i,:);
    R_L2(i+1,:) = momentum * R_L2(i,:) + decay * eta_rep * delta_R_L2;
    R_L2(i+1,:) = max(-1, min(1, R_L2(i+1,:)));
    
    % ==============================================================
    % STEP 7: WEIGHT LEARNING (Hebbian Rule - 3D)
    % ==============================================================
    
    layer_scale_L1 = max(0.1, mean(abs(R_L2(i,:))));
    layer_scale_L2 = max(0.1, mean(abs(R_L3(i,:))));
    
    % Learn mapping from motor basis to proprioceptive state
    dW_L1 = -(eta_W * pi_L1 / layer_scale_L1) * (E_L1(i,:)' * R_L2(i,:));
    W_L1_from_L2 = W_L1_from_L2 + dW_L1;
    
    % Learn mapping from goals to motor commands
    dW_L2 = -(eta_W * pi_L2 / layer_scale_L2) * (E_L2(i,:)' * R_L3(i,:));
    W_L2_from_L3 = W_L2_from_L3 + dW_L2;
    
    % Weight regularization
    W_L1_from_L2 = W_L1_from_L2 * weight_decay;
    W_L2_from_L3 = W_L2_from_L3 * weight_decay;
    
    learning_trace_W(i) = norm(dW_L1, 'fro') + norm(dW_L2, 'fro');
    
end  % End main loop

fprintf('\n\n');

% ====================================================================
% 3D VISUALIZATION
% ====================================================================

figure('Position', [100, 100, 1600, 1000]);

% Define colors for trials
colors = {'r', 'g', 'b', 'm'};

% Plot 1: 3D Ground Truth Trajectories (Actual Reaching Paths)
subplot(2, 4, 1);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    traj_x = x_true(trial_idx);
    traj_y = y_true(trial_idx);
    traj_z = z_true(trial_idx);
    
    % Plot trajectory with color gradient (time progression)
    plot_trajectory_3d(traj_x, traj_y, traj_z, colors{trial}, 0.7);
    
    % Plot starting position
    plot3(trial_start_positions(trial,1), trial_start_positions(trial,2), trial_start_positions(trial,3), ...
        's', 'Color', colors{trial}, 'MarkerSize', 10, 'MarkerFaceColor', colors{trial}, 'DisplayName', sprintf('Trial %d Start', trial));
    % Plot target
    plot3(targets(trial, 1), targets(trial, 2), targets(trial, 3), ...
        'o', 'Color', colors{trial}, 'MarkerSize', 12, 'MarkerFaceColor', colors{trial}, 'DisplayName', sprintf('Trial %d Target', trial));
end
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Ground Truth Trajectories (Actual Reaching)');
view(45, 45);
legend('Location', 'best', 'FontSize', 8);

% Plot 2: 3D Learned Position Predictions (Model Learned Trajectories)
subplot(2, 4, 2);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    traj_x = R_L1(trial_idx,1);
    traj_y = R_L1(trial_idx,2);
    traj_z = R_L1(trial_idx,3);
    
    % Plot trajectory with color gradient (time progression) - dashed
    plot_trajectory_3d_dashed(traj_x, traj_y, traj_z, colors{trial}, 0.7);
    
    % Plot starting position
    plot3(trial_start_positions(trial,1), trial_start_positions(trial,2), trial_start_positions(trial,3), ...
        's', 'Color', colors{trial}, 'MarkerSize', 10, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k');
    % Plot target
    plot3(targets(trial, 1), targets(trial, 2), targets(trial, 3), ...
        'o', 'Color', colors{trial}, 'MarkerSize', 12, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k');
end
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Learned Position Predictions (Model Trajectories)');
view(45, 45);

% Plot 3: 3D Overlay - Ground Truth vs Learned Predictions
subplot(2, 4, 3);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Ground truth trajectory (solid)
    traj_x = x_true(trial_idx);
    traj_y = y_true(trial_idx);
    traj_z = z_true(trial_idx);
    plot_trajectory_3d(traj_x, traj_y, traj_z, colors{trial}, 0.7);
    
    % Learned prediction trajectory (dashed)
    traj_x_pred = R_L1(trial_idx,1);
    traj_y_pred = R_L1(trial_idx,2);
    traj_z_pred = R_L1(trial_idx,3);
    plot_trajectory_3d_dashed(traj_x_pred, traj_y_pred, traj_z_pred, colors{trial}, 0.7);
    
    % Plot starting position
    plot3(trial_start_positions(trial,1), trial_start_positions(trial,2), trial_start_positions(trial,3), ...
        's', 'Color', colors{trial}, 'MarkerSize', 8, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    % Plot target
    plot3(targets(trial, 1), targets(trial, 2), targets(trial, 3), ...
        'o', 'Color', colors{trial}, 'MarkerSize', 10, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Overlay: Truth (solid) vs Learned (dashed)');
view(45, 45);

% Plot 4: Alternative 3D View (different angle for better perspective)
subplot(2, 4, 4);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    
    % Ground truth trajectory (solid)
    traj_x = x_true(trial_idx);
    traj_y = y_true(trial_idx);
    traj_z = z_true(trial_idx);
    plot_trajectory_3d(traj_x, traj_y, traj_z, colors{trial}, 0.7);
    
    % Learned prediction trajectory (dashed)
    traj_x_pred = R_L1(trial_idx,1);
    traj_y_pred = R_L1(trial_idx,2);
    traj_z_pred = R_L1(trial_idx,3);
    plot_trajectory_3d_dashed(traj_x_pred, traj_y_pred, traj_z_pred, colors{trial}, 0.7);
    
    % Plot starting position
    plot3(trial_start_positions(trial,1), trial_start_positions(trial,2), trial_start_positions(trial,3), ...
        's', 'Color', colors{trial}, 'MarkerSize', 8, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    % Plot target
    plot3(targets(trial, 1), targets(trial, 2), targets(trial, 3), ...
        'o', 'Color', colors{trial}, 'MarkerSize', 10, 'MarkerFaceColor', colors{trial}, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Overlay (Top View)');
view(0, 90);  % Top-down view

% Calculate position and velocity errors
pos_error = sqrt((x_true - R_L1(:,1)').^2 + (y_true - R_L1(:,2)').^2 + (z_true - R_L1(:,3)').^2);
vel_error = sqrt((vx_true - R_L1(:,4)').^2 + (vy_true - R_L1(:,5)').^2 + (vz_true - R_L1(:,6)').^2);

% Plot 5: Position Error by Trial
subplot(2, 4, 5);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    plot(trial_idx, pos_error(trial_idx), 'Color', colors{trial}, 'LineWidth', 1.5, 'DisplayName', sprintf('Trial %d', trial));
end
grid on; xlabel('Time (steps)'); ylabel('Position Error (m)');
title('Position RMSE: Truth vs Learned');
legend('Location', 'best', 'FontSize', 8);

% Plot 6: Velocity Error by Trial
subplot(2, 4, 6);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    plot(trial_idx, vel_error(trial_idx), 'Color', colors{trial}, 'LineWidth', 1.5, 'DisplayName', sprintf('Trial %d', trial));
end
grid on; xlabel('Time (steps)'); ylabel('Velocity Error (m/s)');
title('Velocity RMSE: Truth vs Learned');
legend('Location', 'best', 'FontSize', 8);

% Plot 7: Free Energy Trajectory
subplot(2, 4, 7);
plot(free_energy_all, 'k-', 'LineWidth', 1.5);
grid on; xlabel('Time (steps)'); ylabel('Free Energy');
title('Free Energy Minimization');
xlim([0 N]);

% Plot 8: Trial Indicator
subplot(2, 4, 8);
trial_vals = zeros(1, N);
for trial = 1:n_trials
    trial_vals(phases_indices{trial}) = trial;
end
image(reshape(trial_vals, 1, N));
colorbar; caxis([1 4]);
xlabel('Time (steps)');
title('Trial Label');
set(gca, 'YTickLabel', '');

% Compute summary statistics by trial
pos_rmse_by_trial = [];
vel_rmse_by_trial = [];
reaching_distance_by_trial = [];

for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    pos_rmse_by_trial(trial) = sqrt(mean(pos_error(trial_idx).^2));
    vel_rmse_by_trial(trial) = sqrt(mean(vel_error(trial_idx).^2));
    reaching_distance_by_trial(trial) = reaching_error_all(trial_idx(end));
end

sgtitle(sprintf(['3D Multi-Trial Sensorimotor Reaching - Rao & Ballard Predictive Coding\n', ...
    'Position RMSE by Trial: [%.4f, %.4f, %.4f, %.4f] m | ', ...
    'Velocity RMSE: [%.4f, %.4f, %.4f, %.4f] m/s'], ...
    pos_rmse_by_trial(1), pos_rmse_by_trial(2), pos_rmse_by_trial(3), pos_rmse_by_trial(4), ...
    vel_rmse_by_trial(1), vel_rmse_by_trial(2), vel_rmse_by_trial(3), vel_rmse_by_trial(4)), ...
    'FontSize', 10, 'FontWeight', 'bold');

% ====================================================================
% SAVE FIGURE
% ====================================================================

output_dir = './figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

figure_filename = fullfile(output_dir, '3D_reaching_trajectories.png');
saveas(gcf, figure_filename);
fprintf('\n✓ Figure saved to: %s\n', figure_filename);

% Also save as high-resolution PDF
pdf_filename = fullfile(output_dir, '3D_reaching_trajectories.pdf');
saveas(gcf, pdf_filename);
fprintf('✓ Figure saved to: %s\n', pdf_filename);

% ====================================================================
% ANALYSIS SUMMARY (3D Multi-Trial)
% ====================================================================

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  3D MULTI-TRIAL REACHING - GROUND TRUTH vs LEARNED        ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('POSITION PREDICTIONS (from R_L1(:,1:3)) vs Ground Truth:\n');
fprintf('─────────────────────────────────────────────────────────\n');
overall_pos_rmse = sqrt(mean(pos_error.^2));
fprintf('Overall Position RMSE: %.6f m\n\n', overall_pos_rmse);

fprintf('By Trial:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_pos_error = pos_error(trial_idx);
    trial_rmse = sqrt(mean(trial_pos_error.^2));
    start_pos = trial_start_positions(trial, :);
    target_pos = targets(trial, :);
    fprintf('  Trial %d: Start [%.2f, %.2f, %.2f] → Target [%.2f, %.2f, %.2f]\n', ...
        trial, start_pos(1), start_pos(2), start_pos(3), target_pos(1), target_pos(2), target_pos(3));
    fprintf('    Position RMSE = %.6f m (mean error: %.6f m)\n', trial_rmse, mean(trial_pos_error));
end

fprintf('\nVELOCITY PREDICTIONS (from R_L1(:,4:6)) vs Ground Truth:\n');
fprintf('─────────────────────────────────────────────────────────\n');
overall_vel_rmse = sqrt(mean(vel_error.^2));
fprintf('Overall Velocity RMSE: %.6f m/s\n\n', overall_vel_rmse);

fprintf('By Trial:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_vel_error = vel_error(trial_idx);
    trial_rmse = sqrt(mean(trial_vel_error.^2));
    fprintf('  Trial %d: Velocity RMSE = %.6f m/s (mean error: %.6f m/s)\n', ...
        trial, trial_rmse, mean(trial_vel_error));
end

fprintf('\nFINAL REACHING DISTANCE TO TARGET:\n');
fprintf('─────────────────────────────────────────────────────────\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    final_reaching_dist = reaching_error_all(trial_idx(end));
    initial_reaching_dist = reaching_error_all(trial_idx(1));
    fprintf('  Trial %d: Initial distance = %.6f m, Final distance = %.6f m\n', ...
        trial, initial_reaching_dist, final_reaching_dist);
end

fprintf('\nLEARNING EFFICIENCY:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('Final Free Energy:           %.6e\n', free_energy_all(end));
fprintf('Free Energy Reduction Rate:  %.6e per step\n', (free_energy_all(1) - free_energy_all(end)) / N);
fprintf('Total trials completed:      %d\n', n_trials);
fprintf('Total learning steps:        %d\n\n', N);