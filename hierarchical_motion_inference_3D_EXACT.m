fprintf('╔═════════════════════════════════════════════════════════════╗\n');
fprintf('║  SENSORIMOTOR LEARNING: 3D REACHING & GRASPING             ║\n');
fprintf('║  Learning to Reach Targets in 3D Space via Predictive Coding ║\n');
fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% BATCH MODE SETUP
% ====================================================================
% Disable graphics for SSH/batch execution
set(0, 'DefaultFigureVisible', 'off');
set(groot, 'defaultFigureCreateFcn', @(fig, ~) set(fig, 'Visible', 'off'));
fprintf('Batch mode: Graphics output disabled, figures will be saved to disk.\n\n');

% ====================================================================
% 3D REACHING TASK CONFIGURATION
% ====================================================================

dt = 0.01;              % Time step (s)
T_per_trial = 40;      % Duration per trial (s)
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
fprintf('  Phase 1 (0-%.2fs):  Reach to target [%.2f, %.2f, %.2f]\n', T_per_trial, targets(1,1), targets(1,2), targets(1,3));
fprintf('  Phase 2 (%.2f-%.2fs):  Reach to target [%.2f, %.2f, %.2f]\n', T_per_trial, 2*T_per_trial, targets(2,1), targets(2,2), targets(2,3));
fprintf('  Phase 3 (%.2f-%.2fs):  Reach to target [%.2f, %.2f, %.2f]\n', 2*T_per_trial, 3*T_per_trial, targets(3,1), targets(3,2), targets(3,3));
fprintf('  Phase 4 (%.2f-%.2fs): Reach to target [%.2f, %.2f, %.2f]\n\n', 3*T_per_trial, 4*T_per_trial, targets(4,1), targets(4,2), targets(4,3));

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

eta_rep = 0.005;         % Representation learning rate (increased from 0.001)
eta_W = 0.0005;          % Weight matrix learning rate (increased from 0.0001)
momentum = 0.90;         % Momentum for representation updates (decreased for faster learning)
weight_decay = 0.98;     % L2 regularization on weights (decreased for faster learning)
pi_L1 = 100;             % Precision (reliability) of L1 sensory input
pi_L2 = 10;              % Precision of L2 motor basis
pi_L3 = 1;               % Precision of L3 goal representation

fprintf('LEARNING PARAMETERS:\n');
fprintf('  η_rep = %.6f (representation learning rate)\n', eta_rep);
fprintf('  η_W   = %.6f (weight matrix learning rate)\n', eta_W);
fprintf('  Momentum = %.4f\n', momentum);
fprintf('  Weight Decay = %.4f\n', weight_decay);
fprintf('  π_L1  = %.0f, π_L2  = %.0f, π_L3  = %.0f\n\n', pi_L1, pi_L2, pi_L3);

fprintf('HIERARCHICAL PREDICTIVE CODING WITH ACTIVE INFERENCE:\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('L3 (Goal): ACTIVE INFERENCE NODE\n');
fprintf('  • NOT clamped to task target (varies over time)\n');
fprintf('  • Infers goal from proprioceptive errors (bottom-up)\n');
fprintf('  • Pulled toward task target by soft constraint (top-down)\n');
fprintf('  • Creates continuous prediction errors for learning\n');
fprintf('  • Enables generalization across phase transitions\n\n');
fprintf('L2 (Motor Basis): Learned motor primitives\n');
fprintf('  • Learns to predict motor commands from goal\n');
fprintf('  • Updated via proprioceptive coupling errors\n\n');
fprintf('L1 (Proprioception): Sensory prediction layer\n');
fprintf('  • Predicts position and velocity from motor commands\n');
fprintf('  • Compared against actual sensory input\n');
fprintf('  • Generates error signals that drive learning\n');
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

% Pre-allocate representation matrices for ALL timesteps (CRITICAL!)
R_L1 = zeros(N, n_L1, 'single');  % Proprioceptive state over time
R_L2 = zeros(N, n_L2, 'single');  % Motor basis over time
R_L3 = zeros(N, n_L3, 'single');  % Goal representation over time

% Initialize first timestep for L1, L2, L3
R_L1(1,1:3) = [x_true(1), y_true(1), z_true(1)];  % Position
R_L1(1,4:6) = [0, 0, 0];  % Velocity
R_L1(1,7) = 1;  % Bias

% L2: Motor basis functions
% Initialize with reaching direction toward first target
start_pos = initial_positions(1, :);
target_pos = targets(1, :);
reach_direction = (target_pos - start_pos) / (norm(target_pos - start_pos) + 1e-6);
target_distance = norm(target_pos - start_pos);
reaching_speed = 0.2 * target_distance;  % Scale with distance

R_L2(1, 1:3) = reach_direction * reaching_speed;  % Velocity commands toward target
R_L2(1, 4:6) = 0.01 * randn(1, 3);  % Auxiliary motor channels

% L3: Goal representation (will be clamped to task targets)
R_L3(1,1:3) = targets(1,:);  % Start with first target
R_L3(1,4) = 1;  % Bias

fprintf('INITIAL CONDITIONS (Trial 1):\n');
fprintf('  Start position: [%.2f, %.2f, %.2f]\n', x_true(1), y_true(1), z_true(1));
fprintf('  Target position: [%.2f, %.2f, %.2f]\n', targets(1,1), targets(1,2), targets(1,3));
fprintf('  R_L2(1,:) = reaching velocity [%.4f, %.4f, %.4f] m/s (toward target)\n', R_L2(1,1), R_L2(1,2), R_L2(1,3));
fprintf('  R_L3(1,:) = [target_x=%.2f, target_y=%.2f, target_z=%.2f, bias=1]\n\n', targets(1,1), targets(1,2), targets(1,3));

% Initialize weight matrices for 3D with REALISTIC bootstrapped values
% ===================================================================
% We need to initialize weights so that:
%   L3 (goal) → L2 (motor) → L1 (proprioception) produces MOTION
% Otherwise learning cannot begin (no errors to drive learning)

% Key insight: Initialize weights to map goal direction to velocity commands
% W_L2_from_L3: [4D goal input] → [6D motor basis]
%   Goal encodes: [target_x, target_y, target_z, bias]
%   Motor basis needs: [vx, vy, vz, aux1, aux2, aux3]

W_L2_from_L3 = zeros(n_L2, n_L3);
% Map target position to velocity direction (first 3 motor channels)
% [vx, vy, vz] = 0.2 * [target_x, target_y, target_z]
W_L2_from_L3(1:3, 1:3) = 0.2 * eye(3, 3);  % Direct coupling: target → velocity (increased from 0.1)
% Small random values for auxiliary channels
W_L2_from_L3(4:6, 1:3) = 0.01 * randn(3, 3);
W_L2_from_L3(1:6, 4) = 0.01 * randn(6, 1);  % Coupling from bias

% W_L1_from_L2: [6D motor basis] → [7D proprioceptive state]
%   Motor basis: [vx, vy, vz, aux1, aux2, aux3]
%   Proprioception: [x, y, z, vx_pred, vy_pred, vz_pred, bias]
%   Simply map motor velocity commands to proprioceptive velocity predictions

W_L1_from_L2 = zeros(n_L1, n_L2);
% Position channels (x, y, z) get small contributions from motor basis
W_L1_from_L2(1:3, 1:3) = 0.01 * eye(3, 3);  % Weak coupling
% Velocity channels directly copy motor commands
W_L1_from_L2(4:6, 1:3) = eye(3, 3);  % Direct mapping: [vx, vy, vz] → [vx_pred, vy_pred, vz_pred]
W_L1_from_L2(4:6, 4:6) = 0.1 * randn(3, 3);  % Auxiliary motor channels
W_L1_from_L2(7, :) = 0.01 * randn(1, 6);  % Bias prediction

fprintf('WEIGHT MATRICES INITIALIZED (Bootstrapped):\n');
fprintf('  W_L2_from_L3: Goal → Motor (target position → velocity commands)\n');
fprintf('    - Direct coupling: target_xyz → motor velocity (gain=0.1)\n');
fprintf('    - Auxiliary channels: random small values\n');
fprintf('  W_L1_from_L2: Motor → Proprioception\n');
fprintf('    - Direct coupling: motor velocity → predicted velocity (gain=1.0)\n');
fprintf('    - Position predictions: weak coupling (gain=0.01)\n\n');

% Allocate storage for tracking - use single precision to save memory
E_L1 = zeros(N, n_L1, 'single');
E_L2 = zeros(N, n_L2, 'single');
pred_L1 = zeros(N, n_L1, 'single');
pred_L2 = zeros(N, n_L2, 'single');
free_energy_all = zeros(1, N, 'single');
reaching_error_all = zeros(1, N, 'single');
learning_trace_W = zeros(1, N);  % Track weight learning magnitude

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
    % At the start of each new trial, reinitialize position and motor basis
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
                
                % HARD RESET L3: Jump to new task target (phase transition)
                % This ensures L3 can respond to the new task immediately
                R_L3(i, 1:3) = targets(trial, :);
                R_L3(i, 4) = 1;
                
                % Re-initialize L2 motor basis with reaching direction toward new target
                start_pos = initial_positions(trial, :);
                target_pos = targets(trial, :);
                reach_direction = (target_pos - start_pos) / (norm(target_pos - start_pos) + 1e-6);
                target_distance = norm(target_pos - start_pos);
                reaching_speed = 0.2 * target_distance;  % Scale with distance (0.2 = fraction of distance per second)
                
                R_L2(i, 1:3) = reach_direction * reaching_speed;
                R_L2(i, 4:6) = 0.01 * randn(1, 3);
                
                current_trial = trial;
                fprintf('\n[Trial %d started at step %d - L3 reset to target, L2 reinitialized]\n', trial, i);
                break;
            end
        end
    end
    
    % ==============================================================
    % STEP 0: GOAL REPRESENTATION (ACTIVE INFERENCE - L3)
    % ==============================================================
    % L3 is NOT clamped but inferred from proprioceptive error
    % This allows learning to persist across phase transitions
    
    % Task signal: preferred target for this trial (soft constraint)
    task_target = targets(current_trial, :);
    
    % L3 starts at current value (will be updated based on errors below)
    
    % ==============================================================
    % STEP 1: MOTOR COMMAND GENERATION (Top-Down: L3 → L2 → L1)
    % ==============================================================
    
    % L3 predicts L2 motor commands via learned weights
    pred_L2(i,:) = R_L3(i,:) * W_L2_from_L3';  % [6D motor basis prediction]
    
    % L2 executes: actual motor basis is learned representation
    motor_basis = R_L2(i,:);  % [6D: actual learned motor commands]
    
    % L2 predicts L1 proprioceptive state via learned weights
    pred_L1(i,:) = motor_basis * W_L1_from_L2';  % [7D: predicted proprioception]
    
    % Extract predicted velocities from L1 prediction
    pred_vx = pred_L1(i, 4);
    pred_vy = pred_L1(i, 5);
    pred_vz = pred_L1(i, 6);
    
    % ==============================================================
    % STEP 2: KINEMATICS - MOTOR COMMAND EXECUTION
    % ==============================================================
    
    % Apply motor gain (scales learned velocity prediction to actual command)
    motor_vx(i) = motor_gain * pred_vx;
    motor_vy(i) = motor_gain * pred_vy;
    motor_vz(i) = motor_gain * pred_vz;
    
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
    
    % Sensory input: actual proprioceptive state
    sensory_pos = [x_true(i+1), y_true(i+1), z_true(i+1)];  % Actual 3D position
    sensory_vel = [vx_true(i+1), vy_true(i+1), vz_true(i+1)];  % Actual 3D velocity
    
    % ==============================================================
    % STEP 4: ERROR COMPUTATION (Bottom-Up Predictive Coding)
    % ==============================================================
    
    % L1 errors: sensory input vs prediction (prediction error)
    E_L1(i,1:3) = sensory_pos - pred_L1(i,1:3);  % Position error
    E_L1(i,4:6) = sensory_vel - pred_L1(i,4:6);  % Velocity error
    E_L1(i,7) = 1 - pred_L1(i,7);  % Bias error
    
    % L2 errors: actual motor basis vs L3's prediction
    E_L2(i,:) = motor_basis - pred_L2(i,:);
    
    % Calculate 3D reaching error (distance to current target)
    reaching_error_all(i) = sqrt((x_true(i+1) - targets(current_trial,1))^2 + ...
                                 (y_true(i+1) - targets(current_trial,2))^2 + ...
                                 (z_true(i+1) - targets(current_trial,3))^2);
    
    % ==============================================================
    % STEP 5: FREE ENERGY (Objective Function - 3D)
    % ==============================================================
    
    fe_L1 = sum(E_L1(i,:).^2) / (2 * pi_L1);
    fe_L2 = sum(E_L2(i,:).^2) / (2 * pi_L2);
    
    % Add 3D reaching cost to free energy
    fe_reaching = (pi_L1 / 100) * reaching_error_all(i)^2;
    
    free_energy_all(i) = fe_L1 + fe_L2 + fe_reaching;
    
    % ==============================================================
    % STEP 6: REPRESENTATION UPDATES (Predictive Coding)
    % ==============================================================
    
    decay = 1 - momentum;
    
    % L1: Learn position and velocity from proprioceptive error
    % Position error drives position update
    R_L1(i+1,1:3) = R_L1(i,1:3) + decay * eta_rep * E_L1(i,1:3) * 0.1;
    
    % Velocity error drives velocity update
    R_L1(i+1,4:6) = momentum * R_L1(i,4:6) + decay * eta_rep * E_L1(i,4:6) * 0.1;
    R_L1(i+1,4:6) = max(-2, min(2, R_L1(i+1,4:6)));  % Bounds [-2, 2] m/s
    
    % Clamp position to workspace to prevent unbounded learning
    R_L1(i+1,1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), R_L1(i+1,1)));
    R_L1(i+1,2) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), R_L1(i+1,2)));
    R_L1(i+1,3) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), R_L1(i+1,3)));
    
    R_L1(i+1,7) = 1;  % Bias fixed
    
    % L2: Learn motor basis from motor error and proprioceptive coupling
    % Motor error directly drives motor basis learning
    coupling_from_L1 = E_L1(i,:) * W_L1_from_L2;  % Proprioceptive error → motor error
    norm_W1 = max(0.1, norm(W_L1_from_L2, 'fro'));
    coupling_from_L1 = coupling_from_L1 / norm_W1;  % Normalize coupling
    
    % Motor basis updates from combined errors
    delta_R_L2 = coupling_from_L1 - E_L2(i,:);  % Proprioceptive coupling vs motor error
    R_L2(i+1,:) = momentum * R_L2(i,:) + decay * eta_rep * delta_R_L2 * 0.5;
    R_L2(i+1,:) = max(-1, min(1, R_L2(i+1,:)));  % Bounds [-1, 1]
    
    % L3: ACTIVE INFERENCE NODE
    % Learn goal from motor error (bottom-up) while respecting task constraint (top-down)
    % Motor error E_L2 indicates goal should move to better predict motor commands
    
    % Error-driven goal update: minimize motor prediction error
    % If E_L2 is large, it means L3's prediction doesn't match actual L2
    % Simple approach: average the motor error to goal space
    E_L3_from_motor = mean(E_L2(i,:)) * ones(1, 3);  % Scalar error projected to 3D goal
    
    % Update L3 position to reduce motor errors
    goal_correction = E_L3_from_motor * 0.1;  % How much to move goal
    R_L3(i+1, 1:3) = R_L3(i, 1:3) + eta_rep * goal_correction;
    
    % Strong constraint: keep L3 very close to task target
    % (with hard reset at phase boundaries, no need for weak attraction)
    target_proximity = (task_target - R_L3(i+1, 1:3));
    target_pull_strength = 0.2;  % Stronger pull toward target
    R_L3(i+1, 1:3) = R_L3(i+1, 1:3) + eta_rep * target_proximity * target_pull_strength;
    
    % Keep L3 in workspace bounds
    R_L3(i+1, 1:3) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), R_L3(i+1,1)));
    R_L3(i+1, 2) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), R_L3(i+1,2)));
    R_L3(i+1, 3) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), R_L3(i+1,3)));
    
    R_L3(i+1, 4) = 1;  % Bias fixed
    
    % ==============================================================
    % STEP 7: WEIGHT LEARNING (Hebbian Rule - 3D)
    % ==============================================================
    
    layer_scale_L1 = max(0.1, mean(abs(R_L2(i,:))));
    layer_scale_L2 = max(0.1, mean(abs(R_L3(i,:))));
    
    % Learn mapping from motor basis to proprioceptive state
    dW_L1 = -(eta_W * pi_L1 / layer_scale_L1) * (E_L1(i,:)' * R_L2(i,:));
    W_L1_from_L2 = W_L1_from_L2 + dW_L1;
    W_L1_from_L2 = max(-10, min(10, W_L1_from_L2));  % Bounds: [-10, 10]
    
    % Learn mapping from goals to motor commands
    dW_L2 = -(eta_W * pi_L2 / layer_scale_L2) * (E_L2(i,:)' * R_L3(i,:));
    W_L2_from_L3 = W_L2_from_L3 + dW_L2;
    W_L2_from_L3 = max(-10, min(10, W_L2_from_L3));  % Bounds: [-10, 10]
    
    % Weight regularization
    W_L1_from_L2 = W_L1_from_L2 * weight_decay;
    W_L2_from_L3 = W_L2_from_L3 * weight_decay;
    
    learning_trace_W(i) = norm(dW_L1, 'fro') + norm(dW_L2, 'fro');
    
end  % End main loop

fprintf('\n\n');

% ====================================================================
% SAVE NUMERICAL RESULTS (No Graphics - Batch Mode Optimized)
% ====================================================================

fprintf('Saving results...\n');

output_dir = './figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Calculate position and velocity errors
pos_error = sqrt((x_true - R_L1(:,1)').^2 + (y_true - R_L1(:,2)').^2 + (z_true - R_L1(:,3)').^2);
vel_error = sqrt((vx_true - R_L1(:,4)').^2 + (vy_true - R_L1(:,5)').^2 + (vz_true - R_L1(:,6)').^2);

% Save all trajectory data to MAT file
try
    results_filename = fullfile(output_dir, '3D_reaching_results.mat');
    save(results_filename, 'x_true', 'y_true', 'z_true', 'vx_true', 'vy_true', 'vz_true', ...
        'R_L1', 'R_L2', 'R_L3', 'pos_error', 'vel_error', 'free_energy_all', ...
        'reaching_error_all', 'targets', 'phases_indices', 'trial_start_positions', '-v7.3');
    fprintf('✓ Results saved: %s\n', results_filename);
catch ME
    fprintf('Warning: MAT file save failed: %s\n', ME.message);
end

% Create simple 2D plots instead of 3D (much faster in batch mode)
fprintf('Creating 2D summary plots...\n');

fig = figure('Position', [100, 100, 1400, 800], 'Visible', 'off');

colors = {'r', 'g', 'b', 'm'};

% Plot 1: Position Error Over Time
subplot(2, 3, 1);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    plot(trial_idx, pos_error(trial_idx), 'Color', colors{trial}, 'LineWidth', 2, 'DisplayName', sprintf('Trial %d', trial));
end
grid on; xlabel('Time (steps)'); ylabel('Position Error (m)');
title('Position Error: Truth vs Learned');
legend off;

% Plot 2: Velocity Error Over Time
subplot(2, 3, 2);
hold on;
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    plot(trial_idx, vel_error(trial_idx), 'Color', colors{trial}, 'LineWidth', 2, 'DisplayName', sprintf('Trial %d', trial));
end
grid on; xlabel('Time (steps)'); ylabel('Velocity Error (m/s)');
title('Velocity Error: Truth vs Learned');
legend off;

% Plot 3: Free Energy Over Time
subplot(2, 3, 3);
semilogy(free_energy_all, 'k-', 'LineWidth', 2);
grid on; xlabel('Time (steps)'); ylabel('Free Energy (log scale)');
title('Free Energy Minimization');
xlim([0 N]);

% Plot 4: X Position Over Time
subplot(2, 3, 4);
hold on;
plot(x_true, 'b-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
plot(R_L1(:,1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Learned');
grid on; xlabel('Time (steps)'); ylabel('X Position (m)');
title('X Coordinate Prediction');
legend off;

% Plot 5: Y Position Over Time
subplot(2, 3, 5);
hold on;
plot(y_true, 'b-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
plot(R_L1(:,2), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Learned');
grid on; xlabel('Time (steps)'); ylabel('Y Position (m)');
title('Y Coordinate Prediction');
legend off;

% Plot 6: Z Position Over Time
subplot(2, 3, 6);
hold on;
plot(z_true, 'b-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
plot(R_L1(:,3), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Learned');
grid on; xlabel('Time (steps)'); ylabel('Z Position (m)');
title('Z Coordinate Prediction');
legend off;

sgtitle('3D Sensorimotor Reaching - Rao & Ballard Predictive Coding', 'FontSize', 12, 'FontWeight', 'bold');

% Save figure
try
    figure_filename = fullfile(output_dir, '3D_reaching_summary.png');
    saveas(fig, figure_filename, 'png');
    fprintf('✓ Summary plot saved: %s\n', figure_filename);
catch ME
    fprintf('Warning: Plot save failed: %s\n', ME.message);
end

close(fig);

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