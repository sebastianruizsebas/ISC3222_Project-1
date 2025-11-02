function results = hierarchical_motion_inference_dual_hierarchy(params, make_plots)
    % DUAL-HIERARCHY PREDICTIVE CODING MODEL
    % ========================================================================
    % Motor Region: Learns stable forward models (how commands produce motion)
    % Planning Region: Learns task-specific reaching strategies
    % Task Context (L0): Explicit one-hot encoding of current task (trial)
    %
    % Task: "Player Chasing Moving Ball"
    % - Ball moves continuously with smooth trajectory
    % - Player learns to intercept moving target
    % - Motor region decoupled from target motion (learns stable dynamics)
    % - Planning region learns task-specific interception strategies
    %
    % Usage:
    %   hierarchical_motion_inference_dual_hierarchy()                    % Run with defaults, make plots
    %   hierarchical_motion_inference_dual_hierarchy(params)              % Run with custom params, make plots
    %   hierarchical_motion_inference_dual_hierarchy(params, true)        % Run with custom params, make plots
    %   hierarchical_motion_inference_dual_hierarchy(params, false)       % Run with custom params, NO plots (for optimization)
    
    % Default: make plots unless explicitly disabled
    if nargin < 2
        make_plots = true;
    end

    % Suppress verbose initialization and parameter printouts when running under PSO (parallel optimization)
    % Use the actual argument name `params` (PSO passes a struct named dh_params)
    if ~(exist('params','var') && isstruct(params) && isfield(params,'suppress_init_log') && params.suppress_init_log)
        % ...existing code for printing initialization and parameters...
        % (If you want to see these logs, set dh_params.suppress_init_log = false)
    end

    weight_decay = 0.98;
% --------------------------------------------------------------------
% PARAMETER OVERRIDES, PHYSICS, TIMING, AND TASK SETUP
% (Copied from the reference 'copy' implementation to ensure PSO
%  passes minimal params and the function defines all runtime vars.)

if nargin > 0 && isstruct(params)
    % Override a subset of defaults with provided parameters
    if isfield(params, 'eta_rep'), eta_rep = params.eta_rep; end
    if isfield(params, 'eta_W'), eta_W = params.eta_W; end
    if isfield(params, 'momentum'), momentum = params.momentum; end
    if isfield(params, 'weight_decay'), weight_decay = params.weight_decay; end
    if isfield(params, 'motor_gain'), motor_gain = params.motor_gain; end
    if isfield(params, 'damping'), damping = params.damping; end
    if isfield(params, 'reaching_speed_scale'), reaching_speed_scale = params.reaching_speed_scale; end
    if isfield(params, 'decay_motor'), decay_motor = params.decay_motor; end
    if isfield(params, 'decay_plan'), decay_plan = params.decay_plan; end
    if isfield(params, 'W_plan_gain'), W_plan_gain = params.W_plan_gain; end
    if isfield(params, 'W_motor_gain'), W_motor_gain = params.W_motor_gain; end
    optimizer_mode = true;
else
    optimizer_mode = false;
end

% Default learning rates and parameters (if not already set by params)
if ~exist('eta_rep', 'var'), eta_rep = 0.01; end           % Representation learning rate
if ~exist('eta_W', 'var'), eta_W = 0.001; end              % Weight matrix learning rate
if ~exist('momentum', 'var'), momentum = 0.9; end           % Momentum for learning
if ~exist('weight_decay', 'var'), weight_decay = 0.9; end  % Weight decay per step

% Task-conditional learning parameter
if ~exist('interference_penalty_weight', 'var'), interference_penalty_weight = 0.01; end

% --- Physics parameters (can be provided via params) ---
if nargin > 0 && isstruct(params)
    if isfield(params, 'gravity'), gravity = params.gravity; end
    if isfield(params, 'restitution'), restitution = params.restitution; end
    if isfield(params, 'ground_friction'), ground_friction = params.ground_friction; end
    if isfield(params, 'air_drag'), air_drag = params.air_drag; end
end

% Default physics params (if not provided)
if ~exist('gravity', 'var'), gravity = 9.81; end            % m/s^2 downward
if ~exist('restitution', 'var'), restitution = 0.75; end    % 0..1 bounce energy retained
if ~exist('ground_friction', 'var'), ground_friction = 0.90; end % 0..1 lateral speed retained on bounce
if ~exist('air_drag', 'var'), air_drag = 0.001; end         % small fractional velocity loss per step

% --------------------------------------------------------------------
% Tunable trajectory-generation parameters exposed via `params`
% (provide defaults here so the candidate generator below can reference them)
if nargin > 0 && isstruct(params) && isfield(params, 'vmax_ball')
    vmax_ball = params.vmax_ball;
else
    vmax_ball = 8.0; % m/s, tuneable upper limit for ball initial speed
end
if nargin > 0 && isstruct(params) && isfield(params, 'min_start_sep')
    min_start_sep = params.min_start_sep;
else
    min_start_sep = 0.5; % meters
end
% --------------------------------------------------------------------

% ====================================================================
% TASK CONFIGURATION: PLAYER CHASING MOVING BALL
% ====================================================================

% Timing defaults (can be overridden by params)
dt = 0.01;              % Time step (s)
T_per_trial = 20;      % Duration per trial (s) - smaller default for quicker runs
n_trials = 1;           % Number of different ball trajectories

if nargin > 0 && isstruct(params)
    if isfield(params, 'dt'), dt = params.dt; end
    if isfield(params, 'T_per_trial'), T_per_trial = params.T_per_trial; end
    if isfield(params, 'n_trials'), n_trials = params.n_trials; end
end

T = T_per_trial * n_trials;  % Total duration
t = 0:dt:T;
N = length(t);

% Construct trial phase indices (start:end step indices for each trial)
trial_duration_steps = round(T_per_trial / dt);
phases_indices = cell(n_trials, 1);
for trial = 1:n_trials
    start_idx = (trial - 1) * trial_duration_steps + 1;
    end_idx = min(trial * trial_duration_steps, N);
    phases_indices{trial} = start_idx:end_idx;
end

% Workspace bounds (used for generating player starts and ball trajectories)
workspace_bounds = [
    -10, 10;      % X bounds
    -10, 10;      % Y bounds
    0, 5;        % Z bounds
];

% Initial player positions for each trial
% X and Y are randomized within workspace bounds; Z is fixed to ground
% level (z=0) so the player always starts on the ground and can reach
% bouncing passes more predictably. The z value can be overridden via
% params.player_start_z if desired.
initial_positions = zeros(n_trials, 3);
for trial = 1:n_trials
    % Random X, Y within workspace
    initial_positions(trial, 1) = workspace_bounds(1, 1) + rand() * (workspace_bounds(1, 2) - workspace_bounds(1, 1));
    initial_positions(trial, 2) = workspace_bounds(2, 1) + rand() * (workspace_bounds(2, 2) - workspace_bounds(2, 1));
    % Fixed Z on ground (allow override via params.player_start_z)
    if nargin > 0 && isstruct(params) && isfield(params, 'player_start_z')
        initial_positions(trial, 3) = params.player_start_z;
    else
        initial_positions(trial, 3) = 0;
    end
end

% Enforce player starts on ground plane (z = 0) regardless of any overrides
initial_positions(:,3) = 0;

% FIXED DETERMINISTIC BALL TRAJECTORIES
% Each trial has a predefined, identical trajectory (no randomization)
% This ensures perfect reproducibility across all runs
ball_trajectories = {};

fprintf('GENERATING FIXED BALL TRAJECTORIES (DETERMINISTIC):\n');

% Define four distinct but fixed ball trajectories
% Format: struct with 'start_pos', 'velocity', 'acceleration' (row vectors)

% Trial 1: Ball moving diagonally upward then forward
ball_trajectories{1} = struct(...
    'start_pos', [2.0, 2.0, 1.0], ...
    'velocity', [2.5, 1.5, 1.0], ...
    'acceleration', [0.1, 0.0, 0.0]);

% Trial 2: Ball moving primarily in X direction with some vertical component
ball_trajectories{2} = struct(...
    'start_pos', [2.0, 2.0, 1.0], ...
    'velocity', [3.0, 0.5, 0.5], ...
    'acceleration', [0.0, 0.05, -0.1]);

% Trial 3: Ball moving in Y direction with upward arc
ball_trajectories{3} = struct(...
    'start_pos', [2.0, 2.0, 1.0], ...
    'velocity', [0.5, 2.0, 2.0], ...
    'acceleration', [0.0, 0.1, -0.05]);

% Trial 4: Ball moving in mixed 3D trajectory
ball_trajectories{4} = struct(...
    'start_pos', [2.0, 2.0, 1.0], ...
    'velocity', [2.0, 2.0, 1.0], ...
    'acceleration', [0.05, -0.05, 0.0]);

fprintf('✓ Fixed ball trajectories defined for all trials (DETERMINISTIC):\n');
for trial = 1:min(n_trials, numel(ball_trajectories))
    fprintf('  Trial %d: start=[%.1f, %.1f, %.1f], v=[%.1f, %.1f, %.1f], a=[%.3f, %.3f, %.3f]\n', ...
        trial, ...
        ball_trajectories{trial}.start_pos(1), ball_trajectories{trial}.start_pos(2), ball_trajectories{trial}.start_pos(3), ...
        ball_trajectories{trial}.velocity(1), ball_trajectories{trial}.velocity(2), ball_trajectories{trial}.velocity(3), ...
        ball_trajectories{trial}.acceleration(1), ball_trajectories{trial}.acceleration(2), ball_trajectories{trial}.acceleration(3));
end
fprintf('\n');

% Ensure ball starts are sufficiently far from player initial positions
% Default minimum separation (meters)
min_start_sep = 0.5;
if nargin > 0 && isstruct(params) && isfield(params, 'min_start_sep')
    min_start_sep = params.min_start_sep;
end

for trial = 1:n_trials
    player_pos = initial_positions(trial, :);
    start_pos = ball_trajectories{trial}.start_pos;
    sep = norm(start_pos - player_pos);
    attempts = 0;
    while sep < min_start_sep && attempts < 100
        start_pos = [workspace_bounds(1,1) + rand()*(workspace_bounds(1,2)-workspace_bounds(1,1)), ...
                     workspace_bounds(2,1) + rand()*(workspace_bounds(2,2)-workspace_bounds(2,1)), ...
                     workspace_bounds(3,1) + rand()*(workspace_bounds(3,2)-workspace_bounds(3,1))];
        ball_trajectories{trial}.start_pos = start_pos;
        sep = norm(start_pos - player_pos);
        attempts = attempts + 1;
    end
    if sep < min_start_sep
        % fallback: place ball on the boundary of required separation
        dir = randn(1,3); dir = dir / (norm(dir)+1e-9);
        ball_trajectories{trial}.start_pos = player_pos + dir * min_start_sep;
    end
end

% Layer dimensions (needed later when initializing representations)
% NOTE: scale_factor controls how much to enlarge internal layers.
scale_factor = 2.0;  % 200% -> 2x

n_L0 = n_trials;        % One-hot encoding: which trial/task is active
n_L1_motor = 7;         % keep L1 semantics [x,y,z,vx,vy,vz,bias] unchanged
% scale internal layers (L2/L3) by factor (round to integer, at least 1)
n_L2_motor = max(1, round(scale_factor * 6));
n_L3_motor = max(1, round(scale_factor * 3));

n_L1_plan = 7;          % keep L1 planning semantics unchanged
n_L2_plan = max(1, round(scale_factor * 6));
n_L3_plan = max(1, round(scale_factor * 3));

% Semantic indices for L1 (position, velocity, bias). Placing these here
% ensures idx_* are available before representations are initialized.
n_pos = 3; n_vel = 3; n_bias = 1;
idx_pos = 1:n_pos;
idx_vel = n_pos + (1:n_vel);
idx_bias = n_pos + n_vel + 1;

% (Semantic L1 indices are defined above near the layer-dimension block)

% Initialize runtime arrays (positions, velocities, motors)
x_player = zeros(1, N); y_player = zeros(1, N); z_player = zeros(1, N);
vx_player = zeros(1, N); vy_player = zeros(1, N); vz_player = zeros(1, N);

x_ball = zeros(1, N); y_ball = zeros(1, N); z_ball = zeros(1, N);
vx_ball = zeros(1, N); vy_ball = zeros(1, N); vz_ball = zeros(1, N);

motor_vx_motor = zeros(1, N); motor_vy_motor = zeros(1, N); motor_vz_motor = zeros(1, N);
motor_vx_plan = zeros(1, N); motor_vy_plan = zeros(1, N); motor_vz_plan = zeros(1, N);

% Motor dynamics defaults
if ~exist('motor_gain', 'var'), motor_gain = 0.5; end
if ~exist('damping', 'var'), damping = 0.85; end
if ~exist('reaching_speed_scale', 'var'), reaching_speed_scale = 0.5; end

% End insertion of runtime/task defaults
% --------------------------------------------------------------------

% NEW: Separate decay rates for motor vs. planning regions
if ~exist('decay_motor', 'var')
    decay_motor = 0.95;  % Motor: preserve across tasks (95% retained)
end
if ~exist('decay_plan', 'var')
    decay_plan = 0.70;   % Planning: forget old targets (70% retained)
end

% Weight initialization gains
if ~exist('W_motor_gain', 'var')
    W_motor_gain = 0.5;  % Motor weight initialization
end
if ~exist('W_plan_gain', 'var')
    W_plan_gain = 0.5;   % Planning weight initialization
end

pi_L1_motor = 100;       % Proprioceptive precision
pi_L2_motor = 10;        % Motor basis precision
pi_L3_motor = 1;         % Motor output precision

pi_L1_plan = 100;        % Planning goal precision
pi_L2_plan = 10;         % Planning policy precision
pi_L3_plan = 1;          % Planning output precision

% Keep base/reference precision values for adaptive updates
pi_L1_motor_base = pi_L1_motor;
pi_L2_motor_base = pi_L2_motor;
pi_L1_plan_base = pi_L1_plan;
pi_L2_plan_base = pi_L2_plan;

% ====================================================================
% ADAPTIVE PRECISION PARAMETERS (NEW - Error-Driven Precision Scaling)
% ====================================================================
% Prediction-error-driven precision: precision_new = precision_old * exp(alpha * error_magnitude)
% Higher error → higher precision (tighter bounds on predictions)
% Lower error → lower precision (allow flexibility/exploration)

if nargin > 0 && isstruct(params)
    if isfield(params, 'alpha_precision_gain')
        alpha_precision_gain = params.alpha_precision_gain;
    else
        alpha_precision_gain = 0.5;  % Default sensitivity to error magnitude
    end
    if isfield(params, 'pi_L1_motor_min')
        pi_L1_motor_min = params.pi_L1_motor_min;
    else
        pi_L1_motor_min = 10;   % Minimum precision (allow exploration)
    end
    if isfield(params, 'pi_L1_motor_max')
        pi_L1_motor_max = params.pi_L1_motor_max;
    else
        pi_L1_motor_max = 500;  % Maximum precision (tight bounds)
    end
    if isfield(params, 'pi_L2_motor_min')
        pi_L2_motor_min = params.pi_L2_motor_min;
    else
        pi_L2_motor_min = 1;
    end
    if isfield(params, 'pi_L2_motor_max')
        pi_L2_motor_max = params.pi_L2_motor_max;
    else
        pi_L2_motor_max = 100;
    end
    if isfield(params, 'pi_L1_plan_min')
        pi_L1_plan_min = params.pi_L1_plan_min;
    else
        pi_L1_plan_min = 10;
    end
    if isfield(params, 'pi_L1_plan_max')
        pi_L1_plan_max = params.pi_L1_plan_max;
    else
        pi_L1_plan_max = 500;
    end
    if isfield(params, 'pi_L2_plan_min')
        pi_L2_plan_min = params.pi_L2_plan_min;
    else
        pi_L2_plan_min = 1;
    end
    if isfield(params, 'pi_L2_plan_max')
        pi_L2_plan_max = params.pi_L2_plan_max;
    else
        pi_L2_plan_max = 100;
    end
else
    % Defaults (no PSO optimization)
    alpha_precision_gain = 0.5;
    pi_L1_motor_min = 10;    pi_L1_motor_max = 500;
    pi_L2_motor_min = 1;     pi_L2_motor_max = 100;
    pi_L1_plan_min = 10;     pi_L1_plan_max = 500;
    pi_L2_plan_min = 1;      pi_L2_plan_max = 100;
end

% Store bounds in P struct for hierarchical_step_update helper
P_pi_bounds = struct();
P_pi_bounds.L1_motor = [pi_L1_motor_min, pi_L1_motor_max];
P_pi_bounds.L2_motor = [pi_L2_motor_min, pi_L2_motor_max];
P_pi_bounds.L1_plan = [pi_L1_plan_min, pi_L1_plan_max];
P_pi_bounds.L2_plan = [pi_L2_plan_min, pi_L2_plan_max];

    if ~(exist('params','var') && isstruct(params) && isfield(params,'save_results') && params.save_results == false)
        fprintf('LEARNING PARAMETERS:\n');
        fprintf('  eta_rep = %.6f (representation learning rate)\n', eta_rep);
        fprintf('  eta_W   = %.6f (weight matrix learning rate)\n', eta_W);
        fprintf('  Momentum = %.4f\n', momentum);
        fprintf('  Weight Decay (per-step) = %.4f\n', weight_decay);
        fprintf('  Decay at Phase (Motor) = %.4f (95%%-98%% retained)\n', decay_motor);
        fprintf('  Decay at Phase (Planning) = %.4f (70%%-80%% retained)\n', decay_plan);
        fprintf('  pi_motor   = [%.0f, %.0f, %.0f]\n', pi_L1_motor, pi_L2_motor, pi_L3_motor);
        fprintf('  pi_plan    = [%.0f, %.0f, %.0f]\n', pi_L1_plan, pi_L2_plan, pi_L3_plan);
        fprintf('\n  ADAPTIVE PRECISION (Error-Driven):\n');
        fprintf('    alpha_precision_gain = %.4f (sensitivity to error magnitude)\n', alpha_precision_gain);
        fprintf('    pi_L1_motor bounds: [%.1f, %.1f]\n', pi_L1_motor_min, pi_L1_motor_max);
        fprintf('    pi_L2_motor bounds: [%.1f, %.1f]\n', pi_L2_motor_min, pi_L2_motor_max);
        fprintf('    pi_L1_plan bounds:  [%.1f, %.1f]\n', pi_L1_plan_min, pi_L1_plan_max);
        fprintf('    pi_L2_plan bounds:  [%.1f, %.1f]\n\n', pi_L2_plan_min, pi_L2_plan_max);
    end

% ====================================================================
% INITIALIZE REPRESENTATIONS
% ====================================================================

% Task context (L0) - one-hot encoding
R_L0 = zeros(N, n_L0, 'single');

% Motor region representations
R_L1_motor = zeros(N, n_L1_motor, 'single');
R_L2_motor = zeros(N, n_L2_motor, 'single');
R_L3_motor = zeros(N, n_L3_motor, 'single');

% Planning region representations
R_L1_plan = zeros(N, n_L1_plan, 'single');
R_L2_plan = zeros(N, n_L2_plan, 'single');
R_L3_plan = zeros(N, n_L3_plan, 'single');

% Initialize first timestep
% Task context: Trial 1 active
R_L0(1, 1) = 1;

% Player initial state
x_player(1) = initial_positions(1, 1);
y_player(1) = initial_positions(1, 2);
z_player(1) = initial_positions(1, 3);

% Motor L1 (proprioception) - use semantic indices for positions/vel/bias
R_L1_motor(1, idx_pos) = [x_player(1), y_player(1), z_player(1)];
% velocity channels (pad/truncate to fit)
tmp_vel_init = zeros(1, numel(idx_vel)); tmp_vel_init(1:min(3,numel(tmp_vel_init))) = 0;
R_L1_motor(1, idx_vel) = tmp_vel_init;
R_L1_motor(1, idx_bias) = 1;

% Ball initial state
x_ball(1) = ball_trajectories{1}.start_pos(1);
y_ball(1) = ball_trajectories{1}.start_pos(2);
z_ball(1) = ball_trajectories{1}.start_pos(3);
vx_ball(1) = ball_trajectories{1}.velocity(1);
vy_ball(1) = ball_trajectories{1}.velocity(2);
vz_ball(1) = ball_trajectories{1}.velocity(3);

% Motor L2/L3: initial velocity commands
reach_direction = ([x_ball(1), y_ball(1), z_ball(1)] - [x_player(1), y_player(1), z_player(1)]) / ...
                   (norm([x_ball(1), y_ball(1), z_ball(1)] - [x_player(1), y_player(1), z_player(1)]) + 1e-6);
target_distance = norm([x_ball(1), y_ball(1), z_ball(1)] - [x_player(1), y_player(1), z_player(1)]);
reaching_speed = reaching_speed_scale * target_distance;

R_L2_motor(1, 1:3) = reach_direction * reaching_speed;
R_L2_motor(1, 4:6) = 0.01 * randn(1, 3);
R_L3_motor(1, 1:3) = reach_direction * reaching_speed;

% Planning L1: ball position + goal (use semantic idx)
R_L1_plan(1, idx_pos) = [x_ball(1), y_ball(1), z_ball(1)];  % Ball position
% planning velocity/goal channels (use ball pos as initial goal for available vel slots)
tmp_goal = zeros(1, numel(idx_vel));
vals = [x_ball(1), y_ball(1), z_ball(1)];
ncopy = min(3, numel(tmp_goal));
tmp_goal(1:ncopy) = vals(1:ncopy);
R_L1_plan(1, idx_vel) = tmp_goal;
R_L1_plan(1, idx_bias) = 1;

% Planning L2/L3: initial policies
R_L2_plan(1, 1:3) = reach_direction * reaching_speed;
R_L2_plan(1, 4:6) = 0.01 * randn(1, 3);
R_L3_plan(1, 1:3) = reach_direction * reaching_speed;

fprintf('INITIAL CONDITIONS (Trial 1):\n');
fprintf('  Player start: [%.2f, %.2f, %.2f]\n', x_player(1), y_player(1), z_player(1));
fprintf('  Ball start:   [%.2f, %.2f, %.2f]\n', x_ball(1), y_ball(1), z_ball(1));
fprintf('  Initial reach direction: [%.4f, %.4f, %.4f], speed: %.4f m/s\n', ...
    reach_direction(1), reach_direction(2), reach_direction(3), reaching_speed);
fprintf('  R_L1_motor (proprioception) initialized\n');
fprintf('  R_L1_plan (ball + goal) initialized\n\n');

% ====================================================================
% INITIALIZE WEIGHT MATRICES - DUAL HIERARCHY (TASK-INDEXED)
% ====================================================================
% NEW: Maintain separate weight matrices per task to prevent interference learning.
% Each task gets its own learned mappings, reducing catastrophic forgetting.

% Motor region weights: task-indexed (one copy per trial)
W_motor_L2_to_L1 = cell(n_trials, 1);  % task-specific motor basis->proprioception
W_motor_L3_to_L2 = cell(n_trials, 1);  % task-specific output->motor basis
W_motor_L1_lat = cell(n_trials, 1);    % task-specific lateral motor L1
W_motor_L2_lat = cell(n_trials, 1);    % task-specific lateral motor L2
W_motor_L3_lat = cell(n_trials, 1);    % task-specific lateral motor L3

% Planning region weights: task-indexed
W_plan_L2_to_L1 = cell(n_trials, 1);   % task-specific policy->goal
W_plan_L3_to_L2 = cell(n_trials, 1);   % task-specific output->policy
W_plan_L1_lat = cell(n_trials, 1);     % task-specific lateral planning L1
W_plan_L2_lat = cell(n_trials, 1);     % task-specific lateral planning L2
W_plan_L3_lat = cell(n_trials, 1);     % task-specific lateral planning L3

% Initialize per-task weight matrices
for task_idx = 1:n_trials
    % --- Motor L2->L1 mappings (basis to proprioception) ---
    W_motor_L2_to_L1{task_idx} = zeros(n_L1_motor, n_L2_motor);
    
    % Map L3_motor (velocity-like outputs) into L1 velocity rows.
    map_vel = min(n_vel, n_L3_motor);
    W_motor_L2_to_L1{task_idx}(idx_vel(1:map_vel), 1:map_vel) = eye(map_vel);
    
    % Weak position coupling from same L3 channels (if available)
    map_pos = min(n_pos, n_L3_motor);
    W_motor_L2_to_L1{task_idx}(idx_pos(1:map_pos), 1:map_pos) = 0.01 * eye(map_pos);
    
    % Bias / offset row -- small random init
    W_motor_L2_to_L1{task_idx}(idx_bias, :) = 0.01 * randn(1, n_L2_motor);
    
    % --- Motor L3->L2 mappings (output to basis) ---
    W_motor_L3_to_L2{task_idx} = zeros(n_L2_motor, n_L3_motor);
    
    % Initialize L3->L2 mapping with structured identity on the overlapping block
    map_block = min(n_L2_motor, n_L3_motor);
    fan_in3 = max(1, n_L3_motor);
    W_motor_L3_to_L2{task_idx}(1:map_block, 1:map_block) = W_motor_gain * eye(map_block);
    % remaining rows (if any) get small random init scaled by fan-in
    if n_L2_motor > map_block
        W_motor_L3_to_L2{task_idx}(map_block+1:end, 1:n_L3_motor) = (W_motor_gain / sqrt(fan_in3)) * 0.01 * randn(n_L2_motor-map_block, n_L3_motor);
    end
    
    % --- Motor lateral weights ---
    rng(task_idx);  % reproducible per-task lateral init
    W_motor_L1_lat{task_idx} = 0.01 * randn(n_L1_motor, n_L1_motor);
    W_motor_L2_lat{task_idx} = 0.01 * randn(n_L2_motor, n_L2_motor);
    W_motor_L3_lat{task_idx} = 0.01 * randn(n_L3_motor, n_L3_motor);
    
    % Remove strong self-connections
    W_motor_L1_lat{task_idx}(1:size(W_motor_L1_lat{task_idx},1)+1:end) = 0;
    W_motor_L2_lat{task_idx}(1:size(W_motor_L2_lat{task_idx},1)+1:end) = 0;
    W_motor_L3_lat{task_idx}(1:size(W_motor_L3_lat{task_idx},1)+1:end) = 0;
    
    % --- Planning L2->L1 mappings (policy to goal) ---
    W_plan_L2_to_L1{task_idx} = zeros(n_L1_plan, n_L2_plan);
    
    map_vel_p = min(n_vel, n_L3_plan);
    map_pos_p = min(n_pos, n_L3_plan);
    W_plan_L2_to_L1{task_idx}(idx_pos(1:map_pos_p), 1:map_pos_p) = 0.01 * eye(map_pos_p);
    W_plan_L2_to_L1{task_idx}(idx_vel(1:map_vel_p), 1:map_vel_p) = 0.1 * eye(map_vel_p);
    W_plan_L2_to_L1{task_idx}(idx_bias, :) = 0.01 * randn(1, n_L2_plan);
    
    % --- Planning L3->L2 mappings ---
    W_plan_L3_to_L2{task_idx} = zeros(n_L2_plan, n_L3_plan);
    
    map_block_p = min(n_L2_plan, n_L3_plan);
    fan_in3_p = max(1, n_L3_plan);
    W_plan_L3_to_L2{task_idx}(1:map_block_p, 1:map_block_p) = W_plan_gain * eye(map_block_p);
    if n_L2_plan > map_block_p
        W_plan_L3_to_L2{task_idx}(map_block_p+1:end, 1:n_L3_plan) = (W_plan_gain / sqrt(fan_in3_p)) * 0.01 * randn(n_L2_plan-map_block_p, n_L3_plan);
    end
    
    % --- Planning lateral weights ---
    rng(task_idx + n_trials);  % different seed for planning
    W_plan_L1_lat{task_idx} = 0.01 * randn(n_L1_plan, n_L1_plan);
    W_plan_L2_lat{task_idx} = 0.01 * randn(n_L2_plan, n_L2_plan);
    W_plan_L3_lat{task_idx} = 0.01 * randn(n_L3_plan, n_L3_plan);
    
    % Remove strong self-connections
    W_plan_L1_lat{task_idx}(1:size(W_plan_L1_lat{task_idx},1)+1:end) = 0;
    W_plan_L2_lat{task_idx}(1:size(W_plan_L2_lat{task_idx},1)+1:end) = 0;
    W_plan_L3_lat{task_idx}(1:size(W_plan_L3_lat{task_idx},1)+1:end) = 0;
end


fprintf('WEIGHT MATRICES INITIALIZED (TASK-INDEXED):\n');
fprintf('  Motor Region: %d separate task copies\n', n_trials);
fprintf('    W_motor_L2_to_L1{task}: Motor Basis → Proprioception (%dx%d each)\n', n_L1_motor, n_L2_motor);
fprintf('    W_motor_L3_to_L2{task}: Output → Basis (%dx%d each)\n', n_L2_motor, n_L3_motor);
fprintf('  Planning Region: %d separate task copies\n', n_trials);
fprintf('    W_plan_L2_to_L1{task}: Policies → Goal State (%dx%d each)\n', n_L1_plan, n_L2_plan);
fprintf('    W_plan_L3_to_L2{task}: Output → Policies (%dx%d each)\n\n', n_L2_plan, n_L3_plan);

% ====================================================================
% ERROR AND LEARNING TRACKING
% ====================================================================

E_L1_motor = zeros(N, n_L1_motor, 'single');
E_L2_motor = zeros(N, n_L2_motor, 'single');
pred_L1_motor = zeros(N, n_L1_motor, 'single');
pred_L2_motor = zeros(N, n_L2_motor, 'single');

E_L1_plan = zeros(N, n_L1_plan, 'single');
E_L2_plan = zeros(N, n_L2_plan, 'single');
pred_L1_plan = zeros(N, n_L1_plan, 'single');
pred_L2_plan = zeros(N, n_L2_plan, 'single');

free_energy_all = zeros(1, N, 'single');
interception_error_all = zeros(1, N, 'single');
learning_trace_W = zeros(1, N);

% Traces for dynamic precision (for offline inspection)
pi_trace_L1_motor = zeros(1, N);
pi_trace_L2_motor = zeros(1, N);
pi_trace_L1_plan  = zeros(1, N);
pi_trace_L2_plan  = zeros(1, N);

% Additional diagnostic traces and smoothing params for precision updates
pi_raw_trace_L1_motor = zeros(1, N);
pi_raw_trace_L2_motor = zeros(1, N);
pi_raw_trace_L1_plan  = zeros(1, N);
pi_raw_trace_L2_plan  = zeros(1, N);
denom_trace_L1_motor = zeros(1, N);
denom_trace_L2_motor = zeros(1, N);
denom_trace_L1_plan  = zeros(1, N);
denom_trace_L2_plan  = zeros(1, N);

% Smoothing and (tighter) step‑limiting for precision updates to avoid sudden spikes
% Make precision updates more conservative: slower smoothing and smaller per‑step multiplicative changes
pi_smooth_alpha = 0.999;        % stronger smoothing (closer to 1 => much slower changes)
pi_max_step_ratio = 1.05;       % tighter max allowed multiplicative change per step (~5%)

% Dynamic precision histories
window_size = 100;
L1_motor_error_history = [];
L2_motor_error_history = [];
L1_plan_error_history = [];
L2_plan_error_history = [];

% NEW: Per-task error tracking for interference monitoring
% task_errors_motor(i, t) = error of motor predictions IF task t were active at step i
% task_errors_plan(i, t) = error of planning predictions IF task t were active at step i
task_errors_motor = zeros(N, n_trials, 'single');
task_errors_plan = zeros(N, n_trials, 'single');

% ====================================================================
% MAIN LEARNING LOOP - DUAL HIERARCHY
% ====================================================================

fprintf('Running dual-hierarchy learning with player chasing moving ball...\n');
fprintf('Total iterations: %d (dt=%.4fs per step, ~%.1f seconds estimated)\n', N-1, dt, (N-1)*dt);

current_trial = 1;

%---------------------------------------------------------------------
% Prepare S (state) and P (parameters) structs for step helper
%---------------------------------------------------------------------
S = struct();
% runtime arrays
S.x_player = x_player; S.y_player = y_player; S.z_player = z_player;
S.vx_player = vx_player; S.vy_player = vy_player; S.vz_player = vz_player;
S.x_ball = x_ball; S.y_ball = y_ball; S.z_ball = z_ball;
S.vx_ball = vx_ball; S.vy_ball = vy_ball; S.vz_ball = vz_ball;
S.motor_vx_motor = motor_vx_motor; S.motor_vy_motor = motor_vy_motor; S.motor_vz_motor = motor_vz_motor;
S.motor_vx_plan = motor_vx_plan; S.motor_vy_plan = motor_vy_plan; S.motor_vz_plan = motor_vz_plan;

% Representations
S.R_L0 = R_L0;
S.R_L1_motor = R_L1_motor; S.R_L2_motor = R_L2_motor; S.R_L3_motor = R_L3_motor;
S.R_L1_plan = R_L1_plan; S.R_L2_plan = R_L2_plan; S.R_L3_plan = R_L3_plan;

% Predictions and errors
S.pred_L1_motor = pred_L1_motor; S.pred_L2_motor = pred_L2_motor;
S.pred_L1_plan = pred_L1_plan; S.pred_L2_plan = pred_L2_plan;
S.E_L1_motor = E_L1_motor; S.E_L2_motor = E_L2_motor;
S.E_L1_plan = E_L1_plan; S.E_L2_plan = E_L2_plan;

% Weight matrices (task-indexed cells)
S.W_motor_L2_to_L1 = W_motor_L2_to_L1; S.W_motor_L3_to_L2 = W_motor_L3_to_L2;
S.W_plan_L2_to_L1 = W_plan_L2_to_L1; S.W_plan_L3_to_L2 = W_plan_L3_to_L2;
S.W_motor_L1_lat = W_motor_L1_lat; S.W_motor_L2_lat = W_motor_L2_lat; S.W_motor_L3_lat = W_motor_L3_lat;
S.W_plan_L1_lat = W_plan_L1_lat; S.W_plan_L2_lat = W_plan_L2_lat; S.W_plan_L3_lat = W_plan_L3_lat;

% Learning traces
S.free_energy_all = free_energy_all; S.interception_error_all = interception_error_all;
S.learning_trace_W = learning_trace_W;

% Per-task error tracking (for interference monitoring)
S.task_errors_motor = task_errors_motor;
S.task_errors_plan = task_errors_plan;

% Precision traces and raw/denom traces
S.pi_trace_L1_motor = pi_trace_L1_motor; S.pi_trace_L2_motor = pi_trace_L2_motor;
S.pi_trace_L1_plan = pi_trace_L1_plan; S.pi_trace_L2_plan = pi_trace_L2_plan;
S.pi_raw_trace_L1_motor = pi_raw_trace_L1_motor; S.pi_raw_trace_L2_motor = pi_raw_trace_L2_motor;
S.pi_raw_trace_L1_plan = pi_raw_trace_L1_plan; S.pi_raw_trace_L2_plan = pi_raw_trace_L2_plan;
S.denom_trace_L1_motor = denom_trace_L1_motor; S.denom_trace_L2_motor = denom_trace_L2_motor;
S.denom_trace_L1_plan = denom_trace_L1_plan; S.denom_trace_L2_plan = denom_trace_L2_plan;

% Dynamic precision state
% IMPORTANT: Clip initial precision values to be within bounds before simulation starts
S.pi_L1_motor = max(P_pi_bounds.L1_motor(1), min(P_pi_bounds.L1_motor(2), pi_L1_motor));
S.pi_L2_motor = max(P_pi_bounds.L2_motor(1), min(P_pi_bounds.L2_motor(2), pi_L2_motor));
S.pi_L1_plan = max(P_pi_bounds.L1_plan(1), min(P_pi_bounds.L1_plan(2), pi_L1_plan));
S.pi_L2_plan = max(P_pi_bounds.L2_plan(1), min(P_pi_bounds.L2_plan(2), pi_L2_plan));
S.pi_L1_motor_base = S.pi_L1_motor;
S.pi_L2_motor_base = S.pi_L2_motor;
S.pi_L1_plan_base = S.pi_L1_plan;
S.pi_L2_plan_base = S.pi_L2_plan;
% L3 precisions (fixed small values)
S.pi_L3_motor = pi_L3_motor; S.pi_L3_plan = pi_L3_plan;
S.pi_L3_motor_base = pi_L3_motor; S.pi_L3_plan_base = pi_L3_plan;

% Error histories
S.L1_motor_error_history = L1_motor_error_history; S.L2_motor_error_history = L2_motor_error_history;
S.L1_plan_error_history = L1_plan_error_history; S.L2_plan_error_history = L2_plan_error_history;

% CLIPPING COUNTER (Nov 1, 2025): Track how many Inf/NaN clipping events occur
S.clipping_count = 0;  % Counter incremented each time clipping is triggered

% Misc
S.current_trial = current_trial;
S.phases_indices = phases_indices;
S.ball_trajectories = ball_trajectories;

%---------------------------------------------------------------------
% Parameter struct (constants passed to step helper)
%---------------------------------------------------------------------
P = struct();
P.dt = dt; P.gravity = gravity; P.restitution = restitution; P.ground_friction = ground_friction; P.air_drag = air_drag;
P.workspace_bounds = workspace_bounds; P.motor_gain = motor_gain; P.damping = damping; P.reaching_speed_scale = reaching_speed_scale;
P.eta_rep = eta_rep; P.eta_W = eta_W; P.momentum = momentum; P.weight_decay = weight_decay;
P.decay_motor = decay_motor; P.decay_plan = decay_plan; P.W_plan_gain = W_plan_gain; P.W_motor_gain = W_motor_gain;
P.pi_smooth_alpha = pi_smooth_alpha; P.pi_max_step_ratio = pi_max_step_ratio; P.window_size = window_size;
% Pass semantic indices to helper so it can be agnostic to L1 sizing
P.idx_pos = idx_pos; P.idx_vel = idx_vel; P.idx_bias = idx_bias;
% Pass adaptive precision parameters to helper
P.alpha_precision_gain = alpha_precision_gain;
P.pi_bounds = P_pi_bounds;  % Bounds for dynamic precision updates
% FIX (Nov 2, 2025): Add missing interference_penalty_weight parameter
P.interference_penalty_weight = interference_penalty_weight;  % Cross-task error penalty weight

% VALIDATION: Ensure precision bounds are sensible (min < max)
if ~isfinite(P.alpha_precision_gain) || P.alpha_precision_gain <= 0
    P.alpha_precision_gain = 0.5;  % Reset to default if invalid
end

% Check if bounds are valid - if any min >= max, use defaults
if P.pi_bounds.L1_motor(1) >= P.pi_bounds.L1_motor(2) || P.pi_bounds.L2_motor(1) >= P.pi_bounds.L2_motor(2) || ...
   P.pi_bounds.L1_plan(1) >= P.pi_bounds.L1_plan(2) || P.pi_bounds.L2_plan(1) >= P.pi_bounds.L2_plan(2)
    % BOUNDS INVALID - reset to defaults
    P.pi_bounds.L1_motor = [10, 500];
    P.pi_bounds.L2_motor = [0.5, 50];
    P.pi_bounds.L1_plan = [10, 500];
    P.pi_bounds.L2_plan = [0.5, 50];
end

% Termination distance: when player is within this distance of ball the session ends
P.termination_distance = 0.15;
if nargin > 0 && isstruct(params) && isfield(params, 'termination_distance')
    P.termination_distance = params.termination_distance;
end

% Ground plane override: prefer explicit params.ground_z if given, otherwise use workspace lower bound
if nargin > 0 && isstruct(params) && isfield(params, 'ground_z')
    P.ground_z = params.ground_z;
else
    P.ground_z = workspace_bounds(3,1);
end

for i = 1:N-1
    if mod(i, 100) == 0, fprintf('.'); end
    
    % ==============================================================
    % CHECK FOR TRIAL TRANSITION
    % ==============================================================
    if i > 1
        for trial = 2:n_trials
            if i == phases_indices{trial}(1)
                % Reset player position and ball trajectory for new trial (write into S so helper uses authoritative state)
                S.x_player(i) = initial_positions(trial, 1);
                S.y_player(i) = initial_positions(trial, 2);
                S.z_player(i) = initial_positions(trial, 3);
                S.vx_player(i) = 0;
                S.vy_player(i) = 0;
                S.vz_player(i) = 0;

                % Reset ball for new trial (write into S)
                S.x_ball(i) = ball_trajectories{trial}.start_pos(1);
                S.y_ball(i) = ball_trajectories{trial}.start_pos(2);
                S.z_ball(i) = ball_trajectories{trial}.start_pos(3);
                S.vx_ball(i) = ball_trajectories{trial}.velocity(1);
                S.vy_ball(i) = ball_trajectories{trial}.velocity(2);
                S.vz_ball(i) = ball_trajectories{trial}.velocity(3);
                
                % Update task context (L0) in S
                S.R_L0(i, :) = 0;
                S.R_L0(i, trial) = 1;
                
                % Reset motor region representations (write into S using semantic L1 indices)
                S.R_L1_motor(i, idx_pos) = [S.x_player(i), S.y_player(i), S.z_player(i)];
                tmpv = zeros(1, numel(idx_vel)); tmpv(1:min(3,numel(tmpv))) = 0;
                S.R_L1_motor(i, idx_vel) = tmpv;
                S.R_L1_motor(i, idx_bias) = 1;
                
                reach_direction = ([S.x_ball(i), S.y_ball(i), S.z_ball(i)] - [S.x_player(i), S.y_player(i), S.z_player(i)]) / ...
                                   (norm([S.x_ball(i), S.y_ball(i), S.z_ball(i)] - [S.x_player(i), S.y_player(i), S.z_player(i)]) + 1e-6);
                target_distance = norm([S.x_ball(i), S.y_ball(i), S.z_ball(i)] - [S.x_player(i), S.y_player(i), S.z_player(i)]);
                reaching_speed = reaching_speed_scale * target_distance;
                
                S.R_L2_motor(i, 1:3) = reach_direction * reaching_speed;
                S.R_L2_motor(i, 4:6) = 0.01 * randn(1, 3);
                S.R_L3_motor(i, 1:3) = reach_direction * reaching_speed;
                
                % Reset planning region (write into S)
                S.R_L1_plan(i, idx_pos) = [S.x_ball(i), S.y_ball(i), S.z_ball(i)];
                tmpg = zeros(1, numel(idx_vel)); vals = [S.x_ball(i), S.y_ball(i), S.z_ball(i)];
                ncopy = min(3, numel(tmpg)); tmpg(1:ncopy) = vals(1:ncopy);
                S.R_L1_plan(i, idx_vel) = tmpg;
                S.R_L1_plan(i, idx_bias) = 1;

                S.R_L2_plan(i, 1:3) = reach_direction * reaching_speed;
                S.R_L2_plan(i, 4:6) = 0.01 * randn(1, 3);
                S.R_L3_plan(i, 1:3) = reach_direction * reaching_speed;
                
                % Apply phase transition decay - differential for motor vs. planning
                % Weight matrices are stored as cell arrays (one per task). Apply
                % decay to each task-specific matrix so we don't attempt numeric
                % ops on cell containers.
                for tt = 1:numel(S.W_motor_L2_to_L1)
                    S.W_motor_L2_to_L1{tt} = decay_motor * S.W_motor_L2_to_L1{tt};
                    S.W_motor_L3_to_L2{tt} = decay_motor * S.W_motor_L3_to_L2{tt};
                end

                for tt = 1:numel(S.W_plan_L2_to_L1)
                    S.W_plan_L2_to_L1{tt} = decay_plan * S.W_plan_L2_to_L1{tt};
                    S.W_plan_L3_to_L2{tt} = decay_plan * S.W_plan_L3_to_L2{tt};
                end

                % Restore critical motor mappings (use semantic idx_vel for robustness)
                % Write this into the current trial's motor mapping so the active
                % task retains the basic velocity-to-output mapping.
                map_vel_idx = idx_vel(1:min(3, numel(idx_vel)));
                S.W_motor_L2_to_L1{trial}(map_vel_idx, 1:3) = eye(numel(map_vel_idx), 3);

                current_trial = trial;
                S.current_trial = current_trial; % ensure helper uses the updated trial index
                
                % Safety: ensure ball and player are not exactly coincident after reset
                if isstruct(params) && isfield(params, 'min_start_sep')
                    local_min_sep = params.min_start_sep;
                else
                    local_min_sep = 0.5;
                end
                sep_now = norm([S.x_ball(i), S.y_ball(i), S.z_ball(i)] - [S.x_player(i), S.y_player(i), S.z_player(i)]);
                if sep_now < local_min_sep
                    dirn = randn(1,3); dirn = dirn / (norm(dirn)+1e-9);
                    newpos = [S.x_player(i), S.y_player(i), S.z_player(i)] + dirn * local_min_sep;
                    newpos(1) = min(max(newpos(1), workspace_bounds(1,1)), workspace_bounds(1,2));
                    newpos(2) = min(max(newpos(2), workspace_bounds(2,1)), workspace_bounds(2,2));
                    newpos(3) = min(max(newpos(3), workspace_bounds(3,1)), workspace_bounds(3,2));
                    S.x_ball(i) = newpos(1); S.y_ball(i) = newpos(2); S.z_ball(i) = newpos(3);
                    S.vx_ball(i) = 0; S.vy_ball(i) = 0; S.vz_ball(i) = 0;
                end

                fprintf('\n[Trial %d started at step %d, Task Context: R_L0(i,%d)=1]\n', trial, i, trial);
                fprintf('  Player reset to: [%.2f, %.2f, %.2f]\n', S.x_player(i), S.y_player(i), S.z_player(i));
                fprintf('  Ball reset to: [%.2f, %.2f, %.2f]\n', S.x_ball(i), S.y_ball(i), S.z_ball(i));
                fprintf('  Weight decay (Motor: %.2f→%.0f%%, Planning: %.2f→%.0f%%)\n', ...
                    decay_motor, 100*decay_motor, decay_plan, 100*decay_plan);
                
                break;
            end
        end
    end
    
    % ==============================================================
    % NOTE: Ball physics (integration + collisions) are now centralized
    %       inside `hierarchical_step_update.m`. The helper operates on
    %       S.* arrays and is the authoritative place for kinematics.
    % ==============================================================
    
    % Delegate predictive coding + update work to the helper (type-stable, JIT-friendly)
    S = hierarchical_step_update(i, S, P);

    % Update current trial if helper changed it
    current_trial = S.current_trial;

    % If the helper signaled session end (player close to ball), stop early
    if isfield(S, 'session_end') && S.session_end
        if isfield(S, 'termination_reason')
            fprintf('\n⚠  Session terminated early at step %d: %s\n', S.termination_step, S.termination_reason);
        else
            fprintf('\nSession terminated early at step %d (player within %.3fm of ball)\n', S.termination_step, P.termination_distance);
        end
        break;
    end

    % Only print summary for the last step of the last trial when running under PSO
    if i == N-1 && exist('params','var') && isstruct(params) && isfield(params,'save_results') && params.save_results == false
        last_trial = n_trials;
        last_trial_indices = phases_indices{last_trial};
        last_step_idx = last_trial_indices(end);
        particle_num = -1;
        pso_iter = -1;
        pso_iter_total = -1;
        if isfield(params, 'particle_num'), particle_num = params.particle_num; end
        if isfield(params, 'pso_iter'), pso_iter = params.pso_iter; end
        if isfield(params, 'pso_iter_total'), pso_iter_total = params.pso_iter_total; end
        fprintf('PSO Particle %d | Iteration %d/%d | ', particle_num, pso_iter, pso_iter_total);
        fprintf('eta_rep=%.6f, eta_W=%.6f, momentum=%.6f, weight_decay=%.6f, decay_motor=%.6f, decay_plan=%.6f, motor_gain=%.6f, damping=%.6f, reaching_speed_scale=%.6f, W_plan_gain=%.6f, W_motor_gain=%.6f | ', P.eta_rep, P.eta_W, P.momentum, P.weight_decay, P.decay_motor, P.decay_plan, P.motor_gain, P.damping, P.reaching_speed_scale, P.W_plan_gain, P.W_motor_gain);
        fprintf('Final interception error (step %d, trial %d): %.6f\n', last_step_idx, last_trial, S.interception_error_all(last_step_idx));
    end
    
end  % End main loop

% Pull arrays/state back from S for saving and plotting
x_player = S.x_player; y_player = S.y_player; z_player = S.z_player;
vx_player = S.vx_player; vy_player = S.vy_player; vz_player = S.vz_player;
x_ball = S.x_ball; y_ball = S.y_ball; z_ball = S.z_ball;
vx_ball = S.vx_ball; vy_ball = S.vy_ball; vz_ball = S.vz_ball;
R_L0 = S.R_L0;
R_L1_motor = S.R_L1_motor; R_L2_motor = S.R_L2_motor; R_L3_motor = S.R_L3_motor;
R_L1_plan = S.R_L1_plan; R_L2_plan = S.R_L2_plan; R_L3_plan = S.R_L3_plan;
interception_error_all = S.interception_error_all;
free_energy_all = S.free_energy_all;

% ====================================================================
% FINAL INF/NAN CLIPPING PASS (Nov 1, 2025)
% ====================================================================
% Ensure all critical output arrays are finite before returning to caller
% This prevents NaN/Inf from propagating through optimization / plotting stages

max_finite_value = 1e8;
min_valid_free_energy = 0;
max_valid_free_energy = 1e7;

% Clip free energy to safe range
free_energy_all = max(min_valid_free_energy, min(max_valid_free_energy, free_energy_all));
free_energy_all(~isfinite(free_energy_all)) = max_valid_free_energy;

% Clip interception error to safe range
interception_error_all = max(0, min(max_finite_value, interception_error_all));
interception_error_all(~isfinite(interception_error_all)) = max_finite_value;

% Clip all representation matrices
R_L1_motor = max(-max_finite_value, min(max_finite_value, R_L1_motor));
R_L2_motor = max(-max_finite_value, min(max_finite_value, R_L2_motor));
R_L3_motor = max(-max_finite_value, min(max_finite_value, R_L3_motor));
R_L1_plan = max(-max_finite_value, min(max_finite_value, R_L1_plan));
R_L2_plan = max(-max_finite_value, min(max_finite_value, R_L2_plan));
R_L3_plan = max(-max_finite_value, min(max_finite_value, R_L3_plan));

% Clip trajectory arrays
x_player = max(-max_finite_value, min(max_finite_value, x_player));
y_player = max(-max_finite_value, min(max_finite_value, y_player));
z_player = max(-max_finite_value, min(max_finite_value, z_player));
x_ball = max(-max_finite_value, min(max_finite_value, x_ball));
y_ball = max(-max_finite_value, min(max_finite_value, y_ball));
z_ball = max(-max_finite_value, min(max_finite_value, z_ball));

fprintf('✓ Applied final Inf/NaN clipping pass (safety check for free energy and trajectories)\n\n');

% Check if trial was terminated due to excessive clipping
if isfield(S, 'termination_reason') && contains(S.termination_reason, 'Excessive consecutive Inf/NaN')
    fprintf('⚠  TRIAL SUMMARY: Early termination triggered due to excessive Inf/NaN clipping.\n');
    fprintf('   Reason: %s\n\n', S.termination_reason);
end

phases_indices = S.phases_indices;
W_motor_L2_to_L1 = S.W_motor_L2_to_L1; W_motor_L3_to_L2 = S.W_motor_L3_to_L2;
W_plan_L2_to_L1 = S.W_plan_L2_to_L1; W_plan_L3_to_L2 = S.W_plan_L3_to_L2;
W_motor_L1_lat = S.W_motor_L1_lat; W_motor_L2_lat = S.W_motor_L2_lat; W_motor_L3_lat = S.W_motor_L3_lat;
W_plan_L1_lat = S.W_plan_L1_lat; W_plan_L2_lat = S.W_plan_L2_lat; W_plan_L3_lat = S.W_plan_L3_lat;
learning_trace_W = S.learning_trace_W;
pi_trace_L1_motor = S.pi_trace_L1_motor; pi_trace_L2_motor = S.pi_trace_L2_motor;
pi_trace_L1_plan = S.pi_trace_L1_plan; pi_trace_L2_plan = S.pi_trace_L2_plan;
pi_raw_trace_L1_motor = S.pi_raw_trace_L1_motor; pi_raw_trace_L2_motor = S.pi_raw_trace_L2_motor;
pi_raw_trace_L1_plan = S.pi_raw_trace_L1_plan; pi_raw_trace_L2_plan = S.pi_raw_trace_L2_plan;
denom_trace_L1_motor = S.denom_trace_L1_motor; denom_trace_L2_motor = S.denom_trace_L2_motor;
denom_trace_L1_plan = S.denom_trace_L1_plan; denom_trace_L2_plan = S.denom_trace_L2_plan;

fprintf('\n✓ Main loop complete (%d iterations executed)\n\n', N-1);

% ====================================================================
% SAVE RESULTS
% ====================================================================

% Prepare results struct (always returned)
interception_error_all = interception_error_all(1:N);

results = struct();
results.x_player = x_player; results.y_player = y_player; results.z_player = z_player;
results.vx_player = vx_player; results.vy_player = vy_player; results.vz_player = vz_player;
results.x_ball = x_ball; results.y_ball = y_ball; results.z_ball = z_ball;
results.vx_ball = vx_ball; results.vy_ball = vy_ball; results.vz_ball = vz_ball;
results.R_L0 = R_L0;
results.R_L1_motor = R_L1_motor; results.R_L2_motor = R_L2_motor; results.R_L3_motor = R_L3_motor;
results.R_L1_plan = R_L1_plan; results.R_L2_plan = R_L2_plan; results.R_L3_plan = R_L3_plan;
results.interception_error_all = interception_error_all;
results.free_energy_all = free_energy_all;
results.phases_indices = phases_indices;
results.W_motor_L2_to_L1 = W_motor_L2_to_L1; results.W_motor_L3_to_L2 = W_motor_L3_to_L2;
results.W_plan_L2_to_L1 = W_plan_L2_to_L1; results.W_plan_L3_to_L2 = W_plan_L3_to_L2;
results.W_motor_L1_lat = W_motor_L1_lat; results.W_motor_L2_lat = W_motor_L2_lat; results.W_motor_L3_lat = W_motor_L3_lat;
results.W_plan_L1_lat = W_plan_L1_lat; results.W_plan_L2_lat = W_plan_L2_lat; results.W_plan_L3_lat = W_plan_L3_lat;
results.learning_trace_W = learning_trace_W;
results.pi_trace_L1_motor = pi_trace_L1_motor; results.pi_trace_L2_motor = pi_trace_L2_motor;
results.pi_trace_L1_plan = pi_trace_L1_plan; results.pi_trace_L2_plan = pi_trace_L2_plan;
results.pi_raw_trace_L1_motor = pi_raw_trace_L1_motor; results.pi_raw_trace_L2_motor = pi_raw_trace_L2_motor;
results.pi_raw_trace_L1_plan = pi_raw_trace_L1_plan; results.pi_raw_trace_L2_plan = pi_raw_trace_L2_plan;
results.denom_trace_L1_motor = denom_trace_L1_motor; results.denom_trace_L2_motor = denom_trace_L2_motor;
results.denom_trace_L1_plan = denom_trace_L1_plan; results.denom_trace_L2_plan = denom_trace_L2_plan;

% ====================================================================
% CLIPPING STATISTICS (Nov 1, 2025)
% ====================================================================
% Track total number of Inf/NaN clipping events that occurred during this run
clipping_count = S.clipping_count;
results.clipping_count = clipping_count;  % Add to results for easy tracking

% Track early termination due to excessive clipping
if isfield(S, 'termination_reason')
    results.early_termination = true;
    results.termination_reason = S.termination_reason;
    results.termination_step = S.termination_step;
else
    results.early_termination = false;
    results.termination_reason = 'Completed normally';
    results.termination_step = N;
end

if clipping_count > 0
    fprintf('⚠  CLIPPING SUMMARY: %d Inf/NaN clipping event(s) detected and handled during this run.\n', clipping_count);
else
    fprintf('✓ CLIPPING SUMMARY: No Inf/NaN clipping events occurred (clean run).\n');
end

% Decide whether to save a MAT file to disk. Default: true (backwards compatible).
save_results = true;
if nargin > 0 && isstruct(params) && isfield(params, 'save_results')
    save_results = params.save_results;
end

output_dir = './figures';
if save_results
    fprintf('Saving results...\n');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    try
        results_filename = fullfile(output_dir, '3D_dual_hierarchy_results.mat');
        save(results_filename, '-struct', 'results', '-v7.3');
        fprintf('✓ Results saved: %s\n', results_filename);
        fprintf('   (includes clipping_count = %d)\n', clipping_count);
    catch ME
        fprintf('Warning: MAT file save failed: %s\n', ME.message);
    end
else
    fprintf('Skipping MAT-file save (params.save_results=false). Returning results struct only.\n');
end

% ====================================================================
% PLOTTING (if enabled)
% ====================================================================

if make_plots
    fprintf('Creating summary plots...\n');
    
    fig = figure('Position', [100, 100, 1400, 900], 'Visible', 'off');
    
    colors = {'r', 'g', 'b', 'm'};
    
    % Plot 1: Interception Error Over Time
    subplot(2, 3, 1);
    hold on;
    for trial = 1:n_trials
        trial_idx = phases_indices{trial};
        plot(trial_idx, interception_error_all(trial_idx), 'Color', colors{trial}, 'LineWidth', 2, 'DisplayName', sprintf('Trial %d', trial));
    end
    grid on; xlabel('Time (steps)'); ylabel('Distance to Ball (m)');
    title('Interception Error: Player to Ball');
    legend off;
    
    % Plot 2: Free Energy Over Time
    subplot(2, 3, 2);
    semilogy(free_energy_all, 'k-', 'LineWidth', 2);
    grid on; xlabel('Time (steps)'); ylabel('Free Energy (log scale)');
    title('Free Energy Minimization (Dual Hierarchy)');
    xlim([0 N]);
    
    % Plot 3: Player X vs Ball X
    subplot(2, 3, 3);
    hold on;
    plot(x_ball, 'b-', 'LineWidth', 2, 'DisplayName', 'Ball');
    plot(x_player, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Player');
    grid on; xlabel('Time (steps)'); ylabel('X Position (m)');
    title('X Coordinate: Player Chasing Ball');
    legend off;
    
    % Plot 4: Player Y vs Ball Y
    subplot(2, 3, 4);
    hold on;
    plot(y_ball, 'b-', 'LineWidth', 2, 'DisplayName', 'Ball');
    plot(y_player, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Player');
    grid on; xlabel('Time (steps)'); ylabel('Y Position (m)');
    title('Y Coordinate: Player Chasing Ball');
    legend off;
    
    % Plot 5: Player Z vs Ball Z
    subplot(2, 3, 5);
    hold on;
    plot(z_ball, 'b-', 'LineWidth', 2, 'DisplayName', 'Ball');
    plot(z_player, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Player');
    grid on; xlabel('Time (steps)'); ylabel('Z Position (m)');
    title('Z Coordinate: Player Chasing Ball');
    legend off;
    
    % Plot 6: Learning Trace
    subplot(2, 3, 6);
    semilogy(learning_trace_W + 1e-10, 'k-', 'LineWidth', 2);
    grid on; xlabel('Time (steps)'); ylabel('Weight Change Magnitude (log scale)');
    title('Learning Trace: Weight Updates');
    xlim([0 N]);
    
    sgtitle('Dual-Hierarchy Predictive Coding: Player Chasing Moving Ball', 'FontSize', 12, 'FontWeight', 'bold');
    
    try
        figure_filename = fullfile(output_dir, '3D_dual_hierarchy_summary.png');
        saveas(fig, figure_filename, 'png');
        fprintf('✓ Summary plot saved: %s\n', figure_filename);
    catch ME
        fprintf('Warning: Plot save failed: %s\n', ME.message);
    end
    
    close(fig);
else
    fprintf('Skipping plot generation (make_plots=false)\n');
end

% ====================================================================
% ANALYSIS SUMMARY
% ====================================================================

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  DUAL-HIERARCHY LEARNING: PLAYER CHASING MOVING BALL       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('INTERCEPTION PERFORMANCE:\n');
fprintf('─────────────────────────────────────────────────────────\n');
overall_interception_rmse = sqrt(mean(interception_error_all.^2));
fprintf('Overall Interception RMSE: %.6f m\n\n', overall_interception_rmse);

fprintf('By Trial:\n');
for trial = 1:n_trials
    trial_idx = phases_indices{trial};
    trial_errors = interception_error_all(trial_idx);
    trial_rmse = sqrt(mean(trial_errors.^2));
    fprintf('  Trial %d: Interception RMSE = %.6f m (mean distance: %.6f m)\n', ...
        trial, trial_rmse, mean(trial_errors));
end

fprintf('\nLEARNING EFFICIENCY:\n');
fprintf('─────────────────────────────────────────────────────────\n');
fprintf('Final Free Energy:           %.6e\n', free_energy_all(end-1));
fprintf('Free Energy Reduction Rate:  %.6e per step\n', (free_energy_all(1) - free_energy_all(end)) / N);
fprintf('Total trials completed:      %d\n', n_trials);
fprintf('Total learning steps:        %d\n', N);

fprintf('\nMOTOR REGION STATUS:\n');
fprintf('  • Always learning (goal-independent motor laws)\n');
fprintf('  • Weight decay at phase boundaries: %.0f%% retained\n', 100*decay_motor);

fprintf('\nPLANNING REGION STATUS:\n');
fprintf('  • Task-gated learning (gates in 0.3-1.0 range)\n');
fprintf('  • Weight decay at phase boundaries: %.0f%% retained\n', 100*decay_plan);

fprintf('\nTASK CONTEXT (L0):\n');
fprintf('  • One-hot encoding of current trial\n');
fprintf('  • Explicit representation enables task-specific learning\n');

fprintf('\n');

end  % End of hierarchical_motion_inference_dual_hierarchy function
