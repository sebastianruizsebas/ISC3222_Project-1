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
    
    if nargin > 0 && isstruct(params)
        % Override defaults with provided parameters
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

fprintf('╔═════════════════════════════════════════════════════════════╗\n');
fprintf('║  DUAL-HIERARCHY PREDICTIVE CODING: PLAYER CHASING BALL    ║\n');
fprintf('║  Motor Region (Stable) + Planning Region (Task-Specific)  ║\n');
fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% BATCH MODE SETUP
% ====================================================================
if ~exist('optimizer_mode', 'var')
    optimizer_mode = false;
end

set(0, 'DefaultFigureVisible', 'off');
set(groot, 'defaultFigureCreateFcn', @(fig, ~) set(fig, 'Visible', 'off'));

if ~optimizer_mode
    fprintf('Batch mode: Graphics output disabled, figures will be saved to disk.\n\n');
end

% ====================================================================
% TASK CONFIGURATION: PLAYER CHASING MOVING BALL
% ====================================================================

dt = 0.01;              % Time step (s)
T_per_trial = 4000;      % Duration per trial (s)
n_trials = 4;           % Number of different ball trajectories

% Allow overriding of timing parameters via params for faster debug runs or PSO
if nargin > 0 && isstruct(params)
    if isfield(params, 'dt'), dt = params.dt; end
    if isfield(params, 'T_per_trial'), T_per_trial = params.T_per_trial; end
    if isfield(params, 'n_trials'), n_trials = params.n_trials; end
end

T = T_per_trial * n_trials;  % Total duration
t = 0:dt:T;
N = length(t);

% Define reaching task phases
trial_duration_steps = round(T_per_trial / dt);  % ~250 steps per trial
phase_masks = {};

% Construct trial phase indices (start:end step indices for each trial)
% phases_indices is used throughout the main loop to detect trial transitions
phases_indices = cell(n_trials, 1);
for trial = 1:n_trials
    start_idx = (trial - 1) * trial_duration_steps + 1;
    end_idx = min(trial * trial_duration_steps, N);
    phases_indices{trial} = start_idx:end_idx;
end

% Define ball trajectory parameters for each trial
% Each trial has a different ball trajectory (start_pos, velocity, acceleration)
rng(42);  % For reproducibility
ball_trajectories = {};
for trial = 1:n_trials
    ball_trajectories{trial} = struct(...
        'start_pos', randn(1, 3) * 0.5, ...  % Random starting position
        'velocity', randn(1, 3) * 0.5, ...   % Random initial velocity
        'acceleration', randn(1, 3) * 0.1 ... % Random acceleration pattern
    );
end

% Workspace bounds
workspace_bounds = [
    -2, 2;      % X bounds
    -2, 2;      % Y bounds
    -1, 2       % Z bounds
];

% Initial player positions for each trial (random inside workspace)
initial_positions = zeros(n_trials, 3);
for trial = 1:n_trials
    for dim = 1:3
        initial_positions(trial, dim) = workspace_bounds(dim, 1) + ...
            rand() * (workspace_bounds(dim, 2) - workspace_bounds(dim, 1));
    end
end
    
% ====================================================================
% LAYER DIMENSIONS - DUAL HIERARCHY
% ====================================================================

% Task Context Layer (L0)
n_L0 = n_trials;        % One-hot encoding: which trial/task is active

% Motor Region
n_L1_motor = 7;         % [x, y, z, vx, vy, vz, bias] - proprioceptive state
n_L2_motor = 6;         % Learned motor primitives (stable velocity commands)
n_L3_motor = 3;         % Motor output (3D velocity commands to muscles)

% Planning Region
n_L1_plan = 7;          % [ball_x, ball_y, ball_z, goal_x, goal_y, goal_z, bias]
n_L2_plan = 6;          % Learned planning/interception policies
n_L3_plan = 3;          % Planning output (target velocity from planning region)

fprintf('DUAL-HIERARCHY ARCHITECTURE:\n');
fprintf('═════════════════════════════════════════════════════════════\n');
fprintf('TASK CONTEXT (L0): %d neurons [one-hot trial encoding]\n', n_L0);
fprintf('  • Explicit task identity for trial-specific learning\n');
fprintf('  • Gates learning in motor vs. planning regions\n\n');

fprintf('MOTOR REGION (Learns Stable Dynamics):\n');
fprintf('  L1_motor (Proprioception): %d neurons [x, y, z, vx, vy, vz, bias]\n', n_L1_motor);
fprintf('    → Stable proprioceptive state (decoupled from target motion)\n');
fprintf('  L2_motor (Basis):           %d neurons [learned motor primitives]\n', n_L2_motor);
fprintf('    → Goal-independent motion primitives\n');
fprintf('  L3_motor (Output):          %d neurons [3D velocity commands]\n', n_L3_motor);
fprintf('    → Direct motor commands to execution system\n');
fprintf('  Weight Decay: 95%%-98%% (preserve motor knowledge across tasks)\n\n');

fprintf('PLANNING REGION (Learns Task-Specific Strategies):\n');
fprintf('  L1_plan (Goal State):       %d neurons [ball_x, ball_y, ball_z, goal_x, goal_y, goal_z, bias]\n', n_L1_plan);
fprintf('    → Current ball position + desired interception goal\n');
fprintf('  L2_plan (Policies):         %d neurons [learned interception policies]\n', n_L2_plan);
fprintf('    → Task-specific reaching strategies\n');
fprintf('  L3_plan (Output):           %d neurons [target velocity from planning]\n', n_L3_plan);
fprintf('    → Planned velocity to send to motor region\n');
fprintf('  Weight Decay: 70%%-80%% (forget old targets, learn new tasks)\n\n');

fprintf('LEARNING GATING:\n');
fprintf('  Motor region: Always learning (goal-independent motor laws)\n');
fprintf('  Planning region: Task-gated learning (gate = 0.3-1.0 based on task context)\n\n');

% ====================================================================
% 3D KINEMATICS: INTEGRATING MOTOR COMMANDS INTO POSITION
% ====================================================================

% Initialize 3D position and velocity
x_player = zeros(1, N);
y_player = zeros(1, N);
z_player = zeros(1, N);
vx_player = zeros(1, N);
vy_player = zeros(1, N);
vz_player = zeros(1, N);

% Ball trajectories
x_ball = zeros(1, N);
y_ball = zeros(1, N);
z_ball = zeros(1, N);
vx_ball = zeros(1, N);
vy_ball = zeros(1, N);
vz_ball = zeros(1, N);

% Motor commands from each region
motor_vx_motor = zeros(1, N);  % Velocity from motor region
motor_vy_motor = zeros(1, N);
motor_vz_motor = zeros(1, N);

motor_vx_plan = zeros(1, N);   % Velocity from planning region
motor_vy_plan = zeros(1, N);
motor_vz_plan = zeros(1, N);

% Motor dynamics parameters
if ~exist('motor_gain', 'var')
    motor_gain = 0.5;
end
if ~exist('damping', 'var')
    damping = 0.85;
end
if ~exist('reaching_speed_scale', 'var')
    reaching_speed_scale = 0.5;
end

fprintf('3D MOTOR DYNAMICS:\n');
fprintf('  Motor gain: %.2f (scaling of commands to actual motion)\n', motor_gain);
fprintf('  Damping: %.2f (velocity decay per timestep)\n', damping);
fprintf('  Reaching speed scale: %.2f (target-distance-dependent)\n\n', reaching_speed_scale);

% ====================================================================
% LEARNING PARAMETERS
% ====================================================================

if ~exist('eta_rep', 'var')
    eta_rep = 0.005;
end
if ~exist('eta_W', 'var')
    eta_W = 0.0005;
end
if ~exist('momentum', 'var')
    momentum = 0.98;
end
if ~exist('weight_decay', 'var')
    weight_decay = 0.98;
end

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

fprintf('LEARNING PARAMETERS:\n');
fprintf('  η_rep = %.6f (representation learning rate)\n', eta_rep);
fprintf('  η_W   = %.6f (weight matrix learning rate)\n', eta_W);
fprintf('  Momentum = %.4f\n', momentum);
fprintf('  Weight Decay (per-step) = %.4f\n', weight_decay);
fprintf('  Decay at Phase (Motor) = %.4f (95%%-98%% retained)\n', decay_motor);
fprintf('  Decay at Phase (Planning) = %.4f (70%%-80%% retained)\n', decay_plan);
fprintf('  π_motor   = [%.0f, %.0f, %.0f]\n', pi_L1_motor, pi_L2_motor, pi_L3_motor);
fprintf('  π_plan    = [%.0f, %.0f, %.0f]\n\n', pi_L1_plan, pi_L2_plan, pi_L3_plan);

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

% Motor L1 (proprioception)
R_L1_motor(1, 1:3) = [x_player(1), y_player(1), z_player(1)];
R_L1_motor(1, 4:6) = [0, 0, 0];
R_L1_motor(1, 7) = 1;

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

% Planning L1: ball position + goal
R_L1_plan(1, 1:3) = [x_ball(1), y_ball(1), z_ball(1)];  % Ball position
R_L1_plan(1, 4:6) = [x_ball(1), y_ball(1), z_ball(1)];  % Interception goal (initially same as ball)
R_L1_plan(1, 7) = 1;

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
% INITIALIZE WEIGHT MATRICES - DUAL HIERARCHY
% ====================================================================

% Motor region weights
W_motor_L2_to_L1 = zeros(n_L1_motor, n_L2_motor);
W_motor_L3_to_L2 = zeros(n_L2_motor, n_L3_motor);

% Map L3_motor (3D velocity) to L1_motor (proprioceptive prediction)
W_motor_L2_to_L1(4:6, 1:3) = eye(3, 3);  % Velocity commands → velocity predictions
W_motor_L2_to_L1(1:3, 1:3) = 0.01 * eye(3, 3);  % Weak position coupling
W_motor_L2_to_L1(7, :) = 0.01 * randn(1, n_L2_motor);

% Map L2_motor (primitives) to L3_motor (output)
W_motor_L3_to_L2(1:3, 1:3) = W_motor_gain * eye(3, 3);  % Primitives → output
W_motor_L3_to_L2(4:6, 1:3) = 0.01 * randn(3, 3);

% Planning region weights
W_plan_L2_to_L1 = zeros(n_L1_plan, n_L2_plan);
W_plan_L3_to_L2 = zeros(n_L2_plan, n_L3_plan);

% Map L3_plan (target velocity from planning) to L1_plan (ball + goal)
W_plan_L2_to_L1(1:3, 1:3) = 0.01 * eye(3, 3);  % Weak ball position prediction
W_plan_L2_to_L1(4:6, 1:3) = 0.1 * eye(3, 3);   % Goal prediction from policies
W_plan_L2_to_L1(7, :) = 0.01 * randn(1, n_L2_plan);

% Map L2_plan (policies) to L3_plan (planning output)
W_plan_L3_to_L2(1:3, 1:3) = W_plan_gain * eye(3, 3);
W_plan_L3_to_L2(4:6, 1:3) = 0.01 * randn(3, 3);

% --- NEW: LATERAL (WITHIN-LAYER) WEIGHTS (small random init) ---
rng(0); % reproducible lateral init
W_motor_L1_lat = 0.01 * randn(n_L1_motor, n_L1_motor);
W_motor_L2_lat = 0.01 * randn(n_L2_motor, n_L2_motor);
W_motor_L3_lat = 0.01 * randn(n_L3_motor, n_L3_motor);
W_plan_L1_lat  = 0.01 * randn(n_L1_plan,  n_L1_plan);
W_plan_L2_lat  = 0.01 * randn(n_L2_plan,  n_L2_plan);
W_plan_L3_lat  = 0.01 * randn(n_L3_plan,  n_L3_plan);

% Remove strong self-connections initially
W_motor_L1_lat(1:n_L1_motor+1:end) = 0;
W_motor_L2_lat(1:n_L2_motor+1:end) = 0;
W_motor_L3_lat(1:n_L3_motor+1:end) = 0;
W_plan_L1_lat(1:n_L1_plan+1:end)   = 0;
W_plan_L2_lat(1:n_L2_plan+1:end)   = 0;
W_plan_L3_lat(1:n_L3_plan+1:end)   = 0;

fprintf('WEIGHT MATRICES INITIALIZED:\n');
fprintf('  Motor Region:\n');
fprintf('    W_motor_L2_to_L1: Motor Basis → Proprioception (6×7)\n');
fprintf('    W_motor_L3_to_L2: Output → Basis (3×6)\n');
fprintf('  Planning Region:\n');
fprintf('    W_plan_L2_to_L1: Policies → Goal State (6×7)\n');
fprintf('    W_plan_L3_to_L2: Output → Policies (3×6)\n\n');

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

% Smoothing and step-limiting for precision updates to avoid saturation
pi_smooth_alpha = 0.995;        % strong smoothing (closer to 1 => slower changes)
pi_max_step_ratio = 1.2;        % max allowed multiplicative change per step (20%)

% Dynamic precision histories
window_size = 100;
L1_motor_error_history = [];
L2_motor_error_history = [];
L1_plan_error_history = [];
L2_plan_error_history = [];

% ====================================================================
% MAIN LEARNING LOOP - DUAL HIERARCHY
% ====================================================================

fprintf('Running dual-hierarchy learning with player chasing moving ball...\n');
fprintf('Total iterations: %d (dt=%.4fs per step, ~%.1f seconds estimated)\n', N-1, dt, (N-1)*dt);

current_trial = 1;

for i = 1:N-1
    if mod(i, 100) == 0, fprintf('.'); end
    
    % ==============================================================
    % CHECK FOR TRIAL TRANSITION
    % ==============================================================
    if i > 1
        for trial = 2:n_trials
            if i == phases_indices{trial}(1)
                % Reset player position and ball trajectory for new trial
                x_player(i) = initial_positions(trial, 1);
                y_player(i) = initial_positions(trial, 2);
                z_player(i) = initial_positions(trial, 3);
                vx_player(i) = 0;
                vy_player(i) = 0;
                vz_player(i) = 0;
                
                % Reset ball for new trial
                x_ball(i) = ball_trajectories{trial}.start_pos(1);
                y_ball(i) = ball_trajectories{trial}.start_pos(2);
                z_ball(i) = ball_trajectories{trial}.start_pos(3);
                vx_ball(i) = ball_trajectories{trial}.velocity(1);
                vy_ball(i) = ball_trajectories{trial}.velocity(2);
                vz_ball(i) = ball_trajectories{trial}.velocity(3);
                
                % Update task context (L0)
                R_L0(i, :) = 0;
                R_L0(i, trial) = 1;
                
                % Reset motor region representations
                R_L1_motor(i, 1:3) = [x_player(i), y_player(i), z_player(i)];
                R_L1_motor(i, 4:6) = [0, 0, 0];
                R_L1_motor(i, 7) = 1;
                
                reach_direction = ([x_ball(i), y_ball(i), z_ball(i)] - [x_player(i), y_player(i), z_player(i)]) / ...
                                   (norm([x_ball(i), y_ball(i), z_ball(i)] - [x_player(i), y_player(i), z_player(i)]) + 1e-6);
                target_distance = norm([x_ball(i), y_ball(i), z_ball(i)] - [x_player(i), y_player(i), z_player(i)]);
                reaching_speed = reaching_speed_scale * target_distance;
                
                R_L2_motor(i, 1:3) = reach_direction * reaching_speed;
                R_L2_motor(i, 4:6) = 0.01 * randn(1, 3);
                R_L3_motor(i, 1:3) = reach_direction * reaching_speed;
                
                % Reset planning region
                R_L1_plan(i, 1:3) = [x_ball(i), y_ball(i), z_ball(i)];
                R_L1_plan(i, 4:6) = [x_ball(i), y_ball(i), z_ball(i)];
                R_L1_plan(i, 7) = 1;
                
                R_L2_plan(i, 1:3) = reach_direction * reaching_speed;
                R_L2_plan(i, 4:6) = 0.01 * randn(1, 3);
                R_L3_plan(i, 1:3) = reach_direction * reaching_speed;
                
                % Apply phase transition decay - differential for motor vs. planning
                W_motor_L2_to_L1 = decay_motor * W_motor_L2_to_L1;
                W_motor_L3_to_L2 = decay_motor * W_motor_L3_to_L2;
                
                W_plan_L2_to_L1 = decay_plan * W_plan_L2_to_L1;
                W_plan_L3_to_L2 = decay_plan * W_plan_L3_to_L2;
                
                % Restore critical motor mappings
                W_motor_L2_to_L1(4:6, 1:3) = eye(3, 3);
                
                current_trial = trial;
                
                fprintf('\n[Trial %d started at step %d, Task Context: R_L0(i,%d)=1]\n', trial, i, trial);
                fprintf('  Player reset to: [%.2f, %.2f, %.2f]\n', x_player(i), y_player(i), z_player(i));
                fprintf('  Ball reset to: [%.2f, %.2f, %.2f]\n', x_ball(i), y_ball(i), z_ball(i));
                fprintf('  Weight decay (Motor: %.2f→%.0f%%, Planning: %.2f→%.0f%%)\n', ...
                    decay_motor, 100*decay_motor, decay_plan, 100*decay_plan);
                
                break;
            end
        end
    end
    
    % ============================================================== 
    % UPDATE BALL TRAJECTORY (Continuous Motion)  -- REPLACED WITH BOUNCE PHYSICS
    % ============================================================== 
    time_in_trial = i - phases_indices{current_trial}(1);

    % Smooth oscillating acceleration (internal drive)
    acc_x = ball_trajectories{current_trial}.acceleration(1) * sin(time_in_trial * 0.001);
    acc_y = ball_trajectories{current_trial}.acceleration(2) * sin(time_in_trial * 0.001 + 1);
    acc_z = ball_trajectories{current_trial}.acceleration(3) * sin(time_in_trial * 0.001 + 2);

    % Gravity acts downward on z
    ax = acc_x;
    ay = acc_y;
    az = acc_z - gravity;

    % Integrate velocity
    vx_ball(i+1) = vx_ball(i) + ax * dt;
    vy_ball(i+1) = vy_ball(i) + ay * dt;
    vz_ball(i+1) = vz_ball(i) + az * dt;

    % Air drag (simple fractional damping)
    vx_ball(i+1) = vx_ball(i+1) * (1 - air_drag);
    vy_ball(i+1) = vy_ball(i+1) * (1 - air_drag);
    vz_ball(i+1) = vz_ball(i+1) * (1 - air_drag);

    % Integrate position
    x_ball(i+1) = x_ball(i) + dt * vx_ball(i+1);
    y_ball(i+1) = y_ball(i) + dt * vy_ball(i+1);
    z_ball(i+1) = z_ball(i) + dt * vz_ball(i+1);

    % Collision with ground plane (workspace_bounds(3,1))
    ground_z = workspace_bounds(3,1);
    if z_ball(i+1) <= ground_z
        % Place on ground
        z_ball(i+1) = ground_z;

        % Invert vertical velocity with restitution
        if vz_ball(i+1) < 0
            vz_ball(i+1) = -restitution * vz_ball(i+1);
        end

        % Apply ground friction to horizontal velocity
        vx_ball(i+1) = vx_ball(i+1) * ground_friction;
        vy_ball(i+1) = vy_ball(i+1) * ground_friction;

        % Kill very small bounces to avoid jitter
        if abs(vz_ball(i+1)) < 1e-3
            vz_ball(i+1) = 0;
        end
    end

    % Clamp to workspace
    x_ball(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), x_ball(i+1)));
    y_ball(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), y_ball(i+1)));
    z_ball(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), z_ball(i+1)));
    
    % ==============================================================
    % MOTOR REGION PREDICTIVE CODING
    % ==============================================================

    % Top-down: L3_motor → L2_motor
    pred_L2_motor(i,:) = R_L3_motor(i,:) * W_motor_L3_to_L2';

    % Lateral: L2 lateral projection
    pred_L2_motor(i,:) = pred_L2_motor(i,:) + R_L2_motor(i,:) * W_motor_L2_lat';

    % Top-down: L2_motor → L1_motor
    pred_L1_motor(i,:) = R_L2_motor(i,:) * W_motor_L2_to_L1';

    % Lateral: L1 lateral projection
    pred_L1_motor(i,:) = pred_L1_motor(i,:) + R_L1_motor(i,:) * W_motor_L1_lat';

    % Extract predicted velocities
    pred_vx_motor = pred_L1_motor(i, 4);
    pred_vy_motor = pred_L1_motor(i, 5);
    pred_vz_motor = pred_L1_motor(i, 6);

    % Motor execution
    motor_vx_motor(i) = motor_gain * pred_vx_motor;
    motor_vy_motor(i) = motor_gain * pred_vy_motor;
    motor_vz_motor(i) = motor_gain * pred_vz_motor;
    
    % ==============================================================
    % PLANNING REGION PREDICTIVE CODING
    % ==============================================================
    
    % Planning region: predict policies from planning goals
    pred_L2_plan(i,:) = R_L3_plan(i,:) * W_plan_L3_to_L2';
    
    % Lateral: L2 planning lateral
    pred_L2_plan(i,:) = pred_L2_plan(i,:) + R_L2_plan(i,:) * W_plan_L2_lat';
    
    % Plan predicts ball state
    pred_L1_plan(i,:) = R_L2_plan(i,:) * W_plan_L2_to_L1';
    
    % Lateral: L1 planning lateral
    pred_L1_plan(i,:) = pred_L1_plan(i,:) + R_L1_plan(i,:) * W_plan_L1_lat';

    % Extract predicted velocities from planning
    pred_vx_plan = pred_L1_plan(i, 4);
    pred_vy_plan = pred_L1_plan(i, 5);
    pred_vz_plan = pred_L1_plan(i, 6);
    
    motor_vx_plan(i) = motor_gain * pred_vx_plan;
    motor_vy_plan(i) = motor_gain * pred_vy_plan;
    motor_vz_plan(i) = motor_gain * pred_vz_plan;
    
    % ==============================================================
    % COMBINED MOTOR OUTPUT & KINEMATICS
    % ==============================================================
    % Combine motor and planning outputs (equal weighting for now)
    
    combined_vx = 0.5 * motor_vx_motor(i) + 0.5 * motor_vx_plan(i);
    combined_vy = 0.5 * motor_vy_motor(i) + 0.5 * motor_vy_plan(i);
    combined_vz = 0.5 * motor_vz_motor(i) + 0.5 * motor_vz_plan(i);
    
    % Update player velocity
    vx_player(i+1) = damping * vx_player(i) + combined_vx;
    vy_player(i+1) = damping * vy_player(i) + combined_vy;
    vz_player(i+1) = damping * vz_player(i) + combined_vz;
    
    % Update player position
    x_player(i+1) = x_player(i) + dt * vx_player(i+1);
    y_player(i+1) = y_player(i) + dt * vy_player(i+1);
    z_player(i+1) = z_player(i) + dt * vz_player(i+1);
    
    % Clamp to workspace
    x_player(i+1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), x_player(i+1)));
    y_player(i+1) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), y_player(i+1)));
    z_player(i+1) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), z_player(i+1)));
    
    % ==============================================================
    % ERROR COMPUTATION - MOTOR REGION
    % ==============================================================
    
    % Proprioceptive error
    E_L1_motor(i, 1:3) = [x_player(i+1), y_player(i+1), z_player(i+1)] - pred_L1_motor(i, 1:3);
    E_L1_motor(i, 4:6) = [vx_player(i+1), vy_player(i+1), vz_player(i+1)] - pred_L1_motor(i, 4:6);
    E_L1_motor(i, 7) = 1 - pred_L1_motor(i, 7);
    
    % Motor basis error
    E_L2_motor(i,:) = R_L2_motor(i,:) - pred_L2_motor(i,:);
    
    % ==============================================================
    % ERROR COMPUTATION - PLANNING REGION
    % ==============================================================
    
    % Ball position error
    E_L1_plan(i, 1:3) = [x_ball(i+1), y_ball(i+1), z_ball(i+1)] - pred_L1_plan(i, 1:3);
    
    % Goal: try to intercept ball (goal = current ball position in simple version)
    ball_pos_now = [x_ball(i+1), y_ball(i+1), z_ball(i+1)];
    E_L1_plan(i, 4:6) = ball_pos_now - pred_L1_plan(i, 4:6);
    
    E_L1_plan(i, 7) = 1 - pred_L1_plan(i, 7);
    
    % Policy error
    E_L2_plan(i,:) = R_L2_plan(i,:) - pred_L2_plan(i,:);
    
    % Interception error: distance from player to ball
    interception_error_all(i) = sqrt((x_player(i+1) - x_ball(i+1))^2 + ...
                                      (y_player(i+1) - y_ball(i+1))^2 + ...
                                      (z_player(i+1) - z_ball(i+1))^2);
    
    % ==============================================================
    % FREE ENERGY & METRICS
    % ==============================================================
    
    fe_motor = sum(E_L1_motor(i,:).^2) / (2 * pi_L1_motor) + sum(E_L2_motor(i,:).^2) / (2 * pi_L2_motor);
    fe_plan = sum(E_L1_plan(i,:).^2) / (2 * pi_L1_plan) + sum(E_L2_plan(i,:).^2) / (2 * pi_L2_plan);
    fe_interception = (pi_L1_motor / 100) * interception_error_all(i)^2;
    
    free_energy_all(i) = fe_motor + fe_plan + fe_interception;
    
    % ==============================================================
    % REPRESENTATION UPDATES (MOTOR REGION)
    % ==============================================================
    
    decay = 1 - momentum;
    
    % L1_motor: proprioceptive learning
    R_L1_motor(i+1, 1:3) = R_L1_motor(i, 1:3) + decay * eta_rep * E_L1_motor(i, 1:3) * 0.1;
    R_L1_motor(i+1, 4:6) = momentum * R_L1_motor(i, 4:6) + decay * eta_rep * E_L1_motor(i, 4:6) * 0.1;
    R_L1_motor(i+1, 4:6) = max(-2, min(2, R_L1_motor(i+1, 4:6)));
    R_L1_motor(i+1, 1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), R_L1_motor(i+1,1)));
    R_L1_motor(i+1, 2) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), R_L1_motor(i+1,2)));
    R_L1_motor(i+1, 3) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), R_L1_motor(i+1,3)));
    R_L1_motor(i+1, 7) = 1;
    
    % L2_motor: motor basis learning
    coupling_motor = E_L1_motor(i,:) * W_motor_L2_to_L1;
    norm_W_motor = max(0.1, norm(W_motor_L2_to_L1, 'fro'));
    coupling_motor = coupling_motor / norm_W_motor;
    
    delta_R_L2_motor = coupling_motor - E_L2_motor(i,:);
    R_L2_motor(i+1,:) = momentum * R_L2_motor(i,:) + decay * eta_rep * delta_R_L2_motor * 0.5;
    R_L2_motor(i+1,:) = max(-1, min(1, R_L2_motor(i+1,:)));
    
    % L3_motor: output learning
    E_L3_motor = mean(E_L2_motor(i,:)) * ones(1, 3);
    R_L3_motor(i+1, 1:3) = R_L3_motor(i, 1:3) + eta_rep * E_L3_motor * 0.1;
    R_L3_motor(i+1, 1:3) = max(-1, min(1, R_L3_motor(i+1, 1:3)));
    
    % ==============================================================
    % REPRESENTATION UPDATES (PLANNING REGION)
    % ==============================================================
    
    % L1_plan: goal state learning
    R_L1_plan(i+1, 1:3) = R_L1_plan(i, 1:3) + decay * eta_rep * E_L1_plan(i, 1:3) * 0.1;
    R_L1_plan(i+1, 4:6) = R_L1_plan(i, 4:6) + decay * eta_rep * E_L1_plan(i, 4:6) * 0.1;
    R_L1_plan(i+1, 4:6) = max(-2, min(2, R_L1_plan(i+1, 4:6)));
    R_L1_plan(i+1, 1) = max(workspace_bounds(1,1), min(workspace_bounds(1,2), R_L1_plan(i+1,1)));
    R_L1_plan(i+1, 2) = max(workspace_bounds(2,1), min(workspace_bounds(2,2), R_L1_plan(i+1,2)));
    R_L1_plan(i+1, 3) = max(workspace_bounds(3,1), min(workspace_bounds(3,2), R_L1_plan(i+1,3)));
    R_L1_plan(i+1, 7) = 1;
    
    % L2_plan: policy learning (WITH TASK GATING)
    % Task gate: full learning when this task is active (R_L0(i, current_trial) = 1)
    task_gate = R_L0(i, current_trial) * 0.7 + 0.3;  % Range [0.3, 1.0]
    
    coupling_plan = E_L1_plan(i,:) * W_plan_L2_to_L1;
    norm_W_plan = max(0.1, norm(W_plan_L2_to_L1, 'fro'));
    coupling_plan = coupling_plan / norm_W_plan;
    
    delta_R_L2_plan = coupling_plan - E_L2_plan(i,:);
    R_L2_plan(i+1,:) = momentum * R_L2_plan(i,:) + decay * eta_rep * delta_R_L2_plan * 0.5 * task_gate;
    R_L2_plan(i+1,:) = max(-1, min(1, R_L2_plan(i+1,:)));
    
    % L3_plan: planning output learning
    E_L3_plan = mean(E_L2_plan(i,:)) * ones(1, 3);
    R_L3_plan(i+1, 1:3) = R_L3_plan(i, 1:3) + eta_rep * E_L3_plan * 0.1 * task_gate;
    R_L3_plan(i+1, 1:3) = max(-1, min(1, R_L3_plan(i+1, 1:3)));
    
    % ==============================================================
    % WEIGHT LEARNING WITH DYNAMIC PRECISION - MOTOR REGION
    % ==============================================================
    
    layer_scale_motor_1 = max(0.1, mean(abs(R_L2_motor(i,:))));
    layer_scale_motor_3 = max(0.1, mean(abs(R_L3_motor(i,:))));
    
    % Motor weights: always learning (motor region is always active)
    dW_motor_1 = -(eta_W * pi_L1_motor / layer_scale_motor_1) * (E_L1_motor(i,:)' * R_L2_motor(i,:));
    W_motor_L2_to_L1 = W_motor_L2_to_L1 + dW_motor_1;
    
    dW_motor_3 = -(eta_W * pi_L2_motor / layer_scale_motor_3) * (E_L2_motor(i,:)' * R_L3_motor(i,:));
    W_motor_L3_to_L2 = W_motor_L3_to_L2 + dW_motor_3;
    
    % --- NEW: lateral weight updates (motor) ---
    dW_motor_L1_lat = -(eta_W * pi_L1_motor / max(0.1, mean(abs(R_L1_motor(i,:))))) * (E_L1_motor(i,:)' * R_L1_motor(i,:));
    dW_motor_L2_lat = -(eta_W * pi_L2_motor / max(0.1, mean(abs(R_L2_motor(i,:))))) * (E_L2_motor(i,:)' * R_L2_motor(i,:));
    dW_motor_L3_lat = -(eta_W * pi_L3_motor / max(0.1, mean(abs(R_L3_motor(i,:))))) * (mean(E_L2_motor(i,:))' * R_L3_motor(i,:));
    
    W_motor_L1_lat = W_motor_L1_lat + dW_motor_L1_lat;
    W_motor_L2_lat = W_motor_L2_lat + dW_motor_L2_lat;
    W_motor_L3_lat = W_motor_L3_lat + dW_motor_L3_lat;
    
    % Regularize lateral weights lightly and zero self-connections
    W_motor_L1_lat = W_motor_L1_lat * 0.9999; W_motor_L1_lat(1:n_L1_motor+1:end) = 0;
    W_motor_L2_lat = W_motor_L2_lat * 0.9999; W_motor_L2_lat(1:n_L2_motor+1:end) = 0;
    W_motor_L3_lat = W_motor_L3_lat * 0.9999; W_motor_L3_lat(1:n_L3_motor+1:end) = 0;

    % ============================================================== 
    % WEIGHT LEARNING WITH DYNAMIC PRECISION - PLANNING REGION
    % ==============================================================
    % Planning weights: task-gated learning
    % Ensure layer-scale normalization variables exist (analogous to motor region)
    layer_scale_plan_1 = max(0.1, mean(abs(R_L2_plan(i,:))));
    layer_scale_plan_3 = max(0.1, mean(abs(R_L3_plan(i,:))));

    dW_plan_1 = -(eta_W * pi_L1_plan / layer_scale_plan_1) * (E_L1_plan(i,:)' * R_L2_plan(i,:)) * task_gate;
    W_plan_L2_to_L1 = W_plan_L2_to_L1 + dW_plan_1;

    dW_plan_3 = -(eta_W * pi_L2_plan / layer_scale_plan_3) * (E_L2_plan(i,:)' * R_L3_plan(i,:)) * task_gate;
    W_plan_L3_to_L2 = W_plan_L3_to_L2 + dW_plan_3;

    % --- NEW: lateral weight updates (planning, task-gated) ---
    dW_plan_L1_lat = -(eta_W * pi_L1_plan / max(0.1, mean(abs(R_L1_plan(i,:))))) * (E_L1_plan(i,:)' * R_L1_plan(i,:)) * task_gate;
    dW_plan_L2_lat = -(eta_W * pi_L2_plan / max(0.1, mean(abs(R_L2_plan(i,:))))) * (E_L2_plan(i,:)' * R_L2_plan(i,:)) * task_gate;
    dW_plan_L3_lat = -(eta_W * pi_L3_plan / max(0.1, mean(abs(R_L3_plan(i,:))))) * (mean(E_L2_plan(i,:))' * R_L3_plan(i,:)) * task_gate;

    W_plan_L1_lat = W_plan_L1_lat + dW_plan_L1_lat;
    W_plan_L2_lat = W_plan_L2_lat + dW_plan_L2_lat;
    W_plan_L3_lat = W_plan_L3_lat + dW_plan_L3_lat;

    % Regularize lateral planning weights and zero self-connections
    W_plan_L1_lat = W_plan_L1_lat * 0.9999; W_plan_L1_lat(1:n_L1_plan+1:end) = 0;
    W_plan_L2_lat = W_plan_L2_lat * 0.9999; W_plan_L2_lat(1:n_L2_plan+1:end) = 0;
    W_plan_L3_lat = W_plan_L3_lat * 0.9999; W_plan_L3_lat(1:n_L3_plan+1:end) = 0;

    learning_trace_W(i) = norm(dW_motor_1, 'fro') + norm(dW_motor_3, 'fro') + ...
                          norm(dW_plan_1, 'fro') + norm(dW_plan_3, 'fro') + ...
                          norm(dW_motor_L1_lat,'fro') + norm(dW_motor_L2_lat,'fro') + norm(dW_motor_L3_lat,'fro') + ...
                          norm(dW_plan_L1_lat,'fro') + norm(dW_plan_L2_lat,'fro') + norm(dW_plan_L3_lat,'fro');
    
    % ==============================================================
    % DYNAMIC PRECISION UPDATES
    % ==============================================================
    
    L1_motor_error_mag = sqrt(sum(E_L1_motor(i,:).^2));
    L2_motor_error_mag = sqrt(sum(E_L2_motor(i,:).^2));
    L1_plan_error_mag = sqrt(sum(E_L1_plan(i,:).^2));
    L2_plan_error_mag = sqrt(sum(E_L2_plan(i,:).^2));
    
    L1_motor_error_history = [L1_motor_error_history, L1_motor_error_mag];
    L2_motor_error_history = [L2_motor_error_history, L2_motor_error_mag];
    L1_plan_error_history = [L1_plan_error_history, L1_plan_error_mag];
    L2_plan_error_history = [L2_plan_error_history, L2_plan_error_mag];
    
    if length(L1_motor_error_history) > window_size
        L1_motor_error_history = L1_motor_error_history(end-window_size+1:end);
        L2_motor_error_history = L2_motor_error_history(end-window_size+1:end);
        L1_plan_error_history = L1_plan_error_history(end-window_size+1:end);
        L2_plan_error_history = L2_plan_error_history(end-window_size+1:end);
    end
    
    % Update dynamic precisions (robust, smoothed, capped)
    % Use robust normalization to avoid numerical explosion from variance/division
    % Normalize variance to [0,1] using a small floor to avoid /0
    epsilon_var = 1e-6;

    % Helper anonymous to compute smoothed, step-limited precision (not used — updates written inline)

    % L1 motor
    if length(L1_motor_error_history) > 10
        err_val = L1_motor_error_history(end);
        var_val = var(L1_motor_error_history);
        var_norm = var_val / (var_val + epsilon_var);
        denom = 1 + 0.8 * err_val + 0.2 * var_norm;                  % bounded denominator
        if ~isfinite(denom) || denom <= 0, denom = 1; end

        % Compute raw precision relative to base and clamp raw to a reasonable range
        raw_pi = pi_L1_motor_base / denom;
        raw_pi = max(pi_L1_motor_base * 0.01, min(pi_L1_motor_base * 10, raw_pi));

        % Smooth and limit per-step multiplicative change
        pi_candidate = pi_smooth_alpha * pi_L1_motor + (1 - pi_smooth_alpha) * raw_pi;
        if pi_L1_motor > 0
            ratio = pi_candidate / pi_L1_motor;
            if ratio > pi_max_step_ratio, pi_candidate = pi_L1_motor * pi_max_step_ratio; end
            if ratio < 1/pi_max_step_ratio, pi_candidate = pi_L1_motor / pi_max_step_ratio; end
        end

        % Final clamp
        pi_L1_motor = max(1, min(1000, pi_candidate));

        % Diagnostics
        pi_trace_L1_motor(i) = pi_L1_motor;
        pi_raw_trace_L1_motor(i) = raw_pi;
        denom_trace_L1_motor(i) = denom;
    else
        pi_trace_L1_motor(i) = pi_L1_motor;
        pi_raw_trace_L1_motor(i) = pi_L1_motor;
        denom_trace_L1_motor(i) = 1;
    end

    % L2 motor
    if length(L2_motor_error_history) > 10
        err_val = L2_motor_error_history(end);
        var_val = var(L2_motor_error_history);
        var_norm = var_val / (var_val + epsilon_var);
        denom = 1 + 0.8 * err_val + 0.2 * var_norm;
        if ~isfinite(denom) || denom <= 0, denom = 1; end

        raw_pi = pi_L2_motor_base / denom;
        raw_pi = max(pi_L2_motor_base * 0.01, min(pi_L2_motor_base * 10, raw_pi));

        pi_candidate = pi_smooth_alpha * pi_L2_motor + (1 - pi_smooth_alpha) * raw_pi;
        if pi_L2_motor > 0
            ratio = pi_candidate / pi_L2_motor;
            if ratio > pi_max_step_ratio, pi_candidate = pi_L2_motor * pi_max_step_ratio; end
            if ratio < 1/pi_max_step_ratio, pi_candidate = pi_L2_motor / pi_max_step_ratio; end
        end

        pi_L2_motor = max(0.1, min(100, pi_candidate));

        pi_trace_L2_motor(i) = pi_L2_motor;
        pi_raw_trace_L2_motor(i) = raw_pi;
        denom_trace_L2_motor(i) = denom;
    else
        pi_trace_L2_motor(i) = pi_L2_motor;
        pi_raw_trace_L2_motor(i) = pi_L2_motor;
        denom_trace_L2_motor(i) = 1;
    end

    % L1 planning
    if length(L1_plan_error_history) > 10
        err_val = L1_plan_error_history(end);
        var_val = var(L1_plan_error_history);
        var_norm = var_val / (var_val + epsilon_var);
        denom = 1 + 0.8 * err_val + 0.2 * var_norm;
        if ~isfinite(denom) || denom <= 0, denom = 1; end

        raw_pi = pi_L1_plan_base / denom;
        raw_pi = max(pi_L1_plan_base * 0.01, min(pi_L1_plan_base * 10, raw_pi));

        pi_candidate = pi_smooth_alpha * pi_L1_plan + (1 - pi_smooth_alpha) * raw_pi;
        if pi_L1_plan > 0
            ratio = pi_candidate / pi_L1_plan;
            if ratio > pi_max_step_ratio, pi_candidate = pi_L1_plan * pi_max_step_ratio; end
            if ratio < 1/pi_max_step_ratio, pi_candidate = pi_L1_plan / pi_max_step_ratio; end
        end

        pi_L1_plan = max(1, min(1000, pi_candidate));

        pi_trace_L1_plan(i) = pi_L1_plan;
        pi_raw_trace_L1_plan(i) = raw_pi;
        denom_trace_L1_plan(i) = denom;
    else
        pi_trace_L1_plan(i) = pi_L1_plan;
        pi_raw_trace_L1_plan(i) = pi_L1_plan;
        denom_trace_L1_plan(i) = 1;
    end

    % L2 planning
    if length(L2_plan_error_history) > 10
        err_val = L2_plan_error_history(end);
        var_val = var(L2_plan_error_history);
        var_norm = var_val / (var_val + epsilon_var);
        denom = 1 + 0.8 * err_val + 0.2 * var_norm;
        if ~isfinite(denom) || denom <= 0, denom = 1; end

        raw_pi = pi_L2_plan_base / denom;
        raw_pi = max(pi_L2_plan_base * 0.01, min(pi_L2_plan_base * 10, raw_pi));

        pi_candidate = pi_smooth_alpha * pi_L2_plan + (1 - pi_smooth_alpha) * raw_pi;
        if pi_L2_plan > 0
            ratio = pi_candidate / pi_L2_plan;
            if ratio > pi_max_step_ratio, pi_candidate = pi_L2_plan * pi_max_step_ratio; end
            if ratio < 1/pi_max_step_ratio, pi_candidate = pi_L2_plan / pi_max_step_ratio; end
        end

        pi_L2_plan = max(0.1, min(100, pi_candidate));

        pi_trace_L2_plan(i) = pi_L2_plan;
        pi_raw_trace_L2_plan(i) = raw_pi;
        denom_trace_L2_plan(i) = denom;
    else
        pi_trace_L2_plan(i) = pi_L2_plan;
        pi_raw_trace_L2_plan(i) = pi_L2_plan;
        denom_trace_L2_plan(i) = 1;
    end
    
    % Debug output
    if mod(i, 1000) == 0 && i > 1000
        fprintf('  Step %d: Motor π=[%.2f,%.2f] Plan π=[%.2f,%.2f] | Interception error=%.4f\n', ...
            i, pi_L1_motor, pi_L2_motor, pi_L1_plan, pi_L2_plan, interception_error_all(i));
    end
    
end  % End main loop

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
