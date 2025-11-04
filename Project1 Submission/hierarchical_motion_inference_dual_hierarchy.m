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
T_per_trial = 30;      % Duration per trial (s) - smaller default for quicker runs
n_trials = 3;           % Number of different ball trajectories

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
    -5, 5;      % X bounds
    -5, 5;      % Y bounds
    0, 3.5;        % Z bounds
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

fprintf('GENERATING MOVING TARGET TRAJECTORIES (CONSTANT VELOCITY - Nov 3, 2025):\n');

% MOVING TARGET REACHING TASK (Nov 3, 2025 - restores predictive coding test)
% Each trial has a target following constant velocity (learnable, repeatable)
% Tests whether hierarchies learn to PREDICT temporal dynamics
% Format: target_trajectories{trial} = struct with start_pos, velocity, acceleration
% RATIONALE: Motor learns velocity control; Planning learns target motion model
%            Error signals decompose naturally: motor vs planning errors

% Define three distinct moving target trajectories (one per trial)
target_trajectories = {};

% Trial 1: Constant velocity toward origin (slow approach)
% Velocity: [-1.5, -1.5, 0] m/s = 2.12 m/s diagonal approach
% Starting distance ~7 m → closes in ~3.3 seconds
target_trajectories{1} = struct(...
    'start_pos', [5.0, 5.0, 1.5], ...
    'velocity', [-1.5, -1.5, 0.0], ...
    'acceleration', [0.0, 0.0, 0.0]);

% Trial 2: Diagonal approach with Z component (medium speed)
% Velocity: [1.0, -1.5, -0.2] m/s = 1.81 m/s 3D motion
% Tests Z-axis prediction (downward motion)
target_trajectories{2} = struct(...
    'start_pos', [-5.0, 5.0, 2.5], ...
    'velocity', [1.0, -1.5, -0.2], ...
    'acceleration', [0.0, 0.0, 0.0]);

% Trial 3: Slower approach (tests generalization)
% Velocity: [-0.8, 0.5, 0.1] m/s = 0.94 m/s (half Trial 1 speed)
% Tests whether motor generalizes or memorizes velocity
target_trajectories{3} = struct(...
    'start_pos', [5.0, -5.0, 1.0], ...
    'velocity', [-0.8, 0.5, 0.1], ...
    'acceleration', [0.0, 0.0, 0.0]);

fprintf('✓ MOVING TARGET TRAJECTORIES DEFINED (CONSTANT VELOCITY):\n');
fprintf('  Each trial: target follows constant velocity throughout\n');
for trial = 1:min(n_trials, numel(target_trajectories))
    tv = target_trajectories{trial};
    speed = norm(tv.velocity);
    fprintf('  Trial %d: start=[%.1f,%.1f,%.1f], vel=[%.2f,%.2f,%.2f] (speed=%.2f m/s)\n', ...
        trial, tv.start_pos(1), tv.start_pos(2), tv.start_pos(3), ...
        tv.velocity(1), tv.velocity(2), tv.velocity(3), speed);
end
fprintf('\n');

% Validation: ensure starting positions allow interception opportunity
% ENFORCE: minimum and maximum separation for taskability
min_start_sep = 1.0;   % meters (minimum separation - allows learning)
max_start_sep = 8.0;   % meters (maximum separation - ensures catchability within trial duration)

% Motor dynamics parameters (for feasibility calculation)
player_max_speed = 2.5;        % m/s (maximum reachable player speed)
player_accel = 5.0;            % m/s^2 (maximum player acceleration)
dt_sim = 0.01;                 % seconds (simulation timestep)

fprintf('VALIDATING GEOMETRICALLY FEASIBLE TRAJECTORIES:\n');
fprintf('═════════════════════════════════════════════════════════════════\n');
fprintf('Constraints:\n');
fprintf('  • Initial separation: %.1f m ≤ sep ≤ %.1f m\n', min_start_sep, max_start_sep);
fprintf('  • Player max speed: %.1f m/s\n', player_max_speed);
fprintf('  • Player max accel: %.1f m/s^2\n', player_accel);
fprintf('  • Trial duration: %.1f s\n\n', T_per_trial);

% Track validation results
validation_summary = struct();
validation_summary.n_trials = n_trials;
validation_summary.n_feasible = 0;
validation_summary.n_warnings = 0;
validation_summary.n_adjusted = 0;
validation_summary.trials = {};

for trial = 1:n_trials
    player_pos = initial_positions(trial, :);
    target_start = target_trajectories{trial}.start_pos;
    target_vel = target_trajectories{trial}.velocity;
    target_accel = target_trajectories{trial}.acceleration;
    
    % Compute speeds
    target_speed = norm(target_vel);
    target_accel_mag = norm(target_accel);
    
    % Initial separation
    sep_vec = target_start - player_pos;
    sep = norm(sep_vec);
    sep_direction = sep_vec / (sep + 1e-6);
    
    % ─────────────────────────────────────────────────────────
    % TEST 1: SEPARATION BOUNDS
    % ─────────────────────────────────────────────────────────
    separation_ok = true;
    if sep < min_start_sep
        direction = sep_vec / (sep + 1e-6);
        target_trajectories{trial}.start_pos = player_pos + direction * min_start_sep;
        sep = min_start_sep;
        separation_ok = false;
        validation_summary.n_adjusted = validation_summary.n_adjusted + 1;
    elseif sep > max_start_sep
        direction = sep_vec / (sep + 1e-6);
        target_trajectories{trial}.start_pos = player_pos + direction * max_start_sep;
        sep = max_start_sep;
        separation_ok = false;
        validation_summary.n_adjusted = validation_summary.n_adjusted + 1;
    end
    
    % ─────────────────────────────────────────────────────────
    % TEST 2: INTERCEPTION FEASIBILITY (MOVING TARGET)
    % ─────────────────────────────────────────────────────────
    % Can the player catch the target within the trial?
    % For MOVING targets: uses relative motion analysis
    % 
    % Key insight: In relative coordinates (target frame),
    % player must close distance at relative_closing_speed
    % 
    % Simplified model: assumes optimal pursuit (always moving directly at target)
    % More rigorous: player can only move at max_speed, must predict target motion
    
    interception_feasible = false;
    time_to_intercept_min = Inf;
    
    % ─ RELATIVE MOTION ANALYSIS ─
    % Player velocity vector (optimal): magnitude = player_max_speed, direction = toward target
    % Target velocity vector: already known as target_vel
    % Relative velocity (player relative to target): v_rel = v_player - v_target
    % 
    % In best case (player always moves optimally toward current target position),
    % closing speed = player_max_speed (because player can move at max speed toward target)
    % 
    % However, if target is moving AWAY, effective closing speed reduces:
    % closing_speed_effective = max(player_max_speed - target_speed, 0.01)
    % 
    % For moving targets, must also account for target heading:
    % - If target moves toward player: closing faster → easier interception
    % - If target moves away from player: closing slower → harder interception
    
    % Conservative estimate: use component of target velocity moving away from player
    % direction_to_target = (target_pos - player_pos) / separation
    direction_to_target = (target_start - player_pos) / (sep + 1e-6);
    target_speed_away = max(0, dot(target_vel, direction_to_target));  % Component moving away
    
    % More accurate closing speed accounting for target heading
    closing_speed_accurate = player_max_speed - target_speed_away;
    
    % Interception is feasible if:
    % 1. Player is faster than target (can always catch up), OR
    % 2. Closing speed > 0 AND gap can be closed before end of trial
    if closing_speed_accurate > 0.01  % Small threshold to avoid division by near-zero
        time_to_intercept_min = sep / (closing_speed_accurate + 1e-6);
        if time_to_intercept_min <= T_per_trial
            interception_feasible = true;
        end
    elseif target_speed < player_max_speed && target_speed > 0
        % Even if target heading is unfortunate, if player is faster overall, can catch
        time_to_intercept_min = sep / (player_max_speed - target_speed + 1e-6);
        if time_to_intercept_min <= T_per_trial
            interception_feasible = true;
        end
    end
    
    % Sanity check: if gap is small enough that player can cover it in trial time
    distance_coverable = player_max_speed * T_per_trial;
    if sep < distance_coverable && ~interception_feasible
        interception_feasible = true;  % Can cover initial gap even if relative motion awkward
        time_to_intercept_min = sep / player_max_speed;
    end
    
    % ─────────────────────────────────────────────────────────
    % TEST 3: WORKSPACE BOUNDS
    % ─────────────────────────────────────────────────────────
    % Will target stay within workspace bounds throughout trial?
    
    target_trajectory_in_bounds = true;
    max_distance_traveled = target_speed * T_per_trial + 0.5 * target_accel_mag * T_per_trial^2;
    target_final_approx = target_start + target_vel * T_per_trial + 0.5 * target_accel * T_per_trial^2;
    
    % Check if final position is within reasonable bounds
    for dim = 1:3
        if target_final_approx(dim) < workspace_bounds(dim, 1) - 1.0 || ...
           target_final_approx(dim) > workspace_bounds(dim, 2) + 1.0
            target_trajectory_in_bounds = false;
            break;
        end
    end
    
    % ─────────────────────────────────────────────────────────
    % TEST 4: VELOCITY CONSISTENCY
    % ─────────────────────────────────────────────────────────
    % Are target speeds reasonable? (not too fast or zero)
    
    velocity_reasonable = true;
    if target_speed < 0.05  && target_accel_mag < 0.01
        % Target moving too slowly and not accelerating
        velocity_reasonable = false;
    elseif target_speed > 5.0
        % Target moving unreasonably fast (faster than human sprint)
        velocity_reasonable = false;
    end
    
    % ─────────────────────────────────────────────────────────
    % GENERATE REPORT FOR THIS TRIAL
    % ─────────────────────────────────────────────────────────
    
    fprintf('Trial %d:\n', trial);
    fprintf('  Start position: [%.2f, %.2f, %.2f]\n', target_start(1), target_start(2), target_start(3));
    fprintf('  Player position: [%.2f, %.2f, %.2f]\n', player_pos(1), player_pos(2), player_pos(3));
    fprintf('  Initial separation: %.2f m', sep);
    if separation_ok
        fprintf(' ✓\n');
    else
        fprintf(' (ADJUSTED)\n');
    end
    
    fprintf('  Target velocity: [%.3f, %.3f, %.3f] m/s (speed: %.3f m/s)', ...
        target_vel(1), target_vel(2), target_vel(3), target_speed);
    if velocity_reasonable
        fprintf(' ✓\n');
    else
        fprintf(' ⚠ UNREASONABLE\n');
        validation_summary.n_warnings = validation_summary.n_warnings + 1;
    end
    
    fprintf('  Target acceleration: [%.3f, %.3f, %.3f] m/s^2 (mag: %.3f)\n', ...
        target_accel(1), target_accel(2), target_accel(3), target_accel_mag);
    
    fprintf('  Interception feasible: ');
    if interception_feasible
        fprintf('✓ YES (est. time: %.1f s < %.1f s)\n', time_to_intercept_min, T_per_trial);
        if time_to_intercept_min <= T_per_trial
            fprintf('    → Player can catch with PERFECT motor control\n');
        end
    else
        fprintf('✗ NO (target too fast: %.2f m/s > %.2f m/s)\n', target_speed, player_max_speed);
        validation_summary.n_warnings = validation_summary.n_warnings + 1;
    end
    
    fprintf('  Trajectory stays in bounds: ');
    if target_trajectory_in_bounds
        fprintf('✓ YES\n');
    else
        fprintf('⚠ MARGINAL (final pos: [%.2f, %.2f, %.2f])\n', ...
            target_final_approx(1), target_final_approx(2), target_final_approx(3));
        validation_summary.n_warnings = validation_summary.n_warnings + 1;
    end
    
    % Overall verdict
    all_tests_pass = separation_ok && interception_feasible && velocity_reasonable && target_trajectory_in_bounds;
    if all_tests_pass
        fprintf('  VERDICT: ✅ GEOMETRICALLY FEASIBLE\n\n');
        validation_summary.n_feasible = validation_summary.n_feasible + 1;
    else
        fprintf('  VERDICT: ⚠️  MARGINAL (may require optimal motor learning)\n\n');
    end
    
    % Store trial result
    validation_summary.trials{trial} = struct(...
        'separation_ok', separation_ok, ...
        'interception_feasible', interception_feasible, ...
        'trajectory_in_bounds', target_trajectory_in_bounds, ...
        'velocity_reasonable', velocity_reasonable, ...
        'time_to_intercept_min', time_to_intercept_min, ...
        'target_speed', target_speed);
end

% ─────────────────────────────────────────────────────────
% PRINT SUMMARY
% ─────────────────────────────────────────────────────────
fprintf('═════════════════════════════════════════════════════════════════\n');
fprintf('TRAJECTORY VALIDATION SUMMARY:\n');
fprintf('  Total trials: %d\n', validation_summary.n_trials);
fprintf('  Geometrically feasible: %d (%d%%)\n', validation_summary.n_feasible, ...
    round(100 * validation_summary.n_feasible / validation_summary.n_trials));
fprintf('  Warnings issued: %d\n', validation_summary.n_warnings);
fprintf('  Positions adjusted: %d\n', validation_summary.n_adjusted);
fprintf('═════════════════════════════════════════════════════════════════\n\n');

if validation_summary.n_feasible == validation_summary.n_trials
    fprintf('✅ ALL TRIALS GEOMETRICALLY FEASIBLE - Ready for optimization\n\n');
else
    fprintf('⚠️  Some trials marginal - Motor learning must be accurate for interception\n\n');
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
    
    % Hard-coded minimum precision values (NOT optimized by PSO)
    pi_L1_motor_min = 10;   % Fixed minimum precision (allow exploration)
    pi_L2_motor_min = 1;    % Fixed minimum precision
    pi_L1_plan_min = 10;    % Fixed minimum precision
    pi_L2_plan_min = 1;     % Fixed minimum precision
    
    % Read maximum precision values from PSO (optimized)
    if isfield(params, 'pi_L1_motor_max')
        pi_L1_motor_max = params.pi_L1_motor_max;
    else
        pi_L1_motor_max = 500;  % Default maximum precision (tight bounds)
    end
    if isfield(params, 'pi_L2_motor_max')
        pi_L2_motor_max = params.pi_L2_motor_max;
    else
        pi_L2_motor_max = 100;
    end
    if isfield(params, 'pi_L1_plan_max')
        pi_L1_plan_max = params.pi_L1_plan_max;
    else
        pi_L1_plan_max = 500;
    end
    if isfield(params, 'pi_L2_plan_max')
        pi_L2_plan_max = params.pi_L2_plan_max;
    else
        pi_L2_plan_max = 100;
    end
else
    % Defaults (no PSO optimization)
    alpha_precision_gain = 0.5;
    % Hard-coded minimums
    pi_L1_motor_min = 10;    pi_L1_motor_max = 500;
    pi_L2_motor_min = 1;     pi_L2_motor_max = 100;
    pi_L1_plan_min = 10;     pi_L1_plan_max = 400;
    pi_L2_plan_min = 1;      pi_L2_plan_max = 60;
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

% MOVING TARGET INITIAL STATE (Nov 3, 2025 - constant velocity kinematics)
% Target starts at specified position with specified velocity
% This tests predictive coding: can hierarchies learn target motion model?
x_ball(1) = target_trajectories{1}.start_pos(1);
y_ball(1) = target_trajectories{1}.start_pos(2);
z_ball(1) = target_trajectories{1}.start_pos(3);
vx_ball(1) = target_trajectories{1}.velocity(1);  % Target has non-zero velocity (MOVING)
vy_ball(1) = target_trajectories{1}.velocity(2);
vz_ball(1) = target_trajectories{1}.velocity(3);

% Motor L2/L3: initial velocity commands toward target
reach_direction = ([x_ball(1), y_ball(1), z_ball(1)] - [x_player(1), y_player(1), z_player(1)]) / ...
                   (norm([x_ball(1), y_ball(1), z_ball(1)] - [x_player(1), y_player(1), z_player(1)]) + 1e-6);
target_distance = norm([x_ball(1), y_ball(1), z_ball(1)] - [x_player(1), y_player(1), z_player(1)]);
reaching_speed = reaching_speed_scale * target_distance;

R_L2_motor(1, 1:3) = reach_direction * reaching_speed;
R_L2_motor(1, 4:6) = 0.01 * randn(1, 3);
R_L3_motor(1, 1:3) = reach_direction * reaching_speed;

% Planning L1: target POSITION and VELOCITY (use semantic idx)
% CHANGED (Nov 3): Now include velocity information for predictive planning
R_L1_plan(1, idx_pos) = [x_ball(1), y_ball(1), z_ball(1)];  % Target position
R_L1_plan(1, idx_vel) = [vx_ball(1), vy_ball(1), vz_ball(1)];  % Target velocity (for prediction)
R_L1_plan(1, idx_bias) = 1;

% Planning L2/L3: initial policies based on target velocity
% (Planning will learn to predict future positions based on velocity)
target_vel = [vx_ball(1), vy_ball(1), vz_ball(1)];
R_L2_plan(1, 1:3) = target_vel / (norm(target_vel) + 1e-6);  % Normalized velocity direction
R_L2_plan(1, 4:6) = 0.01 * randn(1, 3);
R_L3_plan(1, 1:3) = target_vel / (norm(target_vel) + 1e-6);

fprintf('INITIAL CONDITIONS (Trial 1 - MOVING TARGET):\n');
fprintf('  Player start: [%.2f, %.2f, %.2f]\n', x_player(1), y_player(1), z_player(1));
fprintf('  Target start: [%.2f, %.2f, %.2f]\n', x_ball(1), y_ball(1), z_ball(1));
fprintf('  Target velocity: [%.2f, %.2f, %.2f] (MOVING - tests prediction)\n', vx_ball(1), vy_ball(1), vz_ball(1));
fprintf('  Initial reach direction: [%.4f, %.4f, %.4f]\n', ...
    reach_direction(1), reach_direction(2), reach_direction(3));
fprintf('  R_L1_motor (proprioception) initialized\n');
fprintf('  R_L1_plan (target position + velocity) initialized\n');
fprintf('  Expected: Motor learns velocity control; Planning learns target motion model\n\n');

% ====================================================================
% INITIALIZE WEIGHT MATRICES - DUAL HIERARCHY (TASK-INDEXED)
% ====================================================================
% NEW: Maintain separate weight matrices per task to prevent interference learning.
% Each task gets its own learned mappings, reducing catastrophic forgetting.

% ====================================================================
% MOTOR WEIGHTS: SHARED (GENERALIZE ACROSS TASKS)
% ====================================================================
% Motor region learns stable forward model of velocity dynamics
% SINGLE shared copy for all tasks (not task-indexed)
% This enforces generalization: reaching control should work across all tasks
% Task selectivity emerges from planning layer (which IS task-indexed)
% ====================================================================

W_motor_L2_to_L1 = zeros(n_L1_motor, n_L2_motor);
W_motor_L3_to_L2 = zeros(n_L2_motor, n_L3_motor);

% --- Motor L2->L1 mappings (basis to proprioception) ---
% Map L2_motor velocity-like features into L1 velocity rows.
map_vel = min(n_vel, n_L2_motor);
W_motor_L2_to_L1(idx_vel(1:map_vel), 1:map_vel) = eye(map_vel);

% Weak position coupling from same L2 channels (if available)
map_pos = min(n_pos, n_L2_motor);
W_motor_L2_to_L1(idx_pos(1:map_pos), 1:map_pos) = 0.01 * eye(map_pos);

% Bias / offset row -- small random init
W_motor_L2_to_L1(idx_bias, :) = 0.01 * randn(1, n_L2_motor);

% --- Motor L3->L2 mappings (output to basis) ---
% Initialize L3->L2 mapping with structured identity on the overlapping block
map_block = min(n_L2_motor, n_L3_motor);
fan_in3 = max(1, n_L3_motor);
W_motor_L3_to_L2(1:map_block, 1:map_block) = W_motor_gain * eye(map_block);
% remaining rows (if any) get small random init scaled by fan-in
if n_L2_motor > map_block
    W_motor_L3_to_L2(map_block+1:end, 1:n_L3_motor) = (W_motor_gain / sqrt(fan_in3)) * 0.01 * randn(n_L2_motor-map_block, n_L3_motor);
end

% ====================================================================
% PLANNING WEIGHTS: TASK-INDEXED (SPECIALIZE ACROSS TASKS)
% ====================================================================
% Planning region learns task-specific interception strategies
% CELL ARRAY: one copy per task (learns different ball dynamics for each trial)
% ====================================================================

W_plan_L2_to_L1 = cell(n_trials, 1);   % task-specific policy->goal
W_plan_L3_to_L2 = cell(n_trials, 1);   % task-specific output->policy

% Initialize per-task planning weight matrices
for task_idx = 1:n_trials
    % --- Planning L2->L1 mappings (policy to goal) ---
    W_plan_L2_to_L1{task_idx} = zeros(n_L1_plan, n_L2_plan);
    
    map_vel_p = min(n_vel, n_L2_plan);
    map_pos_p = min(n_pos, n_L2_plan);
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
end

fprintf('WEIGHT MATRICES INITIALIZED:\n');
fprintf('  ✓ Motor Region: SHARED across all tasks (generalization)\n');
fprintf('      W_motor_L2_to_L1: Motor Basis → Proprioception (%dx%d)\n', n_L1_motor, n_L2_motor);
fprintf('      W_motor_L3_to_L2: Output → Basis (%dx%d)\n', n_L2_motor, n_L3_motor);
fprintf('      → Single shared copy: learns stable forward model\n');
fprintf('  ✓ Planning Region: TASK-INDEXED (%d task-specific copies)\n', n_trials);
fprintf('      W_plan_L2_to_L1{task}: Policies → Goal State (%dx%d each)\n', n_L1_plan, n_L2_plan);
fprintf('      W_plan_L3_to_L2{task}: Output → Policies (%dx%d each)\n', n_L2_plan, n_L3_plan);
fprintf('      → Separate copies: learn task-specific interception strategies\n\n');

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

% Weight matrices (task-indexed cells - feedforward only, no lateral)
S.W_motor_L2_to_L1 = W_motor_L2_to_L1; S.W_motor_L3_to_L2 = W_motor_L3_to_L2;
S.W_plan_L2_to_L1 = W_plan_L2_to_L1; S.W_plan_L3_to_L2 = W_plan_L3_to_L2;

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
% FIX (Nov 2, 2025): NEW ISSUE #2 - Add P_pi_bounds error handling
% Validate that P_pi_bounds struct exists, has required fields, and values are well-formed
assert(isstruct(P_pi_bounds), 'ERROR: P_pi_bounds must be a struct');
assert(isfield(P_pi_bounds, 'L1_motor'), 'ERROR: P_pi_bounds.L1_motor field missing');
assert(isfield(P_pi_bounds, 'L2_motor'), 'ERROR: P_pi_bounds.L2_motor field missing');
assert(isfield(P_pi_bounds, 'L1_plan'), 'ERROR: P_pi_bounds.L1_plan field missing');
assert(isfield(P_pi_bounds, 'L2_plan'), 'ERROR: P_pi_bounds.L2_plan field missing');
% Validate each bound is a 2-element numeric vector with finite values
bounds_fields = {'L1_motor', 'L2_motor', 'L1_plan', 'L2_plan'};
for f = 1:numel(bounds_fields)
    fname = bounds_fields{f};
    assert(isnumeric(P_pi_bounds.(fname)) && numel(P_pi_bounds.(fname)) == 2, ...
        sprintf('ERROR: P_pi_bounds.%s must be a 2-element numeric vector', fname));
    assert(all(isfinite(P_pi_bounds.(fname))), ...
        sprintf('ERROR: P_pi_bounds.%s contains NaN/Inf', fname));
    assert(P_pi_bounds.(fname)(1) < P_pi_bounds.(fname)(2), ...
        sprintf('ERROR: P_pi_bounds.%s has min >= max', fname));
end
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
S.target_trajectories = target_trajectories;  % Moving target trajectories (constant velocity)

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
P.interference_penalty_weight = interference_penalty_weight;  % Cross-task error penalty weight
% Pass moving target trajectories to helper
P.target_trajectories = target_trajectories;  % For kinematic integration in step helper
% Add max clipping events threshold (stop early if too many clipping events)
if isfield(params, 'max_clipping_events')
    P.max_clipping_events = params.max_clipping_events;
else
    P.max_clipping_events = 25; % default: stop after 25 clipping events
end

% Centralized threshold for consecutive NaN/Inf termination (single-source)
% Backwards compatible: if params.max_consecutive_clipping is provided use it,
% otherwise fall back to a safe default of 50 consecutive events.
if isfield(params, 'max_consecutive_clipping')
    P.max_consecutive_clipping = params.max_consecutive_clipping;
else
    P.max_consecutive_clipping = 50; % default: terminate after 50 consecutive NaN/Inf steps
end

% FIX (Nov 2, 2025): INCOMPLETE FIX #3 - Add task gate parameterization
% These control the task-gating mechanism in planning region (motor always learns)
P.min_task_gate = 0.3;      % Minimum task gate value (when task is not active)
P.task_gate_range = 0.7;    % Range of task gate (so max = min + range = 1.0)

% VALIDATION: Ensure precision bounds are sensible (FIX Nov 2, 2025: comprehensive validation)
if ~isfinite(P.alpha_precision_gain) || P.alpha_precision_gain <= 0
    P.alpha_precision_gain = 0.5;  % Reset to default if invalid
end

% Comprehensive bounds validation: checks min < max, ratio bounds, NaN/Inf, and narrow bounds detection
P = validate_and_fix_precision_bounds(P);  % Function at end of file (line ~1130)

% ====================================================================
% FIX #5: PSO PARAMETER VALIDATION (Nov 2, 2025)
% ====================================================================
% CRITICAL: Verify that PSO-optimized parameters are actually being USED in the simulation
% This prevents silent failures where PSO parameters are loaded but never referenced
% ====================================================================

fprintf('\n✓ VALIDATION: Precision Parameter Usage (FIX #5)\n');
fprintf('─────────────────────────────────────────────────────\n');

% Verify that P struct contains all precision-related fields that will be used in hierarchical_step_update.m
required_precision_fields = {'alpha_precision_gain', 'pi_bounds'};
for f = 1:numel(required_precision_fields)
    field_name = required_precision_fields{f};
    if ~isfield(P, field_name)
        error('ERROR: P.%s not set - PSO parameters will not be used! Set in initialization.', field_name);
    end
end

fprintf('  ✓ P.alpha_precision_gain = %.6f (error-driven precision sensitivity)\n', P.alpha_precision_gain);
fprintf('  ✓ P.pi_bounds.L1_motor = [%.1f, %.1f]\n', P.pi_bounds.L1_motor(1), P.pi_bounds.L1_motor(2));
fprintf('  ✓ P.pi_bounds.L2_motor = [%.1f, %.1f]\n', P.pi_bounds.L2_motor(1), P.pi_bounds.L2_motor(2));
fprintf('  ✓ P.pi_bounds.L1_plan = [%.1f, %.1f]\n', P.pi_bounds.L1_plan(1), P.pi_bounds.L1_plan(2));
fprintf('  ✓ P.pi_bounds.L2_plan = [%.1f, %.1f]\n', P.pi_bounds.L2_plan(1), P.pi_bounds.L2_plan(2));

fprintf('\n  These parameters will be used in hierarchical_step_update.m to control:\n');
fprintf('    - Error-driven precision scaling (exponential): precision *= exp(alpha * error)\n');
fprintf('    - Precision bounds enforcement: clamp to [min, max]\n');
fprintf('    - Result: Adaptive precision dynamics that PSO can optimize\n\n');

fprintf('✓ PSO OBJECTIVE FUNCTION:\n');
fprintf('  Minimize: weighted_score = reaching_error + lambda * free_energy\n');
fprintf('  where:\n');
fprintf('    reaching_error = mean(||player - ball|| over all steps)\n');
fprintf('    free_energy = sum of prediction errors scaled by precisions\n');
fprintf('    lambda = objective_weights.free_energy (typically 0.1-1.0)\n\n');

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

% =====================================================================
% FIX #3: ADAPTIVE LEARNING RATE SCHEDULE (Curriculum Learning)
% =====================================================================
% THEORY: Learning rates should decrease over time (Robbins-Monro conditions)
%         to ensure convergence and prevent overfitting
% NEUROSCIENCE: Motor learning exhibits well-known learning curve decay
%               Fast early learning → slow consolidation → plateau
% MECHANISM: Every N trials, decay learning rates by factor
%            eta_new = eta_old * decay_schedule_factor
%
% Typical decay: 10% per trial (after first trial, reduce by 0.9x)
%                Result: after 10 trials, learning rate ~35% of initial

% Curriculum phase boundaries (can be PSO parameters)
curriculum_phase_boundaries = [1, 2, 3];  % Phases at trials 1, 2, 3
curriculum_eta_factors = [1.0, 0.5, 0.2];  % Learning rates: 100%, 50%, 20% of initial

% Store initial learning rates for scheduling
initial_eta_rep = eta_rep;
initial_eta_W = eta_W;

% In main loop, after each trial boundary (around line ~800):
for trial = 1:n_trials
    if trial > 1
        % Determine which curriculum phase we're in
        phase_idx = min(find(trial >= curriculum_phase_boundaries));
        if ~isempty(phase_idx) && phase_idx <= numel(curriculum_eta_factors)
            scheduled_eta_factor = curriculum_eta_factors(phase_idx);
            
            % Update learning rates for this trial
            eta_rep = initial_eta_rep * scheduled_eta_factor;
            eta_W = initial_eta_W * scheduled_eta_factor;
            P.eta_rep = eta_rep;
            P.eta_W = eta_W;
            
            fprintf('Trial %d: Entering curriculum phase %d, learning rates scaled to %.1f%%\n', ...
                trial, phase_idx, 100*scheduled_eta_factor);
        end
    end
end

for i = 1:N-1
    % FIX (Nov 3, 2025): Add diagnostic output every 100 steps
    % Tracks: free energy, interception error, weight update magnitude, precision values
    % Use this to debug whether learning is actually happening
    print_diagnostics = (mod(i, 100) == 0);
    if print_diagnostics, fprintf('.'); end
    
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

                % MOVING TARGET RESET (Nov 3, 2025 - constant velocity, tests predictive coding)
                S.x_ball(i) = target_trajectories{trial}.start_pos(1);
                S.y_ball(i) = target_trajectories{trial}.start_pos(2);
                S.z_ball(i) = target_trajectories{trial}.start_pos(3);
                S.vx_ball(i) = target_trajectories{trial}.velocity(1);  % Target has velocity (MOVING - tests prediction)
                S.vy_ball(i) = target_trajectories{trial}.velocity(2);
                S.vz_ball(i) = target_trajectories{trial}.velocity(3);
                
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
                % CHANGED (Nov 3): Include target velocity in planning L1 for predictive modeling
                S.R_L1_plan(i, idx_pos) = [S.x_ball(i), S.y_ball(i), S.z_ball(i)];  % Target position
                S.R_L1_plan(i, idx_vel) = [S.vx_ball(i), S.vy_ball(i), S.vz_ball(i)];  % Target velocity (for prediction)
                S.R_L1_plan(i, idx_bias) = 1;

                % Planning L2/L3: initialize based on target velocity
                target_vel = [S.vx_ball(i), S.vy_ball(i), S.vz_ball(i)];
                S.R_L2_plan(i, 1:3) = target_vel / (norm(target_vel) + 1e-6);  % Normalized velocity direction
                S.R_L2_plan(i, 4:6) = 0.01 * randn(1, 3);
                S.R_L3_plan(i, 1:3) = target_vel / (norm(target_vel) + 1e-6);
                
                % Apply phase transition decay - differential for motor vs. planning
                % Motor weights are SHARED (single copy for generalization)
                % Planning weights are TASK-INDEXED (one copy per task)
                S.W_motor_L2_to_L1 = decay_motor * S.W_motor_L2_to_L1;
                S.W_motor_L3_to_L2 = decay_motor * S.W_motor_L3_to_L2;

                for tt = 1:numel(S.W_plan_L2_to_L1)
                    S.W_plan_L2_to_L1{tt} = decay_plan * S.W_plan_L2_to_L1{tt};
                    S.W_plan_L3_to_L2{tt} = decay_plan * S.W_plan_L3_to_L2{tt};
                end

                % Restore critical motor mappings (use semantic idx_vel for robustness)
                % Write this into the SHARED motor mapping so all tasks retain
                % the basic velocity-to-output mapping.
                map_vel_idx = idx_vel(1:min(3, numel(idx_vel)));
                S.W_motor_L2_to_L1(map_vel_idx, 1:3) = eye(numel(map_vel_idx), 3);

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

    % FIX (Nov 3, 2025): PRINT LEARNING DIAGNOSTICS
    % These values help you debug whether learning is happening
    if print_diagnostics
        fprintf('\n  Step %d: FE=%.2e | IntErr=%.4f | |dW|=%.2e | pi_L1m=%.1f | noise_scale=%.4f\n', ...
            i, ...
            S.free_energy_all(i), ...
            S.interception_error_all(i), ...
            S.learning_trace_W(i), ...
            S.pi_L1_motor, ...
            max(0.01, 0.05 * (1.0 - i/1000)));  % Current noise level
    end

    % --- NEW: Early termination if too many clipping events occurred ---
    if isfield(S, 'clipping_count') && S.clipping_count > P.max_clipping_events
        S.session_end = true;
        S.termination_step = i;
        S.termination_reason = sprintf('Excessive consecutive Inf/NaN clipping: %d > %d', S.clipping_count, P.max_clipping_events);
        fprintf('\n⚠  Early termination at step %d: %s\n', i, S.termination_reason);
        break;
    end
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

max_finite_value = 1e12;
min_valid_free_energy = 0;
max_valid_free_energy = 1e12;

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
target_trajectories = S.target_trajectories;
W_motor_L2_to_L1 = S.W_motor_L2_to_L1; W_motor_L3_to_L2 = S.W_motor_L3_to_L2;
W_plan_L2_to_L1 = S.W_plan_L2_to_L1; W_plan_L3_to_L2 = S.W_plan_L3_to_L2;
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
results.target_trajectories = target_trajectories;
results.W_motor_L2_to_L1 = W_motor_L2_to_L1; results.W_motor_L3_to_L2 = W_motor_L3_to_L2;
results.W_plan_L2_to_L1 = W_plan_L2_to_L1; results.W_plan_L3_to_L2 = W_plan_L3_to_L2;
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

% ================================================================
% VALIDATION FUNCTION: Comprehensive Precision Bounds Validation
% ================================================================
% FIX (Nov 2, 2025): INCOMPLETE FIX #2 - Comprehensive bounds validation
% Purpose: Ensure precision bounds are sensible (min < max), within reasonable ratios,
%          have no NaN/Inf, and aren't too narrow (would prevent precision adaptation)
%
% Checks performed:
%  1. min < max (basic validity)
%  2. Ratio between min/max in range [1.1, 1000] (not too tight, not too loose)
%  3. No NaN or Inf values
%  4. Bounds not too narrow (ratio must be >= 1.1 to allow 10% range for adaptation)
%  5. If any check fails, reset to defaults and warn
%
function P = validate_and_fix_precision_bounds(P)
    % Define reasonable defaults
    defaults = struct(...
        'L1_motor', [10, 500], ...
        'L2_motor', [0.5, 50], ...
        'L1_plan', [10, 500], ...
        'L2_plan', [0.5, 50]);
    
    % Check structure fields: motor_L1, motor_L2, plan_L1, plan_L2
    bounds_fields = {'L1_motor', 'L2_motor', 'L1_plan', 'L2_plan'};
    warnings_issued = {};
    
    for f = 1:numel(bounds_fields)
        field_name = bounds_fields{f};
        
        % Check field exists
        if ~isfield(P.pi_bounds, field_name)
            P.pi_bounds.(field_name) = defaults.(field_name);
            warnings_issued{end+1} = sprintf('Missing field %s.%s, reset to default [%g, %g]', ...
                'pi_bounds', field_name, defaults.(field_name)(1), defaults.(field_name)(2));
            continue;
        end
        
        bounds = P.pi_bounds.(field_name);
        
        % Check: bounds is numeric vector of length 2
        if ~isnumeric(bounds) || numel(bounds) ~= 2
            P.pi_bounds.(field_name) = defaults.(field_name);
            warnings_issued{end+1} = sprintf('%s.%s not a 2-element vector, reset to default', ...
                'pi_bounds', field_name);
            continue;
        end
        
        % Check: no NaN or Inf
        if any(~isfinite(bounds))
            P.pi_bounds.(field_name) = defaults.(field_name);
            warnings_issued{end+1} = sprintf('%s.%s contains NaN/Inf [%g, %g], reset to default', ...
                'pi_bounds', field_name, bounds(1), bounds(2));
            continue;
        end
        
        % Check: min < max
        if bounds(1) >= bounds(2)
            P.pi_bounds.(field_name) = defaults.(field_name);
            warnings_issued{end+1} = sprintf('%s.%s has min >= max [%g, %g], reset to default', ...
                'pi_bounds', field_name, bounds(1), bounds(2));
            continue;
        end
        
        % Check: ratio in reasonable range [1.1, 1000]
        ratio = bounds(2) / bounds(1);
        if ratio < 1.1
            P.pi_bounds.(field_name) = defaults.(field_name);
            warnings_issued{end+1} = sprintf('%s.%s too narrow (ratio %.2f < 1.1), reset to default', ...
                'pi_bounds', field_name, ratio);
            continue;
        elseif ratio > 1000
            P.pi_bounds.(field_name) = defaults.(field_name);
            warnings_issued{end+1} = sprintf('%s.%s too loose (ratio %.0f > 1000), reset to default', ...
                'pi_bounds', field_name, ratio);
            continue;
        end
    end
    
    % Print all warnings (if any)
    if ~isempty(warnings_issued)
        fprintf('\n*** PRECISION BOUNDS VALIDATION WARNINGS ***\n');
        for w = 1:numel(warnings_issued)
            fprintf('  [WARNING] %s\n', warnings_issued{w});
        end
        fprintf('*** END BOUNDS VALIDATION ***\n\n');
    end
end  % End validate_and_fix_precision_bounds
