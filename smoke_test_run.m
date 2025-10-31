% build params (map CSV/PSO names -> function names)
params = struct();
params.eta_rep = 0.00121192463296075;
params.eta_W  = 2.63321983368896e-05;
params.momentum = 0.952164271513988;
params.decay_plan  = 0.99;                 % decay_L2_goal
params.decay_motor = 0.100892693348881;   % decay_L1_motor
params.motor_gain = 1;
params.damping = 0.711876080199316;
params.reaching_speed_scale = 0.460970909671357;
params.W_plan_gain  = 0.311927505462757;  % W_L2_goal_gain
params.W_motor_gain = 0.001;              % W_L1_pos_gain
params.weight_decay = 0.999;

% task / physics options you asked about
params.T_per_trial = 4000;      % seconds per trial
params.n_trials = 4;           % number of different trajectories
params.ensure_opportunity = true;
params.opportunity_radius = 1.0;
params.opportunity_max_attempts = 1000;
params.opportunity_directed_prob = 0.3;
params.opportunity_directed_speed_range = [0.8, 2.5];
params.ensure_reachable = true;
params.player_max_speed = 2.5;          % m/s
params.player_max_accel = 6.0;          % m/s^2
params.reachable_tolerance = 0.10;

% other useful toggles
params.save_results = true;   % don't write MAT file for smoke test
params.ground_z = 0;           % set ground plane to z=0

% run without plots (fast / for PSO)
results = hierarchical_motion_inference_dual_hierarchy(params, true);