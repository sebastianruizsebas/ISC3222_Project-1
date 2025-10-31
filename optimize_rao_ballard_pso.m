% PARTICLE SWARM OPTIMIZATION FOR 3D RAO & BALLARD MODEL
% ========================================================
%
% Uses Particle Swarm Optimization to find optimal learning parameters
% for the hierarchical motion inference model with 3D reaching task.
%
% PSO is more efficient than random search, typically finding better parameters
% in fewer evaluations by using social intelligence (particle interactions).

clear all;
close all;
clc;

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  PARTICLE SWARM OPTIMIZATION - 3D RAO & BALLARD MODEL       ║\n');
fprintf('║  Hierarchical Motion Inference with Active Inference        ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% PSO CONFIGURATION
% ====================================================================

% Number of particles (swarm size)
num_particles = 20;  % Each particle = one parameter set

% Number of PSO iterations (generations)
num_iterations = 30;  % Each iteration = all particles tested

% Total evaluations will be: num_particles * num_iterations = 600 trials
total_evals = num_particles * num_iterations;

fprintf('PSO CONFIGURATION:\n');
fprintf('  Number of particles (swarm size): %d\n', num_particles);
fprintf('  Number of iterations (generations): %d\n', num_iterations);
fprintf('  Total model evaluations: %d\n\n', total_evals);

% PSO hyperparameters (standard values)
w = 0.7;        % Inertia weight (controls momentum of particles)
c1 = 1.5;       % Cognitive parameter (attraction to particle's best)
c2 = 1.5;       % Social parameter (attraction to swarm's best)
noise_scale = 0.05;  % Noise scale for stochastic perturbations (5% of parameter range)

fprintf('PSO HYPERPARAMETERS:\n');
fprintf('  Inertia weight (w): %.2f\n', w);
fprintf('  Cognitive parameter (c1): %.2f\n', c1);
fprintf('  Social parameter (c2): %.2f\n', c2);
fprintf('  Noise scale (stochastic exploration): %.2f\n\n', noise_scale);

% ====================================================================
% DEFINE SEARCH SPACE FOR PARAMETERS
% ====================================================================

% Parameter bounds - now includes 11 parameters (extended from 3)
param_bounds = struct();

% LEARNING RATES (log scale)
param_bounds.eta_rep.log_min = -4;      % 10^-4 = 0.0001
param_bounds.eta_rep.log_max = -1;      % 10^-1 = 0.1
param_bounds.eta_W.log_min = -6;        % 10^-6 = 0.000001
param_bounds.eta_W.log_max = -1;        % 10^-1 = 0.1
param_bounds.momentum.min = 0.70;       % Linear scale
param_bounds.momentum.max = 0.98;

% WEIGHT DECAY (linear scale, affects learning across trials)
param_bounds.decay_L2_goal.min = 0.10;  % L2 weight decay
param_bounds.decay_L2_goal.max = 0.99;
param_bounds.decay_L1_motor.min = 0.10; % L1 weight decay
param_bounds.decay_L1_motor.max = 0.99;

% MOTOR DYNAMICS (linear scale, affects trajectory quality)
param_bounds.motor_gain.min = 0.1;      % Initial motor command strength
param_bounds.motor_gain.max = 1.0;
param_bounds.damping.min = 0.70;        % Velocity dampening
param_bounds.damping.max = 0.99;
param_bounds.reaching_speed_scale.min = 0.1;  % Scale for initial reaching speed
param_bounds.reaching_speed_scale.max = 1.0;

% WEIGHT INITIALIZATION GAINS (linear scale, affects convergence)
param_bounds.W_L2_goal_gain.min = 0.1;  % L3→L2 weight initialization
param_bounds.W_L2_goal_gain.max = 1.0;
param_bounds.W_L1_pos_gain.min = 0.001; % L2→L1 weight initialization
param_bounds.W_L1_pos_gain.max = 0.1;

% Objective function weights
% For 3D reaching, primary metric is reaching distance improvement
objective_weights = struct('reaching_distance', 1.0, 'position_rmse', 0.5);

fprintf('PARAMETER SEARCH SPACE (11-DIMENSIONAL):\n');
fprintf('═════════════════════════════════════════\n');
fprintf('LEARNING RATES:\n');
fprintf('  eta_rep:  [%.6f, %.6f] (log scale: 10^[%d, %d])\n', ...
    10^param_bounds.eta_rep.log_min, 10^param_bounds.eta_rep.log_max, ...
    param_bounds.eta_rep.log_min, param_bounds.eta_rep.log_max);
fprintf('  eta_W:    [%.6f, %.6f] (log scale: 10^[%d, %d])\n', ...
    10^param_bounds.eta_W.log_min, 10^param_bounds.eta_W.log_max, ...
    param_bounds.eta_W.log_min, param_bounds.eta_W.log_max);
fprintf('  momentum: [%.2f, %.2f] (linear scale)\n', ...
    param_bounds.momentum.min, param_bounds.momentum.max);

fprintf('WEIGHT DECAY (trial transfer):\n');
fprintf('  decay_L2_goal: [%.2f, %.2f]\n', ...
    param_bounds.decay_L2_goal.min, param_bounds.decay_L2_goal.max);
fprintf('  decay_L1_motor: [%.2f, %.2f]\n', ...
    param_bounds.decay_L1_motor.min, param_bounds.decay_L1_motor.max);

fprintf('MOTOR DYNAMICS (trajectory quality):\n');
fprintf('  motor_gain: [%.2f, %.2f]\n', ...
    param_bounds.motor_gain.min, param_bounds.motor_gain.max);
fprintf('  damping: [%.2f, %.2f]\n', ...
    param_bounds.damping.min, param_bounds.damping.max);
fprintf('  reaching_speed_scale: [%.2f, %.2f]\n', ...
    param_bounds.reaching_speed_scale.min, param_bounds.reaching_speed_scale.max);

fprintf('WEIGHT INITIALIZATION (convergence speed):\n');
fprintf('  W_L2_goal_gain: [%.3f, %.2f]\n', ...
    param_bounds.W_L2_goal_gain.min, param_bounds.W_L2_goal_gain.max);
fprintf('  W_L1_pos_gain: [%.4f, %.3f]\n\n', ...
    param_bounds.W_L1_pos_gain.min, param_bounds.W_L1_pos_gain.max);

fprintf('OBJECTIVE FUNCTION WEIGHTS:\n');
fprintf('  Reaching distance improvement: %.1f\n', objective_weights.reaching_distance);
fprintf('  Position RMSE:                 %.1f\n\n', objective_weights.position_rmse);

% ====================================================================
% INITIALIZE PARTICLE SWARM
% ====================================================================

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('Initializing particle swarm with SPREAD-OUT initialization...\n\n');

% Initialize particle positions (SPREAD OUT across entire parameter space)
particles = struct();
for p = 1:num_particles
    % Use Latin hypercube or stratified sampling for better spread
    % Divide parameter space into cells for each particle
    
    % LEARNING RATES
    % eta_rep: spread particles across log scale
    log_eta_rep_min = param_bounds.eta_rep.log_min;
    log_eta_rep_max = param_bounds.eta_rep.log_max;
    log_eta_rep_cell = log_eta_rep_min + (p-1) * (log_eta_rep_max - log_eta_rep_min) / num_particles;
    log_eta_rep = log_eta_rep_cell + rand() * (log_eta_rep_max - log_eta_rep_min) / num_particles;
    particles(p).eta_rep = 10^log_eta_rep;
    
    % eta_W: spread particles across log scale
    log_eta_W_min = param_bounds.eta_W.log_min;
    log_eta_W_max = param_bounds.eta_W.log_max;
    log_eta_W_cell = log_eta_W_min + (p-1) * (log_eta_W_max - log_eta_W_min) / num_particles;
    log_eta_W = log_eta_W_cell + rand() * (log_eta_W_max - log_eta_W_min) / num_particles;
    particles(p).eta_W = 10^log_eta_W;
    
    % momentum: spread particles linearly
    mom_min = param_bounds.momentum.min;
    mom_max = param_bounds.momentum.max;
    mom_cell = mom_min + (p-1) * (mom_max - mom_min) / num_particles;
    particles(p).momentum = mom_cell + rand() * (mom_max - mom_min) / num_particles;
    
    % WEIGHT DECAY
    dec_L2_min = param_bounds.decay_L2_goal.min;
    dec_L2_max = param_bounds.decay_L2_goal.max;
    dec_L2_cell = dec_L2_min + (p-1) * (dec_L2_max - dec_L2_min) / num_particles;
    particles(p).decay_L2_goal = dec_L2_cell + rand() * (dec_L2_max - dec_L2_min) / num_particles;
    
    dec_L1_min = param_bounds.decay_L1_motor.min;
    dec_L1_max = param_bounds.decay_L1_motor.max;
    dec_L1_cell = dec_L1_min + (p-1) * (dec_L1_max - dec_L1_min) / num_particles;
    particles(p).decay_L1_motor = dec_L1_cell + rand() * (dec_L1_max - dec_L1_min) / num_particles;
    
    % MOTOR DYNAMICS
    mg_min = param_bounds.motor_gain.min;
    mg_max = param_bounds.motor_gain.max;
    mg_cell = mg_min + (p-1) * (mg_max - mg_min) / num_particles;
    particles(p).motor_gain = mg_cell + rand() * (mg_max - mg_min) / num_particles;
    
    damp_min = param_bounds.damping.min;
    damp_max = param_bounds.damping.max;
    damp_cell = damp_min + (p-1) * (damp_max - damp_min) / num_particles;
    particles(p).damping = damp_cell + rand() * (damp_max - damp_min) / num_particles;
    
    rss_min = param_bounds.reaching_speed_scale.min;
    rss_max = param_bounds.reaching_speed_scale.max;
    rss_cell = rss_min + (p-1) * (rss_max - rss_min) / num_particles;
    particles(p).reaching_speed_scale = rss_cell + rand() * (rss_max - rss_min) / num_particles;
    
    % WEIGHT INITIALIZATION
    wl2_min = param_bounds.W_L2_goal_gain.min;
    wl2_max = param_bounds.W_L2_goal_gain.max;
    wl2_cell = wl2_min + (p-1) * (wl2_max - wl2_min) / num_particles;
    particles(p).W_L2_goal_gain = wl2_cell + rand() * (wl2_max - wl2_min) / num_particles;
    
    wl1_min = param_bounds.W_L1_pos_gain.min;
    wl1_max = param_bounds.W_L1_pos_gain.max;
    wl1_cell = wl1_min + (p-1) * (wl1_max - wl1_min) / num_particles;
    particles(p).W_L1_pos_gain = wl1_cell + rand() * (wl1_max - wl1_min) / num_particles;
    
    % Initialize velocity with larger range for exploration
    particles(p).vel_eta_rep = -2 * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min) + ...
        rand() * 4 * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min);
    particles(p).vel_eta_W = -2 * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min) + ...
        rand() * 4 * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min);
    particles(p).vel_momentum = -2 * (param_bounds.momentum.max - param_bounds.momentum.min) + ...
        rand() * 4 * (param_bounds.momentum.max - param_bounds.momentum.min);
    particles(p).vel_decay_L2_goal = -2 * (param_bounds.decay_L2_goal.max - param_bounds.decay_L2_goal.min) + ...
        rand() * 4 * (param_bounds.decay_L2_goal.max - param_bounds.decay_L2_goal.min);
    particles(p).vel_decay_L1_motor = -2 * (param_bounds.decay_L1_motor.max - param_bounds.decay_L1_motor.min) + ...
        rand() * 4 * (param_bounds.decay_L1_motor.max - param_bounds.decay_L1_motor.min);
    particles(p).vel_motor_gain = -2 * (param_bounds.motor_gain.max - param_bounds.motor_gain.min) + ...
        rand() * 4 * (param_bounds.motor_gain.max - param_bounds.motor_gain.min);
    particles(p).vel_damping = -2 * (param_bounds.damping.max - param_bounds.damping.min) + ...
        rand() * 4 * (param_bounds.damping.max - param_bounds.damping.min);
    particles(p).vel_reaching_speed_scale = -2 * (param_bounds.reaching_speed_scale.max - param_bounds.reaching_speed_scale.min) + ...
        rand() * 4 * (param_bounds.reaching_speed_scale.max - param_bounds.reaching_speed_scale.min);
    particles(p).vel_W_L2_goal_gain = -2 * (param_bounds.W_L2_goal_gain.max - param_bounds.W_L2_goal_gain.min) + ...
        rand() * 4 * (param_bounds.W_L2_goal_gain.max - param_bounds.W_L2_goal_gain.min);
    particles(p).vel_W_L1_pos_gain = -2 * (param_bounds.W_L1_pos_gain.max - param_bounds.W_L1_pos_gain.min) + ...
        rand() * 4 * (param_bounds.W_L1_pos_gain.max - param_bounds.W_L1_pos_gain.min);
    
    % Initialize particle's best position and score
    particles(p).best_eta_rep = particles(p).eta_rep;
    particles(p).best_eta_W = particles(p).eta_W;
    particles(p).best_momentum = particles(p).momentum;
    particles(p).best_decay_L2_goal = particles(p).decay_L2_goal;
    particles(p).best_decay_L1_motor = particles(p).decay_L1_motor;
    particles(p).best_motor_gain = particles(p).motor_gain;
    particles(p).best_damping = particles(p).damping;
    particles(p).best_reaching_speed_scale = particles(p).reaching_speed_scale;
    particles(p).best_W_L2_goal_gain = particles(p).W_L2_goal_gain;
    particles(p).best_W_L1_pos_gain = particles(p).W_L1_pos_gain;
    particles(p).best_score = inf;
end

% Global best tracking
global_best_score = inf;
global_best_params = struct();

fprintf('Swarm initialized with %d SPREAD-OUT particles.\n', num_particles);
fprintf('Particles distributed across entire parameter space (stratified sampling).\n\n');

% ====================================================================
% PSO MAIN LOOP
% ====================================================================

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('Starting PSO optimization...\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Store history for analysis
iteration_history = struct();
iteration_history.best_scores = [];
iteration_history.avg_scores = [];
iteration_history.best_params_history = [];

% Quick debug mode: when true, PSO runs use short trials (fast) so you can validate
% the optimization loop without waiting for full-length simulations. Set to false
% to run the full-duration model during PSO (much slower).
fast_debug_mode = false;    % <-- set to false for full/production PSO runs
debug_T_per_trial = 2.5;   % seconds per trial for fast debug mode (2.5s -> ~250 steps at dt=0.01)
debug_dt = 0.02;           % larger dt for faster debug runs

for iteration = 1:num_iterations
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║ PSO Iteration %d/%d (Evaluating %d particles)              ║\n', ...
        iteration, num_iterations, num_particles);
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    iteration_scores = [];
    
    % Evaluate each particle in the swarm (parallelized)
    % We evaluate particles in parallel using parfor. Each worker returns
    % a small result struct and a numeric score. We avoid writing files in
    % the worker (dh_params.save_results=false) to prevent I/O contention.

    % Prepare output containers
    scores = inf(1, num_particles);
    loaded_data_cell = cell(1, num_particles);
    personal_best_scores = [particles.best_score];
    personal_best_params_cell = cell(1, num_particles);

    % Start parallel pool if needed
    if isempty(gcp('nocreate'))
        try
            parpool('local');
        catch
            % If parpool can't be started, fallback to serial loop below
        end
    end

    % Start a safe parpool sized automatically from system specs
    % Estimate per_worker_MB based on available memory and desired fraction
    m = memory;
    availMB = m.MemAvailableAllArrays/1024^2;
    % Use 1/num_particles of available memory, but cap at 2000MB per worker
    est_per_worker_MB = min(2000, max(500, floor(availMB / num_particles)));
    suggested_workers = start_safe_parpool(est_per_worker_MB);

    % Use parfor to evaluate particles in parallel. Each iteration must be
    % independent and write to separate cells/arrays.
    parfor p = 1:num_particles
        try
            % Minimal per-particle reporting (not printed in parfor workers)

            % Map particle to dual-hierarchy params
            dh_params = struct();
            dh_params.eta_rep = particles(p).eta_rep;
            dh_params.eta_W = particles(p).eta_W;
            dh_params.momentum = particles(p).momentum;
            dh_params.decay_plan = particles(p).decay_L2_goal;
            dh_params.decay_motor = particles(p).decay_L1_motor;
            dh_params.motor_gain = particles(p).motor_gain;
            dh_params.damping = particles(p).damping;
            dh_params.reaching_speed_scale = particles(p).reaching_speed_scale;
            dh_params.W_plan_gain = particles(p).W_L2_goal_gain;
            dh_params.W_motor_gain = particles(p).W_L1_pos_gain;
            % (fast debug overrides are disabled inside parfor for compiler safety)
            dh_params.save_results = false;

            % Run model (worker returns results struct)
            res = hierarchical_motion_inference_dual_hierarchy(dh_params, false);

            % Compute score from returned struct (same as serial path)
            if isfield(res, 'interception_error_all') && isfield(res, 'phases_indices')
                interception_error_all_local = res.interception_error_all;
                phases_indices_local = res.phases_indices;
                % use final interception distance per trial
                n_trials_model = numel(phases_indices_local);
                trial_final = zeros(1, n_trials_model);
                for tt = 1:n_trials_model
                    idx_range = phases_indices_local{tt};
                    trial_final(tt) = interception_error_all_local(idx_range(end));
                end
                avg_final = mean(trial_final);
                scores(p) = avg_final;
                loaded_data_cell{p} = res;
            else
                scores(p) = inf;
                loaded_data_cell{p} = struct();
            end

            % compute new personal best info (returned to main thread)
            personal_best_params_cell{p} = struct('eta_rep', particles(p).eta_rep, 'eta_W', particles(p).eta_W, ...
                'momentum', particles(p).momentum, 'decay_L2_goal', particles(p).decay_L2_goal, 'decay_L1_motor', particles(p).decay_L1_motor, ...
                'motor_gain', particles(p).motor_gain, 'damping', particles(p).damping, 'reaching_speed_scale', particles(p).reaching_speed_scale, ...
                'W_L2_goal_gain', particles(p).W_L2_goal_gain, 'W_L1_pos_gain', particles(p).W_L1_pos_gain);

        catch MEpar
            scores(p) = inf;
            loaded_data_cell{p} = struct();
            personal_best_params_cell{p} = struct();
        end
    end

    % Merge results from parallel workers back into particles and statistics
    for p = 1:num_particles
        current_score = scores(p);
        iteration_scores = [iteration_scores, current_score];
        particles(p).score = current_score;

        % Update personal best if improved
        if current_score < particles(p).best_score
            particles(p).best_score = current_score;
            pb = personal_best_params_cell{p};
            if ~isempty(fieldnames(pb))
                particles(p).best_eta_rep = pb.eta_rep;
                particles(p).best_eta_W = pb.eta_W;
                particles(p).best_momentum = pb.momentum;
                particles(p).best_decay_L2_goal = pb.decay_L2_goal;
                particles(p).best_decay_L1_motor = pb.decay_L1_motor;
                particles(p).best_motor_gain = pb.motor_gain;
                particles(p).best_damping = pb.damping;
                particles(p).best_reaching_speed_scale = pb.reaching_speed_scale;
                particles(p).best_W_L2_goal_gain = pb.W_L2_goal_gain;
                particles(p).best_W_L1_pos_gain = pb.W_L1_pos_gain;
            end
            fprintf('    ★ Particle %d new personal best: %.6f\n', p, current_score);
        end

        % Update global best and save best simulation snapshot if improved
        if current_score < global_best_score
            global_best_score = current_score;
            global_best_params.eta_rep = particles(p).eta_rep;
            global_best_params.eta_W = particles(p).eta_W;
            global_best_params.momentum = particles(p).momentum;
            global_best_params.decay_L2_goal = particles(p).decay_L2_goal;
            global_best_params.decay_L1_motor = particles(p).decay_L1_motor;
            global_best_params.motor_gain = particles(p).motor_gain;
            global_best_params.damping = particles(p).damping;
            global_best_params.reaching_speed_scale = particles(p).reaching_speed_scale;
            global_best_params.W_L2_goal_gain = particles(p).W_L2_goal_gain;
            global_best_params.W_L1_pos_gain = particles(p).W_L1_pos_gain;
            fprintf('    ✯ NEW GLOBAL BEST (particle %d): %.6f ✯\n', p, global_best_score);
            try
                out_dir = './figures'; if ~exist(out_dir, 'dir'), mkdir(out_dir); end
                best_fname = fullfile(out_dir, '3D_dual_hierarchy_results_best.mat');
                best_data = loaded_data_cell{p};
                save(best_fname, 'best_data', '-v7.3');
                fprintf('    ✓ Best results saved: %s\n', best_fname);
            catch MESAVE
                fprintf('    Warning: failed to save best results: %s\n', MESAVE.message);
            end
        end
        fprintf('  Particle %d → Score: %.6f\n', p, current_score);
    end
    
    % Record iteration statistics
    iteration_history.best_scores = [iteration_history.best_scores, global_best_score];
    iteration_history.avg_scores = [iteration_history.avg_scores, mean(iteration_scores)];
    iteration_history.best_params_history = [iteration_history.best_params_history; global_best_params];
    
    fprintf('\n  Iteration %d Summary:\n', iteration);
    fprintf('    Global best score:  %.6f\n', global_best_score);
    fprintf('    Iteration avg:      %.6f\n', mean(iteration_scores));
    fprintf('    Best particle:      %.6f\n', min(iteration_scores));
    
    % --- UPDATE PARTICLE VELOCITIES AND POSITIONS ---
    fprintf('\n  Updating particle positions and velocities with stochastic noise...\n\n');
    
    for p = 1:num_particles
        % Velocity update equation (standard PSO with stochastic noise):
        % v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x) + noise
        
        % eta_rep (log scale)
        r1 = rand();
        r2 = rand();
        eta_rep_range = param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min;
        noise_eta_rep = noise_scale * eta_rep_range * randn();
        particles(p).vel_eta_rep = w * particles(p).vel_eta_rep + ...
            c1 * r1 * (log10(particles(p).best_eta_rep) - log10(particles(p).eta_rep)) + ...
            c2 * r2 * (log10(global_best_params.eta_rep) - log10(particles(p).eta_rep)) + ...
            noise_eta_rep;
        
        % eta_W (log scale)
        r1 = rand();
        r2 = rand();
        eta_W_range = param_bounds.eta_W.log_max - param_bounds.eta_W.log_min;
        noise_eta_W = noise_scale * eta_W_range * randn();
        particles(p).vel_eta_W = w * particles(p).vel_eta_W + ...
            c1 * r1 * (log10(particles(p).best_eta_W) - log10(particles(p).eta_W)) + ...
            c2 * r2 * (log10(global_best_params.eta_W) - log10(particles(p).eta_W)) + ...
            noise_eta_W;
        
        % momentum (linear scale)
        r1 = rand();
        r2 = rand();
        momentum_range = param_bounds.momentum.max - param_bounds.momentum.min;
        noise_momentum = noise_scale * momentum_range * randn();
        particles(p).vel_momentum = w * particles(p).vel_momentum + ...
            c1 * r1 * (particles(p).best_momentum - particles(p).momentum) + ...
            c2 * r2 * (global_best_params.momentum - particles(p).momentum) + ...
            noise_momentum;
        
        % decay_L2_goal (linear scale)
        r1 = rand();
        r2 = rand();
        decay_L2_range = param_bounds.decay_L2_goal.max - param_bounds.decay_L2_goal.min;
        noise_decay_L2 = noise_scale * decay_L2_range * randn();
        particles(p).vel_decay_L2_goal = w * particles(p).vel_decay_L2_goal + ...
            c1 * r1 * (particles(p).best_decay_L2_goal - particles(p).decay_L2_goal) + ...
            c2 * r2 * (global_best_params.decay_L2_goal - particles(p).decay_L2_goal) + ...
            noise_decay_L2;
        
        % decay_L1_motor (linear scale)
        r1 = rand();
        r2 = rand();
        decay_L1_range = param_bounds.decay_L1_motor.max - param_bounds.decay_L1_motor.min;
        noise_decay_L1 = noise_scale * decay_L1_range * randn();
        particles(p).vel_decay_L1_motor = w * particles(p).vel_decay_L1_motor + ...
            c1 * r1 * (particles(p).best_decay_L1_motor - particles(p).decay_L1_motor) + ...
            c2 * r2 * (global_best_params.decay_L1_motor - particles(p).decay_L1_motor) + ...
            noise_decay_L1;
        
        % motor_gain (linear scale)
        r1 = rand();
        r2 = rand();
        motor_gain_range = param_bounds.motor_gain.max - param_bounds.motor_gain.min;
        noise_motor_gain = noise_scale * motor_gain_range * randn();
        particles(p).vel_motor_gain = w * particles(p).vel_motor_gain + ...
            c1 * r1 * (particles(p).best_motor_gain - particles(p).motor_gain) + ...
            c2 * r2 * (global_best_params.motor_gain - particles(p).motor_gain) + ...
            noise_motor_gain;
        
        % damping (linear scale)
        r1 = rand();
        r2 = rand();
        damping_range = param_bounds.damping.max - param_bounds.damping.min;
        noise_damping = noise_scale * damping_range * randn();
        particles(p).vel_damping = w * particles(p).vel_damping + ...
            c1 * r1 * (particles(p).best_damping - particles(p).damping) + ...
            c2 * r2 * (global_best_params.damping - particles(p).damping) + ...
            noise_damping;
        
        % reaching_speed_scale (linear scale)
        r1 = rand();
        r2 = rand();
        rss_range = param_bounds.reaching_speed_scale.max - param_bounds.reaching_speed_scale.min;
        noise_rss = noise_scale * rss_range * randn();
        particles(p).vel_reaching_speed_scale = w * particles(p).vel_reaching_speed_scale + ...
            c1 * r1 * (particles(p).best_reaching_speed_scale - particles(p).reaching_speed_scale) + ...
            c2 * r2 * (global_best_params.reaching_speed_scale - particles(p).reaching_speed_scale) + ...
            noise_rss;
        
        % W_L2_goal_gain (linear scale)
        r1 = rand();
        r2 = rand();
        wl2_range = param_bounds.W_L2_goal_gain.max - param_bounds.W_L2_goal_gain.min;
        noise_wl2 = noise_scale * wl2_range * randn();
        particles(p).vel_W_L2_goal_gain = w * particles(p).vel_W_L2_goal_gain + ...
            c1 * r1 * (particles(p).best_W_L2_goal_gain - particles(p).W_L2_goal_gain) + ...
            c2 * r2 * (global_best_params.W_L2_goal_gain - particles(p).W_L2_goal_gain) + ...
            noise_wl2;
        
        % W_L1_pos_gain (linear scale)
        r1 = rand();
        r2 = rand();
        wl1_range = param_bounds.W_L1_pos_gain.max - param_bounds.W_L1_pos_gain.min;
        noise_wl1 = noise_scale * wl1_range * randn();
        particles(p).vel_W_L1_pos_gain = w * particles(p).vel_W_L1_pos_gain + ...
            c1 * r1 * (particles(p).best_W_L1_pos_gain - particles(p).W_L1_pos_gain) + ...
            c2 * r2 * (global_best_params.W_L1_pos_gain - particles(p).W_L1_pos_gain) + ...
            noise_wl1;
        
        % Position updates
        % For log-scale parameters, position is updated on log scale then converted
        log_eta_rep_new = log10(particles(p).eta_rep) + particles(p).vel_eta_rep;
        particles(p).eta_rep = 10^log_eta_rep_new;
        
        log_eta_W_new = log10(particles(p).eta_W) + particles(p).vel_eta_W;
        particles(p).eta_W = 10^log_eta_W_new;
        
        particles(p).momentum = particles(p).momentum + particles(p).vel_momentum;
        particles(p).decay_L2_goal = particles(p).decay_L2_goal + particles(p).vel_decay_L2_goal;
        particles(p).decay_L1_motor = particles(p).decay_L1_motor + particles(p).vel_decay_L1_motor;
        particles(p).motor_gain = particles(p).motor_gain + particles(p).vel_motor_gain;
        particles(p).damping = particles(p).damping + particles(p).vel_damping;
        particles(p).reaching_speed_scale = particles(p).reaching_speed_scale + particles(p).vel_reaching_speed_scale;
        particles(p).W_L2_goal_gain = particles(p).W_L2_goal_gain + particles(p).vel_W_L2_goal_gain;
        particles(p).W_L1_pos_gain = particles(p).W_L1_pos_gain + particles(p).vel_W_L1_pos_gain;
        
        % Enforce bounds on all parameters
        particles(p).eta_rep = max(10^param_bounds.eta_rep.log_min, ...
                                   min(10^param_bounds.eta_rep.log_max, particles(p).eta_rep));
        particles(p).eta_W = max(10^param_bounds.eta_W.log_min, ...
                                min(10^param_bounds.eta_W.log_max, particles(p).eta_W));
        particles(p).momentum = max(param_bounds.momentum.min, ...
                                   min(param_bounds.momentum.max, particles(p).momentum));
        particles(p).decay_L2_goal = max(param_bounds.decay_L2_goal.min, ...
                                        min(param_bounds.decay_L2_goal.max, particles(p).decay_L2_goal));
        particles(p).decay_L1_motor = max(param_bounds.decay_L1_motor.min, ...
                                         min(param_bounds.decay_L1_motor.max, particles(p).decay_L1_motor));
        particles(p).motor_gain = max(param_bounds.motor_gain.min, ...
                                     min(param_bounds.motor_gain.max, particles(p).motor_gain));
        particles(p).damping = max(param_bounds.damping.min, ...
                                  min(param_bounds.damping.max, particles(p).damping));
        particles(p).reaching_speed_scale = max(param_bounds.reaching_speed_scale.min, ...
                                              min(param_bounds.reaching_speed_scale.max, particles(p).reaching_speed_scale));
        particles(p).W_L2_goal_gain = max(param_bounds.W_L2_goal_gain.min, ...
                                         min(param_bounds.W_L2_goal_gain.max, particles(p).W_L2_goal_gain));
        particles(p).W_L1_pos_gain = max(param_bounds.W_L1_pos_gain.min, ...
                                        min(param_bounds.W_L1_pos_gain.max, particles(p).W_L1_pos_gain));
    end
end

% ====================================================================
% PSO COMPLETE - SAVE RESULTS
% ====================================================================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PSO OPTIMIZATION COMPLETE\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Best Parameters Found (11-DIMENSIONAL OPTIMIZATION):\n');
fprintf('  Score (weighted objective):  %.6f\n', global_best_score);
fprintf('\n  LEARNING RATES:\n');
fprintf('    eta_rep:                   %.6f\n', global_best_params.eta_rep);
fprintf('    eta_W:                     %.6f\n', global_best_params.eta_W);
fprintf('    momentum:                  %.6f\n', global_best_params.momentum);
fprintf('\n  WEIGHT DECAY (trial transfer):\n');
fprintf('    decay_L2_goal:             %.6f\n', global_best_params.decay_L2_goal);
fprintf('    decay_L1_motor:            %.6f\n', global_best_params.decay_L1_motor);
fprintf('\n  MOTOR DYNAMICS (trajectory quality):\n');
fprintf('    motor_gain:                %.6f\n', global_best_params.motor_gain);
fprintf('    damping:                   %.6f\n', global_best_params.damping);
fprintf('    reaching_speed_scale:      %.6f\n', global_best_params.reaching_speed_scale);
fprintf('\n  WEIGHT INITIALIZATION (convergence speed):\n');
fprintf('    W_L2_goal_gain:            %.6f\n', global_best_params.W_L2_goal_gain);
fprintf('    W_L1_pos_gain:             %.6f\n\n', global_best_params.W_L1_pos_gain);

% Create results struct for saving
results = struct();
results.best_score = global_best_score;
results.best_params = global_best_params;
results.iteration_history = iteration_history;
results.particles = particles;
results.optimization_method = 'Particle Swarm Optimization (PSO)';
results.num_particles = num_particles;
results.num_iterations = num_iterations;
results.total_evaluations = total_evals;
results.pso_inertia_weight = w;
results.pso_cognitive = c1;
results.pso_social = c2;

% ====================================================================
% BUILD TOP-20 LEADERBOARD (from particle personal bests)
% ====================================================================
try
    n_leader = min(20, num_particles);
    % Collect personal bests
    all_scores = zeros(num_particles,1);
    for pp = 1:num_particles
        all_scores(pp) = particles(pp).best_score;
    end
    [sorted_scores, idx] = sort(all_scores, 'ascend');
    % Determine how many valid (finite) bests we have
    valid_mask = isfinite(sorted_scores);
    top_n = min(n_leader, sum(valid_mask));

    leader_list = struct('score', cell(top_n,1), 'params', cell(top_n,1));
    for k = 1:top_n
        ip = idx(k);
        ps = struct();
        ps.eta_rep = particles(ip).best_eta_rep;
        ps.eta_W = particles(ip).best_eta_W;
        ps.momentum = particles(ip).best_momentum;
        ps.decay_L2_goal = particles(ip).best_decay_L2_goal;
        ps.decay_L1_motor = particles(ip).best_decay_L1_motor;
        ps.motor_gain = particles(ip).best_motor_gain;
        ps.damping = particles(ip).best_damping;
        ps.reaching_speed_scale = particles(ip).best_reaching_speed_scale;
        ps.W_L2_goal_gain = particles(ip).best_W_L2_goal_gain;
        ps.W_L1_pos_gain = particles(ip).best_W_L1_pos_gain;

        leader_list(k).score = sorted_scores(k);
        leader_list(k).params = ps;
    end

    % Attach to results
    results.top20 = leader_list;

    % Save leaderboard to figures dir
    out_dir = './figures';
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    top_fname = fullfile(out_dir, 'pso_top20_best_params.mat');
    save(top_fname, 'leader_list');
    fprintf('✓ Top %d PSO parameter sets saved: %s\n', top_n, top_fname);
catch ME
    fprintf('Warning: failed to build/save top-20 leaderboard: %s\n', ME.message);
end

% Save results with timestamp
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
results_filename = sprintf('optimization_results_3D_PSO_%s.mat', timestamp);
save(results_filename, 'results');
fprintf('Saving optimization results to %s\n\n', results_filename);

% ====================================================================
% VISUALIZE PSO CONVERGENCE
% ====================================================================

figure('Name', 'PSO Convergence', 'NumberTitle', 'off', 'Visible', 'off');

% Plot 1: Global best score over iterations
subplot(2, 2, 1);
plot(1:num_iterations, iteration_history.best_scores, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('Global Best Score');
title('PSO Convergence: Best Score');
grid on;
set(gca, 'YScale', 'log');

% Plot 2: Average score over iterations
subplot(2, 2, 2);
plot(1:num_iterations, iteration_history.avg_scores, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:num_iterations, iteration_history.best_scores, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Iteration');
ylabel('Score');
title('PSO Convergence: Best vs Average Score');
legend('Average Score', 'Best Score');
grid on;
set(gca, 'YScale', 'log');

% Plot 3: eta_rep convergence
subplot(2, 2, 3);
eta_rep_history = [iteration_history.best_params_history.eta_rep];
plot(1:num_iterations, eta_rep_history, 'g-d', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('eta_rep');
title('Parameter Convergence: eta_rep');
grid on;
set(gca, 'YScale', 'log');

% Plot 4: momentum convergence
subplot(2, 2, 4);
momentum_history = [iteration_history.best_params_history.momentum];
plot(1:num_iterations, momentum_history, 'm-^', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Iteration');
ylabel('momentum');
title('Parameter Convergence: momentum');
grid on;

sgtitle('Particle Swarm Optimization - Convergence Analysis');
saveas(gcf, 'optimization_results_PSO_convergence.png');
fprintf('Convergence visualization saved: optimization_results_PSO_convergence.png\n\n');

fprintf('You can now use these optimal parameters in hierarchical_motion_inference_3D_EXACT:\n');
fprintf('  eta_rep = %.6f;\n', global_best_params.eta_rep);
fprintf('  eta_W = %.6f;\n', global_best_params.eta_W);
fprintf('  momentum = %.6f;\n\n', global_best_params.momentum);

fprintf('Or call with struct:\n');
fprintf('  params = struct(...\n');
fprintf('      ''eta_rep'', %.6f, ...\n', global_best_params.eta_rep);
fprintf('      ''eta_W'', %.6f, ...\n', global_best_params.eta_W);
fprintf('      ''momentum'', %.6f, ...\n', global_best_params.momentum);
fprintf('      ''weight_decay'', 0.98);\n');
fprintf('  hierarchical_motion_inference_3D_EXACT(params);\n\n');

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PSO optimization complete!\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
