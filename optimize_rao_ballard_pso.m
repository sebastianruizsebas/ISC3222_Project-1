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
num_particles = 25;  % Each particle = one parameter set

% Number of PSO iterations (generations)
num_iterations = 35;  % Each iteration = all particles tested

% Total evaluations will be: num_particles * num_iterations = 875 trials
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

% Parameter bounds - now includes 15 parameters (extended with task-conditional learning)
param_bounds = struct();

% LEARNING RATES (log scale)
param_bounds.eta_rep.log_min = -4;      % 10^-4 = 0.0001
param_bounds.eta_rep.log_max = -1;      % 10^-1 = 0.1
param_bounds.eta_W.log_min = -6;        % 10^-6 = 0.000001
param_bounds.eta_W.log_max = -1;        % 10^-1 = 0.1
param_bounds.momentum.min = 0.70;       % Linear scale
param_bounds.momentum.max = 0.98;

% WEIGHT DECAY (linear scale, affects learning across trials)
param_bounds.weight_decay.min = 0.60;
param_bounds.weight_decay.max = 0.999;

% TASK-CONDITIONAL DECAY RATES (NEW - Nov 1, 2025)
% Motor region: preserve stable dynamics across tasks (high retention)
param_bounds.decay_motor.min = 0.90;    % Motor: 90-99% weight retention
param_bounds.decay_motor.max = 0.99;
% Planning region: forget old task-specific targets (lower retention)
param_bounds.decay_plan.min = 0.50;     % Planning: 50-80% weight retention
param_bounds.decay_plan.max = 0.80;

% MOTOR DYNAMICS (linear scale, affects trajectory quality)
param_bounds.motor_gain.min = 0.1;      % Initial motor command strength
param_bounds.motor_gain.max = 1.0;
param_bounds.damping.min = 0.70;        % Velocity dampening
param_bounds.damping.max = 0.99;
param_bounds.reaching_speed_scale.min = 0.1;  % Scale for initial reaching speed
param_bounds.reaching_speed_scale.max = 1.0;

% WEIGHT INITIALIZATION GAINS (linear scale, affects convergence)
param_bounds.W_motor_gain.min = 0.01;   % Motor weight init gain
param_bounds.W_motor_gain.max = 1.0;
param_bounds.W_plan_gain.min = 0.01;    % Planning weight init gain
param_bounds.W_plan_gain.max = 1.0;

% TASK-CONDITIONAL LEARNING PARAMETERS (NEW - Nov 1, 2025)
% Interference penalty: encourages task-specific weight specialization
param_bounds.interference_penalty_weight.min = 0.0;   % No penalty
param_bounds.interference_penalty_weight.max = 0.1;   % Max cross-task error penalty

% Objective function weights
% For 3D reaching, primary metric is reaching distance improvement
objective_weights = struct('reaching_distance', 1.0, 'position_rmse', 0.5);

fprintf('PARAMETER SEARCH SPACE (15-DIMENSIONAL - TASK-CONDITIONAL LEARNING):\n');
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

fprintf('WEIGHT DECAY:\n');
fprintf('  weight_decay (global): [%.3f, %.3f]\n', ...
    param_bounds.weight_decay.min, param_bounds.weight_decay.max);

fprintf('TASK-CONDITIONAL DECAY RATES (NEW - Nov 1, 2025):\n');
fprintf('  decay_motor:  [%.2f, %.2f] (motor: preserve across tasks)\n', ...
    param_bounds.decay_motor.min, param_bounds.decay_motor.max);
fprintf('  decay_plan:   [%.2f, %.2f] (planning: forget old targets)\n', ...
    param_bounds.decay_plan.min, param_bounds.decay_plan.max);

fprintf('MOTOR DYNAMICS (trajectory quality):\n');
fprintf('  motor_gain: [%.2f, %.2f]\n', ...
    param_bounds.motor_gain.min, param_bounds.motor_gain.max);
fprintf('  damping: [%.2f, %.2f]\n', ...
    param_bounds.damping.min, param_bounds.damping.max);
fprintf('  reaching_speed_scale: [%.2f, %.2f]\n', ...
    param_bounds.reaching_speed_scale.min, param_bounds.reaching_speed_scale.max);

fprintf('WEIGHT INITIALIZATION GAINS (convergence speed) (NEW format):\n');
fprintf('  W_motor_gain: [%.3f, %.2f]\n', ...
    param_bounds.W_motor_gain.min, param_bounds.W_motor_gain.max);
fprintf('  W_plan_gain:  [%.3f, %.2f]\n', ...
    param_bounds.W_plan_gain.min, param_bounds.W_plan_gain.max);

fprintf('TASK-CONDITIONAL LEARNING - INTERFERENCE PENALTY (NEW - Nov 1, 2025):\n');
fprintf('  interference_penalty_weight: [%.4f, %.4f] (cross-task error penalty)\n\n', ...
    param_bounds.interference_penalty_weight.min, param_bounds.interference_penalty_weight.max);

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
    
    % WEIGHT DECAY (global)
    wd_min = param_bounds.weight_decay.min;
    wd_max = param_bounds.weight_decay.max;
    wd_cell = wd_min + (p-1) * (wd_max - wd_min) / num_particles;
    particles(p).weight_decay = wd_cell + rand() * (wd_max - wd_min) / num_particles;
    
    % TASK-CONDITIONAL DECAY RATES (NEW - Nov 1, 2025)
    % decay_motor: preserve stable motor dynamics across tasks
    dm_min = param_bounds.decay_motor.min;
    dm_max = param_bounds.decay_motor.max;
    dm_cell = dm_min + (p-1) * (dm_max - dm_min) / num_particles;
    particles(p).decay_motor = dm_cell + rand() * (dm_max - dm_min) / num_particles;
    
    % decay_plan: allow planning region to forget old task-specific targets
    dp_min = param_bounds.decay_plan.min;
    dp_max = param_bounds.decay_plan.max;
    dp_cell = dp_min + (p-1) * (dp_max - dp_min) / num_particles;
    particles(p).decay_plan = dp_cell + rand() * (dp_max - dp_min) / num_particles;
    
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
    
    % WEIGHT INITIALIZATION GAINS (NEW format - Nov 1, 2025)
    % Motor weight initialization
    wm_min = param_bounds.W_motor_gain.min;
    wm_max = param_bounds.W_motor_gain.max;
    wm_cell = wm_min + (p-1) * (wm_max - wm_min) / num_particles;
    particles(p).W_motor_gain = wm_cell + rand() * (wm_max - wm_min) / num_particles;
    
    % Planning weight initialization
    wp_min = param_bounds.W_plan_gain.min;
    wp_max = param_bounds.W_plan_gain.max;
    wp_cell = wp_min + (p-1) * (wp_max - wp_min) / num_particles;
    particles(p).W_plan_gain = wp_cell + rand() * (wp_max - wp_min) / num_particles;
    
    % TASK-CONDITIONAL LEARNING - INTERFERENCE PENALTY (NEW - Nov 1, 2025)
    ipw_min = param_bounds.interference_penalty_weight.min;
    ipw_max = param_bounds.interference_penalty_weight.max;
    ipw_cell = ipw_min + (p-1) * (ipw_max - ipw_min) / num_particles;
    particles(p).interference_penalty_weight = ipw_cell + rand() * (ipw_max - ipw_min) / num_particles;
    
    % Initialize velocity with larger range for exploration
    particles(p).vel_eta_rep = -2 * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min) + ...
        rand() * 4 * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min);
    particles(p).vel_eta_W = -2 * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min) + ...
        rand() * 4 * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min);
    particles(p).vel_momentum = -2 * (param_bounds.momentum.max - param_bounds.momentum.min) + ...
        rand() * 4 * (param_bounds.momentum.max - param_bounds.momentum.min);
    particles(p).vel_weight_decay = -2 * (param_bounds.weight_decay.max - param_bounds.weight_decay.min) + ...
        rand() * 4 * (param_bounds.weight_decay.max - param_bounds.weight_decay.min);
    particles(p).vel_decay_motor = -2 * (param_bounds.decay_motor.max - param_bounds.decay_motor.min) + ...
        rand() * 4 * (param_bounds.decay_motor.max - param_bounds.decay_motor.min);
    particles(p).vel_decay_plan = -2 * (param_bounds.decay_plan.max - param_bounds.decay_plan.min) + ...
        rand() * 4 * (param_bounds.decay_plan.max - param_bounds.decay_plan.min);
    particles(p).vel_motor_gain = -2 * (param_bounds.motor_gain.max - param_bounds.motor_gain.min) + ...
        rand() * 4 * (param_bounds.motor_gain.max - param_bounds.motor_gain.min);
    particles(p).vel_damping = -2 * (param_bounds.damping.max - param_bounds.damping.min) + ...
        rand() * 4 * (param_bounds.damping.max - param_bounds.damping.min);
    particles(p).vel_reaching_speed_scale = -2 * (param_bounds.reaching_speed_scale.max - param_bounds.reaching_speed_scale.min) + ...
        rand() * 4 * (param_bounds.reaching_speed_scale.max - param_bounds.reaching_speed_scale.min);
    particles(p).vel_W_motor_gain = -2 * (param_bounds.W_motor_gain.max - param_bounds.W_motor_gain.min) + ...
        rand() * 4 * (param_bounds.W_motor_gain.max - param_bounds.W_motor_gain.min);
    particles(p).vel_W_plan_gain = -2 * (param_bounds.W_plan_gain.max - param_bounds.W_plan_gain.min) + ...
        rand() * 4 * (param_bounds.W_plan_gain.max - param_bounds.W_plan_gain.min);
    particles(p).vel_interference_penalty_weight = -2 * (param_bounds.interference_penalty_weight.max - param_bounds.interference_penalty_weight.min) + ...
        rand() * 4 * (param_bounds.interference_penalty_weight.max - param_bounds.interference_penalty_weight.min);
    
    % Initialize particle's best position and score
    particles(p).best_eta_rep = particles(p).eta_rep;
    particles(p).best_eta_W = particles(p).eta_W;
    particles(p).best_momentum = particles(p).momentum;
    particles(p).best_weight_decay = particles(p).weight_decay;
    particles(p).best_decay_motor = particles(p).decay_motor;
    particles(p).best_decay_plan = particles(p).decay_plan;
    particles(p).best_motor_gain = particles(p).motor_gain;
    particles(p).best_damping = particles(p).damping;
    particles(p).best_reaching_speed_scale = particles(p).reaching_speed_scale;
    particles(p).best_W_motor_gain = particles(p).W_motor_gain;
    particles(p).best_W_plan_gain = particles(p).W_plan_gain;
    particles(p).best_interference_penalty_weight = particles(p).interference_penalty_weight;
    particles(p).best_score = inf;
end

% Global best tracking
    global_best_score = inf;
    global_best_params = struct('eta_rep', nan, 'eta_W', nan, 'momentum', nan, 'weight_decay', nan, ...
        'decay_motor', nan, 'decay_plan', nan, 'motor_gain', nan, 'damping', nan, 'reaching_speed_scale', nan, ...
        'W_motor_gain', nan, 'W_plan_gain', nan, 'interference_penalty_weight', nan);

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

    % On clusters, memory() may not be available. Use a fixed per_worker_MB or cluster-specific logic.
    est_per_worker_MB = 1500; % Default to 1500MB per worker for clusters
    suggested_workers = start_safe_parpool(est_per_worker_MB);

    % Use parfor to evaluate particles in parallel. Each iteration must be
    % independent and write to separate cells/arrays.
    parfor p = 1:num_particles
        try
            % Map particle to dual-hierarchy params
            dh_params = struct();
            dh_params.eta_rep = particles(p).eta_rep;
            dh_params.eta_W = particles(p).eta_W;
            dh_params.momentum = particles(p).momentum;
            dh_params.weight_decay = particles(p).weight_decay;
            dh_params.decay_motor = particles(p).decay_motor;
            dh_params.decay_plan = particles(p).decay_plan;
            dh_params.motor_gain = particles(p).motor_gain;
            dh_params.damping = particles(p).damping;
            dh_params.reaching_speed_scale = particles(p).reaching_speed_scale;
            dh_params.W_motor_gain = particles(p).W_motor_gain;
            dh_params.W_plan_gain = particles(p).W_plan_gain;
            dh_params.interference_penalty_weight = particles(p).interference_penalty_weight;
            dh_params.save_results = false;
            % Pass PSO context for printout
            dh_params.particle_num = p;
            dh_params.pso_iter = iteration;
            dh_params.pso_iter_total = num_iterations;

            % Run model (worker returns results struct)
            res = hierarchical_motion_inference_dual_hierarchy(dh_params, false);

            if isfield(res, 'interception_error_all') && isfield(res, 'phases_indices')
                interception_error_all_local = res.interception_error_all;
                phases_indices_local = res.phases_indices;
                n_trials_model = numel(phases_indices_local);
                trial_final = zeros(1, n_trials_model);
                for tt = 1:n_trials_model
                    idx_range = phases_indices_local{tt};
                    trial_final(tt) = interception_error_all_local(idx_range(end));
                end
                avg_final = mean(trial_final);
                
                % EARLY TERMINATION PENALTY (NEW - Nov 1, 2025)
                % If the trial was terminated early (e.g., due to Inf/NaN clipping),
                % penalize it heavily to avoid PSO converging on those parameters
                early_term_penalty = 0;
                if isfield(res, 'early_termination') && res.early_termination
                    % Compute completion ratio: how many steps vs. total attempted
                    total_steps = length(interception_error_all_local);
                    if total_steps > 0
                        % Penalize based on incompleteness: 0-100 penalty based on %missing
                        completion_ratio = (total_steps - 1) / max(1, length(interception_error_all_local));
                        early_term_penalty = 100.0 * (1.0 - max(0, min(1, completion_ratio)));
                    else
                        early_term_penalty = 100.0;  % Maximum penalty if almost no steps executed
                    end
                    
                    % Add severe penalty if quit due to Inf/NaN clipping (not just normal end)
                    if isfield(res, 'termination_reason') && contains(res.termination_reason, 'Excessive')
                        early_term_penalty = early_term_penalty + 500.0;  % Severe penalty
                    end
                end
                
                scores(p) = avg_final + early_term_penalty;
                loaded_data_cell{p} = res;
            else
                scores(p) = inf;
                loaded_data_cell{p} = struct();
            end

            personal_best_params_cell{p} = struct('eta_rep', particles(p).eta_rep, 'eta_W', particles(p).eta_W, ...
                'momentum', particles(p).momentum, 'weight_decay', particles(p).weight_decay, ...
                'decay_motor', particles(p).decay_motor, 'decay_plan', particles(p).decay_plan, ...
                'motor_gain', particles(p).motor_gain, 'damping', particles(p).damping, 'reaching_speed_scale', particles(p).reaching_speed_scale, ...
                'W_motor_gain', particles(p).W_motor_gain, 'W_plan_gain', particles(p).W_plan_gain, ...
                'interference_penalty_weight', particles(p).interference_penalty_weight);

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
                particles(p).best_weight_decay = pb.weight_decay;
                particles(p).best_decay_motor = pb.decay_motor;
                particles(p).best_decay_plan = pb.decay_plan;
                particles(p).best_motor_gain = pb.motor_gain;
                particles(p).best_damping = pb.damping;
                particles(p).best_reaching_speed_scale = pb.reaching_speed_scale;
                particles(p).best_W_motor_gain = pb.W_motor_gain;
                particles(p).best_W_plan_gain = pb.W_plan_gain;
                if isfield(pb, 'interference_penalty_weight')
                    particles(p).best_interference_penalty_weight = pb.interference_penalty_weight;
                end
            end
            fprintf('    ★ Particle %d new personal best: %.6f\n', p, current_score);
        end

        % Update global best and save best simulation snapshot if improved
        if current_score < global_best_score
            global_best_score = current_score;
            global_best_params.eta_rep = particles(p).eta_rep;
            global_best_params.eta_W = particles(p).eta_W;
            global_best_params.momentum = particles(p).momentum;
            global_best_params.weight_decay = particles(p).weight_decay;
            global_best_params.decay_motor = particles(p).decay_motor;
            global_best_params.decay_plan = particles(p).decay_plan;
            global_best_params.motor_gain = particles(p).motor_gain;
            global_best_params.damping = particles(p).damping;
            global_best_params.reaching_speed_scale = particles(p).reaching_speed_scale;
            global_best_params.W_motor_gain = particles(p).W_motor_gain;
            global_best_params.W_plan_gain = particles(p).W_plan_gain;
            global_best_params.interference_penalty_weight = particles(p).interference_penalty_weight;
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
        r1 = rand(); r2 = rand();
        eta_rep_range = param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min;
        noise_eta_rep = noise_scale * eta_rep_range * randn();
        particles(p).vel_eta_rep = w * particles(p).vel_eta_rep + ...
            c1 * r1 * (log10(particles(p).best_eta_rep) - log10(particles(p).eta_rep)) + ...
            c2 * r2 * (log10(global_best_params.eta_rep) - log10(particles(p).eta_rep)) + noise_eta_rep;
        
        % eta_W (log scale)
        r1 = rand(); r2 = rand();
        eta_W_range = param_bounds.eta_W.log_max - param_bounds.eta_W.log_min;
        noise_eta_W = noise_scale * eta_W_range * randn();
        particles(p).vel_eta_W = w * particles(p).vel_eta_W + ...
            c1 * r1 * (log10(particles(p).best_eta_W) - log10(particles(p).eta_W)) + ...
            c2 * r2 * (log10(global_best_params.eta_W) - log10(particles(p).eta_W)) + noise_eta_W;
        
        % momentum (linear scale)
        r1 = rand(); r2 = rand();
        momentum_range = param_bounds.momentum.max - param_bounds.momentum.min;
        noise_momentum = noise_scale * momentum_range * randn();
        particles(p).vel_momentum = w * particles(p).vel_momentum + ...
            c1 * r1 * (particles(p).best_momentum - particles(p).momentum) + ...
            c2 * r2 * (global_best_params.momentum - particles(p).momentum) + noise_momentum;
        
        % weight_decay (linear scale)
        r1 = rand(); r2 = rand();
        wd_range = param_bounds.weight_decay.max - param_bounds.weight_decay.min;
        noise_wd = noise_scale * wd_range * randn();
        particles(p).vel_weight_decay = w * particles(p).vel_weight_decay + ...
            c1 * r1 * (particles(p).best_weight_decay - particles(p).weight_decay) + ...
            c2 * r2 * (global_best_params.weight_decay - particles(p).weight_decay) + noise_wd;
        
        % decay_motor (linear scale) - NEW
        r1 = rand(); r2 = rand();
        decay_motor_range = param_bounds.decay_motor.max - param_bounds.decay_motor.min;
        noise_decay_motor = noise_scale * decay_motor_range * randn();
        particles(p).vel_decay_motor = w * particles(p).vel_decay_motor + ...
            c1 * r1 * (particles(p).best_decay_motor - particles(p).decay_motor) + ...
            c2 * r2 * (global_best_params.decay_motor - particles(p).decay_motor) + noise_decay_motor;
        
        % decay_plan (linear scale) - NEW
        r1 = rand(); r2 = rand();
        decay_plan_range = param_bounds.decay_plan.max - param_bounds.decay_plan.min;
        noise_decay_plan = noise_scale * decay_plan_range * randn();
        particles(p).vel_decay_plan = w * particles(p).vel_decay_plan + ...
            c1 * r1 * (particles(p).best_decay_plan - particles(p).decay_plan) + ...
            c2 * r2 * (global_best_params.decay_plan - particles(p).decay_plan) + noise_decay_plan;
        
        % motor_gain (linear scale)
        r1 = rand(); r2 = rand();
        motor_gain_range = param_bounds.motor_gain.max - param_bounds.motor_gain.min;
        noise_motor_gain = noise_scale * motor_gain_range * randn();
        particles(p).vel_motor_gain = w * particles(p).vel_motor_gain + ...
            c1 * r1 * (particles(p).best_motor_gain - particles(p).motor_gain) + ...
            c2 * r2 * (global_best_params.motor_gain - particles(p).motor_gain) + noise_motor_gain;
        
        % damping (linear scale)
        r1 = rand(); r2 = rand();
        damping_range = param_bounds.damping.max - param_bounds.damping.min;
        noise_damping = noise_scale * damping_range * randn();
        particles(p).vel_damping = w * particles(p).vel_damping + ...
            c1 * r1 * (particles(p).best_damping - particles(p).damping) + ...
            c2 * r2 * (global_best_params.damping - particles(p).damping) + noise_damping;
        
        % reaching_speed_scale (linear scale)
        r1 = rand(); r2 = rand();
        rss_range = param_bounds.reaching_speed_scale.max - param_bounds.reaching_speed_scale.min;
        noise_rss = noise_scale * rss_range * randn();
        particles(p).vel_reaching_speed_scale = w * particles(p).vel_reaching_speed_scale + ...
            c1 * r1 * (particles(p).best_reaching_speed_scale - particles(p).reaching_speed_scale) + ...
            c2 * r2 * (global_best_params.reaching_speed_scale - particles(p).reaching_speed_scale) + noise_rss;
        
        % W_motor_gain (linear scale) - NEW
        r1 = rand(); r2 = rand();
        wm_range = param_bounds.W_motor_gain.max - param_bounds.W_motor_gain.min;
        noise_wm = noise_scale * wm_range * randn();
        particles(p).vel_W_motor_gain = w * particles(p).vel_W_motor_gain + ...
            c1 * r1 * (particles(p).best_W_motor_gain - particles(p).W_motor_gain) + ...
            c2 * r2 * (global_best_params.W_motor_gain - particles(p).W_motor_gain) + noise_wm;
        
        % W_plan_gain (linear scale) - NEW
        r1 = rand(); r2 = rand();
        wp_range = param_bounds.W_plan_gain.max - param_bounds.W_plan_gain.min;
        noise_wp = noise_scale * wp_range * randn();
        particles(p).vel_W_plan_gain = w * particles(p).vel_W_plan_gain + ...
            c1 * r1 * (particles(p).best_W_plan_gain - particles(p).W_plan_gain) + ...
            c2 * r2 * (global_best_params.W_plan_gain - particles(p).W_plan_gain) + noise_wp;
        
        % interference_penalty_weight (linear scale) - NEW
        r1 = rand(); r2 = rand();
        ipw_range = param_bounds.interference_penalty_weight.max - param_bounds.interference_penalty_weight.min;
        noise_ipw = noise_scale * ipw_range * randn();
        particles(p).vel_interference_penalty_weight = w * particles(p).vel_interference_penalty_weight + ...
            c1 * r1 * (particles(p).best_interference_penalty_weight - particles(p).interference_penalty_weight) + ...
            c2 * r2 * (global_best_params.interference_penalty_weight - particles(p).interference_penalty_weight) + noise_ipw;
        
        % Position updates
        % For log-scale parameters, position is updated on log scale then converted
        log_eta_rep_new = log10(particles(p).eta_rep) + particles(p).vel_eta_rep;
        particles(p).eta_rep = 10^log_eta_rep_new;
        
        log_eta_W_new = log10(particles(p).eta_W) + particles(p).vel_eta_W;
        particles(p).eta_W = 10^log_eta_W_new;
        
        particles(p).momentum = particles(p).momentum + particles(p).vel_momentum;
        particles(p).weight_decay = particles(p).weight_decay + particles(p).vel_weight_decay;
        particles(p).decay_motor = particles(p).decay_motor + particles(p).vel_decay_motor;
        particles(p).decay_plan = particles(p).decay_plan + particles(p).vel_decay_plan;
        particles(p).motor_gain = particles(p).motor_gain + particles(p).vel_motor_gain;
        particles(p).damping = particles(p).damping + particles(p).vel_damping;
        particles(p).reaching_speed_scale = particles(p).reaching_speed_scale + particles(p).vel_reaching_speed_scale;
        particles(p).W_motor_gain = particles(p).W_motor_gain + particles(p).vel_W_motor_gain;
        particles(p).W_plan_gain = particles(p).W_plan_gain + particles(p).vel_W_plan_gain;
        particles(p).interference_penalty_weight = particles(p).interference_penalty_weight + particles(p).vel_interference_penalty_weight;
        
        % Enforce bounds on all parameters
        particles(p).eta_rep = max(10^param_bounds.eta_rep.log_min, min(10^param_bounds.eta_rep.log_max, particles(p).eta_rep));
        particles(p).eta_W = max(10^param_bounds.eta_W.log_min, min(10^param_bounds.eta_W.log_max, particles(p).eta_W));
        particles(p).momentum = max(param_bounds.momentum.min, min(param_bounds.momentum.max, particles(p).momentum));
        particles(p).weight_decay = max(param_bounds.weight_decay.min, min(param_bounds.weight_decay.max, particles(p).weight_decay));
        particles(p).decay_motor = max(param_bounds.decay_motor.min, min(param_bounds.decay_motor.max, particles(p).decay_motor));
        particles(p).decay_plan = max(param_bounds.decay_plan.min, min(param_bounds.decay_plan.max, particles(p).decay_plan));
        particles(p).motor_gain = max(param_bounds.motor_gain.min, min(param_bounds.motor_gain.max, particles(p).motor_gain));
        particles(p).damping = max(param_bounds.damping.min, min(param_bounds.damping.max, particles(p).damping));
        particles(p).reaching_speed_scale = max(param_bounds.reaching_speed_scale.min, min(param_bounds.reaching_speed_scale.max, particles(p).reaching_speed_scale));
        particles(p).W_motor_gain = max(param_bounds.W_motor_gain.min, min(param_bounds.W_motor_gain.max, particles(p).W_motor_gain));
        particles(p).W_plan_gain = max(param_bounds.W_plan_gain.min, min(param_bounds.W_plan_gain.max, particles(p).W_plan_gain));
        particles(p).interference_penalty_weight = max(param_bounds.interference_penalty_weight.min, min(param_bounds.interference_penalty_weight.max, particles(p).interference_penalty_weight));
    end
end

% ====================================================================
% PSO COMPLETE - SAVE RESULTS
% ====================================================================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PSO OPTIMIZATION COMPLETE\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Best Parameters Found (15-DIMENSIONAL TASK-CONDITIONAL OPTIMIZATION - Nov 1, 2025):\n');
fprintf('  Score (weighted objective):  %.6f\n', global_best_score);
fprintf('\n  LEARNING RATES:\n');
fprintf('    eta_rep:                   %.6f\n', global_best_params.eta_rep);
fprintf('    eta_W:                     %.6f\n', global_best_params.eta_W);
fprintf('    momentum:                  %.6f\n', global_best_params.momentum);
fprintf('\n  WEIGHT DECAY:\n');
fprintf('    weight_decay (global):     %.6f\n', global_best_params.weight_decay);
fprintf('\n  TASK-CONDITIONAL DECAY RATES (NEW - Nov 1, 2025):\n');
fprintf('    decay_motor:               %.6f (preserve motor across tasks)\n', global_best_params.decay_motor);
fprintf('    decay_plan:                %.6f (forget old task targets)\n', global_best_params.decay_plan);
fprintf('\n  MOTOR DYNAMICS (trajectory quality):\n');
fprintf('    motor_gain:                %.6f\n', global_best_params.motor_gain);
fprintf('    damping:                   %.6f\n', global_best_params.damping);
fprintf('    reaching_speed_scale:      %.6f\n', global_best_params.reaching_speed_scale);
fprintf('\n  WEIGHT INITIALIZATION GAINS (NEW format - Nov 1, 2025):\n');
fprintf('    W_motor_gain:              %.6f (motor weight initialization)\n', global_best_params.W_motor_gain);
fprintf('    W_plan_gain:               %.6f (planning weight initialization)\n', global_best_params.W_plan_gain);
fprintf('\n  TASK-CONDITIONAL LEARNING PARAMETERS (NEW - Nov 1, 2025):\n');
fprintf('    interference_penalty_weight: %.6f (cross-task error penalty)\n\n', global_best_params.interference_penalty_weight);

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
    ps.weight_decay = particles(ip).best_weight_decay;

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
