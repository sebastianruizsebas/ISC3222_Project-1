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

fprintf('PSO HYPERPARAMETERS:\n');
fprintf('  Inertia weight (w): %.2f\n', w);
fprintf('  Cognitive parameter (c1): %.2f\n', c1);
fprintf('  Social parameter (c2): %.2f\n\n', c2);

% ====================================================================
% DEFINE SEARCH SPACE FOR PARAMETERS
% ====================================================================

% Parameter bounds
param_bounds = struct();
param_bounds.eta_rep.log_min = -4;      % 10^-4 = 0.0001
param_bounds.eta_rep.log_max = -1;      % 10^-1 = 0.1
param_bounds.eta_W.log_min = -6;        % 10^-6 = 0.000001
param_bounds.eta_W.log_max = -1;        % 10^-1 = 0.1
param_bounds.momentum.min = 0.70;       % Linear scale
param_bounds.momentum.max = 0.98;

% Objective function weights
% For 3D reaching, primary metric is reaching distance improvement
objective_weights = struct('reaching_distance', 1.0, 'position_rmse', 0.5);

fprintf('PARAMETER SEARCH SPACE:\n');
fprintf('  eta_rep:  [%.6f, %.6f] (log scale: 10^[%d, %d])\n', ...
    10^param_bounds.eta_rep.log_min, 10^param_bounds.eta_rep.log_max, ...
    param_bounds.eta_rep.log_min, param_bounds.eta_rep.log_max);
fprintf('  eta_W:    [%.6f, %.6f] (log scale: 10^[%d, %d])\n', ...
    10^param_bounds.eta_W.log_min, 10^param_bounds.eta_W.log_max, ...
    param_bounds.eta_W.log_min, param_bounds.eta_W.log_max);
fprintf('  momentum: [%.2f, %.2f] (linear scale)\n\n', ...
    param_bounds.momentum.min, param_bounds.momentum.max);

fprintf('OBJECTIVE FUNCTION WEIGHTS:\n');
fprintf('  Reaching distance improvement: %.1f\n', objective_weights.reaching_distance);
fprintf('  Position RMSE:                 %.1f\n\n', objective_weights.position_rmse);

% ====================================================================
% INITIALIZE PARTICLE SWARM
% ====================================================================

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('Initializing particle swarm...\n\n');

% Initialize particle positions (random within bounds)
particles = struct();
for p = 1:num_particles
    % Sample on log scale for learning rates
    log_eta_rep = param_bounds.eta_rep.log_min + ...
        rand() * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min);
    particles(p).eta_rep = 10^log_eta_rep;
    
    log_eta_W = param_bounds.eta_W.log_min + ...
        rand() * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min);
    particles(p).eta_W = 10^log_eta_W;
    
    % Sample momentum on linear scale
    particles(p).momentum = param_bounds.momentum.min + ...
        rand() * (param_bounds.momentum.max - param_bounds.momentum.min);
    
    % Fixed parameter
    particles(p).weight_decay = 0.98;
    
    % Initialize velocity (random, typically in [-1, 1] * parameter_range)
    particles(p).vel_eta_rep = -2 * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min) + ...
        rand() * 4 * (param_bounds.eta_rep.log_max - param_bounds.eta_rep.log_min);
    particles(p).vel_eta_W = -2 * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min) + ...
        rand() * 4 * (param_bounds.eta_W.log_max - param_bounds.eta_W.log_min);
    particles(p).vel_momentum = -2 * (param_bounds.momentum.max - param_bounds.momentum.min) + ...
        rand() * 4 * (param_bounds.momentum.max - param_bounds.momentum.min);
    
    % Initialize particle's best position and score
    particles(p).best_eta_rep = particles(p).eta_rep;
    particles(p).best_eta_W = particles(p).eta_W;
    particles(p).best_momentum = particles(p).momentum;
    particles(p).best_score = inf;
end

% Global best tracking
global_best_score = inf;
global_best_params = struct();

fprintf('Swarm initialized with %d particles.\n\n', num_particles);

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

for iteration = 1:num_iterations
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║ PSO Iteration %d/%d (Evaluating %d particles)              ║\n', ...
        iteration, num_iterations, num_particles);
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    iteration_scores = [];
    
    % Evaluate each particle in the swarm
    for p = 1:num_particles
        fprintf('  Particle %d/%d: ', p, num_particles);
        fprintf('η_rep=%.6f, η_W=%.6f, mom=%.4f\n', ...
            particles(p).eta_rep, particles(p).eta_W, particles(p).momentum);
        
        % Create parameter struct for this particle
        current_params = struct();
        current_params.eta_rep = particles(p).eta_rep;
        current_params.eta_W = particles(p).eta_W;
        current_params.momentum = particles(p).momentum;
        current_params.weight_decay = particles(p).weight_decay;
        
        % --- RUN THE 3D MODEL WITH THESE PARAMETERS ---
        try
            % Turn off graphics
            old_visible = get(0, 'DefaultFigureVisible');
            set(0, 'DefaultFigureVisible', 'off');
            
            % Run the 3D model with parameters passed as struct
            hierarchical_motion_inference_3D_EXACT(current_params);
            
            % Restore visibility
            set(0, 'DefaultFigureVisible', old_visible);
            
            % Load results from the saved MAT file
            results_file = './figures/3D_reaching_results.mat';
            if ~isfile(results_file)
                error('Expected results file not found: %s', results_file);
            end
            loaded_data = load(results_file);
            
            % Extract variables from loaded data
            x_true = loaded_data.x_true;
            y_true = loaded_data.y_true;
            z_true = loaded_data.z_true;
            R_L1 = loaded_data.R_L1;
            reaching_error_all = loaded_data.reaching_error_all;
            phases_indices = loaded_data.phases_indices;
            
        catch ME
            fprintf('    ✗ Simulation failed: %s\n', ME.message);
            set(0, 'DefaultFigureVisible', old_visible);
            current_score = inf;
            particles(p).score = inf;
            iteration_scores = [iteration_scores, inf];
            continue;
        end
        
        % --- CALCULATE OBJECTIVE SCORE ---
        n_trials_model = 4;
        trial_reaching_dists = {};
        for t = 1:n_trials_model
            trial_idx = phases_indices{t};
            trial_reaching_dists{t} = reaching_error_all(trial_idx(end));
        end
        
        avg_final_reaching_dist = mean([trial_reaching_dists{:}]);
        
        % Calculate position RMSE for secondary metric
        pos_error_trial = sqrt((x_true - R_L1(:,1)').^2 + ...
                              (y_true - R_L1(:,2)').^2 + ...
                              (z_true - R_L1(:,3)').^2);
        pos_rmse_trial = sqrt(mean(pos_error_trial.^2));
        
        % Check for NaN or Inf
        if isnan(avg_final_reaching_dist) || isinf(avg_final_reaching_dist) || ...
           isnan(pos_rmse_trial) || isinf(pos_rmse_trial)
            fprintf('    ✗ Unstable (NaN/Inf). Assigning high penalty.\n');
            current_score = inf;
        else
            % Objective: minimize reaching distance (primary) and position error (secondary)
            current_score = objective_weights.reaching_distance * avg_final_reaching_dist + ...
                           objective_weights.position_rmse * pos_rmse_trial;
        end
        
        particles(p).score = current_score;
        iteration_scores = [iteration_scores, current_score];
        
        fprintf('    → Score: %.6f (Reach Dist=%.4f, Pos RMSE=%.4f)\n', ...
            current_score, avg_final_reaching_dist, pos_rmse_trial);
        
        % Update particle's personal best
        if current_score < particles(p).best_score
            particles(p).best_score = current_score;
            particles(p).best_eta_rep = particles(p).eta_rep;
            particles(p).best_eta_W = particles(p).eta_W;
            particles(p).best_momentum = particles(p).momentum;
            fprintf('    ★ New personal best: %.6f\n', current_score);
        end
        
        % Update global best
        if current_score < global_best_score
            global_best_score = current_score;
            global_best_params.eta_rep = particles(p).eta_rep;
            global_best_params.eta_W = particles(p).eta_W;
            global_best_params.momentum = particles(p).momentum;
            global_best_params.weight_decay = 0.98;
            fprintf('    ✯ NEW GLOBAL BEST: %.6f ✯\n', global_best_score);
        end
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
    fprintf('\n  Updating particle positions and velocities...\n\n');
    
    for p = 1:num_particles
        % Velocity update equation (standard PSO):
        % v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        
        % eta_rep (log scale)
        r1 = rand();
        r2 = rand();
        particles(p).vel_eta_rep = w * particles(p).vel_eta_rep + ...
            c1 * r1 * (log10(particles(p).best_eta_rep) - log10(particles(p).eta_rep)) + ...
            c2 * r2 * (log10(global_best_params.eta_rep) - log10(particles(p).eta_rep));
        
        % eta_W (log scale)
        r1 = rand();
        r2 = rand();
        particles(p).vel_eta_W = w * particles(p).vel_eta_W + ...
            c1 * r1 * (log10(particles(p).best_eta_W) - log10(particles(p).eta_W)) + ...
            c2 * r2 * (log10(global_best_params.eta_W) - log10(particles(p).eta_W));
        
        % momentum (linear scale)
        r1 = rand();
        r2 = rand();
        particles(p).vel_momentum = w * particles(p).vel_momentum + ...
            c1 * r1 * (particles(p).best_momentum - particles(p).momentum) + ...
            c2 * r2 * (global_best_params.momentum - particles(p).momentum);
        
        % Position update
        % For log-scale parameters, position is updated on log scale then converted
        log_eta_rep_new = log10(particles(p).eta_rep) + particles(p).vel_eta_rep;
        particles(p).eta_rep = 10^log_eta_rep_new;
        
        log_eta_W_new = log10(particles(p).eta_W) + particles(p).vel_eta_W;
        particles(p).eta_W = 10^log_eta_W_new;
        
        particles(p).momentum = particles(p).momentum + particles(p).vel_momentum;
        
        % Enforce bounds
        particles(p).eta_rep = max(10^param_bounds.eta_rep.log_min, ...
                                   min(10^param_bounds.eta_rep.log_max, particles(p).eta_rep));
        particles(p).eta_W = max(10^param_bounds.eta_W.log_min, ...
                                min(10^param_bounds.eta_W.log_max, particles(p).eta_W));
        particles(p).momentum = max(param_bounds.momentum.min, ...
                                   min(param_bounds.momentum.max, particles(p).momentum));
    end
end

% ====================================================================
% PSO COMPLETE - SAVE RESULTS
% ====================================================================

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PSO OPTIMIZATION COMPLETE\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Best Parameters Found:\n');
fprintf('  Score (weighted objective):  %.6f\n', global_best_score);
fprintf('  eta_rep:                     %.6f\n', global_best_params.eta_rep);
fprintf('  eta_W:                       %.6f\n', global_best_params.eta_W);
fprintf('  momentum:                    %.6f\n', global_best_params.momentum);
fprintf('  weight_decay (FIXED):        %.6f\n\n', global_best_params.weight_decay);

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
