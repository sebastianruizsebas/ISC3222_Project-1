% filepath: c:\Users\srseb\OneDrive\School\FSU\Fall 2025\Symbolic Numeric Computation w Alan Lemmon\Project1\optimize_rao_ballard_parameters.m
%
% PARAMETER OPTIMIZATION FOR 2D RAO & BALLARD MODEL
% ===================================================
%
% This script performs a random search to find the optimal set of
% hyperparameters for the hierarchical_motion_inference_2D_EXACT model.
%
% It aims to minimize a weighted combination of position, velocity,
% and acceleration errors.
%
% METHOD:
%   1. Define a search space for key parameters (learning rates, momentum).
%   2. Run a specified number of trials.
%   3. In each trial, randomly sample parameters from the search space.
%   4. Run the simulation with the sampled parameters.
%   5. Calculate a single objective score based on the resulting errors.
%   6. Track the best parameters that yield the lowest score.
%   7. Report the best parameters and visualize the search results.
%

function [best_params, results] = optimize_rao_ballard_parameters()

clc;
clear global;

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  PARAMETER OPTIMIZATION - 3D RAO & BALLARD MODEL            ║\n');
fprintf('║  Hierarchical Motion Inference with Active Inference        ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% OPTIMIZATION CONFIGURATION
% ====================================================================

num_trials = 2000;  % Number of random parameter sets to test

% Define the search space for each parameter.
% Learning rates are sampled on a log scale, which is standard practice.
param_space = struct();
param_space.eta_rep     = struct('log_min', -4, 'log_max', -1);   % Search 10^-4 to 10^-1 (0.0001 to 0.1)
param_space.eta_W       = struct('log_min', -6, 'log_max', -2);   % Search 10^-6 to 10^-2 (0.000001 to 0.01)
param_space.momentum    = struct('min', 0.80, 'max', 0.98);       % Search 0.80 to 0.98
% Note: weight_decay and other architecture parameters are NOT optimized
% They have theoretical justification and are kept fixed

% Define weights for the objective function.
% For 3D reaching, we primarily care about reaching distance improvement
objective_weights = struct('reaching_distance', 1.0, 'position_rmse', 0.5);

fprintf('OPTIMIZATION SETUP:\n');
fprintf('  Target Model: hierarchical_motion_inference_3D_EXACT\n');
fprintf('  Number of trials: %d\n', num_trials);
fprintf('  Search Space (eta_rep):     [%.6f, %.4f]\n', 10^param_space.eta_rep.log_min, 10^param_space.eta_rep.log_max);
fprintf('  Search Space (eta_W):       [%.6f, %.4f]\n', 10^param_space.eta_W.log_min, 10^param_space.eta_W.log_max);
fprintf('  Search Space (momentum):    [%.2f, %.2f]\n', param_space.momentum.min, param_space.momentum.max);
fprintf('\n');
fprintf('  Objective Weights:\n');
fprintf('    - Reaching distance improvement: %.1f\n', objective_weights.reaching_distance);
fprintf('    - Position RMSE:                  %.1f\n', objective_weights.position_rmse);
fprintf('\n');
fprintf('  FIXED PARAMETERS (NOT optimized):\n');
fprintf('    - weight_decay = 0.98 (prevents catastrophic forgetting at phase boundaries)\n');
fprintf('    - Motor dynamics (gain, damping, reaching_speed scaling) - set by task\n');
fprintf('    - Precision weights (pi_L1, pi_L2, pi_L3) - set by uncertainty model\n\n');

% ====================================================================
% RANDOM SEARCH LOOP
% ====================================================================

% *** FIX 1: Pre-allocate results arrays for efficiency ***
% Use cell arrays and regular arrays for parfor compatibility
trial_params = cell(num_trials, 1);
trial_scores = ones(num_trials, 1) * inf;
trial_errors = cell(num_trials, 1);

best_score = inf;
best_params = struct();

fprintf('Starting random search...\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

% Use for loop (not parfor) because evalc cannot work in parallel mode
for trial = 1:num_trials
    
    fprintf('Trial %d/%d...\n', trial, num_trials);
    
    % --- 1. Sample a new set of parameters ---
    current_params = struct();
    
    % Sample learning rates on a log scale
    log_eta_rep = param_space.eta_rep.log_min + rand() * (param_space.eta_rep.log_max - param_space.eta_rep.log_min);
    current_params.eta_rep = 10^log_eta_rep;
    
    log_eta_W = param_space.eta_W.log_min + rand() * (param_space.eta_W.log_max - param_space.eta_W.log_min);
    current_params.eta_W = 10^log_eta_W;
    
    % Sample momentum on a linear scale
    current_params.momentum = param_space.momentum.min + rand() * (param_space.momentum.max - param_space.momentum.min);
    
    % weight_decay is FIXED - not optimized
    current_params.weight_decay = 0.98;
    
    fprintf('  Testing params: η_rep=%.6f, η_W=%.6f, mom=%.4f\n', ...
        current_params.eta_rep, current_params.eta_W, current_params.momentum);
    
    % --- 2. Run the simulation with these parameters ---
    try
        % Call the 3D hierarchical motion inference model
        % Run in batch mode with no graphics output
        old_visible = get(0, 'DefaultFigureVisible');
        set(0, 'DefaultFigureVisible', 'off');
        
        % Note: hierarchical_motion_inference_3D_EXACT is a script, not a function
        % We need to run it in the workspace with the parameters pre-set
        % Save current params to workspace
        eta_rep = current_params.eta_rep;
        eta_W = current_params.eta_W;
        momentum = current_params.momentum;
        weight_decay = current_params.weight_decay;
        
        % Suppress output
        evalc('hierarchical_motion_inference_3D_EXACT');
        
        % The script populates: x_true, y_true, z_true, R_L1, phases_indices, targets, reaching_error_all
        
        set(0, 'DefaultFigureVisible', old_visible);
    catch ME
        fprintf('  ✗ Simulation failed for this parameter set: %s\n', ME.message);
        set(0, 'DefaultFigureVisible', old_visible);
        trial_params{trial} = current_params;
        trial_scores(trial) = inf;
        trial_errors{trial} = struct('reaching_dist', inf, 'pos_rmse', inf);
        continue; % Skip to the next trial
    end
    
    % --- 3. Calculate the objective score ---
    % For 3D reaching task, primary metric is reaching distance improvement
    % reaching_error_all is automatically calculated in the script
    
    % Calculate average reaching distance across all 4 trials
    % (lower is better - goal is to reach targets)
    n_trials_model = 4;  % Fixed in the 3D model
    trial_reaching_dists = {};
    for t = 1:n_trials_model
        trial_idx = phases_indices{t};
        % Final reaching distance for this trial (last timestep of trial)
        trial_reaching_dists{t} = reaching_error_all(trial_idx(end));
    end
    
    avg_final_reaching_dist = mean([trial_reaching_dists{:}]);
    
    % Also calculate position RMSE for secondary metric
    pos_error_trial = sqrt((x_true - R_L1(:,1)').^2 + (y_true - R_L1(:,2)').^2 + (z_true - R_L1(:,3)').^2);
    pos_rmse_trial = sqrt(mean(pos_error_trial.^2));
    
    % Check for NaN or Inf, which indicates instability
    if isnan(avg_final_reaching_dist) || isinf(avg_final_reaching_dist) || ...
       isnan(pos_rmse_trial) || isinf(pos_rmse_trial)
        fprintf('  ✗ Unstable simulation (NaN/Inf error). Assigning high penalty.\n');
        current_score = inf;
    else
        % Objective: minimize reaching distance (primary) and position error (secondary)
        current_score = objective_weights.reaching_distance * avg_final_reaching_dist + ...
                       objective_weights.position_rmse * pos_rmse_trial;
    end
    
    fprintf('  Result: Avg Final Reaching Dist=%.5f m, Pos RMSE=%.5f m -> Score=%.5f\n', ...
        avg_final_reaching_dist, pos_rmse_trial, current_score);
    
    % --- 4. Store results ---
    trial_params{trial} = current_params;
    trial_scores(trial) = current_score;
    trial_errors{trial} = struct('reaching_dist', avg_final_reaching_dist, 'pos_rmse', pos_rmse_trial);
    
    % Update best score if this is better
    if current_score < best_score
        best_score = current_score;
        best_params = current_params;
        fprintf('  ★ New best score: %.6f\n', best_score);
    end
    
    fprintf('\n');
end

% --- Convert temporary variables to results structure ---
results = struct();
results.params = [trial_params{:}]';
results.score = trial_scores;
results.errors = [trial_errors{:}]';

% best_score and best_params already updated inside the loop

if ~isinf(best_score)
    fprintf('  ★ Best score found at trial %d! ★\n\n', find([trial_scores(:)] == best_score, 1));
else
    fprintf('  ✗ No stable parameter set was found.\n\n');
    best_params = struct();
end


fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% ====================================================================
% FINAL REPORT AND VISUALIZATION
% ====================================================================

if isinf(best_score)
    fprintf('No stable parameter set was found.\n');
else
    fprintf('Best Parameters Found:\n');
    fprintf('  - Score (weighted objective):  %.6f\n', best_score);
    fprintf('  - eta_rep:                     %.6f\n', best_params.eta_rep);
    fprintf('  - eta_W:                       %.6f\n', best_params.eta_W);
    fprintf('  - momentum:                    %.6f\n', best_params.momentum);
    fprintf('  - weight_decay (FIXED):        %.6f\n\n', best_params.weight_decay);
    
    % --- MODIFIED: Save results to a file ---
    results_filename = sprintf('optimization_results_3D_%s.mat', datestr(now,'yyyy-mm-dd_HH-MM-SS'));
    fprintf('Saving optimization results to %s\n', results_filename);
    save(results_filename, 'best_params', 'results');
    
    fprintf('You can now use these parameters in hierarchical_motion_inference_3D_EXACT for optimal performance.\n');
    fprintf('Update the LEARNING PARAMETERS section with:\n');
    fprintf('  eta_rep = %.6f;\n', best_params.eta_rep);
    fprintf('  eta_W = %.6f;\n', best_params.eta_W);
    fprintf('  momentum = %.6f;\n\n', best_params.momentum);
    
    % --- Visualize the search results ---
    visualize_optimization_results(results);
end

end


function visualize_optimization_results(results)
    % Create plots to show how parameters affect the score
    % For 3D reaching optimization
    
    figure('Name', 'Parameter Optimization Results - 3D Reaching', 'NumberTitle', 'off', ...
           'Position', [100 100 1200 800], 'Visible', 'off');
    
    valid_idx = ~isinf(results.score);
    
    if ~any(valid_idx)
        disp('No valid trials to plot.');
        return;
    end
    
    % Extract data from struct array for plotting
    scores = results.score(valid_idx);
    params_array = results.params(valid_idx);
    
    eta_rep_vals = arrayfun(@(p) p.eta_rep, params_array);
    eta_W_vals = arrayfun(@(p) p.eta_W, params_array);
    momentum_vals = arrayfun(@(p) p.momentum, params_array);
    reaching_dist_vals = arrayfun(@(e) e.reaching_dist, results.errors(valid_idx));
    pos_rmse_vals = arrayfun(@(e) e.pos_rmse, results.errors(valid_idx));
    
    % Plot 1: Score vs. eta_rep
    subplot(2,3,1);
    semilogx(eta_rep_vals, scores, 'b.', 'MarkerSize', 15);
    xlabel('Representation Learning Rate (eta_rep)');
    ylabel('Objective Score');
    title('Score vs. Representation Learning Rate');
    grid on;
    
    % Plot 2: Score vs. eta_W
    subplot(2,3,2);
    semilogx(eta_W_vals, scores, 'r.', 'MarkerSize', 15);
    xlabel('Weight Learning Rate (eta_W)');
    ylabel('Objective Score');
    title('Score vs. Weight Learning Rate');
    grid on;
    
    % Plot 3: Score vs. momentum
    subplot(2,3,3);
    plot(momentum_vals, scores, 'g.', 'MarkerSize', 15);
    xlabel('Momentum');
    ylabel('Objective Score');
    title('Score vs. Momentum');
    grid on;
    
    % Plot 4: Final Reaching Distance vs. eta_rep
    subplot(2,3,4);
    semilogx(eta_rep_vals, reaching_dist_vals, 'b.', 'MarkerSize', 15);
    xlabel('Representation Learning Rate (eta_rep)');
    ylabel('Final Reaching Distance (m)');
    title('Reaching Performance vs. eta_rep');
    grid on;
    
    % Plot 5: Position RMSE vs. momentum
    subplot(2,3,5);
    plot(momentum_vals, pos_rmse_vals, 'g.', 'MarkerSize', 15);
    xlabel('Momentum');
    ylabel('Position RMSE (m)');
    title('Position Accuracy vs. Momentum');
    grid on;
    
    % Plot 6: 3D scatter plot of key parameters
    subplot(2,3,6);
    scatter3(eta_rep_vals, eta_W_vals, momentum_vals, 40, scores, 'filled');
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('eta_rep');
    ylabel('eta_W');
    zlabel('momentum');
    title('Parameter Space vs. Score');
    cb = colorbar;
    ylabel(cb, 'Objective Score');
    grid on;
    view(3);
    
    sgtitle('3D Reaching Optimization - Parameter Search Results', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Save figure
    try
        saveas(gcf, 'optimization_results_3D_visualization.png');
        fprintf('✓ Visualization saved: optimization_results_3D_visualization.png\n');
    catch
        fprintf('Warning: Could not save visualization\n');
    end
end