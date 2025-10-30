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
fprintf('║  PARAMETER OPTIMIZATION - 2D RAO & BALLARD MODEL            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% OPTIMIZATION CONFIGURATION
% ====================================================================

num_trials = 50;  % Number of random parameter sets to test

% Define the search space for each parameter.
% Learning rates are sampled on a log scale, which is standard practice.
param_space = struct();
param_space.eta_rep     = struct('log_min', -3, 'log_max', 0);   % Search 10^-3 to 10^0 (0.001 to 1.0)
param_space.eta_W       = struct('log_min', -4, 'log_max', -1);  % Search 10^-4 to 10^-1 (0.0001 to 0.1)
param_space.momentum    = struct('min', 0.52, 'max', 0.99);     % Search 0.52 to 0.99
param_space.weight_decay = struct('min', 0.900, 'max', 1.0);    % Search 0.900 to 1.0 (no decay)

% Define weights for the objective function.
% Higher weights prioritize minimizing errors in higher-order derivatives.
objective_weights = struct('pos', 1, 'vel', 2, 'acc', 3);

fprintf('OPTIMIZATION SETUP:\n');
fprintf('  Number of trials: %d\n', num_trials);
fprintf('  Search Space (eta_rep):     [%.4f, %.2f]\n', 10^param_space.eta_rep.log_min, 10^param_space.eta_rep.log_max);
fprintf('  Search Space (eta_W):       [%.4f, %.2f]\n', 10^param_space.eta_W.log_min, 10^param_space.eta_W.log_max);
fprintf('  Search Space (momentum):    [%.2f, %.2f]\n', param_space.momentum.min, param_space.momentum.max);
fprintf('  Search Space (weight_decay):[%.4f, %.2f]\n', param_space.weight_decay.min, param_space.weight_decay.max);
fprintf('  Objective Weights: pos=%.1f, vel=%.1f, acc=%.1f\n\n', ...
    objective_weights.pos, objective_weights.vel, objective_weights.acc);

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

% Use parfor for parallel execution if Parallel Computing Toolbox is available
% Change 'parfor' to 'for' if you don't have the toolbox.
parfor trial = 1:num_trials
    
    fprintf('Trial %d/%d...\n', trial, num_trials);
    
    % --- 1. Sample a new set of parameters ---
    current_params = struct();
    
    % Sample learning rates on a log scale
    log_eta_rep = param_space.eta_rep.log_min + rand() * (param_space.eta_rep.log_max - param_space.eta_rep.log_min);
    current_params.eta_rep = 10^log_eta_rep;
    
    log_eta_W = param_space.eta_W.log_min + rand() * (param_space.eta_W.log_max - param_space.eta_W.log_min);
    current_params.eta_W = 10^log_eta_W;
    
    % Sample momentum and decay on a linear scale
    current_params.momentum = param_space.momentum.min + rand() * (param_space.momentum.max - param_space.momentum.min);
    current_params.weight_decay = param_space.weight_decay.min + rand() * (param_space.weight_decay.max - param_space.weight_decay.min);
    
    fprintf('  Testing params: η_rep=%.5f, η_W=%.5f, mom=%.4f, decay=%.5f\n', ...
        current_params.eta_rep, current_params.eta_W, current_params.momentum, current_params.weight_decay);
    
    % --- 2. Run the simulation with these parameters ---
    try
        % Call the simulation function directly (evalc is not compatible with parfor)
        [~, R_L2, R_L3, ~, ~, ~, ~, ~, ~, ~, ~, vx_true, vy_true, ax_true, ay_true, ~, ~] = ...
            hierarchical_motion_inference_2D_EXACT(current_params);
    catch ME
        fprintf('  ✗ Simulation failed for this parameter set: %s\n', ME.message);
        continue; % Skip to the next trial
    end
    
    % --- 3. Calculate the objective score ---
    % Use Root Mean Squared Error (RMSE) for robustness
    vel_rmse = sqrt(mean((R_L2(:,1)' - vx_true).^2 + (R_L2(:,2)' - vy_true).^2));
    acc_rmse = sqrt(mean((R_L3(:,1)' - ax_true).^2 + (R_L3(:,2)' - ay_true).^2));
    
    % Check for NaN or Inf, which indicates instability
    if isnan(vel_rmse) || isinf(vel_rmse) || isnan(acc_rmse) || isinf(acc_rmse)
        fprintf('  ✗ Unstable simulation (NaN/Inf error). Assigning high penalty.\n');
        current_score = inf;
    else
        % Weighted sum of errors (we ignore position error as it's not directly inferred)
        current_score = objective_weights.vel * vel_rmse + objective_weights.acc * acc_rmse;
    end
    
    fprintf('  Result: Vel RMSE=%.5f, Acc RMSE=%.5f -> Score=%.5f\n', vel_rmse, acc_rmse, current_score);
    
    % --- 4. Store results ---
    % *** FIX 2: Store in parfor-compatible variables ***
    trial_params{trial} = current_params;
    trial_scores(trial) = current_score;
    trial_errors{trial} = struct('vel_rmse', vel_rmse, 'acc_rmse', acc_rmse);
    
    % Note: Cannot update 'best_score' inside a parfor loop directly.
    % We will find the best score after the loop finishes.
    fprintf('\n');
end

% --- Convert temporary variables to results structure ---
results = struct();
results.params = [trial_params{:}]';
results.score = trial_scores;
results.errors = [trial_errors{:}]';

% --- Find the best score after the loop ---
[best_score, best_idx] = min(results.score);
if ~isinf(best_score)
    best_params = results.params(best_idx);
    fprintf('  ★ Best score found at trial %d! ★\n\n', best_idx);
else
    best_params = struct(); % No best params found
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
    fprintf('  - Score:          %.6f\n', best_score);
    fprintf('  - eta_rep:        %.6f\n', best_params.eta_rep);
    fprintf('  - eta_W:          %.6f\n', best_params.eta_W);
    fprintf('  - momentum:       %.6f\n', best_params.momentum);
    fprintf('  - weight_decay:   %.6f\n\n', best_params.weight_decay);
    
    % --- MODIFIED: Save results to a file ---
    results_filename = sprintf('optimization_results_%s.mat', datestr(now,'yyyy-mm-dd_HH-MM-SS'));
    fprintf('Saving optimization results to %s\n', results_filename);
    save(results_filename, 'best_params', 'results');
    
    fprintf('You can now use these parameters in the main script for optimal performance.\n');
    
    % --- Visualize the search results ---
    visualize_optimization_results(results);
end

end


function visualize_optimization_results(results)
    % Create plots to show how parameters affect the score
    
    figure('Name', 'Parameter Optimization Results', 'NumberTitle', 'off', ...
           'Position', [100 100 1200 800]);
    
    valid_idx = ~isinf(results.score);
    
    if ~any(valid_idx)
        disp('No valid trials to plot.');
        return;
    end
    
    % *** FIX 3: Robustly extract data from struct array for plotting ***
    scores = results.score(valid_idx);
    params_array = results.params(valid_idx);
    
    eta_rep_vals = arrayfun(@(p) p.eta_rep, params_array);
    eta_W_vals = arrayfun(@(p) p.eta_W, params_array);
    momentum_vals = arrayfun(@(p) p.momentum, params_array);
    
    % Plot 1: Score vs. eta_rep
    subplot(2,2,1);
    semilogx(eta_rep_vals, scores, 'b.', 'MarkerSize', 15);
    xlabel('Representation Learning Rate (eta_rep)');
    ylabel('Objective Score');
    title('Score vs. Representation Learning Rate');
    grid on;
    
    % Plot 2: Score vs. eta_W
    subplot(2,2,2);
    semilogx(eta_W_vals, scores, 'r.', 'MarkerSize', 15);
    xlabel('Weight Learning Rate (eta_W)');
    ylabel('Objective Score');
    title('Score vs. Weight Learning Rate');
    grid on;
    
    % Plot 3: Score vs. momentum
    subplot(2,2,3);
    plot(momentum_vals, scores, 'g.', 'MarkerSize', 15);
    xlabel('Momentum');
    ylabel('Objective Score');
    title('Score vs. Momentum');
    grid on;
    
    % Plot 4: 3D scatter plot of key parameters
    subplot(2,2,4);
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
end