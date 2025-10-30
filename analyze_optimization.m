% ANALYZE OPTIMIZATION RESULTS
% ============================
%
% This script loads the results from the parameter optimization search
% and displays the top 3 best-performing parameter sets.

% Clear workspace and command window
clc;
clear;
close all;

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  ANALYSIS OF TOP OPTIMIZATION RESULTS                       ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% --- 1. Load the most recent optimization results file ---
fprintf('Searching for optimization results file...\n');
result_files = dir('optimization_results_3D_2025-10-30_12-46-29.mat');

if isempty(result_files)
    error('No optimization results file found. Please run "optimize_rao_ballard_parameters.m" first.');
end

% Find the most recent file by date
[~, latest_idx] = max([result_files.datenum]);
results_filename = result_files(latest_idx).name;

fprintf('Loading data from: %s\n\n', results_filename);
load(results_filename, 'results'); % Loads the 'results' struct

% --- 2. Find the top 3 parameter sets ---
% Sort the results by score in ascending order
[sorted_scores, sorted_indices] = sort(results.score, 'ascend');

% Determine how many top results to show (up to 20)
num_valid_results = sum(~isinf(sorted_scores));
num_to_display = min(20, num_valid_results);

if num_to_display == 0
    fprintf('No valid (non-infinite score) results found in the data file.\n');
    return;
end

fprintf('Displaying Top %d Parameter Sets:\n', num_to_display);
fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n');
fprintf('%-5s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n', ...
    'Rank', 'Score', 'Reach Dist', 'Pos RMSE', 'eta_rep', 'eta_W', 'Momentum', 'Weight Decay');
fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n');

% --- 3. Display the details of the top sets ---
for i = 1:num_to_display
    % Get the index for the i-th best result
    idx = sorted_indices(i);
    
    % Extract the data for this rank
    score = results.score(idx);
    params = results.params(idx);
    errors = results.errors(idx);
    
    % Print the formatted row
    fprintf('%-5d | %-12.4f | %-12.4f | %-12.4f | %-12.6f | %-12.6f | %-12.4f | %-12.4f\n', ...
        i, ...
        score, ...
        errors.reaching_dist, ...
        errors.pos_rmse, ...
        params.eta_rep, ...
        params.eta_W, ...
        params.momentum, ...
        params.weight_decay);
end

fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n');
fprintf('Analysis complete.\n');