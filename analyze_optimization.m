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

% --- 1. Load PSO top-20 leaderboard (preferred) or fallback to older optimization results ---
fprintf('Searching for PSO leaderboard (./figures/pso_top20_best_params.mat)...\n');
leader_file = fullfile('./figures', 'pso_top20_best_params.mat');
if isfile(leader_file)
    fprintf('Found leaderboard file: %s\n\n', leader_file);
    S = load(leader_file, 'leader_list');
    leader_list = S.leader_list;
    use_leader_list = true;
else
    fprintf('Leaderboard file not found, falling back to legacy optimization results in ./optimization_results/...\n');
    result_files = dir('./optimization_results/rao_ballard_3D_optimization_*.mat');
    if isempty(result_files)
        error('No optimization results file or PSO leaderboard found. Please run PSO first.');
    end
    [~, latest_idx] = max([result_files.datenum]);
    results_filename = fullfile(result_files(latest_idx).folder, result_files(latest_idx).name);
    fprintf('Loading legacy optimization results from: %s\n\n', results_filename);
    load(results_filename, 'results'); % Loads the 'results' struct
    use_leader_list = false;
end

if use_leader_list
    % leader_list is a struct array with fields: score, params
    n_leader = numel(leader_list);
    num_to_display = min(20, n_leader);
    if num_to_display == 0
        fprintf('Leaderboard is empty.\n');
        return;
    end

    fprintf('Displaying Top %d PSO Parameter Sets (from %s):\n', num_to_display, leader_file);
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n');
    fprintf('%-5s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', ...
        'Rank', 'Score', 'eta_rep', 'eta_W', 'momentum', 'decay_L2', 'decay_L1', 'motor_gain', 'damping', 'reach_scale', 'W_L2');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n');

    for i = 1:num_to_display
        sc = leader_list(i).score;
        p = leader_list(i).params;
        fprintf('%-5d | %-12.6f | %-10.6g | %-10.6g | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', ...
            i, sc, p.eta_rep, p.eta_W, p.momentum, p.decay_L2_goal, p.decay_L1_motor, p.motor_gain, p.damping, p.reaching_speed_scale, p.W_L2_goal_gain);
    end
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n');
    fprintf('Analysis complete.\n');
else
    % Legacy results struct handling (best-effort mapping)
    if ~isfield(results, 'top20') && ~isfield(results, 'score')
        % Try to construct a score / params list from results.particles (personal bests)
        fprintf('Legacy results detected: constructing leaderboard from particles.personal bests...\n');
        num_particles = numel(results.particles);
        all_scores = inf(num_particles,1);
        for pp = 1:num_particles
            if isfield(results.particles(pp), 'best_score')
                all_scores(pp) = results.particles(pp).best_score;
            end
        end
        [sorted_scores, sorted_idx] = sort(all_scores, 'ascend');
        valid_mask = isfinite(sorted_scores);
        num_to_display = min(20, sum(valid_mask));
        fprintf('Displaying Top %d (constructed from particles)\n', num_to_display);
        fprintf('════════════════════════════════════════════════════════════════\n');
        fprintf('%-5s | %-12s | %-10s | %-10s | %-10s\n', 'Rank', 'Score', 'eta_rep', 'eta_W', 'momentum');
        fprintf('════════════════════════════════════════════════════════════════\n');
        for i = 1:num_to_display
            ip = sorted_idx(i);
            sc = sorted_scores(i);
            p = results.particles(ip);
            fprintf('%-5d | %-12.6f | %-10.6g | %-10.6g | %-10.4f\n', i, sc, p.best_eta_rep, p.best_eta_W, p.best_momentum);
        end
        fprintf('════════════════════════════════════════════════════════════════\n\n');
    else
        % If results.top20 exists (some older scripts), prefer it
        if isfield(results, 'top20')
            leader_list = results.top20;
            n_leader = numel(leader_list);
            num_to_display = min(20, n_leader);
            fprintf('Displaying Top %d Parameter Sets (from results.top20):\n', num_to_display);
            fprintf('════════════════════════════════════════════════════════════════\n');
            fprintf('%-5s | %-12s | %-10s | %-10s | %-10s\n', 'Rank', 'Score', 'eta_rep', 'eta_W', 'momentum');
            fprintf('════════════════════════════════════════════════════════════════\n');
            for i = 1:num_to_display
                sc = leader_list(i).score;
                p = leader_list(i).params;
                fprintf('%-5d | %-12.6f | %-10.6g | %-10.6g | %-10.4f\n', i, sc, p.eta_rep, p.eta_W, p.momentum);
            end
            fprintf('════════════════════════════════════════════════════════════════\n\n');
        else
            error('Cannot interpret legacy results structure. Please run PSO or convert results.');
        end
    end
end

% After leader_list exists (use the top entry)
best_params = leader_list(1).params;

% Map to main function names if desired (optional)
params_to_run = struct( ...
    'eta_rep', best_params.eta_rep, ...
    'eta_W', best_params.eta_W, ...
    'momentum', best_params.momentum, ...
    'decay_plan', best_params.decay_L2_goal, ...
    'decay_motor', best_params.decay_L1_motor, ...
    'motor_gain', best_params.motor_gain, ...
    'damping', best_params.damping, ...
    'reaching_speed_scale', best_params.reaching_speed_scale, ...
    'W_plan_gain', best_params.W_L2_goal_gain, ...
    'W_motor_gain', best_params.W_L1_pos_gain, ...
    'save_results', true ...
);

save(fullfile('./figures','pso_best_params.mat'),'params_to_run');
fprintf('Saved best params to ./figures/pso_best_params.mat\n');