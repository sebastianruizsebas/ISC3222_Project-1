%% ANALYZE PSO OPTIMIZATION RESULTS
% Reads the latest PSO results file and extracts/creates top20 leaderboard

clear all; close all; clc;

fprintf('════════════════════════════════════════════════════════════════\n');
fprintf('PSO RESULTS ANALYZER\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

% Find the latest optimization_results_3D_PSO_*.mat file
pso_files = dir('./figures/pso_top200_best_params.mat');
if isempty(pso_files)
    error('No PSO results files found. Run optimize_rao_ballard_pso.m first.');
end

[~, idx] = sort([pso_files.datenum], 'descend');
latest_file = pso_files(idx(1));
pso_filepath = fullfile(latest_file.folder, latest_file.name);
fprintf('Found latest PSO results: %s\n', latest_file.name);
fprintf('  File size: %.2f MB\n', latest_file.bytes / 1e6);
fprintf('  Date: %s\n\n', datestr(latest_file.datenum));

% Load PSO results
fprintf('Loading PSO results...\n');
S = load(pso_filepath); %#ok<NASGU>

% Support two file formats:
%  - Legacy: contains 'results' struct (results.particles, results.best_score, ...)
%  - New: contains 'leader_list' and 'param_stats' directly (saved by optimize_rao_ballard_pso)
if isfield(S, 'results')
    results = S.results;
    fprintf('✓ PSO results (legacy results struct) loaded\n');
    if isfield(results, 'best_score'), best_score = results.best_score; else best_score = NaN; end
    if isfield(results, 'num_particles'), num_particles = results.num_particles; else num_particles = NaN; end
    if isfield(results, 'num_iterations'), num_iterations = results.num_iterations; else num_iterations = NaN; end
    if isfield(results, 'total_evaluations'), total_evals = results.total_evaluations; else total_evals = NaN; end

    % If leader_list is embedded, prefer it; otherwise build later from particles
    if isfield(results, 'leader_list')
        leader_list = results.leader_list;
    else
        leader_list = [];
    end
elseif isfield(S, 'leader_list')
    leader_list = S.leader_list;
    param_stats = [];
    if isfield(S, 'param_stats'), param_stats = S.param_stats; end
    fprintf('✓ PSO leaderboard file loaded (leader_list + param_stats)\n');
    best_score = leader_list(1).score;
    % num_particles/iterations unknown in this file; try to recover from param_stats or set NaN
    num_particles = NaN;
    num_iterations = NaN;
    total_evals = NaN;
else
    error('Loaded file does not contain expected variables (results or leader_list). Found: %s', strjoin(fieldnames(S), ', '));
end

fprintf('  Best score (est): %.6g\n', best_score);
fprintf('  Total particles (est): %s\n', mat2str(num_particles));
fprintf('  Total iterations (est): %s\n', mat2str(num_iterations));
fprintf('  Total evaluations (est): %s\n\n', mat2str(total_evals));

% ====================================================================
% Extract top20 leaderboard from particles
% ====================================================================
fprintf('Extracting top-200 best parameter sets...\n');
% If a pre-built leader_list was loaded from the .mat file, use it directly.
if exist('leader_list', 'var') && ~isempty(leader_list)
    fprintf('Using pre-saved leader_list with %d entries\n', length(leader_list));
    % Ensure top_n reflects available entries
    top_n = length(leader_list);
else
    % Build leader_list from legacy results.particles
    if ~exist('results', 'var') || ~isfield(results, 'particles')
        error('Results structure does not contain particles field and no leader_list was loaded.');
    end

    particles = results.particles;
    num_particles = length(particles);

    % Collect all particle best scores
    all_scores = zeros(num_particles, 1);
    for p = 1:num_particles
        all_scores(p) = particles(p).best_score;
    end

    % Sort by score (ascending = better)
    [sorted_scores, idx] = sort(all_scores, 'ascend');

    % Count valid (finite) scores
    valid_mask = isfinite(sorted_scores);
    num_valid = sum(valid_mask);
    fprintf('  Valid (finite) scores: %d / %d\n', num_valid, num_particles);

    % Take top 20 valid scores (legacy behavior)
    top_n = min(20, num_valid);
    fprintf('  Building top-%d leaderboard...\n\n', top_n);

    % Build leader_list structure
    leader_list = struct('score', cell(top_n,1), 'params', cell(top_n,1), ...
                         'particle_id', cell(top_n,1), 'rank', cell(top_n,1));

    for k = 1:top_n
        ip = idx(k);  % particle index

        % Create params structure with all expected parameters
        ps = struct();
        ps.eta_rep = particles(ip).best_eta_rep;
        ps.eta_W = particles(ip).best_eta_W;
        ps.momentum = particles(ip).best_momentum;
        ps.weight_decay = particles(ip).best_weight_decay;
        ps.decay_motor = particles(ip).best_decay_motor;
        ps.decay_plan = particles(ip).best_decay_plan;
        ps.motor_gain = particles(ip).best_motor_gain;
        ps.damping = particles(ip).best_damping;
        ps.reaching_speed_scale = particles(ip).best_reaching_speed_scale;
        ps.W_motor_gain = particles(ip).best_W_motor_gain;
        ps.W_plan_gain = particles(ip).best_W_plan_gain;
        ps.interference_penalty_weight = particles(ip).best_interference_penalty_weight;

        % Add precision parameters if they exist (may not be in older particle records)
        if isfield(particles(ip), 'best_alpha_precision_gain')
            ps.alpha_precision_gain = particles(ip).best_alpha_precision_gain;
        end
        if isfield(particles(ip), 'best_pi_L1_motor_min')
            ps.pi_L1_motor_min = particles(ip).best_pi_L1_motor_min;
        end
        if isfield(particles(ip), 'best_pi_L1_motor_max')
            ps.pi_L1_motor_max = particles(ip).best_pi_L1_motor_max;
        end
        if isfield(particles(ip), 'best_pi_L2_motor_min')
            ps.pi_L2_motor_min = particles(ip).best_pi_L2_motor_min;
        end
        if isfield(particles(ip), 'best_pi_L2_motor_max')
            ps.pi_L2_motor_max = particles(ip).best_pi_L2_motor_max;
        end
        if isfield(particles(ip), 'best_pi_L1_plan_min')
            ps.pi_L1_plan_min = particles(ip).best_pi_L1_plan_min;
        end
        if isfield(particles(ip), 'best_pi_L1_plan_max')
            ps.pi_L1_plan_max = particles(ip).best_pi_L1_plan_max;
        end
        if isfield(particles(ip), 'best_pi_L2_plan_min')
            ps.pi_L2_plan_min = particles(ip).best_pi_L2_plan_min;
        end
        if isfield(particles(ip), 'best_pi_L2_plan_max')
            ps.pi_L2_plan_max = particles(ip).best_pi_L2_plan_max;
        end

        % Store in leader_list
        leader_list(k).score = sorted_scores(k);
        leader_list(k).params = ps;
        leader_list(k).particle_id = ip;
        leader_list(k).rank = k;
    end
end

% ====================================================================
% Display Top-20 Leaderboard
% ====================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('TOP 20 PARAMETER SETS\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Rank │   Score   │ Particle │ eta_rep  │ eta_W    │ momentum\n');
fprintf('─────┼───────────┼──────────┼──────────┼──────────┼──────────\n');

for k = 1:top_n
    % Safe access for optional fields (particle_id may not exist in saved leader_list)
    if isfield(leader_list, 'particle_id') && ~isempty(leader_list(k).particle_id)
        pid = leader_list(k).particle_id;
    else
        pid = -1; % unknown particle id
    end
    % Some older leader entries might lack full params; guard access
    eta_rep = NaN; eta_W = NaN; momentum_val = NaN;
    if isfield(leader_list(k), 'params')
        p = leader_list(k).params;
        if isfield(p, 'eta_rep'), eta_rep = p.eta_rep; end
        if isfield(p, 'eta_W'), eta_W = p.eta_W; end
        if isfield(p, 'momentum'), momentum_val = p.momentum; end
    end
    fprintf('%4d │ %9.6f │ %8d │ %.6e │ %.6e │ %.6e\n', ...
        k, leader_list(k).score, pid, eta_rep, eta_W, momentum_val);
end

fprintf('\n');

% ====================================================================
% Save Top-20 to pso_top20_best_params.mat
% ====================================================================
fprintf('Saving top-20 leaderboard...\n');

out_dir = './figures';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

save_path = fullfile(out_dir, 'pso_top20_best_params.mat');
save(save_path, 'leader_list');
fprintf('✓ Top-20 saved to: %s\n\n', save_path);

% ====================================================================
% Save best_params for quick access
% ====================================================================
fprintf('Saving best parameters...\n');
best_params = leader_list(1).params;
best_score = leader_list(1).score;

save_path_best = fullfile(out_dir, 'pso_best_params.mat');
save(save_path_best, 'best_params', 'best_score');
fprintf('✓ Best params saved to: %s\n\n', save_path_best);

% ====================================================================
% Summary Statistics
% ====================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('SUMMARY STATISTICS\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% If param_stats was saved with the PSO leaderboard, print a compact summary
if exist('param_stats','var') && ~isempty(param_stats)
    try
        fprintf('\nPARAMETER STATS (saved bounds vs observed across saved top-N)\n');
        fprintf('  %-28s %-12s %-12s %-12s %-12s\n', 'parameter', 'saved_min', 'saved_max', 'obs_min', 'obs_max');
        fprintf('  %-28s %-12s %-12s %-12s %-12s\n', repmat('-',1,28), repmat('-',1,12), repmat('-',1,12), repmat('-',1,12), repmat('-',1,12));
        % param_stats can be either: struct with fieldnames per parameter
        % (param_stats.<pname> = struct(...)) OR an array of structs with
        % .name/.bounds_min/.topN_observed_min fields. Handle both.
        if isstruct(param_stats) && ~isempty(fieldnames(param_stats)) && ~all(cellfun(@isempty, fieldnames(param_stats))) && ~isfield(param_stats, 'name')
            fn = fieldnames(param_stats);
            for f = 1:numel(fn)
                pname = fn{f};
                ps = param_stats.(pname);
                % saved bounds
                if isfield(ps, 'bounds_min')
                    smin = ps.bounds_min;
                else
                    smin = NaN;
                end
                if isfield(ps, 'bounds_max')
                    smax = ps.bounds_max;
                else
                    smax = NaN;
                end
                % observed
                if isfield(ps, 'topN_observed_min')
                    omin = ps.topN_observed_min;
                else
                    omin = NaN;
                end
                if isfield(ps, 'topN_observed_max')
                    omax = ps.topN_observed_max;
                else
                    omax = NaN;
                end
                fprintf('  %-28s %12.4g %12.4g %12.4g %12.4g\n', pname, smin, smax, omin, omax);
            end
        else
            % Assume array-of-structs format
            for pi = 1:numel(param_stats)
                ps = param_stats(pi);
                if isfield(ps, 'name')
                    pname = ps.name;
                elseif isfield(ps, 'param_name')
                    pname = ps.param_name;
                else
                    pname = sprintf('param_%d', pi);
                end

                if isfield(ps, 'min')
                    smin = ps.min;
                elseif isfield(ps, 'saved_min')
                    smin = ps.saved_min;
                elseif isfield(ps, 'bound_min')
                    smin = ps.bound_min;
                else
                    smin = NaN;
                end

                if isfield(ps, 'max')
                    smax = ps.max;
                elseif isfield(ps, 'saved_max')
                    smax = ps.saved_max;
                elseif isfield(ps, 'bound_max')
                    smax = ps.bound_max;
                else
                    smax = NaN;
                end

                if isfield(ps, 'observed_min')
                    omin = ps.observed_min;
                elseif isfield(ps, 'obs_min')
                    omin = ps.obs_min;
                elseif isfield(ps, 'min_observed')
                    omin = ps.min_observed;
                else
                    omin = NaN;
                end

                if isfield(ps, 'observed_max')
                    omax = ps.observed_max;
                elseif isfield(ps, 'obs_max')
                    omax = ps.obs_max;
                elseif isfield(ps, 'max_observed')
                    omax = ps.max_observed;
                else
                    omax = NaN;
                end
                fprintf('  %-28s %12.4g %12.4g %12.4g %12.4g\n', pname, smin, smax, omin, omax);
            end
        end
        fprintf('\n');
    catch ME
        fprintf('Warning: failed to print param_stats summary: %s\n', ME.message);
    end
end

fprintf('Score Statistics:\n');
fprintf('  Best (Top-1):    %.6f\n', leader_list(1).score);
fprintf('  Top-5 mean:      %.6f\n', mean([leader_list(1:5).score]));
fprintf('  Top-10 mean:     %.6f\n', mean([leader_list(1:10).score]));
fprintf('  Top-20 mean:     %.6f\n', mean([leader_list(1:top_n).score]));

fprintf('\nTop-1 Parameters:\n');
fprintf('  eta_rep:                   %.6e\n', best_params.eta_rep);
fprintf('  eta_W:                     %.6e\n', best_params.eta_W);
fprintf('  momentum:                  %.6e\n', best_params.momentum);
fprintf('  weight_decay:              %.6e\n', best_params.weight_decay);
fprintf('  decay_motor:               %.6e\n', best_params.decay_motor);
fprintf('  decay_plan:                %.6e\n', best_params.decay_plan);
fprintf('  motor_gain:                %.6e\n', best_params.motor_gain);
fprintf('  damping:                   %.6e\n', best_params.damping);
fprintf('  reaching_speed_scale:      %.6e\n', best_params.reaching_speed_scale);
fprintf('  W_motor_gain:              %.6e\n', best_params.W_motor_gain);
fprintf('  W_plan_gain:               %.6e\n', best_params.W_plan_gain);
fprintf('  interference_penalty_weight: %.6e\n', best_params.interference_penalty_weight);

fprintf('\n════════════════════════════════════════════════════════════════\n');
fprintf('✓ Analysis complete!\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('Next steps:\n');
fprintf('  1. Run load_best_pso_and_run.m to test best parameters\n');
fprintf('  2. Or manually use best_params from pso_best_params.mat\n');
fprintf('  3. Access top20 from figures/pso_top20_best_params.mat\n\n');