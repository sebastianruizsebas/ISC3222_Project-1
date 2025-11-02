%% ANALYZE PSO OPTIMIZATION RESULTS
% Reads the latest PSO results file and extracts/creates top20 leaderboard

clear all; close all; clc;

fprintf('════════════════════════════════════════════════════════════════\n');
fprintf('PSO RESULTS ANALYZER\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

% Find the latest optimization_results_3D_PSO_*.mat file
pso_files = dir('./optimization_results_3D_PSO_2025-11-02_02-45-43.mat');
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
load(pso_filepath, 'results');

fprintf('✓ PSO results loaded\n');
fprintf('  Best score: %.6f\n', results.best_score);
fprintf('  Total particles: %d\n', results.num_particles);
fprintf('  Total iterations: %d\n', results.num_iterations);
fprintf('  Total evaluations: %d\n\n', results.total_evaluations);

% ====================================================================
% Extract top20 leaderboard from particles
% ====================================================================
fprintf('Extracting top-20 best parameter sets...\n');

if ~isfield(results, 'particles')
    error('Results structure does not contain particles field.');
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

% Take top 20 valid scores
top_n = min(20, num_valid);
fprintf('  Building top-%d leaderboard...\n\n', top_n);

% Build leader_list structure
leader_list = struct('score', cell(top_n,1), 'params', cell(top_n,1), ...
                     'particle_id', cell(top_n,1), 'rank', cell(top_n,1));

for k = 1:top_n
    ip = idx(k);  % particle index
    
    % Create params structure with all 20 parameters
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

% ====================================================================
% Display Top-20 Leaderboard
% ====================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('TOP 20 PARAMETER SETS\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Rank │   Score   │ Particle │ eta_rep  │ eta_W    │ momentum\n');
fprintf('─────┼───────────┼──────────┼──────────┼──────────┼──────────\n');

for k = 1:top_n
    fprintf('%4d │ %9.6f │ %8d │ %.6e │ %.6e │ %.6e\n', ...
        k, leader_list(k).score, leader_list(k).particle_id, ...
        leader_list(k).params.eta_rep, leader_list(k).params.eta_W, ...
        leader_list(k).params.momentum);
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