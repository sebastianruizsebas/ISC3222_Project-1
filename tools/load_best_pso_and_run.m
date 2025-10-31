function results = load_best_pso_and_run(leaderboard_file, idx, save_results)
%LOAD_BEST_PSO_AND_RUN Load top PSO parameters and run the main experiment
%
%   results = load_best_pso_and_run()
%   results = load_best_pso_and_run(leaderboard_file, idx, save_results)
%
% Defaults:
%   leaderboard_file = './figures/pso_top20_best_params.mat'
%   idx = 1
%   save_results = true
%
% This helper loads the saved leaderboard (expects a variable `leader_list`)
% and extracts the parameter struct from the top entry, then calls
% `hierarchical_motion_inference_dual_hierarchy(params, true)` so you can
% visually inspect or validate the best parameter set.

% Resolve project root from the script location so helper works regardless
% of the current working directory (for example, when user does `cd tools`).
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

if nargin < 1 || isempty(leaderboard_file)
    leaderboard_file = fullfile(project_root, './tools/figures', 'pso_top20_best_params.mat');
end

if nargin < 2 || isempty(idx)
    idx = 1;
end
if nargin < 3 || isempty(save_results)
    save_results = true;
end

if ~exist(leaderboard_file, 'file')
    alt = fullfile(project_root, 'figures', 'pso_best_params.mat');
    if exist(alt, 'file')
        leaderboard_file = alt;
        fprintf('Using fallback leaderboard file: %s\n', leaderboard_file);
    else
        error('Leaderboard file not found: %s (also tried %s)', leaderboard_file, alt);
    end
end

loaded = load(leaderboard_file);

% Expect either leader_list or top-level struct
if isfield(loaded, 'leader_list')
    leader_list = loaded.leader_list;
elseif isfield(loaded, 'results') && isfield(loaded.results, 'leader_list')
    leader_list = loaded.results.leader_list;
else
    % try to find a struct array in the file
    vars = fieldnames(loaded);
    leader_list = [];
    for k = 1:numel(vars)
        v = loaded.(vars{k});
        if isstruct(v) && numel(v) > 0 && isfield(v, 'params')
            leader_list = v;
            break
        end
    end
    if isempty(leader_list)
        error('Could not find a leaderboard structure with field ''params'' in %s', leaderboard_file);
    end
end

if idx < 1 || idx > numel(leader_list)
    error('Index idx (=%d) is out of range [1..%d]', idx, numel(leader_list));
end

entry = leader_list(idx);

if isfield(entry, 'params')
    params = entry.params;
elseif isfield(entry, 'particle')
    % older naming convention
    params = entry.particle;
else
    error('Leaderboard entry does not contain a params or particle field.');
end

% Ensure save_results flag is respected
params.save_results = logical(save_results);

% Clear PSO-specific bookkeeping if present to avoid confusing the main function
if isfield(params, 'pso_iter'), params = rmfield(params, 'pso_iter'); end
if isfield(params, 'pso_iter_total'), params = rmfield(params, 'pso_iter_total'); end
if isfield(params, 'particle_num'), params = rmfield(params, 'particle_num'); end

fprintf('Running hierarchical_motion_inference_dual_hierarchy with leaderboard entry %d/%d\n', idx, numel(leader_list));
disp(params);

try
    results = hierarchical_motion_inference_dual_hierarchy(params, true);
catch ME
    fprintf('Error running hierarchical_motion_inference_dual_hierarchy: %s\n', ME.message);
    rethrow(ME);
end

% Optionally save the results in figures/
outdir = fullfile(project_root, 'figures');
if save_results
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    % create a timestamped base name and ensure uniqueness by appending
    % a run index suffix (_run2, _run3, ...) if a file with the same
    % timestamped name already exists
    ts = datestr(now,'yyyymmdd_HHMM');
    base = sprintf('run_best_pso_idx%d_%s', idx, ts);
    outname = fullfile(outdir, [base '.mat']);
    run_idx = 1;
    while exist(outname, 'file')
        run_idx = run_idx + 1; % start at 2 when first conflict
        outname = fullfile(outdir, sprintf('%s_run%d.mat', base, run_idx));
    end
    save(outname, 'params', 'results');
    fprintf('Saved run results to %s\n', outname);
end

% --- Create a visualization-friendly MAT file if common fields exist ---
% This helps scripts like visualize_3d_reaching.m which expect top-level
% variables (phases_indices, x_true, R_L1, targets, trial_start_positions, ...)
try
    res = results; %#ok<NASGU>
    vis = struct();

    % Map of canonical variable names -> candidate aliases to search for in results
    alias_map = struct();
    alias_map.phases_indices = {{'phases_indices','phase_indices','phases_idx','phase_idx','phasesIndices'}};
    alias_map.x_true = {{'x_true','x_player','x_ball','x_true_all','x'}};
    alias_map.y_true = {{'y_true','y_player','y_ball','y_true_all','y'}};
    alias_map.z_true = {{'z_true','z_player','z_ball','z_true_all','z'}};
    alias_map.vx_true = {{'vx_true','vx_player','vx_ball','vx_true_all','vx'}};
    alias_map.vy_true = {{'vy_true','vy_player','vy_ball','vy_true_all','vy'}};
    alias_map.vz_true = {{'vz_true','vz_player','vz_ball','vz_true_all','vz'}};
    alias_map.R_L1 = {{'R_L1','R_L1_all','R_L1_pred','R_L1_predictions','R_L1s','R_L1_matrix'}};
    alias_map.targets = {{'targets','target_positions','target','targets_all'}};
    alias_map.trial_start_positions = {{'trial_start_positions','start_positions','trial_starts','trial_start_pos','initial_positions'}};
    alias_map.reaching_error_all = {{'reaching_error_all','interception_error_all','reaching_error','reaching_error_all_trials'}};
    alias_map.free_energy_all = {{'free_energy_all','free_energy','F_all'}};

    added = {};
    % For each canonical name, look for the first alias that exists in results
    keys = fieldnames(alias_map);
    for ki = 1:numel(keys)
        key = keys{ki};
        candidates = alias_map.(key){1};
        found = false;
        for ci = 1:numel(candidates)
            cand = candidates{ci};
            if isfield(results, cand)
                vis.(key) = results.(cand);
                added{end+1} = key; %#ok<AGROW>
                found = true;
                break;
            end
        end
        % Also check for top-level presence (in case some code saved top-level vars)
        if ~found
            if exist('loaded','var') && isfield(loaded, key)
                vis.(key) = loaded.(key);
                added{end+1} = key; %#ok<AGROW>
                found = true;
            end
        end
    end

    % If we found at least one of the expected variables, save a compact
    % visualization MAT so visualization scripts can load the variables
    if ~isempty(added)
        % unique vis MAT name using same timestamp + optional run suffix
        ts_vis = datestr(now,'yyyymmdd_HHMM');
        base_vis = sprintf('run_best_pso_idx%d_%s_vis', idx, ts_vis);
        visname = fullfile(outdir, [base_vis '.mat']);
        run_idx_vis = 1;
        while exist(visname, 'file')
            run_idx_vis = run_idx_vis + 1;
            visname = fullfile(outdir, sprintf('%s_run%d.mat', base_vis, run_idx_vis));
        end
        save(visname, '-struct', 'vis');
        fprintf('Saved visualization MAT with fields: %s -> %s\n', strjoin(added, ', '), visname);
    else
        fprintf('No common visualization fields found in results; no vis MAT created.\n');
    end
catch ME
    fprintf('Warning: could not create visualization MAT: %s\n', ME.message);
end

end
