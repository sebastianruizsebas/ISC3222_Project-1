function results = load_best_pso_and_run(pso_file, idx, save_results)
%LOAD_BEST_PSO_AND_RUN Load top PSO parameters and run the main experiment
%
%   results = load_best_pso_and_run()
%   results = load_best_pso_and_run(pso_file, idx, save_results)
%
% Defaults:
%   pso_file = './figures/optimization_results_3D_PSO_*.mat'
%   idx = 1 (best particle)
%   save_results = true
%
% This helper loads the PSO optimization results, extracts the best
% parameters, and runs the main dual-hierarchy model.

% Resolve project root from the script location
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

if nargin < 1 || isempty(pso_file)
    pso_file = fullfile(project_root, 'tools/figures/pso_best_params.mat');
end

if nargin < 2 || isempty(idx)
    idx = 1;
end
if nargin < 3 || isempty(save_results)
    save_results = true;
end

% ====================================================================
% FIND AND LOAD PSO RESULTS FILE
% ====================================================================

candidate_files = {
    pso_file;
    fullfile(project_root, 'figures', 'optimization_results_3D_PSO_*.mat');
    fullfile(project_root, 'figures', '3D_dual_hierarchy_results_best.mat');
    fullfile(project_root, 'figures', 'pso_*.mat');
};

loaded = struct();
loaded_file = '';

for cf_idx = 1:numel(candidate_files)
    cf = candidate_files{cf_idx};
    
    % Handle wildcards
    if contains(cf, '*')
        dir_result = dir(cf);
        if ~isempty(dir_result)
            % Sort by date, take most recent
            [~, newest_idx] = max([dir_result.datenum]);
            cf = fullfile(fileparts(cf), dir_result(newest_idx).name);
        else
            continue;
        end
    end
    
    if exist(cf, 'file')
        try
            loaded = load(cf);
            loaded_file = cf;
            fprintf('Successfully loaded PSO results: %s\n', cf);
            break;
        catch
            continue;
        end
    end
end

if isempty(loaded_file)
    error('Could not find PSO results file. Tried: %s', strjoin(candidate_files, ', '));
end

% ====================================================================
% EXTRACT BEST PARAMETERS FROM PSO RESULTS
% ====================================================================

params = struct();

% Check for leaderboard structure (results.top20)
if isfield(loaded, 'results') && isstruct(loaded.results)
    if isfield(loaded.results, 'top20')
        leader_list = loaded.results.top20;
        fprintf('Found %d particles in leaderboard\n', numel(leader_list));
        
        if idx < 1 || idx > numel(leader_list)
            error('Index idx=%d is out of range [1..%d]', idx, numel(leader_list));
        end
        
        entry = leader_list(idx);
        
        % Extract params from entry
        if isfield(entry, 'params')
            params = entry.params;
            fprintf('Extracted parameters from leaderboard entry %d\n', idx);
        elseif isfield(entry, 'particle')
            params = entry.particle;
            fprintf('Extracted parameters from particle field\n');
        else
            % Try to use the entry itself as parameters
            params = entry;
        end
    end
end

% Handle top-level leader_list files (e.g. figures/pso_top20_best_params.mat)
if isempty(fieldnames(params)) && isfield(loaded, 'leader_list')
    leader_list = loaded.leader_list;
    fprintf('Found leader_list in loaded file with %d entries\n', numel(leader_list));
    if idx < 1 || idx > numel(leader_list)
        error('Index idx=%d is out of range [1..%d] for leader_list', idx, numel(leader_list));
    end
    entry = leader_list(idx);
    if isfield(entry, 'params')
        params = entry.params;
        fprintf('Extracted parameters from leader_list(%d).params\n', idx);
    elseif isfield(entry, 'particle')
        params = entry.particle;
        fprintf('Extracted parameters from leader_list(%d).particle\n', idx);
    else
        % If entry directly contains parameters, try that
        % e.g., entry may be the params struct with a score field
        if isstruct(entry) && isfield(entry, 'score') && isfield(entry, 'params')
            params = entry.params;
        else
            params = entry;
        end
        fprintf('Extracted parameters from leader_list entry %d (fallback)\n', idx);
    end
end

% Fallback: Check for global best_particle or best_params
if isempty(fieldnames(params))
    if isfield(loaded, 'best_particle')
        params = loaded.best_particle;
        fprintf('Extracted best_particle\n');
    elseif isfield(loaded, 'best_params')
        params = loaded.best_params;
        fprintf('Extracted best_params\n');
    elseif isfield(loaded, 'global_best')
        params = loaded.global_best;
        fprintf('Extracted global_best\n');
    end
end

% Last resort: scan for any struct that looks like parameters
if isempty(fieldnames(params))
    fprintf('Warning: Could not find parameters in standard locations, scanning file...\n');
    vars = fieldnames(loaded);
    for k = 1:numel(vars)
        v = loaded.(vars{k});
        if isstruct(v) && numel(v) > 0
            % Check if this looks like a parameter set (has eta_rep, eta_W, etc.)
            if isfield(v(1), 'eta_rep') || isfield(v(1), 'eta_W') || isfield(v(1), 'momentum')
                params = v(1);
                fprintf('Found parameters in field: %s\n', vars{k});
                break;
            end
        end
    end
end

if isempty(fieldnames(params))
    error(['Could not extract parameters from %s.\n' ...
        'Expected structure: results.top20[idx].params or best_particle'], loaded_file);
end

% ====================================================================
% CLEAN UP PARAMETERS (REMOVE TRACE DATA, KEEP ONLY LEARNING PARAMS)
% ====================================================================

fprintf('\nCleaning parameters...\n');

% List of PSO metadata fields to remove
pso_metadata = {
    'score', 'best_score', 'fitness', 'velocity', 'best_position', ...
    'pso_iter', 'pso_iter_total', 'particle_num', 'iteration'
};

for mi = 1:numel(pso_metadata)
    if isfield(params, pso_metadata{mi})
        params = rmfield(params, pso_metadata{mi});
    end
end

% List of essential learning parameters to KEEP
essential_params = {
    'eta_rep', 'eta_W', 'momentum', 'decay_motor', 'decay_plan', ...
    'motor_gain', 'damping', 'reaching_speed_scale', 'W_plan_gain', 'W_motor_gain', ...
    'weight_decay', 'interference_penalty_weight', ...
    'dt', 'gravity', 'restitution', 'ground_friction', 'air_drag', ...
    'T_per_trial', 'n_trials', 'termination_distance'
};

params_clean = struct();
for pi = 1:numel(essential_params)
    pname = essential_params{pi};
    if isfield(params, pname)
        params_clean.(pname) = params.(pname);
    end
end

% Check how many parameters we extracted
extracted_count = numel(fieldnames(params_clean));
fprintf('Extracted %d essential parameters\n', extracted_count);

if extracted_count == 0
    fprintf('Warning: No essential parameters found, using all available parameters\n');
    params_clean = params;
end

params = params_clean;

% Ensure save_results flag
params.save_results = logical(save_results);

% ====================================================================
% DISPLAY PARAMETERS
% ====================================================================

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  Running Best PSO Parameters (Leaderboard Entry %d)        ║\n', idx);
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

fprintf('Extracted Parameters:\n');
fprintf('─────────────────────────────────────────────────────────────\n');
param_names = fieldnames(params);
for pi = 1:numel(param_names)
    pname = param_names{pi};
    pval = params.(pname);
    if isnumeric(pval)
        fprintf('  %-35s = %.8g\n', pname, pval);
    else
        fprintf('  %-35s = %s\n', pname, class(pval));
    end
end
fprintf('─────────────────────────────────────────────────────────────\n\n');

% ====================================================================
% RUN MAIN MODEL WITH EXTRACTED PARAMETERS
% ====================================================================

try
    fprintf('Running hierarchical_motion_inference_dual_hierarchy...\n\n');
    results = hierarchical_motion_inference_dual_hierarchy(params, true);
catch ME
    fprintf('\n❌ Error running main model:\n%s\n', ME.message);
    rethrow(ME);
end

% ====================================================================
% SAVE RESULTS
% ====================================================================

outdir = fullfile(project_root, 'figures');
if save_results
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    
    ts = datestr(now, 'yyyymmdd_HHMM');
    base = sprintf('run_best_pso_%s', ts);
    outname = fullfile(outdir, [base '.mat']);
    run_idx = 1;
    
    while exist(outname, 'file')
        run_idx = run_idx + 1;
        outname = fullfile(outdir, sprintf('%s_run%d.mat', base, run_idx));
    end
    
    save(outname, 'params', 'results');
    fprintf('\n✓ Saved run results to:\n  %s\n', outname);
end

% ====================================================================
% CREATE VISUALIZATION MAT FILE (with key traces)
% ====================================================================

try
    vis = struct();
    
    % Map canonical variable names to candidates
    alias_map = struct();
    alias_map.phases_indices = {{'phases_indices', 'phase_indices'}};
    alias_map.x_player = {{'x_player', 'x_true'}};
    alias_map.y_player = {{'y_player', 'y_true'}};
    alias_map.z_player = {{'z_player', 'z_true'}};
    alias_map.R_L1_motor = {{'R_L1_motor', 'R_L1'}};
    alias_map.R_L2_motor = {{'R_L2_motor', 'R_L2'}};
    alias_map.R_L0 = {{'R_L0'}};
    alias_map.interception_error_all = {{'interception_error_all', 'reaching_error_all'}};
    alias_map.free_energy_all = {{'free_energy_all'}};
    alias_map.learning_trace_W = {{'learning_trace_W'}};
    
    added = {};
    keys = fieldnames(alias_map);
    
    for ki = 1:numel(keys)
        key = keys{ki};
        candidates = alias_map.(key){1};
        
        for ci = 1:numel(candidates)
            cand = candidates{ci};
            if isfield(results, cand)
                vis.(key) = results.(cand);
                added{end+1} = key; %#ok<AGROW>
                break;
            end
        end
    end
    
    if ~isempty(added)
        ts_vis = datestr(now, 'yyyymmdd_HHMM');
        base_vis = sprintf('run_best_pso_%s_vis', ts_vis);
        visname = fullfile(outdir, [base_vis '.mat']);
        run_idx_vis = 1;
        
        while exist(visname, 'file')
            run_idx_vis = run_idx_vis + 1;
            visname = fullfile(outdir, sprintf('%s_run%d.mat', base_vis, run_idx_vis));
        end
        
        save(visname, '-struct', 'vis');
        fprintf('✓ Saved visualization MAT with %d fields:\n  %s\n', numel(added), visname);
    end
    
catch ME
    fprintf('Warning: Could not create visualization MAT: %s\n', ME.message);
end

fprintf('\n✓ Completed successfully!\n\n');

end
