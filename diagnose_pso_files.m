%% Diagnose available PSO results files
clear all; close all; clc;

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PSO FILES DIAGNOSTIC\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Check figures directory
fig_dir = './figures';
if ~exist(fig_dir, 'dir')
    fprintf('❌ Figures directory not found: %s\n', fig_dir);
    return;
end

% Find all PSO-related files
pso_patterns = {
    'pso_*.mat';
    'optimization_results_3D_PSO_*.mat';
    '3D_dual_hierarchy_results_*.mat';
    '*results*.mat';
};

all_files = {};
for p_idx = 1:numel(pso_patterns)
    pattern = fullfile(fig_dir, pso_patterns{p_idx});
    found = dir(pattern);
    all_files = [all_files; {found.name}'];
end

% Remove duplicates
all_files = unique(all_files);

fprintf('Found %d potential PSO/results files:\n', numel(all_files));
fprintf('─────────────────────────────────────────────────────────────\n\n');

for f_idx = 1:numel(all_files)
    fname = all_files{f_idx};
    fpath = fullfile(fig_dir, fname);
    finfo = dir(fpath);
    
    fprintf('File %d: %s\n', f_idx, fname);
    fprintf('  Size: %.2f MB\n', finfo.bytes / 1e6);
    fprintf('  Date: %s\n', datestr(finfo.datenum));
    
    % Load and inspect structure
    try
        loaded = load(fpath);
        vars = fieldnames(loaded);
        fprintf('  Contents (%d fields):\n', numel(vars));
        for v_idx = 1:min(10, numel(vars))
            var = vars{v_idx};
            val = loaded.(var);
            if isstruct(val)
                fprintf('    - %s (struct, %d elements)\n', var, numel(val));
                % Check if it has params-like fields
                if numel(val) > 0
                    val1 = val(1);
                    fields = fieldnames(val1);
                    if numel(fields) <= 20
                        fprintf('      Fields: %s\n', strjoin(fields(1:min(5, end)), ', '));
                    else
                        fprintf('      Fields: %s [+%d more]\n', ...
                            strjoin(fields(1:5), ', '), numel(fields) - 5);
                    end
                end
            elseif iscell(val)
                fprintf('    - %s (cell, %s)\n', var, mat2str(size(val)));
            elseif isnumeric(val)
                fprintf('    - %s (numeric, %s)\n', var, mat2str(size(val)));
            else
                fprintf('    - %s (%s)\n', var, class(val));
            end
        end
    catch ME
        fprintf('  ❌ Error loading: %s\n', ME.message);
    end
    fprintf('\n');
end

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('RECOMMENDATION:\n');
fprintf('Run: analyze_optimization.m to generate pso_top20_best_params.mat\n');
fprintf('Then: load_best_pso_and_run()\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
