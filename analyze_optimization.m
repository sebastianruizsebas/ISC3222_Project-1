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
leader_file = fullfile('./optimization_results_3D_PSO_2025-10-31_01-06-54.mat');
% Try to load the preferred PSO leaderboard file, but be defensive: the file
% may exist but not contain the expected variable name 'leader_list'. If so,
% fall back to the legacy-results search below.
if isfile(leader_file)
    fprintf('Found leaderboard file: %s\n', leader_file);
    % inspect variables inside the mat file first
    try
        vars_in_file = who('-file', leader_file);
    catch
        vars_in_file = {};
    end
    if ismember('leader_list', vars_in_file)
        S = load(leader_file, 'leader_list');
        leader_list = S.leader_list;
        use_leader_list = true;
        loaded_mat_filename = leader_file;
        fprintf('Loaded ''leader_list'' from leaderboard file.\n\n');
    else
        fprintf('Leaderboard file exists but does not contain ''leader_list''; falling back to legacy optimization results search.\n\n');
        use_leader_list = false;
    end
else
    fprintf('Leaderboard file not found, falling back to legacy optimization results in ./optimization_results/...\n');
    use_leader_list = false;
end

% If we didn't obtain a leader_list above, search the legacy results folder
if ~exist('use_leader_list','var') || ~use_leader_list
    result_files = dir('./optimization_results/rao_ballard_3D_optimization_*.mat');
    if isempty(result_files)
        error('No optimization results file or PSO leaderboard found. Please run PSO first.');
    end
    [~, latest_idx] = max([result_files.datenum]);
    results_filename = fullfile(result_files(latest_idx).folder, result_files(latest_idx).name);
    fprintf('Loading legacy optimization results from: %s\n\n', results_filename);
    load(results_filename, 'results'); % Loads the 'results' struct
    loaded_mat_filename = results_filename;
end

% --- Debug: print contents of the loaded MAT file (top-level fields & summaries) ---
if exist('loaded_mat_filename','var') && isfile(loaded_mat_filename)
    try
        fprintf('\n--- Loaded MAT file summary: %s ---\n', loaded_mat_filename);
        fullS = load(loaded_mat_filename);
        fns = fieldnames(fullS);
        for ii = 1:numel(fns)
            fname = fns{ii};
            val = fullS.(fname);
            cls = class(val);
            sz = mat2str(size(val));
            fprintf(' * %s : class=%s size=%s\n', fname, cls, sz);
            % Print small numeric arrays; otherwise print a short summary
            try
                if isnumeric(val) || islogical(val)
                    cnt = numel(val);
                    if cnt == 0
                        fprintf('   (empty)\n');
                    elseif cnt <= 20
                        fprintf('   values: ');
                        disp(val);
                    else
                        vv = val(isfinite(val));
                        if isempty(vv)
                            fprintf('   numeric, but all NaN/Inf or non-finite\n');
                        else
                            fprintf('   summary: min=%.6g, max=%.6g, mean=%.6g, n=%d\n', min(vv), max(vv), mean(vv), numel(vv));
                        end
                    end
                elseif ischar(val)
                    s = val(:)';
                    s = s(1:min(numel(s),200));
                    fprintf('   char: "%s"\n', s);
                elseif isstruct(val)
                    subf = fieldnames(val);
                    fprintf('   struct with %d element(s); fields: %s\n', numel(val), strjoin(subf', ', '));
                    % show first element's small fields
                    if numel(val) > 0
                        fn2 = fieldnames(val(1));
                        for jj = 1:min(6,numel(fn2))
                            try
                                v2 = val(1).(fn2{jj});
                                if isnumeric(v2) && numel(v2) <= 10
                                    fprintf('     - %s: %s\n', fn2{jj}, mat2str(v2));
                                end
                            catch
                            end
                        end
                    end
                else
                    fprintf('   (class %s not expanded)\n', cls);
                end
            catch
                fprintf('   (error printing this field)\n');
            end
        end
        fprintf('--- end of MAT summary ---\n\n');
    catch ME
        fprintf('Could not print MAT file contents: %s\n', ME.message);
    end
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

% -----------------------------
% Statistical influence analysis
% -----------------------------
% Build a unified list of entries (leader_list or legacy results.particles)
if exist('leader_list','var') && ~isempty(leader_list)
    entries = leader_list;
elseif exist('results','var') && isfield(results, 'particles')
    % Construct entries from particles' personal bests
    np = numel(results.particles);
    entries = repmat(struct('score', [], 'params', struct()), np, 1);
    for k = 1:np
        p = results.particles(k);
        entries(k).score = NaN;
        if isfield(p, 'best_score'), entries(k).score = p.best_score; end
        params = struct();
        if isfield(p, 'best_eta_rep'), params.eta_rep = p.best_eta_rep; end
        if isfield(p, 'best_eta_W'), params.eta_W = p.best_eta_W; end
        if isfield(p, 'best_momentum'), params.momentum = p.best_momentum; end
        % fallback names
        if isfield(p, 'best_decay_L2_goal'), params.decay_L2_goal = p.best_decay_L2_goal; end
        if isfield(p, 'best_decay_L1_motor'), params.decay_L1_motor = p.best_decay_L1_motor; end
        if isfield(p, 'best_motor_gain'), params.motor_gain = p.best_motor_gain; end
        if isfield(p, 'best_damping'), params.damping = p.best_damping; end
        if isfield(p, 'best_reaching_speed_scale'), params.reaching_speed_scale = p.best_reaching_speed_scale; end
        if isfield(p, 'best_W_L2_goal_gain'), params.W_L2_goal_gain = p.best_W_L2_goal_gain; end
        if isfield(p, 'best_W_L1_pos_gain'), params.W_L1_pos_gain = p.best_W_L1_pos_gain; end
        if isfield(p, 'best_weight_decay'), params.weight_decay = p.best_weight_decay; end
        entries(k).params = params;
    end
else
    error('No usable leaderboard or legacy particles found for statistical analysis.');
end

% Candidate parameter names to analyze (common set)
param_names = {'eta_rep','eta_W','momentum','decay_L2_goal','decay_L1_motor','motor_gain','damping','reaching_speed_scale','W_L2_goal_gain','W_L1_pos_gain','weight_decay'};
N = numel(entries);
M = numel(param_names);
X = nan(N, M);
Y = nan(N, 1);

% helper to try multiple possible field name variants
function v = get_field_safe(s, names)
    v = NaN;
    for ii = 1:numel(names)
        if isfield(s, names{ii})
            v = s.(names{ii}); return;
        end
    end
end

for i = 1:N
    if isfield(entries(i), 'score') && ~isempty(entries(i).score)
        Y(i) = entries(i).score;
    end
    p = entries(i).params;
    % try canonical mappings and fallbacks
    X(i,1) = get_field_safe(p, {'eta_rep','best_eta_rep'});
    X(i,2) = get_field_safe(p, {'eta_W','best_eta_W'});
    X(i,3) = get_field_safe(p, {'momentum','best_momentum'});
    X(i,4) = get_field_safe(p, {'decay_L2_goal','decay_L2','best_decay_L2_goal'});
    X(i,5) = get_field_safe(p, {'decay_L1_motor','decay_L1','best_decay_L1_motor'});
    X(i,6) = get_field_safe(p, {'motor_gain','best_motor_gain'});
    X(i,7) = get_field_safe(p, {'damping','best_damping'});
    X(i,8) = get_field_safe(p, {'reaching_speed_scale','reach_scale','best_reaching_speed_scale'});
    X(i,9) = get_field_safe(p, {'W_L2_goal_gain','W_L2','best_W_L2_goal_gain'});
    X(i,10)= get_field_safe(p, {'W_L1_pos_gain','W_L1','best_W_L1_pos_gain'});
    X(i,11)= get_field_safe(p, {'weight_decay','best_weight_decay'});
end

valid_rows = isfinite(Y);
if sum(valid_rows) < 3
    warning('Not enough valid scored entries (%d) for statistical analysis.', sum(valid_rows));
else
    % Compute Pearson correlation for each parameter vs score
    corrs = nan(M,1);
    for j = 1:M
        xv = X(valid_rows,j);
        yv = Y(valid_rows);
        ok = isfinite(xv) & isfinite(yv);
        if sum(ok) >= 3 && std(xv(ok)) > 0
            R = corrcoef(xv(ok), yv(ok)); corrs(j) = R(1,2);
        else
            corrs(j) = NaN;
        end
    end

    [~, idx_sorted] = sort(abs(corrs), 'descend', 'MissingPlacement', 'last');
    fprintf('\nParameter correlation with score (top 5):\n');
    for k = 1:min(5, M)
        j = idx_sorted(k);
        fprintf('  %2d) %-18s : corr = % .4f\n', k, param_names{j}, corrs(j));
    end

    % Pairwise linear model R^2 for all parameter pairs
    pairs = nchoosek(1:M,2);
    R2_pairs = nan(size(pairs,1),1);
    for pi = 1:size(pairs,1)
        a = pairs(pi,1); b = pairs(pi,2);
        xv = X(valid_rows,[a b]); yv = Y(valid_rows);
        ok = all(isfinite(xv),2) & isfinite(yv);
        if sum(ok) >= 4
            Xreg = [ones(sum(ok),1), xv(ok,1), xv(ok,2)];
            beta = Xreg \ yv(ok);
            yhat = Xreg * beta;
            ssres = sum((yv(ok) - yhat).^2);
            sst = sum((yv(ok) - mean(yv(ok))).^2);
            R2_pairs(pi) = 1 - ssres / max(sst, eps);
        else
            R2_pairs(pi) = NaN;
        end
    end

    [bestR2, bestIdx] = max(R2_pairs);
    if ~isfinite(bestR2)
        fprintf('\nPairwise R^2 analysis: insufficient data to evaluate pairs.\n');
    else
        best_pair = pairs(bestIdx,:);
        fprintf('\nTop parameter pair by linear R^2 (explained variance):\n');
        fprintf('  %s + %s -> R^2 = %.4f\n', param_names{best_pair(1)}, param_names{best_pair(2)}, bestR2);
        % Show top 5 pairs
        [sortedR2, sidx] = sort(R2_pairs, 'descend', 'MissingPlacement', 'last');
        fprintf('\nTop 5 parameter pairs by R^2:\n');
        for k = 1:min(5, numel(sortedR2))
            if ~isfinite(sortedR2(k)), break; end
            pr = pairs(sidx(k),:);
            fprintf('  %d) %s + %s : R^2 = %.4f\n', k, param_names{pr(1)}, param_names{pr(2)}, sortedR2(k));
        end
    end

    % Save influence summary for later inspection
    out_dir = './figures'; if ~exist(out_dir,'dir'), mkdir(out_dir); end
    save(fullfile(out_dir,'pso_parameter_influence.mat'), 'param_names', 'corrs', 'pairs', 'R2_pairs');
    fprintf('\nSaved parameter influence summary to ./figures/pso_parameter_influence.mat\n');
end

% --- Create a 3D surface visualization for the top parameter pair (score as height) ---
try
    if exist('best_pair','var') && exist('bestR2','var') && isfinite(bestR2)
        a = best_pair(1); b = best_pair(2);
        % Use only rows with finite score and parameter values
        vr = valid_rows;
        pa = X(vr,a); pb = X(vr,b); ps = Y(vr);
        ok = isfinite(pa) & isfinite(pb) & isfinite(ps);
        if sum(ok) >= 6
            % Build grid and interpolate scattered scores onto it
            na = max(20, ceil(sqrt(sum(ok))*2)); nb = na;
            xa = linspace(min(pa(ok)), max(pa(ok)), na);
            xb = linspace(min(pb(ok)), max(pb(ok)), nb);
            [XA, XB] = meshgrid(xa, xb);
            % Prefer scatteredInterpolant; fallback to griddata
            try
                F = scatteredInterpolant(pa(ok), pb(ok), ps(ok), 'natural', 'none');
                Z = F(XA, XB);
                if all(isnan(Z),'all')
                    Z = griddata(pa(ok), pb(ok), ps(ok), XA, XB, 'linear');
                end
            catch
                Z = griddata(pa(ok), pb(ok), ps(ok), XA, XB, 'linear');
            end

            % If interpolation produced NaNs in regions, try a smoother fill using nearest
            if any(isnan(Z),'all')
                Zn = Z;
                nanidx = isnan(Z);
                if any(nanidx,'all')
                    Zn(nanidx) = griddata(pa(ok), pb(ok), ps(ok), XA(nanidx), XB(nanidx), 'nearest');
                    Z = Zn;
                end
            end

            % Create figure and plot
            hFig = figure('Name','PSO score surface','NumberTitle','off');
            surf(XA, XB, Z, 'EdgeColor','none'); hold on;
            scatter3(pa(ok), pb(ok), ps(ok), 40, 'k', 'filled');
            xlabel(param_names{a}, 'Interpreter', 'none');
            ylabel(param_names{b}, 'Interpreter', 'none');
            zlabel('Score');
            title(sprintf('PSO score surface: %s vs %s (pair R^2 = %.3f)', param_names{a}, param_names{b}, bestR2), 'Interpreter', 'none');
            colorbar; view(45,30); grid on; shading interp;

            % Save to figures
            out_dir = './figures'; if ~exist(out_dir,'dir'), mkdir(out_dir); end
            safeA = matlab.lang.makeValidName(param_names{a});
            safeB = matlab.lang.makeValidName(param_names{b});
            outfn = fullfile(out_dir, sprintf('pso_score_surface_%s_%s.png', safeA, safeB));
            try
                saveas(hFig, outfn);
                fprintf('Saved pairwise surface plot to %s\n', outfn);
            catch
                fprintf('Could not save surface plot to %s (saveas failed)\n', outfn);
            end
        else
            fprintf('Not enough finite points (%d) to build surface for the top pair (%s + %s).\n', sum(ok), param_names{a}, param_names{b});
        end
    else
        fprintf('Top parameter pair or R^2 not available; skipping surface visualization.\n');
    end
catch ME
    fprintf('Error while creating pairwise surface plot: %s\n', ME.message);
end


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