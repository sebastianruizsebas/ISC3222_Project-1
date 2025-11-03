%% ANALYZE PARAMETER VARIANCE FROM PSO TOP-N
% Loads a saved top-N PSO leaderboard file (e.g. ./figures/pso_top110_best_params.mat)
% Extracts `leader_list(k).params` for k=1..N, computes mean/std/median and
% generates distribution plots (histogram + boxplot) for each parameter.

clearvars -except TOP_N; close all; clc;

out_dir = './figures';
top_fname = fullfile(out_dir, 'pso_top110_best_params.mat');
if ~exist(top_fname, 'file')
    error('Expected file not found: %s\nPlease run analyze_optimization.m (which saves top-N) first.', top_fname);
end

S = load(top_fname);

% Leader list can be stored either at top-level or inside a struct named results
if isfield(S, 'leader_list')
    leader_list = S.leader_list;
elseif isfield(S, 'results') && isfield(S.results, 'leader_list')
    leader_list = S.results.leader_list;
else
    error('Loaded file does not contain leader_list. Found fields: %s', strjoin(fieldnames(S), ', '));
end

N = numel(leader_list);
if N == 0
    error('leader_list is empty in %s', top_fname);
end

% Collect parameter names from first non-empty params struct
first_params = leader_list(1).params;
param_names = fieldnames(first_params);
n_params = numel(param_names);
fprintf('Found %d parameters in leader_list.params; top-N = %d\n', n_params, N);

% Preallocate values matrix (N x n_params)
vals = nan(N, n_params);

for k = 1:N
    if ~isfield(leader_list(k), 'params') || isempty(leader_list(k).params)
        warning('leader_list(%d) missing params - filling with NaN', k);
        continue;
    end
    p = leader_list(k).params;
    for j = 1:n_params
        name = param_names{j};
        if isfield(p, name) && ~isempty(p.(name))
            vals(k,j) = double(p.(name));
        else
            vals(k,j) = NaN;
        end
    end
end

% Compute statistics (ignore NaNs)
stats = struct();
for j = 1:n_params
    col = vals(:,j);
    stats.(param_names{j}).mean = mean(col(~isnan(col)));
    stats.(param_names{j}).std = std(col(~isnan(col)));
    stats.(param_names{j}).median = median(col(~isnan(col)));
    stats.(param_names{j}).min = min(col(~isnan(col)));
    stats.(param_names{j}).max = max(col(~isnan(col)));
    stats.(param_names{j}).n_nonan = sum(~isnan(col));
end

% Print a compact table
fprintf('\nPARAMETER STATISTICS (top-%d):\n', N);
fprintf(' %-28s %10s %10s %10s %10s %6s\n', 'parameter', 'mean', 'std', 'median', 'min', 'n');
fprintf('%s\n', repmat('-',1,80));
for j = 1:n_params
    s = stats.(param_names{j});
    fprintf(' %-28s %10.4g %10.4g %10.4g %10.4g %6d\n', param_names{j}, s.mean, s.std, s.median, s.min, s.n_nonan);
end

% Create output directory for distributions
dist_dir = fullfile(out_dir, sprintf('pso_top%d_param_distributions', N));
if ~exist(dist_dir, 'dir'), mkdir(dist_dir); end

% Generate per-parameter distribution plots
for j = 1:n_params
    pname = param_names{j};
    col = vals(:,j);
    col_nonan = col(~isnan(col));

    fig = figure('Name', sprintf('Param: %s', pname), 'Visible', 'off');
    % Histogram
    ax1 = subplot(2,1,1);
    histogram(col_nonan, 'Normalization', 'pdf'); hold on;
    title(sprintf('%s (n=%d) - histogram', pname, numel(col_nonan)), 'Interpreter', 'none');
    xlabel(pname, 'Interpreter', 'none'); ylabel('Density'); grid on;
    % Add kernel density estimate if available
    try
        [f, xi] = ksdensity(col_nonan);
        plot(xi, f, 'r-', 'LineWidth', 1.5);
    catch
        % ignore
    end

    % Boxplot on second axis
    ax2 = subplot(2,1,2);
    boxplot(col_nonan, 'Notch','on');
    title(sprintf('%s - boxplot', pname), 'Interpreter', 'none');
    ylabel(pname, 'Interpreter', 'none');

    % Save figure
    fig_fname = fullfile(dist_dir, sprintf('%s_distribution.png', pname));
    try
        saveas(fig, fig_fname);
    catch ME
        warning('Failed to save figure for %s: %s', pname, ME.message);
    end
    close(fig);
end

% Save numeric stats and raw values for downstream analysis
save(fullfile(dist_dir, sprintf('pso_top%d_param_stats.mat', N)), 'param_names', 'vals', 'stats', 'N');

fprintf('\nSaved %d distribution plots and stats to: %s\n', n_params, dist_dir);
fprintf('Saved stats matfile: %s\n', fullfile(dist_dir, sprintf('pso_top%d_param_stats.mat', N)));

% Quick summary figure: multi-panel of histograms (arrange up to 4x5)
cols = 5; rows = ceil(n_params/cols);
fig = figure('Name', sprintf('Top-%d parameter distributions', N), 'Visible', 'off', 'Position',[100 100 min(1600, 320*cols) min(900, 200*rows)]);
for j = 1:n_params
    subplot(rows, cols, j);
    col_nonan = vals(~isnan(vals(:,j)), j);
    if isempty(col_nonan)
        text(0.5,0.5,'no data','HorizontalAlignment','center'); axis off; continue;
    end
    histogram(col_nonan); title(param_names{j}, 'Interpreter', 'none');
end
saveas(fig, fullfile(dist_dir, sprintf('pso_top%d_all_param_histograms.png', N)));
close(fig);

%% Pairwise plots: parameter value vs score
% Extract scores (robust to a few possible field names)
scores = nan(N,1);
for k = 1:N
    if isfield(leader_list(k), 'score') && ~isempty(leader_list(k).score)
        scores(k) = double(leader_list(k).score);
    elseif isfield(leader_list(k), 'obj') && ~isempty(leader_list(k).obj)
        scores(k) = double(leader_list(k).obj);
    elseif isfield(leader_list(k), 'cost') && ~isempty(leader_list(k).cost)
        scores(k) = double(leader_list(k).cost);
    elseif isfield(leader_list(k), 'fitness') && ~isempty(leader_list(k).fitness)
        scores(k) = double(leader_list(k).fitness);
    else
        scores(k) = NaN;
    end
end

% Per-parameter scatter plots of value vs score
corr_vals = nan(n_params,1);
for j = 1:n_params
    pname = param_names{j};
    x = vals(:,j);
    mask = ~isnan(x) & ~isnan(scores);
    xnon = x(mask);
    snon = scores(mask);

    fig = figure('Name', sprintf('%s vs score', pname), 'Visible', 'off');
    scatter(xnon, snon, 20, 'filled'); hold on;
    xlabel(pname, 'Interpreter', 'none'); ylabel('score'); grid on;
    title(sprintf('%s vs score (n=%d)', pname, numel(snon)), 'Interpreter', 'none');

    if numel(snon) > 2
        % Linear fit and overlay
        pfit = polyfit(xnon, snon, 1);
        xx = linspace(min(xnon), max(xnon), 200);
        plot(xx, polyval(pfit, xx), 'r-', 'LineWidth', 1.2);
        % Pearson correlation
        R = corrcoef(xnon, snon);
        corr_vals(j) = R(1,2);
        text(0.05, 0.95, sprintf('r=%.3g', corr_vals(j)), 'Units','normalized', 'VerticalAlignment','top');
    end

    fig_fname = fullfile(dist_dir, sprintf('%s_vs_score.png', pname));
    try
        saveas(fig, fig_fname);
    catch ME
        warning('Failed to save score-plot for %s: %s', pname, ME.message);
    end
    close(fig);
end

% Combined multipanel scatter figure
cols = 5; rows = ceil(n_params/cols);
fig = figure('Name', sprintf('Top-%d param vs score', N), 'Visible', 'off', 'Position', [100 100 min(1600,320*cols) min(900,200*rows)]);
for j = 1:n_params
    subplot(rows, cols, j);
    x = vals(:,j);
    mask = ~isnan(x) & ~isnan(scores);
    if sum(mask) < 2
        text(0.5,0.5,'no data','HorizontalAlignment','center'); axis off; continue;
    end
    scatter(x(mask), scores(mask), 10, 'filled');
    title(param_names{j}, 'Interpreter', 'none');
    if ~isnan(corr_vals(j))
        % annotate r
        text(0.02,0.95, sprintf('r=%.3g', corr_vals(j)), 'Units','normalized', 'VerticalAlignment','top');
    end
    xlabel(''); ylabel('');
end
saveas(fig, fullfile(dist_dir, sprintf('pso_top%d_all_param_vs_score.png', N)));
close(fig);

% Print ranked correlations
[~, idx] = sort(abs(corr_vals), 'descend', 'MissingPlacement','last');
fprintf('\nTop parameter correlations with score (abs r desc):\n');
fprintf(' %-28s %8s %8s\n', 'parameter','r','n');
fprintf('%s\n', repmat('-',1,50));
for ii = 1:min(20, n_params)
    j = idx(ii);
    if isnan(corr_vals(j)), continue; end
    mask = ~isnan(vals(:,j)) & ~isnan(scores);
    fprintf(' %-28s %8.4g %8d\n', param_names{j}, corr_vals(j), sum(mask));
end

fprintf('\nAll done.\n');
