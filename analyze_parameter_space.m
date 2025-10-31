function analyze_parameter_space(matfile)
% analyze_parameter_space Export results.top20 params and scores to CSV.
%   This simplified function expects the MAT file to contain a struct
%   `results` with field `top20`, where each element has fields `score`
%   and `params` (a struct of parameter values). It writes a CSV with one
%   row per top20 entry and columns for each parameter plus `score`.
%
%   Usage:
%     analyze_parameter_space; % uses default filename
%     analyze_parameter_space('my_results.mat');

if nargin < 1 || isempty(matfile)
    matfile = 'optimization_results_3D_PSO_2025-10-31_01-06-54.mat';
end

if ~isfile(matfile)
    error('File not found: %s', matfile);
end

S = load(matfile);
if ~isfield(S, 'results') || ~isstruct(S.results) || ~isfield(S.results, 'top20')
    error('MAT file does not contain results.top20. Expected structure: results.top20(i).params and .score');
end

T = S.results.top20;
if isempty(T)
    error('results.top20 is empty. Nothing to export.');
end

m = numel(T);
% collect parameter fieldnames from the first element
firstParams = T(1).params;
if ~isstruct(firstParams)
    error('results.top20(1).params is not a struct');
end
pfields = fieldnames(firstParams);
np = numel(pfields);

% build numeric/table-friendly cell array
data = cell(m, np + 1); % params..., score
for i = 1:m
    pi = T(i).params;
    for j = 1:np
        fld = pfields{j};
        if isfield(pi, fld)
            v = pi.(fld);
            % convert numeric/scalar/logical to value; else store as string
            if isnumeric(v) && isscalar(v)
                data{i,j} = v;
            elseif islogical(v) && isscalar(v)
                data{i,j} = double(v);
            else
                try
                    data{i,j} = v; % allow non-numeric; writetable will handle
                catch
                    data{i,j} = mat2str(v);
                end
            end
        else
            data{i,j} = NaN;
        end
    end
    % score
    if isfield(T(i), 'score')
        data{i, np+1} = T(i).score;
    else
        data{i, np+1} = NaN;
    end
end

% Create table and write CSV
colnames = [pfields; {'score'}];
try
    % convert to table; ensure variable names are valid
    Ttab = cell2table(data, 'VariableNames', matlab.lang.makeValidName(colnames));
catch
    % fallback: build table column-by-column
    Ttab = table();
    for j = 1:np
        col = data(:,j);
        Ttab.(matlab.lang.makeValidName(pfields{j})) = col;
    end
    Ttab.score = cell2mat(data(:,np+1));
end

out_dir = './figures'; if ~exist(out_dir, 'dir'), mkdir(out_dir); end
outfn = fullfile(out_dir, 'top20_params_scores.csv');
try
    writetable(Ttab, outfn);
    fprintf('Wrote results.top20 to %s (rows=%d, cols=%d)\n', outfn, m, width(Ttab));
    % --- Post-export analysis: find two params with highest variance and plot surf(score) ---
    try
        vn = Ttab.Properties.VariableNames;
        scoreName = 'score';
        if ~ismember(scoreName, vn)
            idxs = find(strcmpi(vn, 'score'));
            if isempty(idxs)
                fprintf('No score column found in table; skipping surface plot.\n');
            else
                scoreName = vn{idxs(1)};
            end
        end
        paramNames = setdiff(vn, scoreName, 'stable');
        D = nan(height(Ttab), numel(paramNames));
        validCol = false(1, numel(paramNames));
        for ci = 1:numel(paramNames)
            col = Ttab.(paramNames{ci});
            if isnumeric(col)
                v = double(col);
            elseif iscell(col)
                if all(cellfun(@(c) isnumeric(c) && isscalar(c), col))
                    v = cell2mat(col);
                else
                    s = cellfun(@(c) (ischar(c) || isstring(c)), col);
                    if all(s)
                        v = str2double(cellfun(@char, col, 'UniformOutput', false));
                    else
                        v = nan(size(col));
                    end
                end
            elseif isstring(col)
                v = str2double(col);
            else
                v = nan(size(col));
            end
            if isvector(v) && numel(v)==height(Ttab)
                D(:,ci) = double(v(:));
                validCol(ci) = any(isfinite(D(:,ci)));
            else
                D(:,ci) = nan(height(Ttab),1);
                validCol(ci) = false;
            end
        end
        if sum(validCol) < 2
            fprintf('Not enough numeric parameter columns (%d) to build surface plot.\n', sum(validCol));
        else
            idxMap = find(validCol);
            vars = nan(1, numel(idxMap));
            for k = 1:numel(idxMap)
                colv = D(:, idxMap(k));
                vars(k) = nanvar(colv);
            end
            [~, sidx] = sort(vars, 'descend');
            topidx = idxMap(sidx(1:2));
            nameA = paramNames{topidx(1)}; nameB = paramNames{topidx(2)};
            Xs = D(:, topidx(1)); Ys = D(:, topidx(2)); Zs = Ttab.(scoreName);
            ok = isfinite(Xs) & isfinite(Ys) & isfinite(Zs);
            if sum(ok) < 6
                fprintf('Too few finite samples (%d) for interpolation plot.\n', sum(ok));
            else
                nx = 120; ny = 120;
                xi = linspace(min(Xs(ok)), max(Xs(ok)), nx);
                yi = linspace(min(Ys(ok)), max(Ys(ok)), ny);
                [Xi, Yi] = meshgrid(xi, yi);
                try
                    F = scatteredInterpolant(Xs(ok), Ys(ok), Zs(ok), 'natural', 'none');
                    Zi = F(Xi, Yi);
                    if all(isnan(Zi),'all')
                        Zi = griddata(Xs(ok), Ys(ok), Zs(ok), Xi, Yi, 'linear');
                    end
                catch
                    Zi = griddata(Xs(ok), Ys(ok), Zs(ok), Xi, Yi, 'linear');
                end
                if any(isnan(Zi),'all')
                    nanidx = isnan(Zi);
                    Zi(nanidx) = griddata(Xs(ok), Ys(ok), Zs(ok), Xi(nanidx), Yi(nanidx), 'nearest');
                end
                h = figure('Name','Top-variance parameter surface','NumberTitle','off');
                surf(Xi, Yi, Zi, 'EdgeColor','none'); hold on;
                scatter3(Xs(ok), Ys(ok), Zs(ok), 36, 'k', 'filled');
                xlabel(nameA, 'Interpreter', 'none'); ylabel(nameB, 'Interpreter', 'none'); zlabel('score');
                title(sprintf('Score surface: %s vs %s (by variance)', nameA, nameB), 'Interpreter', 'none');
                colorbar; shading interp; view(45,30); grid on;
                safeA = matlab.lang.makeValidName(nameA); safeB = matlab.lang.makeValidName(nameB);
                outsurf = fullfile(out_dir, sprintf('top20_score_surface_%s_%s.png', safeA, safeB));
                try
                    saveas(h, outsurf);
                    fprintf('Saved surface plot to %s\n', outsurf);
                catch
                    fprintf('Could not save surface plot.\n');
                end
            end
        end
    catch ME
        fprintf('Post-export analysis failed: %s\n', ME.message);
    end
catch ME
    error('Failed to write CSV: %s', ME.message);
end
end