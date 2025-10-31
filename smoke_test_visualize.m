function smoke_test_visualize(smoke_mat)
% SMOKE_TEST_VISUALIZE  Load and plot results saved by smoke_test_run
%
% smoke_test_visualize()                 % uses ./figures/smoke_test_results.mat
% smoke_test_visualize(path_to_mat)

if nargin < 1 || isempty(smoke_mat)
    smoke_mat = fullfile('.', 'figures', 'smoke_test_results.mat');
end

if ~exist(smoke_mat, 'file')
    error('Smoke-test MAT not found: %s\nRun smoke_test_run.m first.', smoke_mat);
end

fprintf('Loading smoke test results: %s\n', smoke_mat);
data = load(smoke_mat);
if isfield(data, 'results')
    results = data.results;
else
    % backwards-compatible: file might contain variables directly
    results = data;
end

% Defensive extraction with defaults
N = length(results.interception_error_all);
time = 1:N;

% Create figure
fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 900]);
cols = 3;

% 1) Interception error
subplot(3, cols, 1);
if isfield(results, 'interception_error_all')
    plot(time, results.interception_error_all, 'k-', 'LineWidth', 1.5);
    ylabel('Distance (m)'); xlabel('Step'); title('Interception Error'); grid on;
else
    text(0.1, 0.5, 'No interception_error_all in results', 'Units','normalized');
end

% 2) Free energy
subplot(3, cols, 2);
if isfield(results, 'free_energy_all')
    semilogy(time, results.free_energy_all, 'b-'); xlabel('Step'); title('Free Energy'); grid on;
else
    text(0.1, 0.5, 'No free_energy_all in results', 'Units','normalized');
end

% 3) Learning trace
subplot(3, cols, 3);
if isfield(results, 'learning_trace_W')
    semilogy(time, results.learning_trace_W + 1e-12, 'k-'); xlabel('Step'); title('Learning Trace (W updates)'); grid on;
else
    text(0.1, 0.5, 'No learning_trace_W in results', 'Units','normalized');
end

% 4-6) Player vs Ball positions (X/Y/Z)
subplot(3, cols, 4);
if isfield(results, 'x_ball') && isfield(results, 'x_player')
    plot(time, results.x_ball, 'b-', time, results.x_player, 'r--'); xlabel('Step'); title('X Position'); legend({'Ball','Player'}); grid on;
else
    text(0.1, 0.5, 'No X position data', 'Units','normalized');
end

subplot(3, cols, 5);
if isfield(results, 'y_ball') && isfield(results, 'y_player')
    plot(time, results.y_ball, 'b-', time, results.y_player, 'r--'); xlabel('Step'); title('Y Position'); grid on;
else
    text(0.1, 0.5, 'No Y position data', 'Units','normalized');
end

subplot(3, cols, 6);
if isfield(results, 'z_ball') && isfield(results, 'z_player')
    plot(time, results.z_ball, 'b-', time, results.z_player, 'r--'); xlabel('Step'); title('Z Position'); grid on;
else
    text(0.1, 0.5, 'No Z position data', 'Units','normalized');
end

% 7-9) π traces (if available)
subplot(3, cols, 7);
if isfield(results, 'pi_trace_L1_motor')
    plot(time, results.pi_trace_L1_motor, 'r-'); xlabel('Step'); title('π L1 Motor'); grid on;
else
    text(0.1, 0.5, 'No π L1 Motor trace', 'Units','normalized');
end

subplot(3, cols, 8);
if isfield(results, 'pi_trace_L2_motor')
    plot(time, results.pi_trace_L2_motor, 'r-'); xlabel('Step'); title('π L2 Motor'); grid on;
else
    text(0.1, 0.5, 'No π L2 Motor trace', 'Units','normalized');
end

subplot(3, cols, 9);
if isfield(results, 'pi_trace_L1_plan')
    plot(time, results.pi_trace_L1_plan, 'b-'); xlabel('Step'); title('π L1 Plan'); grid on;
else
    text(0.1, 0.5, 'No π L1 Plan trace', 'Units','normalized');
end

sgtitle('Smoke Test Summary', 'FontSize', 14);

outdir = fullfile('.', 'figures');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
outfn = fullfile(outdir, 'smoke_test_summary.png');
try
    saveas(fig, outfn);
    fprintf('Saved smoke test summary: %s\n', outfn);
catch ME
    fprintf('Failed to save summary figure: %s\n', ME.message);
end

close(fig);
fprintf('Visualization complete.\n');
end
