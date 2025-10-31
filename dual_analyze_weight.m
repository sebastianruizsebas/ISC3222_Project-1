%% DUAL HIERARCHY WEIGHT MATRIX ANALYSIS
% Analyze learned weight matrices for the dual-hierarchy experiment.
% Produces summary statistics, heatmaps and saves a text report.

clear all; close all; clc;

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  DUAL-HIERARCHY WEIGHT MATRIX ANALYSIS                     ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Default results file produced by hierarchical_motion_inference_dual_hierarchy
results_file = './tools/figures/3D_dual_hierarchy_results.mat';
if ~isfile(results_file)
    % try a best or alternate name
    alt = './tools/figures/3D_dual_hierarchy_results_best.mat';
    if isfile(alt)
        results_file = alt;
    else
        error('Results file not found: %s (also tried %s)', './figures/3D_dual_hierarchy_results.mat', alt);
    end
end

fprintf('Loading results: %s\n', results_file);
loaded = load(results_file);
fprintf('✓ Loaded: %s\n\n', results_file);

% Determine variable names used in results
% For dual hierarchy we expect motor/planning matrices named like:
% W_motor_L2_to_L1, W_motor_L3_to_L2, W_plan_L2_to_L1, W_plan_L3_to_L2
vars = fieldnames(loaded);

% Helper to assert existence or fallback
getvar = @(name) (isfield(loaded, name) && ~isempty(loaded.(name))) * loaded.(name) + (~isfield(loaded, name) && error('Required variable %s not found in %s', name, results_file));

% Check for motor and plan weight matrices
required_motor = {'W_motor_L2_to_L1', 'W_motor_L3_to_L2'};
required_plan = {'W_plan_L2_to_L1', 'W_plan_L3_to_L2'};

for k = 1:numel(required_motor)
    if ~isfield(loaded, required_motor{k})
        error('Missing required motor weight matrix: %s (file: %s)', required_motor{k}, results_file);
    end
end
for k = 1:numel(required_plan)
    if ~isfield(loaded, required_plan{k})
        error('Missing required planning weight matrix: %s (file: %s)', required_plan{k}, results_file);
    end
end

W_motor_L2_to_L1 = loaded.W_motor_L2_to_L1;
W_motor_L3_to_L2 = loaded.W_motor_L3_to_L2;
W_plan_L2_to_L1  = loaded.W_plan_L2_to_L1;
W_plan_L3_to_L2  = loaded.W_plan_L3_to_L2;

% Optional lateral weights
W_motor_L1_lat = [];
if isfield(loaded, 'W_motor_L1_lat'), W_motor_L1_lat = loaded.W_motor_L1_lat; end
W_plan_L1_lat = [];
if isfield(loaded, 'W_plan_L1_lat'), W_plan_L1_lat = loaded.W_plan_L1_lat; end

fprintf('Variables found and loaded:\n');
fprintf('  W_motor_L2_to_L1: %dx%d\n', size(W_motor_L2_to_L1));
fprintf('  W_motor_L3_to_L2: %dx%d\n', size(W_motor_L3_to_L2));
fprintf('  W_plan_L2_to_L1 : %dx%d\n', size(W_plan_L2_to_L1));
fprintf('  W_plan_L3_to_L2 : %dx%d\n\n', size(W_plan_L3_to_L2));

% =====================
% MOTOR REGION ANALYSIS
% =====================

fprintf('\n=== MOTOR REGION: Forward Model (Motor→Proprioception) ===\n');
% W_motor_L2_to_L1: maps motor basis (cols) to proprioceptive outputs (rows)
W_forward = W_motor_L2_to_L1; % rows: proprio (e.g., 7), cols: motor basis (e.g., 6)

fprintf('Forward model shape: [%d × %d] (proprio × motor_basis)\n', size(W_forward,1), size(W_forward,2));

% Position rows: assume first 3 rows are positions, next 3 velocities, final bias
if size(W_forward,1) >= 6
    W_pos = W_forward(1:3, :);
    W_vel = W_forward(4:6, :);
else
    W_pos = W_forward(1:min(3,size(W_forward,1)), :);
    W_vel = W_forward(max(1,size(W_forward,1)-2):size(W_forward,1), :);
end

% Velocity mapping expected to be near-identity on first 3 motor cols
vel_identity_error = NaN;
if size(W_vel,2) >= 3 && size(W_vel,1) >= 3
    vel_identity_error = norm(W_vel(:,1:3) - eye(3));
    fprintf('Velocity submatrix error from identity (motor→velocity): %.6f\n', vel_identity_error);
else
    fprintf('Velocity submatrix too small to test identity assumption.\n');
end

% Auxiliary motor contribution magnitude
aux_mag = norm(W_vel(:,4:end),'fro');
fprintf('Auxiliary motor contribution magnitude: %.6f\n', aux_mag);

% =====================
% PLANNING REGION ANALYSIS
% =====================

fprintf('\n=== PLANNING REGION: Goal→Policy and Policy→Output ===\n');
% W_plan_L3_to_L2 maps plan output (e.g., 3D) to policies (cols?), but naming varies.
% We'll analyze W_plan_L2_to_L1 (policies → planning L1) as goal mapping and
% W_plan_L3_to_L2 as policy→output mapping.

W_plan_goal = W_plan_L2_to_L1; % rows: planning L1 dims (7), cols: planning policies (6)
W_plan_output = W_plan_L3_to_L2; % rows: policies dims (6), cols: output dims (3) or vice-versa

fprintf('W_plan_L2_to_L1 shape: [%d × %d]\n', size(W_plan_goal,1), size(W_plan_goal,2));
fprintf('W_plan_L3_to_L2 shape: [%d × %d]\n', size(W_plan_output,1), size(W_plan_output,2));

% Try to interpret goal coupling: look for columns in W_plan_goal that map goals
% into planning L1 entries (we expect indices 1:3 to be ball position or target)
if size(W_plan_goal,1) >= 3
    goal_coupling = norm(W_plan_goal(1:3,:), 'fro');
    fprintf('Goal coupling (norm of first 3 rows): %.6f\n', goal_coupling);
else
    fprintf('Planning goal rows too small to compute coupling metric.\n');
end

% =====================
% SUMMARY AND VISUALIZATIONS
% =====================

fig = figure('Name', 'Dual Hierarchy Weight Matrices', 'Position', [100, 100, 1400, 900]);

% Panel 1: Motor forward model heatmap
ax1 = subplot(2,2,1);
imagesc(W_motor_L2_to_L1);
colorbar;
title('W_{motor: L1 \leftarrow L2} (Proprioception \leftarrow Motor Basis)');
xlabel('Motor basis channels'); ylabel('Proprioceptive outputs');

% Panel 2: Motor output mapping
ax2 = subplot(2,2,2);
imagesc(W_motor_L3_to_L2);
colorbar;
title('W_{motor: L2 \leftarrow L3} (Basis \leftarrow Output)');
xlabel('L3 output dims'); ylabel('L2 basis channels');

% Panel 3: Planning goal coupling (heatmap of first 3 rows)
ax3 = subplot(2,2,3);
imagesc(W_plan_L2_to_L1);
colorbar;
title('W_{plan: L1 \leftarrow L2} (Planning L1 \leftarrow Policies)');
xlabel('Policy channels'); ylabel('Planning L1 channels');

% Panel 4: Planning output mapping
ax4 = subplot(2,2,4);
imagesc(W_plan_L3_to_L2);
colorbar;
title('W_{plan: L2 \leftarrow L3} (Policies \leftarrow Output)');
xlabel('L3 outputs'); ylabel('Policy channels');

sgtitle('Dual-Hierarchy Learned Weight Matrices', 'FontSize', 13, 'FontWeight', 'bold');

out_png = './figures/dual_hierarchy_weight_analysis.png';
try
    saveas(fig, out_png);
    fprintf('\nSaved figure: %s\n', out_png);
catch ME
    fprintf('Warning: could not save figure: %s\n', ME.message);
end

% Export analysis summary file
analysis_file = './figures/DUAL_WEIGHT_MATRIX_ANALYSIS.txt';
fid = fopen(analysis_file, 'w');
fprintf(fid, 'DUAL HIERARCHY WEIGHT MATRIX ANALYSIS\n');
fprintf(fid, 'Source: %s\n\n', results_file);
fprintf(fid, 'W_motor_L2_to_L1 shape: %d x %d\n', size(W_motor_L2_to_L1));
fprintf(fid, 'W_motor_L3_to_L2 shape: %d x %d\n', size(W_motor_L3_to_L2));
fprintf(fid, 'W_plan_L2_to_L1 shape : %d x %d\n', size(W_plan_L2_to_L1));
fprintf(fid, 'W_plan_L3_to_L2 shape : %d x %d\n\n', size(W_plan_L3_to_L2));

fprintf(fid, 'Velocity identity error (motor velocity rows vs identity): %.6f\n', vel_identity_error);
fprintf(fid, 'Auxiliary motor contribution magnitude: %.6f\n', aux_mag);
if exist('goal_coupling', 'var')
    fprintf(fid, 'Planning goal coupling (first 3 rows norm): %.6f\n', goal_coupling);
end

fclose(fid);
fprintf('Saved analysis text: %s\n', analysis_file);

fprintf('\n✓ Dual-hierarchy weight analysis complete.\n');
