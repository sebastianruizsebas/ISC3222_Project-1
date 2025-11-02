% Test single particle with valid parameters
clear all;
clc;

fprintf('Testing single PSO particle with valid bounds...\n\n');

% Create test parameters with valid bounds (min < max)
test_params = struct();

% Bounds must satisfy: min < max
test_params.pi_L1_motor_min = 5;      % [5, 50]
test_params.pi_L1_motor_max = 100;    % Valid: 5 < 100 ✓
test_params.pi_L2_motor_min = 0.5;    % [0.5, 5]  
test_params.pi_L2_motor_max = 50;     % Valid: 0.5 < 50 ✓
test_params.pi_L1_plan_min = 10;      % [10, 50]
test_params.pi_L1_plan_max = 500;     % Valid: 10 < 500 ✓
test_params.pi_L2_plan_min = 0.1;     % [0.1, 5]
test_params.pi_L2_plan_max = 100;     % Valid: 0.1 < 100 ✓

% Precision gain
test_params.alpha_precision_gain = 1.5;

fprintf('Test Parameters:\n');
fprintf('  pi_L1_motor bounds: [%.1f, %.1f]\n', test_params.pi_L1_motor_min, test_params.pi_L1_motor_max);
fprintf('  pi_L2_motor bounds: [%.1f, %.1f]\n', test_params.pi_L2_motor_min, test_params.pi_L2_motor_max);
fprintf('  pi_L1_plan bounds: [%.1f, %.1f]\n', test_params.pi_L1_plan_min, test_params.pi_L1_plan_max);
fprintf('  pi_L2_plan bounds: [%.1f, %.1f]\n', test_params.pi_L2_plan_min, test_params.pi_L2_plan_max);
fprintf('  alpha_precision_gain: %.2f\n\n', test_params.alpha_precision_gain);

fprintf('Running simulation...\n');
tic;
result = hierarchical_motion_inference_dual_hierarchy(test_params, false);
elapsed = toc;

fprintf('\nSimulation completed in %.2f seconds\n', elapsed);

% Compute score from interception error (same as PSO does)
if isfield(result, 'interception_error_all') && numel(result.interception_error_all) > 0
    score = mean(result.interception_error_all(max(1, end-500):end));  % Average last 500 steps
else
    score = Inf;
end

fprintf('Score (avg final interception error): %g\n', score);
fprintf('Is finite: %d\n', isfinite(score));

if isfinite(score)
    fprintf('\n✓ SUCCESS: Score is finite!\n');
else
    fprintf('\n✗ FAILURE: Score is Inf or NaN\n');
end
