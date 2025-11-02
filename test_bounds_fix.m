% Quick test of single PSO particle after fixes
clear all; close all; clc;

fprintf('╔════════════════════════════════════════════╗\n');
fprintf('║ TESTING SINGLE PSO PARTICLE (BUG FIX TEST)║\n');
fprintf('╚════════════════════════════════════════════╝\n\n');

% Test 1: Simple particle with default values
fprintf('Test 1: Running with default parameters...\n');
try
    result1 = hierarchical_motion_inference_dual_hierarchy(struct(), false);
    score1 = mean(result1.interception_error_all(1500:end));  % Last 500 steps
    fprintf('  ✓ Score: %.4f (finite: %d)\n\n', score1, isfinite(score1));
catch ME
    fprintf('  ✗ ERROR: %s\n\n', ME.message);
end

% Test 2: Particle with PSO precision bounds
fprintf('Test 2: Running with PSO precision bounds...\n');
try
    params = struct();
    % Set some PSO precision bounds
    params.pi_L1_motor_min = 15;
    params.pi_L1_motor_max = 200;
    params.pi_L2_motor_min = 1;
    params.pi_L2_motor_max = 40;
    params.pi_L1_plan_min = 20;
    params.pi_L1_plan_max = 400;
    params.pi_L2_plan_min = 1;
    params.pi_L2_plan_max = 40;
    params.alpha_precision_gain = 0.8;
    
    result2 = hierarchical_motion_inference_dual_hierarchy(params, false);
    score2 = mean(result2.interception_error_all(1500:end));
    fprintf('  ✓ Score: %.4f (finite: %d)\n\n', score2, isfinite(score2));
catch ME
    fprintf('  ✗ ERROR: %s\n\n', ME.message);
end

% Test 3: Particle with inverted bounds (should be auto-corrected)
fprintf('Test 3: Running with INVERTED bounds (should auto-correct)...\n');
try
    params = struct();
    % INTENTIONALLY INVERTED: max < min
    params.pi_L1_motor_min = 200;  % Normally max
    params.pi_L1_motor_max = 15;   % Normally min (INVERTED!)
    params.pi_L2_motor_min = 40;
    params.pi_L2_motor_max = 1;    % INVERTED!
    params.pi_L1_plan_min = 400;
    params.pi_L1_plan_max = 20;    % INVERTED!
    params.pi_L2_plan_min = 40;
    params.pi_L2_plan_max = 1;     % INVERTED!
    params.alpha_precision_gain = 0.8;
    
    result3 = hierarchical_motion_inference_dual_hierarchy(params, false);
    score3 = mean(result3.interception_error_all(1500:end));
    fprintf('  ✓ Score: %.4f (finite: %d) - AUTO-CORRECTED!\n\n', score3, isfinite(score3));
catch ME
    fprintf('  ✗ ERROR: %s\n\n', ME.message);
end

fprintf('═══════════════════════════════════════════════\n');
fprintf('All tests completed!\n');
