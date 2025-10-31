% Quick validation script to test 2D model and test framework
% This ensures all files exist and basic syntax is correct

clear all; close all; clc;

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('VALIDATING 2D SPATIOTEMPORAL MOTION MODEL\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

try
    fprintf('[1/3] Testing hierarchical_motion_inference_2D_EXACT()...\n');
    [R_L1, R_L2, R_L3, E_L1, E_L2, E_L3, W_L1_from_L2, W_L2_from_L3, free_energy, true_x, true_y, true_vx, true_vy, true_ax, true_ay] = hierarchical_motion_inference_2D_EXACT();
    fprintf('      ✓ Model executed successfully\n');
    fprintf('      - R_L1 size: [%d × %d]\n', size(R_L1));
    fprintf('      - Free energy converged: %.4f → %.4f\n', free_energy(1), free_energy(end));
    fprintf('      - Weight matrices learned: W_L1 [%d × %d], W_L2 [%d × %d]\n', ...
        size(W_L1_from_L2), size(W_L2_from_L3));
    
catch ME
    fprintf('      ✗ ERROR: %s\n', ME.message);
    fprintf('      Stack trace:\n');
    disp(ME.stack);
    return;
end

try
    fprintf('\n[2/3] Testing analyze_learned_motion_filters()...\n');
    analyze_learned_motion_filters(W_L1_from_L2, W_L2_from_L3);
    fprintf('      ✓ Analysis script executed successfully\n');
    
catch ME
    fprintf('      ✗ ERROR: %s\n', ME.message);
    fprintf('      Stack trace:\n');
    disp(ME.stack);
    return;
end

try
    fprintf('\n[3/3] Testing test_rao_ballard_2D()...\n');
    test_rao_ballard_2D();
    fprintf('      ✓ Test framework executed successfully\n');
    
catch ME
    fprintf('      ✗ ERROR: %s\n', ME.message);
    fprintf('      Stack trace:\n');
    disp(ME.stack);
    return;
end

fprintf('\n═══════════════════════════════════════════════════════════\n');
fprintf('ALL VALIDATION TESTS PASSED\n');
fprintf('═══════════════════════════════════════════════════════════\n');
