% filepath: analyze_learned_motion_filters.m
%
% ANALYZE LEARNED MOTION FILTERS (like V1 complex cells)
% Shows how Rao & Ballard learning discovers direction selectivity
% 
% Updated for 2D motion with x, y, vx, vy, ax, ay components

function [] = analyze_learned_motion_filters(W_L1_from_L2, W_L2_from_L3)

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  LEARNED MOTION FILTERS - DIRECTION SELECTIVITY           ║\n');
fprintf('║  Analysis of 2D Motion Filter Properties                  ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% W_L1_from_L2 predicts 2D position (x, y) and velocity (vx, vy) from motion filters
% Extract the filter-to-sensory mappings to understand learned representations

fprintf('WEIGHT MATRIX W^(L1): Position/Velocity ← Motion Filters\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('  Dimensions: %d sensory components × %d learned filters\n', size(W_L1_from_L2));
fprintf('  Sensory components: [x, y, vx, vy, ax, ay, bias1, bias2]\n\n');

% Analyze each learned motion filter
n_filters = size(W_L1_from_L2, 2);

for j = 1:n_filters
    w_filter = W_L1_from_L2(:, j);
    
    % Extract components
    w_x = w_filter(1);
    w_y = w_filter(2);
    w_vx = w_filter(3);
    w_vy = w_filter(4);
    w_ax = w_filter(5);
    w_ay = w_filter(6);
    
    % Compute filter properties
    position_component = sqrt(w_x^2 + w_y^2);
    velocity_component = sqrt(w_vx^2 + w_vy^2);
    acceleration_component = sqrt(w_ax^2 + w_ay^2);
    filter_magnitude = norm(w_filter);
    
    % Direction selectivity: ratio of max to next largest component
    magnitudes = [position_component, velocity_component, acceleration_component];
    [max_mag, max_idx] = max(magnitudes);
    selectivity = max_mag / (mean(magnitudes) + 1e-6);
    
    % Compute motion direction angle (velocity direction in 2D)
    motion_angle = atan2(w_vy, w_vx) * 180 / pi;
    
    fprintf('Motion Filter %d:\n', j);
    fprintf('  Total magnitude: %.6f\n', filter_magnitude);
    fprintf('  Position component: %.6f (x=%.4f, y=%.4f)\n', position_component, w_x, w_y);
    fprintf('  Velocity component: %.6f (vx=%.4f, vy=%.4f)\n', velocity_component, w_vx, w_vy);
    fprintf('  Acceleration component: %.6f (ax=%.4f, ay=%.4f)\n', acceleration_component, w_ax, w_ay);
    fprintf('  Motion direction: %.1f° from x-axis\n', motion_angle);
    fprintf('  Selectivity index: %.4f\n', selectivity);
    fprintf('  → Primary component: ');
    
    component_names = {'Position', 'Velocity', 'Acceleration'};
    fprintf('%s\n\n', component_names{max_idx});
end

% W_L2_from_L3 predicts velocity from acceleration
fprintf('\nWEIGHT MATRIX W^(L2): Velocity ← Acceleration\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('  Dimensions: %d motion filters × %d acceleration components\n', size(W_L2_from_L3));
fprintf('  Acceleration components: [ax, ay, bias]\n\n');

for j = 1:size(W_L2_from_L3, 2)
    w_filter = W_L2_from_L3(:, j);
    
    magnitude = norm(w_filter);
    
    fprintf('Acceleration Component %d:\n', j);
    fprintf('  Total filter magnitude: %.6f\n', magnitude);
    fprintf('  Motion filter tuning: [%s]\n', sprintf('%.4f ', w_filter));
    fprintf('  → Predicts velocity change based on acceleration\n\n');
end

% Compute population statistics
fprintf('\nPOPULATION STATISTICS:\n');
fprintf('─────────────────────────────────────────────────────────────\n\n');

w1_norms = vecnorm(W_L1_from_L2, 2, 1);
w2_norms = vecnorm(W_L2_from_L3, 2, 1);

fprintf('W^(L1) Motion Filter Magnitudes:\n');
fprintf('  Mean: %.6f\n', mean(w1_norms));
fprintf('  Std:  %.6f\n', std(w1_norms));
fprintf('  Range: [%.6f, %.6f]\n', min(w1_norms), max(w1_norms));
fprintf('  → Indicates learned diversity in motion representations\n\n');

fprintf('W^(L2) Acceleration Mapping Magnitudes:\n');
fprintf('  Mean: %.6f\n', mean(w2_norms));
fprintf('  Std:  %.6f\n', std(w2_norms));
fprintf('  Range: [%.6f, %.6f]\n\n', min(w2_norms), max(w2_norms));

% Visualize learned filters
figure('Name', 'Learned Motion Filters', 'NumberTitle', 'off', 'Position', [100 100 1000 700]);

subplot(2,1,1);
imagesc(W_L1_from_L2);
colormap(gca, 'redblue');
colorbar;
xlabel('Learned Motion Filter Index');
ylabel('Sensory Components (x, y, vx, vy, ax, ay, bias1, bias2)');
title('W^(L1): How Learned Filters Predict Position/Velocity');
set(gca, 'YTickLabel', {'x', 'y', 'vx', 'vy', 'ax', 'ay', 'b1', 'b2'});

subplot(2,1,2);
imagesc(W_L2_from_L3);
colormap(gca, 'redblue');
colorbar;
xlabel('Acceleration Components (ax, ay, bias)');
ylabel('Learned Motion Filter Index');
title('W^(L2): How Acceleration Drives Motion Filters');
set(gca, 'XTickLabel', {'ax', 'ay', 'bias'});

fprintf('═══════════════════════════════════════════════════════════\n\n');

end