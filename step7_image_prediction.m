%% Step 7: Image Prediction with Predictive Coding
% Extends the 1D velocity inference to 2D spatial predictions
% Uses the same free energy minimization framework from Steps 2-4

clear; clc;

fprintf('Step 7: 2D Image Prediction with Predictive Coding\n');
fprintf('==================================================\n\n');

%% Load parameters from previous steps for consistency
if exist('simulation_results.mat', 'file')
    prev = load('simulation_results.mat');
    dt = prev.t(2) - prev.t(1);
    % Use default values since these aren't saved in Step 2
    kappa_v = 2.0;
    sigma_x = 0.1;  % Match Step 2's sigma_x value
    fprintf('Loaded time step from Step 2:\n');
    fprintf('  dt = %.4f s\n', dt);
    fprintf('  kappa_v = %.2f (default)\n', kappa_v);
    fprintf('  sigma_x = %.2f (default)\n\n', sigma_x);
else
    % Default parameters
    dt = 0.01;
    kappa_v = 2.0;
    sigma_x = 0.1;
    fprintf('Using default parameters (Step 2 not found)\n\n');
end

%% Simulation parameters
T_total = 10;
t = 0:dt:T_total;
N = length(t);

% Image parameters
img_size = 64;
[X, Y] = meshgrid(1:img_size, 1:img_size);
center_x = img_size/2;
center_y = img_size/2;

% Gabor parameters
sigma_gabor = 8;  % Spatial extent
lambda = 12;      % Wavelength
theta = 0;        % Orientation

%% True velocity (same pattern as Steps 2-6)
v_true = 2 * ones(size(t));
v_true(t >= 5) = 4;  % Velocity change at t=5s

% Inferred velocity (initialize)
v_est = 2.0;  % Initial guess
v_history = zeros(size(t));

% Prior precision on velocity (from Step 3 concepts)
sigma_v = 1.0;  % Medium prior strength
pi_v = 1 / sigma_v^2;

% Observation precision
pi_x = 1 / sigma_x^2;

%% Initialize tracking
phase_history = zeros(size(t));
prediction_error_history = zeros(size(t));
free_energy_spatial = zeros(size(t));

% Initial phase
phase = 0;

fprintf('Running 2D spatial prediction simulation...\n');

%% Main simulation loop
for t_idx = 1:N
    % True Gabor patch (moving rightward)
    true_phase = 2*pi * ((X - center_x) * cos(theta) + ...
                         (Y - center_y) * sin(theta) + ...
                         v_true(t_idx) * t(t_idx)) / lambda;
    gabor_true = exp(-((X-center_x).^2 + (Y-center_y).^2)/(2*sigma_gabor^2)) .* ...
                 cos(true_phase);
    
    % Observed image (with noise)
    image_obs = gabor_true + randn(size(gabor_true)) * sigma_x;
    
    % Predicted image based on current velocity estimate
    pred_phase = 2*pi * ((X - center_x) * cos(theta) + ...
                         (Y - center_y) * sin(theta) + ...
                         v_est * t(t_idx)) / lambda;
    gabor_pred = exp(-((X-center_x).^2 + (Y-center_y).^2)/(2*sigma_gabor^2)) .* ...
                 cos(pred_phase);
    
    % Spatial prediction error (pixel-wise)
    spatial_error = image_obs - gabor_pred;
    
    % Aggregate prediction error (sum over spatial dimensions)
    total_pred_error = sum(spatial_error(:).^2);
    prediction_error_history(t_idx) = total_pred_error;
    
    % Free energy (same form as Steps 2-4, but summed over space)
    free_energy_spatial(t_idx) = 0.5 * pi_x * total_pred_error + ...
                                 0.5 * pi_v * (v_est - 2)^2;
    
    % Velocity update (gradient descent on free energy)
    % Compute how prediction error changes with velocity
    dpred_dv = 2*pi * t(t_idx) / lambda * ...
               exp(-((X-center_x).^2 + (Y-center_y).^2)/(2*sigma_gabor^2)) .* ...
               (-sin(pred_phase));
    
    % Gradient of prediction error w.r.t. velocity
    grad_v = sum(spatial_error(:) .* dpred_dv(:));
    
    % Update velocity estimate (predictive coding update)
    dv_dt = -kappa_v * (pi_x * grad_v + pi_v * (v_est - 2));
    v_est = v_est + dt * dv_dt;
    
    % Store results
    v_history(t_idx) = v_est;
    phase_history(t_idx) = mean(pred_phase(:));
end

fprintf('Simulation complete!\n\n');

%% Visualization
fig = figure('Position', [100, 100, 1400, 900]);
sgtitle('Step 7: 2D Spatial Prediction with Predictive Coding', ...
        'FontSize', 16, 'FontWeight', 'bold');

% 1. Velocity inference (compare with 1D case)
subplot(2, 3, 1);
plot(t, v_true, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True Velocity');
hold on;
plot(t, v_history, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated (from images)');
xline(5, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Velocity');
title('Velocity Inference from 2D Images');
legend('Location', 'best');
grid on;

% 2. Spatial prediction error over time
subplot(2, 3, 2);
plot(t, prediction_error_history, 'b-', 'LineWidth', 2);
xline(5, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Sum of Squared Errors');
title('Spatial Prediction Error');
grid on;

% 3. Free energy
subplot(2, 3, 3);
plot(t, free_energy_spatial, 'Color', [0.6 0.2 0.8], 'LineWidth', 2);
xline(5, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Free Energy');
title('Free Energy Minimization');
grid on;

% 4. Example image at t=2s (before change)
t_early = find(t >= 2, 1);
subplot(2, 3, 4);
true_phase_early = 2*pi * ((X - center_x) * cos(theta) + ...
                           (Y - center_y) * sin(theta) + ...
                           v_true(t_early) * t(t_early)) / lambda;
gabor_early = exp(-((X-center_x).^2 + (Y-center_y).^2)/(2*sigma_gabor^2)) .* ...
              cos(true_phase_early);
imagesc(gabor_early);
colormap(subplot(2,3,4), gray);
axis square;
title(sprintf('True Image at t=%.1fs', t(t_early)));
colorbar;

% 5. Example image at t=7s (after change)
t_late = find(t >= 7, 1);
subplot(2, 3, 5);
true_phase_late = 2*pi * ((X - center_x) * cos(theta) + ...
                          (Y - center_y) * sin(theta) + ...
                          v_true(t_late) * t(t_late)) / lambda;
gabor_late = exp(-((X-center_x).^2 + (Y-center_y).^2)/(2*sigma_gabor^2)) .* ...
             cos(true_phase_late);
imagesc(gabor_late);
colormap(subplot(2,3,5), gray);
axis square;
title(sprintf('True Image at t=%.1fs', t(t_late)));
colorbar;

% 6. Comparison with 1D results (if available)
subplot(2, 3, 6);
if exist('simulation_results.mat', 'file')
    prev = load('simulation_results.mat');
    plot(prev.t, prev.v_history, 'b-', 'LineWidth', 2, 'DisplayName', '1D (Step 2)');
    hold on;
    plot(t, v_history, 'r--', 'LineWidth', 2, 'DisplayName', '2D (Step 7)');
    plot(prev.t, prev.true_velocity, 'k:', 'LineWidth', 2.5, 'DisplayName', 'True');
    xline(5, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Velocity');
    title('1D vs 2D Comparison');
    legend('Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'Run Step 2 first for comparison', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
    axis off;
    title('1D vs 2D Comparison');
end

%% Save results
save('image_prediction_results.mat', 't', 'v_history', 'v_true', ...
     'prediction_error_history', 'free_energy_spatial', 'X', 'Y', ...
     'sigma_gabor', 'lambda', 'img_size');

fprintf('Results saved to: image_prediction_results.mat\n\n');

%% Print summary
fprintf('Step 7 Summary - 2D Spatial Prediction:\n');
fprintf('  Image size: %dx%d pixels\n', img_size, img_size);
fprintf('  Mean velocity error: %.4f\n', mean(abs(v_history - v_true)));
fprintf('  Final velocity estimate: %.2f (true: %.2f)\n', v_history(end), v_true(end));
fprintf('  Final free energy: %.4f\n', free_energy_spatial(end));

if exist('simulation_results.mat', 'file')
    prev = load('simulation_results.mat');
    err_1d = mean(abs(prev.v_history - prev.true_velocity));
    err_2d = mean(abs(v_history - v_true));
    fprintf('\nComparison with 1D inference:\n');
    fprintf('  1D mean error: %.4f\n', err_1d);
    fprintf('  2D mean error: %.4f\n', err_2d);
    fprintf('  Ratio (2D/1D): %.2fx\n', err_2d/err_1d);
end