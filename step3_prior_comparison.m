% STEP 3: COMPARING STABILITY UNDER DIFFERENT PRIORS
% ====================================================
% This script explores how different prior strengths affect system behavior.
% This is critical for understanding psychiatric modeling:
%   - Strong priors = rigid beliefs (e.g., autism spectrum traits)
%   - Weak priors = unstable beliefs (e.g., psychotic spectrum traits)
%
% EXPERIMENT:
% We test the same sensory input with three different prior strengths
% and compare adaptation speed and stability.

clear; clc; close all;
fprintf('=== STEP 3: PRIOR COMPARISON ===\n\n');

%% Simulation Setup
fprintf('Setting up comparison experiment...\n');

% Time parameters
t_span = [0 10];
dt = 0.01;
t = 0:dt:t_span(2);
N = length(t);

% Sensory input (same as Step 2)
true_velocity = zeros(1, N);
mid = floor(N/2);
true_velocity(1:mid) = 2;
true_velocity(mid+1:end) = -1;
true_position = cumsum(true_velocity) * dt;

% Add noise
sensory_noise_std = 0.05;
x_obs = true_position + sensory_noise_std * randn(1, N);

fprintf('  Sensory input: velocity changes from +2 to -1 at t=5s\n\n');

%% Define Different Prior Strengths
fprintf('Testing three prior configurations:\n');

% Prior strengths to test
sigma_v_values = [0.1, 1.0, 10.0];
labels = {'Strong Prior (σ_v=0.1)', ...
          'Medium Prior (σ_v=1.0)', ...
          'Weak Prior (σ_v=10.0)'};
colors = [0.8 0.2 0.2;    % Red for strong (rigid)
          0.2 0.6 0.2;    % Green for medium
          0.2 0.2 0.8];   % Blue for weak (flexible)

% Interpretation
fprintf('  1. Strong Prior (σ_v = %.1f): Rigid beliefs, slow adaptation\n', sigma_v_values(1));
fprintf('     → Models: Autism spectrum (sensory over-reliance on priors)\n\n');

fprintf('  2. Medium Prior (σ_v = %.1f): Balanced beliefs\n', sigma_v_values(2));
fprintf('     → Models: Typical perception\n\n');

fprintf('  3. Weak Prior (σ_v = %.1f): Flexible but noisy beliefs\n', sigma_v_values(3));
fprintf('     → Models: Psychotic spectrum (volatile predictions)\n\n');

%% Fixed Parameters
sigma_x = 0.1;    % Sensory precision (same for all)
mu_v = 0;         % Expected velocity (stationary)

%% Run Simulations for Each Prior
fprintf('Running simulations');

results = struct();

for j = 1:length(sigma_v_values)
    fprintf('.');
    
    sigma_v = sigma_v_values(j);
    
    % Initialize
    x_est = 0;
    v_est = 0;
    
    % Storage
    x_history = zeros(1, N);
    v_history = zeros(1, N);
    free_energy = zeros(1, N);
    
    % Simulation loop
    for i = 1:N
        x_current = x_obs(i);
        
        % Prediction errors
        epsilon_x = x_current - v_est;
        epsilon_v = v_est - mu_v;
        
        % Update rules
        dx_dt = epsilon_x / sigma_x^2;
        dv_dt = epsilon_x / sigma_x^2 - epsilon_v / sigma_v^2;
        
        % Integrate
        x_est = x_est + dx_dt * dt;
        v_est = v_est + dv_dt * dt;
        
        % Free energy
        F = 0.5 * (epsilon_x^2 / sigma_x^2 + epsilon_v^2 / sigma_v^2);
        
        % Store
        x_history(i) = x_est;
        v_history(i) = v_est;
        free_energy(i) = F;
    end
    
    % Save results
    results(j).sigma_v = sigma_v;
    results(j).x_history = x_history;
    results(j).v_history = v_history;
    results(j).free_energy = free_energy;
    results(j).label = labels{j};
    results(j).color = colors(j,:);
end

fprintf(' Done!\n\n');

%% Visualization: Comparison Plots
fprintf('Generating comparison plots...\n');

figure('Position', [100 100 1600 1000], 'Name', 'Prior Strength Comparison');

% Subplot 1: Velocity inference comparison
subplot(2,3,1);
plot(t, true_velocity, 'k--', 'LineWidth', 3, 'DisplayName', 'True Velocity'); hold on;
for j = 1:length(results)
    plot(t, results(j).v_history, 'LineWidth', 2.5, ...
        'Color', results(j).color, 'DisplayName', results(j).label);
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity Estimate', 'FontSize', 11);
title('Belief Dynamics', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 9);
grid on; box on;
xline(5, 'k:', 'LineWidth', 1, 'Alpha', 0.3);

% Subplot 2: Adaptation error over time
subplot(2,3,2);
for j = 1:length(results)
    adaptation_error = abs(results(j).v_history - true_velocity);
    plot(t, adaptation_error, 'LineWidth', 2.5, ...
        'Color', results(j).color, 'DisplayName', results(j).label); hold on;
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('|v_{est} - v_{true}|', 'FontSize', 11);
title('Adaptation Speed', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on; box on;
xline(5, 'k:', 'LineWidth', 1, 'Alpha', 0.3);

% Subplot 3: Free energy comparison
subplot(2,3,3);
for j = 1:length(results)
    plot(t, results(j).free_energy, 'LineWidth', 2.5, ...
        'Color', results(j).color, 'DisplayName', results(j).label); hold on;
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Free Energy', 'FontSize', 11);
title('Model Evidence', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on; box on;
xline(5, 'k:', 'LineWidth', 1, 'Alpha', 0.3);

% Subplot 4: Position tracking
subplot(2,3,4);
plot(t, true_position, 'k--', 'LineWidth', 2, 'DisplayName', 'True'); hold on;
for j = 1:length(results)
    plot(t, results(j).x_history, 'LineWidth', 1.5, ...
        'Color', results(j).color, 'DisplayName', results(j).label);
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Position', 'FontSize', 11);
title('Position Tracking', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 9);
grid on; box on;

% Subplot 5: Zoom on adaptation period (t = 4.5 to 6.5)
subplot(2,3,5);
zoom_idx = (t >= 4.5) & (t <= 6.5);
plot(t(zoom_idx), true_velocity(zoom_idx), 'k--', 'LineWidth', 3, 'DisplayName', 'True'); hold on;
for j = 1:length(results)
    plot(t(zoom_idx), results(j).v_history(zoom_idx), 'LineWidth', 2.5, ...
        'Color', results(j).color, 'DisplayName', results(j).label);
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
title('Adaptation Period (Zoomed)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on; box on;
xline(5, 'k:', 'LineWidth', 1.5);

% Subplot 6: Variance in velocity estimates (stability)
subplot(2,3,6);
window = 50;  % 0.5 second window (50 samples)
v_variance = zeros(length(results), N - window + 1);
for j = 1:length(results)
    for i = 1:(N - window + 1)
        v_variance(j, i) = var(results(j).v_history(i:i+window-1));
    end
end
t_var = t(1:N-window+1);
for j = 1:length(results)
    plot(t_var, v_variance(j,:), 'LineWidth', 2.5, ...
        'Color', results(j).color, 'DisplayName', results(j).label); hold on;
end
xlabel('Time (s)', 'FontSize', 11);
ylabel('Variance (0.5s window)', 'FontSize', 11);
title('Belief Stability', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on; box on;

sgtitle('Effect of Prior Strength on Predictive Coding', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Quantitative Comparison
fprintf('\nQuantitative Analysis:\n');
fprintf('%-25s | %10s | %10s | %10s\n', 'Prior Type', 'Adapt Time', 'Final Error', 'Avg Variance');
fprintf('%s\n', repmat('-', 1, 70));

change_idx = mid + 1;    % first index after change (t ≈ 5s)
post_change = change_idx:N;

for j = 1:length(results)
    % Adaptation time (to within 20% of true value)
    vel_error = abs(results(j).v_history(post_change) - true_velocity(post_change));
    threshold = 0.2 * abs(true_velocity(change_idx) - true_velocity(change_idx-1));
    adapted_idx = find(vel_error < threshold, 1);
    
    if ~isempty(adapted_idx)
        adapt_time = adapted_idx * dt;
    else
        adapt_time = inf;
    end
    
    % Final velocity error
    final_error = abs(results(j).v_history(end) - true_velocity(end));
    
    % Average variance (stability measure) using 50-sample non-overlapping windows
    vh = results(j).v_history(post_change);
    win = 50;
    m = numel(vh);
    nwin = floor(m / win);
    if nwin > 0
        data = reshape(vh(1:nwin*win), win, nwin);  % each column is one window
        avg_variance = mean(var(data, 0, 1));       % variance per window, then average
    else
        avg_variance = var(vh);                    % fallback for very short vectors
    end
    
    fprintf('%-25s | %8.2fs | %10.4f | %10.6f\n', ...
        results(j).label, adapt_time, final_error, avg_variance);
end

%% Psychiatric Modeling Interpretation
fprintf('\n=== PSYCHIATRIC MODELING INTERPRETATION ===\n\n');

fprintf('STRONG PRIOR (σ_v = %.1f):\n', sigma_v_values(1));
fprintf('  • Slow to update beliefs despite clear sensory evidence\n');
fprintf('  • Models: Autism spectrum (over-reliance on predictions)\n');
fprintf('  • Clinical: Difficulty adapting to change, rigid thinking\n');
fprintf('  • Advantage: Stable in noisy environments\n\n');

fprintf('MEDIUM PRIOR (σ_v = %.1f):\n', sigma_v_values(2));
fprintf('  • Balanced adaptation and stability\n');
fprintf('  • Models: Typical perception\n');
fprintf('  • Optimal trade-off between flexibility and noise\n\n');

fprintf('WEAK PRIOR (σ_v = %.1f):\n', sigma_v_values(3));
fprintf('  • Rapid adaptation but volatile estimates\n');
fprintf('  • Models: Psychotic spectrum (weak prior beliefs)\n');
fprintf('  • Clinical: Hallucinations, delusions (sensory over-reliance)\n');
fprintf('  • Disadvantage: Susceptible to noise, instability\n\n');

%% Save Results
fprintf('Saving comparison results...\n');
save('prior_comparison_results.mat', 'results', 't', 'true_velocity', 'sigma_v_values');

fprintf('\n=== STEP 3 COMPLETE ===\n');
fprintf('Key findings:\n');
fprintf('  1. Prior strength controls adaptation vs. stability trade-off\n');
fprintf('  2. Strong priors → slow but stable (autism-like)\n');
fprintf('  3. Weak priors → fast but noisy (psychosis-like)\n');
fprintf('  4. Psychiatric conditions can be modeled as prior imbalances\n\n');
fprintf('Next: Run step4_ode45_version.m for high-precision simulation.\n');
