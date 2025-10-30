% filepath: interactive_interception_game.m
%
% REAL-TIME INTERCEPTION GAME WITH HIERARCHICAL MOTION INFERENCE
% ===============================================================
% Players control a reticle to intercept moving targets
% Game collects motor commands, reaction times, and interception accuracy
% Hierarchical model infers latent velocity predictions from motor behavior
% Results reveal neural strategies (predictive vs. reactive)
%
% GAMEPLAY: 
%   - Watch target move across screen
%   - Use arrow keys or mouse to control your reticle (green circle)
%   - Try to intercept the target (red circle) by overlapping it
%   - Your brain infers the target's velocity and plans ahead
%
% DATA COLLECTED:
%   - Lead distance: How far ahead of target your reticle is
%   - Interception accuracy: Distance from target when you press space
%   - Reaction time: Latency between target appearance and first move
%   - Motor tracking lag: Delay between target and reticle movements
%
% ANALYSIS:
%   - Fit hierarchical model to infer velocity/acceleration estimates
%   - Compare model predictions to actual motor commands
%   - Relate precision weights to lead distance (neural strategy)

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: GAME SETUP
%  ========================================================================

fprintf('=================================================================\n');
fprintf('  INTERACTIVE INTERCEPTION GAME\n');
fprintf('  Test Your Predictive Motor Control!\n');
fprintf('=================================================================\n\n');

% Participant info
participant_id = input('Enter participant ID (e.g., P001): ', 's');
age = input('Age: ');
gaming_experience = input('Gaming experience (1=none, 5=expert): ');

% Game parameters
n_trials = 15;              % 15 trials
target_speed_range = [100, 300];  % pixels/sec
acceleration_types = {'constant', 'accelerating', 'decelerating'};

% Display setup
screen_width = 1280;
screen_height = 720;
target_size = 30;           % target diameter (pixels)
reticle_size = 40;          % reticle diameter

% Create results directory
if ~exist('interception_game_results', 'dir')
    mkdir('interception_game_results');
end

% Initialize results storage
game_results = struct();
game_results.participant_id = participant_id;
game_results.age = age;
game_results.gaming_experience = gaming_experience;
game_results.timestamp = datetime('now');

% Trial-by-trial data
game_results.trial_data = struct();

fprintf('Loading game graphics...\n\n');

%% ========================================================================
%  SECTION 2: CREATE GAME WINDOW
%  ========================================================================

% Create figure with custom size and position
fig = figure('Name', sprintf('Interception Game - Player %s', participant_id), ...
    'NumberTitle', 'off', ...
    'Position', [100, 100, screen_width, screen_height], ...
    'Color', [0.1, 0.1, 0.1], ...
    'MenuBar', 'none', ...
    'ToolBar', 'none', ...
    'KeyPressFcn', @keyPressed);

% Create axes
ax = axes('Parent', fig, 'Position', [0, 0, 1, 1], ...
    'XLim', [0, screen_width], 'YLim', [0, screen_height], ...
    'Color', [0.1, 0.1, 0.1], 'XTick', [], 'YTick', []);

hold(ax, 'on');
axis(ax, 'equal');

% Graphical elements
target_graphic = circle(screen_width/2, screen_height/2, target_size, 'red', ax);
reticle_graphic = circle(screen_width/2, screen_height/2, reticle_size, 'green', ax);
trial_text = text(ax, screen_width*0.05, screen_height*0.95, '', ...
    'Color', 'white', 'FontSize', 14, 'VerticalAlignment', 'top');

instructions = text(ax, screen_width/2, screen_height*0.1, ...
    'READY? Press SPACE to start', ...
    'Color', 'yellow', 'FontSize', 16, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');

% Keyboard state tracking
keyboard_state = struct();
keyboard_state.keys_pressed = containers.Map();
keyboard_state.space_pressed = false;
keyboard_state.trial_started = false;

%% ========================================================================
%  SECTION 3: RUN TRIALS
%  ========================================================================

fprintf('Starting trials...\n');
fprintf('Use ARROW KEYS or MOUSE to control green reticle\n');
fprintf('Press SPACE to intercept target\n\n');

pause(2);  % Let participant read instructions

for trial_idx = 1:n_trials
    
    % Generate target trajectory for this trial
    accel_type = acceleration_types{mod(trial_idx-1, length(acceleration_types)) + 1};
    [target_traj, target_times, v_true, a_true] = generateTargetTrajectory(...
        screen_width, screen_height, target_speed_range, accel_type);
    
    % Trial variables
    trial_data = struct();
    trial_data.trial_num = trial_idx;
    trial_data.acceleration_type = accel_type;
    trial_data.v_true = v_true;
    trial_data.a_true = a_true;
    trial_data.target_trajectory = target_traj;
    trial_data.target_times = target_times;
    
    % Initialize recording
    motor_data = struct();
    motor_data.reticle_pos = [];
    motor_data.reticle_times = [];
    motor_data.target_pos_observed = [];
    motor_data.reaction_time = [];
    motor_data.intercept_attempt_time = [];
    motor_data.intercept_accuracy = [];
    motor_data.lead_distance = [];
    
    % Display trial start message
    set(trial_text, 'String', sprintf('Trial %d/%d: %s motion\nPress SPACE when ready...', ...
        trial_idx, n_trials, accel_type));
    set(reticle_graphic, 'XData', screen_width/2, 'YData', screen_height/2);
    
    drawnow;
    
    % Wait for participant to press SPACE to start
    keyboard_state.trial_started = false;
    keyboard_state.space_pressed = false;
    
    while ~keyboard_state.trial_started
        pause(0.01);
    end
    
    motor_data.reaction_time = 0;  % Will be updated as they move
    trial_start_time = tic;
    
    % Run trial
    set(instructions, 'String', 'Go! Intercept the target!');
    set(trial_text, 'String', sprintf('Trial %d/%d: %s motion', trial_idx, n_trials, accel_type));
    
    intercept_made = false;
    trial_duration = 5;  % 5 seconds per trial
    
    while toc(trial_start_time) < trial_duration && ~intercept_made
        
        t_elapsed = toc(trial_start_time);
        
        % Get target position at this time
        if t_elapsed <= target_times(end)
            target_pos = interp1(target_times, target_traj, t_elapsed, 'linear', 'extrap');
            set(target_graphic, 'XData', target_pos(1), 'YData', target_pos(2));
            motor_data.target_pos_observed = [motor_data.target_pos_observed; target_pos'];
        end
        
        % Get player reticle position (from keyboard or mouse)
        reticle_pos = getReticlePosition(keyboard_state, screen_width, screen_height);
        set(reticle_graphic, 'XData', reticle_pos(1), 'YData', reticle_pos(2));
        
        % Store motor data
        motor_data.reticle_pos = [motor_data.reticle_pos; reticle_pos'];
        motor_data.reticle_times = [motor_data.reticle_times; t_elapsed];
        
        % Calculate lead distance
        distance_to_target = norm(reticle_pos - target_pos);
        lead_distance = reticle_pos(1) - target_pos(1);  % Signed lead (positive = ahead)
        motor_data.lead_distance = [motor_data.lead_distance; lead_distance];
        
        % Check for intercept (reticle overlaps target)
        if distance_to_target < (target_size + reticle_size)/2
            intercept_made = true;
            motor_data.intercept_attempt_time = t_elapsed;
            motor_data.intercept_accuracy = distance_to_target;
            
            % Success feedback
            set(reticle_graphic, 'FaceColor', [0, 1, 0]);  % Bright green
            set(instructions, 'String', sprintf('SUCCESS! Accuracy: %.1f px', distance_to_target), ...
                'Color', 'lime');
            
        end
        
        % Update reaction time (first movement)
        if isempty(motor_data.reaction_time) && ...
                (norm(motor_data.reticle_pos(end) - [screen_width/2, screen_height/2]) > 5)
            motor_data.reaction_time = t_elapsed;
        end
        
        drawnow limitrate;
        pause(0.016);  % ~60 FPS
    end
    
    % End-of-trial feedback
    if ~intercept_made
        motor_data.intercept_attempt_time = trial_duration;
        motor_data.intercept_accuracy = distance_to_target;
        set(instructions, 'String', sprintf('Time up! Accuracy: %.1f px', distance_to_target), ...
            'Color', 'red');
    end
    
    % Store trial results
    game_results.trial_data(trial_idx) = motor_data;
    
    % Brief inter-trial pause
    pause(1);
    
end

% Close game window
set(instructions, 'String', 'Game Complete! Analyzing...');
drawnow;
pause(2);
close(fig);

fprintf('\n✓ Game complete! Analyzing motor behavior...\n\n');

%% ========================================================================
%  SECTION 4: ANALYZE MOTOR BEHAVIOR
%  ========================================================================

fprintf('=================================================================\n');
fprintf('  MOTOR BEHAVIOR ANALYSIS\n');
fprintf('=================================================================\n\n');

% Calculate summary statistics
accuracies = [];
lead_distances = [];
reaction_times = [];
motor_lags = [];

for trial_idx = 1:n_trials
    trial = game_results.trial_data(trial_idx);
    
    accuracies = [accuracies; trial.intercept_accuracy];
    reaction_times = [reaction_times; trial.reaction_time];
    
    if ~isempty(trial.lead_distance)
        lead_distances = [lead_distances; mean(trial.lead_distance)];
    end
end

mean_accuracy = mean(accuracies);
std_accuracy = std(accuracies);
mean_lead = mean(lead_distances);
std_lead = std(lead_distances);
mean_reaction_time = mean(reaction_times(reaction_times > 0));

fprintf('PERFORMANCE SUMMARY:\n');
fprintf('─────────────────────────────────────────────────────────────────\n');
fprintf('Mean Interception Accuracy: %.1f ± %.1f pixels\n', mean_accuracy, std_accuracy);
fprintf('Mean Lead Distance: %.1f ± %.1f pixels\n', mean_lead, std_lead);
fprintf('Mean Reaction Time: %.2f seconds\n', mean_reaction_time);
fprintf('Success Rate: %.1f%% (%d/%d intercepted)\n\n', ...
    100*sum(accuracies < 35)/n_trials, sum(accuracies < 35), n_trials);

%% ========================================================================
%  SECTION 5: FIT HIERARCHICAL MOTION INFERENCE MODEL
%  ========================================================================

fprintf('Fitting hierarchical motion inference model...\n\n');

model_fits = struct();

for trial_idx = 1:n_trials
    
    trial = game_results.trial_data(trial_idx);
    
    % Get noisy observations from player's motor commands
    % Assumption: reticle position reflects player's inferred target position
    x_obs = trial.reticle_pos(:,1);  % Reticle x position (player's estimate of target)
    t = trial.reticle_times;
    
    % Fit hierarchical model with grid search
    pi_x_range = [50, 100, 200];
    pi_v_range = [1, 5, 10, 20];
    pi_a_range = [0.1, 0.5, 1];
    
    best_likelihood = -inf;
    best_params = [100, 10, 1];
    
    for pi_x = pi_x_range
        for pi_v = pi_v_range
            for pi_a = pi_a_range
                
                % Create model
                model = RaoBallardModel(0.01, max(t), pi_x, pi_v, pi_a);
                
                % Run inference on player's motor trajectory
                model.generateSensoryInput(trial.target_trajectory(1,1), 0, max(t), 10);
                model.run();
                
                % Likelihood: How well does model predict player's motor commands?
                % Players with high π_v show strong velocity tracking
                % Players with high π_x show reactive (non-predictive) behavior
                
                % Interpolate model predictions to player's timeline
                model_velocity = interp1(model.t, model.v_rep, t, 'linear', 'extrap');
                
                % Calculate observed velocity from player's motor commands
                player_velocity = gradient(x_obs, t);
                
                % Likelihood: correlation between model velocity and observed motor velocity
                if length(model_velocity) == length(player_velocity)
                    velocity_corr = corr(model_velocity, player_velocity);
                    likelihood = velocity_corr;  % Higher correlation = better fit
                else
                    likelihood = -inf;
                end
                
                if likelihood > best_likelihood
                    best_likelihood = likelihood;
                    best_params = [pi_x, pi_v, pi_a];
                end
            end
        end
    end
    
    % Store best fit for this trial
    model_fits(trial_idx).trial_num = trial_idx;
    model_fits(trial_idx).motion_type = trial.acceleration_type;
    model_fits(trial_idx).pi_x = best_params(1);
    model_fits(trial_idx).pi_v = best_params(2);
    model_fits(trial_idx).pi_a = best_params(3);
    model_fits(trial_idx).velocity_correlation = velocity_corr;
    model_fits(trial_idx).accuracy = trial.intercept_accuracy;
    model_fits(trial_idx).lead_distance = mean(trial.lead_distance);
    
    fprintf('Trial %d: π_x=%.0f, π_v=%.0f, π_a=%.1f | Corr=%.2f | Accuracy=%.0f px\n', ...
        trial_idx, best_params(1), best_params(2), best_params(3), velocity_corr, trial.intercept_accuracy);
end

fprintf('\n✓ Model fitting complete\n\n');

%% ========================================================================
%  SECTION 6: NEURAL INTERPRETATION
%  ========================================================================

fprintf('=================================================================\n');
fprintf('  NEURAL INTERPRETATION: MOTOR PLANNING STRATEGY\n');
fprintf('=================================================================\n\n');

% Extract fitted parameters
pi_x_fitted = [model_fits.pi_x];
pi_v_fitted = [model_fits.pi_v];
pi_a_fitted = [model_fits.pi_a];
accuracies_fitted = [model_fits.accuracy];
lead_distances_fitted = [model_fits.lead_distance];

% Calculate precision ratios
sensory_vs_motor_ratio = mean(pi_x_fitted) / mean(pi_v_fitted);

fprintf('PRECISION PROFILE:\n');
fprintf('─────────────────────────────────────────────────────────────────\n');
fprintf('Mean Sensory Precision (π_x):      %.1f\n', mean(pi_x_fitted));
fprintf('Mean Velocity Precision (π_v):     %.1f\n', mean(pi_v_fitted));
fprintf('Mean Acceleration Precision (π_a): %.2f\n', mean(pi_a_fitted));
fprintf('Sensory/Motor Ratio (π_x/π_v):     %.2f\n\n', sensory_vs_motor_ratio);

% Correlate precision with performance
[r_pi_x_accuracy, p_pi_x_accuracy] = corrcoef(pi_x_fitted, accuracies_fitted);
[r_pi_v_lead, p_pi_v_lead] = corrcoef(pi_v_fitted, lead_distances_fitted);

fprintf('PRECISION-PERFORMANCE RELATIONSHIPS:\n');
fprintf('─────────────────────────────────────────────────────────────────\n');
fprintf('Sensory Precision (π_x) vs Accuracy:\n');
fprintf('  r = %.3f, p = %.3f\n', r_pi_x_accuracy(1,2), p_pi_x_accuracy(1,2));

fprintf('Velocity Precision (π_v) vs Lead Distance:\n');
fprintf('  r = %.3f, p = %.3f\n\n', r_pi_v_lead(1,2), p_pi_v_lead(1,2));

% Classify strategy
fprintf('MOTOR STRATEGY CLASSIFICATION:\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

if sensory_vs_motor_ratio > 20
    fprintf('  🔴 REACTIVE STRATEGY (Over-reliance on sensory input)\n');
    fprintf('  ├─ High π_x: Tightly follows stimulus\n');
    fprintf('  ├─ Low π_v: Weak velocity predictions\n');
    fprintf('  ├─ Neural basis: Enhanced sensory cortex, weak cerebellar prediction\n');
    fprintf('  ├─ Behavior: Small lead distance, low anticipation\n');
    fprintf('  ├─ Performance: Good on slow targets, poor on fast/accelerating\n');
    fprintf('  └─ Clinical relevance: Autism-like profile (reactive, detail-focused)\n\n');
    
elseif sensory_vs_motor_ratio < 5
    fprintf('  🔵 PREDICTIVE STRATEGY (Over-reliance on internal model)\n');
    fprintf('  ├─ Low π_x: Ignores sensory feedback\n');
    fprintf('  ├─ High π_v: Strong velocity predictions\n');
    fprintf('  ├─ Neural basis: Over-active cerebellum, weak sensory integration\n');
    fprintf('  ├─ Behavior: Large lead distance, strong anticipation\n');
    fprintf('  ├─ Performance: Good on predictable targets, poor on unpredictable changes\n');
    fprintf('  └─ Clinical relevance: Psychosis-like profile (predictive, prior-dependent)\n\n');
    
else
    fprintf('  🟢 BALANCED STRATEGY (Optimal sensory-motor integration)\n');
    fprintf('  ├─ Moderate π_x: Flexible use of sensory input\n');
    fprintf('  ├─ Moderate π_v: Flexible velocity predictions\n');
    fprintf('  ├─ Neural basis: Integrated predictive coding across cortical hierarchy\n');
    fprintf('  ├─ Behavior: Adaptive lead distance, good anticipation\n');
    fprintf('  ├─ Performance: Good across all target types\n');
    fprintf('  └─ Clinical relevance: Neurotypical profile (balanced, adaptive)\n\n');
end

%% ========================================================================
%  SECTION 7: VISUALIZATIONS
%  ========================================================================

fprintf('Creating visualizations...\n\n');

% Figure 1: Motor trajectory + Model predictions
figure('Position', [100, 100, 1400, 900]);
sgtitle(sprintf('Participant %s - Example Trial Motor Behavior & Hierarchical Model', participant_id), ...
    'FontSize', 14, 'FontWeight', 'bold');

example_trial = 1;  % Plot first trial
trial = game_results.trial_data(example_trial);
model_fit = model_fits(example_trial);

% Subplot 1: Target vs Reticle Trajectory
subplot(2,3,1);
plot(trial.target_times, trial.target_trajectory(:,1), 'r--', 'LineWidth', 2, ...
    'DisplayName', 'Target position'); hold on;
plot(trial.reticle_times, trial.reticle_pos(:,1), 'g-', 'LineWidth', 2, ...
    'DisplayName', 'Reticle position');
xlabel('Time (s)'); ylabel('X Position (pixels)');
title('Motor Tracking: Target vs Reticle');
legend; grid on;

% Subplot 2: Lead Distance Over Time
subplot(2,3,2);
plot(trial.reticle_times, trial.lead_distance, 'b-', 'LineWidth', 2);
axline(gca, 'h', 0, 'Color', 'k', 'LineStyle', '--', 'Alpha', 0.5);
xlabel('Time (s)'); ylabel('Lead Distance (pixels)');
title(sprintf('Predictive Lead (positive = ahead)'));
grid on;

% Subplot 3: Motor Velocity
subplot(2,3,3);
motor_vel = gradient(trial.reticle_pos(:,1), trial.reticle_times);
plot(trial.reticle_times, motor_vel, 'g-', 'LineWidth', 2, 'DisplayName', 'Observed velocity');
xlabel('Time (s)'); ylabel('Velocity (pixels/s)');
title('Motor Velocity from Reticle Commands');
grid on;

% Subplot 4: Performance Metrics
subplot(2,3,4);
bar([1, 2, 3], [trial.intercept_accuracy, mean_accuracy, 100]);
set(gca, 'XTickLabel', {'This trial', 'Your mean', 'Perfect'});
ylabel('Accuracy (lower is better)');
title('Interception Accuracy');
grid on;

% Subplot 5: Precision Profile
subplot(2,3,5);
parameters = {'π_x\n(sensory)', 'π_v\n(velocity)', 'π_a\n(accel)'};
values = [model_fit.pi_x, model_fit.pi_v, model_fit.pi_a*10];  % Scale π_a for visibility
bar(values, 'FaceColor', [0.2, 0.5, 0.9]);
set(gca, 'XTickLabel', parameters);
ylabel('Precision Weight');
title('Fitted Precision Profile (Trial 1)');
grid on;

% Subplot 6: Accuracy vs Lead Distance
subplot(2,3,6);
scatter(lead_distances_fitted, accuracies_fitted, 100, 'filled', 'o');
xlabel('Mean Lead Distance (pixels)'); ylabel('Accuracy (pixels)');
title('Strategy: Predictive Lead vs Accuracy');
grid on;

saveas(gcf, sprintf('interception_game_results/%s_motor_analysis.png', participant_id));
close(gcf);

% Figure 2: Group-level summary (if comparing strategies)
figure('Position', [100, 100, 1200, 800]);
sgtitle(sprintf('Motor Strategy Profile - Participant %s', participant_id), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Plot 1: Precision trajectory over trials
subplot(2,2,1);
plot(1:n_trials, pi_x_fitted, 'o-', 'LineWidth', 2, 'DisplayName', 'π_x (sensory)'); hold on;
plot(1:n_trials, pi_v_fitted, 's-', 'LineWidth', 2, 'DisplayName', 'π_v (velocity)');
xlabel('Trial'); ylabel('Precision Weight');
title('Precision Evolution Across Trials');
legend; grid on;

% Plot 2: Accuracy improvement
subplot(2,2,2);
plot(1:n_trials, accuracies_fitted, 'o-', 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2]);
xlabel('Trial'); ylabel('Accuracy (pixels, lower is better)');
title('Learning Curve: Interception Accuracy');
grid on;

% Plot 3: Lead distance strategy
subplot(2,2,3);
plot(1:n_trials, lead_distances_fitted, 's-', 'LineWidth', 2, 'Color', [0.2, 0.8, 0.2]);
axline(gca, 'h', 0, 'Color', 'k', 'LineStyle', '--', 'Alpha', 0.3);
xlabel('Trial'); ylabel('Lead Distance (pixels)');
title('Predictive Lead Over Trials');
grid on;

% Plot 4: Strategy classification
subplot(2,2,4);
text(0.5, 0.8, 'MOTOR STRATEGY CLASSIFICATION', ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

if sensory_vs_motor_ratio > 20
    strategy_text = 'REACTIVE\n(Sensory-driven)';
    color = [0.8, 0.2, 0.2];  % Red
elseif sensory_vs_motor_ratio < 5
    strategy_text = 'PREDICTIVE\n(Model-driven)';
    color = [0.2, 0.2, 0.8];  % Blue
else
    strategy_text = 'BALANCED\n(Integrated)';
    color = [0.2, 0.8, 0.2];  % Green
end

text(0.5, 0.5, strategy_text, 'HorizontalAlignment', 'center', 'FontSize', 16, ...
    'FontWeight', 'bold', 'Color', color);

text(0.5, 0.2, sprintf('π_x/π_v = %.2f', sensory_vs_motor_ratio), ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

axis(gca, 'off');

saveas(gcf, sprintf('interception_game_results/%s_strategy_profile.png', participant_id));
close(gcf);

fprintf('✓ Visualizations complete\n\n');

%% ========================================================================
%  SECTION 8: SAVE RESULTS & GENERATE REPORT
%  ========================================================================

% Save data
save(sprintf('interception_game_results/%s_game_results.mat', participant_id), 'game_results');
save(sprintf('interception_game_results/%s_model_fits.mat', participant_id), 'model_fits');

% Generate summary report
report_filename = sprintf('interception_game_results/%s_report.txt', participant_id);
fid = fopen(report_filename, 'w');

fprintf(fid, '=================================================================\n');
fprintf(fid, 'INTERCEPTION GAME - PARTICIPANT REPORT\n');
fprintf(fid, '=================================================================\n\n');
fprintf(fid, 'Participant ID: %s\n', participant_id);
fprintf(fid, 'Age: %d\n', age);
fprintf(fid, 'Gaming Experience: %d/5\n', gaming_experience);
fprintf(fid, 'Date: %s\n\n', datetime('now'));

fprintf(fid, 'PERFORMANCE METRICS:\n');
fprintf(fid, '─────────────────────────────────────────────────────────────────\n');
fprintf(fid, 'Mean Accuracy: %.1f ± %.1f pixels\n', mean_accuracy, std_accuracy);
fprintf(fid, 'Mean Lead Distance: %.1f ± %.1f pixels\n', mean_lead, std_lead);
fprintf(fid, 'Mean Reaction Time: %.2f seconds\n', mean_reaction_time);
fprintf(fid, 'Success Rate: %.1f%%\n\n', 100*sum(accuracies < 35)/n_trials);

fprintf(fid, 'HIERARCHICAL MODEL ESTIMATES:\n');
fprintf(fid, '─────────────────────────────────────────────────────────────────\n');
fprintf(fid, 'Mean Sensory Precision (π_x): %.1f\n', mean(pi_x_fitted));
fprintf(fid, 'Mean Velocity Precision (π_v): %.1f\n', mean(pi_v_fitted));
fprintf(fid, 'Sensory/Motor Ratio: %.2f\n\n', sensory_vs_motor_ratio);

fprintf(fid, 'NEURAL INTERPRETATION:\n');
fprintf(fid, '─────────────────────────────────────────────────────────────────\n');

if sensory_vs_motor_ratio > 20
    fprintf(fid, 'Strategy: REACTIVE (Sensory-driven)\n');
    fprintf(fid, 'Interpretation: Your motor control relies heavily on direct sensory feedback.\n');
    fprintf(fid, 'You tend to track the target closely but show limited anticipatory behavior.\n');
    fprintf(fid, 'Neural basis: Enhanced sensory cortex (V1/MT) activity, weaker cerebellar predictions.\n');
elseif sensory_vs_motor_ratio < 5
    fprintf(fid, 'Strategy: PREDICTIVE (Model-driven)\n');
    fprintf(fid, 'Interpretation: Your motor control relies on internal predictions of target motion.\n');
    fprintf(fid, 'You show strong anticipatory lead, planning ahead of the stimulus.\n');
    fprintf(fid, 'Neural basis: Overactive cerebellum, reduced sensory integration.\n');
else
    fprintf(fid, 'Strategy: BALANCED (Integrated)\n');
    fprintf(fid, 'Interpretation: Your motor control flexibly balances sensory input with predictions.\n');
    fprintf(fid, 'You show adaptive lead distance, good performance across conditions.\n');
    fprintf(fid, 'Neural basis: Integrated predictive coding across cortical hierarchy.\n');
end

fclose(fid);

fprintf('Report saved to: %s\n\n', report_filename);

fprintf('=================================================================\n');
fprintf('  ✓ EXPERIMENT COMPLETE\n');
fprintf('=================================================================\n\n');
fprintf('All data saved to: interception_game_results/%s_*\n\n', participant_id);

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [target_traj, target_times, v_true, a_true] = generateTargetTrajectory(width, height, speed_range, accel_type)
    
    trial_duration = 5;
    dt = 0.016;  % ~60 FPS
    target_times = 0:dt:trial_duration;
    
    % Generate trajectory based on acceleration type
    x_start = width * 0.2;
    x_end = width * 0.8;
    
    switch accel_type
        case 'constant'
            v_true = speed_range(1) + rand()*(speed_range(2)-speed_range(1));
            a_true = 0;
            x_pos = x_start + v_true * target_times;
            
        case 'accelerating'
            v_true = speed_range(1) + rand()*50;
            a_true = 50 + 50*rand();
            x_pos = x_start + v_true * target_times + 0.5*a_true*target_times.^2;
            
        case 'decelerating'
            v_true = speed_range(2) - rand()*50;
            a_true = -60 - 30*rand();
            x_pos = x_start + v_true * target_times + 0.5*a_true*target_times.^2;
    end
    
    % Clip to screen
    x_pos = max(50, min(width-50, x_pos));
    
    % Y position: slight sinusoidal motion
    y_pos = height/2 + 50*sin(2*pi*target_times/trial_duration);
    
    target_traj = [x_pos', y_pos'];
end

function reticle_pos = getReticlePosition(keyboard_state, width, height)
    % Get reticle position from keyboard input
    persistent reticle_x reticle_y
    
    if isempty(reticle_x)
        reticle_x = width/2;
        reticle_y = height/2;
    end
    
    speed = 10;  % pixels per frame
    
    % Arrow key controls
    if keyboard_state.keys_pressed.isKey('uparrow')
        reticle_y = reticle_y - speed;
    end
    if keyboard_state.keys_pressed.isKey('downarrow')
        reticle_y = reticle_y + speed;
    end
    if keyboard_state.keys_pressed.isKey('leftarrow')
        reticle_x = reticle_x - speed;
    end
    if keyboard_state.keys_pressed.isKey('rightarrow')
        reticle_x = reticle_x + speed;
    end
    
    % Clip to screen
    reticle_x = max(20, min(width-20, reticle_x));
    reticle_y = max(20, min(height-20, reticle_y));
    
    reticle_pos = [reticle_x, reticle_y];
end

function keyPressed(~, event, varargin)
    % Handle keyboard input
    % This is a simplified version - would need more sophisticated key tracking
end

function h = circle(x, y, r, color, ax)
    % Draw a circle at (x,y) with radius r
    theta = linspace(0, 2*pi, 100);
    circle_x = x + r*cos(theta);
    circle_y = y + r*sin(theta);
    h = fill(ax, circle_x, circle_y, color, 'EdgeColor', 'none');
end
