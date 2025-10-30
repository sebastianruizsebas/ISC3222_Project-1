%% Neural Mass Model - Custom Scenario Builder
%
% Interactive script to create custom parameter sets based on your experimental conditions
% Use this to:
%   - Test hypotheses about neural mechanisms
%   - Calibrate model to your specific task
%   - Generate predictions for new conditions
%==========================================================================

clear all; close all; clc

fprintf('\n');
fprintf('===========================================================\n');
fprintf('NEURAL MASS MODEL - CUSTOM SCENARIO BUILDER\n');
fprintf('===========================================================\n\n');

%% Define Your Custom Scenario
%==========================================================================

% Scenario name
scenario_name = 'Motor Interception Task';

% Description of your experimental condition
description = ['Neural activity during visuomotor interception task. ' ...
    'Predict neural response to target motion cues combined with ' ...
    'motor feedback from participant actions.'];

% Task parameters (modify as needed)
%--------------------------------------------------------------------------

% 1. Stimulus timing
stimulus_onset = 0.05;      % When does stimulus appear? (seconds)
stimulus_duration = 0.05;   % How long is it present?
stimulus_intensity = 0.8;   % Relative strength (0-1)

% 2. Expected response characteristics
% Based on your interception game, what neural response do you expect?
%
% Possibilities:
%   - Fast visomotor response: ~100-150ms (sensory guidance)
%   - Slow feedback response: ~300-400ms (error correction)
%   - Conflict response: ~150-250ms (motor-sensory mismatch)
%   - Predictive component: ~200-300ms (anticipatory)

expected_latency = 150;     % Expected peak latency (ms)
expected_amplitude = 0.15;  % Expected peak amplitude (mV)
response_type = 'fast';     % 'fast', 'slow', 'conflict', 'predictive'

% 3. Task demands (affect connectivity)
require_attention = true;   % Does task require focused attention?
require_prediction = false; % Does task require prediction?
require_error_correction = true; % Does task need motor error feedback?

% 4. Participant state
arousal_level = 0.7;        % 0=drowsy, 1=highly alert
difficulty = 0.6;           % 0=easy, 1=very difficult

%% Generate Parameters Based on Task
%==========================================================================

fprintf('Generating parameters for: %s\n', scenario_name);
fprintf('Description: %s\n\n', description);

n_sources = 1;
n_states = 13;

M.x = zeros(1, 13);
M.pF.E = [32 16 4];
M.pF.H = [1 1 1/2 1/2 1/32]*128;
M.pF.D = [2 4];
M.pF.G = [8 32];
M.pF.T = [4 16];
M.pF.R = [1 2];

P = struct();

% Set connectivity based on response type and task demands
%--------------------------------------------------------------------------

fprintf('Setting connectivity parameters:\n');

switch response_type
    case 'fast'
        % Early sensory-driven response
        % Strong forward, weak backward
        forward_strength = 0.18;
        backward_strength = 0.08;
        fprintf('  Response type: Fast sensory-driven\n');
        
    case 'slow'
        % Late attention-dependent response
        % Strong forward and backward
        forward_strength = 0.20;
        backward_strength = 0.20;
        fprintf('  Response type: Slow attention-dependent\n');
        
    case 'conflict'
        % Motor-sensory conflict (mismatch)
        % Very strong forward (prediction error)
        forward_strength = 0.25;
        backward_strength = 0.15;
        fprintf('  Response type: Motor-sensory conflict\n');
        
    case 'predictive'
        % Anticipatory response (predictive)
        % Strong backward (expected future state)
        forward_strength = 0.12;
        backward_strength = 0.22;
        fprintf('  Response type: Predictive anticipation\n');
        
    otherwise
        % Default: balanced
        forward_strength = 0.15;
        backward_strength = 0.12;
        fprintf('  Response type: Default balanced\n');
end

% Modulate by attention and difficulty
if require_attention
    backward_strength = backward_strength * (0.8 + 0.4*arousal_level);
    fprintf('  Attention effect: Backward +%.0f%%\n', (0.8 + 0.4*arousal_level - 1)*100);
end

if require_error_correction
    forward_strength = forward_strength * (1.0 + 0.3*difficulty);
    fprintf('  Error correction: Forward +%.0f%%\n', 0.3*difficulty*100);
end

% Set values
P.A{1} = log(forward_strength)*ones(n_sources, n_sources);
P.A{2} = log(backward_strength)*ones(n_sources, n_sources);
P.A{3} = log(forward_strength*0.4)*ones(n_sources, n_sources);

fprintf('  Forward (A1): %.3f\n', forward_strength);
fprintf('  Backward (A2): %.3f\n', backward_strength);
fprintf('  Lateral (A3): %.3f\n\n', forward_strength*0.4);

% Intrinsic connectivity
P.H = log(0.05 + 0.1*arousal_level)*ones(n_sources, n_sources);

% Input scaling - based on stimulus intensity
input_gain = 0.0001 + stimulus_intensity * 0.012;
P.C = log(input_gain);

% Time constants - faster for fast responses, slower for late responses
if strcmp(response_type, 'fast')
    time_scale = 0.85;  % Faster
elseif strcmp(response_type, 'slow')
    time_scale = 1.15;  % Slower
else
    time_scale = 1.0;   % Standard
end
P.T = log(time_scale)*ones(n_sources, 2);

% Delays
P.D = log(0.08)*ones(1, 1);
P.I = log(0.08)*ones(1, 1);

% Receptor densities
P.G = log(0.5 + 0.3*difficulty)*ones(n_sources, 1);
P.R = log(1 + 0.2*difficulty)*ones(2, 1);

% Initial state - activity depends on task
x = zeros(n_sources, n_states);
if arousal_level > 0.5
    x(1, [1 2 7 9 12]) = arousal_level * [0.3 0.2 0.1 0.1 0.05];
end

fprintf('Input parameters:\n');
fprintf('  Stimulus onset: %.0f ms\n', stimulus_onset*1000);
fprintf('  Stimulus duration: %.0f ms\n', stimulus_duration*1000);
fprintf('  Stimulus intensity: %.2f\n', stimulus_intensity);
fprintf('  Input gain (C): %.5f\n', exp(P.C));
fprintf('  Arousal level: %.2f\n', arousal_level);
fprintf('  Difficulty: %.2f\n\n', difficulty);

%% Run Simulation
%==========================================================================

fprintf('Running simulation...\n');

u_baseline = 0.01;

% Time integration
dt = 0.001;
t_end = 1.0;
t = 0:dt:t_end;
nt = length(t);

X_sim = zeros(n_states, nt);
X_sim(:, 1) = x(:);
U_trace = zeros(1, nt);

for i = 2:nt
    % Define stimulus input
    if t(i) >= stimulus_onset && t(i) < (stimulus_onset + stimulus_duration)
        u_t = u_baseline + stimulus_intensity * exp(P.C);
    else
        u_t = u_baseline;
    end
    
    U_trace(i) = u_t;
    
    % Add motor feedback for error correction (if enabled)
    if require_error_correction && t(i) > 0.15
        % Simulate proprioceptive/motor feedback after initial response
        motor_feedback = 0.3 * sin(2*pi*2*(t(i)-0.15));  % 2Hz feedback oscillation
        u_t = u_t + motor_feedback * 0.01;
    end
    
    x_current = X_sim(:, i-1);
    [f, ~] = test(x_current, u_t, P, M);
    if size(f, 2) > 1
        f = f';
    end
    X_sim(:, i) = X_sim(:, i-1) + f * dt;
end

fprintf('✓ Simulation complete\n\n');

% Extract response
V_py = X_sim(9, :);
V_ii = X_sim(12, :);
ERP = V_py + V_ii;

[peak_amp, peak_idx] = max(abs(ERP));
peak_latency = t(peak_idx);

%% Display Results
%==========================================================================

fprintf('===========================================================\n');
fprintf('RESULTS\n');
fprintf('===========================================================\n\n');

fprintf('Peak Response:\n');
fprintf('  Measured latency: %.1f ms\n', peak_latency*1000);
fprintf('  Expected latency: %.0f ms\n', expected_latency);
fprintf('  Latency match: %s\n\n', ...
    iif(abs(peak_latency*1000 - expected_latency) < 50, '✓ GOOD', '✗ MISMATCH'));

fprintf('  Measured amplitude: %.4f mV\n', peak_amp);
fprintf('  Expected amplitude: %.4f mV\n', expected_amplitude);
fprintf('  Amplitude match: %s\n\n', ...
    iif(peak_amp > expected_amplitude*0.5 && peak_amp < expected_amplitude*2.0, ...
        '✓ REASONABLE', '✗ MISMATCH'));

fprintf('Component Analysis:\n');
fprintf('  Pyramidal peak: %.4f mV\n', max(abs(V_py)));
fprintf('  II peak: %.4f mV\n', max(abs(V_ii)));
fprintf('  Ratio (Pyr/II): %.2f\n', max(abs(V_py)) / max(abs(V_ii)));

%% Create Visualization
%==========================================================================

fig = figure('Name', scenario_name, 'NumberTitle', 'off', 'Position', [100, 100, 1400, 800]);

% Panel 1: Input stimulus
subplot(3, 3, 1);
plot(t*1000, U_trace, 'k-', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Input');
title('Stimulus Input');
xlim([0 1000]);
grid on;

% Panel 2: ERP
subplot(3, 3, 2);
plot(t*1000, ERP, 'b-', 'LineWidth', 2.5);
hold on;
plot(peak_latency*1000, peak_amp, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('ERP (mV)');
title(sprintf('ERP Response (Peak: %.1f ms, %.4f mV)', peak_latency*1000, peak_amp));
xlim([0 1000]);
grid on;

% Panel 3: Components
subplot(3, 3, 3);
plot(t*1000, V_py, 'b-', 'LineWidth', 2, 'DisplayName', 'Pyramidal');
hold on;
plot(t*1000, V_ii, 'r-', 'LineWidth', 2, 'DisplayName', 'II');
xlabel('Time (ms)');
ylabel('Voltage (mV)');
title('Neural Components');
xlim([0 1000]);
legend;
grid on;

% Panel 4-6: State evolution
state_indices = [1 2 4 5 7 8];
state_names = {'SS Volt', 'Pyr+ Volt', 'SS Curr', 'Pyr+ Curr', 'II+ Volt', 'II+ Curr'};

for idx = 1:3
    subplot(3, 3, 3+idx);
    state_num = state_indices(idx);
    plot(t*1000, X_sim(state_num, :), 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('State');
    title(state_names{idx});
    xlim([0 1000]);
    grid on;
end

for idx = 4:6
    subplot(3, 3, 3+idx);
    state_num = state_indices(idx);
    plot(t*1000, X_sim(state_num, :), 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('State');
    title(state_names{idx});
    xlim([0 1000]);
    grid on;
end

% Panel 9: Parameter summary
subplot(3, 3, 9);
axis off;
summary_text = sprintf(['SCENARIO PARAMETERS\n\n' ...
    'Task: %s\n' ...
    'Response: %s\n' ...
    'Forward: %.3f\n' ...
    'Backward: %.3f\n' ...
    'Input: %.5f\n' ...
    'Arousal: %.2f\n' ...
    'Difficulty: %.2f'], ...
    scenario_name, response_type, forward_strength, backward_strength, ...
    exp(P.C), arousal_level, difficulty);
text(0.1, 0.5, summary_text, 'FontSize', 9, 'FontName', 'Monospaced', ...
    'VerticalAlignment', 'middle');

sgtitle(sprintf('Neural Mass Model: %s', scenario_name), 'FontSize', 12, 'FontWeight', 'bold');

%% Save Results
%==========================================================================

fprintf('\nSaving results...\n');

output_dir = 'neural_mass_custom';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save figure
saveas(fig, fullfile(output_dir, [scenario_name '_response.png']));

% Save data
results.name = scenario_name;
results.description = description;
results.response_type = response_type;
results.parameters = P;
results.initial_state = x;
results.t = t;
results.X_sim = X_sim;
results.ERP = ERP;
results.peak_latency = peak_latency;
results.peak_amplitude = peak_amp;
results.expected_latency = expected_latency;
results.expected_amplitude = expected_amplitude;

save(fullfile(output_dir, [scenario_name '_results.mat']), 'results');

fprintf('✓ Results saved to: %s\n', output_dir);
fprintf('  - Figure: %s_response.png\n', scenario_name);
fprintf('  - Data: %s_results.mat\n\n', scenario_name);

%% Helper function for conditional text
%==========================================================================

function result = iif(condition, if_true, if_false)
    % Inline if: iif(condition, value_if_true, value_if_false)
    if condition
        result = if_true;
    else
        result = if_false;
    end
end

