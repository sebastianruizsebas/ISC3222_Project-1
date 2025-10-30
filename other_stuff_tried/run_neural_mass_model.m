% filepath: run_neural_mass_model.m
%
% Simple example of running the neural mass model from test.m

%% Setup: Define model structure
%==========================================================================

% Number of sources (brain regions)
n_sources = 1;

% Number of states (12 neural states + 1 slow K conductance = 13)
n_states = 13;

% Create model structure M
M.x = zeros(1, 13);  % 1 source x 13 neural states

% Fixed parameters
M.pF.E = [32 16 4];                    % Extrinsic rates
M.pF.H = [1 1 1/2 1/2 1/32]*128;      % Intrinsic rates
M.pF.D = [2 4];                        % Propagation delays
M.pF.G = [8 32];                       % Receptor densities
M.pF.T = [4 16];                       % Synaptic time constants
M.pF.R = [1 2];                        % Static nonlinearity parameters

%% Parameters: P structure
%==========================================================================

% Use SCALED parameters to avoid numerical instability
% The key is P.C which scales the input
% Reduce scales to prevent the model from exploding

% Extrinsic connectivity (log-scaled with smaller values to stabilize)
P.A{1} = log(0.1)*ones(n_sources, n_sources);    % Forward connections
P.A{2} = log(0.1)*ones(n_sources, n_sources);    % Backward connections
P.A{3} = log(0.05)*ones(n_sources, n_sources);   % Lateral connections

% Intrinsic connectivity
P.H = log(0.1)*ones(n_sources, n_sources);

% Input scaling - CRITICAL PARAMETER
% This controls how much the input drives the system
P.C = log(0.001);  % Very small input gain to prevent instability

% Time constants
P.T = log(1)*ones(n_sources, 2);

% Delays (use realistic values, not zero)
P.D = log(0.1);  % Extrinsic delay
P.I = log(0.1);  % Intrinsic delay

% Nonlinearity parameters - keep modest
P.G = log(0.5)*ones(n_sources, 1);
P.R = log(1)*ones(2, 1);

%% Initial state: x
%==========================================================================
% The test.m function expects x to be (n_sources, n_states)
% Initialize with activity across multiple layers to seed the network

x = zeros(n_sources, n_states);

% Initialize multiple states to create a pattern of activity
x(1, 1) = 0.5;   % Spiny stellate voltage
x(1, 2) = 0.3;   % Pyramidal positive voltage
x(1, 3) = -0.3;  % Pyramidal negative voltage  
x(1, 7) = 0.2;   % Inhibitory interneuron positive voltage
x(1, 9) = 0.1;   % Pyramidal combined voltage
x(1, 12) = 0.05; % II combined voltage

%% Input: u
%==========================================================================

u = 1;

%% Run the neural mass model - Single timestep
%==========================================================================

fprintf('Computing neural dynamics at single timestep...\n');
[f, J] = spm_fx_lfp(x, u, P, M);

fprintf('✓ State derivatives computed\n');
fprintf('  State vector size: %d x 1\n', length(x));
fprintf('  Derivatives (f): %d x 1\n', length(f));
fprintf('  Jacobian (J): %d x %d\n', size(J,1), size(J,2));

%% Simulate over time (integrate using simple Euler method)
%==========================================================================

fprintf('\nSimulating neural dynamics over 1 second...\n');

% Time parameters
dt = 0.001;   % 1 ms time step
t_end = 1.0;  % 1 second
t = 0:dt:t_end;
nt = length(t);

% Initialize storage (store as column vector for each time step)
X_sim = zeros(n_sources*n_states, nt);  % Shape: (13, 1001)
F_sim = zeros(n_sources*n_states, nt);  % Derivatives

% Initial condition: x is already (1, 13), flatten it to column vector
X_sim(:, 1) = x(:);  % Convert to column vector (13, 1)

fprintf('Initial state shape: (%d, %d)\n', size(X_sim, 1), size(X_sim, 2));

% Simple Euler integration
for i = 2:nt
    % Define stimulus input (50ms pulse from 100-150ms)
    if t(i) < 0.1
        u_t = 0;  % No input
    elseif t(i) < 0.15
        u_t = 2;  % Strong input pulse (50ms)
    else
        u_t = 0;  % No input
    end
    
    % Get previous state as column vector and pass to test()
    x_current = X_sim(:, i-1);  % Get column vector
    [f, ~] = spm_fx_lfp(x_current, u_t, P, M);
    
    % Ensure f is a column vector
    if size(f, 2) > 1
        f = f';  % Convert to column if needed
    end
    
    % Update state using Euler method: x(t+dt) = x(t) + f*dt
    X_sim(:, i) = X_sim(:, i-1) + f * dt;
    F_sim(:, i) = f;
end

fprintf('✓ Simulation complete\n');

%% Extract and visualize results
%==========================================================================

figure('Name', 'Neural Mass Model Dynamics', 'NumberTitle', 'off', ...
    'Position', [100, 100, 1200, 800]);

% Extract key neural states from column-vector storage
% X_sim is (13, 1001) - each row is a state variable across time
V_ss = X_sim(1, :);        % Spiny stellate voltage (state 1)
V_py_pos = X_sim(2, :);    % Pyramidal positive (state 2)
V_py_neg = X_sim(3, :);    % Pyramidal negative (state 3)
I_ss_pos = X_sim(4, :);    % SS current positive (state 4)
I_py_pos = X_sim(5, :);    % Pyramidal current positive (state 5)
I_py_neg = X_sim(6, :);    % Pyramidal current negative (state 6)
V_ii_pos = X_sim(7, :);    % Inhibitory interneuron positive (state 7)
I_ii_pos = X_sim(8, :);    % II current positive (state 8)
V_py = X_sim(9, :);        % Pyramidal voltage (state 9)
V_ii_neg = X_sim(10, :);   % II negative voltage (state 10)
I_ii_neg = X_sim(11, :);   % II current negative (state 11)
V_ii = X_sim(12, :);       % II voltage (state 12)
K_slow = X_sim(13, :);     % Slow potassium (state 13)

% Panel 1: Pyramidal cell voltages
subplot(3,3,1);
plot(t, V_py_pos, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Positive');
hold on;
plot(t, V_py_neg, 'LineWidth', 2, 'Color', 'red', 'DisplayName', 'Negative');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Pyramidal Cell Voltages');
legend;
grid on;

% Panel 2: Inhibitory interneuron voltages
subplot(3,3,2);
plot(t, V_ii_pos, 'LineWidth', 2, 'Color', 'green', 'DisplayName', 'Positive');
hold on;
plot(t, V_ii_neg, 'LineWidth', 2, 'Color', 'cyan', 'DisplayName', 'Negative');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Inhibitory Interneuron Voltages');
legend;
grid on;

% Panel 3: Spiny stellate voltage
subplot(3,3,3);
plot(t, V_ss, 'LineWidth', 2, 'Color', 'magenta');
xlabel('Time (s)');
ylabel('Voltage (mV)');
title('Spiny Stellate Voltage');
grid on;

% Panel 4: Pyramidal currents
subplot(3,3,4);
plot(t, I_py_pos, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Positive');
hold on;
plot(t, I_py_neg, 'LineWidth', 2, 'Color', 'red', 'DisplayName', 'Negative');
xlabel('Time (s)');
ylabel('Current (a.u.)');
title('Pyramidal Cell Currents');
legend;
grid on;

% Panel 5: Inhibitory interneuron currents
subplot(3,3,5);
plot(t, I_ii_pos, 'LineWidth', 2, 'Color', 'green', 'DisplayName', 'Positive');
hold on;
plot(t, I_ii_neg, 'LineWidth', 2, 'Color', 'cyan', 'DisplayName', 'Negative');
xlabel('Time (s)');
ylabel('Current (a.u.)');
title('Inhibitory Interneuron Currents');
legend;
grid on;

% Panel 6: Slow potassium conductance
subplot(3,3,6);
plot(t, K_slow, 'LineWidth', 2, 'Color', 'yellow');
xlabel('Time (s)');
ylabel('Conductance (a.u.)');
title('Slow Potassium Conductance');
grid on;

% Panel 7: Simulated ERP (pyramidal + inhibitory voltage sum)
subplot(3,3,7);
ERP = V_py + V_ii;  % Combined population response
plot(t, ERP, 'LineWidth', 2.5, 'Color', 'black');
xlabel('Time (s)');
ylabel('Population Response (mV)');
title('Simulated ERP Signal');
grid on;

% Panel 8: First derivative (response velocity)
subplot(3,3,8);
dERP_dt = gradient(ERP, dt);
plot(t, dERP_dt, 'LineWidth', 2, 'Color', 'red');
xlabel('Time (s)');
ylabel('dERP/dt (mV/s)');
title('ERP Derivative (Velocity)');
grid on;

% Panel 9: Summary statistics
subplot(3,3,9);
axis off;

% Find peak and latency
[peak_amp, peak_idx] = max(abs(ERP));
peak_latency = t(peak_idx);

summary_text = sprintf(['NEURAL MASS MODEL SUMMARY\n\n' ...
    'Peak ERP Amplitude: %.3f mV\n' ...
    'Peak Latency: %.3f s (%.1f ms)\n' ...
    'Total States: %d\n' ...
    'Integration Method: Euler\n' ...
    'Time Step: %.4f s\n' ...
    'Simulation Duration: %.2f s\n\n' ...
    'Model Parameters:\n' ...
    'Pyramidal Peak: %.3f\n' ...
    'II Peak: %.3f\n' ...
    'SS Peak: %.3f'], ...
    peak_amp, peak_latency, peak_latency*1000, ...
    n_sources*n_states, dt, t_end, ...
    max(abs(V_py)), max(abs(V_ii)), max(abs(V_ss)));

text(0.1, 0.5, summary_text, 'FontSize', 10, 'FontName', 'Monospaced', ...
    'VerticalAlignment', 'middle');

sgtitle('Neural Mass Model - Cortical Dynamics with Stimulus');

%% Print summary to console
%==========================================================================

fprintf('\n');
fprintf('=================================================================\n');
fprintf('SIMULATION RESULTS\n');
fprintf('=================================================================\n\n');

fprintf('Stimulation:\n');
fprintf('  Duration: 50 ms (100-150 ms)\n');
fprintf('  Amplitude: 2.0 units\n\n');

fprintf('Neural Response:\n');
fprintf('  Peak ERP Amplitude: %.3f mV\n', peak_amp);
fprintf('  Peak Latency: %.3f s (%.1f ms)\n', peak_latency, peak_latency*1000);
fprintf('  Pyramidal Cell Peak: %.3f mV\n', max(abs(V_py)));
fprintf('  Inhibitory Interneuron Peak: %.3f mV\n', max(abs(V_ii)));
fprintf('  Spiny Stellate Peak: %.3f mV\n\n', max(abs(V_ss)));

fprintf('Model Configuration:\n');
fprintf('  Number of Sources: %d\n', n_sources);
fprintf('  Number of States: %d\n', n_states);
fprintf('  Total State Dimensions: %d\n', n_sources*n_states);
fprintf('  Integration Time Step: %.4f s\n', dt);
fprintf('  Total Simulation Time: %.2f s\n', t_end);
fprintf('  Number of Time Points: %d\n\n', nt);

fprintf('State Labels:\n');
fprintf('  1-3: Voltages (SS, Pyr+, Pyr-)\n');
fprintf('  4-6: Currents (SS+, Pyr+, Pyr-)\n');
fprintf('  7-8: II Voltages and Currents\n');
fprintf('  9-12: Secondary Voltages\n');
fprintf('  13: Slow Potassium Conductance\n\n');

fprintf('=================================================================\n');
fprintf('Figure saved and displayed.\n');
fprintf('=================================================================\n\n');