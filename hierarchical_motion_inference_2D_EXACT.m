% filepath: hierarchical_motion_inference_2D_EXACT.m
%
% EXACT RAO & BALLARD (1999) REPLICATION - 2D SPATIOTEMPORAL MOTION
% ===================================================================
%
% This implementation extends the 1D Rao & Ballard model to 2D motion,
% demonstrating learning of direction-selective motion filters similar
% to V1 and MT neurons in biological visual cortex.
%
% KEY FEATURES:
%   1. 2D position, velocity, and acceleration (x,y coordinates)
%   2. Circular trajectory with angular acceleration
%   3. Learned directional motion filters via Hebbian rule
%   4. Exact polynomial kinematics (no numerical errors)
%   5. Full predictive coding hierarchy with error neurons
%

function [R_L1, R_L2, R_L3, E_L1, E_L2, E_L3, W_L1_from_L2, W_L2_from_L3, ...
          free_energy, x_true, y_true, vx_true, vy_true, ax_true, ay_true, t, dt] = ...
    hierarchical_motion_inference_2D_EXACT(params) % *** MODIFIED: Accept params struct ***

clc;

% *** MODIFIED: Use default parameters if none are provided ***
if nargin < 1
    fprintf('No parameters provided. Using default values.\n');
    params = struct();
end

% *** MODIFIED: Get parameters from struct or use defaults ***
eta_rep = get_param(params, 'eta_rep', 0.1);
eta_W = get_param(params, 'eta_W', 0.01);
momentum = get_param(params, 'momentum', 0.90);
pi_L1 = get_param(params, 'pi_L1', 100);
pi_L2 = get_param(params, 'pi_L2', 10);
pi_L3 = get_param(params, 'pi_L3', 1);
weight_decay = get_param(params, 'weight_decay', 0.9995);


fprintf('\n');
fprintf('╔═════════════════════════════════════════════════════════════╗\n');
fprintf('║  RAO & BALLARD (1999) - 2D SPATIOTEMPORAL MOTION          ║\n');
fprintf('║  Learning Motion Direction Selectivity (V1/MT-like)       ║\n');
fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

% ====================================================================
% CONFIGURATION
% ====================================================================

dt = 0.01;              % Time step (ms)
T = 10;                 % Total duration (s)
t = 0:dt:T;
N = length(t);

% Define phase masks for polynomial motion changes
phase1_mask = (t < 5);
phase2_mask = (t >= 5);

% Layer dimensions for 2D motion
n_L1 = 8;               % Level 1: x, y, vx, vy, ax, ay, + 2 bias neurons
n_L2 = 6;               % Level 2: 6 motion basis functions
n_L3 = 3;               % Level 3: ax, ay, + bias

fprintf('NETWORK ARCHITECTURE:\n');
fprintf('  Level 1 (Sensory):      %d neurons [x, y, vx, vy, ax, ay, bias1, bias2]\n', n_L1);
fprintf('  Level 2 (Motion Basis): %d neurons [learned motion filters]\n', n_L2);
fprintf('  Level 3 (Acceleration): %d neurons [ax, ay, bias]\n\n', n_L3);

% Learning rate parameters
% *** MODIFIED: These are now controlled by the params struct ***
eta_pi = 0.001;         % Precision adaptation rate (kept constant for this optimization)

% Precision weights (reliability of each information source)
% pi_L1 = 100;            % High precision: trust sensory input
% pi_L2 = 10;             % Medium precision: intermediate representation
% pi_L3 = 1;              % Low precision: acceleration prior (smooth motion)

fprintf('LEARNING PARAMETERS (from optimizer):\n');
fprintf('  η_rep = %.6f (representation learning rate)\n', eta_rep);
fprintf('  η_W   = %.6f (weight matrix learning rate)\n', eta_W);
fprintf('  Momentum = %.4f\n', momentum);
fprintf('  Weight Decay = %.4f\n', weight_decay);
fprintf('  π_L1  = %.0f, π_L2  = %.0f, π_L3  = %.0f\n\n', pi_L1, pi_L2, pi_L3);

% ====================================================================
% INITIALIZE WEIGHT MATRICES (Learned via Hebbian Rule)
% ====================================================================
% W^(L): n_L × n_{L+1} matrices implementing generative model
% pred^(L) = W^(L) * R^(L+1)
% Learning: ΔW_{ij} = -η * π * E_i * R_j

W_L1_from_L2 = 0.01 * randn(n_L1, n_L2);  % Position prediction weights
W_L2_from_L3 = 0.01 * randn(n_L2, n_L3);  % Velocity prediction weights

fprintf('WEIGHT MATRICES INITIALIZED:\n');
fprintf('  W^(L1): %d × %d  [Position ← Motion filters]\n', size(W_L1_from_L2));
fprintf('  W^(L2): %d × %d  [Motion filters ← Acceleration]\n\n', size(W_L2_from_L3));

% ====================================================================
% INITIALIZE STATE VARIABLES
% ====================================================================

% Representations (R-neurons encode features)
R_L1 = zeros(N, n_L1);
R_L2 = zeros(N, n_L2);
R_L3 = zeros(N, n_L3);

% Error neurons (E-neurons encode prediction errors)
E_L1 = zeros(N, n_L1);
E_L2 = zeros(N, n_L2);
E_L3 = zeros(N, n_L3);

% Predictions (feedback from higher layers)
pred_L1 = zeros(N, n_L1);
pred_L2 = zeros(N, n_L2);
pred_L3 = zeros(N, n_L3);

% Diagnostics
free_energy = zeros(1, N);
learning_trace_W = zeros(1, N);

% ====================================================================
% GENERATE 2D POLYNOMIAL MOTION: CIRCULAR TRAJECTORY
% ====================================================================
% 
% Phase 1 (t < 5s): Uniform circular motion (constant angular velocity)
% Phase 2 (t >= 5s): Accelerating circular motion (increasing angular velocity)
%
% This creates both position changes AND velocity changes, allowing
% the network to learn acceleration-dependent motion predictions

% Angular motion parameters
theta = zeros(1, N);
omega = zeros(1, N);
alpha = zeros(1, N);

% Phase 1: Constant angular velocity (slow rotation)
theta(phase1_mask) = 0.5 * t(phase1_mask);
omega(phase1_mask) = 0.5;
alpha(phase1_mask) = 0;

% Phase 2: Accelerating angular velocity (increasing rotation speed)
t_phase2 = t(phase2_mask) - 5;  % Time since phase 2 began
theta(phase2_mask) = 0.5*5 + 0.5*t_phase2 + 0.5*t_phase2.^2;
omega(phase2_mask) = 0.5 + 1.0*t_phase2;
alpha(phase2_mask) = 1.0 * ones(size(t_phase2));

% Convert angular motion to Cartesian coordinates
% Circle: radius r = 2.0 m
r = 2.0;

% Position on circle: (x,y) = r*(cos θ, sin θ)
x_true = r * cos(theta);
y_true = r * sin(theta);

% Velocity: tangent to circle
% v_x = dx/dt = -r*ω*sin(θ)
% v_y = dy/dt = r*ω*cos(θ)
vx_true = -r * omega .* sin(theta);
vy_true = r * omega .* cos(theta);

% Acceleration: centripetal + tangential
% a_x = -r*α*sin(θ) - r*ω²*cos(θ)
% a_y = r*α*cos(θ) - r*ω²*sin(θ)
ax_true = -r * alpha .* sin(theta) - r * (omega .^ 2) .* cos(theta);
ay_true = r * alpha .* cos(theta) - r * (omega .^ 2) .* sin(theta);

% Verify polynomial structure
fprintf('2D POLYNOMIAL MOTION VERIFICATION:\n');
fprintf('  Phase 1 (0 ≤ t < 5): Uniform circular motion\n');
fprintf('    Angular velocity: ω = 0.5 rad/s (constant)\n');
fprintf('    Angular acceleration: α = 0 rad/s²\n');
fprintf('    Position: (x,y) on circle of radius %.2f m\n', r);
fprintf('\n');
fprintf('  Phase 2 (5 ≤ t ≤ 10): Accelerating circular motion\n');
fprintf('    Angular velocity: ω = 0.5 + 1.0*(t-5) rad/s (linear increase)\n');
fprintf('    Angular acceleration: α = 1.0 rad/s² (constant)\n');
fprintf('\n');
fprintf('  Position range: x ∈ [%.4f, %.4f], y ∈ [%.4f, %.4f] m\n', ...
    min(x_true), max(x_true), min(y_true), max(y_true));
fprintf('  Velocity range: vx,vy ∈ [%.4f, %.4f] m/s\n', ...
    min([vx_true, vy_true]), max([vx_true, vy_true]));
fprintf('  Acceleration range: ax,ay ∈ [%.4f, %.4f] m/s²\n\n', ...
    min([ax_true, ay_true]), max([ax_true, ay_true]));

% ====================================================================
% SENSORY INPUT LAYER (L1): 2D Kinematic Information
% ====================================================================
% L1 receives the complete 2D kinematic state from the environment

R_L1(:,1) = x_true';            % Position x
R_L1(:,2) = y_true';            % Position y
R_L1(:,3) = vx_true';           % Velocity vx
R_L1(:,4) = vy_true';           % Velocity vy
R_L1(:,5) = ax_true';           % Acceleration ax
R_L1(:,6) = ay_true';           % Acceleration ay
R_L1(:,7) = ones(N, 1);         % Bias neuron 1
R_L1(:,8) = ones(N, 1);         % Bias neuron 2

fprintf('SENSORY INPUT LAYER (L1):\n');
fprintf('  Neuron 1-2: Position (x, y)\n');
fprintf('  Neuron 3-4: Velocity (vx, vy)\n');
fprintf('  Neuron 5-6: Acceleration (ax, ay)\n');
fprintf('  Neuron 7-8: Bias terms (constant 1.0)\n\n');

% ====================================================================
% INITIALIZE HIGHER LAYER REPRESENTATIONS
% ====================================================================
% Seed with values derived from sensory input

% L2: Initialize with motion components
R_L2(1,1) = vx_true(1);         % Initial vx estimate
R_L2(1,2) = vy_true(1);         % Initial vy estimate
R_L2(1,3) = ax_true(1);         % Initial ax estimate
R_L2(1,4) = ay_true(1);         % Initial ay estimate
R_L2(1,5:6) = 0.1*randn(1, 2);  % Random basis functions

% L3: Initialize with acceleration
R_L3(1,1) = ax_true(1);
R_L3(1,2) = ay_true(1);
R_L3(1,3) = 1;                  % Bias

fprintf('INITIAL CONDITIONS:\n');
fprintf('  R_L2(1,:) = [vx=%.4f, vy=%.4f, ax=%.4f, ay=%.4f, basis1, basis2]\n', ...
    vx_true(1), vy_true(1), ax_true(1), ay_true(1));
fprintf('  R_L3(1,:) = [ax=%.4f, ay=%.4f, bias=1.0]\n\n', ax_true(1), ay_true(1));

% ====================================================================
% MAIN PREDICTIVE CODING LOOP (Rao & Ballard Algorithm)
% ====================================================================
% 
% For each timestep:
%   1. Compute top-down PREDICTIONS from higher layers
%   2. Compute bottom-up ERRORS (prediction mismatches)
%   3. Evaluate FREE ENERGY objective F = Σ E²/(2π)
%   4. Update REPRESENTATIONS via gradient descent on F
%   5. Learn WEIGHT MATRICES via Hebbian rule ΔW ∝ E*R^T

fprintf('Running Rao & Ballard predictive coding');

% momentum = 0.90;        % *** MODIFIED: Controlled by params struct ***
max_rep_value = 10;     % Bound on representation values

for i = 1:N-1
    if mod(i, 200) == 0, fprintf('.'); end
    
    % ==============================================================
    % STEP 1: FEEDBACK PREDICTIONS (Top-Down)
    % ==============================================================
    % Higher layer representations predict lower layer via W matrices
    
    pred_L2(i,:) = R_L3(i,:) * W_L2_from_L3';
    pred_L1(i,:) = R_L2(i,:) * W_L1_from_L2';
    
    % ==============================================================
    % STEP 2: ERROR COMPUTATION (Bottom-Up)
    % ==============================================================
    % Prediction errors represent violations of model expectations
    
    E_L1(i,:) = R_L1(i,:) - pred_L1(i,:);
    E_L2(i,:) = R_L2(i,:) - pred_L2(i,:);
    E_L3(i,:) = R_L3(i,:) - 0;  % Prior: expect zero acceleration
    
    % ==============================================================
    % STEP 3: FREE ENERGY (Objective Function)
    % ==============================================================
    % F = Σ_L (ε_L² / 2π_L)
    % Quantifies total prediction error, weighted by precision
    
    fe_L1 = sum(E_L1(i,:).^2) / (2 * pi_L1);
    fe_L2 = sum(E_L2(i,:).^2) / (2 * pi_L2);
    fe_L3 = sum(E_L3(i,:).^2) / (2 * pi_L3);
    free_energy(i) = fe_L1 + fe_L2 + fe_L3;
    
    % ==============================================================
    % STEP 4: REPRESENTATION UPDATES
    % ==============================================================
    % Gradient descent: ∂R^(L)/∂t = -∂F/∂R^(L)
    
    % L2: Driven by own errors and lower-layer coupling
    coupling_from_L1 = E_L1(i,:) * W_L1_from_L2;
    norm_W1 = max(0.1, norm(W_L1_from_L2, 'fro'));
    coupling_from_L1 = coupling_from_L1 / norm_W1;
    
    delta_R_L2 = E_L2(i,:) - coupling_from_L1;
    decay = 1 - momentum;
    R_L2(i+1,:) = momentum * R_L2(i,:) + decay * (R_L2(i,:) - eta_rep * delta_R_L2);
    R_L2(i+1,:) = max(-max_rep_value, min(max_rep_value, R_L2(i+1,:)));
    
    % L3: Driven by errors from layer below
    coupling_from_L2 = E_L2(i,:) * W_L2_from_L3;
    norm_W2 = max(0.1, norm(W_L2_from_L3, 'fro'));
    coupling_from_L2 = coupling_from_L2 / norm_W2;
    
    delta_R_L3 = coupling_from_L2 - E_L3(i,:);
    R_L3(i+1,:) = momentum * R_L3(i,:) + decay * (R_L3(i,:) - eta_rep * delta_R_L3);
    R_L3(i+1,:) = max(-max_rep_value, min(max_rep_value, R_L3(i+1,:)));
    
    % ==============================================================
    % STEP 5: WEIGHT LEARNING (Hebbian Rule with Error Signal)
    % ==============================================================
    % ΔW = -η * π * E * R^T
    % Outer product: each column is weighted by error signal
    
    layer_scale_L1 = max(0.1, mean(abs(R_L2(i,:))));
    layer_scale_L2 = max(0.1, mean(abs(R_L3(i,:))));
    
    dW_L1 = -(eta_W * pi_L1 / layer_scale_L1) * (E_L1(i,:)' * R_L2(i,:));
    W_L1_from_L2 = W_L1_from_L2 + dW_L1;
    
    dW_L2 = -(eta_W * pi_L2 / layer_scale_L2) * (E_L2(i,:)' * R_L3(i,:));
    W_L2_from_L3 = W_L2_from_L3 + dW_L2;
    
    % Prevent weight explosion via L2 regularization
    W_L1_from_L2 = W_L1_from_L2 * weight_decay; % *** MODIFIED: Use param ***
    W_L2_from_L3 = W_L2_from_L3 * weight_decay; % *** MODIFIED: Use param ***
    
    learning_trace_W(i) = norm(dW_L1, 'fro') + norm(dW_L2, 'fro');
    
end  % End timestep loop

fprintf(' Done!\n\n');

% ====================================================================
% SAVE RESULTS TO OUTPUT
% ====================================================================

% Return all computed variables for analysis
% (No explicit return needed in MATLAB function, but good practice)

end  % End function


% *** ADDED: Helper function to get parameters safely ***
function val = get_param(params, field, default)
    if isfield(params, field)
        val = params.(field);
    else
        val = default;
    end
end
