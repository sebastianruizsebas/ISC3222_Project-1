% filepath: hierarchical_motion_inference_2D_SPATIOTEMPORAL.m
%
% EXTEND RAO & BALLARD TO 2D SPATIOTEMPORAL MOTION
% This shows how the predictive coding model learns oriented motion filters,
% similar to V1 complex cells and MT neurons
%

% Expand layer dimensions for 2D motion
n_L1 = 8;   % Level 1: x, y, vx, vy, ax, ay, + bias terms
n_L2 = 6;   % Level 2: vx, vy, ax, ay, + basis functions
n_L3 = 3;   % Level 3: ax, ay, + global context

% Generate 2D polynomial motion with rotation
% Create circular trajectory with accelerating angular velocity

theta = zeros(1, N);
omega = zeros(1, N);
alpha = zeros(1, N);

% Phase 1: Slow rotation
theta(phase1_mask) = 0.5 * t(phase1_mask);
omega(phase1_mask) = 0.5;
alpha(phase1_mask) = 0;

% Phase 2: Accelerating rotation
theta(phase2_mask) = 0.5*5 + 0.5*(t(phase2_mask)-5) + 0.5*(t(phase2_mask)-5).^2;
omega(phase2_mask) = 0.5 + 1.0*(t(phase2_mask)-5);
alpha(phase2_mask) = 1.0;

% Position on circle
r = 2.0;  % Circle radius
x_2d = r * cos(theta);
y_2d = r * sin(theta);

% Velocity (tangent to circle)
vx_2d = -r * omega .* sin(theta);
vy_2d = r * omega .* cos(theta);

% Acceleration (centripetal + tangential)
ax_2d = -r * alpha .* sin(theta) - r * (omega .^ 2) .* cos(theta);
ay_2d = r * alpha .* cos(theta) - r * (omega .^ 2) .* sin(theta);

% Sensory input: 2D representation
R_L1(:,1) = x_2d';
R_L1(:,2) = y_2d';
R_L1(:,3) = vx_2d';
R_L1(:,4) = vy_2d';
R_L1(:,5) = ax_2d';
R_L1(:,6) = ay_2d';
R_L1(:,7:8) = 0.1*randn(N, 2);  % Bias neurons

fprintf('2D SPATIOTEMPORAL MOTION:\n');
fprintf('  Circular trajectory with angular acceleration\n');
fprintf('  Position: (x,y) on circle of radius %.2f m\n', r);
fprintf('  Radius of motion: %.4f to %.4f m\n', min(sqrt(x_2d.^2+y_2d.^2)), max(sqrt(x_2d.^2+y_2d.^2)));
fprintf('  Angular velocity range: [%.4f, %.4f] rad/s\n', min(omega), max(omega));
fprintf('  Angular acceleration: %.4f rad/s²\n\n', max(alpha));
