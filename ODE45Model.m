classdef ODE45Model < PredictiveCodingModel
    % ODE45MODEL Implements adaptive Runge-Kutta for predictive coding
    % Uses MATLAB's ode45 solver for high-precision simulation
    
    properties
        params          % Structure with additional parameters
        options         % ODE45 solver options
    end
    
    methods
        function obj = ODE45Model(dt, T, sigma_x, sigma_v)
            % Call superclass constructor
            obj@PredictiveCodingModel(dt, T, sigma_x, sigma_v);
            
            % ODE45 solver options (high precision)
            obj.options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 0.1);
            
            % Store parameters for ODE function
            obj.params = struct();
            obj.params.sigma_x = sigma_x;
            obj.params.sigma_v = sigma_v;
            obj.params.mu_v = 0;
            obj.params.noise_std = 0.05;
            obj.params.change_time = T / 2;
            obj.params.v_before = 2.0;
            obj.params.v_after = -1.0;
        end
        
        function generateSensoryInput(obj, noise_std, v_before, v_after, change_time)
            % Override parent method to store parameters
            if nargin >= 2
                obj.params.noise_std = noise_std;
            end
            if nargin >= 3
                obj.params.v_before = v_before;
            end
            if nargin >= 4
                obj.params.v_after = v_after;
            end
            if nargin >= 5
                obj.params.change_time = change_time;
            end
            
            % Call parent method
            generateSensoryInput@PredictiveCodingModel(obj, obj.params.noise_std, ...
                obj.params.v_before, obj.params.v_after, obj.params.change_time);
        end
        
        function run(obj)
            % ODE45 adaptive integration
            fprintf('Running ODE45 simulation (adaptive Runge-Kutta)...\n');
            
            % Define dynamics function
            ode_fn = @(t, y) obj.dynamics(t, y);
            
            % Time span
            t_span = [obj.t(1), obj.t(end)];
            
            % Initial conditions: [x; v]
            y0 = [0; 0];
            
            % Solve with ODE45
            tic;
            [t_ode, y_ode] = ode45(ode_fn, t_span, y0, obj.options);
            solve_time = toc;
            
            fprintf('  Solution complete in %.4f seconds\n', solve_time);
            fprintf('  Time steps taken: %d (adaptive)\n', length(t_ode));
            fprintf('  Average dt: %.4f seconds\n', mean(diff(t_ode)));
            
            % Interpolate to uniform grid for compatibility
            obj.x_history = interp1(t_ode, y_ode(:,1), obj.t);
            obj.v_history = interp1(t_ode, y_ode(:,2), obj.t);
            
            % Compute prediction errors and free energy on uniform grid
            for i = 1:obj.N
                epsilon_x = obj.x_obs(i) - obj.v_history(i);
                epsilon_v = obj.v_history(i) - obj.mu_v;
                
                obj.prediction_error_x(i) = abs(epsilon_x);
                obj.prediction_error_v(i) = abs(epsilon_v);
                obj.free_energy(i) = 0.5 * (epsilon_x^2 / obj.sigma_x^2 + ...
                                            epsilon_v^2 / obj.sigma_v^2);
            end
        end
        
        function dydt = dynamics(obj, t, y)
            % Predictive coding dynamics for ODE45
            % y = [x_est; v_est]
            
            v_est = y(2);  % Estimated velocity
            
            % Get current sensory observation
            if t < obj.params.change_time
                x_obs = obj.params.v_before * t;
            else
                x_obs = obj.params.v_before * obj.params.change_time + ...
                        obj.params.v_after * (t - obj.params.change_time);
            end
            
            % Add pseudo-noise (deterministic for ODE)
            x_obs = x_obs + obj.params.noise_std * sin(100*t);
            
            % Compute prediction errors
            epsilon_x = x_obs - v_est;
            epsilon_v = v_est - obj.params.mu_v;
            
            % Update rules (gradient descent on free energy)
            dx_dt = epsilon_x / obj.params.sigma_x^2;
            dv_dt = epsilon_x / obj.params.sigma_x^2 - epsilon_v / obj.params.sigma_v^2;
            
            dydt = [dx_dt; dv_dt];
        end
        
        function fig = visualize(obj, title_str)
            % Override to add ODE45-specific plots
            if nargin < 2
                title_str = 'High-Precision Simulation with ODE45';
            end
            
            fig = figure('Position', [100, 100, 1400, 900]);
            
            % Standard plots (1-4)
            subplot(2,3,1);
            plot(obj.t, obj.true_position, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Position'); hold on;
            plot(obj.t, obj.x_history, 'b-', 'LineWidth', 2, 'DisplayName', 'Estimated (ODE45)');
            xlabel('Time (s)', 'FontSize', 11); ylabel('Position', 'FontSize', 11);
            title('Position Tracking', 'FontSize', 12, 'FontWeight', 'bold');
            legend('Location', 'northwest', 'FontSize', 10);
            grid on; box on;
            xline(obj.params.change_time, 'r--', 'LineWidth', 1.5);
            
            subplot(2,3,2);
            plot(obj.t, obj.true_velocity, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Velocity'); hold on;
            plot(obj.t, obj.v_history, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated (ODE45)');
            yline(obj.mu_v, 'g--', 'Prior', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11); ylabel('Velocity', 'FontSize', 11);
            title('Velocity Inference', 'FontSize', 12, 'FontWeight', 'bold');
            legend('Location', 'northeast', 'FontSize', 10);
            grid on; box on;
            xline(obj.params.change_time, 'r--', 'LineWidth', 1.5);
            
            subplot(2,3,3);
            plot(obj.x_history, obj.v_history, 'b-', 'LineWidth', 2); hold on;
            plot(obj.x_history(1), obj.v_history(1), 'go', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Start');
            plot(obj.x_history(end), obj.v_history(end), 'rs', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'End');
            xlabel('Position Estimate', 'FontSize', 11);
            ylabel('Velocity Estimate', 'FontSize', 11);
            title('Phase Portrait', 'FontSize', 12, 'FontWeight', 'bold');
            legend('FontSize', 10);
            grid on; box on;
            
            subplot(2,3,4);
            pos_error = abs(obj.true_position - obj.x_history);
            semilogy(obj.t, pos_error, 'b-', 'LineWidth', 2);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Position Error (log scale)', 'FontSize', 11);
            title('Tracking Error', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(obj.params.change_time, 'r--', 'LineWidth', 1.5);
            
            subplot(2,3,5);
            vel_error = abs(obj.v_history - obj.true_velocity);
            plot(obj.t, vel_error, 'r-', 'LineWidth', 2);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Velocity Error', 'FontSize', 11);
            title('Inference Error', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(obj.params.change_time, 'r--', 'LineWidth', 1.5);
            
            subplot(2,3,6);
            plot(obj.t, obj.free_energy, 'Color', [0.5 0 0.5], 'LineWidth', 2);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Free Energy', 'FontSize', 11);
            title('Free Energy Minimization', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(obj.params.change_time, 'r--', 'LineWidth', 1.5);
            
            sgtitle(title_str, 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end
