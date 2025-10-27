classdef PredictiveCodingModel < handle
    % PREDICTIVECODINGMODEL Base class for hierarchical predictive coding models
    % Implements common functionality for free energy minimization
    
    properties
        % Time parameters
        dt              % Time step
        t               % Time vector
        N               % Number of time steps
        
        % Model parameters
        sigma_x         % Sensory precision
        sigma_v         % Velocity prior precision
        mu_v            % Prior mean velocity
        
        % State variables
        x_est           % Position estimate
        v_est           % Velocity estimate
        
        % History
        x_history
        v_history
        free_energy
        prediction_error_x
        prediction_error_v
        
        % Sensory input
        x_obs
        true_position
        true_velocity
    end
    
    methods
        function obj = PredictiveCodingModel(dt, T, sigma_x, sigma_v)
            % Constructor
            obj.dt = dt;
            obj.t = 0:dt:T;
            obj.N = length(obj.t);
            obj.sigma_x = sigma_x;
            obj.sigma_v = sigma_v;
            obj.mu_v = 0;  % Default: stationary prior
            
            % Initialize state
            obj.x_est = 0;
            obj.v_est = 0;
            
            % Preallocate history
            obj.x_history = zeros(1, obj.N);
            obj.v_history = zeros(1, obj.N);
            obj.free_energy = zeros(1, obj.N);
            obj.prediction_error_x = zeros(1, obj.N);
            obj.prediction_error_v = zeros(1, obj.N);
        end
        
        function generateSensoryInput(obj, noise_std, v_before, v_after, change_time)
            % Generate synthetic sensory input with velocity change
            % Default scenario: velocity changes at t=5s from +2 to -1
            if nargin < 3
                v_before = 2;
            end
            if nargin < 4
                v_after = -1;
            end
            if nargin < 5
                change_time = obj.t(end) / 2;  % Middle of simulation
            end
            
            mid = find(obj.t >= change_time, 1);
            
            obj.true_velocity = zeros(1, obj.N);
            obj.true_velocity(1:mid-1) = v_before;
            obj.true_velocity(mid:end) = v_after;
            
            obj.true_position = cumsum(obj.true_velocity) * obj.dt;
            obj.x_obs = obj.true_position + noise_std * randn(1, obj.N);
        end
        
        function run(obj)
            % Main simulation loop (to be implemented by subclasses)
            error('Subclasses must implement run() method');
        end
        
        function F = computeFreeEnergy(obj, i)
            % Compute free energy at time step i
            epsilon_x = obj.x_obs(i) - obj.v_est;
            epsilon_v = obj.v_est - obj.mu_v;
            
            F = 0.5 * (epsilon_x^2 / obj.sigma_x^2 + ...
                       epsilon_v^2 / obj.sigma_v^2);
        end
        
        function fig = visualize(obj, title_str)
            % Create standard visualization
            if nargin < 2
                title_str = 'Predictive Coding: Two-Level Visual Motion Model';
            end
            
            fig = figure('Position', [100, 100, 1400, 900]);
            
            % Position tracking
            subplot(3, 2, 1);
            plot(obj.t, obj.true_position, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Position'); hold on;
            plot(obj.t, obj.x_obs, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5, 'DisplayName', 'Noisy Observation');
            plot(obj.t, obj.x_history, 'b-', 'LineWidth', 2, 'DisplayName', 'Estimated Position');
            xlabel('Time (s)', 'FontSize', 11); ylabel('Position', 'FontSize', 11);
            title('Sensory Input vs. Prediction', 'FontSize', 12, 'FontWeight', 'bold');
            legend('Location', 'northwest', 'FontSize', 9);
            grid on; box on;
            xline(5, 'r--', 'LineWidth', 1.5);
            
            % Velocity inference
            subplot(3, 2, 2);
            plot(obj.t, obj.true_velocity, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Velocity'); hold on;
            plot(obj.t, obj.v_history, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated Velocity');
            yline(obj.mu_v, 'g--', 'Prior', 'LineWidth', 1.5, 'FontSize', 9);
            xlabel('Time (s)', 'FontSize', 11); ylabel('Velocity', 'FontSize', 11);
            title('Hidden State Inference', 'FontSize', 12, 'FontWeight', 'bold');
            legend('Location', 'northeast', 'FontSize', 9);
            grid on; box on;
            xline(5, 'r--', 'LineWidth', 1.5);
            
            % Position prediction error
            subplot(3, 2, 3);
            plot(obj.t, obj.prediction_error_x, 'b-', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11); ylabel('|ε_x|', 'FontSize', 11);
            title('Sensory Prediction Error', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(5, 'r--', 'LineWidth', 1.5);
            
            % Velocity prediction error
            subplot(3, 2, 4);
            plot(obj.t, obj.prediction_error_v, 'r-', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11); ylabel('|ε_v|', 'FontSize', 11);
            title('Prior Prediction Error', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(5, 'r--', 'LineWidth', 1.5);
            
            % Free energy
            subplot(3, 2, 5);
            plot(obj.t, obj.free_energy, 'Color', [0.5 0 0.5], 'LineWidth', 2);
            xlabel('Time (s)', 'FontSize', 11); ylabel('Free Energy', 'FontSize', 11);
            title('Free Energy Minimization', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(5, 'r--', 'LineWidth', 1.5);
            
            % Position tracking error
            subplot(3, 2, 6);
            position_error = abs(obj.true_position - obj.x_history);
            plot(obj.t, position_error, 'g-', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11); ylabel('|x_{true} - x_{est}|', 'FontSize', 11);
            title('Tracking Error', 'FontSize', 12, 'FontWeight', 'bold');
            grid on; box on;
            xline(5, 'r--', 'LineWidth', 1.5);
            
            % Add overall title
            sgtitle(title_str, 'FontSize', 14, 'FontWeight', 'bold');
        end
        
        function save(obj, filename)
            % Save results to .mat file
            fprintf('\nSaving results...\n');
            
            t = obj.t; %#ok<PROPLC>
            x_history = obj.x_history; %#ok<PROPLC>
            v_history = obj.v_history; %#ok<PROPLC>
            true_position = obj.true_position; %#ok<PROPLC>
            true_velocity = obj.true_velocity; %#ok<PROPLC>
            free_energy = obj.free_energy; %#ok<PROPLC>
            x_obs = obj.x_obs; %#ok<PROPLC>
            
            save(filename, 't', 'x_history', 'v_history', ...
                'true_position', 'true_velocity', 'free_energy', 'x_obs');
        end
        
        function printSummary(obj)
            % Print performance summary
            pos_error = mean(abs(obj.x_history - obj.true_position));
            vel_error = mean(abs(obj.v_history - obj.true_velocity));
            
            fprintf('\nPerformance Metrics:\n');
            fprintf('  Mean position error: %.4f\n', pos_error);
            fprintf('  Mean velocity error: %.4f\n', vel_error);
            fprintf('  Final free energy: %.4f\n', obj.free_energy(end));
            fprintf('  Final velocity: %.2f (true: %.2f)\n', ...
                obj.v_history(end), obj.true_velocity(end));
            
            % Adaptation time
            change_idx = find(obj.t >= 5, 1);
            if ~isempty(change_idx)
                post_change = change_idx:obj.N;
                vel_change_magnitude = abs(obj.true_velocity(change_idx) - obj.true_velocity(change_idx-1));
                adaptation_threshold = 0.2 * vel_change_magnitude;
                vel_error_post_change = abs(obj.v_history(post_change) - obj.true_velocity(post_change));
                adapted_idx = find(vel_error_post_change < adaptation_threshold, 1);
                
                if ~isempty(adapted_idx)
                    adaptation_time = adapted_idx * obj.dt;
                    fprintf('  Adaptation time: %.2f seconds\n', adaptation_time);
                end
            end
        end
    end
end
