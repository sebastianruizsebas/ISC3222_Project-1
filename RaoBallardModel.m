classdef RaoBallardModel < handle
    % RAOBALLARDMODEL Three-level hierarchical predictive coding
    % Based on Rao & Ballard (1999) with explicit prediction and error units
    
    properties
        % Time parameters
        dt
        t
        N
        
        % Precision weights (inverse variances)
        pi_x        % Level 1 sensory precision
        pi_v        % Level 2 prior precision
        pi_a        % Level 3 prior precision
        
        % Learning rates
        eta_rep     % Learning rate for representations
        eta_err     % Learning rate for error units
        
        % Prior expectations
        mu_a        % Expected acceleration
        
        % State variables - Representations
        a_rep       % Acceleration representation
        v_rep       % Velocity representation
        x_rep       % Position representation
        
        % State variables - Errors
        err_x       % Level 1 error (sensory)
        err_v       % Level 2 error (velocity)
        err_a       % Level 3 error (acceleration)
        
        % State variables - Predictions
        pred_x      % Position predicted from velocity
        pred_v      % Velocity predicted from acceleration
        
        % Sensory input
        x_obs
        true_x
        true_v
        true_a
        
        % Free energy
        free_energy
    end
    
    methods
        function obj = RaoBallardModel(dt, T, pi_x, pi_v, pi_a)
            % Constructor
            obj.dt = dt;
            obj.t = 0:dt:T;
            obj.N = length(obj.t);
            
            obj.pi_x = pi_x;
            obj.pi_v = pi_v;
            obj.pi_a = pi_a;
            
            % Learning rates
            obj.eta_rep = 0.1;
            obj.eta_err = 0.5;
            
            % Prior expectations
            obj.mu_a = 0;  % Stationary velocity expected
            
            % Initialize state
            obj.a_rep = zeros(1, obj.N);
            obj.v_rep = zeros(1, obj.N);
            obj.x_rep = zeros(1, obj.N);
            
            obj.err_x = zeros(1, obj.N);
            obj.err_v = zeros(1, obj.N);
            obj.err_a = zeros(1, obj.N);
            
            obj.pred_x = zeros(1, obj.N);
            obj.pred_v = zeros(1, obj.N);
            
            obj.free_energy = zeros(1, obj.N);
        end
        
        function generateSensoryInput(obj, sensor_noise, a_before, a_after, change_time)
            % Generate sensory input with changing acceleration
            if nargin < 3
                a_before = 0;
            end
            if nargin < 4
                a_after = -3;
            end
            if nargin < 5
                change_time = obj.t(end) / 2;
            end
            
            mid = find(obj.t >= change_time, 1);
            
            % True motion: constant acceleration phases
            obj.true_a = [a_before * ones(1, mid-1), a_after * ones(1, obj.N-mid+1)];
            obj.true_v = zeros(1, obj.N);
            obj.true_x = zeros(1, obj.N);
            
            for i = 2:obj.N
                obj.true_v(i) = obj.true_v(i-1) + obj.true_a(i-1) * obj.dt;
                obj.true_x(i) = obj.true_x(i-1) + obj.true_v(i-1) * obj.dt;
            end
            
            % Add sensory noise
            obj.x_obs = obj.true_x + randn(1, obj.N) * sensor_noise;
            
            % Initialize position representation with first observation
            obj.x_rep(1) = obj.x_obs(1);
        end
        
        function run(obj)
            % Rao & Ballard update loop
            fprintf('Running Rao & Ballard predictive coding simulation');
            
            for i = 1:obj.N-1
                if mod(i, obj.N/10) == 0
                    fprintf('.');
                end
                
                % ===== FORWARD PASS: Generate Predictions (Top-Down) =====
                % Level 3 → Level 2: Acceleration predicts velocity change
                obj.pred_v(i) = obj.a_rep(i);  % dv/dt = a
                
                % Level 2 → Level 1: Velocity predicts position change
                obj.pred_x(i) = obj.v_rep(i);  % dx/dt = v
                
                % ===== COMPUTE PREDICTION ERRORS (Bottom-Up) =====
                % Level 1: Sensory prediction error
                obj.err_x(i) = obj.pi_x * (obj.x_obs(i) - obj.x_rep(i));
                
                % Level 2: Velocity prediction error
                observed_v_change = (obj.x_rep(i) - obj.x_rep(max(1, i-1))) / obj.dt;
                obj.err_v(i) = obj.pi_v * (observed_v_change - obj.pred_x(i));
                
                % Level 3: Acceleration prior error
                observed_a_change = (obj.v_rep(i) - obj.v_rep(max(1, i-1))) / obj.dt;
                obj.err_a(i) = obj.pi_a * (observed_a_change - obj.pred_v(i));
                
                % ===== UPDATE REPRESENTATIONS (Error Correction) =====
                % Position update (driven by sensory error)
                obj.x_rep(i+1) = obj.x_rep(i) + obj.dt * obj.eta_rep * obj.err_x(i);
                
                % Velocity update (driven by errors from above AND below)
                obj.v_rep(i+1) = obj.v_rep(i) + obj.dt * obj.eta_rep * ...
                    (obj.err_v(i) / obj.pi_v + obj.err_x(i) / obj.pi_x);
                
                % Acceleration update
                obj.a_rep(i+1) = obj.a_rep(i) + obj.dt * obj.eta_rep * ...
                    (obj.err_a(i) / obj.pi_a + obj.err_v(i) / obj.pi_v - ...
                     (obj.a_rep(i) - obj.mu_a) * obj.pi_a);
                
                % Compute free energy
                obj.free_energy(i) = 0.5 * (obj.err_x(i)^2 / obj.pi_x + ...
                                            obj.err_v(i)^2 / obj.pi_v + ...
                                            obj.err_a(i)^2 / obj.pi_a);
            end
            
            % Final free energy
            obj.free_energy(obj.N) = 0.5 * (obj.err_x(obj.N)^2 / obj.pi_x + ...
                                            obj.err_v(obj.N)^2 / obj.pi_v + ...
                                            obj.err_a(obj.N)^2 / obj.pi_a);
            
            fprintf(' Done!\n');
        end
        
        function fig = visualize(obj)
            % Create Rao & Ballard visualization
            fprintf('Generating Rao & Ballard visualization...\n');
            
            fig = figure('Position', [100, 100, 1600, 1000]);
            
            % 1. Position inference
            subplot(3, 3, 1);
            plot(obj.t, obj.true_x, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
            plot(obj.t, obj.x_obs, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5, ...
                 'DisplayName', 'Noisy Obs');
            plot(obj.t, obj.x_rep, 'b-', 'LineWidth', 2, 'DisplayName', 'Inferred');
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Position', 'FontSize', 11);
            title('Level 1: Position Representation', 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 9);
            grid on;
            
            % 2. Velocity inference
            subplot(3, 3, 2);
            plot(obj.t, obj.true_v, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
            plot(obj.t, obj.v_rep, 'g-', 'LineWidth', 2, 'DisplayName', 'Inferred');
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Velocity', 'FontSize', 11);
            title('Level 2: Velocity Representation', 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 9);
            grid on;
            
            % 3. Acceleration inference
            subplot(3, 3, 3);
            plot(obj.t, obj.true_a, 'k--', 'LineWidth', 2.5, 'DisplayName', 'True'); hold on;
            plot(obj.t, obj.a_rep, 'r-', 'LineWidth', 2, 'DisplayName', 'Inferred');
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Acceleration', 'FontSize', 11);
            title('Level 3: Acceleration Representation', 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 9);
            grid on;
            
            % 4. Sensory prediction error
            subplot(3, 3, 4);
            plot(obj.t, obj.err_x, 'b-', 'LineWidth', 2);
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Error Signal', 'FontSize', 11);
            title('Level 1: Sensory Error (ε_x)', 'FontWeight', 'bold');
            grid on;
            
            % 5. Velocity prediction error
            subplot(3, 3, 5);
            plot(obj.t, obj.err_v, 'g-', 'LineWidth', 2);
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Error Signal', 'FontSize', 11);
            title('Level 2: Velocity Error (ε_v)', 'FontWeight', 'bold');
            grid on;
            
            % 6. Acceleration prediction error
            subplot(3, 3, 6);
            plot(obj.t, obj.err_a, 'r-', 'LineWidth', 2);
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Error Signal', 'FontSize', 11);
            title('Level 3: Acceleration Error (ε_a)', 'FontWeight', 'bold');
            grid on;
            
            % 7. Predictions vs. representations
            subplot(3, 3, 7);
            plot(obj.t, obj.v_rep, 'g-', 'LineWidth', 2, 'DisplayName', 'v (representation)'); hold on;
            plot(obj.t, obj.pred_x, 'b--', 'LineWidth', 2, 'DisplayName', 'prediction from v');
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Value', 'FontSize', 11);
            title('Top-Down Predictions', 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 9);
            grid on;
            
            % 8. Free energy
            subplot(3, 3, 8);
            plot(obj.t, obj.free_energy, 'm-', 'LineWidth', 2);
            xline(5, 'red', ':', 'LineWidth', 1.5);
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('Free Energy', 'FontSize', 11);
            title('Model Evidence (Lower = Better)', 'FontWeight', 'bold');
            grid on;
            
            % 9. Information flow diagram
            subplot(3, 3, 9);
            axis off;
            text(0.5, 0.9, 'Rao & Ballard Architecture', 'FontSize', 13, ...
                 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            
            % Draw hierarchy
            y_levels = [0.7, 0.5, 0.3, 0.1];
            labels = {'Level 3: Acceleration (a)', 'Level 2: Velocity (v)', ...
                      'Level 1: Position (x)', 'Sensory Input (x_{obs})'};
            colors = {'red', 'green', 'blue', 'cyan'};
            
            for i = 1:4
                rectangle('Position', [0.2, y_levels(i)-0.05, 0.6, 0.08], ...
                         'FaceColor', colors{i}, 'EdgeColor', 'black', ...
                         'LineWidth', 2, 'Curvature', 0.2);
                text(0.5, y_levels(i), labels{i}, 'FontSize', 10, 'Color', 'white', ...
                     'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            end
            
            text(0.5, 0.02, '↓ Predictions (green) | ↑ Errors (red)', ...
                'FontSize', 9, 'HorizontalAlignment', 'center');
            
            sgtitle('Rao & Ballard Predictive Coding: Three-Level Hierarchy', ...
                    'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function save(obj, filename)
            % Save results to .mat file
            fprintf('\nSaving results...\n');
            
            t = obj.t; %#ok<PROPLC>
            x_rep = obj.x_rep; %#ok<PROPLC>
            v_rep = obj.v_rep; %#ok<PROPLC>
            a_rep = obj.a_rep; %#ok<PROPLC>
            err_x = obj.err_x; %#ok<PROPLC>
            err_v = obj.err_v; %#ok<PROPLC>
            err_a = obj.err_a; %#ok<PROPLC>
            pred_x = obj.pred_x; %#ok<PROPLC>
            pred_v = obj.pred_v; %#ok<PROPLC>
            true_x = obj.true_x; %#ok<PROPLC>
            true_v = obj.true_v; %#ok<PROPLC>
            true_a = obj.true_a; %#ok<PROPLC>
            x_obs = obj.x_obs; %#ok<PROPLC>
            free_energy = obj.free_energy; %#ok<PROPLC>
            pi_x = obj.pi_x; %#ok<PROPLC>
            pi_v = obj.pi_v; %#ok<PROPLC>
            pi_a = obj.pi_a; %#ok<PROPLC>
            
            save(filename, 't', 'x_rep', 'v_rep', 'a_rep', ...
                 'err_x', 'err_v', 'err_a', 'pred_x', 'pred_v', ...
                 'true_x', 'true_v', 'true_a', 'x_obs', 'free_energy', ...
                 'pi_x', 'pi_v', 'pi_a');
        end
        
        function printSummary(obj)
            % Print performance summary
            fprintf('\nPerformance Metrics:\n');
            fprintf('  Final position error: %.4f\n', abs(obj.x_rep(end) - obj.true_x(end)));
            fprintf('  Final velocity error: %.4f\n', abs(obj.v_rep(end) - obj.true_v(end)));
            fprintf('  Final acceleration error: %.4f\n', abs(obj.a_rep(end) - obj.true_a(end)));
            fprintf('  Final free energy: %.4f\n', obj.free_energy(end));
            
            % Adaptation analysis
            change_idx = find(obj.t >= 5, 1);
            post_change_a_error = abs(obj.a_rep(change_idx:end) - obj.true_a(change_idx:end));
            adapt_idx = find(post_change_a_error < 0.5, 1);
            if ~isempty(adapt_idx)
                adapt_time = obj.t(change_idx + adapt_idx) - 5;
                fprintf('  Adaptation time (acceleration): %.2f seconds\n', adapt_time);
            else
                fprintf('  Adaptation time: > %.1f seconds\n', obj.t(end) - 5);
            end
        end
    end
end
