classdef EulerModel < PredictiveCodingModel
    % EULERMODEL Implements Euler integration for predictive coding
    
    methods
        function obj = EulerModel(dt, T, sigma_x, sigma_v)
            % Call superclass constructor
            obj@PredictiveCodingModel(dt, T, sigma_x, sigma_v);
        end
        
        function run(obj)
            % Euler integration simulation
            fprintf('Running Euler simulation');
            
            for i = 1:obj.N
                if mod(i, obj.N/10) == 0
                    fprintf('.');
                end
                
                % Current sensory observation
                x_current = obj.x_obs(i);
                
                % Compute prediction errors
                epsilon_x = x_current - obj.v_est;
                epsilon_v = obj.v_est - obj.mu_v;
                
                % Update rules (derived from gradient descent)
                dx_dt = epsilon_x / obj.sigma_x^2;
                dv_dt = epsilon_x / obj.sigma_x^2 - epsilon_v / obj.sigma_v^2;
                
                % Euler integration
                obj.x_est = obj.x_est + dx_dt * obj.dt;
                obj.v_est = obj.v_est + dv_dt * obj.dt;
                
                % Compute free energy
                F = 0.5 * (epsilon_x^2 / obj.sigma_x^2 + epsilon_v^2 / obj.sigma_v^2);
                
                % Store history
                obj.x_history(i) = obj.x_est;
                obj.v_history(i) = obj.v_est;
                obj.prediction_error_x(i) = abs(epsilon_x);
                obj.prediction_error_v(i) = abs(epsilon_v);
                obj.free_energy(i) = F;
            end
            
            fprintf(' Done!\n');
        end
    end
end
