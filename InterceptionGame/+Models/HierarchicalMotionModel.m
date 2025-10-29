classdef HierarchicalMotionModel < handle
    % HierarchicalMotionModel - Fits hierarchical inference to motor behavior
    %
    % Infers precision weights (π_x, π_v, π_a) from player's motor commands
    % Implements predictive coding inference
    %
    % Usage:
    %   model = Models.HierarchicalMotionModel(trial_data);
    %   model.fitPrecision();
    %   model.summary();
    
    properties
        trial_data                      % Game.TrialData object
        
        % Fitted parameters
        pi_x (1,1) double = 100         % Sensory precision
        pi_v (1,1) double = 10          % Velocity precision
        pi_a (1,1) double = 1           % Acceleration precision
        
        % Model state
        x_rep (:,1) double              % Position representation
        v_rep (:,1) double              % Velocity representation
        a_rep (:,1) double              % Acceleration representation
        
        % Errors
        err_x (:,1) double
        err_v (:,1) double
        err_a (:,1) double
        
        % Time
        t (:,1) double
        
        % Fit quality
        likelihood (1,1) double = -inf
        velocity_correlation (1,1) double
        
        % Hyperparameters
        learning_rate (1,1) double = 0.1
        dt (1,1) double = 0.01
    end
    
    methods
        function obj = HierarchicalMotionModel(trial_data, varargin)
            % Constructor
            %
            % Args:
            %   trial_data: Game.TrialData object
            %   varargin: name-value pairs (dt, learning_rate, etc.)
            
            obj.trial_data = trial_data;
            
            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'dt', 0.01, @isnumeric);
            addParameter(p, 'learning_rate', 0.1, @isnumeric);
            parse(p, varargin{:});
            
            obj.dt = p.Results.dt;
            obj.learning_rate = p.Results.learning_rate;
        end
        
        function fitPrecision(obj, varargin)
            % Fit precision parameters to trial data
            %
            % Grid search over precision parameters
            % Maximize correlation between model velocity and observed motor velocity
            
            % Parameter search ranges
            p = inputParser;
            addParameter(p, 'pi_x_range', [50, 100, 200], @isnumeric);
            addParameter(p, 'pi_v_range', [1, 5, 10, 20], @isnumeric);
            addParameter(p, 'pi_a_range', [0.1, 0.5, 1], @isnumeric);
            parse(p, varargin{:});
            
            pi_x_range = p.Results.pi_x_range;
            pi_v_range = p.Results.pi_v_range;
            pi_a_range = p.Results.pi_a_range;
            
            % Get motor data from trial
            t_motor = obj.trial_data.reticle_times;
            x_motor = obj.trial_data.reticle_pos(:,1);
            
            if isempty(t_motor) || length(t_motor) < 3
                warning('HierarchicalMotionModel:insufficient_data', ...
                    'Not enough motor data to fit model');
                return;
            end
            
            best_likelihood = -inf;
            best_params = [100, 10, 1];
            
            % Grid search
            for pi_x = pi_x_range
                for pi_v = pi_v_range
                    for pi_a = pi_a_range
                        
                        % Run inference
                        obj.runInference(t_motor, x_motor, pi_x, pi_v, pi_a);
                        
                        % Compute likelihood: correlation between model and observed velocity
                        model_vel = gradient(obj.x_rep, t_motor);
                        obs_vel = gradient(x_motor, t_motor);
                        
                        try
                            corr_val = corr(model_vel, obs_vel);
                            likelihood = corr_val;  % Higher correlation = better fit
                        catch
                            likelihood = -inf;
                        end
                        
                        if likelihood > best_likelihood
                            best_likelihood = likelihood;
                            best_params = [pi_x, pi_v, pi_a];
                        end
                    end
                end
            end
            
            % Set best parameters
            obj.pi_x = best_params(1);
            obj.pi_v = best_params(2);
            obj.pi_a = best_params(3);
            obj.likelihood = best_likelihood;
            
            % Run final inference with best parameters
            obj.runInference(t_motor, x_motor, obj.pi_x, obj.pi_v, obj.pi_a);
        end
        
        function runInference(obj, t, x_obs, pi_x, pi_v, pi_a)
            % Run hierarchical predictive coding inference
            %
            % Args:
            %   t: time vector
            %   x_obs: observed position (motor commands)
            %   pi_x, pi_v, pi_a: precision weights
            
            n = length(t);
            
            % Initialize representations
            obj.x_rep = zeros(n,1);
            obj.v_rep = zeros(n,1);
            obj.a_rep = zeros(n,1);
            
            obj.x_rep(1) = x_obs(1);
            obj.v_rep(1) = 0;
            obj.a_rep(1) = 0;
            
            % Initialize errors
            obj.err_x = zeros(n,1);
            obj.err_v = zeros(n,1);
            obj.err_a = zeros(n,1);
            
            obj.t = t;
            
            % Inference loop
            for i = 1:n-1
                
                % Top-down predictions
                pred_v = obj.a_rep(i);
                pred_x = obj.v_rep(i);
                
                % Bottom-up errors
                obj.err_x(i) = pi_x * (x_obs(i) - obj.x_rep(i));
                
                % Velocity error (from observed velocity change)
                obs_vel_change = (x_obs(i) - obj.x_rep(i)) / obj.dt;
                obj.err_v(i) = pi_v * (obs_vel_change - pred_v);
                
                % Acceleration error (prior on smoothness)
                obj.err_a(i) = pi_a * (obj.a_rep(i) - 0);  % Prior mean = 0
                
                % Update representations (gradient descent on free energy)
                obj.x_rep(i+1) = obj.x_rep(i) + obj.learning_rate * obj.err_x(i) / pi_x;
                obj.v_rep(i+1) = obj.v_rep(i) + obj.learning_rate * (obj.err_v(i)/pi_v - obj.err_x(i)/pi_x);
                obj.a_rep(i+1) = obj.a_rep(i) + obj.learning_rate * (obj.err_a(i)/pi_a - obj.err_v(i)/pi_v);
            end
            
            % Compute final velocity correlation
            model_vel = gradient(obj.x_rep, t);
            obs_vel = gradient(x_obs, t);
            
            try
                obj.velocity_correlation = corr(model_vel, obs_vel);
            catch
                obj.velocity_correlation = 0;
            end
        end
        
        function summary(obj)
            % Print model summary
            
            fprintf('\n╔════════════════════════════════════════════════════╗\n');
            fprintf('║    HIERARCHICAL MOTION INFERENCE - MODEL FIT      ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            
            fprintf('FITTED PRECISION WEIGHTS:\n');
            fprintf('─────────────────────────────────────────────────────\n');
            fprintf('  π_x (Sensory):     %.1f\n', obj.pi_x);
            fprintf('  π_v (Velocity):    %.1f\n', obj.pi_v);
            fprintf('  π_a (Acceleration): %.2f\n\n', obj.pi_a);
            
            fprintf('FIT QUALITY:\n');
            fprintf('─────────────────────────────────────────────────────\n');
            fprintf('  Likelihood:        %.4f\n', obj.likelihood);
            fprintf('  Velocity Corr:     %.4f\n\n', obj.velocity_correlation);
            
            % Classify strategy
            ratio = obj.pi_x / obj.pi_v;
            fprintf('MOTOR STRATEGY:\n');
            fprintf('─────────────────────────────────────────────────────\n');
            fprintf('  π_x / π_v Ratio:   %.2f\n', ratio);
            
            if ratio > 15
                fprintf('  Classification:    REACTIVE (sensory-driven)\n');
            elseif ratio < 5
                fprintf('  Classification:    PREDICTIVE (model-driven)\n');
            else
                fprintf('  Classification:    BALANCED (integrated)\n');
            end
            fprintf('\n');
        end
    end
end
