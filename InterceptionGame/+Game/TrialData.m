classdef TrialData < handle
    % TrialData - Container for single trial results
    %
    % Encapsulates all data collected during one trial
    % Provides validation and analysis methods
    %
    % Usage:
    %   trial = Game.TrialData(trial_num, motion_type, ...);
    %   trial.addMotorCommand(t, reticle_pos);
    %   trial.computeMetrics();
    
    properties
        % Trial identification
        trial_num (1,1) double
        motion_type string
        timestamp datetime
        
        % Target trajectory
        target_trajectory (:,2) double  % [x, y] positions
        target_times (:,1) double       % Time points
        v_true (1,1) double             % True velocity
        a_true (1,1) double             % True acceleration
        
        % Motor commands
        reticle_pos (:,2) double        % Player reticle positions
        reticle_times (:,1) double      % Time points
        
        % Observations (target as seen by player)
        target_obs (:,2) double
        
        % Interception attempt
        intercept_time (1,1) double
        intercept_pos (1,2) double
        intercept_accuracy (1,1) double % Distance from target
        
        % Computed metrics
        reaction_time (1,1) double      % Time to first movement
        mean_lead_distance (1,1) double % Average lead (signed)
        max_lead_distance (1,1) double
        velocity_correlation (1,1) double
        success (1,1) logical           % Did interception succeed?
    end
    
    properties (Constant)
        INTERCEPTION_THRESHOLD = 35     % pixels
    end
    
    methods
        function obj = TrialData(trial_num, motion_type)
            % Constructor
            obj.trial_num = trial_num;
            obj.motion_type = motion_type;
            obj.timestamp = datetime('now');
            
            % Initialize arrays
            obj.reticle_pos = [];
            obj.reticle_times = [];
            obj.target_obs = [];
            obj.reaction_time = 0;
        end
        
        function addMotorCommand(obj, time, reticle_pos)
            % Add a motor command (reticle position) to trial
            %
            % Args:
            %   time: elapsed time in seconds
            %   reticle_pos: [x, y] position
            
            obj.reticle_times = [obj.reticle_times; time];
            obj.reticle_pos = [obj.reticle_pos; reticle_pos];
            
            % Compute reaction time (first movement)
            if isempty(obj.reaction_time) || obj.reaction_time == 0
                if size(obj.reticle_pos, 1) > 1
                    displacement = norm(obj.reticle_pos(end,:) - obj.reticle_pos(1,:));
                    if displacement > 5  % 5 pixel threshold
                        obj.reaction_time = time;
                    end
                end
            end
        end
        
        function addTargetObservation(obj, target_pos)
            % Add target position observation
            obj.target_obs = [obj.target_obs; target_pos];
        end
        
        function addInterception(obj, time, accuracy, success)
            % Record interception attempt
            %
            % Args:
            %   time: when player attempted intercept
            %   accuracy: distance from target
            %   success: logical - did they intercept?
            
            obj.intercept_time = time;
            obj.intercept_accuracy = accuracy;
            obj.success = success;
        end
        
        function computeMetrics(obj)
            % Compute derived metrics
            
            if isempty(obj.reticle_times) || isempty(obj.target_times)
                warning('TrialData:insufficient_data', ...
                    'Cannot compute metrics - missing data');
                return;
            end
            
            % Interpolate target and reticle to common timeline
            t_common = linspace(max(obj.reticle_times(1), obj.target_times(1)), ...
                min(obj.reticle_times(end), obj.target_times(end)), 500);
            
            target_x = interp1(obj.target_times, obj.target_trajectory(:,1), t_common);
            reticle_x = interp1(obj.reticle_times, obj.reticle_pos(:,1), t_common);
            
            % Lead distance: how far ahead is reticle?
            lead = reticle_x - target_x;
            obj.mean_lead_distance = nanmean(lead);
            obj.max_lead_distance = max(abs(lead));
            
            % Velocity correlation
            target_vel = gradient(target_x, t_common);
            reticle_vel = gradient(reticle_x, t_common);
            
            try
                obj.velocity_correlation = corr(target_vel', reticle_vel');
            catch
                obj.velocity_correlation = NaN;
            end
        end
        
        function summary(obj)
            % Print trial summary
            
            fprintf('\nTrial %d (%s):\n', obj.trial_num, obj.motion_type);
            fprintf('  ├─ Reaction Time:      %.2f s\n', obj.reaction_time);
            fprintf('  ├─ Mean Lead Distance: %.1f px\n', obj.mean_lead_distance);
            fprintf('  ├─ Interception:       %.1f px (success: %s)\n', ...
                obj.intercept_accuracy, string(obj.success));
            fprintf('  └─ Velocity Corr:      %.3f\n', obj.velocity_correlation);
        end
    end
end
