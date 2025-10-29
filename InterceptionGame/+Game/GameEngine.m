classdef GameEngine < handle
    % GameEngine - Main game loop and graphics management
    %
    % Handles:
    % - Graphics rendering
    % - Keyboard/mouse input
    % - Trial execution
    % - Real-time data collection
    %
    % Usage:
    %   engine = Game.GameEngine(config);
    %   engine.run();
    
    properties
        config                          % GameConfiguration object
        
        % Graphics
        fig
        ax
        target_graphic
        reticle_graphic
        trial_text_handle
        instructions_handle
        
        % Game state
        is_running logical = false
        current_trial (1,1) double = 0
        trials_data (:,1) Game.TrialData  % Array of TrialData
        
        % Keyboard state
        keys_pressed containers.Map
        space_pressed logical = false
        
        % Performance metrics
        performance_log table
    end
    
    methods
        function obj = GameEngine(config)
            % Constructor
            %
            % Args:
            %   config: GameConfiguration object
            
            if ~isa(config, 'Game.GameConfiguration')
                error('GameEngine:invalid_config', ...
                    'Input must be GameConfiguration object');
            end
            
            obj.config = config;
            obj.config.validate();
            obj.keys_pressed = containers.Map();
            
            % Pre-allocate trial data array
            obj.trials_data(obj.config.n_trials) = Game.TrialData(1, "");
        end
        
        function initialize(obj)
            % Initialize graphics window
            
            obj.fig = figure(...
                'Name', sprintf('Interception Game - Player %s', obj.config.participant_id), ...
                'NumberTitle', 'off', ...
                'Position', [100, 100, obj.config.screen_width, obj.config.screen_height], ...
                'Color', [0.1, 0.1, 0.1], ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'KeyPressFcn', @obj.keyPressCallback, ...
                'KeyReleaseFcn', @obj.keyReleaseCallback, ...
                'CloseRequestFcn', @obj.closeCallback);
            
            % Create axes
            obj.ax = axes('Parent', obj.fig, ...
                'Position', [0, 0, 1, 1], ...
                'XLim', [0, obj.config.screen_width], ...
                'YLim', [0, obj.config.screen_height], ...
                'Color', [0.1, 0.1, 0.1], ...
                'XTick', [], 'YTick', []);
            
            hold(obj.ax, 'on');
            axis(obj.ax, 'equal');
            
            % Create graphical elements
            obj.target_graphic = obj.drawCircle(...
                obj.config.screen_width/2, obj.config.screen_height/2, ...
                obj.config.target_size, 'red');
            
            obj.reticle_graphic = obj.drawCircle(...
                obj.config.screen_width/2, obj.config.screen_height/2, ...
                obj.config.reticle_size, 'green');
            
            % Text elements
            obj.trial_text_handle = text(obj.ax, ...
                obj.config.screen_width*0.05, obj.config.screen_height*0.95, '', ...
                'Color', 'white', 'FontSize', 14, 'VerticalAlignment', 'top');
            
            obj.instructions_handle = text(obj.ax, ...
                obj.config.screen_width/2, obj.config.screen_height*0.1, ...
                'READY? Press SPACE to start', ...
                'Color', 'yellow', 'FontSize', 16, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            
            drawnow;
            fprintf('✓ Graphics initialized\n');
        end
        
        function run(obj)
            % Main game loop
            
            obj.initialize();
            obj.is_running = true;
            
            fprintf('Starting trials...\n');
            fprintf('Use ARROW KEYS to control green reticle\n');
            fprintf('Press SPACE when ready to intercept\n\n');
            
            pause(2);
            
            % Run each trial
            for trial_idx = 1:obj.config.n_trials
                if ~obj.is_running
                    break;
                end
                obj.runTrial(trial_idx);
            end
            
            obj.is_running = false;
            set(obj.instructions_handle, 'String', 'Game Complete!');
            drawnow;
            pause(2);
        end
        
        function runTrial(obj, trial_idx)
            % Execute a single trial
            %
            % Args:
            %   trial_idx: trial number
            
            % Generate target trajectory
            accel_type = obj.config.acceleration_types{...
                mod(trial_idx-1, length(obj.config.acceleration_types)) + 1};
            
            [target_traj, target_times, v_true, a_true] = ...
                Utils.TrajectoryGenerator.generateTargetTrajectory(...
                obj.config.screen_width, obj.config.screen_height, ...
                obj.config.target_speed_range, accel_type, obj.config.trial_duration);
            
            % Create trial data container
            trial = Game.TrialData(trial_idx, accel_type);
            trial.target_trajectory = target_traj;
            trial.target_times = target_times;
            trial.v_true = v_true;
            trial.a_true = a_true;
            
            % Display trial start message
            set(obj.trial_text_handle, ...
                'String', sprintf('Trial %d/%d: %s motion\nPress SPACE when ready...', ...
                trial_idx, obj.config.n_trials, accel_type));
            
            % Reset reticle position
            obj.updateReticleDisplay([obj.config.screen_width/2, obj.config.screen_height/2]);
            set(obj.instructions_handle, 'String', 'Waiting for you to press SPACE...');
            
            drawnow;
            
            % Wait for space press
            obj.space_pressed = false;
            while ~obj.space_pressed && obj.is_running
                pause(0.01);
            end
            
            % Run trial
            set(obj.instructions_handle, 'String', 'Go! Intercept the target!');
            set(obj.trial_text_handle, ...
                'String', sprintf('Trial %d/%d: %s motion', trial_idx, obj.config.n_trials, accel_type));
            
            trial_start_time = tic;
            intercept_made = false;
            reticle_pos = [obj.config.screen_width/2, obj.config.screen_height/2];
            
            while toc(trial_start_time) < obj.config.trial_duration && ~intercept_made
                
                t_elapsed = toc(trial_start_time);
                
                % Get target position
                if t_elapsed <= target_times(end)
                    target_pos = interp1(target_times, target_traj, t_elapsed, 'linear', 'extrap');
                    obj.updateTargetDisplay(target_pos);
                    trial.addTargetObservation(target_pos);
                else
                    target_pos = target_traj(end,:);
                end
                
                % Get reticle position from keyboard
                reticle_pos = obj.getReticlePosition(reticle_pos);
                obj.updateReticleDisplay(reticle_pos);
                trial.addMotorCommand(t_elapsed, reticle_pos);
                
                % Check interception
                distance_to_target = norm(reticle_pos - target_pos);
                
                if distance_to_target < (obj.config.target_size + obj.config.reticle_size)/2
                    intercept_made = true;
                    set(obj.reticle_graphic, 'FaceColor', [0, 1, 0]);
                    set(obj.instructions_handle, ...
                        'String', sprintf('SUCCESS! Accuracy: %.1f px', distance_to_target), ...
                        'Color', 'lime');
                    trial.addInterception(t_elapsed, distance_to_target, true);
                end
                
                drawnow limitrate;
                pause(obj.config.dt);
            end
            
            % End-of-trial feedback
            if ~intercept_made
                distance_to_target = norm(reticle_pos - target_pos);
                set(obj.instructions_handle, ...
                    'String', sprintf('Time up! Accuracy: %.1f px', distance_to_target), ...
                    'Color', 'red');
                trial.addInterception(obj.config.trial_duration, distance_to_target, false);
            end
            
            % Compute metrics and store
            trial.computeMetrics();
            obj.trials_data(trial_idx) = trial;
            trial.summary();
            
            % Inter-trial pause
            pause(1);
        end
        
        function reticle_pos = getReticlePosition(obj, current_pos)
            % Get reticle position based on keyboard input
            
            reticle_pos = current_pos;
            speed = obj.config.reticle_speed;
            
            if obj.keys_pressed.isKey('uparrow')
                reticle_pos(2) = reticle_pos(2) - speed;
            end
            if obj.keys_pressed.isKey('downarrow')
                reticle_pos(2) = reticle_pos(2) + speed;
            end
            if obj.keys_pressed.isKey('leftarrow')
                reticle_pos(1) = reticle_pos(1) - speed;
            end
            if obj.keys_pressed.isKey('rightarrow')
                reticle_pos(1) = reticle_pos(1) + speed;
            end
            
            % Clip to screen
            reticle_pos(1) = max(obj.config.reticle_size, ...
                min(obj.config.screen_width - obj.config.reticle_size, reticle_pos(1)));
            reticle_pos(2) = max(obj.config.reticle_size, ...
                min(obj.config.screen_height - obj.config.reticle_size, reticle_pos(2)));
        end
        
        function updateTargetDisplay(obj, target_pos)
            % Update target graphics
            set(obj.target_graphic, 'XData', target_pos(1), 'YData', target_pos(2));
        end
        
        function updateReticleDisplay(obj, reticle_pos)
            % Update reticle graphics
            set(obj.reticle_graphic, 'XData', reticle_pos(1), 'YData', reticle_pos(2));
            set(obj.reticle_graphic, 'FaceColor', [0, 0.7, 0]);  % Default color
        end
        
        % Callback functions
        function keyPressCallback(obj, ~, event)
            % Handle key press
            key = lower(event.Key);
            obj.keys_pressed(key) = true;
            
            if strcmp(key, 'space')
                obj.space_pressed = true;
            end
        end
        
        function keyReleaseCallback(obj, ~, event)
            % Handle key release
            key = lower(event.Key);
            if obj.keys_pressed.isKey(key)
                remove(obj.keys_pressed, key);
            end
        end
        
        function closeCallback(obj, ~, ~)
            % Handle window close
            obj.is_running = false;
            delete(obj.fig);
        end
    end
    
    methods (Access = private)
        function h = drawCircle(obj, x, y, r, color)
            % Draw a circle
            theta = linspace(0, 2*pi, 100);
            circle_x = x + r*cos(theta);
            circle_y = y + r*sin(theta);
            h = fill(obj.ax, circle_x, circle_y, color, 'EdgeColor', 'none');
        end
    end
end
