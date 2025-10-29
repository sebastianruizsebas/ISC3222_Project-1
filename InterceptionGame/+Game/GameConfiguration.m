classdef GameConfiguration
    % GameConfiguration - Manages all game parameters
    % 
    % Follows open science practices:
    % - All parameters configurable and documented
    % - Easy to reproduce experiments
    % - Version tracking for experimental runs
    %
    % Usage:
    %   config = Game.GameConfiguration();
    %   config.n_trials = 20;
    %   config.validate();
    
    properties
        % Experiment parameters
        participant_id                  % Unique identifier
        age                            % Participant age
        gaming_experience (1,1) double = 1  % Scale 1-5
        
        % Game difficulty
        n_trials (1,1) double = 15
        trial_duration (1,1) double = 5    % seconds
        target_speed_range (1,2) double = [100, 300]  % pixels/sec
        acceleration_types string = ["constant", "accelerating", "decelerating"]
        
        % Display parameters
        screen_width (1,1) double = 1280
        screen_height (1,1) double = 720
        target_size (1,1) double = 30     % diameter in pixels
        reticle_size (1,1) double = 40
        framerate (1,1) double = 60       % FPS
        
        % Motor control parameters
        reticle_speed (1,1) double = 10   % pixels per frame
        interception_threshold (1,1) double = 35  % pixels
        
        % File paths
        output_dir string = "interception_game_results"
        data_dir string = "data"
        
        % Metadata
        version string = "1.0.0"
        creation_date datetime
        experimenter string = "Anonymous"
        notes string = ""
    end
    
    properties (Dependent)
        dt                              % Time step (1/framerate)
    end
    
    methods
        function obj = GameConfiguration(varargin)
            % Constructor with optional name-value pairs
            % 
            % Example:
            %   config = Game.GameConfiguration('n_trials', 20, 'participant_id', 'P001');
            
            obj.creation_date = datetime('now');
            
            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'participant_id', 'default', @isstring);
            addParameter(p, 'age', 0, @isnumeric);
            addParameter(p, 'n_trials', 15, @isnumeric);
            parse(p, varargin{:});
            
            obj.participant_id = p.Results.participant_id;
            obj.age = p.Results.age;
            obj.n_trials = p.Results.n_trials;
        end
        
        function dt_val = get.dt(obj)
            % Calculate time step from framerate
            dt_val = 1 / obj.framerate;
        end
        
        function validate(obj)
            % Validate all parameters for consistency
            
            if obj.n_trials <= 0
                error('GameConfiguration:n_trials must be > 0');
            end
            
            if obj.target_speed_range(1) >= obj.target_speed_range(2)
                error('GameConfiguration:target_speed_range min must be < max');
            end
            
            if obj.reticle_size < 5
                warning('GameConfiguration:reticle_size very small - may be hard to see');
            end
            
            if obj.target_size > obj.screen_width / 2
                error('GameConfiguration:target_size too large for screen');
            end
            
            if obj.framerate < 30
                warning('GameConfiguration:framerate < 30 FPS - may feel laggy');
            end
        end
        
        function summary(obj)
            % Print human-readable summary of configuration
            
            fprintf('\n╔════════════════════════════════════════════════════╗\n');
            fprintf('║        GAME CONFIGURATION SUMMARY                  ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            
            fprintf('PARTICIPANT:\n');
            fprintf('  ID:                 %s\n', obj.participant_id);
            fprintf('  Age:                %d\n', obj.age);
            fprintf('  Gaming Experience:  %d/5\n\n', obj.gaming_experience);
            
            fprintf('GAME PARAMETERS:\n');
            fprintf('  Number of Trials:   %d\n', obj.n_trials);
            fprintf('  Trial Duration:     %.1f seconds\n', obj.trial_duration);
            fprintf('  Target Speed Range: %.0f - %.0f px/sec\n', ...
                obj.target_speed_range(1), obj.target_speed_range(2));
            fprintf('  Acceleration Types: %s\n', strjoin(obj.acceleration_types, ', '));
            
            fprintf('\nDISPLAY SETTINGS:\n');
            fprintf('  Screen Resolution:  %d x %d\n', obj.screen_width, obj.screen_height);
            fprintf('  Target Size:        %d px diameter\n', obj.target_size);
            fprintf('  Reticle Size:       %d px diameter\n', obj.reticle_size);
            fprintf('  Frame Rate:         %d FPS (dt = %.3f s)\n', obj.framerate, obj.dt);
            
            fprintf('\nMOTOR CONTROL:\n');
            fprintf('  Reticle Speed:      %d px/frame\n', obj.reticle_speed);
            fprintf('  Interception Threshold: %d px\n', obj.interception_threshold);
            
            fprintf('\nMETADATA:\n');
            fprintf('  Version:            %s\n', obj.version);
            fprintf('  Created:            %s\n', datetime(obj.creation_date));
            fprintf('  Experimenter:       %s\n', obj.experimenter);
            if ~isempty(obj.notes)
                fprintf('  Notes:              %s\n', obj.notes);
            end
            fprintf('\n');
        end
        
        function save(obj, filename)
            % Save configuration to JSON for reproducibility
            
            if nargin < 2
                filename = sprintf('config_%s_%s.json', obj.participant_id, ...
                    datestr(obj.creation_date, 'yyyymmdd_HHMMSS'));
            end
            
            % Convert to struct for JSON serialization
            config_struct = struct(...
                'participant_id', obj.participant_id, ...
                'age', obj.age, ...
                'gaming_experience', obj.gaming_experience, ...
                'n_trials', obj.n_trials, ...
                'trial_duration', obj.trial_duration, ...
                'version', obj.version, ...
                'creation_date', datestr(obj.creation_date), ...
                'experimenter', obj.experimenter);
            
            % Use jsonencode if available (R2016b+)
            try
                json_str = jsonencode(config_struct);
                fid = fopen(filename, 'w');
                fprintf(fid, json_str);
                fclose(fid);
                fprintf('✓ Configuration saved to: %s\n', filename);
            catch
                warning('JSON encoding not available - saving as .mat file instead');
                save(filename, 'config_struct');
            end
        end
    end
    
    methods (Static)
        function config = load(filename)
            % Load configuration from file
            
            if endsWith(filename, '.json')
                % Load from JSON
                fid = fopen(filename, 'r');
                json_str = fscanf(fid, '%c');
                fclose(fid);
                
                try
                    data = jsondecode(json_str);
                    config = Game.GameConfiguration('participant_id', data.participant_id, ...
                        'age', data.age, 'n_trials', data.n_trials);
                catch
                    error('Failed to load JSON configuration');
                end
            else
                % Load from .mat
                data = load(filename);
                config = Game.GameConfiguration();
            end
        end
    end
end
