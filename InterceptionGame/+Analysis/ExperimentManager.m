classdef ExperimentManager < handle
    % ExperimentManager - Orchestrates entire experiment workflow
    %
    % Handles:
    % - Participant intake
    % - Game execution
    % - Data analysis
    % - Report generation
    %
    % Usage:
    %   exp = Analysis.ExperimentManager();
    %   exp.runFull();
    
    properties
        config                          % GameConfiguration
        engine                          % GameEngine
        trials_data                     % Array of TrialData
        
        % Analysis results
        model_fits                      % Array of fitted models
        summary_statistics table
        
        % Output
        output_dir string = "interception_game_results"
    end
    
    methods
        function obj = ExperimentManager(varargin)
            % Constructor
            %
            % Args:
            %   varargin: name-value pairs for configuration
            
            p = inputParser;
            addParameter(p, 'participant_id', 'P' + string(round(rand()*10000)), @isstring);
            addParameter(p, 'n_trials', 15, @isnumeric);
            addParameter(p, 'output_dir', "interception_game_results", @isstring);
            parse(p, varargin{:});
            
            % Create configuration
            obj.config = Game.GameConfiguration(...
                'participant_id', p.Results.participant_id, ...
                'n_trials', p.Results.n_trials);
            
            obj.output_dir = p.Results.output_dir;
            
            % Create directories
            if ~exist(char(obj.output_dir), 'dir')
                mkdir(char(obj.output_dir));
            end
        end
        
        function runFull(obj)
            % Run complete experiment workflow
            
            fprintf('\n╔════════════════════════════════════════════════════╗\n');
            fprintf('║      INTERCEPTION GAME - FULL EXPERIMENT            ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            
            % 1. Participant intake
            obj.participantIntake();
            
            % 2. Display configuration
            obj.config.summary();
            
            % 3. Run game
            obj.runGame();
            
            % 4. Analyze results
            obj.analyzeResults();
            
            % 5. Generate report
            obj.generateReport();
            
            % 6. Save data
            obj.saveResults();
            
            fprintf('\n╔════════════════════════════════════════════════════╗\n');
            fprintf('║          ✓ EXPERIMENT COMPLETE                      ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            fprintf('Results saved to: %s/\n\n', obj.output_dir);
        end
        
        function participantIntake(obj)
            % Collect participant information
            
            fprintf('╔════════════════════════════════════════════════════╗\n');
            fprintf('║        PARTICIPANT INFORMATION INTAKE              ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            
            obj.config.participant_id = input('Participant ID: ', 's');
            obj.config.age = input('Age: ');
            obj.config.gaming_experience = input('Gaming experience (1-5): ');
            obj.config.experimenter = input('Experimenter name: ', 's');
            obj.config.notes = input('Notes (optional): ', 's');
        end
        
        function runGame(obj)
            % Execute the game
            
            fprintf('\n╔════════════════════════════════════════════════════╗\n');
            fprintf('║           RUNNING INTERCEPTION GAME                 ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            
            % Create and run game engine
            obj.engine = Game.GameEngine(obj.config);
            obj.engine.run();
            
            % Extract trial data
            obj.trials_data = obj.engine.trials_data;
        end
        
        function analyzeResults(obj)
            % Perform all analyses on collected data
            
            fprintf('\n╔════════════════════════════════════════════════════╗\n');
            fprintf('║            ANALYZING RESULTS                        ║\n');
            fprintf('╚════════════════════════════════════════════════════╝\n\n');
            
            % Fit hierarchical models to each trial
            fprintf('Fitting hierarchical motion inference models...\n');
            
            n_trials = length(obj.trials_data);
            obj.model_fits(n_trials) = Models.HierarchicalMotionModel([]);
            
            for trial_idx = 1:n_trials
                trial = obj.trials_data(trial_idx);
                
                if isempty(trial.reticle_times) || length(trial.reticle_times) < 3
                    fprintf('  Trial %d: Skipped (insufficient data)\n', trial_idx);
                    continue;
                end
                
                % Fit model
                model = Models.HierarchicalMotionModel(trial);
                model.fitPrecision();
                obj.model_fits(trial_idx) = model;
                
                fprintf('  Trial %d: π_x=%.0f, π_v=%.0f, π_a=%.2f\n', ...
                    trial_idx, model.pi_x, model.pi_v, model.pi_a);
            end
            
            fprintf('\n✓ Model fitting complete\n\n');
            
            % Compute summary statistics
            obj.computeSummaryStatistics();
        end
        
        function computeSummaryStatistics(obj)
            % Compute group-level statistics
            
            n_trials = length(obj.trials_data);
            
            % Extract metrics
            pi_x_vals = [];
            pi_v_vals = [];
            pi_a_vals = [];
            accuracies = [];
            lead_distances = [];
            reaction_times = [];
            
            for trial_idx = 1:n_trials
                trial = obj.trials_data(trial_idx);
                
                if trial.intercept_accuracy > 0
                    accuracies = [accuracies; trial.intercept_accuracy];
                end
                
                if ~isnan(trial.mean_lead_distance)
                    lead_distances = [lead_distances; trial.mean_lead_distance];
                end
                
                if trial.reaction_time > 0
                    reaction_times = [reaction_times; trial.reaction_time];
                end
                
                if trial_idx <= length(obj.model_fits) && ~isempty(obj.model_fits(trial_idx).trial_data)
                    pi_x_vals = [pi_x_vals; obj.model_fits(trial_idx).pi_x];
                    pi_v_vals = [pi_v_vals; obj.model_fits(trial_idx).pi_v];
                    pi_a_vals = [pi_a_vals; obj.model_fits(trial_idx).pi_a];
                end
            end
            
            % Create summary table
            obj.summary_statistics = table(...
                mean(accuracies), std(accuracies), ...
                mean(lead_distances), std(lead_distances), ...
                mean(reaction_times), std(reaction_times), ...
                mean(pi_x_vals), std(pi_x_vals), ...
                mean(pi_v_vals), std(pi_v_vals), ...
                mean(pi_a_vals), std(pi_a_vals), ...
                'VariableNames', {'Accuracy_mean', 'Accuracy_std', ...
                    'Lead_mean', 'Lead_std', ...
                    'RT_mean', 'RT_std', ...
                    'PiX_mean', 'PiX_std', ...
                    'PiV_mean', 'PiV_std', ...
                    'PiA_mean', 'PiA_std'});
        end
        
        function generateReport(obj)
            % Generate comprehensive summary report
            
            report_file = fullfile(char(obj.output_dir), ...
                sprintf('%s_report.txt', char(obj.config.participant_id)));
            
            fid = fopen(report_file, 'w');
            
            % Header
            fprintf(fid, '═══════════════════════════════════════════════════\n');
            fprintf(fid, '  INTERCEPTION GAME - PARTICIPANT REPORT\n');
            fprintf(fid, '═══════════════════════════════════════════════════\n\n');
            
            % Participant info
            fprintf(fid, 'PARTICIPANT INFORMATION:\n');
            fprintf(fid, '───────────────────────────────────────────────────\n');
            fprintf(fid, 'ID:                %s\n', char(obj.config.participant_id));
            fprintf(fid, 'Age:               %d\n', obj.config.age);
            fprintf(fid, 'Gaming Exp:        %d/5\n', obj.config.gaming_experience);
            fprintf(fid, 'Experimenter:      %s\n', char(obj.config.experimenter));
            fprintf(fid, 'Date:              %s\n\n', datestr(datetime('now')));
            
            % Performance summary
            fprintf(fid, 'PERFORMANCE METRICS:\n');
            fprintf(fid, '───────────────────────────────────────────────────\n');
            
            stats = obj.summary_statistics;
            fprintf(fid, 'Accuracy:          %.1f ± %.1f pixels\n', ...
                stats.Accuracy_mean(1), stats.Accuracy_std(1));
            fprintf(fid, 'Lead Distance:     %.1f ± %.1f pixels\n', ...
                stats.Lead_mean(1), stats.Lead_std(1));
            fprintf(fid, 'Reaction Time:     %.2f ± %.2f seconds\n\n', ...
                stats.RT_mean(1), stats.RT_std(1));
            
            % Model fit
            fprintf(fid, 'HIERARCHICAL MODEL ESTIMATES:\n');
            fprintf(fid, '───────────────────────────────────────────────────\n');
            fprintf(fid, 'π_x (sensory):     %.1f ± %.1f\n', ...
                stats.PiX_mean(1), stats.PiX_std(1));
            fprintf(fid, 'π_v (velocity):    %.1f ± %.1f\n', ...
                stats.PiV_mean(1), stats.PiV_std(1));
            fprintf(fid, 'π_a (accel):       %.2f ± %.2f\n\n', ...
                stats.PiA_mean(1), stats.PiA_std(1));
            
            % Strategy interpretation
            ratio = stats.PiX_mean(1) / stats.PiV_mean(1);
            fprintf(fid, 'MOTOR STRATEGY:\n');
            fprintf(fid, '───────────────────────────────────────────────────\n');
            fprintf(fid, 'π_x / π_v Ratio:   %.2f\n', ratio);
            
            if ratio > 15
                fprintf(fid, 'Classification:    REACTIVE (sensory-driven)\n');
                fprintf(fid, 'Description:\n');
                fprintf(fid, '  • Over-trusts sensory input\n');
                fprintf(fid, '  • Weak velocity predictions\n');
                fprintf(fid, '  • Small lead distance (tracks closely)\n');
                fprintf(fid, '  • Good on slow/constant targets\n');
                fprintf(fid, '  • Poor on accelerating targets\n');
            elseif ratio < 5
                fprintf(fid, 'Classification:    PREDICTIVE (model-driven)\n');
                fprintf(fid, 'Description:\n');
                fprintf(fid, '  • Over-relies on internal models\n');
                fprintf(fid, '  • Strong velocity predictions\n');
                fprintf(fid, '  • Large lead distance (anticipatory)\n');
                fprintf(fid, '  • Good on smooth/predictable targets\n');
                fprintf(fid, '  • Poor on sudden changes\n');
            else
                fprintf(fid, 'Classification:    BALANCED (integrated)\n');
                fprintf(fid, 'Description:\n');
                fprintf(fid, '  • Flexible sensory-motor integration\n');
                fprintf(fid, '  • Adaptive velocity predictions\n');
                fprintf(fid, '  • Moderate, adaptive lead distance\n');
                fprintf(fid, '  • Good across all target types\n');
            end
            
            fprintf(fid, '\n═══════════════════════════════════════════════════\n');
            
            fclose(fid);
            fprintf('✓ Report saved: %s\n', report_file);
        end
        
        function saveResults(obj)
            % Save all data in standard formats
            
            % Save configuration
            config_file = fullfile(char(obj.output_dir), ...
                sprintf('%s_config.json', char(obj.config.participant_id)));
            obj.config.save(config_file);
            
            % Save trial data (MATLAB format for compatibility)
            data_file = fullfile(char(obj.output_dir), ...
                sprintf('%s_trials.mat', char(obj.config.participant_id)));
            trials_data = obj.trials_data;
            model_fits = obj.model_fits;
            summary_statistics = obj.summary_statistics;
            save(data_file, 'trials_data', 'model_fits', 'summary_statistics');
            fprintf('✓ Data saved: %s\n', data_file);
        end
    end
end
