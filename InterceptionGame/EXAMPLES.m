%% EXAMPLE USAGE PATTERNS
%
% This file demonstrates common use cases for the Interception Game

%% EXAMPLE 1: Run default experiment
% Simplest way to run a full experiment

fprintf('EXAMPLE 1: Default experiment\n');
fprintf('─────────────────────────────\n\n');

% Just run it!
% exp = Analysis.ExperimentManager();
% exp.runFull();

fprintf('# Code:\n');
fprintf('exp = Analysis.ExperimentManager();\n');
fprintf('exp.runFull();\n\n');

%% EXAMPLE 2: Custom game parameters
% Customize the game difficulty and configuration

fprintf('EXAMPLE 2: Custom game parameters\n');
fprintf('──────────────────────────────────\n\n');

% Create experiment with custom parameters
% exp = Analysis.ExperimentManager(...
%     'participant_id', 'P001_FastMotion', ...
%     'n_trials', 20);
% 
% % Modify configuration
% exp.config.trial_duration = 3;      % Shorter trials
% exp.config.target_speed_range = [200, 400];  % Faster targets
% exp.config.gaming_experience = 4;   % Advanced player
% 
% % Run
% exp.runFull();

fprintf('# Code:\n');
fprintf('exp = Analysis.ExperimentManager(''participant_id'', ''P001'', ''n_trials'', 20);\n');
fprintf('exp.config.trial_duration = 3;\n');
fprintf('exp.config.target_speed_range = [200, 400];\n');
fprintf('exp.runFull();\n\n');

%% EXAMPLE 3: Run just the game (skip analysis)
% Useful for quick testing or running multiple participants

fprintf('EXAMPLE 3: Game only (no analysis)\n');
fprintf('──────────────────────────────────\n\n');

% config = Game.GameConfiguration(...
%     'participant_id', 'P002', ...
%     'n_trials', 10);
% 
% engine = Game.GameEngine(config);
% engine.run();
% 
% trials = engine.trials_data;

fprintf('# Code:\n');
fprintf('config = Game.GameConfiguration(''participant_id'', ''P002'', ''n_trials'', 10);\n');
fprintf('engine = Game.GameEngine(config);\n');
fprintf('engine.run();\n');
fprintf('trials = engine.trials_data;\n\n');

%% EXAMPLE 4: Analyze existing data
% Load previous results and perform new analyses

fprintf('EXAMPLE 4: Analyze existing data\n');
fprintf('────────────────────────────────\n\n');

% % Load saved data
% data = load('interception_game_results/P001_trials.mat');
% trials_data = data.trials_data;
% 
% % Compute additional metrics
% accuracies = [trials_data.intercept_accuracy];
% lead_distances = [trials_data.mean_lead_distance];
% 
% figure;
% scatter(lead_distances, accuracies);
% xlabel('Lead Distance (px)'); ylabel('Accuracy (px)');
% title('Strategy vs Performance');

fprintf('# Code:\n');
fprintf('data = load(''interception_game_results/P001_trials.mat'');\n');
fprintf('trials_data = data.trials_data;\n');
fprintf('accuracies = [trials_data.intercept_accuracy];\n');
fprintf('scatter(accuracies);\n\n');

%% EXAMPLE 5: Compare multiple participants
% Aggregate data across participants

fprintf('EXAMPLE 5: Compare multiple participants\n');
fprintf('────────────────────────────────────────\n\n');

% % Run experiment for multiple participants
% participants = {'P001', 'P002', 'P003'};
% all_results = table();
% 
% for i = 1:length(participants)
%     exp = Analysis.ExperimentManager('participant_id', participants{i});
%     exp.runFull();
%     
%     % Extract summary stats
%     stats = exp.summary_statistics;
%     stats.participant_id = participants{i};
%     all_results = [all_results; stats];
% end
% 
% % Compare precisions
% figure;
% boxplot([all_results.PiX_mean, all_results.PiV_mean, all_results.PiA_mean]);
% set(gca, 'XTickLabel', {'π_x', 'π_v', 'π_a'});
% ylabel('Precision Weight');
% title('Precision Profile Across Participants');

fprintf('# Code:\n');
fprintf('participants = {''P001'', ''P002'', ''P003''};\n');
fprintf('for i = 1:length(participants)\n');
fprintf('  exp = Analysis.ExperimentManager(''participant_id'', participants{i});\n');
fprintf('  exp.runFull();\n');
fprintf('end\n\n');

%% EXAMPLE 6: Generate custom report
% Create specialized analysis report

fprintf('EXAMPLE 6: Generate custom report\n');
fprintf('─────────────────────────────────\n\n');

% % Load data
% data = load('interception_game_results/P001_trials.mat');
% trials = data.trials_data;
% models = data.model_fits;
% 
% % Separate by motion type
% constant_idx = strcmp({trials.motion_type}, 'constant');
% accel_idx = strcmp({trials.motion_type}, 'accelerating');
% decel_idx = strcmp({trials.motion_type}, 'decelerating');
% 
% % Compare performance by condition
% fprintf('Accuracy by motion type:\n');
% fprintf('  Constant:      %.1f px\n', mean([trials(constant_idx).intercept_accuracy]));
% fprintf('  Accelerating:  %.1f px\n', mean([trials(accel_idx).intercept_accuracy]));
% fprintf('  Decelerating:  %.1f px\n', mean([trials(decel_idx).intercept_accuracy]));

fprintf('# Code:\n');
fprintf('data = load(''interception_game_results/P001_trials.mat'');\n');
fprintf('constant_idx = strcmp({data.trials_data.motion_type}, ''constant'');\n');
fprintf('fprintf(''Accuracy: %%.1f px\\\\n'', mean([data.trials_data(constant_idx).intercept_accuracy]));\n\n');

%% EXAMPLE 7: Run with different difficulty levels
% Progressive difficulty for learning effects

fprintf('EXAMPLE 7: Progressive difficulty\n');
fprintf('─────────────────────────────────\n\n');

% % Easy: slow targets
% config_easy = Game.GameConfiguration(...
%     'participant_id', 'P001_Easy', 'n_trials', 5);
% config_easy.target_speed_range = [50, 150];
% engine1 = Game.GameEngine(config_easy);
% engine1.run();
% 
% % Medium: normal speed
% config_medium = Game.GameConfiguration(...
%     'participant_id', 'P001_Medium', 'n_trials', 5);
% config_medium.target_speed_range = [100, 300];
% engine2 = Game.GameEngine(config_medium);
% engine2.run();
% 
% % Hard: fast targets
% config_hard = Game.GameConfiguration(...
%     'participant_id', 'P001_Hard', 'n_trials', 5);
% config_hard.target_speed_range = [200, 400];
% engine3 = Game.GameEngine(config_hard);
% engine3.run();

fprintf('# Code:\n');
fprintf('for difficulty = [''Easy'', ''Medium'', ''Hard'']\n');
fprintf('  config = Game.GameConfiguration(..)\n');
fprintf('  engine = Game.GameEngine(config);\n');
fprintf('  engine.run();\n');
fprintf('end\n\n');

%% EXAMPLE 8: Test individual components
% Useful for debugging and development

fprintf('EXAMPLE 8: Test components\n');
fprintf('────────────────────────────\n\n');

% % Test configuration
% config = Game.GameConfiguration('participant_id', 'TEST');
% config.summary();
% config.validate();
% 
% % Test trajectory generation
% [traj, times, v, a] = Utils.TrajectoryGenerator.generateTargetTrajectory(...
%     1280, 720, [100, 300], 'accelerating', 5);
% 
% figure;
% plot(traj(:,1));
% xlabel('Frame'); ylabel('X Position');
% title(sprintf('Trajectory (v=%.0f, a=%.0f)', v, a));
% 
% % Test model fitting
% trial = Game.TrialData(1, 'test');
% for i = 1:100
%     trial.addMotorCommand((i-1)*0.01, [100+i, 360]);
% end
% 
% model = Models.HierarchicalMotionModel(trial);
% model.fitPrecision();
% model.summary();

fprintf('# Code:\n');
fprintf('config = Game.GameConfiguration();\n');
fprintf('config.validate();\n');
fprintf('config.summary();\n\n');

fprintf('✓ See code comments above for all usage examples\n');
fprintf('✓ Uncomment any example to run it\n');
fprintf('✓ Modify parameters to test different scenarios\n\n');
