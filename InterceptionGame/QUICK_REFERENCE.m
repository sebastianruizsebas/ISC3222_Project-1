% QUICK REFERENCE GUIDE - Interception Game
% 
% Print this for quick lookup while coding

%% ═══════════════════════════════════════════════════════════════════════
%%  QUICK START (Copy & Paste)
%% ═══════════════════════════════════════════════════════════════════════

% Run full experiment with one command:
run_experiment()

% Or with custom participant:
exp = Analysis.ExperimentManager('participant_id', 'P001');
exp.runFull();

%% ═══════════════════════════════════════════════════════════════════════
%%  CLASS API REFERENCE
%% ═══════════════════════════════════════════════════════════════════════

% GAME CONFIGURATION
% ──────────────────
config = Game.GameConfiguration('participant_id', 'P001', 'age', 25);
config.n_trials = 20;
config.trial_duration = 5;
config.target_speed_range = [100, 300];
config.validate();
config.summary();
config.save('config.json');

% GAME ENGINE
% ───────────
engine = Game.GameEngine(config);
engine.run();
trials = engine.trials_data;
trials(1).summary();

% TRIAL DATA
% ──────────
trial = Game.TrialData(1, "constant_velocity");
trial.addMotorCommand(0.1, [640, 360]);
trial.addInterception(2.0, 50, true);
trial.computeMetrics();
accuracy = trial.intercept_accuracy;
lead = trial.mean_lead_distance;

% TRAJECTORY GENERATION
% ─────────────────────
[traj, times, v, a] = Utils.TrajectoryGenerator.generateTargetTrajectory(...
    1280, 720, [100, 300], 'accelerating', 5);

% HIERARCHICAL MODEL
% ──────────────────
model = Models.HierarchicalMotionModel(trial);
model.fitPrecision();
model.summary();
pi_x = model.pi_x;
pi_v = model.pi_v;
pi_a = model.pi_a;

% EXPERIMENT MANAGER
% ──────────────────
exp = Analysis.ExperimentManager('participant_id', 'P001');
exp.runFull();
exp.participantIntake();
exp.runGame();
exp.analyzeResults();
exp.generateReport();
exp.saveResults();
stats = exp.summary_statistics;

%% ═══════════════════════════════════════════════════════════════════════
%%  COMMON TASKS
%% ═══════════════════════════════════════════════════════════════════════

% Run game only (skip analysis)
% ──────────────────────────────
config = Game.GameConfiguration('participant_id', 'P001', 'n_trials', 15);
engine = Game.GameEngine(config);
engine.run();

% Analyze existing data
% ─────────────────────
data = load('interception_game_results/P001_trials.mat');
trials = data.trials_data;
accuracies = [trials.intercept_accuracy];
leads = [trials.mean_lead_distance];

% Batch process multiple participants
% ────────────────────────────────────
for i = 1:10
    exp = Analysis.ExperimentManager('participant_id', sprintf('P%03d', i));
    exp.runFull();
end

% Extract performance metrics
% ────────────────────────────
accuracy = mean([trials.intercept_accuracy]);
std_acc = std([trials.intercept_accuracy]);
success_rate = sum([trials.success]) / length(trials);

% Extract model parameters
% ──────────────────────────
pi_x_vals = [model_fits.pi_x];
pi_v_vals = [model_fits.pi_v];
ratio = mean(pi_x_vals) / mean(pi_v_vals);

% Compare trials by motion type
% ──────────────────────────────
const_idx = strcmp({trials.motion_type}, 'constant');
accel_idx = strcmp({trials.motion_type}, 'accelerating');
mean_acc_const = mean([trials(const_idx).intercept_accuracy]);
mean_acc_accel = mean([trials(accel_idx).intercept_accuracy]);

%% ═══════════════════════════════════════════════════════════════════════
%%  CONFIGURATION PARAMETERS
%% ═══════════════════════════════════════════════════════════════════════

% Experiment parameters
config.participant_id = 'P001';        % Unique ID
config.age = 25;                       % Age in years
config.gaming_experience = 3;          % 1-5 scale
config.n_trials = 15;                  % Number of trials
config.trial_duration = 5;             % Seconds

% Display
config.screen_width = 1280;            % Pixels
config.screen_height = 720;            % Pixels
config.target_size = 30;               % Diameter pixels
config.reticle_size = 40;              % Diameter pixels
config.framerate = 60;                 % FPS

% Motion
config.target_speed_range = [100, 300];  % [min, max] px/sec
config.acceleration_types = ["constant", "accelerating", "decelerating"];

% Motor control
config.reticle_speed = 10;             % Pixels per frame
config.interception_threshold = 35;    % Pixels

% File paths
config.output_dir = "interception_game_results";
config.data_dir = "data";

%% ═══════════════════════════════════════════════════════════════════════
%%  INTERPRETATION GUIDE
%% ═══════════════════════════════════════════════════════════════════════

% π_x / π_v Ratio Interpretation
% ───────────────────────────────

% Ratio > 15: REACTIVE (Autism-like)
%   • Over-trusts sensory input
%   • Weak velocity predictions
%   • Small lead distance
%   • High accuracy on slow targets
%   • Poor accuracy on accelerating targets
%   • Neural: Enhanced V1, weak MT

% Ratio < 5: PREDICTIVE (Psychosis-like)
%   • Over-relies on internal model
%   • Strong velocity predictions
%   • Large lead distance
%   • Good on smooth predictable targets
%   • Poor on sudden changes
%   • Neural: Weak V1, overactive cerebellum

% Ratio ≈ 10: BALANCED (Neurotypical)
%   • Flexible sensory-motor integration
%   • Moderate lead distance
%   • Good across all target types
%   • Adaptive behavior
%   • Neural: Integrated cortical hierarchy

%% ═══════════════════════════════════════════════════════════════════════
%%  OUTPUT FILES
%% ═══════════════════════════════════════════════════════════════════════

% After running exp.runFull() for participant P001:
% 
% interception_game_results/
% ├── P001_config.json          ← Experiment parameters (JSON)
% ├── P001_trials.mat           ← All data (MATLAB format)
% └── P001_report.txt           ← Summary report (text)
%
% Access:
% data = load('P001_trials.mat');
% trials = data.trials_data;      % Array of TrialData objects
% models = data.model_fits;       % Array of fitted models
% stats = data.summary_statistics % Table with aggregate stats

%% ═══════════════════════════════════════════════════════════════════════
%%  TRIAL DATA FIELDS
%% ═══════════════════════════════════════════════════════════════════════

% For each trial in trials_data:
trial.trial_num                  % Trial number
trial.motion_type                % 'constant', 'accelerating', 'decelerating'
trial.reticle_pos                % [N × 2] player's reticle (x,y)
trial.reticle_times              % [N × 1] times
trial.target_trajectory          % [N × 2] true target position
trial.target_times               % [N × 1] target time points
trial.intercept_accuracy         % Distance from target at intercept (px)
trial.reaction_time              % Time to first movement (s)
trial.mean_lead_distance         % Average lead distance (px)
trial.success                    % Logical - did they intercept?
trial.velocity_correlation       % r value: correlation with model velocity

%% ═══════════════════════════════════════════════════════════════════════
%%  MODEL FIT FIELDS
%% ═══════════════════════════════════════════════════════════════════════

% For each fitted model in model_fits:
model.pi_x                       % Sensory precision (50-200)
model.pi_v                       % Velocity precision (1-20)
model.pi_a                       % Acceleration precision (0.1-1)
model.likelihood                 % Model fit quality (-1 to 1)
model.velocity_correlation       % How well model predicts motor velocity
model.x_rep                      % Position representation (inferred)
model.v_rep                      % Velocity representation (inferred)
model.a_rep                      % Acceleration representation (inferred)

%% ═══════════════════════════════════════════════════════════════════════
%%  USEFUL ONE-LINERS
%% ═══════════════════════════════════════════════════════════════════════

% Get all accuracies:
accuracies = [trials.intercept_accuracy];

% Get all lead distances:
leads = [trials.mean_lead_distance];

% Get accuracies by motion type:
acc_const = [trials(strcmp({trials.motion_type},'constant')).intercept_accuracy];

% Success rate:
pct_success = 100 * sum([trials.success]) / length(trials);

% Mean precision values:
mean_pi_x = mean([model_fits.pi_x]);
mean_pi_v = mean([model_fits.pi_v]);

% Strategy classification:
ratio = mean_pi_x / mean_pi_v;
if ratio > 15
    strategy = 'REACTIVE';
elseif ratio < 5
    strategy = 'PREDICTIVE';
else
    strategy = 'BALANCED';
end

% Plot performance by trial:
figure; plot([trials.intercept_accuracy], 'o-'); ylabel('Accuracy (px)');

% Plot lead distance over trials:
figure; plot([trials.mean_lead_distance], 's-'); ylabel('Lead (px)');

% Scatter: strategy vs performance:
figure; scatter([trials.mean_lead_distance], [trials.intercept_accuracy]);

%% ═══════════════════════════════════════════════════════════════════════
%%  DEBUGGING
%% ═══════════════════════════════════════════════════════════════════════

% Verify installation:
which GameConfiguration

% Check class accessibility:
Game.GameConfiguration
Game.GameEngine
Models.HierarchicalMotionModel

% Run tests:
test_all()

% Print configuration:
config.summary()

% Check trial data:
trial.summary()

% Check model fit:
model.summary()

% Verify graphics:
figure; circle_handle = fill([1 2 2 1]*100, [1 1 2 2]*100, 'r'); 
axis equal; xlim([0 300]); ylim([0 300]);

%% ═══════════════════════════════════════════════════════════════════════
%%  PLOTTING EXAMPLES
%% ═══════════════════════════════════════════════════════════════════════

% Accuracy over trials:
figure; plot([trials.intercept_accuracy]); 
xlabel('Trial'); ylabel('Accuracy (px)'); title('Learning Curve');

% Lead distance over trials:
figure; plot([trials.mean_lead_distance]);
xlabel('Trial'); ylabel('Lead Distance (px)'); title('Predictive Strategy');

% Strategy classification pie:
ratio = mean([model_fits.pi_x]) / mean([model_fits.pi_v]);
figure; text(0.5, 0.5, sprintf('π_x/π_v = %.2f\n%s Strategy', ratio, ...
    iif(ratio>15, 'REACTIVE', iif(ratio<5, 'PREDICTIVE', 'BALANCED'))));

% Box plot by motion type:
accs_const = [trials(strcmp({trials.motion_type},'constant')).intercept_accuracy];
accs_accel = [trials(strcmp({trials.motion_type},'accelerating')).intercept_accuracy];
figure; boxplot({accs_const, accs_accel}, 'Labels', {'Constant', 'Accelerating'});
ylabel('Accuracy (px)');

%% ═══════════════════════════════════════════════════════════════════════

fprintf('\n✓ Quick Reference Guide Loaded\n');
fprintf('For more help:\n');
fprintf('  README.md     - Architecture & API reference\n');
fprintf('  SETUP.md      - Installation & troubleshooting\n');
fprintf('  EXAMPLES.m    - Usage patterns\n');
fprintf('  test_all()    - Run unit tests\n\n');
