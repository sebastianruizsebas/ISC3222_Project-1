%% Wrapper to run the Active Inference Tutorial
% This script handles SPM initialization and runs the Step_by_Step_AI_Guide

% Clear workspace
clear all
close all

%% 1. INITIALIZE SPM
%==========================================================================

fprintf('\n');
fprintf('===========================================================\n');
fprintf('ACTIVE INFERENCE TUTORIAL - INITIALIZATION\n');
fprintf('===========================================================\n\n');

% Add SPM to path if not already there
spm_path = 'C:\Users\srseb\OneDrive\Documents\spm_25.01.02\spm';

if ~exist(spm_path, 'dir')
    error('SPM12 not found at: %s\nPlease verify the SPM installation path.', spm_path);
end

if ~exist('spm', 'file')
    addpath(spm_path);
    fprintf('✓ Added SPM to MATLAB path: %s\n', spm_path);
else
    fprintf('✓ SPM already in MATLAB path\n');
end

% Initialize SPM (sets defaults for fMRI)
try
    spm('defaults', 'fMRI');
    fprintf('✓ SPM defaults initialized\n');
catch ME
    warning('Active Inference Tutorial:SPMInit', 'Could not initialize SPM defaults: %s', ME.message);
end

% Verify required functions are available
required_functions = {'spm_MDP_check', 'spm_MDP_VB_X_tutorial', 'spm_figure', 'spm_BMS'};
fprintf('\nChecking for required SPM functions:\n');

missing_functions = {};
for i = 1:length(required_functions)
    if ~exist(required_functions{i}, 'file')
        fprintf('  ✗ %s - NOT FOUND\n', required_functions{i});
        missing_functions = [missing_functions, required_functions{i}];
    else
        fprintf('  ✓ %s found\n', required_functions{i});
    end
end

if ~isempty(missing_functions)
    error('Missing required SPM functions: %s\n', strjoin(missing_functions, ', '));
end

%% 2. CONFIGURE SIMULATION PARAMETERS
%==========================================================================

fprintf('\n');
fprintf('Simulation Configuration:\n');
fprintf('---\n');

% Simulation selection
fprintf('\nSelect simulation to run:\n');
fprintf('  1 = Single trial with risk-seeking parameter variations (reproduces Fig. 8)\n');
fprintf('  2 = Multi-trial with fixed left context (reproduces Fig. 10)\n');
fprintf('  3 = Reversal learning (left→right context switch) (reproduces Fig. 11)\n');
fprintf('  4 = Parameter estimation from simulated reversal learning data\n');
fprintf('  5 = Model comparison (2-param vs 3-param models) + PEB group analysis\n');

Sim = 4;  % Change this to select different simulations
fprintf('\n→ Selected: Sim = %d\n', Sim);

% Risk-seeking parameter (used for Sim 1-3)
if Sim <= 3
    fprintf('\nRisk-seeking parameter (rs1):\n');
    fprintf('  For Sim 1: use 4 or 8\n');
    fprintf('  For Sim 2-3: use 3 or 4\n');
    rs1 = 4;  % Default value
    fprintf('→ Selected: rs1 = %d\n', rs1);
end

% Model comparison group-level analysis flag (for Sim 5)
PEB = 1;  % Set to 1 to run Parametric Empirical Bayes (hierarchical analysis)
if Sim == 5
    fprintf('\nParametric Empirical Bayes (group-level analysis):\n');
    fprintf('→ PEB = %d (0=disabled, 1=enabled)\n', PEB);
    fprintf('  Note: PEB results are saved after Sim=5 for later reloading\n');
end

%% 3. SET RANDOM SEED FOR REPRODUCIBILITY
%==========================================================================

fprintf('\nRandom number generator:\n');
rng('default');  % Use 'shuffle' for different results each run
fprintf('→ Set to default seed for reproducibility\n');

%% 4. RUN THE TUTORIAL
%==========================================================================

fprintf('\n');
fprintf('===========================================================\n');
fprintf('RUNNING ACTIVE INFERENCE TUTORIAL (Sim = %d)\n', Sim);
fprintf('===========================================================\n\n');

try
    % Add the tutorial directory to path
    tutorial_dir = fileparts(mfilename('fullpath'));
    addpath(fullfile(tutorial_dir, 'DeepActiveInference'));
    
    % Run the main tutorial script
    Step_by_Step_AI_Guide;
    
    fprintf('\n');
    fprintf('===========================================================\n');
    fprintf('✓ TUTORIAL COMPLETED SUCCESSFULLY\n');
    fprintf('===========================================================\n\n');
    
catch ME
    fprintf('\n');
    fprintf('===========================================================\n');
    fprintf('✗ ERROR DURING TUTORIAL EXECUTION\n');
    fprintf('===========================================================\n\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Location: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    fprintf('\nFull stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  [%d] %s (line %d)\n', i, ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end

%% 5. POST-SIMULATION SUMMARY
%==========================================================================

fprintf('\nSimulation Details:\n');
fprintf('---\n');

switch Sim
    case 1
        fprintf('Single trial active inference with risk-seeking parameter: rs=%d\n', rs1);
        fprintf('Outputs: Figure 1 (neural responses), Figure 2 (beliefs & behavior)\n');
    case 2
        fprintf('Multi-trial simulation (30 trials) in fixed left-better context\n');
        fprintf('Outputs: Figure 3 (performance across trials)\n');
    case 3
        fprintf('Reversal learning: 4 trials left-better, then switch to right-better\n');
        fprintf('Outputs: Figure 4 (adaptive behavior during reversal)\n');
    case 4
        fprintf('Parameter estimation: recovering alpha (action precision) and rs (risk-seeking)\n');
        fprintf('From: 32-trial reversal learning task\n');
        fprintf('Outputs: Parameter recovery plots and estimation accuracy\n');
    case 5
        fprintf('Model comparison: 2-parameter model vs 3-parameter model\n');
        fprintf('Data: 6 participants × multiple parameter combinations\n');
        fprintf('Methods: BMS (Bayesian Model Selection), Recoverability analysis\n');
        if PEB == 1
            fprintf('Plus: PEB (Parametric Empirical Bayes) group-level analysis\n');
        end
        fprintf('Outputs: Model evidence, parameter recovery, saved results structures\n');
        fprintf('Saved files: GCM_2.mat, GCM_3.mat, Two_parameter_model_estimates.mat, etc.\n');
end

fprintf('\n');
fprintf('For detailed interpretation, see the Step_by_Step_AI_Guide comments.\n');
fprintf('For publication details, see: Smith et al. (Computational Psychiatry)\n');
fprintf('\n');

%% 6. OPTIONAL: SAVE SESSION LOG
%==========================================================================

% Optionally save a log of this session
log_filename = sprintf('AI_tutorial_log_Sim%d_%s.txt', Sim, datetime('now', 'Format', 'yyyyMMdd_HHmmss'));

try
    % Get current figure positions and info
    fig_handles = get(0, 'Children');
    n_figures = length(fig_handles);
    
    fprintf('\nSession Summary:\n');
    fprintf('  - Simulation: Sim = %d\n', Sim);
    fprintf('  - Figures created: %d\n', n_figures);
    fprintf('  - Timestamp: %s\n', datetime('now'));
    
    % Optional: save figures
    if n_figures > 0
        fprintf('\nOptionally save figures? (currently displaying %d figures)\n', n_figures);
        fprintf('Modify the script to enable automatic figure saving if desired.\n');
    end
    
catch ME
    fprintf('Note: Could not create session summary (%s)\n', ME.message);
end

fprintf('\n✓ Script execution completed.\n\n');

