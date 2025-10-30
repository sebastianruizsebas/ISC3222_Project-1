%% ACTIVE INFERENCE TUTORIAL - EXECUTION GUIDE
%
% This document explains how to run the Active Inference tutorial
% and what each simulation does.
%
% Reference: Smith, Friston, Whyte (2022) 
% "A Step-by-Step Tutorial on Active Inference Modelling and its Application to Empirical Data"

%% HOW TO RUN THE TUTORIAL
%==========================================================================
%
% There are THREE ways to run this tutorial:
%
% 1. INTERACTIVE MODE (Recommended for exploration)
%    -----------------------------------------------
%    Open MATLAB GUI and run:
%    >> run_ai_tutorial_quick
%    
%    Or use the full version with more control:
%    >> run_active_inference_tutorial
%
%    In this mode, figures display interactively and you can explore them.
%
%
% 2. BATCH MODE (For non-interactive/headless execution)
%    ---------------------------------------------------
%    From command line:
%    matlab -batch "run_ai_tutorial_batch"
%    
%    This runs without displaying figures and saves results to files.
%
%
% 3. MANUAL MODE (For debugging)
%    ---------------------------
%    Open DeepActiveInference/Step_by_Step_AI_Guide.m directly
%    Run individual sections using Ctrl+Shift+Enter

%% SIMULATION TYPES
%==========================================================================

% SIMULATION 1: Single Trial
% --------------------------
% Sim = 1;
%
% Runs a single trial of the slot machine task and visualizes:
%   - Neural responses (expected free energy, value, etc.)
%   - Posterior beliefs about context, choice state, etc.
%   - Agent behavior
%
% Outputs:
%   Figure 1: spm_MDP_VB_LFP - Neural response waveforms (like ERPs)
%   Figure 2: spm_MDP_VB_trial - Beliefs, actions, and observations
%
% Reproducible as: Figure 8 (main text)
% 
% Configuration:
%   rs1 = 4  (or 8)  - Risk-seeking parameter
%   Higher rs → more likely to take risky actions (gamble more)
%
%
% SIMULATION 2: Multi-Trial Learning (Fixed Context)
% --------------------------------------------------
% Sim = 2;
%
% Runs 30 consecutive trials where the "left" machine is always better.
% Shows how the agent learns to prefer the winning machine.
%
% Outputs:
%   Figure 3: spm_MDP_VB_game_tutorial
%   - Learning curves (accuracy, reaction time)
%   - Action histograms
%   - Evidence accumulation over trials
%
% Reproducible as: Figure 10 (main text)
%
% Configuration:
%   rs1 = 3 (or 4)  - Risk-seeking parameter
%
%
% SIMULATION 3: Reversal Learning
% --------------------------------
% Sim = 3;
%
% Runs 32 trials where the context switches midway:
%   Trials 1-4: Left machine is better (D{1} = [1 0]')
%   Trials 5-32: Right machine is better (D{1} = [0 1]')
%
% Shows adaptive learning when environment changes.
% This is the key test of learning and flexibility.
%
% Outputs:
%   Figure 4: spm_MDP_VB_game_tutorial
%   - Reversal learning curves
%   - Context inference over time
%   - Switch in choice preferences
%
% Reproducible as: Figure 11 (main text)
%
% Configuration:
%   rs1 = 3 (or 4)
%
%
% SIMULATION 4: Parameter Estimation
% -----------------------------------
% Sim = 4;
%
% Generates simulated behavior under specific parameter values,
% then INVERTS the model to recover those parameters.
% 
% Tests whether model inversion (maximum likelihood estimation)
% can recover the true parameters that generated the behavior.
%
% Estimated parameters:
%   - alpha (action precision): controls action randomness
%     Higher alpha = more deterministic choices
%   - rs (risk-seeking): preference for wins vs. losses
%     Higher rs = more willing to gamble
%
% Outputs:
%   - Parameter recovery plots (true vs. estimated)
%   - Posterior mean and covariance
%   - Accuracy of parameter estimation
%
% Reproducible as: Figure 17 (top panel, main text)
%
% True parameter values used for generation:
%   alpha = 4 (compared to prior of 16 - lower precision)
%   rs = 6 (compared to prior of 5 - higher risk-seeking)
%
%
% SIMULATION 5: Model Comparison
% --------------------------------
% Sim = 5;
%
% Compares two competing models:
%
%   Model 1 (2 parameters):
%     - alpha (action precision)
%     - rs (risk-seeking)
%
%   Model 2 (3 parameters):
%     - alpha (action precision)
%     - rs (risk-seeking)
%     - eta (learning rate)
%
% Generates synthetic data from both models across multiple participants
% and simulated parameter combinations.
%
% Tests:
%   1. Bayesian Model Selection (BMS)
%      - Which model better explains the data?
%      - Provides: protected exceedance probability (pxp)
%   
%   2. Parameter Recoverability
%      - Can we recover true parameters?
%      - Tests correlation between true and estimated values
%
%   3. Log-likelihood and Action Probability
%      - How well does each model predict behavior?
%
% Outputs:
%   - Recoverability plots (5 figures)
%   - Model evidence (protected exceedance probability)
%   - Saved .mat files:
%     * Two_parameter_model_estimates.mat
%     * Three_parameter_model_estimates.mat
%     * GCM_2.mat (Group Conditional Model for 2-param)
%     * GCM_3.mat (Group Conditional Model for 3-param)
%
% Reproducible as: Figure 17 (bottom panel, main text)
%
% Optional: Group-level analysis (requires PEB = 1)
%   - Parametric Empirical Bayes (PEB)
%   - Tests for between-group differences
%   - Tests for age effects (simulated)
%   - Requires the saved GCM_2.mat and GCM_3.mat files
%
%
% NOTE: Sim 5 takes VERY LONG (10-30+ minutes) because it runs:
%   - 6 participants
%   - 2 models × multiple parameter combinations
%   - Parameter estimation for each combination
%   - Model comparison statistics
%
%   Recommendation: Run once, save results, then use PEB = 1 separately

%% CONFIGURATION OPTIONS
%==========================================================================

% Sim (Simulation Number)
% -----------------------
% Valid values: 1, 2, 3, 4, 5
% Default: 1

% rs1 (Risk-Seeking Parameter)
% ----------------------------
% Only used for Sim = 1, 2, 3
% 
% Default values:
%   Sim 1: rs1 = 4 or 8
%   Sim 2: rs1 = 3 or 4
%   Sim 3: rs1 = 3 or 4
%
% Interpretation:
%   Low rs (e.g., 1-2): Risk-averse, prefers information-seeking
%   Medium rs (e.g., 3-5): Moderate risk-taking
%   High rs (e.g., 6-10): Risk-seeking, prefers gambling

% PEB (Parametric Empirical Bayes)
% --------------------------------
% Only used for Sim = 5
% Valid values: 0 (disabled), 1 (enabled)
% Default: 0
%
% If PEB = 1, runs hierarchical (group-level) Bayesian analysis
% Requires significant computation time
%
% Note: Results are saved after Sim = 5, so you can:
%   1. Run Sim = 5 once with PEB = 0 to save GCM_2.mat, GCM_3.mat
%   2. Later, load those .mat files and run PEB = 1 separately
%      (This saves time on repeated runs)

%% KEY PARAMETERS WITHIN THE MODEL
%==========================================================================

% Model Learning Parameters (optional to modify)
% -----------------------------------------------

% eta (learning rate): 0-1
%   Controls how quickly beliefs change with new experience
%   Default: 0.5
%   Higher values: faster learning
%   Lower values: slower, more conservative learning

% omega (forgetting rate): 0-1
%   Controls how much past experience is "forgotten" each trial
%   Default: 1 (no forgetting)
%   Lower values: more forgetting (volatility assumption)

% beta (policy precision): positive number
%   Controls determinism of policy selection (inverse temperature)
%   Default: 1
%   Higher values: more deterministic choice

% alpha (action precision): positive number
%   Controls determinism of action execution
%   Default: 16
%   Higher values: less action noise

% tau (time constant): positive number
%   Controls speed of belief updating
%   Default: 12
%   Affects smoothness of neural responses

% erp (belief resetting): 1 or higher
%   Controls prior confidence carried between timepoints
%   Default: 1 (full carry-over)
%   Higher values: more uncertainty accumulation

%% RUNNING THE TUTORIAL IN PRACTICE
%==========================================================================

% EXAMPLE 1: Quick single-trial test
% -----------------------------------
% Edit line 20 in Step_by_Step_AI_Guide.m:
%   Sim = 1;
% Edit line 47:
%   rs1 = 4;
% Then run: >> run_ai_tutorial_quick

% EXAMPLE 2: Test learning over multiple trials
% -----------------------------------------------
% Edit Step_by_Step_AI_Guide.m:
%   Sim = 2;
%   rs1 = 4;
% Then run: >> run_ai_tutorial_quick

% EXAMPLE 3: Test reversal learning (adaptive behavior)
% -------------------------------------------------------
% Edit Step_by_Step_AI_Guide.m:
%   Sim = 3;
%   rs1 = 3;
% Then run: >> run_ai_tutorial_quick

% EXAMPLE 4: Parameter recovery (single run)
% -----------------------------------------------
% Edit Step_by_Step_AI_Guide.m:
%   Sim = 4;
% Then run: >> run_ai_tutorial_quick
% Takes ~1-2 minutes

% EXAMPLE 5: Full model comparison (long!)
% -----------------------------------------------
% Edit Step_by_Step_AI_Guide.m:
%   Sim = 5;
%   PEB = 0; (first time)
% Then run: >> run_ai_tutorial_quick
% Takes 15-30+ minutes depending on your computer

% EXAMPLE 6: Group analysis after saving data
% -----------------------------------------------
% a) First run Sim = 5 with PEB = 0 to save GCM_2.mat, GCM_3.mat
% b) Then set PEB = 1 and run again (faster because data is saved)

%% TROUBLESHOOTING
%==========================================================================

% Error: "Unrecognized function or variable 'spm_MDP_check'"
% ----------------------------------------------------------
% Solution: Ensure SPM12 and DEM toolbox are in MATLAB path
%   addpath('C:\Users\srseb\OneDrive\Documents\spm_25.01.02\spm')
%   spm('defaults', 'fMRI');
%   addpath(fullfile(spm_path, 'toolbox', 'DEM'));

% Error: "Matrix dimension mismatch in state equations"
% -------------------------------------------------------
% Likely issue: Incorrect state vector format (covered in tutorial)
% See the tutorial section on M.x dimension issues

% Error: "Out of memory" or "MATLAB is slow"
% -------------------------------------------
% For Sim = 5: This is expected. Model comparison is computationally intensive.
% Consider using fewer parameter combinations or running on a high-performance machine.

% Warning: "Figures not displaying in batch mode"
% -----------------------------------------------
% This is expected. Use: run_ai_tutorial_batch.m
% Results are saved as PNG files in ai_results/sim_X/ folder

%% OUTPUT INTERPRETATION
%==========================================================================

% Figure 1 (spm_MDP_VB_LFP) - Neural Responses
% -----------------------------------------------
% Shows simulated ERP-like responses reflecting:
%   - Free energy (expected free energy of policies)
%   - Value (expected outcome)
%   - Uncertainty (entropy)
%   - Estimated likelihood ratios
% These mimic real neural recordings (ERPs, MEG, fMRI)

% Figure 2 (spm_MDP_VB_trial) - Beliefs and Behavior
% ---------------------------------------------
% Shows posterior beliefs about:
%   - Hidden states (context, behavior state)
%   - Observations (what was seen)
%   - Policies (planned future actions)
% Also shows the actual actions taken

% Figure 3/4 (spm_MDP_VB_game_tutorial) - Learning Curves
% -----------------------------------------------
% Panels show:
%   1. Accuracy over trials
%   2. Reaction time
%   3. Action frequency (which options chosen)
%   4. Choice entropy (decision certainty)
%   5. Free energy trajectory

% Recoverability Plots (Sim 4-5)
% ---------------------------------
% Scatter plots with regression line showing:
%   X-axis: True parameter values (used to generate data)
%   Y-axis: Estimated parameter values (recovered by model inversion)
% 
% Perfect recovery = diagonal line (r = 1.0)
% Good recovery = tight correlation around line
% Poor recovery = scatter with low correlation

%% REFERENCES
%==========================================================================

% Smith, R., Friston, K. J., & Whyte, C. J. (2022). 
% A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data. The Computational Psychiatry Course, 
% Zurich.
%
% Available at: https://www.translationalneuromodeling.org/cpcourse/
%
% Original theoretical framework:
% Friston, K. J., Stephan, K. E., Montague, P. R., & Dolan, R. J. (2014).
% Computational psychiatry: the brain as a phantastic organ of adaptation.
% The Lancet Psychiatry, 1(2), 148-158.

%% NEXT STEPS AFTER RUNNING
%==========================================================================

% 1. Examine the code in Step_by_Step_AI_Guide.m
%    - Understand how mdp structure is built
%    - See how priors (D, A, B, C) are specified
%    - Learn how policies are enumerated
%
% 2. Modify model parameters and re-run to understand effects
%    - Change C (preferences) to test risk-aversion
%    - Change B (transitions) to test planning
%    - Change A (likelihoods) to test perceptual learning
%
% 3. Apply to your own data
%    - Use Estimate_parameters.m to fit real participant data
%    - Use model comparison to test competing hypotheses
%    - Use PEB for group-level inference
%
% 4. Integrate with your game task
%    - The interception game can be framed as an active inference problem
%    - Participants' motor commands = actions
%    - Prediction errors = observations vs. expectations
%    - Neural responses could reflect free energy minimization

%% END OF GUIDE
%==========================================================================

