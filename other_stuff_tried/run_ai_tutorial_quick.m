%% Quick Start: Active Inference Tutorial
%
% This is a minimal wrapper that runs the tutorial with sensible defaults
% For more control, use: run_active_inference_tutorial.m

clear all; close all; clc

% Initialize SPM
spm_path = 'C:\Users\srseb\OneDrive\Documents\spm_25.01.02\spm';
if ~exist(spm_path, 'dir')
    error('SPM not found at: %s', spm_path);
end
addpath(spm_path);
spm('defaults', 'fMRI');

fprintf('✓ SPM initialized\n');
fprintf('✓ Starting Active Inference Tutorial...\n\n');

% Set seed for reproducibility
rng('default');

% Configuration: Modify these to run different simulations
Sim = 5;      % 1=single trial, 2=multi-trial, 3=reversal, 4=estimation, 5=model comparison
rs1 = 4;      % Risk-seeking parameter (for Sim 1-3)
PEB = 1;      % Set to 1 for group-level analysis in Sim 5

% Run the tutorial
addpath(fileparts(mfilename('fullpath')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'DeepActiveInference'));

Step_by_Step_AI_Guide;

fprintf('\n✓ Tutorial completed successfully\n');

