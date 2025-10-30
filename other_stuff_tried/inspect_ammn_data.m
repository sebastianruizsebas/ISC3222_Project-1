function [] = inspect_ammn_data()
%INSPECT_AMMN_DATA Inspect and understand the aMMN_critical_events.mat structure
%
% This script loads and displays the complete structure of the aMMN data file,
% helping to understand what variables are available for visualization.
%
% OUTPUT:
%   - Console display of data structure
%   - Data summary table
%   - Dimension analysis
%

clear all; close all; clc;

fprintf('================================================\n');
fprintf('aMMN CRITICAL EVENTS DATA INSPECTOR\n');
fprintf('================================================\n\n');

% File path
mat_file = 'aMMN_critical_events.mat';

if ~exist(mat_file, 'file')
    fprintf('ERROR: %s not found in current directory\n', mat_file);
    fprintf('Current directory: %s\n', pwd);
    return;
end

fprintf('Loading: %s\n\n', mat_file);

%% Load and inspect data
data = load(mat_file);

%% Display all variables
fprintf('VARIABLES IN DATA FILE:\n');
fprintf('================================================\n\n');

var_names = fieldnames(data);
num_vars = length(var_names);

fprintf('Total variables: %d\n\n', num_vars);

for i = 1:num_vars
    var_name = var_names{i};
    var_data = getfield(data, var_name);
    
    % Get basic info
    if isnumeric(var_data)
        var_class = class(var_data);
        var_size = size(var_data);
        var_numel = numel(var_data);
        
        fprintf('[%d] %s\n', i, var_name);
        fprintf('    Class: %s\n', var_class);
        fprintf('    Size: %s\n', sprintf('[%s]', sprintf('%d ', var_size)));
        fprintf('    Elements: %d\n', var_numel);
        
        % Display statistics for numeric arrays
        if numel(var_data) > 0 && ~isa(var_data, 'uint8')
            try
                fprintf('    Min: %.4f, Max: %.4f, Mean: %.4f, Std: %.4f\n', ...
                    min(var_data(:)), max(var_data(:)), mean(var_data(:)), std(var_data(:)));
            catch
                % Skip if can't compute stats
            end
        end
        
    elseif isstruct(var_data)
        fprintf('[%d] %s\n', i, var_name);
        fprintf('    Class: struct\n');
        fprintf('    Size: %s\n', sprintf('[%s]', sprintf('%d ', size(var_data))));
        field_names = fieldnames(var_data);
        field_str = sprintf('%s, ', field_names{:});
        field_str = field_str(1:end-2);  % Remove trailing ', '
        fprintf('    Fields: %s\n', field_str);
        
    elseif iscell(var_data)
        fprintf('[%d] %s\n', i, var_name);
        fprintf('    Class: cell array\n');
        fprintf('    Size: %s\n', sprintf('[%s]', sprintf('%d ', size(var_data))));
        
    else
        fprintf('[%d] %s\n', i, var_name);
        fprintf('    Class: %s\n', class(var_data));
    end
    
    fprintf('\n');
end

%% Provide interpretation
fprintf('\n================================================\n');
fprintf('DATA INTERPRETATION GUIDE\n');
fprintf('================================================\n\n');

% Look for EEG data
if isfield(data, 'EEG') && isnumeric(data.EEG)
    fprintf('✓ EEG data found (main signal data)\n');
    fprintf('  Likely dimensions: [samples × channels × trials]\n');
    fprintf('  Typical sampling rate: 2500 Hz\n\n');
end

% Look for event markers
if isfield(data, 'events')
    fprintf('✓ Events found (trial markers)\n');
    if isstruct(data.events)
        event_fields = fieldnames(data.events);
        event_str = sprintf('%s, ', event_fields{:});
        event_str = event_str(1:end-2);  % Remove trailing ', '
        fprintf('  Contains: %s\n', event_str);
    end
    fprintf('\n');
end

% Look for standard trials
if isfield(data, 'standard_trials')
    fprintf('✓ Standard trials found (common stimuli)\n');
    fprintf('  These are baseline/non-deviant stimuli\n\n');
end

% Look for deviant trials
if isfield(data, 'deviant_trials')
    fprintf('✓ Deviant trials found (oddball stimuli)\n');
    fprintf('  These are the rare/unexpected stimuli that generate MMN\n\n');
end

% Look for MMN component
if isfield(data, 'MMN') || isfield(data, 'mmn')
    fprintf('✓ MMN component found (difference wave)\n');
    fprintf('  This is typically Deviant - Standard\n\n');
end

%% Suggest visualization script
fprintf('================================================\n');
fprintf('RECOMMENDED NEXT STEPS\n');
fprintf('================================================\n\n');

fprintf('Run: visualize_ammn_events()\n');
fprintf('This will create comprehensive ERP visualizations including:\n');
fprintf('  - Event timeline and distribution\n');
fprintf('  - Standard vs Deviant comparison\n');
fprintf('  - MMN component analysis\n');
fprintf('  - Full 64-channel display\n');
fprintf('  - Statistical summaries\n\n');

end
