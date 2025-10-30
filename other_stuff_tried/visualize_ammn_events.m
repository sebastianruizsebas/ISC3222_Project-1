function [] = visualize_ammn_events()
%VISUALIZE_AMMN_EVENTS Load and visualize auditory mismatch negativity (aMMN) EEG data
%
% This script loads the aMMN_critical_events.mat file and creates comprehensive
% visualizations of the event-related potentials (ERPs) from the Mismatch Negativity task.
%
% DATA SOURCE:
%   - Recording: sub-01_ses-MMN_task-Optimum1 (64-channel EEG)
%   - Task: Auditory Oddball (aMMN)
%   - Sampling Rate: 2500 Hz
%   - Reference: Brain Vision Recorder
%
% CHANNEL LOCATIONS:
%   64 channels including: Frontal (Fp, AF, F), Central (C, Cz), Parietal (P, Pz),
%   Temporal (T), Occipital (O, Oz), and electrooculography (VEOG, HEOG)
%
% OUTPUT VISUALIZATIONS:
%   - Figure 1: Critical events timeline and markers
%   - Figure 2: ERP waveforms at key channels (Fz, Cz, Pz)
%   - Figure 3: Topographic map of MMN component
%   - Figure 4: Standard vs Deviant stimulus comparison
%   - Figure 5: Full 64-channel ERP display
%
% AUTHOR: AI Assistant
% DATE: October 30, 2025
% VERSION: 1.0

clear all; close all; clc;

%% 1. SETUP & CONFIGURATION
fprintf('==========================================\n');
fprintf('AUDITORY MISMATCH NEGATIVITY (aMMN) EEG\n');
fprintf('==========================================\n\n');

% Define file paths
datadir = pwd;
mat_file = fullfile(datadir, 'aMMN_critical_events.mat');

% EEG parameters from header file
fs = 2500;  % Sampling rate (Hz)
t_pre = -0.2;   % Pre-stimulus window (s)
t_post = 0.5;   % Post-stimulus window (s)

% Channel information (from vhdr file)
channel_names = {
    'Fp1', 'Fp2', 'Fpz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', ...
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', ...
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', ...
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', ...
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', ...
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz', ...
    'AF7', 'AF3', 'AF8', 'AF4', 'VEOG', 'HEOG'
};

% Key channels of interest for MMN
key_channels = {'Fz', 'FCz', 'Cz', 'CPz', 'Pz'};
key_indices = [];
for i = 1:length(key_channels)
    idx = find(strcmp(channel_names, key_channels{i}));
    if ~isempty(idx)
        key_indices = [key_indices, idx];
    end
end

fprintf('Loading data from: %s\n', mat_file);

%% 2. LOAD DATA
try
    data = load(mat_file);
    fprintf('✓ Successfully loaded aMMN_critical_events.mat\n\n');
catch
    fprintf('✗ ERROR: Could not load %s\n', mat_file);
    fprintf('Please ensure the file exists in the current directory.\n');
    return;
end

% Display loaded data structure
fprintf('DATA STRUCTURE:\n');
fprintf('  Variables in mat file:\n');
var_names = fieldnames(data);
for i = 1:length(var_names)
    var = getfield(data, var_names{i});
    if isnumeric(var)
        fprintf('    - %s: [%s] class: %s\n', var_names{i}, ...
            sprintf('%d ', size(var)), class(var));
    else
        fprintf('    - %s: %s\n', var_names{i}, class(var));
    end
end
fprintf('\n');

%% 3. EXTRACT KEY VARIABLES
% Try common variable names
if isfield(data, 'EEG')
    eeg_data = data.EEG;
elseif isfield(data, 'eeg')
    eeg_data = data.eeg;
else
    eeg_data = [];
end

if isfield(data, 'events')
    events = data.events;
elseif isfield(data, 'markers')
    events = data.markers;
else
    events = [];
end

if isfield(data, 'standard_trials')
    standard_trials = data.standard_trials;
else
    standard_trials = [];
end

if isfield(data, 'deviant_trials')
    deviant_trials = data.deviant_trials;
else
    deviant_trials = [];
end

%% 4. ANALYZE CRITICAL EVENTS
fprintf('CRITICAL EVENTS ANALYSIS:\n');
fprintf('======================================\n\n');

if ~isempty(events)
    if isstruct(events)
        event_types = {events.type};
        event_times = [events.latency];
        fprintf('Number of events: %d\n', length(events));
        fprintf('Event types: %s\n', sprintf('%s, ', event_types{:}));
    else
        fprintf('Number of events: %d\n', length(events));
    end
    fprintf('\n');
end

%% 5. CREATE VISUALIZATIONS

% Figure 1: Event Timeline
fprintf('Creating Figure 1: Critical Events Timeline...\n');
create_event_timeline(events, fs);

% Figure 2: Key Channel ERPs
fprintf('Creating Figure 2: Key Channel ERPs (Fz, Cz, Pz)...\n');
if ~isempty(eeg_data) && ~isempty(standard_trials) && ~isempty(deviant_trials)
    create_key_channel_erps(eeg_data, standard_trials, deviant_trials, ...
        channel_names, key_channels, fs, t_pre, t_post);
end

% Figure 3: Standard vs Deviant Comparison
fprintf('Creating Figure 3: Standard vs Deviant Comparison...\n');
if ~isempty(eeg_data) && ~isempty(standard_trials) && ~isempty(deviant_trials)
    create_oddball_comparison(eeg_data, standard_trials, deviant_trials, ...
        channel_names, fs, t_pre, t_post);
end

% Figure 4: Full 64-Channel Display
fprintf('Creating Figure 4: Full 64-Channel ERP Display...\n');
if ~isempty(eeg_data)
    create_full_channel_display(eeg_data, channel_names, fs, t_pre, t_post);
end

% Figure 5: Statistical Summary
fprintf('Creating Figure 5: Statistical Summary...\n');
if ~isempty(eeg_data) && ~isempty(standard_trials) && ~isempty(deviant_trials)
    create_statistical_summary(eeg_data, standard_trials, deviant_trials, ...
        channel_names, key_channels, fs, t_pre, t_post);
end

fprintf('\n✓ All visualizations complete!\n');
fprintf('==========================================\n');

end

%% HELPER FUNCTIONS

function [] = create_event_timeline(events, fs)
%CREATE_EVENT_TIMELINE Display critical events on a timeline
figure('Name', 'aMMN Events Timeline', 'NumberTitle', 'off', 'Position', [100 100 1200 600]);

if isempty(events)
    text(0.5, 0.5, 'No event data available', 'HorizontalAlignment', 'center');
    return;
end

if isstruct(events)
    event_times_sec = [events.latency] / fs;  % Convert to seconds
    event_types = {events.type};
    
    % Count event types
    unique_types = unique(event_types);
    colors = lines(length(unique_types));
    
    % Plot timeline
    subplot(2,1,1);
    hold on;
    for i = 1:length(unique_types)
        mask = strcmp(event_types, unique_types{i});
        scatter(event_times_sec(mask), ones(sum(mask), 1) * i, 50, colors(i,:), 'filled');
    end
    set(gca, 'YTick', 1:length(unique_types), 'YTickLabel', unique_types);
    xlabel('Time (s)');
    ylabel('Event Type');
    title('Critical Events Timeline');
    grid on;
    
    % Plot event histogram
    subplot(2,1,2);
    histogram(event_times_sec, 50, 'EdgeColor', 'black', 'FaceColor', [0.7 0.7 0.7]);
    xlabel('Time (s)');
    ylabel('Event Count');
    title('Event Frequency Distribution');
    grid on;
else
    text(0.5, 0.5, 'Event format not recognized', 'HorizontalAlignment', 'center');
end

end

function [] = create_key_channel_erps(eeg_data, standard_trials, deviant_trials, ...
    channel_names, key_channels, fs, t_pre, t_post)
%CREATE_KEY_CHANNEL_ERPS Display ERPs at key channels
figure('Name', 'Key Channel ERPs', 'NumberTitle', 'off', 'Position', [100 100 1200 800]);

num_key = length(key_channels);
t_vec = (t_pre:1/fs:t_post);

for i = 1:num_key
    % Find channel index
    ch_idx = find(strcmp(channel_names, key_channels{i}));
    if isempty(ch_idx)
        continue;
    end
    
    subplot(2, 3, i);
    
    % Extract and average standard and deviant responses
    if ~isempty(standard_trials)
        std_erp = mean(standard_trials(:, ch_idx, :), 3);
        plot(t_vec(1:length(std_erp)), std_erp, 'b-', 'LineWidth', 2, 'DisplayName', 'Standard');
        hold on;
    end
    
    if ~isempty(deviant_trials)
        dev_erp = mean(deviant_trials(:, ch_idx, :), 3);
        plot(t_vec(1:length(dev_erp)), dev_erp, 'r-', 'LineWidth', 2, 'DisplayName', 'Deviant');
    end
    
    % Compute MMN (Deviant - Standard)
    if ~isempty(standard_trials) && ~isempty(deviant_trials)
        mmn = dev_erp - std_erp;
        plot(t_vec(1:length(mmn)), mmn, 'g--', 'LineWidth', 2, 'DisplayName', 'MMN (Dev-Std)');
    end
    
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    title(key_channels{i});
    axline(0, 0, 'Color', 'k', 'LineStyle', ':');
    axline([], 0, 'Color', 'k', 'LineStyle', ':');
    grid on;
    if i == 1
        legend('Location', 'best');
    end
end

end

function [] = create_oddball_comparison(eeg_data, standard_trials, deviant_trials, ...
    channel_names, fs, t_pre, t_post)
%CREATE_ODDBALL_COMPARISON Compare standard vs deviant across channels
figure('Name', 'Standard vs Deviant Comparison', 'NumberTitle', 'off', 'Position', [100 100 1400 900]);

% Find central channels for detailed analysis
central_channels = {'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'F3', 'F4', 'C3', 'C4'};
subplot_idx = 1;

t_vec = (t_pre:1/fs:t_post);

for ch_name = central_channels
    ch_idx = find(strcmp(channel_names, ch_name{1}));
    if isempty(ch_idx)
        continue;
    end
    
    subplot(3, 3, subplot_idx);
    
    % Standard response
    if ~isempty(standard_trials)
        std_erp = mean(standard_trials(:, ch_idx, :), 3);
        std_sem = std(standard_trials(:, ch_idx, :), [], 3) / sqrt(size(standard_trials, 3));
        plot(t_vec(1:length(std_erp)), std_erp, 'b-', 'LineWidth', 2);
        hold on;
        fill_between(t_vec(1:length(std_erp)), std_erp - std_sem, std_erp + std_sem, 'b');
    end
    
    % Deviant response
    if ~isempty(deviant_trials)
        dev_erp = mean(deviant_trials(:, ch_idx, :), 3);
        dev_sem = std(deviant_trials(:, ch_idx, :), [], 3) / sqrt(size(deviant_trials, 3));
        plot(t_vec(1:length(dev_erp)), dev_erp, 'r-', 'LineWidth', 2);
        fill_between(t_vec(1:length(dev_erp)), dev_erp - dev_sem, dev_erp + dev_sem, 'r');
    end
    
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    title(ch_name{1});
    axline(0, 0, 'Color', 'k', 'LineStyle', ':');
    axline([], 0, 'Color', 'k', 'LineStyle', ':');
    grid on;
    
    subplot_idx = subplot_idx + 1;
    if subplot_idx > 9
        break;
    end
end

end

function [] = create_full_channel_display(eeg_data, channel_names, fs, t_pre, t_post)
%CREATE_FULL_CHANNEL_DISPLAY Display all 64 channels
figure('Name', 'Full 64-Channel Display', 'NumberTitle', 'off', 'Position', [100 100 1600 1000]);

t_vec = (t_pre:1/fs:t_post);
num_channels = min(64, size(eeg_data, 2));

% Create grid display
ncols = 8;
nrows = ceil(num_channels / ncols);

for i = 1:num_channels
    subplot(nrows, ncols, i);
    
    if i <= length(eeg_data)
        % Get time window
        t_start = max(1, round((t_pre * fs)));
        t_end = min(size(eeg_data, 1), round((t_post * fs)));
        
        if t_start < t_end
            plot(t_vec(1:(t_end - t_start + 1)), eeg_data(t_start:t_end, i), 'k-');
        end
    end
    
    title(channel_names{i}, 'FontSize', 8);
    set(gca, 'FontSize', 7);
    grid on;
end

end

function [] = create_statistical_summary(eeg_data, standard_trials, deviant_trials, ...
    channel_names, key_channels, fs, t_pre, t_post)
%CREATE_STATISTICAL_SUMMARY Compute and display statistical measures
figure('Name', 'Statistical Summary', 'NumberTitle', 'off', 'Position', [100 100 1200 800]);

% Peak analysis
subplot(2,2,1);
key_indices = [];
for i = 1:length(key_channels)
    idx = find(strcmp(channel_names, key_channels{i}));
    if ~isempty(idx)
        key_indices = [key_indices, idx];
    end
end

if ~isempty(standard_trials) && ~isempty(deviant_trials)
    std_peaks = [];
    dev_peaks = [];
    mmn_peaks = [];
    
    for i = key_indices
        std_erp = mean(standard_trials(:, i, :), 3);
        dev_erp = mean(deviant_trials(:, i, :), 3);
        mmn = dev_erp - std_erp;
        
        std_peaks = [std_peaks, min(std_erp)];
        dev_peaks = [dev_peaks, min(dev_erp)];
        mmn_peaks = [mmn_peaks, min(mmn)];
    end
    
    bar([std_peaks; dev_peaks; mmn_peaks]', 'grouped');
    set(gca, 'XTickLabel', key_channels);
    ylabel('Peak Amplitude (µV)');
    title('Peak Amplitude Comparison');
    legend('Standard', 'Deviant', 'MMN');
    grid on;
end

% Latency analysis
subplot(2,2,2);
if ~isempty(standard_trials) && ~isempty(deviant_trials)
    t_vec = (t_pre:1/fs:t_post);
    std_latencies = [];
    dev_latencies = [];
    mmn_latencies = [];
    
    for i = key_indices
        std_erp = mean(standard_trials(:, i, :), 3);
        dev_erp = mean(deviant_trials(:, i, :), 3);
        mmn = dev_erp - std_erp;
        
        % Find peak latencies
        [~, std_idx] = min(std_erp);
        [~, dev_idx] = min(dev_erp);
        [~, mmn_idx] = min(mmn);
        
        std_latencies = [std_latencies, t_vec(std_idx)*1000];  % Convert to ms
        dev_latencies = [dev_latencies, t_vec(dev_idx)*1000];
        mmn_latencies = [mmn_latencies, t_vec(mmn_idx)*1000];
    end
    
    bar([std_latencies; dev_latencies; mmn_latencies]', 'grouped');
    set(gca, 'XTickLabel', key_channels);
    ylabel('Latency (ms)');
    title('Peak Latency Comparison');
    legend('Standard', 'Deviant', 'MMN');
    grid on;
end

% Amplitude distribution across channels
subplot(2,2,3);
if ~isempty(deviant_trials)
    all_dev_peaks = [];
    channel_indices = 1:size(deviant_trials, 2);
    
    for i = channel_indices
        dev_erp = mean(deviant_trials(:, i, :), 3);
        all_dev_peaks = [all_dev_peaks, min(dev_erp)];
    end
    
    scatter(channel_indices, all_dev_peaks, 100, all_dev_peaks, 'filled');
    colormap('jet');
    colorbar;
    xlabel('Channel');
    ylabel('Peak Amplitude (µV)');
    title('Deviant Peak Amplitude Across All Channels');
    grid on;
end

% MMN statistics
subplot(2,2,4);
if ~isempty(standard_trials) && ~isempty(deviant_trials)
    mmn_amplitudes = [];
    
    for i = 1:min(size(deviant_trials, 2), 64)
        std_erp = mean(standard_trials(:, i, :), 3);
        dev_erp = mean(deviant_trials(:, i, :), 3);
        mmn = dev_erp - std_erp;
        mmn_amplitudes = [mmn_amplitudes, min(mmn)];
    end
    
    histogram(mmn_amplitudes, 20, 'EdgeColor', 'black', 'FaceColor', [0.7 0.7 0.7]);
    xlabel('MMN Amplitude (µV)');
    ylabel('Count');
    title('MMN Amplitude Distribution');
    grid on;
    
    % Add statistics text
    mean_mmn = mean(mmn_amplitudes);
    std_mmn = std(mmn_amplitudes);
    text(0.98, 0.97, sprintf('Mean: %.2f µV\nStd: %.2f µV', mean_mmn, std_mmn), ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'Units', 'normalized', 'BackgroundColor', 'white', 'EdgeColor', 'black');
end

end

function [] = fill_between(x, y1, y2, color)
%FILL_BETWEEN Create a filled area between two curves
fill([x, fliplr(x)], [y1, fliplr(y2)], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
