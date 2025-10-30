function [] = visualize_eeg_data()
%VISUALIZE_EEG_DATA Load and visualize Brain Vision EEG data
%
% Loads the BrainVision format EEG file (sub-01_ses-MMN_task-Optimum1_eeg.eeg)
% and creates comprehensive visualizations of the 64-channel recording.
%
% FILE FORMAT:
%   - Header: sub-01_ses-MMN_task-Optimum1_eeg.vhdr
%   - Markers: sub-01_ses-MMN_task-Optimum1_eeg.vmrk
%   - Data: sub-01_ses-MMN_task-Optimum1_eeg.eeg (binary)
%
% RECORDING PARAMETERS:
%   - Channels: 64
%   - Sampling rate: 2500 Hz
%   - Resolution: 0.5 µV per bit
%   - Format: INT16 multiplexed
%

clear all; close all; clc;

fprintf('================================================\n');
fprintf('BRAIN VISION EEG DATA VIEWER\n');
fprintf('Auditory Mismatch Negativity (aMMN) Task\n');
fprintf('================================================\n\n');

%% File paths
vhdr_file = 'sub-01_ses-MMN_task-Optimum1_eeg.vhdr';
eeg_file = 'sub-01_ses-MMN_task-Optimum1_eeg.eeg';
vmrk_file = 'sub-01_ses-MMN_task-Optimum1_eeg.vmrk';

% Check files exist
if ~exist(vhdr_file, 'file')
    fprintf('ERROR: %s not found\n', vhdr_file);
    return;
end

fprintf('Files found:\n');
fprintf('  ✓ %s\n', vhdr_file);
fprintf('  ✓ %s\n', eeg_file);
fprintf('  ✓ %s\n\n', vmrk_file);

%% Parse header file
fprintf('Parsing header file...\n');
header = parse_vhdr(vhdr_file);

fprintf('Recording info:\n');
fprintf('  Channels: %d\n', header.num_channels);
fprintf('  Sampling rate: %d Hz\n', header.fs);
fprintf('  Resolution: %.1f µV/bit\n', header.resolution);
fprintf('  Data format: %s\n', header.data_format);
fprintf('  Byte order: %s\n', header.byte_order);

%% Get file info and load chunked data
finfo = dir(eeg_file);
file_size_bytes = finfo.bytes;
num_samples_total = file_size_bytes / (header.num_channels * 2);
duration_sec = num_samples_total / header.fs;

fprintf('\nFile info:\n');
fprintf('  Size: %.2f MB\n', file_size_bytes / 1e6);
fprintf('  Total samples: %d\n', num_samples_total);
fprintf('  Total duration: %.2f seconds\n\n', duration_sec);

% Load first 30 seconds to avoid memory issues
fprintf('Loading first 30 seconds of EEG data...\n');
load_duration = min(30, duration_sec);
samples_to_load = round(load_duration * header.fs);

fid = fopen(eeg_file, 'r');
if fid == -1
    fprintf('ERROR: Cannot open %s\n', eeg_file);
    return;
end

data_int16 = fread(fid, samples_to_load * header.num_channels, 'int16');
fclose(fid);

% Reshape and convert
data_raw = reshape(data_int16, header.num_channels, samples_to_load)';
eeg_data = data_raw * header.resolution;

fprintf('Loaded: %d samples (%.2f seconds)\n', size(eeg_data, 1), load_duration);

%% Channel info
channel_names = {
    'Fp1', 'Fp2', 'Fpz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', ...
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', ...
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', ...
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', ...
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', ...
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz', ...
    'AF7', 'AF3', 'AF8', 'AF4', 'VEOG', 'HEOG'
};

time_vec = (0:size(eeg_data,1)-1) / header.fs;

%% Figure 1: Full recording overview
fprintf('Creating Figure 1: Full Recording Overview...\n');
figure('Name', 'EEG Full Recording', 'NumberTitle', 'off', 'Position', [50 50 1400 800]);

% Plot 1: One channel
subplot(2,2,1);
plot(time_vec, eeg_data(:, 25), 'b-', 'LineWidth', 0.5);  % Cz channel
xlabel('Time (s)');
ylabel('Amplitude (µV)');
title('Cz - Full Recording');
grid on;

% Plot 2: Early portion (first 5 seconds)
subplot(2,2,2);
plot(time_vec(1:5*header.fs), eeg_data(1:5*header.fs, 25), 'b-', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude (µV)');
title('Cz - First 5 Seconds');
grid on;

% Plot 3: RMS over time (sliding window)
subplot(2,2,3);
win_size = header.fs * 1;  % 1 second windows
rms_vals = zeros(floor(size(eeg_data,1)/win_size), 1);
for i = 1:length(rms_vals)
    idx_start = (i-1)*win_size + 1;
    idx_end = min(i*win_size, size(eeg_data,1));
    rms_vals(i) = rms(eeg_data(idx_start:idx_end, 25));
end
time_rms = linspace(0, time_vec(end), length(rms_vals));
plot(time_rms, rms_vals, 'g-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('RMS Amplitude (µV)');
title('RMS Activity Over Time (Cz)');
grid on;

% Plot 4: Distribution
subplot(2,2,4);
histogram(eeg_data(:, 25), 100, 'FaceColor', 'b', 'FaceAlpha', 0.7);
xlabel('Amplitude (µV)');
ylabel('Frequency');
title('Amplitude Distribution (Cz)');
grid on;

%% Figure 2: Multi-channel view (early data)
fprintf('Creating Figure 2: Multi-Channel Overview (Early Data)...\n');
figure('Name', 'Multi-Channel View', 'NumberTitle', 'off', 'Position', [50 50 1400 900]);

% Select first 2 seconds of data
time_window = 2;
idx_end = min(time_window * header.fs, size(eeg_data, 1));
time_seg = time_vec(1:idx_end);
data_seg = eeg_data(1:idx_end, :);

% Plot key channels
key_channels = {'Fp1', 'Fp2', 'Fz', 'C3', 'Cz', 'C4', 'Pz', 'Oz'};
key_indices = [];
for i = 1:length(key_channels)
    idx = find(strcmp(channel_names, key_channels{i}));
    if ~isempty(idx)
        key_indices = [key_indices, idx];
    end
end

for i = 1:length(key_indices)
    subplot(2, 4, i);
    ch_idx = key_indices(i);
    plot(time_seg, data_seg(:, ch_idx), 'b-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('µV');
    title(channel_names{ch_idx});
    grid on;
end

%% Figure 3: Spectral analysis
fprintf('Creating Figure 3: Spectral Analysis...\n');
figure('Name', 'Spectral Analysis', 'NumberTitle', 'off', 'Position', [50 50 1400 700]);

% FFT of Cz channel
cz_idx = 25;
[pxx, f] = pwelch(eeg_data(:, cz_idx), 1024, 512, 2048, header.fs);

subplot(1,2,1);
semilogy(f, pxx, 'b-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power (µV²/Hz)');
title('Power Spectrum (Cz)');
grid on;
xlim([0, 100]);

% Spectrogram
subplot(1,2,2);
spectrogram(eeg_data(:, cz_idx), 256, 128, 512, header.fs, 'yaxis');
title('Spectrogram (Cz)');
colorbar;

%% Figure 4: Amplitude statistics across channels
fprintf('Creating Figure 4: Channel Statistics...\n');
figure('Name', 'Channel Statistics', 'NumberTitle', 'off', 'Position', [50 50 1400 600]);

% Compute statistics per channel
rms_per_channel = zeros(1, header.num_channels);
mean_per_channel = zeros(1, header.num_channels);
std_per_channel = zeros(1, header.num_channels);

for i = 1:header.num_channels
    rms_per_channel(i) = rms(eeg_data(:, i));
    mean_per_channel(i) = mean(eeg_data(:, i));
    std_per_channel(i) = std(eeg_data(:, i));
end

subplot(1,3,1);
bar(1:header.num_channels, rms_per_channel, 'FaceColor', [0.2 0.2 0.8]);
xlabel('Channel');
ylabel('RMS Amplitude (µV)');
title('RMS per Channel');
grid on;
set(gca, 'XTick', [1, 10, 20, 30, 40, 50, 60]);

subplot(1,3,2);
bar(1:header.num_channels, mean_per_channel, 'FaceColor', [0.2 0.8 0.2]);
xlabel('Channel');
ylabel('Mean Amplitude (µV)');
title('Mean per Channel');
grid on;
set(gca, 'XTick', [1, 10, 20, 30, 40, 50, 60]);

subplot(1,3,3);
bar(1:header.num_channels, std_per_channel, 'FaceColor', [0.8 0.2 0.2]);
xlabel('Channel');
ylabel('Std Dev (µV)');
title('Std Dev per Channel');
grid on;
set(gca, 'XTick', [1, 10, 20, 30, 40, 50, 60]);

%% Print statistics
fprintf('\n================================================\n');
fprintf('DATA STATISTICS\n');
fprintf('================================================\n\n');

fprintf('RECORDING SUMMARY:\n');
fprintf('  Loaded duration: %.2f seconds\n', load_duration);
fprintf('  Total recording: %.2f seconds\n', duration_sec);
fprintf('  Sample rate: %d Hz\n', header.fs);
fprintf('  Channels: %d\n\n', header.num_channels);

fprintf('AMPLITUDE STATISTICS (µV):\n');
fprintf('  Min: %.4f\n', min(eeg_data(:)));
fprintf('  Max: %.4f\n', max(eeg_data(:)));
fprintf('  Mean: %.4f\n', mean(eeg_data(:)));
fprintf('  Std: %.4f\n\n', std(eeg_data(:)));

% Compute RMS and std for all channels
rms_per_ch = zeros(1, header.num_channels);
std_per_ch = zeros(1, header.num_channels);
for i = 1:header.num_channels
    rms_per_ch(i) = rms(eeg_data(:, i));
    std_per_ch(i) = std(eeg_data(:, i));
end

fprintf('TOP CHANNELS BY RMS:\n');
[~, idx_rms] = sort(rms_per_ch, 'descend');
for i = 1:min(10, length(idx_rms))
    ch_idx = idx_rms(i);
    if ch_idx <= length(channel_names)
        fprintf('  %d. %s: %.4f µV\n', i, channel_names{ch_idx}, rms_per_ch(ch_idx));
    end
end

fprintf('\nTOP CHANNELS BY STD DEV:\n');
[~, idx_std] = sort(std_per_ch, 'descend');
for i = 1:min(10, length(idx_std))
    ch_idx = idx_std(i);
    if ch_idx <= length(channel_names)
        fprintf('  %d. %s: %.4f µV\n', i, channel_names{ch_idx}, std_per_ch(ch_idx));
    end
end

fprintf('\n✓ Visualization complete!\n');
fprintf('================================================\n\n');

end

%% Helper function to parse VHDR file
function header = parse_vhdr(vhdr_file)
    header.num_channels = [];
    header.fs = [];
    header.resolution = 0.5;  % Default: 0.5 µV per bit
    header.data_format = 'BINARY';
    header.byte_order = 'little_endian';
    
    fid = fopen(vhdr_file, 'r');
    if fid == -1
        error('Cannot open %s', vhdr_file);
    end
    
    line_no = 0;
    while ~feof(fid)
        line = fgetl(fid);
        line_no = line_no + 1;
        
        if strncmpi(line, 'NumberOfChannels=', 17)
            header.num_channels = sscanf(line(18:end), '%d');
        elseif strncmpi(line, 'SamplingInterval=', 17)
            % SamplingInterval is in microseconds
            sampling_interval_us = sscanf(line(18:end), '%d');
            header.fs = round(1e6 / sampling_interval_us);
        elseif strncmpi(line, 'BinaryFormat=', 13)
            header.byte_order = line(14:end);
        end
    end
    fclose(fid);
    
    % Validate
    if isempty(header.num_channels)
        error('Could not find NumberOfChannels in header');
    end
    if isempty(header.fs)
        error('Could not find SamplingInterval in header');
    end
end
