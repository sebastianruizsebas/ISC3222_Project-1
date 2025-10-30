function [] = visualize_ammn_timeseries()
%VISUALIZE_AMMN_TIMESERIES Visualize aMMN critical events time series data
%
% The aMMN_critical_events.mat contains three time series:
%   - t0z: 10 samples (likely baseline or reference)
%   - t1z: 720 samples (likely standard stimulus responses)
%   - t2z: 720 samples (likely deviant stimulus responses)
%
% This script creates comprehensive visualizations of these time series,
% including comparison plots, spectral analysis, and statistical summaries.

clear all; close all; clc;

fprintf('================================================\n');
fprintf('AUDITORY MISMATCH NEGATIVITY (aMMN) ANALYSIS\n');
fprintf('Critical Events Time Series\n');
fprintf('================================================\n\n');

%% Load data
mat_file = 'aMMN_critical_events.mat';
if ~exist(mat_file, 'file')
    fprintf('ERROR: %s not found\n', mat_file);
    return;
end

data = load(mat_file);
fprintf('Data loaded successfully\n\n');

t0z = data.t0z;  % Baseline/reference (10 samples)
t1z = data.t1z;  % Standard trials (720 samples)
t2z = data.t2z;  % Deviant trials (720 samples)

fprintf('Data Summary:\n');
fprintf('  t0z (Baseline):  %d samples, Range: [%.2f, %.2f]\n', ...
    length(t0z), min(t0z), max(t0z));
fprintf('  t1z (Standard):  %d samples, Range: [%.2f, %.2f]\n', ...
    length(t1z), min(t1z), max(t1z));
fprintf('  t2z (Deviant):   %d samples, Range: [%.2f, %.2f]\n', ...
    length(t2z), min(t2z), max(t2z));
fprintf('\n');

%% Figure 1: Time Series Overview
fprintf('Creating Figure 1: Time Series Overview...\n');
figure('Name', 'aMMN Time Series Overview', 'NumberTitle', 'off', 'Position', [50 100 1400 700]);

% Plot 1: All three time series
subplot(2,3,1);
time_t1 = linspace(0, 1, length(t1z));
time_t2 = linspace(0, 1, length(t2z));
plot(time_t1, t1z, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Standard (t1z)');
hold on;
plot(time_t2, t2z, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Deviant (t2z)');
xlabel('Normalized Time');
ylabel('Amplitude');
title('Standard vs Deviant Time Series');
legend('Location', 'best');
grid on;

% Plot 2: Difference (MMN)
subplot(2,3,2);
mmn = t2z - t1z;
plot(time_t2, mmn, 'g-', 'LineWidth', 2, 'DisplayName', 'MMN (Deviant - Standard)');
xlabel('Normalized Time');
ylabel('Amplitude');
title('Mismatch Negativity (MMN)');
grid on;
legend('Location', 'best');

% Plot 3: Baseline/Reference
subplot(2,3,3);
if length(t0z) > 0
    plot(t0z, 'ko-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Baseline (t0z)');
    xlabel('Sample');
    ylabel('Amplitude');
    title('Baseline/Reference Signal');
    grid on;
    legend('Location', 'best');
end

% Plot 4: Standard close-up (early samples)
subplot(2,3,4);
window = 1:min(100, length(t1z));
plot(time_t1(window), t1z(window), 'b-', 'LineWidth', 2);
xlabel('Normalized Time');
ylabel('Amplitude');
title('Standard - Early Response (first 100 samples)');
grid on;

% Plot 5: Deviant close-up (early samples)
subplot(2,3,5);
plot(time_t2(window), t2z(window), 'r-', 'LineWidth', 2);
xlabel('Normalized Time');
ylabel('Amplitude');
title('Deviant - Early Response (first 100 samples)');
grid on;

% Plot 6: Peak finding
subplot(2,3,6);
[pk_std, loc_std] = findpeaks(-t1z, 'NPeaks', 5);
[pk_dev, loc_dev] = findpeaks(-t2z, 'NPeaks', 5);
plot(time_t1, t1z, 'b-', 'LineWidth', 1);
hold on;
plot(time_t2, t2z, 'r-', 'LineWidth', 1);
scatter(time_t1(loc_std), t1z(loc_std), 100, 'bo', 'filled', 'DisplayName', 'Standard peaks');
scatter(time_t2(loc_dev), t2z(loc_dev), 100, 'rs', 'filled', 'DisplayName', 'Deviant peaks');
xlabel('Normalized Time');
ylabel('Amplitude');
title('Peak Detection');
legend('Location', 'best');
grid on;

%% Figure 2: Statistical Comparison
fprintf('Creating Figure 2: Statistical Comparison...\n');
figure('Name', 'aMMN Statistical Analysis', 'NumberTitle', 'off', 'Position', [50 100 1400 700]);

% Plot 1: Distribution
subplot(2,3,1);
histogram(t1z, 30, 'FaceColor', 'b', 'FaceAlpha', 0.6, 'DisplayName', 'Standard');
hold on;
histogram(t2z, 30, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'DisplayName', 'Deviant');
xlabel('Amplitude');
ylabel('Frequency');
title('Amplitude Distribution');
legend('Location', 'best');
grid on;

% Plot 2: Box plot comparison
subplot(2,3,2);
boxplot([t1z' t2z'], {'Standard', 'Deviant'});
ylabel('Amplitude');
title('Amplitude Distribution (Box Plot)');
grid on;

% Plot 3: Cumulative distribution
subplot(2,3,3);
[f1, x1] = ecdf(t1z);
[f2, x2] = ecdf(t2z);
plot(x1, f1, 'b-', 'LineWidth', 2, 'DisplayName', 'Standard');
hold on;
plot(x2, f2, 'r-', 'LineWidth', 2, 'DisplayName', 'Deviant');
xlabel('Amplitude');
ylabel('Cumulative Probability');
title('Cumulative Distribution Function');
legend('Location', 'best');
grid on;

% Plot 4: Mean and error bars
subplot(2,3,4);
means = [mean(t1z), mean(t2z)];
stds = [std(t1z), std(t2z)];
sems = stds / sqrt([length(t1z), length(t2z)]);
bar([1, 2], means, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'black', 'LineWidth', 2);
hold on;
errorbar([1, 2], means, sems, 'k.', 'LineWidth', 2, 'MarkerSize', 15);
set(gca, 'XTickLabel', {'Standard', 'Deviant'});
ylabel('Mean Amplitude');
title('Mean ± SEM');
grid on;

% Plot 5: Statistics table as text
subplot(2,3,5);
axis off;
stats_text = sprintf('STATISTICS\n\nStandard (t1z):\n  Mean: %.4f\n  Std Dev: %.4f\n  Min: %.4f\n  Max: %.4f\n  N: %d\n\nDeviant (t2z):\n  Mean: %.4f\n  Std Dev: %.4f\n  Min: %.4f\n  Max: %.4f\n  N: %d\n\nDifference (MMN):\n  Mean: %.4f\n  Std Dev: %.4f', ...
    mean(t1z), std(t1z), min(t1z), max(t1z), length(t1z), ...
    mean(t2z), std(t2z), min(t2z), max(t2z), length(t2z), ...
    mean(mmn), std(mmn));
text(0.1, 0.5, stats_text, 'FontSize', 10, 'FontName', 'Courier', ...
    'VerticalAlignment', 'middle', 'BackgroundColor', 'white', 'EdgeColor', 'black');

% Plot 6: T-test results
subplot(2,3,6);
[h, p_val, ci, stats_ttest] = ttest2(t1z, t2z);
axis off;
ttest_text = sprintf('T-TEST RESULTS\n\nH0: Means are equal\n\np-value: %.6f\nt-statistic: %.4f\ndf: %d\n95%% CI: [%.4f, %.4f]\n\nSignificant: %s\nalpha = 0.05', ...
    p_val, stats_ttest.tstat, stats_ttest.df, ci(1), ci(2), onoff(h));
text(0.1, 0.5, ttest_text, 'FontSize', 10, 'FontName', 'Courier', ...
    'VerticalAlignment', 'middle', 'BackgroundColor', 'white', 'EdgeColor', 'black');

%% Figure 3: Spectral Analysis
fprintf('Creating Figure 3: Spectral Analysis...\n');
figure('Name', 'aMMN Spectral Analysis', 'NumberTitle', 'off', 'Position', [50 100 1400 700]);

% Assume sampling rate of 2500 Hz based on header file
fs = 2500;

% Plot 1: Power spectral density - Standard
subplot(2,3,1);
[pxx_std, f_std] = pwelch(t1z, [], [], 1024, fs);
semilogy(f_std, pxx_std, 'b-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (V²/Hz)');
title('Standard - Power Spectrum');
grid on;

% Plot 2: Power spectral density - Deviant
subplot(2,3,2);
[pxx_dev, f_dev] = pwelch(t2z, [], [], 1024, fs);
semilogy(f_dev, pxx_dev, 'r-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (V²/Hz)');
title('Deviant - Power Spectrum');
grid on;

% Plot 3: PSD Comparison
subplot(2,3,3);
semilogy(f_std, pxx_std, 'b-', 'LineWidth', 2, 'DisplayName', 'Standard');
hold on;
semilogy(f_dev, pxx_dev, 'r-', 'LineWidth', 2, 'DisplayName', 'Deviant');
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (V²/Hz)');
title('Power Spectrum Comparison');
legend('Location', 'best');
grid on;
xlim([0, 100]);

% Plot 4: FFT magnitudes
subplot(2,3,4);
fft_std = abs(fft(t1z - mean(t1z)));
fft_dev = abs(fft(t2z - mean(t2z)));
freqs = linspace(0, fs, length(fft_std));
plot(freqs(1:100), fft_std(1:100), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Standard');
hold on;
plot(freqs(1:100), fft_dev(1:100), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Deviant');
xlabel('Frequency (Hz)');
ylabel('FFT Magnitude');
title('FFT Comparison');
legend('Location', 'best');
grid on;

% Plot 5: Coherence between Standard and Deviant
subplot(2,3,5);
[cxy, f_coh] = mscohere(t1z, t2z, [], [], 1024, fs);
plot(f_coh, cxy, 'k-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Coherence');
title('Coherence: Standard vs Deviant');
grid on;
ylim([0 1]);

% Plot 6: Spectrogram
subplot(2,3,6);
% Concatenate for spectrogram
combined = [t1z t2z];
spectrogram(combined, 256, 128, 1024, fs, 'yaxis');
title('Spectrogram: Standard + Deviant');
colorbar;

%% Figure 4: Detailed Waveform Analysis
fprintf('Creating Figure 4: Detailed Waveform Analysis...\n');
figure('Name', 'aMMN Waveform Details', 'NumberTitle', 'off', 'Position', [50 100 1400 700]);

% Divide time series into segments
n_segments = 6;
segment_len = floor(length(t1z) / n_segments);

for i = 1:n_segments
    subplot(2, 3, i);
    start_idx = (i-1) * segment_len + 1;
    end_idx = min(i * segment_len, length(t1z));
    
    seg_idx = start_idx:end_idx;
    segment_time = linspace(0, 1, length(seg_idx));
    
    plot(segment_time, t1z(seg_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'Standard');
    hold on;
    plot(segment_time, t2z(seg_idx), 'r-', 'LineWidth', 2, 'DisplayName', 'Deviant');
    
    xlabel('Time (within segment)');
    ylabel('Amplitude');
    title(sprintf('Segment %d (samples %d-%d)', i, start_idx, end_idx));
    if i == 1
        legend('Location', 'best');
    end
    grid on;
end

%% Summary statistics
fprintf('\n================================================\n');
fprintf('ANALYSIS SUMMARY\n');
fprintf('================================================\n\n');

fprintf('WAVEFORM STATISTICS:\n');
fprintf('Standard (t1z):\n');
fprintf('  Mean: %.6f\n', mean(t1z));
fprintf('  Std Dev: %.6f\n', std(t1z));
fprintf('  Min/Max: %.6f / %.6f\n\n', min(t1z), max(t1z));

fprintf('Deviant (t2z):\n');
fprintf('  Mean: %.6f\n', mean(t2z));
fprintf('  Std Dev: %.6f\n', std(t2z));
fprintf('  Min/Max: %.6f / %.6f\n\n', min(t2z), max(t2z));

fprintf('MMN (Deviant - Standard):\n');
fprintf('  Mean: %.6f\n', mean(mmn));
fprintf('  Std Dev: %.6f\n', std(mmn));
fprintf('  Min/Max: %.6f / %.6f\n\n', min(mmn), max(mmn));

% T-test
fprintf('STATISTICAL TEST (Independent t-test):\n');
fprintf('  p-value: %.6f %s\n', p_val, onoff(h));
fprintf('  t-statistic: %.4f\n', stats_ttest.tstat);
fprintf('  Degrees of freedom: %d\n', stats_ttest.df);
fprintf('  95%% CI of difference: [%.6f, %.6f]\n\n', ci(1), ci(2));

% Peak analysis
fprintf('PEAK ANALYSIS (Top 5 peaks):\n');
fprintf('Standard peaks:\n');
for j = 1:min(5, length(loc_std))
    fprintf('  Peak %d: %.6f at position %.0f\n', j, -pk_std(j), loc_std(j));
end
fprintf('\nDeviant peaks:\n');
for j = 1:min(5, length(loc_dev))
    fprintf('  Peak %d: %.6f at position %.0f\n', j, -pk_dev(j), loc_dev(j));
end

fprintf('\n✓ All visualizations complete!\n');
fprintf('================================================\n\n');

end

function result = onoff(h)
    if h
        result = '(SIGNIFICANT)';
    else
        result = '(not significant)';
    end
end
