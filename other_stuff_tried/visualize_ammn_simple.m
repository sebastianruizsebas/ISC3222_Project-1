function [] = visualize_ammn_simple()
%VISUALIZE_AMMN_SIMPLE Simple and fast visualization of aMMN time series data

clear all; close all; clc;

fprintf('================================================\n');
fprintf('aMMN VISUALIZATION - SIMPLE VERSION\n');
fprintf('================================================\n\n');

% Load data
data = load('aMMN_critical_events.mat');
t0z = data.t0z;
t1z = data.t1z;
t2z = data.t2z;

fprintf('Data loaded: t0z (%d), t1z (%d), t2z (%d)\n\n', ...
    length(t0z), length(t1z), length(t2z));

%% Figure 1: Overview
fprintf('Creating Figure 1: Time Series Comparison...\n');
figure('Position', [100 100 1200 600]);

time_t1 = linspace(0, 1, length(t1z));
time_t2 = linspace(0, 1, length(t2z));

subplot(2,2,1);
plot(time_t1, t1z, 'b-', 'LineWidth', 2);
hold on;
plot(time_t2, t2z, 'r-', 'LineWidth', 2);
xlabel('Normalized Time');
ylabel('Amplitude');
title('Standard (blue) vs Deviant (red)');
legend('Standard', 'Deviant');
grid on;

subplot(2,2,2);
mmn = t2z - t1z;
plot(time_t2, mmn, 'g-', 'LineWidth', 2.5);
xlabel('Normalized Time');
ylabel('Amplitude');
title('MMN (Deviant - Standard)');
grid on;

subplot(2,2,3);
histogram(t1z, 40, 'FaceColor', 'b', 'FaceAlpha', 0.6);
hold on;
histogram(t2z, 40, 'FaceColor', 'r', 'FaceAlpha', 0.6);
xlabel('Amplitude');
ylabel('Frequency');
title('Amplitude Distribution');
legend('Standard', 'Deviant');
grid on;

subplot(2,2,4);
plot(t0z, 'ko-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Sample');
ylabel('Amplitude');
title('Baseline/Reference (t0z)');
grid on;

%% Figure 2: Statistics
fprintf('Creating Figure 2: Statistical Summary...\n');
figure('Position', [100 100 1200 500]);

[h, p_val, ci, stats] = ttest2(t1z, t2z);

% Statistics table
subplot(1,3,1);
hold off;
axis off;
y_pos = 0.9;
line_height = 0.08;

text_str = sprintf('STATISTICAL SUMMARY\n\n');
text_str = [text_str, sprintf('Standard (t1z):\n')];
text_str = [text_str, sprintf('  Mean: %.6f\n', mean(t1z))];
text_str = [text_str, sprintf('  Std:  %.6f\n', std(t1z))];
text_str = [text_str, sprintf('  Min:  %.6f\n', min(t1z))];
text_str = [text_str, sprintf('  Max:  %.6f\n', max(t1z))];

text_str = [text_str, sprintf('\nDeviant (t2z):\n')];
text_str = [text_str, sprintf('  Mean: %.6f\n', mean(t2z))];
text_str = [text_str, sprintf('  Std:  %.6f\n', std(t2z))];
text_str = [text_str, sprintf('  Min:  %.6f\n', min(t2z))];
text_str = [text_str, sprintf('  Max:  %.6f\n', max(t2z))];

text_str = [text_str, sprintf('\nMMN (Deviant-Std):\n')];
text_str = [text_str, sprintf('  Mean: %.6f\n', mean(mmn))];
text_str = [text_str, sprintf('  Std:  %.6f\n', std(mmn))];

text(0.05, y_pos, text_str, 'VerticalAlignment', 'top', 'FontSize', 10);

% T-test results
subplot(1,3,2);
axis off;
ttest_str = sprintf('T-TEST RESULTS\n');
ttest_str = [ttest_str, sprintf('(Independent samples)\n\n')];
ttest_str = [ttest_str, sprintf('p-value: %.6f\n', p_val)];
ttest_str = [ttest_str, sprintf('t-stat: %.4f\n', stats.tstat)];
ttest_str = [ttest_str, sprintf('df: %d\n', stats.df)];
ttest_str = [ttest_str, sprintf('95%% CI: [%.6f, %.6f]\n\n', ci(1), ci(2))];
if h
    ttest_str = [ttest_str, sprintf('Result: SIGNIFICANT\n')];
else
    ttest_str = [ttest_str, sprintf('Result: NOT significant\n')];
end
ttest_str = [ttest_str, sprintf('(alpha = 0.05)\n')];

text(0.05, y_pos, ttest_str, 'VerticalAlignment', 'top', 'FontSize', 10);

% Bar plot comparison
subplot(1,3,3);
means = [mean(t1z), mean(t2z), mean(mmn)];
errs = [std(t1z), std(t2z), std(mmn)];
bar(1:3, means, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'black', 'LineWidth', 2);
hold on;
errorbar(1:3, means, errs, 'k.', 'LineWidth', 2, 'MarkerSize', 10);
set(gca, 'XTickLabel', {'Standard', 'Deviant', 'MMN'});
ylabel('Mean Amplitude');
title('Mean ± Std Dev');
grid on;

%% Figure 3: Spectral Analysis
fprintf('Creating Figure 3: Power Spectrum...\n');
figure('Position', [100 100 1200 500]);

fs = 2500;

[pxx_std, f_std] = pwelch(t1z, [], [], 1024, fs);
[pxx_dev, f_dev] = pwelch(t2z, [], [], 1024, fs);

subplot(1,3,1);
semilogy(f_std, pxx_std, 'b-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power (V²/Hz)');
title('Standard - Power Spectrum');
grid on;
xlim([0, 100]);

subplot(1,3,2);
semilogy(f_dev, pxx_dev, 'r-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power (V²/Hz)');
title('Deviant - Power Spectrum');
grid on;
xlim([0, 100]);

subplot(1,3,3);
semilogy(f_std, pxx_std, 'b-', 'LineWidth', 2);
hold on;
semilogy(f_dev, pxx_dev, 'r-', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power (V²/Hz)');
title('Comparison');
legend('Standard', 'Deviant', 'Location', 'best');
grid on;
xlim([0, 100]);

%% Figure 4: Detailed Analysis
fprintf('Creating Figure 4: Detailed Waveforms...\n');
figure('Position', [100 100 1200 700]);

n_seg = 6;
seg_len = floor(length(t1z) / n_seg);

for i = 1:n_seg
    subplot(2,3,i);
    start_idx = (i-1)*seg_len + 1;
    end_idx = min(i*seg_len, length(t1z));
    idx = start_idx:end_idx;
    
    t_seg = linspace(0, 1, length(idx));
    plot(t_seg, t1z(idx), 'b-', 'LineWidth', 2);
    hold on;
    plot(t_seg, t2z(idx), 'r-', 'LineWidth', 2);
    
    xlabel('Time');
    ylabel('Amplitude');
    title(sprintf('Segment %d (samples %d-%d)', i, start_idx, end_idx));
    if i == 1
        legend('Standard', 'Deviant', 'Location', 'best');
    end
    grid on;
end

%% Print Summary
fprintf('\n================================================\n');
fprintf('ANALYSIS RESULTS\n');
fprintf('================================================\n\n');

fprintf('STANDARD (t1z) - %d samples:\n', length(t1z));
fprintf('  Mean:     %.8f\n', mean(t1z));
fprintf('  Std Dev:  %.8f\n', std(t1z));
fprintf('  Range:    [%.8f, %.8f]\n', min(t1z), max(t1z));
fprintf('  Median:   %.8f\n\n', median(t1z));

fprintf('DEVIANT (t2z) - %d samples:\n', length(t2z));
fprintf('  Mean:     %.8f\n', mean(t2z));
fprintf('  Std Dev:  %.8f\n', std(t2z));
fprintf('  Range:    [%.8f, %.8f]\n', min(t2z), max(t2z));
fprintf('  Median:   %.8f\n\n', median(t2z));

fprintf('MMN (Deviant - Standard):\n');
fprintf('  Mean:     %.8f\n', mean(mmn));
fprintf('  Std Dev:  %.8f\n', std(mmn));
fprintf('  Range:    [%.8f, %.8f]\n', min(mmn), max(mmn));
fprintf('  Median:   %.8f\n\n', median(mmn));

fprintf('T-TEST COMPARISON (Standard vs Deviant):\n');
fprintf('  p-value:  %.8f %s\n', p_val, iif(h, '(SIGNIFICANT)', '(not significant)'));
fprintf('  t-stat:   %.6f\n', stats.tstat);
fprintf('  df:       %d\n', stats.df);
fprintf('  95%% CI:   [%.8f, %.8f]\n\n', ci(1), ci(2));

fprintf('✓ All visualizations complete!\n');
fprintf('================================================\n\n');

end

function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
