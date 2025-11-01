function nan_analysis()
    % NAN_ANALYSIS - Comprehensive analysis of nan_snapshot.mat
    % =========================================================================
    % This script analyzes a snapshot of NaN failures captured during model
    % execution. It identifies which variables contain NaNs, when they occur,
    % and provides diagnostics for debugging numerical instabilities.
    %
    % Usage:
    %   nan_analysis()
    %
    % Output:
    %   - Console diagnostics (NaN locations, statistics)
    %   - Visualizations (heatmaps, time series)
    %   - Exported analysis to CSV for further inspection

    fprintf('\n');
    fprintf('╔═════════════════════════════════════════════════════════════╗\n');
    fprintf('║  NaN SNAPSHOT ANALYSIS - Dual Hierarchy Model Diagnostics  ║\n');
    fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

    % ====================================================================
    % LOAD SNAPSHOT FILE
    % ====================================================================

    snapshot_file = './nan_snapshot.mat';
    if ~isfile(snapshot_file)
        fprintf('ERROR: nan_snapshot.mat not found in ./figures/\n');
        fprintf('Run the model with save_results=false to generate a snapshot on NaN detection.\n');
        return;
    end

    fprintf('Loading snapshot file: %s\n\n', snapshot_file);
    try
        S = load(snapshot_file);
    catch ME
        fprintf('ERROR loading snapshot: %s\n', ME.message);
        return;
    end

    % ====================================================================
    % EXTRACT SNAPSHOT VARIABLES
    % ====================================================================

    % Expected snapshot variables from hierarchical_step_update.m
    % (Note: inline comments removed to avoid cell array syntax issues)
    expected_vars = {
        'i'; ...
        'E_L1_motor'; 'E_L2_motor'; ...
        'E_L1_plan'; 'E_L2_plan'; ...
        'R_L1_motor'; 'R_L2_motor'; 'R_L3_motor'; ...
        'R_L1_plan'; 'R_L2_plan'; 'R_L3_plan'; ...
        'R_L0'; ...
        'W_motor_L2_to_L1'; 'W_motor_L3_to_L2'; ...
        'W_plan_L2_to_L1'; 'W_plan_L3_to_L2'; ...
        'W_motor_L1_lat'; 'W_motor_L2_lat'; 'W_motor_L3_lat'; ...
        'W_plan_L1_lat'; 'W_plan_L2_lat'; 'W_plan_L3_lat'; ...
        'pi_L1_motor'; 'pi_L2_motor'; 'pi_L3_motor'; ...
        'pi_L1_plan'; 'pi_L2_plan'; 'pi_L3_plan'; ...
        'denom_L1_motor'; 'denom_L2_motor'; ...
        'denom_L1_plan'; 'denom_L2_plan'
    };

    fprintf('SNAPSHOT CONTENTS:\n');
    fprintf('─────────────────────────────────────────────────────────────\n');
    available_vars = fieldnames(S);
    fprintf('Available variables (%d total):\n', numel(available_vars));
    for i = 1:numel(available_vars)
        var_name = available_vars{i};
        var_val = S.(var_name);
        if isnumeric(var_val)
            if isscalar(var_val)
                fprintf('  %s: scalar (%.6e)\n', var_name, var_val);
            else
                fprintf('  %s: [%s]\n', var_name, mat2str(size(var_val)));
            end
        elseif iscell(var_val)
            fprintf('  %s: cell array (length %d)\n', var_name, numel(var_val));
        else
            fprintf('  %s: %s\n', var_name, class(var_val));
        end
    end
    fprintf('\n');

    % ====================================================================
    % TIMESTEP INFORMATION
    % ====================================================================

    if isfield(S, 'i')
        fprintf('FAILURE TIMESTEP:\n');
        fprintf('─────────────────────────────────────────────────────────────\n');
        fprintf('  Failed at step: %d\n\n', S.i);
    end

    % ====================================================================
    % ANALYZE EACH VARIABLE FOR NaNS
    % ====================================================================

    fprintf('NaN ANALYSIS BY VARIABLE:\n');
    fprintf('─────────────────────────────────────────────────────────────\n\n');

    nan_summary = struct();
    nan_count_total = 0;

    for v = 1:numel(available_vars)
        var_name = available_vars{v};
        var_val = S.(var_name);

        if ~isnumeric(var_val) && ~iscell(var_val)
            continue; % skip non-numeric variables
        end

        if iscell(var_val)
            % Cell array: check each cell
            nan_cells = 0;
            max_nans = 0;
            for c = 1:numel(var_val)
                if isnumeric(var_val{c})
                    nan_count = sum(isnan(var_val{c}(:)));
                    if nan_count > 0
                        nan_cells = nan_cells + 1;
                        max_nans = max(max_nans, nan_count);
                    end
                end
            end
            if nan_cells > 0
                fprintf('  ★ %s (cell array):\n', var_name);
                fprintf('      - NaN-containing cells: %d / %d\n', nan_cells, numel(var_val));
                fprintf('      - Max NaNs in any cell: %d\n\n', max_nans);
                nan_summary.(var_name) = struct('type', 'cell', 'nan_cells', nan_cells, 'max_nans', max_nans);
                nan_count_total = nan_count_total + max_nans;
            end
        else
            % Numeric array: count NaNs
            nan_count = sum(isnan(var_val(:)));
            inf_count = sum(isinf(var_val(:)));

            if nan_count > 0 || inf_count > 0
                fprintf('  ★ %s:\n', var_name);
                fprintf('      - Shape: %s\n', mat2str(size(var_val)));
                fprintf('      - NaN count: %d (%.2f%%)\n', nan_count, 100 * nan_count / numel(var_val));
                if inf_count > 0
                    fprintf('      - Inf count: %d (%.2f%%)\n', inf_count, 100 * inf_count / numel(var_val));
                end

                % Find first NaN location(s)
                [idx_nan] = find(isnan(var_val));
                if ~isempty(idx_nan)
                    fprintf('      - First NaN at index: %d\n', idx_nan(1));
                end

                % Statistics on non-NaN values
                valid_vals = var_val(~isnan(var_val) & ~isinf(var_val));
                if ~isempty(valid_vals)
                    fprintf('      - Valid values: min=%.6e, max=%.6e, mean=%.6e, std=%.6e\n', ...
                        min(valid_vals), max(valid_vals), mean(valid_vals), std(valid_vals));
                end
                fprintf('\n');

                nan_summary.(var_name) = struct('type', 'numeric', 'nan_count', nan_count, ...
                    'inf_count', inf_count, 'size', size(var_val));
                nan_count_total = nan_count_total + nan_count;
            end
        end
    end

    fprintf('\nTOTAL NaNs ACROSS ALL VARIABLES: %d\n\n', nan_count_total);

    % ====================================================================
    % IDENTIFY ROOT CAUSE PATTERNS
    % ====================================================================

    fprintf('ROOT CAUSE ANALYSIS:\n');
    fprintf('─────────────────────────────────────────────────────────────\n\n');

    issues = {};

    % Check 1: Precision-related issues
    precision_vars = {'pi_L1_motor', 'pi_L2_motor', 'pi_L1_plan', 'pi_L2_plan', ...
                      'denom_L1_motor', 'denom_L2_motor', 'denom_L1_plan', 'denom_L2_plan'};
    precision_nans = 0;
    for pv = 1:numel(precision_vars)
        if isfield(S, precision_vars{pv})
            val = S.(precision_vars{pv});
            if isnumeric(val)
                precision_nans = precision_nans + sum(isnan(val(:)));
            end
        end
    end
    if precision_nans > 0
        issues{end+1} = sprintf('PRECISION UPDATE FAILURE: %d NaNs in pi_* or denom_* variables', precision_nans);
        fprintf('  ⚠ Precision update failure detected (%d NaNs)\n', precision_nans);
        fprintf('     → Check: update_pi function, log(denom) calls, or precision smoothing\n');
        fprintf('     → Remedy: ensure denom > 0 before log, clamp precision > 0.1\n\n');
    end

    % Check 2: Weight-related issues
    weight_vars = {'W_motor_L2_to_L1', 'W_motor_L3_to_L2', 'W_plan_L2_to_L1', 'W_plan_L3_to_L2'};
    weight_nans = 0;
    for wv = 1:numel(weight_vars)
        if isfield(S, weight_vars{wv})
            val = S.(weight_vars{wv});
            if iscell(val)
                for c = 1:numel(val)
                    if isnumeric(val{c})
                        weight_nans = weight_nans + sum(isnan(val{c}(:)));
                    end
                end
            elseif isnumeric(val)
                weight_nans = weight_nans + sum(isnan(val(:)));
            end
        end
    end
    if weight_nans > 0
        issues{end+1} = sprintf('WEIGHT UPDATE FAILURE: %d NaNs in W_* matrices', weight_nans);
        fprintf('  ⚠ Weight update failure detected (%d NaNs)\n', weight_nans);
        fprintf('     → Check: weight update equations, dW clipping, or learning rate instability\n');
        fprintf('     → Remedy: add dW clipping (max norm), reduce eta_W, check momentum\n\n');
    end

    % Check 3: Representation update issues
    rep_vars = {'R_L1_motor', 'R_L2_motor', 'R_L3_motor', 'R_L1_plan', 'R_L2_plan', 'R_L3_plan'};
    rep_nans = 0;
    for rv = 1:numel(rep_vars)
        if isfield(S, rep_vars{rv})
            val = S.(rep_vars{rv});
            if isnumeric(val)
                rep_nans = rep_nans + sum(isnan(val(:)));
            end
        end
    end
    if rep_nans > 0
        issues{end+1} = sprintf('REPRESENTATION UPDATE FAILURE: %d NaNs in R_* variables', rep_nans);
        fprintf('  ⚠ Representation update failure detected (%d NaNs)\n', rep_nans);
        fprintf('     → Check: coupling term computation, R update equations\n');
        fprintf('     → Remedy: add R clipping, reduce eta_rep, check for division by zero\n\n');
    end

    % Check 4: Error computation issues
    error_vars = {'E_L1_motor', 'E_L2_motor', 'E_L1_plan', 'E_L2_plan'};
    error_nans = 0;
    for ev = 1:numel(error_vars)
        if isfield(S, error_vars{ev})
            val = S.(error_vars{ev});
            if isnumeric(val)
                error_nans = error_nans + sum(isnan(val(:)));
            end
        end
    end
    if error_nans > 0
        issues{end+1} = sprintf('ERROR COMPUTATION FAILURE: %d NaNs in E_* variables', error_nans);
        fprintf('  ⚠ Error computation failure detected (%d NaNs)\n', error_nans);
        fprintf('     → Check: prediction computation, observation-prediction mismatch\n');
        fprintf('     → Remedy: clamp predictions to valid range, check physics integrator\n\n');
    end

    if isempty(issues)
        fprintf('  ✓ No obvious failure patterns detected\n');
        fprintf('  → NaNs likely due to accumulation of small numerical errors\n');
        fprintf('  → Remedy: increase precision, reduce learning rates, add gradient clipping\n\n');
    end

    % ====================================================================
    % DETAILED DIAGNOSTICS FOR KEY VARIABLES
    % ====================================================================

    fprintf('DETAILED DIAGNOSTICS:\n');
    fprintf('─────────────────────────────────────────────────────────────\n\n');

    % Motor error
    if isfield(S, 'E_L1_motor')
        fprintf('Motor Error (E_L1_motor):\n');
        E = S.E_L1_motor;
        fprintf('  Shape: %s\n', mat2str(size(E)));
        fprintf('  Range (non-NaN): [%.6e, %.6e]\n', min(E(~isnan(E))), max(E(~isnan(E))));
        fprintf('  Norm: %.6e\n', norm(E(~isnan(E))));
        fprintf('  NaNs: %d, Infs: %d\n\n', sum(isnan(E(:))), sum(isinf(E(:))));
    end

    % Planning error
    if isfield(S, 'E_L1_plan')
        fprintf('Planning Error (E_L1_plan):\n');
        E = S.E_L1_plan;
        fprintf('  Shape: %s\n', mat2str(size(E)));
        fprintf('  Range (non-NaN): [%.6e, %.6e]\n', min(E(~isnan(E))), max(E(~isnan(E))));
        fprintf('  Norm: %.6e\n', norm(E(~isnan(E))));
        fprintf('  NaNs: %d, Infs: %d\n\n', sum(isnan(E(:))), sum(isinf(E(:))));
    end

    % Motor representation
    if isfield(S, 'R_L2_motor')
        fprintf('Motor Representation (R_L2_motor):\n');
        R = S.R_L2_motor;
        fprintf('  Shape: %s\n', mat2str(size(R)));
        if ~all(isnan(R(:)))
            fprintf('  Range (non-NaN): [%.6e, %.6e]\n', min(R(~isnan(R))), max(R(~isnan(R))));
        end
        fprintf('  NaNs: %d, Infs: %d\n\n', sum(isnan(R(:))), sum(isinf(R(:))));
    end

    % ====================================================================
    % VISUALIZATION
    % ====================================================================

    fprintf('GENERATING VISUALIZATIONS...\n\n');

    fig = figure('Name', 'NaN Snapshot Analysis', 'NumberTitle', 'off', 'Visible', 'off');

    % Plot 1: Error traces
    subplot(2, 2, 1);
    if isfield(S, 'E_L1_motor')
        E_motor = S.E_L1_motor;
        plot(E_motor, 'b-', 'LineWidth', 1.5);
        hold on;
        % Mark NaN locations
        nan_idx = find(isnan(E_motor));
        if ~isempty(nan_idx)
            plot(nan_idx, max(E_motor(~isnan(E_motor))) * ones(size(nan_idx)), 'r*', 'MarkerSize', 10);
        end
    end
    xlabel('Timestep');
    ylabel('Error');
    title('Motor Error (E\_L1\_motor) - * marks NaNs');
    grid on;
    set(gca, 'YScale', 'log');

    % Plot 2: Planning error
    subplot(2, 2, 2);
    if isfield(S, 'E_L1_plan')
        E_plan = S.E_L1_plan;
        plot(E_plan, 'r-', 'LineWidth', 1.5);
        hold on;
        nan_idx = find(isnan(E_plan));
        if ~isempty(nan_idx)
            plot(nan_idx, max(E_plan(~isnan(E_plan))) * ones(size(nan_idx)), 'r*', 'MarkerSize', 10);
        end
    end
    xlabel('Timestep');
    ylabel('Error');
    title('Planning Error (E\_L1\_plan) - * marks NaNs');
    grid on;
    set(gca, 'YScale', 'log');

    % Plot 3: Precision traces
    subplot(2, 2, 3);
    if isfield(S, 'pi_L1_motor')
        pi_motor = S.pi_L1_motor;
        plot(pi_motor, 'b-', 'LineWidth', 1.5);
        hold on;
        if isfield(S, 'pi_L2_motor')
            pi_L2 = S.pi_L2_motor;
            plot(pi_L2, 'b--', 'LineWidth', 1);
        end
    end
    if isfield(S, 'pi_L1_plan')
        pi_plan = S.pi_L1_plan;
        plot(pi_plan, 'r-', 'LineWidth', 1.5);
        if isfield(S, 'pi_L2_plan')
            pi_L2_plan = S.pi_L2_plan;
            plot(pi_L2_plan, 'r--', 'LineWidth', 1);
        end
    end
    xlabel('Timestep');
    ylabel('Precision');
    title('Precision Traces (pi\_*) - solid=L1, dashed=L2');
    legend('Motor L1', 'Motor L2', 'Plan L1', 'Plan L2');
    grid on;
    set(gca, 'YScale', 'log');

    % Plot 4: Variable NaN summary
    subplot(2, 2, 4);
    var_names_with_nans = fieldnames(nan_summary);
    nan_counts = [];
    for i = 1:numel(var_names_with_nans)
        s = nan_summary.(var_names_with_nans{i});
        if isfield(s, 'nan_count')
            nan_counts(i) = s.nan_count;
        elseif isfield(s, 'max_nans')
            nan_counts(i) = s.max_nans;
        end
    end
    if ~isempty(nan_counts)
        [sorted_nans, sort_idx] = sort(nan_counts, 'descend');
        sorted_names = var_names_with_nans(sort_idx);
        bar(sorted_nans);
        set(gca, 'XTickLabel', sorted_names, 'XTickLabelRotation', 45);
        ylabel('NaN Count');
        title('NaN Distribution by Variable (Top 10)');
        grid on;
    end

    sgtitle('NaN Snapshot Analysis Summary', 'FontSize', 12, 'FontWeight', 'bold');
    try
        saveas(fig, 'nan_snapshot_analysis.png');
        fprintf('✓ Visualization saved: nan_snapshot_analysis.png\n\n');
    catch
        fprintf('Warning: Could not save visualization\n\n');
    end
    close(fig);

    % ====================================================================
    % EXPORT TO CSV FOR FURTHER INSPECTION
    % ====================================================================

    fprintf('EXPORTING DIAGNOSTICS TO CSV...\n\n');

    try
        % Create summary table
        summary_table = table();
        row_idx = 1;

        for v = 1:numel(available_vars)
            var_name = available_vars{v};
            var_val = S.(var_name);

            if isnumeric(var_val)
                nan_count = sum(isnan(var_val(:)));
                inf_count = sum(isinf(var_val(:)));
                valid_count = numel(var_val) - nan_count - inf_count;

                summary_table(row_idx, :) = table({var_name}, numel(var_val), nan_count, inf_count, valid_count, ...
                    'VariableNames', {'Variable', 'Total_Elements', 'NaN_Count', 'Inf_Count', 'Valid_Count'});
                row_idx = row_idx + 1;
            end
        end

        writetable(summary_table, 'nan_snapshot_summary.csv');
        fprintf('✓ Summary exported: nan_snapshot_summary.csv\n\n');
    catch ME
        fprintf('Warning: Could not export to CSV: %s\n\n', ME.message);
    end

    % ====================================================================
    % RECOMMENDATIONS
    % ====================================================================

    fprintf('╔═════════════════════════════════════════════════════════════╗\n');
    fprintf('║  RECOMMENDATIONS FOR FIXING NaN ISSUES                    ║\n');
    fprintf('╚═════════════════════════════════════════════════════════════╝\n\n');

    fprintf('1. IMMEDIATE FIXES:\n');
    fprintf('   ☐ Add clamping to all representation updates:\n');
    fprintf('       R_L* = max(min_val, min(max_val, R_L* + dR))\n');
    fprintf('   ☐ Add gradient clipping to weight updates:\n');
    fprintf('       dW = max(-clip_val, min(clip_val, dW))\n');
    fprintf('   ☐ Ensure precision stays positive:\n');
    fprintf('       pi_* = max(0.1, pi_* + dpi)\n\n');

    fprintf('2. NUMERICAL STABILITY:\n');
    fprintf('   ☐ Reduce learning rates (eta_rep, eta_W) by 50%%-70%%\n');
    fprintf('   ☐ Increase momentum (0.9+) to smooth updates\n');
    fprintf('   ☐ Add weight decay to regularize large weights\n');
    fprintf('   ☐ Check for division by zero in coupling terms\n\n');

    fprintf('3. MONITORING:\n');
    fprintf('   ☐ Log max/min values before each major computation\n');
    fprintf('   ☐ Check for NaNs after each step (use isfinite)\n');
    fprintf('   ☐ Save intermediate states for debugging\n\n');

    fprintf('4. MODEL CHANGES:\n');
    fprintf('   ☐ Consider batch updates instead of per-step updates\n');
    fprintf('   ☐ Use double precision (default in MATLAB) instead of single\n');
    fprintf('   ☐ Implement adaptive learning rates per layer\n\n');

    fprintf('═════════════════════════════════════════════════════════════\n');
    fprintf('Analysis complete. Check nan_snapshot_analysis.png and nan_snapshot_summary.csv\n');
    fprintf('═════════════════════════════════════════════════════════════\n\n');

end
