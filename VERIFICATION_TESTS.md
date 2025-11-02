# VERIFICATION TESTS FOR CODE REVIEW FINDINGS

## Test 1: Verify Bug #1 (Hardcoded alpha_precision_gain)

**Objective:** Confirm that PSO parameter changes don't affect precision scaling

**Test Code:**
```matlab
% Before fix: should show NO variation in scaling
results1 = hierarchical_motion_inference_dual_hierarchy(struct(...
    'alpha_precision_gain', 0.1, ...
    'suppress_init_log', true), false);

results2 = hierarchical_motion_inference_dual_hierarchy(struct(...
    'alpha_precision_gain', 2.0, ...  % 20x different!
    'suppress_init_log', true), false);

% Compare precision scale factors
scale1 = results1.pi_scale_history.L1_motor;
scale2 = results2.pi_scale_history.L1_motor;

% Before fix: scale1 ≈ scale2 (both always exp(0.5 * error))
% After fix: scale1 << scale2 (0.1 much less sensitive than 2.0)

fprintf('Before fix: scale1 and scale2 are nearly identical\n');
fprintf('After fix: scale2 should be dramatically larger\n');
fprintf('Scale ratio (should be >>1 after fix): %.2f\n', max(scale2) / max(scale1));
```

**Expected result (BEFORE fix):** Ratio ≈ 1.0 (no difference)  
**Expected result (AFTER fix):** Ratio >> 1.0 (significant difference)

---

## Test 2: Verify Bug #2 (Hardcoded precision bounds)

**Objective:** Confirm that PSO precision bound parameters don't take effect

**Test Code:**
```matlab
% Run with different precision bounds
results_low = hierarchical_motion_inference_dual_hierarchy(struct(...
    'pi_L1_motor_min', 5, ...      % Low floor
    'pi_L1_motor_max', 200, ...    % Low ceiling
    'suppress_init_log', true), false);

results_high = hierarchical_motion_inference_dual_hierarchy(struct(...
    'pi_L1_motor_min', 50, ...     % High floor
    'pi_L1_motor_max', 1000, ...   % High ceiling
    'suppress_init_log', true), false);

% Check actual precision values reached
min_low = min(results_low.pi_trace_L1_motor);
max_low = max(results_low.pi_trace_L1_motor);
min_high = min(results_high.pi_trace_L1_motor);
max_high = max(results_high.pi_trace_L1_motor);

fprintf('Low bounds test:   min=%.1f, max=%.1f\n', min_low, max_low);
fprintf('High bounds test:  min=%.1f, max=%.1f\n', min_high, max_high);

% Before fix: both should show [10, 500] hardcoded range
% After fix: low should show [5, 200], high should show [50, 1000]
```

**Expected result (BEFORE fix):** Both tests show [10, 500]  
**Expected result (AFTER fix):** Different ranges for each test

---

## Test 3: Verify Bug #3 (Triple precision update)

**Objective:** Confirm that precision is updated multiple times per step

**Test Code:**
```matlab
% Add debugging to hierarchical_step_update.m before each update:

% Before line ~515 (update_pi call):
fprintf('  Update 1 (update_pi): pi_before=%.2f\n', S.pi_L1_motor);

% Before line ~525 (hardcoded exp):
fprintf('  Update 2 (hardcoded exp): pi_before=%.2f\n', S.pi_L1_motor);

% Before line ~650 (error-driven section):
fprintf('  Update 3 (error-driven): pi_before=%.2f\n', S.pi_L1_motor);

% Before line ~730 (bounds clipping):
fprintf('  Final clip: pi_before=%.2f\n', S.pi_L1_motor);

% Run one timestep and look for all four messages
results = hierarchical_motion_inference_dual_hierarchy(struct(...
    'suppress_init_log', true, 'n_trials', 1, 'T_per_trial', 0.1), false);
```

**Expected result (BEFORE fix):** All four update messages appear (multiple modifications)  
**Expected result (AFTER fix):** Only 1-2 messages (consolidated mechanism)

---

## Test 4: Verify Bug #4 (Missing interference_penalty_weight)

**Objective:** Confirm parameter isn't being passed through to helper

**Test Code:**
```matlab
% Add debug line in hierarchical_step_update.m around line ~280:
if isfield(P, 'interference_penalty_weight')
    fprintf('  interference_penalty_weight in P: %.4f\n', P.interference_penalty_weight);
else
    fprintf('  ⚠ interference_penalty_weight NOT in P (will use default 0.01)\n');
end

% Run with explicit parameter
results = hierarchical_motion_inference_dual_hierarchy(struct(...
    'interference_penalty_weight', 0.05, ...  % Non-default value
    'suppress_init_log', true), false);

% Before fix: should print "NOT in P" message
% After fix: should print "0.0500" message
```

**Expected result (BEFORE fix):** "NOT in P (will use default 0.01)"  
**Expected result (AFTER fix):** "interference_penalty_weight in P: 0.0500"

---

## Test 5: Integration Test - PSO Should Work

**Objective:** Verify PSO optimization responds to parameter changes after fixes

**Test Code:**
```matlab
% Run PSO with 5 particles, 3 iterations (quick test)
% Modify optimize_rao_ballard_pso.m to:
%   num_particles = 5;
%   num_iterations = 3;

% Before fix: all particles should converge to similar parameters
%            (PSO not exploring parameter space effectively)

% After fix: particles should explore diverse parameter values
%           (PSO exploring effectively)

results = optimize_rao_ballard_pso();  % Load from saved results

% Check parameter diversity across top 5 particles
top5_alpha = [results.particles(1).best_alpha_precision_gain, ...
              results.particles(2).best_alpha_precision_gain, ...
              results.particles(3).best_alpha_precision_gain, ...
              results.particles(4).best_alpha_precision_gain, ...
              results.particles(5).best_alpha_precision_gain];

fprintf('Top 5 alpha_precision_gain values:\n');
disp(top5_alpha);
fprintf('Std dev: %.4f\n', std(top5_alpha));

% Before fix: std ~0 (all same due to hardcoding)
% After fix: std should be significant (PSO exploring)
```

**Expected result (BEFORE fix):** std ≈ 0, all values ≈ 0.5  
**Expected result (AFTER fix):** std >> 0, diverse values in [0.1, 2.0]

---

## Test 6: Precision Scaling Shape Test

**Objective:** Verify exponential precision scaling has expected shape

**Test Code:**
```matlab
% Manually test precision scaling formula
% exp(alpha * error_magnitude) should:
% - Equal 1.0 when error=0 (no change)
% - Grow exponentially with error
% - Not explode (capped at error=5)

alpha = 0.5;
errors = [0, 1, 2, 3, 4, 5, 5, 5];  % Last three capped

scales = exp(alpha * min(errors, 5.0));
fprintf('Error → Scale factor:\n');
for k = 1:length(errors)
    fprintf('  %.1f  →  %.4f\n', errors(k), scales(k));
end

% Expected output (roughly):
% 0.0 → 1.0000 (no change)
% 1.0 → 1.6487 (64.9% increase)
% 2.0 → 2.7183 (171.8% increase)
% 3.0 → 4.4817 (348.2% increase)
% 4.0 → 7.3891 (638.9% increase)
% 5.0 → 12.1825 (1118.3% increase)  -- CAPPED
% 5.0 → 12.1825 (same)
% 5.0 → 12.1825 (same)
```

**Expected result:** Exponential growth from 1.0 with error cap at 5

---

## Test 7: Code Review Metrics

**Objective:** Measure parameter dependency before/after fixes

**Test Code:**
```matlab
% Create sensitivity analysis: vary each parameter individually
% measure how much output changes

params_base = struct(...
    'alpha_precision_gain', 0.5, ...
    'pi_L1_motor_min', 10, ...
    'pi_L1_motor_max', 500, ...
    'interference_penalty_weight', 0.01);

% Test with base values
results_base = hierarchical_motion_inference_dual_hierarchy(params_base, false);
score_base = mean(results_base.interception_error_all);

% Test with varied parameters
params_varied = params_base;
params_varied.alpha_precision_gain = 2.0;  % 4x higher

results_varied = hierarchical_motion_inference_dual_hierarchy(params_varied, false);
score_varied = mean(results_varied.interception_error_all);

sensitivity = (score_varied - score_base) / score_base * 100;
fprintf('Parameter sensitivity to alpha_precision_gain:\n');
fprintf('  Base score: %.6f\n', score_base);
fprintf('  Varied (α×4): %.6f\n', score_varied);
fprintf('  Change: %.2f%%\n', sensitivity);

% Before fix: sensitivity ≈ 0% (parameter ignored)
% After fix: sensitivity >> 0% (parameter has effect)
```

**Expected result (BEFORE fix):** Change < 1%  
**Expected result (AFTER fix):** Change > 5-10%

---

## Recommended Test Sequence

1. **Quick verification (5 min):** Run Test #3 (multiple updates)
2. **Parameter isolation (10 min):** Run Tests #1, #2, #4
3. **Integration validation (15 min):** Run Test #5 (PSO exploration)
4. **Sensitivity analysis (20 min):** Run Test #7 (measure parameter effect)
5. **Mathematical validation (5 min):** Run Test #6 (exponential shape)

**Total time:** ~55 minutes for comprehensive verification

---

## Success Criteria

✓ All tests pass before AND after applying fixes  
✓ BEFORE fix: Most tests show parameter blocking  
✓ AFTER fix: All tests show parameters working effectively  
✓ Precision traces show smooth exponential scaling  
✓ PSO optimization explores parameter space effectively  
✓ Parameter sensitivity significantly increased  

---

**Test Plan Created:** November 2, 2025  
**Implementation:** Implement in MATLAB test script after fixes applied
