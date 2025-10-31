% smoke_test_run.m
% Quick smoke test for hierarchical_motion_inference_dual_hierarchy
% Creates a very short run to validate end-to-end behavior and saves a small MAT

% Configure a short test (small T_per_trial and few trials)
params = struct();
params.dt = 0.01;               % time step
params.T_per_trial = 0.5;       % seconds per trial (very short for smoke test)
params.n_trials = 2;            % small number of trials
params.eta_rep = 0.005;
params.eta_W = 0.0005;
params.momentum = 0.98;
params.weight_decay = 0.98;
% Do not write heavy per-particle files during smoke tests
params.save_results = true;     % we will save a tiny results file for inspection

fprintf('\nRunning smoke test: short dual-hierarchy run (T_per_trial=%.3fs, n_trials=%d)\n', params.T_per_trial, params.n_trials);

try
    results = hierarchical_motion_inference_dual_hierarchy(params, false);

    outdir = fullfile('.', 'figures');
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    outfn = fullfile(outdir, 'smoke_test_results.mat');
    save(outfn, 'results', '-v7.3');

    rmse = sqrt(mean(results.interception_error_all.^2));
    fprintf('Smoke test completed successfully. Results saved to: %s\n', outfn);
    fprintf('Interception RMSE: %.6f m\n', rmse);
catch ME
    fprintf('Smoke test FAILED: %s\n', ME.message);
    if isfield(ME, 'stack')
        for k = 1:numel(ME.stack)
            fprintf('  at %s (line %d)\n', ME.stack(k).file, ME.stack(k).line);
        end
    end
    rethrow(ME);
end
