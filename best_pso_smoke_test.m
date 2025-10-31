% matlab
% 1) locate leaderboard (assumes you're in project root)
lb1 = fullfile('./tools/figures','pso_top20_best_params.mat');
lb2 = fullfile('./tools/figures','pso_best_params.mat');
if exist(lb1,'file'), leaderboard_file = lb1; elseif exist(lb2,'file'), leaderboard_file = lb2; else error('Leaderboard not found'); end

% 2) load and pick top entry (same logic as helper)
loaded = load(leaderboard_file);
if isfield(loaded,'leader_list'), leader_list = loaded.leader_list;
else
    vars = fieldnames(loaded); leader_list = [];
    for k=1:numel(vars)
        v = loaded.(vars{k});
        if isstruct(v) && numel(v)>0 && isfield(v,'params'), leader_list = v; break; end
    end
    if isempty(leader_list), error('Could not find leader_list or params struct in leaderboard'); end
end
entry = leader_list(1);
if isfield(entry,'params'), params = entry.params; elseif isfield(entry,'particle'), params = entry.particle; else error('No params found'); end

% 3) override for a fast smoke test
params.dt = 0.02;
params.T_per_trial = 0.5;
params.n_trials = 2;
params.save_results = false;

% 4) run main (no plots)
res = hierarchical_motion_inference_dual_hierarchy(params, false);

% 5) quick check
assert(all(isfinite(res.interception_error_all)), 'NaN/Inf in interception errors');
disp('Smoke test OK: interception_error_all finite');