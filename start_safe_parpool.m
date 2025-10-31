function suggested = start_safe_parpool(numLogical, availMB, per_worker_MB)
% Start a safe parpool sized from the numbers you provide.
% Usage: suggested = start_safe_parpool(8, 16384, 1000);

if nargin < 1 || isempty(numLogical), numLogical = java.lang.Runtime.getRuntime.availableProcessors; end
if nargin < 2 || isempty(availMB),   m = memory; availMB = m.MemAvailableAllArrays/1024^2; end
if nargin < 3 || isempty(per_worker_MB), per_worker_MB = 1000; end

max_by_cpu = max(1, numLogical - 1);       % leave 1 core for OS
max_by_mem = floor(availMB / per_worker_MB);
c = parcluster('local');
max_cluster = c.NumWorkers;

suggested = max(1, min([max_by_cpu, max_by_mem, max_cluster]));
fprintf('Suggested parpool workers: %d (logical=%d, availMB=%.0f, perWorkerMB=%d, clusterMax=%d)\n', ...
    suggested, numLogical, availMB, per_worker_MB, max_cluster);

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', suggested);
else
    if pool.NumWorkers ~= suggested
        delete(pool);
        parpool('local', suggested);
    else
        fprintf('Reusing existing parpool with %d workers.\n', pool.NumWorkers);
    end
end
end