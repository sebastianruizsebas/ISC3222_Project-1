%% INTERCEPTION GAME - MAIN SCRIPT
% 
% Object-Oriented implementation with best practices:
% - Modular class-based architecture
% - Clear separation of concerns
% - Reproducible experiments
% - Open science format
%
% QUICK START:
%   run_experiment();
%
% For custom parameters:
%   exp = Analysis.ExperimentManager('participant_id', 'P001', 'n_trials', 20);
%   exp.runFull();

function run_experiment()
    
    fprintf('\n\n');
    fprintf('╔═══════════════════════════════════════════════════════════╗\n');
    fprintf('║         INTERCEPTION GAME - MOTOR CONTROL STUDY          ║\n');
    fprintf('║     Testing Hierarchical Motion Inference in Humans      ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');
    
    % Create experiment manager with default parameters
    exp = Analysis.ExperimentManager();
    
    % Run full experiment workflow
    exp.runFull();
    
end
