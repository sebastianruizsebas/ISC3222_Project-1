%% UNIT TESTS FOR INTERCEPTION GAME
%
% Run this script to validate all components
% Follows best practices for open science: reproducible, tested code

function test_all()
    
    fprintf('\n╔═══════════════════════════════════════════════════════════╗\n');
    fprintf('║              RUNNING UNIT TESTS                          ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');
    
    test_game_configuration();
    test_trial_data();
    test_trajectory_generator();
    test_hierarchical_model();
    
    fprintf('\n✓ ALL TESTS PASSED!\n\n');
    
end

function test_game_configuration()
    
    fprintf('Testing GameConfiguration...\n');
    
    % Create configuration
    config = Game.GameConfiguration(...
        'participant_id', 'TEST001', ...
        'age', 25, ...
        'n_trials', 10);
    
    % Validate
    config.validate();
    
    % Check properties
    assert(config.n_trials == 10, 'n_trials not set correctly');
    assert(strcmp(config.participant_id, 'TEST001'), 'participant_id not set');
    assert(config.dt > 0, 'dt should be positive');
    assert(config.dt == 1/config.framerate, 'dt calculation incorrect');
    
    fprintf('  ✓ GameConfiguration tests passed\n\n');
    
end

function test_trial_data()
    
    fprintf('Testing TrialData...\n');
    
    % Create trial
    trial = Game.TrialData(1, "test");
    
    % Add motor commands
    trial.addMotorCommand(0.0, [640, 360]);
    trial.addMotorCommand(0.1, [650, 360]);
    trial.addMotorCommand(0.2, [660, 360]);
    
    % Check data storage
    assert(size(trial.reticle_pos, 1) == 3, 'Motor commands not stored correctly');
    assert(trial.reaction_time > 0, 'Reaction time should be computed');
    
    % Add interception
    trial.addInterception(2.0, 50, true);
    assert(trial.success == true, 'Success flag not set');
    
    fprintf('  ✓ TrialData tests passed\n\n');
    
end

function test_trajectory_generator()
    
    fprintf('Testing TrajectoryGenerator...\n');
    
    % Generate constant velocity trajectory
    [traj, times, v, a] = Utils.TrajectoryGenerator.generateTargetTrajectory(...
        1280, 720, [100, 300], 'constant', 5);
    
    assert(size(traj, 1) > 0, 'Trajectory not generated');
    assert(size(traj, 2) == 2, 'Trajectory should have x,y columns');
    assert(length(times) == size(traj, 1), 'Times and trajectory size mismatch');
    assert(a == 0, 'Acceleration should be 0 for constant velocity');
    
    % Generate accelerating trajectory
    [traj_accel, ~, ~, a_accel] = Utils.TrajectoryGenerator.generateTargetTrajectory(...
        1280, 720, [100, 300], 'accelerating', 5);
    
    assert(a_accel > 0, 'Acceleration should be positive');
    
    fprintf('  ✓ TrajectoryGenerator tests passed\n\n');
    
end

function test_hierarchical_model()
    
    fprintf('Testing HierarchicalMotionModel...\n');
    
    % Create dummy trial data
    trial = Game.TrialData(1, "test");
    
    % Add synthetic motion data (constant velocity)
    t = linspace(0, 2, 100)';
    x = 100 + 50 * t;  % Constant velocity = 50 px/sec
    
    for i = 1:length(t)
        trial.addMotorCommand(t(i), [x(i), 360]);
    end
    
    % Create and fit model
    model = Models.HierarchicalMotionModel(trial);
    model.fitPrecision();
    
    % Check that model was fit
    assert(~isinf(model.likelihood), 'Model likelihood should be finite');
    assert(model.pi_x > 0, 'π_x should be positive');
    assert(model.pi_v > 0, 'π_v should be positive');
    
    fprintf('  ✓ HierarchicalMotionModel tests passed\n\n');
    
end
