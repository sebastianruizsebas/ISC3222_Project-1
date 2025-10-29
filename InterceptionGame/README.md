# Interception Game - Object-Oriented Architecture

## Overview

This is a refactored, modular implementation of the interception game experiment following **best practices for open science and reproducible research**.

## Key Features

✓ **Object-Oriented Design** - Encapsulated, reusable classes
✓ **Separation of Concerns** - Each class has single responsibility  
✓ **Reproducible** - All parameters configurable, saved in JSON
✓ **Tested** - Unit tests included
✓ **Open Science** - Following FAIR principles (Findable, Accessible, Interoperable, Reusable)
✓ **Well-Documented** - Inline documentation and examples

## Architecture

```
InterceptionGame/
├── +Game/                    # Core game logic
│   ├── GameConfiguration.m   # Config management
│   ├── GameEngine.m          # Main game loop & graphics
│   └── TrialData.m           # Trial data container
│
├── +Analysis/                # Data analysis
│   └── ExperimentManager.m   # Orchestrates full workflow
│
├── +Models/                  # Statistical models
│   └── HierarchicalMotionModel.m  # Fits precision weights
│
├── +Utils/                   # Utility functions
│   └── TrajectoryGenerator.m # Generate motion stimuli
│
├── run_experiment.m          # Main entry point
├── tests/
│   └── test_all.m            # Unit tests
└── data/                     # Output directory
```

## Quick Start

### 1. Run Full Experiment

```matlab
run_experiment()
```

This will:
- Collect participant information interactively
- Run the game (15 trials, ~5 min)
- Fit hierarchical motion inference model
- Generate report with neural interpretation

### 2. Custom Parameters

```matlab
exp = Analysis.ExperimentManager(...
    'participant_id', 'P001', ...
    'n_trials', 20);
exp.runFull();
```

### 3. Run Tests

```matlab
test_all()
```

## Class Reference

### `Game.GameConfiguration`

Manages all experiment parameters.

**Properties:**
- `participant_id` - Unique identifier
- `n_trials` - Number of trials (default: 15)
- `screen_width`, `screen_height` - Display resolution
- `target_size`, `reticle_size` - Visual element sizes
- `trial_duration` - Length of each trial (default: 5 seconds)

**Methods:**
```matlab
config = Game.GameConfiguration('participant_id', 'P001');
config.validate();              % Check consistency
config.summary();               % Print summary
config.save('config.json');     % Save to JSON
```

### `Game.TrialData`

Container for single trial results.

**Properties:**
- `reticle_pos` - Player's reticle trajectory
- `target_trajectory` - Target motion trajectory
- `intercept_accuracy` - Distance from target at intercept
- `reaction_time` - Latency of first movement

**Methods:**
```matlab
trial = Game.TrialData(1, "constant_velocity");
trial.addMotorCommand(t, pos);  % Add motor command
trial.addInterception(t, acc, success);
trial.computeMetrics();         % Derive statistics
trial.summary();                % Print trial results
```

### `Game.GameEngine`

Main game loop and graphics management.

**Key Methods:**
```matlab
engine = Game.GameEngine(config);
engine.run();          % Execute all trials
```

Handles:
- Graphics rendering (OpenGL-based)
- Keyboard/mouse input
- Trial execution
- Real-time data collection

### `Models.HierarchicalMotionModel`

Fits hierarchical predictive coding model to motor behavior.

**Properties:**
- `pi_x` - Sensory precision (observes position)
- `pi_v` - Velocity precision (motion continuity prior)
- `pi_a` - Acceleration precision (smoothness prior)

**Methods:**
```matlab
model = Models.HierarchicalMotionModel(trial_data);
model.fitPrecision();           % Grid search over parameters
model.summary();                % Print results & interpretation
```

**Interpretation:**
```
π_x / π_v > 15  → REACTIVE (sensory-driven, autism-like)
π_x / π_v < 5   → PREDICTIVE (model-driven, psychosis-like)
π_x / π_v ≈ 10  → BALANCED (neurotypical)
```

### `Analysis.ExperimentManager`

Orchestrates entire experiment workflow.

**Methods:**
```matlab
exp = Analysis.ExperimentManager('participant_id', 'P001');
exp.runFull();           % Full workflow
exp.participantIntake(); % Collect info
exp.runGame();           % Execute game
exp.analyzeResults();    % Fit models & compute stats
exp.generateReport();    % Create summary
exp.saveResults();       % Save all data
```

### `Utils.TrajectoryGenerator`

Generate target motion trajectories.

**Methods:**
```matlab
% Generate standard trajectory
[traj, times, v_true, a_true] = Utils.TrajectoryGenerator.generateTargetTrajectory(...
    screen_width, screen_height, ...
    speed_range, ...           % [min_speed, max_speed]
    accel_type, ...            % 'constant', 'accelerating', 'decelerating'
    duration);

% Generate complex trajectory with direction change
[traj, times] = Utils.TrajectoryGenerator.generateComplexTrajectory(...
    screen_width, screen_height, duration);
```

## Data Format

### Configuration (JSON)
```json
{
  "participant_id": "P001",
  "age": 25,
  "gaming_experience": 3,
  "n_trials": 15,
  "version": "1.0.0",
  "creation_date": "2025-10-29T14:30:00Z"
}
```

### Trial Data (MATLAB .mat)
```
trials_data        struct array of Game.TrialData
model_fits         struct array of Models.HierarchicalMotionModel
summary_statistics table with aggregated metrics
```

## Open Science Best Practices

### 1. **Reproducibility**
- All parameters saved in JSON
- Fixed random seeds possible
- Complete experiment logged

### 2. **Documentation**
- Inline documentation for every class
- Usage examples in docstrings
- README with quick start

### 3. **Testing**
- Unit tests for each component
- Test file included
- Easy to add integration tests

### 4. **Version Control**
- Clear class interfaces
- Backward compatibility maintained
- Version tracking in config

### 5. **Open Formats**
- JSON for human-readable config
- MATLAB .mat for data (compatible with Python scipy)
- Plain text reports

## Output Files

For participant P001:
```
interception_game_results/
├── P001_config.json           # Experiment configuration
├── P001_trials.mat            # All trial data & model fits
├── P001_report.txt            # Human-readable summary
└── P001_neural_interpretation.txt  # Detailed brain interpretation
```

## Neural Interpretation

The fitted precision weights reveal motor control strategy:

| Ratio | Strategy | Neural Basis | Clinical Profile |
|-------|----------|--------------|-----------------|
| π_x/π_v > 15 | REACTIVE | Enhanced V1, weak MT | Autism |
| π_x/π_v < 5 | PREDICTIVE | Weak V1, overactive cerebelum | Psychosis |
| π_x/π_v ≈ 10 | BALANCED | Integrated cortical hierarchy | Neurotypical |

See `neural_interpretation_motor_control.m` for detailed explanation.

## Extending the Code

### Add new motion type:
```matlab
% In TrajectoryGenerator.m
case 'new_pattern'
    % Implement new motion
```

### Add new analysis:
```matlab
% Create new class in +Analysis/
classdef MyAnalysis < handle
    methods
        function analyze(obj, trial_data)
            % Custom analysis
        end
    end
end

% Use in ExperimentManager
analyzer = Analysis.MyAnalysis();
results = analyzer.analyze(obj.trials_data);
```

### Customize game difficulty:
```matlab
config = Game.GameConfiguration();
config.n_trials = 30;
config.target_speed_range = [150, 350];
config.trial_duration = 3;  % Shorter trials = harder
config.validate();
```

## Requirements

- MATLAB R2016b or later
- Image Processing Toolbox (optional, for advanced graphics)
- Statistics and Machine Learning Toolbox

## Citation

If you use this code for research, please cite:

```bibtex
@software{interception_game_2025,
  title={Interception Game: Object-Oriented Hierarchical Motion Inference},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/interception-game}
}
```

## License

MIT License - Free for academic and commercial use

## Authors

Your Name (FSU Symbolic & Numerical Computation Course)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new code
4. Submit a pull request

## Support

For issues or questions:
- Check the documentation
- Run `test_all()` to verify setup
- Open an issue on GitHub

---

**Last updated:** October 2025  
**Version:** 1.0.0  
**Status:** Production Ready ✓
