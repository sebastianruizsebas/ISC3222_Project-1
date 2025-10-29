# SETUP & INSTALLATION GUIDE

## System Requirements

- **MATLAB** R2016b or later
- **Recommended**: MATLAB R2020b+ (better graphics performance)
- **Optional**: Statistics and Machine Learning Toolbox

## Installation

### 1. Clone/Download Repository

```bash
cd ~/Documents/MATLAB
git clone <repository-url> InterceptionGame
cd InterceptionGame
```

### 2. Add to MATLAB Path

In MATLAB:

```matlab
% Add to path
addpath(genpath(pwd))

% Save path for future sessions
savepath
```

Or manually:
- File → Set Path → Add Folder (select InterceptionGame directory)

### 3. Verify Installation

```matlab
% Run tests
test_all()

% Check if classes are accessible
>> Game.GameConfiguration
>> Game.GameEngine
>> Models.HierarchicalMotionModel
```

If you see class definitions printed, installation is successful!

## Quick Start

### Run Full Experiment

```matlab
run_experiment()
```

This will interactively:
1. Ask for participant information
2. Display game instructions
3. Run 15 trials (~5 minutes)
4. Analyze results
5. Generate report

### Expected Output

```
═══════════════════════════════════════════════════════════╗
║      INTERCEPTION GAME - FULL EXPERIMENT                 ║
╚═══════════════════════════════════════════════════════════╝

PARTICIPANT INFORMATION INTAKE
──────────────────────────────

Participant ID: P001
Age: 25
Gaming experience (1-5): 3
Experimenter name: Dr. Smith
Notes (optional): control group

GAME CONFIGURATION SUMMARY
──────────────────────────
...trials running...
...analysis in progress...

PERFORMANCE METRICS
───────────────────
Mean Accuracy: 45.3 ± 12.5 pixels
Mean Lead Distance: 15.2 ± 8.1 pixels
...

✓ EXPERIMENT COMPLETE
Results saved to: interception_game_results/P001_*
```

## Directory Structure After First Run

```
InterceptionGame/
├── +Game/
├── +Analysis/
├── +Models/
├── +Utils/
├── run_experiment.m
├── EXAMPLES.m
├── README.md
└── interception_game_results/         ← Created on first run
    ├── P001_config.json
    ├── P001_trials.mat
    └── P001_report.txt
```

## Troubleshooting

### Issue: "Undefined function or variable 'Game'"

**Solution:**
- Verify installation: `which GameConfiguration`
- If not found, check path: `path`
- Re-add directory: `addpath(genpath(pwd))`

### Issue: "Graphics not displaying"

**Solution:**
- Check MATLAB graphics mode: `opengl` (should output OpenGL info)
- Try software rendering: `opengl('software')`
- Ensure figure is visible: `set(gcf, 'Visible', 'on')`

### Issue: Slow gameplay / dropped frames

**Solution:**
- Reduce screen resolution in config
- Increase frame interval: `config.framerate = 30` (instead of 60)
- Close other applications
- Use hardware accelerated graphics

### Issue: "Cannot save JSON - jsonencode not available"

**Solution:**
- Upgrade to MATLAB R2016b+ (includes jsonencode)
- Or data will save as .mat file automatically (compatible)

### Issue: Model fitting takes too long

**Solution:**
- Reduce search space in `HierarchicalMotionModel.fitPrecision()`
- Fewer parameter grid points:
```matlab
model.fitPrecision(...
    'pi_x_range', [100, 200], ...
    'pi_v_range', [5, 10], ...
    'pi_a_range', [0.5, 1]);
```

## Common Customizations

### 1. Change Number of Trials

```matlab
exp = Analysis.ExperimentManager('n_trials', 30);
exp.runFull();
```

### 2. Adjust Difficulty

```matlab
exp = Analysis.ExperimentManager();
exp.config.target_speed_range = [200, 400];  % Faster
exp.config.trial_duration = 3;               % Shorter
exp.runFull();
```

### 3. Use Different Motion Types

Edit `+Utils/TrajectoryGenerator.m`:
- 'constant': Constant velocity
- 'accelerating': Speeding up
- 'decelerating': Slowing down
- Custom: Add your own motion pattern

### 4. Custom Screen Resolution

```matlab
config = Game.GameConfiguration();
config.screen_width = 1920;
config.screen_height = 1080;
```

## Running Multiple Participants (Batch Mode)

Create `batch_experiment.m`:

```matlab
function batch_experiment(n_participants)
    
    for i = 1:n_participants
        participant_id = sprintf('P%03d', i);
        fprintf('\nParticipant %d/%d: %s\n', i, n_participants, participant_id);
        
        exp = Analysis.ExperimentManager('participant_id', participant_id);
        exp.runFull();
        
        % Pause between participants
        input('Press ENTER for next participant...');
    end
    
    fprintf('\n✓ All participants completed!\n');
    fprintf('Data saved to: interception_game_results/\n');
    
end
```

Run:
```matlab
batch_experiment(10)  % Run 10 participants
```

## Analyzing Results Across Participants

Create `group_analysis.m`:

```matlab
function group_analysis()
    
    % Find all participant files
    files = dir('interception_game_results/*_trials.mat');
    
    all_accuracy = [];
    all_lead_dist = [];
    all_pi_x = [];
    all_pi_v = [];
    
    for i = 1:length(files)
        data = load(fullfile(files(i).folder, files(i).name));
        
        acc = [data.trials_data.intercept_accuracy];
        lead = [data.trials_data.mean_lead_distance];
        
        all_accuracy = [all_accuracy; mean(acc)];
        all_lead_dist = [all_lead_dist; mean(lead)];
        
        % Model fits
        for j = 1:length(data.model_fits)
            model = data.model_fits(j);
            all_pi_x = [all_pi_x; model.pi_x];
            all_pi_v = [all_pi_v; model.pi_v];
        end
    end
    
    % Create group figures
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2,3,1);
    hist(all_accuracy, 20);
    xlabel('Mean Accuracy (px)'); ylabel('Count');
    title(sprintf('Accuracy Distribution (n=%d)', length(files)));
    
    subplot(2,3,2);
    hist(all_lead_dist, 20);
    xlabel('Lead Distance (px)'); ylabel('Count');
    title('Lead Distance Distribution');
    
    subplot(2,3,3);
    scatter(all_lead_dist, all_accuracy);
    xlabel('Lead Distance'); ylabel('Accuracy');
    title('Strategy vs Performance');
    
    subplot(2,3,4);
    hist(all_pi_x, 20);
    xlabel('π_x (sensory precision)');
    title('Sensory Precision Distribution');
    
    subplot(2,3,5);
    hist(all_pi_v, 20);
    xlabel('π_v (velocity precision)');
    title('Velocity Precision Distribution');
    
    subplot(2,3,6);
    ratio = all_pi_x ./ all_pi_v;
    hist(ratio, 20);
    xlabel('π_x / π_v ratio');
    title('Motor Strategy Distribution');
    
    fprintf('\n✓ Group analysis complete\n');
    
end
```

## Version Control & Reproducibility

Track your experiments:

```matlab
exp = Analysis.ExperimentManager();
exp.config.version = "1.0.0";
exp.config.experimenter = "Dr. Smith";
exp.config.notes = "Baseline study - healthy controls";
exp.runFull();

% Configuration automatically saved with version info
```

## Data Export

Export results to other formats:

```matlab
% Load data
data = load('interception_game_results/P001_trials.mat');

% Save as table (for R/Python)
T = struct2table(data.summary_statistics);
writetable(T, 'P001_summary.csv');

% Or export trials individually
for i = 1:length(data.trials_data)
    trial = data.trials_data(i);
    trial_table = table(...
        trial.trial_num, ...
        trial.motion_type, ...
        trial.intercept_accuracy, ...
        trial.mean_lead_distance, ...
        'VariableNames', {'trial_id', 'motion_type', 'accuracy', 'lead'});
    writetable(trial_table, sprintf('P001_trial_%03d.csv', i));
end
```

## Support & Debugging

### Enable Verbose Output

```matlab
% Add to any script
global INTERCEPTION_DEBUG
INTERCEPTION_DEBUG = true;
```

### Save Intermediate Results

```matlab
exp = Analysis.ExperimentManager();
exp.runGame();
save('backup_game_data.mat', 'exp');

exp.analyzeResults();
save('backup_with_analysis.mat', 'exp');
```

### Verify Graphics

```matlab
% Test graphics
figure;
hold on;
theta = linspace(0, 2*pi, 100);
plot(50+30*cos(theta), 360+30*sin(theta), 'r-', 'LineWidth', 2);
plot(50+40*cos(theta), 360+40*sin(theta), 'g-', 'LineWidth', 2);
axis([0 1280 0 720]);
axis equal;
title('Graphics Test');
```

## Performance Tips

1. **Fast mode** (fewer trials):
```matlab
config.n_trials = 5;
```

2. **Lighter graphics**:
```matlab
config.screen_width = 800;
config.screen_height = 600;
```

3. **Parallel processing** (for batch):
```matlab
parfor i = 1:n_participants
    % Requires Parallel Computing Toolbox
end
```

## Next Steps

1. ✓ Install & verify setup
2. ✓ Run `run_experiment()` 
3. ✓ Check output in `interception_game_results/`
4. ✓ Review report and neural interpretation
5. ✓ Run `batch_experiment()` for multiple participants
6. ✓ Use `group_analysis()` to compare results

## Support

- **Documentation**: See README.md
- **Examples**: Run `EXAMPLES.m`
- **Tests**: Run `test_all()`
- **Issues**: Check troubleshooting section above

---

Happy experimenting! 🎮🧠

