# Parameter Optimization Strategy for 3D Hierarchical Motion Inference

## Overview
`optimize_rao_ballard_parameters.m` performs a random search over hyperparameter space to find optimal learning rates and momentum for `hierarchical_motion_inference_3D_EXACT.m`.

## Sound ML Principles Applied

### ✅ Parameters OPTIMIZED (Tunable Hyperparameters)
These are genuinely uncertain and should be optimized via search:

1. **eta_rep** (Representation Learning Rate)
   - Controls how fast neural representations (R_L1, R_L2, R_L3) update
   - Search range: **[1e-4, 1e-1]** (log scale)
   - Current value: 0.005 (middle of range)
   - Rationale: Too high → oscillation; too low → slow learning

2. **eta_W** (Weight Matrix Learning Rate)  
   - Controls how fast weight matrices (W_L1_from_L2, W_L2_from_L3) learn
   - Search range: **[1e-6, 1e-2]** (log scale)
   - Current value: 0.0005
   - Rationale: Weights need much slower learning than representations to prevent catastrophic forgetting

3. **momentum** (Representation Update Momentum)
   - Controls temporal smoothing of representation updates
   - Search range: **[0.80, 0.98]** (linear scale)
   - Current value: 0.90
   - Rationale: Adds inertia to learning; prevents chasing noise; but too high delays convergence

### ❌ Parameters NOT OPTIMIZED (Fixed by Design)

1. **weight_decay = 0.98** (FIXED)
   - Set by phase transition requirements
   - Must prevent catastrophic forgetting while enabling generalization
   - NOT varied because it has theoretical justification in active inference

2. **Motor Dynamics** (FIXED)
   - motor_gain = 0.5
   - damping = 0.95
   - reaching_speed = 0.2 × target_distance
   - Rationale: Set by task physics, not tunable hyperparameters

3. **Precision Weights** (FIXED)
   - pi_L1 = 100, pi_L2 = 10, pi_L3 = 1
   - Rationale: Relative precision defines uncertainty model; changing breaks theoretical interpretation

4. **Weight Initialization** (FIXED)
   - W_L2_from_L3(1:3, 1:3) = 0.2 × I (goal→velocity coupling)
   - W_L1_from_L2(4:6, 1:3) = I (motor→velocity mapping)
   - Rationale: Bootstrap values enable initial motion; critical for learning to start

5. **Architecture** (FIXED)
   - Layer sizes: L1=7, L2=6, L3=4
   - Task: 4 trials × 40 seconds each
   - Rationale: Problem-defined, not hyperparameters

## Objective Function

**Primary Goal:** Minimize final reaching distance to targets across all 4 trials

**Formula:**
```
score = reaching_distance_weight × avg_final_reaching_distance + 
        position_rmse_weight × position_prediction_error
```

**Weights:**
- reaching_distance: 1.0 (primary - task performance)
- position_rmse: 0.5 (secondary - learning quality)

**Why this objective:**
- Reaching distance directly measures task performance
- Position RMSE measures model learning quality
- Balances task success with learning fidelity

## Search Strategy

**Algorithm:** Random sampling from parameter space
- Linear search over 3 dimensions
- No gradient computation needed
- Robust to local optima from non-convex landscape

**Number of trials:** Configurable (default: 100-500)
- Lower trials (50-100): Quick exploration
- Medium trials (200-500): Balanced search
- Higher trials (1000+): Thorough exploration (slower)

**Output:** 
- Best parameters found
- All trial results saved to `optimization_results_3D_YYYYMMDD_HHMMSS.mat`
- Visualization plots saved to `optimization_results_3D_visualization.png`

## How to Use

### Run the optimizer:
```matlab
optimize_rao_ballard_parameters()
```

### Apply results to model:
1. Open `hierarchical_motion_inference_3D_EXACT.m`
2. Find the "LEARNING PARAMETERS" section
3. Update with best values from optimization output
4. Re-run the model

### Adjust search sensitivity:
Edit line ~9 in `optimize_rao_ballard_parameters.m`:
```matlab
num_trials = 200;  % Increase for more thorough search
```

## Expected Parameter Ranges

Based on 3D reaching task characteristics:

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| eta_rep | 0.001 - 0.02 | Higher for faster convergence |
| eta_W | 0.0001 - 0.001 | Much smaller than eta_rep |
| momentum | 0.85 - 0.95 | High momentum = smoother learning |

## Interpretation of Results

**Good optimization outcome:**
- Best score: < 1.0 (low reaching error)
- eta_rep: 0.002 - 0.01 (modest learning rates)
- eta_W: 0.0001 - 0.0005 (conservative weight updates)
- momentum: 0.90 - 0.95 (smooth temporal dynamics)

**Poor optimization outcome:**
- All trials produce inf or nan scores → simulation instability
- Best parameters near boundaries → expand search range
- High variability in scores → increase num_trials

## Mathematical Foundation

This optimizer implements **model-based hyperparameter optimization** by:
1. Sampling from bounded parameter space
2. Running full forward model simulations
3. Computing task-relevant objective metrics
4. Tracking best-performing parameters

This approach is principled because:
- **Ergodic sampling:** Explores full parameter space uniformly
- **Task-aligned objective:** Optimizes for actual goal (reaching)
- **Principled scoring:** Multi-metric with theoretical weights
- **Conservative tuning:** Only modifies proven tunable parameters

## References

- Rao & Ballard (1999): "Predictive Coding in the Visual Cortex"
- Friston et al. (2017): "Active Inference and Learning"
- Hyperparameter optimization: Bergstra & Bengio (2012)
