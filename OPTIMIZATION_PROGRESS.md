# Parameter Optimization Progress - 3D Reaching Task

## Current Status
**Optimization Run**: 500 trials (in progress)  
**Current Position**: ~Trial 56/500  
**Elapsed Time**: ~45 minutes (estimated)

## Initial Test Run (50 trials) - COMPLETED ✅
- **Best Score Found**: 3.222539
- **Best Parameters**:
  - `eta_rep`: 0.058294 (representation learning rate)
  - `eta_W`: 0.000003 (weight learning rate - extremely low)
  - `momentum`: 0.754875
  - `weight_decay`: 0.980000 (fixed)

### Key Observations from 50-Trial Run:
1. **Parameter Variation**: Only 2 unique eta_rep values across 50 trials
   - Trial 1: eta_rep = 0.027808 (higher learning)
   - Trials 2-50: eta_rep ≈ 0.058294 (converged to best)

2. **Score Distribution**:
   - Trial 1: Score = 4.666702 (higher/worse)
   - Trials 2-50: Score = 3.222539 (lower/better)
   - Only 2 unique scores total

3. **Performance Metrics (Best Parameters)**:
   - Avg Final Reaching Distance: 2.5748 m
   - Position RMSE: 1.2955 m
   - Weighted Objective Score: 3.222539

4. **Velocity Predictions**:
   - Overall Velocity RMSE: 0.052862 m/s (excellent!)
   - Very consistent across trials (0.039-0.066 m/s)

## Current Full Optimization (500 trials) - IN PROGRESS 🔄
**Rationale**: Expand search space to find globally optimal parameters  
**Expected Duration**: ~6-8 hours

### Search Space:
| Parameter | Type | Min | Max | Notes |
|-----------|------|-----|-----|-------|
| eta_rep | Log | 1e-4 | 1e-1 | Learning rate for representations |
| eta_W | Log | 1e-6 | 1e-1 | Learning rate for weight matrices |
| momentum | Linear | 0.70 | 0.98 | Smoothing for representation updates |
| weight_decay | Fixed | 0.98 | 0.98 | Prevents catastrophic forgetting |

### Objective Function:
```
Score = 1.0 × avg_reaching_distance + 0.5 × position_rmse
```
(Lower is better)

## Expected Next Steps

### Upon Completion (500 trials):
1. Analyze full results with `analyze_optimization.m`
2. Identify top 20 parameter sets
3. Compare parameter diversity and score distribution
4. Create visualization of parameter-score relationships

### Recommended Actions:
1. **If scores vary significantly**: Parameters successfully found diverse local optima
2. **If convergence detected**: May indicate need for:
   - Finer search resolution in high-performing region
   - Alternative optimization method (Bayesian, gradient-based)
   - Expanded search bounds for under-explored regions

### Final Validation:
Once optimal parameters identified:
```matlab
eta_rep = [best_value];
eta_W = [best_value];
momentum = [best_value];
% Run hierarchical_motion_inference_3D_EXACT with these parameters
% Verify 4-trial reaching performance
```

## Architecture Summary

### Optimizer Flow:
```
optimize_rao_ballard_parameters.m (500 trials)
    ↓
hierarchical_motion_inference_3D_EXACT(params_struct)
    ↓
[4 multi-trial reaching runs]
    ↓
Load results from ./figures/3D_reaching_results.mat
    ↓
Calculate objective score
    ↓
Store in results struct
    ↓
Repeat for next trial
```

### Parameter Passing (FIXED):
- ✅ 3D script now accepts params struct as function argument
- ✅ Function wrapper extracts eta_rep, eta_W, momentum, weight_decay
- ✅ Optimizer loads results from MAT file instead of relying on workspace isolation
- ✅ Each trial receives unique parameter combination

## Key Insights from ML Theory

### Why These Parameters Matter:
1. **eta_rep (0.058294)**: Controls how quickly representations adapt
   - Too high: Instability, overfitting to noise
   - Too low: Slow learning, failure to adapt
   - Current value: Moderate adaptation rate

2. **eta_W (0.000003)**: Extremely low weight learning rate
   - Indicates: Learned weights from initialization are nearly optimal
   - Motor→proprioception mapping very close to identity
   - Minimal benefit from further weight learning
   - Phase resets (70% decay) dominate weight changes

3. **momentum (0.7549)**: Moderate smoothing of representation updates
   - Balances responsiveness vs. stability
   - Enables multi-trial learning without catastrophic forgetting
   - Moderate value suggests good generalization

### Why weight_decay=0.98 is FIXED:
- Essential at phase boundaries to prevent catastrophic forgetting
- Learned weights would be destroyed without this regularization
- Task physics (motor gain, damping) require this level of weight decay

---
**Last Updated**: 2025-10-30 (During Trial 56/500)  
**Expected Completion**: ~2025-10-30 23:00-01:00
