# Parameter Documentation - Dual-Hierarchy Predictive Coding Model

**Date**: November 3, 2025  
**Model**: Hierarchical Motion Inference with Rao & Ballard Active Inference  
**Purpose**: Complete reference for all parameters used in motor learning optimization

---

## Table of Contents

1. [Learning Parameters](#learning-parameters)
2. [Weight Decay Parameters](#weight-decay-parameters)
3. [Motor Dynamics Parameters](#motor-dynamics-parameters)
4. [Weight Initialization Parameters](#weight-initialization-parameters)
5. [Task-Conditional Learning Parameters](#task-conditional-learning-parameters)
6. [Precision Adaptation Parameters](#precision-adaptation-parameters)
7. [PSO Hyperparameters](#pso-hyperparameters)
8. [Simulation Parameters](#simulation-parameters)

---

## Learning Parameters

### `eta_rep` (Representation Learning Rate)
- **Range**: 10^-5 to 10^-4 (typically 0.00001 to 0.0001)
- **Scale**: Logarithmic
- **Purpose**: Controls how quickly hierarchical representations (R_L1, R_L2, R_L3) update based on prediction errors.
- **Typical Value**: 0.00005 (1e-5)
- **Details**: Lower values → slower learning, more stable; Higher values → faster learning, more noise. Affects all layers equally: `R_L1(i+1) = R_L1(i) + eta_rep * error`.

### `eta_W` (Weight Learning Rate)
- **Range**: 10^-7 to 10^-5 (typically 0.0000001 to 0.00001)
- **Scale**: Logarithmic
- **Purpose**: Controls how quickly synaptic weights (W_motor_*, W_plan_*) update based on prediction errors.
- **Typical Value**: 0.001 (1e-3) or 0.0001 (1e-4)
- **Details**: Much smaller than eta_rep because weights are shared across many timesteps and should be more stable. Drives the core learning mechanism: `W(i+1) = W(i) - eta_W * dW`.

### `momentum` (Temporal Smoothing)
- **Range**: 0.70 to 0.99
- **Scale**: Linear [0, 1]
- **Purpose**: Implements exponential moving average for representation updates, creating temporal continuity and smoothing out noise.
- **Typical Value**: 0.90 (90% of previous value retained)
- **Details**: High momentum (0.90+) creates smooth, slow trajectories; low momentum (0.70-0.80) creates reactive, noisy updates. Formula: `R(i+1) = momentum * R(i) + (1-momentum) * delta_R`. Also controls weight decay: `decay = 1 - momentum`.

---

## Weight Decay Parameters

### `weight_decay` (Global Weight Retention)
- **Range**: 0.65 to 0.80
- **Scale**: Linear [0, 1]
- **Purpose**: Global scaling factor applied to all weights during trial transitions to prevent catastrophic forgetting while allowing some plasticity.
- **Typical Value**: 0.98
- **Details**: Applied as `W_new = weight_decay * W_old` at phase boundaries (between trials). Value 0.98 means 98% of weight retained, 2% forgotten. Affects both motor and planning regions uniformly.

### `decay_motor` (Motor Weight Retention at Phase Boundaries)
- **Range**: 0.88 to 0.94 (88-94% retention)
- **Scale**: Linear [0, 1]
- **Purpose**: Task-conditional decay applied ONLY to motor weights during trial transitions—higher retention preserves learned velocity control across different tasks.
- **Typical Value**: 0.95
- **Details**: Motor learns generalizable forward models (velocity→output mapping) that should be stable across tasks, so high retention (95%) is appropriate. Applied as `W_motor = decay_motor * W_motor` at trial boundaries.

### `decay_plan` (Planning Weight Retention at Phase Boundaries)
- **Range**: 0.20 to 0.68 (20-68% retention)
- **Scale**: Linear [0, 1]
- **Purpose**: Task-conditional decay applied ONLY to planning weights during trial transitions—lower retention encourages forgetting of old task-specific strategies when switching tasks.
- **Typical Value**: 0.70
- **Details**: Planning learns task-specific interception strategies that must adapt when target velocity changes. Lower retention (70%) allows faster adaptation to new task structure. Applied as `W_plan = decay_plan * W_plan` at trial boundaries.

---

## Motor Dynamics Parameters

### `motor_gain` (Motor Command Scaling)
- **Range**: 0.25 to 1.1
- **Scale**: Linear [0.25, 1.1]
- **Purpose**: Scales motor commands before integrating into velocity dynamics: `v_player(i+1) = motor_gain * pred_vel + noise`.
- **Typical Value**: 0.5 to 1.0
- **Details**: Controls trajectory speed—higher values create faster, more aggressive movements; lower values create slower, more cautious reaches. Also affects learning dynamics: too high → overshooting errors, too low → insufficient exploration.

### `damping` (Velocity Decay)
- **Range**: 0.53 to 0.92
- **Scale**: Linear [0.53, 0.92]
- **Purpose**: Implements first-order low-pass filtering on player velocity: `v_player(i+1) = damping * v_player(i) + motor_command`.
- **Typical Value**: 0.85
- **Details**: High damping (0.92) creates smooth, sluggish motion; low damping (0.53) creates sharp, responsive motion. Biologically motivated by muscle mechanics and neural motor filtering. Acts as a passive brake on accelerations.

### `reaching_speed_scale` (Curriculum Learning Speed Factor)
- **Range**: 0.20 to 1.35
- **Scale**: Linear [0.20, 1.35]
- **Purpose**: Multiplicative scaling of motor commands during curriculum learning phases—higher values accelerate learning rates for harder tasks.
- **Typical Value**: 0.5 to 1.0
- **Details**: Used in curriculum learning: when trial difficulty increases, reaching_speed_scale is increased to give the learner more "muscle" to work with. Implemented as `eta_W_curriculum = eta_W * reaching_speed_scale` during phase transitions.

---

## Weight Initialization Parameters

### `W_motor_gain` (Motor Weight Initialization Scale)
- **Range**: 0.1 to 0.75
- **Scale**: Linear [0.1, 0.75]
- **Purpose**: Controls initial magnitude of motor weight matrices before training begins: `W_motor_init = randn(...) * W_motor_gain`.
- **Typical Value**: 0.5
- **Details**: Higher values create larger initial weights (faster initial predictions but more errors); lower values create smaller weights (slower initial predictions but more stable learning). Affects early convergence speed: too high → instability, too low → slow initial learning.

### `W_plan_gain` (Planning Weight Initialization Scale)
- **Range**: 0.35 to 0.95
- **Scale**: Linear [0.35, 0.95]
- **Purpose**: Controls initial magnitude of planning weight matrices: `W_plan_init = randn(...) * W_plan_gain`.
- **Typical Value**: 0.5 to 0.65
- **Details**: Planning weights are typically initialized larger than motor weights (0.35-0.95 vs 0.1-0.75) because planning has more freedom to specialize per task. Affects convergence of task-specific strategies.

---

## Task-Conditional Learning Parameters

### `interference_penalty_weight` (Cross-Task Interference Penalty)
- **Range**: 0.0 to 0.05
- **Scale**: Linear [0.0, 0.05]
- **Purpose**: Controls strength of regularization that penalizes off-task errors to prevent catastrophic forgetting and encourage task selectivity.
- **Typical Value**: 0.01
- **Details**: Added to free energy: `FE += interference_penalty * sum(cross_task_errors^2)`. Higher values create stronger task separation (fewer errors on other tasks) but may slow learning on current task. Value 0.01 is recommended baseline (1% of free energy weight).

---

## Precision Adaptation Parameters

### `alpha_precision_gain` (Error-Driven Precision Sensitivity)
- **Range**: 1.1 to 2.0
- **Scale**: Linear [1.1, 2.0]
- **Purpose**: Controls exponential sensitivity of precision adaptation to prediction errors: `precision_new = precision_old * exp(alpha * error)`.
- **Typical Value**: 1.5
- **Details**: Higher alpha (2.0) creates more aggressive precision changes (high errors → massive precision increase); lower alpha (1.1) creates smoother, more gradual precision adaptation. Determines how quickly the model "trusts" or "distrusts" predictions based on errors.

### `pi_L1_motor_max` (L1 Motor Maximum Precision Bound)
- **Range**: 250 to 500
- **Scale**: Linear [250, 500]
- **Purpose**: Upper bound on precision for L1 motor layer (proprioceptive/proprioceptive error scaling): `precision = max(10, min(pi_L1_motor_max, precision))`.
- **Typical Value**: 500
- **Details**: Minimum precision is hard-coded at 10. Higher maximum (500) allows stronger suppression of proprioceptive errors when confident; lower maximum (250) creates more uniform error weighting. Prevents precision from growing unbounded.

### `pi_L2_motor_max` (L2 Motor Maximum Precision Bound)
- **Range**: 65 to 200
- **Scale**: Linear [65, 200]
- **Purpose**: Upper bound on precision for L2 motor layer (basis function error scaling).
- **Typical Value**: 100
- **Details**: Minimum precision hard-coded at 1. Controls how strongly basis function mismatches affect learning. Higher values penalize basis function errors more heavily when the model is confident.

### `pi_L1_plan_max` (L1 Planning Maximum Precision Bound)
- **Range**: 50 to 320
- **Scale**: Linear [50, 320]
- **Purpose**: Upper bound on precision for L1 planning layer (goal/target representation error scaling).
- **Typical Value**: 500
- **Details**: Minimum precision hard-coded at 10. Affects how strongly the model weights ball position prediction errors. Higher values → model more confident in ball trajectory predictions, lower values → model more uncertain.

### `pi_L2_plan_max` (L2 Planning Maximum Precision Bound)
- **Range**: 7 to 60
- **Scale**: Linear [7, 60]
- **Purpose**: Upper bound on precision for L2 planning layer (planning policy error scaling).
- **Typical Value**: 50
- **Details**: Minimum precision hard-coded at 1. Controls how strongly planning basis function errors affect the task-conditional learning signal. Typically much lower than motor L2 because planning policies are more variable across tasks.

---

## PSO Hyperparameters

### `num_particles` (Swarm Size)
- **Range**: Typically 50-500
- **Value**: 250
- **Purpose**: Number of particles (parameter sets) in the swarm—each particle represents one candidate solution.
- **Details**: More particles → better exploration of parameter space but slower iteration. 250 is standard balance between exploration and computational cost. Each particle evaluates the model multiple times per iteration.

### `num_iterations` (PSO Generations)
- **Range**: Typically 20-100
- **Value**: 50
- **Purpose**: Number of generations the swarm evolves—determines total evaluations: `total_evals = num_particles * num_iterations * 2` (2 trials per particle per iteration).
- **Details**: 50 iterations × 250 particles × 2 trials = 25,000 model evaluations total. More iterations → better convergence but exponentially more computation time.

### `w` (PSO Inertia Weight)
- **Range**: 0.4 to 0.9
- **Value**: 0.7
- **Purpose**: Controls momentum of particle velocity updates—how much previous velocity is retained in the next PSO step.
- **Details**: `velocity_new = w * velocity_old + c1 * (best - current) + c2 * (global_best - current)`. Higher w (0.9) → more exploration, slower convergence; lower w (0.4) → more exploitation, faster convergence but may get stuck in local minima.

### `c1` (Cognitive/Attraction to Personal Best)
- **Range**: 0.5 to 2.0
- **Value**: 0.8
- **Purpose**: PSO coefficient controlling each particle's attraction to its own best-found solution.
- **Details**: Encourages particles to "remember" their personal best scores. Higher c1 (2.0) → particles stick closer to their own bests; lower c1 (0.5) → particles can stray further and explore more.

### `c2` (Social/Attraction to Global Best)
- **Range**: 0.5 to 2.0
- **Value**: 1.5
- **Purpose**: PSO coefficient controlling each particle's attraction to the swarm's global best solution.
- **Details**: Encourages swarm convergence toward best known solution. Higher c2 (2.0) → strong convergence pressure, fast but risky; lower c2 (0.5) → loose convergence, more exploration. Usually c2 > c1 to favor swarm intelligence.

### `noise_scale` (PSO Stochastic Perturbation)
- **Range**: 0.01 to 0.2
- **Value**: 0.1
- **Purpose**: Scales random perturbations added to particles to escape local minima and maintain exploration.
- **Details**: At each PSO step, a small random noise (0-10% of parameter range) is added to each particle's position. Higher noise → more exploration but noisier convergence; lower noise → sharper convergence but may miss global optimum.

### `fast_debug_mode` (Quick Testing Flag)
- **Range**: true or false
- **Value**: false (for production runs)
- **Purpose**: When true, PSO uses shorter trial durations (2.5s instead of 5-10s) for faster iteration during development.
- **Details**: During development, set to true to test the PSO loop in minutes; for final optimization, set to false to run full-duration trials (takes hours). Also sets `debug_dt = 0.02` (larger timestep) for speed.

---

## Simulation Parameters

### `dt` (Timestep Size)
- **Range**: 0.01 to 0.02 seconds
- **Value**: 0.01 (default), 0.02 (debug mode)
- **Purpose**: Integration timestep for continuous dynamics—controls temporal resolution of kinematics and learning.
- **Details**: `position(i+1) = position(i) + dt * velocity(i)`. Smaller dt (0.01) → more accurate but slower; larger dt (0.02) → less accurate but faster. Standard value 0.01 gives ~100 steps per second.

### `workspace_bounds` (3D Reachable Workspace)
- **Range**: [-5, 5] meters in x,y; [0, 3] meters in z
- **Value**: `[-5 5; -5 5; 0 3]`
- **Purpose**: Defines the safe operating region for player movements—positions outside bounds are clipped.
- **Details**: Motor L1 representations (player position) are constrained to workspace bounds. Biologically motivated: defines arm's reachable space. z-axis starts at 0 (table surface) and extends 3m up.

### `target_trajectories` (Ball Movement Specifications)
- **Components**: `start_position`, `velocity`, `acceleration`
- **Purpose**: Defines moving target motion across multiple trials with constant-velocity profiles.
- **Details**: Each trial has independent target trajectory. Example: Trial 1 starts at [5,5,1.5] with velocity [-1.5,-1.5,0]. Constant velocity (accel=0) tests predictive coding: can model learn velocity prediction?

### `termination_distance` (Interception Success Criterion)
- **Range**: 0.1 to 1.0 meters
- **Value**: Typically 0.5m
- **Purpose**: Distance threshold below which a trial is considered successfully completed.
- **Details**: When `|player - ball| < termination_distance`, the trial ends early. Enables measurement of convergence speed and success rate. Typical value 0.5m represents successful hand-ball contact.

### `n_L1_motor`, `n_L2_motor`, `n_L3_motor` (Layer Sizes)
- **Purpose**: Number of neurons/units in each motor hierarchy level.
- **Typical Values**: L1=7 (3 pos + 3 vel + 1 bias), L2=12 (basis functions), L3=6 (motor output)
- **Details**: L1 motor encodes proprioception (position, velocity, bias); L2 motor represents learned basis functions for motor control; L3 motor outputs 6 motor commands. Larger layers → more expressivity but slower learning.

### `n_L1_plan`, `n_L2_plan`, `n_L3_plan` (Planning Layer Sizes)
- **Purpose**: Number of neurons/units in each planning hierarchy level.
- **Typical Values**: L1=7 (3 ball_pos + 3 ball_vel + 1 bias), L2=12 (basis functions), L3=6 (planning outputs)
- **Details**: L1 planning encodes ball state (position, velocity); L2 represents task-specific basis functions; L3 outputs planning signals. Task-indexed: separate copies of L2 and L3 for each trial.

### `idx_pos`, `idx_vel`, `idx_bias` (Semantic Indices)
- **Purpose**: Define which L1 elements correspond to position, velocity, and bias.
- **Values**: `idx_pos = [1:3]`, `idx_vel = [4:6]`, `idx_bias = [7]`
- **Details**: Enforces semantic meaning: position updates use position error, velocity updates use velocity error, bias stays 1. Enables robust representation learning and prevents accidental mixing of coordinate systems.

### `objective_weights.reaching_distance` (Reaching Error Weight)
- **Range**: 0.5 to 2.0
- **Value**: 1.2
- **Purpose**: Relative weighting of reaching distance error vs free energy in PSO objective function.
- **Details**: `objective = reaching_distance * reaching_error + position_rmse * position_error`. Higher weight (1.2) → PSO prioritizes accurate interception over minimizing free energy; lower weight → PSO prioritizes model efficiency.

### `objective_weights.position_rmse` (Position RMSE Weight)
- **Range**: 0.5 to 2.0
- **Value**: 1.1
- **Purpose**: Relative weighting of position RMSE vs reaching distance in PSO objective.
- **Details**: Balances fine-grained trajectory accuracy (RMSE) against final interception error. Values close to reaching_distance weight (1.1 vs 1.2) create balanced optimization across trajectory quality and final accuracy.

---

## Derived/Computed Parameters

### `noise_annealing_factor` (Motor Exploration Decay)
- **Computed as**: `max(0.01, 1.0 - (i / 1000))`
- **Purpose**: Anneals motor exploration noise over first ~1000 timesteps.
- **Details**: Starts at 1.0 (100% exploration), decays to 0.01 (1% exploration). `noise_scale = 0.05 * noise_annealing_factor` means initial noise ≈0.05 m/s, final noise ≈0.0005 m/s. Implements exploration→exploitation transition.

### `decay` (Learning Rate Multiplier)
- **Computed as**: `1 - momentum`
- **Purpose**: Converts momentum (retention ratio) into update fraction.
- **Details**: If `momentum = 0.90`, then `decay = 0.10`, meaning 10% of error-driven change is applied per step. Formula: `R_new = momentum * R_old + decay * delta_R`. Inverse relationship ensures high momentum = low decay rate.

### `max_finite_value` (Overflow Protection Threshold)
- **Value**: 1e12
- **Purpose**: Hard upper bound preventing NaN/Inf propagation in error and free energy computations.
- **Details**: Any error or free energy component exceeding 1e12 is clipped: `error_clipped = min(1e12, error)`. Protects against exponential explosion from exponential precision scaling.

### `min_precision`, `max_precision` (Precision Bounds)
- **Values**: min = 1e-12, max = 1e12
- **Purpose**: Hard bounds on all precision parameters to prevent numerical instability.
- **Details**: Ensures `1e-12 <= precision <= 1e12`. Prevents division by zero (too-low precision) and explosive precision updates (too-high precision). Applied to all four precision variables (pi_L1_motor, pi_L2_motor, pi_L1_plan, pi_L2_plan).

---

## Quick Reference: Default Parameter Set

```matlab
params = struct(...
    'eta_rep', 0.00005, ...          % Representation learning rate
    'eta_W', 0.001, ...              % Weight learning rate
    'momentum', 0.90, ...            % Temporal smoothing
    'weight_decay', 0.98, ...        % Global weight retention
    'decay_motor', 0.95, ...         % Motor weight decay
    'decay_plan', 0.70, ...          % Planning weight decay
    'motor_gain', 0.5, ...           % Motor command scaling
    'damping', 0.85, ...             % Velocity filtering
    'reaching_speed_scale', 0.5, ... % Curriculum scaling
    'W_motor_gain', 0.5, ...         % Motor init scale
    'W_plan_gain', 0.65, ...         % Planning init scale
    'interference_penalty_weight', 0.01, ... % Cross-task penalty
    'alpha_precision_gain', 1.5, ... % Error-driven precision sensitivity
    'pi_L1_motor_max', 500, ...      % Max L1 motor precision
    'pi_L2_motor_max', 100, ...      % Max L2 motor precision
    'pi_L1_plan_max', 500, ...       % Max L1 plan precision
    'pi_L2_plan_max', 50 ...         % Max L2 plan precision
);
```

---

## Parameter Tuning Guidelines

### For Faster Learning
- **Increase**: `eta_rep`, `eta_W`, `motor_gain`, `W_motor_gain`
- **Decrease**: `momentum`, `decay_motor`, `decay_plan`
- **Effect**: Model updates more aggressively but risks instability

### For Smoother Learning
- **Increase**: `momentum`, `damping`, `decay_motor`
- **Decrease**: `eta_rep`, `eta_W`, `motor_gain`
- **Effect**: Learning becomes slower but more stable, less noisy

### For Better Task Generalization
- **Increase**: `decay_motor`, `W_motor_gain`
- **Decrease**: `decay_plan`, `interference_penalty_weight`
- **Effect**: Motor weights preserved across tasks, planning can diverge

### For Better Task Specialization
- **Decrease**: `decay_plan`, `weight_decay`
- **Increase**: `interference_penalty_weight`
- **Effect**: Planning weights reset at task boundaries, cross-task interference penalized

### For Faster PSO Convergence
- **Increase**: `c2`, `w` (more social cohesion)
- **Decrease**: `noise_scale` (less exploration)
- **Effect**: Swarm converges faster but may miss global optimum

### For Better Global Optimum Search
- **Decrease**: `c2`, `w` (more exploration)
- **Increase**: `noise_scale`, `num_iterations` (more exploration time)
- **Effect**: Takes longer but finds better parameters

---

## References

- **Predictive Coding**: Friston, K. (2010). The free-energy principle. Nature Reviews Neuroscience.
- **Active Inference**: Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex.
- **PSO**: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
- **Motor Learning**: Shadmehr, R., & Mussa-Ivaldi, F. A. (1994). Adaptive representation of dynamics during learning of a motor task.

