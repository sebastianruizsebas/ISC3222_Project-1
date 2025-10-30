# DUAL-HIERARCHY PREDICTIVE CODING ARCHITECTURE
## "Player Chasing Moving Ball" Task with Explicit Task Context

---

## Overview

This document describes the complete implementation of the **dual-hierarchy predictive coding model** with an explicit task context layer, designed to learn both **stable motor skills** and **task-specific reaching strategies** while chasing a moving ball.

**Key Innovation**: Separates learning into two independent regions with **different decay rates**:
- **Motor Region**: Learns stable, goal-independent dynamics (high weight preservation)
- **Planning Region**: Learns task-specific strategies (low weight preservation)

---

## Architecture Overview

### Four-Layer Hierarchical Organization

```
┌─────────────────────────────────────────────────────┐
│  L0: TASK CONTEXT (One-Hot Encoding)                │
│      • R_L0 ∈ ℝ^4 (one per trial)                   │
│      • Explicitly represents which task is active    │
│      • Used for task-gated learning                  │
└─────────────────────────────────────────────────────┘
         ↓                          ↓
    MOTOR REGION              PLANNING REGION
    (Stable Laws)             (Task-Specific)
    
├─────────────────────┐   ├─────────────────────┐
│ L3_motor (Output)   │   │ L3_plan (Output)    │
│ [vx, vy, vz]        │   │ [vx, vy, vz]        │
│ (3D velocity)       │   │ (target velocity)   │
└─────────────────────┘   └─────────────────────┘
    ↑                          ↑
    │ W_motor_L3_to_L2        │ W_plan_L3_to_L2
    │                         │
├─────────────────────┐   ├─────────────────────┐
│ L2_motor (Basis)    │   │ L2_plan (Policies)  │
│ [6 primitives]      │   │ [6 policies]        │
│ (goal-indep.)       │   │ (task-dep.)         │
└─────────────────────┘   └─────────────────────┘
    ↑                          ↑
    │ W_motor_L2_to_L1        │ W_plan_L2_to_L1
    │                         │
├─────────────────────┐   ├─────────────────────┐
│ L1_motor (Proprioception)  │ L1_plan (Goal)  │
│ [x,y,z,vx,vy,vz,bias]     │ [ball_x,y,z,    │
│ (player state)             │  goal_x,y,z,bias]│
└─────────────────────┐   ├─────────────────────┘
     ↑                         ↑
     │ sensory_motor           │ ball_position
     └──────────────────────────┘
```

---

## Component Descriptions

### L0: Task Context Layer

**Purpose**: Explicit representation of task identity for trial-specific learning.

**State**: 
```matlab
R_L0(i, trial) = 1  % One-hot encoding
R_L0(i, :) = [0, 0, 1, 0]  % Example: Trial 3 active
```

**Usage**:
- Determines which trial/task is currently active
- Gates learning in planning region via `task_gate = R_L0(i, current_trial) * 0.7 + 0.3`
- Updated at phase transitions (lines 451-458)

---

### Motor Region: Stable Dynamics Learning

#### Purpose
Learn **stable, generalizable** forward models:
- How motor commands produce motion
- How to integrate velocity into position
- Proprioceptive state prediction

#### Why Separate?
- Motor dynamics are **goal-independent**: knowing how to reach velocity is useful for ALL reaching tasks
- Should preserve learned motor laws across task switches
- High phase transition decay rate: **95% (0.95) and 98% (0.98)** retained

#### L1_motor: Proprioceptive State
```matlab
R_L1_motor = [x, y, z, vx, vy, vz, bias]  % Player position & velocity
E_L1_motor = sensory_proprioception - pred_L1_motor
```

#### L2_motor: Motor Basis Functions
```matlab
R_L2_motor ∈ ℝ^6  % Learned motor primitives (velocity commands)
E_L2_motor = R_L2_motor - pred_L2_motor
```

#### L3_motor: Motor Output
```matlab
R_L3_motor ∈ ℝ^3  % [vx, vy, vz] velocity commands to muscles
```

#### Weight Matrices
- **W_motor_L3_to_L2** (3×6): L3_motor → L2_motor predictions
- **W_motor_L2_to_L1** (7×6): L2_motor → L1_motor predictions

#### Learning Dynamics
```matlab
% Always learning (no task gating)
dW_motor = -(eta_W * pi) * (E * R')
W_motor = W_motor + dW_motor
W_motor = W_motor * 0.99995  % Per-step decay (light)

% Phase transitions
W_motor = 0.95 * W_motor  % 95% retained → preserves motor knowledge
```

---

### Planning Region: Task-Specific Strategies

#### Purpose
Learn **task-specific reaching strategies**:
- How to intercept moving targets
- Which velocity commands reach goals
- Trial-specific policies

#### Why Separate?
- Task strategies are **highly specific**: reaching target 1 ≠ reaching target 2
- Should forget old strategies when new task arrives
- Low phase transition decay rate: **70% (0.70) and 80% (0.80)** retained

#### L1_plan: Goal State
```matlab
R_L1_plan = [ball_x, ball_y, ball_z, goal_x, goal_y, goal_z, bias]
E_L1_plan_ball = [x_ball, y_ball, z_ball] - pred_L1_plan(1:3)
E_L1_plan_goal = [goal_x, goal_y, goal_z] - pred_L1_plan(4:6)
```

#### L2_plan: Policies
```matlab
R_L2_plan ∈ ℝ^6  % Learned interception policies
E_L2_plan = R_L2_plan - pred_L2_plan
```

#### L3_plan: Planning Output
```matlab
R_L3_plan ∈ ℝ^3  % [vx, vy, vz] target velocity for this task
```

#### Weight Matrices
- **W_plan_L3_to_L2** (3×6): L3_plan → L2_plan predictions
- **W_plan_L2_to_L1** (7×6): L2_plan → L1_plan predictions

#### Learning Dynamics with Task Gating
```matlab
% Task-gated learning (depends on whether task is active)
task_gate = R_L0(i, current_trial) * 0.7 + 0.3  % Range [0.3, 1.0]
dW_plan = -(eta_W * pi) * (E * R') * task_gate
W_plan = W_plan + dW_plan
W_plan = W_plan * 0.99995  % Per-step decay (light)

% Phase transitions
W_plan = 0.70 * W_plan  % 70% retained → forgets old strategies
```

---

## Task: Player Chasing Moving Ball

### Ball Trajectory Model

**Continuous motion** with time-varying position:

```matlab
% Ball initial state (per trial)
x_ball(0) = ball_trajectories{trial}.start_pos(1)
v_ball(0) = ball_trajectories{trial}.velocity(1)
a_ball = ball_trajectories{trial}.acceleration(1) * sin(t * 0.001)

% Ball dynamics (every timestep)
x_ball(i+1) = x_ball(i) + dt * v_ball(i)
v_ball(i+1) = v_ball(i) + a_ball(i) * dt

% Smooth sinusoidal acceleration makes motion interesting
a_x = A_x * sin(t * 0.001)
a_y = A_y * sin(t * 0.001 + 1)
a_z = A_z * sin(t * 0.001 + 2)
```

### Interception Objective

```matlab
% The player's goal: reach the same position as the ball
interception_error = ||player_pos - ball_pos||
```

---

## Learning Flow

### Forward Propagation (Top-Down Prediction)

```matlab
% MOTOR REGION
pred_L2_motor = R_L3_motor * W_motor_L3_to_L2'      % L3 → L2
pred_L1_motor = R_L2_motor * W_motor_L2_to_L1'      % L2 → L1
[pred_vx_m, pred_vy_m, pred_vz_m] = pred_L1_motor(4:6)

% PLANNING REGION
pred_L2_plan = R_L3_plan * W_plan_L3_to_L2'         % L3 → L2
pred_L1_plan = R_L2_plan * W_plan_L2_to_L1'         % L2 → L1
[pred_vx_p, pred_vy_p, pred_vz_p] = pred_L1_plan(4:6)

% Combined motor execution
combined_vx = 0.5 * motor_gain * pred_vx_m + 0.5 * motor_gain * pred_vx_p
combined_vy = 0.5 * motor_gain * pred_vy_m + 0.5 * motor_gain * pred_vy_p
combined_vz = 0.5 * motor_gain * pred_vz_m + 0.5 * motor_gain * pred_vz_p

% Kinematics: velocity → position
v_player(i+1) = damping * v_player(i) + combined_velocity
x_player(i+1) = x_player(i) + dt * v_player(i+1)
```

### Error Computation (Bottom-Up)

```matlab
% MOTOR ERRORS
E_L1_motor = sensory_proprioception - pred_L1_motor
E_L2_motor = R_L2_motor - pred_L2_motor

% PLANNING ERRORS
E_L1_plan_ball = [x_ball, y_ball, z_ball] - pred_L1_plan(1:3)
E_L1_plan_goal = [goal_x, goal_y, goal_z] - pred_L1_plan(4:6)
E_L2_plan = R_L2_plan - pred_L2_plan

% Task metric
interception_error = ||x_player - x_ball||
```

### Representation Updates (Predictive Coding)

```matlab
% MOTOR REGION (always learning)
ΔR_L1_motor = eta_rep * E_L1_motor * 0.1
ΔR_L2_motor = eta_rep * (coupling - E_L2_motor) * 0.5  % From proprioceptive coupling
ΔR_L3_motor = eta_rep * mean(E_L2_motor) * 0.1        % Average error

% PLANNING REGION (task-gated)
task_gate = R_L0(i, current_trial) * 0.7 + 0.3        % 0.3-1.0 range
ΔR_L1_plan = eta_rep * E_L1_plan * 0.1
ΔR_L2_plan = eta_rep * (coupling - E_L2_plan) * 0.5 * task_gate
ΔR_L3_plan = eta_rep * mean(E_L2_plan) * 0.1 * task_gate
```

### Weight Learning (Hebbian with Precision Scaling)

```matlab
% MOTOR WEIGHTS (always learning)
dW_motor = -(eta_W * precision) * (E' * R)
W_motor = W_motor + dW_motor
W_motor = W_motor * 0.99995  % Per-step regularization

% PLANNING WEIGHTS (task-gated)
dW_plan = -(eta_W * precision) * (E' * R) * task_gate
W_plan = W_plan + dW_plan
W_plan = W_plan * 0.99995  % Per-step regularization
```

---

## Key Features

### 1. Task Context Gating

**Purpose**: Different learning rates for motor vs. planning regions

```matlab
% Task gate calculation
task_gate = R_L0(i, current_trial) * 0.7 + 0.3

% When task is active (R_L0(i, current_trial) = 1):
task_gate = 1.0  % Planning region learns at full strength

% When task is inactive (R_L0(i, current_trial) = 0):
task_gate = 0.3  % Planning region learns at 30% strength (consolidation)
```

### 2. Differential Phase Decay

**Motor Region** (preserves knowledge):
```matlab
% 95% and 98% retention
W_motor = 0.95 * W_motor  % Major reset
W_motor = 0.98 * W_motor  % Minor reset
```

**Planning Region** (forgets old targets):
```matlab
% 70% and 80% retention
W_plan = 0.70 * W_plan  % Major reset
W_plan = 0.80 * W_plan  % Minor reset
```

### 3. Dynamic Precision Weights

**Responsive to prediction errors**:

```matlab
% Current error (immediate response)
L_current_error = ||E(i)||

% Historical variance (stability)
L_variance = var(error_history)

% Precision formula
π = π_base / (1 + 0.8 * L_current_error + 0.2 * L_variance / ε)
π = clamp(π, π_min, π_max)
```

---

## Parameter Summary

### Architecture Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_trials` | 4 | Number of different ball trajectories |
| `T_per_trial` | 4000s | Duration per trial |
| `dt` | 0.01s | Timestep |

### Decay Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `decay_motor` | 0.95 | Motor weight decay at phase transitions |
| `decay_plan` | 0.70 | Planning weight decay at phase transitions |
| `weight_decay` | 0.99995 | Per-step regularization (all weights) |

### Motor Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `motor_gain` | 0.5 | Command-to-motion scaling |
| `damping` | 0.85 | Velocity decay per timestep |
| `reaching_speed_scale` | 0.5 | Distance-based reaching speed |

### Learning Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `eta_rep` | 0.005 | Representation learning rate |
| `eta_W` | 0.0005 | Weight matrix learning rate |
| `momentum` | 0.98 | Temporal smoothing |

### Precision Parameters
| Parameter | Default | Range |
|-----------|---------|-------|
| `pi_L1_motor` | 100 | 1-1000 |
| `pi_L2_motor` | 10 | 0.1-100 |
| `pi_L1_plan` | 100 | 1-1000 |
| `pi_L2_plan` | 10 | 0.1-100 |

---

## Usage

### Basic Execution
```matlab
% Run with defaults (4 trials, plotting enabled)
hierarchical_motion_inference_dual_hierarchy()

% Run with custom parameters and plotting
params = struct('eta_rep', 0.01, 'motor_gain', 0.6);
hierarchical_motion_inference_dual_hierarchy(params, true)

% Run without plotting (for optimization)
hierarchical_motion_inference_dual_hierarchy(params, false)
```

### Output Files
- `3D_dual_hierarchy_results.mat` - Complete trajectory, weight, and error data
- `3D_dual_hierarchy_summary.png` - Summary plots of learning performance

---

## How to Adapt for Different Tasks

### Change Ball Trajectory
```matlab
% In task configuration section
ball_trajectories{trial} = struct(...
    'start_pos', [1, 1, 1], ...      % Starting position
    'velocity', [0.5, -0.3, 0.2], ... % Constant velocity
    'acceleration', [0.1, 0, 0.05]... % Acceleration pattern
);
```

### Adjust Weight Decay (Motor vs. Planning)
```matlab
% Motor region: increase decay_motor to preserve more knowledge
decay_motor = 0.98;  % 98% retention (was 0.95)

% Planning region: decrease decay_plan to forget more aggressively
decay_plan = 0.50;   % 50% retention (was 0.70)
```

### Modify Task Gating Strength
```matlab
% In representation update section
task_gate = R_L0(i, current_trial) * 0.9 + 0.1;  % Range [0.1, 1.0] (was [0.3, 1.0])
```

---

## Comparison: Single vs. Dual Hierarchy

### Single Hierarchy (Previous)
```
Problem: Motor primitives constantly disrupted by changing targets
Result: L2 motor basis never stabilizes; learning frozen after few trials
```

### Dual Hierarchy (New)
```
Motor Region: Learns stable basis regardless of current task
Planning Region: Learns task-specific strategies with strong decay
Result: Motor knowledge preserved; planning region refreshes each trial
```

---

## Future Extensions

1. **Multi-Step Prediction**: Predict ball position N steps ahead
2. **Skill Composition**: Learn how to combine primitives for complex behaviors
3. **Exploration vs. Exploitation**: Add novelty bonus for underexplored regions
4. **Curriculum Learning**: Start with stationary targets, gradually add motion
5. **Hierarchical Motor Basis**: Add intermediate layer for synergy groups

---

## References

- Rao & Ballard (1999) "Predictive coding in the visual cortex"
- Friston (2005) "A theory of cortical responses"
- Keramati & Gutkin (2014) "Homeostatic plasticity of excitatory-inhibitory balance"

---

**Last Updated**: October 30, 2025  
**Status**: Ready for PSO optimization  
**Optimization Interface**: Compatible with existing PSO optimizer (11 parameters)
