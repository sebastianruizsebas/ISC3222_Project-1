# Predictive Coding & Active Inference: Two-Level Visual Motion Model

**Course:** Symbolic & Numeric Computation  
**Author:** Sebastian  
**Date:** October 2025

---

## Project Overview

This project implements a **hierarchical predictive coding model** inspired by neuroscience theories of perception and active inference. The model demonstrates how the brain might minimize prediction errors through a hierarchy of beliefs, with applications to understanding psychiatric conditions.

### What This Project Does

1. **Symbolically derives** mathematical update rules for a two-level generative model
2. **Numerically simulates** how the system adapts to changing sensory input
3. **Compares stability** under different prior strengths (psychiatric modeling)
4. **Implements high-precision** ODE45 solution

---

## Theoretical Background

### Predictive Coding Theory

The brain is modeled as a **hierarchical prediction machine**:
- **Higher levels** generate predictions about lower levels
- **Lower levels** send prediction errors upward
- The system minimizes **free energy** (weighted prediction errors)

### Our Two-Level Model

```
Level 2: Velocity (v) ──predicts──> Level 1: Position (x)
   ↑                                       ↑
   │                                       │
 Prior (μ_v)                       Sensory Input (x_obs)
```

**Generative Model:**
- Position changes according to velocity: `dx/dt = v`
- Velocity has a Gaussian prior: `v ~ N(μ_v, σ_v²)`

**Inference (Gradient Descent on Free Energy):**
```
F = (1/2) * [(x_obs - v)²/σ_x² + (v - μ_v)²/σ_v²]

dx/dt = -∂F/∂x = (x_obs - v) / σ_x²
dv/dt = -∂F/∂v = (x_obs - v) / σ_x² - (v - μ_v) / σ_v²
```

### Key Parameters

| Parameter | Meaning | Effect |
|-----------|---------|--------|
| `σ_x` | Sensory precision (noise) | Smaller = trust sensors more |
| `σ_v` | Prior strength | Smaller = rigid beliefs, Larger = flexible beliefs |
| `μ_v` | Expected velocity | Prior expectation (e.g., 0 = stationary) |

---

## Files in This Project

### Experiment Scripts

1. **`step1_symbolic_derivation.m`**
   - Derives update rules using MATLAB Symbolic Math Toolbox
   - Displays equations in readable format
   - Saves symbolic expressions to `symbolic_derivations.mat`

2. **`step2_numerical_simulation.m`**
   - Implements Euler integration of the dynamics
   - Simulates response to sudden velocity change (t=5s)
   - Generates 6-subplot figure showing adaptation
   - Saves results to `simulation_results.mat`

3. **`step3_prior_comparison.m`**
   - Tests three prior strengths: σ_v = {0.1, 1.0, 10.0}
   - Demonstrates psychiatric modeling (autism vs. psychosis spectra)
   - Quantifies adaptation time and stability
   - Saves results to `prior_comparison_results.mat`

4. **`step4_ode45_version.m`**
   - High-precision simulation using MATLAB's `ode45` (Runge-Kutta)
   - Adaptive time-stepping for efficiency
   - Compares accuracy with Euler method
   - Saves results to `ode45_results.mat`

5. **`run_all_experiments.m`**
   - Master script to execute all four steps sequentially
   - Generates summary comparison figure
   - **START HERE** for full demonstration

### Output Files (Generated)

- `symbolic_derivations.mat` - Symbolic expressions
- `simulation_results.mat` - Euler simulation data
- `prior_comparison_results.mat` - Multi-prior comparison
- `ode45_results.mat` - High-precision ODE45 data

---

## How to Run the Experiments

### Quick Start (Run Everything)

```matlab
cd 'Project1'
run_all_experiments
```

This will:
1. Run all four steps in sequence
2. Generate all figures
3. Create all output `.mat` files
4. Display summary results

**Note:** Set `pause_between_steps = false` in `run_all_experiments.m` to run non-interactively.

### Run Individual Steps

```matlab
% Step 1: See the math
step1_symbolic_derivation

% Step 2: Basic simulation
step2_numerical_simulation

% Step 3: Compare priors (psychiatric modeling)
step3_prior_comparison

% Step 4: High-precision ODE45
step4_ode45_version
```

---

## Understanding the Results

### Step 1: Symbolic Derivation

**What you'll see:**
- Mathematical update rules displayed in pretty format
- Interpretation of gradient descent dynamics

**Key insight:**
The system minimizes free energy by balancing:
- **Bottom-up** sensory evidence (`x_obs - v`)
- **Top-down** prior expectations (`v - μ_v`)

---

### Step 2: Numerical Simulation

**Experiment setup:**
- Target moves at velocity +2 for 5 seconds
- Then suddenly changes to velocity -1
- System must infer velocity from noisy position observations

**What you'll see (6 subplots):**

1. **Position tracking**: Estimated vs. true position
2. **Velocity inference**: System discovers hidden velocity
3. **Sensory prediction error**: How surprised the system is
4. **Prior prediction error**: Deviation from expected velocity
5. **Free energy**: Overall model quality (decreases over time)
6. **Tracking error**: Absolute position error

**Key observations:**
- System tracks position despite noise
- Infers hidden velocity (not directly observed!)
- Adapts when velocity changes at t=5s
- Free energy spike at change, then decreases

---

### Step 3: Prior Comparison (Psychiatric Modeling)

**Three prior strengths tested:**

#### 1. Strong Prior (σ_v = 0.1) - "Rigid Beliefs"
- **Behavior:** Slow adaptation, stable estimates
- **Clinical analog:** Autism spectrum traits
  - Over-reliance on predictions
  - Difficulty adapting to change
  - Reduced sensory sensitivity
- **Advantage:** Robust to noise
- **Disadvantage:** Slow to update

#### 2. Medium Prior (σ_v = 1.0) - "Balanced"
- **Behavior:** Moderate adaptation speed
- **Clinical analog:** Typical perception
- **Optimal balance:** Flexibility vs. stability

#### 3. Weak Prior (σ_v = 10.0) - "Flexible Beliefs"
- **Behavior:** Fast but volatile adaptation
- **Clinical analog:** Psychotic spectrum traits
  - Under-reliance on predictions
  - Over-reliance on sensory input
  - Hallucinations/delusions (faulty priors)
- **Advantage:** Rapid adaptation
- **Disadvantage:** Susceptible to noise

**What you'll see (6 subplots):**

1. **Belief dynamics**: How each prior adapts
2. **Adaptation speed**: Time to reach new velocity
3. **Free energy**: Model evidence comparison
4. **Position tracking**: Overall performance
5. **Adaptation zoom**: Close-up of t=4.5-6.5s
6. **Belief stability**: Variance over time

**Quantitative table:**
- Adaptation time (seconds)
- Final velocity error
- Average belief variance

**Key insight:**
Psychiatric conditions may arise from imbalanced priors—too strong or too weak.

---

### Step 4: ODE45 High-Precision

**Why ODE45?**
- Adaptive Runge-Kutta method (4th/5th order)
- Better accuracy than fixed-step Euler
- Efficient adaptive time-stepping
- Industry standard for stiff ODEs

**What you'll see (6 subplots):**

1. **Position tracking**: ODE45 solution
2. **Velocity inference**: Smooth high-precision estimate
3. **Phase portrait**: Trajectory in (x, v) space
4. **Position error**: Log-scale tracking error
5. **Velocity error**: Inference accuracy
6. **Adaptive steps**: How ODE45 adjusts dt

**Comparison with Euler:**
- Reports max and RMS differences
- ODE45 typically more accurate by 2-3 orders of magnitude
- Fewer time steps needed (adaptive)

---

## Interpreting the Figures

### Color Coding (Step 3)
- **Red:** Strong prior (rigid)
- **Green:** Medium prior (balanced)
- **Blue:** Weak prior (flexible)
- **Black dashed:** Ground truth

### Vertical Red Lines
- Mark the velocity change at t=5s
- Look for how each method/prior responds

### What to Look For

#### Good Adaptation:
- Quick convergence after velocity change
- Low free energy after initial transient
- Smooth velocity estimates

#### Poor Adaptation:
- Oscillations in velocity estimate
- Persistent high prediction errors
- Slow return to low free energy

---

## Mathematical Details

### Free Energy Functional

The system minimizes:

$$F = \frac{1}{2} \left[ \frac{(x_{obs} - v)^2}{\sigma_x^2} + \frac{(v - \mu_v)^2}{\sigma_v^2} \right]$$

This is equivalent to:
- Maximum a posteriori (MAP) estimation
- Negative log posterior probability
- Weighted least squares

### Update Rules

Position (perceptual inference):
$$\frac{dx}{dt} = \frac{x_{obs} - v}{\sigma_x^2}$$

Velocity (belief update):
$$\frac{dv}{dt} = \frac{x_{obs} - v}{\sigma_x^2} - \frac{v - \mu_v}{\sigma_v^2}$$

### Interpretation

- **First term** in `dv/dt`: Bottom-up sensory drive
- **Second term** in `dv/dt`: Top-down prior constraint
- **Balance** determined by relative magnitudes of σ_x and σ_v

---

## Extensions & Future Work

### Implemented (in this project):
✅ Two-level hierarchy  
✅ Continuous sensory input  
✅ Prior strength comparison  
✅ Psychiatric modeling interpretation  
✅ Multiple integration methods  

### Possible Extensions:

1. **Three-level hierarchy**
   - Add acceleration inference
   - Test deeper hierarchical effects

2. **Stochastic dynamics**
   - Add process noise to velocity
   - Use stochastic differential equations (SDEs)

3. **Active inference**
   - Add motor actions
   - System controls sensory input to minimize surprise

4. **Real sensory data**
   - Replace synthetic motion with real visual tracking
   - Import eye-tracking or mouse-tracking data

5. **Parameter fitting**
   - Fit σ_x, σ_v to real behavioral data
   - Use Bayesian inference for parameter estimation

6. **Alternative priors**
   - Non-Gaussian priors (Laplacian, mixture models)
   - Time-varying priors (context-dependent)

7. **Psychiatric model validation**
   - Compare with clinical datasets
   - Predict behavioral phenotypes

---

## Connection to Course Material

### Symbolic Computation
- MATLAB Symbolic Math Toolbox for derivations
- Automatic differentiation of free energy
- LaTeX-ready equation output
- Similar to Labs 1-2 (Newton's method symbolic work)

### Numeric Computation
- Euler integration (explicit method)
- ODE45 (adaptive Runge-Kutta)
- Error analysis and comparison
- Similar to Lab 4 (SIRV epidemic modeling with `ode45`)

### System Dynamics
- Coupled differential equations
- Hierarchical dynamical systems
- Phase portraits and stability
- Similar to Lab 4 (population dynamics)

### Visualization
- Multi-panel scientific figures
- Time series and phase plots
- Comparative analysis plots
- Similar to Lab 4 and Lab 6 plotting approaches

---

## Key Takeaways

### 1. Hierarchical Inference Works
The system successfully infers hidden velocity from noisy position observations—demonstrates the power of hierarchical generative models.

### 2. Priors Matter
The balance between sensory evidence and prior beliefs critically determines:
- Adaptation speed
- Stability
- Robustness to noise

### 3. Psychiatric Modeling Potential
Simple parameter changes (prior strength) qualitatively reproduce features of:
- Autism spectrum (rigid priors)
- Psychotic spectrum (weak priors)

### 4. Free Energy Principle
Minimizing weighted prediction errors provides a unified framework for:
- Perception (inferring hidden states)
- Learning (updating beliefs)
- Action (active inference extension)

### 5. Computational Methods
- Symbolic derivation ensures mathematical correctness
- Numerical simulation enables exploration of complex dynamics
- Multiple methods (Euler, ODE45) provide validation

---

## Troubleshooting

### "Symbolic Math Toolbox required"
**Solution:** Step 1 requires this toolbox. If not available, the derived equations are already implemented in Steps 2-4, so you can skip Step 1.

### "Figures not appearing"
**Solution:** Check `close all` commands. Remove or use `figure; ...` to create new figures.

### "Out of memory"
**Solution:** Reduce simulation time (`t_span`) or increase time step (`dt`).

### "Results look different each run"
**Solution:** Sensory noise is random. Set `rng(seed)` at start for reproducibility.

### "Adaptation time = inf"
**Solution:** Prior too strong (σ_v too small) or threshold too strict. Adjust parameters.

---

## References & Further Reading

### Predictive Coding & Free Energy
- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.
- Rao, R.P. & Ballard, D.H. (1999). *Predictive coding in the visual cortex*. Nature Neuroscience.

### Active Inference
- Friston, K. et al. (2017). *Active inference: a process theory*. Neural Computation.
- Parr, T. & Friston, K.J. (2019). *Generalised free energy and active inference*. Biological Cybernetics.

### Psychiatric Modeling
- Adams, R.A. et al. (2013). *The computational anatomy of psychosis*. Frontiers in Psychiatry.
- Pellicano, E. & Burr, D. (2012). *When the world becomes 'too real': a Bayesian explanation of autistic perception*. Trends in Cognitive Sciences.

### Numerical Methods
- Dormand, J.R. & Prince, P.J. (1980). *A family of embedded Runge-Kutta formulae*. Journal of Computational and Applied Mathematics.
- MATLAB Documentation: `ode45`, Symbolic Math Toolbox

---

## License & Acknowledgments

This project was created for educational purposes as part of FSU's Symbolic & Numeric Computation course (Fall 2025).

**Acknowledgments:**
- Course instructor: Professor Alan Lemmon
- Theoretical framework: Karl Friston's Free Energy Principle
- Implementation: MATLAB R2024a+

---

## Contact

For questions about this project, contact the course instructor or refer to the course materials.

**Project completed:** October 2025  
**MATLAB version:** R2024a or later recommended  
**Required toolboxes:** Symbolic Math Toolbox (optional for Step 1 only)

---

*"The brain is fundamentally a prediction machine that minimizes surprise."*  
— Karl Friston
