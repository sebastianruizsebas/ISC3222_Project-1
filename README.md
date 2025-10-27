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

### Model Classes (Object-Oriented Implementation)

1. **`PredictiveCodingModel.m`** (Base Class)
   - Abstract base class for two-level predictive coding models
   - Common functionality: visualization, saving, performance metrics
   - Properties: time parameters, precision, state variables, histories
   - Methods: `generateSensoryInput()`, `run()`, `visualize()`, `save()`, `printSummary()`

2. **`EulerModel.m`** (Inherits from PredictiveCodingModel)
   - Implements Euler integration (fixed-step method)
   - Simple, explicit forward integration
   - Good for understanding basic dynamics

3. **`ODE45Model.m`** (Inherits from PredictiveCodingModel)
   - Implements adaptive Runge-Kutta method (4th/5th order)
   - High-precision with adaptive time-stepping
   - Includes phase portrait visualization
   - Automatically compares with Euler method

4. **`RaoBallardModel.m`** (Standalone Class)
   - Three-level hierarchical model (position → velocity → acceleration)
   - Explicit prediction and error units
   - Separate representations for each hierarchical level
   - Based on Rao & Ballard (1999) cortical architecture

### Experiment Scripts

1. **`step1_symbolic_derivation.m`**
   - Derives update rules using MATLAB Symbolic Math Toolbox
   - Displays equations in readable format
   - Saves symbolic expressions to `symbolic_derivations.mat`

2. **`step2_numerical_simulation.m`**
   - Creates and runs `EulerModel` instance
   - Simulates response to sudden velocity change (t=5s)
   - Generates 6-subplot figure showing adaptation
   - Saves results to `simulation_results.mat`

3. **`step3_prior_comparison.m`**
   - Tests three prior strengths: σ_v = {0.1, 1.0, 10.0}
   - Demonstrates psychiatric modeling (autism vs. psychosis spectra)
   - Quantifies adaptation time and stability
   - Saves results to `prior_comparison_results.mat`

4. **`step4_ode45_version.m`**
   - Creates and runs `ODE45Model` instance
   - High-precision simulation using adaptive Runge-Kutta
   - Compares accuracy with `EulerModel`
   - Saves results to `ode45_results.mat`

5. **`step5_rao_ballard_extension.m`**
   - Creates and runs `RaoBallardModel` instance
   - Three-level hierarchical predictive coding
   - Explicit prediction and error units

6. **`step6_compare_architectures.m`**
   - Compares two-level (Steps 2-4) vs. three-level (Step 5) architectures
   - Analyzes prediction error dynamics and free energy

7. **`step7_image_prediction.m`**
   - Extends to 2D spatial predictions
   - Image-based predictive coding demonstration

8. **`run_all_experiments.m`**
   - Master script to execute all steps sequentially
   - Saves figures to `figures/` directory for remote access
   - **START HERE** for full demonstration

### Output Files (Generated)

- `symbolic_derivations.mat` - Symbolic expressions
- `simulation_results.mat` - Euler simulation data
- `prior_comparison_results.mat` - Multi-prior comparison
- `ode45_results.mat` - High-precision ODE45 data
- `rao_ballard_results.mat` - Three-level model results
- `image_prediction_results.mat` - 2D spatial predictions
- `figures/*.png` - All generated figures saved as PNG files

---

## How to Run the Experiments

### Quick Start (Recommended)

Run all experiments and save figures:

```matlab
cd 'c:\Users\srseb\OneDrive\School\FSU\Fall 2025\Symbolic Numeric Computation w Alan Lemmon\Project1'
run_all_experiments
```

This will:
1. Run all seven steps in sequence
2. Generate all figures and save them to `figures/` directory
3. Create all output `.mat` files
4. Display summary results in console

**For Remote Terminal Access:**
- All figures are automatically saved as PNG files in `figures/` directory
- You can download them after the script completes
- Figure names: `step1.png`, `step2.png`, `step3.png`, etc.

### Run Individual Steps

Each step can be run independently:

```matlab
% Step 1: See the symbolic math derivations
step1_symbolic_derivation

% Step 2: Basic Euler simulation (uses EulerModel class)
step2_numerical_simulation

% Step 3: Compare priors (psychiatric modeling)
step3_prior_comparison

% Step 4: High-precision ODE45 (uses ODE45Model class)
step4_ode45_version

% Step 5: Rao & Ballard three-level model (uses RaoBallardModel class)
step5_rao_ballard_extension

% Step 6: Compare two-level vs. three-level architectures
step6_compare_architectures

% Step 7: 2D spatial predictions
step7_image_prediction
```

### Using the Model Classes Directly

The refactored code uses object-oriented programming. You can create and run models programmatically:

#### Example 1: Basic Euler Simulation

```matlab
% Create model
model = EulerModel(0.01, 10, 0.1, 1.0);  % dt, T, sigma_x, sigma_v

% Generate sensory input (noise_std)
model.generateSensoryInput(0.05);

% Run simulation
model.run();

% Visualize results
model.visualize();

% Print performance metrics
model.printSummary();

% Save results
model.save('my_results.mat');
```

#### Example 2: High-Precision ODE45

```matlab
% Create ODE45 model
model = ODE45Model(0.01, 10, 0.1, 1.0);

% Generate sensory input
model.generateSensoryInput(0.05);

% Run adaptive Runge-Kutta simulation
model.run();

% Visualize (includes phase portrait)
model.visualize();

% Access results
position = model.x_history;
velocity = model.v_history;
errors = model.free_energy;
```

#### Example 3: Three-Level Rao & Ballard Model

```matlab
% Create Rao & Ballard model
% Parameters: dt, T, pi_x, pi_v, pi_a (precision weights)
model = RaoBallardModel(0.01, 10, 100, 10, 1);

% Generate sensory input
% Parameters: noise, a_before, a_after, change_time
model.generateSensoryInput(0.05, 0, -3, 5.0);

% Run three-level hierarchical inference
model.run();

% Visualize all levels
model.visualize();

% Print summary
model.printSummary();

% Access three-level states
position = model.x_rep;
velocity = model.v_rep;
acceleration = model.a_rep;
```

#### Example 4: Custom Experiment

```matlab
% Compare different noise levels
noise_levels = [0.01, 0.05, 0.1];
results = cell(1, 3);

for i = 1:3
    model = EulerModel(0.01, 10, 0.1, 1.0);
    model.generateSensoryInput(noise_levels(i));
    model.run();
    results{i} = model;
end

% Plot comparison
figure;
hold on;
for i = 1:3
    plot(results{i}.t, results{i}.free_energy, ...
         'DisplayName', sprintf('Noise = %.2f', noise_levels(i)));
end
legend; xlabel('Time (s)'); ylabel('Free Energy');
title('Effect of Sensory Noise on Free Energy');
```

### Model Class API Reference

#### Common Methods (PredictiveCodingModel, EulerModel, ODE45Model)

| Method | Description | Parameters |
|--------|-------------|------------|
| `generateSensoryInput(noise_std)` | Create synthetic motion data | `noise_std` - sensory noise level |
| `run()` | Execute simulation | None |
| `visualize()` | Create standard plots | Optional: custom title |
| `save(filename)` | Save results to .mat file | `filename` - path to save |
| `printSummary()` | Display performance metrics | None |

#### RaoBallardModel Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `generateSensoryInput(noise, a_before, a_after, t_change)` | Create motion with acceleration change | 4 parameters for acceleration profile |
| `run()` | Execute three-level inference | None |
| `visualize()` | Create 9-subplot hierarchy figure | None |
| `save(filename)` | Save all levels to .mat | `filename` |
| `printSummary()` | Display three-level metrics | None |

---

## Understanding the Results

### Architecture Overview

The project now uses an **object-oriented design** with three main model classes:

```
PredictiveCodingModel (Base Class)
├── EulerModel          (Step 2: Fixed-step integration)
├── ODE45Model          (Step 4: Adaptive Runge-Kutta)
└── [Common methods: visualize, save, printSummary]

RaoBallardModel (Standalone)
└── (Step 5: Three-level hierarchical inference)
```

**Benefits of Class-Based Design:**
- ✅ Modular and reusable code
- ✅ Consistent interface across methods
- ✅ Easy to extend and customize
- ✅ Less code duplication (~60% reduction in script length)

### Step 1: Symbolic Derivation

**What you'll see:**
- Mathematical update rules displayed in pretty format
- Interpretation of gradient descent dynamics

**Key insight:**
The system minimizes free energy by balancing:
- **Bottom-up** sensory evidence (`x_obs - v`)
- **Top-down** prior expectations (`v - μ_v`)

---

### Step 2: Numerical Simulation (EulerModel)

**Implementation:**
- Uses `EulerModel` class with fixed-step Euler integration
- Object-oriented approach makes code cleaner and more maintainable

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

**Code example:**
```matlab
model = EulerModel(0.01, 10, 0.1, 1.0);
model.generateSensoryInput(0.05);
model.run();
model.visualize();
```

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

### Step 4: ODE45 High-Precision (ODE45Model)

**Implementation:**
- Uses `ODE45Model` class with adaptive Runge-Kutta method
- Automatically compares with `EulerModel` for validation
- Includes dynamics method for ODE solver integration

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
6. **Free energy**: Minimization over time

**Comparison with Euler:**
- Automatically creates `EulerModel` for comparison
- Reports max and RMS differences
- ODE45 typically more accurate by 2-3 orders of magnitude
- Fewer adaptive time steps needed

**Code example:**
```matlab
model = ODE45Model(0.01, 10, 0.1, 1.0);
model.generateSensoryInput(0.05);
model.run();  % Adaptive integration
model.visualize();  % Includes phase portrait
```

---

### Step 5: Rao & Ballard Extension (RaoBallardModel)

**Implementation:**
- Uses standalone `RaoBallardModel` class
- Different architecture from base predictive coding (separate error/prediction units)

**New in this step:**
- Three-level hierarchical model (position → velocity → acceleration)
- Explicit prediction and error units
- Separate learning rates for representations vs. errors
- Cascading inference through multiple levels

**What you'll see (9 subplots):**

1. **Position representation**: Level 1 inference
2. **Velocity representation**: Level 2 inference
3. **Acceleration representation**: Level 3 inference
4. **Sensory error**: ε_x (bottom-up)
5. **Velocity error**: ε_v (middle level)
6. **Acceleration error**: ε_a (top-down)
7. **Predictions**: Top-down signals
8. **Free energy**: Total model evidence
9. **Architecture diagram**: Information flow visualization

**Key architectural differences:**

| Feature | Free Energy (Steps 2-4) | Rao & Ballard (Step 5) |
|---------|-------------------------|------------------------|
| **Predictions** | Implicit | Explicit units |
| **Errors** | Gradient terms | Separate neurons |
| **Levels** | 2 (x, v) | 3 (x, v, a) |
| **Neural realism** | Abstract | Closer to cortex |
| **Computation** | Gradient descent | Error propagation |
| **Base class** | PredictiveCodingModel | Standalone |

**Code example:**
```matlab
model = RaoBallardModel(0.01, 10, 100, 10, 1);
model.generateSensoryInput(0.05, 0, -3, 5.0);
model.run();
model.visualize();  % 9-panel figure
```

**Applications:**
- Visual cortex modeling (V1 → V2 → MT hierarchy)
- Demonstrates cortical microcircuit structure
- Canonical computation across brain regions

---

### Step 6: Compare Architectures

Compares the two-level free energy model (Steps 2-4) with the three-level Rao & Ballard model (Step 5):
- Loads saved results from both approaches
- Analyzes prediction errors and adaptation dynamics
- Highlights architectural trade-offs

---

### Step 7: Image Prediction

Extends the 1D motion model to 2D spatial predictions:
- Demonstrates how predictive coding scales to images
- Shows prediction error patterns in spatial domain

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
✅ Two-level hierarchy with class-based design  
✅ Three-level Rao & Ballard architecture  
✅ Continuous sensory input  
✅ Prior strength comparison  
✅ Psychiatric modeling interpretation  
✅ Multiple integration methods (Euler, ODE45)  
✅ Object-oriented design for modularity  
✅ Remote terminal support (auto-save figures)  
✅ 2D spatial predictions  

### Code Architecture Improvements:
✅ **Base class pattern** - `PredictiveCodingModel` provides common functionality  
✅ **Inheritance** - `EulerModel` and `ODE45Model` extend base class  
✅ **Encapsulation** - Model state and methods bundled together  
✅ **Reusability** - Models can be instantiated multiple times  
✅ **Maintainability** - ~60% reduction in code duplication  

### Possible Extensions:

1. **Additional Integration Methods**
   - Implement RK4 (4th-order Runge-Kutta) as another subclass
   - Stochastic differential equation (SDE) solvers
   - Implicit methods for stiff systems

2. **Parameter Optimization**
   - Add `fitParameters()` method to estimate σ_x, σ_v from data
   - Bayesian parameter estimation
   - Cross-validation for model selection

3. **Active Inference**
   - Add motor actions to control sensory input
   - Extend models to minimize surprise through action
   - Implement action selection policies

4. **Real Sensory Data**
   - Load eye-tracking or mouse-tracking data
   - Fit models to empirical trajectories
   - Validate against human behavioral experiments

5. **Deeper Hierarchies**
   - Extend to 4+ levels
   - Test scalability of inference
   - Compare computational efficiency

6. **Alternative Priors**
   - Non-Gaussian priors (Laplacian, Student-t)
   - Mixture model priors
   - Time-varying context-dependent priors

7. **Clinical Applications**
   - Fit models to patient data
   - Predict behavioral phenotypes
   - Test intervention strategies

8. **Performance Optimization**
   - GPU acceleration for large-scale simulations
   - Parallel execution of multiple models
   - C++ MEX implementations for speed

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

### "Undefined function or variable 'EulerModel'"
**Solution:** Make sure all `.m` class files are in your MATLAB path. Navigate to the project directory first:
```matlab
cd 'c:\Users\srseb\OneDrive\School\FSU\Fall 2025\Symbolic Numeric Computation w Alan Lemmon\Project1'
```

### "Figures not appearing" (Remote Terminal)
**Solution:** If running via SSH/remote terminal, figures are automatically saved to `figures/` directory. Check there for PNG files instead of expecting display windows.

### "Out of memory"
**Solution:** Reduce simulation time (`T`) or increase time step (`dt`) when creating models:
```matlab
model = EulerModel(0.05, 5, 0.1, 1.0);  % Larger dt, shorter T
```

### "Results look different each run"
**Solution:** Sensory noise is random. Set random seed for reproducibility:
```matlab
rng(42);  % Fixed seed
model = EulerModel(0.01, 10, 0.1, 1.0);
model.generateSensoryInput(0.05);
```

### "Adaptation time = inf" or NaN
**Solution:** Prior too strong (σ_v too small) or threshold too strict. Adjust parameters or check the `printSummary()` output for details.

### "Class not found" after editing
**Solution:** MATLAB caches class definitions. Clear classes and reload:
```matlab
clear classes
step2_numerical_simulation  % Re-run
```

### "Figure directory not created"
**Solution:** The `run_all_experiments.m` script creates `figures/` automatically. If running steps individually, create it manually:
```matlab
if ~exist('figures', 'dir')
    mkdir('figures');
end
```

### Performance Issues
**Solution:** For faster execution:
- Use ODE45 with larger tolerances: `odeset('RelTol', 1e-3, 'AbsTol', 1e-5)`
- Reduce simulation duration: `T = 5` instead of `T = 10`
- Increase time step for Euler: `dt = 0.05` instead of `dt = 0.01`

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
