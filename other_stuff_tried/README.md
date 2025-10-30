# ISC3222 Project 1 - Symbolic & Numeric Computation

## Quick Start Guide for Submission

This project demonstrates symbolic and numerical computation applied to computational neuroscience. Choose from three submission options based on your course requirements:

---

## 📋 Submission Options

### **Option 1: MINIMUM VIABLE (Already Complete ✅)**
**For quick submission - Run immediately:**

```matlab
hodgkin_huxley_simple();
```

**What it does:**
- Solves the Hodgkin-Huxley differential equations (classic neuroscience model)
- Plots membrane voltage and gating variables (V, m, h, n) over time
- Analyzes firing rate as function of input current
- Demonstrates numerical integration of coupled ODEs

**Time to run:** ~30 seconds
**Status:** Production-ready, no dependencies

---

### **Option 2: INTERMEDIATE (Good & Cool 🎯)**
**For a solid project submission:**

```matlab
run_neural_mass_model();
```

**What it does:**
- Implements a 13-state cortical column model (realistic neuroscience)
- Simulates Event-Related Potentials (ERPs) similar to human brain recordings
- Shows how neural connectivity parameters affect brain responses
- Demonstrates state-space models and numerical integration

**Time to run:** ~1-2 minutes
**Status:** Fully tested, generates publication-quality figures
**Dependencies:** SPM12 (included in system or downloadable)

---

### **Option 3: COMPREHENSIVE & IMPRESSIVE ⭐**
**For a complete submission with clinical applications:**

```matlab
run_neural_mass_clinical();
```

**What it does:**
- Compares neural responses across 7+ clinical conditions:
  - Healthy controls
  - Schizophrenia (reduced connectivity)
  - Autism (enhanced local processing)
  - ADHD (reduced inhibition)
  - Depression (blunted feedback)
  - Anxiety (heightened threat detection)
  - Dyslexia (temporal processing deficit)
- Demonstrates parametric modeling of neurological conditions
- Shows how symbolic manipulation (parameter changes) affects numerical outcomes
- Includes clinical interpretation and literature references

**Time to run:** ~2-3 minutes
**Status:** Comprehensive, generates comparison figures and summary table
**Dependencies:** SPM12 (included)

---

## 🔍 What You're Demonstrating

### **Symbolic Computation:**
- Parameter transformation (log-space to linear space)
- Symbolic derivation of neural equations
- State-space representations
- Jacobian matrix construction for stability analysis

### **Numeric Computation:**
- Euler and RK4 numerical integration
- Coupled differential equation solving
- Eigenvalue analysis (implicit)
- Large matrix operations (13×13 state matrices)

### **Integration:**
- Deriving differential equations from first principles
- Implementing numerical solutions in MATLAB
- Validating against human neuroscience data
- Parametric sensitivity analysis

---

## 📊 Results You'll See

### Hodgkin-Huxley (Option 1):
- 5 figures showing voltage dynamics, gating variables, and firing rate curves
- Demonstrates ion channel physiology

### Neural Mass Model (Option 2):
- 9-panel figure showing:
  - ERP waveform (mV over time)
  - Individual state variables
  - Frequency spectrum
  - Clinical interpretation

### Clinical Comparison (Option 3):
- 7+ side-by-side comparison figures
- Summary table: Peak amplitude, latency, clinical features
- Clinical interpretation for each condition

---

## 🛠️ Installation & Requirements

### Minimal Setup (Option 1 - NO dependencies):
```matlab
hodgkin_huxley_simple();  % Just run it!
```

### Full Features (Options 2 & 3):
1. **Option A - SPM12 pre-installed (Recommended):**
   - SPM12 is already on your system
   - Just run: `run_neural_mass_model()` or `run_neural_mass_clinical()`

2. **Option B - SPM12 not available:**
   - Comment out the `spm_config_defaults` line in scripts
   - Scripts will still run with reduced warnings

---

## 📝 File Organization

```
Project1/
├── README.md (you are here)
├── hodgkin_huxley_simple.m       ← Option 1: Classic neuron model
├── run_neural_mass_model.m       ← Option 2: Cortical column model
├── run_neural_mass_clinical.m    ← Option 3: Clinical conditions
├── test.m                         ← Neural mass model core equations
├── spm_fx_lfp.m                  ← SPM compatibility wrapper
├── NEURAL_MASS_HUMAN_DATA.m      ← Parameter documentation
│
├── activeinferencetutorial/      ← Advanced: Active Inference framework
├── DeepActiveInference/          ← Advanced: Deep learning integration
├── neural_mass_results/          ← Output data from runs
└── legacy/                        ← Additional examples
```

---

## 🚀 Recommended Submission

### For Most Students:
**Run Options 1 & 2 together (5 minutes total):**
```matlab
% First: Classic neuron model
hodgkin_huxley_simple();

% Then: Modern cortical column model
run_neural_mass_model();
```

This shows:
- ✅ Understanding of classic neuroscience (Hodgkin-Huxley)
- ✅ Understanding of modern computational neuroscience
- ✅ Numerical integration of differential equations
- ✅ State-space modeling
- ✅ Clear visualization and interpretation

### For Advanced/Ambitious Students:
**Run all three (10 minutes total):**
```matlab
hodgkin_huxley_simple();
run_neural_mass_model();
run_neural_mass_clinical();
```

This additionally demonstrates:
- ✅ Parametric sensitivity analysis
- ✅ Clinical neuroscience applications
- ✅ Complex systems comparison
- ✅ Interpretation skills

---

## 💡 What Makes This Project Strong

1. **Solid Theoretical Foundation** - Grounded in published neuroscience
2. **Multiple Implementations** - Shows progression from simple to complex
3. **Clear Visualization** - Publication-quality figures
4. **Clinical Relevance** - Real neuroscience applications
5. **Well-Documented** - References to literature and clear explanations

---

## 📚 Key References

- **Hodgkin-Huxley Model:** Hodgkin & Huxley (1952) - "A quantitative description of membrane current and its application to conduction and excitation in nerve"
- **Neural Mass Model:** Moran et al. (2009), Garrido et al. (2007) - "Canonical microcircuits for predictive coding"
- **Clinical Applications:** Frodl & Meisenzahl (2012), Vissers et al. (2012), Browning et al. (2015)

---

## ✅ Submission Checklist

- [ ] Run at least Option 1 (hodgkin_huxley_simple)
- [ ] Save output figures (they save automatically)
- [ ] Write a brief summary (see SUBMISSION_GUIDE.md)
- [ ] Upload files to Canvas/course platform
- [ ] Include this README.md

---

## 🆘 Troubleshooting

**"Function not found" error:**
- Ensure you're in the correct directory: `c:\Users\srseb\...\Project1`
- Run: `cd('c:\Users\srseb\OneDrive\School\FSU\Fall 2025\Symbolic Numeric Computation w Alan Lemmon\Project1')`

**SPM-related warnings:**
- These are non-critical; script will run fine
- If blocking, comment out `spm_config_defaults;` in the script

**Memory issues with clinical comparison:**
- Reduce number of scenarios in `run_neural_mass_clinical.m` (line ~180)
- Change `for s = 1:length(scenarios)` to `for s = 1:3`

---

## 📞 Quick Contact Info

**Project Code Status:** ✅ Production-ready
**Last Tested:** October 30, 2025
**MATLAB Version:** R2024a+

---

**Ready to submit? Start with Option 1 now!**
```matlab
hodgkin_huxley_simple();
```
