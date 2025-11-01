## Model → Cortical Microcircuit Mapping

This document maps the main variables and structures in the Project1 dual‑hierarchy predictive‑coding model to plausible cortical microcircuit substrates and proposes physiological experiments to validate those mappings. The goal is pragmatic: give clear, testable predictions (laminar signatures, perturbations, and expected effects) so experimentalists can design validation protocols.

Notes on scope and tone
- These mappings are functional correspondences (computational roles) rather than one‑to‑one anatomical assertions. They are motivated by predictive‑coding accounts (Rao & Ballard, 1999), canonical microcircuit / laminar models (Bastos et al., 2012; Friston 2005), and more recent laminar physiology reviews.
- Suggested experiments use laminar probes (CSD/LFP/multiunit), causal perturbations (optogenetic/TMS/pharmacology), and behavioral manipulations (expectation / variance) to test specific model predictions.

### Short contract (what these mappings assume)
- Inputs: sensory/proprioceptive observations (here: ball and proprioceptive position/velocity). 
- Outputs: motor commands / reach velocity (implemented by motor L1–L3 representations and readouts).
- Error channels: prediction errors at L1/L2 (E_L1_*, E_L2_* in the code) that drive updates in representations and weights.

---

## High-level mappings (modules)

- `R_L1_motor` (proprioceptive L1): primary somatosensory / proprioceptive superficial representation (sensory observations). Expect strong sensory responses in superficial layers (L2/3), fast gamma‑band modulation for surprising inputs.
- `R_L2_motor` / `R_L3_motor` (motor intermediate / output): premotor / motor cortical populations implementing policy / motor basis functions. Deep layers (L5/6) and infragranular populations are candidates for holding sustained motor commands and projecting to brainstem/spinal cord.
- `R_L1_plan` (planning L1): target/goal representation (could map to parietal or prefrontal goal representations feeding into premotor areas). Top‑down predictions from planning L3 should arrive in deeper layers of motor cortex.
- `R_L2_plan` / `R_L3_plan`: policy/predictive dynamics in prefrontal / premotor circuits used to generate expected target motion or intercept plans.
- `W_*` matrices: synaptic weight matrices implementing learned mappings between layers. Changes in these correspond to synaptic plasticity measurable with long‑latency potentiation and sensitive to NMDA‑dependent blockade.
- `pi_*` (precision traces): gain/precision signals — hypothesised to map to neuromodulatory gain control (acetylcholine, noradrenaline) and laminar modulation of inhibitory/excitatory balance. Precision dynamics should alter LFP spectral content (beta/gamma balance).

---

## Variable-by-variable mapping and experiment suggestions

Each entry lists: variable(s) → computational role → plausible cortical substrate → testable physiological predictions / experiments.

1) `params.eta_rep` (representation learning rate)
- Role: controls rate at which internal representations `R_*` are updated from prediction errors.
- Substrate mapping: synaptic plasticity in superficial → deep connections (short‑term plastic changes), possibly NMDA‑dependent in L2/3→L5.
- Experiments:
  - Pharmacological: apply NMDA antagonists locally and observe whether rapid trial‑by‑trial adaptation of neuronal responses and anticipatory signals is reduced.
  - Behavioral: repeated perturbation schedule; with reduced eta_rep expect slower adaptation in neural anticipatory signals.

2) `params.eta_W` (weight learning rate)
- Role: controls long‑term synaptic weight updates (`W_*`).
- Substrate mapping: long‑term synaptic plasticity in intracortical pathways (L2/3 horizontal connections, L5→L2 feedback).
- Experiments:
  - Multi‑session training to intercept novel dynamics while measuring evoked potentials; blocking LTP/LTD mechanisms should impair long‑term performance gains.

3) `params.momentum`, `params.weight_decay`
- Role: algorithmic stabilizers for updates (equivalents of synaptic homeostasis/short‑term dynamics).
- Substrate mapping: synaptic scaling, short‑term plasticity, neuromodulator‑mediated stability.
- Experiments:
  - Manipulate neuromodulators (ACh/NE) and measure changes in adaptation stability and learning convergence.

4) Physics params (`P.gravity`, `P.restitution`, `P.ground_friction`, `P.air_drag`, `P.ground_z`)
- Role: parameters of the generative model for object motion used by the planning layer.
- Substrate mapping: internal model in parietal/premotor circuits representing environment dynamics.
- Experiments:
  - Change external dynamics (e.g., make bounce more/less elastic) and measure error signals and how quickly planning representations (`R_L2_plan`, `R_L3_plan`) update.

5) `R_L1_motor`, `pred_L1_motor`, `E_L1_motor`
- Role: proprioceptive observation, prediction, and prediction error at L1.
- Substrate mapping: superficial layers (L2/3) carry prediction errors (gamma), deep layers carry predictions (beta/alpha).
- Experiments:
  - Laminar recording in S1/M1 during unexpected limb perturbations — expect transient gamma increases in superficial layers correlated with `E_L1_motor` amplitude.

6) `R_L2_motor`, `R_L3_motor`
- Role: motor policy primitives and sustained motor outputs.
- Substrate mapping: premotor and motor cortex deep layers (L5) for outgoing motor drive; L2/3 for intermediate policy variables.
- Experiments:
  - Deep‑layer neuronal firing should predict upcoming velocity components (correlate with `S.vx_player`, `S.vy_player`). Transient deep layer inhibition should disrupt accurate velocity generation.

7) `R_L1_plan`, `pred_L1_plan`, `E_L1_plan`
- Role: representation of external object (ball) and planning prediction/errors.
- Substrate mapping: parietal / frontal regions, superficial layers for prediction errors about object motion.
- Experiments:
  - Occlusion paradigms (briefly hide ball): measure whether planning representations maintain expectancies (sustained activity) and whether reappearance causes superficial gamma bursts tied to `E_L1_plan`.

8) Weight matrices (`W_*`)
- Role: learned mappings implementing predictions and mappings between layers.
- Substrate mapping: interlaminar and intracolumnar synaptic pathways.
- Experiments:
  - Stimulate putative source layer and measure evoked responses in target layers over training to estimate plastic change in effective connectivity.

9) Precision traces (`pi_L1_*`, `pi_L2_*`)
- Role: estimated channel reliabilities used to scale prediction errors.
- Substrate mapping: neuromodulatory systems (ACh, NE), local interneuron gain control.
- Experiments:
  - Change sensory uncertainty (add jitter to ball position) and measure whether neuromodulatory proxies (pupil, LFP spectral shifts) and laminar gain signatures change as predicted by the model's `pi_*` dynamics.
  - Pharmacological manipulation of cholinergic signalling should alter the observed precision‑dependent reweighting.

10) State traces (`S.x_player`, `S.vx_player`, `S.x_ball`, `S.vx_ball`)
- Role: observable kinematics used to validate internal states.
- Experiments:
  - Correlate neural population trajectories (dimensionality reduction of laminar population activity) with modeled state trajectories to assess alignment.

11) `free_energy_all`, `interception_error_all`
- Role: global objective and instantaneous error signals.
- Substrate mapping: distributed performance monitoring networks (ACC/dmPFC, neuromodulatory arousal systems).
- Experiments:
  - Trial‑by‑trial increases in interception error should correlate with ACC activation and phasic pupil dilation (LC/NE activity proxy).

---

## Protocol suggestions (to test multiple hypotheses)

1) Moving ball interception with occasional perturbations
- Setup: baseline predictable trajectories, occasional deviations (trajectory perturbation, occlusion, increased noise).
- Recording: laminar probes in motor + parietal/premotor cortex, motion capture, pupilometry.
- Predictions: perturbations → superficial gamma bursts (E_L1), then beta‑band changes as predictions are updated; precision manipulations change gamma/beta balance and learning rate.

2) Causal laminar perturbations
- Temporally targeted silencing of superficial layers during surprise should reduce gamma and slow behavioral correction; silencing deep layers near reach onset should disrupt motor output without removing sensory error signals.

3) Plasticity tests (multi‑session)
- Train on altered dynamics (e.g., different restitution) and measure changes in effective interlaminar connectivity; block NMDA to test reliance on NMDA mechanisms.

---

## Analysis recipes / metrics
- Laminar CSD to separate superficial vs deep sources; compare gamma (30–80Hz) and beta (12–30Hz) aligned to prediction error and reach onset.
- Spike‑field coherence: superficial neurons should align to gamma during errors; deep neurons to beta during strong top‑down predictions.
- Directed coherence / Granger causality (frequency resolved) to test bottom‑up gamma (errors) vs top‑down beta (predictions) flows.

## Recommended readings
- Rao & Ballard (1999). Predictive coding in the visual cortex: a functional interpretation of some extra‑classical receptive‑field effects.
- Bastos et al. (2012). Canonical microcircuits for predictive coding. Neuron.
- Reviews on laminar predictive coding and precision (Auksztulewicz & Friston and related literature).

---

## UPDATE (Nov 1, 2025): Task-Conditional Multiplicative Gating & Interference Prevention

### Task Context (L0) → Prefrontal & Neuromodulatory Gating

**Computational role:** `R_L0(i, task)` is a one-hot encoding of the current task (trial). Each task activates a separate set of learned weights and gates predictions multiplicatively.

**Cortical substrate (Prefrontal Cortex → M1/PMd):**
- **dlPFC/dACC** (dorsolateral prefrontal / anterior cingulate cortex): encode task rules and context representations (Dewan et al., 2020; Frässle et al., 2015)
- **Neuromodulatory systems** (cholinergic, noradrenergic, dopaminergic): implement the multiplicative gain gating
  - **ACh** (acetylcholine from basal forebrain): enhances signal-to-noise in attended/relevant pathways
  - **NE** (noradrenaline from locus coeruleus): increases arousal and gain during task uncertainty/switches
  - **DA** (dopamine from SNc/VTA): signals task relevance and reward prediction

**Mechanism (Mante et al., 2013; Rigotti et al., 2013):**
```
Task context R_L0 (from PFC/ACC)
     ↓ (via acetylcholine / inhibitory interneurons)
Modulates excitatory:inhibitory balance in M1/PMd
     ↓ (multiplicative modulation of activity)
Motor predictions = task_gate * (population activity * weights)
```

**Testable prediction:**
- Lesion/inactivate ACC → loss of task-specificity (predictions same across tasks)
- Optogenetic enhancement of PFC→M1 projections during task-switch → faster prediction re-tuning
- Cholinergic antagonists (atropine) → loss of multiplicative gating (linear gating instead)

---

### Weight Indexing by Task → Distributed Task Memory

**Computational structure:** 
```matlab
W_motor{task_1}, W_motor{task_2}, ..., W_motor{task_N}  % N separate learned maps
```

**Cortical substrate (Population Coding & Context-Dependent Remapping):**
- **M1/PMd population codes rotate/rescale per task** (Kaufman et al., 2016; *Nature*)
  - Same neurons encode velocity, but with different gain/scaling per context
  - Covariance structure of population activity is context-dependent
  
- **Specific circuit candidate: M1 Layer 2/3 horizontal connections**
  - Sparse, recurrent connectivity that learns task-specific input-output mappings
  - Gated by feedback from M1 Layer 5 (which receives PFC input)
  
- **Alternative: Parallel pathways via basal ganglia**
  - Direct pathway (D1 neurons) → facilitates current task's motor plan
  - Indirect pathway (D2 neurons) → suppresses off-task plans
  - Creates functional "multiplexing" of task-specific weights

**Evidence:**
- fMRI/population recordings in PFC and M1 show different neural activity patterns per task (even for same stimuli)
- Plasticity within a task vs. interference across tasks controlled by task-selective feedback
- Lesions of M1 horizontal connections → increased cross-task interference (Sanes & Donoghue, 2000)

**Testable predictions:**
- **Structure:** Record simultaneously from M1 and PFC; compute task-dependent alignment (CCA) of population codes. Should see rotation/rescaling per task.
- **Causality:** Optogenetically perturb M1 Layer 5 feedback during task A → should disrupt task A learning without affecting task B weights
- **Plasticity:** Multi-session training on task A then task B; measure spine enlargement (∆F/%) selectively on layer 2/3 neurons coding task B (not task A)

---

### Task-Selective Learning (Only Active Task Updates)

**Computational principle:** Weight updates only occur for the currently active task. Off-task weights remain frozen.

```matlab
% Active task updates normally
W_motor{current_task} += dW

% Off-task weights DO NOT update (frozen)
W_motor{other_task} remains unchanged
```

**Neurobiological mechanism (Eligibility Traces + Dopamine Gating):**
- **Synaptic tagging**: Synapses active during recent presynaptic/postsynaptic coincidence are "tagged" (stc-protein deposition)
- **Consolidation gate**: LTP/LTD only occurs when specific task's dopamine signal is high
  - dopamine ↑ when task is active and reward/error is present
  - dopamine ↓ when task is off-task or inactive
  
- **Molecular specificity** (Lisman et al., 2002):
  - Each synapse stores AMPAR, NMDAR, CaMKII state (local to that synapse)
  - Plasticity requires both tagged synapse AND permissive dopamine signal
  - Off-task synapses remain tagged but do NOT undergo LTP/LTD without dopamine

**Predicted circuit (Dopamine + Credit Assignment):**
```
PFC/ACC → encodes task identity (context)
  ↓
SNc/VTA → receives task signal
  ↓
Dopamine release → proportional to task-relevance and reward
  ↓
M1 synapses → LTP/LTD only when dopamine is high (for active task)
```

**Empirical evidence:**
- Dopamine release is task-selective (Roy et al., 2022; phasic vs tonic DA)
- Lesion of dopaminergic neurons → severe impairment in multi-task learning (Parkinson's disease, ADHD)
- Direct dopamine manipulation: enhancing DA during task A → accelerates task A learning but NOT task B

**Testable predictions:**
1. **Dopamine recordings + multi-task learning:**
   - Record dopamine (fast-scan cyclic voltammetry, FSCV) during task switching
   - Expect phasic DA increase at task onset (for active task)
   - Off-task dopamine remains baseline even with off-task prediction errors

2. **Pharmacological:** 
   - Broad DA agonist (e.g., bromocriptine) applied globally → all tasks learn simultaneously → catastrophic interference (no task selectivity)
   - Task-specific dopamine manipulation (optogenetics + receptor antagonists) → selectively enhance learning in one task

3. **Synaptic tagging experiments** (in vitro slice):
   - Prime task A neurons with weakly-pairing protocols (tag without LTP)
   - Then apply dopamine agonist only to task A synapses
   - Expect selective LTP in task A (tagged) but not task B (untagged) pathways

---

### Cross-Task Error Computation & Interference Monitoring

**Computational role:** `task_errors_motor(i, t)` computes prediction error of task `t` at step `i`, even if `t` is not active.

**Biological correlate (Simultaneous Multi-Task Representation):**
- **PFC & PMd populations maintain representations of ALL tasks** simultaneously
  - Even off-task neurons continue to encode task-specific hypotheses
  - This is "null space" activity (Remington et al., 2016; *Neuron*)
  
- **Purpose of off-task representations:**
  1. Readiness to switch (if task changes)
  2. Context-dependent remapping (preparing alternative plans)
  3. Error monitoring / conflict detection (ACC monitors task-irrelevant prediction errors)

**Neural implementation:**
```
ACC neurons → compute error signals for ALL tasks (multiplex)
     ↓
Anterior insula → integrates multi-task conflict / surprise
     ↓
Adjusts PFC gain → down-weights off-task errors, up-weights task-relevant errors
```

**Testable predictions:**
1. **fMRI/multiunit recording in ACC:**
   - Decode task-specific prediction errors even for off-task
   - Expect activity correlating with error magnitude for all tasks (not just active)
   - Off-task error signal smaller or slower than on-task (due to weaker top-down attention)

2. **Inactivation of ACC → loss of cross-task error monitoring:**
   - Lesion ACC → animals freeze during conflict or show passive behavior
   - Cannot flexibly switch tasks because off-task signals are not monitored

---

### Optional Interference Penalty in Free Energy

**Mathematical formulation:**
```
F_total(i) = F_task_specific(i) + interference_penalty_weight * sum_off_tasks error^2
```

**Interpretation:**
- Penalizes off-task models from becoming good predictors of current-task data
- Encourages functional specialization (task1's model ≠ task2's model)
- Equivalent to pushing population codes to be orthogonal across tasks

**Neural substrate (Sparse Coding & Inhibitory Control):**
- **VIP+ (Vasoactive Intestinal Peptide) interneurons in PMd:**
  - Encode task identity strongly
  - Inhibit neurons that would activate off-task motor programs
  - Result: off-task models are suppressed (lower activity → smaller plasticity)

- **Cerebellar Purkinje cell inhibition:**
  - Off-task cerebellar modules are inhibited by task-specific gating
  - Prevents crosstalk between task-specific motor modules

**Experimental test:**
- Optogenetically activate off-task cerebellar module during task A → should see:
  1. Increased off-task activity (short term)
  2. Decreased task A performance (interference)
  3. Slow re-suppression as inhibitory gating re-engages

---

### Summary Table: Task-Conditional Learning Circuit

| Computational Variable | Cortical Structure | Neurotransmitter | Experimental Test |
|---|---|---|---|
| `R_L0` (task context) | dPFC, ACC, dlPFC | (state repr.) | fMRI: task decoding; ACC inactivation → loss of context |
| Task gating (multiplicative) | PFC→M1 pathway | ACh, NE (gain modulation) | Atropine → linear instead of multiplicative; optogenetics of PFC→M1 |
| Task-indexed weights `W{t}` | M1 pop. remapping, BG direct/indirect | DA (conditional gating) | Multi-electrode array: CCA per task; dopamine lesion → interference |
| Task-selective updates | DA gating of synaptic tag | DA (consolidation signal) | FSCV dopamine during task switch; DA agonist → non-selective learning |
| Cross-task errors | ACC multiplex, insula conflict | ? | ACC lesion → loss of off-task error; insula fMRI shows all-task errors |
| Interference penalty | VIP+ inhibition, cerebellar gating | GABA (inhibition) | Optogenetics of VIP+ during off-task → increased interference |

---

## Closing notes (Updated)

- Task-conditional learning is deeply embedded in prefrontal-motor circuitry and neuromodulatory systems
- The computational model reveals that **gating**, **indexing**, and **selective plasticity** are key to multi-task learning
- These predictions are testable with current neurotechnology (optogenetics, FSCV, multi-electrode arrays, fMRI)
- Future work: integrate this circuit model with rodent/primate behavioral data and electrophysiology


