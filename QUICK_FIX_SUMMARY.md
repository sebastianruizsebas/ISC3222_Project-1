# QUICK FIX SUMMARY (Nov 2, 2025)

## Three Critical Bugs Making PSO Ineffective

### BUG #1: Hardcoded alpha_precision_gain (Line ~500 in hierarchical_step_update.m)
```matlab
% WRONG:
alpha_precision_gain = 0.5;  % Hardcoded! PSO value ignored

% RIGHT:
alpha_precision_gain = P.alpha_precision_gain;  % Use PSO parameter
```

### BUG #2: Hardcoded Precision Bounds (Lines ~525-530)
```matlab
% WRONG:
S.pi_L1_motor = max(10, min(500, S.pi_L1_motor));  % Hardcoded!

% RIGHT:
S.pi_L1_motor = max(P.pi_bounds.L1_motor(1), min(P.pi_bounds.L1_motor(2), S.pi_L1_motor));
```

### BUG #3: Triple Precision Update (Lines ~515-530 + ~650-700)
Mechanism 1 (update_pi) + Mechanism 2 (hardcoded exp) + Mechanism 3 (correct exp) all running!

**FIX:** Delete lines ~515-530. Keep only the error-driven section (lines ~650-700).

---

## How These Bugs Block PSO

| Parameter | Intended Range | What PSO Sends | What Code Uses | Result |
|-----------|----------------|---|---|---|
| `alpha_precision_gain` | [0.1, 2.0] | ✓ Varies | ❌ Always 0.5 | PSO ineffective |
| `pi_L1_motor_min` | [5, 50] | ✓ Varies | ❌ Always 10 | PSO ineffective |
| `pi_L1_motor_max` | [200, 1000] | ✓ Varies | ❌ Always 500 | PSO ineffective |
| All 9 precision params | [various] | ✓ Varies | ❌ Hardcoded | PSO ineffective |

---

## Additional Issue

**Missing from P struct:** `interference_penalty_weight` (line ~715 in main script)
- Add: `P.interference_penalty_weight = params.interference_penalty_weight;`

---

## Files Affected

- `hierarchical_step_update.m` — Lines ~500-530, ~650-700 (remove one, fix other)
- `hierarchical_motion_inference_dual_hierarchy.m` — Line ~715 (add one line)

---

## Time to Fix: 5-10 minutes per file

See `CRITICAL_CODE_REVIEW_NOV_2.md` for detailed analysis.
