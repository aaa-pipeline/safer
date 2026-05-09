# Static EXP102 SAFER Pressure Interface Calculator

This is the pure static browser calculator for the current EXP102 SAFER results
used in `outputs_experiments_20260507/exp102_selected_three_safe_safer`.

Open:

```text
index.html
```

No Python process, cloud server, or local web server is required. The page loads
`model-data.js` directly in the browser.

## What It Does

The current export contains both:

- browser-side EXP102 model exports for arbitrary pipe/defect inputs
- the 84 locked EXP102 test specimens for exact paper-result reproduction

The specimen dropdown is only a shortcut for reproducing a test row. Users may
also type new values for `D`, `t`, `d`, `L`, `w`, `sigma_y`, and `sigma_u`.

## EXP102 Basis

- Selected good-but-diverse residual backbones: ET, HGB, and XGBoost
- SAFER policy: `SAFER_selective_anchor_cap_val0_cov_eps1`
- `P_SAFER_final`: max of the three SAFER capacities
- Held-out EXP102 test unsafe count: `0/84` for each SAFER backbone and
  `0/84` for `P_SAFER_final`

For a new typed input, `0/84` remains the held-out test-set statistic for the
model family. It is not a claim that a future arbitrary point has been physically
tested.

## Files

```text
index.html            Static UI
style.css             Page styling
app.js                Browser-side formulas and EXP102 inference logic
model-data.js         EXP102 test rows plus browser-side model export
export_exp102_data.py EXP102 model/data exporter
export_model_data.py  Legacy arbitrary-input exporter
```

## Refresh Model Data

If EXP102 output tables or model code are updated, regenerate the web model file:

```bash
python web_calculator/export_exp102_data.py
```

For a small lookup-only export, use:

```bash
python web_calculator/export_exp102_data.py --lookup-only
```

## Current Output

The page returns:

- DNV-RP-F101, ASME B31G, Modified B31G, PCORRC, Modified PCORRC
- raw ET/HGB/XGBoost residual predictions
- SAFER-ET, SAFER-HGB, and SAFER-XGBoost capacities
- `P_SAFER_final = max(SAFER-ET, SAFER-HGB, SAFER-XGBoost)`
- `P_allow = eta * P_SAFER_final`
- `P_design = 2 * sigma_y * t / D * F`
- `P_interface = min(P_allow, P_design)`
- optional operating-pressure screening when `P_op` is entered

The results table separates held-out test-set unsafe counts from MAPE. For
example:

```text
SAFER-ET (MAPE 13.15%) | EXP102 test unsafe 0/84 (0.0%)
```

The default screening setting is `eta = 0.72` and `F = 0.72`. `P_interface` is a
capacity-side pressure input for engineering screening; it is not, by itself, an
approved operating pressure or a repair decision.
