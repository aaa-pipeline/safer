# Static SAFER Pressure Interface Calculator

This is the pure static web version of the local pressure calculator.

Open:

```text
index.html
```

No Python process, cloud server, or local web server is required. The page loads
`model-data.js` directly in the browser.

## Files

```text
index.html          Static UI
style.css           Page styling
app.js              Browser-side formulas and model inference
model-data.js       Exported model data
export_model_data.py
```

## Refresh Model Data

If `local_calculator/artifacts` is updated, regenerate the web model file:

```bash
python web_calculator/export_model_data.py
```

## Current Output

The page returns:

- DNV-RP-F101, ASME B31G, Modified B31G, PCORRC, Modified PCORRC
- Top-3 residual models
- Top-3 SAFER models
- `P_SAFER_final`, defined as the maximum of the available SAFER capacity estimates
- `P_allow = eta * P_SAFER_final`
- `P_design = 2 * sigma_y * t / D * F`
- `P_interface = min(P_allow, P_design)`
- Optional operating-pressure screening when `P_op` is entered
- Defect severity and model-explanation indicators: `d/t`, `L/sqrt(Dt)`, `w/(piD)`, SAFER adjustment, model spread, and selected `p_unsafe`

Each method name includes held-out test-set unsafe rate and MAPE, for example:

```text
SAFER-Cat (unsafe 4.17%, MAPE 13.94%)
```

The default screening setting is `eta = 0.72` and `F = 0.72`. Users can choose more
conservative presets or enter custom factors. `P_interface` is a capacity-side
pressure input for engineering screening; it is not, by itself, an approved operating
pressure or a repair decision.
