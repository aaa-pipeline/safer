# Static Pressure Calculator

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
- Operating-pressure margin and safety ratio for the user-specified `P_op`

Each method name includes held-out test-set unsafe rate and MAPE, for example:

```text
SAFER-Cat (unsafe 4.17%, MAPE 13.94%)
```
