(function () {
  "use strict";

  const EPS = 1e-8;
  const DEFAULTS = {
    D: 762,
    t: 10,
    d: 4,
    L: 120,
    w: 60,
    sigma_y: 450,
    sigma_u: 535,
    eta: 0.72,
    F: 0.72,
    P_op: 8,
  };

  const FACTOR_PRESETS = {
    reference: { eta: 0.72, F: 0.72 },
    conservative60: { eta: 0.60, F: 0.60 },
    liquid54: { eta: 0.54, F: 0.54 },
    class50: { eta: 0.50, F: 0.50 },
    class40: { eta: 0.40, F: 0.40 },
    dnvReference: { eta: 0.648, F: 0.72 },
  };

  function calculate(input, modelData) {
    const data = modelData || globalThis.MODEL_DATA;
    if (!data) {
      throw new Error("Model data is not loaded.");
    }
    const clean = validateInput(input);
    const row = addFeatures(addBaselines(clean));
    const metadata = data.metadata;
    const featureColumns = metadata.feature_columns;
    const x = featureColumns.map((name) => row[name]);

    const results = [];
    const formulaColumns = [
      ["DNV-RP-F101", "P_DNV"],
      ["ASME B31G", "P_ASME_B31G"],
      ["Modified B31G", "P_Mod_B31G"],
      ["PCORRC", "P_PCORRC"],
      ["Modified PCORRC", "P_Mod_PCORRC"],
    ];

    for (const [label, column] of formulaColumns) {
      const metrics = metadata.formula_metrics[label] || {};
      results.push({
        category: "Standard",
        method: label,
        display_method: displayMethod(label, metrics.unsafe_rate, metrics.test_mape),
        prediction: row[column],
        test_mape: metrics.test_mape,
        test_unsafe_rate: metrics.unsafe_rate,
        p_unsafe: null,
        margin: null,
        op_margin: opMargin(row[column], clean.P_op),
        safety_ratio: capacityToOpRatio(row[column], clean.P_op),
      });
    }

    for (const item of metadata.residual_top3) {
      const model = data.residual_models[item.artifact_name];
      const prediction = predictResidual(model, x, row.P_DNV);
      results.push({
        category: "Residual",
        method: item.label,
        display_method: displayMethod(item.label, item.unsafe_rate, item.test_mape),
        prediction,
        test_mape: item.test_mape,
        test_unsafe_rate: item.unsafe_rate,
        p_unsafe: null,
        margin: null,
        op_margin: opMargin(prediction, clean.P_op),
        safety_ratio: capacityToOpRatio(prediction, clean.P_op),
      });
    }

    for (const item of metadata.safer_top3) {
      const residualModel = data.residual_models[item.artifact_name];
      const safer = data.safer_models[item.artifact_name];
      const pRaw = predictResidual(residualModel, x, row.P_DNV);
      const riskX = x.concat([pRaw, pRaw / Math.max(row.P_DNV, EPS)]);
      const pUnsafe = predictModel(safer.risk_classifier, riskX);
      const sigma = Math.max(predictModel(safer.error_scale_model, riskX), metadata.eps || EPS);
      const params = safer.safety_params;
      const dynamic = dynamicMargin(
        pUnsafe,
        sigma,
        Number(params.tau),
        Number(params.gamma),
        Number(params.lambda),
        Number(params.q),
        Number(params.risk_weight_floor || 0)
      );
      const margin = Number(params.fixed_floor || 0) + dynamic;
      const prediction = pRaw - margin;
      results.push({
        category: "SAFER",
        method: item.label,
        display_method: displayMethod(item.label, item.unsafe_rate, item.test_mape),
        prediction,
        test_mape: item.test_mape,
        test_unsafe_rate: item.unsafe_rate,
        p_unsafe: pUnsafe,
        sigma_error: sigma,
        p_raw: pRaw,
        margin,
        op_margin: opMargin(prediction, clean.P_op),
        safety_ratio: capacityToOpRatio(prediction, clean.P_op),
      });
    }

    return buildReport(clean, row, results);
  }

  function buildReport(input, featureRow, rows) {
    const saferRows = rows.filter((row) => row.category === "SAFER");
    if (!saferRows.length) {
      throw new Error("No SAFER models are available in model-data.js.");
    }
    const finalSafer = saferRows.reduce((best, row) => (row.prediction > best.prediction ? row : best), saferRows[0]);
    const pSaferFinal = finalSafer.prediction;
    const pAllow = input.eta * pSaferFinal;
    const pDesign = ((2 * input.sigma_y * input.t) / Math.max(input.D, EPS)) * input.F;
    const pInterface = Math.min(pAllow, pDesign);
    const governing = pAllow <= pDesign ? "SAFER capacity" : "Design envelope";
    const opRatio = input.P_op === null ? null : input.P_op / Math.max(pInterface, EPS);
    const interfaceMargin = input.P_op === null ? null : pInterface - input.P_op;
    const decision =
      input.P_op === null
        ? {
            status: "incomplete",
            label: "Interface only",
            note: "Enter P_op to run the operating pressure check.",
          }
        : input.P_op <= pInterface
          ? {
              status: "pass",
              label: "Pass screening",
              note: `P_op is ${formatNumber(opRatio, 3)} of P_interface.`,
            }
          : {
              status: "review",
              label: "Review required",
              note: `P_op exceeds P_interface by ${formatNumber(-interfaceMargin, 3)} MPa.`,
            };
    const saferPredictions = saferRows.map((row) => row.prediction);
    const residualPredictions = rows.filter((row) => row.category === "Residual").map((row) => row.prediction);
    return {
      input,
      rows,
      summary: {
        p_safer_final: pSaferFinal,
        final_method: finalSafer.method,
        p_interface: pInterface,
        decision,
      },
      interface: {
        p_allow: pAllow,
        p_design: pDesign,
        p_interface: pInterface,
        governing,
        eta: input.eta,
        F: input.F,
        p_op: input.P_op,
        p_op_over_interface: opRatio,
        interface_margin: interfaceMargin,
      },
      explanation: {
        d_over_t: featureRow.d_over_t,
        L_over_sqrtDt: featureRow.L_over_sqrtDt,
        w_over_piD: featureRow.w_over_piD,
        safer_adjustment: finalSafer.margin,
        final_p_unsafe: finalSafer.p_unsafe,
        final_sigma_error: finalSafer.sigma_error,
        safer_spread: maxMinusMin(saferPredictions),
        residual_spread: maxMinusMin(residualPredictions),
        risk_level: riskLevel(finalSafer.p_unsafe),
      },
    };
  }

  function opMargin(prediction, pOp) {
    return pOp === null ? null : prediction - pOp;
  }

  function capacityToOpRatio(prediction, pOp) {
    return pOp === null ? null : prediction / Math.max(pOp, EPS);
  }

  function maxMinusMin(values) {
    const clean = values.filter((value) => Number.isFinite(Number(value))).map(Number);
    if (!clean.length) {
      return null;
    }
    return Math.max(...clean) - Math.min(...clean);
  }

  function riskLevel(pUnsafe) {
    if (!Number.isFinite(Number(pUnsafe))) {
      return "";
    }
    if (pUnsafe >= 0.66) {
      return "High";
    }
    if (pUnsafe >= 0.33) {
      return "Medium";
    }
    return "Low";
  }

  function validateInput(input) {
    const out = {};
    const requiredKeys = ["D", "t", "d", "L", "w", "sigma_y", "sigma_u", "eta", "F"];
    for (const key of requiredKeys) {
      const rawValue = input[key] === undefined ? DEFAULTS[key] : input[key];
      const value = Number(rawValue);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`${key} must be a positive number.`);
      }
      out[key] = value;
    }
    const rawPOp = input.P_op;
    if (rawPOp === undefined || rawPOp === null || String(rawPOp).trim() === "") {
      out.P_op = null;
    } else {
      const value = Number(rawPOp);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error("P_op must be a positive number or left blank.");
      }
      out.P_op = value;
    }
    if (out.d >= out.t) {
      throw new Error("d must be smaller than t.");
    }
    if (out.sigma_u < out.sigma_y) {
      throw new Error("sigma_u must be greater than or equal to sigma_y.");
    }
    return out;
  }

  function addBaselines(row) {
    const out = { ...row };
    const { D, t, d, L, sigma_u, sigma_y } = out;
    const sqrtDt = Math.sqrt(Math.max(D * t, 1e-12));
    const q = Math.sqrt(1 + 0.31 * Math.pow(L / sqrtDt, 2));
    const denom = 1 - d / Math.max(t * q, 1e-12);
    const geom = (1 - d / t) / Math.max(denom, 1e-12);
    const intact = (2 * t) / Math.max(D - t, 1e-12);
    out.P_DNV = 1.05 * sigma_u * intact * geom;

    const lengthParam = (L * L) / Math.max(D * t, 1e-12);
    const dOverT = d / t;
    const mB31g = Math.sqrt(1 + 0.8 * lengthParam);
    const b31gShort =
      2.2 *
      sigma_y *
      (t / D) *
      ((1 - (2 * dOverT) / 3) / Math.max(1 - (2 * dOverT) / (3 * mB31g), 1e-12));
    const b31gLong = 2.2 * sigma_y * (t / D) * (1 - dOverT);
    out.P_ASME_B31G = lengthParam < 20 ? b31gShort : b31gLong;

    const mModB31g =
      lengthParam <= 50
        ? Math.sqrt(Math.max(1 + 0.6275 * lengthParam - 0.003375 * lengthParam * lengthParam, 1e-12))
        : 3.3 + 0.032 * lengthParam;
    out.P_Mod_B31G =
      2 *
      (sigma_y + 68.95) *
      (t / D) *
      ((1 - 0.85 * dOverT) / Math.max(1 - (0.85 * dOverT) / mModB31g, 1e-12));

    const pcorrcLength = L / Math.max(Math.sqrt((D * (t - d)) / 2), 1e-12);
    out.P_PCORRC = 2 * sigma_u * (t / D) * (1 - dOverT * (1 - Math.exp(-0.157 * pcorrcLength)));
    out.P_Mod_PCORRC = 1.8 * sigma_u * (t / D) * (1 - dOverT * (1 - Math.exp(-0.224 * pcorrcLength)));
    return out;
  }

  function addFeatures(row) {
    const out = { ...row };
    out.d_over_t = out.d / out.t;
    out.L_over_sqrtDt = out.L / Math.sqrt(Math.max(out.D * out.t, 1e-12));
    out.w_over_piD = out.w / (Math.PI * out.D);
    out.D_over_t = out.D / out.t;
    out.t_over_D = out.t / out.D;
    out.sigma_u_over_sigma_y = out.sigma_u / out.sigma_y;
    out.P_intact_y = out.sigma_y * ((2 * out.t) / Math.max(out.D - out.t, 1e-12));
    out.P_intact_u = out.sigma_u * ((2 * out.t) / Math.max(out.D - out.t, 1e-12));
    return out;
  }

  function predictResidual(model, x, pDnv) {
    const raw = predictModel(model.base_model, x);
    if (model.target_mode === "log_ratio") {
      return pDnv * Math.exp(raw);
    }
    if (model.target_mode === "ratio") {
      return pDnv * raw;
    }
    return pDnv + raw;
  }

  function predictModel(model, x) {
    switch (model.kind) {
      case "sklearn_forest_regressor":
        return predictForestRegressor(model, x);
      case "sklearn_forest_classifier":
        return predictForestClassifierPositive(model, x);
      case "hist_gradient_boosting_regressor":
        return predictHistGradientBoosting(model, x);
      case "lightgbm_regressor":
        return predictLightgbm(model, x);
      case "catboost_regressor":
        return predictCatboost(model, x);
      default:
        throw new Error(`Unsupported model kind: ${model.kind}`);
    }
  }

  function predictForestRegressor(model, x) {
    let total = 0;
    for (const tree of model.trees) {
      total += predictSklearnTreeValue(tree, x);
    }
    return total / model.trees.length;
  }

  function predictForestClassifierPositive(model, x) {
    let total = 0;
    for (const tree of model.trees) {
      const value = predictSklearnTreeValue(tree, x);
      const denom = value.reduce((a, b) => a + b, 0);
      total += denom > 0 ? value[1] / denom : 0;
    }
    return total / model.trees.length;
  }

  function predictSklearnTreeValue(tree, x) {
    let node = 0;
    while (tree.children_left[node] !== -1) {
      const feature = tree.feature[node];
      const threshold = tree.threshold[node];
      node = x[feature] <= threshold ? tree.children_left[node] : tree.children_right[node];
    }
    return tree.value[node];
  }

  function predictHistGradientBoosting(model, x) {
    let total = model.baseline || 0;
    for (const tree of model.trees) {
      let node = 0;
      while (!tree.is_leaf[node]) {
        const feature = tree.feature_idx[node];
        const threshold = tree.num_threshold[node];
        const value = x[feature];
        const goLeft = Number.isNaN(value) ? Boolean(tree.missing_go_to_left[node]) : value <= threshold;
        node = goLeft ? tree.left[node] : tree.right[node];
      }
      total += tree.value[node];
    }
    return total;
  }

  function predictLightgbm(model, x) {
    let total = 0;
    for (const treeInfo of model.tree_info) {
      total += predictLightgbmNode(treeInfo.tree_structure, x);
    }
    return total;
  }

  function predictLightgbmNode(node, x) {
    if (Object.prototype.hasOwnProperty.call(node, "leaf_value")) {
      return node.leaf_value;
    }
    const featureValue = x[node.split_feature];
    let goLeft;
    if (!Number.isFinite(featureValue)) {
      goLeft = Boolean(node.default_left);
    } else if (node.decision_type === "<=" || node.decision_type === "") {
      goLeft = featureValue <= Number(node.threshold);
    } else {
      throw new Error(`Unsupported LightGBM decision type: ${node.decision_type}`);
    }
    return predictLightgbmNode(goLeft ? node.left_child : node.right_child, x);
  }

  function predictCatboost(model, x) {
    let total = 0;
    for (const tree of model.trees) {
      let index = 0;
      for (let i = 0; i < tree.splits.length; i += 1) {
        const split = tree.splits[i];
        if (x[split.feature] > split.border) {
          index |= 1 << i;
        }
      }
      total += tree.leaf_values[index];
    }
    return model.scale * total + model.bias;
  }

  function dynamicMargin(pUnsafe, sigma, tau, gamma, lambdaValue, q, riskFloor) {
    return lambdaValue * riskWeight(pUnsafe, tau, gamma, riskFloor) * q * sigma;
  }

  function riskWeight(pUnsafe, tau, gamma, riskFloor) {
    const scaled = tau >= 1 ? 0 : Math.max((pUnsafe - tau) / Math.max(1 - tau, 1e-12), 0);
    const clippedFloor = Math.min(Math.max(riskFloor, 0), 1);
    return clippedFloor + (1 - clippedFloor) * Math.pow(scaled, gamma);
  }

  function displayMethod(label, unsafeRate, testMape) {
    const parts = [];
    if (unsafeRate !== null && unsafeRate !== undefined && Number.isFinite(Number(unsafeRate))) {
      parts.push(`unsafe ${(Number(unsafeRate) * 100).toFixed(2)}%`);
    }
    if (testMape !== null && testMape !== undefined && Number.isFinite(Number(testMape))) {
      parts.push(`MAPE ${Number(testMape).toFixed(2)}%`);
    }
    return parts.length ? `${label} (${parts.join(", ")})` : label;
  }

  function formatNumber(value, decimals = 4) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return "";
    }
    return Number(value).toFixed(decimals);
  }

  function formatPressure(value) {
    const text = formatNumber(value, 3);
    return text ? `${text} MPa` : "-";
  }

  function formatRatio(value) {
    const text = formatNumber(value, 3);
    return text || "-";
  }

  function formatPercent(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return "-";
    }
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  const LABEL_HTML = Object.freeze({
    P_SAFER_final: "P<sub>SAFER,final</sub>",
    P_allow: "P<sub>allow</sub>",
    P_design: "P<sub>design</sub>",
    P_op: "P<sub>op</sub>",
    "P_op / P_interface": "P<sub>op</sub> / P<sub>interface</sub>",
    "eta / F": "&eta; / F",
    "d/t": "d/t",
    "L/sqrt(Dt)": "L/&radic;(Dt)",
    "w/(piD)": "w/(&pi;D)",
    "Selected p_unsafe": "Selected p<sub>unsafe</sub>",
  });

  function labelHtml(label) {
    return LABEL_HTML[label] || escapeHtml(label);
  }

  function mathify(text) {
    return escapeHtml(text)
      .replaceAll("P_SAFER_final", "P<sub>SAFER,final</sub>")
      .replaceAll("P_interface", "P<sub>interface</sub>")
      .replaceAll("P_allow", "P<sub>allow</sub>")
      .replaceAll("P_design", "P<sub>design</sub>")
      .replaceAll("P_op", "P<sub>op</sub>")
      .replaceAll("p_unsafe", "p<sub>unsafe</sub>")
      .replaceAll("eta", "&eta;")
      .replaceAll("sigma_y", "&sigma;<sub>y</sub>")
      .replaceAll("sigma_u", "&sigma;<sub>u</sub>")
      .replaceAll("L/sqrt(Dt)", "L/&radic;(Dt)")
      .replaceAll("w/(piD)", "w/(&pi;D)");
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function readForm() {
    const input = {};
    for (const key of Object.keys(DEFAULTS)) {
      input[key] = document.getElementById(key).value;
    }
    input.factorPreset = document.getElementById("factorPreset").value;
    return input;
  }

  function renderReport(report) {
    renderSummary(report);
    renderDetailList("interfaceDetails", [
      ["P_allow", formatPressure(report.interface.p_allow)],
      ["P_design", formatPressure(report.interface.p_design)],
      ["Governing constraint", report.interface.governing],
      ["eta / F", `${formatNumber(report.interface.eta, 3)} / ${formatNumber(report.interface.F, 3)}`],
      ["P_op", report.interface.p_op === null ? "Not provided" : formatPressure(report.interface.p_op)],
      ["P_op / P_interface", formatRatio(report.interface.p_op_over_interface)],
    ]);
    renderDetailList("explanationDetails", [
      ["d/t", formatRatio(report.explanation.d_over_t)],
      ["L/sqrt(Dt)", formatRatio(report.explanation.L_over_sqrtDt)],
      ["w/(piD)", formatRatio(report.explanation.w_over_piD)],
      ["SAFER adjustment", formatPressure(report.explanation.safer_adjustment)],
      ["SAFER model spread", formatPressure(report.explanation.safer_spread)],
      ["Selected p_unsafe", formatPercent(report.explanation.final_p_unsafe)],
      ["Risk level", report.explanation.risk_level || "-"],
    ]);
    renderResults(report.rows);
  }

  function renderSummary(report) {
    setText("saferCapacity", formatPressure(report.summary.p_safer_final));
    setHtml("saferCapacityNote", `${labelHtml("P_SAFER_final")} from ${escapeHtml(report.summary.final_method)}`);
    setText("interfacePressure", formatPressure(report.summary.p_interface));
    setText("interfaceNote", `Controlled by ${report.interface.governing}`);
    setText("decisionText", report.summary.decision.label);
    setHtml("decisionNote", mathify(report.summary.decision.note));
    const decisionCard = document.getElementById("decisionCard");
    decisionCard.classList.remove("pass", "review", "incomplete");
    decisionCard.classList.add(report.summary.decision.status);
  }

  function renderDetailList(id, pairs) {
    const list = document.getElementById(id);
    list.textContent = "";
    for (const [label, value] of pairs) {
      const term = document.createElement("dt");
      term.innerHTML = labelHtml(label);
      const description = document.createElement("dd");
      description.textContent = value;
      list.appendChild(term);
      list.appendChild(description);
    }
  }

  function renderResults(results) {
    const body = document.getElementById("resultsBody");
    body.textContent = "";
    let previousCategory = null;
    for (const row of results) {
      const tr = document.createElement("tr");
      if (previousCategory !== null && previousCategory !== row.category) {
        tr.classList.add("group-start");
      }
      previousCategory = row.category;

      const category = document.createElement("td");
      category.className = "category";
      category.textContent = row.category;
      tr.appendChild(category);

      const method = document.createElement("td");
      method.className = "method";
      method.textContent = row.display_method;
      tr.appendChild(method);

      const prediction = document.createElement("td");
      prediction.className = "number";
      prediction.textContent = formatNumber(row.prediction);
      tr.appendChild(prediction);

      const margin = document.createElement("td");
      margin.className = "number";
      margin.textContent = formatNumber(row.margin);
      tr.appendChild(margin);

      const pUnsafe = document.createElement("td");
      pUnsafe.className = "number";
      pUnsafe.textContent = row.p_unsafe === null || row.p_unsafe === undefined ? "" : formatPercent(row.p_unsafe);
      tr.appendChild(pUnsafe);

      const opMargin = document.createElement("td");
      opMargin.className = "number";
      opMargin.textContent = formatNumber(row.op_margin);
      tr.appendChild(opMargin);

      const safetyRatio = document.createElement("td");
      safetyRatio.className = "number";
      safetyRatio.textContent = formatNumber(row.safety_ratio);
      tr.appendChild(safetyRatio);

      body.appendChild(tr);
    }
  }

  function setText(id, text) {
    document.getElementById(id).textContent = text;
  }

  function setHtml(id, html) {
    document.getElementById(id).innerHTML = html;
  }

  function showError(message) {
    const box = document.getElementById("errorBox");
    box.textContent = message;
    box.hidden = false;
  }

  function clearError() {
    const box = document.getElementById("errorBox");
    box.textContent = "";
    box.hidden = true;
  }

  function setDefaults() {
    for (const [key, value] of Object.entries(DEFAULTS)) {
      document.getElementById(key).value = String(value);
    }
    document.getElementById("factorPreset").value = "reference";
  }

  function runUiCalculation() {
    try {
      clearError();
      renderReport(calculate(readForm()));
    } catch (error) {
      showError(error.message || String(error));
    }
  }

  function applyFactorPreset() {
    const presetName = document.getElementById("factorPreset").value;
    const preset = FACTOR_PRESETS[presetName];
    if (!preset) {
      return;
    }
    document.getElementById("eta").value = String(preset.eta);
    document.getElementById("F").value = String(preset.F);
    runUiCalculation();
  }

  function markCustomPreset() {
    document.getElementById("factorPreset").value = "custom";
  }

  function initializeUi() {
    const status = document.getElementById("modelStatus");
    status.textContent = globalThis.MODEL_DATA ? "Models loaded" : "Models missing";
    document.getElementById("calculatorForm").addEventListener("submit", (event) => {
      event.preventDefault();
      runUiCalculation();
    });
    document.getElementById("factorPreset").addEventListener("change", applyFactorPreset);
    document.getElementById("eta").addEventListener("input", markCustomPreset);
    document.getElementById("F").addEventListener("input", markCustomPreset);
    document.getElementById("resetButton").addEventListener("click", () => {
      setDefaults();
      runUiCalculation();
    });
    runUiCalculation();
  }

  globalThis.PressureCalculator = {
    calculate,
    addBaselines,
    addFeatures,
    predictModel,
    predictResidual,
  };

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", initializeUi);
  }
})();
