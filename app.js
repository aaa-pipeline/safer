(function () {
  "use strict";

  const EPS = 1e-8;
  const DEFAULTS = {
    D: 762,
    t: 17.5,
    d: 8.75,
    L: 50,
    w: 50,
    sigma_y: 495,
    sigma_u: 565,
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

    if (isExp102Mode(data)) {
      return calculateExp102(clean, row, data);
    }

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

  function isExp102Mode(data) {
    return data && (data.mode === "exp102_verified_test_set" || data.mode === "exp102_safer_model_export");
  }

  function hasExp102ModelExport(data) {
    return Boolean(data && data.residual_models && data.safer_models && data.lower_bound_models);
  }

  function calculateExp102(input, featureRow, data) {
    const matchedCase = findMatchingTestCase(input, data.test_cases || []);
    if (hasExp102ModelExport(data)) {
      return calculateExp102ModelExport(input, featureRow, data, matchedCase);
    }
    return calculateVerifiedTestCase(input, featureRow, data, matchedCase);
  }

  function calculateExp102ModelExport(input, featureRow, data, matchedCase) {
    const metadata = data.metadata;
    const featureColumns = metadata.feature_columns;
    const x = featureColumns.map((name) => featureRow[name]);
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
      const prediction = column === "P_DNV" && matchedCase ? matchedCase.P_DNV : featureRow[column];
      results.push({
        category: "Standard",
        method: label,
        display_method: displayMethod(label, metrics.unsafe_rate, metrics.test_mape),
        prediction,
        test_mape: metrics.test_mape,
        test_unsafe_rate: metrics.unsafe_rate,
        test_unsafe_count: metrics.unsafe_count,
        test_n: metrics.n,
        p_unsafe: null,
        margin: null,
        op_margin: opMargin(prediction, input.P_op),
        safety_ratio: capacityToOpRatio(prediction, input.P_op),
      });
    }

    const rawPredictions = {};
    for (const item of metadata.residual_top3 || []) {
      const computed = predictResidual(data.residual_models[item.artifact_name], x, featureRow.P_DNV);
      const prediction =
        matchedCase && matchedCase.raw_predictions[item.artifact_name] !== undefined
          ? matchedCase.raw_predictions[item.artifact_name]
          : computed;
      rawPredictions[item.artifact_name] = computed;
      results.push({
        category: "Residual",
        method: item.label,
        display_method: displayMethod(item.label, item.unsafe_rate, item.test_mape),
        prediction,
        test_mape: item.test_mape,
        test_unsafe_rate: item.unsafe_rate,
        test_unsafe_count: item.unsafe_count,
        test_n: item.n,
        p_unsafe: null,
        margin: null,
        op_margin: opMargin(prediction, input.P_op),
        safety_ratio: capacityToOpRatio(prediction, input.P_op),
      });
    }

    for (const item of metadata.safer_top3 || []) {
      const computed = predictExp102Safer(item, x, featureRow, data, rawPredictions[item.artifact_name]);
      const prediction =
        matchedCase && matchedCase.safer_predictions[item.artifact_name] !== undefined
          ? matchedCase.safer_predictions[item.artifact_name]
          : computed.prediction;
      const pRaw =
        matchedCase && matchedCase.raw_predictions[item.artifact_name] !== undefined
          ? matchedCase.raw_predictions[item.artifact_name]
          : computed.p_raw;
      results.push({
        category: "SAFER",
        method: item.label,
        display_method: displayMethod(item.label, item.unsafe_rate, item.test_mape),
        prediction,
        test_mape: item.test_mape,
        test_unsafe_rate: item.unsafe_rate,
        test_unsafe_count: item.unsafe_count,
        test_n: item.n,
        p_unsafe: computed.p_unsafe,
        sigma_error: computed.sigma_error,
        p_raw: pRaw,
        margin: pRaw - prediction,
        op_margin: opMargin(prediction, input.P_op),
        safety_ratio: capacityToOpRatio(prediction, input.P_op),
      });
    }

    const finalMetric = metadata.final_safer || {};
    const saferRows = results.filter((row) => row.category === "SAFER");
    const computedFinal = saferRows.reduce((best, row) => (row.prediction > best.prediction ? row : best), saferRows[0]);
    const finalPrediction = matchedCase ? matchedCase.P_SAFER_final : computedFinal.prediction;
    const sourceRow =
      saferRows.find((row) => nearlyEqual(row.prediction, finalPrediction)) ||
      computedFinal;
    results.push({
      category: "SAFER",
      method: finalMetric.label || "P_SAFER_final",
      display_method: displayMethod("P_SAFER_final (max of 3)", finalMetric.unsafe_rate, finalMetric.test_mape),
      prediction: finalPrediction,
      test_mape: finalMetric.test_mape,
      test_unsafe_rate: finalMetric.unsafe_rate,
      test_unsafe_count: finalMetric.unsafe_count,
      test_n: finalMetric.n,
      p_unsafe: sourceRow ? sourceRow.p_unsafe : null,
      sigma_error: sourceRow ? sourceRow.sigma_error : null,
      p_raw: sourceRow ? sourceRow.p_raw : null,
      margin: sourceRow ? sourceRow.p_raw - finalPrediction : null,
      op_margin: opMargin(finalPrediction, input.P_op),
      safety_ratio: capacityToOpRatio(finalPrediction, input.P_op),
      is_final: true,
    });

    return buildReport(input, featureRow, results, matchedCase, metadata);
  }

  function calculateVerifiedTestCase(input, featureRow, data, matchedCase = null) {
    const metadata = data.metadata;
    matchedCase = matchedCase || findMatchingTestCase(input, data.test_cases || []);
    if (!matchedCase) {
      throw new Error(
        "This lookup-only export contains the 84 locked EXP102 specimens but no browser-side models. Regenerate model-data.js without --lookup-only to enable arbitrary input."
      );
    }

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
      const prediction = column === "P_DNV" ? matchedCase.P_DNV : featureRow[column];
      results.push({
        category: "Standard",
        method: label,
        display_method: displayMethod(label, metrics.unsafe_rate, metrics.test_mape),
        prediction,
        test_mape: metrics.test_mape,
        test_unsafe_rate: metrics.unsafe_rate,
        test_unsafe_count: metrics.unsafe_count,
        test_n: metrics.n,
        p_unsafe: null,
        margin: null,
        op_margin: opMargin(prediction, input.P_op),
        safety_ratio: capacityToOpRatio(prediction, input.P_op),
      });
    }

    for (const item of metadata.residual_top3 || []) {
      const prediction = matchedCase.raw_predictions[item.artifact_name];
      results.push({
        category: "Residual",
        method: item.label,
        display_method: displayMethod(item.label, item.unsafe_rate, item.test_mape),
        prediction,
        test_mape: item.test_mape,
        test_unsafe_rate: item.unsafe_rate,
        test_unsafe_count: item.unsafe_count,
        test_n: item.n,
        p_unsafe: null,
        margin: null,
        op_margin: opMargin(prediction, input.P_op),
        safety_ratio: capacityToOpRatio(prediction, input.P_op),
      });
    }

    for (const item of metadata.safer_top3 || []) {
      const prediction = matchedCase.safer_predictions[item.artifact_name];
      const rawPrediction = matchedCase.raw_predictions[item.artifact_name];
      const hasRawPrediction =
        rawPrediction !== null && rawPrediction !== undefined && Number.isFinite(Number(rawPrediction));
      results.push({
        category: "SAFER",
        method: item.label,
        display_method: displayMethod(item.label, item.unsafe_rate, item.test_mape),
        prediction,
        test_mape: item.test_mape,
        test_unsafe_rate: item.unsafe_rate,
        test_unsafe_count: item.unsafe_count,
        test_n: item.n,
        p_unsafe: null,
        sigma_error: null,
        p_raw: rawPrediction,
        margin: hasRawPrediction ? rawPrediction - prediction : null,
        op_margin: opMargin(prediction, input.P_op),
        safety_ratio: capacityToOpRatio(prediction, input.P_op),
      });
    }

    const finalMetric = metadata.final_safer || {};
    const finalPrediction = matchedCase.P_SAFER_final;
    const sourceRow = results
      .filter((row) => row.category === "SAFER")
      .find((row) => nearlyEqual(row.prediction, finalPrediction));
    results.push({
      category: "SAFER",
      method: finalMetric.label || "P_SAFER_final",
      display_method: displayMethod("P_SAFER_final (max of 3)", finalMetric.unsafe_rate, finalMetric.test_mape),
      prediction: finalPrediction,
      test_mape: finalMetric.test_mape,
      test_unsafe_rate: finalMetric.unsafe_rate,
      test_unsafe_count: finalMetric.unsafe_count,
      test_n: finalMetric.n,
      p_unsafe: null,
      sigma_error: null,
      p_raw: sourceRow ? sourceRow.p_raw : null,
      margin: sourceRow ? sourceRow.margin : null,
      op_margin: opMargin(finalPrediction, input.P_op),
      safety_ratio: capacityToOpRatio(finalPrediction, input.P_op),
      is_final: true,
    });

    return buildReport(input, featureRow, results, matchedCase, metadata);
  }

  function predictExp102Safer(item, x, featureRow, data, rawPrediction = null) {
    const artifactName = item.artifact_name;
    const safer = data.safer_models[artifactName];
    if (!safer) {
      throw new Error(`Missing EXP102 SAFER model export for ${artifactName}.`);
    }
    const pRaw =
      rawPrediction !== null && rawPrediction !== undefined
        ? rawPrediction
        : predictResidual(data.residual_models[artifactName], x, featureRow.P_DNV);
    const riskX = x.concat([pRaw, pRaw / Math.max(featureRow.P_DNV, EPS)]);
    const pUnsafe = predictModel(safer.risk_classifier, riskX);
    const sigma = Math.max(predictModel(safer.error_scale_model, riskX), data.metadata.eps || EPS);
    const targetPolicy = safer.target_hybrid_policy;
    const pLower = predictLowerBound(data.lower_bound_models[targetPolicy.lb_candidate], x, featureRow.P_DNV);
    const targetPrediction = guardedHybridPrediction(pRaw, pLower, pUnsafe, sigma, targetPolicy);
    const selectivePolicy = safer.selective_anchor_policy;
    const anchorRaw = predictLowerBound(data.lower_bound_models[selectivePolicy.anchor_candidate], x, featureRow.P_DNV);
    const safeAnchor = Math.max(anchorRaw - Number(selectivePolicy.anchor_delta_abs || 0), EPS);
    const useAnchor = selectiveAnchorGate(selectivePolicy, featureRow, pRaw, sigma, safeAnchor);
    const prediction = useAnchor ? Math.min(targetPrediction, safeAnchor) : targetPrediction;
    return {
      prediction,
      p_raw: pRaw,
      p_unsafe: pUnsafe,
      sigma_error: sigma,
      target_prediction: targetPrediction,
      safe_anchor: safeAnchor,
      selective_anchor_gate: useAnchor,
      margin: pRaw - prediction,
    };
  }

  function predictLowerBound(model, x, pDnv) {
    if (!model) {
      throw new Error("Missing lower-bound model export.");
    }
    const raw = predictModel(model.base_model, x);
    if (model.target_mode === "log_ratio") {
      return pDnv * Math.exp(raw);
    }
    if (model.target_mode === "ratio") {
      return pDnv * raw;
    }
    return pDnv + raw;
  }

  function guardedHybridPrediction(pPoint, pLower, risk, sigma, params) {
    const pBound = Math.min(Number(pLower), Number(pPoint));
    const logPoint = Math.log(Math.max(Number(pPoint), EPS));
    const logLower = Math.log(Math.max(pBound, EPS));
    const gap = Math.max(logPoint - logLower, 0);
    const score = exp102Score(gap, risk, sigma, params);
    const gate = clamp01((score - Number(params.gate_threshold || 0)) / Math.max(Number(params.gate_denominator), EPS));
    return Math.exp(logPoint - Number(params.blend || 0) * gate * gap - Number(params.lb_log_delta || 0));
  }

  function exp102Score(gap, risk, sigma, params) {
    const mode = String(params.score_mode || "risk");
    const gapScale = Math.max(Number(params.gap_scale || 0), EPS);
    const sigmaScale = Math.max(Number(params.sigma_scale || 0), EPS);
    const gapN = Math.min(Math.max(gap / gapScale, 0), 2);
    const sigmaN = Math.min(Math.max(Number(sigma) / sigmaScale, 0), 2);
    const pUnsafe = clamp01(Number(risk));
    if (mode === "risk") {
      return pUnsafe;
    }
    if (mode === "lb_gap") {
      return gapN;
    }
    if (mode === "risk_times_gap") {
      return pUnsafe * gapN;
    }
    if (mode === "sigma_times_gap") {
      return sigmaN * gapN;
    }
    if (mode === "max_risk_gap") {
      return Math.max(pUnsafe, gapN);
    }
    throw new Error(`Unsupported EXP102 score mode: ${mode}`);
  }

  function selectiveAnchorGate(params, featureRow, pRaw, sigma, safeAnchor) {
    return (
      signalGate(exp102Signal(params.signal_1, featureRow, pRaw, sigma, safeAnchor), params.direction_1, params.threshold_1) ||
      signalGate(exp102Signal(params.signal_2, featureRow, pRaw, sigma, safeAnchor), params.direction_2, params.threshold_2)
    );
  }

  function exp102Signal(name, featureRow, pRaw, sigma, safeAnchor) {
    switch (String(name || "")) {
      case "dnv_ratio_to_anchor":
        return featureRow.P_DNV / Math.max(safeAnchor, EPS);
      case "raw_over_dnv":
        return pRaw / Math.max(featureRow.P_DNV, EPS);
      case "L_over_sqrtDt":
        return featureRow.L_over_sqrtDt;
      case "sigma_error":
        return sigma;
      default:
        throw new Error(`Unsupported EXP102 selective signal: ${name}`);
    }
  }

  function signalGate(value, direction, threshold) {
    const number = Number(value);
    const cut = Number(threshold);
    if (!Number.isFinite(number) || !Number.isFinite(cut)) {
      return false;
    }
    if (direction === "high") {
      return number >= cut;
    }
    if (direction === "low") {
      return number <= cut;
    }
    throw new Error(`Unsupported EXP102 gate direction: ${direction}`);
  }

  function clamp01(value) {
    if (!Number.isFinite(Number(value))) {
      return 0;
    }
    return Math.min(Math.max(Number(value), 0), 1);
  }

  function findMatchingTestCase(input, cases) {
    if (typeof document !== "undefined") {
      const sampleCase = document.getElementById("sampleCase");
      const selected = sampleCase ? cases[Number(sampleCase.value)] : null;
      if (selected && caseMatchesInput(input, selected)) {
        return selected;
      }
    }
    return cases.find((item) => caseMatchesInput(input, item));
  }

  function caseMatchesInput(input, item) {
    return ["D", "t", "d", "L", "w", "sigma_y", "sigma_u"].every((key) =>
      nearlyEqual(Number(input[key]), Number(item[key]))
    );
  }

  function nearlyEqual(a, b) {
    return Math.abs(Number(a) - Number(b)) <= 1e-6 * Math.max(1, Math.abs(Number(a)), Math.abs(Number(b)));
  }

  function buildReport(input, featureRow, rows, matchedCase = null, metadata = null) {
    const saferRows = rows.filter((row) => row.category === "SAFER");
    if (!saferRows.length) {
      throw new Error("No SAFER models are available in model-data.js.");
    }
    const finalSafer =
      saferRows.find((row) => row.is_final) ||
      saferRows.reduce((best, row) => (row.prediction > best.prediction ? row : best), saferRows[0]);
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
      test_case: matchedCase,
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
        matched_sample: matchedCase ? `ID ${matchedCase.ID} / ${matchedCase.condition_group_id}` : "",
        data_mode: metadata && metadata.created_from ? metadata.created_from : "",
        d_over_t: featureRow.d_over_t,
        L_over_sqrtDt: featureRow.L_over_sqrtDt,
        w_over_piD: featureRow.w_over_piD,
        safer_adjustment: finalSafer.margin,
        final_p_unsafe: finalSafer.p_unsafe,
        final_test_unsafe_rate: finalSafer.test_unsafe_rate,
        final_test_unsafe_count: finalSafer.test_unsafe_count,
        final_test_n: finalSafer.test_n,
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
      case "dummy_classifier":
        return predictDummyClassifierPositive(model);
      case "sklearn_forest_regressor":
        return predictForestRegressor(model, x);
      case "sklearn_forest_classifier":
        return predictForestClassifierPositive(model, x);
      case "sklearn_gradient_boosting_regressor":
        return predictGradientBoosting(model, x);
      case "hist_gradient_boosting_regressor":
        return predictHistGradientBoosting(model, x);
      case "xgboost_regressor":
        return predictXgboost(model, x);
      case "lightgbm_regressor":
        return predictLightgbm(model, x);
      case "catboost_regressor":
        return predictCatboost(model, x);
      default:
        throw new Error(`Unsupported model kind: ${model.kind}`);
    }
  }

  function predictDummyClassifierPositive(model) {
    return Number(model.constant) === 1 ? 1 : 0;
  }

  function predictForestRegressor(model, x) {
    let total = 0;
    for (const tree of model.trees) {
      total += predictSklearnTreeValue(tree, x);
    }
    return total / model.trees.length;
  }

  function predictGradientBoosting(model, x) {
    let total = Number(model.init || 0);
    for (const tree of model.trees) {
      total += Number(model.learning_rate || 1) * predictSklearnTreeValue(tree, x);
    }
    return total;
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

  function predictXgboost(model, x) {
    let total = Number(model.base_score || 0);
    for (const tree of model.trees) {
      total += predictXgboostNode(tree, x);
    }
    return total;
  }

  function predictXgboostNode(node, x) {
    if (Object.prototype.hasOwnProperty.call(node, "leaf")) {
      return Number(node.leaf);
    }
    const value = x[node.feature];
    if (!Number.isFinite(value)) {
      return predictXgboostNode(node.missing, x);
    }
    return predictXgboostNode(value < Number(node.threshold) ? node.yes : node.no, x);
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

  function formatUnsafeCount(count, n, rate) {
    if (count !== null && count !== undefined && n !== null && n !== undefined) {
      return `${count}/${n} (${formatPercent(rate)})`;
    }
    return formatPercent(rate);
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
    "Matched sample": "Matched sample",
    "Data source": "Data source",
    "EXP102 test unsafe": "EXP102 test unsafe",
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
    syncSampleSelection(report.test_case);
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
      ["Matched sample", report.explanation.matched_sample || "-"],
      ["d/t", formatRatio(report.explanation.d_over_t)],
      ["L/sqrt(Dt)", formatRatio(report.explanation.L_over_sqrtDt)],
      ["w/(piD)", formatRatio(report.explanation.w_over_piD)],
      ["SAFER adjustment", formatPressure(report.explanation.safer_adjustment)],
      ["SAFER model spread", formatPressure(report.explanation.safer_spread)],
      [
        "EXP102 test unsafe",
        formatUnsafeCount(
          report.explanation.final_test_unsafe_count,
          report.explanation.final_test_n,
          report.explanation.final_test_unsafe_rate
        ),
      ],
      ["Data source", report.explanation.data_mode || "-"],
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

      const testUnsafe = document.createElement("td");
      testUnsafe.className = "number";
      testUnsafe.textContent = formatUnsafeCount(row.test_unsafe_count, row.test_n, row.test_unsafe_rate);
      tr.appendChild(testUnsafe);

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

  function clearReport() {
    setText("saferCapacity", "-");
    setHtml("saferCapacityNote", labelHtml("P_SAFER_final"));
    setText("interfacePressure", "-");
    setText("interfaceNote", "min(P_allow, P_design)");
    setText("decisionText", "-");
    setText("decisionNote", "P_op / P_interface");
    document.getElementById("interfaceDetails").textContent = "";
    document.getElementById("explanationDetails").textContent = "";
    document.getElementById("resultsBody").textContent = "";
  }

  function setDefaults() {
    for (const [key, value] of Object.entries(DEFAULTS)) {
      document.getElementById(key).value = String(value);
    }
    document.getElementById("factorPreset").value = "reference";
    const sampleCase = document.getElementById("sampleCase");
    if (sampleCase && sampleCase.options.length) {
      sampleCase.selectedIndex = 0;
    }
  }

  function populateSampleCases() {
    const sampleCase = document.getElementById("sampleCase");
    if (!sampleCase) {
      return;
    }
    sampleCase.textContent = "";
    const cases = isExp102Mode(globalThis.MODEL_DATA) ? globalThis.MODEL_DATA.test_cases || [] : [];
    if (!cases.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No EXP102 cases";
      sampleCase.appendChild(option);
      sampleCase.disabled = true;
      return;
    }
    sampleCase.disabled = false;
    cases.forEach((item, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `ID ${item.ID} / ${item.condition_group_id}`;
      sampleCase.appendChild(option);
    });
  }

  function applySampleCase() {
    const data = globalThis.MODEL_DATA;
    if (!isExp102Mode(data)) {
      return;
    }
    const sampleCase = document.getElementById("sampleCase");
    const item = data.test_cases[Number(sampleCase.value)];
    if (!item) {
      return;
    }
    for (const key of ["D", "t", "d", "L", "w", "sigma_y", "sigma_u"]) {
      document.getElementById(key).value = String(item[key]);
    }
    runUiCalculation();
  }

  function syncSampleSelection(matchedCase) {
    if (!matchedCase || !isExp102Mode(globalThis.MODEL_DATA)) {
      return;
    }
    const index = (globalThis.MODEL_DATA.test_cases || []).findIndex((item) => item.row_id === matchedCase.row_id);
    const sampleCase = document.getElementById("sampleCase");
    if (sampleCase && index >= 0) {
      sampleCase.value = String(index);
    }
  }

  function runUiCalculation() {
    try {
      clearError();
      renderReport(calculate(readForm()));
    } catch (error) {
      clearReport();
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
    status.textContent =
      isExp102Mode(globalThis.MODEL_DATA)
        ? hasExp102ModelExport(globalThis.MODEL_DATA)
          ? "EXP102 models loaded"
          : "EXP102 lookup loaded"
        : globalThis.MODEL_DATA
          ? "Models loaded"
          : "Models missing";
    populateSampleCases();
    document.getElementById("calculatorForm").addEventListener("submit", (event) => {
      event.preventDefault();
      runUiCalculation();
    });
    document.getElementById("sampleCase").addEventListener("change", applySampleCase);
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
