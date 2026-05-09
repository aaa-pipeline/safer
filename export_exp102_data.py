from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp101_good_diverse_residual_safer import (
    build_lb_specs,
    build_residual_oof,
    fit_lb_model,
)
from exp102_selected_three_safe_safer import score_for_params
from safer_strict_fem_assisted import load_exp_and_fem
from src.augmentation import augment_train_set
from src.config import Config
from src.features import feature_columns
from src.safety_calibration import fit_safety_calibrator
from src.train_residual import fit_residual_learner, predict_residual_pressure


EXP102_DIR = ROOT / "outputs_experiments_20260507" / "exp102_selected_three_safe_safer"
MODEL_DATA_PATH = Path(__file__).resolve().parent / "model-data.js"

CAPACITY_TABLE = EXP102_DIR / "final_selected_three_engineering_capacity_table.csv"
BENCHMARK_TABLE = EXP102_DIR / "standards_vs_safer_underestimation_test.csv"
SUMMARY_TABLE = EXP102_DIR / "selected_three_safe_safer_summary.csv"
PREDICTIONS_TABLE = EXP102_DIR / "selected_three_safe_safer_predictions.csv"
GRID_TABLE = EXP102_DIR / "selected_three_safe_safer_grids.csv"
METADATA_JSON = EXP102_DIR / "metadata.json"

FINAL_SAFER_POLICY = "SAFER_selective_anchor_cap_val0_cov_eps1"
TARGET_HYBRID_POLICY = "hybrid_SAFER_val_target_q95_min_val_MAPE"

BACKBONES = [
    ("ET-LogRatio", "ET", "ET_LogRatio", "ET"),
    ("HGB-LogRatio", "HGB", "HGB_LogRatio", "HGB"),
    ("XGBoost-LogRatio", "XGBoost", "XGBoost_LogRatio", "XGBoost"),
]
OOF_SEED_INDEX = {
    "RF-LogRatio": 0,
    "ET-LogRatio": 1,
    "HGB-LogRatio": 2,
    "XGBoost-LogRatio": 3,
    "LightGBM-LogRatio": 4,
    "CatBoost-LogRatio": 5,
}
MODEL_TO_ARTIFACT = {model_name: artifact_name for model_name, _, artifact_name, _ in BACKBONES}
FORMULA_LABELS = ["DNV-RP-F101", "ASME B31G", "Modified B31G", "PCORRC", "Modified PCORRC"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def i(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def metric(row: dict[str, str]) -> dict[str, Any]:
    n = i(row.get("n"))
    unsafe_count = i(row.get("unsafe_count"))
    return {
        "test_mape": f(row.get("MAPE")),
        "test_mae": f(row.get("MAE")),
        "test_rmse": f(row.get("RMSE")),
        "test_r2": f(row.get("R2")),
        "unsafe_count": unsafe_count,
        "n": n,
        "unsafe_rate": None if n in (None, 0) or unsafe_count is None else unsafe_count / n,
        "coverage": None if n in (None, 0) or unsafe_count is None else 1 - unsafe_count / n,
    }


def build_metrics() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    benchmark_rows = read_csv(BENCHMARK_TABLE)
    benchmark_by_method = {row["method"]: row for row in benchmark_rows}
    formula_metrics = {label: metric(benchmark_by_method[label]) for label in FORMULA_LABELS}

    summary_rows = read_csv(SUMMARY_TABLE)
    summary_lookup = {
        (row["scope"], row["model_name"], row["safer_policy"]): row
        for row in summary_rows
    }

    residual_top3 = []
    safer_top3 = []
    for model_name, family, artifact_name, short_label in BACKBONES:
        raw_row = summary_lookup[("test_EXP", model_name, "point_prediction_no_margin")]
        residual_top3.append(
            {
                "label": f"raw-{short_label}",
                "canonical_model": model_name,
                "family": family,
                "artifact_name": artifact_name,
                **metric(raw_row),
            }
        )

        safer_row = summary_lookup[("test_EXP", model_name, FINAL_SAFER_POLICY)]
        safer_top3.append(
            {
                "label": f"SAFER-{short_label}",
                "canonical_model": model_name,
                "family": family,
                "artifact_name": artifact_name,
                **metric(safer_row),
            }
        )

    final_metric = metric(benchmark_by_method["SAFER selective-anchor max of 3"])
    final_metric.update(
        {
            "label": "P_SAFER_final",
            "method": "SAFER selective-anchor max of 3",
            "basis": "max_of_3_selected_exact_safe_backbones",
        }
    )
    return formula_metrics, residual_top3, safer_top3, final_metric


def build_raw_prediction_lookup() -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    name_to_artifact = {model_name: artifact_name for model_name, _, artifact_name, _ in BACKBONES}
    for row in read_csv(PREDICTIONS_TABLE):
        if row["split"] != "test" or row["safer_policy"] != "point_prediction_no_margin":
            continue
        artifact_name = name_to_artifact.get(row["model_name"])
        if artifact_name:
            out[(int(row["row_id"]), artifact_name)] = float(row["P_raw"])
    return out


def build_cases() -> list[dict[str, Any]]:
    raw_lookup = build_raw_prediction_lookup()
    cases: list[dict[str, Any]] = []
    for row in read_csv(CAPACITY_TABLE):
        row_id = int(row["row_id"])
        raw_predictions = {
            artifact_name: raw_lookup.get((row_id, artifact_name))
            for _, _, artifact_name, _ in BACKBONES
        }
        safer_predictions = {
            "ET_LogRatio": float(row["P_SAFER_ET_LogRatio"]),
            "HGB_LogRatio": float(row["P_SAFER_HGB_LogRatio"]),
            "XGBoost_LogRatio": float(row["P_SAFER_XGBoost_LogRatio"]),
        }
        cases.append(
            {
                "row_id": row_id,
                "ID": row["ID"],
                "condition_group_id": row["condition_group_id"],
                "D": float(row["D"]),
                "t": float(row["t"]),
                "d": float(row["d"]),
                "L": float(row["L"]),
                "w": float(row["w"]),
                "sigma_y": float(row["sigma_y"]),
                "sigma_u": float(row["sigma_u"]),
                "P_DNV": float(row["P_DNV"]),
                "P_SAFER_final": float(row["P_SAFER_final"]),
                "P_SAFER_final_basis": row["P_SAFER_final_basis"],
                "raw_predictions": raw_predictions,
                "safer_predictions": safer_predictions,
            }
        )
    return cases


def build_target_hybrid_policies() -> dict[str, dict[str, Any]]:
    usecols = [
        "point_model",
        "lb_candidate",
        "lb_family",
        "score_mode",
        "gate_style",
        "gate_quantile",
        "gate_threshold",
        "blend",
        "correction_mode",
        "guard_factor",
        "lb_log_delta",
        "gap_scale",
        "sigma_scale",
        "val_unsafe_count",
        "val_MAPE",
        "val_RMSE",
        "val_mean_margin",
    ]
    grid = pd.read_csv(GRID_TABLE, usecols=usecols)
    policies: dict[str, dict[str, Any]] = {}
    for model_name, _, _, _ in BACKBONES:
        pool = grid.loc[
            (grid["point_model"].eq(model_name))
            & (grid["correction_mode"].eq("target_q95"))
            & (grid["val_unsafe_count"] <= 4)
        ].copy()
        if pool.empty:
            raise ValueError(f"No target hybrid policy was found for {model_name}")
        row = pool.sort_values(["val_MAPE", "val_RMSE", "val_mean_margin"]).iloc[0].to_dict()
        policies[model_name] = clean_policy(row)
    return policies


def build_selective_policies() -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for row in read_csv(SUMMARY_TABLE):
        if row["scope"] != "test_EXP" or row["safer_policy"] != FINAL_SAFER_POLICY:
            continue
        policies[row["model_name"]] = clean_policy(
            {
                "anchor_candidate": row["anchor_candidate"],
                "anchor_delta_abs": row["anchor_delta_abs"],
                "selective_policy": row["selective_policy"],
                "signal_1": row["signal_1"],
                "direction_1": row["direction_1"],
                "q_1": row["q_1"],
                "threshold_1": row["threshold_1"],
                "signal_2": row["signal_2"],
                "direction_2": row["direction_2"],
                "q_2": row["q_2"],
                "threshold_2": row["threshold_2"],
                "test_cap_rate": row["test_cap_rate"],
            }
        )
    missing = {model_name for model_name, _, _, _ in BACKBONES} - set(policies)
    if missing:
        raise ValueError(f"Missing selective policies for: {sorted(missing)}")
    return policies


def clean_policy(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == "":
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                out[key] = value
            else:
                out[key] = number
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
            if math.isfinite(value):
                out[key] = value
        else:
            out[key] = value
    return out


def build_model_export(target_policies: dict[str, dict[str, Any]], selective_policies: dict[str, dict[str, Any]]):
    config = Config(run_robustness=False, make_figures=False, k_folds=5)
    exp_df, fem_df, _ = load_exp_and_fem(config.resolved_data_path)
    fcols = feature_columns(exp_df)
    train_df, val_df, _, _ = protocol_split_for_export(exp_df, fem_df, config)

    lb_specs = build_lb_specs()
    lb_lookup = {spec.candidate_id: (index, spec) for index, spec in enumerate(lb_specs)}
    required_lb_ids = {
        str(policy["lb_candidate"])
        for policy in target_policies.values()
        if policy.get("lb_candidate")
    } | {
        str(policy["anchor_candidate"])
        for policy in selective_policies.values()
        if policy.get("anchor_candidate")
    }

    fitted_lbs: dict[str, Any] = {}
    lower_bound_models: dict[str, Any] = {}
    for lb_id in sorted(required_lb_ids):
        index, spec = lb_lookup[lb_id]
        seed = config.random_state + 7000 + index
        print(f"[fit-lb-export] {lb_id}", flush=True)
        train_aug, _ = augment_train_set(train_df, config, seed=seed)
        model = fit_lb_model(train_aug, fcols, spec, seed)
        fitted_lbs[lb_id] = model
        lower_bound_models[lb_id] = export_lower_bound_model(model)

    residual_models: dict[str, Any] = {}
    safer_models: dict[str, Any] = {}
    selected_names = sorted(model_name for model_name, _, _, _ in BACKBONES)
    for fit_index, model_name in enumerate(selected_names):
        family = next(family for name, family, _, _ in BACKBONES if name == model_name)
        artifact_name = MODEL_TO_ARTIFACT[model_name]
        seed = config.random_state + 11000 + fit_index
        print(f"[fit-model-export] {model_name}", flush=True)
        train_aug, _ = augment_train_set(train_df, config, seed=seed)
        residual_model = fit_residual_learner(train_aug, fcols, model_name, seed)
        p_val = predict_residual_pressure(residual_model, val_df, fcols)

        oof_seed = config.random_state + 1000 * OOF_SEED_INDEX[model_name]
        oof = build_residual_oof(train_df, fcols, model_name, config, oof_seed)
        oof = oof.rename(columns={"P_oof": "P_raw_error_model", "error": "error_for_safety"})
        safety = fit_safety_calibrator(oof, val_df, fcols, p_val, config, seed)

        target_policy = dict(target_policies[model_name])
        p_lb_val = fitted_lbs[str(target_policy["lb_candidate"])].predict_pressure(val_df, fcols)
        risk_val = safety.calibration_predictions["p_unsafe"].to_numpy(dtype=float)
        sigma_val = safety.calibration_predictions["sigma_error"].to_numpy(dtype=float)
        target_policy["gate_denominator"] = gate_denominator_for_policy(
            val_df=val_df,
            fcols=fcols,
            p_raw=p_val,
            p_lb=p_lb_val,
            risk=risk_val,
            sigma=sigma_val,
            params=target_policy,
        )
        target_policy["deployment_gate_denominator_basis"] = "validation_EXP_score_p95_minus_threshold"

        safer_models[artifact_name] = {
            "canonical_model": model_name,
            "family": family,
            "risk_classifier": export_model(safety.risk_classifier),
            "error_scale_model": export_model(safety.error_scale_model),
            "dynamic_safety_params": clean_policy(safety.selected_params),
            "target_hybrid_policy": target_policy,
            "selective_anchor_policy": selective_policies[model_name],
        }
        residual_models[artifact_name] = export_residual_model(residual_model)

    return residual_models, lower_bound_models, safer_models


def protocol_split_for_export(exp_df: pd.DataFrame, fem_df: pd.DataFrame, config: Config):
    from exp101_good_diverse_residual_safer import protocol_split

    return protocol_split(exp_df, fem_df, config)


def gate_denominator_for_policy(
    val_df: pd.DataFrame,
    fcols: list[str],
    p_raw: np.ndarray,
    p_lb: np.ndarray,
    risk: np.ndarray,
    sigma: np.ndarray,
    params: dict[str, Any],
) -> float:
    del fcols
    p_point = np.asarray(p_raw, dtype=float)
    p_lb = np.minimum(np.asarray(p_lb, dtype=float), p_point)
    log_point = np.log(np.maximum(p_point, 1.0e-12))
    log_lb = np.log(np.maximum(p_lb, 1.0e-12))
    gap = np.maximum(log_point - log_lb, 0.0)
    score = score_for_params(gap, risk, sigma, params)
    threshold = float(params["gate_threshold"])
    return max(float(np.percentile(score, 95)) - threshold, 1.0e-12)


def export_residual_model(model: Any) -> dict[str, Any]:
    base_model = getattr(model, "base_model", model)
    return {
        "target_mode": getattr(model, "target_mode", "absolute"),
        "base_model": export_model(base_model),
    }


def export_lower_bound_model(model: Any) -> dict[str, Any]:
    return {
        "target_mode": "log_ratio",
        "base_model": export_model(model.base_model),
    }


def export_model(model: Any) -> dict[str, Any]:
    module = type(model).__module__
    name = type(model).__name__
    if name == "DummyClassifier":
        return export_dummy_classifier(model)
    if name == "GradientBoostingRegressor":
        return export_gradient_boosting(model)
    if name == "HistGradientBoostingRegressor":
        return export_hist_gradient_boosting(model)
    if name == "LGBMRegressor":
        return export_lightgbm(model)
    if name == "XGBRegressor" or "xgboost" in module.lower():
        return export_xgboost(model)
    if "catboost" in module.lower() or name == "CatBoostRegressor":
        return export_catboost(model)
    if hasattr(model, "estimators_"):
        return export_sklearn_forest(model)
    raise TypeError(f"Unsupported model type: {type(model)}")


def export_dummy_classifier(model: Any) -> dict[str, Any]:
    classes = getattr(model, "classes_", np.array([0, 1])).astype(int).tolist()
    constant = int(np.ravel(getattr(model, "constant_", [classes[0]]))[0])
    return {"kind": "dummy_classifier", "classes": classes, "constant": constant}


def export_sklearn_forest(model: Any) -> dict[str, Any]:
    is_classifier = hasattr(model, "classes_")
    trees = []
    for estimator in model.estimators_:
        if isinstance(estimator, (list, tuple, np.ndarray)):
            estimator = estimator[0]
        trees.append(export_sklearn_tree(estimator))
    return {
        "kind": "sklearn_forest_classifier" if is_classifier else "sklearn_forest_regressor",
        "n_estimators": len(trees),
        "classes": getattr(model, "classes_", np.array([])).astype(int).tolist() if is_classifier else None,
        "trees": trees,
    }


def export_gradient_boosting(model: Any) -> dict[str, Any]:
    trees = []
    for estimator in model.estimators_.ravel():
        trees.append(export_sklearn_tree(estimator))
    init = 0.0
    if hasattr(model, "init_") and hasattr(model.init_, "constant_"):
        init = float(np.asarray(model.init_.constant_).ravel()[0])
    return {
        "kind": "sklearn_gradient_boosting_regressor",
        "init": init,
        "learning_rate": float(model.learning_rate),
        "trees": trees,
    }


def export_sklearn_tree(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    values = tree.value
    if values.ndim == 3 and values.shape[2] > 1:
        node_values = values[:, 0, :].astype(float).tolist()
    else:
        node_values = values[:, 0, 0].astype(float).tolist()
    return {
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": node_values,
    }


def export_hist_gradient_boosting(model: Any) -> dict[str, Any]:
    trees = []
    for predictor_group in model._predictors:
        predictor = predictor_group[0]
        nodes = predictor.nodes
        trees.append(
            {
                "value": nodes["value"].astype(float).tolist(),
                "feature_idx": nodes["feature_idx"].astype(int).tolist(),
                "num_threshold": nodes["num_threshold"].astype(float).tolist(),
                "missing_go_to_left": nodes["missing_go_to_left"].astype(int).tolist(),
                "left": nodes["left"].astype(int).tolist(),
                "right": nodes["right"].astype(int).tolist(),
                "is_leaf": nodes["is_leaf"].astype(int).tolist(),
            }
        )
    baseline = np.asarray(getattr(model, "_baseline_prediction", [[0.0]])).ravel()[0]
    return {
        "kind": "hist_gradient_boosting_regressor",
        "baseline": float(baseline),
        "trees": trees,
    }


def export_xgboost(model: Any) -> dict[str, Any]:
    booster = model.get_booster()
    feature_names = booster.feature_names or []
    trees = [convert_xgboost_node(json.loads(text), feature_names) for text in booster.get_dump(dump_format="json")]
    config = json.loads(booster.save_config())
    base_score = parse_xgboost_base_score(config["learner"]["learner_model_param"].get("base_score", 0.0))
    return {
        "kind": "xgboost_regressor",
        "base_score": base_score,
        "trees": trees,
    }


def parse_xgboost_base_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].split(",", 1)[0]
    return float(text)


def convert_xgboost_node(node: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    if "leaf" in node:
        return {"leaf": float(node["leaf"])}
    children = {child["nodeid"]: child for child in node.get("children", [])}
    split = str(node["split"])
    if split in feature_names:
        feature = feature_names.index(split)
    elif split.startswith("f") and split[1:].isdigit():
        feature = int(split[1:])
    else:
        raise ValueError(f"Unknown XGBoost split feature: {split}")
    return {
        "feature": feature,
        "threshold": float(node["split_condition"]),
        "yes": convert_xgboost_node(children[int(node["yes"])], feature_names),
        "no": convert_xgboost_node(children[int(node["no"])], feature_names),
        "missing": convert_xgboost_node(children[int(node["missing"])], feature_names),
    }


def export_lightgbm(model: Any) -> dict[str, Any]:
    dump = model.booster_.dump_model()
    return {
        "kind": "lightgbm_regressor",
        "tree_info": dump["tree_info"],
    }


def export_catboost(model: Any) -> dict[str, Any]:
    import tempfile

    tmp = Path(tempfile.gettempdir()) / f"catboost_export_{id(model)}.json"
    model.save_model(str(tmp), format="json")
    data = json.loads(tmp.read_text(encoding="utf-8"))
    trees = []
    for tree in data["oblivious_trees"]:
        splits = []
        for split in tree["splits"]:
            if split.get("split_type") != "FloatFeature":
                raise TypeError(f"Unsupported CatBoost split type: {split}")
            splits.append(
                {
                    "feature": int(split["float_feature_index"]),
                    "border": float(split["border"]),
                }
            )
        trees.append(
            {
                "splits": splits,
                "leaf_values": [float(v) for v in tree["leaf_values"]],
            }
        )
    scale, bias = data.get("scale_and_bias", [1.0, [0.0]])
    return {
        "kind": "catboost_regressor",
        "scale": float(scale),
        "bias": float(bias[0] if isinstance(bias, list) else bias),
        "trees": trees,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookup-only",
        action="store_true",
        help="Export only the locked EXP102 test rows, without browser-side models.",
    )
    args = parser.parse_args()

    metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    formula_metrics, residual_top3, safer_top3, final_safer = build_metrics()
    cases = build_cases()
    target_policies = build_target_hybrid_policies()
    selective_policies = build_selective_policies()

    payload: dict[str, Any] = {
        "version": 3,
        "mode": "exp102_safer_model_export",
        "metadata": {
            "version": 3,
            "created_from": str(EXP102_DIR.relative_to(ROOT)).replace("\\", "/"),
            "data_path": metadata.get("data_path"),
            "n_test": metadata["split"]["test"]["n"],
            "feature_columns": [
                "D",
                "t",
                "d",
                "L",
                "w",
                "sigma_y",
                "sigma_u",
                "d_over_t",
                "L_over_sqrtDt",
                "w_over_piD",
                "D_over_t",
                "t_over_D",
                "sigma_u_over_sigma_y",
                "P_intact_y",
                "P_intact_u",
                "P_DNV",
            ],
            "formula_metrics": formula_metrics,
            "residual_top3": residual_top3,
            "safer_top3": safer_top3,
            "final_safer": final_safer,
            "eps": 1.0e-8,
            "selection_rules": {
                "residual_top3": "EXP102 good-but-diverse residual backbones selected from train-only OOF results",
                "safer_top3": FINAL_SAFER_POLICY,
                "target_hybrid": TARGET_HYBRID_POLICY,
                "final_safer": "P_SAFER_final is the per-sample maximum of ET/HGB/XGBoost SAFER capacities",
            },
        },
        "test_cases": cases,
    }

    if args.lookup_only:
        payload["mode"] = "exp102_verified_test_set"
    else:
        residual_models, lower_bound_models, safer_models = build_model_export(target_policies, selective_policies)
        payload["residual_models"] = residual_models
        payload["lower_bound_models"] = lower_bound_models
        payload["safer_models"] = safer_models

    js = "window.MODEL_DATA=" + json.dumps(_json_ready(payload), ensure_ascii=False, separators=(",", ":")) + ";\n"
    MODEL_DATA_PATH.write_text(js, encoding="utf-8")
    print(f"Wrote {MODEL_DATA_PATH}")
    print(f"Mode: {payload['mode']}")
    print(f"Test cases: {len(cases)}")
    print(f"Size: {MODEL_DATA_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
