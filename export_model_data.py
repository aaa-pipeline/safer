from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOCAL_ARTIFACT_DIR = PROJECT_ROOT / "local_calculator" / "artifacts"
WEB_DIR = Path(__file__).resolve().parent
MODEL_DATA_PATH = WEB_DIR / "model-data.js"


def main() -> None:
    metadata = json.loads((LOCAL_ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    model_names = sorted(
        {
            item["artifact_name"]
            for item in metadata["residual_top3"] + metadata["safer_top3"]
        }
    )

    residual_models: dict[str, Any] = {}
    for artifact_name in model_names:
        path = LOCAL_ARTIFACT_DIR / "residual_models" / f"{artifact_name}.joblib"
        residual_models[artifact_name] = export_residual_model(joblib.load(path))

    safer_models: dict[str, Any] = {}
    for item in metadata["safer_top3"]:
        artifact_name = item["artifact_name"]
        model_dir = LOCAL_ARTIFACT_DIR / "safer_models" / artifact_name
        safer_models[artifact_name] = {
            "risk_classifier": export_model(joblib.load(model_dir / "risk_classifier.joblib")),
            "error_scale_model": export_model(joblib.load(model_dir / "error_scale_model.joblib")),
            "safety_params": json.loads((model_dir / "safety_params.json").read_text(encoding="utf-8")),
        }

    payload = {
        "version": 1,
        "metadata": metadata,
        "residual_models": residual_models,
        "safer_models": safer_models,
    }

    js = "window.MODEL_DATA = " + json.dumps(_json_ready(payload), separators=(",", ":"), ensure_ascii=False) + ";\n"
    MODEL_DATA_PATH.write_text(js, encoding="utf-8")
    print(f"Wrote {MODEL_DATA_PATH}")
    print(f"Size: {MODEL_DATA_PATH.stat().st_size / (1024 * 1024):.2f} MB")


def export_residual_model(model: Any) -> dict[str, Any]:
    base_model = getattr(model, "base_model", model)
    return {
        "target_mode": getattr(model, "target_mode", "absolute"),
        "base_model": export_model(base_model),
    }


def export_model(model: Any) -> dict[str, Any]:
    module = type(model).__module__
    name = type(model).__name__
    if hasattr(model, "estimators_"):
        return export_sklearn_forest(model)
    if name == "HistGradientBoostingRegressor":
        return export_hist_gradient_boosting(model)
    if name == "LGBMRegressor":
        return export_lightgbm(model)
    if "catboost" in module.lower() or name == "CatBoostRegressor":
        return export_catboost(model)
    raise TypeError(f"Unsupported model type: {type(model)}")


def export_sklearn_forest(model: Any) -> dict[str, Any]:
    is_classifier = hasattr(model, "classes_")
    trees = []
    for estimator in model.estimators_:
        if isinstance(estimator, (list, tuple, np.ndarray)):
            estimator = estimator[0]
        tree = estimator.tree_
        values = tree.value
        if is_classifier:
            node_values = values[:, 0, :].astype(float).tolist()
        else:
            node_values = values[:, 0, 0].astype(float).tolist()
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "value": node_values,
            }
        )
    return {
        "kind": "sklearn_forest_classifier" if is_classifier else "sklearn_forest_regressor",
        "n_estimators": len(trees),
        "classes": getattr(model, "classes_", np.array([])).astype(int).tolist() if is_classifier else None,
        "trees": trees,
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


def export_lightgbm(model: Any) -> dict[str, Any]:
    dump = model.booster_.dump_model()
    return {
        "kind": "lightgbm_regressor",
        "tree_info": dump["tree_info"],
    }


def export_catboost(model: Any) -> dict[str, Any]:
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


if __name__ == "__main__":
    main()
