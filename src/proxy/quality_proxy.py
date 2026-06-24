"""Train and load the lightweight MLP task-quality proxy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.proxy.image_features import IMAGE_FEATURE_COLUMNS


TARGET_FIELD = "task_quality"
MODEL_TYPE_MLP = "mlp"
MODEL_TYPE_HIST_GBDT = "hist_gbdt"
MODEL_TYPE_RANDOM_FOREST = "random_forest"
MODEL_TYPE_EXTRA_TREES = "extra_trees"
MODEL_TYPES = (
    MODEL_TYPE_MLP,
    MODEL_TYPE_HIST_GBDT,
    MODEL_TYPE_RANDOM_FOREST,
    MODEL_TYPE_EXTRA_TREES,
)

# Keep this list restricted to values that are available online or available
# from offline cost tables. Do not add supervision-only diagnostics here.
CATEGORICAL_FEATURES = (
    "predicted_class",
    "compression_level",
)

BASE_NUMERIC_FEATURES = (
    "class_id",
    "detector_confidence",
    "area_ratio",
    "semantic_value",
    "grid_id",
    "class_priority",
    "exit_level",
    "compressed_bytes",
    "output_width",
    "output_height",
)
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + IMAGE_FEATURE_COLUMNS

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LEAKAGE_FIELDS = (
    "quality_label",
    TARGET_FIELD,
    "predicted_class_id",
    "exit_confidence",
)

REQUIRED_PROFILE_FIELDS = (
    "image_id",
    TARGET_FIELD,
    *FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    summary: dict[str, object]


@dataclass
class QualityProxyModel:
    """Saved sklearn pipeline plus the feature policy needed for safe inference."""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    leakage_fields: tuple[str, ...]
    target_field: str
    model_type: str = MODEL_TYPE_MLP

    def predict_quality(self, records: Iterable[Mapping[str, object]] | pd.DataFrame) -> np.ndarray:
        """Predict clipped task quality for candidate scheduling actions."""

        frame = records_to_frame(records)
        _ensure_optional_image_features(frame)
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"records missing feature fields: {missing}")
        predictions = self.pipeline.predict(frame[list(self.feature_columns)])
        return np.clip(predictions, 0.0, 1.0)


def predict_quality(
    model: QualityProxyModel,
    records: Iterable[Mapping[str, object]] | pd.DataFrame,
) -> np.ndarray:
    return model.predict_quality(records)


def load_profile_csv(path: Path, *, include_image_features: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_profile_frame(frame, include_image_features=include_image_features)
    return frame


def validate_feature_policy() -> None:
    """Guard against accidentally training on labels or model-output diagnostics."""

    overlap = sorted(set(FEATURE_COLUMNS) & set(LEAKAGE_FIELDS))
    if overlap:
        raise ValueError(f"feature columns contain leakage fields: {overlap}")


def feature_columns_for(*, include_image_features: bool = True) -> tuple[str, ...]:
    """Return the safe online/table feature set for one proxy training run."""

    if include_image_features:
        return FEATURE_COLUMNS
    return CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES


def numeric_features_for(*, include_image_features: bool = True) -> tuple[str, ...]:
    """Return numeric feature columns matching feature_columns_for."""

    if include_image_features:
        return NUMERIC_FEATURES
    return BASE_NUMERIC_FEATURES


def _ensure_optional_image_features(frame: pd.DataFrame) -> None:
    for column in IMAGE_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0


def validate_profile_frame(frame: pd.DataFrame, *, include_image_features: bool = True) -> None:
    validate_feature_policy()
    if include_image_features:
        _ensure_optional_image_features(frame)
    required_fields = {"image_id", TARGET_FIELD, *feature_columns_for(include_image_features=include_image_features)}
    missing = sorted(required_fields - set(frame.columns))
    if missing:
        raise ValueError(f"profile data missing required fields: {missing}")
    if len(frame) < 2:
        raise ValueError("profile data must contain at least two rows")
    for column in numeric_features_for(include_image_features=include_image_features) + (TARGET_FIELD,):
        frame[column] = pd.to_numeric(frame[column], errors="raise")


def records_to_frame(records: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    _ensure_optional_image_features(frame)
    missing = sorted(set(CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"records missing feature fields: {missing}")
    return frame


def split_profile_frame(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    include_image_features: bool = True,
) -> SplitResult:
    """Split by original image when possible to avoid ROI-level data leakage."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    validate_profile_frame(frame, include_image_features=include_image_features)
    image_count = frame["image_id"].nunique()
    if image_count >= 2:
        # Grouping by image_id preserves the project rule that one original
        # image cannot appear in both proxy training and proxy testing.
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_indices, test_indices = next(
            splitter.split(frame, frame[TARGET_FIELD], groups=frame["image_id"])
        )
        train = frame.iloc[train_indices].reset_index(drop=True)
        test = frame.iloc[test_indices].reset_index(drop=True)
        strategy = "group_shuffle_split"
    else:
        # Tiny smoke fixtures often contain a single image, so they cannot
        # satisfy image-level isolation; mark that case explicitly in summary.
        n_test = max(1, int(round(len(frame) * test_size)))
        n_test = min(n_test, len(frame) - 1)
        train = frame.iloc[:-n_test].reset_index(drop=True)
        test = frame.iloc[-n_test:].reset_index(drop=True)
        strategy = "row_split_smoke_only"

    summary = {
        "strategy": strategy,
        "test_size": test_size,
        "seed": seed,
        "train_row_count": len(train),
        "test_row_count": len(test),
        "train_image_count": train["image_id"].nunique(),
        "test_image_count": test["image_id"].nunique(),
    }
    return SplitResult(train=train, test=test, summary=summary)


def validate_explicit_profile_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    include_image_features: bool = True,
) -> dict[str, object]:
    """Validate fixed train/val/test profiling splits without reshuffling rows."""

    for name, frame in (("train", train), ("val", val), ("test", test)):
        validate_profile_frame(frame, include_image_features=include_image_features)
        if frame.empty:
            raise ValueError(f"{name} profile split must not be empty")

    train_images = set(train["image_id"].astype(str))
    val_images = set(val["image_id"].astype(str))
    test_images = set(test["image_id"].astype(str))
    overlaps = {
        "train_val": bool(train_images & val_images),
        "train_test": bool(train_images & test_images),
        "val_test": bool(val_images & test_images),
    }
    if any(overlaps.values()):
        details = {
            "train_val": sorted(train_images & val_images),
            "train_test": sorted(train_images & test_images),
            "val_test": sorted(val_images & test_images),
        }
        raise ValueError(f"explicit profile split image_id overlap detected: {details}")

    return {
        "strategy": "explicit_train_val_test",
        "train_row_count": len(train),
        "val_row_count": len(val),
        "test_row_count": len(test),
        "train_image_count": train["image_id"].nunique(),
        "val_image_count": val["image_id"].nunique(),
        "test_image_count": test["image_id"].nunique(),
        "image_overlaps": overlaps,
    }


def build_quality_proxy_pipeline(
    *,
    model_type: str = MODEL_TYPE_HIST_GBDT,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    hidden_layer_sizes: Sequence[int] = (32, 16),
    max_iter: int = 500,
    seed: int = 42,
) -> Pipeline:
    """Build one compact tabular proxy candidate for online scheduling simulation."""

    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric_features)),
            ("categorical", _dense_one_hot_encoder(), list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    if model_type == MODEL_TYPE_MLP:
        regressor = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            random_state=seed,
            max_iter=max_iter,
            learning_rate_init=1e-3,
        )
    elif model_type == MODEL_TYPE_HIST_GBDT:
        regressor = HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=max_iter,
            learning_rate=0.05,
        )
    elif model_type == MODEL_TYPE_RANDOM_FOREST:
        regressor = RandomForestRegressor(
            n_estimators=64,
            random_state=seed,
            min_samples_leaf=1,
            n_jobs=-1,
        )
    else:
        regressor = ExtraTreesRegressor(
            n_estimators=64,
            random_state=seed,
            min_samples_leaf=1,
            n_jobs=-1,
        )
    return Pipeline([("features", transformer), ("regressor", regressor)])


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else 0.0
    if not np.isfinite(r2):
        r2 = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2),
    }


def threshold_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    true_positive_labels = y_true >= threshold
    predicted_positive_labels = y_pred >= threshold
    tp = int(np.logical_and(true_positive_labels, predicted_positive_labels).sum())
    fp = int(np.logical_and(~true_positive_labels, predicted_positive_labels).sum())
    fn = int(np.logical_and(true_positive_labels, ~predicted_positive_labels).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def high_value_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    threshold: float = 0.7,
) -> dict[str, float]:
    mask = pd.to_numeric(frame["semantic_value"], errors="coerce").fillna(0.0).to_numpy() >= threshold
    if not mask.any():
        return {
            "semantic_value_threshold": float(threshold),
            "count": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "r2": 0.0,
        }
    metrics = regression_metrics(frame[TARGET_FIELD].to_numpy()[mask], predictions[mask])
    return {
        "semantic_value_threshold": float(threshold),
        "count": float(mask.sum()),
        **metrics,
    }


def action_ranking_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    scored = frame.copy()
    scored["_prediction"] = predictions
    agreements = 0
    regrets: list[float] = []
    evaluated = 0
    for _, group in scored.groupby("roi_id", dropna=False):
        if group.empty:
            continue
        oracle_index = group[TARGET_FIELD].astype(float).idxmax()
        predicted_index = group["_prediction"].astype(float).idxmax()
        oracle_quality = float(group.loc[oracle_index, TARGET_FIELD])
        selected_true_quality = float(group.loc[predicted_index, TARGET_FIELD])
        agreements += int(oracle_index == predicted_index)
        regrets.append(max(oracle_quality - selected_true_quality, 0.0))
        evaluated += 1
    return {
        "evaluated_roi_count": float(evaluated),
        "top1_agreement": agreements / max(evaluated, 1),
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "p95_regret": float(np.percentile(regrets, 95)) if regrets else 0.0,
    }


def full_quality_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    quality_threshold: float,
    high_value_threshold: float = 0.7,
) -> dict[str, object]:
    y_true = frame[TARGET_FIELD].to_numpy()
    return {
        **regression_metrics(y_true, predictions),
        "threshold_metrics": threshold_classification_metrics(
            y_true,
            predictions,
            threshold=quality_threshold,
        ),
        "high_value": high_value_metrics(
            frame,
            predictions,
            threshold=high_value_threshold,
        ),
        "action_ranking": action_ranking_metrics(frame, predictions),
    }


def grouped_error_summary(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, list[dict[str, object]]]:
    scored = frame.copy()
    scored["_prediction"] = predictions
    scored["_abs_error"] = (scored[TARGET_FIELD] - scored["_prediction"]).abs()
    scored["_squared_error"] = (scored[TARGET_FIELD] - scored["_prediction"]) ** 2

    summaries: dict[str, list[dict[str, object]]] = {}
    for group_field in ("predicted_class", "exit_level", "compression_level"):
        group_rows: list[dict[str, object]] = []
        for group_value, group in scored.groupby(group_field, dropna=False):
            group_rows.append(
                {
                    group_field: str(group_value),
                    "count": int(len(group)),
                    "mae": float(group["_abs_error"].mean()),
                    "mse": float(group["_squared_error"].mean()),
                }
            )
        summaries[group_field] = group_rows
    return summaries


def save_quality_proxy(path: Path, model: QualityProxyModel) -> None:
    """Persist both weights and the feature contract for reproducible inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model.pipeline,
            "feature_columns": model.feature_columns,
            "categorical_features": model.categorical_features,
            "numeric_features": model.numeric_features,
            "leakage_fields": model.leakage_fields,
            "target_field": model.target_field,
            "model_type": model.model_type,
        },
        path,
    )


def load_quality_proxy(path: Path) -> QualityProxyModel:
    """Load the saved proxy and its feature contract."""

    payload = joblib.load(path)
    return QualityProxyModel(
        pipeline=payload["pipeline"],
        feature_columns=tuple(payload["feature_columns"]),
        categorical_features=tuple(payload["categorical_features"]),
        numeric_features=tuple(payload["numeric_features"]),
        leakage_fields=tuple(payload["leakage_fields"]),
        target_field=payload["target_field"],
        model_type=payload.get("model_type", MODEL_TYPE_MLP),
    )


def train_quality_proxy_smoke(
    profile_csv: Path,
    output_dir: Path,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    max_iter: int = 500,
    hidden_layer_sizes: Sequence[int] = (32, 16),
    model_type: str = MODEL_TYPE_HIST_GBDT,
    include_image_features: bool = True,
    quality_threshold: float = 0.5,
) -> dict[str, object]:
    """Train the proxy and write metrics plus grouped error diagnostics."""

    frame = load_profile_csv(profile_csv, include_image_features=include_image_features)
    split = split_profile_frame(
        frame,
        test_size=test_size,
        seed=seed,
        include_image_features=include_image_features,
    )
    feature_columns = feature_columns_for(include_image_features=include_image_features)
    numeric_features = numeric_features_for(include_image_features=include_image_features)
    pipeline = build_quality_proxy_pipeline(
        model_type=model_type,
        numeric_features=numeric_features,
        categorical_features=CATEGORICAL_FEATURES,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        seed=seed,
    )
    pipeline.fit(split.train[list(feature_columns)], split.train[TARGET_FIELD].to_numpy())

    train_pred = np.clip(pipeline.predict(split.train[list(feature_columns)]), 0.0, 1.0)
    test_pred = np.clip(pipeline.predict(split.test[list(feature_columns)]), 0.0, 1.0)
    model = QualityProxyModel(
        pipeline=pipeline,
        feature_columns=tuple(feature_columns),
        categorical_features=tuple(CATEGORICAL_FEATURES),
        numeric_features=tuple(numeric_features),
        leakage_fields=tuple(LEAKAGE_FIELDS),
        target_field=TARGET_FIELD,
        model_type=model_type,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "quality_proxy.joblib"
    save_quality_proxy(model_path, model)
    summary = {
        "purpose": "Smoke task-quality proxy training; not a formal experiment.",
        "profile_csv": str(profile_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "feature_columns": list(feature_columns),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(numeric_features),
        "leakage_fields": list(LEAKAGE_FIELDS),
        "target_field": TARGET_FIELD,
        "split": split.summary,
        "configuration": {
            "test_size": test_size,
            "seed": seed,
            "max_iter": max_iter,
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "model_type": model_type,
            "include_image_features": include_image_features,
            "quality_threshold": quality_threshold,
        },
        "metrics": {
            "train": full_quality_metrics(
                split.train,
                train_pred,
                quality_threshold=quality_threshold,
            ),
            "test": full_quality_metrics(
                split.test,
                test_pred,
                quality_threshold=quality_threshold,
            ),
        },
        "grouped_errors": grouped_error_summary(split.test, test_pred),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def train_quality_proxy_strict(
    train_profile_csv: Path,
    val_profile_csv: Path,
    test_profile_csv: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    max_iter: int = 500,
    hidden_layer_sizes: Sequence[int] = (32, 16),
    model_type: str = MODEL_TYPE_HIST_GBDT,
    include_image_features: bool = True,
    quality_threshold: float = 0.5,
) -> dict[str, object]:
    """Train on an explicit train split and report validation/test metrics separately."""

    train_frame = load_profile_csv(train_profile_csv, include_image_features=include_image_features)
    val_frame = load_profile_csv(val_profile_csv, include_image_features=include_image_features)
    test_frame = load_profile_csv(test_profile_csv, include_image_features=include_image_features)
    split_summary = validate_explicit_profile_splits(
        train_frame,
        val_frame,
        test_frame,
        include_image_features=include_image_features,
    )

    feature_columns = feature_columns_for(include_image_features=include_image_features)
    numeric_features = numeric_features_for(include_image_features=include_image_features)
    pipeline = build_quality_proxy_pipeline(
        model_type=model_type,
        numeric_features=numeric_features,
        categorical_features=CATEGORICAL_FEATURES,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        seed=seed,
    )
    pipeline.fit(train_frame[list(feature_columns)], train_frame[TARGET_FIELD].to_numpy())

    train_pred = np.clip(pipeline.predict(train_frame[list(feature_columns)]), 0.0, 1.0)
    val_pred = np.clip(pipeline.predict(val_frame[list(feature_columns)]), 0.0, 1.0)
    test_pred = np.clip(pipeline.predict(test_frame[list(feature_columns)]), 0.0, 1.0)
    model = QualityProxyModel(
        pipeline=pipeline,
        feature_columns=tuple(feature_columns),
        categorical_features=tuple(CATEGORICAL_FEATURES),
        numeric_features=tuple(numeric_features),
        leakage_fields=tuple(LEAKAGE_FIELDS),
        target_field=TARGET_FIELD,
        model_type=model_type,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "quality_proxy.joblib"
    save_quality_proxy(model_path, model)
    summary = {
        "purpose": "Strict task-quality proxy training with explicit train/val/test profiling splits.",
        "profile_csvs": {
            "train": str(train_profile_csv.resolve()),
            "val": str(val_profile_csv.resolve()),
            "test": str(test_profile_csv.resolve()),
        },
        "output_dir": str(output_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "model_size_bytes": model_path.stat().st_size,
        "feature_columns": list(feature_columns),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(numeric_features),
        "leakage_fields": list(LEAKAGE_FIELDS),
        "target_field": TARGET_FIELD,
        "split": split_summary,
        "configuration": {
            "seed": seed,
            "max_iter": max_iter,
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "model_type": model_type,
            "include_image_features": include_image_features,
            "quality_threshold": quality_threshold,
        },
        "metrics": {
            "train": full_quality_metrics(
                train_frame,
                train_pred,
                quality_threshold=quality_threshold,
            ),
            "val": full_quality_metrics(
                val_frame,
                val_pred,
                quality_threshold=quality_threshold,
            ),
            "test": full_quality_metrics(
                test_frame,
                test_pred,
                quality_threshold=quality_threshold,
            ),
        },
        "grouped_errors": {
            "val": grouped_error_summary(val_frame, val_pred),
            "test": grouped_error_summary(test_frame, test_pred),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
