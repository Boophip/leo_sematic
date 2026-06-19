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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_FIELD = "task_quality"

# Keep this list restricted to values that are available online or available
# from offline cost tables. Do not add supervision-only diagnostics here.
CATEGORICAL_FEATURES = (
    "predicted_class",
    "compression_level",
)

NUMERIC_FEATURES = (
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

    def predict_quality(self, records: Iterable[Mapping[str, object]] | pd.DataFrame) -> np.ndarray:
        """Predict clipped task quality for candidate scheduling actions."""

        frame = records_to_frame(records)
        predictions = self.pipeline.predict(frame[list(self.feature_columns)])
        return np.clip(predictions, 0.0, 1.0)


def predict_quality(
    model: QualityProxyModel,
    records: Iterable[Mapping[str, object]] | pd.DataFrame,
) -> np.ndarray:
    return model.predict_quality(records)


def load_profile_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_profile_frame(frame)
    return frame


def validate_feature_policy() -> None:
    """Guard against accidentally training on labels or model-output diagnostics."""

    overlap = sorted(set(FEATURE_COLUMNS) & set(LEAKAGE_FIELDS))
    if overlap:
        raise ValueError(f"feature columns contain leakage fields: {overlap}")


def validate_profile_frame(frame: pd.DataFrame) -> None:
    validate_feature_policy()
    missing = sorted(set(REQUIRED_PROFILE_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"profile data missing required fields: {missing}")
    if len(frame) < 2:
        raise ValueError("profile data must contain at least two rows")
    for column in NUMERIC_FEATURES + (TARGET_FIELD,):
        frame[column] = pd.to_numeric(frame[column], errors="raise")


def records_to_frame(records: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"records missing feature fields: {missing}")
    return frame


def split_profile_frame(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> SplitResult:
    """Split by original image when possible to avoid ROI-level data leakage."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    validate_profile_frame(frame)
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


def build_quality_proxy_pipeline(
    *,
    hidden_layer_sizes: Sequence[int] = (32, 16),
    max_iter: int = 500,
    seed: int = 42,
) -> Pipeline:
    """Build the compact tabular MLP used by online scheduling simulation."""

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    regressor = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        random_state=seed,
        max_iter=max_iter,
        learning_rate_init=1e-3,
    )
    return Pipeline([("features", transformer), ("regressor", regressor)])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else 0.0
    if not np.isfinite(r2):
        r2 = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2),
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
    )


def train_quality_proxy_smoke(
    profile_csv: Path,
    output_dir: Path,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    max_iter: int = 500,
    hidden_layer_sizes: Sequence[int] = (32, 16),
) -> dict[str, object]:
    """Train the proxy and write metrics plus grouped error diagnostics."""

    frame = load_profile_csv(profile_csv)
    split = split_profile_frame(frame, test_size=test_size, seed=seed)
    pipeline = build_quality_proxy_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        seed=seed,
    )
    pipeline.fit(split.train[list(FEATURE_COLUMNS)], split.train[TARGET_FIELD].to_numpy())

    train_pred = np.clip(pipeline.predict(split.train[list(FEATURE_COLUMNS)]), 0.0, 1.0)
    test_pred = np.clip(pipeline.predict(split.test[list(FEATURE_COLUMNS)]), 0.0, 1.0)
    model = QualityProxyModel(
        pipeline=pipeline,
        feature_columns=tuple(FEATURE_COLUMNS),
        categorical_features=tuple(CATEGORICAL_FEATURES),
        numeric_features=tuple(NUMERIC_FEATURES),
        leakage_fields=tuple(LEAKAGE_FIELDS),
        target_field=TARGET_FIELD,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "quality_proxy.joblib"
    save_quality_proxy(model_path, model)
    summary = {
        "purpose": "Smoke task-quality proxy training; not a formal experiment.",
        "profile_csv": str(profile_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "leakage_fields": list(LEAKAGE_FIELDS),
        "target_field": TARGET_FIELD,
        "split": split.summary,
        "configuration": {
            "test_size": test_size,
            "seed": seed,
            "max_iter": max_iter,
            "hidden_layer_sizes": list(hidden_layer_sizes),
        },
        "metrics": {
            "train": regression_metrics(split.train[TARGET_FIELD].to_numpy(), train_pred),
            "test": regression_metrics(split.test[TARGET_FIELD].to_numpy(), test_pred),
        },
        "grouped_errors": grouped_error_summary(split.test, test_pred),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
