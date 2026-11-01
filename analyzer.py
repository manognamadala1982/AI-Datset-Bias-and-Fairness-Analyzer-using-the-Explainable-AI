"""Model training and fairness analysis for the crime dataset demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_FACTORIES = {
    "Logistic regression": lambda: LogisticRegression(
        max_iter=1500, class_weight="balanced", random_state=42
    ),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra trees": lambda: ExtraTreesClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


@dataclass
class DatasetInfo:
    data: pd.DataFrame
    original_rows: int
    duplicate_rows: int


def load_dataset(file) -> DatasetInfo:
    """Load a CSV, normalize text, infer numeric columns, and remove duplicates."""
    data = pd.read_csv(file, encoding="utf-8-sig")
    data.columns = [str(column).strip() for column in data.columns]
    data = data.loc[:, ~data.columns.duplicated()].copy()

    for column in data.select_dtypes(include=["object", "string"]).columns:
        values = data[column].astype("string").str.strip()
        values = values.replace("", pd.NA)
        numeric = pd.to_numeric(values, errors="coerce")
        non_missing = int(values.notna().sum())
        if non_missing and numeric.notna().sum() / non_missing >= 0.9:
            data[column] = numeric
        else:
            data[column] = values

    original_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    return DatasetInfo(
        data=data,
        original_rows=original_rows,
        duplicate_rows=original_rows - len(data),
    )


def _make_pipeline(features: pd.DataFrame, model_name: str) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", MODEL_FACTORIES[model_name]()),
        ]
    )


def train_and_compare(
    data: pd.DataFrame,
    target: str,
    sensitive_columns: list[str],
    model_names: list[str],
    positive_label: Any,
    minimum_group_size: int = 30,
) -> dict[str, Any]:
    """Train selected classifiers and calculate test-set performance and fairness."""
    if target not in data:
        raise ValueError(f"Target column {target!r} was not found.")
    if not model_names:
        raise ValueError("Select at least one model.")
    if not sensitive_columns:
        raise ValueError("Select at least one sensitive attribute.")

    usable = data.dropna(subset=[target]).copy()
    if usable[target].nunique() < 2:
        raise ValueError("The target needs at least two classes to train a model.")
    if usable[target].value_counts().min() < 2:
        raise ValueError(
            "At least one target class has fewer than two rows after duplicate "
            "removal. Choose another target or obtain more representative data."
        )

    sensitive_columns = [column for column in sensitive_columns if column in usable]
    if not sensitive_columns:
        raise ValueError("None of the selected sensitive columns are present.")

    y = usable[target].astype(str)
    sensitive = usable[sensitive_columns].copy()

    # Sensitive attributes are intentionally held out from model inputs and used
    # only to evaluate group outcomes.
    features = usable.drop(columns=[target, *sensitive_columns]).copy()
    constant_columns = [
        column for column in features if features[column].nunique(dropna=False) <= 1
    ]
    features = features.drop(columns=constant_columns)
    if features.empty:
        raise ValueError("No usable input features remain after excluding the target.")

    class_counts = y.value_counts()
    test_count = int(np.ceil(len(y) * 0.25))
    if test_count < y.nunique() or len(y) - test_count < y.nunique():
        raise ValueError("There are too few rows to place every class in both splits.")

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = (
        train_test_split(
            features,
            y,
            sensitive,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    )

    if str(positive_label) not in set(y):
        positive_label = y.value_counts().index[0]
    positive_label = str(positive_label)

    results: dict[str, Any] = {
        "target": target,
        "positive_label": positive_label,
        "class_counts": class_counts,
        "excluded_constant_columns": constant_columns,
        "comparison": [],
        "models": {},
        "fairness": {},
        "test_size": len(y_test),
    }

    for model_name in model_names:
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Unsupported model: {model_name}")

        model = _make_pipeline(X_train, model_name)
        model.fit(X_train, y_train)
        predictions = pd.Series(model.predict(X_test), index=y_test.index)
        result_metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
            "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        results["comparison"].append(result_metrics)

        importance = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="balanced_accuracy",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        importance_frame = pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": importance.importances_mean,
                "Std. dev.": importance.importances_std,
            }
        ).sort_values("Importance", ascending=False, ignore_index=True)

        results["models"][model_name] = {
            "pipeline": model,
            "y_test": y_test,
            "predictions": predictions,
            "importance": importance_frame,
        }

        for sensitive_column in sensitive_columns:
            results["fairness"].setdefault(sensitive_column, {})[model_name] = (
                group_fairness_metrics(
                    y_test=y_test,
                    predictions=predictions,
                    groups=sensitive_test[sensitive_column],
                    positive_label=positive_label,
                    minimum_group_size=minimum_group_size,
                )
            )

    results["comparison"] = pd.DataFrame(results["comparison"]).sort_values(
        "Balanced accuracy", ascending=False, ignore_index=True
    )
    return results


def group_fairness_metrics(
    y_test: pd.Series,
    predictions: pd.Series,
    groups: pd.Series,
    positive_label: str,
    minimum_group_size: int,
) -> dict[str, Any]:
    """Calculate selection rate, TPR, and FPR by group and their max-min gaps."""
    actual_positive = y_test.astype(str) == str(positive_label)
    predicted_positive = predictions.astype(str) == str(positive_label)
    group_values = groups.fillna("Missing").astype(str)
    rows = []

    for group_name in sorted(group_values.unique()):
        mask = group_values == group_name
        count = int(mask.sum())
        actual = actual_positive.loc[mask]
        predicted = predicted_positive.loc[mask]
        true_positive = int((actual & predicted).sum())
        false_positive = int((~actual & predicted).sum())
        actual_positive_count = int(actual.sum())
        actual_negative_count = int((~actual).sum())
        eligible = count >= minimum_group_size
        rows.append(
            {
                "Group": group_name,
                "Samples": count,
                "Selection rate": float(predicted.mean()),
                "True positive rate": (
                    true_positive / actual_positive_count
                    if eligible and actual_positive_count
                    else np.nan
                ),
                "False positive rate": (
                    false_positive / actual_negative_count
                    if eligible and actual_negative_count
                    else np.nan
                ),
                "Support": (
                    "Included"
                    if eligible
                    else f"Below minimum ({minimum_group_size})"
                ),
            }
        )

    table = pd.DataFrame(rows)
    # Small groups remain visible in the table but do not influence disparity
    # summaries, which would otherwise be dominated by noisy estimates.
    eligible = table["Support"] == "Included"
    eligible_rates = table.loc[eligible]
    gaps = {}
    for metric in ("Selection rate", "True positive rate", "False positive rate"):
        rates = eligible_rates[metric].dropna()
        gaps[metric] = float(rates.max() - rates.min()) if len(rates) >= 2 else np.nan

    gaps["Selection-rate parity"] = selection_rate_parity(table)

    return {"by_group": table, "gaps": gaps}


def selection_rate_parity(group_metrics: pd.DataFrame) -> float:
    """Return min/max selection-rate ratio for groups marked as supported."""
    required_columns = {"Support", "Selection rate"}
    if not required_columns.issubset(group_metrics.columns):
        return float("nan")

    supported = group_metrics.loc[group_metrics["Support"] == "Included"]
    rates = pd.to_numeric(supported["Selection rate"], errors="coerce").dropna()
    if len(rates) < 2 or rates.max() <= 0:
        return float("nan")
    return float(rates.min() / rates.max())
