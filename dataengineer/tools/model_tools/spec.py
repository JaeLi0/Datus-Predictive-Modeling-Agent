# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""TaskSpec: the structured modeling task the LLM decides in step 1 (define_task).

The LLM fills the spec; code validates it against the data (columns exist, the target matches the task
type, the time setup is coherent) and fills safe defaults for the primary metric and baseline. Every later
tool reads the spec instead of re-asking the LLM, so a run is reproducible from spec + rules + templates.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

TaskType = Literal["binary", "multiclass", "regression", "timeseries"]

DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse", "timeseries": "wape"}
ALLOWED_METRICS = {
    "binary": {"roc_auc", "pr_auc", "log_loss", "f1"},
    "multiclass": {"log_loss", "accuracy", "macro_f1"},
    "regression": {"rmse", "mae", "r2", "mape"},
    "timeseries": {"wape", "mase", "mae", "rmse"},
}
DEFAULT_BASELINE = {
    "binary": "prior",
    "multiclass": "prior",
    "regression": "mean",
    "timeseries": "seasonal_naive",
}


class TaskSpec(BaseModel):
    """What to predict, for whom, as of when, and how success is measured."""

    name: str = Field(description="Short task id, e.g. adult_income")
    task_type: TaskType
    target: str = Field(description="Target column in the modeling table")
    entity: Optional[str] = Field(default=None, description="Entity id column (customer, series id ...)")
    time_col: Optional[str] = Field(default=None, description="Event/observation time column")
    cutoff: Optional[str] = Field(
        default=None,
        description="Prediction time point (ISO date). Features may only use data strictly before it; "
        "the label window starts at it. Required for point-in-time tasks built from event tables.",
    )
    cutoffs: List[str] = Field(
        default_factory=list,
        description="Several prediction time points for point-in-time snapshots (one row per entity per cutoff); "
        "the latest becomes the out-of-time test set.",
    )
    horizon: Optional[int] = Field(
        default=None, description="Forecast horizon in periods (time series), or label window in days (snapshots)"
    )
    freq: Optional[str] = Field(default=None, description="Series frequency for time series: H, D, W")
    season_length: Optional[int] = Field(default=None, description="Seasonal period in freq units, e.g. 7 for D")
    positive_label: Optional[str] = Field(default=None, description="Positive class value for binary tasks")
    primary_metric: Optional[str] = None
    baseline: Optional[str] = None
    drop_columns: List[str] = Field(
        default_factory=list, description="Ids/free text/post-outcome columns that must never be features"
    )
    notes: str = ""

    @field_validator("positive_label", mode="before")
    @classmethod
    def _label_as_text(cls, v):
        # LLMs often send true/1 for a boolean/int target; class values are compared as text
        return None if v is None else str(v)

    @model_validator(mode="after")
    def _defaults(self):
        if self.primary_metric is None:
            self.primary_metric = DEFAULT_METRIC[self.task_type]
        if self.primary_metric not in ALLOWED_METRICS[self.task_type]:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} not valid for {self.task_type}; "
                f"choose from {sorted(ALLOWED_METRICS[self.task_type])}"
            )
        if self.baseline is None:
            self.baseline = DEFAULT_BASELINE[self.task_type]
        if self.task_type == "timeseries":
            missing = [k for k in ("time_col", "horizon", "freq") if getattr(self, k) in (None, "")]
            if missing:
                raise ValueError(f"time series task needs {missing}")
            if self.season_length is None:
                self.season_length = {"H": 24, "D": 7, "W": 52, "M": 12}.get(self.freq.upper()[:1], 1)
        return self

    @property
    def is_classification(self) -> bool:
        return self.task_type in ("binary", "multiclass")


def validate_spec(spec: TaskSpec, df: pd.DataFrame) -> Dict:
    """Check the spec against the data. Returns {errors, warnings, target_summary}; errors block the run."""
    errors: List[str] = []
    warnings: List[str] = []
    cols = set(df.columns)
    for field_name in ("target", "entity", "time_col"):
        col = getattr(spec, field_name)
        if col and col not in cols:
            errors.append(f"{field_name} column {col!r} not in data")
    for c in spec.drop_columns:
        if c not in cols:
            warnings.append(f"drop column {c!r} not in data")
    if errors:
        return {"errors": errors, "warnings": warnings, "target_summary": None}

    y = df[spec.target]
    summary: Dict = {"n_rows": int(len(df)), "missing": int(y.isna().sum())}
    if y.isna().all():
        errors.append("target is entirely missing")
    nunique = int(y.nunique(dropna=True))
    if spec.task_type == "binary":
        vc = y.value_counts(dropna=True)
        summary["classes"] = {str(k): int(v) for k, v in vc.items()}
        if nunique != 2:
            errors.append(f"binary task but target has {nunique} distinct values")
        else:
            pos = spec.positive_label
            if pos is None:
                pos = str(vc.index[-1])  # minority class
                spec.positive_label = pos
                warnings.append(f"positive_label not given; using minority class {pos!r}")
            elif pos not in {str(k) for k in vc.index}:
                errors.append(f"positive_label {pos!r} not among target values {list(summary['classes'])}")
            rate = float((y.astype(str) == spec.positive_label).mean())
            summary["positive_rate"] = round(rate, 4)
            if rate < 0.2 and spec.primary_metric == "roc_auc":
                warnings.append(f"positive rate {rate:.1%} is low; report pr_auc alongside roc_auc")
    elif spec.task_type == "multiclass":
        summary["classes"] = {str(k): int(v) for k, v in y.value_counts().head(20).items()}
        if nunique < 3:
            errors.append(f"multiclass task but target has {nunique} distinct values")
    else:
        if not pd.api.types.is_numeric_dtype(y):
            errors.append(f"{spec.task_type} target must be numeric, got {y.dtype}")
        else:
            q = y.quantile([0, 0.01, 0.5, 0.99, 1]).tolist()
            summary.update({"min": q[0], "p01": q[1], "median": q[2], "p99": q[3], "max": q[4], "mean": float(y.mean())})
            if nunique <= 2:
                errors.append("regression target has <= 2 distinct values; is this a classification task?")
            if spec.primary_metric == "mape" and (y == 0).any():
                errors.append("mape is undefined with zero targets; use mae or wape")
            if (y < 0).any():
                warnings.append("target has negative values")

    if spec.time_col and spec.time_col in cols:
        t = pd.to_datetime(df[spec.time_col], errors="coerce")
        if t.isna().mean() > 0.01:
            errors.append(f"time column {spec.time_col!r} has {t.isna().mean():.1%} unparseable values")
        else:
            summary["time_range"] = [str(t.min()), str(t.max())]
            if spec.cutoff and not (t.min() <= pd.Timestamp(spec.cutoff) <= t.max()):
                warnings.append(f"cutoff {spec.cutoff} outside data time range {summary['time_range']}")
    if spec.task_type == "timeseries" and spec.entity and spec.entity in cols and spec.time_col in cols:
        dup = int(df.duplicated([spec.entity, spec.time_col]).sum())
        if dup:
            errors.append(f"{dup} duplicate (entity, time) rows; aggregate to one row per series per period first")
    for c in cols - {spec.target}:
        if c in spec.drop_columns or str(c).startswith("__"):
            continue
        s = df[c]
        if s.dtype == object and s.nunique() > 0.9 * len(s) and len(s) > 50:
            warnings.append(f"{c!r} looks like an id or free text (unique ratio {s.nunique() / len(s):.2f}); consider drop_columns")
    return {"errors": errors, "warnings": warnings, "target_summary": _jsonable(summary)}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    return obj
