# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""profile_data: column statistics and a list of data issues for the LLM to read in step 2.

The report is compact on purpose (it goes into the LLM context): per-column stats plus `issues`, each with
a machine-readable kind so the LLM can map issues to cleaning rules without parsing prose.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dataengineer.tools.model_tools.spec import TaskSpec, _jsonable

HIGH_MISSING = 0.5
ID_UNIQUE_RATIO = 0.95
HIGH_CARDINALITY = 50
OUTLIER_SHARE = 0.01


def profile_frame(df: pd.DataFrame, spec: Optional[TaskSpec] = None, max_columns: int = 80) -> Dict:
    n = len(df)
    issues: List[Dict] = []
    columns: Dict[str, Dict] = {}
    for col in list(df.columns)[:max_columns]:
        s = df[col]
        info: Dict = {"dtype": str(s.dtype), "missing_rate": float(s.isna().mean()), "nunique": int(s.nunique(dropna=True))}
        nonnull = s.dropna()
        if pd.api.types.is_bool_dtype(s):
            info["true_rate"] = float(nonnull.mean()) if len(nonnull) else None
        elif pd.api.types.is_numeric_dtype(s):
            if len(nonnull):
                q1, med, q3 = nonnull.quantile([0.25, 0.5, 0.75]).tolist()
                info.update({"min": nonnull.min(), "median": med, "max": nonnull.max(), "mean": nonnull.mean(),
                             "zero_rate": float((nonnull == 0).mean()), "negative_rate": float((nonnull < 0).mean())})
                iqr = q3 - q1
                if iqr > 0:
                    share = float(((nonnull < q1 - 3 * iqr) | (nonnull > q3 + 3 * iqr)).mean())
                    info["outlier_rate_3iqr"] = share
                    if share > OUTLIER_SHARE:
                        issues.append({"kind": "outliers", "column": col, "detail": f"{share:.1%} beyond 3*IQR"})
        elif pd.api.types.is_datetime64_any_dtype(s):
            if len(nonnull):
                info.update({"min": str(nonnull.min()), "max": str(nonnull.max())})
        else:
            info["top"] = {str(k): int(v) for k, v in nonnull.astype(str).value_counts().head(5).items()}
            stripped = nonnull.astype(str).str.strip()
            blank = float((stripped.isin(["", "?", "NA", "N/A", "null", "None"])).mean()) if len(nonnull) else 0.0
            if blank > 0:
                info["placeholder_missing_rate"] = blank
                issues.append({"kind": "placeholder_missing", "column": col, "detail": f"{blank:.1%} values like '?' or ''"})
            numeric_like = pd.to_numeric(stripped, errors="coerce").notna().mean() if len(nonnull) else 0
            if numeric_like > 0.9:
                issues.append({"kind": "numeric_as_text", "column": col, "detail": f"{numeric_like:.0%} parse as numbers"})
        if info["missing_rate"] > HIGH_MISSING:
            issues.append({"kind": "high_missing", "column": col, "detail": f"{info['missing_rate']:.0%} missing"})
        elif info["missing_rate"] > 0:
            issues.append({"kind": "missing", "column": col, "detail": f"{info['missing_rate']:.1%} missing"})
        if info["nunique"] <= 1:
            issues.append({"kind": "constant", "column": col, "detail": "single value"})
        if n > 50 and info["nunique"] >= ID_UNIQUE_RATIO * n and not pd.api.types.is_float_dtype(s):
            issues.append({"kind": "id_like", "column": col, "detail": f"{info['nunique']} distinct of {n} rows"})
        elif s.dtype == object and info["nunique"] > HIGH_CARDINALITY:
            issues.append({"kind": "high_cardinality", "column": col, "detail": f"{info['nunique']} categories"})
        columns[col] = info

    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        issues.append({"kind": "duplicate_rows", "column": None, "detail": f"{dup_rows} fully duplicated rows"})
    report: Dict = {"n_rows": n, "n_columns": int(df.shape[1]), "duplicate_rows": dup_rows, "columns": columns}
    if df.shape[1] > max_columns:
        report["truncated_columns"] = int(df.shape[1] - max_columns)

    if spec is not None:
        if spec.target in df.columns:
            y = df[spec.target]
            if spec.is_classification:
                report["target"] = {str(k): int(v) for k, v in y.value_counts(dropna=False).items()}
            else:
                report["target"] = {"mean": y.mean(), "std": y.std(), "min": y.min(), "max": y.max(),
                                    "zero_rate": float((y == 0).mean()), "skew": float(y.skew())}
                if abs(y.skew()) > 2:
                    issues.append({"kind": "skewed_target", "column": spec.target, "detail": f"skew {y.skew():.1f}"})
        keys = [c for c in (spec.entity, spec.time_col) if c and c in df.columns]
        if keys and spec.task_type == "timeseries":
            dk = int(df.duplicated(keys).sum())
            if dk:
                issues.append({"kind": "duplicate_keys", "column": ",".join(keys), "detail": f"{dk} duplicate rows"})
        if spec.time_col and spec.time_col in df.columns:
            report["time"] = _time_coverage(df, spec)
            if report["time"].get("gaps"):
                issues.append({"kind": "time_gaps", "column": spec.time_col, "detail": f"{report['time']['gaps']} missing periods"})
    report["issues"] = issues
    return _jsonable(report)


def _time_coverage(df: pd.DataFrame, spec: TaskSpec) -> Dict:
    t = pd.to_datetime(df[spec.time_col], errors="coerce")
    out: Dict = {"min": str(t.min()), "max": str(t.max()), "unparseable": int(t.isna().sum())}
    freq = (spec.freq or "").upper()[:1]
    if freq in ("H", "D", "W", "M"):
        pfreq = {"H": "h", "D": "D", "W": "W", "M": "M"}[freq]
        periods = t.dropna().dt.to_period(pfreq)
        if len(periods):
            full = pd.period_range(periods.min(), periods.max(), freq=pfreq)
            present = periods.unique()
            out["periods"] = int(len(full))
            out["gaps"] = int(len(full) - len(present))
            if spec.entity and spec.entity in df.columns:
                per_series = df.assign(_p=periods).groupby(spec.entity)["_p"].nunique()
                out["series"] = int(per_series.size)
                out["periods_per_series"] = {"min": int(per_series.min()), "median": float(per_series.median()),
                                             "max": int(per_series.max())}
    return out
