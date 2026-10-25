# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""clean_data: execute LLM-proposed cleaning rules from a fixed vocabulary and log their effect.

The LLM never writes pandas code; it emits a list of rules like
{"op": "drop_rows", "column": "Quantity", "cmp": "<=", "value": 0, "reason": "returns"}.
Each rule is applied in order and logged with rows before/after and cells changed.

Statistical imputation (median/mean) and quantile clipping are deliberately absent: statistics fitted on
the full table leak test information into training, so they belong inside the training pipeline. Cleaning
only does row/column-level fixes and constant fills.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

CMP = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "in": lambda s, v: s.isin(v),
    "not_in": lambda s, v: ~s.isin(v),
    "isnull": lambda s, v: s.isna(),
    "notnull": lambda s, v: s.notna(),
    "startswith": lambda s, v: s.astype(str).str.startswith(str(v)),
    "contains": lambda s, v: s.astype(str).str.contains(str(v), regex=False),
}
OPS = ("drop_duplicates", "drop_columns", "drop_rows", "replace_values", "strip_strings", "cast",
       "fill_missing", "add_missing_indicator", "clip", "derive")


class CleaningError(ValueError):
    pass


def _need(rule: Dict, *keys: str) -> None:
    missing = [k for k in keys if k not in rule]
    if missing:
        raise CleaningError(f"rule {rule.get('op')!r} missing {missing}")


def _col(df: pd.DataFrame, rule: Dict) -> str:
    _need(rule, "column")
    if rule["column"] not in df.columns:
        raise CleaningError(f"column {rule['column']!r} not found")
    return rule["column"]


def apply_rule(df: pd.DataFrame, rule: Dict) -> Tuple[pd.DataFrame, Dict]:
    op = rule.get("op")
    before = len(df)
    changed = 0
    if op == "drop_duplicates":
        subset = rule.get("subset") or [c for c in df.columns if not str(c).startswith("__")]
        df = df.drop_duplicates(subset=subset)
    elif op == "drop_columns":
        _need(rule, "columns")
        cols = [c for c in rule["columns"] if c in df.columns]
        df = df.drop(columns=cols)
        changed = len(cols)
    elif op == "drop_rows":
        col = _col(df, rule)
        cmp = rule.get("cmp", "==")
        if cmp not in CMP:
            raise CleaningError(f"unknown cmp {cmp!r}; use one of {list(CMP)}")
        mask = CMP[cmp](df[col], rule.get("value")).fillna(False).astype(bool)
        df = df[~mask]
    elif op == "replace_values":
        col = _col(df, rule)
        _need(rule, "mapping")
        mapping = {k: (np.nan if v is None else v) for k, v in rule["mapping"].items()}
        new = df[col].replace(mapping)
        changed = int((new.astype(str) != df[col].astype(str)).sum())
        df = df.assign(**{col: new})
    elif op == "strip_strings":
        cols = rule.get("columns") or [c for c in df.columns if df[c].dtype == object]
        for c in cols:
            if c in df.columns and df[c].dtype == object:
                new = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
                changed += int((new != df[c]).sum())
                df = df.assign(**{c: new})
    elif op == "cast":
        col = _col(df, rule)
        dtype = rule.get("dtype")
        s = df[col]
        if dtype == "numeric":
            new = pd.to_numeric(s, errors="coerce")
        elif dtype == "datetime":
            new = pd.to_datetime(s, errors="coerce")
        elif dtype == "string":
            new = s.astype("string").astype(object).where(s.notna(), None)
        elif dtype == "category":
            new = s.astype(str).where(s.notna(), None)
        else:
            raise CleaningError(f"cast dtype must be numeric/datetime/string/category, got {dtype!r}")
        changed = int(new.isna().sum() - s.isna().sum())  # values that failed to parse
        df = df.assign(**{col: new})
    elif op == "fill_missing":
        col = _col(df, rule)
        _need(rule, "value")
        changed = int(df[col].isna().sum())
        df = df.assign(**{col: df[col].fillna(rule["value"])})
    elif op == "add_missing_indicator":
        col = _col(df, rule)
        df = df.assign(**{f"{col}_is_missing": df[col].isna().astype(int)})
        changed = int(df[col].isna().sum())
    elif op == "clip":
        col = _col(df, rule)
        lo, hi = rule.get("lower"), rule.get("upper")
        if lo is None and hi is None:
            raise CleaningError("clip needs lower and/or upper (fixed values read from the profile)")
        new = df[col].clip(lower=lo, upper=hi)
        changed = int((new != df[col]).sum())
        df = df.assign(**{col: new})
    elif op == "derive":
        _need(rule, "name", "expr")
        try:
            new = df.eval(rule["expr"], engine="python")
        except Exception as e:
            raise CleaningError(f"derive expr failed: {e}") from e
        df = df.assign(**{rule["name"]: new})
        changed = len(df)
    else:
        raise CleaningError(f"unknown op {op!r}; allowed: {OPS}")
    return df, {"op": op, "column": rule.get("column") or rule.get("columns") or rule.get("name"),
                "rows_before": before, "rows_after": len(df), "rows_removed": before - len(df),
                "cells_changed": changed, "reason": rule.get("reason", "")}


def apply_rules(df: pd.DataFrame, rules: List[Dict[str, Any]], protected: List[str] = ()) -> Tuple[pd.DataFrame, List[Dict]]:
    """Apply rules in order. Rules that would drop a protected column (target, entity, time) are refused."""
    log: List[Dict] = []
    for rule in rules:
        if rule.get("op") == "drop_columns" and set(rule.get("columns", [])) & set(protected):
            raise CleaningError(f"cannot drop protected columns {sorted(set(rule['columns']) & set(protected))}")
        df, entry = apply_rule(df, rule)
        log.append(entry)
    total = log[0]["rows_before"] if log else len(df)
    if total and len(df) < 0.5 * total:
        log.append({"op": "warning", "detail": f"cleaning removed {1 - len(df) / total:.0%} of rows; check the rules"})
    return df.reset_index(drop=True), log
