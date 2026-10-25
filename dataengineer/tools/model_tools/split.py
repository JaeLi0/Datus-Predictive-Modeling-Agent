# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""make_split: holdout test set plus cross-validation folds inside the training part.

Methods
- random / stratified: i.i.d. rows (stratified keeps the class ratio)
- group: an entity never appears on both sides (GroupKFold-style)
- time: out-of-time holdout (latest rows are test), expanding-window CV inside train
- rolling_origin: time-series backtest, each fold's validation is the next `horizon` periods after an origin
- snapshot: point-in-time snapshots; the last cutoff is test, earlier cutoffs rotate as validation

Guardrail (enforce=True): the method must fit the spec. Time series must use rolling_origin, tasks with a
time column must split by time or snapshot, and tasks whose entity repeats must at least group by entity.
The no-guardrail ablation passes enforce=False.

Output: a frame aligned with the modeling table with column `role` (train/test) and one column per CV fold
`cv<k>` with values train/valid/'' (unused).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataengineer.tools.model_tools.spec import TaskSpec

METHODS = ("random", "stratified", "group", "time", "rolling_origin", "snapshot")


class SplitError(ValueError):
    pass


def required_methods(spec: TaskSpec, df: pd.DataFrame) -> Tuple[List[str], str]:
    """Methods allowed by the guardrail, with the reason."""
    if spec.task_type == "timeseries":
        return ["rolling_origin"], "time series: validate by backtesting on later periods only"
    if "snapshot" in df.columns:
        return ["snapshot", "time"], "point-in-time snapshots: test on the latest cutoff"
    if spec.time_col and spec.time_col in df.columns:
        return ["time", "rolling_origin"], "rows carry a time column: test must be later than train"
    if spec.entity and spec.entity in df.columns and df[spec.entity].duplicated().any():
        return ["group"], f"entity {spec.entity!r} repeats across rows: keep each entity on one side"
    if spec.is_classification:
        return ["stratified", "random", "group"], "i.i.d. rows"
    return ["random", "group"], "i.i.d. rows"


def make_split(df: pd.DataFrame, spec: TaskSpec, method: str, test_size: float = 0.2, n_folds: int = 5,
               seed: int = 0, enforce: bool = True, gap: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    if method not in METHODS:
        raise SplitError(f"unknown split method {method!r}; allowed {METHODS}")
    allowed, why = required_methods(spec, df)
    if enforce and method not in allowed:
        raise SplitError(f"split {method!r} not allowed for this task ({why}); use one of {allowed}")
    n = len(df)
    rng = np.random.RandomState(seed)
    role = np.array(["train"] * n, dtype=object)
    folds: List[np.ndarray] = []  # each: array of 'train'/'valid'/'' per row
    info: Dict = {"method": method, "test_size": test_size, "n_folds": n_folds, "seed": seed, "guardrail_reason": why}

    if method in ("random", "stratified", "group"):
        if method == "group":
            if not spec.entity:
                raise SplitError("group split needs spec.entity")
            groups = df[spec.entity].astype(str).to_numpy()
            uniq = np.unique(groups)
            rng.shuffle(uniq)
            n_test = max(1, int(round(len(uniq) * test_size)))
            test_groups = set(uniq[:n_test])
            role[np.isin(groups, list(test_groups))] = "test"
            train_groups = uniq[n_test:]
            fold_of = {g: i % n_folds for i, g in enumerate(train_groups)}
            fid = np.array([fold_of.get(g, -1) for g in groups])
        else:
            idx = np.arange(n)
            if method == "stratified":
                y = df[spec.target].astype(str).to_numpy()
                test_idx, fid = [], np.full(n, -1)
                for cls in np.unique(y):
                    cidx = idx[y == cls]
                    rng.shuffle(cidx)
                    k = int(round(len(cidx) * test_size))
                    test_idx.extend(cidx[:k])
                    fid[cidx[k:]] = np.arange(len(cidx) - k) % n_folds
                role[test_idx] = "test"
            else:
                rng.shuffle(idx)
                k = int(round(n * test_size))
                role[idx[:k]] = "test"
                fid = np.full(n, -1)
                fid[idx[k:]] = np.arange(n - k) % n_folds
        for f in range(n_folds):
            folds.append(np.where(role == "test", "", np.where(fid == f, "valid", "train")))

    elif method == "time":
        key = df["snapshot"] if "snapshot" in df.columns and not spec.time_col else df[spec.time_col]
        t = pd.to_datetime(key)
        boundary = t.quantile(1 - test_size)
        role[(t >= boundary).to_numpy()] = "test"
        info["test_from"] = str(boundary)
        tr_times = np.sort(t[role == "train"].unique())
        edges = np.array_split(tr_times, n_folds + 1)
        for f in range(1, n_folds + 1):
            v0, v1 = edges[f][0], edges[f][-1]
            col = np.where(role == "test", "", np.where(t < v0, "train", np.where(t <= v1, "valid", "")))
            folds.append(col)

    elif method == "snapshot":
        if "snapshot" not in df.columns:
            raise SplitError("snapshot split needs a snapshot column (define_task with cutoffs)")
        snaps = sorted(df["snapshot"].astype(str).unique())
        if len(snaps) < 3:
            raise SplitError("snapshot split needs at least 3 cutoffs (train, valid, test)")
        s = df["snapshot"].astype(str).to_numpy()
        role[s == snaps[-1]] = "test"
        info["test_snapshot"] = snaps[-1]
        horizon_days = int(spec.horizon or 0)
        for v in snaps[1:-1]:
            # train only on snapshots whose label window closed before the validation cutoff
            ok_train = [x for x in snaps[:-1] if pd.Timestamp(x) + pd.Timedelta(days=horizon_days) <= pd.Timestamp(v)]
            if not ok_train:
                continue
            folds.append(np.where(s == v, "valid", np.where(np.isin(s, ok_train), "train", "")))
        info["note"] = "train snapshots must have their label window closed before the validation/test cutoff"
        last_train = [x for x in snaps[:-1] if pd.Timestamp(x) + pd.Timedelta(days=horizon_days) <= pd.Timestamp(snaps[-1])]
        role[(role == "train") & ~np.isin(s, last_train)] = "unused"

    elif method == "rolling_origin":
        if not spec.time_col:
            raise SplitError("rolling_origin needs spec.time_col")
        h = int(spec.horizon or 1)
        g = int(gap or 0)
        t = pd.to_datetime(df[spec.time_col])
        periods = np.sort(t.unique())
        if len(periods) < h * (n_folds + 2):
            raise SplitError(f"only {len(periods)} periods; need at least {h * (n_folds + 2)} for {n_folds} folds of horizon {h}")
        test_start = periods[-h]
        role[(t >= test_start).to_numpy()] = "test"
        info.update({"horizon": h, "gap": g, "test_from": str(pd.Timestamp(test_start))})
        origins = []
        for f in range(n_folds, 0, -1):
            v_start_i = len(periods) - h - f * h
            v_start, v_end = periods[v_start_i], periods[v_start_i + h - 1]
            train_end = periods[v_start_i - g - 1] if g else periods[v_start_i - 1]
            col = np.where(t <= train_end, "train", np.where((t >= v_start) & (t <= v_end), "valid", ""))
            col = np.where(role == "test", "", col)
            folds.append(col)
            origins.append(str(pd.Timestamp(v_start)))
        info["fold_origins"] = origins

    out = pd.DataFrame({"role": role})
    for k, col in enumerate(folds):
        out[f"cv{k}"] = col
    info["n_folds"] = len(folds)
    info["counts"] = {"train": int((role == "train").sum()), "test": int((role == "test").sum()),
                      "unused": int((role == "unused").sum())}
    if spec.entity and spec.entity in df.columns and method in ("random", "stratified"):
        ent = df[spec.entity].astype(str)
        overlap = set(ent[role == "train"]) & set(ent[role == "test"])
        info["entity_overlap"] = len(overlap)
    return out, info
