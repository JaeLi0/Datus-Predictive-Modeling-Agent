# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""train_select (step 6): baseline -> candidate models -> AutoML tuning of the best two -> rule-based choice.

The LLM never writes training code. It may restrict the candidate list, exclude features and set the time
budget; everything else here is fixed so a run is reproducible.

Candidates
- binary / multiclass: prior (baseline), logistic regression, random forest, LightGBM
- regression: mean (baseline), ridge, random forest, LightGBM
- time series: seasonal naive (baseline), ETS per series, ridge on lag features, LightGBM on lag features
  (one global model across series, the series id as a categorical feature)

Procedure
1. Every candidate is scored on the CV folds of the run's split with default settings. Folds with many
   training rows are subsampled for this coarse pass.
2. The two best tunable candidates (random forest, LightGBM) are tuned by FLAML within the time budget,
   validated on the most recent / last CV fold, then re-scored on all folds without subsampling.
3. Selection rule: best mean primary metric; candidates within one fold standard deviation of the best count
   as tied, and among tied ones the more interpretable wins (baseline < linear < ets < rf < lgbm), then the
   more stable, then the faster.

Preprocessing is fitted on the training rows of each fold only: category levels, imputation medians,
scaling, and out-of-fold target encodings (feature_meta entries with deferred=True).
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataengineer.tools.model_tools.features import oof_target_encode
from dataengineer.tools.model_tools.spec import TaskSpec, _jsonable

HIGHER_IS_BETTER = {"roc_auc", "pr_auc", "f1", "r2", "accuracy", "macro_f1"}
METRICS = {
    "binary": ["roc_auc", "pr_auc", "log_loss", "f1"],
    "multiclass": ["log_loss", "accuracy", "macro_f1"],
    "regression": ["rmse", "mae", "r2", "mape"],
    "timeseries": ["wape", "mase", "mae", "rmse"],
}
CANDIDATES = {
    "binary": ["prior", "linear", "rf", "lgbm"],
    "multiclass": ["prior", "linear", "rf", "lgbm"],
    "regression": ["mean", "linear", "rf", "lgbm"],
    "timeseries": ["seasonal_naive", "ets", "linear", "lgbm"],
}
BASELINES = {"prior", "mean", "seasonal_naive"}
TUNABLE = {"rf": "rf", "lgbm": "lgbm"}
INTERPRETABILITY = {"prior": 0, "mean": 0, "seasonal_naive": 0, "linear": 1, "ets": 2, "rf": 3, "lgbm": 4}
COARSE_MAX_TRAIN = 30000
RF_MAX_TREES = 500
MAX_LEVELS = 50
N_JOBS = int(os.environ.get("MODEL_N_JOBS", "4"))
INTERNAL = {"__rid", "snapshot"}


class TrainError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# target and metrics
# ---------------------------------------------------------------------------------------------
def target_vector(df: pd.DataFrame, spec: TaskSpec) -> Tuple[np.ndarray, Optional[List[str]]]:
    y = df[spec.target]
    if spec.task_type == "binary":
        return (y.astype(str) == str(spec.positive_label)).astype(int).to_numpy(), None
    if spec.task_type == "multiclass":
        classes = sorted(y.astype(str).unique())
        return y.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy(), classes
    return pd.to_numeric(y, errors="coerce").astype(float).to_numpy(), None


def mase_scale(df: pd.DataFrame, spec: TaskSpec, rows: np.ndarray) -> float:
    """Mean absolute seasonal difference on the training rows (pooled over series)."""
    m = int(spec.season_length or 1)
    d = df.loc[rows, [c for c in (spec.entity, spec.time_col, spec.target) if c]].sort_values(
        [c for c in (spec.entity, spec.time_col) if c])
    g = d.groupby(spec.entity)[spec.target] if spec.entity else d[spec.target]
    diff = (d[spec.target] - g.shift(m)).abs().dropna()
    if not len(diff) or diff.mean() == 0:
        diff = (d[spec.target] - g.shift(1)).abs().dropna()
    return float(diff.mean()) if len(diff) and diff.mean() > 0 else 1.0


def score(metric: str, y: np.ndarray, pred: np.ndarray, scale: float = 1.0) -> Optional[float]:
    from sklearn import metrics as M

    y = np.asarray(y)
    pred = np.asarray(pred)
    try:
        if metric == "roc_auc":
            return float(M.roc_auc_score(y, pred)) if len(np.unique(y)) > 1 else None
        if metric == "pr_auc":
            return float(M.average_precision_score(y, pred)) if len(np.unique(y)) > 1 else None
        if metric == "log_loss":
            if pred.ndim == 1:
                return float(M.log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6), labels=[0, 1]))
            return float(M.log_loss(y, np.clip(pred, 1e-6, 1), labels=list(range(pred.shape[1]))))
        if metric == "f1":
            return float(M.f1_score(y, (pred >= 0.5).astype(int), zero_division=0))
        if metric == "accuracy":
            return float(M.accuracy_score(y, pred.argmax(1)))
        if metric == "macro_f1":
            return float(M.f1_score(y, pred.argmax(1), average="macro", zero_division=0))
        if metric == "rmse":
            return float(np.sqrt(np.mean((y - pred) ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(y - pred)))
        if metric == "r2":
            return float(M.r2_score(y, pred))
        if metric == "mape":
            nz = y != 0
            return float(np.mean(np.abs((y[nz] - pred[nz]) / y[nz]))) if nz.any() else None
        if metric == "wape":
            den = np.sum(np.abs(y))
            return float(np.sum(np.abs(y - pred)) / den) if den else None
        if metric == "mase":
            return float(np.mean(np.abs(y - pred)) / (scale or 1.0))
    except ValueError:
        return None
    raise TrainError(f"unknown metric {metric!r}")


def all_scores(spec: TaskSpec, y, pred, scale: float = 1.0) -> Dict[str, Optional[float]]:
    return {m: score(m, y, pred, scale) for m in METRICS[spec.task_type]}


# ---------------------------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame, spec: TaskSpec, meta: List[Dict], exclude: Optional[List[str]] = None) -> Dict:
    """Columns usable as model inputs, split into numeric / categorical / deferred target encodings."""
    skip = {spec.target, spec.time_col, *INTERNAL, *spec.drop_columns, *(exclude or [])}
    if spec.task_type != "timeseries":
        skip.add(spec.entity)
    num, cat, dropped = [], [], []
    for c in df.columns:
        if c in skip or c is None:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            dropped.append((c, "datetime column; use date_parts"))
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            nun = s.nunique(dropna=True)
            if c != spec.entity and nun > max(MAX_LEVELS * 20, 0.5 * len(s)):
                dropped.append((c, f"categorical with {nun} levels looks like an id"))
            else:
                cat.append(c)
    te = [m for m in meta if m.get("deferred") and m.get("template") == "target_encoding"
          and m["sources"][0] in df.columns and m["name"] not in (exclude or [])]
    return {"numeric": num, "categorical": cat, "target_encoded": te, "dropped": dropped}


class Prep:
    """Fitted on training rows only. mode 'tree' -> numeric matrix with category codes (NaN kept);
    mode 'linear' -> imputed, scaled numerics plus one-hot categories."""

    def __init__(self, cols: Dict, mode: str):
        self.num = list(cols["numeric"])
        self.cat = list(cols["categorical"])
        self.te = list(cols["target_encoded"])
        self.mode = mode

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.levels = {}
        for c in self.cat:
            vc = X[c].astype(str).value_counts()
            self.levels[c] = list(vc.index[:MAX_LEVELS])
        self.te_maps = {}
        out = self._base(X)
        for m in self.te:
            src = m["sources"][0]
            yy = pd.Series(y.astype(float), index=X.index)
            enc, _ = oof_target_encode(X[src].astype(str), yy, None, folds=int(m.get("folds", 5)),
                                       smoothing=float(m.get("smoothing", 20)))
            stats = yy.groupby(X[src].astype(str)).agg(["sum", "count"])
            prior, k = float(yy.mean()), float(m.get("smoothing", 20))
            self.te_maps[m["name"]] = (src, ((stats["sum"] + prior * k) / (stats["count"] + k)).to_dict(), prior)
            out[m["name"]] = enc.to_numpy()
        if self.mode == "linear":
            self.medians = out.median(numeric_only=True)
            filled = out.fillna(self.medians).fillna(0)
            self.mu, self.sd = filled.mean(), filled.std().replace(0, 1).fillna(1)
            return (filled - self.mu) / self.sd
        self.columns = list(out.columns)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._base(X)
        for name, (src, mapping, prior) in self.te_maps.items():
            out[name] = X[src].astype(str).map(mapping).fillna(prior).to_numpy()
        if self.mode == "linear":
            out = out.reindex(columns=self.mu.index, fill_value=0)
            return (out.fillna(self.medians).fillna(0) - self.mu) / self.sd
        return out.reindex(columns=self.columns)

    def _base(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = {c: pd.to_numeric(X[c], errors="coerce").astype(float).to_numpy() for c in self.num}
        for c in self.cat:
            s = X[c].astype(str)
            if self.mode == "linear":
                for lv in self.levels[c]:
                    parts[f"{c}={lv}"] = (s == lv).astype(float).to_numpy()
            else:
                code = {lv: i for i, lv in enumerate(self.levels[c])}
                parts[c] = s.map(code).fillna(len(code)).astype(float).to_numpy()
        return pd.DataFrame(parts, index=X.index)

    def input_names(self) -> List[str]:
        return self.num + self.cat + [m["name"] for m in self.te]


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
def make_estimator(name: str, spec: TaskSpec, params: Optional[Dict] = None, seed: int = 0):
    params = dict(params or {})
    clf = spec.is_classification
    if name == "linear":
        from sklearn.linear_model import LogisticRegression, Ridge

        return (LogisticRegression(max_iter=2000, C=params.get("C", 1.0)) if clf
                else Ridge(alpha=params.get("alpha", 1.0)))
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        base = {"n_estimators": 300, "min_samples_leaf": 2, "n_jobs": N_JOBS, "random_state": seed}
        base.update(params)
        base["n_jobs"] = N_JOBS
        return RandomForestClassifier(**base) if clf else RandomForestRegressor(**base)
    if name == "lgbm":
        import lightgbm as lgb

        base = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "random_state": seed, "verbose": -1,
                "n_jobs": N_JOBS}
        base.update(params)
        base.update(verbose=-1, n_jobs=N_JOBS)
        return lgb.LGBMClassifier(**base) if clf else lgb.LGBMRegressor(**base)
    raise TrainError(f"{name!r} is not an estimator model")


class FittedModel:
    """A fitted candidate: preprocessing + estimator (or the rule for baselines / per-series models)."""

    def __init__(self, name: str, spec: TaskSpec, cols: Dict, params: Optional[Dict] = None):
        self.name, self.spec, self.cols, self.params = name, spec, cols, dict(params or {})
        self.prep: Optional[Prep] = None
        self.est = None
        self.classes: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "FittedModel":
        y, self.classes = target_vector(df, self.spec)
        if self.name in ("prior", "mean"):
            self.const = (np.bincount(y.astype(int)) / len(y) if self.spec.task_type == "multiclass"
                          else float(np.mean(y)))
            return self
        if self.name in ("seasonal_naive", "ets"):
            self.history = df[[self.spec.entity, self.spec.time_col, self.spec.target]].copy() if self.spec.entity \
                else df[[self.spec.time_col, self.spec.target]].copy()
            return self
        self.prep = Prep(self.cols, "linear" if self.name == "linear" else "tree")
        X = self.prep.fit_transform(df, y)
        self.est = make_estimator(self.name, self.spec, self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Probability of the positive class (binary), class probabilities (multiclass) or values."""
        n = len(df)
        if self.name in ("prior", "mean"):
            return np.tile(self.const, (n, 1)) if self.spec.task_type == "multiclass" else np.full(n, self.const)
        if self.name == "seasonal_naive":
            return seasonal_naive(self.history if history is None else history, df, self.spec)
        if self.name == "ets":
            return ets_forecast(self.history if history is None else history, df, self.spec)
        X = self.prep.transform(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.spec.task_type == "binary":
                return self.est.predict_proba(X)[:, 1]
            if self.spec.task_type == "multiclass":
                return self.est.predict_proba(X)
            return self.est.predict(X)

    def describe(self) -> Dict:
        return {"model": self.name, "params": self.params, "inputs": self.prep.input_names() if self.prep else []}


def _series_key(spec: TaskSpec, df: pd.DataFrame) -> pd.Series:
    return df[spec.entity].astype(str) if spec.entity else pd.Series("_", index=df.index)


def _positions(history: pd.DataFrame, target_rows: pd.DataFrame, spec: TaskSpec):
    """Integer period positions shared by history and target rows; origin = first target period."""
    periods = pd.DatetimeIndex(np.sort(pd.to_datetime(pd.concat([history[spec.time_col],
                                                                 target_rows[spec.time_col]])).unique()))
    tpos = periods.get_indexer(pd.to_datetime(target_rows[spec.time_col]))
    h = history.assign(_k=_series_key(spec, history).to_numpy(),
                       _p=periods.get_indexer(pd.to_datetime(history[spec.time_col])))
    return h, tpos, int(tpos.min())


def seasonal_naive(history: pd.DataFrame, target_rows: pd.DataFrame, spec: TaskSpec) -> np.ndarray:
    """y_hat(t) = y(t - k*m) for the smallest k that lands before the forecast origin (the first period of
    target_rows); series without that much history fall back to their last value before the origin."""
    m = int(spec.season_length or 1)
    h, tpos, origin = _positions(history, target_rows, spec)
    h = h[h["_p"] < origin]
    lookup = dict(zip(zip(h["_k"], h["_p"]), h[spec.target].astype(float)))
    last = h.sort_values("_p").groupby("_k")[spec.target].last().astype(float).to_dict()
    glob = float(h[spec.target].mean()) if len(h) else 0.0
    keys = _series_key(spec, target_rows).to_numpy()
    out = np.empty(len(target_rows))
    for i, (k, p) in enumerate(zip(keys, tpos)):
        q = p - m
        while q >= origin:
            q -= m
        v = lookup.get((k, q))
        out[i] = v if v is not None and np.isfinite(v) else last.get(k, glob)
    return out


def ets_forecast(history: pd.DataFrame, target_rows: pd.DataFrame, spec: TaskSpec) -> np.ndarray:
    """Exponential smoothing per series fitted on periods before the forecast origin (additive seasonality
    when at least two seasons of history exist), forecasting each target period by its step ahead."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    m = int(spec.season_length or 1)
    h, tpos, origin = _positions(history, target_rows, spec)
    h = h[h["_p"] < origin].sort_values("_p")
    keys = _series_key(spec, target_rows).to_numpy()
    out = np.full(len(target_rows), np.nan)
    for k in np.unique(keys):
        y = h.loc[h["_k"] == k, spec.target].astype(float).to_numpy()
        idx = np.where(keys == k)[0]
        if len(y) < 3:
            out[idx] = y[-1] if len(y) else 0.0
            continue
        steps = tpos[idx] - (origin - 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seasonal = "add" if m > 1 and len(y) >= 2 * m else None
                fit = ExponentialSmoothing(y, trend=None, seasonal=seasonal, seasonal_periods=m if seasonal else None,
                                           initialization_method="estimated").fit()
                fc = fit.forecast(int(steps.max()))
            out[idx] = np.asarray(fc)[steps - 1]
        except Exception:
            out[idx] = y[-1]
    return np.nan_to_num(out, nan=float(h[spec.target].mean()) if len(h) else 0.0)


# ---------------------------------------------------------------------------------------------
# cross-validation and selection
# ---------------------------------------------------------------------------------------------
def fit_predict(name: str, df: pd.DataFrame, spec: TaskSpec, cols: Dict, train: np.ndarray, test: np.ndarray,
                params: Optional[Dict] = None) -> Tuple[np.ndarray, FittedModel]:
    model = FittedModel(name, spec, cols, params).fit(df[train])
    hist = df[train] if name in ("seasonal_naive", "ets") else None
    return model.predict(df[test], history=hist), model


def cross_validate(name: str, df: pd.DataFrame, spec: TaskSpec, cols: Dict, split: pd.DataFrame,
                   params: Optional[Dict] = None, max_train: Optional[int] = None, seed: int = 0) -> Dict:
    folds = [c for c in split.columns if c.startswith("cv")]
    if not folds:
        raise TrainError("the split has no CV folds; call make_split with n_folds >= 2")
    y, _ = target_vector(df, spec)
    per_fold, t0 = [], time.time()
    rng = np.random.RandomState(seed)
    for f in folds:
        tr = (split[f] == "train").to_numpy()
        va = (split[f] == "valid").to_numpy()
        if not tr.any() or not va.any():
            continue
        if max_train and tr.sum() > max_train and spec.task_type != "timeseries":
            keep = rng.choice(np.where(tr)[0], max_train, replace=False)
            tr = np.zeros(len(df), bool)
            tr[keep] = True
        pred, _ = fit_predict(name, df, spec, cols, tr, va, params)
        scale = mase_scale(df, spec, tr) if spec.task_type == "timeseries" else 1.0
        per_fold.append(all_scores(spec, y[va], pred, scale))
    if not per_fold:
        raise TrainError("no usable CV fold (each needs train and valid rows)")
    pm = spec.primary_metric
    vals = [p[pm] for p in per_fold if p.get(pm) is not None]
    means = {m: (float(np.mean([p[m] for p in per_fold if p.get(m) is not None]))
                 if any(p.get(m) is not None for p in per_fold) else None) for m in per_fold[0]}
    return {"model": name, "params": params or {}, "folds": len(per_fold), "mean": means,
            "primary_mean": float(np.mean(vals)) if vals else None,
            "primary_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "seconds": round(time.time() - t0, 2), "subsampled": bool(max_train and spec.task_type != "timeseries"
                                                                       and (split["role"] == "train").sum() > max_train)}


def flaml_tune(name: str, df: pd.DataFrame, spec: TaskSpec, cols: Dict, split: pd.DataFrame, time_budget: float,
               seed: int = 0) -> Dict:
    """Tune one estimator family with FLAML, validating on the last CV fold. Returns sklearn params."""
    from flaml import AutoML

    fold = [c for c in split.columns if c.startswith("cv")][-1]
    tr = (split[fold] == "train").to_numpy()
    va = (split[fold] == "valid").to_numpy()
    y, _ = target_vector(df, spec)
    prep = Prep(cols, "tree")
    Xtr = prep.fit_transform(df[tr], y[tr])
    Xva = prep.transform(df[va])
    metric = {"roc_auc": "roc_auc", "pr_auc": "ap", "log_loss": "log_loss", "f1": "f1", "accuracy": "accuracy",
              "macro_f1": "micro_f1", "rmse": "rmse", "mae": "mae", "r2": "r2", "mape": "mape",
              "wape": "mae", "mase": "mae"}[spec.primary_metric]
    task = {"binary": "classification", "multiclass": "classification"}.get(spec.task_type, "regression")
    automl = AutoML()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        automl.fit(Xtr, y[tr], X_val=Xva, y_val=y[va], task=task, metric=metric, estimator_list=[TUNABLE[name]],
                   time_budget=time_budget, eval_method="holdout", seed=seed, verbose=0, n_jobs=N_JOBS,
                   retrain_full=False)
    est = automl.best_model_for_estimator(TUNABLE[name])
    params = est.estimator.get_params() if est is not None and est.estimator is not None else {}
    keep = {"n_estimators", "num_leaves", "min_child_samples", "learning_rate", "colsample_bytree", "reg_alpha",
            "reg_lambda", "max_bin", "subsample", "max_features", "max_leaf_nodes", "criterion", "min_samples_leaf",
            "max_depth"}
    params = {k: v for k, v in params.items() if k in keep and v is not None}
    if name == "rf" and params.get("n_estimators", 0) > RF_MAX_TREES:
        params["n_estimators"] = RF_MAX_TREES  # beyond a few hundred trees the forest only gets slower
    return {"params": _jsonable(params), "iterations": len(getattr(automl, "config_history", {}) or {})}


def _better(a: float, b: float, metric: str) -> bool:
    return a > b if metric in HIGHER_IS_BETTER else a < b


def select(results: List[Dict], metric: str) -> Tuple[Dict, str]:
    ok = [r for r in results if r.get("primary_mean") is not None]
    if not ok:
        raise TrainError("no candidate produced a score")
    best = ok[0]
    for r in ok[1:]:
        if _better(r["primary_mean"], best["primary_mean"], metric):
            best = r
    tol = max(best["primary_std"], 1e-9)
    tied = [r for r in ok if abs(r["primary_mean"] - best["primary_mean"]) <= tol and r["model"] not in BASELINES]
    if not tied:
        tied = [best]
    tied.sort(key=lambda r: (INTERPRETABILITY.get(r["model"], 9), r["primary_std"], r["seconds"]))
    pick = tied[0]
    if pick is best:
        reason = (f"{pick['label']} has the best mean CV {metric} ({pick['primary_mean']:.4f} ± {pick['primary_std']:.4f})")
    else:
        reason = (f"{pick['label']} ({pick['primary_mean']:.4f}) is within one fold std ({tol:.4f}) of the best "
                  f"{best['label']} ({best['primary_mean']:.4f}) and is more interpretable / stable")
    baseline = next((r for r in ok if r["model"] in BASELINES), None)
    if baseline and not _better(pick["primary_mean"], baseline["primary_mean"], metric):
        reason += f"; WARNING: it does not beat the baseline ({baseline['primary_mean']:.4f})"
    return pick, reason


def train_select(df: pd.DataFrame, spec: TaskSpec, meta: List[Dict], split: pd.DataFrame,
                 candidates: Optional[List[str]] = None, exclude: Optional[List[str]] = None,
                 time_budget: float = 60, seed: int = 0) -> Dict:
    allowed = CANDIDATES[spec.task_type]
    cands = [c for c in (candidates or allowed)]
    bad = [c for c in cands if c not in allowed]
    if bad:
        raise TrainError(f"candidates {bad} not available for {spec.task_type}; choose from {allowed}")
    base = next(c for c in allowed if c in BASELINES)
    if base not in cands:
        cands = [base] + cands  # the baseline is always reported
    cols = feature_columns(df, spec, meta, exclude)
    if not (cols["numeric"] or cols["categorical"] or cols["target_encoded"]) and any(
            c not in BASELINES and c != "ets" for c in cands):
        raise TrainError("no usable feature columns")
    n_train = int((split["role"] == "train").sum())
    results = []
    for c in cands:
        r = cross_validate(c, df, spec, cols, split, max_train=COARSE_MAX_TRAIN, seed=seed)
        r.update(label=c, stage="default")
        results.append(r)
    tunable = [r for r in results if r["model"] in TUNABLE and r["primary_mean"] is not None]
    tunable.sort(key=lambda r: r["primary_mean"], reverse=spec.primary_metric in HIGHER_IS_BETTER)
    for r in tunable[:2]:
        t0 = time.time()
        tuned = flaml_tune(r["model"], df, spec, cols, split, time_budget / max(1, min(2, len(tunable))), seed)
        rt = cross_validate(r["model"], df, spec, cols, split, params=tuned["params"], seed=seed)
        rt.update(label=f"{r['model']}_tuned", stage="flaml", tune_seconds=round(time.time() - t0, 1),
                  flaml_iterations=tuned["iterations"])
        results.append(rt)
    pick, reason = select(results, spec.primary_metric)
    table = [{"label": r["label"], "model": r["model"], "params": r["params"], "stage": r["stage"],
              "cv_primary": r["primary_mean"],
              "cv_std": r["primary_std"], "cv_all": r["mean"], "seconds": r["seconds"], "subsampled": r["subsampled"]}
             for r in results]
    return _jsonable({
        "primary_metric": spec.primary_metric, "higher_is_better": spec.primary_metric in HIGHER_IS_BETTER,
        "n_train_rows": n_train, "features": {k: cols[k] for k in ("numeric", "categorical")},
        "target_encoded": [m["name"] for m in cols["target_encoded"]],
        "dropped_columns": [{"column": c, "why": w} for c, w in cols["dropped"]],
        "excluded": exclude or [], "table": table,
        "chosen": {"label": pick["label"], "model": pick["model"], "params": pick["params"]},
        "reason": reason,
        "rule": "best mean CV primary metric; within one fold std counts as tied -> more interpretable, "
                "more stable, faster",
    })
