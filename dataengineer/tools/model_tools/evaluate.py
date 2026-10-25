# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""evaluate (step 7) and explain (step 8).

evaluate fits the chosen model on every training row of the split, scores the held-out test set once, and
adds the diagnostics the LLM needs to judge whether the result can be trusted:
- all task metrics on test next to the CV estimate, and the CV -> test gap (metric_gap leakage check)
- baseline on the same test rows, so "better than naive" is explicit
- calibration table and Brier score (binary)
- error by segment (worst levels of low-cardinality categoricals; per series for time series)
- PSI of each input between train and test (drift between the periods / groups)
- plots: ROC/PR/calibration, predicted vs actual, or forecast vs actual for sample series

explain uses SHAP for tree models (TreeExplainer on a test sample) and standardized coefficients for linear
models; it reports global importance and the top contributions for a few individual predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dataengineer.tools.model_tools import train as T
from dataengineer.tools.model_tools.leakage import metric_gap_check
from dataengineer.tools.model_tools.spec import TaskSpec, _jsonable

PSI_WARN = 0.2
SEGMENT_MIN_ROWS = 50


def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> Optional[float]:
    a, b = a.dropna(), b.dropna()
    if len(a) < 20 or len(b) < 20:
        return None
    if pd.api.types.is_numeric_dtype(a) and a.nunique() > bins:
        edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return None
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a, edges)[0] / len(a)
        pb = np.histogram(b, edges)[0] / len(b)
    else:
        levels = a.astype(str).value_counts().index[:20]
        pa = np.array([(a.astype(str) == lv).mean() for lv in levels])
        pb = np.array([(b.astype(str) == lv).mean() for lv in levels])
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Dict:
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, max(0, len(q) - 2))
    rows = []
    for k in np.unique(idx):
        m = idx == k
        rows.append({"bin": int(k), "n": int(m.sum()), "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    ece = sum(r["n"] * abs(r["mean_pred"] - r["observed"]) for r in rows) / max(1, len(y))
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "bins": rows}


def segment_errors(df: pd.DataFrame, spec: TaskSpec, cols: Dict, y: np.ndarray, pred: np.ndarray, scale: float,
                   top: int = 3) -> List[Dict]:
    out = []
    metric = spec.primary_metric
    if spec.task_type == "timeseries" and spec.entity:
        key = df[spec.entity].astype(str).to_numpy()
        rows = []
        for k in np.unique(key):
            m = key == k
            rows.append({"segment": f"{spec.entity}={k}", "n": int(m.sum()),
                         metric: T.score(metric, y[m], pred[m], scale)})
        rows = [r for r in rows if r[metric] is not None]
        rows.sort(key=lambda r: r[metric], reverse=metric not in T.HIGHER_IS_BETTER)
        return rows[:top * 2]
    cands = [c for c in cols["categorical"] + cols["numeric"] if 2 <= df[c].nunique() <= 12][:6]
    for c in cands:
        for lv, g in df.groupby(df[c].astype(str)).groups.items():
            m = df.index.isin(g)
            if m.sum() < SEGMENT_MIN_ROWS:
                continue
            s = T.score(metric, y[m], pred[m], scale)
            if s is not None:
                out.append({"segment": f"{c}={lv}", "n": int(m.sum()), metric: s})
    out.sort(key=lambda r: r[metric], reverse=metric not in T.HIGHER_IS_BETTER)
    return out[:top * 2]


def drift(df: pd.DataFrame, cols: Dict, train: np.ndarray, test: np.ndarray, top: int = 5) -> List[Dict]:
    rows = []
    for c in cols["numeric"] + cols["categorical"]:
        v = psi(df.loc[train, c], df.loc[test, c])
        if v is not None:
            rows.append({"feature": c, "psi": round(v, 4)})
    rows.sort(key=lambda r: -r["psi"])
    return rows[:top]


def _plots(out_dir: Path, spec: TaskSpec, df_test: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if spec.task_type == "binary":
        from sklearn.metrics import precision_recall_curve, roc_curve

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        fpr, tpr, _ = roc_curve(y, pred)
        ax[0].plot(fpr, tpr)
        ax[0].plot([0, 1], [0, 1], "--", color="grey")
        ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
        pr, rc, _ = precision_recall_curve(y, pred)
        ax[1].plot(rc, pr)
        ax[1].axhline(y.mean(), ls="--", color="grey")
        ax[1].set(title="Precision-recall", xlabel="recall", ylabel="precision")
        cal = calibration_table(y, pred)["bins"]
        ax[2].plot([r["mean_pred"] for r in cal], [r["observed"] for r in cal], "o-")
        ax[2].plot([0, 1], [0, 1], "--", color="grey")
        ax[2].set(title="Calibration", xlabel="predicted", ylabel="observed")
    elif spec.task_type == "regression":
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].scatter(y, pred, s=4, alpha=0.4)
        lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax[0].plot(lim, lim, "--", color="grey")
        ax[0].set(title="Predicted vs actual", xlabel="actual", ylabel="predicted")
        ax[1].hist(pred - y, bins=50)
        ax[1].set(title="Residuals (pred - actual)")
    elif spec.task_type == "timeseries":
        keys = df_test[spec.entity].astype(str).unique()[:3] if spec.entity else ["_"]
        fig, ax = plt.subplots(len(keys), 1, figsize=(9, 2.6 * len(keys)), squeeze=False)
        for i, k in enumerate(keys):
            m = (df_test[spec.entity].astype(str) == k).to_numpy() if spec.entity else np.ones(len(df_test), bool)
            t = pd.to_datetime(df_test[spec.time_col]).to_numpy()[m]
            o = np.argsort(t)
            ax[i, 0].plot(t[o], y[m][o], "o-", label="actual")
            ax[i, 0].plot(t[o], pred[m][o], "s--", label="forecast")
            ax[i, 0].set(title=str(k))
            ax[i, 0].legend()
    else:
        return paths
    fig.tight_layout()
    p = out_dir / "evaluation.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    paths.append(str(p))
    return paths


def evaluate(df: pd.DataFrame, spec: TaskSpec, meta: List[Dict], split: pd.DataFrame, selection: Dict,
             out_dir: Path, model: Optional[Dict] = None, guardrails: bool = True):
    """Returns (report, fitted model)."""
    chosen = model or selection["chosen"]
    cols = T.feature_columns(df, spec, meta, selection.get("excluded"))
    train = (split["role"] == "train").to_numpy()
    test = (split["role"] == "test").to_numpy()
    if not test.any():
        raise T.TrainError("the split has no test rows")
    y, _ = T.target_vector(df, spec)
    scale = T.mase_scale(df, spec, train) if spec.task_type == "timeseries" else 1.0
    pred, fitted = T.fit_predict(chosen["model"], df, spec, cols, train, test, chosen.get("params"))
    base_name = next(c for c in T.CANDIDATES[spec.task_type] if c in T.BASELINES)
    base_pred, _ = T.fit_predict(base_name, df, spec, cols, train, test)
    test_scores = T.all_scores(spec, y[test], pred, scale)
    base_scores = T.all_scores(spec, y[test], base_pred, scale)
    pm = spec.primary_metric
    row = next((r for r in selection["table"] if r["label"] == chosen.get("label", chosen["model"])), None)
    cv_primary = row["cv_primary"] if row else None
    findings = []
    if guardrails and cv_primary is not None and test_scores.get(pm) is not None:
        findings = metric_gap_check(cv_primary, test_scores[pm], pm, pm in T.HIGHER_IS_BETTER)
    report = {
        "model": chosen, "n_train": int(train.sum()), "n_test": int(test.sum()),
        "test": test_scores, "baseline": {"model": base_name, **base_scores},
        "cv_primary": cv_primary, "primary_metric": pm,
        "beats_baseline": (T._better(test_scores[pm], base_scores[pm], pm)
                           if test_scores.get(pm) is not None and base_scores.get(pm) is not None else None),
        "findings": findings,
        "segments": segment_errors(df[test].reset_index(drop=True), spec, cols, y[test],
                                   pred if pred.ndim == 1 else pred.max(1), scale)
        if spec.task_type != "multiclass" else [],
        "drift_psi": drift(df, cols, train, test),
    }
    if spec.task_type == "binary":
        cal = calibration_table(y[test], pred)
        report["calibration"] = {"brier": cal["brier"], "ece": cal["ece"]}
    report["drift_warning"] = [d["feature"] for d in report["drift_psi"] if d["psi"] > PSI_WARN]
    try:
        report["plots"] = _plots(out_dir / "plots", spec, df[test].reset_index(drop=True), y[test],
                                 pred if pred.ndim == 1 else pred.max(1))
    except Exception as e:  # plotting is a convenience; never fail the evaluation for it
        report["plots"] = []
        report["plot_error"] = str(e)
    preds = pd.DataFrame({"__rid": df.loc[test, "__rid"].to_numpy() if "__rid" in df.columns else np.where(test)[0],
                          "actual": y[test]})
    preds["prediction"] = pred if pred.ndim == 1 else pred.argmax(1)
    return _jsonable(report), fitted, preds


def explain(fitted: T.FittedModel, df: pd.DataFrame, test: np.ndarray, out_dir: Path, top: int = 15,
            n_individual: int = 3, sample: int = 2000, seed: int = 0) -> Dict:
    if fitted.prep is None:
        return {"method": "none", "note": f"{fitted.name} has no feature inputs to explain (rule-based or per-series "
                                          "model); explain the baseline logic instead"}
    rows = df[test]
    if len(rows) > sample:
        rows = rows.sample(sample, random_state=seed)
    X = fitted.prep.transform(rows)
    pred = fitted.predict(rows)
    if fitted.name == "linear":
        coef = np.asarray(fitted.est.coef_)
        coef = coef[0] if coef.ndim > 1 and coef.shape[0] == 1 else (np.abs(coef).mean(0) if coef.ndim > 1 else coef)
        contrib = X.to_numpy() * coef
        method = "standardized coefficients (contribution = coef * standardized value)"
    else:
        import shap

        explainer = shap.TreeExplainer(fitted.est)
        sv = explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else np.mean(np.abs(np.stack(sv)), 0)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] == 2 else np.abs(sv).mean(2)
        contrib = sv
        method = "SHAP TreeExplainer on a test sample"
    names = list(X.columns)
    # one-hot columns of linear models are folded back into their source column
    src = [n.split("=", 1)[0] if "=" in n else n for n in names]
    imp = pd.Series(np.abs(contrib).mean(0), index=src).groupby(level=0).sum().sort_values(ascending=False)
    signed = pd.Series(contrib.mean(0), index=src).groupby(level=0).sum()
    global_imp = [{"feature": f, "mean_abs_contribution": float(v), "mean_signed": float(signed[f])}
                  for f, v in imp.head(top).items()]
    order = np.argsort(-(pred if np.ndim(pred) == 1 else np.max(pred, 1)))[:n_individual]
    individual = []
    for i in order:
        c = pd.Series(contrib[i], index=src).groupby(level=0).sum()
        c = c.reindex(c.abs().sort_values(ascending=False).index)[:5]
        individual.append({"row": rows.index[i].item() if hasattr(rows.index[i], "item") else int(rows.index[i]),
                           "prediction": float(pred[i] if np.ndim(pred) == 1 else np.max(pred[i])),
                           "top_contributions": {k: round(float(v), 5) for k, v in c.items()}})
    plot = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        h = imp.head(top)[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(h) + 1))
        ax.barh(h.index.astype(str), h.values)
        ax.set(title=f"Global importance ({fitted.name})", xlabel="mean |contribution|")
        fig.tight_layout()
        plot = str(out_dir / "importance.png")
        fig.savefig(plot, dpi=90)
        plt.close(fig)
    except Exception:
        plot = None
    return _jsonable({"method": method, "model": fitted.name, "global_importance": global_imp,
                      "individual": individual, "plot": plot, "rows_explained": len(rows)})
