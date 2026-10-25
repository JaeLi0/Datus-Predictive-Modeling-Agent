# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""leakage_check: find features that would not be available at prediction time, or splits that let test
information into training.

Two families of checks.

Provenance (exact, needs a feature definition):
- sql_time_filter: parse the feature SQL with SQLGlot, collect bounds on the time column, and require an
  upper bound strictly before the cutoff. Missing or later bounds mean the feature reads the future.
- window_offsets: a lag/rolling definition covers offsets [hi - window + 1, hi] relative to the target
  period (hi = -shift, centered windows reach forward by window // 2); every offset must be <= -gap where
  gap is the forecast horizon.

Empirical (needs only data, catches undocumented columns):
- too_good: one feature alone predicts the target better than a threshold (CV AUC / R^2 of a shallow tree)
- identity: a handful of numeric features reproduce the target almost exactly by a linear formula
  (e.g. casual + registered = cnt)
- naive_target_encoding: a feature is constant within the groups of some categorical column and equals
  that group's mean target, i.e. it was encoded with in-sample labels
- future_alignment: in time-ordered series a feature anticipates the target's future innovations (moves its
  own past could not predict) better than it explains the current one
- split_overlap: entities (or identical rows) appear in both train and test
- metric_gap: offline CV score much better than the out-of-time score (used after training)

Each finding has severity "leak" (drop/fix before training) or "suspect" (needs a human look), and a
suggested fix.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

from dataengineer.tools.model_tools.spec import TaskSpec, _jsonable

TOO_GOOD_AUC = 0.95
TOO_GOOD_R2 = 0.90
IDENTITY_R2 = 0.999
TE_CORR = 0.95
ALIGN_MIN = 0.3
OVERLAP_MAX_SHARE = 0.01
SAMPLE = 20000


def _finding(check: str, features, severity: str, evidence: str, fix: str) -> Dict:
    if isinstance(features, str):
        features = [features]
    return {"check": check, "features": list(features) if features else [], "severity": severity,
            "evidence": evidence, "fix": fix}


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------
def _eval_time(node: exp.Expression) -> Optional[pd.Timestamp]:
    """Evaluate a constant time expression: literals, CAST/TIMESTAMP '...', +/- INTERVAL n unit."""
    if isinstance(node, exp.Paren):
        return _eval_time(node.this)
    if isinstance(node, exp.Literal) and node.is_string:
        try:
            return pd.Timestamp(node.this)
        except Exception:
            return None
    if isinstance(node, (exp.Cast, exp.TryCast, exp.TimeStrToTime, exp.StrToTime)):
        return _eval_time(node.this)
    if isinstance(node, (exp.Add, exp.Sub, exp.DateAdd, exp.DateSub, exp.TsOrDsAdd)):
        base = _eval_time(node.this)
        delta = _interval(node.expression)
        if base is None or delta is None:
            return None
        sign = -1 if isinstance(node, (exp.Sub, exp.DateSub)) else 1
        return base + sign * delta
    return None


def _interval(node: exp.Expression) -> Optional[pd.Timedelta]:
    if isinstance(node, exp.Interval):
        try:
            n = float(str(node.this.name if hasattr(node.this, "name") else node.this).strip("'"))
        except Exception:
            return None
        unit = str(node.unit.name if node.unit is not None else "DAY").upper().rstrip("S")
        days = {"DAY": 1, "WEEK": 7, "MONTH": 30.4375, "YEAR": 365.25, "HOUR": 1 / 24}.get(unit)
        return pd.Timedelta(days=n * days) if days else None
    if isinstance(node, exp.Literal) and not node.is_string:
        return pd.Timedelta(days=float(node.this))
    return None


def _is_time_col(node: exp.Expression, time_cols: Iterable[str]) -> bool:
    names = {c.lower() for c in time_cols}
    return isinstance(node, exp.Column) and node.name.lower() in names


def sql_time_bounds(sql: str, time_cols: Iterable[str], dialect: str = "duckdb") -> Dict:
    """Upper/lower bounds the WHERE clause places on any of time_cols (constant bounds only)."""
    tree = sqlglot.parse_one(sql, read=dialect)
    uppers, lowers, unresolved = [], [], 0
    time_cols = list(time_cols)
    for cmp in tree.find_all(exp.LT, exp.LTE, exp.GT, exp.GTE, exp.EQ, exp.Between):
        if isinstance(cmp, exp.Between):
            if _is_time_col(cmp.this, time_cols):
                lo, hi = _eval_time(cmp.args["low"]), _eval_time(cmp.args["high"])
                lowers.append(lo) if lo is not None else None
                if hi is not None:
                    uppers.append((hi, True))
                else:
                    unresolved += 1
            continue
        left, right = cmp.this, cmp.expression
        if _is_time_col(left, time_cols):
            col_left, other = True, right
        elif _is_time_col(right, time_cols):
            col_left, other = False, left
        else:
            continue
        val = _eval_time(other)
        if val is None:
            unresolved += 1
            continue
        op = type(cmp)
        if op is exp.EQ:
            uppers.append((val, True))
            lowers.append(val)
            continue
        is_upper = (op in (exp.LT, exp.LTE)) == col_left
        inclusive = op in (exp.LTE, exp.GTE)
        if is_upper:
            uppers.append((val, inclusive))
        else:
            lowers.append(val)
    return {"uppers": uppers, "lowers": lowers, "unresolved": unresolved}


def check_sql_feature(name: str, sql: str, time_cols: Iterable[str], cutoff: str) -> Optional[Dict]:
    cut = pd.Timestamp(cutoff)
    try:
        b = sql_time_bounds(sql, time_cols)
    except Exception as e:
        return _finding("sql_time_filter", name, "suspect", f"SQL could not be parsed: {e}", "rewrite the feature SQL")
    if not b["uppers"]:
        if b["unresolved"]:
            return _finding("sql_time_filter", name, "suspect",
                            "time filter compares to a non-constant expression; cannot prove it ends before the cutoff",
                            "use a constant cutoff bound")
        return _finding("sql_time_filter", name, "leak", "no upper bound on the time column: aggregates all history "
                        "including rows after the cutoff", "add time < cutoff")
    upper, inclusive = min(b["uppers"], key=lambda x: (x[0], x[1]))
    if upper > cut or (upper == cut and inclusive):
        return _finding("sql_time_filter", name, "leak",
                        f"time upper bound {'<=' if inclusive else '<'} {upper} is not before cutoff {cut}",
                        "end the feature window strictly before the cutoff")
    return None


def window_offsets(defn: Dict) -> tuple:
    """(lo, hi) offsets in periods relative to the target period covered by a lag/rolling definition."""
    shift = int(defn.get("shift", 0))
    window = int(defn.get("window", 1))
    hi = -shift
    if defn.get("center"):
        hi += window // 2
    return hi - window + 1, hi


def check_window_feature(name: str, defn: Dict, gap: int) -> Optional[Dict]:
    lo, hi = window_offsets(defn)
    if hi > -gap:
        when = "the target period itself" if hi == 0 else ("future periods" if hi > 0 else f"t{hi:+d}")
        return _finding("window_offsets", name, "leak",
                        f"uses offsets [{lo}, {hi}] relative to the target; reaches {when}, but only values at "
                        f"t-{gap} or earlier are known at forecast time",
                        f"recompute with shift >= {gap} and center=False")
    return None


def provenance_checks(spec: TaskSpec, definitions: List[Dict], cutoffs: Optional[List[str]] = None) -> List[Dict]:
    """definitions: feature meta from build_features and/or a feature catalog. Each item may carry `sql`
    (+ `time_expr`) or a window definition (`template` lag/rolling with shift/window/center)."""
    out = []
    gap = max(1, int(spec.horizon or 1)) if spec.task_type == "timeseries" else 1
    for d in definitions:
        name = d.get("name")
        if not name:
            continue
        if d.get("sql"):
            raw = d.get("time_column") or d.get("time_expr") or spec.time_col or ""
            raw = raw if isinstance(raw, list) else [raw]
            time_cols = [str(c).split(".")[-1].strip('"') for c in raw if c]
            cut = d.get("cutoff")
            try:
                pd.Timestamp(cut) if cut else None
            except (ValueError, TypeError):
                cut = None  # e.g. the literal "{cutoff}" copied from a catalog: use the run's own cutoff
            cut = cut or (min(cutoffs) if cutoffs else spec.cutoff)
            if not cut:
                continue
            if not time_cols:
                out.append(_finding("sql_time_filter", name, "suspect", "time column unknown; cannot check the SQL "
                                    "time filter", "pass time_column with the definition"))
                continue
            f = check_sql_feature(name, d["sql"].replace("{cutoff}", str(pd.Timestamp(cut))), time_cols, cut)
            if f:
                out.append(f)
        elif d.get("template") in ("lag", "rolling") or "shift" in d or d.get("center"):
            if d.get("template") == "lag":
                d = {**d, "window": 1}
            f = check_window_feature(name, d, gap)
            if f:
                out.append(f)
    return out


# ---------------------------------------------------------------------------------------------
# empirical
# ---------------------------------------------------------------------------------------------
def _sample(df: pd.DataFrame, n: int = SAMPLE, seed: int = 0) -> pd.DataFrame:
    return df.sample(n, random_state=seed) if len(df) > n else df


def _y_numeric(y: pd.Series, spec: TaskSpec) -> pd.Series:
    if spec.task_type == "binary":
        pos = spec.positive_label
        return (y.astype(str) == str(pos)).astype(int) if pos is not None else pd.factorize(y)[0]
    if spec.task_type == "multiclass":
        return pd.Series(pd.factorize(y)[0], index=y.index)
    return pd.to_numeric(y, errors="coerce")


def _single_feature_score(x: pd.Series, y: pd.Series, classification: bool) -> Optional[float]:
    from sklearn.model_selection import cross_val_predict
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if x.dtype == object or str(x.dtype) == "category":
        x = pd.Series(pd.factorize(x)[0], index=x.index)
    x = pd.to_numeric(x, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 50 or x[ok].nunique() < 2:
        return None
    X = x[ok].to_numpy().reshape(-1, 1)
    yy = y[ok].to_numpy()
    if classification:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(yy)) != 2:
            return None
        p = cross_val_predict(DecisionTreeClassifier(max_depth=4, min_samples_leaf=20), X, yy, cv=3,
                              method="predict_proba")[:, 1]
        return float(roc_auc_score(yy, p))
    from sklearn.metrics import r2_score

    p = cross_val_predict(DecisionTreeRegressor(max_depth=6, min_samples_leaf=20), X, yy, cv=3)
    return float(r2_score(yy, p))


def too_good_check(df: pd.DataFrame, spec: TaskSpec, features: List[str]) -> List[Dict]:
    s = _sample(df)
    y = _y_numeric(s[spec.target], spec)
    out = []
    clf = spec.task_type == "binary"
    thr = TOO_GOOD_AUC if clf else TOO_GOOD_R2
    for f in features:
        score = _single_feature_score(s[f], y, clf)
        if score is not None and score >= thr:
            out.append(_finding("too_good", f, "leak",
                                f"this feature alone reaches {'AUC' if clf else 'R2'} {score:.3f} (threshold {thr}); "
                                "a single legitimate feature rarely explains the target this well",
                                "confirm when the value becomes known; drop it if it is recorded after the outcome"))
    return out


def identity_check(df: pd.DataFrame, spec: TaskSpec, features: List[str], max_terms: int = 4) -> List[Dict]:
    """Greedy forward search for a small linear formula of features that reproduces the target."""
    if spec.is_classification:
        return []
    s = _sample(df)
    y = pd.to_numeric(s[spec.target], errors="coerce")
    num = [f for f in features if pd.api.types.is_numeric_dtype(s[f]) and s[f].notna().mean() > 0.95]
    if not num or y.var() == 0:
        return []
    X = s[num].astype(float)
    ok = X.notna().all(axis=1) & y.notna()
    X, yv = X[ok], y[ok].to_numpy()
    chosen: List[str] = []
    best = 0.0
    tss = ((yv - yv.mean()) ** 2).sum()
    for _ in range(max_terms):
        cand_best = None
        for f in num:
            if f in chosen:
                continue
            A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in chosen + [f]])
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            r2 = 1 - ((yv - A @ coef) ** 2).sum() / tss
            if cand_best is None or r2 > cand_best[1]:
                cand_best = (f, r2)
        if cand_best is None:
            break
        chosen.append(cand_best[0])
        best = cand_best[1]
        if best >= IDENTITY_R2:
            return [_finding("identity", chosen, "leak",
                             f"target is a linear function of {chosen} (R2 {best:.5f}); these are components or "
                             "restatements of the target", "drop these features")]
    return []


def naive_te_check(df: pd.DataFrame, spec: TaskSpec, features: List[str], group_cols: List[str]) -> List[Dict]:
    s = df  # no sampling: group means need every row of the group
    y = _y_numeric(s[spec.target], spec).astype(float)
    out = []
    for g in group_cols:
        if g not in s.columns:
            continue
        ng = s[g].nunique()
        if ng < 2 or ng > len(s) / 2:
            continue
        gmean = y.groupby(s[g]).transform("mean")
        for f in features:
            if f == g or not pd.api.types.is_numeric_dtype(s[f]):
                continue
            within = s.groupby(g)[f].transform("std").fillna(0)
            if (within < 1e-9).mean() < 0.99:
                continue
            c = np.corrcoef(s[f].fillna(s[f].mean()), gmean)[0, 1]
            if np.isfinite(c) and c >= TE_CORR:
                out.append(_finding("naive_target_encoding", f, "leak",
                                    f"constant within each {g!r} group and equal to that group's mean target (corr {c:.3f}): "
                                    "encoded with the rows' own labels",
                                    f"replace with an out-of-fold target encoding of {g!r} (template target_encoding)"))
    return out


def _innovations(d: pd.DataFrame, spec: TaskSpec, grp: pd.Series, n_lags: int) -> pd.Series:
    """Residual of a pooled AR model on lags gap..gap+n_lags-1 (series-demeaned): the part of y_t that was
    unpredictable from the target's own past at forecast time."""
    gap = max(1, int(spec.horizon or 1)) if spec.task_type == "timeseries" else 1
    y = pd.to_numeric(d[spec.target], errors="coerce")
    y = y - y.groupby(grp).transform("mean")
    X = pd.concat({k: y.groupby(grp).shift(k) for k in range(gap, gap + n_lags)}, axis=1)
    ok = X.notna().all(axis=1) & y.notna()
    e = pd.Series(np.nan, index=d.index)
    if ok.sum() < 3 * (n_lags + 1):
        return e
    A = np.column_stack([np.ones(int(ok.sum())), X[ok].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
    e[ok] = y[ok].to_numpy() - A @ coef
    return e


def future_alignment_check(df: pd.DataFrame, spec: TaskSpec, features: List[str], k_max: int = 3) -> List[Dict]:
    """Within each series, correlate a feature with the target's innovations (what its own past could not
    predict). Information known at time t cannot anticipate innovations after t; a feature that correlates
    with future innovations more than with the current one was computed from future values."""
    if not spec.time_col or spec.time_col not in df.columns:
        return []
    keys = [c for c in (spec.entity, spec.time_col) if c and c in df.columns]
    d = df.sort_values(keys)
    grp = d[spec.entity] if spec.entity and spec.entity in d.columns else pd.Series(0, index=d.index)
    e = _innovations(d, spec, grp, n_lags=max(3, int(spec.season_length or 7)))
    if e.notna().sum() < 50:
        return []
    out = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(d[f]):
            continue
        x = d[f] - d[f].groupby(grp).transform("mean")
        cur = abs(x.corr(e))
        fut = np.nanmax([abs(x.corr(e.groupby(grp).shift(-k))) for k in range(1, k_max + 1)])
        if np.isfinite(fut) and fut >= ALIGN_MIN and fut > (cur if np.isfinite(cur) else 0):
            out.append(_finding("future_alignment", f, "leak",
                                f"correlates with the target's unpredictable moves in the next {k_max} periods "
                                f"(|r| {fut:.2f}) more than with the current one (|r| {cur:.2f}); information "
                                "available at forecast time cannot do that", "recompute from data before the forecast origin only"))
    return out


def split_overlap_check(df: pd.DataFrame, spec: TaskSpec, split: pd.DataFrame, features: List[str]) -> List[Dict]:
    out = []
    train, test = split["role"].to_numpy() == "train", split["role"].to_numpy() == "test"
    if spec.entity and spec.entity in df.columns and spec.task_type != "timeseries" and "snapshot" not in df.columns:
        ent = df[spec.entity].astype(str).to_numpy()
        overlap = np.isin(ent[test], ent[train]).mean() if test.any() else 0
        if overlap > 0:
            out.append(_finding("split_overlap", spec.entity, "leak",
                                f"{overlap:.1%} of test rows belong to entities also in train",
                                "split by entity (method group or time)"))
    cols = [f for f in features if f in df.columns]
    if cols and test.any():
        tr = pd.util.hash_pandas_object(df.loc[train, cols], index=False)
        te = pd.util.hash_pandas_object(df.loc[test, cols], index=False)
        share = float(te.isin(set(tr)).mean())
        if share > OVERLAP_MAX_SHARE:
            out.append(_finding("split_overlap", "<all features>", "suspect",
                                f"{share:.1%} of test rows have an identical feature vector in train (duplicated records?)",
                                "deduplicate or split by the entity the duplicates share"))
    return out


def metric_gap_check(cv_score: float, holdout_score: float, metric: str, higher_is_better: bool,
                     rel_tol: float = 0.1) -> List[Dict]:
    if cv_score is None or holdout_score is None:
        return []
    gap = (cv_score - holdout_score) if higher_is_better else (holdout_score - cv_score)
    base = abs(cv_score) or 1.0
    if gap / base > rel_tol:
        return [_finding("metric_gap", [], "suspect",
                         f"offline CV {metric} {cv_score:.4f} vs out-of-time/holdout {holdout_score:.4f} "
                         f"({gap / base:.0%} worse): validation is leakier or less realistic than deployment",
                         "re-check the split method and features that differ between periods")]
    return []


def run_checks(df: pd.DataFrame, spec: TaskSpec, features: List[str], definitions: Optional[List[Dict]] = None,
               split: Optional[pd.DataFrame] = None, group_cols: Optional[List[str]] = None,
               cutoffs: Optional[List[str]] = None, checks: Optional[List[str]] = None) -> Dict:
    """Run all applicable checks. Features with a passing provenance definition skip the empirical
    too_good / future_alignment checks (they are proven point-in-time)."""
    checks = set(checks or ["provenance", "too_good", "identity", "naive_target_encoding", "future_alignment",
                            "split_overlap"])
    findings: List[Dict] = []
    definitions = definitions or []
    if "provenance" in checks:
        findings += provenance_checks(spec, definitions, cutoffs)
    defined = {d["name"] for d in definitions if d.get("name")}
    flagged = {f for x in findings for f in x["features"]}
    proven = defined - flagged
    unproven = [f for f in features if f not in proven]
    if group_cols is None:
        # categorical keys, including integer-coded ones (codes for region, traffic source, age ...)
        group_cols = [c for c in df.columns if c != spec.target and (
            df[c].dtype == object or str(df[c].dtype) == "category"
            or (pd.api.types.is_integer_dtype(df[c]) and 2 <= df[c].nunique() <= min(200, len(df) // 20)))]
        if spec.entity and spec.entity not in group_cols:
            group_cols.append(spec.entity)
    if "too_good" in checks:
        findings += too_good_check(df, spec, [f for f in unproven if f not in flagged])
    if "identity" in checks:
        findings += identity_check(df, spec, features)
    if "naive_target_encoding" in checks:
        findings += naive_te_check(df, spec, features, group_cols)
    if "future_alignment" in checks and spec.time_col:
        findings += future_alignment_check(df, spec, [f for f in unproven if f not in flagged])
    if "split_overlap" in checks and split is not None:
        findings += split_overlap_check(df, spec, split, features)
    leaks = sorted({f for x in findings if x["severity"] == "leak" for f in x["features"] if f in df.columns})
    return _jsonable({"findings": findings, "leaky_features": leaks,
                      "suspect_features": sorted({f for x in findings if x["severity"] == "suspect"
                                                  for f in x["features"] if f in df.columns} - set(leaks)),
                      "checked_features": len(features), "proven_point_in_time": sorted(proven & set(features))})
