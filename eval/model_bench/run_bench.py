"""Modeling benchmark: reference pipeline, three comparison groups, independent scoring.

Groups
- a: full gen_model agent            (model_once.py --group a)
- b: agent without guardrails         (no leakage_check, split method not enforced)
- c: pure FLAML on the raw wide table, random split, no agent

Independent scoring (the agent never scores itself)
- Every task has a fixed ground-truth (GT) test set: the GT split method (inject.py ground_truth) applied with
  seed 0 to the task table. The reference pipeline below defines it and stores its row keys (__rid).
- For an agent run the scorer takes the agent's final feature table, its chosen model and parameters and its
  excluded columns, refits that model on the rows that are legitimately before / outside the GT test set, and
  scores the GT test rows. Columns the scorer knows are not available at prediction time (injected leaks and
  real leaks, and every feature derived from them) are blanked on the test rows, as they would be in production.
  A model that relied on them loses accuracy; a model that dropped them is unaffected.
- The agent's own test score (on its own split) is reported next to it as "self-reported".
- Reference pipeline (= "clean reference"): GT spec, leaks removed, standard lag features for time series,
  the same train_select + evaluate tools. Normalized score = (score - baseline) / (reference - baseline) on the
  GT test set, so 1.0 means as good as the clean reference and 0 means no better than the naive baseline.

Usage
  python eval/model_bench/run_bench.py reference [ids]
  python eval/model_bench/run_bench.py run --round round1 --groups a,b,c --repeats 3 --workers 2 [ids]
  python eval/model_bench/run_bench.py score --round round1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataengineer.tools.func_tool.join_path_tools import JoinGraphBuilder, solve  # noqa: E402
from dataengineer.tools.model_tools import train as T  # noqa: E402
from dataengineer.tools.model_tools.spec import TaskSpec  # noqa: E402
from dataengineer.tools.model_tools.tools import ModelingCore  # noqa: E402
from dataengineer.tools.model_tools.workspace import RunWorkspace  # noqa: E402
sys.path.insert(0, str(ROOT / "eval/model_bench"))
from inject import _gt_spec, load_tasks  # noqa: E402

PREP = ROOT / "eval/model_bench/prepared"
OUT = ROOT / "outputs/experiments/model_bench"
REF = OUT / "reference"
TIME_BUDGET = 40


def _vis(tid):
    return json.loads((PREP / tid / "visible.json").read_text())


def _hid(tid):
    return json.loads((PREP / tid / "hidden.json").read_text())


def leak_columns(hid: dict) -> set:
    return {n for n, r in hid["roles"].items() if r["role"] == "leak" and r["leak_type"] != "duplicate_entity"} \
        | set(hid.get("real_leaks", []))


def dup_columns(hid: dict) -> set:
    return {n for n, r in hid["roles"].items() if r["role"] == "leak" and r["leak_type"] == "duplicate_entity"}


def decoy_columns(hid: dict) -> set:
    return {n for n, r in hid["roles"].items() if r["role"] == "decoy"}


def core_for(vis: dict, runs_dir: Path, time_budget: float = TIME_BUDGET):
    con = duckdb.connect(str(ROOT / vis["db"]), read_only=True)
    sem = ROOT / vis["semantic_dir"] if vis.get("semantic_dir") else None
    graph = JoinGraphBuilder(lambda sql: con.execute(sql).fetchall(), "duckdb", sem).build() if sem else None
    core = ModelingCore(runs_dir, query=lambda sql: con.execute(sql).df(), semantic_dir=sem,
                        join_solver=(lambda tables: solve(graph, tables)) if graph else None,
                        time_budget=time_budget, db_label=str(ROOT / vis["db"]))
    return core, con


def predict_windows(name: str, params, frame: pd.DataFrame, spec: TaskSpec, cols: dict, tr: np.ndarray,
                    te: np.ndarray, h: int) -> np.ndarray:
    """Time series scored over several forecast origins: the test periods are cut into windows of h periods.
    Per-series models (seasonal naive, ETS) are refitted at every origin on all history before it; feature
    models are fitted once on the training rows (their lag features already end h periods before each target)."""
    if name not in ("seasonal_naive", "ets"):
        return T.fit_predict(name, frame, spec, cols, tr, te, params)[0]
    t = pd.to_datetime(frame[spec.time_col])
    periods = np.sort(t[te].unique())
    out = pd.Series(np.nan, index=frame.index)
    for k in range(0, len(periods), h):
        w = periods[k:k + h]
        m = te & t.isin(w).to_numpy()
        hist = (t < w[0]).to_numpy()
        out[m] = T.fit_predict(name, frame, spec, cols, hist, m, params)[0]
    return out[te].to_numpy()


def ts_templates(spec: dict) -> list:
    h = int(spec["horizon"])
    if spec.get("freq", "D").upper().startswith("W"):
        return [{"template": "lag", "lags": [h, 2 * h, 52]}, {"template": "rolling", "windows": [4, 12], "stats": ["mean"]},
                {"template": "date_parts", "parts": ["weekofyear", "month"]}]
    return [{"template": "lag", "lags": [h, 2 * h, 3 * h, 4 * h]},
            {"template": "rolling", "windows": [7, 28], "stats": ["mean", "std"]},
            {"template": "date_parts", "parts": ["dayofweek", "month"]}]


# ---------------------------------------------------------------------------------------------
# reference pipeline (defines the GT test set)
# ---------------------------------------------------------------------------------------------
def reference(task: dict) -> dict:
    tid = task["id"]
    vis, hid = _vis(tid), _hid(tid)
    runs = REF / "runs"
    core, con = core_for(vis, runs)
    gt = task["ground_truth"]
    spec = _gt_spec(task)
    leaks = leak_columns(hid)
    out = {}
    for variant in ("clean", "wide"):
        run = f"{tid}_{variant}"
        shutil.rmtree(runs / run, ignore_errors=True)
        src = {"label": gt["label"], "population": gt["population"]} if gt.get("label") else {"table": vis["tables"][0]}
        res = core.define_task(run, spec, src)
        if not res["ok"]:
            raise RuntimeError(res["errors"])
        tpl = [{"template": "sql_feature", "name": f["name"], "sql": f["sql"]} for f in vis["feature_catalog"]
               if f.get("sql") and (variant == "wide" or f["name"] not in leaks)]
        if variant == "clean":
            tpl.append({"template": "exclude", "columns": sorted(leaks | dup_columns(hid) - {spec.get("entity")})})
            if gt["task_type"] == "timeseries":
                tpl += ts_templates(spec)
        if tpl:
            core.build_features(run, tpl)
        core.make_split(run, gt["acceptable_splits"][0], test_size=0.2, n_folds=3)
        if variant == "wide":
            continue
        core.train_select(run)
        ev = core.evaluate(run)
        pm = spec["primary_metric"]
        out.update(reference=ev["test"][pm], baseline=ev["baseline"][pm], model=ev["model"], metric=pm,
                   all_test=ev["test"], all_baseline=ev["baseline"])
    ws = RunWorkspace(runs / f"{tid}_clean")
    df = ws.load_frame("raw")
    split_frame = ws.load_frame("split")
    test_rids = df.loc[(split_frame["role"] == "test").to_numpy(), "__rid"]
    ref_spec = ws.load_spec()
    if gt.get("test_windows"):
        # GT test = the last test_windows * horizon periods, scored origin by origin
        h = int(ref_spec.horizon)
        feat = ws.load_frame(ws.latest_table())
        t = pd.to_datetime(feat[ref_spec.time_col])
        start = np.sort(t.unique())[-int(gt["test_windows"]) * h]
        te, tr = (t >= start).to_numpy(), (t < start).to_numpy()
        cols = T.feature_columns(feat, ref_spec, ws.load_json("feature_meta", default=[]))
        y = T.target_vector(feat, ref_spec)[0][te]
        scale = T.mase_scale(feat, ref_spec, tr)
        pm = ref_spec.primary_metric
        out["reference"] = T.score(pm, y, predict_windows(out["model"]["model"], out["model"].get("params"), feat,
                                                          ref_spec, cols, tr, te, h), scale)
        out["baseline"] = T.score(pm, y, predict_windows("seasonal_naive", None, feat, ref_spec, cols, tr, te, h), scale)
        test_rids = feat.loc[te, "__rid"]
    info = {"task": tid, "metric": out["metric"], "reference": out["reference"], "baseline": out["baseline"],
            "reference_model": out["model"], "reference_all": out["all_test"], "baseline_all": out["all_baseline"],
            "test_rids": [str(r) for r in test_rids], "split": gt["acceptable_splits"][0],
            "spec": ref_spec.model_dump(), "wide_run": f"{tid}_wide"}
    REF.mkdir(parents=True, exist_ok=True)
    (REF / f"{tid}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    con.close()
    print(f"{tid}: reference {out['metric']}={out['reference']:.4f} baseline={out['baseline']:.4f} "
          f"model={out['model']['label']} test_rows={len(test_rids)}")
    return info


def gt_context(tid: str) -> dict:
    """Ground-truth test rows and, per raw row id, the time / entity the scorer needs."""
    info = json.loads((REF / f"{tid}.json").read_text())
    ws = RunWorkspace(REF / "runs" / f"{tid}_clean")
    raw = ws.load_frame("raw")
    spec = TaskSpec(**info["spec"])
    raw["__rid"] = raw["__rid"].astype(str)
    test = set(info["test_rids"])
    ctx = {"info": info, "spec": spec, "raw": raw.set_index("__rid"), "test": test}
    if "snapshot" in raw.columns:
        cut = max(raw.loc[raw["__rid"].isin(test), "snapshot"])
        ctx["test_cutoff"] = pd.Timestamp(cut)
    elif spec.time_col:
        ctx["test_start"] = pd.to_datetime(raw.loc[raw["__rid"].isin(test), spec.time_col]).min()
    return ctx


def train_mask(rids: pd.Series, ctx: dict) -> np.ndarray:
    """Rows that may be used to train a model evaluated on the GT test set."""
    not_test = ~rids.isin(ctx["test"]).to_numpy()
    if "test_cutoff" in ctx:
        h = int(ctx["spec"].horizon or 0)
        snap = pd.to_datetime(rids.str.split("|").str[0], errors="coerce")
        return not_test & (snap + pd.Timedelta(days=h) <= ctx["test_cutoff"]).to_numpy()
    if "test_start" in ctx:
        t = pd.to_datetime(ctx["raw"][ctx["spec"].time_col].reindex(rids).to_numpy())
        return not_test & (t < ctx["test_start"])
    if ctx["spec"].entity and ctx["spec"].entity in ctx["raw"].columns:
        ent = ctx["raw"][ctx["spec"].entity].astype(str)
        test_ents = set(ent.reindex(list(ctx["test"])).dropna())
        return not_test & ~ent.reindex(rids).isin(test_ents).to_numpy()
    return not_test


# ---------------------------------------------------------------------------------------------
# group c: pure FLAML
# ---------------------------------------------------------------------------------------------
def run_flaml(tid: str, out: Path, seed: int) -> dict:
    from flaml import AutoML

    ctx = gt_context(tid)
    spec = ctx["spec"]
    hid = _hid(tid)
    ws = RunWorkspace(REF / "runs" / ctx["info"]["wide_run"])
    df = ws.load_frame(ws.latest_table())
    rids = df["__rid"].astype(str)
    y, _ = T.target_vector(df, spec)
    X = df.drop(columns=[spec.target, "__rid"])
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
        elif pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype("int64") // 10**9
    task = "classification" if spec.is_classification else "regression"
    metric = {"roc_auc": "roc_auc", "pr_auc": "ap", "rmse": "rmse", "mae": "mae", "wape": "mae", "mase": "mae"}[spec.primary_metric]
    t0 = time.time()

    def fit(Xtr, ytr):
        m = AutoML()
        m.fit(Xtr, ytr, task=task, metric=metric, time_budget=TIME_BUDGET, seed=seed, verbose=0,
              n_jobs=int(os.environ.get("MODEL_N_JOBS", "2")))
        return m

    def pred(m, Xte):
        return m.predict_proba(Xte)[:, 1] if spec.task_type == "binary" else m.predict(Xte)

    rng = np.random.RandomState(seed)
    rand_test = rng.rand(len(df)) < 0.2
    m = fit(X[~rand_test], y[~rand_test])
    scale = T.mase_scale(df, spec, ~rand_test) if spec.task_type == "timeseries" else 1.0
    self_score = T.score(spec.primary_metric, y[rand_test], pred(m, X[rand_test]), scale)
    tr = train_mask(rids, ctx)
    te = rids.isin(ctx["test"]).to_numpy()
    m2 = fit(X[tr], y[tr])
    Xte = X[te].copy()
    for c in leak_columns(hid) & set(Xte.columns):
        Xte[c] = np.nan
    scale = T.mase_scale(df, spec, tr) if spec.task_type == "timeseries" else 1.0
    indep = T.score(spec.primary_metric, y[te], pred(m2, Xte), scale)
    rec = {"task": tid, "group": "c", "seed": seed, "completed": True, "self_score": self_score, "indep_score": indep,
           "seconds": round(time.time() - t0, 1), "tokens": 0, "interactions": 0,
           "leaks_excluded": [], "leaks_used": sorted(leak_columns(hid) & set(X.columns)),
           "dup_handled": {d: False for d in dup_columns(hid)},
           "decoys_used": sorted(decoy_columns(hid) & set(X.columns)), "decoys_excluded": [],
           "model": m2.best_estimator, "split": "random"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


# ---------------------------------------------------------------------------------------------
# agent runs (a, b)
# ---------------------------------------------------------------------------------------------
def run_agent(tid: str, group: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "MODEL_N_JOBS": os.environ.get("MODEL_N_JOBS", "2")}
    with open(out / "agent.log", "w") as log:
        subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "eval/model_bench/model_once.py"), "--task", tid,
                        "--group", group, "--out", str(out), "--time-budget", str(TIME_BUDGET)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)


def _inputs_using(cols_used: list, meta: list) -> dict:
    """feature -> set of original columns it was derived from (through feature_meta sources)."""
    src = {m["name"]: m.get("sources", []) for m in meta if m.get("name")}

    def roots(c, seen=()):
        if c in seen or c not in src or not src[c]:
            return {c}
        out = {c}
        for s in src[c]:
            out |= roots(s, seen + (c,))
        return out

    return {c: roots(c) for c in cols_used}


def score_agent(tid: str, group: str, out: Path) -> dict:
    rec = json.loads((out / "record.json").read_text()) if (out / "record.json").exists() else {"error": "no record"}
    hid = _hid(tid)
    res = {"task": tid, "group": group, "seconds": rec.get("seconds"), "tokens": rec.get("tokens"),
           "interactions": rec.get("interactions"), "error": rec.get("error"),
           "tool_calls": len(rec.get("trace", [])), "tool_errors": sum(1 for t in rec.get("trace", []) if not t["ok"])}
    run_id = rec.get("run_id")
    run = out / "runs" / run_id if run_id else None
    res["completed"] = bool(run and (run / "evaluation.json").exists())
    res["exported"] = bool(run and (run / "export" / "model_card.md").exists())
    if not res["completed"]:
        return res
    ws = RunWorkspace(run)
    spec = ws.load_spec()
    gtx = gt_context(tid)
    gspec = gtx["spec"]
    sel, ev = ws.load_json("selection"), ws.load_json("evaluation")
    meta = ws.load_json("feature_meta", default=[])
    df = ws.load_frame(ws.latest_table())
    split = ws.load_frame("split")
    # regression and time-series framings of a numeric target are both scorable with the GT metric
    compatible = spec.task_type == gspec.task_type or {spec.task_type, gspec.task_type} == {"regression", "timeseries"}
    res.update(task_type=spec.task_type, task_type_ok=compatible, model=ev["model"],
               split=json.loads((run / "split.json").read_text()).get("method"),
               agent_metric=spec.primary_metric, cv_primary=ev.get("cv_primary"))
    # self-reported on the agent's own test split, in the GT metric
    preds = ws.load_frame("test_predictions")
    m = gspec.primary_metric
    scale = T.mase_scale(df, spec, (split["role"] == "train").to_numpy()) if spec.task_type == "timeseries" else 1.0
    if compatible and spec.task_type != "multiclass":
        p = preds["prediction"].to_numpy()
        yv = preds["actual"].to_numpy()
        if spec.task_type == "binary" and str(spec.positive_label) != str(gspec.positive_label):
            p, yv = 1 - p, 1 - yv
        res["self_score"] = T.score(m, yv, p, scale)
    # leak handling
    inputs = sel["features"]["numeric"] + sel["features"]["categorical"] + sel.get("target_encoded", [])
    if ev["model"]["model"] in T.BASELINES or ev["model"]["model"] == "ets":
        inputs = []
    used_roots = set().union(*_inputs_using(inputs, meta).values()) if inputs else set()
    leaks, decoys = leak_columns(hid), decoy_columns(hid)
    present = set(ws.load_frame("raw").columns)
    if "snapshot" in df.columns:
        present |= {f["name"] for f in _vis(tid)["feature_catalog"]}
    res["leaks_used"] = sorted(leaks & used_roots)
    res["leaks_excluded"] = sorted(leaks - used_roots)
    res["decoys_used"] = sorted(decoys & used_roots)
    res["decoys_excluded"] = sorted(decoys - used_roots)
    leak_rep = ws.load_json("leakage") if ws.has("leakage") else None
    res["tool_flagged"] = sorted(set(leak_rep["leaky_features"]) | set(leak_rep["suspect_features"])) if leak_rep else []
    res["dup_handled"] = {}
    rids = df["__rid"].astype(str)
    keyed = False
    # rows loaded through the agent's own SQL carry its row order, not the table's: re-key time-series rows by
    # (series, period) against the GT table
    if "snapshot" not in df.columns and gspec.time_col and spec.time_col in df.columns:
        graw = gtx["raw"].reset_index()
        g_ent = gspec.entity if gspec.entity in graw.columns else None
        a_ent = spec.entity if (spec.entity in df.columns and g_ent) else None
        gkey = list(zip(graw[g_ent].astype(str) if g_ent else [""] * len(graw), pd.to_datetime(graw[gspec.time_col])))
        akey = list(zip(df[a_ent].astype(str) if a_ent else [""] * len(df), pd.to_datetime(df[spec.time_col])))
        lookup = dict(zip(gkey, graw["__rid"].astype(str)))
        mapped = pd.Series([lookup.get(k) for k in akey], index=df.index)
        if mapped.notna().mean() > 0.95 and len(set(gkey)) == len(gkey):
            rids = mapped.fillna("__none__")
            keyed = True
    for d in dup_columns(hid):
        ent = gtx["raw"][d].astype(str).reindex(rids) if d in gtx["raw"].columns else None
        if ent is None or "__rid" not in df.columns:
            continue
        roles = split["role"].to_numpy()
        tr_e = set(ent[roles == "train"].dropna())
        te_e = ent[roles == "test"].dropna()
        res["dup_handled"][d] = bool(len(te_e) and te_e.isin(tr_e).mean() < 0.001)
    # independent score on the GT test set
    te = rids.isin(gtx["test"]).to_numpy()
    res["gt_test_coverage"] = round(te.sum() / max(1, len(gtx["test"])), 3)
    aligned = True
    if "snapshot" not in df.columns and te.any() and not keyed:
        shared = rids[te][:300]
        a_y = df.loc[te, spec.target].head(300)
        g_y = gtx["raw"][gspec.target].reindex(shared)
        a_n, g_n = pd.to_numeric(a_y, errors="coerce").to_numpy(), pd.to_numeric(g_y, errors="coerce").to_numpy()
        if np.isfinite(a_n).all() and np.isfinite(g_n).all():
            aligned = np.isclose(a_n, g_n).mean() > 0.99
        else:
            aligned = (a_y.astype(str).to_numpy() == g_y.astype(str).to_numpy()).mean() > 0.99
    res["rows_aligned"] = bool(aligned)
    if not compatible or res["gt_test_coverage"] < 0.5 or not aligned:
        res["indep_score"] = None
        return res
    tr = train_mask(rids, gtx)
    frame = df.copy()
    blank = [c for c in frame.columns if c in used_roots and _inputs_using([c], meta)[c] & leaks]
    frame.loc[te, blank] = np.nan
    cols = T.feature_columns(frame, spec, meta, sel.get("excluded"))
    task_cfg = next(t for t in load_tasks([tid]))
    if spec.task_type == "timeseries" and task_cfg["ground_truth"].get("test_windows"):
        pred = predict_windows(ev["model"]["model"], ev["model"].get("params"), frame, spec, cols, tr, te,
                               int(gspec.horizon))
    else:
        pred, _ = T.fit_predict(ev["model"]["model"], frame, spec, cols, tr, te, ev["model"].get("params"))
    y, _ = T.target_vector(frame, spec)
    yy = y[te]
    if keyed:  # score against the untouched GT values (the agent may have cleaned/clipped its target)
        yy = pd.to_numeric(gtx["raw"][gspec.target].reindex(rids[te]), errors="coerce").to_numpy()
    if spec.task_type == "binary" and str(spec.positive_label) != str(gspec.positive_label):
        pred, yy = 1 - pred, 1 - yy
    scale = T.mase_scale(frame, spec, tr) if spec.task_type == "timeseries" else 1.0
    res["indep_score"] = T.score(m, yy, pred, scale)
    res["blanked_on_test"] = blank
    return res


def normalized(score, info):
    if score is None:
        return None
    # only defined when the clean reference beats the naive baseline on the GT test set
    higher = info["metric"] in T.HIGHER_IS_BETTER
    den = info["reference"] - info["baseline"]
    if not den or (den < 0) == higher:
        return None
    return (score - info["baseline"]) / den


def cmd_run(args):
    ids = args.ids or [t["id"] for t in load_tasks()]
    rd = OUT / args.round
    jobs = []
    for rep in range(args.repeats):
        for tid in ids:
            for g in args.groups.split(","):
                d = rd / f"{tid}_{g}_r{rep}"
                if (d / ("score.json" if g == "c" else "record.json")).exists():
                    continue
                jobs.append((tid, g, rep, d))
    print(f"{len(jobs)} jobs", flush=True)

    def work(job):
        tid, g, rep, d = job
        t0 = time.time()
        try:
            if g == "c":
                run_flaml(tid, d, seed=rep)
            else:
                run_agent(tid, g, d)
            print(f"done {tid} {g} r{rep} in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"FAILED {tid} {g} r{rep}: {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, jobs))


def cmd_score(args):
    rd = OUT / args.round
    rows = []
    for d in sorted(p for p in rd.iterdir() if p.is_dir() and not p.name.startswith("_")):
        tid, g, rep = d.name.split("_")
        if g == "c":
            if not (d / "score.json").exists():
                continue
            r = json.loads((d / "score.json").read_text())
        else:
            try:
                r = score_agent(tid, g, d)
            except Exception as e:
                r = {"task": tid, "group": g, "completed": False, "error": f"scoring failed: {e}"}
            (d / "score.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
        info = json.loads((REF / f"{tid}.json").read_text())
        r["rep"] = rep
        r["metric"] = info["metric"]
        r["normalized"] = normalized(r.get("indep_score"), info)
        n_leaks = len(leak_columns(_hid(tid))) + len(dup_columns(_hid(tid)))
        caught = len(r.get("leaks_excluded", [])) + sum(bool(v) for v in r.get("dup_handled", {}).values())
        r["leak_recall"] = caught / n_leaks if n_leaks and r.get("completed") else None
        r["decoy_false_alarms"] = len(r.get("decoys_excluded", [])) if r.get("completed") else None
        hidden = {n for n, x in _hid(tid)["roles"].items() if x.get("hidden") and x["role"] == "leak"}
        if hidden and r.get("completed"):
            r["hidden_leak_recall"] = len(hidden & set(r.get("leaks_excluded", []))) / len(hidden)
            shown = leak_columns(_hid(tid)) - hidden
            r["described_leak_recall"] = len(shown & set(r.get("leaks_excluded", []))) / len(shown) if shown else None
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = ["task", "group", "rep", "completed", "exported", "metric", "self_score", "indep_score", "normalized",
            "leak_recall", "hidden_leak_recall", "described_leak_recall", "decoy_false_alarms", "leaks_used", "split", "model", "seconds", "tokens", "interactions",
            "tool_errors", "error"]
    df[[c for c in keep if c in df.columns]].to_csv(rd / "results.csv", index=False)
    print(df[[c for c in keep[:12] if c in df.columns]].to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("ids", nargs="*")
    r2 = sub.add_parser("run")
    r2.add_argument("ids", nargs="*")
    r2.add_argument("--round", default="round1")
    r2.add_argument("--groups", default="a,b,c")
    r2.add_argument("--repeats", type=int, default=3)
    r2.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("score")
    s.add_argument("--round", default="round1")
    args = ap.parse_args()
    if args.cmd == "reference":
        for t in load_tasks(args.ids or None):
            reference(t)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
