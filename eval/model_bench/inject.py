"""Prepare benchmark tasks: build each task's table, inject leak and decoy features, and write what the
agent may see apart from what only the scorer may see.

For every task in eval/model_bench/tasks/*.yaml:
- eval/model/bench/<id>.duckdb: a copy of the task data with injected columns (source DBs untouched)
- eval/model_bench/prepared/<id>/visible.json: request, table names, data dictionary for injected columns
  and feature-catalog definitions (what a feature store would show)
- eval/model_bench/prepared/<id>/hidden.json: roles (leak/decoy/neutral) and leak types, for scoring only

`--validate` then runs the deterministic leakage_check with the ground-truth spec and split on the prepared
data and reports, per task, which leaks the tools catch and which decoys they wrongly flag. That measures
the tools alone; the agent's own recall/false alarms are measured in stage B.

Usage: .venv/bin/python eval/model_bench/inject.py [--validate] [C1 C2 ...]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataengineer.tools.func_tool.join_path_tools import JoinGraphBuilder, solve  # noqa: E402
from dataengineer.tools.model_tools.features import oof_target_encode  # noqa: E402
from dataengineer.tools.model_tools.spec import TaskSpec  # noqa: E402
from dataengineer.tools.model_tools.tools import ModelingCore  # noqa: E402

TASKS = ROOT / "eval/model_bench/tasks"
PREP = ROOT / "eval/model_bench/prepared"
BENCH_DB = ROOT / "eval/model/bench"


def load_tasks(ids=None):
    out = []
    for f in sorted(TASKS.glob("*.yaml")):
        t = yaml.safe_load(f.read_text())
        if not ids or t["id"] in ids:
            out.append(t)
    return out


def _y01(df, gt):
    y = df[gt["target"]]
    if gt["task_type"] == "binary":
        return (y.astype(str) == str(gt["positive_label"])).astype(int)
    return pd.to_numeric(y)


def inject_columns(df: pd.DataFrame, task: dict) -> pd.DataFrame:
    inj, gt = task["injection"], task["ground_truth"]
    rng = np.random.RandomState(inj["seed"])
    y = _y01(df, gt)
    order, group = inj.get("order_by"), inj.get("group_by")
    if order:
        df = df.sort_values([c for c in (group, order) if c]).reset_index(drop=True)
        y = _y01(df, gt)
    dup_specs = []
    for c in inj.get("columns", []):
        kind, name = c["kind"], c["name"]
        if kind == "synthetic_key":
            df[name] = [f"K{v:05d}" for v in rng.randint(0, c["n_groups"], len(df))]
        elif kind == "target_proxy":
            if gt["task_type"] == "binary":
                lab = np.where(rng.rand(len(df)) < c["overlap"], 1 - y, y)
                if c.get("levels"):
                    pos, neg = c["levels"]["pos"], c["levels"]["neg"]
                    df[name] = np.where(lab == 1, rng.choice(pos, len(df)), rng.choice(neg, len(df)))
                else:
                    df[name] = np.round(np.where(lab == 1, rng.normal(70, 8, len(df)), rng.normal(30, 8, len(df))), 1)
            else:
                df[name] = y * (1 + rng.normal(0, c["noise"], len(df)))
        elif kind == "naive_te":
            df[name] = y.groupby(df[c["key"]]).transform("mean").round(4)
        elif kind == "oof_te":
            df[name] = oof_target_encode(df[c["key"]].astype(str), y.astype(float), seed=inj["seed"])[0].round(4)
        elif kind == "arith":
            a, b = df[c["left"]].astype(float), df[c["right"]].astype(float)
            df[name] = {"+": a + b, "-": a - b, "*": a * b, "/": a / b.replace(0, np.nan)}[c["op"]]
        elif kind == "window":
            src = df[c["source"]].astype(float)
            g = src.groupby(df[group]) if group else src
            shifted = g.shift(c["shift"])
            sg = shifted.groupby(df[group]) if group else shifted
            w, center = int(c["window"]), bool(c.get("center", False))
            df[name] = sg.transform(lambda s: s.rolling(w, center=center, min_periods=w).agg(c.get("stat", "mean")))
        elif kind == "duplicate_entity":
            dup_specs.append(c)
        else:
            raise ValueError(f"unknown injection kind {kind}")
    for c in dup_specs:
        df.insert(0, c["name"], np.arange(len(df)))
        pick = rng.rand(len(df)) < c["share"]
        extra = pd.concat([df[pick]] * (int(c.get("copies", 2)) - 1), ignore_index=True)
        df = pd.concat([df, extra], ignore_index=True).sample(frac=1, random_state=inj["seed"]).reset_index(drop=True)
    return df


def prepare(task: dict) -> dict:
    ds = task["dataset"]
    src_db = ROOT / ds["db"]
    BENCH_DB.mkdir(parents=True, exist_ok=True)
    out_db = BENCH_DB / f"{task['id']}.duckdb"
    if out_db.exists():
        out_db.unlink()
    shutil.copy(src_db, out_db)
    con = duckdb.connect(str(out_db))
    table = ds.get("table")
    if ds.get("panel_sql"):
        table = "weekly_product_units"
        df = con.execute(ds["panel_sql"]).df()
    elif table:
        df = con.execute(f'SELECT * FROM "{table}"').df()
    else:
        df = None
    if df is not None and ds.get("sample_series"):
        ent = task["ground_truth"]["entity"]
        keep = pd.Series(sorted(df[ent].unique())).sample(ds["sample_series"], random_state=task["injection"]["seed"])
        df = df[df[ent].isin(set(keep))].reset_index(drop=True)
    if df is not None:
        df = inject_columns(df, task)
        con.register("_df", df)
        con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _df')
        con.unregister("_df")
    con.close()

    cols = task["injection"].get("columns", [])
    catalog = task["injection"].get("catalog", [])
    visible = {
        "id": task["id"], "request": task["request"], "db": str(out_db.relative_to(ROOT)),
        "tables": [table] if table else ds["tables"], "source": ds["source"],
        "semantic_dir": ds.get("semantic_dir"),
        "data_dictionary": {c["name"]: c["description"] for c in cols if not c.get("hidden")},
        "feature_catalog": [
            {"name": c["name"], "description": c["description"], "definition": c["definition"]}
            for c in cols if c.get("definition") and not c.get("hidden")
        ] + [{"name": c["name"], "description": c.get("description", ""), "sql": c["sql"]} for c in catalog],
    }
    hidden = {
        "id": task["id"], "ground_truth": task["ground_truth"], "reference": task["reference"],
        "roles": {c["name"]: {"role": c["role"], "leak_type": c.get("leak_type"), "hidden": bool(c.get("hidden"))}
                  for c in cols + catalog},
        "real_leaks": task["ground_truth"].get("real_leaks", []),
    }
    d = PREP / task["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "visible.json").write_text(json.dumps(visible, ensure_ascii=False, indent=2))
    (d / "hidden.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=2))
    n = len(df) if df is not None else None
    print(f"{task['id']}: prepared {out_db.relative_to(ROOT)} table={table or ds['tables']} rows={n}")
    return {"db": out_db, "table": table, "visible": visible, "hidden": hidden}


# ---------------------------------------------------------------------------------------------
def _gt_spec(task: dict) -> dict:
    gt = task["ground_truth"]
    keys = ("task_type", "target", "positive_label", "entity", "time_col", "cutoffs", "horizon", "freq")
    spec = {k: gt[k] for k in keys if gt.get(k) is not None}
    if gt.get("entity_after_injection"):
        spec["entity"] = gt["entity_after_injection"]
    spec["drop_columns"] = gt.get("must_drop", [])
    pm = gt.get("primary_metric")
    spec["primary_metric"] = pm[0] if isinstance(pm, list) else pm
    return {"name": task["name"], **spec}


def validate(task: dict, prepared: dict, runs_dir: Path) -> dict:
    con = duckdb.connect(str(prepared["db"]), read_only=True)
    query = lambda sql: con.execute(sql).df()  # noqa: E731
    sem = ROOT / prepared["visible"]["semantic_dir"] if prepared["visible"].get("semantic_dir") else None
    graph = None
    if sem:
        graph = JoinGraphBuilder(lambda sql: con.execute(sql).fetchall(), "duckdb", sem).build()
    core = ModelingCore(runs_dir, query=query, semantic_dir=sem,
                        join_solver=(lambda tables: solve(graph, tables)) if graph else None)
    gt = task["ground_truth"]
    run = f"validate_{task['id']}"
    shutil.rmtree(runs_dir / run, ignore_errors=True)
    spec = _gt_spec(task)
    if gt.get("label"):
        res = core.define_task(run, spec, {"label": gt["label"], "population": gt["population"]})
    else:
        res = core.define_task(run, spec, {"table": prepared["table"]})
    if not res["ok"]:
        raise RuntimeError(f"{task['id']}: define_task failed {res['errors']}")
    templates = []
    defs = []
    for f in prepared["visible"]["feature_catalog"]:
        if f.get("sql"):
            templates.append({"template": "sql_feature", "name": f["name"], "sql": f["sql"]})
        else:
            defs.append({"name": f["name"], **f["definition"]})
    core.build_features(run, templates)
    split_method = gt["acceptable_splits"][0]
    core.make_split(run, split_method, n_folds=3)
    rep = core.leakage_check(run, definitions=defs)
    roles = prepared["hidden"]["roles"]
    leaks = {n for n, r in roles.items() if r["role"] == "leak"} | set(prepared["hidden"]["real_leaks"])
    decoys = {n for n, r in roles.items() if r["role"] == "decoy"}
    flagged_leak = set(rep["leaky_features"])
    flagged_any = flagged_leak | set(rep["suspect_features"])
    caught = {n for n in leaks if n in flagged_any}
    dup = [n for n, r in roles.items() if r["role"] == "leak" and r["leak_type"] == "duplicate_entity"]
    if dup:
        # the right (group) split prevents this leak; check it is also *detected* when a random split is used
        from dataengineer.tools.model_tools import leakage as L
        from dataengineer.tools.model_tools.split import make_split

        ws = core.ws(run)
        df = ws.load_frame(ws.latest_table())
        tspec = ws.load_spec()
        rsplit, _ = make_split(df, tspec, "random", enforce=False)
        feats = core.feature_columns(ws, df, tspec)
        if L.split_overlap_check(df, tspec, rsplit, feats):
            caught.update(dup)
    summary = {
        "task": task["id"], "split": split_method, "leaks": sorted(leaks), "caught": sorted(caught),
        "missed": sorted(leaks - caught), "decoys_flagged": sorted(decoys & flagged_any),
        "other_flagged": sorted(flagged_any - leaks - decoys),
        "findings": [{k: f[k] for k in ("check", "features", "severity", "evidence")} for f in rep["findings"]],
    }
    con.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    results = []
    for task in load_tasks(args.ids):
        prepared = prepare(task)
        if args.validate:
            s = validate(task, prepared, ROOT / "eval/model_bench/prepared/_runs")
            results.append(s)
            print(f"  caught {len(s['caught'])}/{len(s['leaks'])} missed={s['missed']} decoys_flagged={s['decoys_flagged']} "
                  f"other={s['other_flagged']}")
    if results:
        out = ROOT / "eval/model_bench/prepared/validation.json"
        prev = json.loads(out.read_text()) if out.exists() else {}
        prev.update({r["task"]: r for r in results})
        out.write_text(json.dumps(prev, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
