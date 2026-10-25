"""EX evaluation for the DE agent.

For every (group, repeat, question) it runs eval/agent_once.py in a subprocess, executes the predicted
and gold SQL against the database, compares results with ``compare_pandas_table`` from
benchmark/scripts/evaluation.py (float tolerance, column matching) and attributes errors to one of:
schema (table/column recall), join, filter (business rule / filter / shape), exec (syntax or execution).

Usage:
  .venv/bin/python eval/run_eval.py --groups A,B,C,D --repeats 2 --workers 3 --run-id ablation_v1
  .venv/bin/python eval/run_eval.py --summarize outputs/experiments/ablation_v1
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import queue
import time
from pathlib import Path

import duckdb
import pandas as pd
import sqlglot
import yaml
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark/scripts"))
from evaluation import compare_pandas_table  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
SQLITE = ROOT / "sample_data/california_schools/california_schools.sqlite"
DUCK = ROOT / "eval/kb/duckdb-demo-eval.duckdb"
BASE_CFG = ROOT / "conf/eval_agent.yml"
SUFFIX = "\n\n(Return one final SQL query whose result contains exactly what the question asks for.)"
DIALECT = {"california_schools": "sqlite", "demo": "duckdb"}


def execute(ds, sql, duck_path=DUCK):
    if ds == "california_schools":
        con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
        finally:
            con.close()
    con = duckdb.connect(str(duck_path), read_only=True)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def _tables(tree):
    return {t.name.lower() for t in tree.find_all(exp.Table)}


def _join_sig(tree):
    sig = set()
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if on is None:
            continue
        cols = tuple(sorted(c.name.lower() for c in on.find_all(exp.Column)))
        normalized = any(True for _ in on.find_all(exp.Func) if not isinstance(_, exp.Column))
        sig.add((cols, normalized))
    return sig


def attribute(ds, pred_sql, gold_sql, exec_error):
    """Rule-based error attribution: exec > schema > join > filter."""
    if exec_error or not pred_sql:
        return "exec"
    d = DIALECT[ds]
    try:
        p, g = sqlglot.parse_one(pred_sql, read=d), sqlglot.parse_one(gold_sql, read=d)
    except Exception:
        return "exec"
    pt, gt = _tables(p), _tables(g)
    if not gt <= pt:
        return "schema"
    if len(gt) >= 2:
        gs, ps = _join_sig(g), _join_sig(p)
        g_norm = any(n for _, n in gs)
        p_norm = any(n for _, n in ps) or ("printf" in pred_sql.lower() or "lpad" in pred_sql.lower())
        if g_norm and not p_norm:
            return "join"
        if {c for c, _ in gs} - {c for c, _ in ps} and not ("where" in pred_sql.lower() and "in (" in pred_sql.lower()):
            return "join"
    gcols = {c.name.lower() for c in g.find_all(exp.Column)}
    pcols = {c.name.lower() for c in p.find_all(exp.Column)}
    if gcols - pcols and len(gcols - pcols) >= max(1, len(gcols) // 2):
        return "schema"
    return "filter"


def score(item, rec, duck_path=DUCK):
    pred = rec.get("final_sql") or rec.get("last_ok_sql")
    out = {"pred_sql": pred, "exec_error": None, "ex": 0}
    if rec.get("error"):
        out["exec_error"] = rec["error"][:300]
    gold_df = execute(item["datasource"], item["gold_sql"], duck_path)
    if pred:
        try:
            pred_df = execute(item["datasource"], pred, duck_path)
            out["ex"] = int(compare_pandas_table(pred_df, gold_df, ignore_order=True))
            out["exec_error"] = None
        except Exception as e:
            out["exec_error"] = f"{type(e).__name__}: {e}"[:300]
    out["error_type"] = "" if out["ex"] else attribute(item["datasource"], pred, item["gold_sql"], out["exec_error"])
    return out


def worker_config(run_dir, wid):
    """Per-worker config: its own DuckDB copy (DuckDB allows one writer per file)."""
    wdir = run_dir / "workers" / f"w{wid}"
    wdir.mkdir(parents=True, exist_ok=True)
    duck = wdir / "demo.duckdb"
    if not duck.exists():
        shutil.copy(DUCK, duck)
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["agent"]["services"]["datasources"]["demo"]["uri"] = f"duckdb:///{duck}"
    path = wdir / "agent.yml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    os.chmod(path, 0o600)
    return path, duck


SLOTS: "queue.Queue" = queue.Queue()


def run_one(job):
    item, group, rep, run_dir = job
    rec_path = run_dir / "records" / f"{group}_r{rep}_{item['id']}.json"
    cfg, duck = SLOTS.get()  # one agent process per worker copy of the DuckDB file
    try:
        _run_agent(item, group, rec_path, cfg)
        rec = json.loads(rec_path.read_text())
        s = score(item, rec, duck)
    finally:
        SLOTS.put((cfg, duck))
    return _row(item, group, rep, rec, s)


def _run_agent(item, group, rec_path, cfg):
    if not rec_path.exists():
        env = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True")
        cmd = [PY, str(ROOT / "eval/agent_once.py"), "--config", str(cfg), "--datasource", item["datasource"],
               "--subagent", f"eval_{group}", "--question", item["question"] + SUFFIX, "--out", str(rec_path)]
        try:
            subprocess.run(cmd, cwd=ROOT, env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            rec_path.write_text(json.dumps({"error": "timeout 600s", "seconds": 600, "trace": []}))
        if not rec_path.exists():
            rec_path.write_text(json.dumps({"error": "runner crashed", "trace": []}))


def _row(item, group, rep, rec, s):
    tools = [t["tool"] for t in rec.get("trace", [])]
    return {
        "group": group, "repeat": rep, "id": item["id"], "tier": item["tier"], "datasource": item["datasource"],
        "ex": s["ex"], "error_type": s["error_type"], "exec_error": s["exec_error"], "pred_sql": s["pred_sql"],
        "tokens": rec.get("tokens", 0), "seconds": rec.get("seconds", 0), "n_tool_calls": len(tools),
        "join_tool_calls": tools.count("suggest_join_path"), "kb_tool_calls": sum(t.startswith(("search_", "get_", "list_subject")) for t in tools),
        "interactions": rec.get("interactions", 0), "tools": "|".join(tools),
    }


def summarize(run_dir):
    df = pd.read_csv(run_dir / "results.csv")
    lines = [f"# Ablation summary: {run_dir.name}", "", f"questions={df['id'].nunique()} rows={len(df)}", ""]
    per_rep = df.groupby(["group", "repeat"])["ex"].mean().unstack() * 100
    lines += ["## EX by group and repeat (%)", "", per_rep.round(1).to_markdown(), ""]
    g = df.groupby("group")
    overall = pd.DataFrame({
        "EX%": g["ex"].mean() * 100,
        "avg_tokens": g["tokens"].mean(),
        "avg_seconds": g["seconds"].mean(),
        "avg_tool_calls": g["n_tool_calls"].mean(),
        "join_tool_used%": g["join_tool_calls"].apply(lambda s: (s > 0).mean() * 100),
    })
    lines += ["## Overall (mean of repeats)", "", overall.round(1).to_markdown(), ""]
    tier = df.pivot_table(index="group", columns="tier", values="ex", aggfunc="mean") * 100
    lines += ["## EX by tier (%)", "", tier.round(1).to_markdown(), ""]
    err = df[df.ex == 0].pivot_table(index="group", columns="error_type", values="id", aggfunc="count", fill_value=0)
    lines += ["## Error attribution (count of wrong answers, all repeats)", "", err.to_markdown(), ""]
    text = "\n".join(lines)
    (run_dir / "summary.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="A,B,C,D")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--run-id", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--dataset", default=str(ROOT / "eval/datasets/base.jsonl"))
    ap.add_argument("--ids", default="", help="comma-separated subset of question ids")
    ap.add_argument("--summarize", default="")
    a = ap.parse_args()
    if a.summarize:
        print(summarize(Path(a.summarize)))
        return

    run_dir = ROOT / "outputs/experiments" / a.run_id
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(a.dataset)]
    if a.ids:
        keep = set(a.ids.split(","))
        items = [i for i in items if i["id"] in keep]
    for w in range(a.workers):
        SLOTS.put(worker_config(run_dir, w))
    jobs = [(item, group, rep, run_dir) for rep in range(1, a.repeats + 1) for group in a.groups.split(",") for item in items]

    rows, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run_one, j): j for j in jobs}
        for f in cf.as_completed(futs):
            rows.append(f.result())
            done += 1
            r = rows[-1]
            print(f"[{done}/{len(jobs)}] {r['group']} r{r['repeat']} {r['id']} ex={r['ex']} {r['error_type']} {r['seconds']}s", flush=True)
            if done % 10 == 0 or done == len(jobs):
                pd.DataFrame(rows).sort_values(["group", "repeat", "id"]).to_csv(run_dir / "results.csv", index=False)
    print(summarize(run_dir))


if __name__ == "__main__":
    main()
