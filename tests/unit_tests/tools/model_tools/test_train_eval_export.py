# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for steps 6-8 of the modeling tools: train_select, evaluate, explain, export (no LLM)."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from dataengineer.tools.model_tools import train as T
from dataengineer.tools.model_tools.export import replay
from dataengineer.tools.model_tools.spec import TaskSpec
from dataengineer.tools.model_tools.tools import ModelingCore

RNG = np.random.RandomState(1)


@pytest.fixture
def clf_df():
    n = 1500
    x1, x2 = RNG.normal(size=n), RNG.normal(size=n)
    seg = RNG.choice(["a", "b", "c"], size=n)
    y = (x1 + 0.5 * x2 + (seg == "a") + RNG.normal(scale=1.0, size=n) > 1.0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "seg": seg, "y": y})


@pytest.fixture
def panel():
    rows = []
    for s in range(4):
        for t, d in enumerate(pd.date_range("2022-01-01", periods=140, freq="D")):
            rows.append({"sid": f"s{s}", "day": d, "y": 40 + 10 * s + 8 * np.sin(2 * np.pi * t / 7) + RNG.normal(scale=2)})
    return pd.DataFrame(rows)


def test_metrics_and_selection_rule():
    y = np.array([0, 0, 1, 1])
    assert T.score("roc_auc", y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert T.score("wape", np.array([10.0, 10.0]), np.array([9.0, 11.0])) == pytest.approx(0.1)
    rows = [{"label": "lgbm", "model": "lgbm", "primary_mean": 0.80, "primary_std": 0.02, "seconds": 3, "params": {}},
            {"label": "linear", "model": "linear", "primary_mean": 0.79, "primary_std": 0.01, "seconds": 1, "params": {}},
            {"label": "prior", "model": "prior", "primary_mean": 0.5, "primary_std": 0.0, "seconds": 0, "params": {}}]
    pick, reason = T.select(rows, "roc_auc")
    assert pick["label"] == "linear" and "within one fold std" in reason  # tie -> more interpretable
    pick, _ = T.select([dict(r, primary_std=0.001) for r in rows], "roc_auc")
    assert pick["label"] == "lgbm"


def test_seasonal_naive_uses_only_history(panel):
    spec = TaskSpec(name="p", task_type="timeseries", target="y", entity="sid", time_col="day", horizon=7, freq="D")
    cut = panel["day"].max() - pd.Timedelta(days=6)
    hist, fut = panel[panel["day"] < cut], panel[panel["day"] >= cut]
    pred = T.seasonal_naive(hist, fut, spec)
    expected = hist.set_index(["sid", "day"])["y"]
    keys = list(zip(fut["sid"], fut["day"] - pd.Timedelta(days=7)))
    assert np.allclose(pred, expected.reindex(keys).to_numpy())


def test_prep_fits_on_train_only(clf_df):
    cols = {"numeric": ["x1"], "categorical": ["seg"], "target_encoded": []}
    p = T.Prep(cols, "linear")
    tr = clf_df.iloc[:1000]
    X = p.fit_transform(tr, tr["y"].to_numpy())
    assert abs(X["x1"].mean()) < 1e-9
    X2 = p.transform(clf_df.iloc[1000:].assign(seg="zzz"))  # unseen level -> all one-hot zeros
    assert (X2.filter(like="seg=") == (0 - p.mu.filter(like="seg=")) / p.sd.filter(like="seg=")).all().all()


def test_end_to_end_classification(tmp_path, clf_df):
    core = ModelingCore(tmp_path, time_budget=4)
    res = core.define_task("r", {"name": "t", "task_type": "binary", "target": "y", "positive_label": "1"},
                           {"frame": clf_df})
    assert res["ok"]
    core.build_features("r", [{"template": "target_encoding", "column": "seg"}])
    core.make_split("r", "stratified", n_folds=3)
    sel = core.train_select("r", candidates=["linear", "lgbm"])
    labels = [r["label"] for r in sel["table"]]
    assert labels[0] == "prior" and "lgbm_tuned" in labels and "te_seg" in sel["target_encoded"]
    ev = core.evaluate("r")
    assert ev["test"]["roc_auc"] > 0.75 and ev["beats_baseline"] and "calibration" in ev
    ex = core.explain("r")
    assert {g["feature"] for g in ex["global_importance"][:2]} & {"x1", "seg", "te_seg"}
    out = core.export_artifacts("r", "synthetic data", "test only")
    assert {"model_card.md", "recipe.json", "train.py", "score.py", "model.joblib"} <= set(out["files"])
    recipe = json.loads((tmp_path / "r" / "export" / "recipe.json").read_text())
    rep = replay(core, {**recipe, "source": {"frame": clf_df}}, "r_replay")
    assert rep["test"]["roc_auc"] == pytest.approx(ev["test"]["roc_auc"])


def test_end_to_end_timeseries_beats_naive_and_replays_on_duckdb(tmp_path, panel):
    db = tmp_path / "p.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE panel AS SELECT * FROM panel")
    q = lambda sql: con.execute(sql).df()  # noqa: E731
    core = ModelingCore(tmp_path / "runs", query=q, time_budget=4)
    spec = {"name": "p", "task_type": "timeseries", "target": "y", "entity": "sid", "time_col": "day",
            "horizon": 7, "freq": "D"}
    assert core.define_task("ts", spec, {"table": "panel"})["ok"]
    core.build_features("ts", [{"template": "lag", "lags": [7, 14]},
                               {"template": "date_parts", "parts": ["dayofweek"]}])
    with pytest.raises(Exception):
        core.make_split("ts", "random")  # guardrail
    core.make_split("ts", "rolling_origin", n_folds=3)
    sel = core.train_select("ts", candidates=["seasonal_naive", "ets", "lgbm"])
    assert sel["table"][0]["model"] == "seasonal_naive"
    ev = core.evaluate("ts")
    assert ev["test"]["wape"] < 0.1 and ev["segments"]
    core.export_artifacts("ts", "-", "-")
    recipe = json.loads((tmp_path / "runs" / "ts" / "export" / "recipe.json").read_text())
    rep = replay(core, recipe, "ts_replay")
    assert rep["test"]["wape"] == pytest.approx(ev["test"]["wape"])
    con.close()


def test_no_guardrail_core_skips_gap_check(tmp_path, clf_df):
    core = ModelingCore(tmp_path, time_budget=2, guardrails=False, enforce_split=False)
    core.define_task("r", {"name": "t", "task_type": "binary", "target": "y", "positive_label": "1"}, {"frame": clf_df})
    core.make_split("r", "random", n_folds=2)
    core.train_select("r", candidates=["linear"])
    assert core.evaluate("r")["findings"] == []


def test_stage_c_tool_fixes(panel):
    from dataengineer.tools.model_tools import leakage as L
    from dataengineer.tools.model_tools.features import build_features

    spec = TaskSpec(name="p", task_type="timeseries", target="y", entity="sid", time_col="day", horizon=7, freq="D")
    df = panel.assign(y_lag7=0.0)  # an input column with the name a lag template would generate
    out, meta, _ = build_features(df, spec, [{"template": "lag", "lags": [7]},
                                             {"template": "arith", "left": "y_lag7", "op": "*", "right": 2},
                                             {"template": "exclude", "columns": ["y_lag7"]}])
    assert "y_lag7_gen" in out.columns and "y_lag7" not in out.columns and out["y_lag7_gen"].notna().any()
    sql = "SELECT c AS entity_id, SUM(v) AS f FROM t WHERE ts >= TIMESTAMP '{cutoff}' - INTERVAL 30 DAY AND ts < TIMESTAMP '{cutoff}' + INTERVAL 5 DAY GROUP BY 1"
    found = L.provenance_checks(TaskSpec(name="s", task_type="regression", target="y", cutoff="2011-01-01"),
                                [{"name": "f", "sql": sql, "time_column": "ts", "cutoff": "{cutoff}"}])
    assert found and found[0]["severity"] == "leak"
