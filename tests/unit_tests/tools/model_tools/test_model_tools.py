# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the gen_model modeling tools (synthetic data, real DuckDB, no LLM)."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from dataengineer.tools.model_tools import leakage as L
from dataengineer.tools.model_tools.clean import CleaningError, apply_rules
from dataengineer.tools.model_tools.features import FeatureError, build_features, oof_target_encode
from dataengineer.tools.model_tools.pit import SemanticCatalog, WindowAgg, snapshot_frame, window_sql
from dataengineer.tools.model_tools.profile import profile_frame
from dataengineer.tools.model_tools.spec import TaskSpec, validate_spec
from dataengineer.tools.model_tools.split import SplitError, make_split
from dataengineer.tools.model_tools.tools import ModelingCore

RNG = np.random.RandomState(0)


@pytest.fixture
def clf_df():
    n = 2000
    x1 = RNG.normal(size=n)
    city = RNG.choice([f"c{i}" for i in range(40)], size=n)
    y = (x1 + RNG.normal(scale=1.5, size=n) > 0.8).astype(int)
    return pd.DataFrame({"id": np.arange(n), "x1": x1, "city": city, "y": y})


@pytest.fixture
def panel():
    """3 daily series with weekly seasonality."""
    rows = []
    for s in range(3):
        temp = 0.0
        for t, d in enumerate(pd.date_range("2022-01-01", periods=200, freq="D")):
            temp = 0.8 * temp + RNG.normal()  # persistent exogenous driver, legitimately known at t
            rows.append({"sid": f"s{s}", "day": d, "temp": temp,
                         "y": 50 + 10 * s + 8 * np.sin(2 * np.pi * t / 7) + 3 * temp + RNG.normal(scale=3)})
    return pd.DataFrame(rows)


def ts_spec(h=1):
    return TaskSpec(name="p", task_type="timeseries", target="y", entity="sid", time_col="day", horizon=h, freq="D")


# ---- spec ----------------------------------------------------------------------------------
def test_spec_defaults_and_validation(clf_df):
    spec = TaskSpec(name="t", task_type="binary", target="y", drop_columns=["id"])
    assert spec.primary_metric == "roc_auc" and spec.baseline == "prior"
    res = validate_spec(spec, clf_df)
    assert res["errors"] == []
    assert spec.positive_label == "1"  # minority class inferred
    with pytest.raises(ValueError):
        TaskSpec(name="t", task_type="binary", target="y", primary_metric="rmse")
    with pytest.raises(ValueError):
        TaskSpec(name="t", task_type="timeseries", target="y")  # needs time_col/horizon/freq
    assert ts_spec().season_length == 7
    bad = validate_spec(TaskSpec(name="t", task_type="regression", target="city"), clf_df)
    assert any("numeric" in e for e in bad["errors"])


def test_profile_flags_issues(clf_df):
    df = clf_df.assign(const=1, txt=np.where(RNG.rand(len(clf_df)) < 0.1, "?", "a"))
    rep = profile_frame(df, TaskSpec(name="t", task_type="binary", target="y"))
    kinds = {(i["kind"], i["column"]) for i in rep["issues"]}
    assert ("constant", "const") in kinds and ("id_like", "id") in kinds and ("placeholder_missing", "txt") in kinds


# ---- clean ---------------------------------------------------------------------------------
def test_clean_rules_and_protection():
    df = pd.DataFrame({"q": [1, -2, 3, 3], "c": [" a", "b ", "?", "?"], "y": [1, 0, 1, 1]})
    out, log = apply_rules(df, [
        {"op": "drop_duplicates"},
        {"op": "drop_rows", "column": "q", "cmp": "<=", "value": 0, "reason": "returns"},
        {"op": "replace_values", "column": "c", "mapping": {"?": None}},
        {"op": "strip_strings", "columns": ["c"]},
    ])
    assert len(out) == 2 and out["c"].tolist()[0] == "a" and out["c"].isna().sum() == 1
    assert [e["rows_removed"] for e in log[:2]] == [1, 1]
    with pytest.raises(CleaningError):
        apply_rules(df, [{"op": "drop_columns", "columns": ["y"]}], protected=["y"])
    with pytest.raises(CleaningError):
        apply_rules(df, [{"op": "impute_median", "column": "q"}])


# ---- features ------------------------------------------------------------------------------
def test_lags_are_forced_past_horizon(panel):
    spec = ts_spec(h=7)
    out, meta, warns = build_features(panel, spec, [
        {"template": "lag", "lags": [1, 14]},
        {"template": "rolling", "windows": [7], "stats": ["mean"]},
        {"template": "date_parts", "parts": ["dayofweek"]},
    ])
    names = {m["name"] for m in meta}
    assert {"y_lag7", "y_lag14", "y_roll7_mean_s7", "day_dayofweek"} <= names
    assert any("raised to 7" in w for w in warns)
    s0 = out[out.sid == "s0"].reset_index(drop=True)
    assert s0.loc[20, "y_lag7"] == pytest.approx(s0.loc[13, "y"])
    assert s0.loc[20, "y_roll7_mean_s7"] == pytest.approx(s0.loc[7:13, "y"].mean())
    assert out["sid"].tolist() == panel["sid"].tolist()  # input row order kept
    assert all(L.check_window_feature(m["name"], {**m, "window": m.get("window", 1)}, 7) is None
               for m in meta if m["template"] in ("lag", "rolling"))


def test_feature_guards(clf_df):
    spec = TaskSpec(name="t", task_type="binary", target="y")
    with pytest.raises(FeatureError):
        build_features(clf_df, spec, [{"template": "arith", "left": "y", "right": "x1"}])
    _, meta, _ = build_features(clf_df, spec, [{"template": "target_encoding", "column": "city"}])
    assert meta[0]["deferred"] is True


def test_oof_target_encoding_not_constant_within_group(clf_df):
    enc, test_enc = oof_target_encode(clf_df["city"], clf_df["y"], clf_df["city"].head(5))
    assert clf_df.assign(e=enc).groupby("city")["e"].nunique().max() > 1
    assert len(test_enc) == 5


# ---- split ---------------------------------------------------------------------------------
def test_split_guardrail_and_methods(panel, clf_df):
    spec = ts_spec(h=7)
    with pytest.raises(SplitError):
        make_split(panel, spec, "random")
    split, info = make_split(panel, spec, "rolling_origin", n_folds=3)
    t = panel["day"]
    assert t[split.role == "test"].min() > t[split.role == "train"].max()
    for k in range(3):
        assert t[split[f"cv{k}"] == "train"].max() < t[split[f"cv{k}"] == "valid"].min()
        assert panel[split[f"cv{k}"] == "valid"]["day"].nunique() == 7
    # repeated entity -> group split, no overlap
    rep = clf_df.assign(cust=clf_df["id"] // 4)
    gspec = TaskSpec(name="t", task_type="binary", target="y", entity="cust")
    with pytest.raises(SplitError):
        make_split(rep, gspec, "stratified")
    gsplit, _ = make_split(rep, gspec, "group")
    assert not set(rep.cust[gsplit.role == "test"]) & set(rep.cust[gsplit.role == "train"])
    # guardrail off: allowed, and leakage check sees the overlap
    rsplit, _ = make_split(rep, gspec, "stratified", enforce=False)
    found = L.split_overlap_check(rep, gspec, rsplit, ["x1"])
    assert found and found[0]["severity"] == "leak"


# ---- leakage: provenance -------------------------------------------------------------------
@pytest.mark.parametrize("where,leak", [
    ("t.ts >= TIMESTAMP '2011-01-01' - INTERVAL 90 DAY AND t.ts < TIMESTAMP '2011-01-01'", False),
    ("t.ts < TIMESTAMP '2011-01-01' + INTERVAL 30 DAY", True),
    ("t.ts <= DATE '2011-01-01'", True),
    ("t.ts BETWEEN '2010-06-01' AND '2010-12-31'", False),
    ("t.amount > 0", True),
])
def test_sql_time_filter(where, leak):
    sql = f"SELECT t.cid, SUM(t.amount) FROM t WHERE {where} GROUP BY 1"
    f = L.check_sql_feature("f", sql, ["ts"], "2011-01-01")
    assert (f is not None and f["severity"] == "leak") == leak


def test_window_offsets():
    assert L.window_offsets({"shift": 1, "window": 7}) == (-7, -1)
    assert L.check_window_feature("ok", {"shift": 1, "window": 7}, gap=1) is None
    assert L.check_window_feature("c", {"shift": 0, "window": 7, "center": True}, gap=1)["severity"] == "leak"
    assert L.check_window_feature("short", {"shift": 1, "window": 1}, gap=7)["severity"] == "leak"


# ---- leakage: empirical --------------------------------------------------------------------
def test_too_good_and_identity():
    n = 3000
    hr = RNG.randint(0, 24, n)
    casual = RNG.poisson(20 + 10 * (hr > 9), n)
    registered = RNG.poisson(100 + 50 * np.sin(hr / 24 * np.pi), n)
    df = pd.DataFrame({"hr": hr, "temp": RNG.rand(n), "casual": casual, "registered": registered})
    df["cnt"] = df.casual + df.registered
    df["cnt_noisy"] = df.cnt * (1 + RNG.normal(scale=0.03, size=n))
    spec = TaskSpec(name="b", task_type="regression", target="cnt")
    tg = {x["features"][0] for x in L.too_good_check(df, spec, ["hr", "temp", "cnt_noisy"])}
    assert tg == {"cnt_noisy"}
    ident = L.identity_check(df, spec, ["hr", "temp", "casual", "registered"])
    assert ident and set(ident[0]["features"]) == {"casual", "registered"}
    assert L.identity_check(df, spec, ["hr", "temp"]) == []


def test_naive_target_encoding_detected(clf_df):
    spec = TaskSpec(name="t", task_type="binary", target="y", positive_label="1")
    df = clf_df.assign(te_naive=clf_df.groupby("city")["y"].transform("mean"),
                       te_oof=oof_target_encode(clf_df["city"], clf_df["y"])[0])
    found = L.naive_te_check(df, spec, ["x1", "te_naive", "te_oof"], ["city"])
    assert [f["features"][0] for f in found] == ["te_naive"]


def test_future_alignment(panel):
    spec = ts_spec(h=1)
    g = panel.sort_values(["sid", "day"]).groupby("sid")["y"]
    df = panel.sort_values(["sid", "day"]).assign(
        lag1=g.shift(1),
        fut_mean3=g.transform(lambda s: s[::-1].rolling(3).mean()[::-1].shift(-1)),
    )
    found = {f["features"][0] for f in L.future_alignment_check(df, spec, ["lag1", "temp", "fut_mean3"])}
    assert found == {"fut_mean3"}


def test_run_checks_skips_proven_features(panel):
    spec = ts_spec(h=1)
    out, meta, _ = build_features(panel, spec, [{"template": "lag", "lags": [1, 7]}])
    rep = L.run_checks(out, spec, ["y_lag1", "y_lag7"], definitions=meta)
    assert rep["leaky_features"] == [] and set(rep["proven_point_in_time"]) == {"y_lag1", "y_lag7"}
    bad = [{"name": "y_c", "template": "rolling", "shift": 0, "window": 3, "center": True}]
    rep2 = L.run_checks(out.assign(y_c=out["y"]), spec, ["y_lag1", "y_c"], definitions=meta + bad)
    assert rep2["leaky_features"] == ["y_c"]


def test_metric_gap():
    assert L.metric_gap_check(0.95, 0.80, "roc_auc", True)
    assert not L.metric_gap_check(0.85, 0.83, "roc_auc", True)


# ---- point in time over a semantic layer -----------------------------------------------------
@pytest.fixture
def shop(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE invoices (invoice VARCHAR, customer_id VARCHAR, invoice_date TIMESTAMP)")
    con.execute("CREATE TABLE invoice_lines (invoice VARCHAR, quantity INT, price DOUBLE)")
    con.execute("""INSERT INTO invoices VALUES ('i1','a','2010-10-01'),('i2','a','2010-12-20'),('i3','b','2011-01-10'),
                   ('i4','a','2011-01-15'),('i5','c','2009-01-01')""")
    con.execute("INSERT INTO invoice_lines VALUES ('i1',2,5.0),('i2',1,10.0),('i3',3,1.0),('i4',1,100.0),('i5',1,1.0)")
    sem = tmp_path / "sem"
    sem.mkdir()
    (sem / "invoices.yml").write_text("""data_source:
  name: invoices
  sql_table: invoices
  identifiers:
    - {name: invoice, type: PRIMARY, expr: invoice}
    - {name: customer, type: FOREIGN, expr: customer_id}
  dimensions:
    - {name: invoice_date, type: TIME, expr: invoice_date}
  measures:
    - {name: invoice_count, agg: COUNT_DISTINCT, expr: invoice}
""")
    (sem / "invoice_lines.yml").write_text("""data_source:
  name: invoice_lines
  sql_table: invoice_lines
  identifiers:
    - {name: invoice, type: FOREIGN, expr: invoice}
  measures:
    - {name: revenue, agg: SUM, expr: quantity * price}
""")
    cols = {"invoices": ["invoice", "customer_id", "invoice_date"], "invoice_lines": ["invoice", "quantity", "price"]}

    def solver(tables):
        return {"from_clause": 'FROM invoice_lines\nJOIN invoices ON invoice_lines."invoice" = invoices."invoice"',
                "joins": [], "ambiguous": False}

    return con, SemanticCatalog(sem, cols), solver


def test_window_sql_and_snapshot(shop):
    con, cat, solver = shop
    q = lambda sql: con.execute(sql).df()
    feat = window_sql(WindowAgg("rev_90d", "revenue", 90, "customer", "invoice_date"), cat, solver,
                      "TIMESTAMP '2011-01-01'", "feature")
    assert 'invoice_lines."quantity" * invoice_lines."price"' in feat["sql"]
    assert L.check_sql_feature("rev_90d", feat["sql"], ["invoice_date"], "2011-01-01") is None
    vals = dict(q(feat["sql"]).itertuples(index=False))
    assert vals == {"a": 10.0}  # only i2; i1 (2010-10-01) is 92 days before the cutoff
    built = snapshot_frame(["2011-01-01"], WindowAgg("y", "revenue", 30, "customer", "invoice_date"),
                           WindowAgg("_a", "invoice_count", 365, "customer", "invoice_date"), cat, solver, q)
    f = built["frame"].set_index("entity_id")["y"].to_dict()
    assert f == {"a": 100.0}  # b has no history before the cutoff, c is outside the 365-day population window
    leak = window_sql(WindowAgg("x", "revenue", 30, "customer", "invoice_date"), cat, solver,
                      "TIMESTAMP '2011-01-01'", "label")
    assert L.check_sql_feature("x", leak["sql"], ["invoice_date"], "2011-01-01")["severity"] == "leak"


def test_core_end_to_end(tmp_path, clf_df):
    core = ModelingCore(tmp_path)
    r = core.define_task("run1", {"name": "t", "task_type": "binary", "target": "y", "drop_columns": ["id"]},
                         {"frame": clf_df.assign(leak=clf_df["y"] + RNG.normal(scale=0.05, size=len(clf_df)))})
    assert r["ok"]
    assert core.profile_data("run1")["n_rows"] == len(clf_df)
    assert core.clean_data("run1", [{"op": "drop_duplicates"}])["rows_after"] == len(clf_df)
    core.build_features("run1", [{"template": "target_encoding", "column": "city"}])
    sp = core.make_split("run1", "stratified")
    assert sp["counts"]["test"] == pytest.approx(0.2 * len(clf_df), abs=2)
    rep = core.leakage_check("run1")
    assert rep["leaky_features"] == ["leak"]
    steps = [d["step"] for d in core.ws("run1").decisions()]
    assert steps == ["define_task", "profile_data", "clean_data", "build_features", "make_split", "leakage_check"]


def test_sql_feature_catalog_is_checked_not_trusted(shop):
    con, cat, solver = shop
    q = lambda sql: con.execute(sql).df()
    spec = TaskSpec(name="r", task_type="regression", target="y", entity="cid", cutoffs=["2011-01-01", "2011-02-01"])
    frame = pd.DataFrame({"cid": ["a", "a", "b"], "snapshot": ["2011-01-01", "2011-02-01", "2011-02-01"], "y": [1.0, 2.0, 3.0]})
    good = ("SELECT customer_id AS entity_id, COUNT(*) AS n_before FROM invoices "
            "WHERE invoice_date < TIMESTAMP '{cutoff}' GROUP BY 1")
    bad = "SELECT customer_id AS entity_id, COUNT(*) AS n_all FROM invoices GROUP BY 1"
    out, meta, _ = build_features(frame, spec, [{"template": "sql_feature", "name": "n_before", "sql": good},
                                                {"template": "sql_feature", "name": "n_all", "sql": bad}],
                                  query=q, catalog=cat)
    assert out["n_before"].tolist() == [2, 3, 1]  # a: i1,i2 before 2011-01-01, +i4 before 2011-02-01; b: i3
    assert out["n_all"].tolist() == [3, 3, 1]
    assert meta[0]["time_column"] == ["invoice_date"] and meta[0]["cutoff"] == "2011-01-01"
    found = L.provenance_checks(spec, meta)
    assert [f["features"] for f in found] == [["n_all"]]
