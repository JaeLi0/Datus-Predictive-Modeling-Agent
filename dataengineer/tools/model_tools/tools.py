# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Function-tool wrapper for the 8 modeling steps plus leakage_check.

Each tool works on a run workspace (run_id) and returns a compact summary for the LLM; full tables stay on
disk. The LLM supplies decisions as JSON (spec, rules, templates, split method); the code does the work.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import json

import numpy as np
import pandas as pd
from agents import Tool

from dataengineer.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from dataengineer.tools.model_tools import clean as clean_mod
from dataengineer.tools.model_tools import evaluate as eval_mod
from dataengineer.tools.model_tools import export as export_mod
from dataengineer.tools.model_tools import features as feat_mod
from dataengineer.tools.model_tools import leakage as leak_mod
from dataengineer.tools.model_tools import split as split_mod
from dataengineer.tools.model_tools import train as train_mod
from dataengineer.tools.model_tools.pit import SemanticCatalog, WindowAgg, snapshot_frame
from dataengineer.tools.model_tools.profile import profile_frame
from dataengineer.tools.model_tools.spec import TaskSpec, validate_spec
from dataengineer.tools.model_tools.workspace import ROW_ID, RunWorkspace
from dataengineer.utils.loggings import get_logger

logger = get_logger(__name__)


class ModelingCore:
    """Database-agnostic implementation; the agent wrapper and the benchmark both use it."""

    def __init__(self, runs_dir: Path, query: Optional[Callable[[str], pd.DataFrame]] = None,
                 semantic_dir: Optional[Path] = None, join_solver: Optional[Callable[[List[str]], Dict]] = None,
                 enforce_split: bool = True, guardrails: bool = True, time_budget: float = 60,
                 db_label: str = ""):
        self.runs_dir = Path(runs_dir)
        self.guardrails = guardrails
        self.time_budget = time_budget
        self.db_label = db_label
        self.query = query
        self.semantic_dir = Path(semantic_dir) if semantic_dir else None
        self.join_solver = join_solver
        self.enforce_split = enforce_split
        self._catalog: Optional[SemanticCatalog] = None

    def ws(self, run_id: str) -> RunWorkspace:
        return RunWorkspace(self.runs_dir / run_id)

    def catalog(self) -> SemanticCatalog:
        if self._catalog is None:
            if not self.semantic_dir:
                raise ValueError("no semantic model directory configured for point-in-time features")
            cols = {}
            if self.query is not None:
                try:
                    info = self.query("SELECT table_name, column_name FROM information_schema.columns")
                    for t, c in info.itertuples(index=False):
                        cols.setdefault(str(t), []).append(str(c))
                except Exception as e:
                    logger.debug(f"column listing failed: {e}")
            self._catalog = SemanticCatalog(self.semantic_dir, cols)
        return self._catalog

    def _read_table(self, name: str) -> pd.DataFrame:
        """Read a table given as name, schema.name or catalog.schema.name (quoted or not); qualifiers that do not
        resolve in this connection are dropped from the left."""
        parts = [p.strip().strip('"`') for p in str(name).split(".") if p.strip()]
        last_err = None
        for i in range(len(parts)):
            ref = ".".join(f'"{p}"' for p in parts[i:])
            try:
                return self.query(f"SELECT * FROM {ref}")
            except Exception as e:
                last_err = e
        raise ValueError(f"table {name!r} not found: {last_err}")

    # ---- step 1 -------------------------------------------------------------------------
    def define_task(self, run_id: str, spec: Dict, source: Dict) -> Dict:
        """source: {"table": name} | {"sql": query} | {"label": {...}, "population": {...}} (snapshots)."""
        ws = self.ws(run_id)
        task = TaskSpec(**spec)
        if "label" in source:
            cutoffs = task.cutoffs or ([task.cutoff] if task.cutoff else [])
            if not cutoffs:
                raise ValueError("snapshot task needs spec.cutoff or spec.cutoffs")
            lab = WindowAgg(name=task.target, **source["label"])
            pop = WindowAgg(name="_activity", **source["population"])
            built = snapshot_frame(cutoffs, lab, pop, self.catalog(), self.join_solver, self.query)
            df = built["frame"].rename(columns={"entity_id": task.entity or "entity_id"})
            task.entity = task.entity or "entity_id"
            ws.save_json("label_sql", built["sql"])
            task.horizon = task.horizon or lab.window_days
        elif "table" in source or "sql" in source:
            if self.query is None:
                raise ValueError("no database connection")
            df = self.query(source["sql"]) if source.get("sql") else self._read_table(source["table"])
        elif "frame" in source:  # in-process use (tests, benchmark)
            df = source["frame"]
        else:
            raise ValueError("source needs table, sql, or label+population")
        if ROW_ID not in df.columns:
            if "snapshot" in df.columns:
                df[ROW_ID] = df["snapshot"].astype(str) + "|" + df[task.entity].astype(str)
            else:
                df[ROW_ID] = np.arange(len(df))
        check = validate_spec(task, df)
        ws.save_spec(task)
        if not check["errors"]:
            ws.save_frame("raw", df)
        ws.log("define_task", decision={"spec": spec, "source": {k: v for k, v in source.items() if k != "frame"}},
               result=check)
        return {"spec": task.model_dump(), **check, "ok": not check["errors"]}

    # ---- step 2 -------------------------------------------------------------------------
    def profile_data(self, run_id: str, table: str = "") -> Dict:
        ws = self.ws(run_id)
        name = table or ws.latest_table()
        rep = profile_frame(ws.load_frame(name).drop(columns=[ROW_ID], errors="ignore"), ws.load_spec())
        ws.save_json(f"profile_{name}", rep)
        ws.log("profile_data", result={"table": name, "issues": len(rep["issues"])})
        return {"table": name, **rep}

    # ---- step 3 -------------------------------------------------------------------------
    def clean_data(self, run_id: str, rules: List[Dict]) -> Dict:
        ws = self.ws(run_id)
        spec = ws.load_spec()
        df = ws.load_frame("raw")
        protected = [c for c in (spec.target, spec.entity, spec.time_col, ROW_ID) if c]
        out, log = clean_mod.apply_rules(df, rules, protected=protected)
        if spec.target in out.columns and out[spec.target].isna().any():
            n = int(out[spec.target].isna().sum())
            out = out[out[spec.target].notna()].reset_index(drop=True)
            log.append({"op": "drop_rows", "column": spec.target, "rows_removed": n, "reason": "missing target"})
        ws.save_frame("clean", out)
        ws.save_json("clean_log", log)
        ws.log("clean_data", decision={"rules": rules}, result={"rows": [len(df), len(out)]})
        return {"rows_before": len(df), "rows_after": len(out), "log": log, "columns": list(out.columns)}

    # ---- step 4 -------------------------------------------------------------------------
    def build_features(self, run_id: str, templates: List[Dict]) -> Dict:
        ws = self.ws(run_id)
        spec = ws.load_spec()
        df = ws.load_frame("clean" if ws.has("clean") else "raw")
        needs_pit = any(t.get("template") in ("pit_window", "sql_feature") for t in templates)
        out, meta, warnings = feat_mod.build_features(
            df, spec, templates, query=self.query,
            catalog=self.catalog() if needs_pit and self.semantic_dir else None,
            join_solver=self.join_solver)
        ws.save_frame("features", out)
        ws.save_json("feature_meta", meta)
        ws.log("build_features", decision={"templates": templates}, result={"n_features": len(meta), "warnings": warnings})
        generated = [m["name"] for m in meta if m.get("name") and not m.get("deferred")]
        return {"rows": len(out), "columns": list(out.columns), "generated": generated,
                "deferred": [m["name"] for m in meta if m.get("deferred")], "warnings": warnings,
                "null_rates": {c: round(float(out[c].isna().mean()), 4) for c in generated if c in out.columns}}

    # ---- step 5 -------------------------------------------------------------------------
    def make_split(self, run_id: str, method: str, test_size: float = 0.2, n_folds: int = 5, seed: int = 0) -> Dict:
        ws = self.ws(run_id)
        spec = ws.load_spec()
        df = ws.load_frame(ws.latest_table())
        allowed, why = split_mod.required_methods(spec, df)
        split, info = split_mod.make_split(df, spec, method, test_size=test_size, n_folds=n_folds, seed=seed,
                                           enforce=self.enforce_split)
        ws.save_frame("split", split)
        ws.save_json("split", info)
        ws.log("make_split", decision={"method": method, "test_size": test_size, "n_folds": n_folds},
               result=info)
        return {**info, "allowed_methods": allowed}

    # ---- leakage ------------------------------------------------------------------------
    def feature_columns(self, ws: RunWorkspace, df: pd.DataFrame, spec: TaskSpec) -> List[str]:
        skip = {spec.target, spec.entity, spec.time_col, "snapshot", ROW_ID, *spec.drop_columns}
        return [c for c in df.columns if c not in skip]

    def leakage_check(self, run_id: str, definitions: Optional[List[Dict]] = None, features: Optional[List[str]] = None,
                      checks: Optional[List[str]] = None) -> Dict:
        ws = self.ws(run_id)
        spec = ws.load_spec()
        df = ws.load_frame(ws.latest_table()).drop(columns=[ROW_ID], errors="ignore")
        feats = features or self.feature_columns(ws, df, spec)
        defs = list(ws.load_json("feature_meta", default=[])) + list(definitions or [])
        split = ws.load_frame("split") if (ws.root / "split.parquet").exists() else None
        cutoffs = sorted(df["snapshot"].astype(str).unique()) if "snapshot" in df.columns else None
        rep = leak_mod.run_checks(df, spec, feats, definitions=defs, split=split, cutoffs=cutoffs, checks=checks)
        ws.save_json("leakage", rep)
        ws.log("leakage_check", result={"leaky": rep["leaky_features"], "suspect": rep["suspect_features"]})
        return rep

    # ---- step 6 -------------------------------------------------------------------------
    def train_select(self, run_id: str, candidates: Optional[List[str]] = None, exclude: Optional[List[str]] = None,
                     time_budget: Optional[float] = None) -> Dict:
        ws = self.ws(run_id)
        spec = ws.load_spec()
        df = ws.load_frame(ws.latest_table())
        if not ws.has("split"):
            raise ValueError("no split yet; call make_split first")
        missing = [c for c in (exclude or []) if c not in df.columns]
        exclude = [c for c in (exclude or []) if c in df.columns]
        budget = float(time_budget or self.time_budget)
        sel = train_mod.train_select(df, spec, ws.load_json("feature_meta", default=[]), ws.load_frame("split"),
                                     candidates=candidates, exclude=exclude, time_budget=budget)
        if missing:
            sel["note"] = f"exclude columns already absent from the table (ignored): {missing}"
        ws.save_json("selection", sel)
        ws.log("train_select", decision={"candidates": candidates, "exclude": exclude, "time_budget": budget},
               result={"chosen": sel["chosen"], "reason": sel["reason"]})
        return sel

    # ---- step 7 -------------------------------------------------------------------------
    def evaluate(self, run_id: str, label: str = "", model: Optional[Dict] = None,
                 excluded: Optional[List[str]] = None) -> Dict:
        """label: a candidate from the selection table (default: the chosen one). model: an explicit
        {"model", "params"} (replay from a recipe)."""
        import joblib

        ws = self.ws(run_id)
        spec = ws.load_spec()
        df = ws.load_frame(ws.latest_table())
        if ws.has("selection"):
            sel = ws.load_json("selection")
        elif model:
            sel = {"chosen": model, "table": [], "excluded": excluded or []}
        else:
            raise ValueError("no model selection yet; call train_select first")
        if model is None:
            if label:
                row = next((r for r in sel["table"] if r["label"] == label), None)
                if row is None:
                    raise ValueError(f"unknown candidate {label!r}; choose from {[r['label'] for r in sel['table']]}")
                model = {"label": row["label"], "model": row["model"], "params": row.get("params", {})}
            else:
                model = sel["chosen"]
        rep, fitted, preds = eval_mod.evaluate(df, spec, ws.load_json("feature_meta", default=[]),
                                               ws.load_frame("split"), sel, ws.root, model=model,
                                               guardrails=self.guardrails)
        joblib.dump(fitted, ws.root / "model.joblib")
        ws.save_frame("test_predictions", preds)
        ws.save_json("evaluation", rep)
        ws.log("evaluate", decision={"model": rep["model"]}, result={"test": rep["test"], "findings": rep["findings"]})
        return rep

    # ---- step 8 -------------------------------------------------------------------------
    def explain(self, run_id: str) -> Dict:
        import joblib

        ws = self.ws(run_id)
        if not (ws.root / "model.joblib").exists():
            raise ValueError("no fitted model; call evaluate first")
        fitted = joblib.load(ws.root / "model.joblib")
        df = ws.load_frame(ws.latest_table())
        test = (ws.load_frame("split")["role"] == "test").to_numpy()
        rep = eval_mod.explain(fitted, df, test, ws.root / "plots")
        ws.save_json("explanation", rep)
        ws.log("explain", result={"top": [g["feature"] for g in rep.get("global_importance", [])[:5]]})
        return rep

    def export_artifacts(self, run_id: str, limitations: str, usage: str) -> Dict:
        ws = self.ws(run_id)
        spec = ws.load_spec()
        if not ws.has("evaluation"):
            raise ValueError("no evaluation yet; call evaluate first")
        sel = ws.load_json("selection")
        evaluation = ws.load_json("evaluation")
        recipe = export_mod.build_recipe(ws.decisions(), sel, evaluation["model"])
        out = ws.root / "export"
        out.mkdir(exist_ok=True)
        (out / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2, default=str))
        import shutil

        shutil.copy(ws.root / "model.joblib", out / "model.joblib")
        leak = ws.load_json("leakage") if ws.has("leakage") else None
        expl = ws.load_json("explanation") if ws.has("explanation") else None
        card = export_mod.model_card(spec, recipe, sel, evaluation, leak, expl, limitations, usage)
        (out / "model_card.md").write_text(card)
        shutil.copy(ws.root / "decisions.jsonl", out / "decisions.jsonl")
        repo = Path(__file__).resolve().parents[3]
        scripts = export_mod.write_scripts(out, repo, self.db_label, str(self.semantic_dir) if self.semantic_dir else None)
        ws.log("export_artifacts", decision={"limitations": limitations, "usage": usage})
        return {"dir": str(out), "files": sorted(p.name for p in out.iterdir()), "scripts": scripts,
                "model_card_preview": card[:1200]}


class ModelTools:
    """Agent-facing function tools. Runs live under <project data dir>/model_runs/<run_id>."""

    def __init__(self, agent_config, db_func_tool=None, enforce_split: bool = True, guardrails: bool = True,
                 time_budget: float = 60, runs_dir: Optional[Path] = None):
        self.agent_config = agent_config
        self.db_func_tool = db_func_tool
        self.enforce_split = enforce_split
        self.guardrails = guardrails
        self.time_budget = time_budget
        self.runs_dir = runs_dir
        self._cores: Dict[str, ModelingCore] = {}

    def available_tools(self) -> List[Tool]:
        fns = [self.define_task, self.profile_data, self.clean_data, self.build_features, self.make_split]
        if self.guardrails:
            fns.append(self.leakage_check)
        fns += [self.train_select, self.evaluate, self.explain, self.export_artifacts]
        return [trans_to_function_tool(f) for f in fns]

    def _core(self, datasource: str = "") -> ModelingCore:
        if datasource in self._cores:
            return self._cores[datasource]
        from dataengineer.tools.func_tool.join_path_tools import JoinPathTools, solve

        if self.db_func_tool is None:
            from dataengineer.tools.func_tool.database import DBFuncTool

            self.db_func_tool = DBFuncTool(agent_config=self.agent_config)
        connector = self.db_func_tool._get_connector(datasource or None)

        def query(sql: str) -> pd.DataFrame:
            res = connector.execute_pandas(sql)
            if not res.success:
                raise RuntimeError(res.error)
            return res.sql_return

        jp = JoinPathTools(self.agent_config, db_func_tool=self.db_func_tool)
        ds_name = datasource or getattr(self.agent_config, "current_datasource", "") or ""
        try:
            semantic_dir = self.agent_config.path_manager.semantic_model_path(ds_name)
        except Exception:
            semantic_dir = None
        runs_dir = self.runs_dir or Path(self.agent_config.path_manager.project_data_dir) / "model_runs"
        db_label = ""
        try:
            db_label = str(self.agent_config.current_db_config(ds_name).uri or "").replace("duckdb:///", "")
        except Exception:
            pass
        core = ModelingCore(runs_dir, query=query, semantic_dir=semantic_dir,
                            join_solver=lambda tables: solve(jp._graph(datasource), tables),
                            enforce_split=self.enforce_split, guardrails=self.guardrails,
                            time_budget=self.time_budget, db_label=db_label)
        self._cores[datasource] = core
        return core

    @staticmethod
    def _json(value, name: str):
        """Structured arguments arrive as JSON text (the strict tool schema has no free-form objects)."""
        if value is None or isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value) if str(value).strip() else None
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} is not valid JSON: {e}") from e

    @staticmethod
    def _wrap(fn, *args, **kwargs) -> FuncToolResult:
        try:
            return FuncToolResult(result=fn(*args, **kwargs))
        except Exception as e:
            logger.warning(f"{getattr(fn, '__name__', 'model tool')} failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    def define_task(self, run_id: str, spec: str, source: str, datasource: str = "") -> FuncToolResult:
        """
        Step 1. Define the modeling task and load its table into a new run workspace.

        Args:
            run_id: Name for this modeling run, e.g. "adult_v1". Reuse it in every later step.
            spec: TaskSpec as a JSON object string: name, task_type (binary|multiclass|regression|timeseries), target, entity,
                time_col, cutoff or cutoffs (ISO dates, point-in-time tasks), horizon, freq (H|D|W),
                positive_label, primary_metric, baseline, drop_columns (ids, free text, post-outcome columns).
            source: JSON object string. {"table": "<table>"} or {"sql": "<select>"} for a ready modeling table; for a
                point-in-time task built from event tables: {"label": {"measure", "window_days", "entity",
                "time_dim", "filters"}, "population": {same keys; entities active in this window before the
                cutoff}} using semantic-model names.
            datasource: Datasource name; empty means the current one.

        Returns:
            dict with the validated spec, errors (blocking), warnings and a target summary.
        """
        try:
            spec_d, source_d = self._json(spec, "spec"), self._json(source, "source")
        except ValueError as e:
            return FuncToolResult(success=0, error=str(e))
        return self._wrap(self._core(datasource).define_task, run_id, spec_d, source_d)

    def profile_data(self, run_id: str, table: str = "", datasource: str = "") -> FuncToolResult:
        """
        Step 2. Profile the run's current table: per-column missing rate, type, cardinality, outliers,
        duplicates, target distribution, time coverage, and a list of issues (each with a kind).

        Args:
            run_id: The run created by define_task.
            table: raw | clean | features; empty means the most processed one.
            datasource: Datasource name; empty means the current one.
        """
        return self._wrap(self._core(datasource).profile_data, run_id, table)

    def clean_data(self, run_id: str, rules: str, datasource: str = "") -> FuncToolResult:
        """
        Step 3. Apply cleaning rules to the raw table, in order, and log rows/cells affected by each.

        Each rule is {"op": ..., "reason": "..."} with op one of:
        drop_duplicates {subset?}; drop_columns {columns}; drop_rows {column, cmp (== != < <= > >= in not_in
        isnull notnull startswith contains), value}; replace_values {column, mapping}; strip_strings {columns?};
        cast {column, dtype numeric|datetime|string|category}; fill_missing {column, value};
        add_missing_indicator {column}; clip {column, lower?, upper?}; derive {name, expr (pandas eval)}.
        Statistical imputation is done later inside training, not here.

        Args:
            run_id: The run created by define_task.
            rules: The ordered rule list as a JSON array string.
            datasource: Datasource name; empty means the current one.
        """
        try:
            rules_l = self._json(rules, "rules") or []
        except ValueError as e:
            return FuncToolResult(success=0, error=str(e))
        return self._wrap(self._core(datasource).clean_data, run_id, rules_l)

    def build_features(self, run_id: str, templates: str, datasource: str = "") -> FuncToolResult:
        """
        Step 4. Build features from templates. Point-in-time safety is enforced by the code.

        Templates: {"template": "date_parts", "column", "parts": [year, month, day, dayofweek, dayofyear,
        weekofyear, hour, is_weekend, quarter]}; {"template": "log1p", "column"}; {"template": "arith", "left",
        "op" (+ - * /), "right", "name"}; {"template": "lag", "column" (default target), "lags": [..]};
        {"template": "rolling", "column", "windows": [..], "stats": [mean, std, min, max, sum]} (lags/rolling
        are shifted by at least the horizon); {"template": "pit_window", "measure", "windows": [days],
        "entity_ref", "time_dim", "filters", "name"} (semantic measure aggregated before each cutoff);
        {"template": "sql_feature", "name", "sql", "time_column"} (feature-catalog SQL returning entity_id and
        <name>, with a {cutoff} placeholder, run per snapshot; its time filter is verified by leakage_check);
        {"template": "target_encoding", "column"} (fitted out-of-fold during training);
        {"template": "exclude", "columns"}.

        Args:
            run_id: The run created by define_task.
            templates: Template list as a JSON array string.
            datasource: Datasource name; empty means the current one.
        """
        try:
            tpl = self._json(templates, "templates") or []
        except ValueError as e:
            return FuncToolResult(success=0, error=str(e))
        return self._wrap(self._core(datasource).build_features, run_id, tpl)

    def make_split(self, run_id: str, method: str, test_size: float = 0.2, n_folds: int = 5,
                   datasource: str = "") -> FuncToolResult:
        """
        Step 5. Create the holdout test set and CV folds.

        Methods: random, stratified, group (by spec.entity), time (out-of-time test, expanding CV),
        rolling_origin (time-series backtest, folds of `horizon` periods), snapshot (latest cutoff is test).
        The method must fit the task: time series -> rolling_origin; a time column -> time/snapshot;
        repeated entities -> group. A disallowed method returns an error naming the allowed ones.

        Args:
            run_id: The run created by define_task.
            method: Split method.
            test_size: Share of rows (or time) held out for test where applicable.
            n_folds: Number of CV folds / backtest origins.
            datasource: Datasource name; empty means the current one.
        """
        return self._wrap(self._core(datasource).make_split, run_id, method, test_size, n_folds)

    def leakage_check(self, run_id: str, definitions: str = "", datasource: str = "") -> FuncToolResult:
        """
        Check features and split for leakage before training. Combines provenance checks (feature SQL time
        filters, lag/rolling offsets vs the horizon) with data checks (single feature too predictive, target
        reproduced by a formula of features, in-sample target encoding, future-tracking features, entity or
        duplicate overlap between train and test). Returns findings with severity leak|suspect and a fix.

        Args:
            run_id: The run created by define_task.
            definitions: JSON array string of extra feature definitions from a feature catalog (or ""), each {"name", "sql", "time_column",
                "cutoff"} or {"name", "template": "rolling"|"lag", "shift", "window", "center"}.
            datasource: Datasource name; empty means the current one.
        """
        try:
            defs = self._json(definitions, "definitions")
        except ValueError as e:
            return FuncToolResult(success=0, error=str(e))
        return self._wrap(self._core(datasource).leakage_check, run_id, defs)

    def train_select(self, run_id: str, candidates: str = "", exclude: str = "", time_budget: float = 0,
                     datasource: str = "") -> FuncToolResult:
        """
        Step 6. Compare a baseline and candidate models on the split's CV folds, tune the best two with FLAML
        (AutoML) inside a time budget, and recommend one by rule: best mean CV primary metric; candidates within
        one fold std are tied and the more interpretable, more stable, faster one wins. The test set is not used.

        Candidates: binary/multiclass prior, linear, rf, lgbm; regression mean, linear, rf, lgbm; time series
        seasonal_naive, ets, linear, lgbm. The baseline is always included.

        Args:
            run_id: The run.
            candidates: JSON array string of candidate names, or "" for all.
            exclude: JSON array string of feature columns to keep out of the model (leaks, ids, post-outcome
                fields), or "".
            time_budget: Total FLAML tuning seconds; 0 uses the default.
            datasource: Datasource name; empty means the current one.

        Returns:
            dict with the comparison table (CV mean/std per candidate), chosen model and the reason.
        """
        try:
            cands, excl = self._json(candidates, "candidates"), self._json(exclude, "exclude")
        except ValueError as e:
            return FuncToolResult(success=0, error=str(e))
        return self._wrap(self._core(datasource).train_select, run_id, cands, excl, time_budget or None)

    def evaluate(self, run_id: str, label: str = "", datasource: str = "") -> FuncToolResult:
        """
        Step 7. Fit the chosen model on all training rows and score the held-out test set once. Returns test
        metrics next to the CV estimate and the baseline, the CV-to-test gap check, calibration (binary), worst
        segments, train-vs-test drift (PSI) and plot paths.

        Args:
            run_id: The run.
            label: Candidate label from the train_select table to evaluate instead of the chosen one ("" = chosen).
            datasource: Datasource name; empty means the current one.
        """
        return self._wrap(self._core(datasource).evaluate, run_id, label)

    def explain(self, run_id: str, datasource: str = "") -> FuncToolResult:
        """
        Step 8a. Explain the evaluated model: SHAP for tree models, standardized coefficients for linear models.
        Returns global importance and the top contributions for the highest-scored test rows.

        Args:
            run_id: The run.
            datasource: Datasource name; empty means the current one.
        """
        return self._wrap(self._core(datasource).explain, run_id)

    def export_artifacts(self, run_id: str, limitations: str, usage: str, datasource: str = "") -> FuncToolResult:
        """
        Step 8b. Export the model card, recipe.json (every decision of the run), model.joblib, the decision log
        and reproducible train.py / score.py into <run>/export.

        Args:
            run_id: The run.
            limitations: Known limitations to write into the model card (data period, leakage risks you could
                not rule out, segments with poor accuracy, drift).
            usage: Intended use and how to act on the predictions (thresholds, refresh cadence, monitoring).
            datasource: Datasource name; empty means the current one.
        """
        return self._wrap(self._core(datasource).export_artifacts, run_id, limitations, usage)
