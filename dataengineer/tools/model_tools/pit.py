# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Point-in-time SQL over the semantic layer: features before a cutoff, labels after it.

A feature or label is declared in business terms: a semantic measure ("revenue"), the entity identifier
("customer"), the time dimension ("invoice_date") and a window. This module resolves each name to its
table and column expression from the semantic-model YAML, asks the JOIN path solver how to connect the
tables involved, and renders SQL whose time filter is generated, never written by the LLM:

    feature: time >= cutoff - window AND time < cutoff
    label:   time >= cutoff AND time < cutoff + window

That generated filter is what the static leakage check later verifies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

AGG_SQL = {"SUM": "SUM", "COUNT": "COUNT", "COUNT_DISTINCT": "COUNT(DISTINCT", "AVERAGE": "AVG", "AVG": "AVG",
           "MAX": "MAX", "MIN": "MIN"}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _qualify(table: str, expr: str, columns: List[str]) -> str:
    """Prefix bare column names in a measure expression with their table (longest names first)."""
    import re

    out = expr
    for c in sorted(columns, key=len, reverse=True):
        out = re.sub(rf'(?<![\w."]){re.escape(c)}(?![\w"])', f'{table}.{_q(c)}', out)
        out = re.sub(rf'(?<![\w.])"{re.escape(c)}"', f'{table}.{_q(c)}', out)
    return out


@dataclass
class SemanticObject:
    kind: str  # measure | dimension | identifier
    name: str
    table: str
    expr: str
    agg: Optional[str] = None
    type: Optional[str] = None


class SemanticCatalog:
    """Name -> (table, expression) lookup over a directory of semantic-model YAML files."""

    def __init__(self, semantic_dir: Path, table_columns: Optional[Dict[str, List[str]]] = None):
        self.objects: Dict[str, List[SemanticObject]] = {}
        self.table_columns = table_columns or {}
        for f in sorted(Path(semantic_dir).glob("*.y*ml")):
            for doc in yaml.safe_load_all(f.read_text()):
                ds = (doc or {}).get("data_source")
                if not ds:
                    continue
                table = (ds.get("sql_table") or ds["name"]).split(".")[-1].strip('`"')
                for m in ds.get("measures") or []:
                    self._add(SemanticObject("measure", m["name"], table, str(m.get("expr") or m["name"]),
                                             agg=str(m.get("agg", "SUM")).upper()))
                for d in ds.get("dimensions") or []:
                    self._add(SemanticObject("dimension", d["name"], table, str(d.get("expr") or d["name"]),
                                             type=str(d.get("type", "")).upper()))
                for i in ds.get("identifiers") or []:
                    self._add(SemanticObject("identifier", i["name"], table, str(i.get("expr") or i["name"]),
                                             type=str(i.get("type", "")).upper()))

    def _add(self, obj: SemanticObject) -> None:
        self.objects.setdefault(obj.name.lower(), []).append(obj)

    def get(self, name: str, kind: str, prefer_tables: Optional[List[str]] = None) -> SemanticObject:
        """Resolve a name. Objects on prefer_tables win (fewer joins); otherwise a PRIMARY identifier."""
        cands = [o for o in self.objects.get(name.lower(), []) if o.kind == kind]
        if not cands:
            known = sorted({o.name for objs in self.objects.values() for o in objs if o.kind == kind})
            raise KeyError(f"no {kind} named {name!r}; known: {known}")
        for t in prefer_tables or []:
            same = [o for o in cands if o.table == t]
            if same:
                return same[0]
        primary = [o for o in cands if o.type == "PRIMARY"]
        return (primary or cands)[0]

    def col(self, obj: SemanticObject) -> str:
        cols = self.table_columns.get(obj.table)
        if cols:
            return _qualify(obj.table, obj.expr, cols)
        if obj.expr.replace("_", "").isalnum():
            return f"{obj.table}.{_q(obj.expr)}"
        return obj.expr


@dataclass
class WindowAgg:
    """One aggregate of a semantic measure over a time window relative to the cutoff."""

    name: str
    measure: str
    window_days: int
    entity: str
    time_dim: str
    filters: Optional[List[str]] = None  # extra semantic-free SQL predicates, e.g. ["invoices.\"Invoice\" NOT LIKE 'C%'"]


def window_sql(spec: WindowAgg, catalog: SemanticCatalog, join_solver: Callable[[List[str]], Dict], cutoff_expr: str,
               side: str) -> Dict:
    """Render SQL for a window aggregate. side='feature' looks back from the cutoff, side='label' forward.

    cutoff_expr is a SQL expression for the cutoff (a literal like DATE '2011-01-01', or a column such as
    s.cutoff when computing many snapshots in one query).
    """
    measure = catalog.get(spec.measure, "measure")
    tdim = catalog.get(spec.time_dim, "dimension", prefer_tables=[measure.table])
    ent = catalog.get(spec.entity, "identifier", prefer_tables=[measure.table, tdim.table])
    tables = list(dict.fromkeys([measure.table, tdim.table, ent.table]))
    if len(tables) > 1:
        plan = join_solver(tables)
        if plan.get("ambiguous"):
            raise ValueError(f"ambiguous join between {tables}: {plan.get('alternatives')}")
        from_clause = plan["from_clause"]
    else:
        plan = None
        from_clause = f"FROM {tables[0]}"
    agg = measure.agg if measure.agg in AGG_SQL else "SUM"
    mexpr = catalog.col(measure)
    agg_expr = f"COUNT(DISTINCT {mexpr})" if agg == "COUNT_DISTINCT" else f"{AGG_SQL[agg]}({mexpr})"
    t = catalog.col(tdim)
    w = int(spec.window_days)
    if side == "feature":
        time_filter = f"{t} >= {cutoff_expr} - INTERVAL {w} DAY AND {t} < {cutoff_expr}"
    elif side == "label":
        time_filter = f"{t} >= {cutoff_expr} AND {t} < {cutoff_expr} + INTERVAL {w} DAY"
    else:
        raise ValueError("side must be feature or label")
    where = [time_filter] + list(spec.filters or [])
    sql = (f"SELECT {catalog.col(ent)} AS entity_id, {agg_expr} AS {_q(spec.name)}\n{from_clause}\n"
           f"WHERE {' AND '.join('(' + p + ')' for p in where)}\nGROUP BY 1")
    return {"name": spec.name, "sql": sql, "time_expr": t, "tables": tables,
            "join_plan": plan and {k: plan[k] for k in ("from_clause", "joins", "intermediate_tables") if k in plan},
            "window": [-w, 0] if side == "feature" else [0, w]}


def snapshot_frame(cutoffs: List[str], label: WindowAgg, population: WindowAgg, catalog: SemanticCatalog,
                   join_solver: Callable[[List[str]], Dict], query: Callable[[str], "object"]) -> Dict:
    """Build the modeling table for a point-in-time task: one row per (entity active in the population window
    before the cutoff, cutoff), with the label aggregated over the window after the cutoff (0 when absent).

    Returns {"frame": DataFrame[entity_id, snapshot, <label.name>], "sql": {...}}.
    """
    import pandas as pd

    parts = []
    sqls = {}
    for c in sorted(cutoffs):
        cut = f"TIMESTAMP '{pd.Timestamp(c)}'"
        pop = window_sql(population, catalog, join_solver, cut, side="feature")
        lab = window_sql(label, catalog, join_solver, cut, side="label")
        p = query(pop["sql"])[["entity_id"]].dropna()  # rows without an entity (e.g. guest orders) are not a population
        l_ = query(lab["sql"]).dropna(subset=["entity_id"])
        p["entity_id"] = p["entity_id"].astype(str)
        l_["entity_id"] = l_["entity_id"].astype(str)
        f = p.merge(l_, on="entity_id", how="left")
        f[label.name] = f[label.name].fillna(0)
        f["snapshot"] = str(pd.Timestamp(c).date())
        parts.append(f)
        sqls.setdefault("population", pop["sql"])
        sqls.setdefault("label", lab["sql"])
    return {"frame": pd.concat(parts, ignore_index=True), "sql": sqls}
