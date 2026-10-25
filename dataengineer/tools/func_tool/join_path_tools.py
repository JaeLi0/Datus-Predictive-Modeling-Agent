# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""JOIN path inference over a table-key graph.

The graph has one node per table and one edge per candidate join key, collected from three sources:
declared foreign keys, semantic-model identifiers (same entity name across data sources) and a
same-column-name heuristic for key-like columns. Each edge is profiled against the data: the share of
distinct keys on one side found on the other. When the raw match rate is low the tool tries a key
normalization (zero padding to a common length, trimming) and keeps the normalized condition if it
matches clearly better, which catches keys that silently lost leading zeros.

For two tables the tool returns the shortest path; for three or more an approximate Steiner tree
(Kou-Markowsky-Berman: metric closure, MST, expand, prune). Equal-cost alternatives are reported as
ambiguous so the agent can ask the user instead of guessing.
"""

import heapq
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml
from agents import Tool

from dataengineer.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from dataengineer.utils.loggings import get_logger

logger = get_logger(__name__)

SOURCE_WEIGHT = {"declared_fk": 1.0, "semantic_model": 1.0, "name_match": 1.5}
KEY_NAME_RE = re.compile(r"(^id$|_id$|id$|code$|_key$|_no$|^key$)", re.IGNORECASE)
GOOD_MATCH = 0.98
JOIN_PATH_PROMPT_HINT = (
    "\n\n## Multi-table queries\n"
    "When a question needs 2 or more tables, call suggest_join_path with those tables before writing SQL, "
    "use the join conditions it returns (they may include key normalization), and if it reports "
    "ambiguous=true ask the user which relationship is meant.\n"
)
LOW_MATCH_PENALTY = 2.0


def _bare(table: str) -> str:
    return table.split(".")[-1].strip('`"').lower()


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass
class JoinEdge:
    left_table: str
    left_col: str
    right_table: str
    right_col: str
    sources: List[str] = field(default_factory=list)
    match_rate: Optional[float] = None
    normalized_match_rate: Optional[float] = None
    left_expr: Optional[str] = None  # normalized expression templates, "{t}" is the table alias
    right_expr: Optional[str] = None

    @property
    def key(self) -> frozenset:
        return frozenset({(self.left_table, self.left_col.lower()), (self.right_table, self.right_col.lower())})

    @property
    def weight(self) -> float:
        w = min(SOURCE_WEIGHT.get(s, 2.0) for s in self.sources) if self.sources else 2.0
        best = max(r for r in (self.match_rate, self.normalized_match_rate) if r is not None) if (
            self.match_rate is not None or self.normalized_match_rate is not None
        ) else None
        if best is not None and best < 0.5:
            w += LOW_MATCH_PENALTY
        return w

    def other(self, table: str) -> str:
        return self.right_table if table == self.left_table else self.left_table

    def condition(self) -> str:
        left = (self.left_expr or "{t}.{c}").format(t=self.left_table, c=_q(self.left_col))
        right = (self.right_expr or "{t}.{c}").format(t=self.right_table, c=_q(self.right_col))
        return f"{left} = {right}"

    def to_dict(self) -> dict:
        d = {
            "left": f"{self.left_table}.{self.left_col}",
            "right": f"{self.right_table}.{self.right_col}",
            "condition": self.condition(),
            "evidence": sorted(set(self.sources)),
        }
        if self.match_rate is not None:
            d["key_match_rate"] = round(self.match_rate, 4)
        if self.normalized_match_rate is not None:
            d["normalized_key_match_rate"] = round(self.normalized_match_rate, 4)
            d["note"] = (
                "Direct equality loses rows; the keys only line up after normalization. "
                "Use the condition above, not a plain column equality."
            )
        return d


class JoinGraph:
    """Undirected multigraph of tables; parallel edges are kept (they are distinct join options)."""

    def __init__(self):
        self.edges: Dict[frozenset, JoinEdge] = {}
        self.adj: Dict[str, List[JoinEdge]] = {}

    def add_table(self, table: str):
        self.adj.setdefault(table, [])

    def add_edge(self, edge: JoinEdge) -> JoinEdge:
        if edge.left_table == edge.right_table:
            return edge
        existing = self.edges.get(edge.key)
        if existing:
            for s in edge.sources:
                if s not in existing.sources:
                    existing.sources.append(s)
            return existing
        self.edges[edge.key] = edge
        self.add_table(edge.left_table)
        self.add_table(edge.right_table)
        self.adj[edge.left_table].append(edge)
        self.adj[edge.right_table].append(edge)
        return edge

    def shortest_paths(self, src: str, dst: str, limit: int = 5) -> Tuple[float, List[List[JoinEdge]]]:
        """All equal-cost shortest paths (as edge lists) from src to dst."""
        if src not in self.adj or dst not in self.adj:
            return float("inf"), []
        dist = {src: 0.0}
        preds: Dict[str, List[Tuple[str, JoinEdge]]] = {src: []}
        heap = [(0.0, src)]
        eps = 1e-9
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")) + eps:
                continue
            for e in self.adj[u]:
                v = e.other(u)
                nd = d + e.weight
                if nd < dist.get(v, float("inf")) - eps:
                    dist[v] = nd
                    preds[v] = [(u, e)]
                    heapq.heappush(heap, (nd, v))
                elif abs(nd - dist.get(v, float("inf"))) <= eps:
                    preds[v].append((u, e))
        if dst not in dist:
            return float("inf"), []

        paths: List[List[JoinEdge]] = []

        def walk(node: str, acc: List[JoinEdge]):
            if len(paths) >= limit:
                return
            if node == src:
                paths.append(list(reversed(acc)))
                return
            for p, e in preds[node]:
                walk(p, acc + [e])

        walk(dst, [])
        return dist[dst], paths

    def steiner(self, terminals: List[str]) -> Tuple[List[JoinEdge], List[dict]]:
        """KMB 2-approximation of the minimum Steiner tree. Returns (edges, ambiguities)."""
        closure: Dict[Tuple[str, str], Tuple[float, List[List[JoinEdge]]]] = {}
        for a, b in itertools.combinations(terminals, 2):
            closure[(a, b)] = self.shortest_paths(a, b)
        # Prim's MST on the metric closure
        in_tree = {terminals[0]}
        mst_pairs: List[Tuple[str, str]] = []
        while len(in_tree) < len(terminals):
            best = None
            for (a, b), (cost, paths) in closure.items():
                if (a in in_tree) != (b in in_tree) and paths and (best is None or cost < best[0]):
                    best = (cost, a, b)
            if best is None:
                raise ValueError("tables are not connected: " + ", ".join(t for t in terminals if t not in in_tree))
            _, a, b = best
            mst_pairs.append((a, b))
            in_tree.update({a, b})
        ambiguities = []
        chosen: Dict[frozenset, JoinEdge] = {}
        for a, b in mst_pairs:
            cost, paths = closure[(a, b)]
            if len(paths) > 1:
                ambiguities.append({"between": [a, b], "options": [[e.to_dict() for e in p] for p in paths]})
            for e in paths[0]:
                chosen[e.key] = e
        # Prune: MST over the expanded subgraph, then drop non-terminal leaves
        edges = sorted(chosen.values(), key=lambda e: e.weight)
        parent: Dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree = []
        for e in edges:
            ra, rb = find(e.left_table), find(e.right_table)
            if ra != rb:
                parent[ra] = rb
                tree.append(e)
        terms = set(terminals)
        changed = True
        while changed:
            changed = False
            degree: Dict[str, int] = {}
            for e in tree:
                degree[e.left_table] = degree.get(e.left_table, 0) + 1
                degree[e.right_table] = degree.get(e.right_table, 0) + 1
            for e in list(tree):
                for t in (e.left_table, e.right_table):
                    if degree.get(t) == 1 and t not in terms:
                        tree.remove(e)
                        changed = True
                        break
        return tree, ambiguities


def solve(graph: JoinGraph, tables: List[str]) -> dict:
    """Pure planning step on a built graph (unit-testable without a database)."""
    terminals = list(dict.fromkeys(_bare(t) for t in tables))
    unknown = [t for t in terminals if t not in graph.adj]
    if unknown:
        raise ValueError(f"no join keys known for table(s): {', '.join(unknown)}")
    if len(terminals) < 2:
        return {"tables": terminals, "joins": [], "ambiguous": False, "advice": "Single table, no join needed."}
    if len(terminals) == 2:
        _, paths = graph.shortest_paths(terminals[0], terminals[1])
        if not paths:
            raise ValueError(f"no join path between {terminals[0]} and {terminals[1]}")
        tree = paths[0]
        ambiguities = (
            [{"between": terminals, "options": [[e.to_dict() for e in p] for p in paths]}] if len(paths) > 1 else []
        )
        method = "shortest_path"
    else:
        tree, ambiguities = graph.steiner(terminals)
        method = "steiner_tree_approx"

    used = {t for e in tree for t in (e.left_table, e.right_table)}
    order = [terminals[0]]
    remaining = list(tree)
    from_lines = [f"FROM {terminals[0]}"]
    while remaining:
        for e in remaining:
            a, b = e.left_table, e.right_table
            if (a in order) != (b in order):
                new = b if a in order else a
                order.append(new)
                from_lines.append(f"JOIN {new} ON {e.condition()}")
                remaining.remove(e)
                break
        else:
            break
    result = {
        "method": method,
        "tables": terminals,
        "intermediate_tables": sorted(used - set(terminals)),
        "joins": [e.to_dict() for e in tree],
        "from_clause": "\n".join(from_lines),
        "ambiguous": bool(ambiguities),
    }
    if ambiguities:
        result["alternatives"] = ambiguities
        result["advice"] = (
            "Several join paths have the same cost. Ask the user which relationship they mean "
            "(ask_user) before writing SQL; the from_clause above is only the first option."
        )
    else:
        result["advice"] = "Use from_clause as the FROM/JOIN part of the query."
    return result


class JoinGraphBuilder:
    """Builds a JoinGraph for one datasource from FKs, semantic models, name heuristics and data profiling."""

    def __init__(self, query: Callable[[str], List[tuple]], dialect: str, semantic_dir: Optional[Path] = None):
        self.query = query
        d = str(getattr(dialect, "value", dialect) or "").lower()
        self.dialect = next((k for k in ("sqlite", "duckdb") if k in d), d)
        self.semantic_dir = semantic_dir

    # ---- metadata -------------------------------------------------------------------------
    def tables(self) -> List[str]:
        if self.dialect == "sqlite":
            rows = self.query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        else:
            rows = self.query("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")
        return [r[0] for r in rows]

    def columns(self, table: str) -> List[Tuple[str, bool]]:
        """(column, is_primary_key)"""
        if self.dialect == "sqlite":
            return [(r[1], bool(r[5])) for r in self.query(f"PRAGMA table_info({_q(table)})")]
        pk = set()
        try:
            for r in self.query(
                "SELECT constraint_column_names FROM duckdb_constraints() "
                f"WHERE table_name = '{table}' AND constraint_type = 'PRIMARY KEY'"
            ):
                pk.update(r[0])
        except Exception:
            pass
        rows = self.query(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' "
            "AND table_schema = current_schema() ORDER BY ordinal_position"
        )
        return [(r[0], r[0] in pk) for r in rows]

    def declared_fks(self, table: str) -> List[Tuple[str, str, str]]:
        """(from_col, ref_table, ref_col)"""
        if self.dialect == "sqlite":
            return [(r[3], r[2], r[4]) for r in self.query(f"PRAGMA foreign_key_list({_q(table)})")]
        try:
            rows = self.query(
                "SELECT constraint_column_names, referenced_table, referenced_column_names FROM duckdb_constraints() "
                f"WHERE table_name = '{table}' AND constraint_type = 'FOREIGN KEY'"
            )
        except Exception:
            return []
        return [(c, r[1], rc) for r in rows for c, rc in zip(r[0], r[2])]

    def semantic_identifiers(self) -> List[Tuple[str, str, str, str]]:
        """(table, identifier_name, type, column_expr) from semantic-model YAML files."""
        out = []
        if not self.semantic_dir or not Path(self.semantic_dir).exists():
            return out
        for f in sorted(Path(self.semantic_dir).glob("*.y*ml")):
            try:
                for doc in yaml.safe_load_all(f.read_text()):
                    ds = (doc or {}).get("data_source")
                    if not ds:
                        continue
                    table = _bare(ds.get("sql_table") or ds.get("name", ""))
                    for ident in ds.get("identifiers") or []:
                        expr = str(ident.get("expr") or ident.get("name")).strip('`"')
                        out.append((table, ident.get("name"), str(ident.get("type", "")).upper(), expr))
            except Exception as e:
                logger.warning(f"skip semantic file {f}: {e}")
        return out

    # ---- profiling ------------------------------------------------------------------------
    def _rate(self, lt: str, lexpr: str, rt: str, rexpr: str, sample: int = 2000) -> Optional[float]:
        sql = (
            f"SELECT COUNT(*), SUM(CASE WHEN v IN (SELECT {rexpr} FROM {_q(rt)} AS r) THEN 1 ELSE 0 END) "
            f"FROM (SELECT DISTINCT {lexpr} AS v FROM {_q(lt)} AS l WHERE {lexpr} IS NOT NULL LIMIT {sample}) AS s"
        )
        try:
            n, m = self.query(sql)[0]
            return (m or 0) / n if n else None
        except Exception as e:
            logger.debug(f"profiling failed for {lt}/{rt}: {e}")
            return None

    def _pad(self, col: str, width: int) -> str:
        if self.dialect == "sqlite":
            return f"printf('%0{width}d', CAST({col} AS INTEGER))"
        return f"lpad(CAST({col} AS VARCHAR), {width}, '0')"

    def profile(self, edge: JoinEdge):
        lc, rc = "l." + _q(edge.left_col), "r." + _q(edge.right_col)
        fwd = self._rate(edge.left_table, lc, edge.right_table, rc)
        bwd = self._rate(edge.right_table, rc.replace("r.", "l.", 1), edge.left_table, lc.replace("l.", "r.", 1))
        rates = [r for r in (fwd, bwd) if r is not None]
        if not rates:
            return
        edge.match_rate = max(rates)
        if edge.match_rate >= GOOD_MATCH:
            return
        try:
            (lmin, lmax), (rmin, rmax) = (
                self.query(
                    f"SELECT MIN(LENGTH(CAST({_q(c)} AS TEXT))), MAX(LENGTH(CAST({_q(c)} AS TEXT))) FROM {_q(t)}"
                )[0]
                for t, c in ((edge.left_table, edge.left_col), (edge.right_table, edge.right_col))
            )
        except Exception:
            return
        width = max(lmax or 0, rmax or 0)
        if not width or (lmin == lmax == rmin == rmax):
            return
        candidates = []
        if (lmin or 0) < width:
            candidates.append(("left", self._pad("{t}.{c}", width)))
        if (rmin or 0) < width:
            candidates.append(("right", self._pad("{t}.{c}", width)))
        best = None
        for side, tmpl in candidates:
            if side == "left":
                lx, rx = tmpl.format(t="l", c=_q(edge.left_col)), rc
                rate = max(
                    r for r in (
                        self._rate(edge.left_table, lx, edge.right_table, rx),
                        self._rate(edge.right_table, rc.replace("r.", "l.", 1), edge.left_table, tmpl.format(t="r", c=_q(edge.left_col))),
                    ) if r is not None
                )
            else:
                lx, rx = lc, tmpl.format(t="r", c=_q(edge.right_col))
                rate = max(
                    r for r in (
                        self._rate(edge.left_table, lx, edge.right_table, rx),
                        self._rate(edge.right_table, tmpl.format(t="l", c=_q(edge.right_col)), edge.left_table, lc.replace("l.", "r.", 1)),
                    ) if r is not None
                )
            if best is None or rate > best[0]:
                best = (rate, side, tmpl)
        if best and best[0] >= edge.match_rate + 0.02:
            edge.normalized_match_rate = best[0]
            if best[1] == "left":
                edge.left_expr = best[2]
            else:
                edge.right_expr = best[2]

    # ---- build ----------------------------------------------------------------------------
    def build(self, profile: bool = True) -> JoinGraph:
        g = JoinGraph()
        tables = {_bare(t): t for t in self.tables()}
        cols: Dict[str, List[Tuple[str, bool]]] = {}
        for bt, t in tables.items():
            g.add_table(bt)
            try:
                cols[bt] = self.columns(t)
            except Exception:
                cols[bt] = []
            for fc, rt, rcol in self.declared_fks(t):
                if _bare(rt) in tables:
                    g.add_edge(JoinEdge(bt, fc, _bare(rt), rcol, ["declared_fk"]))

        by_ident: Dict[str, List[Tuple[str, str, str]]] = {}
        for table, name, typ, expr in self.semantic_identifiers():
            if table in tables and name:
                by_ident.setdefault(name.lower(), []).append((table, typ, expr))
        for holders in by_ident.values():
            for (t1, ty1, c1), (t2, ty2, c2) in itertools.combinations(holders, 2):
                if t1 != t2 and ("PRIMARY" in (ty1, ty2) or ty1 == ty2 == "UNIQUE"):
                    g.add_edge(JoinEdge(t1, c1, t2, c2, ["semantic_model"]))

        for t1, t2 in itertools.combinations(sorted(cols), 2):
            c2map = {c.lower(): (c, pk) for c, pk in cols[t2]}
            for c1, pk1 in cols[t1]:
                hit = c2map.get(c1.lower())
                if hit and (pk1 or hit[1] or KEY_NAME_RE.search(c1)):
                    g.add_edge(JoinEdge(t1, c1, t2, hit[0], ["name_match"]))

        if profile:
            for e in g.edges.values():
                self.profile(e)
        return g


class JoinPathTools:
    """Function tool wrapper: suggest_join_path for the agent."""

    def __init__(self, agent_config, db_func_tool=None):
        self.agent_config = agent_config
        self.db_func_tool = db_func_tool
        self._graphs: Dict[str, JoinGraph] = {}

    def available_tools(self) -> List[Tool]:
        return [trans_to_function_tool(self.suggest_join_path)]

    def _graph(self, datasource: str) -> JoinGraph:
        if datasource in self._graphs:
            return self._graphs[datasource]
        if self.db_func_tool is None:
            from dataengineer.tools.func_tool.database import DBFuncTool

            self.db_func_tool = DBFuncTool(agent_config=self.agent_config)
        connector = self.db_func_tool._get_connector(datasource or None)

        def query(sql: str) -> List[tuple]:
            res = connector.execute_pandas(sql)
            if not res.success:
                raise RuntimeError(res.error)
            df = res.sql_return
            return [tuple(r) for r in df.itertuples(index=False, name=None)]

        ds_name = datasource or getattr(self.agent_config, "current_datasource", "") or ""
        semantic_dir = None
        try:
            semantic_dir = self.agent_config.path_manager.semantic_model_path(ds_name)
        except Exception:
            pass
        graph = JoinGraphBuilder(query, str(getattr(connector, "dialect", "")), semantic_dir).build()
        self._graphs[datasource] = graph
        return graph

    def suggest_join_path(self, tables: List[str], datasource: str = "") -> FuncToolResult:
        """
        Recommend how to JOIN a set of tables. Call this BEFORE writing any SQL that uses 2 or more tables.

        It builds a table-key graph from declared foreign keys, semantic-model entities and key-like
        column names, checks each key against the data (how many keys actually match, and whether a
        normalization such as zero padding is needed), and returns the minimal connecting join tree:
        the join conditions, any intermediate tables, and a ready FROM/JOIN clause.
        If several paths are equally good it returns ambiguous=true with the alternatives; then ask the
        user which relationship is meant instead of guessing.

        Args:
            tables: Table names the question needs, e.g. ["frpm", "satscores"].
            datasource: Datasource name; empty means the current one.

        Returns:
            dict with joins (condition, evidence, key match rates), intermediate_tables, from_clause,
            ambiguous and alternatives.
        """
        try:
            return FuncToolResult(result=solve(self._graph(datasource), tables))
        except Exception as e:
            logger.warning(f"suggest_join_path failed: {e}")
            return FuncToolResult(success=0, error=str(e))
