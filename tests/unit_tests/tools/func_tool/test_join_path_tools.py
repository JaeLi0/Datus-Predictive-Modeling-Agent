# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for JOIN path inference (real SQLite, no LLM)."""

import sqlite3

import pytest

from dataengineer.tools.func_tool.join_path_tools import JoinEdge, JoinGraph, JoinGraphBuilder, solve


def _builder(con, semantic_dir=None):
    return JoinGraphBuilder(lambda sql: con.execute(sql).fetchall(), "sqlite", semantic_dir)


@pytest.fixture
def shop():
    """orders has two FKs to customers (bill_to / ship_to): an ambiguous pair."""
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, bill_to INTEGER, ship_to INTEGER,
            FOREIGN KEY (bill_to) REFERENCES customers (customer_id),
            FOREIGN KEY (ship_to) REFERENCES customers (customer_id));
        CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders (order_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id));
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, title TEXT);
        INSERT INTO customers VALUES (1, 'a'), (2, 'b');
        INSERT INTO orders VALUES (10, 1, 2), (11, 2, 2);
        INSERT INTO order_items VALUES (100, 10, 7), (101, 11, 8);
        INSERT INTO products VALUES (7, 'x'), (8, 'y');
        """
    )
    return con


def test_two_tables_direct_fk(shop):
    g = _builder(shop).build()
    r = solve(g, ["order_items", "products"])
    assert r["method"] == "shortest_path"
    assert not r["ambiguous"]
    assert r["joins"][0]["condition"] == 'order_items."product_id" = products."product_id"'


def test_parallel_fks_are_ambiguous(shop):
    g = _builder(shop).build()
    r = solve(g, ["orders", "customers"])
    assert r["ambiguous"] is True
    conds = {opt[0]["condition"] for opt in r["alternatives"][0]["options"]}
    assert conds == {'orders."bill_to" = customers."customer_id"', 'orders."ship_to" = customers."customer_id"'}
    assert "ask" in r["advice"].lower()


def test_intermediate_table_found(shop):
    g = _builder(shop).build()
    r = solve(g, ["order_items", "products", "orders"])
    assert r["method"] == "steiner_tree_approx"
    assert len(r["joins"]) == 2
    assert r["from_clause"].count("JOIN") == 2


def test_steiner_adds_bridge_table(shop):
    g = _builder(shop).build()
    r = solve(g, ["products", "orders"])
    assert r["intermediate_tables"] == ["order_items"]
    assert len(r["joins"]) == 2


def test_key_normalization_detected():
    """A key that lost its leading zero only matches after zero padding."""
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE school (code TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE score (code TEXT, val INTEGER, FOREIGN KEY (code) REFERENCES school (code));
        """
    )
    codes = [f"{i:05d}" for i in range(1, 41)]  # 00001..00040, all with leading zeros
    con.executemany("INSERT INTO school VALUES (?, 'n')", [(c,) for c in codes])
    con.executemany("INSERT INTO score VALUES (?, 1)", [(str(int(c)),) for c in codes[:30]] + [(c,) for c in codes[30:]])
    g = _builder(con).build()
    r = solve(g, ["score", "school"])
    j = r["joins"][0]
    assert j["key_match_rate"] < 0.5
    assert j["normalized_key_match_rate"] == 1.0
    assert "printf('%05d'" in j["condition"]
    joined = con.execute(f"SELECT COUNT(*) FROM score JOIN school ON {j['condition']}").fetchone()[0]
    assert joined == 40


def test_semantic_identifier_edge(tmp_path):
    con = sqlite3.connect(":memory:")
    con.executescript("CREATE TABLE a (k TEXT PRIMARY KEY); CREATE TABLE b (ref TEXT); INSERT INTO a VALUES ('1'); INSERT INTO b VALUES ('1');")
    (tmp_path / "a.yml").write_text("data_source:\n  name: a\n  sql_table: a\n  identifiers:\n    - {name: thing, type: PRIMARY, expr: k}\n")
    (tmp_path / "b.yml").write_text("data_source:\n  name: b\n  sql_table: b\n  identifiers:\n    - {name: thing, type: FOREIGN, expr: ref}\n")
    g = _builder(con, tmp_path).build()
    r = solve(g, ["a", "b"])
    assert r["joins"][0]["evidence"] == ["semantic_model"]
    assert r["joins"][0]["key_match_rate"] == 1.0


def test_unknown_table_raises(shop):
    g = _builder(shop).build()
    with pytest.raises(ValueError):
        solve(g, ["orders", "nope"])


def test_graph_merges_sources():
    g = JoinGraph()
    g.add_edge(JoinEdge("a", "id", "b", "a_id", ["declared_fk"]))
    g.add_edge(JoinEdge("b", "a_id", "a", "id", ["name_match"]))
    assert len(g.edges) == 1
    assert sorted(next(iter(g.edges.values())).sources) == ["declared_fk", "name_match"]
