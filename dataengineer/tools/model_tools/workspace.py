# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Run workspace: one directory per modeling run holding every intermediate table and decision.

Tools hand frames to each other through this directory (parquet + json) rather than through the LLM
context, so the LLM only ever sees summaries and a run can be replayed or exported later.

Layout: spec.json, raw.parquet, clean.parquet, features.parquet, feature_meta.json, split.parquet,
split.json, and decisions.jsonl (an append-only log of every LLM decision and tool result summary).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dataengineer.tools.model_tools.spec import TaskSpec

# Internal row key added by define_task. It follows every row through cleaning and features so a run can be
# re-scored on a fixed holdout later; it is never profiled, deduplicated on, or used as a feature.
ROW_ID = "__rid"


class RunWorkspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- frames -------------------------------------------------------------------------
    def save_frame(self, name: str, df: pd.DataFrame) -> Path:
        path = self.root / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def load_frame(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"run has no {name!r} table yet; run the step that produces it first")
        return pd.read_parquet(path)

    def has(self, name: str) -> bool:
        return (self.root / f"{name}.parquet").exists() or (self.root / f"{name}.json").exists()

    def latest_table(self) -> str:
        """The most processed modeling table available: features > clean > raw."""
        for name in ("features", "clean", "raw"):
            if (self.root / f"{name}.parquet").exists():
                return name
        raise FileNotFoundError("run has no data yet; call define_task first")

    # ---- json ---------------------------------------------------------------------------
    def save_json(self, name: str, obj: Any) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
        return path

    def load_json(self, name: str, default: Any = None) -> Any:
        path = self.root / f"{name}.json"
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"run has no {name}.json yet")
        return json.loads(path.read_text())

    def save_spec(self, spec: TaskSpec) -> None:
        self.save_json("spec", spec.model_dump())

    def load_spec(self) -> TaskSpec:
        return TaskSpec(**self.load_json("spec"))

    # ---- decision log -------------------------------------------------------------------
    def log(self, step: str, decision: Optional[Dict] = None, result: Optional[Dict] = None) -> None:
        rec = {"ts": round(time.time(), 3), "step": step, "decision": decision, "result": result}
        with open(self.root / "decisions.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def decisions(self) -> List[Dict]:
        path = self.root / "decisions.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
