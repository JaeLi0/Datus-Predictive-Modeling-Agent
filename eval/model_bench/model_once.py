"""Run the gen_model agent once on one benchmark task and dump a JSON record.

Same streaming path as ``dataengineer-cli --print`` (PrintModeRunner) with eval-only changes:
- the task's injected DuckDB copy is registered as datasource ``bench_<id>`` (semantic models copied to its
  semantic-model directory when the task has them);
- persistent memory is off and the ``task`` delegation tool is removed;
- ask_user gets a fixed "go with your proposal" reply and every such call is counted (the confirmation points a
  human would have answered);
- group ``a`` = full agent, group ``b`` = no guardrails (no leakage_check tool, split method not enforced).

Usage: python eval/model_bench/model_once.py --task C1 --group a --out <dir>
"""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataengineer.cli.print_mode import PrintModeRunner  # noqa: E402
from dataengineer.schemas.action_history import ActionRole, ActionStatus  # noqa: E402
from dataengineer.utils.async_utils import run_async  # noqa: E402
from dataengineer.utils.loggings import configure_logging  # noqa: E402

CFG = str(ROOT / "conf/eval_agent.yml")
PREP = ROOT / "eval/model_bench/prepared"
AUTO_REPLY = "（无人值守运行，没有人可以确认）请按你的提议继续。"


def build_message(vis: dict) -> str:
    parts = [vis["request"].strip(), "", f"数据在数据源 bench_{vis['id']} 中，表：{', '.join(vis['tables'])}。"]
    if vis.get("data_dictionary"):
        parts += ["", "部分字段说明："] + [f"- {k}: {v}" for k, v in vis["data_dictionary"].items()]
    if vis.get("feature_catalog"):
        parts += ["", "特征库中可选用的现成特征（定义如下）："]
        for f in vis["feature_catalog"]:
            d = f.get("sql") or json.dumps(f.get("definition"), ensure_ascii=False)
            parts.append(f"- {f['name']}: {f['description']}\n  定义: {d.strip()}")
    if vis.get("semantic_dir"):
        parts += ["", "该数据源已配置语义模型（实体、度量、时间维），可用于按时间点取数。"]
    return "\n".join(parts)


class BenchRunner(PrintModeRunner):
    def __init__(self, args, task_id: str, group: str, runs_dir: Path, time_budget: float):
        super().__init__(args)
        self.interactions = 0
        self.task_id, self.group, self.runs_dir, self.time_budget = task_id, group, runs_dir, time_budget

    def run(self):
        from dataengineer.agent.node.gen_model_agentic_node import GenModelAgenticNode
        from dataengineer.agent.node.node_factory import create_node_input

        node = GenModelAgenticNode(self.agent_config, execution_mode="interactive", guardrails=self.group == "a",
                                   time_budget=self.time_budget, runs_dir=self.runs_dir)
        node.memory_enabled = False
        broker = node._get_or_create_broker()
        orig_request = broker.request

        async def counted_request(*a, **kw):  # remember how many questions the pending ask_user carries
            self._n_questions = len(kw.get("contents") or (a[0] if a else []) or [1])
            return await orig_request(*a, **kw)

        broker.request = counted_request
        node.tools = [t for t in node.tools if t.name != "task"]
        self.tool_names = [t.name for t in node.tools]
        node.input = create_node_input(user_message=self.message, node=node, catalog=None,
                                       database=f"bench_{self.task_id}", db_schema=None,
                                       at_tables=[], at_metrics=[], at_sqls=[])
        run_async(self._stream_chat(node))

    def _write_payload(self, payload):
        pass

    def _read_interaction_input(self) -> str:
        self.interactions += 1
        n = getattr(self, "_n_questions", 1)
        return AUTO_REPLY if n <= 1 else json.dumps([AUTO_REPLY] * n, ensure_ascii=False)


def register_datasource(runner_cls_args, vis: dict):
    """Patch the loaded config: add bench_<id> pointing at the task's DuckDB copy."""
    from dataengineer.configuration import agent_config_loader as L

    orig = L.load_agent_config

    def patched(*a, **kw):
        kw["datasource"] = "demo"
        cfg = orig(*a, **kw)
        name = f"bench_{vis['id']}"
        ds = copy.copy(cfg.services.datasources["demo"])
        ds.uri = f"duckdb:///{vis['db']}"
        ds.database = Path(vis["db"]).stem
        ds.logic_name = name
        cfg.services.datasources[name] = ds
        cfg._current_datasource = name
        if vis.get("semantic_dir"):
            target = cfg.path_manager.semantic_model_path(name)
            for y in (ROOT / vis["semantic_dir"]).glob("*.yml"):
                shutil.copy(y, target / y.name)
        return cfg

    L.load_agent_config = patched
    import dataengineer.cli.print_mode as PM

    if hasattr(PM, "load_agent_config"):
        PM.load_agent_config = patched


def latest_run(runs_dir: Path):
    runs = [p for p in runs_dir.glob("*") if (p / "spec.json").exists()]
    if not runs:
        return None
    exported = [p for p in runs if (p / "export" / "model_card.md").exists()]
    done = [p for p in runs if (p / "evaluation.json").exists()]
    pick = max(exported or done or runs, key=lambda p: (p / "decisions.jsonl").stat().st_mtime if (p / "decisions.jsonl").exists() else 0)
    return pick.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--group", choices=["a", "b"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time-budget", type=float, default=40)
    a = p.parse_args()
    configure_logging(False, console_output=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    vis = json.loads((PREP / a.task / "visible.json").read_text())
    # a private copy per run: the agent's connector locks the file, and runs/scorers work in parallel
    local_db = out / f"bench_{a.task}.duckdb"  # file stem = catalog name = datasource name
    shutil.copy(ROOT / vis["db"], local_db)
    vis = {**vis, "db": str(local_db)}
    message = build_message(vis)
    register_datasource(None, vis)
    ns = argparse.Namespace(
        config=CFG, datasource=f"bench_{a.task}", print_mode=message, subagent="gen_model",
        resume=None, proxy_tools=None, session_scope="model_bench", stream_thinking=False, debug=False,
        db_type=None, db_path=None, history_file=None, no_color=True, save_llm_trace=False,
        filesystem_strict=None, web=False, port=None, host=None, chatbot_dist=None,
    )
    t0 = time.time()
    rec = {"task": a.task, "group": a.group, "message": message}
    runner = None
    try:
        runner = BenchRunner(ns, a.task, a.group, runs_dir, a.time_budget)
        runner.run()
        rec["error"] = None
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:1000]
    rec["seconds"] = round(time.time() - t0, 1)
    trace, tokens, final = [], 0, None
    if runner is not None:
        rec["tools_available"] = getattr(runner, "tool_names", [])
        rec["interactions"] = runner.interactions
        rec["last_actions"] = [f"{x.role}:{x.action_type}:{x.status}:{str(x.messages)[:400]}"
                               for x in runner.actions.get_actions()[-5:]]
        errs = [x for x in runner.actions.get_actions() if x.action_type == "error"]
        if errs and not rec.get("error"):
            rec["error"] = str(errs[-1].messages)[:1000]
        for act in runner.actions.get_actions():
            o = act.output if isinstance(act.output, dict) else {}
            if act.role == ActionRole.TOOL and act.status != ActionStatus.PROCESSING:
                inp = act.input if isinstance(act.input, dict) else {}
                args = inp.get("arguments", inp)
                res = o.get("raw_output", o)
                err = None
                if isinstance(res, dict) and res.get("success") == 0:
                    err = str(res.get("error"))[:300]
                trace.append({"tool": act.action_type, "ok": act.status == ActionStatus.SUCCESS and not err,
                              "error": err, "args": (args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str))[:1500]})
            if act.action_type and act.action_type.endswith("_response") and o:
                final = o.get("response") or final
                tokens = max(tokens, int(o.get("tokens_used") or 0))
            usage = o.get("usage") if isinstance(o.get("usage"), dict) else None
            if usage and usage.get("total_tokens"):
                tokens = max(tokens, int(usage["total_tokens"]))
    rec.update(tokens=tokens, final_response=final, trace=trace, run_id=latest_run(runs_dir) if runs_dir.exists() else None)
    (out / "record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
