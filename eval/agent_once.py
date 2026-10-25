"""Run one question through a sub-agent non-interactively and dump a JSON record.

Same path as ``dataengineer-cli --print`` (PrintModeRunner), with three eval-only changes:
- persistent memory is disabled, so answers cannot leak between questions;
- the ``task`` delegation tool is removed, so a group cannot reach tools outside its whitelist;
- ask_user interactions get a fixed reply (no human in the loop) and are counted.

Usage:
  python eval/agent_once.py --config CFG --datasource DS --subagent eval_A --question "..." --out rec.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataengineer.cli.print_mode import PrintModeRunner  # noqa: E402
from dataengineer.schemas.action_history import ActionRole, ActionStatus  # noqa: E402
from dataengineer.utils.async_utils import run_async  # noqa: E402
from dataengineer.utils.loggings import configure_logging  # noqa: E402

AUTO_REPLY = "No clarification is available. Pick the most reasonable interpretation and continue."


class EvalRunner(PrintModeRunner):
    def __init__(self, args):
        super().__init__(args)
        self.interactions = 0

    def run(self):
        from dataengineer.agent.node.node_factory import create_interactive_node, create_node_input

        node = create_interactive_node(self.subagent_name, self.agent_config, node_id_suffix="_eval", scope=self.scope)
        node.memory_enabled = False
        node.tools = [t for t in node.tools if t.name != "task"]
        self.tool_names = [t.name for t in node.tools]
        node.input = create_node_input(
            user_message=self.message,
            node=node,
            catalog=None,
            database=self.database,
            db_schema=None,
            at_tables=[],
            at_metrics=[],
            at_sqls=[],
        )
        run_async(self._stream_chat(node))

    def _write_payload(self, payload):
        pass

    def _read_interaction_input(self) -> str:
        self.interactions += 1
        return AUTO_REPLY


def _short(v, n=2000):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--datasource", required=True)
    p.add_argument("--subagent", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    configure_logging(False, console_output=False)

    ns = argparse.Namespace(
        config=a.config, datasource=a.datasource, print_mode=a.question, subagent=a.subagent,
        resume=None, proxy_tools=None, session_scope="eval", stream_thinking=False, debug=False,
        db_type=None, db_path=None, history_file=None, no_color=True, save_llm_trace=False,
        filesystem_strict=None, web=False, port=None, host=None, chatbot_dist=None,
    )
    t0 = time.time()
    rec = {"question": a.question, "subagent": a.subagent, "datasource": a.datasource}
    runner = None
    try:
        runner = EvalRunner(ns)
        runner.run()
        rec["error"] = None
    except Exception as e:  # keep the record; the eval counts it as an execution error
        rec["error"] = f"{type(e).__name__}: {e}"[:1000]
    rec["seconds"] = round(time.time() - t0, 1)

    final_sql, tokens, trace, last_ok_sql = None, 0, [], None
    if runner is not None:
        rec["tools_available"] = getattr(runner, "tool_names", [])
        rec["interactions"] = runner.interactions
        rec["actions"] = [f"{a.role}:{a.action_type}:{a.status}" for a in runner.actions.get_actions()][-6:]
        for act in runner.actions.get_actions():
            out = act.output if isinstance(act.output, dict) else {}
            if act.role == ActionRole.TOOL:
                inp = act.input if isinstance(act.input, dict) else {}
                args = inp.get("arguments", inp)
                trace.append({"tool": act.action_type, "ok": act.status == ActionStatus.SUCCESS, "args": _short(args, 600)})
                if act.action_type == "read_query" and act.status == ActionStatus.SUCCESS:
                    try:
                        a_ = json.loads(args) if isinstance(args, str) else args
                        last_ok_sql = a_.get("sql") or last_ok_sql
                    except Exception:
                        pass
            if act.action_type and act.action_type.endswith("_response") and out:
                final_sql = out.get("sql") or final_sql
                stats = out.get("execution_stats") or {}
                tokens = stats.get("total_tokens") or tokens
            usage = out.get("usage") if isinstance(out.get("usage"), dict) else None
            if usage and usage.get("total_tokens"):
                tokens = max(tokens, int(usage["total_tokens"]))
    rec.update(final_sql=final_sql, last_ok_sql=last_ok_sql, tokens=tokens, trace=trace)
    Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
