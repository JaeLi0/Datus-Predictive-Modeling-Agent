# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenModelAgenticNode: the predictive-modeling sub-agent.

The LLM makes the modeling decisions (task definition, cleaning rules, feature templates, split method,
which leakage findings to act on, model choice, conclusions); the model_tools functions do every computation
so a run is reproducible from its decision log. Eight steps: define_task, profile_data, clean_data,
build_features, make_split (+ leakage_check), train_select, evaluate, explain + export_artifacts.

Node config (agent.yml ``agentic_nodes.gen_model``): ``max_turns`` (default 60), ``guardrails`` (default true;
false removes leakage_check and the split-method guardrail, for the no-guardrail ablation), ``time_budget``
(FLAML seconds per train_select, default 60).
"""

from pathlib import Path
from typing import Literal, Optional

from dataengineer.agent.node.agentic_node import AgenticNode
from dataengineer.agent.node.gen_job_agentic_node import GenJobAgenticNode
from dataengineer.configuration.agent_config import AgentConfig
from dataengineer.configuration.node_type import NodeType
from dataengineer.tools.func_tool import DBFuncTool
from dataengineer.utils.exceptions import DataEngineerException, ErrorCode
from dataengineer.utils.loggings import get_logger

logger = get_logger(__name__)


class GenModelAgenticNode(GenJobAgenticNode):
    """Modeling sub-agent. Reuses GenJobAgenticNode's prompt rendering and streaming loop (and therefore its
    SemanticNodeInput handling); only the tools, prompt template and settings differ."""

    NODE_NAME = "gen_model"
    DEFAULT_SKILLS = ""

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        is_subagent: bool = False,
        guardrails: Optional[bool] = None,
        time_budget: Optional[float] = None,
        runs_dir: Optional[Path] = None,
    ):
        self.execution_mode = execution_mode
        cfg = {}
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            raw = agent_config.agentic_nodes[self.NODE_NAME]
            cfg = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {})
        self.max_turns = int(cfg.get("max_turns", 60))
        self.guardrails = bool(cfg.get("guardrails", True)) if guardrails is None else guardrails
        self.time_budget = float(cfg.get("time_budget", 60)) if time_budget is None else time_budget
        self.runs_dir = runs_dir

        AgenticNode.__init__(
            self,
            node_id=f"{self.NODE_NAME}_node",
            description="Predictive modeling node: gen_model",
            node_type=NodeType.TYPE_GEN_MODEL,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            is_subagent=is_subagent,
        )
        self.db_func_tool: Optional[DBFuncTool] = None
        self.filesystem_func_tool = None
        self.ask_user_tool = None
        self.model_tools = None
        self.setup_tools()

    def setup_tools(self):
        if not self.agent_config:
            return
        self.tools = []
        try:
            from dataengineer.tools.model_tools import ModelTools

            self.db_func_tool = DBFuncTool(agent_config=self.agent_config, sub_agent_name=self.NODE_NAME)
            self.tools.extend(self.db_func_tool.available_tools())  # read-only: explore before defining the task
            self.model_tools = ModelTools(self.agent_config, db_func_tool=self.db_func_tool,
                                          enforce_split=self.guardrails, guardrails=self.guardrails,
                                          time_budget=self.time_budget, runs_dir=self.runs_dir)
            self.tools.extend(self.model_tools.available_tools())
        except Exception as e:
            logger.exception("Failed to setup modeling tools")
            raise DataEngineerException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Failed to setup tools for {self.NODE_NAME}: {e}"},
            ) from e
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        logger.debug(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[t.name for t in self.tools]}")

    def _prepare_template_context(self, user_input) -> dict:
        context = super()._prepare_template_context(user_input)
        context["guardrails"] = self.guardrails
        context["interactive"] = self.execution_mode == "interactive" and self.ask_user_tool is not None
        return context
