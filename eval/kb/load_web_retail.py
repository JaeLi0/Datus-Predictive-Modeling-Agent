"""Register Online Retail II with the web agent (conf/agent.yml, datasource ``online_retail``).

Copies the reviewed semantic models (eval/semantic/online_retail_ii/*.yml) into the project's semantic-model
directory and syncs them to the semantic store, so the web agent can search entities/measures and gen_model
can build point-in-time features from them.

Usage: .venv/bin/python eval/kb/load_web_retail.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataengineer.configuration.agent_config_loader import load_agent_config  # noqa: E402
from dataengineer.storage.semantic_model.semantic_model_init import process_semantic_yaml_file  # noqa: E402
from dataengineer.utils.loggings import configure_logging  # noqa: E402

DS = "online_retail"


def main():
    configure_logging(False, console_output=False)
    cfg = load_agent_config(config=str(ROOT / "conf/agent.yml"), datasource=DS)
    target = cfg.path_manager.semantic_model_path(DS)
    for yml in sorted((ROOT / "eval/semantic/online_retail_ii").glob("*.yml")):
        shutil.copy(yml, target / yml.name)
        ok, err = process_semantic_yaml_file(str(target / yml.name), cfg)
        print(f"semantic {DS}/{yml.name}: {'ok' if ok else err}")


if __name__ == "__main__":
    main()
