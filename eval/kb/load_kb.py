"""Load the eval knowledge base (isolated project ``de_eval``, root eval/kb).

- semantic models + metrics: eval/semantic/<datasource>/*.yml (synced to the semantic/metric stores)
- business rules: eval/kb/rules.md -> ext_knowledge store (search_text + explanation per rule)

Knowledge is written from business definitions only; no gold SQL from eval/datasets is copied.

Usage: .venv/bin/python eval/kb/load_kb.py
"""

import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataengineer.configuration.agent_config_loader import load_agent_config  # noqa: E402
from dataengineer.storage.ext_knowledge.store import ExtKnowledgeRAG  # noqa: E402
from dataengineer.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge  # noqa: E402
from dataengineer.storage.semantic_model.semantic_model_init import process_semantic_yaml_file  # noqa: E402
from dataengineer.utils.loggings import configure_logging  # noqa: E402

CFG = str(ROOT / "conf/eval_agent.yml")
SUBJECT = {
    "california_schools": "Education/California/Schools",
    "demo": "Finance/Banking/BankFailures",
}
BANK_RULES = {"R11", "R12", "R13"}


def rules_csv(out: Path) -> int:
    rows = []
    for line in (ROOT / "eval/kb/rules.md").read_text().splitlines():
        m = re.match(r"\|\s*(R\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", line)
        if not m:
            continue
        rid, term, definition = m.groups()
        subject = SUBJECT["demo"] if rid in BANK_RULES else SUBJECT["california_schools"]
        name = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
        rows.append({
            "subject_path": subject,
            "name": name,
            "search_text": f"{term} definition business rule",
            "explanation": f"Business rule '{term}': {definition}",
        })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_path", "name", "search_text", "explanation"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    configure_logging(False, console_output=False)
    for ds in ("california_schools", "demo"):
        cfg = load_agent_config(config=CFG, datasource=ds)
        target = cfg.path_manager.semantic_model_path(ds)  # where gen_semantic_model would write them
        target.mkdir(parents=True, exist_ok=True)
        for yml in sorted((ROOT / "eval/semantic" / ds).glob("*.yml")):
            shutil.copy(yml, target / yml.name)
            ok, err = process_semantic_yaml_file(str(yml), cfg)
            print(f"semantic {ds}/{yml.name}: {'ok' if ok else err}")

    cfg = load_agent_config(config=CFG, datasource="california_schools")
    csv_path = ROOT / "eval/kb/rules.csv"
    n = rules_csv(csv_path)
    rag = ExtKnowledgeRAG(cfg)
    rag.truncate()
    init_ext_knowledge(rag.store, str(csv_path), build_mode="overwrite", pool_size=2)
    print(f"ext_knowledge: {n} rules -> store size {rag.store.table_size()}")


if __name__ == "__main__":
    main()
