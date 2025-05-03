#!/usr/bin/env python
"""
Generate the synthetic train / dev / test splits used in the paper.

Each sample = { "nl": <query>, "json": <gold_workflow_dict> }
Outputs three JSONL files in data/ (train.jsonl, dev.jsonl, test.jsonl).
"""
import json, random, argparse, uuid, pathlib
from typing import List, Dict

STEP_POOL = [
    "create_ticket", "assign_agent", "send_email", "send_sms",
    "post_to_slack", "generate_report", "update_record",
    "archive_record", "close_ticket", "escalate_issue",
    "notify_manager", "reopen_ticket",
]
TABLE_POOL = ["incident", "change_request", "user", "task", "problem"]

TEMPLATE = (
    "When a new record is inserted into the {table} table, "
    "{steps}."
)

def make_steps(step_names: List[str]) -> List[Dict]:
    """Return JSON list of steps with ids / parents chained."""
    steps = []
    parent = None
    for i, name in enumerate(step_names, 1):
        sid = f"{i}"
        steps.append({"id": sid, "name": name, "parent": parent})
        parent = sid
    return steps

def make_example() -> Dict:
    table = random.choice(TABLE_POOL)
    k = random.randint(2, 4)
    step_names = random.sample(STEP_POOL, k)
    query_steps = ", ".join(name.replace("_", " ") for name in step_names)
    nl = TEMPLATE.format(table=table, steps=query_steps)
    gold = {
        "trigger": {"type": "on_insert", "table": table},
        "steps": make_steps(step_names),
    }
    return {"nl": nl, "json": gold}

def main(n_train: int, n_dev: int, n_test: int, out_dir: str = "data"):
    random.seed(42)
    splits = {"train": n_train, "dev": n_dev, "test": n_test}
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    for split, n in splits.items():
        path = pathlib.Path(out_dir) / f"{split}.jsonl"
        with path.open("w") as f:
            for _ in range(n):
                json.dump(make_example(), f)
                f.write("\n")
        print(f"Wrote {n} examples to {path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_dev",   type=int, default=50)
    p.add_argument("--n_test",  type=int, default=50)
    args = p.parse_args()
    main(args.n_train, args.n_dev, args.n_test)
