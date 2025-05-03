#!/usr/bin/env python
"""
Build TF‑IDF or MiniLM indices for step + table pools.
Outputs pickled retriever objects under rag/retrieval/indices/.
"""
import pickle, argparse, json, pathlib
from rag.retrieval.tfidf_retriever import TFIDFRetriever
from rag.retrieval.minilm_retriever import MiniLMRetriever
from rag.utils.json_io import read_jsonl
from tqdm import tqdm

STEP_POOL = [
    "create_ticket", "assign_agent", "send_email", "send_sms",
    "post_to_slack", "generate_report", "update_record",
    "archive_record", "close_ticket", "escalate_issue",
    "notify_manager", "reopen_ticket",
]
TABLE_POOL = ["incident", "change_request", "user", "task", "problem"]

def collect_positive_pairs(train_path: str):
    pairs = []
    for ex in read_jsonl(train_path):
        q = ex["nl"]
        for step in ex["json"]["steps"]:
            pairs.append((q, step["name"]))
        pairs.append((q, ex["json"]["trigger"]["table"]))
    return pairs

def main(args):
    out_dir = pathlib.Path("rag/retrieval/indices")
    out_dir.mkdir(parents=True, exist_ok=True)

    # choose retriever
    if args.retriever == "tfidf":
        retriever = TFIDFRetriever()
    else:  # minilm
        retriever = MiniLMRetriever()
        if args.finetune:
            pairs = collect_positive_pairs(args.train)
            retriever.finetune(pairs, epochs=5)

    # build separate indices for steps and tables
    retriever.build_index(STEP_POOL + TABLE_POOL)
    out_file = out_dir / f"{args.retriever}.pkl"
    with out_file.open("wb") as f:
        pickle.dump(retriever, f)
    print(f"Saved retriever to {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", choices=["tfidf", "minilm"], default="minilm")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--train", type=str, default="data/train.jsonl")
    main(p.parse_args())
