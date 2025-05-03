#!/usr/bin/env python
"""
Compute average metrics for a set of predictions.
"""
import argparse, json, pandas as pd
from rag.utils.json_io import read_jsonl
from rag.evaluation import metrics as M

def safe_load(text: str):
    text = text.strip("` \n")
    first, last = text.find("{"), text.rfind("}")
    return json.loads(text[first:last+1])

def main(args):
    gold = {ex["nl"]: ex["json"] for ex in read_jsonl(args.gold)}
    preds = read_jsonl(args.pred)
    rows = []
    for ex in preds:
        nl = ex["nl"]
        pred_json = safe_load(ex["pred"])
        gold_json = gold[nl]
        rows.append({
            "TriggerEM":    M.trigger_em(pred_json, gold_json),
            "BofS":         M.bag_of_steps(pred_json, gold_json),
            "HallucSteps":  M.halluc_step(pred_json, gold_json),
            "HallucTable":  M.halluc_table(pred_json, gold_json),
        })
    df = pd.DataFrame(rows).mean()
    print(df.round(3).to_string())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=str, required=True)
    p.add_argument("--pred", type=str, required=True)
    main(p.parse_args())
