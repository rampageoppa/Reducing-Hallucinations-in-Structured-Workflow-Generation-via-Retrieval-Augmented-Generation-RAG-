#!/usr/bin/env python
"""
Generate JSON workflows with either BASE or RAG prompt.

Example:
python rag/generation/run_generation.py \
    --prompt rag --retriever minilm --k 3 \
    --input data/test.jsonl --output outputs/rag_minilm/preds.jsonl
"""
import argparse, json, pathlib, pickle, time, os, sys
from typing import Dict, List
import openai, tiktoken
from rag.utils.json_io import read_jsonl, write_jsonl
from rag.generation.prompt_templates import BASE_TEMPLATE, RAG_TEMPLATE

# ------------------ OpenAI key ------------------
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # set env var

def call_openai(prompt: str, model: str = "gpt-4o-mini-2024-07-18",
                max_tokens: int = 350) -> Dict:
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ------------------ generation loop ------------------
def main(args):
    data = read_jsonl(args.input)
    preds = []

    # load retriever if RAG
    if args.prompt == "rag":
        with open(f"rag/retrieval/indices/{args.retriever}.pkl", "rb") as f:
            retriever = pickle.load(f)

    for ex in data:
        nl = ex["nl"]
        # --------- build prompt -------------
        if args.prompt == "base":
            prompt = BASE_TEMPLATE.format(nl=nl)
        else:
            steps = retriever.retrieve(nl, k=args.k)
            # split step/table lists
            step_sugg = [s for s in steps if "_" in s]
            tab_sugg  = [s for s in steps if "_" not in s]
            prompt = RAG_TEMPLATE.format(
                steps="\n".join(f"- {s}" for s in step_sugg),
                tables="\n".join(f"- {t}" for t in tab_sugg),
                nl=nl
            )
        # --------- call LLM -------------
        raw = call_openai(prompt)
        preds.append({"nl": nl, "pred": raw})

        time.sleep(0.4)  # avoid rate limit

    write_jsonl(preds, args.output)
    print(f"Wrote {len(preds)} predictions to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", choices=["base", "rag"], default="base")
    p.add_argument("--retriever", choices=["tfidf", "minilm"], default="minilm")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--input",  type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    main(p.parse_args())
