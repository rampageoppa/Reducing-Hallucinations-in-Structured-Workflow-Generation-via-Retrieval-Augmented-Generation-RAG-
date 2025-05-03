# experiments/run_baseline.sh
python rag/generation/run_generation.py \
  --prompt base --input data/test.jsonl \
  --output outputs/base/preds.jsonl
python rag/evaluation/evaluate.py \
  --gold data/test.jsonl --pred outputs/base/preds.jsonl
