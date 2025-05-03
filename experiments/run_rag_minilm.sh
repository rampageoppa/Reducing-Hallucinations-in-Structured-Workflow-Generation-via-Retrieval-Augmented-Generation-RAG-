# experiments/run_rag_minilm.sh
python rag/retrieval/build_index.py --retriever minilm --finetune --train data/train.jsonl
python rag/generation/run_generation.py \
  --prompt rag --retriever minilm --k 3 \
  --input data/test.jsonl \
  --output outputs/rag_minilm/preds.jsonl
python rag/evaluation/evaluate.py \
  --gold data/test.jsonl --pred outputs/rag_minilm/preds.jsonl
