# experiments/run_ablation_tfidf.sh
python rag/retrieval/build_index.py --retriever tfidf
python rag/generation/run_generation.py \
  --prompt rag --retriever tfidf --k 3 \
  --input data/test.jsonl \
  --output outputs/rag_tfidf/preds.jsonl
python rag/evaluation/evaluate.py \
  --gold data/test.jsonl --pred outputs/rag_tfidf/preds.jsonl
