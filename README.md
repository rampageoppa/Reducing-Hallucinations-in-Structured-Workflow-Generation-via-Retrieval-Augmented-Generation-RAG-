# Reducing Hallucinations in Structured Workflow Generation via Retrieval‑Augmented Generation

Official code to reproduce the paper (Gao & Li).

## 1. Quick Start

```bash
# clone and create environment (Conda recommended)
conda env create -f env.yaml          # or: pip install -r requirements.txt
conda activate rag-workflow

# 1) generate synthetic dataset
python data/generate_synthetic.py --n_train 200 --n_dev 50 --n_test 50

# 2) build TF‑IDF or dense indices
python rag/retrieval/build_index.py --retriever tfidf
python rag/retrieval/build_index.py --retriever minilm --finetune

# 3) run generation
bash experiments/run_baseline.sh         # LLM without retrieval
bash experiments/run_rag_minilm.sh       # RAG with dense MiniLM

# 4) evaluate
python rag/evaluation/evaluate.py --pred_dir outputs/rag_minilm
```

## 2. Code Structure

```
rag-workflow-hallucination/
├── README.md
├── LICENSE
├── requirements.txt
├── env.yaml                # optional Conda env spec
├── data/
│   ├── generate_synthetic.py
│   ├── workflow_schema.json
│   └── README.md
├── rag/
│   ├── __init__.py
│   ├── retrieval/
│   │   ├── retriever_base.py
│   │   ├── tfidf_retriever.py
│   │   ├── minilm_retriever.py
│   │   └── build_index.py
│   ├── generation/
│   │   ├── prompt_templates.py
│   │   └── run_generation.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   └── utils/
│       ├── json_io.py
│       └── seed.py
├── experiments/
│   ├── run_baseline.sh
│   ├── run_rag_minilm.sh
│   └── run_ablation_tfidf.sh
└── figures/
    └── base_vs_rag_metrics.png   # auto‑generated
```
