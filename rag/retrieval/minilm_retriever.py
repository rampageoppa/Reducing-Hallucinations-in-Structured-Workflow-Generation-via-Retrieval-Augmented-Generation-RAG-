from typing import List, Tuple
import faiss, numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from .retriever_base import BaseRetriever

class MiniLMRetriever(BaseRetriever):
    """
    Dense retriever using paraphrase‑MiniLM‑L6‑v2 with optional fine‑tuning.
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self._cands: List[str] = []

    # ---------- training ----------
    def finetune(self, pairs: List[Tuple[str, str]], epochs: int = 5) -> None:
        examples = [InputExample(texts=[q, pos]) for q, pos in pairs]
        loader = DataLoader(examples, batch_size=16, shuffle=True)
        loss_fn = losses.CosineSimilarityLoss(self.encoder)
        self.encoder.fit(train_objectives=[(loader, loss_fn)],
                         epochs=epochs, show_progress_bar=True)

    # ---------- indexing ----------
    def build_index(self, candidates: List[str]) -> None:
        self._cands = candidates
        vecs = self.encoder.encode(candidates, normalize_embeddings=True)
        d = vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vecs.astype("float32"))

    # ---------- inference ----------
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(vec, k)
        return [self._cands[i] for i in I[0]]
