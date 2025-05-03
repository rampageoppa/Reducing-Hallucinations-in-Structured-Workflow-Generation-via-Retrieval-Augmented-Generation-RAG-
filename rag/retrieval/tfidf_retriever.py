from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .retriever_base import BaseRetriever

class TFIDFRetriever(BaseRetriever):
    """Simple bag‑of‑words lexical retriever."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._cands: List[str] = []

    def build_index(self, candidates: List[str]) -> None:
        self._cands = candidates
        self._tfidf = self.vectorizer.fit_transform(candidates)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._tfidf).flatten()
        idxs = sims.argsort()[::-1][:k]
        return [self._cands[i] for i in idxs]
