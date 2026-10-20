from sentence_transformers import CrossEncoder
from langsmith import traceable

from src.retriever import load_store

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.reranker = CrossEncoder(CROSS_ENCODER)
        self.fetch_k = fetch_k
        self.top_k = top_k

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]