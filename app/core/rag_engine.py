import os
import glob
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import settings

class SimpleBM25:
    """Lightweight, self-contained Okapi BM25 implementation for hybrid search."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / max(self.corpus_size, 1)
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.doc_lens = [len(doc) for doc in corpus]
        
        df: Dict[str, int] = {}
        for document in corpus:
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] = df.get(word, 0) + 1

        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(token, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1)
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_lens[idx] / self.avg_doc_len))
                scores[idx] += idf_val * (num / denom)
        return scores


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine
    Fulfills Layer 4 (Knowledge and Retrieval):
    - Document chunking with overlap (prevents splitting facts across boundaries)
    - Hybrid retrieval: Dense (TF-IDF vector cosine similarity) + Sparse (BM25 keyword)
    - Reciprocal Rank Fusion (RRF) to merge candidate lists
    - Cross-scoring re-ranking pass with category & exact match boosts
    - Relevance cutoff threshold: returns "not found" instead of hallucinating on weak context
    - Document versioning & metadata
    """

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or (Path(__file__).resolve().parent.parent / "data" / "knowledge_base")
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self.bm25: Optional[SimpleBM25] = None
        self.cutoff_threshold: float = settings.RAG_CUTOFF_THRESHOLD
        self.load_and_index_documents()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def chunk_document_with_overlap(self, text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
        """Split text into overlapping chunks so context at boundaries is not lost."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for p in paragraphs:
            if len(current) + len(p) <= chunk_size:
                current = (current + "\n\n" + p).strip()
            else:
                if current:
                    chunks.append(current)
                    # Keep tail as overlap
                    current = current[-overlap:] + " " + p if len(current) > overlap else p
                else:
                    chunks.append(p)
                    current = ""
        if current and current not in chunks:
            chunks.append(current)

        return [c for c in chunks if len(c.strip()) > 25]

    def load_and_index_documents(self):
        """Loads all policy documents, generates overlapping chunks, and creates Hybrid indexes."""
        self.documents = []
        if not self.kb_dir.exists():
            return

        files = glob.glob(str(self.kb_dir / "*.md"))
        doc_id = 1
        raw_texts = []
        tokenized_corpus = []

        for filepath in sorted(files):
            filename = os.path.basename(filepath)
            title = filename.replace(".md", "").replace("_", " ").title()
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self.chunk_document_with_overlap(content)
            for idx, chunk in enumerate(chunks):
                doc_obj = {
                    "id": f"DOC-{doc_id}",
                    "title": f"{title} (Part {idx+1})",
                    "file_name": filename,
                    "category": title,
                    "version": "1.0",
                    "content": chunk
                }
                self.documents.append(doc_obj)
                doc_text = f"{title} {chunk}"
                raw_texts.append(doc_text)
                tokenized_corpus.append(self.tokenize(doc_text))
                doc_id += 1

        if raw_texts:
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.doc_vectors = self.vectorizer.fit_transform(raw_texts)
            self.bm25 = SimpleBM25(tokenized_corpus)

    @staticmethod
    def rrf(rank_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion (Cormack et al., 2009) to merge ranked candidate lists."""
        scores: Dict[int, float] = {}
        for ranked in rank_lists:
            for rank, doc_idx in enumerate(ranked, start=1):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + (1.0 / (k + rank))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval (Dense + Sparse + RRF) with re-ranking and relevance cutoff.
        """
        if not self.documents or self.vectorizer is None or self.doc_vectors is None or self.bm25 is None:
            return []

        # 1. Dense Vector Search (Cosine Similarity)
        query_vec = self.vectorizer.transform([query])
        dense_scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        dense_ranked = dense_scores.argsort()[::-1][:top_k * 4]

        # 2. Sparse BM25 Keyword Search
        q_tokens = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 4]

        # 3. Reciprocal Rank Fusion
        fused = self.rrf([list(dense_ranked), list(bm25_ranked)], k=60)

        # 4. Re-ranking Pass with domain-specific boosts
        query_lower = query.lower()
        reranked = []
        for doc_idx, rrf_score in fused[:top_k * 3]:
            doc = dict(self.documents[doc_idx])
            dense_s = float(dense_scores[doc_idx])
            
            # Base combined score
            score = (dense_s * 0.7) + (min(rrf_score * 30.0, 1.0) * 0.3)

            # Metadata category boost if matched
            if category_filter and category_filter.lower() in doc["category"].lower():
                score += 0.15

            # Keyword query boosts
            if any(term in query_lower for term in doc["category"].lower().split()):
                score += 0.15
            if "refund" in query_lower and "refund" in doc["title"].lower():
                score += 0.20
            if "cancellation" in query_lower and "cancellation" in doc["title"].lower():
                score += 0.20
            if "delivery" in query_lower and "delivery" in doc["title"].lower():
                score += 0.15

            doc["score"] = round(min(score, 0.99), 4)
            reranked.append(doc)

        reranked.sort(key=lambda x: x["score"], reverse=True)

        # 5. Relevance Cutoff Gate (returns 'not found' instead of hallucinating on weak context)
        kept = [d for d in reranked if d["score"] >= self.cutoff_threshold]
        return kept[:top_k]

# Global singleton RAG instance
rag_service = RAGEngine()
